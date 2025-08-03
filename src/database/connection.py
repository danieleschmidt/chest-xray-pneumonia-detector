"""
Database connection management for the pneumonia detection system.
Implements SQLAlchemy with async support and connection pooling.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy import create_engine, MetaData, event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool


logger = logging.getLogger(__name__)

# Create base class for ORM models
Base = declarative_base()

# Naming convention for constraints (for Alembic)
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}

Base.metadata = MetaData(naming_convention=NAMING_CONVENTION)


class DatabaseManager:
    """
    Database manager with async support and connection pooling.
    Provides enterprise-grade database management for medical applications.
    """
    
    def __init__(self, database_url: str, echo: bool = False):
        self.database_url = database_url
        self.echo = echo
        self._engine = None
        self._async_engine = None
        self._session_factory = None
        self._async_session_factory = None
        
    async def initialize(self) -> None:
        """Initialize database connections and create tables."""
        try:
            # Create async engine
            self._async_engine = create_async_engine(
                self._convert_to_async_url(self.database_url),
                echo=self.echo,
                pool_pre_ping=True,
                pool_recycle=3600,  # Recycle connections every hour
                future=True
            )
            
            # Create sync engine for migrations
            self._engine = create_engine(
                self.database_url,
                echo=self.echo,
                pool_pre_ping=True,
                pool_recycle=3600,
                future=True
            )
            
            # Create session factories
            self._async_session_factory = async_sessionmaker(
                self._async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            self._session_factory = sessionmaker(
                self._engine,
                expire_on_commit=False
            )
            
            # Create tables if they don't exist
            await self._create_tables()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    def _convert_to_async_url(self, url: str) -> str:
        """Convert synchronous database URL to async version."""
        if url.startswith("sqlite:///"):
            return url.replace("sqlite:///", "sqlite+aiosqlite:///")
        elif url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://")
        elif url.startswith("mysql://"):
            return url.replace("mysql://", "mysql+aiomysql://")
        else:
            return url
    
    async def _create_tables(self) -> None:
        """Create database tables."""
        async with self._async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session with automatic cleanup."""
        if not self._async_session_factory:
            raise RuntimeError("Database not initialized")
        
        session = self._async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    @asynccontextmanager
    async def get_sync_session(self):
        """Get synchronous database session (for migrations)."""
        if not self._session_factory:
            raise RuntimeError("Database not initialized")
        
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    async def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            async with self.get_session() as session:
                await session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False
    
    async def close(self) -> None:
        """Close database connections."""
        if self._async_engine:
            await self._async_engine.dispose()
        if self._engine:
            self._engine.dispose()
        logger.info("Database connections closed")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


async def initialize_database(database_url: Optional[str] = None, echo: bool = False) -> DatabaseManager:
    """Initialize the global database manager."""
    global _db_manager
    
    if not database_url:
        database_url = os.getenv("DATABASE_URL", "sqlite:///./pneumonia_detector.db")
    
    _db_manager = DatabaseManager(database_url, echo)
    await _db_manager.initialize()
    
    return _db_manager


def get_database() -> DatabaseManager:
    """Get the global database manager instance."""
    if not _db_manager:
        raise RuntimeError("Database not initialized. Call initialize_database() first.")
    return _db_manager


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Convenience function to get database session."""
    db = get_database()
    async with db.get_session() as session:
        yield session


# Add database event listeners for security and auditing
@event.listens_for(Base.metadata, "before_create")
def receive_before_create(target, connection, tables, **kw):
    """Log table creation events."""
    logger.info(f"Creating database tables: {[t.name for t in tables]}")


@event.listens_for(Base.metadata, "after_create")  
def receive_after_create(target, connection, tables, **kw):
    """Log successful table creation."""
    logger.info(f"Successfully created tables: {[t.name for t in tables]}")


class DatabaseHealthCheck:
    """Database health monitoring utilities."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    async def check_connectivity(self) -> dict:
        """Check database connectivity and return status."""
        try:
            start_time = asyncio.get_event_loop().time()
            is_healthy = await self.db_manager.health_check()
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return {
                "healthy": is_healthy,
                "response_time_ms": response_time,
                "database_url": self.db_manager.database_url.split("@")[-1] if "@" in self.db_manager.database_url else "local",
                "timestamp": asyncio.get_event_loop().time()
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }
    
    async def check_table_integrity(self) -> dict:
        """Check database table integrity."""
        try:
            async with self.db_manager.get_session() as session:
                # Check if required tables exist
                tables_to_check = ["prediction_records", "audit_logs", "model_versions"]
                
                existing_tables = []
                for table_name in tables_to_check:
                    try:
                        await session.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
                        existing_tables.append(table_name)
                    except Exception:
                        pass
                
                return {
                    "healthy": len(existing_tables) == len(tables_to_check),
                    "existing_tables": existing_tables,
                    "missing_tables": list(set(tables_to_check) - set(existing_tables)),
                    "timestamp": asyncio.get_event_loop().time()
                }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }