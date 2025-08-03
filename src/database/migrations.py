"""
Database migration management for the pneumonia detection system.
Provides version control for database schema changes.
"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from sqlalchemy import text, MetaData, Table, Column, String, DateTime, Integer
from sqlalchemy.ext.asyncio import AsyncSession

from .connection import DatabaseManager, Base


logger = logging.getLogger(__name__)


@dataclass
class Migration:
    """Database migration definition."""
    version: str
    name: str
    description: str
    up_sql: str
    down_sql: str
    created_at: datetime


class MigrationManager:
    """
    Database migration manager with version control.
    Provides safe schema evolution for production systems.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.migrations_dir = Path("migrations")
        self.migrations: List[Migration] = []
        self._migration_table_name = "schema_migrations"
    
    async def initialize(self) -> None:
        """Initialize migration system and create migration table."""
        await self._ensure_migration_table()
        await self._load_migrations()
        logger.info("Migration manager initialized")
    
    async def _ensure_migration_table(self) -> None:
        """Create migration tracking table if it doesn't exist."""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self._migration_table_name} (
            version VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            checksum VARCHAR(64) NOT NULL
        )
        """
        
        async with self.db_manager.get_session() as session:
            await session.execute(text(create_table_sql))
            await session.commit()
    
    async def _load_migrations(self) -> None:
        """Load migration files from disk."""
        if not self.migrations_dir.exists():
            self.migrations_dir.mkdir(parents=True)
            logger.info(f"Created migrations directory: {self.migrations_dir}")
            return
        
        migration_files = sorted(self.migrations_dir.glob("*.sql"))
        
        for file_path in migration_files:
            try:
                migration = self._parse_migration_file(file_path)
                self.migrations.append(migration)
            except Exception as e:
                logger.error(f"Failed to parse migration {file_path}: {str(e)}")
        
        logger.info(f"Loaded {len(self.migrations)} migrations")
    
    def _parse_migration_file(self, file_path: Path) -> Migration:
        """Parse migration file and extract metadata."""
        content = file_path.read_text()
        
        # Extract version from filename (e.g., 001_create_tables.sql)
        filename = file_path.stem
        parts = filename.split("_", 1)
        version = parts[0]
        name = parts[1] if len(parts) > 1 else filename
        
        # Look for migration metadata in comments
        lines = content.split("\n")
        description = ""
        up_sql = ""
        down_sql = ""
        
        current_section = "up"
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("-- Description:"):
                description = line.replace("-- Description:", "").strip()
            elif line.startswith("-- UP"):
                current_section = "up"
                continue
            elif line.startswith("-- DOWN"):
                current_section = "down"
                continue
            elif line.startswith("--"):
                continue
            
            if current_section == "up":
                up_sql += line + "\n"
            elif current_section == "down":
                down_sql += line + "\n"
        
        return Migration(
            version=version,
            name=name,
            description=description,
            up_sql=up_sql.strip(),
            down_sql=down_sql.strip(),
            created_at=datetime.fromtimestamp(file_path.stat().st_mtime)
        )
    
    async def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions."""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                text(f"SELECT version FROM {self._migration_table_name} ORDER BY version")
            )
            return [row[0] for row in result.fetchall()]
    
    async def get_pending_migrations(self) -> List[Migration]:
        """Get list of pending migrations."""
        applied = await self.get_applied_migrations()
        return [m for m in self.migrations if m.version not in applied]
    
    async def apply_migration(self, migration: Migration) -> bool:
        """Apply a single migration."""
        try:
            async with self.db_manager.get_session() as session:
                # Execute the migration
                if migration.up_sql:
                    await session.execute(text(migration.up_sql))
                
                # Record the migration
                checksum = self._calculate_checksum(migration.up_sql)
                await session.execute(
                    text(f"""
                        INSERT INTO {self._migration_table_name} 
                        (version, name, description, checksum)
                        VALUES (:version, :name, :description, :checksum)
                    """),
                    {
                        "version": migration.version,
                        "name": migration.name,
                        "description": migration.description,
                        "checksum": checksum
                    }
                )
                
                await session.commit()
                logger.info(f"Applied migration {migration.version}: {migration.name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to apply migration {migration.version}: {str(e)}")
            return False
    
    async def rollback_migration(self, migration: Migration) -> bool:
        """Rollback a single migration."""
        try:
            async with self.db_manager.get_session() as session:
                # Execute rollback SQL
                if migration.down_sql:
                    await session.execute(text(migration.down_sql))
                
                # Remove migration record
                await session.execute(
                    text(f"DELETE FROM {self._migration_table_name} WHERE version = :version"),
                    {"version": migration.version}
                )
                
                await session.commit()
                logger.info(f"Rolled back migration {migration.version}: {migration.name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to rollback migration {migration.version}: {str(e)}")
            return False
    
    async def migrate_up(self, target_version: Optional[str] = None) -> int:
        """Apply all pending migrations up to target version."""
        pending = await self.get_pending_migrations()
        
        if target_version:
            pending = [m for m in pending if m.version <= target_version]
        
        applied_count = 0
        
        for migration in pending:
            if await self.apply_migration(migration):
                applied_count += 1
            else:
                logger.error(f"Migration failed, stopping at {migration.version}")
                break
        
        logger.info(f"Applied {applied_count} migrations")
        return applied_count
    
    async def migrate_down(self, target_version: str) -> int:
        """Rollback migrations down to target version."""
        applied = await self.get_applied_migrations()
        
        # Find migrations to rollback (in reverse order)
        to_rollback = [
            m for m in reversed(self.migrations) 
            if m.version in applied and m.version > target_version
        ]
        
        rolled_back_count = 0
        
        for migration in to_rollback:
            if await self.rollback_migration(migration):
                rolled_back_count += 1
            else:
                logger.error(f"Rollback failed, stopping at {migration.version}")
                break
        
        logger.info(f"Rolled back {rolled_back_count} migrations")
        return rolled_back_count
    
    def _calculate_checksum(self, content: str) -> str:
        """Calculate checksum for migration content."""
        import hashlib
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def create_migration(
        self,
        name: str,
        description: str = "",
        up_sql: str = "",
        down_sql: str = ""
    ) -> Path:
        """Create a new migration file."""
        # Generate version number
        existing_versions = [m.version for m in self.migrations]
        if existing_versions:
            last_version = max(existing_versions)
            next_version = f"{int(last_version) + 1:03d}"
        else:
            next_version = "001"
        
        # Create filename
        filename = f"{next_version}_{name.replace(' ', '_').lower()}.sql"
        file_path = self.migrations_dir / filename
        
        # Create migration content
        content = f"""-- Description: {description}
-- Created: {datetime.now().isoformat()}

-- UP
{up_sql}

-- DOWN
{down_sql}
"""
        
        file_path.write_text(content)
        logger.info(f"Created migration: {file_path}")
        
        return file_path
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get comprehensive migration status."""
        applied = await self.get_applied_migrations()
        pending = await self.get_pending_migrations()
        
        return {
            "total_migrations": len(self.migrations),
            "applied_count": len(applied),
            "pending_count": len(pending),
            "applied_versions": applied,
            "pending_versions": [m.version for m in pending],
            "last_applied": applied[-1] if applied else None,
            "next_pending": pending[0].version if pending else None
        }


# Built-in migrations for initial schema
INITIAL_MIGRATIONS = [
    {
        "version": "001",
        "name": "create_prediction_tables",
        "description": "Create tables for prediction records and audit logs",
        "up_sql": """
        -- Create prediction records table
        CREATE TABLE prediction_records (
            id VARCHAR(36) PRIMARY KEY,
            request_id VARCHAR(255) UNIQUE NOT NULL,
            session_id VARCHAR(36),
            image_hash VARCHAR(64) NOT NULL,
            image_metadata_id VARCHAR(36),
            prediction VARCHAR(50) NOT NULL,
            confidence FLOAT NOT NULL,
            class_probabilities JSON NOT NULL,
            model_version VARCHAR(255) NOT NULL,
            model_id VARCHAR(36),
            inference_time_ms FLOAT NOT NULL,
            processing_time_ms FLOAT NOT NULL,
            image_quality_score FLOAT,
            diagnostic_quality BOOLEAN DEFAULT TRUE,
            clinical_recommendation TEXT,
            risk_level VARCHAR(50),
            urgency_level VARCHAR(50),
            status VARCHAR(50) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Create indexes for prediction records
        CREATE INDEX idx_prediction_created_at ON prediction_records(created_at);
        CREATE INDEX idx_prediction_model_version ON prediction_records(model_version);
        CREATE INDEX idx_prediction_status ON prediction_records(status);
        CREATE INDEX idx_prediction_confidence ON prediction_records(confidence);
        CREATE INDEX idx_prediction_image_hash ON prediction_records(image_hash);
        """,
        "down_sql": """
        DROP TABLE IF EXISTS prediction_records;
        """
    },
    {
        "version": "002", 
        "name": "create_audit_tables",
        "description": "Create HIPAA-compliant audit logging tables",
        "up_sql": """
        -- Create audit logs table
        CREATE TABLE audit_logs (
            id VARCHAR(36) PRIMARY KEY,
            event_type VARCHAR(50) NOT NULL,
            event_description TEXT NOT NULL,
            user_id VARCHAR(255),
            session_id VARCHAR(36),
            client_ip VARCHAR(255),
            user_agent TEXT,
            request_id VARCHAR(255),
            endpoint VARCHAR(255),
            method VARCHAR(10),
            phi_accessed BOOLEAN DEFAULT FALSE,
            data_classification VARCHAR(50),
            system_component VARCHAR(100) NOT NULL,
            process_id VARCHAR(50),
            event_data JSON,
            success BOOLEAN NOT NULL,
            error_message TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            checksum VARCHAR(64)
        );

        -- Create indexes for audit logs
        CREATE INDEX idx_audit_timestamp ON audit_logs(timestamp);
        CREATE INDEX idx_audit_event_type ON audit_logs(event_type);
        CREATE INDEX idx_audit_user_id ON audit_logs(user_id);
        CREATE INDEX idx_audit_phi_accessed ON audit_logs(phi_accessed);
        CREATE INDEX idx_audit_success ON audit_logs(success);
        """,
        "down_sql": """
        DROP TABLE IF EXISTS audit_logs;
        """
    },
    {
        "version": "003",
        "name": "create_model_management_tables",
        "description": "Create tables for model version management",
        "up_sql": """
        -- Create model versions table
        CREATE TABLE model_versions (
            id VARCHAR(36) PRIMARY KEY,
            model_name VARCHAR(255) NOT NULL,
            version VARCHAR(100) NOT NULL,
            model_type VARCHAR(100) NOT NULL,
            file_path VARCHAR(500) NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            file_checksum VARCHAR(64) NOT NULL,
            training_dataset VARCHAR(255),
            training_date TIMESTAMP,
            training_duration_hours FLOAT,
            accuracy FLOAT,
            precision FLOAT,
            recall FLOAT,
            f1_score FLOAT,
            auc_roc FLOAT,
            status VARCHAR(50) DEFAULT 'development',
            deployment_date TIMESTAMP,
            architecture_config JSON,
            hyperparameters JSON,
            description TEXT,
            tags JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Create unique constraint and indexes
        CREATE UNIQUE INDEX idx_model_name_version ON model_versions(model_name, version);
        CREATE INDEX idx_model_status ON model_versions(status);
        CREATE INDEX idx_model_created_at ON model_versions(created_at);
        """,
        "down_sql": """
        DROP TABLE IF EXISTS model_versions;
        """
    },
    {
        "version": "004",
        "name": "create_session_management_tables", 
        "description": "Create tables for user session management",
        "up_sql": """
        -- Create user sessions table
        CREATE TABLE user_sessions (
            id VARCHAR(36) PRIMARY KEY,
            session_token VARCHAR(255) UNIQUE NOT NULL,
            anonymous_user_id VARCHAR(255) NOT NULL,
            client_ip_hash VARCHAR(64),
            user_agent_hash VARCHAR(64),
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE,
            prediction_count INTEGER DEFAULT 0
        );

        -- Create indexes for user sessions
        CREATE INDEX idx_session_token ON user_sessions(session_token);
        CREATE INDEX idx_session_active ON user_sessions(is_active);
        CREATE INDEX idx_session_started_at ON user_sessions(started_at);

        -- Create image metadata table
        CREATE TABLE image_metadata (
            id VARCHAR(36) PRIMARY KEY,
            image_hash VARCHAR(64) UNIQUE NOT NULL,
            original_filename VARCHAR(255),
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            channels INTEGER NOT NULL,
            format VARCHAR(50) NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            sharpness_score FLOAT,
            contrast_score FLOAT,
            brightness_score FLOAT,
            quality_score FLOAT,
            preprocessing_applied JSON,
            enhancement_methods JSON,
            imaging_modality VARCHAR(50) DEFAULT 'X-RAY',
            body_part VARCHAR(50) DEFAULT 'CHEST',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Create indexes for image metadata
        CREATE INDEX idx_image_hash ON image_metadata(image_hash);
        CREATE INDEX idx_image_quality ON image_metadata(quality_score);
        CREATE INDEX idx_image_created_at ON image_metadata(created_at);
        """,
        "down_sql": """
        DROP TABLE IF EXISTS image_metadata;
        DROP TABLE IF EXISTS user_sessions;
        """
    }
]


async def initialize_default_migrations(migration_manager: MigrationManager) -> None:
    """Initialize default migrations if none exist."""
    if not migration_manager.migrations_dir.exists():
        migration_manager.migrations_dir.mkdir(parents=True)
    
    # Create initial migration files if they don't exist
    for migration_data in INITIAL_MIGRATIONS:
        filename = f"{migration_data['version']}_{migration_data['name']}.sql"
        file_path = migration_manager.migrations_dir / filename
        
        if not file_path.exists():
            await migration_manager.create_migration(
                name=migration_data['name'],
                description=migration_data['description'],
                up_sql=migration_data['up_sql'],
                down_sql=migration_data['down_sql']
            )
    
    # Reload migrations
    await migration_manager._load_migrations()