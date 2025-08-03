"""
Repository pattern implementation for data access layer.
Provides clean abstraction between business logic and database operations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from uuid import uuid4

from sqlalchemy import and_, or_, desc, asc, func, select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .models import (
    PredictionRecord, AuditLog, ModelVersion, UserSession, 
    ImageMetadata, SystemConfiguration, PredictionStatus, 
    AuditEventType, ModelStatus
)


logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository with common database operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def commit(self):
        """Commit the session."""
        await self.session.commit()
    
    async def rollback(self):
        """Rollback the session."""
        await self.session.rollback()
    
    async def refresh(self, instance):
        """Refresh an instance from the database."""
        await self.session.refresh(instance)


class PredictionRepository(BaseRepository):
    """Repository for prediction records with comprehensive querying."""
    
    async def create_prediction(
        self,
        request_id: str,
        image_hash: str,
        prediction: str,
        confidence: float,
        class_probabilities: Dict[str, float],
        model_version: str,
        inference_time_ms: float,
        processing_time_ms: float,
        session_id: Optional[str] = None,
        image_quality_score: Optional[float] = None,
        clinical_recommendation: Optional[str] = None,
        risk_level: Optional[str] = None,
        urgency_level: Optional[str] = None
    ) -> PredictionRecord:
        """Create a new prediction record."""
        
        prediction_record = PredictionRecord(
            request_id=request_id,
            session_id=session_id,
            image_hash=image_hash,
            prediction=prediction,
            confidence=confidence,
            class_probabilities=class_probabilities,
            model_version=model_version,
            inference_time_ms=inference_time_ms,
            processing_time_ms=processing_time_ms,
            image_quality_score=image_quality_score,
            clinical_recommendation=clinical_recommendation,
            risk_level=risk_level,
            urgency_level=urgency_level,
            status=PredictionStatus.COMPLETED
        )
        
        self.session.add(prediction_record)
        await self.session.flush()
        return prediction_record
    
    async def get_by_id(self, prediction_id: str) -> Optional[PredictionRecord]:
        """Get prediction by ID."""
        result = await self.session.execute(
            select(PredictionRecord)
            .options(selectinload(PredictionRecord.image_metadata))
            .where(PredictionRecord.id == prediction_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_request_id(self, request_id: str) -> Optional[PredictionRecord]:
        """Get prediction by request ID."""
        result = await self.session.execute(
            select(PredictionRecord)
            .where(PredictionRecord.request_id == request_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_session(
        self, 
        session_id: str, 
        limit: int = 100,
        offset: int = 0
    ) -> List[PredictionRecord]:
        """Get predictions for a user session."""
        result = await self.session.execute(
            select(PredictionRecord)
            .where(PredictionRecord.session_id == session_id)
            .order_by(desc(PredictionRecord.created_at))
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()
    
    async def get_predictions_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        model_version: Optional[str] = None,
        prediction_type: Optional[str] = None,
        min_confidence: Optional[float] = None
    ) -> List[PredictionRecord]:
        """Get predictions within date range with filters."""
        
        query = select(PredictionRecord).where(
            and_(
                PredictionRecord.created_at >= start_date,
                PredictionRecord.created_at <= end_date
            )
        )
        
        if model_version:
            query = query.where(PredictionRecord.model_version == model_version)
        
        if prediction_type:
            query = query.where(PredictionRecord.prediction == prediction_type)
        
        if min_confidence:
            query = query.where(PredictionRecord.confidence >= min_confidence)
        
        query = query.order_by(desc(PredictionRecord.created_at))
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get prediction statistics."""
        
        # Base query
        base_query = select(PredictionRecord)
        
        if start_date:
            base_query = base_query.where(PredictionRecord.created_at >= start_date)
        if end_date:
            base_query = base_query.where(PredictionRecord.created_at <= end_date)
        
        # Total predictions
        total_result = await self.session.execute(
            select(func.count(PredictionRecord.id)).select_from(base_query.subquery())
        )
        total_predictions = total_result.scalar()
        
        # Predictions by type
        type_result = await self.session.execute(
            select(
                PredictionRecord.prediction,
                func.count(PredictionRecord.id)
            )
            .select_from(base_query.subquery())
            .group_by(PredictionRecord.prediction)
        )
        predictions_by_type = dict(type_result.all())
        
        # Average confidence
        confidence_result = await self.session.execute(
            select(func.avg(PredictionRecord.confidence))
            .select_from(base_query.subquery())
        )
        avg_confidence = confidence_result.scalar()
        
        # Average processing time
        time_result = await self.session.execute(
            select(func.avg(PredictionRecord.processing_time_ms))
            .select_from(base_query.subquery())
        )
        avg_processing_time = time_result.scalar()
        
        # High confidence predictions
        high_confidence_result = await self.session.execute(
            select(func.count(PredictionRecord.id))
            .select_from(base_query.subquery())
            .where(PredictionRecord.confidence >= 0.8)
        )
        high_confidence_count = high_confidence_result.scalar()
        
        return {
            "total_predictions": total_predictions,
            "predictions_by_type": predictions_by_type,
            "average_confidence": float(avg_confidence) if avg_confidence else 0.0,
            "average_processing_time_ms": float(avg_processing_time) if avg_processing_time else 0.0,
            "high_confidence_predictions": high_confidence_count,
            "high_confidence_rate": high_confidence_count / total_predictions if total_predictions > 0 else 0.0
        }
    
    async def update_status(self, prediction_id: str, status: PredictionStatus) -> bool:
        """Update prediction status."""
        result = await self.session.execute(
            update(PredictionRecord)
            .where(PredictionRecord.id == prediction_id)
            .values(status=status, updated_at=func.now())
        )
        return result.rowcount > 0


class AuditRepository(BaseRepository):
    """Repository for HIPAA-compliant audit logging."""
    
    async def log_event(
        self,
        event_type: AuditEventType,
        event_description: str,
        system_component: str,
        success: bool = True,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
        phi_accessed: bool = False,
        data_classification: Optional[str] = None,
        event_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> AuditLog:
        """Create audit log entry."""
        
        audit_log = AuditLog(
            event_type=event_type,
            event_description=event_description,
            user_id=user_id,
            session_id=session_id,
            client_ip=client_ip,
            user_agent=user_agent,
            request_id=request_id,
            endpoint=endpoint,
            method=method,
            phi_accessed=phi_accessed,
            data_classification=data_classification,
            system_component=system_component,
            event_data=event_data,
            success=success,
            error_message=error_message
        )
        
        self.session.add(audit_log)
        await self.session.flush()
        return audit_log
    
    async def get_audit_trail(
        self,
        start_date: datetime,
        end_date: datetime,
        event_types: Optional[List[AuditEventType]] = None,
        user_id: Optional[str] = None,
        phi_only: bool = False,
        limit: int = 1000,
        offset: int = 0
    ) -> List[AuditLog]:
        """Get audit trail with filters."""
        
        query = select(AuditLog).where(
            and_(
                AuditLog.timestamp >= start_date,
                AuditLog.timestamp <= end_date
            )
        )
        
        if event_types:
            query = query.where(AuditLog.event_type.in_(event_types))
        
        if user_id:
            query = query.where(AuditLog.user_id == user_id)
        
        if phi_only:
            query = query.where(AuditLog.phi_accessed == True)
        
        query = query.order_by(desc(AuditLog.timestamp)).limit(limit).offset(offset)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_phi_access_events(
        self,
        start_date: datetime,
        end_date: datetime,
        user_id: Optional[str] = None
    ) -> List[AuditLog]:
        """Get PHI access events for compliance reporting."""
        
        query = select(AuditLog).where(
            and_(
                AuditLog.timestamp >= start_date,
                AuditLog.timestamp <= end_date,
                AuditLog.phi_accessed == True
            )
        )
        
        if user_id:
            query = query.where(AuditLog.user_id == user_id)
        
        query = query.order_by(desc(AuditLog.timestamp))
        
        result = await self.session.execute(query)
        return result.scalars().all()


class ModelVersionRepository(BaseRepository):
    """Repository for model version management."""
    
    async def create_model_version(
        self,
        model_name: str,
        version: str,
        model_type: str,
        file_path: str,
        file_size_bytes: int,
        file_checksum: str,
        status: ModelStatus = ModelStatus.DEVELOPMENT,
        **kwargs
    ) -> ModelVersion:
        """Create new model version."""
        
        model_version = ModelVersion(
            model_name=model_name,
            version=version,
            model_type=model_type,
            file_path=file_path,
            file_size_bytes=file_size_bytes,
            file_checksum=file_checksum,
            status=status,
            **kwargs
        )
        
        self.session.add(model_version)
        await self.session.flush()
        return model_version
    
    async def get_by_name_and_version(
        self, 
        model_name: str, 
        version: str
    ) -> Optional[ModelVersion]:
        """Get model by name and version."""
        result = await self.session.execute(
            select(ModelVersion).where(
                and_(
                    ModelVersion.model_name == model_name,
                    ModelVersion.version == version
                )
            )
        )
        return result.scalar_one_or_none()
    
    async def get_latest_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get latest version of a model."""
        result = await self.session.execute(
            select(ModelVersion)
            .where(ModelVersion.model_name == model_name)
            .order_by(desc(ModelVersion.created_at))
            .limit(1)
        )
        return result.scalar_one_or_none()
    
    async def get_production_models(self) -> List[ModelVersion]:
        """Get all models in production status."""
        result = await self.session.execute(
            select(ModelVersion)
            .where(ModelVersion.status == ModelStatus.PRODUCTION)
            .order_by(desc(ModelVersion.deployment_date))
        )
        return result.scalars().all()
    
    async def promote_to_production(
        self, 
        model_id: str, 
        deployment_date: Optional[datetime] = None
    ) -> bool:
        """Promote model to production status."""
        if not deployment_date:
            deployment_date = datetime.utcnow()
        
        result = await self.session.execute(
            update(ModelVersion)
            .where(ModelVersion.id == model_id)
            .values(
                status=ModelStatus.PRODUCTION,
                deployment_date=deployment_date,
                updated_at=func.now()
            )
        )
        return result.rowcount > 0


class UserRepository(BaseRepository):
    """Repository for user session management."""
    
    async def create_session(
        self,
        session_token: str,
        anonymous_user_id: str,
        client_ip_hash: Optional[str] = None,
        user_agent_hash: Optional[str] = None
    ) -> UserSession:
        """Create new user session."""
        
        session = UserSession(
            session_token=session_token,
            anonymous_user_id=anonymous_user_id,
            client_ip_hash=client_ip_hash,
            user_agent_hash=user_agent_hash
        )
        
        self.session.add(session)
        await self.session.flush()
        return session
    
    async def get_session_by_token(self, session_token: str) -> Optional[UserSession]:
        """Get session by token."""
        result = await self.session.execute(
            select(UserSession)
            .where(
                and_(
                    UserSession.session_token == session_token,
                    UserSession.is_active == True
                )
            )
        )
        return result.scalar_one_or_none()
    
    async def update_session_activity(self, session_id: str) -> bool:
        """Update session last activity."""
        result = await self.session.execute(
            update(UserSession)
            .where(UserSession.id == session_id)
            .values(last_activity=func.now())
        )
        return result.rowcount > 0
    
    async def increment_prediction_count(self, session_id: str) -> bool:
        """Increment prediction count for session."""
        result = await self.session.execute(
            update(UserSession)
            .where(UserSession.id == session_id)
            .values(
                prediction_count=UserSession.prediction_count + 1,
                last_activity=func.now()
            )
        )
        return result.rowcount > 0
    
    async def end_session(self, session_id: str) -> bool:
        """End user session."""
        result = await self.session.execute(
            update(UserSession)
            .where(UserSession.id == session_id)
            .values(
                is_active=False,
                ended_at=func.now()
            )
        )
        return result.rowcount > 0
    
    async def cleanup_expired_sessions(self, expiry_hours: int = 24) -> int:
        """Clean up expired sessions."""
        expiry_time = datetime.utcnow() - timedelta(hours=expiry_hours)
        
        result = await self.session.execute(
            update(UserSession)
            .where(
                and_(
                    UserSession.last_activity < expiry_time,
                    UserSession.is_active == True
                )
            )
            .values(is_active=False, ended_at=func.now())
        )
        return result.rowcount


class ImageMetadataRepository(BaseRepository):
    """Repository for image metadata management."""
    
    async def create_image_metadata(
        self,
        image_hash: str,
        width: int,
        height: int,
        channels: int,
        format: str,
        file_size_bytes: int,
        **kwargs
    ) -> ImageMetadata:
        """Create image metadata record."""
        
        metadata = ImageMetadata(
            image_hash=image_hash,
            width=width,
            height=height,
            channels=channels,
            format=format,
            file_size_bytes=file_size_bytes,
            **kwargs
        )
        
        self.session.add(metadata)
        await self.session.flush()
        return metadata
    
    async def get_by_hash(self, image_hash: str) -> Optional[ImageMetadata]:
        """Get image metadata by hash."""
        result = await self.session.execute(
            select(ImageMetadata).where(ImageMetadata.image_hash == image_hash)
        )
        return result.scalar_one_or_none()
    
    async def get_duplicate_images(self) -> List[ImageMetadata]:
        """Find duplicate images by hash."""
        subquery = select(
            ImageMetadata.image_hash,
            func.count(ImageMetadata.id).label('count')
        ).group_by(ImageMetadata.image_hash).having(func.count(ImageMetadata.id) > 1).subquery()
        
        result = await self.session.execute(
            select(ImageMetadata)
            .join(subquery, ImageMetadata.image_hash == subquery.c.image_hash)
            .order_by(ImageMetadata.image_hash, ImageMetadata.created_at)
        )
        return result.scalars().all()