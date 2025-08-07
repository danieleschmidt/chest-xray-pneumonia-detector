"""Robust API with comprehensive error handling and security."""

import logging
import traceback
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import json

from .advanced_scheduler import QuantumMLScheduler
from .security_framework import QuantumSchedulerSecurity, Permission, SecurityLevel
from .health_monitoring import QuantumSchedulerHealthMonitor
from .quantum_scheduler import TaskPriority, TaskStatus

logger = logging.getLogger(__name__)


@dataclass
class APIResponse:
    """Standardized API response format."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: datetime = None
    request_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class QuantumSchedulerRobustAPI:
    """Robust API with comprehensive error handling and security."""
    
    def __init__(self):
        self.scheduler = QuantumMLScheduler(max_parallel_tasks=8)
        self.security = QuantumSchedulerSecurity()
        self.health_monitor = QuantumSchedulerHealthMonitor(self.scheduler)
        
        # API configuration
        self.max_request_size = 10 * 1024 * 1024  # 10MB
        self.request_timeout = 30.0  # seconds
        
        # Initialize default admin user
        self._initialize_default_users()
        
        # Start health monitoring
        self.health_monitor.start_monitoring()
        
    def _initialize_default_users(self) -> None:
        """Initialize default users for testing."""
        try:
            # Create admin user
            self.security.create_user(
                user_id="admin",
                username="admin",
                email="admin@example.com",
                password="AdminPass123!",
                permissions=[p for p in Permission],
                security_level=SecurityLevel.TOP_SECRET
            )
            
            # Create regular user
            self.security.create_user(
                user_id="user",
                username="user",
                email="user@example.com",
                password="UserPass123!",
                permissions=[
                    Permission.READ_TASKS,
                    Permission.CREATE_TASKS,
                    Permission.VIEW_METRICS
                ],
                security_level=SecurityLevel.INTERNAL
            )
            
        except Exception as e:
            logger.warning(f"Error initializing default users: {e}")
    
    def authenticate(self, username: str, password: str, 
                    ip_address: Optional[str] = None) -> APIResponse:
        """Authenticate user and return session token."""
        try:
            session_token = self.security.authenticate_user(
                username=username,
                password=password,
                ip_address=ip_address
            )
            
            if session_token:
                return APIResponse(
                    success=True,
                    data={"session_token": session_token, "expires_in": 28800}  # 8 hours
                )
            else:
                return APIResponse(
                    success=False,
                    error="Invalid credentials",
                    error_code="AUTH_FAILED"
                )
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return APIResponse(
                success=False,
                error="Authentication service unavailable",
                error_code="AUTH_SERVICE_ERROR"
            )
    
    def create_task(self, session_token: str, task_data: Dict[str, Any]) -> APIResponse:
        """Create new task with security validation."""
        try:
            # Validate session
            user = self.security.validate_session(session_token)
            if not user:
                return APIResponse(
                    success=False,
                    error="Invalid or expired session",
                    error_code="SESSION_INVALID"
                )
            
            # Check permissions
            if not self.security.check_permission(session_token, Permission.CREATE_TASKS):
                return APIResponse(
                    success=False,
                    error="Insufficient permissions",
                    error_code="PERMISSION_DENIED"
                )
            
            # Validate and sanitize task data
            validated_data = self.security.secure_task_creation(session_token, task_data)
            
            # Parse priority
            priority = TaskPriority.MEDIUM
            if 'priority' in task_data:
                priority_map = {
                    'low': TaskPriority.LOW,
                    'medium': TaskPriority.MEDIUM,
                    'high': TaskPriority.HIGH,
                    'critical': TaskPriority.CRITICAL
                }
                priority = priority_map.get(task_data['priority'].lower(), TaskPriority.MEDIUM)
            
            # Create task
            task_id = self.scheduler.add_intelligent_task(
                name=validated_data.get('name', 'Untitled Task'),
                description=validated_data.get('description', ''),
                priority=priority,
                tags=task_data.get('tags', [])
            )
            
            # Get created task details
            task = self.scheduler.get_task(task_id)
            task_details = self._serialize_task(task) if task else None
            
            return APIResponse(
                success=True,
                data={
                    "task_id": task_id,
                    "task": task_details
                }
            )
            
        except ValueError as e:
            return APIResponse(
                success=False,
                error=str(e),
                error_code="VALIDATION_ERROR"
            )
        except PermissionError as e:
            return APIResponse(
                success=False,
                error=str(e),
                error_code="PERMISSION_DENIED"
            )
        except Exception as e:
            logger.error(f"Task creation error: {e}")
            logger.error(traceback.format_exc())
            return APIResponse(
                success=False,
                error="Internal server error",
                error_code="INTERNAL_ERROR"
            )
    
    def get_tasks(self, session_token: str, status: Optional[str] = None,
                  limit: int = 100, offset: int = 0) -> APIResponse:
        """Get tasks with filtering and pagination."""
        try:
            # Validate session
            user = self.security.validate_session(session_token)
            if not user:
                return APIResponse(
                    success=False,
                    error="Invalid or expired session",
                    error_code="SESSION_INVALID"
                )
            
            # Check permissions
            if not self.security.check_permission(session_token, Permission.READ_TASKS):
                return APIResponse(
                    success=False,
                    error="Insufficient permissions",
                    error_code="PERMISSION_DENIED"
                )
            
            # Get tasks
            all_tasks = list(self.scheduler.tasks.values())
            
            # Filter by status if specified
            if status:
                status_filter = None
                try:
                    status_map = {
                        'pending': TaskStatus.PENDING,
                        'running': TaskStatus.RUNNING,
                        'completed': TaskStatus.COMPLETED,
                        'blocked': TaskStatus.BLOCKED,
                        'cancelled': TaskStatus.CANCELLED
                    }
                    status_filter = status_map.get(status.lower())
                except:
                    pass
                
                if status_filter:
                    all_tasks = [t for t in all_tasks if t.status == status_filter]
            
            # Apply pagination
            total_count = len(all_tasks)
            paginated_tasks = all_tasks[offset:offset + limit]
            
            # Serialize tasks
            serialized_tasks = [self._serialize_task(task) for task in paginated_tasks]
            
            return APIResponse(
                success=True,
                data={
                    "tasks": serialized_tasks,
                    "total_count": total_count,
                    "limit": limit,
                    "offset": offset
                }
            )
            
        except Exception as e:
            logger.error(f"Get tasks error: {e}")
            return APIResponse(
                success=False,
                error="Internal server error",
                error_code="INTERNAL_ERROR"
            )
    
    def start_task(self, session_token: str, task_id: str) -> APIResponse:
        """Start a specific task."""
        try:
            # Validate session and permissions
            user = self.security.validate_session(session_token)
            if not user:
                return APIResponse(
                    success=False,
                    error="Invalid or expired session",
                    error_code="SESSION_INVALID"
                )
            
            if not self.security.check_permission(session_token, Permission.START_TASKS):
                return APIResponse(
                    success=False,
                    error="Insufficient permissions",
                    error_code="PERMISSION_DENIED"
                )
            
            # Start task
            success = self.scheduler.start_task(task_id)
            
            if success:
                task = self.scheduler.get_task(task_id)
                return APIResponse(
                    success=True,
                    data={
                        "task_id": task_id,
                        "task": self._serialize_task(task) if task else None
                    }
                )
            else:
                return APIResponse(
                    success=False,
                    error="Failed to start task",
                    error_code="TASK_START_FAILED"
                )
                
        except Exception as e:
            logger.error(f"Start task error: {e}")
            return APIResponse(
                success=False,
                error="Internal server error",
                error_code="INTERNAL_ERROR"
            )
    
    def complete_task(self, session_token: str, task_id: str) -> APIResponse:
        """Complete a specific task."""
        try:
            # Validate session and permissions
            user = self.security.validate_session(session_token)
            if not user:
                return APIResponse(
                    success=False,
                    error="Invalid or expired session",
                    error_code="SESSION_INVALID"
                )
            
            if not self.security.check_permission(session_token, Permission.MODIFY_TASKS):
                return APIResponse(
                    success=False,
                    error="Insufficient permissions",
                    error_code="PERMISSION_DENIED"
                )
            
            # Complete task with learning
            success = self.scheduler.complete_task_with_learning(task_id)
            
            if success:
                task = self.scheduler.get_task(task_id)
                return APIResponse(
                    success=True,
                    data={
                        "task_id": task_id,
                        "task": self._serialize_task(task) if task else None
                    }
                )
            else:
                return APIResponse(
                    success=False,
                    error="Failed to complete task",
                    error_code="TASK_COMPLETE_FAILED"
                )
                
        except Exception as e:
            logger.error(f"Complete task error: {e}")
            return APIResponse(
                success=False,
                error="Internal server error",
                error_code="INTERNAL_ERROR"
            )
    
    def get_next_tasks(self, session_token: str, count: int = 5) -> APIResponse:
        """Get next recommended tasks."""
        try:
            # Validate session and permissions
            user = self.security.validate_session(session_token)
            if not user:
                return APIResponse(
                    success=False,
                    error="Invalid or expired session",
                    error_code="SESSION_INVALID"
                )
            
            if not self.security.check_permission(session_token, Permission.READ_TASKS):
                return APIResponse(
                    success=False,
                    error="Insufficient permissions",
                    error_code="PERMISSION_DENIED"
                )
            
            # Get next tasks using ML algorithm
            next_tasks = self.scheduler.get_intelligent_next_tasks()[:count]
            
            # Serialize tasks with scores
            serialized_tasks = []
            for task in next_tasks:
                task_data = self._serialize_task(task)
                task_data['ml_score'] = self.scheduler._calculate_ml_priority_score(task)
                task_data['success_probability'] = getattr(task, 'predicted_success_probability', 0.8)
                serialized_tasks.append(task_data)
            
            return APIResponse(
                success=True,
                data={
                    "recommended_tasks": serialized_tasks,
                    "count": len(serialized_tasks)
                }
            )
            
        except Exception as e:
            logger.error(f"Get next tasks error: {e}")
            return APIResponse(
                success=False,
                error="Internal server error",
                error_code="INTERNAL_ERROR"
            )
    
    def get_scheduler_insights(self, session_token: str) -> APIResponse:
        """Get ML scheduling insights."""
        try:
            # Validate session and permissions
            user = self.security.validate_session(session_token)
            if not user:
                return APIResponse(
                    success=False,
                    error="Invalid or expired session",
                    error_code="SESSION_INVALID"
                )
            
            if not self.security.check_permission(session_token, Permission.VIEW_METRICS):
                return APIResponse(
                    success=False,
                    error="Insufficient permissions",
                    error_code="PERMISSION_DENIED"
                )
            
            # Get insights
            insights = self.scheduler.get_scheduling_insights()
            
            return APIResponse(
                success=True,
                data=insights
            )
            
        except Exception as e:
            logger.error(f"Get insights error: {e}")
            return APIResponse(
                success=False,
                error="Internal server error",
                error_code="INTERNAL_ERROR"
            )
    
    def get_health_status(self, session_token: str) -> APIResponse:
        """Get system health status."""
        try:
            # Validate session and permissions
            user = self.security.validate_session(session_token)
            if not user:
                return APIResponse(
                    success=False,
                    error="Invalid or expired session",
                    error_code="SESSION_INVALID"
                )
            
            if not self.security.check_permission(session_token, Permission.VIEW_METRICS):
                return APIResponse(
                    success=False,
                    error="Insufficient permissions",
                    error_code="PERMISSION_DENIED"
                )
            
            # Get health status
            health = self.health_monitor.get_current_health()
            
            health_data = {
                "status": health.status,
                "score": health.score,
                "alerts": health.alerts,
                "last_updated": health.last_updated.isoformat(),
                "metrics": {
                    name: {
                        "value": metric.value,
                        "unit": metric.unit,
                        "status": metric.status,
                        "timestamp": metric.timestamp.isoformat()
                    }
                    for name, metric in health.metrics.items()
                }
            }
            
            return APIResponse(
                success=True,
                data=health_data
            )
            
        except Exception as e:
            logger.error(f"Get health status error: {e}")
            return APIResponse(
                success=False,
                error="Internal server error",
                error_code="INTERNAL_ERROR"
            )
    
    def export_data(self, session_token: str, export_type: str = "tasks") -> APIResponse:
        """Export system data securely."""
        try:
            # Validate session and permissions
            user = self.security.validate_session(session_token)
            if not user:
                return APIResponse(
                    success=False,
                    error="Invalid or expired session",
                    error_code="SESSION_INVALID"
                )
            
            # Check export permissions
            if not self.security.secure_data_export(session_token, export_type):
                return APIResponse(
                    success=False,
                    error="Insufficient permissions for data export",
                    error_code="EXPORT_PERMISSION_DENIED"
                )
            
            # Export data based on type
            if export_type == "tasks":
                tasks_data = [self._serialize_task(task) for task in self.scheduler.tasks.values()]
                return APIResponse(
                    success=True,
                    data={
                        "export_type": export_type,
                        "tasks": tasks_data,
                        "count": len(tasks_data),
                        "exported_at": datetime.now().isoformat()
                    }
                )
            
            elif export_type == "insights":
                insights = self.scheduler.get_scheduling_insights()
                return APIResponse(
                    success=True,
                    data={
                        "export_type": export_type,
                        "insights": insights,
                        "exported_at": datetime.now().isoformat()
                    }
                )
            
            elif export_type == "health":
                health_report = self.health_monitor.export_health_report()
                return APIResponse(
                    success=True,
                    data={
                        "export_type": export_type,
                        "health_report": health_report,
                        "exported_at": datetime.now().isoformat()
                    }
                )
            
            else:
                return APIResponse(
                    success=False,
                    error="Invalid export type",
                    error_code="INVALID_EXPORT_TYPE"
                )
                
        except Exception as e:
            logger.error(f"Export data error: {e}")
            return APIResponse(
                success=False,
                error="Internal server error",
                error_code="INTERNAL_ERROR"
            )
    
    def get_security_report(self, session_token: str) -> APIResponse:
        """Get security audit report."""
        try:
            # Validate session and permissions
            user = self.security.validate_session(session_token)
            if not user:
                return APIResponse(
                    success=False,
                    error="Invalid or expired session",
                    error_code="SESSION_INVALID"
                )
            
            if not self.security.check_permission(session_token, Permission.ADMIN_ACCESS):
                return APIResponse(
                    success=False,
                    error="Insufficient permissions",
                    error_code="PERMISSION_DENIED"
                )
            
            # Get security report
            security_report = self.security.get_security_report()
            
            return APIResponse(
                success=True,
                data=security_report
            )
            
        except Exception as e:
            logger.error(f"Get security report error: {e}")
            return APIResponse(
                success=False,
                error="Internal server error",
                error_code="INTERNAL_ERROR"
            )
    
    def _serialize_task(self, task) -> Dict[str, Any]:
        """Serialize task object for API response."""
        return {
            "id": task.id,
            "name": task.name,
            "description": task.description,
            "priority": task.priority.name,
            "status": task.status.value,
            "dependencies": list(task.dependencies),
            "estimated_duration": task.estimated_duration.total_seconds(),
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "resource_requirements": task.resource_requirements,
            "entangled_tasks": list(task.entangled_tasks),
            "superposition_weight": task.superposition_weight,
            "tags": list(getattr(task, 'tags', [])),
            "complexity_score": getattr(task, 'complexity_score', 1.0),
            "predicted_success_probability": getattr(task, 'predicted_success_probability', 0.8)
        }
    
    def shutdown(self) -> None:
        """Graceful shutdown of API services."""
        try:
            self.health_monitor.stop_monitoring()
            self.security.cleanup_expired_sessions()
            logger.info("API services shut down gracefully")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Demo CLI for testing the robust API
def demo_robust_api():
    """Demonstrate robust API functionality."""
    print("🛡️  Robust API Demo")
    print("-" * 30)
    
    api = QuantumSchedulerRobustAPI()
    
    try:
        # Authenticate
        print("🔐 Authenticating...")
        auth_response = api.authenticate("admin", "AdminPass123!")
        print(f"Auth Response: {auth_response.success}")
        
        if not auth_response.success:
            print(f"Authentication failed: {auth_response.error}")
            return
        
        session_token = auth_response.data["session_token"]
        
        # Create tasks
        print("\n📝 Creating tasks...")
        task_data = {
            "name": "ML Model Training",
            "description": "Train deep learning model for image recognition",
            "priority": "high",
            "tags": ["ml", "training"]
        }
        
        create_response = api.create_task(session_token, task_data)
        print(f"Create Task: {create_response.success}")
        
        # Get tasks
        print("\n📋 Getting tasks...")
        tasks_response = api.get_tasks(session_token)
        print(f"Get Tasks: {tasks_response.success}, Count: {len(tasks_response.data['tasks']) if tasks_response.success else 0}")
        
        # Get next recommended tasks
        print("\n🎯 Getting recommendations...")
        next_response = api.get_next_tasks(session_token)
        print(f"Next Tasks: {next_response.success}, Count: {next_response.data['count'] if next_response.success else 0}")
        
        # Get health status
        print("\n🏥 Getting health status...")
        health_response = api.get_health_status(session_token)
        print(f"Health Status: {health_response.success}")
        if health_response.success:
            print(f"  System Status: {health_response.data['status']}")
            print(f"  Health Score: {health_response.data['score']:.1f}")
        
        # Get insights
        print("\n🧠 Getting ML insights...")
        insights_response = api.get_scheduler_insights(session_token)
        print(f"Insights: {insights_response.success}")
        
        print("\n✅ Robust API Demo Completed Successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        api.shutdown()


if __name__ == "__main__":
    demo_robust_api()