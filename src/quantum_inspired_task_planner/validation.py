"""Validation utilities for quantum task planner.

Provides comprehensive input validation, constraint checking,
and data integrity verification for quantum scheduling operations.
"""

import re
from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from .quantum_scheduler import QuantumTask, TaskPriority, TaskStatus

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class TaskValidator:
    """Validates task creation and modification operations."""
    
    # Validation constraints
    MAX_TASK_NAME_LENGTH = 200
    MAX_DESCRIPTION_LENGTH = 2000
    MAX_DEPENDENCIES = 50
    MIN_DURATION_MINUTES = 1
    MAX_DURATION_HOURS = 720  # 30 days
    
    # Valid resource types
    VALID_RESOURCE_TYPES = {"cpu", "memory", "gpu", "storage", "network"}
    
    @classmethod
    def validate_task_creation(cls, name: str, description: str = "",
                             priority: str = "medium", dependencies: Optional[List[str]] = None,
                             estimated_duration: Optional[timedelta] = None,
                             resource_requirements: Optional[Dict[str, float]] = None) -> ValidationResult:
        """Validate task creation parameters."""
        errors = []
        warnings = []
        
        # Validate name
        name_result = cls._validate_task_name(name)
        errors.extend(name_result.errors)
        warnings.extend(name_result.warnings)
        
        # Validate description
        desc_result = cls._validate_description(description)
        errors.extend(desc_result.errors)
        warnings.extend(desc_result.warnings)
        
        # Validate priority
        priority_result = cls._validate_priority(priority)
        errors.extend(priority_result.errors)
        
        # Validate dependencies
        if dependencies:
            dep_result = cls._validate_dependencies(dependencies)
            errors.extend(dep_result.errors)
            warnings.extend(dep_result.warnings)
        
        # Validate duration
        if estimated_duration:
            duration_result = cls._validate_duration(estimated_duration)
            errors.extend(duration_result.errors)
            warnings.extend(duration_result.warnings)
        
        # Validate resource requirements
        if resource_requirements:
            resource_result = cls._validate_resource_requirements(resource_requirements)
            errors.extend(resource_result.errors)
            warnings.extend(resource_result.warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    @classmethod
    def _validate_task_name(cls, name: str) -> ValidationResult:
        """Validate task name."""
        errors = []
        warnings = []
        
        if not name or not name.strip():
            errors.append("Task name cannot be empty")
        elif len(name.strip()) > cls.MAX_TASK_NAME_LENGTH:
            errors.append(f"Task name exceeds maximum length of {cls.MAX_TASK_NAME_LENGTH} characters")
        
        # Check for potentially problematic characters
        if re.search(r'[<>"\']', name):
            warnings.append("Task name contains special characters that may cause display issues")
        
        # Check for SQL injection patterns (defensive)
        sql_patterns = [r'\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b', r'[;\'"]']
        for pattern in sql_patterns:
            if re.search(pattern, name, re.IGNORECASE):
                errors.append("Task name contains potentially unsafe characters")
                break
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    @classmethod
    def _validate_description(cls, description: str) -> ValidationResult:
        """Validate task description."""
        errors = []
        warnings = []
        
        if description and len(description) > cls.MAX_DESCRIPTION_LENGTH:
            errors.append(f"Description exceeds maximum length of {cls.MAX_DESCRIPTION_LENGTH} characters")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    @classmethod
    def _validate_priority(cls, priority: str) -> ValidationResult:
        """Validate task priority."""
        errors = []
        
        valid_priorities = {"low", "medium", "high", "critical"}
        if priority.lower() not in valid_priorities:
            errors.append(f"Invalid priority '{priority}'. Must be one of: {', '.join(valid_priorities)}")
        
        return ValidationResult(len(errors) == 0, errors, [])
    
    @classmethod
    def _validate_dependencies(cls, dependencies: List[str]) -> ValidationResult:
        """Validate task dependencies."""
        errors = []
        warnings = []
        
        if len(dependencies) > cls.MAX_DEPENDENCIES:
            errors.append(f"Too many dependencies. Maximum allowed: {cls.MAX_DEPENDENCIES}")
        
        # Check for duplicate dependencies
        unique_deps = set(dependencies)
        if len(unique_deps) != len(dependencies):
            warnings.append("Duplicate dependencies detected and will be removed")
        
        # Validate dependency ID format (UUID-like)
        uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
        for dep_id in dependencies:
            if not uuid_pattern.match(dep_id):
                errors.append(f"Invalid dependency ID format: {dep_id}")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    @classmethod
    def _validate_duration(cls, duration: timedelta) -> ValidationResult:
        """Validate estimated task duration."""
        errors = []
        warnings = []
        
        total_minutes = duration.total_seconds() / 60
        
        if total_minutes < cls.MIN_DURATION_MINUTES:
            errors.append(f"Duration too short. Minimum: {cls.MIN_DURATION_MINUTES} minutes")
        
        max_minutes = cls.MAX_DURATION_HOURS * 60
        if total_minutes > max_minutes:
            errors.append(f"Duration too long. Maximum: {cls.MAX_DURATION_HOURS} hours")
        
        # Warn about very long durations
        if total_minutes > 480:  # 8 hours
            warnings.append("Task duration exceeds 8 hours - consider breaking into smaller tasks")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    @classmethod
    def _validate_resource_requirements(cls, requirements: Dict[str, float]) -> ValidationResult:
        """Validate resource requirements."""
        errors = []
        warnings = []
        
        for resource_type, amount in requirements.items():
            # Validate resource type
            if resource_type not in cls.VALID_RESOURCE_TYPES:
                errors.append(f"Invalid resource type: {resource_type}")
            
            # Validate amount
            if amount <= 0:
                errors.append(f"Resource amount must be positive for {resource_type}")
            elif amount > 10000:
                warnings.append(f"Very high resource requirement for {resource_type}: {amount}")
        
        return ValidationResult(len(errors) == 0, errors, warnings)


class ScheduleValidator:
    """Validates scheduling operations and constraints."""
    
    @classmethod
    def validate_schedule_integrity(cls, tasks: Dict[str, QuantumTask]) -> ValidationResult:
        """Validate overall schedule integrity and detect issues."""
        errors = []
        warnings = []
        
        # Check for circular dependencies
        circular_deps = cls._detect_circular_dependencies(tasks)
        if circular_deps:
            errors.extend([f"Circular dependency detected: {cycle}" for cycle in circular_deps])
        
        # Check for orphaned tasks
        orphaned_tasks = cls._find_orphaned_dependencies(tasks)
        if orphaned_tasks:
            warnings.extend([f"Task {task_id} has non-existent dependency: {dep}" 
                           for task_id, dep in orphaned_tasks])
        
        # Check for resource constraint violations
        resource_violations = cls._check_resource_constraints(tasks)
        errors.extend(resource_violations)
        
        # Check for scheduling anomalies
        anomalies = cls._detect_scheduling_anomalies(tasks)
        warnings.extend(anomalies)
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    @classmethod
    def _detect_circular_dependencies(cls, tasks: Dict[str, QuantumTask]) -> List[str]:
        """Detect circular dependencies in task graph."""
        def dfs_visit(task_id: str, path: Set[str], visited: Set[str]) -> Optional[List[str]]:
            if task_id in path:
                # Found cycle
                cycle_start = list(path).index(task_id) if task_id in path else 0
                return list(path)[cycle_start:] + [task_id]
            
            if task_id in visited:
                return None
            
            visited.add(task_id)
            path.add(task_id)
            
            task = tasks.get(task_id)
            if task:
                for dep_id in task.dependencies:
                    cycle = dfs_visit(dep_id, path.copy(), visited)
                    if cycle:
                        return cycle
            
            path.discard(task_id)
            return None
        
        cycles = []
        visited = set()
        
        for task_id in tasks:
            if task_id not in visited:
                cycle = dfs_visit(task_id, set(), visited)
                if cycle:
                    cycles.append(" -> ".join(cycle))
        
        return cycles
    
    @classmethod
    def _find_orphaned_dependencies(cls, tasks: Dict[str, QuantumTask]) -> List[Tuple[str, str]]:
        """Find dependencies that reference non-existent tasks."""
        orphaned = []
        task_ids = set(tasks.keys())
        
        for task_id, task in tasks.items():
            for dep_id in task.dependencies:
                if dep_id not in task_ids:
                    orphaned.append((task_id, dep_id))
        
        return orphaned
    
    @classmethod
    def _check_resource_constraints(cls, tasks: Dict[str, QuantumTask]) -> List[str]:
        """Check for resource constraint violations."""
        errors = []
        
        # Aggregate resource requirements for running tasks
        total_requirements: Dict[str, float] = {}
        
        for task in tasks.values():
            if task.status == TaskStatus.RUNNING:
                for resource_type, amount in task.resource_requirements.items():
                    total_requirements[resource_type] = total_requirements.get(resource_type, 0) + amount
        
        # Check against reasonable limits
        resource_limits = {
            "cpu": 1000.0,      # 1000 CPU units
            "memory": 10000.0,  # 10GB memory
            "gpu": 16.0,        # 16 GPU units
            "storage": 100000.0, # 100GB storage
            "network": 10000.0   # 10Gbps network
        }
        
        for resource_type, total_used in total_requirements.items():
            limit = resource_limits.get(resource_type, float('inf'))
            if total_used > limit:
                errors.append(f"Total {resource_type} requirements ({total_used}) exceed system limit ({limit})")
        
        return errors
    
    @classmethod
    def _detect_scheduling_anomalies(cls, tasks: Dict[str, QuantumTask]) -> List[str]:
        """Detect potential scheduling anomalies."""
        warnings = []
        
        # Check for very old pending tasks
        now = datetime.now()
        for task in tasks.values():
            if task.status == TaskStatus.PENDING:
                age = now - task.created_at
                if age.days > 7:
                    warnings.append(f"Task '{task.name}' has been pending for {age.days} days")
        
        # Check for too many high-priority tasks
        high_priority_pending = [
            task for task in tasks.values()
            if task.status == TaskStatus.PENDING and task.priority in [TaskPriority.HIGH, TaskPriority.CRITICAL]
        ]
        
        if len(high_priority_pending) > 10:
            warnings.append(f"Many high-priority tasks pending ({len(high_priority_pending)}). Consider task prioritization review.")
        
        # Check for tasks with very long durations
        for task in tasks.values():
            if task.estimated_duration.total_seconds() > 86400 * 7:  # 1 week
                warnings.append(f"Task '{task.name}' has very long estimated duration ({task.estimated_duration})")
        
        return warnings


class ResourceValidator:
    """Validates resource allocation and management operations."""
    
    @classmethod
    def validate_resource_allocation(cls, resource_id: str, resource_type: str,
                                   capacity: float, current_allocation: float) -> ValidationResult:
        """Validate resource allocation request."""
        errors = []
        warnings = []
        
        # Validate capacity
        if capacity <= 0:
            errors.append("Resource capacity must be positive")
        
        if capacity > 100000:  # Very large capacity
            warnings.append(f"Very large resource capacity specified: {capacity}")
        
        # Check allocation vs capacity
        if current_allocation > capacity:
            errors.append(f"Current allocation ({current_allocation}) exceeds capacity ({capacity})")
        
        # Validate resource ID format
        if not resource_id or len(resource_id) < 2:
            errors.append("Resource ID must be at least 2 characters")
        
        if len(resource_id) > 50:
            errors.append("Resource ID too long (max 50 characters)")
        
        # Check for invalid characters in resource ID
        if not re.match(r'^[a-zA-Z0-9_-]+$', resource_id):
            errors.append("Resource ID can only contain letters, numbers, underscores, and hyphens")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    @classmethod
    def validate_allocation_request(cls, task_requirements: Dict[str, float],
                                  available_resources: Dict[str, float]) -> ValidationResult:
        """Validate that allocation request can be satisfied."""
        errors = []
        warnings = []
        
        for resource_type, required_amount in task_requirements.items():
            available_amount = available_resources.get(resource_type, 0.0)
            
            if required_amount > available_amount:
                errors.append(f"Insufficient {resource_type}: required {required_amount}, available {available_amount}")
            
            # Warn about high utilization
            if available_amount > 0 and (required_amount / available_amount) > 0.8:
                warnings.append(f"High {resource_type} utilization: {(required_amount/available_amount)*100:.1f}%")
        
        return ValidationResult(len(errors) == 0, errors, warnings)


class SecurityValidator:
    """Security validation for quantum task planner operations."""
    
    # Patterns that might indicate security issues
    SUSPICIOUS_PATTERNS = [
        r'\b(eval|exec|import os|subprocess|shell)\b',
        r'[<>"\'].*[<>"\']',  # Potential XSS
        r'\b(password|secret|key|token)\s*[:=]\s*["\']',  # Credentials
        r'\b(rm|del|delete|drop|truncate)\b.*-[rf]',  # Destructive commands
    ]
    
    @classmethod
    def validate_input_security(cls, input_data: Any) -> ValidationResult:
        """Validate input data for security issues."""
        errors = []
        warnings = []
        
        # Convert input to string for pattern matching
        if isinstance(input_data, dict):
            input_str = str(input_data)
        elif isinstance(input_data, (list, tuple)):
            input_str = " ".join(str(item) for item in input_data)
        else:
            input_str = str(input_data)
        
        # Check for suspicious patterns
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if re.search(pattern, input_str, re.IGNORECASE):
                warnings.append(f"Potentially suspicious content detected: {pattern}")
        
        # Check for very long inputs (potential DoS)
        if len(input_str) > 100000:  # 100KB
            errors.append("Input data too large - potential denial of service")
        
        # Check for null bytes (potential security issue)
        if '\x00' in input_str:
            errors.append("Null bytes detected in input")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    @classmethod
    def validate_quantum_parameters(cls, parameters: Dict[str, Any]) -> ValidationResult:
        """Validate quantum algorithm parameters for safety."""
        errors = []
        warnings = []
        
        # Check temperature parameters
        if 'temperature' in parameters:
            temp = parameters['temperature']
            if not isinstance(temp, (int, float)) or temp <= 0:
                errors.append("Temperature must be a positive number")
            elif temp > 10000:
                warnings.append(f"Very high temperature value: {temp}")
        
        # Check iteration limits
        if 'max_iterations' in parameters:
            max_iter = parameters['max_iterations']
            if not isinstance(max_iter, int) or max_iter < 1:
                errors.append("Max iterations must be a positive integer")
            elif max_iter > 100000:
                warnings.append(f"Very high iteration count may cause performance issues: {max_iter}")
        
        # Check cooling rate
        if 'cooling_rate' in parameters:
            cooling = parameters['cooling_rate']
            if not isinstance(cooling, (int, float)) or not (0 < cooling < 1):
                errors.append("Cooling rate must be between 0 and 1")
        
        return ValidationResult(len(errors) == 0, errors, warnings)


def validate_and_log(validation_result: ValidationResult, operation: str,
                    logger_instance: Optional[logging.Logger] = None) -> bool:
    """Validate result and log any issues."""
    log = logger_instance or logger
    
    if validation_result.errors:
        log.error(f"Validation failed for {operation}: {', '.join(validation_result.errors)}")
    
    if validation_result.warnings:
        log.warning(f"Validation warnings for {operation}: {', '.join(validation_result.warnings)}")
    
    return validation_result.is_valid