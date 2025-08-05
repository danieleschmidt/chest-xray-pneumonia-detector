"""Tests for quantum task planner validation system."""

import pytest
from datetime import timedelta
from unittest.mock import patch

from src.quantum_inspired_task_planner.validation import (
    TaskValidator, ScheduleValidator, SecurityValidator, ResourceValidator,
    ValidationResult, validate_and_log
)
from src.quantum_inspired_task_planner.quantum_scheduler import (
    QuantumTask, TaskPriority, TaskStatus
)


class TestValidationResult:
    """Test ValidationResult functionality."""
    
    def test_validation_result_creation(self):
        """Test validation result creation."""
        result = ValidationResult(
            is_valid=True,
            errors=["error1", "error2"],
            warnings=["warning1"]
        )
        
        assert result.is_valid is True
        assert result.errors == ["error1", "error2"]
        assert result.warnings == ["warning1"]


class TestTaskValidator:
    """Test task validation functionality."""
    
    def test_valid_task_creation(self):
        """Test validation of valid task parameters."""
        result = TaskValidator.validate_task_creation(
            name="Valid Task Name",
            description="A valid task description",
            priority="high",
            dependencies=["12345678-1234-1234-1234-123456789012"],
            estimated_duration=timedelta(hours=2),
            resource_requirements={"cpu": 4.0, "memory": 8.0}
        )
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_invalid_task_name(self):
        """Test validation of invalid task names."""
        # Empty name
        result = TaskValidator.validate_task_creation(name="")
        assert result.is_valid is False
        assert any("empty" in error.lower() for error in result.errors)
        
        # Too long name
        long_name = "x" * 300
        result = TaskValidator.validate_task_creation(name=long_name)
        assert result.is_valid is False
        assert any("maximum length" in error for error in result.errors)
        
        # Potentially unsafe characters
        result = TaskValidator.validate_task_creation(name="SELECT * FROM tasks;")
        assert result.is_valid is False
        assert any("unsafe" in error.lower() for error in result.errors)
    
    def test_invalid_priority(self):
        """Test validation of invalid priorities."""
        result = TaskValidator.validate_task_creation(
            name="Test Task",
            priority="invalid_priority"
        )
        
        assert result.is_valid is False
        assert any("invalid priority" in error.lower() for error in result.errors)
    
    def test_invalid_dependencies(self):
        """Test validation of invalid dependencies."""
        # Too many dependencies
        many_deps = [f"dep_{i}" for i in range(60)]  # Exceeds MAX_DEPENDENCIES
        result = TaskValidator.validate_task_creation(
            name="Test Task",
            dependencies=many_deps
        )
        
        assert result.is_valid is False
        assert any("too many dependencies" in error.lower() for error in result.errors)
        
        # Invalid dependency ID format
        result = TaskValidator.validate_task_creation(
            name="Test Task",
            dependencies=["invalid_id_format"]
        )
        
        assert result.is_valid is False
        assert any("invalid dependency id format" in error.lower() for error in result.errors)
        
        # Duplicate dependencies (should be warning, not error)
        result = TaskValidator.validate_task_creation(
            name="Test Task",
            dependencies=["12345678-1234-1234-1234-123456789012", "12345678-1234-1234-1234-123456789012"]
        )
        
        assert any("duplicate" in warning.lower() for warning in result.warnings)
    
    def test_invalid_duration(self):
        """Test validation of invalid durations."""
        # Too short duration
        result = TaskValidator.validate_task_creation(
            name="Test Task",
            estimated_duration=timedelta(seconds=30)  # Less than 1 minute
        )
        
        assert result.is_valid is False
        assert any("too short" in error.lower() for error in result.errors)
        
        # Too long duration
        result = TaskValidator.validate_task_creation(
            name="Test Task",
            estimated_duration=timedelta(days=40)  # Exceeds max
        )
        
        assert result.is_valid is False
        assert any("too long" in error.lower() for error in result.errors)
        
        # Very long duration (warning)
        result = TaskValidator.validate_task_creation(
            name="Test Task",
            estimated_duration=timedelta(hours=12)
        )
        
        assert any("8 hours" in warning for warning in result.warnings)
    
    def test_invalid_resource_requirements(self):
        """Test validation of invalid resource requirements."""
        # Negative resource amount
        result = TaskValidator.validate_task_creation(
            name="Test Task",
            resource_requirements={"cpu": -5.0}
        )
        
        assert result.is_valid is False
        assert any("must be positive" in error for error in result.errors)
        
        # Invalid resource type
        result = TaskValidator.validate_task_creation(
            name="Test Task",
            resource_requirements={"invalid_resource": 5.0}
        )
        
        assert result.is_valid is False
        assert any("invalid resource type" in error.lower() for error in result.errors)
        
        # Very high resource requirement (warning)
        result = TaskValidator.validate_task_creation(
            name="Test Task",
            resource_requirements={"cpu": 15000.0}
        )
        
        assert any("very high" in warning.lower() for warning in result.warnings)


class TestScheduleValidator:
    """Test schedule validation functionality."""
    
    def create_test_tasks(self):
        """Create test tasks for schedule validation."""
        tasks = {}
        
        task1 = QuantumTask(id="task_1", name="Task 1", status=TaskStatus.PENDING)
        task2 = QuantumTask(id="task_2", name="Task 2", status=TaskStatus.PENDING,
                          dependencies={"task_1"})
        task3 = QuantumTask(id="task_3", name="Task 3", status=TaskStatus.RUNNING,
                          resource_requirements={"cpu": 100.0})
        
        tasks["task_1"] = task1
        tasks["task_2"] = task2
        tasks["task_3"] = task3
        
        return tasks
    
    def test_valid_schedule_integrity(self):
        """Test validation of valid schedule."""
        tasks = self.create_test_tasks()
        
        result = ScheduleValidator.validate_schedule_integrity(tasks)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        tasks = {}
        
        # Create circular dependency: task1 -> task2 -> task3 -> task1
        task1 = QuantumTask(id="task_1", name="Task 1", dependencies={"task_3"})
        task2 = QuantumTask(id="task_2", name="Task 2", dependencies={"task_1"})
        task3 = QuantumTask(id="task_3", name="Task 3", dependencies={"task_2"})
        
        tasks["task_1"] = task1
        tasks["task_2"] = task2
        tasks["task_3"] = task3
        
        result = ScheduleValidator.validate_schedule_integrity(tasks)
        
        assert result.is_valid is False
        assert any("circular dependency" in error.lower() for error in result.errors)
    
    def test_orphaned_dependency_detection(self):
        """Test detection of orphaned dependencies."""
        tasks = {}
        
        # Task with dependency on non-existent task
        task1 = QuantumTask(id="task_1", name="Task 1", dependencies={"non_existent_task"})
        tasks["task_1"] = task1
        
        result = ScheduleValidator.validate_schedule_integrity(tasks)
        
        # Should have warning about orphaned dependency
        assert any("non-existent dependency" in warning.lower() for warning in result.warnings)
    
    def test_resource_constraint_validation(self):
        """Test resource constraint validation."""
        tasks = {}
        
        # Create tasks that exceed resource limits
        task1 = QuantumTask(id="task_1", name="Task 1", status=TaskStatus.RUNNING,
                          resource_requirements={"cpu": 2000.0})  # Exceeds limit
        task2 = QuantumTask(id="task_2", name="Task 2", status=TaskStatus.RUNNING,
                          resource_requirements={"memory": 20000.0})  # Exceeds limit
        
        tasks["task_1"] = task1
        tasks["task_2"] = task2
        
        result = ScheduleValidator.validate_schedule_integrity(tasks)
        
        assert result.is_valid is False
        assert any("exceed system limit" in error for error in result.errors)
    
    def test_scheduling_anomaly_detection(self):
        """Test detection of scheduling anomalies."""
        tasks = {}
        
        # Create very old pending task
        import datetime
        old_date = datetime.datetime.now() - datetime.timedelta(days=10)
        
        with patch.object(QuantumTask, 'created_at', old_date):
            old_task = QuantumTask(id="task_1", name="Old Task", status=TaskStatus.PENDING)
            old_task.created_at = old_date
        
        tasks["task_1"] = old_task
        
        # Add many high-priority tasks
        for i in range(15):
            task = QuantumTask(id=f"high_task_{i}", name=f"High Task {i}",
                             priority=TaskPriority.HIGH, status=TaskStatus.PENDING)
            tasks[f"high_task_{i}"] = task
        
        result = ScheduleValidator.validate_schedule_integrity(tasks)
        
        # Should detect anomalies
        assert any("pending for" in warning for warning in result.warnings)
        assert any("many high-priority tasks" in warning.lower() for warning in result.warnings)


class TestSecurityValidator:
    """Test security validation functionality."""
    
    def test_valid_input_security(self):
        """Test validation of safe inputs."""
        safe_inputs = [
            "normal task name",
            {"name": "task", "priority": "high"},
            ["task_1", "task_2"],
            123.45
        ]
        
        for input_data in safe_inputs:
            result = SecurityValidator.validate_input_security(input_data)
            assert result.is_valid is True
    
    def test_suspicious_pattern_detection(self):
        """Test detection of suspicious patterns."""
        suspicious_inputs = [
            "eval('malicious code')",
            "import os; os.system('rm -rf /')",
            "SELECT * FROM users WHERE password='secret'",
            "rm -rf /important/data"
        ]
        
        for input_data in suspicious_inputs:
            result = SecurityValidator.validate_input_security(input_data)
            # Should have warnings for suspicious patterns
            assert len(result.warnings) > 0
    
    def test_large_input_detection(self):
        """Test detection of oversized inputs."""
        # Very large input (potential DoS)
        large_input = "x" * 200000  # 200KB
        
        result = SecurityValidator.validate_input_security(large_input)
        
        assert result.is_valid is False
        assert any("too large" in error.lower() for error in result.errors)
    
    def test_null_byte_detection(self):
        """Test detection of null bytes in input."""
        input_with_null = "normal text\x00hidden content"
        
        result = SecurityValidator.validate_input_security(input_with_null)
        
        assert result.is_valid is False
        assert any("null bytes" in error.lower() for error in result.errors)
    
    def test_quantum_parameter_validation(self):
        """Test validation of quantum algorithm parameters."""
        # Valid parameters
        valid_params = {
            "temperature": 50.0,
            "max_iterations": 1000,
            "cooling_rate": 0.95
        }
        
        result = SecurityValidator.validate_quantum_parameters(valid_params)
        assert result.is_valid is True
        
        # Invalid temperature
        invalid_params = {"temperature": -10.0}
        result = SecurityValidator.validate_quantum_parameters(invalid_params)
        assert result.is_valid is False
        
        # Invalid iteration count
        invalid_params = {"max_iterations": "not_a_number"}
        result = SecurityValidator.validate_quantum_parameters(invalid_params)
        assert result.is_valid is False
        
        # Invalid cooling rate
        invalid_params = {"cooling_rate": 1.5}
        result = SecurityValidator.validate_quantum_parameters(invalid_params)
        assert result.is_valid is False


class TestResourceValidator:
    """Test resource validation functionality."""
    
    def test_valid_resource_allocation(self):
        """Test validation of valid resource allocation."""
        result = ResourceValidator.validate_resource_allocation(
            resource_id="cpu_pool_1",
            resource_type="cpu",
            capacity=100.0,
            current_allocation=25.0
        )
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_invalid_resource_capacity(self):
        """Test validation of invalid resource capacity."""
        # Zero capacity
        result = ResourceValidator.validate_resource_allocation(
            resource_id="cpu_pool",
            resource_type="cpu",
            capacity=0.0,
            current_allocation=0.0
        )
        
        assert result.is_valid is False
        assert any("must be positive" in error for error in result.errors)
        
        # Negative capacity
        result = ResourceValidator.validate_resource_allocation(
            resource_id="cpu_pool",
            resource_type="cpu",
            capacity=-50.0,
            current_allocation=0.0
        )
        
        assert result.is_valid is False
        assert any("must be positive" in error for error in result.errors)
    
    def test_allocation_exceeds_capacity(self):
        """Test detection when allocation exceeds capacity."""
        result = ResourceValidator.validate_resource_allocation(
            resource_id="cpu_pool",
            resource_type="cpu",
            capacity=100.0,
            current_allocation=150.0  # Exceeds capacity
        )
        
        assert result.is_valid is False
        assert any("exceeds capacity" in error for error in result.errors)
    
    def test_invalid_resource_id(self):
        """Test validation of invalid resource IDs."""
        # Too short
        result = ResourceValidator.validate_resource_allocation(
            resource_id="x",
            resource_type="cpu",
            capacity=100.0,
            current_allocation=0.0
        )
        
        assert result.is_valid is False
        assert any("at least 2 characters" in error for error in result.errors)
        
        # Too long
        long_id = "x" * 60
        result = ResourceValidator.validate_resource_allocation(
            resource_id=long_id,
            resource_type="cpu",
            capacity=100.0,
            current_allocation=0.0
        )
        
        assert result.is_valid is False
        assert any("too long" in error for error in result.errors)
        
        # Invalid characters
        result = ResourceValidator.validate_resource_allocation(
            resource_id="cpu@pool#1",
            resource_type="cpu",
            capacity=100.0,
            current_allocation=0.0
        )
        
        assert result.is_valid is False
        assert any("can only contain" in error for error in result.errors)
    
    def test_allocation_request_validation(self):
        """Test validation of allocation requests."""
        # Valid request
        requirements = {"cpu": 20.0, "memory": 500.0}
        available = {"cpu": 100.0, "memory": 1000.0}
        
        result = ResourceValidator.validate_allocation_request(requirements, available)
        assert result.is_valid is True
        
        # Insufficient resources
        requirements = {"cpu": 150.0}
        available = {"cpu": 100.0}
        
        result = ResourceValidator.validate_allocation_request(requirements, available)
        assert result.is_valid is False
        assert any("insufficient" in error.lower() for error in result.errors)
        
        # High utilization warning
        requirements = {"cpu": 90.0}
        available = {"cpu": 100.0}
        
        result = ResourceValidator.validate_allocation_request(requirements, available)
        assert any("high" in warning.lower() for warning in result.warnings)


class TestValidateAndLog:
    """Test validation logging utility."""
    
    def test_successful_validation_logging(self):
        """Test logging of successful validation."""
        result = ValidationResult(is_valid=True, errors=[], warnings=["minor warning"])
        
        with patch('src.quantum_inspired_task_planner.validation.logger') as mock_logger:
            is_valid = validate_and_log(result, "test_operation")
            
            assert is_valid is True
            mock_logger.warning.assert_called_once()
            mock_logger.error.assert_not_called()
    
    def test_failed_validation_logging(self):
        """Test logging of failed validation."""
        result = ValidationResult(
            is_valid=False, 
            errors=["critical error"], 
            warnings=["warning"]
        )
        
        with patch('src.quantum_inspired_task_planner.validation.logger') as mock_logger:
            is_valid = validate_and_log(result, "test_operation")
            
            assert is_valid is False
            mock_logger.error.assert_called_once()
            mock_logger.warning.assert_called_once()
    
    def test_custom_logger_usage(self):
        """Test using custom logger instance."""
        custom_logger = Mock()
        result = ValidationResult(is_valid=False, errors=["error"], warnings=[])
        
        is_valid = validate_and_log(result, "test_operation", custom_logger)
        
        assert is_valid is False
        custom_logger.error.assert_called_once()


class TestValidationIntegration:
    """Integration tests for validation system."""
    
    def test_full_task_validation_pipeline(self):
        """Test complete task validation pipeline."""
        # Test with realistic task parameters
        result = TaskValidator.validate_task_creation(
            name="Deploy Web Application",
            description="Deploy the web application to production environment with monitoring",
            priority="high",
            dependencies=["12345678-1234-1234-1234-123456789012", "87654321-4321-4321-4321-210987654321"],
            estimated_duration=timedelta(hours=4),
            resource_requirements={"cpu": 8.0, "memory": 16.0, "storage": 50.0}
        )
        
        assert result.is_valid is True
        
        # Log the validation
        with patch('src.quantum_inspired_task_planner.validation.logger') as mock_logger:
            validate_and_log(result, "task_creation")
            
            # Should only have warnings if any
            if result.warnings:
                mock_logger.warning.assert_called()
            mock_logger.error.assert_not_called()
    
    def test_schedule_validation_with_realistic_tasks(self):
        """Test schedule validation with realistic task setup."""
        # Create realistic task scenario
        tasks = {}
        
        # Main application deployment task
        main_task = QuantumTask(
            id="main_deploy",
            name="Main Application Deployment",
            status=TaskStatus.PENDING,
            priority=TaskPriority.HIGH,
            resource_requirements={"cpu": 50.0, "memory": 1000.0}
        )
        
        # Database migration task (dependency of main)
        db_task = QuantumTask(
            id="db_migration",
            name="Database Migration",
            status=TaskStatus.COMPLETED,
            priority=TaskPriority.CRITICAL,
            resource_requirements={"cpu": 20.0, "memory": 500.0}
        )
        
        # Frontend build task (dependency of main)
        frontend_task = QuantumTask(
            id="frontend_build",
            name="Frontend Build",
            status=TaskStatus.RUNNING,
            priority=TaskPriority.HIGH,
            resource_requirements={"cpu": 30.0, "memory": 200.0}
        )
        
        # Set up dependencies
        main_task.dependencies = {"db_migration", "frontend_build"}
        
        tasks["main_deploy"] = main_task
        tasks["db_migration"] = db_task
        tasks["frontend_build"] = frontend_task
        
        result = ScheduleValidator.validate_schedule_integrity(tasks)
        
        # Should be valid realistic scenario
        assert result.is_valid is True
    
    def test_security_validation_integration(self):
        """Test security validation in realistic scenarios."""
        # Test task names that might come from user input
        potentially_malicious_names = [
            "Normal Task Name",  # Safe
            "Deploy <script>alert('xss')</script>",  # XSS attempt
            "'; DROP TABLE tasks; --",  # SQL injection attempt
            "$(rm -rf /)"  # Command injection attempt
        ]
        
        for name in potentially_malicious_names:
            security_result = SecurityValidator.validate_input_security(name)
            
            if "Normal Task Name" in name:
                assert security_result.is_valid is True
            else:
                # Should detect security issues
                assert len(security_result.warnings) > 0 or len(security_result.errors) > 0
    
    def test_comprehensive_validation_workflow(self):
        """Test complete validation workflow for task planner operations."""
        # Simulate complete task creation workflow
        task_name = "Critical System Maintenance"
        description = "Perform critical system maintenance including security updates"
        priority = "critical"
        dependencies = ["12345678-1234-1234-1234-123456789012"]
        duration = timedelta(hours=6)
        resources = {"cpu": 40.0, "memory": 800.0, "storage": 100.0}
        
        # 1. Validate task creation
        task_result = TaskValidator.validate_task_creation(
            name=task_name,
            description=description,
            priority=priority,
            dependencies=dependencies,
            estimated_duration=duration,
            resource_requirements=resources
        )
        
        # 2. Validate security
        security_result = SecurityValidator.validate_input_security({
            "name": task_name,
            "description": description,
            "dependencies": dependencies
        })
        
        # 3. Validate resource allocation
        available_resources = {"cpu": 100.0, "memory": 1000.0, "storage": 500.0}
        resource_result = ResourceValidator.validate_allocation_request(
            resources, available_resources
        )
        
        # All validations should pass for this realistic scenario
        assert task_result.is_valid is True
        assert security_result.is_valid is True
        assert resource_result.is_valid is True
        
        # May have warnings but no errors
        total_errors = (len(task_result.errors) + 
                       len(security_result.errors) + 
                       len(resource_result.errors))
        assert total_errors == 0