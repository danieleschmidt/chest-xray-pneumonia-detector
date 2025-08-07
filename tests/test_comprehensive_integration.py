"""Comprehensive integration tests for quantum task planner."""

import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from quantum_inspired_task_planner.quantum_scheduler import (
    QuantumScheduler, QuantumTask, TaskPriority, TaskStatus
)
from quantum_inspired_task_planner.advanced_scheduler import QuantumMLScheduler
from quantum_inspired_task_planner.security_framework import (
    QuantumSchedulerSecurity, Permission, SecurityLevel
)
from quantum_inspired_task_planner.performance_optimizer import (
    PerformanceOptimizer, OptimizationStrategy
)
from quantum_inspired_task_planner.simple_health_monitoring import SimpleHealthMonitor


class TestQuantumSchedulerIntegration:
    """Integration tests for quantum scheduler core functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scheduler = QuantumScheduler(max_parallel_tasks=4)
    
    def test_task_lifecycle(self):
        """Test complete task lifecycle."""
        # Create task
        task_id = self.scheduler.create_task(
            name="Integration Test Task",
            priority=TaskPriority.HIGH,
            estimated_duration=timedelta(minutes=30)
        )
        
        assert task_id is not None
        assert len(task_id) > 0
        
        # Retrieve task
        task = self.scheduler.get_task(task_id)
        assert task is not None
        assert task.name == "Integration Test Task"
        assert task.priority == TaskPriority.HIGH
        assert task.status == TaskStatus.PENDING
        
        # Get task statistics
        stats = self.scheduler.get_task_statistics()
        assert stats['pending'] == 1
        assert stats['running'] == 0
        assert stats['completed'] == 0
        
        # Start task
        success = self.scheduler.start_task(task_id)
        assert success is True
        assert task.status == TaskStatus.RUNNING
        
        stats = self.scheduler.get_task_statistics()
        assert stats['pending'] == 0
        assert stats['running'] == 1
        assert stats['completed'] == 0
        
        # Complete task
        success = self.scheduler.complete_task(task_id)
        assert success is True
        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None
        
        stats = self.scheduler.get_task_statistics()
        assert stats['pending'] == 0
        assert stats['running'] == 0
        assert stats['completed'] == 1
    
    def test_task_dependencies(self):
        """Test task dependency handling."""
        # Create dependent tasks
        parent_id = self.scheduler.create_task("Parent Task", priority=TaskPriority.HIGH)
        child_id = self.scheduler.create_task(
            "Child Task", 
            dependencies=[parent_id],
            priority=TaskPriority.MEDIUM
        )
        
        # Child should not be in next tasks due to dependency
        next_tasks = self.scheduler.get_next_tasks()
        next_task_ids = [t.id for t in next_tasks]
        
        assert parent_id in next_task_ids
        assert child_id not in next_task_ids
        
        # Complete parent task
        self.scheduler.start_task(parent_id)
        self.scheduler.complete_task(parent_id)
        
        # Now child should be available
        next_tasks = self.scheduler.get_next_tasks()
        next_task_ids = [t.id for t in next_tasks]
        assert child_id in next_task_ids
    
    def test_parallel_task_limit(self):
        """Test parallel task execution limits."""
        # Create more tasks than parallel limit
        task_ids = []
        for i in range(6):  # More than max_parallel_tasks (4)
            task_id = self.scheduler.create_task(f"Task {i+1}")
            task_ids.append(task_id)
        
        # Start all available tasks
        started_count = 0
        for task_id in task_ids:
            if self.scheduler.start_task(task_id):
                started_count += 1
        
        # Should be limited to max_parallel_tasks
        assert started_count == 4
        assert len(self.scheduler.running_tasks) == 4
    
    def test_quantum_entanglement(self):
        """Test quantum entanglement between tasks."""
        # Create two tasks
        task1_id = self.scheduler.create_task("Task 1")
        task2_id = self.scheduler.create_task("Task 2")
        
        # Create entanglement
        success = self.scheduler.create_entanglement(task1_id, task2_id)
        assert success is True
        
        task1 = self.scheduler.get_task(task1_id)
        task2 = self.scheduler.get_task(task2_id)
        
        assert task2_id in task1.entangled_tasks
        assert task1_id in task2.entangled_tasks
        
        # Complete one task to test entanglement effects
        self.scheduler.start_task(task1_id)
        self.scheduler.complete_task(task1_id)
        
        # Task 2 should have boosted superposition weight
        assert task2.superposition_weight > 1.0
    
    def test_state_export_import(self):
        """Test scheduler state export and import."""
        # Create some tasks
        task_ids = []
        for i in range(3):
            task_id = self.scheduler.create_task(f"Export Task {i+1}")
            task_ids.append(task_id)
        
        # Start one task
        self.scheduler.start_task(task_ids[0])
        
        # Export state
        state_json = self.scheduler.export_state()
        assert state_json is not None
        assert len(state_json) > 0
        
        # Create new scheduler and import state
        new_scheduler = QuantumScheduler()
        new_scheduler.import_state(state_json)
        
        # Verify imported state
        assert len(new_scheduler.tasks) == 3
        assert len(new_scheduler.running_tasks) == 1
        
        for task_id in task_ids:
            imported_task = new_scheduler.get_task(task_id)
            assert imported_task is not None


class TestMLSchedulerIntegration:
    """Integration tests for ML-enhanced scheduler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ml_scheduler = QuantumMLScheduler(max_parallel_tasks=4)
    
    def test_intelligent_task_creation(self):
        """Test intelligent task creation with ML features."""
        task_id = self.ml_scheduler.add_intelligent_task(
            name="ML Test Task",
            description="Machine learning enhanced task creation",
            priority=TaskPriority.HIGH,
            tags=["ml", "test", "integration"]
        )
        
        assert task_id is not None
        
        task = self.ml_scheduler.get_task(task_id)
        assert task is not None
        assert hasattr(task, 'predicted_success_probability')
        assert hasattr(task, 'complexity_score')
        assert hasattr(task, 'tags')
        
        # ML attributes should have reasonable values
        assert 0.1 <= task.predicted_success_probability <= 0.99
        assert task.complexity_score >= 1.0
        assert len(task.tags) >= 3
    
    def test_ml_task_recommendations(self):
        """Test ML-enhanced task recommendations."""
        # Create diverse tasks
        tasks_data = [
            ("High Priority ML Task", TaskPriority.CRITICAL, ["ml", "critical"]),
            ("Medium Priority Task", TaskPriority.MEDIUM, ["standard"]),
            ("Low Priority Task", TaskPriority.LOW, ["cleanup"]),
            ("Complex Analysis Task", TaskPriority.HIGH, ["analysis", "complex"])
        ]
        
        for name, priority, tags in tasks_data:
            self.ml_scheduler.add_intelligent_task(
                name=name, priority=priority, tags=tags
            )
        
        # Get ML recommendations
        next_tasks = self.ml_scheduler.get_intelligent_next_tasks()
        
        assert len(next_tasks) <= 4  # Should respect parallel limit
        
        # Verify scoring and ordering
        for i in range(len(next_tasks) - 1):
            current_score = self.ml_scheduler._calculate_ml_priority_score(next_tasks[i])
            next_score = self.ml_scheduler._calculate_ml_priority_score(next_tasks[i + 1])
            assert current_score >= next_score  # Should be in descending order
    
    def test_adaptive_learning(self):
        """Test adaptive learning from completed tasks."""
        # Create and complete a task
        task_id = self.ml_scheduler.add_intelligent_task(
            name="Learning Test Task",
            description="Task for testing adaptive learning",
            tags=["learning", "test"]
        )
        
        # Record initial parameters
        initial_learning_rate = self.ml_scheduler.adaptive_params.learning_rate
        
        # Start and complete task with learning
        self.ml_scheduler.start_task(task_id)
        success = self.ml_scheduler.complete_task_with_learning(task_id)
        
        assert success is True
        assert len(self.ml_scheduler.scheduling_history) > 0
        
        # Check that pattern model was updated
        assert len(self.ml_scheduler.pattern_recognition_model) > 0
    
    def test_scheduling_insights(self):
        """Test scheduling insights generation."""
        # Create and execute some tasks to generate data
        for i in range(5):
            task_id = self.ml_scheduler.add_intelligent_task(
                name=f"Insight Task {i+1}",
                tags=["insights", "test"]
            )
            
            if i < 3:  # Complete some tasks
                self.ml_scheduler.start_task(task_id)
                self.ml_scheduler.complete_task_with_learning(task_id)
        
        # Get insights
        insights = self.ml_scheduler.get_scheduling_insights()
        
        assert 'performance_metrics' in insights
        assert 'adaptive_parameters' in insights
        assert 'optimization_recommendations' in insights
        
        # Verify metrics structure
        metrics = insights['performance_metrics']
        assert 'priority_satisfaction_rate' in metrics
        
        # Verify parameters structure
        params = insights['adaptive_parameters']
        assert 'learning_rate' in params
        assert 'temperature_adaptation_rate' in params


class TestSecurityIntegration:
    """Integration tests for security framework."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.security = QuantumSchedulerSecurity()
    
    def test_user_lifecycle(self):
        """Test complete user lifecycle."""
        # Create user
        user = self.security.create_user(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            password="TestPass123!",
            permissions=[Permission.CREATE_TASKS, Permission.READ_TASKS],
            security_level=SecurityLevel.INTERNAL
        )
        
        assert user.user_id == "test_user"
        assert user.username == "testuser"
        assert Permission.CREATE_TASKS in user.permissions
        
        # Authenticate user
        session_token = self.security.authenticate_user("testuser", "TestPass123!")
        assert session_token is not None
        assert len(session_token) > 0
        
        # Validate session
        validated_user = self.security.validate_session(session_token)
        assert validated_user is not None
        assert validated_user.user_id == "test_user"
        
        # Check permissions
        has_create = self.security.check_permission(session_token, Permission.CREATE_TASKS)
        has_admin = self.security.check_permission(session_token, Permission.ADMIN_ACCESS)
        
        assert has_create is True
        assert has_admin is False
    
    def test_authentication_failures(self):
        """Test authentication failure handling."""
        # Create user
        self.security.create_user(
            user_id="test_user2",
            username="testuser2",
            email="test2@example.com",
            password="TestPass123!"
        )
        
        # Test wrong password
        session_token = self.security.authenticate_user("testuser2", "wrongpassword")
        assert session_token is None
        
        # Test non-existent user
        session_token = self.security.authenticate_user("nonexistent", "anypassword")
        assert session_token is None
        
        # Check audit logging
        assert len(self.security.audit_logger.events) > 0
        
        # Verify failed login events
        failed_events = [e for e in self.security.audit_logger.events 
                        if e.event_type == 'login_failure']
        assert len(failed_events) >= 2
    
    def test_secure_task_creation(self):
        """Test secure task creation with validation."""
        # Create user with permissions
        user = self.security.create_user(
            user_id="task_creator",
            username="creator",
            email="creator@example.com",
            password="CreatePass123!",
            permissions=[Permission.CREATE_TASKS]
        )
        
        session_token = self.security.authenticate_user("creator", "CreatePass123!")
        assert session_token is not None
        
        # Test valid task creation
        task_data = {
            "name": "Secure Test Task",
            "description": "A task created through secure validation",
            "type": "standard"
        }
        
        validated_data = self.security.secure_task_creation(session_token, task_data)
        
        assert "name" in validated_data
        assert validated_data["name"] == "Secure Test Task"
        assert "description" in validated_data
        
        # Test invalid task creation (without permissions)
        invalid_user = self.security.create_user(
            user_id="no_perms",
            username="noperms",
            email="noperms@example.com",
            password="NoPerms123!"
        )
        
        invalid_session = self.security.authenticate_user("noperms", "NoPerms123!")
        
        with pytest.raises(PermissionError):
            self.security.secure_task_creation(invalid_session, task_data)
    
    def test_security_reporting(self):
        """Test security reporting functionality."""
        # Generate some security events
        self.security.create_user(
            user_id="report_user",
            username="reporter",
            email="reporter@example.com",
            password="ReportPass123!",
            permissions=[Permission.ADMIN_ACCESS]
        )
        
        session_token = self.security.authenticate_user("reporter", "ReportPass123!")
        
        # Perform some operations to generate events
        self.security.check_permission(session_token, Permission.ADMIN_ACCESS)
        self.security.authenticate_user("reporter", "wrongpassword")  # Failed auth
        
        # Generate security report
        report = self.security.get_security_report()
        
        assert 'user_statistics' in report
        assert 'session_statistics' in report
        assert 'security_events' in report
        
        # Verify user statistics
        user_stats = report['user_statistics']
        assert user_stats['total_users'] >= 1
        assert user_stats['active_users'] >= 1
        
        # Verify security events
        events = report['security_events']
        assert events['total_events'] > 0


class TestPerformanceOptimization:
    """Integration tests for performance optimization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scheduler = QuantumMLScheduler(max_parallel_tasks=4)
        self.optimizer = PerformanceOptimizer(self.scheduler)
    
    def test_performance_profiling(self):
        """Test performance profiling functionality."""
        # Perform operations with profiling
        with self.optimizer.profiler.profile_operation("test_operation"):
            time.sleep(0.01)  # Simulate work
        
        with self.optimizer.profiler.profile_operation("test_operation"):
            time.sleep(0.02)  # Different duration
        
        # Get performance summary
        summary = self.optimizer.profiler.get_performance_summary()
        
        assert 'operations' in summary
        assert 'test_operation' in summary['operations']
        
        op_stats = summary['operations']['test_operation']
        assert op_stats['count'] == 2
        assert op_stats['avg_time'] > 0
        assert op_stats['min_time'] > 0
        assert op_stats['max_time'] > 0
    
    def test_cache_functionality(self):
        """Test quantum cache functionality."""
        cache = self.optimizer.cache
        
        # Test cache operations
        cache.put("test_key", {"data": "test_value"})
        cached_value = cache.get("test_key")
        
        assert cached_value is not None
        assert cached_value["data"] == "test_value"
        
        # Test cache miss
        missing_value = cache.get("nonexistent_key")
        assert missing_value is None
        
        # Test cache statistics
        stats = cache.get_cache_stats()
        assert 'total_entries' in stats
        assert stats['total_entries'] >= 1
    
    def test_parallel_execution(self):
        """Test parallel task execution."""
        def simple_task(x):
            return x * 2
        
        # Create task functions
        task_functions = [lambda i=i: simple_task(i) for i in range(5)]
        
        # Execute in parallel
        results = self.optimizer.execute_parallel_tasks(task_functions, max_workers=3)
        
        assert len(results) == 5
        assert results == [0, 2, 4, 6, 8]  # Expected results
    
    def test_performance_reporting(self):
        """Test performance reporting."""
        # Create some tasks to generate metrics
        for i in range(3):
            task_id = self.scheduler.add_intelligent_task(f"Perf Task {i+1}")
            self.scheduler.start_task(task_id)
            self.scheduler.complete_task_with_learning(task_id)
        
        # Get performance report
        report = self.optimizer.get_performance_report()
        
        assert 'timestamp' in report
        assert 'current_capacity' in report
        assert 'profiler_summary' in report
        assert 'cache_statistics' in report
        assert 'performance_recommendations' in report
        
        # Verify report structure
        capacity = report['current_capacity']
        assert 'thread_pool_workers' in capacity
        assert 'max_parallel_tasks' in capacity


class TestHealthMonitoring:
    """Integration tests for health monitoring."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scheduler = QuantumMLScheduler(max_parallel_tasks=4)
        self.health_monitor = SimpleHealthMonitor(self.scheduler, check_interval=1)
    
    def test_health_monitoring_lifecycle(self):
        """Test health monitoring start/stop lifecycle."""
        # Start monitoring
        self.health_monitor.start_monitoring()
        assert self.health_monitor.is_running is True
        assert self.health_monitor.monitor_thread is not None
        
        # Let it run briefly
        time.sleep(2)
        
        # Check that metrics are being collected
        assert len(self.health_monitor.metrics_history) > 0
        
        # Stop monitoring
        self.health_monitor.stop_monitoring()
        assert self.health_monitor.is_running is False
    
    def test_health_metrics_collection(self):
        """Test health metrics collection."""
        # Create some tasks to generate metrics
        for i in range(5):
            task_id = self.scheduler.add_intelligent_task(f"Health Task {i+1}")
            if i < 3:
                self.scheduler.start_task(task_id)
                if i < 2:
                    self.scheduler.complete_task_with_learning(task_id)
        
        # Start monitoring briefly
        self.health_monitor.start_monitoring()
        time.sleep(1.5)  # Wait for metrics collection
        self.health_monitor.stop_monitoring()
        
        # Check collected metrics
        assert len(self.health_monitor.metrics_history) > 0
        
        # Verify specific metrics exist
        metric_names = list(self.health_monitor.metrics_history.keys())
        expected_metrics = ['total_tasks', 'completed_tasks', 'running_tasks', 
                           'pending_tasks', 'task_completion_rate', 'queue_length']
        
        for expected in expected_metrics:
            assert expected in metric_names
    
    def test_health_status_evaluation(self):
        """Test health status evaluation."""
        # Create tasks with different states
        for i in range(10):
            task_id = self.scheduler.add_intelligent_task(f"Status Task {i+1}")
            if i < 7:  # Complete most tasks for good health
                self.scheduler.start_task(task_id)
                self.scheduler.complete_task_with_learning(task_id)
        
        # Manually trigger metrics collection
        self.health_monitor._collect_metrics()
        
        # Get health status
        health = self.health_monitor.get_current_health()
        
        assert health.status in ['healthy', 'degraded', 'critical']
        assert 0 <= health.score <= 100
        assert isinstance(health.alerts, list)
        assert len(health.metrics) > 0
    
    def test_health_report_export(self):
        """Test health report export."""
        # Generate some health data
        self.health_monitor._collect_metrics()
        
        # Export health report
        report = self.health_monitor.export_health_report()
        
        assert 'timestamp' in report
        assert 'overall_health' in report
        assert 'current_metrics' in report
        assert 'alerts' in report
        
        # Verify report structure
        overall_health = report['overall_health']
        assert 'status' in overall_health
        assert 'score' in overall_health
        assert 'alerts_count' in overall_health


class TestEndToEndIntegration:
    """End-to-end integration tests combining all components."""
    
    def setup_method(self):
        """Set up comprehensive test environment."""
        self.scheduler = QuantumMLScheduler(max_parallel_tasks=4)
        self.security = QuantumSchedulerSecurity()
        self.optimizer = PerformanceOptimizer(self.scheduler)
        self.health_monitor = SimpleHealthMonitor(self.scheduler, check_interval=1)
        
        # Create test user
        self.user = self.security.create_user(
            user_id="e2e_user",
            username="e2euser",
            email="e2e@example.com",
            password="E2EPass123!",
            permissions=[Permission.CREATE_TASKS, Permission.START_TASKS, 
                        Permission.VIEW_METRICS, Permission.MODIFY_TASKS]
        )
        
        self.session_token = self.security.authenticate_user("e2euser", "E2EPass123!")
    
    def test_complete_workflow(self):
        """Test complete workflow from task creation to completion with all features."""
        # Start monitoring and optimization
        self.health_monitor.start_monitoring()
        self.optimizer.start_optimization(OptimizationStrategy.BALANCED)
        
        try:
            # 1. Secure task creation
            task_data = {
                "name": "E2E Workflow Task",
                "description": "End-to-end integration test task",
                "priority": "high"
            }
            
            validated_data = self.security.secure_task_creation(self.session_token, task_data)
            
            # 2. Create intelligent task
            task_id = self.scheduler.add_intelligent_task(
                name=validated_data["name"],
                description=validated_data["description"],
                priority=TaskPriority.HIGH,
                tags=["e2e", "integration", "test"]
            )
            
            assert task_id is not None
            
            # 3. Get ML recommendations
            next_tasks = self.scheduler.get_intelligent_next_tasks()
            assert len(next_tasks) >= 1
            assert any(t.id == task_id for t in next_tasks)
            
            # 4. Execute task with performance monitoring
            with self.optimizer.profiler.profile_operation("e2e_task_execution"):
                success = self.scheduler.start_task(task_id)
                assert success is True
                
                # Simulate task work
                time.sleep(0.1)
                
                success = self.scheduler.complete_task_with_learning(task_id)
                assert success is True
            
            # 5. Wait for monitoring to collect data
            time.sleep(2)
            
            # 6. Verify all systems recorded the activity
            
            # Security audit
            security_report = self.security.get_security_report()
            assert security_report['security_events']['total_events'] > 0
            
            # Performance metrics
            perf_report = self.optimizer.get_performance_report()
            assert perf_report['profiler_summary']['total_operations'] > 0
            
            # Health monitoring
            health_report = self.health_monitor.export_health_report()
            assert health_report['overall_health']['score'] >= 0
            
            # ML insights
            insights = self.scheduler.get_scheduling_insights()
            assert 'performance_metrics' in insights
            
            # Task completion
            task = self.scheduler.get_task(task_id)
            assert task.status == TaskStatus.COMPLETED
            
        finally:
            # Cleanup
            self.health_monitor.stop_monitoring()
            self.optimizer.stop_optimization()
    
    def test_system_resilience(self):
        """Test system resilience under various failure conditions."""
        # Test invalid session handling
        invalid_session = "invalid_session_token"
        
        with pytest.raises(PermissionError):
            self.security.secure_task_creation(invalid_session, {"name": "Test"})
        
        # Test scheduler with no tasks
        next_tasks = self.scheduler.get_intelligent_next_tasks()
        assert isinstance(next_tasks, list)
        
        # Test health monitoring with empty scheduler
        self.health_monitor._collect_metrics()
        health = self.health_monitor.get_current_health()
        assert health.status in ['healthy', 'degraded', 'critical', 'unknown']
        
        # Test performance optimizer with no operations
        report = self.optimizer.get_performance_report()
        assert 'timestamp' in report
    
    def teardown_method(self):
        """Clean up test environment."""
        # Stop any running services
        if hasattr(self, 'health_monitor') and self.health_monitor.is_running:
            self.health_monitor.stop_monitoring()
        
        if hasattr(self, 'optimizer') and self.optimizer.is_monitoring:
            self.optimizer.stop_optimization()


# Test runner for comprehensive validation
if __name__ == "__main__":
    print("🧪 Running Comprehensive Integration Tests")
    print("=" * 50)
    
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])