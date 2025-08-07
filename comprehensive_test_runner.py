#!/usr/bin/env python3
"""Comprehensive test runner for quantum task scheduler without external dependencies."""

import sys
import os
import time
import traceback
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

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


class TestResult:
    """Simple test result tracking."""
    
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.duration = 0.0
        self.start_time = None
    
    def start(self):
        """Start test timing."""
        self.start_time = time.time()
    
    def finish(self, passed: bool, error=None):
        """Finish test with result."""
        self.duration = time.time() - self.start_time
        self.passed = passed
        self.error = error


class TestRunner:
    """Simple test runner."""
    
    def __init__(self):
        self.results = []
        self.setup_data = {}
    
    def run_test(self, test_func, test_name: str):
        """Run a single test function."""
        result = TestResult(test_name)
        result.start()
        
        try:
            print(f"  Running {test_name}...", end=" ")
            test_func()
            result.finish(True)
            print("✅ PASS")
        except Exception as e:
            result.finish(False, e)
            print(f"❌ FAIL: {str(e)}")
        
        self.results.append(result)
        return result.passed
    
    def print_summary(self):
        """Print test summary."""
        passed = len([r for r in self.results if r.passed])
        total = len(self.results)
        
        print(f"\n{'='*50}")
        print(f"Test Summary: {passed}/{total} passed")
        print(f"Total time: {sum(r.duration for r in self.results):.2f}s")
        
        if passed < total:
            print(f"\n❌ Failed Tests:")
            for result in self.results:
                if not result.passed:
                    print(f"  - {result.name}: {result.error}")
        
        return passed == total


def test_quantum_scheduler_basic():
    """Test basic quantum scheduler functionality."""
    scheduler = QuantumScheduler(max_parallel_tasks=4)
    
    # Create task
    task_id = scheduler.create_task(
        name="Test Task",
        priority=TaskPriority.HIGH,
        estimated_duration=timedelta(minutes=30)
    )
    
    assert task_id is not None
    assert len(task_id) > 0
    
    # Retrieve task
    task = scheduler.get_task(task_id)
    assert task is not None
    assert task.name == "Test Task"
    assert task.priority == TaskPriority.HIGH
    
    # Task lifecycle
    stats = scheduler.get_task_statistics()
    assert stats['pending'] == 1
    
    success = scheduler.start_task(task_id)
    assert success is True
    
    success = scheduler.complete_task(task_id)
    assert success is True
    
    final_stats = scheduler.get_task_statistics()
    assert final_stats['completed'] == 1


def test_task_dependencies():
    """Test task dependency handling."""
    scheduler = QuantumScheduler()
    
    # Create dependent tasks
    parent_id = scheduler.create_task("Parent Task", priority=TaskPriority.HIGH)
    child_id = scheduler.create_task(
        "Child Task", 
        dependencies=[parent_id],
        priority=TaskPriority.MEDIUM
    )
    
    # Child should not be available due to dependency
    next_tasks = scheduler.get_next_tasks()
    next_task_ids = [t.id for t in next_tasks]
    
    assert parent_id in next_task_ids
    assert child_id not in next_task_ids
    
    # Complete parent task
    scheduler.start_task(parent_id)
    scheduler.complete_task(parent_id)
    
    # Now child should be available
    next_tasks = scheduler.get_next_tasks()
    next_task_ids = [t.id for t in next_tasks]
    assert child_id in next_task_ids


def test_parallel_task_limits():
    """Test parallel task execution limits."""
    scheduler = QuantumScheduler(max_parallel_tasks=2)
    
    # Create more tasks than limit
    task_ids = []
    for i in range(4):
        task_id = scheduler.create_task(f"Task {i+1}")
        task_ids.append(task_id)
    
    # Try to start all tasks
    started_count = 0
    for task_id in task_ids:
        if scheduler.start_task(task_id):
            started_count += 1
    
    # Should be limited to max_parallel_tasks
    assert started_count == 2
    assert len(scheduler.running_tasks) == 2


def test_ml_scheduler_features():
    """Test ML scheduler enhanced features."""
    ml_scheduler = QuantumMLScheduler(max_parallel_tasks=4)
    
    # Create intelligent task
    task_id = ml_scheduler.add_intelligent_task(
        name="ML Test Task",
        description="Machine learning enhanced task",
        priority=TaskPriority.HIGH,
        tags=["ml", "test"]
    )
    
    task = ml_scheduler.get_task(task_id)
    assert hasattr(task, 'predicted_success_probability')
    assert hasattr(task, 'complexity_score')
    assert hasattr(task, 'tags')
    
    # Test ML recommendations
    next_tasks = ml_scheduler.get_intelligent_next_tasks()
    assert isinstance(next_tasks, list)
    
    # Test learning
    ml_scheduler.start_task(task_id)
    success = ml_scheduler.complete_task_with_learning(task_id)
    assert success is True
    assert len(ml_scheduler.scheduling_history) > 0


def test_security_framework():
    """Test security framework functionality."""
    security = QuantumSchedulerSecurity()
    
    # Create user
    user = security.create_user(
        user_id="test_user",
        username="testuser",
        email="test@example.com",
        password="TestPass123!",
        permissions=[Permission.CREATE_TASKS, Permission.READ_TASKS]
    )
    
    assert user.user_id == "test_user"
    
    # Authenticate
    session_token = security.authenticate_user("testuser", "TestPass123!")
    assert session_token is not None
    
    # Check permissions
    has_create = security.check_permission(session_token, Permission.CREATE_TASKS)
    has_admin = security.check_permission(session_token, Permission.ADMIN_ACCESS)
    
    assert has_create is True
    assert has_admin is False
    
    # Test secure task creation
    task_data = {"name": "Secure Task", "description": "Test task"}
    validated_data = security.secure_task_creation(session_token, task_data)
    assert "name" in validated_data


def test_performance_optimizer():
    """Test performance optimization features."""
    scheduler = QuantumMLScheduler()
    optimizer = PerformanceOptimizer(scheduler)
    
    # Test profiling
    with optimizer.profiler.profile_operation("test_op"):
        time.sleep(0.01)
    
    summary = optimizer.profiler.get_performance_summary()
    assert 'operations' in summary
    assert 'test_op' in summary['operations']
    
    # Test cache
    cache = optimizer.cache
    cache.put("test_key", {"data": "value"})
    value = cache.get("test_key")
    assert value is not None
    assert value["data"] == "value"
    
    # Test parallel execution
    def simple_task(x):
        return x * 2
    
    tasks = [lambda i=i: simple_task(i) for i in range(3)]
    results = optimizer.execute_parallel_tasks(tasks)
    assert len(results) == 3
    assert results == [0, 2, 4]


def test_health_monitoring():
    """Test health monitoring functionality."""
    scheduler = QuantumMLScheduler()
    monitor = SimpleHealthMonitor(scheduler, check_interval=1)
    
    # Create some tasks
    for i in range(3):
        task_id = scheduler.add_intelligent_task(f"Health Task {i+1}")
        if i < 2:
            scheduler.start_task(task_id)
            if i == 0:
                scheduler.complete_task_with_learning(task_id)
    
    # Collect metrics
    monitor._collect_metrics()
    
    # Get health status
    health = monitor.get_current_health()
    assert health.status in ['healthy', 'degraded', 'critical', 'unknown']
    assert 0 <= health.score <= 100
    
    # Export report
    report = monitor.export_health_report()
    assert 'timestamp' in report
    assert 'overall_health' in report


def test_state_persistence():
    """Test state export and import functionality."""
    scheduler = QuantumScheduler()
    
    # Create tasks
    task_ids = []
    for i in range(3):
        task_id = scheduler.create_task(f"Persist Task {i+1}")
        task_ids.append(task_id)
    
    # Start one task
    scheduler.start_task(task_ids[0])
    
    # Export state
    state_json = scheduler.export_state()
    assert state_json is not None
    assert len(state_json) > 0
    
    # Import into new scheduler
    new_scheduler = QuantumScheduler()
    new_scheduler.import_state(state_json)
    
    # Verify
    assert len(new_scheduler.tasks) == 3
    assert len(new_scheduler.running_tasks) == 1


def test_end_to_end_workflow():
    """Test complete end-to-end workflow."""
    # Initialize components
    scheduler = QuantumMLScheduler(max_parallel_tasks=4)
    security = QuantumSchedulerSecurity()
    optimizer = PerformanceOptimizer(scheduler)
    monitor = SimpleHealthMonitor(scheduler, check_interval=1)
    
    try:
        # Create user
        user = security.create_user(
            user_id="e2e_user",
            username="e2euser", 
            email="e2e@example.com",
            password="E2EPass123!",
            permissions=[Permission.CREATE_TASKS, Permission.START_TASKS, Permission.MODIFY_TASKS]
        )
        
        # Authenticate
        session_token = security.authenticate_user("e2euser", "E2EPass123!")
        assert session_token is not None
        
        # Create secure task
        task_data = {"name": "E2E Task", "description": "End-to-end test task"}
        validated_data = security.secure_task_creation(session_token, task_data)
        
        # Create intelligent task
        task_id = scheduler.add_intelligent_task(
            name=validated_data["name"],
            description=validated_data["description"],
            priority=TaskPriority.HIGH,
            tags=["e2e", "test"]
        )
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Execute task with profiling
        with optimizer.profiler.profile_operation("e2e_execution"):
            scheduler.start_task(task_id)
            time.sleep(0.1)  # Simulate work
            scheduler.complete_task_with_learning(task_id)
        
        # Wait for monitoring
        time.sleep(1.5)
        
        # Verify all components worked
        task = scheduler.get_task(task_id)
        assert task.status == TaskStatus.COMPLETED
        
        # Check reports
        security_report = security.get_security_report()
        assert security_report['security_events']['total_events'] > 0
        
        perf_report = optimizer.get_performance_report()
        assert perf_report['profiler_summary']['total_operations'] > 0
        
        health_report = monitor.export_health_report()
        assert health_report['overall_health']['score'] >= 0
        
    finally:
        if monitor.is_running:
            monitor.stop_monitoring()


def main():
    """Run comprehensive test suite."""
    print("🧪 Quantum Task Scheduler - Comprehensive Test Suite")
    print("=" * 60)
    
    runner = TestRunner()
    
    # Define test cases
    test_cases = [
        (test_quantum_scheduler_basic, "Quantum Scheduler Basic Functionality"),
        (test_task_dependencies, "Task Dependency Handling"),
        (test_parallel_task_limits, "Parallel Task Limits"),
        (test_ml_scheduler_features, "ML Scheduler Features"),
        (test_security_framework, "Security Framework"),
        (test_performance_optimizer, "Performance Optimizer"),
        (test_health_monitoring, "Health Monitoring"),
        (test_state_persistence, "State Persistence"),
        (test_end_to_end_workflow, "End-to-End Workflow")
    ]
    
    print(f"\n🚀 Running {len(test_cases)} test cases...")
    print("-" * 40)
    
    # Run all tests
    for test_func, test_name in test_cases:
        runner.run_test(test_func, test_name)
    
    # Print summary
    all_passed = runner.print_summary()
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! System is ready for production.")
        print("\n✅ Quality Gates:")
        print("  ✅ Core functionality validated")
        print("  ✅ ML features operational") 
        print("  ✅ Security framework verified")
        print("  ✅ Performance optimization working")
        print("  ✅ Health monitoring active")
        print("  ✅ End-to-end workflow successful")
        print("  ✅ State persistence confirmed")
    else:
        print("\n❌ Some tests failed. Review and fix issues before production.")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)