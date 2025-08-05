"""Simple validation test for quantum task planner using only standard library."""

import sys
import json
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, '/root/repo/src')

def test_quantum_scheduler():
    """Test basic quantum scheduler functionality."""
    try:
        from quantum_inspired_task_planner.quantum_scheduler import (
            QuantumScheduler, QuantumTask, TaskPriority, TaskStatus
        )
        
        # Test scheduler creation
        scheduler = QuantumScheduler()
        print("✓ QuantumScheduler created successfully")
        
        # Test task creation
        task_id = scheduler.create_task("Test Task", priority=TaskPriority.HIGH)
        print(f"✓ Task created with ID: {task_id[:8]}...")
        
        # Test task retrieval
        task = scheduler.get_task(task_id)
        assert task is not None
        assert task.name == "Test Task"
        print("✓ Task retrieval working")
        
        # Test statistics
        stats = scheduler.get_task_statistics()
        assert stats["pending"] == 1
        print("✓ Task statistics working")
        
        # Test task starting
        success = scheduler.start_task(task_id)
        assert success is True
        assert task.status == TaskStatus.RUNNING
        print("✓ Task starting working")
        
        # Test task completion
        success = scheduler.complete_task(task_id)
        assert success is True
        assert task.status == TaskStatus.COMPLETED
        print("✓ Task completion working")
        
        return True
        
    except Exception as e:
        print(f"✗ QuantumScheduler test failed: {e}")
        return False


def test_resource_allocator():
    """Test basic resource allocator functionality."""
    try:
        from quantum_inspired_task_planner.resource_allocator import (
            QuantumResourceAllocator, ResourceType
        )
        
        # Test allocator creation
        allocator = QuantumResourceAllocator()
        print("✓ QuantumResourceAllocator created successfully")
        
        # Test resource addition
        allocator.add_resource("cpu_pool", ResourceType.CPU, 100.0)
        print("✓ Resource addition working")
        
        # Test resource allocation
        success = allocator.allocate_resources("task_1", {"cpu": 25.0})
        assert success is True
        print("✓ Resource allocation working")
        
        # Test utilization stats
        utilization = allocator.get_resource_utilization()
        assert "cpu_pool" in utilization
        assert utilization["cpu_pool"]["utilization_percent"] == 25.0
        print("✓ Resource utilization stats working")
        
        return True
        
    except Exception as e:
        print(f"✗ ResourceAllocator test failed: {e}")
        return False


def test_validation_system():
    """Test validation system functionality."""
    try:
        from quantum_inspired_task_planner.validation import TaskValidator
        
        # Test valid task validation
        result = TaskValidator.validate_task_creation(
            name="Valid Task",
            priority="high"
        )
        assert result.is_valid is True
        print("✓ Valid task validation working")
        
        # Test invalid task validation
        result = TaskValidator.validate_task_creation(name="")
        assert result.is_valid is False
        assert len(result.errors) > 0
        print("✓ Invalid task validation working")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation test failed: {e}")
        return False


def test_error_handling():
    """Test error handling system."""
    try:
        from quantum_inspired_task_planner.error_handling import (
            ErrorHandler, QuantumError, ErrorCategory, ErrorSeverity
        )
        
        # Test error creation
        error = QuantumError(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            message="Test error",
            details={},
            timestamp=datetime.now(),
            operation="test_operation"
        )
        print("✓ QuantumError creation working")
        
        # Test error handler
        handler = ErrorHandler()
        handler.handle_error(error)
        assert len(handler.error_history) == 1
        print("✓ Error handling working")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False


def run_all_tests():
    """Run all validation tests."""
    print("Running Quantum Task Planner Validation Tests")
    print("=" * 50)
    
    tests = [
        ("Quantum Scheduler", test_quantum_scheduler),
        ("Resource Allocator", test_resource_allocator),
        ("Validation System", test_validation_system),
        ("Error Handling", test_error_handling)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}:")
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")
    
    print(f"\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Quantum Task Planner is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Check implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)