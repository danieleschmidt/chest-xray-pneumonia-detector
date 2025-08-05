"""Comprehensive tests for quantum scheduler functionality."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import uuid

from src.quantum_inspired_task_planner.quantum_scheduler import (
    QuantumScheduler, QuantumTask, TaskPriority, TaskStatus
)


class TestQuantumTask:
    """Test quantum task functionality."""
    
    def test_task_creation_defaults(self):
        """Test task creation with default values."""
        task = QuantumTask(name="Test Task")
        
        assert task.name == "Test Task"
        assert task.priority == TaskPriority.MEDIUM
        assert task.status == TaskStatus.PENDING
        assert len(task.dependencies) == 0
        assert task.estimated_duration == timedelta(hours=1)
        assert task.probability_amplitude == complex(1.0, 0.0)
        assert task.superposition_weight == 1.0
        assert isinstance(task.id, str)
        assert len(task.id) == 36  # UUID length
    
    def test_task_creation_with_parameters(self):
        """Test task creation with custom parameters."""
        dependencies = {"dep1", "dep2"}
        duration = timedelta(hours=2)
        resources = {"cpu": 4.0, "memory": 8.0}
        
        task = QuantumTask(
            name="Complex Task",
            description="A complex test task",
            priority=TaskPriority.HIGH,
            dependencies=dependencies,
            estimated_duration=duration,
            resource_requirements=resources
        )
        
        assert task.name == "Complex Task"
        assert task.description == "A complex test task"
        assert task.priority == TaskPriority.HIGH
        assert task.dependencies == dependencies
        assert task.estimated_duration == duration
        assert task.resource_requirements == resources


class TestQuantumScheduler:
    """Test quantum scheduler functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.scheduler = QuantumScheduler(max_parallel_tasks=4)
    
    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        assert self.scheduler.max_parallel_tasks == 4
        assert len(self.scheduler.tasks) == 0
        assert len(self.scheduler.running_tasks) == 0
        assert len(self.scheduler.completed_tasks) == 0
    
    def test_add_task(self):
        """Test adding tasks to scheduler."""
        task = QuantumTask(name="Test Task")
        task_id = self.scheduler.add_task(task)
        
        assert task_id == task.id
        assert task_id in self.scheduler.tasks
        assert self.scheduler.tasks[task_id] == task
    
    def test_create_task(self):
        """Test task creation via scheduler."""
        task_id = self.scheduler.create_task(
            name="New Task",
            description="Test task creation",
            priority=TaskPriority.HIGH
        )
        
        assert task_id in self.scheduler.tasks
        task = self.scheduler.tasks[task_id]
        assert task.name == "New Task"
        assert task.description == "Test task creation"
        assert task.priority == TaskPriority.HIGH
    
    def test_get_task(self):
        """Test task retrieval."""
        task_id = self.scheduler.create_task("Test Task")
        
        retrieved_task = self.scheduler.get_task(task_id)
        assert retrieved_task is not None
        assert retrieved_task.id == task_id
        assert retrieved_task.name == "Test Task"
        
        # Test non-existent task
        non_existent = self.scheduler.get_task("non-existent-id")
        assert non_existent is None
    
    def test_task_statistics(self):
        """Test task statistics calculation."""
        # Initially empty
        stats = self.scheduler.get_task_statistics()
        assert stats["pending"] == 0
        assert stats["running"] == 0
        assert stats["completed"] == 0
        
        # Add tasks in different states
        task1_id = self.scheduler.create_task("Task 1")
        task2_id = self.scheduler.create_task("Task 2")
        
        # Start one task
        self.scheduler.start_task(task1_id)
        
        # Complete one task
        self.scheduler.complete_task(task1_id)
        
        stats = self.scheduler.get_task_statistics()
        assert stats["pending"] == 1  # task2
        assert stats["running"] == 0
        assert stats["completed"] == 1  # task1
    
    def test_start_task(self):
        """Test task starting."""
        task_id = self.scheduler.create_task("Test Task")
        
        # Test successful start
        success = self.scheduler.start_task(task_id)
        assert success is True
        
        task = self.scheduler.get_task(task_id)
        assert task.status == TaskStatus.RUNNING
        assert task.started_at is not None
        assert task_id in self.scheduler.running_tasks
        
        # Test starting already running task
        success = self.scheduler.start_task(task_id)
        assert success is False
    
    def test_complete_task(self):
        """Test task completion."""
        task_id = self.scheduler.create_task("Test Task")
        
        # Can't complete pending task
        success = self.scheduler.complete_task(task_id)
        assert success is False
        
        # Start then complete
        self.scheduler.start_task(task_id)
        success = self.scheduler.complete_task(task_id)
        assert success is True
        
        task = self.scheduler.get_task(task_id)
        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None
        assert task_id not in self.scheduler.running_tasks
        assert task_id in self.scheduler.completed_tasks
    
    def test_dependencies_satisfied(self):
        """Test dependency satisfaction checking."""
        # Create tasks with dependencies
        task1_id = self.scheduler.create_task("Task 1")
        task2_id = self.scheduler.create_task("Task 2", dependencies=[task1_id])
        
        task1 = self.scheduler.get_task(task1_id)
        task2 = self.scheduler.get_task(task2_id)
        
        # Task 2 should not be able to start while Task 1 is pending
        assert self.scheduler._are_dependencies_satisfied(task1) is True
        assert self.scheduler._are_dependencies_satisfied(task2) is False
        
        # Complete Task 1
        self.scheduler.start_task(task1_id)
        self.scheduler.complete_task(task1_id)
        
        # Now Task 2 should be able to start
        assert self.scheduler._are_dependencies_satisfied(task2) is True
    
    def test_get_next_tasks(self):
        """Test getting next tasks for execution."""
        # Create tasks with different priorities
        high_task_id = self.scheduler.create_task("High Priority", priority=TaskPriority.HIGH)
        medium_task_id = self.scheduler.create_task("Medium Priority", priority=TaskPriority.MEDIUM)
        low_task_id = self.scheduler.create_task("Low Priority", priority=TaskPriority.LOW)
        
        next_tasks = self.scheduler.get_next_tasks()
        
        # Should return tasks sorted by priority score
        assert len(next_tasks) > 0
        assert next_tasks[0].priority == TaskPriority.HIGH
        
        # Test with max parallel constraint
        for _ in range(4):  # Fill up parallel slots
            task_id = self.scheduler.create_task(f"Filler Task {_}")
            self.scheduler.start_task(task_id)
        
        next_tasks = self.scheduler.get_next_tasks()
        assert len(next_tasks) == 0  # No slots available
    
    def test_priority_score_calculation(self):
        """Test quantum-inspired priority score calculation."""
        task = QuantumTask(name="Test Task", priority=TaskPriority.HIGH)
        
        # Mock created_at to test urgency factor
        with patch.object(task, 'created_at', datetime.now() - timedelta(hours=5)):
            score = self.scheduler._calculate_priority_score(task)
            
            # Score should include base priority + urgency
            assert score > TaskPriority.HIGH.value
    
    def test_entanglement_creation(self):
        """Test quantum entanglement between tasks."""
        task1_id = self.scheduler.create_task("Task 1")
        task2_id = self.scheduler.create_task("Task 2")
        
        # Create entanglement
        success = self.scheduler.create_entanglement(task1_id, task2_id)
        assert success is True
        
        task1 = self.scheduler.get_task(task1_id)
        task2 = self.scheduler.get_task(task2_id)
        
        assert task2_id in task1.entangled_tasks
        assert task1_id in task2.entangled_tasks
        
        # Test with non-existent task
        success = self.scheduler.create_entanglement(task1_id, "non-existent")
        assert success is False
    
    def test_entanglement_effects(self):
        """Test quantum entanglement effects on task completion."""
        task1_id = self.scheduler.create_task("Task 1")
        task2_id = self.scheduler.create_task("Task 2")
        
        # Create entanglement
        self.scheduler.create_entanglement(task1_id, task2_id)
        
        task2 = self.scheduler.get_task(task2_id)
        original_weight = task2.superposition_weight
        
        # Complete task1 and check entanglement effect on task2
        self.scheduler.start_task(task1_id)
        self.scheduler.complete_task(task1_id)
        
        # Task2's superposition weight should be boosted
        assert task2.superposition_weight > original_weight
    
    def test_optimize_schedule(self):
        """Test schedule optimization."""
        # Create multiple tasks
        for i in range(5):
            self.scheduler.create_task(f"Task {i}", priority=TaskPriority.MEDIUM)
        
        optimized_schedule = self.scheduler.optimize_schedule()
        
        assert len(optimized_schedule) == 5
        
        # Check that results are sorted by score (descending)
        scores = [score for _, score in optimized_schedule]
        assert scores == sorted(scores, reverse=True)
    
    def test_state_export_import(self):
        """Test scheduler state export and import."""
        # Create some tasks
        task1_id = self.scheduler.create_task("Task 1", priority=TaskPriority.HIGH)
        task2_id = self.scheduler.create_task("Task 2", dependencies=[task1_id])
        
        # Start one task
        self.scheduler.start_task(task1_id)
        
        # Export state
        state_json = self.scheduler.export_state()
        assert isinstance(state_json, str)
        
        # Create new scheduler and import state
        new_scheduler = QuantumScheduler()
        new_scheduler.import_state(state_json)
        
        # Verify state was imported correctly
        assert len(new_scheduler.tasks) == 2
        assert task1_id in new_scheduler.tasks
        assert task2_id in new_scheduler.tasks
        assert task1_id in new_scheduler.running_tasks
        
        imported_task1 = new_scheduler.get_task(task1_id)
        imported_task2 = new_scheduler.get_task(task2_id)
        
        assert imported_task1.name == "Task 1"
        assert imported_task1.status == TaskStatus.RUNNING
        assert imported_task2.name == "Task 2"
        assert task1_id in imported_task2.dependencies
    
    def test_quantum_state_matrix_update(self):
        """Test quantum state matrix updates."""
        initial_matrix_shape = self.scheduler.quantum_state_matrix.shape
        
        # Add tasks and check matrix update
        task1_id = self.scheduler.create_task("Task 1")
        task2_id = self.scheduler.create_task("Task 2")
        
        # Matrix should be updated
        assert self.scheduler.quantum_state_matrix.shape != initial_matrix_shape
        assert self.scheduler.quantum_state_matrix.shape[0] == 2  # 2 tasks
    
    def test_max_parallel_constraint(self):
        """Test maximum parallel tasks constraint."""
        # Create more tasks than parallel limit
        task_ids = []
        for i in range(6):  # More than max_parallel_tasks (4)
            task_id = self.scheduler.create_task(f"Task {i}")
            task_ids.append(task_id)
        
        # Start tasks up to limit
        started_count = 0
        for task_id in task_ids:
            if self.scheduler.start_task(task_id):
                started_count += 1
        
        assert started_count == 4  # Should be limited to max_parallel_tasks
        assert len(self.scheduler.running_tasks) == 4
    
    def test_circular_dependency_detection(self):
        """Test that circular dependencies are handled properly."""
        # This would be tested in the validation module
        # Here we test basic dependency chain
        task1_id = self.scheduler.create_task("Task 1")
        task2_id = self.scheduler.create_task("Task 2", dependencies=[task1_id])
        task3_id = self.scheduler.create_task("Task 3", dependencies=[task2_id])
        
        # Task 3 should not be able to start until Task 1 and 2 are complete
        task3 = self.scheduler.get_task(task3_id)
        assert not self.scheduler._are_dependencies_satisfied(task3)
        
        # Complete dependencies in order
        self.scheduler.start_task(task1_id)
        self.scheduler.complete_task(task1_id)
        
        # Task 2 can now start
        assert self.scheduler._are_dependencies_satisfied(self.scheduler.get_task(task2_id))
        
        self.scheduler.start_task(task2_id)
        self.scheduler.complete_task(task2_id)
        
        # Now Task 3 can start
        assert self.scheduler._are_dependencies_satisfied(task3)


@pytest.fixture
def sample_scheduler():
    """Fixture providing a scheduler with sample tasks."""
    scheduler = QuantumScheduler()
    
    # Add sample tasks
    task1_id = scheduler.create_task("High Priority Task", priority=TaskPriority.HIGH)
    task2_id = scheduler.create_task("Medium Priority Task", priority=TaskPriority.MEDIUM)
    task3_id = scheduler.create_task("Dependent Task", dependencies=[task1_id])
    
    return scheduler, [task1_id, task2_id, task3_id]


class TestQuantumSchedulerIntegration:
    """Integration tests for quantum scheduler."""
    
    def test_full_task_lifecycle(self, sample_scheduler):
        """Test complete task lifecycle."""
        scheduler, task_ids = sample_scheduler
        task1_id, task2_id, task3_id = task_ids
        
        # Get next tasks (should prioritize high priority)
        next_tasks = scheduler.get_next_tasks()
        assert len(next_tasks) >= 1
        assert next_tasks[0].priority == TaskPriority.HIGH
        
        # Start high priority task
        scheduler.start_task(task1_id)
        assert scheduler.get_task(task1_id).status == TaskStatus.RUNNING
        
        # Complete it
        scheduler.complete_task(task1_id)
        assert scheduler.get_task(task1_id).status == TaskStatus.COMPLETED
        
        # Now dependent task should be available
        next_tasks = scheduler.get_next_tasks()
        dependent_task = scheduler.get_task(task3_id)
        assert dependent_task in next_tasks
    
    def test_quantum_optimization_integration(self, sample_scheduler):
        """Test quantum optimization with realistic scenario."""
        scheduler, task_ids = sample_scheduler
        
        # Add more tasks for meaningful optimization
        for i in range(5):
            scheduler.create_task(f"Optimization Task {i}", priority=TaskPriority.MEDIUM)
        
        # Get optimized schedule
        optimized_schedule = scheduler.optimize_schedule()
        
        assert len(optimized_schedule) > 0
        
        # Verify schedule is ordered by priority score
        scores = [score for _, score in optimized_schedule]
        assert scores == sorted(scores, reverse=True)
    
    def test_entanglement_priority_boost(self, sample_scheduler):
        """Test that entanglement boosts task priorities."""
        scheduler, task_ids = sample_scheduler
        task1_id, task2_id, _ = task_ids
        
        # Create entanglement between tasks
        scheduler.create_entanglement(task1_id, task2_id)
        
        task2 = scheduler.get_task(task2_id)
        original_weight = task2.superposition_weight
        
        # Complete entangled task
        scheduler.start_task(task1_id)
        scheduler.complete_task(task1_id)
        
        # Entangled task should have boosted weight
        assert task2.superposition_weight > original_weight
    
    def test_concurrent_task_operations(self, sample_scheduler):
        """Test thread safety of scheduler operations."""
        scheduler, _ = sample_scheduler
        
        import threading
        import time
        
        results = []
        errors = []
        
        def create_and_start_task(task_num):
            try:
                task_id = scheduler.create_task(f"Concurrent Task {task_num}")
                success = scheduler.start_task(task_id)
                results.append((task_num, task_id, success))
            except Exception as e:
                errors.append((task_num, e))
        
        # Create multiple threads to test concurrency
        threads = []
        for i in range(10):
            thread = threading.Thread(target=create_and_start_task, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Concurrent operations failed: {errors}"
        assert len(results) == 10
        
        # Verify scheduler state consistency
        assert len(scheduler.tasks) >= 10  # Original + new tasks
        assert len(scheduler.running_tasks) <= scheduler.max_parallel_tasks
    
    def test_performance_under_load(self, sample_scheduler):
        """Test scheduler performance under high load."""
        scheduler, _ = sample_scheduler
        
        # Create many tasks
        start_time = time.time()
        task_ids = []
        
        for i in range(1000):
            task_id = scheduler.create_task(f"Load Test Task {i}")
            task_ids.append(task_id)
        
        creation_time = time.time() - start_time
        
        # Performance assertion: should create 1000 tasks in reasonable time
        assert creation_time < 5.0, f"Task creation too slow: {creation_time:.2f}s"
        
        # Test next task calculation performance
        start_time = time.time()
        next_tasks = scheduler.get_next_tasks()
        calculation_time = time.time() - start_time
        
        assert calculation_time < 1.0, f"Next task calculation too slow: {calculation_time:.2f}s"
        assert len(next_tasks) <= scheduler.max_parallel_tasks
    
    def test_state_consistency_after_operations(self, sample_scheduler):
        """Test that scheduler state remains consistent after operations."""
        scheduler, task_ids = sample_scheduler
        
        # Perform various operations
        task1_id, task2_id, task3_id = task_ids
        
        # Start and complete tasks
        scheduler.start_task(task1_id)
        scheduler.complete_task(task1_id)
        scheduler.start_task(task2_id)
        
        # Verify state consistency
        stats = scheduler.get_task_statistics()
        
        assert stats["completed"] == 1
        assert stats["running"] == 1
        assert len(scheduler.running_tasks) == stats["running"]
        assert len(scheduler.completed_tasks) == stats["completed"]
        
        # Verify quantum state matrix dimensions
        expected_size = len(scheduler.tasks)
        actual_size = scheduler.quantum_state_matrix.shape[0]
        assert actual_size == expected_size


class TestQuantumSchedulerEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_scheduler_operations(self):
        """Test operations on empty scheduler."""
        scheduler = QuantumScheduler()
        
        # Should handle empty state gracefully
        assert scheduler.get_next_tasks() == []
        assert scheduler.optimize_schedule() == []
        
        stats = scheduler.get_task_statistics()
        assert all(count == 0 for count in stats.values())
    
    def test_invalid_task_operations(self):
        """Test operations with invalid task IDs."""
        scheduler = QuantumScheduler()
        
        # Operations with non-existent task IDs should fail gracefully
        assert scheduler.start_task("non-existent-id") is False
        assert scheduler.complete_task("non-existent-id") is False
        assert scheduler.get_task("non-existent-id") is None
        assert scheduler.create_entanglement("non-existent-1", "non-existent-2") is False
    
    def test_duplicate_task_operations(self):
        """Test duplicate operations on same task."""
        scheduler = QuantumScheduler()
        task_id = scheduler.create_task("Test Task")
        
        # Start task twice
        assert scheduler.start_task(task_id) is True
        assert scheduler.start_task(task_id) is False  # Already running
        
        # Complete task twice
        assert scheduler.complete_task(task_id) is True
        assert scheduler.complete_task(task_id) is False  # Already completed
    
    def test_large_dependency_chains(self):
        """Test performance with large dependency chains."""
        scheduler = QuantumScheduler()
        
        # Create chain of 100 dependent tasks
        task_ids = []
        for i in range(100):
            deps = [task_ids[-1]] if task_ids else []
            task_id = scheduler.create_task(f"Chain Task {i}", dependencies=deps)
            task_ids.append(task_id)
        
        # Should handle large chains without performance issues
        start_time = time.time()
        next_tasks = scheduler.get_next_tasks()
        calculation_time = time.time() - start_time
        
        assert calculation_time < 2.0  # Should be fast even with large chains
        assert len(next_tasks) == 1  # Only first task in chain should be available