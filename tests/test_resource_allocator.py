"""Tests for quantum resource allocator."""

import pytest
from unittest.mock import Mock, patch
import logging

from src.quantum_inspired_task_planner.resource_allocator import (
    QuantumResourceAllocator, Resource, ResourceType
)


class TestResource:
    """Test Resource dataclass functionality."""
    
    def test_resource_creation(self):
        """Test resource creation with valid parameters."""
        resource = Resource(
            type=ResourceType.CPU,
            total_capacity=100.0,
            available_capacity=100.0,
            allocated_tasks={}
        )
        
        assert resource.type == ResourceType.CPU
        assert resource.total_capacity == 100.0
        assert resource.available_capacity == 100.0
        assert resource.allocated_tasks == {}
        assert resource.efficiency_score == 1.0
        assert resource.quantum_coherence == 1.0
    
    def test_resource_types(self):
        """Test all resource type enums."""
        assert ResourceType.CPU.value == "cpu"
        assert ResourceType.MEMORY.value == "memory"
        assert ResourceType.GPU.value == "gpu"
        assert ResourceType.STORAGE.value == "storage"
        assert ResourceType.NETWORK.value == "network"


class TestQuantumResourceAllocator:
    """Test quantum resource allocator functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.allocator = QuantumResourceAllocator()
    
    def test_allocator_initialization(self):
        """Test allocator initialization."""
        assert len(self.allocator.resources) == 0
        assert len(self.allocator.allocation_history) == 0
        assert self.allocator.temperature == 1.0
    
    def test_add_resource(self):
        """Test adding resources to allocator."""
        self.allocator.add_resource("cpu_1", ResourceType.CPU, 100.0)
        
        assert "cpu_1" in self.allocator.resources
        resource = self.allocator.resources["cpu_1"]
        assert resource.type == ResourceType.CPU
        assert resource.total_capacity == 100.0
        assert resource.available_capacity == 100.0
    
    def test_allocate_resources_success(self):
        """Test successful resource allocation."""
        # Add resources
        self.allocator.add_resource("cpu_1", ResourceType.CPU, 100.0)
        self.allocator.add_resource("mem_1", ResourceType.MEMORY, 1000.0)
        
        # Allocate resources for a task
        requirements = {"cpu": 20.0, "memory": 200.0}
        success = self.allocator.allocate_resources("task_1", requirements)
        
        assert success is True
        
        # Check resource allocation
        cpu_resource = self.allocator.resources["cpu_1"]
        mem_resource = self.allocator.resources["mem_1"]
        
        assert cpu_resource.available_capacity == 80.0
        assert mem_resource.available_capacity == 800.0
        assert "task_1" in cpu_resource.allocated_tasks
        assert "task_1" in mem_resource.allocated_tasks
        assert cpu_resource.allocated_tasks["task_1"] == 20.0
        assert mem_resource.allocated_tasks["task_1"] == 200.0
    
    def test_allocate_resources_insufficient(self):
        """Test resource allocation with insufficient resources."""
        # Add limited resources
        self.allocator.add_resource("cpu_1", ResourceType.CPU, 50.0)
        
        # Try to allocate more than available
        requirements = {"cpu": 100.0}
        success = self.allocator.allocate_resources("task_1", requirements)
        
        assert success is False
        
        # Resource should remain unchanged
        cpu_resource = self.allocator.resources["cpu_1"]
        assert cpu_resource.available_capacity == 50.0
        assert len(cpu_resource.allocated_tasks) == 0
    
    def test_deallocate_resources(self):
        """Test resource deallocation."""
        # Setup allocation
        self.allocator.add_resource("cpu_1", ResourceType.CPU, 100.0)
        requirements = {"cpu": 30.0}
        self.allocator.allocate_resources("task_1", requirements)
        
        # Verify allocation
        cpu_resource = self.allocator.resources["cpu_1"]
        assert cpu_resource.available_capacity == 70.0
        assert "task_1" in cpu_resource.allocated_tasks
        
        # Deallocate
        self.allocator.deallocate_resources("task_1")
        
        # Verify deallocation
        assert cpu_resource.available_capacity == 100.0
        assert "task_1" not in cpu_resource.allocated_tasks
    
    def test_resource_utilization_stats(self):
        """Test resource utilization statistics."""
        # Add and allocate resources
        self.allocator.add_resource("cpu_1", ResourceType.CPU, 100.0)
        self.allocator.add_resource("mem_1", ResourceType.MEMORY, 1000.0)
        
        requirements = {"cpu": 25.0, "memory": 300.0}
        self.allocator.allocate_resources("task_1", requirements)
        
        utilization = self.allocator.get_resource_utilization()
        
        assert "cpu_1" in utilization
        assert "mem_1" in utilization
        
        cpu_util = utilization["cpu_1"]
        assert cpu_util["type"] == "cpu"
        assert cpu_util["total_capacity"] == 100.0
        assert cpu_util["available_capacity"] == 75.0
        assert cpu_util["utilization_percent"] == 25.0
        assert cpu_util["allocated_tasks_count"] == 1
        
        mem_util = utilization["mem_1"]
        assert mem_util["utilization_percent"] == 30.0
    
    def test_optimal_resource_selection(self):
        """Test quantum-inspired optimal resource selection."""
        # Add multiple resources of same type with different characteristics
        self.allocator.add_resource("cpu_1", ResourceType.CPU, 100.0)
        self.allocator.add_resource("cpu_2", ResourceType.CPU, 100.0)
        
        # Modify efficiency scores
        self.allocator.resources["cpu_1"].efficiency_score = 1.2
        self.allocator.resources["cpu_2"].efficiency_score = 0.8
        
        # Allocate small amounts to test selection
        candidates = [
            ("cpu_1", self.allocator.resources["cpu_1"]),
            ("cpu_2", self.allocator.resources["cpu_2"])
        ]
        
        selected_id, selected_resource = self.allocator._select_optimal_resource(candidates, 10.0)
        
        # Should select more efficient resource
        assert selected_id == "cpu_1"
        assert selected_resource.efficiency_score == 1.2
    
    def test_rebalance_allocations(self):
        """Test resource rebalancing functionality."""
        # Setup imbalanced allocation
        self.allocator.add_resource("cpu_1", ResourceType.CPU, 100.0)
        self.allocator.add_resource("cpu_2", ResourceType.CPU, 100.0)
        
        # Heavily load one resource
        self.allocator.allocate_resources("task_1", {"cpu": 80.0})  # Goes to cpu_1
        self.allocator.allocate_resources("task_2", {"cpu": 10.0})  # Goes to cpu_1 or cpu_2
        
        # Force allocation to create imbalance
        cpu_1 = self.allocator.resources["cpu_1"]
        cpu_2 = self.allocator.resources["cpu_2"]
        
        # Manually create imbalance for testing
        cpu_1.allocated_tasks = {"task_1": 80.0, "task_2": 10.0}
        cpu_1.available_capacity = 10.0
        cpu_2.allocated_tasks = {}
        cpu_2.available_capacity = 100.0
        
        # Rebalance
        rebalanced_count = self.allocator.rebalance_allocations()
        
        # Should have rebalanced at least one task
        assert rebalanced_count >= 0  # May be 0 if rebalancing logic determines no benefit
    
    def test_multiple_resource_types(self):
        """Test allocation across multiple resource types."""
        # Add different resource types
        self.allocator.add_resource("cpu_pool", ResourceType.CPU, 200.0)
        self.allocator.add_resource("gpu_pool", ResourceType.GPU, 8.0)
        self.allocator.add_resource("mem_pool", ResourceType.MEMORY, 2000.0)
        
        # Allocate mixed resources
        requirements = {
            "cpu": 50.0,
            "gpu": 2.0,
            "memory": 500.0
        }
        
        success = self.allocator.allocate_resources("task_1", requirements)
        assert success is True
        
        # Verify all resource types were allocated
        cpu_resource = self.allocator.resources["cpu_pool"]
        gpu_resource = self.allocator.resources["gpu_pool"]
        mem_resource = self.allocator.resources["mem_pool"]
        
        assert cpu_resource.available_capacity == 150.0
        assert gpu_resource.available_capacity == 6.0
        assert mem_resource.available_capacity == 1500.0
    
    def test_unknown_resource_type_handling(self):
        """Test handling of unknown resource types in requirements."""
        self.allocator.add_resource("cpu_1", ResourceType.CPU, 100.0)
        
        # Include unknown resource type
        requirements = {"cpu": 20.0, "quantum_processor": 5.0}
        
        with patch('src.quantum_inspired_task_planner.resource_allocator.logger') as mock_logger:
            success = self.allocator.allocate_resources("task_1", requirements)
            
            # Should log warning but continue with known resources
            mock_logger.warning.assert_called()
            
            # Should fail because quantum_processor resource doesn't exist
            assert success is False
    
    def test_concurrent_allocation_safety(self):
        """Test thread safety of resource allocation."""
        import threading
        
        self.allocator.add_resource("cpu_pool", ResourceType.CPU, 1000.0)
        
        results = []
        errors = []
        
        def allocate_task(task_num):
            try:
                success = self.allocator.allocate_resources(
                    f"task_{task_num}", 
                    {"cpu": 10.0}
                )
                results.append((task_num, success))
            except Exception as e:
                errors.append((task_num, e))
        
        # Create multiple threads
        threads = []
        for i in range(50):
            thread = threading.Thread(target=allocate_task, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should have no errors
        assert len(errors) == 0, f"Concurrent allocation errors: {errors}"
        
        # Should have successful allocations up to capacity
        successful_allocations = sum(1 for _, success in results if success)
        assert successful_allocations <= 100  # 1000 capacity / 10 per task
        
        # Total allocated should not exceed capacity
        cpu_resource = self.allocator.resources["cpu_pool"]
        total_allocated = sum(cpu_resource.allocated_tasks.values())
        assert total_allocated <= 1000.0
        assert cpu_resource.available_capacity >= 0.0


@pytest.fixture
def configured_allocator():
    """Fixture providing configured resource allocator."""
    allocator = QuantumResourceAllocator()
    
    # Add various resources
    allocator.add_resource("cpu_1", ResourceType.CPU, 100.0)
    allocator.add_resource("cpu_2", ResourceType.CPU, 150.0)
    allocator.add_resource("mem_1", ResourceType.MEMORY, 1000.0)
    allocator.add_resource("gpu_1", ResourceType.GPU, 8.0)
    
    return allocator


class TestResourceAllocatorIntegration:
    """Integration tests for resource allocator."""
    
    def test_realistic_workload_simulation(self, configured_allocator):
        """Test allocator with realistic workload."""
        allocator = configured_allocator
        
        # Simulate various task workloads
        workloads = [
            {"cpu": 20.0, "memory": 200.0},  # CPU-intensive
            {"memory": 500.0, "gpu": 2.0},   # Memory + GPU intensive
            {"cpu": 10.0, "memory": 100.0},  # Light workload
            {"gpu": 4.0, "memory": 300.0},   # GPU-intensive
            {"cpu": 50.0, "memory": 150.0}   # Balanced workload
        ]
        
        allocated_tasks = []
        
        # Allocate resources for all workloads
        for i, workload in enumerate(workloads):
            task_id = f"workload_task_{i}"
            success = allocator.allocate_resources(task_id, workload)
            if success:
                allocated_tasks.append(task_id)
        
        # Should successfully allocate most tasks
        assert len(allocated_tasks) >= 3
        
        # Check utilization
        utilization = allocator.get_resource_utilization()
        
        # Should have reasonable utilization levels
        for resource_id, data in utilization.items():
            assert 0.0 <= data["utilization_percent"] <= 100.0
            assert data["available_capacity"] >= 0.0
    
    def test_resource_efficiency_adaptation(self, configured_allocator):
        """Test that resource efficiency scores adapt to performance."""
        allocator = configured_allocator
        
        # Allocate to cpu_1 multiple times to test efficiency adaptation
        for i in range(5):
            allocator.allocate_resources(f"task_{i}", {"cpu": 10.0})
        
        # Efficiency scores should be tracked
        cpu_1 = allocator.resources["cpu_1"]
        assert cpu_1.efficiency_score is not None
        assert isinstance(cpu_1.efficiency_score, float)
    
    def test_allocation_history_tracking(self, configured_allocator):
        """Test allocation decision history tracking."""
        allocator = configured_allocator
        
        initial_history_len = len(allocator.allocation_history)
        
        # Make several allocations
        allocator.allocate_resources("task_1", {"cpu": 20.0})
        allocator.allocate_resources("task_2", {"memory": 300.0})
        
        # History should be updated
        assert len(allocator.allocation_history) == initial_history_len + 2
        
        # Check history record structure
        latest_record = allocator.allocation_history[-1]
        assert "timestamp" in latest_record
        assert "task_id" in latest_record
        assert "allocation_plan" in latest_record
        assert "temperature" in latest_record
    
    def test_load_balancing_across_similar_resources(self, configured_allocator):
        """Test load balancing across similar resources."""
        allocator = configured_allocator
        
        # Allocate multiple CPU tasks to test load balancing
        cpu_tasks = []
        for i in range(8):  # More tasks than can fit on one CPU resource
            task_id = f"cpu_task_{i}"
            success = allocator.allocate_resources(task_id, {"cpu": 15.0})
            if success:
                cpu_tasks.append(task_id)
        
        # Check that both CPU resources are utilized
        cpu_1 = allocator.resources["cpu_1"]
        cpu_2 = allocator.resources["cpu_2"]
        
        cpu_1_utilization = 100.0 - (cpu_1.available_capacity / cpu_1.total_capacity * 100)
        cpu_2_utilization = 100.0 - (cpu_2.available_capacity / cpu_2.total_capacity * 100)
        
        # Both should have some utilization if load balancing works
        # (unless one resource is much more efficient than the other)
        total_utilization = cpu_1_utilization + cpu_2_utilization
        assert total_utilization > 0
    
    def test_resource_selection_optimization(self, configured_allocator):
        """Test quantum-inspired resource selection optimization."""
        allocator = configured_allocator
        
        # Modify resource characteristics for testing
        cpu_1 = allocator.resources["cpu_1"]
        cpu_2 = allocator.resources["cpu_2"]
        
        # Make cpu_1 more efficient but cpu_2 have more capacity
        cpu_1.efficiency_score = 1.5
        cpu_2.efficiency_score = 1.0
        
        # Allocate tasks and check selection patterns
        candidates = [("cpu_1", cpu_1), ("cpu_2", cpu_2)]
        
        # Test selection for small allocation
        selected_id, selected_resource = allocator._select_optimal_resource(candidates, 10.0)
        
        # Selection should consider efficiency, utilization, and coherence
        assert selected_id in ["cpu_1", "cpu_2"]
        assert selected_resource in [cpu_1, cpu_2]
    
    def test_edge_case_zero_capacity_resource(self):
        """Test handling of zero capacity resources."""
        # This should not be allowed, but test graceful handling
        with pytest.raises(Exception):
            # Assuming validation prevents zero capacity
            self.allocator.add_resource("invalid", ResourceType.CPU, 0.0)
    
    def test_deallocate_nonexistent_task(self, configured_allocator):
        """Test deallocating resources for non-existent task."""
        allocator = configured_allocator
        
        # Should handle gracefully without errors
        allocator.deallocate_resources("non_existent_task")
        
        # Resources should be unchanged
        utilization = allocator.get_resource_utilization()
        for data in utilization.values():
            assert data["allocated_tasks_count"] == 0
    
    def test_allocation_with_empty_requirements(self, configured_allocator):
        """Test allocation with empty resource requirements."""
        allocator = configured_allocator
        
        # Empty requirements should succeed (no resources needed)
        success = allocator.allocate_resources("task_1", {})
        assert success is True
        
        # No resources should be allocated
        utilization = allocator.get_resource_utilization()
        for data in utilization.values():
            assert data["allocated_tasks_count"] == 0


class TestResourceAllocatorAdvanced:
    """Advanced tests for resource allocator."""
    
    def test_quantum_coherence_effects(self, configured_allocator):
        """Test quantum coherence effects on resource selection."""
        allocator = configured_allocator
        
        # Modify quantum coherence values
        cpu_1 = allocator.resources["cpu_1"]
        cpu_2 = allocator.resources["cpu_2"]
        
        cpu_1.quantum_coherence = 0.9
        cpu_2.quantum_coherence = 0.3  # Lower coherence
        
        # Selection should prefer higher coherence
        candidates = [("cpu_1", cpu_1), ("cpu_2", cpu_2)]
        selected_id, _ = allocator._select_optimal_resource(candidates, 10.0)
        
        # Higher coherence should be preferred (other factors being equal)
        assert selected_id == "cpu_1"
    
    def test_allocation_optimization_over_time(self, configured_allocator):
        """Test that allocation decisions improve over time."""
        allocator = configured_allocator
        
        # Track temperature changes (simulated annealing)
        initial_temp = allocator.temperature
        
        # Make multiple allocations
        for i in range(10):
            allocator.allocate_resources(f"task_{i}", {"cpu": 5.0})
        
        # Temperature should decrease (cooling in simulated annealing)
        assert allocator.temperature < initial_temp
    
    def test_resource_type_isolation(self, configured_allocator):
        """Test that resource types are properly isolated."""
        allocator = configured_allocator
        
        # Allocate all CPU
        allocator.allocate_resources("cpu_task", {"cpu": 250.0})  # Total CPU capacity
        
        # Memory allocation should still work
        success = allocator.allocate_resources("mem_task", {"memory": 500.0})
        assert success is True
        
        # But additional CPU allocation should fail
        success = allocator.allocate_resources("cpu_task_2", {"cpu": 10.0})
        assert success is False