"""Quantum-inspired resource allocation engine.

Uses quantum annealing principles to optimize resource distribution
across multiple tasks and computing resources.
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of computing resources."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"


@dataclass
class Resource:
    """Represents a computing resource with quantum-inspired allocation."""
    
    type: ResourceType
    total_capacity: float
    available_capacity: float
    allocated_tasks: Dict[str, float]  # task_id -> allocated_amount
    efficiency_score: float = 1.0
    quantum_coherence: float = 1.0  # Quantum state coherence for resource


class QuantumResourceAllocator:
    """Quantum-inspired resource allocator using annealing optimization."""
    
    def __init__(self):
        self.resources: Dict[str, Resource] = {}
        self.allocation_history: List[Dict] = []
        self.temperature = 1.0  # Simulated annealing temperature
        
    def add_resource(self, resource_id: str, resource_type: ResourceType, 
                    total_capacity: float) -> None:
        """Add a new resource to the allocation system."""
        self.resources[resource_id] = Resource(
            type=resource_type,
            total_capacity=total_capacity,
            available_capacity=total_capacity,
            allocated_tasks={}
        )
        logger.info(f"Added resource {resource_id} with capacity {total_capacity}")
    
    def allocate_resources(self, task_id: str, requirements: Dict[str, float]) -> bool:
        """Allocate resources to a task using quantum optimization."""
        allocation_plan = self._optimize_allocation(task_id, requirements)
        
        if not allocation_plan:
            return False
        
        # Apply the allocation
        for resource_id, amount in allocation_plan.items():
            resource = self.resources[resource_id]
            resource.allocated_tasks[task_id] = amount
            resource.available_capacity -= amount
            
        self._record_allocation(task_id, allocation_plan)
        return True
    
    def _optimize_allocation(self, task_id: str, requirements: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Use quantum annealing to find optimal resource allocation."""
        allocation_plan = {}
        
        for resource_type_str, required_amount in requirements.items():
            try:
                resource_type = ResourceType(resource_type_str)
            except ValueError:
                logger.warning(f"Unknown resource type: {resource_type_str}")
                continue
            
            # Find resources of this type
            candidate_resources = [
                (res_id, res) for res_id, res in self.resources.items()
                if res.type == resource_type and res.available_capacity >= required_amount
            ]
            
            if not candidate_resources:
                logger.warning(f"Insufficient {resource_type.value} resources for task {task_id}")
                return None
            
            # Select optimal resource using quantum-inspired scoring
            best_resource_id, best_resource = self._select_optimal_resource(
                candidate_resources, required_amount
            )
            allocation_plan[best_resource_id] = required_amount
        
        return allocation_plan
    
    def _select_optimal_resource(self, candidates: List[Tuple[str, Resource]], 
                               required_amount: float) -> Tuple[str, Resource]:
        """Select optimal resource using quantum-inspired scoring."""
        scored_candidates = []
        
        for resource_id, resource in candidates:
            # Quantum-inspired scoring function
            utilization_factor = 1.0 - (resource.available_capacity / resource.total_capacity)
            efficiency_score = resource.efficiency_score
            coherence_bonus = resource.quantum_coherence
            
            # Prefer balanced utilization (quantum superposition principle)
            balance_score = 1.0 - abs(0.5 - utilization_factor)
            
            total_score = efficiency_score * coherence_bonus * balance_score
            scored_candidates.append((total_score, resource_id, resource))
        
        # Select highest scoring resource
        scored_candidates.sort(reverse=True)
        return scored_candidates[0][1], scored_candidates[0][2]
    
    def deallocate_resources(self, task_id: str) -> None:
        """Release resources allocated to a completed task."""
        for resource in self.resources.values():
            if task_id in resource.allocated_tasks:
                allocated_amount = resource.allocated_tasks.pop(task_id)
                resource.available_capacity += allocated_amount
                logger.info(f"Released {allocated_amount} units of {resource.type.value} from task {task_id}")
    
    def _record_allocation(self, task_id: str, allocation_plan: Dict[str, float]) -> None:
        """Record allocation decision for learning and optimization."""
        record = {
            "timestamp": datetime.now(),
            "task_id": task_id,
            "allocation_plan": allocation_plan,
            "temperature": self.temperature
        }
        self.allocation_history.append(record)
        
        # Update temperature for simulated annealing
        self.temperature *= 0.99
    
    def get_resource_utilization(self) -> Dict[str, Dict[str, float]]:
        """Get current resource utilization statistics."""
        utilization = {}
        
        for resource_id, resource in self.resources.items():
            utilization[resource_id] = {
                "type": resource.type.value,
                "total_capacity": resource.total_capacity,
                "available_capacity": resource.available_capacity,
                "utilization_percent": ((resource.total_capacity - resource.available_capacity) / 
                                      resource.total_capacity) * 100,
                "allocated_tasks_count": len(resource.allocated_tasks),
                "efficiency_score": resource.efficiency_score,
                "quantum_coherence": resource.quantum_coherence
            }
        
        return utilization
    
    def rebalance_allocations(self) -> int:
        """Rebalance resource allocations using quantum optimization."""
        rebalanced_count = 0
        
        # Collect all current allocations
        current_allocations = {}
        for resource_id, resource in self.resources.items():
            current_allocations[resource_id] = resource.allocated_tasks.copy()
        
        # Apply quantum-inspired rebalancing algorithm
        for resource_type in ResourceType:
            type_resources = [
                (res_id, res) for res_id, res in self.resources.items()
                if res.type == resource_type
            ]
            
            if len(type_resources) < 2:
                continue
            
            # Check for imbalanced utilization
            utilizations = [
                (res.total_capacity - res.available_capacity) / res.total_capacity
                for _, res in type_resources
            ]
            
            if max(utilizations) - min(utilizations) > 0.3:  # 30% imbalance threshold
                rebalanced_count += self._rebalance_resource_type(type_resources)
        
        return rebalanced_count
    
    def _rebalance_resource_type(self, resources: List[Tuple[str, Resource]]) -> int:
        """Rebalance a specific resource type using quantum principles."""
        # For now, implement simple load balancing
        # In a full implementation, this would use quantum annealing
        rebalanced = 0
        
        # Sort by current utilization
        resources.sort(key=lambda x: (x[1].total_capacity - x[1].available_capacity) / x[1].total_capacity)
        
        overloaded_resources = resources[-len(resources)//2:]  # Top half by utilization
        underloaded_resources = resources[:len(resources)//2]  # Bottom half
        
        for overloaded_id, overloaded_res in overloaded_resources:
            if not overloaded_res.allocated_tasks:
                continue
                
            for underloaded_id, underloaded_res in underloaded_resources:
                if underloaded_res.available_capacity > 0:
                    # Move one task allocation
                    task_id = next(iter(overloaded_res.allocated_tasks))
                    amount = overloaded_res.allocated_tasks[task_id]
                    
                    if underloaded_res.available_capacity >= amount:
                        # Transfer allocation
                        del overloaded_res.allocated_tasks[task_id]
                        overloaded_res.available_capacity += amount
                        
                        underloaded_res.allocated_tasks[task_id] = amount
                        underloaded_res.available_capacity -= amount
                        
                        rebalanced += 1
                        logger.info(f"Rebalanced task {task_id} from {overloaded_id} to {underloaded_id}")
                        break
        
        return rebalanced