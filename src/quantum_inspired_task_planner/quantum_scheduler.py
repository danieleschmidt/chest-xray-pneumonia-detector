"""Quantum-inspired task scheduling engine.

This module implements quantum computing concepts like superposition and entanglement
for advanced task scheduling and resource allocation.
"""

import math
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime, timedelta
import json


class TaskPriority(Enum):
    """Task priority levels using quantum-inspired energy states."""
    CRITICAL = 4  # High energy state
    HIGH = 3
    MEDIUM = 2
    LOW = 1       # Ground state


class TaskStatus(Enum):
    """Task status states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


@dataclass
class QuantumTask:
    """Represents a task in quantum superposition with multiple possible states."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    dependencies: Set[str] = field(default_factory=set)
    estimated_duration: timedelta = field(default_factory=lambda: timedelta(hours=1))
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Quantum-inspired properties
    probability_amplitude: complex = complex(1.0, 0.0)
    entangled_tasks: Set[str] = field(default_factory=set)
    superposition_weight: float = 1.0


class QuantumScheduler:
    """Quantum-inspired task scheduler using superposition and entanglement principles."""
    
    def __init__(self, max_parallel_tasks: int = 4):
        self.tasks: Dict[str, QuantumTask] = {}
        self.max_parallel_tasks = max_parallel_tasks
        self.running_tasks: Set[str] = set()
        self.completed_tasks: Set[str] = set()
        self.quantum_state_matrix = [[1.0]]
        
    def add_task(self, task: QuantumTask) -> str:
        """Add a new task to the quantum scheduling system."""
        self.tasks[task.id] = task
        self._update_quantum_state()
        return task.id
    
    def create_task(self, name: str, description: str = "", priority: TaskPriority = TaskPriority.MEDIUM,
                   dependencies: Optional[List[str]] = None,
                   estimated_duration: Optional[timedelta] = None,
                   resource_requirements: Optional[Dict[str, float]] = None) -> str:
        """Create and add a new task to the scheduler."""
        task = QuantumTask(
            name=name,
            description=description,
            priority=priority,
            dependencies=set(dependencies or []),
            estimated_duration=estimated_duration or timedelta(hours=1),
            resource_requirements=resource_requirements or {}
        )
        return self.add_task(task)
    
    def get_task(self, task_id: str) -> Optional[QuantumTask]:
        """Retrieve a task by ID."""
        return self.tasks.get(task_id)
    
    def _update_quantum_state(self) -> None:
        """Update the quantum state matrix based on current tasks."""
        n_tasks = len(self.tasks)
        if n_tasks == 0:
            self.quantum_state_matrix = [[1.0]]
            return
        
        # Create quantum state matrix with task interactions
        self.quantum_state_matrix = [[complex(0.0, 0.0) for _ in range(n_tasks)] for _ in range(n_tasks)]
        task_ids = list(self.tasks.keys())
        
        for i, task_id in enumerate(task_ids):
            task = self.tasks[task_id]
            # Diagonal elements represent task probability amplitudes
            self.quantum_state_matrix[i][i] = task.probability_amplitude
            
            # Off-diagonal elements represent entanglement between tasks
            for j, other_task_id in enumerate(task_ids):
                if i != j and other_task_id in task.entangled_tasks:
                    self.quantum_state_matrix[i][j] = complex(0.5, 0.0)
    
    def _calculate_priority_score(self, task: QuantumTask) -> float:
        """Calculate quantum-inspired priority score for task scheduling."""
        base_score = task.priority.value
        
        # Apply quantum superposition weighting
        superposition_bonus = task.superposition_weight * 0.5
        
        # Consider entanglement effects
        entanglement_factor = 1.0 + (len(task.entangled_tasks) * 0.1)
        
        # Time urgency factor
        age_hours = (datetime.now() - task.created_at).total_seconds() / 3600
        urgency_factor = 1.0 + (age_hours * 0.01)
        
        return base_score * entanglement_factor * urgency_factor + superposition_bonus
    
    def get_next_tasks(self) -> List[QuantumTask]:
        """Get the next optimal tasks to execute using quantum-inspired selection."""
        available_tasks = []
        
        for task in self.tasks.values():
            if (task.status == TaskStatus.PENDING and 
                self._are_dependencies_satisfied(task) and
                len(self.running_tasks) < self.max_parallel_tasks):
                available_tasks.append(task)
        
        if not available_tasks:
            return []
        
        # Sort by quantum-inspired priority score
        available_tasks.sort(key=self._calculate_priority_score, reverse=True)
        
        # Select tasks considering quantum parallelism constraints
        selected_tasks = []
        slots_available = self.max_parallel_tasks - len(self.running_tasks)
        
        for task in available_tasks[:slots_available]:
            selected_tasks.append(task)
        
        return selected_tasks
    
    def _are_dependencies_satisfied(self, task: QuantumTask) -> bool:
        """Check if all task dependencies are completed."""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        return True
    
    def start_task(self, task_id: str) -> bool:
        """Start execution of a task."""
        task = self.tasks.get(task_id)
        if not task or task.status != TaskStatus.PENDING:
            return False
        
        if not self._are_dependencies_satisfied(task):
            return False
        
        if len(self.running_tasks) >= self.max_parallel_tasks:
            return False
        
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        self.running_tasks.add(task_id)
        self._update_quantum_state()
        return True
    
    def complete_task(self, task_id: str) -> bool:
        """Mark a task as completed."""
        task = self.tasks.get(task_id)
        if not task or task.status != TaskStatus.RUNNING:
            return False
        
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        self.running_tasks.discard(task_id)
        self.completed_tasks.add(task_id)
        self._update_quantum_state()
        
        # Apply quantum entanglement effects
        self._process_entanglement_effects(task)
        return True
    
    def _process_entanglement_effects(self, completed_task: QuantumTask) -> None:
        """Process quantum entanglement effects when a task completes."""
        for entangled_task_id in completed_task.entangled_tasks:
            entangled_task = self.tasks.get(entangled_task_id)
            if entangled_task and entangled_task.status == TaskStatus.PENDING:
                # Boost priority of entangled tasks
                entangled_task.superposition_weight *= 1.2
    
    def get_task_statistics(self) -> Dict[str, int]:
        """Get current task statistics."""
        stats = {status.value: 0 for status in TaskStatus}
        for task in self.tasks.values():
            stats[task.status.value] += 1
        return stats
    
    def create_entanglement(self, task_id1: str, task_id2: str) -> bool:
        """Create quantum entanglement between two tasks."""
        task1 = self.tasks.get(task_id1)
        task2 = self.tasks.get(task_id2)
        
        if not task1 or not task2:
            return False
        
        task1.entangled_tasks.add(task_id2)
        task2.entangled_tasks.add(task_id1)
        self._update_quantum_state()
        return True
    
    def optimize_schedule(self) -> List[Tuple[str, float]]:
        """Generate optimized schedule using quantum-inspired algorithms."""
        pending_tasks = [task for task in self.tasks.values() 
                        if task.status == TaskStatus.PENDING]
        
        # Calculate optimal execution order using quantum interference patterns
        schedule = []
        for task in pending_tasks:
            if self._are_dependencies_satisfied(task):
                score = self._calculate_priority_score(task)
                schedule.append((task.id, score))
        
        # Sort by quantum score
        schedule.sort(key=lambda x: x[1], reverse=True)
        return schedule
    
    def export_state(self) -> str:
        """Export current scheduler state as JSON."""
        state = {
            "tasks": {},
            "running_tasks": list(self.running_tasks),
            "completed_tasks": list(self.completed_tasks),
            "max_parallel_tasks": self.max_parallel_tasks
        }
        
        for task_id, task in self.tasks.items():
            state["tasks"][task_id] = {
                "id": task.id,
                "name": task.name,
                "description": task.description,
                "priority": task.priority.value,
                "status": task.status.value,
                "dependencies": list(task.dependencies),
                "estimated_duration": task.estimated_duration.total_seconds(),
                "resource_requirements": task.resource_requirements,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "probability_amplitude": [task.probability_amplitude.real, task.probability_amplitude.imag],
                "entangled_tasks": list(task.entangled_tasks),
                "superposition_weight": task.superposition_weight
            }
        
        return json.dumps(state, indent=2)
    
    def import_state(self, state_json: str) -> None:
        """Import scheduler state from JSON."""
        state = json.loads(state_json)
        
        self.tasks.clear()
        self.running_tasks = set(state["running_tasks"])
        self.completed_tasks = set(state["completed_tasks"])
        self.max_parallel_tasks = state["max_parallel_tasks"]
        
        for task_data in state["tasks"].values():
            task = QuantumTask(
                id=task_data["id"],
                name=task_data["name"],
                description=task_data["description"],
                priority=TaskPriority(task_data["priority"]),
                status=TaskStatus(task_data["status"]),
                dependencies=set(task_data["dependencies"]),
                estimated_duration=timedelta(seconds=task_data["estimated_duration"]),
                resource_requirements=task_data["resource_requirements"],
                created_at=datetime.fromisoformat(task_data["created_at"]),
                started_at=datetime.fromisoformat(task_data["started_at"]) if task_data["started_at"] else None,
                completed_at=datetime.fromisoformat(task_data["completed_at"]) if task_data["completed_at"] else None,
                probability_amplitude=complex(task_data["probability_amplitude"][0], task_data["probability_amplitude"][1]),
                entangled_tasks=set(task_data["entangled_tasks"]),
                superposition_weight=task_data["superposition_weight"]
            )
            self.tasks[task.id] = task
        
        self._update_quantum_state()