"""Quantum-inspired task scheduling engine.

This module implements quantum computing concepts like superposition and entanglement
for advanced task scheduling and resource allocation.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..caching.intelligent_cache import IntelligentCache
from .error_handling import (
    ErrorCategory,
    ErrorSeverity,
    SchedulingException,
    quantum_error_handler,
)

logger = logging.getLogger(__name__)


@dataclass
class QuantumMetrics:
    """Metrics for quantum scheduler performance monitoring."""
    total_tasks: int = 0
    running_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_completion_time: float = 0.0
    quantum_state_updates: int = 0
    entanglement_operations: int = 0
    scheduling_operations: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


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
        self._quantum_state_matrix = np.array([[1.0]], dtype=complex)
        self._quantum_state_dirty = False
        self.metrics = QuantumMetrics()
        self.failed_tasks: Set[str] = set()

        # Caching for performance optimization
        self._priority_cache = IntelligentCache(max_entries=1000, default_ttl_seconds=300)
        self._dependency_cache = IntelligentCache(max_entries=500, default_ttl_seconds=600)
        self._schedule_cache = IntelligentCache(max_entries=100, default_ttl_seconds=60)

        # Auto-scaling parameters
        self._min_parallel_tasks = max(1, max_parallel_tasks // 4)
        self._max_parallel_tasks = max_parallel_tasks * 4
        self._scale_up_threshold = 0.8  # Scale up when 80% capacity used
        self._scale_down_threshold = 0.3  # Scale down when 30% capacity used
        self._last_scale_time = datetime.now()
        self._scale_cooldown_seconds = 60  # Minimum time between scaling operations

    @property
    def quantum_state_matrix(self):
        """Get the current quantum state matrix, updating if necessary."""
        self._update_quantum_state()
        return self._quantum_state_matrix

    def add_task(self, task: QuantumTask) -> str:
        """Add a new task to the quantum scheduling system."""
        self.tasks[task.id] = task
        self._quantum_state_dirty = True
        self._update_metrics()
        return task.id

    @quantum_error_handler(ErrorCategory.VALIDATION, "task_creation", ErrorSeverity.MEDIUM)
    def create_task(self, name: str, description: str = "", priority: TaskPriority = TaskPriority.MEDIUM,
                   dependencies: Optional[List[str]] = None,
                   estimated_duration: Optional[timedelta] = None,
                   resource_requirements: Optional[Dict[str, float]] = None) -> str:
        """Create and add a new task to the scheduler."""
        # Input validation
        if not name or not isinstance(name, str):
            raise ValueError("Task name must be a non-empty string")
        if len(name) > 200:
            raise ValueError("Task name must be less than 200 characters")
        if dependencies:
            for dep_id in dependencies:
                if dep_id not in self.tasks:
                    raise ValueError(f"Dependency task {dep_id} does not exist")

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
        if not self._quantum_state_dirty:
            return

        n_tasks = len(self.tasks)
        if n_tasks == 0:
            self._quantum_state_matrix = np.array([[1.0]], dtype=complex)
            self._quantum_state_dirty = False
            return

        # Create quantum state matrix with task interactions
        self._quantum_state_matrix = np.zeros((n_tasks, n_tasks), dtype=complex)
        task_ids = list(self.tasks.keys())

        for i, task_id in enumerate(task_ids):
            task = self.tasks[task_id]
            # Diagonal elements represent task probability amplitudes
            self._quantum_state_matrix[i][i] = task.probability_amplitude

            # Off-diagonal elements represent entanglement between tasks
            for j, other_task_id in enumerate(task_ids):
                if i != j and other_task_id in task.entangled_tasks:
                    self._quantum_state_matrix[i][j] = complex(0.5, 0.0)

        self._quantum_state_dirty = False
        self.metrics.quantum_state_updates += 1

    def _calculate_priority_score(self, task: QuantumTask) -> float:
        """Calculate quantum-inspired priority score for task scheduling (cached)."""
        # Create cache key based on task state
        cache_key = f"priority_{task.id}_{task.status.value}_{len(task.entangled_tasks)}_{task.superposition_weight}"

        # Check cache first
        cached_score = self._priority_cache.get(cache_key)
        if cached_score is not None:
            return cached_score

        # Calculate score
        base_score = task.priority.value

        # Apply quantum superposition weighting
        superposition_bonus = task.superposition_weight * 0.5

        # Consider entanglement effects
        entanglement_factor = 1.0 + (len(task.entangled_tasks) * 0.1)

        # Time urgency factor
        age_hours = (datetime.now() - task.created_at).total_seconds() / 3600
        urgency_factor = 1.0 + (age_hours * 0.01)

        score = base_score * entanglement_factor * urgency_factor + superposition_bonus

        # Cache the result
        self._priority_cache.set(cache_key, score, ttl_seconds=120)  # Cache for 2 minutes
        return score

    def get_next_tasks(self) -> List[QuantumTask]:
        """Get the next optimal tasks to execute using quantum-inspired selection."""
        # Create cache key based on current state
        state_hash = hash((
            frozenset(self.running_tasks),
            frozenset(self.completed_tasks),
            len(self.tasks),
            self.max_parallel_tasks
        ))
        cache_key = f"next_tasks_{state_hash}"

        # Check cache first
        cached_result = self._schedule_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Try auto-scaling first (but not on every call to avoid performance overhead)
        current_time = datetime.now()
        if (current_time - self._last_scale_time).total_seconds() > 30:  # Only try scaling every 30 seconds
            self._auto_scale()

        available_tasks = []

        # Optimize for large task sets - early exit when we have enough tasks
        slots_available = self.max_parallel_tasks - len(self.running_tasks)
        if slots_available <= 0:
            result = []
            self._schedule_cache.set(cache_key, result, ttl_seconds=30)
            return result

        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING and self._are_dependencies_satisfied(task):
                available_tasks.append(task)
                # Early exit optimization for performance
                if len(available_tasks) >= slots_available * 3:  # Get 3x what we need for good selection
                    break

        if not available_tasks:
            result = []
            self._schedule_cache.set(cache_key, result, ttl_seconds=30)
            return result

        # Sort by quantum-inspired priority score
        available_tasks.sort(key=self._calculate_priority_score, reverse=True)

        # Select tasks considering quantum parallelism constraints
        selected_tasks = available_tasks[:slots_available]

        # Cache result
        self._schedule_cache.set(cache_key, selected_tasks, ttl_seconds=30)
        return selected_tasks

    def _are_dependencies_satisfied(self, task: QuantumTask) -> bool:
        """Check if all task dependencies are completed (cached)."""
        if not task.dependencies:
            return True

        # Create cache key based on dependencies and completed tasks
        deps_hash = hash(frozenset(task.dependencies))
        completed_hash = hash(frozenset(self.completed_tasks))
        cache_key = f"deps_{task.id}_{deps_hash}_{completed_hash}"

        cached_result = self._dependency_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Check dependencies
        result = all(dep_id in self.completed_tasks for dep_id in task.dependencies)

        # Cache result
        self._dependency_cache.set(cache_key, result, ttl_seconds=180)
        return result

    def start_task(self, task_id: str) -> bool:
        """Start execution of a task (graceful failure mode)."""
        try:
            return self.start_task_robust(task_id)
        except (ValueError, SchedulingException) as e:
            logger.warning(f"Failed to start task {task_id}: {e}")
            return False

    @quantum_error_handler(ErrorCategory.SCHEDULING, "start_task", ErrorSeverity.MEDIUM)
    def start_task_robust(self, task_id: str) -> bool:
        """Start execution of a task with robust error handling."""
        if not task_id or not isinstance(task_id, str):
            raise ValueError("Task ID must be a non-empty string")

        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        if task.status != TaskStatus.PENDING:
            raise ValueError(f"Task {task_id} is not in pending state (current: {task.status.value})")

        if not self._are_dependencies_satisfied(task):
            raise ValueError(f"Dependencies not satisfied for task {task_id}")

        if len(self.running_tasks) >= self.max_parallel_tasks:
            raise ValueError(f"Maximum parallel tasks limit ({self.max_parallel_tasks}) reached")

        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        self.running_tasks.add(task_id)
        self._quantum_state_dirty = True

        # Invalidate caches when task state changes
        self._invalidate_task_caches(task_id)

        logger.info(f"Started task {task_id}: {task.name}")
        return True

    def complete_task(self, task_id: str) -> bool:
        """Mark a task as completed (graceful failure mode)."""
        try:
            return self.complete_task_robust(task_id)
        except (ValueError, SchedulingException) as e:
            logger.warning(f"Failed to complete task {task_id}: {e}")
            return False

    @quantum_error_handler(ErrorCategory.SCHEDULING, "complete_task", ErrorSeverity.MEDIUM)
    def complete_task_robust(self, task_id: str) -> bool:
        """Mark a task as completed with robust error handling."""
        if not task_id or not isinstance(task_id, str):
            raise ValueError("Task ID must be a non-empty string")

        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        if task.status != TaskStatus.RUNNING:
            raise ValueError(f"Task {task_id} is not running (current: {task.status.value})")

        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        self.running_tasks.discard(task_id)
        self.completed_tasks.add(task_id)
        self._quantum_state_dirty = True

        # Invalidate caches when task state changes
        self._invalidate_task_caches(task_id)

        logger.info(f"Completed task {task_id}: {task.name}")

        # Apply quantum entanglement effects
        self._process_entanglement_effects(task)
        self._update_metrics()
        return True

    def _update_metrics(self) -> None:
        """Update performance metrics."""
        self.metrics.total_tasks = len(self.tasks)
        self.metrics.running_tasks = len(self.running_tasks)
        self.metrics.completed_tasks = len(self.completed_tasks)
        self.metrics.failed_tasks = len(self.failed_tasks)
        self.metrics.last_updated = datetime.now()

        # Calculate average completion time
        if self.completed_tasks:
            completion_times = []
            for task_id in self.completed_tasks:
                task = self.tasks.get(task_id)
                if task and task.started_at and task.completed_at:
                    duration = (task.completed_at - task.started_at).total_seconds()
                    completion_times.append(duration)

            if completion_times:
                self.metrics.average_completion_time = sum(completion_times) / len(completion_times)

    def _invalidate_task_caches(self, task_id: str) -> None:
        """Invalidate caches related to a specific task."""
        # Priority cache keys start with "priority_{task_id}"
        priority_keys = [key for key in self._priority_cache._entries.keys() if key.startswith(f"priority_{task_id}")]
        for key in priority_keys:
            self._priority_cache.delete(key)

        # Dependency cache keys contain the task_id
        dep_keys = [key for key in self._dependency_cache._entries.keys() if f"deps_{task_id}" in key]
        for key in dep_keys:
            self._dependency_cache.delete(key)

        # Clear schedule cache when any task changes
        self._schedule_cache.clear()

    def _auto_scale(self) -> bool:
        """Automatically adjust parallel task capacity based on demand."""
        # Check cooldown
        if (datetime.now() - self._last_scale_time).total_seconds() < self._scale_cooldown_seconds:
            return False

        # Calculate current utilization
        utilization = len(self.running_tasks) / max(1, self.max_parallel_tasks)

        # Count pending tasks that could run
        ready_tasks = sum(1 for task in self.tasks.values()
                         if task.status == TaskStatus.PENDING and self._are_dependencies_satisfied(task))

        scaled = False

        # Scale up if high utilization and pending work
        if (utilization >= self._scale_up_threshold and
            ready_tasks > 0 and
            self.max_parallel_tasks < self._max_parallel_tasks):
            new_capacity = min(self._max_parallel_tasks,
                              int(self.max_parallel_tasks * 1.5))
            old_capacity = self.max_parallel_tasks
            self.max_parallel_tasks = new_capacity
            logger.info(f"Auto-scaled UP: {old_capacity} -> {new_capacity} parallel tasks")
            scaled = True

        # Scale down if low utilization
        elif (utilization <= self._scale_down_threshold and
              self.max_parallel_tasks > self._min_parallel_tasks):
            new_capacity = max(self._min_parallel_tasks,
                              int(self.max_parallel_tasks * 0.7))
            old_capacity = self.max_parallel_tasks
            self.max_parallel_tasks = new_capacity
            logger.info(f"Auto-scaled DOWN: {old_capacity} -> {new_capacity} parallel tasks")
            scaled = True

        if scaled:
            self._last_scale_time = datetime.now()
            self.metrics.scheduling_operations += 1

        return scaled

    def _process_entanglement_effects(self, completed_task: QuantumTask) -> None:
        """Process quantum entanglement effects when a task completes."""
        for entangled_task_id in completed_task.entangled_tasks:
            entangled_task = self.tasks.get(entangled_task_id)
            if entangled_task and entangled_task.status == TaskStatus.PENDING:
                # Boost priority of entangled tasks
                entangled_task.superposition_weight *= 1.2
                self.metrics.entanglement_operations += 1

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

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        self._update_metrics()
        return {
            "total_tasks": self.metrics.total_tasks,
            "running_tasks": self.metrics.running_tasks,
            "completed_tasks": self.metrics.completed_tasks,
            "failed_tasks": self.metrics.failed_tasks,
            "completion_rate": self.metrics.completed_tasks / max(1, self.metrics.total_tasks),
            "average_completion_time_seconds": self.metrics.average_completion_time,
            "quantum_state_updates": self.metrics.quantum_state_updates,
            "entanglement_operations": self.metrics.entanglement_operations,
            "scheduling_operations": self.metrics.scheduling_operations,
            "last_updated": self.metrics.last_updated.isoformat(),
            "system_health": self._assess_system_health()
        }

    def _assess_system_health(self) -> str:
        """Assess overall system health based on metrics."""
        # Simple health assessment logic
        completion_rate = self.metrics.completed_tasks / max(1, self.metrics.total_tasks)

        if completion_rate >= 0.9 and self.metrics.failed_tasks == 0:
            return "healthy"
        elif completion_rate >= 0.7 and self.metrics.failed_tasks < 3:
            return "degraded"
        else:
            return "unhealthy"

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
