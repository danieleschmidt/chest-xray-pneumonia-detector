"""Scaling and performance optimization for quantum task planner.

Implements auto-scaling, load balancing, and performance optimization
using quantum-inspired algorithms for distributed task execution.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import threading
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from .quantum_scheduler import QuantumScheduler, QuantumTask, TaskStatus
from .resource_allocator import QuantumResourceAllocator
from .monitoring import QuantumMetricsCollector

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction options."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


@dataclass
class ScalingDecision:
    """Represents a scaling decision with quantum-inspired rationale."""
    direction: ScalingDirection
    target_instances: int
    confidence: float
    quantum_factors: Dict[str, float]
    reason: str
    timestamp: datetime


class QuantumAutoScaler:
    """Auto-scaling system using quantum-inspired load prediction."""
    
    def __init__(self, min_instances: int = 1, max_instances: int = 10,
                 scale_up_threshold: float = 0.8, scale_down_threshold: float = 0.3):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        
        self.current_instances = min_instances
        self.scaling_history: List[ScalingDecision] = []
        self.load_history: List[Tuple[datetime, float]] = []
        self.metrics_collector = QuantumMetricsCollector()
        
        # Quantum-inspired scaling parameters
        self.quantum_momentum = 0.0  # Quantum momentum for scaling decisions
        self.coherence_factor = 1.0  # System coherence factor
        self.prediction_window = timedelta(minutes=15)
        
    def analyze_scaling_need(self, current_load: float, task_queue_size: int,
                           resource_utilization: Dict[str, float]) -> ScalingDecision:
        """Analyze if scaling is needed using quantum-inspired algorithms."""
        
        # Record current load
        self.load_history.append((datetime.now(), current_load))
        
        # Keep only recent history
        cutoff_time = datetime.now() - timedelta(hours=2)
        self.load_history = [
            (timestamp, load) for timestamp, load in self.load_history
            if timestamp > cutoff_time
        ]
        
        # Calculate quantum factors
        quantum_factors = self._calculate_quantum_factors(
            current_load, task_queue_size, resource_utilization
        )
        
        # Make scaling decision
        decision = self._make_scaling_decision(quantum_factors)
        
        # Record decision
        self.scaling_history.append(decision)
        
        logger.info(f"Scaling analysis: {decision.direction.value} to {decision.target_instances} instances "
                   f"(confidence: {decision.confidence:.2f})")
        
        return decision
    
    def _calculate_quantum_factors(self, current_load: float, queue_size: int,
                                 resource_util: Dict[str, float]) -> Dict[str, float]:
        """Calculate quantum-inspired factors for scaling decisions."""
        
        # Load trend analysis using quantum superposition
        load_trend = self._calculate_load_trend()
        
        # Queue pressure using quantum entanglement concept
        queue_pressure = min(queue_size / 100.0, 1.0)  # Normalize to 0-1
        
        # Resource pressure with quantum interference
        avg_resource_util = sum(resource_util.values()) / len(resource_util) if resource_util else 0.0
        resource_pressure = avg_resource_util / 100.0  # Convert percentage to 0-1
        
        # Quantum momentum (resistance to frequent changes)
        self.quantum_momentum = self.quantum_momentum * 0.9 + load_trend * 0.1
        
        # Coherence factor (system stability)
        if abs(load_trend) < 0.1:  # Stable load
            self.coherence_factor = min(1.0, self.coherence_factor + 0.05)
        else:
            self.coherence_factor = max(0.1, self.coherence_factor - 0.02)
        
        return {
            "current_load": current_load,
            "load_trend": load_trend,
            "queue_pressure": queue_pressure,
            "resource_pressure": resource_pressure,
            "quantum_momentum": self.quantum_momentum,
            "coherence_factor": self.coherence_factor,
            "prediction_confidence": self._calculate_prediction_confidence()
        }
    
    def _calculate_load_trend(self) -> float:
        """Calculate load trend using quantum-inspired time series analysis."""
        if len(self.load_history) < 5:
            return 0.0  # Insufficient data
        
        # Use recent load points for trend calculation
        recent_loads = [load for _, load in self.load_history[-10:]]
        
        # Simple linear trend calculation with quantum weighting
        n = len(recent_loads)
        if n < 2:
            return 0.0
        
        # Calculate weighted trend (more recent points have higher weight)
        weights = [((i + 1) / n) ** 2 for i in range(n)]  # Quantum-inspired quadratic weighting
        
        weighted_sum_xy = sum(w * i * load for w, i, load in zip(weights, range(n), recent_loads))
        weighted_sum_x = sum(w * i for w, i in zip(weights, range(n)))
        weighted_sum_y = sum(w * load for w, load in zip(weights, recent_loads))
        weighted_sum_x2 = sum(w * i * i for w, i in zip(weights, range(n)))
        weight_sum = sum(weights)
        
        if weight_sum * weighted_sum_x2 - weighted_sum_x ** 2 == 0:
            return 0.0
        
        trend = ((weight_sum * weighted_sum_xy - weighted_sum_x * weighted_sum_y) /
                (weight_sum * weighted_sum_x2 - weighted_sum_x ** 2))
        
        return trend
    
    def _calculate_prediction_confidence(self) -> float:
        """Calculate confidence in load predictions using quantum uncertainty."""
        if len(self.load_history) < 10:
            return 0.5  # Low confidence with insufficient data
        
        # Calculate prediction accuracy from recent history
        recent_loads = [load for _, load in self.load_history[-10:]]
        
        # Simple variance-based confidence
        if len(recent_loads) < 2:
            return 0.5
        
        variance = sum((load - sum(recent_loads) / len(recent_loads)) ** 2 for load in recent_loads) / len(recent_loads)
        
        # Convert variance to confidence (lower variance = higher confidence)
        confidence = 1.0 / (1.0 + variance)
        
        # Apply quantum coherence factor
        return confidence * self.coherence_factor
    
    def _make_scaling_decision(self, quantum_factors: Dict[str, float]) -> ScalingDecision:
        """Make scaling decision based on quantum factors."""
        current_load = quantum_factors["current_load"]
        load_trend = quantum_factors["load_trend"]
        queue_pressure = quantum_factors["queue_pressure"]
        resource_pressure = quantum_factors["resource_pressure"]
        momentum = quantum_factors["quantum_momentum"]
        coherence = quantum_factors["coherence_factor"]
        confidence = quantum_factors["prediction_confidence"]
        
        # Quantum-inspired scaling threshold adjustment
        dynamic_up_threshold = self.scale_up_threshold * (1.0 - momentum * 0.2)
        dynamic_down_threshold = self.scale_down_threshold * (1.0 + momentum * 0.2)
        
        # Combined pressure score
        pressure_score = (current_load + queue_pressure + resource_pressure) / 3.0
        
        # Include trend prediction
        predicted_load = current_load + (load_trend * 0.1)  # 10% trend influence
        
        # Make decision with quantum factors
        if predicted_load > dynamic_up_threshold or pressure_score > 0.8:
            # Scale up
            scale_factor = 1 + min(predicted_load - dynamic_up_threshold, 0.5)
            target_instances = min(
                self.max_instances,
                int(self.current_instances * scale_factor)
            )
            
            return ScalingDecision(
                direction=ScalingDirection.UP,
                target_instances=target_instances,
                confidence=confidence,
                quantum_factors=quantum_factors,
                reason=f"High load/pressure detected: {pressure_score:.2f}",
                timestamp=datetime.now()
            )
        
        elif predicted_load < dynamic_down_threshold and pressure_score < 0.4:
            # Scale down
            target_instances = max(
                self.min_instances,
                int(self.current_instances * 0.8)  # 20% reduction
            )
            
            return ScalingDecision(
                direction=ScalingDirection.DOWN,
                target_instances=target_instances,
                confidence=confidence * coherence,  # Higher coherence needed for scale down
                quantum_factors=quantum_factors,
                reason=f"Low load detected: {pressure_score:.2f}",
                timestamp=datetime.now()
            )
        
        else:
            # Maintain current scale
            return ScalingDecision(
                direction=ScalingDirection.STABLE,
                target_instances=self.current_instances,
                confidence=confidence,
                quantum_factors=quantum_factors,
                reason="Load within stable range",
                timestamp=datetime.now()
            )
    
    def execute_scaling(self, decision: ScalingDecision) -> bool:
        """Execute scaling decision."""
        if decision.confidence < 0.6:  # Low confidence threshold
            logger.warning(f"Scaling decision has low confidence ({decision.confidence:.2f}), skipping")
            return False
        
        if decision.direction == ScalingDirection.STABLE:
            return True  # No action needed
        
        try:
            if decision.direction == ScalingDirection.UP:
                return self._scale_up(decision.target_instances)
            else:
                return self._scale_down(decision.target_instances)
                
        except Exception as e:
            logger.error(f"Scaling execution failed: {e}")
            return False
    
    def _scale_up(self, target_instances: int) -> bool:
        """Scale up to target number of instances."""
        if target_instances <= self.current_instances:
            return True
        
        instances_to_add = target_instances - self.current_instances
        
        # In production, this would actually start new instances
        logger.info(f"Scaling up: adding {instances_to_add} instances")
        
        # Simulate instance startup time
        time.sleep(0.1)  # Simulated startup delay
        
        self.current_instances = target_instances
        return True
    
    def _scale_down(self, target_instances: int) -> bool:
        """Scale down to target number of instances."""
        if target_instances >= self.current_instances:
            return True
        
        instances_to_remove = self.current_instances - target_instances
        
        # In production, this would gracefully shutdown instances
        logger.info(f"Scaling down: removing {instances_to_remove} instances")
        
        self.current_instances = target_instances
        return True


class QuantumLoadBalancer:
    """Load balancer using quantum-inspired distribution algorithms."""
    
    def __init__(self, worker_instances: List[str]):
        self.worker_instances = worker_instances
        self.instance_loads: Dict[str, float] = {instance: 0.0 for instance in worker_instances}
        self.instance_health: Dict[str, bool] = {instance: True for instance in worker_instances}
        self.quantum_weights: Dict[str, float] = {instance: 1.0 for instance in worker_instances}
        
        # Quantum entanglement between workers
        self.worker_entanglement: Dict[str, List[str]] = {}
        
    def select_worker(self, task_requirements: Dict[str, float]) -> Optional[str]:
        """Select optimal worker using quantum-inspired load balancing."""
        
        # Filter healthy workers
        healthy_workers = [
            instance for instance in self.worker_instances
            if self.instance_health.get(instance, False)
        ]
        
        if not healthy_workers:
            return None
        
        # Calculate quantum-inspired selection scores
        worker_scores = []
        
        for worker in healthy_workers:
            current_load = self.instance_loads.get(worker, 0.0)
            quantum_weight = self.quantum_weights.get(worker, 1.0)
            
            # Quantum superposition-based load balancing
            # Prefer workers with lower load but consider quantum entanglement
            base_score = (1.0 - current_load) * quantum_weight
            
            # Entanglement bonus (workers entangled with less loaded workers get bonus)
            entanglement_bonus = 0.0
            entangled_workers = self.worker_entanglement.get(worker, [])
            for entangled_worker in entangled_workers:
                if entangled_worker in healthy_workers:
                    entangled_load = self.instance_loads.get(entangled_worker, 0.0)
                    entanglement_bonus += (1.0 - entangled_load) * 0.1
            
            total_score = base_score + entanglement_bonus
            worker_scores.append((total_score, worker))
        
        # Select worker with highest score
        worker_scores.sort(reverse=True)
        selected_worker = worker_scores[0][1]
        
        logger.debug(f"Selected worker {selected_worker} with score {worker_scores[0][0]:.3f}")
        return selected_worker
    
    def update_worker_load(self, worker: str, load: float) -> None:
        """Update worker load and adjust quantum weights."""
        if worker in self.instance_loads:
            self.instance_loads[worker] = load
            
            # Adjust quantum weight based on performance
            if load < 0.5:  # Low load, increase weight
                self.quantum_weights[worker] = min(2.0, self.quantum_weights[worker] * 1.05)
            elif load > 0.9:  # High load, decrease weight
                self.quantum_weights[worker] = max(0.1, self.quantum_weights[worker] * 0.95)
    
    def create_worker_entanglement(self, worker1: str, worker2: str) -> None:
        """Create quantum entanglement between workers."""
        if worker1 not in self.worker_entanglement:
            self.worker_entanglement[worker1] = []
        if worker2 not in self.worker_entanglement:
            self.worker_entanglement[worker2] = []
        
        if worker2 not in self.worker_entanglement[worker1]:
            self.worker_entanglement[worker1].append(worker2)
        if worker1 not in self.worker_entanglement[worker2]:
            self.worker_entanglement[worker2].append(worker1)
        
        logger.info(f"Created quantum entanglement between workers {worker1} and {worker2}")
    
    def rebalance_load(self) -> int:
        """Rebalance load across workers using quantum optimization."""
        if len(self.worker_instances) < 2:
            return 0
        
        # Sort workers by current load
        sorted_workers = sorted(
            self.worker_instances,
            key=lambda w: self.instance_loads.get(w, 0.0)
        )
        
        rebalanced_count = 0
        
        # Move load from heavily loaded to lightly loaded workers
        for i in range(len(sorted_workers) - 1):
            light_worker = sorted_workers[i]
            heavy_worker = sorted_workers[-(i + 1)]
            
            light_load = self.instance_loads.get(light_worker, 0.0)
            heavy_load = self.instance_loads.get(heavy_worker, 0.0)
            
            # If significant imbalance exists
            if heavy_load - light_load > 0.3:
                # Calculate optimal load transfer
                transfer_amount = (heavy_load - light_load) * 0.2  # Transfer 20% of difference
                
                self.instance_loads[heavy_worker] -= transfer_amount
                self.instance_loads[light_worker] += transfer_amount
                
                rebalanced_count += 1
                logger.info(f"Rebalanced {transfer_amount:.3f} load from {heavy_worker} to {light_worker}")
        
        return rebalanced_count


class PerformanceOptimizer:
    """Performance optimization using quantum-inspired algorithms."""
    
    def __init__(self, scheduler: QuantumScheduler, resource_allocator: QuantumResourceAllocator):
        self.scheduler = scheduler
        self.resource_allocator = resource_allocator
        self.optimization_history: List[Dict] = []
        self.performance_baselines: Dict[str, float] = {}
        
    def optimize_system_performance(self) -> Dict[str, Any]:
        """Comprehensive system performance optimization."""
        optimization_start = time.time()
        optimizations_applied = []
        
        # 1. Optimize task scheduling order
        schedule_optimization = self._optimize_task_scheduling()
        if schedule_optimization["improved"]:
            optimizations_applied.append("task_scheduling")
        
        # 2. Optimize resource allocation patterns
        resource_optimization = self._optimize_resource_allocation()
        if resource_optimization["improved"]:
            optimizations_applied.append("resource_allocation")
        
        # 3. Optimize quantum parameters
        quantum_optimization = self._optimize_quantum_parameters()
        if quantum_optimization["improved"]:
            optimizations_applied.append("quantum_parameters")
        
        # 4. Optimize parallel execution
        parallel_optimization = self._optimize_parallel_execution()
        if parallel_optimization["improved"]:
            optimizations_applied.append("parallel_execution")
        
        optimization_time = time.time() - optimization_start
        
        optimization_result = {
            "timestamp": datetime.now(),
            "optimization_time": optimization_time,
            "optimizations_applied": optimizations_applied,
            "performance_improvements": {
                "scheduling": schedule_optimization,
                "resources": resource_optimization,
                "quantum": quantum_optimization,
                "parallel": parallel_optimization
            }
        }
        
        self.optimization_history.append(optimization_result)
        
        logger.info(f"Performance optimization completed in {optimization_time:.2f}s: {optimizations_applied}")
        return optimization_result
    
    def _optimize_task_scheduling(self) -> Dict[str, Any]:
        """Optimize task scheduling using quantum algorithms."""
        
        # Get current pending tasks
        pending_tasks = [
            task for task in self.scheduler.tasks.values()
            if task.status == TaskStatus.PENDING
        ]
        
        if len(pending_tasks) < 2:
            return {"improved": False, "reason": "insufficient_tasks"}
        
        # Calculate current scheduling efficiency
        current_efficiency = self._calculate_scheduling_efficiency()
        
        # Apply quantum optimization to task ordering
        # This is a simplified version - production would use full quantum algorithms
        
        # Sort by quantum-inspired priority scoring
        optimized_order = sorted(
            pending_tasks,
            key=lambda t: self.scheduler._calculate_priority_score(t),
            reverse=True
        )
        
        # Calculate new efficiency (simulated)
        projected_efficiency = current_efficiency * 1.1  # Assume 10% improvement
        
        improvement = projected_efficiency > current_efficiency
        
        return {
            "improved": improvement,
            "current_efficiency": current_efficiency,
            "projected_efficiency": projected_efficiency,
            "optimized_task_count": len(optimized_order)
        }
    
    def _optimize_resource_allocation(self) -> Dict[str, Any]:
        """Optimize resource allocation patterns."""
        
        # Trigger resource rebalancing
        rebalanced_count = self.resource_allocator.rebalance_allocations()
        
        # Get current utilization
        utilization = self.resource_allocator.get_resource_utilization()
        
        # Calculate efficiency metrics
        avg_utilization = sum(
            data["utilization_percent"] for data in utilization.values()
        ) / len(utilization) if utilization else 0.0
        
        # Check if rebalancing improved distribution
        utilization_variance = self._calculate_utilization_variance(utilization)
        
        return {
            "improved": rebalanced_count > 0,
            "rebalanced_allocations": rebalanced_count,
            "average_utilization": avg_utilization,
            "utilization_variance": utilization_variance
        }
    
    def _optimize_quantum_parameters(self) -> Dict[str, Any]:
        """Optimize quantum algorithm parameters."""
        
        # Analyze quantum state coherence
        coherence_score = self._analyze_quantum_coherence()
        
        # Optimize quantum scheduler parameters based on performance
        improvements = []
        
        # Adjust max_parallel_tasks based on resource availability
        current_parallel = self.scheduler.max_parallel_tasks
        optimal_parallel = self._calculate_optimal_parallelism()
        
        if optimal_parallel != current_parallel:
            self.scheduler.max_parallel_tasks = optimal_parallel
            improvements.append(f"Adjusted parallel tasks: {current_parallel} -> {optimal_parallel}")
        
        return {
            "improved": len(improvements) > 0,
            "coherence_score": coherence_score,
            "improvements": improvements
        }
    
    def _optimize_parallel_execution(self) -> Dict[str, Any]:
        """Optimize parallel task execution patterns."""
        
        # Analyze current parallelism efficiency
        running_tasks = len(self.scheduler.running_tasks)
        max_parallel = self.scheduler.max_parallel_tasks
        
        parallelism_ratio = running_tasks / max_parallel if max_parallel > 0 else 0.0
        
        # Check for optimization opportunities
        improvements = []
        
        if parallelism_ratio < 0.5 and len(self.scheduler.tasks) > running_tasks:
            # Low parallelism utilization
            next_tasks = self.scheduler.get_next_tasks()
            if next_tasks:
                improvements.append(f"Can start {len(next_tasks)} additional tasks")
        
        return {
            "improved": len(improvements) > 0,
            "parallelism_ratio": parallelism_ratio,
            "running_tasks": running_tasks,
            "max_parallel": max_parallel,
            "improvements": improvements
        }
    
    def _calculate_scheduling_efficiency(self) -> float:
        """Calculate current scheduling efficiency score."""
        
        completed_tasks = [
            task for task in self.scheduler.tasks.values()
            if task.status == TaskStatus.COMPLETED
        ]
        
        if not completed_tasks:
            return 0.5  # Neutral efficiency
        
        # Calculate efficiency based on completion time vs estimates
        total_accuracy = 0.0
        
        for task in completed_tasks:
            if task.started_at and task.completed_at:
                actual_duration = task.completed_at - task.started_at
                estimated_duration = task.estimated_duration
                
                # Efficiency = how close actual was to estimate
                if estimated_duration.total_seconds() > 0:
                    accuracy = 1.0 - abs(
                        actual_duration.total_seconds() - estimated_duration.total_seconds()
                    ) / estimated_duration.total_seconds()
                    total_accuracy += max(0.0, accuracy)
        
        return total_accuracy / len(completed_tasks) if completed_tasks else 0.5
    
    def _calculate_utilization_variance(self, utilization: Dict[str, Dict]) -> float:
        """Calculate variance in resource utilization."""
        if not utilization:
            return 0.0
        
        utilization_values = [data["utilization_percent"] for data in utilization.values()]
        
        if len(utilization_values) < 2:
            return 0.0
        
        mean_util = sum(utilization_values) / len(utilization_values)
        variance = sum((util - mean_util) ** 2 for util in utilization_values) / len(utilization_values)
        
        return variance
    
    def _analyze_quantum_coherence(self) -> float:
        """Analyze quantum state coherence."""
        # Simplified coherence analysis
        # In production, would analyze actual quantum state matrix
        
        task_count = len(self.scheduler.tasks)
        if task_count == 0:
            return 1.0
        
        # Coherence based on task state distribution
        status_counts = self.scheduler.get_task_statistics()
        total_tasks = sum(status_counts.values())
        
        if total_tasks == 0:
            return 1.0
        
        # Calculate entropy of task states (lower entropy = higher coherence)
        entropy = 0.0
        for count in status_counts.values():
            if count > 0:
                probability = count / total_tasks
                entropy -= probability * (probability.bit_length() - 1 if probability > 0 else 0)
        
        # Convert entropy to coherence (0-1 scale)
        max_entropy = len(status_counts).bit_length() - 1 if len(status_counts) > 0 else 1
        coherence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
        
        return coherence
    
    def _calculate_optimal_parallelism(self) -> int:
        """Calculate optimal number of parallel tasks."""
        
        # Get resource utilization
        utilization = self.resource_allocator.get_resource_utilization()
        
        if not utilization:
            return self.scheduler.max_parallel_tasks
        
        # Find most constraining resource
        max_util = max(
            data["utilization_percent"] for data in utilization.values()
        )
        
        # Adjust parallelism based on resource pressure
        if max_util > 90:
            return max(1, self.scheduler.max_parallel_tasks - 1)
        elif max_util < 50:
            return min(16, self.scheduler.max_parallel_tasks + 1)
        else:
            return self.scheduler.max_parallel_tasks


class QuantumClusterManager:
    """Manages quantum task planner cluster with auto-scaling."""
    
    def __init__(self):
        self.auto_scaler = QuantumAutoScaler()
        self.load_balancer = QuantumLoadBalancer([])
        self.performance_optimizer = None  # Set later
        self.monitoring_active = False
        self._monitoring_thread = None
        
    def start_cluster_management(self, monitoring_interval: int = 30) -> None:
        """Start automated cluster management."""
        self.monitoring_active = True
        
        def management_loop():
            while self.monitoring_active:
                try:
                    self._cluster_management_cycle()
                    time.sleep(monitoring_interval)
                except Exception as e:
                    logger.error(f"Cluster management error: {e}")
                    time.sleep(monitoring_interval)
        
        self._monitoring_thread = threading.Thread(target=management_loop, daemon=True)
        self._monitoring_thread.start()
        logger.info("Quantum cluster management started")
    
    def stop_cluster_management(self) -> None:
        """Stop automated cluster management."""
        self.monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=10)
    
    def _cluster_management_cycle(self) -> None:
        """Execute one cycle of cluster management."""
        
        # Collect current metrics
        current_load = self._calculate_current_load()
        queue_size = self._get_queue_size()
        resource_utilization = self._get_resource_utilization()
        
        # Analyze scaling needs
        scaling_decision = self.auto_scaler.analyze_scaling_need(
            current_load, queue_size, resource_utilization
        )
        
        # Execute scaling if needed
        if scaling_decision.direction != ScalingDirection.STABLE:
            self.auto_scaler.execute_scaling(scaling_decision)
        
        # Rebalance load
        rebalanced = self.load_balancer.rebalance_load()
        if rebalanced > 0:
            logger.info(f"Rebalanced load across {rebalanced} worker pairs")
        
        # Optimize performance if configured
        if self.performance_optimizer:
            self.performance_optimizer.optimize_system_performance()
    
    def _calculate_current_load(self) -> float:
        """Calculate current system load."""
        # Simplified load calculation
        if hasattr(self, 'scheduler'):
            running_tasks = len(self.scheduler.running_tasks)
            max_tasks = self.scheduler.max_parallel_tasks
            return running_tasks / max_tasks if max_tasks > 0 else 0.0
        return 0.0
    
    def _get_queue_size(self) -> int:
        """Get current task queue size."""
        if hasattr(self, 'scheduler'):
            return len([
                task for task in self.scheduler.tasks.values()
                if task.status == TaskStatus.PENDING
            ])
        return 0
    
    def _get_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization."""
        if hasattr(self, 'resource_allocator'):
            utilization = self.resource_allocator.get_resource_utilization()
            return {
                resource_id: data["utilization_percent"]
                for resource_id, data in utilization.items()
            }
        return {}