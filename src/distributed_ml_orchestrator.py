"""Distributed Machine Learning Orchestrator for Scaled Medical AI Systems.

This module provides advanced distributed training, inference orchestration,
and automatic resource optimization for large-scale medical AI deployments.
"""

import asyncio
import logging
import time
import threading
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable, Set
import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
import multiprocessing as mp
from queue import Queue, PriorityQueue
import heapq
import numpy as np


class NodeType(Enum):
    """Types of compute nodes in the distributed system."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    PARAMETER_SERVER = "parameter_server"
    INFERENCE_ENGINE = "inference_engine"
    DATA_NODE = "data_node"
    EDGE_DEVICE = "edge_device"


class TaskType(Enum):
    """Types of distributed tasks."""
    TRAINING = "training"
    INFERENCE = "inference"
    DATA_PREPROCESSING = "data_preprocessing"
    MODEL_VALIDATION = "model_validation"
    FEDERATED_AGGREGATION = "federated_aggregation"
    HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization"


class ResourceType(Enum):
    """Types of computational resources."""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    TPU = "tpu"


@dataclass
class ComputeResource:
    """Represents available computational resources."""
    resource_type: ResourceType
    total_capacity: float
    available_capacity: float
    utilization_rate: float = 0.0
    performance_score: float = 1.0
    
    @property
    def utilization_percentage(self) -> float:
        """Get utilization as percentage."""
        if self.total_capacity == 0:
            return 0.0
        return ((self.total_capacity - self.available_capacity) / self.total_capacity) * 100


@dataclass
class ComputeNode:
    """Represents a compute node in the distributed system."""
    node_id: str
    node_type: NodeType
    endpoint: str
    resources: Dict[ResourceType, ComputeResource]
    status: str = "active"
    last_heartbeat: float = field(default_factory=time.time)
    current_tasks: Set[str] = field(default_factory=set)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    geographic_location: Optional[str] = None
    network_latency: float = 0.0
    reliability_score: float = 1.0
    
    @property
    def total_gpu_memory(self) -> float:
        """Get total GPU memory across all GPUs."""
        gpu_resource = self.resources.get(ResourceType.GPU)
        return gpu_resource.total_capacity if gpu_resource else 0.0
        
    @property
    def available_gpu_memory(self) -> float:
        """Get available GPU memory across all GPUs."""
        gpu_resource = self.resources.get(ResourceType.GPU)
        return gpu_resource.available_capacity if gpu_resource else 0.0
        
    @property
    def cpu_cores(self) -> int:
        """Get number of CPU cores."""
        cpu_resource = self.resources.get(ResourceType.CPU)
        return int(cpu_resource.total_capacity) if cpu_resource else 0


@dataclass
class DistributedTask:
    """Represents a task in the distributed system."""
    task_id: str
    task_type: TaskType
    priority: int = 0
    resource_requirements: Dict[ResourceType, float] = field(default_factory=dict)
    estimated_duration: float = 0.0
    deadline: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    assigned_nodes: List[str] = field(default_factory=list)
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_overdue(self) -> bool:
        """Check if task is overdue."""
        if self.deadline is None:
            return False
        return time.time() > self.deadline
        
    @property
    def execution_time(self) -> Optional[float]:
        """Get actual execution time if completed."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


class IntelligentScheduler:
    """Advanced scheduler for distributed ML tasks."""
    
    def __init__(self):
        self.task_queue = PriorityQueue()
        self.node_performance_history: Dict[str, List[float]] = {}
        self.task_execution_history: Dict[str, List[float]] = {}
        self.scheduling_policies = {
            TaskType.TRAINING: self._schedule_training_task,
            TaskType.INFERENCE: self._schedule_inference_task,
            TaskType.DATA_PREPROCESSING: self._schedule_data_task,
            TaskType.FEDERATED_AGGREGATION: self._schedule_federated_task,
        }
        
    def schedule_task(
        self, 
        task: DistributedTask, 
        available_nodes: List[ComputeNode]
    ) -> List[ComputeNode]:
        """Schedule a task on the most appropriate nodes."""
        
        # Filter nodes that can handle the task
        capable_nodes = self._filter_capable_nodes(task, available_nodes)
        
        if not capable_nodes:
            return []
            
        # Use task-specific scheduling policy
        scheduler_func = self.scheduling_policies.get(
            task.task_type, 
            self._default_scheduling_policy
        )
        
        selected_nodes = scheduler_func(task, capable_nodes)
        
        # Update scheduling history
        self._update_scheduling_history(task, selected_nodes)
        
        return selected_nodes
        
    def _filter_capable_nodes(
        self, 
        task: DistributedTask, 
        nodes: List[ComputeNode]
    ) -> List[ComputeNode]:
        """Filter nodes that can handle the task requirements."""
        capable_nodes = []
        
        for node in nodes:
            if self._can_node_handle_task(node, task):
                capable_nodes.append(node)
                
        return capable_nodes
        
    def _can_node_handle_task(self, node: ComputeNode, task: DistributedTask) -> bool:
        """Check if a node can handle a specific task."""
        
        # Check if node is active
        if node.status != "active":
            return False
            
        # Check resource requirements
        for resource_type, required_amount in task.resource_requirements.items():
            if resource_type not in node.resources:
                return False
                
            available = node.resources[resource_type].available_capacity
            if available < required_amount:
                return False
                
        # Check task-specific requirements
        if task.task_type == TaskType.TRAINING:
            # Training tasks prefer GPU-enabled nodes
            if ResourceType.GPU not in node.resources:
                return False
                
        elif task.task_type == TaskType.INFERENCE:
            # Inference tasks need low-latency nodes
            if node.network_latency > 100:  # >100ms latency
                return False
                
        return True
        
    def _schedule_training_task(
        self, 
        task: DistributedTask, 
        nodes: List[ComputeNode]
    ) -> List[ComputeNode]:
        """Schedule training tasks with emphasis on GPU resources."""
        
        # Sort nodes by GPU capacity and performance
        def training_score(node: ComputeNode) -> float:
            gpu_score = node.available_gpu_memory / max(node.total_gpu_memory, 1)
            performance_score = node.reliability_score
            utilization_penalty = node.resources[ResourceType.GPU].utilization_rate
            
            return gpu_score * performance_score * (1 - utilization_penalty)
            
        sorted_nodes = sorted(nodes, key=training_score, reverse=True)
        
        # For distributed training, select multiple nodes
        num_nodes_needed = min(len(sorted_nodes), task.payload.get("num_replicas", 1))
        return sorted_nodes[:num_nodes_needed]
        
    def _schedule_inference_task(
        self, 
        task: DistributedTask, 
        nodes: List[ComputeNode]
    ) -> List[ComputeNode]:
        """Schedule inference tasks with emphasis on latency."""
        
        def inference_score(node: ComputeNode) -> float:
            latency_score = 1.0 / (1.0 + node.network_latency / 1000)  # Lower latency is better
            cpu_availability = node.resources[ResourceType.CPU].available_capacity / node.resources[ResourceType.CPU].total_capacity
            load_penalty = len(node.current_tasks) / 10.0  # Penalty for high load
            
            return latency_score * cpu_availability * (1 - load_penalty)
            
        sorted_nodes = sorted(nodes, key=inference_score, reverse=True)
        return [sorted_nodes[0]]  # Single node for inference
        
    def _schedule_data_task(
        self, 
        task: DistributedTask, 
        nodes: List[ComputeNode]
    ) -> List[ComputeNode]:
        """Schedule data processing tasks with emphasis on storage and CPU."""
        
        def data_score(node: ComputeNode) -> float:
            storage_score = node.resources.get(ResourceType.STORAGE, ComputeResource(ResourceType.STORAGE, 0, 0)).available_capacity
            cpu_score = node.resources[ResourceType.CPU].available_capacity
            network_score = 1.0 / (1.0 + node.network_latency / 1000)
            
            return (storage_score + cpu_score) * network_score
            
        sorted_nodes = sorted(nodes, key=data_score, reverse=True)
        
        # Parallel data processing on multiple nodes
        num_nodes = min(len(sorted_nodes), task.payload.get("parallelism", 2))
        return sorted_nodes[:num_nodes]
        
    def _schedule_federated_task(
        self, 
        task: DistributedTask, 
        nodes: List[ComputeNode]
    ) -> List[ComputeNode]:
        """Schedule federated learning aggregation tasks."""
        
        # Prefer nodes with high reliability and low latency
        def federated_score(node: ComputeNode) -> float:
            reliability = node.reliability_score
            latency_score = 1.0 / (1.0 + node.network_latency / 1000)
            cpu_availability = node.resources[ResourceType.CPU].available_capacity / node.resources[ResourceType.CPU].total_capacity
            
            return reliability * latency_score * cpu_availability
            
        sorted_nodes = sorted(nodes, key=federated_score, reverse=True)
        return [sorted_nodes[0]]  # Single coordinator node
        
    def _default_scheduling_policy(
        self, 
        task: DistributedTask, 
        nodes: List[ComputeNode]
    ) -> List[ComputeNode]:
        """Default scheduling policy based on overall node score."""
        
        def overall_score(node: ComputeNode) -> float:
            # Weighted average of resource availability
            scores = []
            weights = []
            
            for resource_type, resource in node.resources.items():
                if resource.total_capacity > 0:
                    availability = resource.available_capacity / resource.total_capacity
                    scores.append(availability * resource.performance_score)
                    weights.append(1.0)
                    
            if not scores:
                return 0.0
                
            weighted_avg = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            return weighted_avg * node.reliability_score
            
        sorted_nodes = sorted(nodes, key=overall_score, reverse=True)
        return [sorted_nodes[0]]
        
    def _update_scheduling_history(
        self, 
        task: DistributedTask, 
        selected_nodes: List[ComputeNode]
    ) -> None:
        """Update scheduling history for learning."""
        for node in selected_nodes:
            if node.node_id not in self.node_performance_history:
                self.node_performance_history[node.node_id] = []
                
        # Track task assignment
        task_key = f"{task.task_type.value}_{task.priority}"
        if task_key not in self.task_execution_history:
            self.task_execution_history[task_key] = []


class ResourceMonitor:
    """Advanced resource monitoring and prediction."""
    
    def __init__(self):
        self.resource_history: Dict[str, List[Dict[str, float]]] = {}
        self.prediction_models = {}
        self.monitoring_interval = 30.0  # seconds
        self._running = False
        
    def start_monitoring(self, nodes: Dict[str, ComputeNode]) -> None:
        """Start continuous resource monitoring."""
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(nodes,), 
            daemon=True
        )
        self._monitor_thread.start()
        
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self._running = False
        
    def _monitoring_loop(self, nodes: Dict[str, ComputeNode]) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                self._collect_metrics(nodes)
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logging.error(f"Resource monitoring error: {e}")
                time.sleep(self.monitoring_interval * 2)
                
    def _collect_metrics(self, nodes: Dict[str, ComputeNode]) -> None:
        """Collect resource metrics from all nodes."""
        current_time = time.time()
        
        for node_id, node in nodes.items():
            # Simulate resource collection (in real implementation, would query nodes)
            metrics = self._simulate_resource_collection(node)
            
            # Update node resources
            self._update_node_resources(node, metrics)
            
            # Store in history
            if node_id not in self.resource_history:
                self.resource_history[node_id] = []
                
            self.resource_history[node_id].append({
                "timestamp": current_time,
                **metrics
            })
            
            # Keep only recent history (last 1000 measurements)
            if len(self.resource_history[node_id]) > 1000:
                self.resource_history[node_id] = self.resource_history[node_id][-1000:]
                
    def _simulate_resource_collection(self, node: ComputeNode) -> Dict[str, float]:
        """Simulate resource metric collection."""
        import random
        
        # Simulate realistic resource fluctuations
        base_cpu_usage = 0.3 + random.uniform(-0.1, 0.3)
        base_memory_usage = 0.4 + random.uniform(-0.1, 0.2)
        base_gpu_usage = 0.2 + random.uniform(-0.1, 0.4) if ResourceType.GPU in node.resources else 0.0
        
        return {
            "cpu_usage": max(0.0, min(1.0, base_cpu_usage)),
            "memory_usage": max(0.0, min(1.0, base_memory_usage)),
            "gpu_usage": max(0.0, min(1.0, base_gpu_usage)),
            "network_latency": max(1.0, node.network_latency + random.uniform(-10, 10)),
            "disk_io": random.uniform(0.1, 0.8),
            "network_throughput": random.uniform(100, 1000)  # MB/s
        }
        
    def _update_node_resources(self, node: ComputeNode, metrics: Dict[str, float]) -> None:
        """Update node resource information based on collected metrics."""
        
        # Update CPU
        if ResourceType.CPU in node.resources:
            cpu_resource = node.resources[ResourceType.CPU]
            cpu_resource.utilization_rate = metrics["cpu_usage"]
            cpu_resource.available_capacity = cpu_resource.total_capacity * (1 - metrics["cpu_usage"])
            
        # Update Memory
        if ResourceType.MEMORY in node.resources:
            memory_resource = node.resources[ResourceType.MEMORY]
            memory_resource.utilization_rate = metrics["memory_usage"]
            memory_resource.available_capacity = memory_resource.total_capacity * (1 - metrics["memory_usage"])
            
        # Update GPU
        if ResourceType.GPU in node.resources:
            gpu_resource = node.resources[ResourceType.GPU]
            gpu_resource.utilization_rate = metrics["gpu_usage"]
            gpu_resource.available_capacity = gpu_resource.total_capacity * (1 - metrics["gpu_usage"])
            
        # Update network latency
        node.network_latency = metrics["network_latency"]
        
        # Update performance metrics
        node.performance_metrics.update(metrics)
        
    def predict_resource_usage(
        self, 
        node_id: str, 
        time_horizon: float = 300.0
    ) -> Dict[str, float]:
        """Predict resource usage for a node over a time horizon."""
        
        if node_id not in self.resource_history:
            return {}
            
        history = self.resource_history[node_id]
        if len(history) < 10:  # Need minimum data for prediction
            return {}
            
        # Simple moving average prediction (in real implementation, would use ML models)
        recent_data = history[-10:]  # Last 10 measurements
        
        predictions = {}
        for metric in ["cpu_usage", "memory_usage", "gpu_usage"]:
            values = [data.get(metric, 0.0) for data in recent_data]
            predicted_value = sum(values) / len(values)
            predictions[metric] = predicted_value
            
        return predictions


class DistributedMLOrchestrator:
    """Main orchestrator for distributed machine learning operations."""
    
    def __init__(self):
        self.nodes: Dict[str, ComputeNode] = {}
        self.tasks: Dict[str, DistributedTask] = {}
        self.scheduler = IntelligentScheduler()
        self.resource_monitor = ResourceMonitor()
        self.task_executor = DistributedTaskExecutor()
        
        self._running = False
        self._orchestration_thread = None
        
    def start_orchestrator(self) -> None:
        """Start the distributed orchestrator."""
        self._running = True
        self.resource_monitor.start_monitoring(self.nodes)
        
        self._orchestration_thread = threading.Thread(
            target=self._orchestration_loop, 
            daemon=True
        )
        self._orchestration_thread.start()
        
        logging.info("Distributed ML Orchestrator started")
        
    def stop_orchestrator(self) -> None:
        """Stop the distributed orchestrator."""
        self._running = False
        self.resource_monitor.stop_monitoring()
        
        if self._orchestration_thread:
            self._orchestration_thread.join(timeout=5.0)
            
        logging.info("Distributed ML Orchestrator stopped")
        
    def register_node(self, node: ComputeNode) -> None:
        """Register a new compute node."""
        self.nodes[node.node_id] = node
        logging.info(f"Registered compute node: {node.node_id} ({node.node_type.value})")
        
    def unregister_node(self, node_id: str) -> None:
        """Unregister a compute node."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            logging.info(f"Unregistered compute node: {node_id}")
            
    def submit_task(self, task: DistributedTask) -> str:
        """Submit a task for distributed execution."""
        self.tasks[task.task_id] = task
        logging.info(f"Submitted task: {task.task_id} ({task.task_type.value})")
        return task.task_id
        
    def get_task_status(self, task_id: str) -> Optional[str]:
        """Get the status of a task."""
        task = self.tasks.get(task_id)
        return task.status if task else None
        
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        total_nodes = len(self.nodes)
        active_nodes = sum(1 for node in self.nodes.values() if node.status == "active")
        
        total_tasks = len(self.tasks)
        running_tasks = sum(1 for task in self.tasks.values() if task.status == "running")
        completed_tasks = sum(1 for task in self.tasks.values() if task.status == "completed")
        
        # Calculate total resources
        total_cpus = sum(node.cpu_cores for node in self.nodes.values())
        total_gpu_memory = sum(node.total_gpu_memory for node in self.nodes.values())
        available_gpu_memory = sum(node.available_gpu_memory for node in self.nodes.values())
        
        return {
            "cluster_info": {
                "total_nodes": total_nodes,
                "active_nodes": active_nodes,
                "total_cpus": total_cpus,
                "total_gpu_memory_gb": total_gpu_memory,
                "available_gpu_memory_gb": available_gpu_memory,
                "gpu_utilization_percent": ((total_gpu_memory - available_gpu_memory) / max(total_gpu_memory, 1)) * 100
            },
            "task_info": {
                "total_tasks": total_tasks,
                "running_tasks": running_tasks,
                "completed_tasks": completed_tasks,
                "pending_tasks": total_tasks - running_tasks - completed_tasks
            },
            "node_details": [
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type.value,
                    "status": node.status,
                    "cpu_cores": node.cpu_cores,
                    "gpu_memory_gb": node.total_gpu_memory,
                    "current_tasks": len(node.current_tasks),
                    "reliability_score": node.reliability_score
                }
                for node in self.nodes.values()
            ]
        }
        
    def _orchestration_loop(self) -> None:
        """Main orchestration loop."""
        while self._running:
            try:
                self._process_pending_tasks()
                self._monitor_running_tasks()
                self._optimize_resource_allocation()
                time.sleep(5.0)  # Check every 5 seconds
            except Exception as e:
                logging.error(f"Orchestration loop error: {e}")
                time.sleep(10.0)
                
    def _process_pending_tasks(self) -> None:
        """Process pending tasks and schedule them."""
        pending_tasks = [task for task in self.tasks.values() if task.status == "pending"]
        
        for task in pending_tasks:
            # Check dependencies
            if not self._are_dependencies_satisfied(task):
                continue
                
            # Find available nodes
            available_nodes = [
                node for node in self.nodes.values() 
                if node.status == "active"
            ]
            
            # Schedule task
            selected_nodes = self.scheduler.schedule_task(task, available_nodes)
            
            if selected_nodes:
                # Assign task to nodes
                task.assigned_nodes = [node.node_id for node in selected_nodes]
                task.status = "scheduled"
                task.started_at = time.time()
                
                # Update node assignments
                for node in selected_nodes:
                    node.current_tasks.add(task.task_id)
                    
                # Start task execution
                self.task_executor.execute_task(task, selected_nodes)
                
                logging.info(f"Scheduled task {task.task_id} on nodes: {task.assigned_nodes}")
                
    def _are_dependencies_satisfied(self, task: DistributedTask) -> bool:
        """Check if task dependencies are satisfied."""
        for dependency_id in task.dependencies:
            if dependency_id in self.tasks:
                dependency_task = self.tasks[dependency_id]
                if dependency_task.status != "completed":
                    return False
            else:
                return False  # Dependency not found
        return True
        
    def _monitor_running_tasks(self) -> None:
        """Monitor and update status of running tasks."""
        running_tasks = [task for task in self.tasks.values() if task.status in ["scheduled", "running"]]
        
        for task in running_tasks:
            # Check if task is completed (simplified check)
            if self._is_task_completed(task):
                task.status = "completed"
                task.completed_at = time.time()
                
                # Free up assigned nodes
                for node_id in task.assigned_nodes:
                    if node_id in self.nodes:
                        self.nodes[node_id].current_tasks.discard(task.task_id)
                        
                logging.info(f"Task {task.task_id} completed in {task.execution_time:.2f}s")
                
            # Check for task timeouts
            elif task.is_overdue:
                task.status = "failed"
                logging.warning(f"Task {task.task_id} failed due to timeout")
                
    def _is_task_completed(self, task: DistributedTask) -> bool:
        """Check if a task is completed (simplified implementation)."""
        # In real implementation, would check with actual task execution system
        if task.started_at:
            elapsed_time = time.time() - task.started_at
            # Simulate task completion based on estimated duration
            estimated_duration = task.estimated_duration or 60.0  # Default 1 minute
            return elapsed_time >= estimated_duration
        return False
        
    def _optimize_resource_allocation(self) -> None:
        """Optimize resource allocation across the cluster."""
        # Identify overloaded and underutilized nodes
        overloaded_nodes = []
        underutilized_nodes = []
        
        for node in self.nodes.values():
            if node.status != "active":
                continue
                
            cpu_utilization = node.resources[ResourceType.CPU].utilization_rate
            
            if cpu_utilization > 0.9:  # >90% utilization
                overloaded_nodes.append(node)
            elif cpu_utilization < 0.3:  # <30% utilization
                underutilized_nodes.append(node)
                
        # Implement load balancing (simplified)
        if overloaded_nodes and underutilized_nodes:
            logging.info(f"Rebalancing load: {len(overloaded_nodes)} overloaded, {len(underutilized_nodes)} underutilized")
            # In real implementation, would migrate tasks or adjust scheduling


class DistributedTaskExecutor:
    """Executes distributed tasks on assigned nodes."""
    
    def __init__(self):
        self.execution_pool = ThreadPoolExecutor(max_workers=50)
        
    def execute_task(self, task: DistributedTask, nodes: List[ComputeNode]) -> Future:
        """Execute a task on the assigned nodes."""
        
        # Submit task for asynchronous execution
        future = self.execution_pool.submit(self._execute_task_implementation, task, nodes)
        
        # Update task status
        task.status = "running"
        
        return future
        
    def _execute_task_implementation(self, task: DistributedTask, nodes: List[ComputeNode]) -> Any:
        """Actual task execution implementation."""
        
        try:
            if task.task_type == TaskType.TRAINING:
                return self._execute_training_task(task, nodes)
            elif task.task_type == TaskType.INFERENCE:
                return self._execute_inference_task(task, nodes)
            elif task.task_type == TaskType.DATA_PREPROCESSING:
                return self._execute_data_preprocessing_task(task, nodes)
            elif task.task_type == TaskType.FEDERATED_AGGREGATION:
                return self._execute_federated_aggregation_task(task, nodes)
            else:
                return self._execute_generic_task(task, nodes)
                
        except Exception as e:
            logging.error(f"Task execution failed: {task.task_id}, Error: {e}")
            task.status = "failed"
            raise e
            
    def _execute_training_task(self, task: DistributedTask, nodes: List[ComputeNode]) -> Dict[str, Any]:
        """Execute distributed training task."""
        logging.info(f"Executing training task {task.task_id} on {len(nodes)} nodes")
        
        # Simulate distributed training
        time.sleep(task.estimated_duration or 30.0)
        
        return {
            "task_id": task.task_id,
            "result": "training_completed",
            "model_accuracy": 0.95,
            "training_time": task.estimated_duration,
            "nodes_used": [node.node_id for node in nodes]
        }
        
    def _execute_inference_task(self, task: DistributedTask, nodes: List[ComputeNode]) -> Dict[str, Any]:
        """Execute inference task."""
        logging.info(f"Executing inference task {task.task_id} on node {nodes[0].node_id}")
        
        # Simulate inference
        time.sleep(min(task.estimated_duration or 1.0, 5.0))
        
        return {
            "task_id": task.task_id,
            "result": "inference_completed",
            "prediction": "normal",
            "confidence": 0.92,
            "inference_time": task.estimated_duration,
            "node_used": nodes[0].node_id
        }
        
    def _execute_data_preprocessing_task(self, task: DistributedTask, nodes: List[ComputeNode]) -> Dict[str, Any]:
        """Execute data preprocessing task."""
        logging.info(f"Executing data preprocessing task {task.task_id} on {len(nodes)} nodes")
        
        # Simulate data processing
        time.sleep(task.estimated_duration or 10.0)
        
        return {
            "task_id": task.task_id,
            "result": "preprocessing_completed",
            "processed_samples": 10000,
            "processing_time": task.estimated_duration,
            "nodes_used": [node.node_id for node in nodes]
        }
        
    def _execute_federated_aggregation_task(self, task: DistributedTask, nodes: List[ComputeNode]) -> Dict[str, Any]:
        """Execute federated learning aggregation task."""
        logging.info(f"Executing federated aggregation task {task.task_id}")
        
        # Simulate model aggregation
        time.sleep(task.estimated_duration or 5.0)
        
        return {
            "task_id": task.task_id,
            "result": "aggregation_completed",
            "global_model_accuracy": 0.94,
            "participating_clients": 10,
            "aggregation_time": task.estimated_duration
        }
        
    def _execute_generic_task(self, task: DistributedTask, nodes: List[ComputeNode]) -> Dict[str, Any]:
        """Execute generic task."""
        logging.info(f"Executing generic task {task.task_id}")
        
        # Simulate generic computation
        time.sleep(task.estimated_duration or 1.0)
        
        return {
            "task_id": task.task_id,
            "result": "task_completed",
            "execution_time": task.estimated_duration
        }


def create_sample_cluster() -> DistributedMLOrchestrator:
    """Create a sample distributed cluster for testing."""
    orchestrator = DistributedMLOrchestrator()
    
    # Create sample nodes
    # GPU Training Nodes
    for i in range(3):
        node = ComputeNode(
            node_id=f"gpu_node_{i}",
            node_type=NodeType.WORKER,
            endpoint=f"http://gpu-node-{i}:8080",
            resources={
                ResourceType.CPU: ComputeResource(ResourceType.CPU, 32, 24),  # 32 cores, 24 available
                ResourceType.MEMORY: ComputeResource(ResourceType.MEMORY, 128, 96),  # 128GB, 96GB available
                ResourceType.GPU: ComputeResource(ResourceType.GPU, 32, 24),  # 32GB GPU, 24GB available
                ResourceType.STORAGE: ComputeResource(ResourceType.STORAGE, 1000, 800),  # 1TB, 800GB available
            },
            geographic_location=f"datacenter_{i % 3}",
            network_latency=10 + i * 5,
            reliability_score=0.98
        )
        orchestrator.register_node(node)
    
    # CPU Inference Nodes
    for i in range(5):
        node = ComputeNode(
            node_id=f"cpu_node_{i}",
            node_type=NodeType.INFERENCE_ENGINE,
            endpoint=f"http://cpu-node-{i}:8080",
            resources={
                ResourceType.CPU: ComputeResource(ResourceType.CPU, 16, 12),  # 16 cores, 12 available
                ResourceType.MEMORY: ComputeResource(ResourceType.MEMORY, 64, 48),  # 64GB, 48GB available
                ResourceType.STORAGE: ComputeResource(ResourceType.STORAGE, 500, 400),  # 500GB, 400GB available
            },
            geographic_location="edge_datacenter",
            network_latency=20 + i * 2,
            reliability_score=0.95
        )
        orchestrator.register_node(node)
    
    # Parameter Server
    param_server = ComputeNode(
        node_id="param_server_0",
        node_type=NodeType.PARAMETER_SERVER,
        endpoint="http://param-server:8080",
        resources={
            ResourceType.CPU: ComputeResource(ResourceType.CPU, 64, 48),  # High CPU for aggregation
            ResourceType.MEMORY: ComputeResource(ResourceType.MEMORY, 256, 192),  # Large memory
            ResourceType.STORAGE: ComputeResource(ResourceType.STORAGE, 2000, 1500),  # Large storage
        },
        geographic_location="main_datacenter",
        network_latency=5,
        reliability_score=0.99
    )
    orchestrator.register_node(param_server)
    
    return orchestrator


if __name__ == "__main__":
    # Example usage
    
    # Create and start orchestrator
    orchestrator = create_sample_cluster()
    orchestrator.start_orchestrator()
    
    # Submit various tasks
    tasks = []
    
    # Training task
    training_task = DistributedTask(
        task_id=str(uuid.uuid4()),
        task_type=TaskType.TRAINING,
        priority=1,
        resource_requirements={
            ResourceType.GPU: 16,  # 16GB GPU memory
            ResourceType.CPU: 8,   # 8 CPU cores
            ResourceType.MEMORY: 32  # 32GB RAM
        },
        estimated_duration=120.0,  # 2 minutes
        payload={"num_replicas": 2, "model_type": "pneumonia_cnn"}
    )
    tasks.append(training_task)
    
    # Inference tasks
    for i in range(10):
        inference_task = DistributedTask(
            task_id=str(uuid.uuid4()),
            task_type=TaskType.INFERENCE,
            priority=2,
            resource_requirements={
                ResourceType.CPU: 2,   # 2 CPU cores
                ResourceType.MEMORY: 4  # 4GB RAM
            },
            estimated_duration=2.0,  # 2 seconds
            payload={"patient_id": f"patient_{i}", "image_id": f"xray_{i}"}
        )
        tasks.append(inference_task)
    
    # Data preprocessing task
    data_task = DistributedTask(
        task_id=str(uuid.uuid4()),
        task_type=TaskType.DATA_PREPROCESSING,
        priority=0,
        resource_requirements={
            ResourceType.CPU: 16,   # 16 CPU cores
            ResourceType.MEMORY: 64,  # 64GB RAM
            ResourceType.STORAGE: 100  # 100GB storage
        },
        estimated_duration=60.0,  # 1 minute
        payload={"dataset_size": "10GB", "preprocessing_steps": ["resize", "normalize", "augment"]}
    )
    tasks.append(data_task)
    
    # Submit all tasks
    for task in tasks:
        orchestrator.submit_task(task)
    
    # Monitor cluster for a while
    for _ in range(12):  # Monitor for 1 minute
        time.sleep(10)
        status = orchestrator.get_cluster_status()
        print(f"\\nCluster Status at {time.strftime('%H:%M:%S')}:")
        print(f"Active Nodes: {status['cluster_info']['active_nodes']}")
        print(f"Running Tasks: {status['task_info']['running_tasks']}")
        print(f"Completed Tasks: {status['task_info']['completed_tasks']}")
        print(f"GPU Utilization: {status['cluster_info']['gpu_utilization_percent']:.1f}%")
    
    # Stop orchestrator
    orchestrator.stop_orchestrator()
    print("\\nDistributed ML Orchestrator demonstration complete")