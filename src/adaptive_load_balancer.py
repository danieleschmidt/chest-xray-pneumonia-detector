"""Adaptive Load Balancer for Medical AI Inference Systems.

This module provides intelligent load balancing with health-aware routing,
automatic scaling, and performance optimization for medical AI workloads.
"""

import asyncio
import logging
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor, Future


class NodeStatus(Enum):
    """Health status of inference nodes."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    OFFLINE = "offline"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies for different scenarios."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    HEALTH_AWARE = "health_aware"
    RESOURCE_AWARE = "resource_aware"
    INTELLIGENT_ADAPTIVE = "intelligent_adaptive"


@dataclass
class NodeMetrics:
    """Comprehensive metrics for inference nodes."""
    node_id: str
    timestamp: float = field(default_factory=time.time)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    active_connections: int = 0
    requests_per_second: float = 0.0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    model_accuracy: float = 1.0
    queue_length: int = 0
    inference_latency_p95: float = 0.0
    throughput_capacity: float = 100.0
    
    @property
    def health_score(self) -> float:
        """Calculate overall health score (0.0 to 1.0)."""
        # Weighted health score calculation
        cpu_score = max(0, 1.0 - self.cpu_usage)
        memory_score = max(0, 1.0 - self.memory_usage)
        gpu_score = max(0, 1.0 - self.gpu_usage) if self.gpu_usage > 0 else 1.0
        error_score = max(0, 1.0 - self.error_rate)
        latency_score = max(0, 1.0 - min(self.average_response_time / 5.0, 1.0))
        accuracy_score = self.model_accuracy
        
        # Weighted average
        weights = [0.2, 0.2, 0.15, 0.2, 0.15, 0.1]
        scores = [cpu_score, memory_score, gpu_score, error_score, latency_score, accuracy_score]
        
        return sum(w * s for w, s in zip(weights, scores))


@dataclass
class InferenceNode:
    """Represents an inference node in the cluster."""
    node_id: str
    endpoint: str
    weight: float = 1.0
    max_concurrent_requests: int = 100
    status: NodeStatus = NodeStatus.HEALTHY
    metrics: Optional[NodeMetrics] = None
    last_health_check: float = field(default_factory=time.time)
    failure_count: int = 0
    circuit_breaker_open: bool = False
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = NodeMetrics(node_id=self.node_id)


class RequestContext:
    """Context for inference requests with priority and routing hints."""
    
    def __init__(
        self,
        request_id: str,
        priority: int = 0,
        patient_id: Optional[str] = None,
        medical_urgency: str = "routine",
        preferred_model: Optional[str] = None,
        timeout: float = 30.0,
        retry_count: int = 0
    ):
        self.request_id = request_id
        self.priority = priority
        self.patient_id = patient_id
        self.medical_urgency = medical_urgency  # "emergency", "urgent", "routine"
        self.preferred_model = preferred_model
        self.timeout = timeout
        self.retry_count = retry_count
        self.created_at = time.time()
        
    @property
    def urgency_priority(self) -> int:
        """Get numeric priority based on medical urgency."""
        urgency_map = {
            "emergency": 100,
            "urgent": 50,
            "routine": 0
        }
        return urgency_map.get(self.medical_urgency, 0)
        
    @property
    def effective_priority(self) -> int:
        """Calculate effective priority considering urgency and base priority."""
        return self.priority + self.urgency_priority


class AdaptiveLoadBalancer:
    """Intelligent load balancer for medical AI inference systems."""
    
    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.INTELLIGENT_ADAPTIVE,
        health_check_interval: float = 10.0,
        metrics_window_size: int = 100
    ):
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.metrics_window_size = metrics_window_size
        
        self.nodes: Dict[str, InferenceNode] = {}
        self.request_queue = []  # Priority queue for requests
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=metrics_window_size))
        self.routing_table: Dict[str, str] = {}  # Patient -> preferred node mapping
        
        self._round_robin_index = 0
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._running = False
        self._health_check_task = None
        
        # Performance tracking
        self.request_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "requests_per_second": 0.0
        }
        
    def add_node(self, node: InferenceNode) -> None:
        """Add an inference node to the cluster."""
        with self._lock:
            self.nodes[node.node_id] = node
            logging.info(f"Added inference node: {node.node_id} at {node.endpoint}")
            
    def remove_node(self, node_id: str) -> None:
        """Remove an inference node from the cluster."""
        with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                logging.info(f"Removed inference node: {node_id}")
                
    def start(self) -> None:
        """Start the load balancer and health monitoring."""
        self._running = True
        self._health_check_task = threading.Thread(target=self._health_check_loop, daemon=True)
        self._health_check_task.start()
        logging.info("Adaptive load balancer started")
        
    def stop(self) -> None:
        """Stop the load balancer."""
        self._running = False
        if self._health_check_task:
            self._health_check_task.join(timeout=5.0)
        self._executor.shutdown(wait=True)
        logging.info("Adaptive load balancer stopped")
        
    def route_request(self, context: RequestContext) -> Optional[InferenceNode]:
        """Route a request to the best available node."""
        with self._lock:
            healthy_nodes = [
                node for node in self.nodes.values()
                if node.status in [NodeStatus.HEALTHY, NodeStatus.DEGRADED]
                and not node.circuit_breaker_open
            ]
            
            if not healthy_nodes:
                logging.error("No healthy nodes available for routing")
                return None
                
            # Apply routing strategy
            if self.strategy == LoadBalancingStrategy.INTELLIGENT_ADAPTIVE:
                return self._intelligent_adaptive_routing(healthy_nodes, context)
            elif self.strategy == LoadBalancingStrategy.HEALTH_AWARE:
                return self._health_aware_routing(healthy_nodes, context)
            elif self.strategy == LoadBalancingStrategy.RESOURCE_AWARE:
                return self._resource_aware_routing(healthy_nodes, context)
            elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                return self._least_response_time_routing(healthy_nodes)
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections_routing(healthy_nodes)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_routing(healthy_nodes)
            else:  # ROUND_ROBIN
                return self._round_robin_routing(healthy_nodes)
                
    def _intelligent_adaptive_routing(
        self, 
        nodes: List[InferenceNode], 
        context: RequestContext
    ) -> InferenceNode:
        """Intelligent adaptive routing based on multiple factors."""
        
        # Check for patient affinity (stick to same node for consistency)
        if context.patient_id and context.patient_id in self.routing_table:
            preferred_node_id = self.routing_table[context.patient_id]
            preferred_node = next((n for n in nodes if n.node_id == preferred_node_id), None)
            if preferred_node and preferred_node.metrics.health_score > 0.7:
                return preferred_node
                
        # Score nodes based on multiple criteria
        node_scores = []
        for node in nodes:
            score = self._calculate_node_score(node, context)
            node_scores.append((score, node))
            
        # Sort by score (highest first) and return best node
        node_scores.sort(reverse=True, key=lambda x: x[0])
        best_node = node_scores[0][1]
        
        # Update patient affinity
        if context.patient_id:
            self.routing_table[context.patient_id] = best_node.node_id
            
        return best_node
        
    def _calculate_node_score(self, node: InferenceNode, context: RequestContext) -> float:
        """Calculate comprehensive node score for routing decisions."""
        metrics = node.metrics
        
        # Base health score
        score = metrics.health_score
        
        # Adjust for current load
        load_factor = min(metrics.active_connections / node.max_concurrent_requests, 1.0)
        score *= (1.0 - load_factor * 0.5)  # Reduce score based on load
        
        # Adjust for response time
        if metrics.average_response_time > 0:
            response_time_factor = min(metrics.average_response_time / 2.0, 1.0)
            score *= (1.0 - response_time_factor * 0.3)
            
        # Boost score for emergency requests on high-performance nodes
        if context.medical_urgency == "emergency":
            if metrics.inference_latency_p95 < 1.0:  # Fast node
                score *= 1.2
                
        # Adjust for model accuracy (critical for medical applications)
        accuracy_boost = metrics.model_accuracy * 0.1
        score += accuracy_boost
        
        # Penalty for high error rates
        error_penalty = metrics.error_rate * 0.5
        score -= error_penalty
        
        return max(0.0, min(1.0, score))
        
    def _health_aware_routing(
        self, 
        nodes: List[InferenceNode], 
        context: RequestContext
    ) -> InferenceNode:
        """Route based primarily on node health scores."""
        best_node = max(nodes, key=lambda n: n.metrics.health_score)
        return best_node
        
    def _resource_aware_routing(
        self, 
        nodes: List[InferenceNode], 
        context: RequestContext
    ) -> InferenceNode:
        """Route based on available resources."""
        
        def resource_score(node: InferenceNode) -> float:
            m = node.metrics
            cpu_availability = 1.0 - m.cpu_usage
            memory_availability = 1.0 - m.memory_usage
            gpu_availability = 1.0 - m.gpu_usage if m.gpu_usage > 0 else 1.0
            
            # Weight GPU more heavily for AI workloads
            return 0.3 * cpu_availability + 0.2 * memory_availability + 0.5 * gpu_availability
            
        best_node = max(nodes, key=resource_score)
        return best_node
        
    def _least_response_time_routing(self, nodes: List[InferenceNode]) -> InferenceNode:
        """Route to node with lowest average response time."""
        return min(nodes, key=lambda n: n.metrics.average_response_time or float('inf'))
        
    def _least_connections_routing(self, nodes: List[InferenceNode]) -> InferenceNode:
        """Route to node with fewest active connections."""
        return min(nodes, key=lambda n: n.metrics.active_connections)
        
    def _weighted_round_robin_routing(self, nodes: List[InferenceNode]) -> InferenceNode:
        """Weighted round-robin routing based on node weights."""
        total_weight = sum(node.weight for node in nodes)
        
        if total_weight == 0:
            return nodes[self._round_robin_index % len(nodes)]
            
        # Select based on weight distribution
        target_weight = random.uniform(0, total_weight)
        current_weight = 0
        
        for node in nodes:
            current_weight += node.weight
            if current_weight >= target_weight:
                return node
                
        return nodes[-1]  # Fallback
        
    def _round_robin_routing(self, nodes: List[InferenceNode]) -> InferenceNode:
        """Simple round-robin routing."""
        node = nodes[self._round_robin_index % len(nodes)]
        self._round_robin_index += 1
        return node
        
    def update_node_metrics(self, node_id: str, metrics: NodeMetrics) -> None:
        """Update metrics for a specific node."""
        with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id].metrics = metrics
                self.metrics_history[node_id].append(metrics)
                
                # Update node status based on health score
                health_score = metrics.health_score
                if health_score >= 0.8:
                    self.nodes[node_id].status = NodeStatus.HEALTHY
                elif health_score >= 0.5:
                    self.nodes[node_id].status = NodeStatus.DEGRADED
                else:
                    self.nodes[node_id].status = NodeStatus.UNHEALTHY
                    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        with self._lock:
            healthy_nodes = sum(1 for n in self.nodes.values() if n.status == NodeStatus.HEALTHY)
            degraded_nodes = sum(1 for n in self.nodes.values() if n.status == NodeStatus.DEGRADED)
            unhealthy_nodes = sum(1 for n in self.nodes.values() if n.status == NodeStatus.UNHEALTHY)
            
            total_capacity = sum(n.metrics.throughput_capacity for n in self.nodes.values())
            current_load = sum(n.metrics.active_connections for n in self.nodes.values())
            
            avg_response_time = sum(n.metrics.average_response_time for n in self.nodes.values()) / max(len(self.nodes), 1)
            avg_accuracy = sum(n.metrics.model_accuracy for n in self.nodes.values()) / max(len(self.nodes), 1)
            
            return {
                "total_nodes": len(self.nodes),
                "healthy_nodes": healthy_nodes,
                "degraded_nodes": degraded_nodes,
                "unhealthy_nodes": unhealthy_nodes,
                "cluster_capacity": total_capacity,
                "current_load": current_load,
                "load_percentage": (current_load / max(total_capacity, 1)) * 100,
                "average_response_time": avg_response_time,
                "average_model_accuracy": avg_accuracy,
                "strategy": self.strategy.value,
                "request_stats": self.request_stats.copy()
            }
            
    def _health_check_loop(self) -> None:
        """Background health monitoring loop."""
        while self._running:
            try:
                self._perform_health_checks()
                time.sleep(self.health_check_interval)
            except Exception as e:
                logging.error(f"Health check loop error: {e}")
                time.sleep(self.health_check_interval * 2)  # Back off on errors
                
    def _perform_health_checks(self) -> None:
        """Perform health checks on all nodes."""
        with self._lock:
            for node in self.nodes.values():
                try:
                    # Simulate health check (in real implementation, would ping node)
                    health_status = self._check_node_health(node)
                    
                    if health_status:
                        node.failure_count = 0
                        node.circuit_breaker_open = False
                        node.last_health_check = time.time()
                    else:
                        node.failure_count += 1
                        if node.failure_count >= 3:
                            node.status = NodeStatus.UNHEALTHY
                            node.circuit_breaker_open = True
                            logging.warning(f"Node {node.node_id} marked as unhealthy")
                            
                except Exception as e:
                    logging.error(f"Health check failed for node {node.node_id}: {e}")
                    node.failure_count += 1
                    
    def _check_node_health(self, node: InferenceNode) -> bool:
        """Check if a node is healthy (placeholder implementation)."""
        # In real implementation, would make HTTP request to health endpoint
        # For now, simulate based on metrics
        if node.metrics:
            return (
                node.metrics.health_score > 0.3 and
                node.metrics.error_rate < 0.5 and
                node.metrics.cpu_usage < 0.95
            )
        return True  # Assume healthy if no metrics yet


class MedicalAIClusterManager:
    """High-level cluster manager for medical AI inference systems."""
    
    def __init__(self):
        self.load_balancer = AdaptiveLoadBalancer()
        self.auto_scaler = AutoScaler()
        self.request_processor = RequestProcessor(self.load_balancer)
        
    def start_cluster(self) -> None:
        """Start the entire cluster management system."""
        self.load_balancer.start()
        self.auto_scaler.start()
        logging.info("Medical AI cluster manager started")
        
    def stop_cluster(self) -> None:
        """Stop the cluster management system."""
        self.load_balancer.stop()
        self.auto_scaler.stop()
        logging.info("Medical AI cluster manager stopped")
        
    def add_inference_node(self, endpoint: str, node_id: Optional[str] = None) -> str:
        """Add a new inference node to the cluster."""
        if node_id is None:
            node_id = f"node_{hashlib.md5(endpoint.encode()).hexdigest()[:8]}"
            
        node = InferenceNode(node_id=node_id, endpoint=endpoint)
        self.load_balancer.add_node(node)
        return node_id
        
    async def process_inference_request(
        self,
        image_data: Any,
        patient_id: Optional[str] = None,
        urgency: str = "routine",
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Process an inference request through the cluster."""
        request_id = f"req_{int(time.time() * 1000)}"
        context = RequestContext(
            request_id=request_id,
            patient_id=patient_id,
            medical_urgency=urgency,
            timeout=timeout
        )
        
        result = await self.request_processor.process_request(context, image_data)
        return result


class AutoScaler:
    """Automatic scaling for inference nodes based on load."""
    
    def __init__(self, target_cpu_usage: float = 0.7, scale_up_threshold: float = 0.8):
        self.target_cpu_usage = target_cpu_usage
        self.scale_up_threshold = scale_up_threshold
        self._running = False
        
    def start(self) -> None:
        """Start auto-scaling monitoring."""
        self._running = True
        
    def stop(self) -> None:
        """Stop auto-scaling monitoring."""
        self._running = False
        
    def should_scale_up(self, cluster_metrics: Dict[str, Any]) -> bool:
        """Determine if cluster should scale up."""
        return cluster_metrics.get("load_percentage", 0) > self.scale_up_threshold * 100
        
    def should_scale_down(self, cluster_metrics: Dict[str, Any]) -> bool:
        """Determine if cluster should scale down."""
        return (
            cluster_metrics.get("load_percentage", 100) < self.target_cpu_usage * 50 and
            cluster_metrics.get("healthy_nodes", 0) > 2  # Keep minimum nodes
        )


class RequestProcessor:
    """Process inference requests through the load balancer."""
    
    def __init__(self, load_balancer: AdaptiveLoadBalancer):
        self.load_balancer = load_balancer
        
    async def process_request(self, context: RequestContext, data: Any) -> Dict[str, Any]:
        """Process a single inference request."""
        start_time = time.time()
        
        try:
            # Route request to best node
            node = self.load_balancer.route_request(context)
            if not node:
                raise Exception("No healthy nodes available")
                
            # Simulate inference (in real implementation, would call node endpoint)
            result = await self._call_inference_node(node, context, data)
            
            # Update metrics
            response_time = time.time() - start_time
            self._update_request_stats(True, response_time)
            
            return {
                "prediction": result,
                "node_id": node.node_id,
                "response_time": response_time,
                "request_id": context.request_id
            }
            
        except Exception as e:
            response_time = time.time() - start_time
            self._update_request_stats(False, response_time)
            
            return {
                "error": str(e),
                "response_time": response_time,
                "request_id": context.request_id
            }
            
    async def _call_inference_node(
        self, 
        node: InferenceNode, 
        context: RequestContext, 
        data: Any
    ) -> Dict[str, Any]:
        """Call inference endpoint on selected node."""
        # Simulate network call and inference
        await asyncio.sleep(random.uniform(0.1, 0.5))  # Simulate processing time
        
        # Simulate occasional failures
        if random.random() < 0.05:  # 5% failure rate
            raise Exception(f"Inference failed on node {node.node_id}")
            
        return {
            "prediction": "normal" if random.random() > 0.3 else "pneumonia",
            "confidence": random.uniform(0.7, 0.99),
            "model_version": "v2.1.0"
        }
        
    def _update_request_stats(self, success: bool, response_time: float) -> None:
        """Update request statistics."""
        stats = self.load_balancer.request_stats
        stats["total_requests"] += 1
        
        if success:
            stats["successful_requests"] += 1
        else:
            stats["failed_requests"] += 1
            
        # Update average response time (exponential moving average)
        alpha = 0.1
        current_avg = stats["average_response_time"]
        stats["average_response_time"] = alpha * response_time + (1 - alpha) * current_avg


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Create cluster manager
        cluster = MedicalAIClusterManager()
        
        # Add some nodes
        cluster.add_inference_node("http://node1:8080", "node1")
        cluster.add_inference_node("http://node2:8080", "node2")
        cluster.add_inference_node("http://node3:8080", "node3")
        
        # Start cluster
        cluster.start_cluster()
        
        # Simulate some requests
        for i in range(10):
            urgency = "emergency" if i % 5 == 0 else "routine"
            result = await cluster.process_inference_request(
                image_data=f"image_{i}",
                patient_id=f"patient_{i % 3}",
                urgency=urgency
            )
            print(f"Request {i}: {result}")
            
        # Get cluster status
        status = cluster.load_balancer.get_cluster_status()
        print(f"Cluster status: {json.dumps(status, indent=2)}")
        
        # Stop cluster
        cluster.stop_cluster()
        
    asyncio.run(main())