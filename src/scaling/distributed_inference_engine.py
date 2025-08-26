"""Distributed Inference Engine for Scalable Medical AI Systems.

Implements high-performance distributed inference with automatic scaling,
load balancing, and fault tolerance for medical AI applications.
"""

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable, AsyncIterator
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import queue

import numpy as np

logger = logging.getLogger(__name__)


class InferenceStatus(Enum):
    """Inference request status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class WorkerStatus(Enum):
    """Worker node status."""
    HEALTHY = "healthy"
    BUSY = "busy"
    DEGRADED = "degraded"
    OFFLINE = "offline"


@dataclass
class InferenceRequest:
    """Inference request data structure."""
    request_id: str
    data: Any
    model_name: str
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=critical
    timeout: float = 30.0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: InferenceStatus = InferenceStatus.PENDING
    worker_id: Optional[str] = None
    result: Any = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class WorkerNode:
    """Worker node information."""
    worker_id: str
    endpoint: str
    status: WorkerStatus
    current_load: int = 0
    max_capacity: int = 10
    model_capabilities: List[str] = field(default_factory=list)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    average_processing_time: float = 1.0
    total_processed: int = 0
    error_rate: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0


@dataclass
class InferenceMetrics:
    """Inference performance metrics."""
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    average_latency: float = 0.0
    p95_latency: float = 0.0
    throughput_rps: float = 0.0
    current_queue_size: int = 0
    active_workers: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class RequestQueue:
    """Priority-based request queue."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.queues = {
            4: deque(),  # Critical
            3: deque(),  # High
            2: deque(),  # Medium
            1: deque()   # Low
        }
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
    
    def put(self, request: InferenceRequest) -> bool:
        """Add request to queue."""
        with self.lock:
            current_size = sum(len(q) for q in self.queues.values())
            
            if current_size >= self.max_size:
                # Queue is full, drop lowest priority request
                if self.queues[1]:
                    dropped_request = self.queues[1].popleft()
                    logger.warning(f"Dropped low priority request: {dropped_request.request_id}")
                else:
                    logger.error("Queue is full, cannot add request")
                    return False
            
            self.queues[request.priority].append(request)
            self.condition.notify()
            return True
    
    def get(self, timeout: Optional[float] = None) -> Optional[InferenceRequest]:
        """Get highest priority request from queue."""
        with self.condition:
            # Try to get request immediately
            request = self._get_next_request()
            if request:
                return request
            
            # Wait for request if timeout specified
            if timeout is None or timeout <= 0:
                return None
            
            end_time = time.time() + timeout
            while time.time() < end_time:
                remaining_timeout = end_time - time.time()
                if remaining_timeout <= 0:
                    break
                
                self.condition.wait(timeout=remaining_timeout)
                request = self._get_next_request()
                if request:
                    return request
            
            return None
    
    def _get_next_request(self) -> Optional[InferenceRequest]:
        """Get next request from highest priority queue."""
        for priority in sorted(self.queues.keys(), reverse=True):
            if self.queues[priority]:
                return self.queues[priority].popleft()
        return None
    
    def size(self) -> int:
        """Get total queue size."""
        with self.lock:
            return sum(len(q) for q in self.queues.values())
    
    def priority_distribution(self) -> Dict[int, int]:
        """Get distribution of requests by priority."""
        with self.lock:
            return {priority: len(queue) for priority, queue in self.queues.items()}


class WorkerPool:
    """Pool of worker nodes for distributed inference."""
    
    def __init__(self):
        self.workers: Dict[str, WorkerNode] = {}
        self.worker_lock = threading.RLock()
        self.metrics_history = deque(maxlen=1000)
        
        # Worker health monitoring
        self.health_check_interval = 30.0
        self.health_check_thread = None
        self.is_monitoring = False
    
    def add_worker(self, 
                   worker_id: str,
                   endpoint: str,
                   max_capacity: int = 10,
                   model_capabilities: List[str] = None) -> bool:
        """Add worker to pool."""
        with self.worker_lock:
            if worker_id in self.workers:
                logger.warning(f"Worker {worker_id} already exists, updating...")
            
            self.workers[worker_id] = WorkerNode(
                worker_id=worker_id,
                endpoint=endpoint,
                status=WorkerStatus.HEALTHY,
                max_capacity=max_capacity,
                model_capabilities=model_capabilities or []
            )
            
            logger.info(f"Added worker {worker_id} with capacity {max_capacity}")
            return True
    
    def remove_worker(self, worker_id: str) -> bool:
        """Remove worker from pool."""
        with self.worker_lock:
            if worker_id in self.workers:
                del self.workers[worker_id]
                logger.info(f"Removed worker {worker_id}")
                return True
            return False
    
    def get_available_workers(self, 
                             model_name: str = None,
                             min_capacity: int = 1) -> List[WorkerNode]:
        """Get available workers for inference."""
        with self.worker_lock:
            available_workers = []
            
            for worker in self.workers.values():
                # Check if worker is healthy
                if worker.status not in [WorkerStatus.HEALTHY, WorkerStatus.BUSY]:
                    continue
                
                # Check if worker has available capacity
                if worker.current_load >= worker.max_capacity:
                    continue
                
                # Check if worker has required capacity
                remaining_capacity = worker.max_capacity - worker.current_load
                if remaining_capacity < min_capacity:
                    continue
                
                # Check if worker supports the model
                if model_name and worker.model_capabilities:
                    if model_name not in worker.model_capabilities:
                        continue
                
                available_workers.append(worker)
            
            # Sort by load (prefer less loaded workers)
            available_workers.sort(key=lambda w: w.current_load / w.max_capacity)
            
            return available_workers
    
    def select_best_worker(self, 
                          model_name: str,
                          priority: int = 1) -> Optional[WorkerNode]:
        """Select best worker for inference request."""
        available_workers = self.get_available_workers(model_name)
        
        if not available_workers:
            return None
        
        # Scoring function for worker selection
        def worker_score(worker: WorkerNode) -> float:
            # Factors: load, processing time, error rate
            load_factor = 1.0 - (worker.current_load / worker.max_capacity)
            speed_factor = 1.0 / (worker.average_processing_time + 0.1)
            reliability_factor = 1.0 - worker.error_rate
            
            # Weight factors based on priority
            if priority >= 3:  # High/Critical priority
                # Prioritize reliability and speed
                score = reliability_factor * 0.5 + speed_factor * 0.3 + load_factor * 0.2
            else:  # Normal/Low priority
                # Prioritize load balancing
                score = load_factor * 0.5 + reliability_factor * 0.3 + speed_factor * 0.2
            
            return score
        
        # Select worker with highest score
        best_worker = max(available_workers, key=worker_score)
        return best_worker
    
    def assign_request(self, worker_id: str, request: InferenceRequest) -> bool:
        """Assign request to worker."""
        with self.worker_lock:
            if worker_id not in self.workers:
                return False
            
            worker = self.workers[worker_id]
            
            if worker.current_load >= worker.max_capacity:
                return False
            
            worker.current_load += 1
            if worker.current_load >= worker.max_capacity:
                worker.status = WorkerStatus.BUSY
            
            request.worker_id = worker_id
            request.started_at = datetime.now()
            request.status = InferenceStatus.PROCESSING
            
            return True
    
    def complete_request(self, 
                        worker_id: str, 
                        request: InferenceRequest,
                        success: bool,
                        processing_time: float):
        """Mark request as completed and update worker metrics."""
        with self.worker_lock:
            if worker_id not in self.workers:
                return
            
            worker = self.workers[worker_id]
            
            # Update worker load
            worker.current_load = max(0, worker.current_load - 1)
            if worker.current_load < worker.max_capacity:
                worker.status = WorkerStatus.HEALTHY
            
            # Update worker metrics
            worker.total_processed += 1
            
            # Update average processing time (exponential moving average)
            alpha = 0.1
            worker.average_processing_time = (
                (1 - alpha) * worker.average_processing_time + 
                alpha * processing_time
            )
            
            # Update error rate
            if success:
                worker.error_rate = worker.error_rate * 0.95  # Decay error rate on success
            else:
                worker.error_rate = min(1.0, worker.error_rate + 0.1)  # Increase on failure
            
            # Update heartbeat
            worker.last_heartbeat = datetime.now()
    
    def start_health_monitoring(self):
        """Start worker health monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.health_check_thread = threading.Thread(target=self._health_monitoring_loop)
        self.health_check_thread.daemon = True
        self.health_check_thread.start()
        
        logger.info("Worker health monitoring started")
    
    def stop_health_monitoring(self):
        """Stop worker health monitoring."""
        self.is_monitoring = False
        if self.health_check_thread:
            self.health_check_thread.join()
        
        logger.info("Worker health monitoring stopped")
    
    def _health_monitoring_loop(self):
        """Health monitoring loop."""
        while self.is_monitoring:
            try:
                self._check_worker_health()
                time.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(5)
    
    def _check_worker_health(self):
        """Check health of all workers."""
        current_time = datetime.now()
        unhealthy_threshold = timedelta(seconds=self.health_check_interval * 2)
        
        with self.worker_lock:
            for worker_id, worker in list(self.workers.items()):
                time_since_heartbeat = current_time - worker.last_heartbeat
                
                if time_since_heartbeat > unhealthy_threshold:
                    if worker.status != WorkerStatus.OFFLINE:
                        logger.warning(f"Worker {worker_id} appears offline")
                        worker.status = WorkerStatus.OFFLINE
                        worker.current_load = 0  # Reset load for offline worker
                
                elif worker.error_rate > 0.5:  # High error rate
                    if worker.status != WorkerStatus.DEGRADED:
                        logger.warning(f"Worker {worker_id} is degraded (error rate: {worker.error_rate:.2f})")
                        worker.status = WorkerStatus.DEGRADED
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get worker pool status."""
        with self.worker_lock:
            total_workers = len(self.workers)
            healthy_workers = sum(1 for w in self.workers.values() if w.status == WorkerStatus.HEALTHY)
            busy_workers = sum(1 for w in self.workers.values() if w.status == WorkerStatus.BUSY)
            degraded_workers = sum(1 for w in self.workers.values() if w.status == WorkerStatus.DEGRADED)
            offline_workers = sum(1 for w in self.workers.values() if w.status == WorkerStatus.OFFLINE)
            
            total_capacity = sum(w.max_capacity for w in self.workers.values())
            current_load = sum(w.current_load for w in self.workers.values())
            
            avg_processing_time = np.mean([w.average_processing_time for w in self.workers.values()]) if self.workers else 0
            avg_error_rate = np.mean([w.error_rate for w in self.workers.values()]) if self.workers else 0
            
            return {
                'total_workers': total_workers,
                'healthy_workers': healthy_workers,
                'busy_workers': busy_workers,
                'degraded_workers': degraded_workers,
                'offline_workers': offline_workers,
                'total_capacity': total_capacity,
                'current_load': current_load,
                'utilization': current_load / total_capacity if total_capacity > 0 else 0,
                'average_processing_time': avg_processing_time,
                'average_error_rate': avg_error_rate
            }


class DistributedInferenceEngine:
    """Main distributed inference engine."""
    
    def __init__(self, max_queue_size: int = 10000):
        self.request_queue = RequestQueue(max_queue_size)
        self.worker_pool = WorkerPool()
        self.active_requests: Dict[str, InferenceRequest] = {}
        
        # Processing threads
        self.num_dispatcher_threads = 4
        self.dispatcher_threads = []
        self.is_running = False
        
        # Metrics and monitoring
        self.metrics = InferenceMetrics()
        self.metrics_history = deque(maxlen=1000)
        self.request_latencies = deque(maxlen=1000)
        
        # Auto-scaling parameters
        self.auto_scaling_enabled = True
        self.scale_up_threshold = 0.8  # Scale up when utilization > 80%
        self.scale_down_threshold = 0.3  # Scale down when utilization < 30%
        self.min_workers = 1
        self.max_workers = 100
        
        # Request tracking
        self.request_lock = threading.RLock()
        
    def start(self):
        """Start the distributed inference engine."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start dispatcher threads
        for i in range(self.num_dispatcher_threads):
            thread = threading.Thread(target=self._dispatcher_loop, args=(f"dispatcher_{i}",))
            thread.daemon = True
            thread.start()
            self.dispatcher_threads.append(thread)
        
        # Start worker health monitoring
        self.worker_pool.start_health_monitoring()
        
        # Start metrics collection
        self._start_metrics_collection()
        
        logger.info("Distributed inference engine started")
    
    def stop(self):
        """Stop the distributed inference engine."""
        self.is_running = False
        
        # Wait for dispatcher threads to finish
        for thread in self.dispatcher_threads:
            thread.join(timeout=5.0)
        
        # Stop worker health monitoring
        self.worker_pool.stop_health_monitoring()
        
        logger.info("Distributed inference engine stopped")
    
    async def submit_inference_request(self, 
                                     data: Any,
                                     model_name: str,
                                     priority: int = 2,
                                     timeout: float = 30.0) -> str:
        """Submit inference request and return request ID."""
        request_id = str(uuid.uuid4())
        
        request = InferenceRequest(
            request_id=request_id,
            data=data,
            model_name=model_name,
            priority=priority,
            timeout=timeout
        )
        
        # Add to active requests
        with self.request_lock:
            self.active_requests[request_id] = request
        
        # Add to queue
        if not self.request_queue.put(request):
            # Remove from active requests if queueing failed
            with self.request_lock:
                del self.active_requests[request_id]
            raise Exception("Request queue is full")
        
        logger.info(f"Submitted inference request {request_id} for model {model_name}")
        return request_id
    
    async def get_inference_result(self, 
                                  request_id: str,
                                  timeout: float = None) -> Optional[InferenceRequest]:
        """Get inference result for request ID."""
        start_time = time.time()
        timeout = timeout or 60.0
        
        while time.time() - start_time < timeout:
            with self.request_lock:
                if request_id in self.active_requests:
                    request = self.active_requests[request_id]
                    
                    if request.status in [InferenceStatus.COMPLETED, 
                                        InferenceStatus.FAILED, 
                                        InferenceStatus.TIMEOUT]:
                        # Remove from active requests
                        del self.active_requests[request_id]
                        return request
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
        
        # Timeout reached
        with self.request_lock:
            if request_id in self.active_requests:
                request = self.active_requests[request_id]
                request.status = InferenceStatus.TIMEOUT
                request.completed_at = datetime.now()
                del self.active_requests[request_id]
                return request
        
        return None
    
    async def inference(self, 
                       data: Any,
                       model_name: str,
                       priority: int = 2,
                       timeout: float = 30.0) -> Any:
        """Perform inference and return result directly."""
        request_id = await self.submit_inference_request(data, model_name, priority, timeout)
        result_request = await self.get_inference_result(request_id, timeout)
        
        if result_request is None:
            raise TimeoutError(f"Inference request {request_id} timed out")
        
        if result_request.status == InferenceStatus.COMPLETED:
            return result_request.result
        elif result_request.status == InferenceStatus.FAILED:
            raise Exception(f"Inference failed: {result_request.error_message}")
        else:
            raise Exception(f"Inference request in unexpected state: {result_request.status}")
    
    def _dispatcher_loop(self, dispatcher_name: str):
        """Main dispatcher loop for processing requests."""
        logger.info(f"Started dispatcher: {dispatcher_name}")
        
        while self.is_running:
            try:
                # Get request from queue
                request = self.request_queue.get(timeout=1.0)
                if request is None:
                    continue
                
                # Check if request has timed out
                if self._is_request_expired(request):
                    self._complete_request_with_timeout(request)
                    continue
                
                # Find available worker
                worker = self.worker_pool.select_best_worker(request.model_name, request.priority)
                if worker is None:
                    # No available workers, put request back in queue
                    # For simplicity, we'll just wait and try again
                    time.sleep(0.1)
                    if not self.request_queue.put(request):
                        self._complete_request_with_error(request, "Failed to requeue request")
                    continue
                
                # Assign request to worker
                if not self.worker_pool.assign_request(worker.worker_id, request):
                    # Assignment failed, put request back in queue
                    if not self.request_queue.put(request):
                        self._complete_request_with_error(request, "Failed to assign to worker")
                    continue
                
                # Process request on worker
                self._process_request_on_worker(request, worker)
                
            except Exception as e:
                logger.error(f"Error in dispatcher {dispatcher_name}: {e}")
                time.sleep(1)
        
        logger.info(f"Stopped dispatcher: {dispatcher_name}")
    
    def _is_request_expired(self, request: InferenceRequest) -> bool:
        """Check if request has expired."""
        if request.timeout <= 0:
            return False
        
        elapsed = (datetime.now() - request.created_at).total_seconds()
        return elapsed > request.timeout
    
    def _complete_request_with_timeout(self, request: InferenceRequest):
        """Complete request with timeout status."""
        request.status = InferenceStatus.TIMEOUT
        request.completed_at = datetime.now()
        request.error_message = "Request timed out"
        
        logger.warning(f"Request {request.request_id} timed out")
        self._update_metrics_for_completed_request(request, False, 0)
    
    def _complete_request_with_error(self, request: InferenceRequest, error_message: str):
        """Complete request with error status."""
        request.status = InferenceStatus.FAILED
        request.completed_at = datetime.now()
        request.error_message = error_message
        
        logger.error(f"Request {request.request_id} failed: {error_message}")
        self._update_metrics_for_completed_request(request, False, 0)
    
    def _process_request_on_worker(self, request: InferenceRequest, worker: WorkerNode):
        """Process request on worker node."""
        start_time = time.time()
        
        try:
            # Simulate inference processing
            # In real implementation, this would call the actual worker endpoint
            result = self._simulate_inference(request.data, request.model_name, worker)
            
            processing_time = time.time() - start_time
            
            # Complete request successfully
            request.status = InferenceStatus.COMPLETED
            request.completed_at = datetime.now()
            request.result = result
            
            # Update worker metrics
            self.worker_pool.complete_request(worker.worker_id, request, True, processing_time)
            
            # Update global metrics
            self._update_metrics_for_completed_request(request, True, processing_time)
            
            logger.debug(f"Request {request.request_id} completed successfully on worker {worker.worker_id}")
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Complete request with error
            request.status = InferenceStatus.FAILED
            request.completed_at = datetime.now()
            request.error_message = str(e)
            
            # Update worker metrics
            self.worker_pool.complete_request(worker.worker_id, request, False, processing_time)
            
            # Update global metrics
            self._update_metrics_for_completed_request(request, False, processing_time)
            
            logger.error(f"Request {request.request_id} failed on worker {worker.worker_id}: {e}")
    
    def _simulate_inference(self, data: Any, model_name: str, worker: WorkerNode) -> Any:
        """Simulate inference processing."""
        # Simulate processing time based on worker performance
        base_time = worker.average_processing_time
        actual_time = max(0.1, np.random.normal(base_time, base_time * 0.2))
        time.sleep(actual_time)
        
        # Simulate occasional failures based on worker error rate
        if np.random.random() < worker.error_rate:
            raise Exception(f"Simulated inference failure on worker {worker.worker_id}")
        
        # Return simulated result
        return {
            'prediction': np.random.random(),
            'confidence': np.random.uniform(0.7, 0.99),
            'model_name': model_name,
            'worker_id': worker.worker_id,
            'processing_time': actual_time
        }
    
    def _update_metrics_for_completed_request(self, 
                                            request: InferenceRequest,
                                            success: bool,
                                            processing_time: float):
        """Update metrics for completed request."""
        # Calculate latency
        latency = (request.completed_at - request.created_at).total_seconds()
        self.request_latencies.append(latency)
        
        # Update metrics
        self.metrics.total_requests += 1
        if success:
            self.metrics.completed_requests += 1
        else:
            self.metrics.failed_requests += 1
        
        # Update average latency
        if len(self.request_latencies) > 0:
            self.metrics.average_latency = np.mean(list(self.request_latencies))
            self.metrics.p95_latency = np.percentile(list(self.request_latencies), 95)
        
        # Update throughput (requests per second over last minute)
        recent_requests = [
            r for r in self.request_latencies
            if r > (time.time() - 60)  # Last minute
        ]
        self.metrics.throughput_rps = len(recent_requests) / 60.0
        
        # Update queue size
        self.metrics.current_queue_size = self.request_queue.size()
        
        # Update active workers
        pool_status = self.worker_pool.get_pool_status()
        self.metrics.active_workers = pool_status['healthy_workers'] + pool_status['busy_workers']
        
        self.metrics.timestamp = datetime.now()
    
    def _start_metrics_collection(self):
        """Start metrics collection thread."""
        def metrics_collection_loop():
            while self.is_running:
                try:
                    # Save current metrics to history
                    self.metrics_history.append(self.metrics)
                    
                    # Perform auto-scaling if enabled
                    if self.auto_scaling_enabled:
                        self._auto_scale()
                    
                    time.sleep(10)  # Collect metrics every 10 seconds
                    
                except Exception as e:
                    logger.error(f"Metrics collection error: {e}")
                    time.sleep(5)
        
        metrics_thread = threading.Thread(target=metrics_collection_loop)
        metrics_thread.daemon = True
        metrics_thread.start()
    
    def _auto_scale(self):
        """Perform auto-scaling based on current load."""
        pool_status = self.worker_pool.get_pool_status()
        utilization = pool_status['utilization']
        current_workers = pool_status['healthy_workers'] + pool_status['busy_workers']
        
        if utilization > self.scale_up_threshold and current_workers < self.max_workers:
            # Scale up - add more workers
            num_to_add = min(2, self.max_workers - current_workers)
            for i in range(num_to_add):
                worker_id = f"auto_worker_{int(time.time())}_{i}"
                self.worker_pool.add_worker(
                    worker_id=worker_id,
                    endpoint=f"http://auto-worker-{worker_id}:8000",
                    max_capacity=10,
                    model_capabilities=["default_model"]
                )
            
            logger.info(f"Auto-scaled up: added {num_to_add} workers (utilization: {utilization:.2f})")
        
        elif utilization < self.scale_down_threshold and current_workers > self.min_workers:
            # Scale down - remove workers (in real implementation, this would be more careful)
            logger.info(f"Auto-scaling down opportunity detected (utilization: {utilization:.2f})")
    
    def add_worker(self, 
                   worker_id: str,
                   endpoint: str,
                   max_capacity: int = 10,
                   model_capabilities: List[str] = None) -> bool:
        """Add worker to the pool."""
        return self.worker_pool.add_worker(worker_id, endpoint, max_capacity, model_capabilities)
    
    def remove_worker(self, worker_id: str) -> bool:
        """Remove worker from the pool."""
        return self.worker_pool.remove_worker(worker_id)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        pool_status = self.worker_pool.get_pool_status()
        queue_distribution = self.request_queue.priority_distribution()
        
        return {
            'engine_status': 'running' if self.is_running else 'stopped',
            'metrics': {
                'total_requests': self.metrics.total_requests,
                'completed_requests': self.metrics.completed_requests,
                'failed_requests': self.metrics.failed_requests,
                'success_rate': (
                    self.metrics.completed_requests / self.metrics.total_requests
                    if self.metrics.total_requests > 0 else 0
                ),
                'average_latency': self.metrics.average_latency,
                'p95_latency': self.metrics.p95_latency,
                'throughput_rps': self.metrics.throughput_rps,
                'current_queue_size': self.metrics.current_queue_size
            },
            'worker_pool': pool_status,
            'queue_distribution': queue_distribution,
            'active_requests': len(self.active_requests),
            'auto_scaling': {
                'enabled': self.auto_scaling_enabled,
                'scale_up_threshold': self.scale_up_threshold,
                'scale_down_threshold': self.scale_down_threshold
            }
        }


async def demonstrate_distributed_inference():
    """Demonstrate distributed inference engine."""
    print("Distributed Inference Engine Demo")
    print("=" * 50)
    
    # Create inference engine
    engine = DistributedInferenceEngine()
    
    # Add some workers
    workers = [
        ("worker_1", "http://worker1:8000", 5, ["pneumonia_model", "xray_classifier"]),
        ("worker_2", "http://worker2:8000", 8, ["pneumonia_model"]),
        ("worker_3", "http://worker3:8000", 6, ["xray_classifier", "general_model"]),
    ]
    
    print("\n1. Adding workers to pool:")
    for worker_id, endpoint, capacity, models in workers:
        engine.add_worker(worker_id, endpoint, capacity, models)
        print(f"  Added {worker_id} (capacity: {capacity}, models: {models})")
    
    # Start engine
    print("\n2. Starting inference engine...")
    engine.start()
    
    # Submit some inference requests
    print("\n3. Submitting inference requests...")
    request_ids = []
    
    # Submit requests with different priorities
    test_requests = [
        ("sample_xray_1.jpg", "pneumonia_model", 3),  # High priority
        ("sample_xray_2.jpg", "pneumonia_model", 2),  # Medium priority
        ("sample_xray_3.jpg", "xray_classifier", 1),  # Low priority
        ("sample_xray_4.jpg", "pneumonia_model", 4),  # Critical priority
        ("sample_xray_5.jpg", "xray_classifier", 2),  # Medium priority
    ]
    
    for data, model, priority in test_requests:
        request_id = await engine.submit_inference_request(data, model, priority, timeout=30.0)
        request_ids.append(request_id)
        print(f"  Submitted request {request_id[:8]}... for {model} (priority: {priority})")
    
    # Wait for results
    print("\n4. Waiting for inference results...")
    
    for request_id in request_ids:
        try:
            result = await engine.get_inference_result(request_id, timeout=10.0)
            if result and result.status == InferenceStatus.COMPLETED:
                print(f"  ✓ Request {request_id[:8]}... completed successfully")
                print(f"    Prediction: {result.result['prediction']:.3f}")
                print(f"    Confidence: {result.result['confidence']:.3f}")
                print(f"    Worker: {result.result['worker_id']}")
            else:
                print(f"  ✗ Request {request_id[:8]}... failed or timed out")
                if result:
                    print(f"    Status: {result.status.value}")
                    if result.error_message:
                        print(f"    Error: {result.error_message}")
        
        except Exception as e:
            print(f"  ✗ Request {request_id[:8]}... error: {e}")
    
    # Show system status
    print("\n5. System Status:")
    status = engine.get_system_status()
    
    print(f"  Engine Status: {status['engine_status']}")
    print(f"  Total Requests: {status['metrics']['total_requests']}")
    print(f"  Success Rate: {status['metrics']['success_rate']:.1%}")
    print(f"  Average Latency: {status['metrics']['average_latency']:.3f}s")
    print(f"  Throughput: {status['metrics']['throughput_rps']:.2f} rps")
    print(f"  Active Workers: {status['worker_pool']['healthy_workers'] + status['worker_pool']['busy_workers']}")
    print(f"  Pool Utilization: {status['worker_pool']['utilization']:.1%}")
    print(f"  Queue Size: {status['metrics']['current_queue_size']}")
    
    # Test high-level inference method
    print("\n6. Testing direct inference method...")
    try:
        result = await engine.inference(
            data="test_image_data",
            model_name="pneumonia_model",
            priority=3,
            timeout=10.0
        )
        print(f"  Direct inference result: {result}")
    except Exception as e:
        print(f"  Direct inference failed: {e}")
    
    # Wait a bit more to see metrics
    print("\n7. Letting system run for a few more seconds...")
    await asyncio.sleep(3)
    
    # Final status
    final_status = engine.get_system_status()
    print(f"\nFinal Status:")
    print(f"  Total Processed: {final_status['metrics']['completed_requests']}")
    print(f"  Final Success Rate: {final_status['metrics']['success_rate']:.1%}")
    
    # Stop engine
    print("\n8. Stopping inference engine...")
    engine.stop()
    print("Demo completed.")


if __name__ == "__main__":
    asyncio.run(demonstrate_distributed_inference())