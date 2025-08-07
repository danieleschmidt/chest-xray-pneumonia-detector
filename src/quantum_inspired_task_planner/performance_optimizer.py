"""Performance optimization and auto-scaling for quantum task scheduler."""

import logging
import threading
import time
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import json
from enum import Enum

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    AGGRESSIVE = "aggressive"  # Maximum performance, higher resource usage
    BALANCED = "balanced"     # Balance performance and resources
    CONSERVATIVE = "conservative"  # Minimum resources, stable performance


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization decisions."""
    throughput: float = 0.0  # tasks per second
    latency: float = 0.0     # average task completion time
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    queue_length: int = 0
    error_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ScalingDecision:
    """Auto-scaling decision with rationale."""
    action: str  # 'scale_up', 'scale_down', 'maintain'
    target_capacity: int
    current_capacity: int
    confidence: float  # 0.0 to 1.0
    reasoning: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class PerformanceProfiler:
    """Advanced performance profiler for task scheduling operations."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.operation_times = defaultdict(lambda: deque(maxlen=window_size))
        self.operation_counts = defaultdict(int)
        self.bottlenecks = {}
        
    def profile_operation(self, operation_name: str):
        """Context manager for profiling operations."""
        return self.OperationTimer(self, operation_name)
    
    class OperationTimer:
        """Timer context manager for operation profiling."""
        
        def __init__(self, profiler, operation_name: str):
            self.profiler = profiler
            self.operation_name = operation_name
            self.start_time = None
        
        def __enter__(self):
            self.start_time = time.perf_counter()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = time.perf_counter() - self.start_time
            self.profiler.record_operation(self.operation_name, duration)
    
    def record_operation(self, operation_name: str, duration: float) -> None:
        """Record operation timing."""
        self.operation_times[operation_name].append(duration)
        self.operation_counts[operation_name] += 1
        
        # Detect bottlenecks
        if len(self.operation_times[operation_name]) >= 10:
            avg_time = sum(self.operation_times[operation_name]) / len(self.operation_times[operation_name])
            if avg_time > 0.1:  # Operations taking more than 100ms
                self.bottlenecks[operation_name] = avg_time
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            'operations': {},
            'bottlenecks': dict(self.bottlenecks),
            'total_operations': sum(self.operation_counts.values())
        }
        
        for op_name, times in self.operation_times.items():
            if times:
                summary['operations'][op_name] = {
                    'count': self.operation_counts[op_name],
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'recent_avg': sum(list(times)[-10:]) / min(len(times), 10)
                }
        
        return summary


class QuantumCacheManager:
    """High-performance caching with quantum-inspired coherence."""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict] = {}
        self.access_times: Dict[str, datetime] = {}
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.lock = threading.RLock()
        
        # Quantum coherence tracking
        self.coherence_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with coherence tracking."""
        with self.lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            entry = self.cache[key]
            if datetime.now() > entry['expires_at']:
                self._evict(key)
                return None
            
            # Update access patterns
            self.access_times[key] = datetime.now()
            self.access_counts[key] += 1
            
            # Update quantum coherence
            self._update_coherence(key)
            
            return entry['value']
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put value in cache with quantum coherence."""
        with self.lock:
            expires_at = datetime.now() + timedelta(seconds=ttl or self.ttl_seconds)
            
            self.cache[key] = {
                'value': value,
                'created_at': datetime.now(),
                'expires_at': expires_at,
                'size': self._estimate_size(value)
            }
            
            self.access_times[key] = datetime.now()
            self.access_counts[key] = 1
            
            # Maintain cache size
            if len(self.cache) > self.max_size:
                self._evict_lru()
    
    def _update_coherence(self, accessed_key: str) -> None:
        """Update quantum coherence matrix based on access patterns."""
        current_time = datetime.now()
        
        # Find recently accessed keys (within last 60 seconds)
        recent_keys = [
            k for k, access_time in self.access_times.items()
            if (current_time - access_time).total_seconds() < 60 and k != accessed_key
        ]
        
        # Increase coherence between recently accessed keys
        for related_key in recent_keys:
            current_coherence = self.coherence_matrix[accessed_key].get(related_key, 0.0)
            self.coherence_matrix[accessed_key][related_key] = min(1.0, current_coherence + 0.1)
            self.coherence_matrix[related_key][accessed_key] = min(1.0, current_coherence + 0.1)
    
    def get_coherent_keys(self, key: str, threshold: float = 0.5) -> List[str]:
        """Get keys with high coherence to given key."""
        return [
            related_key for related_key, coherence
            in self.coherence_matrix.get(key, {}).items()
            if coherence >= threshold
        ]
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._evict(lru_key)
    
    def _evict(self, key: str) -> None:
        """Evict specific key."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.access_counts.pop(key, None)
        self.coherence_matrix.pop(key, None)
        
        # Clean up coherence references
        for coherence_dict in self.coherence_matrix.values():
            coherence_dict.pop(key, None)
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of cached value."""
        try:
            return len(json.dumps(value, default=str))
        except:
            return 1000  # Default estimate
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self.lock:
            total_size = sum(entry.get('size', 0) for entry in self.cache.values())
            
            return {
                'total_entries': len(self.cache),
                'total_size_estimate': total_size,
                'hit_ratio': self._calculate_hit_ratio(),
                'avg_coherence': self._calculate_avg_coherence(),
                'most_accessed': sorted(
                    self.access_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            }
    
    def _calculate_hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        # Simplified calculation - would need request tracking in practice
        return 0.75  # Placeholder
    
    def _calculate_avg_coherence(self) -> float:
        """Calculate average coherence score."""
        if not self.coherence_matrix:
            return 0.0
        
        total_coherence = 0.0
        total_pairs = 0
        
        for key_coherences in self.coherence_matrix.values():
            total_coherence += sum(key_coherences.values())
            total_pairs += len(key_coherences)
        
        return total_coherence / max(1, total_pairs)


class AutoScaler:
    """Intelligent auto-scaling based on performance metrics."""
    
    def __init__(self, min_capacity: int = 1, max_capacity: int = 32):
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.current_capacity = min_capacity
        self.metrics_history: deque = deque(maxlen=100)
        self.scaling_history: List[ScalingDecision] = []
        
        # Scaling thresholds
        self.scale_up_thresholds = {
            'cpu_utilization': 70.0,
            'queue_length': 20,
            'throughput_ratio': 0.8  # Current/desired throughput
        }
        
        self.scale_down_thresholds = {
            'cpu_utilization': 30.0,
            'queue_length': 5,
            'throughput_ratio': 1.5
        }
        
        # Prevent rapid scaling changes
        self.cooldown_period = timedelta(minutes=5)
        self.last_scaling_action = datetime.now() - self.cooldown_period
    
    def add_metrics(self, metrics: PerformanceMetrics) -> None:
        """Add performance metrics for scaling decisions."""
        self.metrics_history.append(metrics)
    
    def make_scaling_decision(self, strategy: OptimizationStrategy = OptimizationStrategy.BALANCED) -> ScalingDecision:
        """Make intelligent scaling decision based on metrics and strategy."""
        if not self.metrics_history:
            return ScalingDecision(
                action='maintain',
                target_capacity=self.current_capacity,
                current_capacity=self.current_capacity,
                confidence=0.0,
                reasoning=["No metrics available"]
            )
        
        # Check cooldown period
        if datetime.now() - self.last_scaling_action < self.cooldown_period:
            return ScalingDecision(
                action='maintain',
                target_capacity=self.current_capacity,
                current_capacity=self.current_capacity,
                confidence=1.0,
                reasoning=["In cooldown period"]
            )
        
        # Analyze recent metrics
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 data points
        avg_metrics = self._calculate_average_metrics(recent_metrics)
        
        # Determine scaling action
        scale_up_score = self._calculate_scale_up_score(avg_metrics, strategy)
        scale_down_score = self._calculate_scale_down_score(avg_metrics, strategy)
        
        reasoning = []
        
        if scale_up_score > 0.7:
            action = 'scale_up'
            target_capacity = min(self.max_capacity, int(self.current_capacity * 1.5))
            confidence = scale_up_score
            reasoning.append(f"High resource utilization (score: {scale_up_score:.2f})")
            
        elif scale_down_score > 0.7:
            action = 'scale_down'
            target_capacity = max(self.min_capacity, int(self.current_capacity * 0.7))
            confidence = scale_down_score
            reasoning.append(f"Low resource utilization (score: {scale_down_score:.2f})")
            
        else:
            action = 'maintain'
            target_capacity = self.current_capacity
            confidence = 1.0 - max(scale_up_score, scale_down_score)
            reasoning.append("Metrics within stable range")
        
        # Add strategy-specific reasoning
        reasoning.extend(self._get_strategy_reasoning(avg_metrics, strategy))
        
        decision = ScalingDecision(
            action=action,
            target_capacity=target_capacity,
            current_capacity=self.current_capacity,
            confidence=confidence,
            reasoning=reasoning
        )
        
        self.scaling_history.append(decision)
        
        if action != 'maintain':
            self.last_scaling_action = datetime.now()
            self.current_capacity = target_capacity
        
        return decision
    
    def _calculate_average_metrics(self, metrics_list: List[PerformanceMetrics]) -> PerformanceMetrics:
        """Calculate average metrics from list."""
        if not metrics_list:
            return PerformanceMetrics()
        
        return PerformanceMetrics(
            throughput=sum(m.throughput for m in metrics_list) / len(metrics_list),
            latency=sum(m.latency for m in metrics_list) / len(metrics_list),
            cpu_utilization=sum(m.cpu_utilization for m in metrics_list) / len(metrics_list),
            memory_utilization=sum(m.memory_utilization for m in metrics_list) / len(metrics_list),
            queue_length=int(sum(m.queue_length for m in metrics_list) / len(metrics_list)),
            error_rate=sum(m.error_rate for m in metrics_list) / len(metrics_list)
        )
    
    def _calculate_scale_up_score(self, metrics: PerformanceMetrics, 
                                  strategy: OptimizationStrategy) -> float:
        """Calculate score for scaling up."""
        score = 0.0
        
        # CPU utilization factor
        if metrics.cpu_utilization > self.scale_up_thresholds['cpu_utilization']:
            score += 0.4 * (metrics.cpu_utilization / 100.0)
        
        # Queue length factor
        if metrics.queue_length > self.scale_up_thresholds['queue_length']:
            score += 0.3 * min(1.0, metrics.queue_length / 50.0)
        
        # Latency factor
        if metrics.latency > 5.0:  # 5 second threshold
            score += 0.2 * min(1.0, metrics.latency / 20.0)
        
        # Error rate factor
        if metrics.error_rate > 0.01:  # 1% threshold
            score += 0.1 * min(1.0, metrics.error_rate * 100)
        
        # Strategy adjustments
        if strategy == OptimizationStrategy.AGGRESSIVE:
            score *= 1.3
        elif strategy == OptimizationStrategy.CONSERVATIVE:
            score *= 0.7
        
        return min(1.0, score)
    
    def _calculate_scale_down_score(self, metrics: PerformanceMetrics,
                                    strategy: OptimizationStrategy) -> float:
        """Calculate score for scaling down."""
        score = 0.0
        
        # Low CPU utilization
        if metrics.cpu_utilization < self.scale_down_thresholds['cpu_utilization']:
            score += 0.4 * (1.0 - metrics.cpu_utilization / 100.0)
        
        # Low queue length
        if metrics.queue_length < self.scale_down_thresholds['queue_length']:
            score += 0.3
        
        # Good latency
        if metrics.latency < 2.0:
            score += 0.2
        
        # Low error rate
        if metrics.error_rate < 0.001:
            score += 0.1
        
        # Strategy adjustments
        if strategy == OptimizationStrategy.CONSERVATIVE:
            score *= 1.2
        elif strategy == OptimizationStrategy.AGGRESSIVE:
            score *= 0.8
        
        # Don't scale down if already at minimum
        if self.current_capacity <= self.min_capacity:
            score = 0.0
        
        return min(1.0, score)
    
    def _get_strategy_reasoning(self, metrics: PerformanceMetrics,
                               strategy: OptimizationStrategy) -> List[str]:
        """Get strategy-specific reasoning."""
        reasoning = []
        
        if strategy == OptimizationStrategy.AGGRESSIVE:
            reasoning.append("Aggressive strategy: prioritizing performance")
            if metrics.latency > 1.0:
                reasoning.append("Latency exceeds aggressive threshold")
        
        elif strategy == OptimizationStrategy.CONSERVATIVE:
            reasoning.append("Conservative strategy: minimizing resource usage")
            if metrics.cpu_utilization < 50.0:
                reasoning.append("CPU utilization allows resource reduction")
        
        else:  # BALANCED
            reasoning.append("Balanced strategy: optimizing performance/cost ratio")
        
        return reasoning


class PerformanceOptimizer:
    """Comprehensive performance optimizer for quantum task scheduler."""
    
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.profiler = PerformanceProfiler()
        self.cache = QuantumCacheManager()
        self.auto_scaler = AutoScaler()
        
        # Performance monitoring
        self.is_monitoring = False
        self.monitor_thread = None
        self.optimization_history = []
        
        # Thread pool for parallel task execution
        self.max_workers = 8
        self.thread_pool = None
        self.process_pool = None
        
    def start_optimization(self, strategy: OptimizationStrategy = OptimizationStrategy.BALANCED) -> None:
        """Start continuous performance optimization."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.optimization_strategy = strategy
        
        # Initialize thread pools
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"Performance optimization started with {strategy.value} strategy")
    
    def stop_optimization(self) -> None:
        """Stop performance optimization."""
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        logger.info("Performance optimization stopped")
    
    def _optimization_loop(self) -> None:
        """Main optimization monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect performance metrics
                metrics = self._collect_performance_metrics()
                
                # Make scaling decision
                scaling_decision = self.auto_scaler.make_scaling_decision(self.optimization_strategy)
                
                # Apply optimizations
                self._apply_optimizations(metrics, scaling_decision)
                
                # Clean up caches
                self._cleanup_caches()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                time.sleep(30)
    
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        # Get task statistics
        total_tasks = len(self.scheduler.tasks)
        completed_tasks = len([t for t in self.scheduler.tasks.values() 
                             if hasattr(t, 'status') and str(t.status).endswith('COMPLETED')])
        running_tasks = len(getattr(self.scheduler, 'running_tasks', []))
        queue_length = total_tasks - completed_tasks - running_tasks
        
        # Calculate throughput and latency
        throughput = self._calculate_throughput()
        latency = self._calculate_average_latency()
        
        # Estimate resource utilization (simplified)
        cpu_utilization = min(100.0, running_tasks * 10.0)  # Simplified estimate
        memory_utilization = min(100.0, total_tasks * 2.0)  # Simplified estimate
        
        # Error rate
        error_rate = self._calculate_error_rate()
        
        metrics = PerformanceMetrics(
            throughput=throughput,
            latency=latency,
            cpu_utilization=cpu_utilization,
            memory_utilization=memory_utilization,
            queue_length=queue_length,
            error_rate=error_rate
        )
        
        self.auto_scaler.add_metrics(metrics)
        return metrics
    
    def _calculate_throughput(self) -> float:
        """Calculate current task throughput."""
        # Simple throughput calculation based on completed tasks
        performance_summary = self.profiler.get_performance_summary()
        
        if 'complete_task' in performance_summary['operations']:
            op_stats = performance_summary['operations']['complete_task']
            if op_stats['avg_time'] > 0:
                return 1.0 / op_stats['avg_time']  # tasks per second
        
        return 0.1  # Default low throughput
    
    def _calculate_average_latency(self) -> float:
        """Calculate average task latency."""
        total_latency = 0.0
        latency_count = 0
        
        for task in self.scheduler.tasks.values():
            if (hasattr(task, 'started_at') and hasattr(task, 'completed_at') and
                task.started_at and task.completed_at):
                latency = (task.completed_at - task.started_at).total_seconds()
                total_latency += latency
                latency_count += 1
        
        return total_latency / max(1, latency_count)
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate."""
        # Simplified error rate calculation
        failed_tasks = len([t for t in self.scheduler.tasks.values() 
                          if hasattr(t, 'status') and str(t.status).endswith('BLOCKED')])
        total_tasks = len(self.scheduler.tasks)
        
        return failed_tasks / max(1, total_tasks)
    
    def _apply_optimizations(self, metrics: PerformanceMetrics, 
                           scaling_decision: ScalingDecision) -> None:
        """Apply performance optimizations based on metrics."""
        optimizations_applied = []
        
        # Apply scaling decision
        if scaling_decision.action == 'scale_up':
            new_capacity = min(self.max_workers * 2, scaling_decision.target_capacity)
            if new_capacity > self.max_workers:
                self._scale_thread_pool(new_capacity)
                optimizations_applied.append(f"Scaled up to {new_capacity} workers")
        
        elif scaling_decision.action == 'scale_down':
            new_capacity = max(2, scaling_decision.target_capacity)
            if new_capacity < self.max_workers:
                self._scale_thread_pool(new_capacity)
                optimizations_applied.append(f"Scaled down to {new_capacity} workers")
        
        # Cache optimization
        if metrics.cpu_utilization > 60:
            # Increase cache size for high CPU usage
            self.cache.max_size = min(20000, self.cache.max_size * 1.1)
            optimizations_applied.append("Increased cache size")
        
        # Task prioritization optimization
        if metrics.queue_length > 10:
            self._optimize_task_prioritization()
            optimizations_applied.append("Optimized task prioritization")
        
        # Record optimization history
        if optimizations_applied:
            self.optimization_history.append({
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'scaling_decision': scaling_decision,
                'optimizations': optimizations_applied
            })
    
    def _scale_thread_pool(self, new_capacity: int) -> None:
        """Dynamically scale thread pool capacity."""
        # Shutdown old pool
        if self.thread_pool:
            self.thread_pool.shutdown(wait=False)
        
        # Create new pool with updated capacity
        self.max_workers = new_capacity
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Update scheduler's parallel task limit
        self.scheduler.max_parallel_tasks = new_capacity
    
    def _optimize_task_prioritization(self) -> None:
        """Optimize task prioritization for better throughput."""
        # Get pending tasks
        pending_tasks = [t for t in self.scheduler.tasks.values() 
                        if hasattr(t, 'status') and str(t.status).endswith('PENDING')]
        
        if not pending_tasks:
            return
        
        # Re-calculate priority scores with performance optimization
        for task in pending_tasks:
            # Boost priority of tasks that are likely to complete quickly
            if hasattr(task, 'estimated_duration'):
                duration_seconds = task.estimated_duration.total_seconds()
                if duration_seconds < 300:  # Tasks under 5 minutes
                    task.superposition_weight *= 1.2
    
    def _cleanup_caches(self) -> None:
        """Clean up caches and expired data."""
        # Cache cleanup is handled internally by the cache manager
        pass
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        profiler_summary = self.profiler.get_performance_summary()
        cache_stats = self.cache.get_cache_stats()
        
        # Recent scaling decisions
        recent_scaling = self.auto_scaler.scaling_history[-10:] if self.auto_scaler.scaling_history else []
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_capacity': {
                'thread_pool_workers': self.max_workers,
                'max_parallel_tasks': self.scheduler.max_parallel_tasks
            },
            'profiler_summary': profiler_summary,
            'cache_statistics': cache_stats,
            'recent_scaling_decisions': [
                {
                    'action': decision.action,
                    'target_capacity': decision.target_capacity,
                    'confidence': decision.confidence,
                    'timestamp': decision.timestamp.isoformat(),
                    'reasoning': decision.reasoning
                }
                for decision in recent_scaling
            ],
            'optimization_history_count': len(self.optimization_history),
            'performance_recommendations': self._generate_performance_recommendations()
        }
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        # Analyze profiler data
        profiler_summary = self.profiler.get_performance_summary()
        
        # Check for bottlenecks
        if profiler_summary['bottlenecks']:
            bottleneck_ops = list(profiler_summary['bottlenecks'].keys())
            recommendations.append(f"Optimize bottleneck operations: {', '.join(bottleneck_ops)}")
        
        # Cache recommendations
        cache_stats = self.cache.get_cache_stats()
        if cache_stats['hit_ratio'] < 0.7:
            recommendations.append("Improve cache hit ratio by optimizing cache keys and TTL")
        
        # Scaling recommendations
        recent_decisions = self.auto_scaler.scaling_history[-5:]
        scale_up_count = len([d for d in recent_decisions if d.action == 'scale_up'])
        
        if scale_up_count >= 3:
            recommendations.append("Consider increasing base capacity due to frequent scale-up events")
        
        # Resource utilization recommendations
        if hasattr(self, 'optimization_strategy'):
            if self.optimization_strategy == OptimizationStrategy.CONSERVATIVE:
                recommendations.append("Consider switching to BALANCED strategy for better performance")
        
        return recommendations
    
    def execute_parallel_tasks(self, task_functions: List[Callable], 
                             max_workers: Optional[int] = None) -> List[Any]:
        """Execute multiple tasks in parallel with optimization."""
        if not task_functions:
            return []
        
        workers = max_workers or self.max_workers
        results = []
        
        # Use appropriate executor based on task type
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(func): i for i, func in enumerate(task_functions)}
            
            # Collect results
            for future in as_completed(future_to_task):
                task_index = future_to_task[future]
                try:
                    result = future.result()
                    results.append((task_index, result))
                except Exception as e:
                    logger.error(f"Parallel task {task_index} failed: {e}")
                    results.append((task_index, None))
        
        # Sort results by original task order
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]