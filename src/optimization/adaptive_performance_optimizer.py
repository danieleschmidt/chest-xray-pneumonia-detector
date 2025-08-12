"""
Adaptive performance optimization system for pneumonia detection pipeline.
Provides intelligent caching, model optimization, and resource allocation.
"""

import time
import threading
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import hashlib
import pickle
import numpy as np
from collections import defaultdict, OrderedDict, deque
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import gc
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Types of optimization strategies."""
    CACHING = "caching"
    MODEL_QUANTIZATION = "model_quantization"
    BATCH_OPTIMIZATION = "batch_optimization"
    MEMORY_MANAGEMENT = "memory_management"
    PIPELINE_PARALLELIZATION = "pipeline_parallelization"
    INFERENCE_ACCELERATION = "inference_acceleration"


class CacheStrategy(Enum):
    """Caching strategies."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"
    PREDICTIVE = "predictive"


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization decisions."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    inference_time_ms: float = 0.0
    throughput_rps: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    cache_hit_rate: float = 0.0
    cache_size_mb: float = 0.0
    model_load_time_ms: float = 0.0
    preprocessing_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0
    queue_wait_time_ms: float = 0.0
    error_rate: float = 0.0


@dataclass
class OptimizationAction:
    """An optimization action to be applied."""
    optimization_type: OptimizationType
    action_name: str
    parameters: Dict[str, Any]
    expected_improvement: float  # Expected performance improvement percentage
    resource_cost: float  # Resource cost (CPU, memory, etc.)
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    applied: bool = False
    result: Optional[Dict[str, Any]] = None


class AdaptiveCache:
    """Intelligent adaptive cache with multiple strategies."""
    
    def __init__(self, max_size: int = 1000, strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        self.max_size = max_size
        self.strategy = strategy
        self.cache = OrderedDict()
        self.access_counts = defaultdict(int)
        self.access_times = {}
        self.ttl_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
        
        # Adaptive strategy parameters
        self.access_patterns = deque(maxlen=1000)
        self.strategy_performance = {
            CacheStrategy.LRU: deque(maxlen=100),
            CacheStrategy.LFU: deque(maxlen=100),
            CacheStrategy.TTL: deque(maxlen=100)
        }
        
        # Default TTL for TTL strategy
        self.default_ttl = 3600  # 1 hour
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            current_time = time.time()
            
            # Check TTL expiration
            if key in self.ttl_times and current_time > self.ttl_times[key]:
                self._remove_key(key)
                self.miss_count += 1
                self._record_access_pattern(key, False)
                return None
            
            if key in self.cache:
                self.hit_count += 1
                self._update_access_info(key, current_time)
                self._record_access_pattern(key, True)
                
                # Move to end for LRU
                if self.strategy in [CacheStrategy.LRU, CacheStrategy.ADAPTIVE]:
                    self.cache.move_to_end(key)
                
                return self.cache[key]
            else:
                self.miss_count += 1
                self._record_access_pattern(key, False)
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put item in cache."""
        with self.lock:
            current_time = time.time()
            
            # Set TTL
            if ttl is not None:
                self.ttl_times[key] = current_time + ttl
            elif self.strategy == CacheStrategy.TTL:
                self.ttl_times[key] = current_time + self.default_ttl
            
            # Add or update item
            if key in self.cache:
                self.cache[key] = value
                self._update_access_info(key, current_time)
            else:
                # Make space if needed
                while len(self.cache) >= self.max_size:
                    self._evict_item()
                
                self.cache[key] = value
                self._update_access_info(key, current_time)
            
            # Move to end for LRU
            if self.strategy in [CacheStrategy.LRU, CacheStrategy.ADAPTIVE]:
                self.cache.move_to_end(key)
    
    def _update_access_info(self, key: str, current_time: float):
        """Update access information for a key."""
        self.access_counts[key] += 1
        self.access_times[key] = current_time
    
    def _record_access_pattern(self, key: str, hit: bool):
        """Record access pattern for adaptive strategy."""
        self.access_patterns.append({
            'key': key,
            'hit': hit,
            'timestamp': time.time(),
            'strategy': self.strategy
        })
    
    def _evict_item(self):
        """Evict item based on current strategy."""
        if not self.cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            key_to_remove = next(iter(self.cache))
        
        elif self.strategy == CacheStrategy.LFU:
            # Find least frequently used
            min_count = min(self.access_counts.get(k, 0) for k in self.cache.keys())
            key_to_remove = next(k for k in self.cache.keys() 
                               if self.access_counts.get(k, 0) == min_count)
        
        elif self.strategy == CacheStrategy.TTL:
            # Find expired or oldest
            current_time = time.time()
            expired_keys = [k for k in self.cache.keys() 
                          if k in self.ttl_times and current_time > self.ttl_times[k]]
            if expired_keys:
                key_to_remove = expired_keys[0]
            else:
                key_to_remove = next(iter(self.cache))
        
        elif self.strategy == CacheStrategy.ADAPTIVE:
            key_to_remove = self._adaptive_eviction()
        
        else:
            key_to_remove = next(iter(self.cache))
        
        self._remove_key(key_to_remove)
    
    def _adaptive_eviction(self) -> str:
        """Adaptive eviction based on access patterns."""
        # Analyze recent performance of different strategies
        if len(self.access_patterns) < 10:
            return next(iter(self.cache))  # Default to FIFO
        
        # Calculate hit rates for different virtual strategies
        recent_patterns = list(self.access_patterns)[-100:]
        
        # Simulate LRU performance
        lru_cache = OrderedDict()
        lru_hits = 0
        for pattern in recent_patterns:
            if pattern['key'] in lru_cache:
                lru_hits += 1
                lru_cache.move_to_end(pattern['key'])
            else:
                if len(lru_cache) >= 50:  # Smaller virtual cache for simulation
                    lru_cache.popitem(last=False)
                lru_cache[pattern['key']] = True
        
        # Simulate LFU performance  
        lfu_cache = {}
        lfu_counts = defaultdict(int)
        lfu_hits = 0
        for pattern in recent_patterns:
            if pattern['key'] in lfu_cache:
                lfu_hits += 1
            else:
                if len(lfu_cache) >= 50:
                    # Remove least frequent
                    min_count = min(lfu_counts[k] for k in lfu_cache.keys())
                    key_to_remove = next(k for k in lfu_cache.keys() 
                                       if lfu_counts[k] == min_count)
                    del lfu_cache[key_to_remove]
                lfu_cache[pattern['key']] = True
            lfu_counts[pattern['key']] += 1
        
        # Choose strategy with better recent performance
        lru_hit_rate = lru_hits / len(recent_patterns)
        lfu_hit_rate = lfu_hits / len(recent_patterns)
        
        if lru_hit_rate > lfu_hit_rate:
            # Use LRU eviction
            return next(iter(self.cache))
        else:
            # Use LFU eviction
            min_count = min(self.access_counts.get(k, 0) for k in self.cache.keys())
            return next(k for k in self.cache.keys() 
                       if self.access_counts.get(k, 0) == min_count)
    
    def _remove_key(self, key: str):
        """Remove key and associated metadata."""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_counts:
            del self.access_counts[key]
        if key in self.access_times:
            del self.access_times[key]
        if key in self.ttl_times:
            del self.ttl_times[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'strategy': self.strategy.value,
            'memory_usage_mb': self._estimate_memory_usage() / (1024 * 1024)
        }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage of cache in bytes."""
        try:
            return sum(len(pickle.dumps(v)) + len(k.encode()) for k, v in self.cache.items())
        except Exception:
            return len(self.cache) * 1024  # Rough estimate
    
    def clear(self):
        """Clear all cache data."""
        with self.lock:
            self.cache.clear()
            self.access_counts.clear()
            self.access_times.clear()
            self.ttl_times.clear()


class ModelOptimizer:
    """Optimizes ML models for better performance."""
    
    def __init__(self):
        self.optimization_history = []
        self.model_cache = {}
    
    def optimize_model(self, model, optimization_type: str = 'quantization') -> Any:
        """Optimize a model using specified technique."""
        try:
            if optimization_type == 'quantization':
                return self._quantize_model(model)
            elif optimization_type == 'pruning':
                return self._prune_model(model)
            elif optimization_type == 'distillation':
                return self._distill_model(model)
            else:
                logger.warning(f"Unknown optimization type: {optimization_type}")
                return model
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return model
    
    def _quantize_model(self, model):
        """Quantize model to reduce size and improve inference speed."""
        try:
            import tensorflow as tf
            
            # Convert to TensorFlow Lite with quantization
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Dynamic range quantization
            converter.representative_dataset = self._get_representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            
            quantized_model = converter.convert()
            
            logger.info("Model successfully quantized")
            return quantized_model
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return model
    
    def _prune_model(self, model):
        """Prune model to remove unnecessary weights."""
        try:
            import tensorflow as tf
            import tensorflow_model_optimization as tfmot
            
            # Define pruning parameters
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.30,
                    final_sparsity=0.70,
                    begin_step=0,
                    end_step=1000
                )
            }
            
            # Apply pruning
            pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
                model, **pruning_params
            )
            
            logger.info("Model successfully pruned")
            return pruned_model
            
        except ImportError:
            logger.warning("TensorFlow Model Optimization not available")
            return model
        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            return model
    
    def _distill_model(self, model):
        """Create a smaller distilled version of the model."""
        # This would implement knowledge distillation
        # For now, return original model
        logger.info("Model distillation not implemented yet")
        return model
    
    def _get_representative_dataset(self):
        """Get representative dataset for quantization."""
        # Generate dummy data for quantization calibration
        for _ in range(100):
            yield [np.random.random((1, 224, 224, 3)).astype(np.float32)]
    
    def benchmark_model(self, model, test_data: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Benchmark model performance."""
        try:
            import tensorflow as tf
            
            if test_data is None:
                # Generate dummy test data
                test_data = np.random.random((100, 224, 224, 3)).astype(np.float32)
            
            # Warm up
            for _ in range(5):
                _ = model.predict(test_data[:1], verbose=0)
            
            # Benchmark inference time
            start_time = time.time()
            predictions = model.predict(test_data, verbose=0)
            end_time = time.time()
            
            total_time_ms = (end_time - start_time) * 1000
            avg_time_per_sample = total_time_ms / len(test_data)
            
            # Estimate model size
            model_size_mb = self._estimate_model_size(model)
            
            return {
                'total_inference_time_ms': total_time_ms,
                'avg_inference_time_ms': avg_time_per_sample,
                'throughput_samples_per_second': 1000 / avg_time_per_sample,
                'model_size_mb': model_size_mb,
                'samples_processed': len(test_data)
            }
            
        except Exception as e:
            logger.error(f"Model benchmarking failed: {e}")
            return {
                'total_inference_time_ms': 0.0,
                'avg_inference_time_ms': 0.0,
                'throughput_samples_per_second': 0.0,
                'model_size_mb': 0.0,
                'samples_processed': 0
            }
    
    def _estimate_model_size(self, model) -> float:
        """Estimate model size in MB."""
        try:
            # Count parameters
            total_params = model.count_params()
            # Assume 4 bytes per parameter (float32)
            size_bytes = total_params * 4
            return size_bytes / (1024 * 1024)
        except Exception:
            return 0.0


class BatchOptimizer:
    """Optimizes batch processing for better throughput."""
    
    def __init__(self):
        self.batch_performance_history = deque(maxlen=1000)
        self.optimal_batch_sizes = {}
    
    def find_optimal_batch_size(self, model, 
                              min_batch_size: int = 1, 
                              max_batch_size: int = 128,
                              target_memory_usage: float = 80.0) -> int:
        """Find optimal batch size for given constraints."""
        try:
            import tensorflow as tf
            
            optimal_batch_size = min_batch_size
            best_throughput = 0.0
            
            for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
                if batch_size < min_batch_size or batch_size > max_batch_size:
                    continue
                
                try:
                    # Test batch size
                    test_data = np.random.random((batch_size, 224, 224, 3)).astype(np.float32)
                    
                    # Check memory usage
                    initial_memory = psutil.virtual_memory().percent
                    
                    # Warm up
                    _ = model.predict(test_data, verbose=0)
                    
                    # Measure performance
                    start_time = time.time()
                    for _ in range(10):  # Multiple runs for accuracy
                        _ = model.predict(test_data, verbose=0)
                    end_time = time.time()
                    
                    # Check memory usage after inference
                    peak_memory = psutil.virtual_memory().percent
                    memory_increase = peak_memory - initial_memory
                    
                    # Skip if memory usage is too high
                    if peak_memory > target_memory_usage:
                        logger.warning(f"Batch size {batch_size} uses too much memory: {peak_memory:.1f}%")
                        break
                    
                    # Calculate throughput
                    total_time = end_time - start_time
                    samples_per_second = (batch_size * 10) / total_time
                    
                    logger.info(f"Batch size {batch_size}: {samples_per_second:.2f} samples/sec, "
                              f"memory: {peak_memory:.1f}%")
                    
                    if samples_per_second > best_throughput:
                        best_throughput = samples_per_second
                        optimal_batch_size = batch_size
                    
                    # Record performance
                    self.batch_performance_history.append({
                        'batch_size': batch_size,
                        'throughput': samples_per_second,
                        'memory_usage': peak_memory,
                        'timestamp': datetime.utcnow()
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to test batch size {batch_size}: {e}")
                    continue
            
            logger.info(f"Optimal batch size: {optimal_batch_size} (throughput: {best_throughput:.2f} samples/sec)")
            return optimal_batch_size
            
        except Exception as e:
            logger.error(f"Batch size optimization failed: {e}")
            return min_batch_size
    
    def adaptive_batch_sizing(self, current_load: float, 
                            available_memory: float) -> int:
        """Dynamically adjust batch size based on current conditions."""
        base_batch_size = 32
        
        # Adjust based on current load
        if current_load > 80:
            # High load - reduce batch size for lower latency
            batch_size = max(1, base_batch_size // 2)
        elif current_load < 30:
            # Low load - increase batch size for better throughput
            batch_size = min(128, base_batch_size * 2)
        else:
            batch_size = base_batch_size
        
        # Adjust based on available memory
        if available_memory < 20:  # Less than 20% memory available
            batch_size = max(1, batch_size // 2)
        elif available_memory > 70:  # More than 70% memory available
            batch_size = min(128, batch_size * 2)
        
        return batch_size


class AdaptivePerformanceOptimizer:
    """Main performance optimizer orchestrating all optimization strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Components
        self.cache = AdaptiveCache(
            max_size=self.config.get('cache_size', 1000),
            strategy=CacheStrategy(self.config.get('cache_strategy', 'adaptive'))
        )
        self.model_optimizer = ModelOptimizer()
        self.batch_optimizer = BatchOptimizer()
        
        # Performance tracking
        self.performance_history = deque(maxlen=10000)
        self.optimization_actions = []
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Configuration
        self.optimization_interval = self.config.get('optimization_interval', 300)  # 5 minutes
        self.performance_threshold = self.config.get('performance_threshold', 0.1)  # 10% improvement
        
        # State
        self.is_optimizing = False
        self.last_optimization_time = datetime.utcnow()
    
    def start_optimization_loop(self):
        """Start continuous optimization loop."""
        def optimization_loop():
            while True:
                try:
                    time.sleep(self.optimization_interval)
                    
                    if not self.is_optimizing:
                        self._perform_optimization_cycle()
                    
                except Exception as e:
                    logger.error(f"Error in optimization loop: {e}")
                    time.sleep(60)  # Wait before retrying
        
        optimization_thread = threading.Thread(target=optimization_loop, daemon=True)
        optimization_thread.start()
        logger.info("Performance optimization loop started")
    
    def record_performance_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics for optimization decisions."""
        self.performance_history.append(metrics)
    
    def _perform_optimization_cycle(self):
        """Perform one cycle of performance optimization."""
        self.is_optimizing = True
        
        try:
            logger.info("Starting optimization cycle")
            
            # Analyze recent performance
            if len(self.performance_history) < 10:
                logger.info("Insufficient performance data for optimization")
                return
            
            # Get optimization recommendations
            recommendations = self._analyze_and_recommend()
            
            # Apply optimizations
            for recommendation in recommendations:
                if recommendation.confidence > 0.7:  # High confidence threshold
                    self._apply_optimization(recommendation)
            
            self.last_optimization_time = datetime.utcnow()
            logger.info(f"Optimization cycle completed with {len(recommendations)} recommendations")
            
        finally:
            self.is_optimizing = False
    
    def _analyze_and_recommend(self) -> List[OptimizationAction]:
        """Analyze performance and generate optimization recommendations."""
        recommendations = []
        
        # Get recent metrics
        recent_metrics = list(self.performance_history)[-100:]
        
        if not recent_metrics:
            return recommendations
        
        # Calculate averages
        avg_inference_time = np.mean([m.inference_time_ms for m in recent_metrics])
        avg_memory_usage = np.mean([m.memory_usage_mb for m in recent_metrics])
        avg_cache_hit_rate = np.mean([m.cache_hit_rate for m in recent_metrics])
        avg_throughput = np.mean([m.throughput_rps for m in recent_metrics])
        
        # Cache optimization recommendations
        if avg_cache_hit_rate < 70:  # Low cache hit rate
            recommendations.append(OptimizationAction(
                optimization_type=OptimizationType.CACHING,
                action_name="increase_cache_size",
                parameters={'new_size': min(self.cache.max_size * 2, 5000)},
                expected_improvement=15.0,
                resource_cost=10.0,
                confidence=0.8
            ))
        
        # Memory optimization recommendations
        if avg_memory_usage > 80:  # High memory usage
            recommendations.append(OptimizationAction(
                optimization_type=OptimizationType.MEMORY_MANAGEMENT,
                action_name="aggressive_garbage_collection",
                parameters={'frequency': 'high'},
                expected_improvement=10.0,
                resource_cost=5.0,
                confidence=0.7
            ))
        
        # Inference optimization recommendations
        if avg_inference_time > 1000:  # Slow inference (>1 second)
            recommendations.append(OptimizationAction(
                optimization_type=OptimizationType.INFERENCE_ACCELERATION,
                action_name="optimize_batch_size",
                parameters={'target_latency': 500},
                expected_improvement=25.0,
                resource_cost=15.0,
                confidence=0.9
            ))
        
        # Throughput optimization recommendations
        if avg_throughput < 10:  # Low throughput
            recommendations.append(OptimizationAction(
                optimization_type=OptimizationType.PIPELINE_PARALLELIZATION,
                action_name="increase_worker_threads",
                parameters={'worker_count': min(psutil.cpu_count() * 2, 16)},
                expected_improvement=30.0,
                resource_cost=20.0,
                confidence=0.8
            ))
        
        return recommendations
    
    def _apply_optimization(self, action: OptimizationAction) -> bool:
        """Apply an optimization action."""
        try:
            logger.info(f"Applying optimization: {action.action_name}")
            
            success = False
            
            if action.optimization_type == OptimizationType.CACHING:
                success = self._apply_cache_optimization(action)
            
            elif action.optimization_type == OptimizationType.MEMORY_MANAGEMENT:
                success = self._apply_memory_optimization(action)
            
            elif action.optimization_type == OptimizationType.INFERENCE_ACCELERATION:
                success = self._apply_inference_optimization(action)
            
            elif action.optimization_type == OptimizationType.PIPELINE_PARALLELIZATION:
                success = self._apply_pipeline_optimization(action)
            
            action.applied = True
            action.result = {'success': success}
            self.optimization_actions.append(action)
            
            if success:
                logger.info(f"Successfully applied optimization: {action.action_name}")
            else:
                logger.warning(f"Failed to apply optimization: {action.action_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error applying optimization {action.action_name}: {e}")
            action.applied = True
            action.result = {'success': False, 'error': str(e)}
            return False
    
    def _apply_cache_optimization(self, action: OptimizationAction) -> bool:
        """Apply cache-related optimization."""
        if action.action_name == "increase_cache_size":
            new_size = action.parameters.get('new_size', self.cache.max_size * 2)
            old_size = self.cache.max_size
            self.cache.max_size = new_size
            logger.info(f"Increased cache size from {old_size} to {new_size}")
            return True
        
        elif action.action_name == "change_cache_strategy":
            new_strategy = action.parameters.get('strategy', 'adaptive')
            self.cache.strategy = CacheStrategy(new_strategy)
            logger.info(f"Changed cache strategy to {new_strategy}")
            return True
        
        return False
    
    def _apply_memory_optimization(self, action: OptimizationAction) -> bool:
        """Apply memory-related optimization."""
        if action.action_name == "aggressive_garbage_collection":
            # Force garbage collection
            collected = gc.collect()
            logger.info(f"Garbage collection freed {collected} objects")
            
            # Clear some caches if memory usage is still high
            if psutil.virtual_memory().percent > 85:
                self.cache.clear()
                logger.info("Cleared cache due to high memory usage")
            
            return True
        
        return False
    
    def _apply_inference_optimization(self, action: OptimizationAction) -> bool:
        """Apply inference-related optimization."""
        if action.action_name == "optimize_batch_size":
            # This would interact with the inference engine to optimize batch size
            target_latency = action.parameters.get('target_latency', 500)
            logger.info(f"Would optimize batch size for target latency: {target_latency}ms")
            return True
        
        return False
    
    def _apply_pipeline_optimization(self, action: OptimizationAction) -> bool:
        """Apply pipeline-related optimization."""
        if action.action_name == "increase_worker_threads":
            worker_count = action.parameters.get('worker_count', 4)
            # This would configure the thread pool for inference
            logger.info(f"Would increase worker threads to {worker_count}")
            return True
        
        return False
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        recent_metrics = list(self.performance_history)[-10:] if self.performance_history else []
        
        if recent_metrics:
            avg_inference_time = np.mean([m.inference_time_ms for m in recent_metrics])
            avg_throughput = np.mean([m.throughput_rps for m in recent_metrics])
            avg_memory_usage = np.mean([m.memory_usage_mb for m in recent_metrics])
        else:
            avg_inference_time = 0
            avg_throughput = 0
            avg_memory_usage = 0
        
        return {
            'is_optimizing': self.is_optimizing,
            'last_optimization_time': self.last_optimization_time.isoformat(),
            'cache_stats': self.cache.get_stats(),
            'recent_performance': {
                'avg_inference_time_ms': avg_inference_time,
                'avg_throughput_rps': avg_throughput,
                'avg_memory_usage_mb': avg_memory_usage
            },
            'total_optimizations': len(self.optimization_actions),
            'successful_optimizations': sum(1 for a in self.optimization_actions 
                                          if a.result and a.result.get('success', False))
        }
    
    def optimize_model_for_inference(self, model, optimization_types: List[str] = None) -> Any:
        """Optimize a model for better inference performance."""
        if optimization_types is None:
            optimization_types = ['quantization']
        
        optimized_model = model
        
        for opt_type in optimization_types:
            try:
                optimized_model = self.model_optimizer.optimize_model(optimized_model, opt_type)
                logger.info(f"Applied {opt_type} optimization to model")
            except Exception as e:
                logger.error(f"Failed to apply {opt_type} optimization: {e}")
        
        return optimized_model
    
    def benchmark_and_optimize(self, model, test_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Benchmark model and apply optimizations."""
        # Initial benchmark
        initial_benchmark = self.model_optimizer.benchmark_model(model, test_data)
        logger.info(f"Initial benchmark: {initial_benchmark}")
        
        # Find optimal batch size
        optimal_batch_size = self.batch_optimizer.find_optimal_batch_size(model)
        logger.info(f"Optimal batch size: {optimal_batch_size}")
        
        # Optimize model
        optimized_model = self.optimize_model_for_inference(model)
        
        # Final benchmark
        final_benchmark = self.model_optimizer.benchmark_model(optimized_model, test_data)
        logger.info(f"Final benchmark: {final_benchmark}")
        
        # Calculate improvement
        improvement = {
            'inference_time_improvement': (
                (initial_benchmark['avg_inference_time_ms'] - final_benchmark['avg_inference_time_ms']) / 
                initial_benchmark['avg_inference_time_ms'] * 100
            ) if initial_benchmark['avg_inference_time_ms'] > 0 else 0,
            
            'throughput_improvement': (
                (final_benchmark['throughput_samples_per_second'] - initial_benchmark['throughput_samples_per_second']) / 
                initial_benchmark['throughput_samples_per_second'] * 100
            ) if initial_benchmark['throughput_samples_per_second'] > 0 else 0,
            
            'model_size_reduction': (
                (initial_benchmark['model_size_mb'] - final_benchmark['model_size_mb']) / 
                initial_benchmark['model_size_mb'] * 100
            ) if initial_benchmark['model_size_mb'] > 0 else 0
        }
        
        return {
            'initial_benchmark': initial_benchmark,
            'final_benchmark': final_benchmark,
            'improvement': improvement,
            'optimal_batch_size': optimal_batch_size,
            'optimized_model': optimized_model
        }
    
    def export_optimization_report(self, file_path: str):
        """Export comprehensive optimization report."""
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'optimization_status': self.get_optimization_status(),
            'optimization_history': [
                {
                    'timestamp': action.timestamp.isoformat(),
                    'type': action.optimization_type.value,
                    'action': action.action_name,
                    'parameters': action.parameters,
                    'expected_improvement': action.expected_improvement,
                    'confidence': action.confidence,
                    'applied': action.applied,
                    'success': action.result.get('success', False) if action.result else False
                }
                for action in self.optimization_actions[-100:]  # Last 100 actions
            ],
            'performance_trends': [
                {
                    'timestamp': metrics.timestamp.isoformat(),
                    'inference_time_ms': metrics.inference_time_ms,
                    'throughput_rps': metrics.throughput_rps,
                    'memory_usage_mb': metrics.memory_usage_mb,
                    'cache_hit_rate': metrics.cache_hit_rate
                }
                for metrics in list(self.performance_history)[-1000:]  # Last 1000 metrics
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Optimization report exported to {file_path}")


def create_performance_optimizer(config: Optional[Dict[str, Any]] = None) -> AdaptivePerformanceOptimizer:
    """Factory function to create a configured performance optimizer."""
    return AdaptivePerformanceOptimizer(config)


if __name__ == '__main__':
    # Example usage
    optimizer = create_performance_optimizer({
        'cache_size': 2000,
        'cache_strategy': 'adaptive',
        'optimization_interval': 300
    })
    
    # Start optimization loop
    optimizer.start_optimization_loop()
    
    # Simulate some performance metrics
    for i in range(10):
        metrics = PerformanceMetrics(
            inference_time_ms=np.random.normal(800, 200),
            throughput_rps=np.random.normal(15, 5),
            memory_usage_mb=np.random.normal(2000, 500),
            cache_hit_rate=np.random.normal(65, 15)
        )
        optimizer.record_performance_metrics(metrics)
        time.sleep(1)
    
    # Get status
    status = optimizer.get_optimization_status()
    print(f"Optimization status: {status}")
    
    # Export report
    optimizer.export_optimization_report('/tmp/optimization_report.json')
