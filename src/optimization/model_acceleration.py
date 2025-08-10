# Advanced Model Optimization and Acceleration
# GPU optimization, model pruning, quantization, and distributed inference

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import time
import psutil
import GPUtil
from pathlib import Path
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import lru_cache
import pickle
import json
from datetime import datetime
import threading
import queue


@dataclass
class OptimizationConfig:
    """Configuration for model optimization strategies."""
    enable_mixed_precision: bool = True
    enable_xla: bool = True
    enable_gpu_memory_growth: bool = True
    enable_model_pruning: bool = False
    pruning_sparsity: float = 0.5
    enable_quantization: bool = False
    quantization_type: str = 'dynamic'  # dynamic, static, int8
    batch_size_optimization: bool = True
    enable_model_parallelism: bool = False
    enable_pipeline_parallelism: bool = False
    cache_predictions: bool = True
    max_cache_size: int = 10000


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    inference_time: float
    throughput: float  # samples per second
    memory_usage: float  # MB
    gpu_utilization: float  # percentage
    cpu_utilization: float  # percentage
    batch_size: int
    model_size: float  # MB
    accuracy: Optional[float] = None


class GPUOptimizer:
    """Optimizes GPU usage for maximum performance."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._configure_gpu()
    
    def _configure_gpu(self):
        """Configure GPU settings for optimal performance."""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    # Enable memory growth to avoid allocation issues
                    tf.config.experimental.set_memory_growth(gpu, True)
                    
                # Enable mixed precision for faster training/inference
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                
                self.logger.info(f"Configured {len(gpus)} GPU(s) with mixed precision")
            else:
                self.logger.warning("No GPUs available, using CPU")
                
        except Exception as e:
            self.logger.error(f"GPU configuration failed: {e}")
    
    def optimize_model_for_gpu(self, model: tf.keras.Model) -> tf.keras.Model:
        """Optimize model architecture for GPU acceleration."""
        
        # Enable XLA compilation for faster execution
        model.compile(
            optimizer=model.optimizer,
            loss=model.loss,
            metrics=model.metrics,
            jit_compile=True  # Enable XLA
        )
        
        return model
    
    def get_optimal_batch_size(self, model: tf.keras.Model, 
                             input_shape: Tuple[int, ...],
                             max_memory_usage: float = 0.8) -> int:
        """Automatically determine optimal batch size."""
        
        if not tf.config.list_physical_devices('GPU'):
            return 32  # Conservative default for CPU
        
        # Start with a reasonable batch size and increase until memory limit
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        optimal_batch_size = 1
        
        for batch_size in batch_sizes:
            try:
                # Test memory usage with dummy data
                dummy_input = tf.random.normal((batch_size,) + input_shape)
                
                # Clear any existing computation graphs
                tf.keras.backend.clear_session()
                
                # Test forward pass
                with tf.device('/GPU:0'):
                    _ = model(dummy_input, training=False)
                
                # Check GPU memory usage
                gpu_info = GPUtil.getGPUs()[0]
                memory_usage_ratio = gpu_info.memoryUsed / gpu_info.memoryTotal
                
                if memory_usage_ratio < max_memory_usage:
                    optimal_batch_size = batch_size
                else:
                    break
                    
            except tf.errors.ResourceExhaustedError:
                break
            except Exception as e:
                self.logger.warning(f"Error testing batch size {batch_size}: {e}")
                break
        
        return optimal_batch_size


class ModelPruner:
    """Implements structured and unstructured model pruning."""
    
    def __init__(self, target_sparsity: float = 0.5):
        self.target_sparsity = target_sparsity
        self.logger = logging.getLogger(__name__)
    
    def prune_model(self, model: tf.keras.Model) -> tf.keras.Model:
        """Apply magnitude-based pruning to reduce model size."""
        
        try:
            import tensorflow_model_optimization as tfmot
            
            # Define pruning schedule
            end_step = 1000  # Adjust based on your training steps
            
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.0,
                    final_sparsity=self.target_sparsity,
                    begin_step=100,
                    end_step=end_step
                )
            }
            
            # Apply pruning to convolutional and dense layers
            def apply_pruning_to_dense(layer):
                if isinstance(layer, tf.keras.layers.Dense):
                    return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
                return layer
            
            def apply_pruning_to_conv(layer):
                if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                    return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
                return layer
            
            # Clone model and apply pruning
            pruned_model = tf.keras.models.clone_model(
                model,
                clone_function=apply_pruning_to_dense
            )
            
            pruned_model = tf.keras.models.clone_model(
                pruned_model,
                clone_function=apply_pruning_to_conv
            )
            
            self.logger.info(f"Applied pruning with target sparsity: {self.target_sparsity}")
            return pruned_model
            
        except ImportError:
            self.logger.error("TensorFlow Model Optimization not available")
            return model
        except Exception as e:
            self.logger.error(f"Pruning failed: {e}")
            return model
    
    def remove_pruning_wrappers(self, pruned_model: tf.keras.Model) -> tf.keras.Model:
        """Remove pruning wrappers after training to get final sparse model."""
        
        try:
            import tensorflow_model_optimization as tfmot
            return tfmot.sparsity.keras.strip_pruning(pruned_model)
        except ImportError:
            return pruned_model


class ModelQuantizer:
    """Implements various quantization strategies for model compression."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def dynamic_quantize(self, model: tf.keras.Model) -> tf.lite.TFLiteConverter:
        """Apply dynamic quantization for reduced inference time."""
        
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Enable dynamic range quantization
            converter.target_spec.supported_types = [tf.float16]
            
            quantized_model = converter.convert()
            
            self.logger.info("Applied dynamic quantization")
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"Dynamic quantization failed: {e}")
            return None
    
    def int8_quantize(self, model: tf.keras.Model, 
                     representative_dataset: Callable) -> tf.lite.TFLiteConverter:
        """Apply INT8 quantization with representative dataset."""
        
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            
            # Ensure INT8 quantization
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            quantized_model = converter.convert()
            
            self.logger.info("Applied INT8 quantization")
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"INT8 quantization failed: {e}")
            return None


class PredictionCache:
    """Intelligent caching system for model predictions."""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.cache_stats = {'hits': 0, 'misses': 0, 'evictions': 0}
        self.lock = threading.Lock()
    
    def _hash_input(self, input_data: np.ndarray) -> str:
        """Generate hash for input data."""
        return hashlib.sha256(input_data.tobytes()).hexdigest()[:16]
    
    def get(self, input_data: np.ndarray) -> Optional[np.ndarray]:
        """Retrieve cached prediction if available."""
        
        with self.lock:
            key = self._hash_input(input_data)
            current_time = time.time()
            
            if key in self.cache:
                entry_time = self.access_times[key]
                if current_time - entry_time < self.ttl_seconds:
                    self.cache_stats['hits'] += 1
                    self.access_times[key] = current_time
                    return self.cache[key]
                else:
                    # Expired entry
                    del self.cache[key]
                    del self.access_times[key]
            
            self.cache_stats['misses'] += 1
            return None
    
    def put(self, input_data: np.ndarray, prediction: np.ndarray):
        """Store prediction in cache."""
        
        with self.lock:
            key = self._hash_input(input_data)
            current_time = time.time()
            
            # Evict if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = prediction.copy()
            self.access_times[key] = current_time
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self.access_times:
            return
            
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]
        self.cache_stats['evictions'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            **self.cache_stats
        }


class DistributedInferenceEngine:
    """Distributed inference engine for high-throughput predictions."""
    
    def __init__(self, model: tf.keras.Model, num_workers: int = None):
        self.model = model
        self.num_workers = num_workers or mp.cpu_count()
        self.prediction_cache = PredictionCache()
        self.gpu_optimizer = GPUOptimizer()
        self.logger = logging.getLogger(__name__)
        
        # Performance monitoring
        self.performance_history = []
        
    def predict_batch_parallel(self, 
                             X_batch: np.ndarray,
                             use_cache: bool = True,
                             batch_size: Optional[int] = None) -> Tuple[np.ndarray, PerformanceMetrics]:
        """Parallel batch prediction with caching and optimization."""
        
        start_time = time.time()
        initial_memory = psutil.virtual_memory().used / 1024**2  # MB
        
        # Determine optimal batch size if not provided
        if batch_size is None:
            batch_size = self.gpu_optimizer.get_optimal_batch_size(
                self.model, X_batch.shape[1:]
            )
        
        predictions = []
        cache_hits = 0
        
        # Process in optimized batches
        for i in range(0, len(X_batch), batch_size):
            batch = X_batch[i:i + batch_size]
            
            batch_predictions = []
            for sample in batch:
                # Check cache first
                if use_cache:
                    cached_pred = self.prediction_cache.get(sample)
                    if cached_pred is not None:
                        batch_predictions.append(cached_pred)
                        cache_hits += 1
                        continue
                
                # Predict if not cached
                sample_batch = np.expand_dims(sample, axis=0)
                pred = self.model.predict(sample_batch, verbose=0)
                
                # Cache the prediction
                if use_cache:
                    self.prediction_cache.put(sample, pred[0])
                
                batch_predictions.append(pred[0])
            
            if batch_predictions:
                predictions.extend(batch_predictions)
        
        # Calculate performance metrics
        end_time = time.time()
        inference_time = end_time - start_time
        throughput = len(X_batch) / inference_time
        final_memory = psutil.virtual_memory().used / 1024**2
        memory_usage = final_memory - initial_memory
        
        # Get GPU utilization if available
        gpu_utilization = 0.0
        try:
            if GPUtil.getGPUs():
                gpu_utilization = GPUtil.getGPUs()[0].load * 100
        except:
            pass
        
        cpu_utilization = psutil.cpu_percent()
        
        # Model size estimation
        model_size = sum([param.numpy().nbytes for param in self.model.trainable_variables]) / 1024**2
        
        metrics = PerformanceMetrics(
            inference_time=inference_time,
            throughput=throughput,
            memory_usage=memory_usage,
            gpu_utilization=gpu_utilization,
            cpu_utilization=cpu_utilization,
            batch_size=batch_size,
            model_size=model_size
        )
        
        self.performance_history.append(metrics)
        
        self.logger.info(f"Processed {len(X_batch)} samples in {inference_time:.2f}s "
                        f"(throughput: {throughput:.1f} samples/sec, cache hits: {cache_hits})")
        
        return np.array(predictions), metrics
    
    def benchmark_performance(self, 
                            test_data: np.ndarray,
                            batch_sizes: List[int] = None,
                            num_runs: int = 3) -> Dict[str, Any]:
        """Comprehensive performance benchmarking."""
        
        if batch_sizes is None:
            batch_sizes = [1, 8, 16, 32, 64, 128]
        
        benchmark_results = {
            'timestamp': datetime.now().isoformat(),
            'test_samples': len(test_data),
            'batch_size_results': {},
            'optimal_batch_size': 1,
            'peak_throughput': 0.0
        }
        
        for batch_size in batch_sizes:
            batch_results = []
            
            for run in range(num_runs):
                try:
                    _, metrics = self.predict_batch_parallel(
                        test_data, 
                        use_cache=False,  # Disable cache for fair comparison
                        batch_size=batch_size
                    )
                    batch_results.append(metrics)
                    
                except Exception as e:
                    self.logger.warning(f"Batch size {batch_size} failed: {e}")
                    break
            
            if batch_results:
                avg_throughput = np.mean([m.throughput for m in batch_results])
                avg_memory = np.mean([m.memory_usage for m in batch_results])
                
                benchmark_results['batch_size_results'][batch_size] = {
                    'avg_throughput': float(avg_throughput),
                    'avg_memory_usage': float(avg_memory),
                    'avg_inference_time': float(np.mean([m.inference_time for m in batch_results])),
                    'std_throughput': float(np.std([m.throughput for m in batch_results]))
                }
                
                if avg_throughput > benchmark_results['peak_throughput']:
                    benchmark_results['peak_throughput'] = avg_throughput
                    benchmark_results['optimal_batch_size'] = batch_size
        
        return benchmark_results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        
        if not self.performance_history:
            return {'error': 'No performance data available'}
        
        throughputs = [m.throughput for m in self.performance_history]
        memory_usage = [m.memory_usage for m in self.performance_history]
        inference_times = [m.inference_time for m in self.performance_history]
        
        cache_stats = self.prediction_cache.get_stats()
        
        return {
            'total_predictions': len(self.performance_history),
            'avg_throughput': float(np.mean(throughputs)),
            'peak_throughput': float(np.max(throughputs)),
            'avg_memory_usage': float(np.mean(memory_usage)),
            'avg_inference_time': float(np.mean(inference_times)),
            'cache_performance': cache_stats,
            'model_size_mb': self.performance_history[0].model_size if self.performance_history else 0
        }


class AutoOptimizer:
    """Automated optimization pipeline for medical AI models."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize optimization components
        self.gpu_optimizer = GPUOptimizer()
        self.model_pruner = ModelPruner(config.pruning_sparsity) if config.enable_model_pruning else None
        self.model_quantizer = ModelQuantizer() if config.enable_quantization else None
    
    def optimize_model(self, 
                      model: tf.keras.Model,
                      validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """Apply comprehensive optimization pipeline."""
        
        optimization_results = {
            'original_model_size': 0,
            'optimized_model_size': 0,
            'optimization_steps': [],
            'performance_improvement': {},
            'optimized_model': model
        }
        
        # Calculate original model size
        original_size = sum([param.numpy().nbytes for param in model.trainable_variables]) / 1024**2
        optimization_results['original_model_size'] = original_size
        
        current_model = model
        
        # Step 1: GPU Optimization
        if self.config.enable_mixed_precision or self.config.enable_xla:
            self.logger.info("Applying GPU optimizations...")
            current_model = self.gpu_optimizer.optimize_model_for_gpu(current_model)
            optimization_results['optimization_steps'].append('gpu_optimization')
        
        # Step 2: Model Pruning
        if self.config.enable_model_pruning and self.model_pruner:
            self.logger.info(f"Applying model pruning (sparsity: {self.config.pruning_sparsity})...")
            current_model = self.model_pruner.prune_model(current_model)
            optimization_results['optimization_steps'].append('model_pruning')
        
        # Step 3: Quantization
        if self.config.enable_quantization and self.model_quantizer:
            self.logger.info(f"Applying {self.config.quantization_type} quantization...")
            
            if self.config.quantization_type == 'dynamic':
                quantized_model = self.model_quantizer.dynamic_quantize(current_model)
                if quantized_model:
                    optimization_results['quantized_model'] = quantized_model
                    optimization_results['optimization_steps'].append('dynamic_quantization')
        
        # Calculate optimized model size
        optimized_size = sum([param.numpy().nbytes for param in current_model.trainable_variables]) / 1024**2
        optimization_results['optimized_model_size'] = optimized_size
        optimization_results['size_reduction_ratio'] = original_size / optimized_size if optimized_size > 0 else 1.0
        
        # Performance benchmarking if validation data provided
        if validation_data is not None:
            X_val, y_val = validation_data
            
            # Benchmark original vs optimized model
            engine_original = DistributedInferenceEngine(model)
            engine_optimized = DistributedInferenceEngine(current_model)
            
            # Test with small sample for performance comparison
            test_sample = X_val[:100] if len(X_val) > 100 else X_val
            
            _, metrics_original = engine_original.predict_batch_parallel(test_sample, use_cache=False)
            _, metrics_optimized = engine_optimized.predict_batch_parallel(test_sample, use_cache=False)
            
            optimization_results['performance_improvement'] = {
                'throughput_improvement': metrics_optimized.throughput / metrics_original.throughput,
                'memory_reduction': (metrics_original.memory_usage - metrics_optimized.memory_usage) / metrics_original.memory_usage,
                'inference_time_reduction': (metrics_original.inference_time - metrics_optimized.inference_time) / metrics_original.inference_time
            }
        
        optimization_results['optimized_model'] = current_model
        
        self.logger.info(f"Optimization complete. Size reduction: {optimization_results['size_reduction_ratio']:.2f}x")
        
        return optimization_results


if __name__ == "__main__":
    # Demonstration of model optimization pipeline
    
    # Create sample model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Create optimization config
    config = OptimizationConfig(
        enable_mixed_precision=True,
        enable_xla=True,
        enable_model_pruning=True,
        pruning_sparsity=0.3,
        batch_size_optimization=True
    )
    
    # Initialize auto-optimizer
    optimizer = AutoOptimizer(config)
    
    # Create synthetic validation data
    X_val = np.random.randn(1000, 224, 224, 3)
    y_val = np.random.randint(0, 2, 1000)
    
    # Run optimization
    results = optimizer.optimize_model(model, (X_val, y_val))
    
    print(f"Optimization Results:")
    print(f"- Original model size: {results['original_model_size']:.2f} MB")
    print(f"- Optimized model size: {results['optimized_model_size']:.2f} MB")
    print(f"- Size reduction: {results['size_reduction_ratio']:.2f}x")
    print(f"- Optimization steps: {results['optimization_steps']}")
    
    if 'performance_improvement' in results:
        perf = results['performance_improvement']
        print(f"- Throughput improvement: {perf['throughput_improvement']:.2f}x")
        print(f"- Memory reduction: {perf['memory_reduction']*100:.1f}%")
        print(f"- Inference time reduction: {perf['inference_time_reduction']*100:.1f}%")