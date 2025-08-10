# Adaptive Auto-Scaling System for Medical AI Workloads
# Intelligent resource management with predictive scaling

import time
import threading
import queue
import psutil
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from collections import deque
import pickle
from abc import ABC, abstractmethod


@dataclass
class ResourceMetrics:
    """System resource metrics for scaling decisions."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    gpu_memory: float
    disk_io: float
    network_io: float
    active_requests: int
    queue_depth: int
    response_time: float


@dataclass
class ScalingDecision:
    """Scaling decision with reasoning."""
    action: str  # scale_up, scale_down, maintain
    target_workers: int
    confidence: float
    reasoning: List[str]
    expected_improvement: float
    estimated_cost: float


class MetricsCollector:
    """Collects and monitors system metrics."""
    
    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self.metrics_history = deque(maxlen=1000)
        self.is_collecting = False
        self.collection_thread = None
        self.logger = logging.getLogger(__name__)
        
        # Request tracking
        self.active_requests = 0
        self.request_queue_depth = 0
        self.response_times = deque(maxlen=100)
        
    def start_collection(self):
        """Start metrics collection in background thread."""
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collect_metrics)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        self.logger.info("Started metrics collection")
    
    def stop_collection(self):
        """Stop metrics collection."""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        self.logger.info("Stopped metrics collection")
    
    def _collect_metrics(self):
        """Background metrics collection loop."""
        try:
            import GPUtil
            gpu_available = True
        except ImportError:
            gpu_available = False
            self.logger.warning("GPUtil not available, GPU metrics disabled")
        
        while self.is_collecting:
            try:
                # CPU and Memory
                cpu_usage = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                memory_usage = memory.percent
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                disk_io_rate = (disk_io.read_bytes + disk_io.write_bytes) / 1024**2  # MB/s
                
                # Network I/O
                network_io = psutil.net_io_counters()
                network_io_rate = (network_io.bytes_sent + network_io.bytes_recv) / 1024**2  # MB/s
                
                # GPU metrics
                gpu_usage = 0.0
                gpu_memory = 0.0
                if gpu_available:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]
                            gpu_usage = gpu.load * 100
                            gpu_memory = (gpu.memoryUsed / gpu.memoryTotal) * 100
                    except:
                        pass
                
                # Response time
                avg_response_time = np.mean(list(self.response_times)) if self.response_times else 0.0
                
                # Create metrics object
                metrics = ResourceMetrics(
                    timestamp=datetime.now(),
                    cpu_usage=cpu_usage,
                    memory_usage=memory_usage,
                    gpu_usage=gpu_usage,
                    gpu_memory=gpu_memory,
                    disk_io=disk_io_rate,
                    network_io=network_io_rate,
                    active_requests=self.active_requests,
                    queue_depth=self.request_queue_depth,
                    response_time=avg_response_time
                )
                
                self.metrics_history.append(metrics)
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")
                time.sleep(self.collection_interval)
    
    def update_request_metrics(self, active_requests: int, queue_depth: int, response_time: float):
        """Update request-specific metrics."""
        self.active_requests = active_requests
        self.request_queue_depth = queue_depth
        if response_time > 0:
            self.response_times.append(response_time)
    
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get most recent metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_window(self, window_minutes: int = 5) -> List[ResourceMetrics]:
        """Get metrics from specified time window."""
        if not self.metrics_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]


class LoadPredictor:
    """Predicts future load patterns for proactive scaling."""
    
    def __init__(self, prediction_horizon: int = 60):  # seconds
        self.prediction_horizon = prediction_horizon
        self.historical_patterns = {}
        self.model_weights = np.array([0.5, 0.3, 0.2])  # Simple weighted average
        self.logger = logging.getLogger(__name__)
    
    def predict_load(self, metrics_history: List[ResourceMetrics]) -> Dict[str, float]:
        """Predict future resource usage."""
        if len(metrics_history) < 3:
            return {'cpu': 50.0, 'memory': 50.0, 'requests': 10}
        
        # Extract time series
        recent_metrics = metrics_history[-10:]  # Last 10 data points
        
        cpu_values = [m.cpu_usage for m in recent_metrics]
        memory_values = [m.memory_usage for m in recent_metrics]
        request_values = [m.active_requests for m in recent_metrics]
        
        # Simple trend analysis
        cpu_trend = np.polyfit(range(len(cpu_values)), cpu_values, 1)[0]
        memory_trend = np.polyfit(range(len(memory_values)), memory_values, 1)[0]
        request_trend = np.polyfit(range(len(request_values)), request_values, 1)[0]
        
        # Predict future values
        cpu_prediction = cpu_values[-1] + (cpu_trend * self.prediction_horizon / 60)
        memory_prediction = memory_values[-1] + (memory_trend * self.prediction_horizon / 60)
        request_prediction = max(0, request_values[-1] + (request_trend * self.prediction_horizon / 60))
        
        # Apply bounds
        cpu_prediction = np.clip(cpu_prediction, 0, 100)
        memory_prediction = np.clip(memory_prediction, 0, 100)
        
        # Detect patterns (simplified)
        current_hour = datetime.now().hour
        if current_hour not in self.historical_patterns:
            self.historical_patterns[current_hour] = {
                'cpu_avg': cpu_prediction,
                'memory_avg': memory_prediction,
                'requests_avg': request_prediction,
                'samples': 1
            }
        else:
            # Update running average
            pattern = self.historical_patterns[current_hour]
            pattern['cpu_avg'] = (pattern['cpu_avg'] * pattern['samples'] + cpu_prediction) / (pattern['samples'] + 1)
            pattern['memory_avg'] = (pattern['memory_avg'] * pattern['samples'] + memory_prediction) / (pattern['samples'] + 1)
            pattern['requests_avg'] = (pattern['requests_avg'] * pattern['samples'] + request_prediction) / (pattern['samples'] + 1)
            pattern['samples'] += 1
        
        return {
            'cpu': float(cpu_prediction),
            'memory': float(memory_prediction),
            'requests': float(request_prediction),
            'confidence': min(0.9, len(recent_metrics) / 10.0)
        }


class AutoScaler:
    """Intelligent auto-scaling system."""
    
    def __init__(self, 
                 min_workers: int = 1,
                 max_workers: int = 10,
                 target_cpu_usage: float = 70.0,
                 target_memory_usage: float = 80.0,
                 scale_up_threshold: float = 85.0,
                 scale_down_threshold: float = 30.0,
                 cooldown_period: int = 300):  # seconds
        
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
        self.target_cpu_usage = target_cpu_usage
        self.target_memory_usage = target_memory_usage
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period
        
        self.last_scaling_action = datetime.now() - timedelta(seconds=cooldown_period)
        self.scaling_history = deque(maxlen=100)
        self.logger = logging.getLogger(__name__)
        
        # Load predictor
        self.load_predictor = LoadPredictor()
        
        # Worker pool management
        self.worker_pool: Optional[ThreadPoolExecutor] = None
        self.worker_pool_lock = threading.Lock()
        
    def make_scaling_decision(self, 
                            current_metrics: ResourceMetrics,
                            predicted_load: Dict[str, float]) -> ScalingDecision:
        """Make intelligent scaling decision based on current and predicted metrics."""
        
        reasoning = []
        action = "maintain"
        target_workers = self.current_workers
        confidence = 0.5
        
        # Check cooldown period
        time_since_last_scaling = (datetime.now() - self.last_scaling_action).total_seconds()
        if time_since_last_scaling < self.cooldown_period:
            reasoning.append(f"In cooldown period ({self.cooldown_period - time_since_last_scaling:.0f}s remaining)")
            return ScalingDecision(
                action=action,
                target_workers=target_workers,
                confidence=0.9,
                reasoning=reasoning,
                expected_improvement=0.0,
                estimated_cost=0.0
            )
        
        # Analyze current metrics
        high_cpu = current_metrics.cpu_usage > self.scale_up_threshold
        high_memory = current_metrics.memory_usage > self.scale_up_threshold
        high_response_time = current_metrics.response_time > 5.0  # 5 seconds
        high_queue_depth = current_metrics.queue_depth > 10
        
        low_cpu = current_metrics.cpu_usage < self.scale_down_threshold
        low_memory = current_metrics.memory_usage < self.scale_down_threshold
        low_response_time = current_metrics.response_time < 1.0
        low_queue_depth = current_metrics.queue_depth < 2
        
        # Analyze predicted metrics
        predicted_high_cpu = predicted_load['cpu'] > self.scale_up_threshold
        predicted_high_memory = predicted_load['memory'] > self.scale_up_threshold
        predicted_high_requests = predicted_load['requests'] > self.current_workers * 5
        
        # Scaling up conditions
        scale_up_score = 0
        
        if high_cpu:
            scale_up_score += 2
            reasoning.append(f"High CPU usage: {current_metrics.cpu_usage:.1f}%")
        
        if high_memory:
            scale_up_score += 2
            reasoning.append(f"High memory usage: {current_metrics.memory_usage:.1f}%")
        
        if high_response_time:
            scale_up_score += 3
            reasoning.append(f"High response time: {current_metrics.response_time:.2f}s")
        
        if high_queue_depth:
            scale_up_score += 2
            reasoning.append(f"High queue depth: {current_metrics.queue_depth}")
        
        if predicted_high_cpu:
            scale_up_score += 1
            reasoning.append(f"Predicted high CPU: {predicted_load['cpu']:.1f}%")
        
        if predicted_high_memory:
            scale_up_score += 1
            reasoning.append(f"Predicted high memory: {predicted_load['memory']:.1f}%")
        
        if predicted_high_requests:
            scale_up_score += 1
            reasoning.append(f"Predicted high request load: {predicted_load['requests']:.0f}")
        
        # Scaling down conditions
        scale_down_score = 0
        
        if low_cpu and low_memory and low_response_time and low_queue_depth:
            scale_down_score = 3
            reasoning.append("All metrics indicate low utilization")
        elif low_cpu and low_memory:
            scale_down_score = 2
            reasoning.append("CPU and memory usage are low")
        
        # Make decision
        if scale_up_score >= 4 and self.current_workers < self.max_workers:
            action = "scale_up"
            target_workers = min(self.max_workers, self.current_workers + 1)
            confidence = min(0.9, scale_up_score / 6.0)
            
            # More aggressive scaling for very high load
            if scale_up_score >= 7:
                target_workers = min(self.max_workers, self.current_workers + 2)
                confidence = 0.95
                
        elif scale_down_score >= 2 and self.current_workers > self.min_workers:
            action = "scale_down"
            target_workers = max(self.min_workers, self.current_workers - 1)
            confidence = min(0.8, scale_down_score / 3.0)
        
        # Calculate expected improvement and cost
        expected_improvement = 0.0
        estimated_cost = 0.0
        
        if action == "scale_up":
            # Estimate performance improvement
            cpu_relief = max(0, current_metrics.cpu_usage - self.target_cpu_usage)
            memory_relief = max(0, current_metrics.memory_usage - self.target_memory_usage)
            expected_improvement = (cpu_relief + memory_relief) / 2.0
            estimated_cost = (target_workers - self.current_workers) * 10.0  # Simplified cost model
            
        elif action == "scale_down":
            # Estimate cost savings
            estimated_cost = (self.current_workers - target_workers) * -10.0  # Negative cost = savings
            expected_improvement = 0.0
        
        return ScalingDecision(
            action=action,
            target_workers=target_workers,
            confidence=confidence,
            reasoning=reasoning,
            expected_improvement=expected_improvement,
            estimated_cost=estimated_cost
        )
    
    def execute_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Execute the scaling decision."""
        
        if decision.action == "maintain":
            return True
        
        self.logger.info(f"Executing scaling decision: {decision.action} to {decision.target_workers} workers")
        self.logger.info(f"Reasoning: {'; '.join(decision.reasoning)}")
        
        try:
            old_workers = self.current_workers
            
            with self.worker_pool_lock:
                # Update worker pool
                if self.worker_pool:
                    self.worker_pool.shutdown(wait=False)
                
                self.worker_pool = ThreadPoolExecutor(max_workers=decision.target_workers)
                self.current_workers = decision.target_workers
            
            # Record scaling action
            scaling_record = {
                'timestamp': datetime.now(),
                'action': decision.action,
                'old_workers': old_workers,
                'new_workers': self.current_workers,
                'confidence': decision.confidence,
                'reasoning': decision.reasoning
            }
            
            self.scaling_history.append(scaling_record)
            self.last_scaling_action = datetime.now()
            
            self.logger.info(f"Successfully scaled from {old_workers} to {self.current_workers} workers")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to execute scaling decision: {e}")
            return False
    
    def get_scaling_recommendations(self, metrics_history: List[ResourceMetrics]) -> Dict[str, Any]:
        """Get scaling recommendations based on historical data."""
        
        if not metrics_history:
            return {'error': 'No metrics available'}
        
        current_metrics = metrics_history[-1]
        predicted_load = self.load_predictor.predict_load(metrics_history)
        
        decision = self.make_scaling_decision(current_metrics, predicted_load)
        
        # Analyze recent scaling effectiveness
        recent_scaling_actions = [r for r in self.scaling_history 
                                if r['timestamp'] > datetime.now() - timedelta(hours=1)]
        
        scaling_effectiveness = 0.5  # Default
        if recent_scaling_actions:
            # Simple effectiveness metric based on whether scaling reduced high resource usage
            scaling_effectiveness = min(1.0, len([r for r in recent_scaling_actions 
                                                 if r['confidence'] > 0.7]) / len(recent_scaling_actions))
        
        return {
            'current_workers': self.current_workers,
            'recommended_action': decision.action,
            'recommended_workers': decision.target_workers,
            'confidence': decision.confidence,
            'reasoning': decision.reasoning,
            'expected_improvement': decision.expected_improvement,
            'estimated_cost': decision.estimated_cost,
            'predicted_load': predicted_load,
            'scaling_effectiveness': scaling_effectiveness,
            'recent_scaling_actions': len(recent_scaling_actions)
        }


class AdaptiveScalingManager:
    """High-level manager for adaptive scaling system."""
    
    def __init__(self, 
                 scaling_config: Optional[Dict[str, Any]] = None,
                 monitoring_interval: float = 30.0):
        
        # Default configuration
        default_config = {
            'min_workers': 1,
            'max_workers': 8,
            'target_cpu_usage': 70.0,
            'scale_up_threshold': 80.0,
            'scale_down_threshold': 30.0,
            'cooldown_period': 300
        }
        
        config = {**default_config, **(scaling_config or {})}
        
        self.metrics_collector = MetricsCollector()
        self.auto_scaler = AutoScaler(**config)
        self.monitoring_interval = monitoring_interval
        
        # Control flags
        self.is_running = False
        self.monitoring_thread = None
        
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start the adaptive scaling system."""
        self.logger.info("Starting adaptive scaling system...")
        
        self.metrics_collector.start_collection()
        self.is_running = True
        
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info("Adaptive scaling system started")
    
    def stop(self):
        """Stop the adaptive scaling system."""
        self.logger.info("Stopping adaptive scaling system...")
        
        self.is_running = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        self.metrics_collector.stop_collection()
        
        # Cleanup worker pool
        with self.auto_scaler.worker_pool_lock:
            if self.auto_scaler.worker_pool:
                self.auto_scaler.worker_pool.shutdown(wait=True)
        
        self.logger.info("Adaptive scaling system stopped")
    
    def _monitoring_loop(self):
        """Main monitoring and scaling decision loop."""
        
        while self.is_running:
            try:
                # Get recent metrics
                metrics_window = self.metrics_collector.get_metrics_window(window_minutes=5)
                
                if len(metrics_window) >= 3:  # Need minimum data for decisions
                    current_metrics = metrics_window[-1]
                    predicted_load = self.auto_scaler.load_predictor.predict_load(metrics_window)
                    
                    # Make scaling decision
                    decision = self.auto_scaler.make_scaling_decision(current_metrics, predicted_load)
                    
                    # Execute if action needed
                    if decision.action != "maintain" and decision.confidence > 0.6:
                        self.auto_scaler.execute_scaling_decision(decision)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        
        current_metrics = self.metrics_collector.get_current_metrics()
        recent_metrics = self.metrics_collector.get_metrics_window(window_minutes=5)
        
        status = {
            'is_running': self.is_running,
            'current_workers': self.auto_scaler.current_workers,
            'worker_limits': {
                'min': self.auto_scaler.min_workers,
                'max': self.auto_scaler.max_workers
            },
            'current_metrics': asdict(current_metrics) if current_metrics else None,
            'recent_scaling_actions': len(self.auto_scaler.scaling_history),
            'metrics_collected': len(recent_metrics),
            'system_health': 'healthy' if current_metrics and current_metrics.cpu_usage < 90 else 'stressed'
        }
        
        # Add recommendations
        if recent_metrics:
            recommendations = self.auto_scaler.get_scaling_recommendations(recent_metrics)
            status['recommendations'] = recommendations
        
        return status
    
    def manual_scale(self, target_workers: int, reason: str = "Manual scaling") -> bool:
        """Manually override scaling decision."""
        
        if not (self.auto_scaler.min_workers <= target_workers <= self.auto_scaler.max_workers):
            self.logger.error(f"Target workers {target_workers} outside allowed range")
            return False
        
        decision = ScalingDecision(
            action="scale_up" if target_workers > self.auto_scaler.current_workers else "scale_down",
            target_workers=target_workers,
            confidence=1.0,
            reasoning=[reason],
            expected_improvement=0.0,
            estimated_cost=0.0
        )
        
        return self.auto_scaler.execute_scaling_decision(decision)


if __name__ == "__main__":
    # Demonstration of adaptive scaling system
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create scaling configuration
    scaling_config = {
        'min_workers': 2,
        'max_workers': 6,
        'target_cpu_usage': 65.0,
        'scale_up_threshold': 75.0,
        'scale_down_threshold': 25.0,
        'cooldown_period': 60  # Shorter for demo
    }
    
    # Initialize scaling manager
    scaling_manager = AdaptiveScalingManager(scaling_config, monitoring_interval=10.0)
    
    try:
        # Start the system
        scaling_manager.start()
        
        # Run for demonstration period
        print("Adaptive scaling system running...")
        print("Monitor the logs to see scaling decisions...")
        
        for i in range(12):  # Run for 2 minutes
            time.sleep(10)
            status = scaling_manager.get_status()
            print(f"\nStatus Update {i+1}:")
            print(f"Workers: {status['current_workers']}")
            print(f"System Health: {status['system_health']}")
            
            if status.get('current_metrics'):
                metrics = status['current_metrics']
                print(f"CPU: {metrics['cpu_usage']:.1f}%, Memory: {metrics['memory_usage']:.1f}%")
        
    finally:
        # Stop the system
        scaling_manager.stop()
        print("\nAdaptive scaling demonstration completed")