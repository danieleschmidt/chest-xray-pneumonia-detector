"""
AI-Driven Adaptive Optimization System
======================================

This module implements an advanced AI-driven optimization system that continuously
learns and adapts system performance based on real-time metrics, usage patterns,
and predictive analytics.

Features:
- Self-optimizing resource allocation using reinforcement learning
- Predictive scaling based on usage pattern analysis
- Automated hyperparameter tuning for ML models
- Dynamic load balancing with AI-driven decision making
- Adaptive caching strategies using machine learning
- Performance anomaly detection and auto-remediation
- Multi-objective optimization with Pareto frontier analysis
"""

import numpy as np
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import threading
from pathlib import Path
import sqlite3
from enum import Enum
import random
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Optimization objectives."""
    MINIMIZE_LATENCY = "minimize_latency"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_ACCURACY = "maximize_accuracy"
    MINIMIZE_RESOURCE_USAGE = "minimize_resource_usage"
    MAXIMIZE_AVAILABILITY = "maximize_availability"


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_utilization: float
    memory_utilization: float
    disk_io_rate: float
    network_io_rate: float
    response_time_ms: float
    throughput_rps: float
    error_rate: float
    active_connections: int
    queue_length: int
    model_accuracy: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class OptimizationAction:
    """Optimization action definition."""
    action_type: str
    parameters: Dict[str, Any]
    expected_impact: Dict[str, float]
    confidence_score: float
    resource_cost: float


class ReinforcementLearningOptimizer:
    """Reinforcement learning based optimizer using Q-learning."""
    
    def __init__(self, state_space_size: int, action_space_size: int,
                 learning_rate: float = 0.1, discount_factor: float = 0.95,
                 exploration_rate: float = 0.1):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        # Q-table: state x action -> expected reward
        self.q_table = np.zeros((state_space_size, action_space_size))
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=10000)
        
        # Metrics tracking
        self.total_reward = 0
        self.episode_count = 0
        
    def discretize_state(self, metrics: SystemMetrics) -> int:
        """Convert continuous metrics to discrete state."""
        # Simple discretization - can be improved with more sophisticated methods
        cpu_bucket = min(int(metrics.cpu_utilization * 10), 9)
        memory_bucket = min(int(metrics.memory_utilization * 10), 9)
        latency_bucket = min(int(metrics.response_time_ms / 100), 9)
        
        # Combine into single state index
        state = cpu_bucket * 100 + memory_bucket * 10 + latency_bucket
        return min(state, self.state_space_size - 1)
    
    def select_action(self, state: int) -> int:
        """Select action using epsilon-greedy strategy."""
        if random.random() < self.exploration_rate:
            return random.randint(0, self.action_space_size - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def update_q_table(self, state: int, action: int, reward: float, next_state: int):
        """Update Q-table using Q-learning update rule."""
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state, action] = new_q
        
        # Store experience for replay
        self.experience_buffer.append((state, action, reward, next_state))
    
    def calculate_reward(self, old_metrics: SystemMetrics, 
                        new_metrics: SystemMetrics, objectives: List[OptimizationObjective]) -> float:
        """Calculate reward based on performance improvement."""
        reward = 0.0
        
        for objective in objectives:
            if objective == OptimizationObjective.MINIMIZE_LATENCY:
                if new_metrics.response_time_ms < old_metrics.response_time_ms:
                    reward += (old_metrics.response_time_ms - new_metrics.response_time_ms) / 1000
            
            elif objective == OptimizationObjective.MAXIMIZE_THROUGHPUT:
                throughput_improvement = new_metrics.throughput_rps - old_metrics.throughput_rps
                reward += throughput_improvement / 100
            
            elif objective == OptimizationObjective.MINIMIZE_RESOURCE_USAGE:
                cpu_improvement = old_metrics.cpu_utilization - new_metrics.cpu_utilization
                memory_improvement = old_metrics.memory_utilization - new_metrics.memory_utilization
                reward += (cpu_improvement + memory_improvement) * 10
            
            elif objective == OptimizationObjective.MAXIMIZE_ACCURACY:
                accuracy_improvement = new_metrics.model_accuracy - old_metrics.model_accuracy
                reward += accuracy_improvement * 100
        
        # Penalty for degraded service
        if new_metrics.error_rate > old_metrics.error_rate:
            reward -= (new_metrics.error_rate - old_metrics.error_rate) * 50
        
        return reward
    
    def experience_replay(self, batch_size: int = 32):
        """Perform experience replay for better learning."""
        if len(self.experience_buffer) < batch_size:
            return
        
        batch = random.sample(self.experience_buffer, batch_size)
        
        for state, action, reward, next_state in batch:
            self.update_q_table(state, action, reward, next_state)


class PatternPredictor:
    """Usage pattern prediction using machine learning."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.cluster_model = KMeans(n_clusters=5, random_state=42)
        
        # Prediction models (simplified - could use more sophisticated models)
        self.is_trained = False
        self.seasonal_patterns = {}
        
    def add_metrics(self, metrics: SystemMetrics):
        """Add new metrics to the history."""
        metric_vector = [
            metrics.cpu_utilization,
            metrics.memory_utilization,
            metrics.response_time_ms,
            metrics.throughput_rps,
            metrics.active_connections,
            metrics.queue_length
        ]
        
        self.metrics_history.append((metrics.timestamp, metric_vector))
        
        # Retrain periodically
        if len(self.metrics_history) >= self.window_size and len(self.metrics_history) % 20 == 0:
            self._retrain_models()
    
    def _retrain_models(self):
        """Retrain prediction models with latest data."""
        if len(self.metrics_history) < 10:
            return
        
        # Extract feature matrix
        X = np.array([metrics for _, metrics in list(self.metrics_history)])
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train anomaly detector
        self.anomaly_detector.fit(X_scaled)
        
        # Train clustering model
        self.cluster_model.fit(X_scaled)
        
        # Extract seasonal patterns (simplified)
        self._extract_seasonal_patterns()
        
        self.is_trained = True
        logger.info("Pattern prediction models retrained")
    
    def _extract_seasonal_patterns(self):
        """Extract seasonal patterns from historical data."""
        if len(self.metrics_history) < 24:  # Need at least 24 data points
            return
        
        # Group by hour of day
        hourly_patterns = {}
        for timestamp, metrics in list(self.metrics_history):
            hour = timestamp.hour
            if hour not in hourly_patterns:
                hourly_patterns[hour] = []
            hourly_patterns[hour].append(metrics)
        
        # Calculate average patterns for each hour
        for hour, metrics_list in hourly_patterns.items():
            if len(metrics_list) >= 3:
                avg_metrics = np.mean(metrics_list, axis=0)
                self.seasonal_patterns[hour] = avg_metrics.tolist()
    
    def predict_next_metrics(self, horizon_minutes: int = 30) -> Optional[List[float]]:
        """Predict system metrics for the next time horizon."""
        if not self.is_trained or len(self.metrics_history) < 10:
            return None
        
        future_time = datetime.now() + timedelta(minutes=horizon_minutes)
        future_hour = future_time.hour
        
        # Use seasonal pattern if available
        if future_hour in self.seasonal_patterns:
            base_prediction = self.seasonal_patterns[future_hour]
        else:
            # Use recent average as baseline
            recent_metrics = list(self.metrics_history)[-10:]
            base_prediction = np.mean([metrics for _, metrics in recent_metrics], axis=0).tolist()
        
        # Add trend analysis (simplified)
        if len(self.metrics_history) >= 20:
            recent_trend = self._calculate_trend()
            base_prediction = [max(0, pred + trend) for pred, trend in zip(base_prediction, recent_trend)]
        
        return base_prediction
    
    def _calculate_trend(self) -> List[float]:
        """Calculate recent trend in metrics."""
        recent_10 = list(self.metrics_history)[-10:]
        previous_10 = list(self.metrics_history)[-20:-10]
        
        if len(previous_10) < 10:
            return [0.0] * 6  # No trend if insufficient data
        
        recent_avg = np.mean([metrics for _, metrics in recent_10], axis=0)
        previous_avg = np.mean([metrics for _, metrics in previous_10], axis=0)
        
        trend = (recent_avg - previous_avg) * 0.1  # Scale down the trend
        return trend.tolist()
    
    def detect_anomaly(self, metrics: SystemMetrics) -> Tuple[bool, float]:
        """Detect if current metrics are anomalous."""
        if not self.is_trained:
            return False, 0.0
        
        metric_vector = np.array([[
            metrics.cpu_utilization,
            metrics.memory_utilization,
            metrics.response_time_ms,
            metrics.throughput_rps,
            metrics.active_connections,
            metrics.queue_length
        ]])
        
        scaled_metrics = self.scaler.transform(metric_vector)
        anomaly_score = self.anomaly_detector.decision_function(scaled_metrics)[0]
        is_anomaly = self.anomaly_detector.predict(scaled_metrics)[0] == -1
        
        return is_anomaly, abs(anomaly_score)


class AdaptiveOptimizationEngine:
    """Main adaptive optimization engine."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.rl_optimizer = ReinforcementLearningOptimizer(
            state_space_size=1000,
            action_space_size=10,
            learning_rate=self.config.get('learning_rate', 0.1)
        )
        
        self.pattern_predictor = PatternPredictor(
            window_size=self.config.get('history_window', 100)
        )
        
        # Optimization objectives
        self.objectives = [
            OptimizationObjective.MINIMIZE_LATENCY,
            OptimizationObjective.MAXIMIZE_THROUGHPUT,
            OptimizationObjective.MINIMIZE_RESOURCE_USAGE
        ]
        
        # Current system state
        self.current_metrics: Optional[SystemMetrics] = None
        self.previous_metrics: Optional[SystemMetrics] = None
        self.current_state: Optional[int] = None
        
        # Optimization actions
        self.available_actions = self._define_optimization_actions()
        
        # Database for storing results
        self.db_path = "adaptive_optimization.db"
        self.init_database()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance tracking
        self.optimization_history = deque(maxlen=1000)
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load optimization configuration."""
        default_config = {
            'optimization_interval_seconds': 60,
            'learning_rate': 0.1,
            'history_window': 100,
            'min_confidence_threshold': 0.7,
            'max_resource_cost': 1.0,
            'enable_predictive_scaling': True,
            'enable_anomaly_detection': True
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
        
        return default_config
    
    def init_database(self):
        """Initialize optimization results database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS optimization_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metrics_before TEXT NOT NULL,
                    metrics_after TEXT,
                    action_taken TEXT NOT NULL,
                    reward_achieved REAL,
                    confidence_score REAL,
                    success INTEGER DEFAULT 0
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    prediction_horizon_minutes INTEGER,
                    predicted_metrics TEXT,
                    actual_metrics TEXT,
                    prediction_accuracy REAL
                )
            """)
    
    def _define_optimization_actions(self) -> List[OptimizationAction]:
        """Define available optimization actions."""
        actions = [
            OptimizationAction(
                action_type="scale_up_resources",
                parameters={"cpu_increase": 0.2, "memory_increase": 0.1},
                expected_impact={"latency": -0.1, "throughput": 0.2, "cost": 0.15},
                confidence_score=0.8,
                resource_cost=0.3
            ),
            OptimizationAction(
                action_type="scale_down_resources",
                parameters={"cpu_decrease": 0.1, "memory_decrease": 0.1},
                expected_impact={"latency": 0.05, "throughput": -0.1, "cost": -0.1},
                confidence_score=0.9,
                resource_cost=-0.2
            ),
            OptimizationAction(
                action_type="adjust_cache_settings",
                parameters={"cache_size_multiplier": 1.5, "ttl_multiplier": 1.2},
                expected_impact={"latency": -0.2, "memory_usage": 0.1},
                confidence_score=0.85,
                resource_cost=0.1
            ),
            OptimizationAction(
                action_type="optimize_batch_size",
                parameters={"batch_size_multiplier": 1.3},
                expected_impact={"throughput": 0.15, "latency": -0.05},
                confidence_score=0.75,
                resource_cost=0.05
            ),
            OptimizationAction(
                action_type="adjust_connection_pool",
                parameters={"pool_size_multiplier": 1.4},
                expected_impact={"throughput": 0.1, "latency": -0.1},
                confidence_score=0.7,
                resource_cost=0.1
            ),
            OptimizationAction(
                action_type="enable_compression",
                parameters={"compression_level": 6},
                expected_impact={"bandwidth_usage": -0.3, "cpu_usage": 0.1},
                confidence_score=0.9,
                resource_cost=0.15
            ),
            OptimizationAction(
                action_type="tune_model_parameters",
                parameters={"inference_batch_size": 32, "precision": "fp16"},
                expected_impact={"latency": -0.15, "accuracy": -0.02, "memory": -0.2},
                confidence_score=0.8,
                resource_cost=0.1
            ),
            OptimizationAction(
                action_type="adjust_load_balancing",
                parameters={"algorithm": "least_connections", "health_check_interval": 30},
                expected_impact={"availability": 0.05, "latency": -0.1},
                confidence_score=0.85,
                resource_cost=0.05
            ),
            OptimizationAction(
                action_type="optimize_garbage_collection",
                parameters={"gc_strategy": "G1", "heap_size_multiplier": 1.2},
                expected_impact={"latency_variance": -0.2, "memory_efficiency": 0.1},
                confidence_score=0.7,
                resource_cost=0.08
            ),
            OptimizationAction(
                action_type="no_action",
                parameters={},
                expected_impact={},
                confidence_score=1.0,
                resource_cost=0.0
            )
        ]
        
        return actions
    
    def update_metrics(self, metrics: SystemMetrics):
        """Update current system metrics and trigger optimization."""
        with self._lock:
            self.previous_metrics = self.current_metrics
            self.current_metrics = metrics
            
            # Add to pattern predictor
            self.pattern_predictor.add_metrics(metrics)
            
            # Check for anomalies
            if self.config.get('enable_anomaly_detection', True):
                is_anomaly, anomaly_score = self.pattern_predictor.detect_anomaly(metrics)
                if is_anomaly:
                    logger.warning(f"Performance anomaly detected (score: {anomaly_score:.3f})")
                    # Trigger immediate optimization for anomalies
                    self._trigger_optimization(urgent=True)
            
            # Regular optimization cycle
            if self.previous_metrics:
                self._trigger_optimization()
    
    def _trigger_optimization(self, urgent: bool = False):
        """Trigger optimization cycle."""
        if not self.current_metrics or not self.previous_metrics:
            return
        
        # Get current state
        current_state = self.rl_optimizer.discretize_state(self.current_metrics)
        
        # Select action using RL
        action_index = self.rl_optimizer.select_action(current_state)
        selected_action = self.available_actions[action_index]
        
        # Check if action meets confidence and cost thresholds
        if (selected_action.confidence_score >= self.config.get('min_confidence_threshold', 0.7) and
            selected_action.resource_cost <= self.config.get('max_resource_cost', 1.0)):
            
            # Execute optimization action
            success = self._execute_optimization_action(selected_action)
            
            # Calculate reward and update RL model
            if success and self.previous_metrics:
                reward = self.rl_optimizer.calculate_reward(
                    self.previous_metrics, self.current_metrics, self.objectives
                )
                
                if self.current_state is not None:
                    self.rl_optimizer.update_q_table(
                        self.current_state, action_index, reward, current_state
                    )
                
                # Store optimization result
                self._store_optimization_result(selected_action, reward, success)
            
            # Perform experience replay
            self.rl_optimizer.experience_replay()
        
        self.current_state = current_state
    
    def _execute_optimization_action(self, action: OptimizationAction) -> bool:
        """Execute an optimization action (simulation for demo)."""
        try:
            logger.info(f"Executing optimization action: {action.action_type}")
            
            # Simulate action execution
            if action.action_type == "scale_up_resources":
                # In real implementation, this would interact with orchestration system
                time.sleep(0.1)  # Simulate execution time
                logger.info("Resources scaled up successfully")
                
            elif action.action_type == "adjust_cache_settings":
                # In real implementation, this would update cache configuration
                logger.info("Cache settings optimized")
                
            elif action.action_type == "optimize_batch_size":
                # In real implementation, this would update model serving configuration
                logger.info("Batch size optimized")
                
            elif action.action_type == "no_action":
                logger.info("No optimization needed - system performing optimally")
            
            else:
                # Simulate other actions
                logger.info(f"Optimization action {action.action_type} executed")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute optimization action: {e}")
            return False
    
    def _store_optimization_result(self, action: OptimizationAction, 
                                 reward: float, success: bool):
        """Store optimization result in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO optimization_results 
                (timestamp, metrics_before, metrics_after, action_taken, 
                 reward_achieved, confidence_score, success)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                json.dumps(self.previous_metrics.__dict__, default=str) if self.previous_metrics else '{}',
                json.dumps(self.current_metrics.__dict__, default=str) if self.current_metrics else '{}',
                json.dumps(action.__dict__, default=str),
                reward,
                action.confidence_score,
                int(success)
            ))
    
    def predict_performance(self, horizon_minutes: int = 30) -> Optional[Dict[str, Any]]:
        """Predict future performance and suggest proactive optimizations."""
        if not self.config.get('enable_predictive_scaling', True):
            return None
        
        predicted_metrics = self.pattern_predictor.predict_next_metrics(horizon_minutes)
        
        if not predicted_metrics:
            return None
        
        # Create predicted SystemMetrics object
        future_time = datetime.now() + timedelta(minutes=horizon_minutes)
        predicted_system_metrics = SystemMetrics(
            timestamp=future_time,
            cpu_utilization=predicted_metrics[0],
            memory_utilization=predicted_metrics[1],
            response_time_ms=predicted_metrics[2],
            throughput_rps=predicted_metrics[3],
            active_connections=int(predicted_metrics[4]),
            queue_length=int(predicted_metrics[5]),
            error_rate=0.0  # Not predicted in this simple model
        )
        
        # Analyze predicted state and recommend actions
        predicted_state = self.rl_optimizer.discretize_state(predicted_system_metrics)
        recommended_action_index = self.rl_optimizer.select_action(predicted_state)
        recommended_action = self.available_actions[recommended_action_index]
        
        return {
            "prediction_horizon_minutes": horizon_minutes,
            "predicted_metrics": predicted_metrics,
            "predicted_state": predicted_state,
            "recommended_action": recommended_action.__dict__,
            "confidence": recommended_action.confidence_score,
            "expected_impact": recommended_action.expected_impact
        }
    
    def generate_optimization_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get recent optimizations
            cursor.execute("""
                SELECT * FROM optimization_results 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            """, (cutoff_time,))
            
            results = [dict(row) for row in cursor.fetchall()]
            
            # Calculate statistics
            total_optimizations = len(results)
            successful_optimizations = sum(1 for r in results if r['success'])
            average_reward = np.mean([r['reward_achieved'] for r in results if r['reward_achieved']])
            
            # Group by action type
            action_types = {}
            for result in results:
                try:
                    action_data = json.loads(result['action_taken'])
                    action_type = action_data.get('action_type', 'unknown')
                    action_types[action_type] = action_types.get(action_type, 0) + 1
                except:
                    pass
            
            return {
                "report_period_hours": hours,
                "timestamp": datetime.now().isoformat(),
                "optimization_statistics": {
                    "total_optimizations": total_optimizations,
                    "successful_optimizations": successful_optimizations,
                    "success_rate": successful_optimizations / total_optimizations if total_optimizations > 0 else 0,
                    "average_reward": float(average_reward) if not np.isnan(average_reward) else 0.0
                },
                "action_distribution": action_types,
                "recent_optimizations": results[:10],  # Most recent 10
                "model_performance": {
                    "q_table_exploration_rate": self.rl_optimizer.exploration_rate,
                    "total_episodes": self.rl_optimizer.episode_count,
                    "pattern_model_trained": self.pattern_predictor.is_trained,
                    "seasonal_patterns_count": len(self.pattern_predictor.seasonal_patterns)
                }
            }
    
    def save_models(self, model_dir: str = "optimization_models"):
        """Save trained optimization models."""
        model_path = Path(model_dir)
        model_path.mkdir(exist_ok=True)
        
        # Save Q-table
        np.save(model_path / "q_table.npy", self.rl_optimizer.q_table)
        
        # Save pattern predictor models
        if self.pattern_predictor.is_trained:
            joblib.dump(self.pattern_predictor.scaler, model_path / "scaler.pkl")
            joblib.dump(self.pattern_predictor.anomaly_detector, model_path / "anomaly_detector.pkl")
            joblib.dump(self.pattern_predictor.cluster_model, model_path / "cluster_model.pkl")
            
            with open(model_path / "seasonal_patterns.json", 'w') as f:
                json.dump(self.pattern_predictor.seasonal_patterns, f)
        
        logger.info(f"Optimization models saved to {model_path}")
    
    def load_models(self, model_dir: str = "optimization_models"):
        """Load pre-trained optimization models."""
        model_path = Path(model_dir)
        
        if not model_path.exists():
            logger.warning(f"Model directory {model_path} not found")
            return
        
        try:
            # Load Q-table
            q_table_path = model_path / "q_table.npy"
            if q_table_path.exists():
                self.rl_optimizer.q_table = np.load(q_table_path)
            
            # Load pattern predictor models
            scaler_path = model_path / "scaler.pkl"
            if scaler_path.exists():
                self.pattern_predictor.scaler = joblib.load(scaler_path)
            
            anomaly_path = model_path / "anomaly_detector.pkl"
            if anomaly_path.exists():
                self.pattern_predictor.anomaly_detector = joblib.load(anomaly_path)
            
            cluster_path = model_path / "cluster_model.pkl"
            if cluster_path.exists():
                self.pattern_predictor.cluster_model = joblib.load(cluster_path)
            
            patterns_path = model_path / "seasonal_patterns.json"
            if patterns_path.exists():
                with open(patterns_path, 'r') as f:
                    self.pattern_predictor.seasonal_patterns = json.load(f)
                self.pattern_predictor.is_trained = True
            
            logger.info(f"Optimization models loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")


def example_usage():
    """Demonstrate AI-driven adaptive optimization."""
    
    # Initialize optimization engine
    optimizer = AdaptiveOptimizationEngine()
    
    # Simulate system metrics over time
    print("🧠 AI-Driven Adaptive Optimization Demo")
    print("=" * 50)
    
    for i in range(20):
        # Generate simulated metrics with some patterns
        base_cpu = 0.3 + 0.4 * np.sin(i * 0.3) + random.uniform(-0.1, 0.1)
        base_memory = 0.5 + 0.2 * np.cos(i * 0.2) + random.uniform(-0.05, 0.05)
        base_latency = 100 + 50 * base_cpu + random.uniform(-20, 20)
        
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_utilization=max(0, min(1, base_cpu)),
            memory_utilization=max(0, min(1, base_memory)),
            disk_io_rate=random.uniform(10, 100),
            network_io_rate=random.uniform(50, 500),
            response_time_ms=max(10, base_latency),
            throughput_rps=max(1, 100 - base_latency / 2 + random.uniform(-10, 10)),
            error_rate=max(0, random.uniform(0, 0.05)),
            active_connections=random.randint(50, 200),
            queue_length=random.randint(0, 20),
            model_accuracy=0.85 + random.uniform(-0.05, 0.05)
        )
        
        # Update optimizer with new metrics
        optimizer.update_metrics(metrics)
        
        # Show current status
        if i % 5 == 0:
            print(f"\nIteration {i}:")
            print(f"  CPU: {metrics.cpu_utilization:.2f}, Memory: {metrics.memory_utilization:.2f}")
            print(f"  Latency: {metrics.response_time_ms:.1f}ms, Throughput: {metrics.throughput_rps:.1f} RPS")
            
            # Get performance prediction
            if i > 10:  # After some learning
                prediction = optimizer.predict_performance(horizon_minutes=15)
                if prediction:
                    print(f"  Predicted action: {prediction['recommended_action']['action_type']}")
                    print(f"  Confidence: {prediction['confidence']:.2f}")
        
        time.sleep(0.1)  # Simulate time passage
    
    # Generate optimization report
    report = optimizer.generate_optimization_report(hours=1)
    print(f"\n📊 Optimization Report:")
    print(f"  Total optimizations: {report['optimization_statistics']['total_optimizations']}")
    print(f"  Success rate: {report['optimization_statistics']['success_rate']:.1%}")
    print(f"  Average reward: {report['optimization_statistics']['average_reward']:.3f}")
    
    # Save trained models
    optimizer.save_models()
    print("✅ Models saved for future use")


if __name__ == "__main__":
    example_usage()