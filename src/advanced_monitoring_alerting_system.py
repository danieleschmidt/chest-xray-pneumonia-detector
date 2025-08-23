#!/usr/bin/env python3
"""
Advanced Monitoring and Alerting System - Generation 4 Enhancement
Comprehensive AI-driven monitoring with quantum-aware alerting and predictive anomaly detection.
"""

import json
import logging
import numpy as np
import threading
import time
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import hashlib
import smtplib
import requests
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import os

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

@dataclass
class MetricThreshold:
    """Metric threshold configuration"""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    comparison_operator: str  # '>', '<', '>=', '<=', '==', '!='
    evaluation_window_minutes: int
    min_samples_required: int
    hysteresis_factor: float = 0.1  # Prevent flapping

@dataclass
class Alert:
    """Comprehensive alert representation"""
    alert_id: str
    severity: AlertSeverity
    status: AlertStatus
    title: str
    description: str
    source_system: str
    metric_name: str
    current_value: float
    threshold_value: float
    quantum_correlation: Optional[float]
    tags: Dict[str, str]
    created_at: datetime
    updated_at: datetime
    acknowledged_at: Optional[datetime]
    resolved_at: Optional[datetime]
    acknowledged_by: Optional[str]
    escalation_level: int
    suppression_rules: List[str]

@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    anomaly_id: str
    metric_name: str
    anomaly_score: float
    anomaly_type: str  # 'spike', 'drop', 'trend', 'pattern', 'quantum'
    confidence_level: float
    baseline_value: float
    current_value: float
    detection_timestamp: datetime
    contextual_factors: Dict[str, Any]
    quantum_influence: Optional[float]

class QuantumAwareMetricsCollector:
    """Quantum-aware metrics collection system"""
    
    def __init__(self):
        self.metrics_buffer = defaultdict(deque)
        self.quantum_correlations = {}
        self.collection_interval = 10  # seconds
        self.is_collecting = False
        self.collector_thread = None
        self._lock = threading.Lock()
        
        # Quantum metrics tracking
        self.quantum_coherence_history = deque(maxlen=1000)
        self.quantum_entanglement_metrics = {}
        self.quantum_phase_tracking = {}
        
    def start_collection(self):
        """Start metrics collection"""
        if self.is_collecting:
            return
            
        self.is_collecting = True
        self.collector_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collector_thread.start()
        logger.info("Started quantum-aware metrics collection")
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.is_collecting = False
        if self.collector_thread and self.collector_thread.is_alive():
            self.collector_thread.join(timeout=1.0)
        logger.info("Stopped metrics collection")
    
    def _collection_loop(self):
        """Main collection loop"""
        while self.is_collecting:
            try:
                self._collect_system_metrics()
                self._collect_quantum_metrics()
                self._calculate_quantum_correlations()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(self.collection_interval * 2)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        timestamp = datetime.now()
        
        # Simulated system metrics collection
        try:
            import psutil
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            system_metrics = {
                'cpu_usage': cpu_usage,
                'memory_usage': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_usage': disk.percent,
                'disk_free_gb': disk.free / (1024**3)
            }
        except ImportError:
            # Fallback simulated metrics
            system_metrics = {
                'cpu_usage': max(0, min(100, np.random.normal(50, 15))),
                'memory_usage': max(0, min(100, np.random.normal(60, 10))),
                'memory_available_gb': max(1, np.random.normal(8, 2)),
                'disk_usage': max(0, min(100, np.random.normal(40, 8))),
                'disk_free_gb': max(10, np.random.normal(100, 20))
            }
        
        # Application-specific metrics (simulated)
        app_metrics = {
            'request_rate_per_sec': max(0, np.random.normal(50, 15)),
            'response_time_ms': max(10, np.random.normal(100, 30)),
            'error_rate_percent': max(0, min(10, np.random.exponential(0.5))),
            'active_connections': max(0, int(np.random.normal(25, 8))),
            'queue_length': max(0, int(np.random.poisson(5))),
            'cache_hit_rate': max(0, min(100, np.random.normal(85, 5)))
        }
        
        # Store metrics with timestamp
        with self._lock:
            for metric_name, value in {**system_metrics, **app_metrics}.items():
                self.metrics_buffer[metric_name].append({
                    'timestamp': timestamp.isoformat(),
                    'value': float(value)
                })
                
                # Keep only recent data
                if len(self.metrics_buffer[metric_name]) > 1000:
                    self.metrics_buffer[metric_name].popleft()
    
    def _collect_quantum_metrics(self):
        """Collect quantum-specific metrics"""
        timestamp = datetime.now()
        
        # Quantum coherence simulation
        base_coherence = 0.8
        noise = np.random.normal(0, 0.05)
        time_factor = np.sin(2 * np.pi * time.time() / 3600) * 0.1  # Hourly variation
        
        quantum_metrics = {
            'quantum_coherence': max(0, min(1, base_coherence + noise + time_factor)),
            'entanglement_strength': max(0, min(1, np.random.beta(8, 2))),
            'decoherence_rate': max(0, min(0.1, np.random.exponential(0.01))),
            'quantum_phase': (time.time() * 0.1) % (2 * np.pi),
            'measurement_fidelity': max(0.5, min(1, np.random.normal(0.9, 0.05))),
            'quantum_error_rate': max(0, min(0.1, np.random.exponential(0.005)))
        }
        
        # Store quantum metrics
        with self._lock:
            self.quantum_coherence_history.append({
                'timestamp': timestamp.isoformat(),
                'coherence': quantum_metrics['quantum_coherence'],
                'entanglement': quantum_metrics['entanglement_strength'],
                'phase': quantum_metrics['quantum_phase']
            })
            
            for metric_name, value in quantum_metrics.items():
                self.metrics_buffer[metric_name].append({
                    'timestamp': timestamp.isoformat(),
                    'value': float(value)
                })
                
                if len(self.metrics_buffer[metric_name]) > 500:
                    self.metrics_buffer[metric_name].popleft()
    
    def _calculate_quantum_correlations(self):
        """Calculate quantum correlations with system metrics"""
        if len(self.quantum_coherence_history) < 10:
            return
        
        # Get recent quantum coherence values
        recent_coherence = [entry['coherence'] for entry in list(self.quantum_coherence_history)[-10:]]
        
        # Calculate correlations with system metrics
        with self._lock:
            for metric_name, metric_data in self.metrics_buffer.items():
                if metric_name.startswith('quantum_') or len(metric_data) < 10:
                    continue
                
                recent_values = [entry['value'] for entry in list(metric_data)[-10:]]
                
                if len(recent_values) == len(recent_coherence):
                    correlation = np.corrcoef(recent_coherence, recent_values)[0, 1]
                    
                    # Filter out NaN correlations
                    if not np.isnan(correlation):
                        self.quantum_correlations[metric_name] = {
                            'correlation': float(correlation),
                            'last_updated': datetime.now().isoformat(),
                            'significance': abs(correlation) > 0.3
                        }
    
    def get_metric_values(self, metric_name: str, time_window_minutes: int = 60) -> List[Dict]:
        """Get metric values within time window"""
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        
        with self._lock:
            if metric_name not in self.metrics_buffer:
                return []
            
            filtered_values = []
            for entry in self.metrics_buffer[metric_name]:
                entry_time = datetime.fromisoformat(entry['timestamp'])
                if entry_time >= cutoff_time:
                    filtered_values.append(entry)
            
            return filtered_values
    
    def get_quantum_correlations(self) -> Dict[str, Dict]:
        """Get current quantum correlations"""
        with self._lock:
            return self.quantum_correlations.copy()

class AIAnomalyDetector:
    """AI-powered anomaly detection system"""
    
    def __init__(self):
        self.detection_models = {}
        self.baseline_calculators = {}
        self.anomaly_history = deque(maxlen=10000)
        self.detection_sensitivity = 0.8
        self._lock = threading.Lock()
        
        # Initialize detection models
        self._initialize_detection_models()
    
    def _initialize_detection_models(self):
        """Initialize anomaly detection models"""
        
        # Statistical threshold model
        self.detection_models['statistical'] = {
            'z_score_threshold': 2.5,
            'iqr_multiplier': 1.5,
            'moving_average_window': 20
        }
        
        # Quantum-aware model
        self.detection_models['quantum'] = {
            'coherence_threshold': 0.5,
            'entanglement_sensitivity': 0.7,
            'phase_variance_threshold': 0.3
        }
        
        # Pattern recognition model
        self.detection_models['pattern'] = {
            'seasonal_components': 24,  # Hours
            'trend_sensitivity': 0.1,
            'pattern_memory': 168  # Week in hours
        }
    
    def detect_anomalies(self, metric_name: str, values: List[Dict], 
                        quantum_correlations: Dict[str, Dict]) -> List[AnomalyDetection]:
        """Detect anomalies in metric values"""
        
        if len(values) < 10:
            return []
        
        anomalies = []
        current_time = datetime.now()
        
        # Extract numeric values and timestamps
        numeric_values = [entry['value'] for entry in values]
        timestamps = [datetime.fromisoformat(entry['timestamp']) for entry in values]
        
        # Statistical anomaly detection
        statistical_anomalies = self._detect_statistical_anomalies(
            metric_name, numeric_values, timestamps
        )
        anomalies.extend(statistical_anomalies)
        
        # Quantum-aware anomaly detection
        if metric_name in quantum_correlations:
            quantum_anomalies = self._detect_quantum_anomalies(
                metric_name, numeric_values, timestamps, quantum_correlations[metric_name]
            )
            anomalies.extend(quantum_anomalies)
        
        # Pattern-based anomaly detection
        pattern_anomalies = self._detect_pattern_anomalies(
            metric_name, numeric_values, timestamps
        )
        anomalies.extend(pattern_anomalies)
        
        # Store detected anomalies
        with self._lock:
            self.anomaly_history.extend(anomalies)
        
        return anomalies
    
    def _detect_statistical_anomalies(self, metric_name: str, values: List[float], 
                                    timestamps: List[datetime]) -> List[AnomalyDetection]:
        """Detect statistical anomalies using Z-score and IQR methods"""
        
        anomalies = []
        model = self.detection_models['statistical']
        
        if len(values) < model['moving_average_window']:
            return anomalies
        
        # Calculate moving statistics
        recent_values = values[-model['moving_average_window']:]
        mean_val = np.mean(recent_values)
        std_val = np.std(recent_values)
        
        # Z-score anomaly detection
        current_value = values[-1]
        if std_val > 0:
            z_score = abs(current_value - mean_val) / std_val
            
            if z_score > model['z_score_threshold']:
                anomaly_type = 'spike' if current_value > mean_val else 'drop'
                
                anomaly = AnomalyDetection(
                    anomaly_id=f"stat_{metric_name}_{int(time.time())}_{hash(str(current_value)) % 10000}",
                    metric_name=metric_name,
                    anomaly_score=float(z_score / model['z_score_threshold']),
                    anomaly_type=anomaly_type,
                    confidence_level=min(1.0, z_score / 5.0),
                    baseline_value=mean_val,
                    current_value=current_value,
                    detection_timestamp=timestamps[-1],
                    contextual_factors={
                        'z_score': float(z_score),
                        'baseline_mean': float(mean_val),
                        'baseline_std': float(std_val),
                        'detection_method': 'statistical_z_score'
                    },
                    quantum_influence=None
                )
                
                anomalies.append(anomaly)
        
        # IQR anomaly detection
        q1 = np.percentile(recent_values, 25)
        q3 = np.percentile(recent_values, 75)
        iqr = q3 - q1
        
        if iqr > 0:
            lower_bound = q1 - model['iqr_multiplier'] * iqr
            upper_bound = q3 + model['iqr_multiplier'] * iqr
            
            if current_value < lower_bound or current_value > upper_bound:
                outlier_type = 'drop' if current_value < lower_bound else 'spike'
                distance = abs(current_value - (lower_bound if current_value < lower_bound else upper_bound))
                
                anomaly = AnomalyDetection(
                    anomaly_id=f"iqr_{metric_name}_{int(time.time())}_{hash(str(current_value)) % 10000}",
                    metric_name=metric_name,
                    anomaly_score=float(distance / iqr),
                    anomaly_type=outlier_type,
                    confidence_level=min(1.0, distance / (iqr * 2)),
                    baseline_value=float((q1 + q3) / 2),
                    current_value=current_value,
                    detection_timestamp=timestamps[-1],
                    contextual_factors={
                        'iqr': float(iqr),
                        'q1': float(q1),
                        'q3': float(q3),
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound),
                        'detection_method': 'statistical_iqr'
                    },
                    quantum_influence=None
                )
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_quantum_anomalies(self, metric_name: str, values: List[float], 
                                timestamps: List[datetime], 
                                quantum_correlation: Dict) -> List[AnomalyDetection]:
        """Detect quantum-influenced anomalies"""
        
        anomalies = []
        model = self.detection_models['quantum']
        
        correlation_strength = abs(quantum_correlation.get('correlation', 0))
        
        # Only detect quantum anomalies for significantly correlated metrics
        if correlation_strength < 0.3:
            return anomalies
        
        current_value = values[-1]
        baseline = np.mean(values[-20:]) if len(values) >= 20 else np.mean(values)
        
        # Quantum influence calculation
        quantum_influence = correlation_strength * (current_value - baseline) / max(baseline, 1)
        
        # Detect quantum decoherence-influenced anomalies
        if abs(quantum_influence) > 0.2:
            anomaly_type = 'quantum'
            confidence = min(1.0, abs(quantum_influence) * 2)
            
            anomaly = AnomalyDetection(
                anomaly_id=f"quantum_{metric_name}_{int(time.time())}_{hash(str(current_value)) % 10000}",
                metric_name=metric_name,
                anomaly_score=float(abs(quantum_influence)),
                anomaly_type=anomaly_type,
                confidence_level=confidence,
                baseline_value=baseline,
                current_value=current_value,
                detection_timestamp=timestamps[-1],
                contextual_factors={
                    'quantum_correlation': correlation_strength,
                    'quantum_influence_strength': abs(quantum_influence),
                    'correlation_significance': quantum_correlation.get('significance', False),
                    'detection_method': 'quantum_correlation'
                },
                quantum_influence=float(quantum_influence)
            )
            
            anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_pattern_anomalies(self, metric_name: str, values: List[float], 
                                timestamps: List[datetime]) -> List[AnomalyDetection]:
        """Detect pattern-based anomalies"""
        
        anomalies = []
        model = self.detection_models['pattern']
        
        if len(values) < model['seasonal_components']:
            return anomalies
        
        # Simple trend detection
        if len(values) >= 10:
            recent_trend = np.polyfit(range(10), values[-10:], 1)[0]
            historical_trends = []
            
            # Calculate historical trends
            window_size = 10
            for i in range(window_size, len(values) - 10, 5):
                trend = np.polyfit(range(window_size), values[i-window_size:i], 1)[0]
                historical_trends.append(trend)
            
            if historical_trends:
                mean_trend = np.mean(historical_trends)
                std_trend = np.std(historical_trends)
                
                if std_trend > 0:
                    trend_z_score = abs(recent_trend - mean_trend) / std_trend
                    
                    if trend_z_score > 2.0:  # Significant trend anomaly
                        anomaly_type = 'trend'
                        
                        anomaly = AnomalyDetection(
                            anomaly_id=f"trend_{metric_name}_{int(time.time())}_{hash(str(recent_trend)) % 10000}",
                            metric_name=metric_name,
                            anomaly_score=float(trend_z_score / 3.0),
                            anomaly_type=anomaly_type,
                            confidence_level=min(1.0, trend_z_score / 4.0),
                            baseline_value=float(mean_trend),
                            current_value=float(recent_trend),
                            detection_timestamp=timestamps[-1],
                            contextual_factors={
                                'trend_z_score': float(trend_z_score),
                                'current_trend': float(recent_trend),
                                'baseline_trend': float(mean_trend),
                                'trend_std': float(std_trend),
                                'detection_method': 'pattern_trend'
                            },
                            quantum_influence=None
                        )
                        
                        anomalies.append(anomaly)
        
        return anomalies

class IntelligentAlertingSystem:
    """Intelligent alerting system with AI-driven routing and escalation"""
    
    def __init__(self):
        self.thresholds = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=10000)
        self.notification_channels = {}
        self.escalation_rules = {}
        self.suppression_rules = {}
        self._lock = threading.Lock()
        
        # Alert correlation tracking
        self.alert_correlations = {}
        self.correlation_window = timedelta(minutes=15)
        
        # Initialize default configuration
        self._initialize_default_config()
    
    def _initialize_default_config(self):
        """Initialize default alerting configuration"""
        
        # Default thresholds
        default_thresholds = [
            MetricThreshold('cpu_usage', 80.0, 95.0, '>', 5, 3),
            MetricThreshold('memory_usage', 85.0, 95.0, '>', 5, 3),
            MetricThreshold('disk_usage', 90.0, 98.0, '>', 10, 2),
            MetricThreshold('response_time_ms', 500.0, 1000.0, '>', 3, 5),
            MetricThreshold('error_rate_percent', 2.0, 5.0, '>', 2, 3),
            MetricThreshold('quantum_coherence', 0.5, 0.3, '<', 5, 3),
        ]
        
        for threshold in default_thresholds:
            self.thresholds[threshold.metric_name] = threshold
        
        # Default notification channels
        self.notification_channels = {
            'console': {'type': 'console', 'enabled': True},
            'webhook': {'type': 'webhook', 'enabled': False, 'url': None},
            'email': {'type': 'email', 'enabled': False, 'recipients': []}
        }
        
        # Default escalation rules
        self.escalation_rules = {
            'default': {
                'levels': [
                    {'severity': AlertSeverity.CRITICAL, 'escalate_after_minutes': 5},
                    {'severity': AlertSeverity.HIGH, 'escalate_after_minutes': 15},
                    {'severity': AlertSeverity.MEDIUM, 'escalate_after_minutes': 60}
                ],
                'max_escalation_level': 3
            }
        }
    
    def add_threshold(self, threshold: MetricThreshold):
        """Add or update a metric threshold"""
        with self._lock:
            self.thresholds[threshold.metric_name] = threshold
        logger.info(f"Added threshold for {threshold.metric_name}")
    
    def evaluate_thresholds(self, metric_name: str, values: List[Dict]) -> List[Alert]:
        """Evaluate thresholds for a metric"""
        
        if metric_name not in self.thresholds:
            return []
        
        threshold = self.thresholds[metric_name]
        
        if len(values) < threshold.min_samples_required:
            return []
        
        alerts = []
        current_time = datetime.now()
        
        # Get values within evaluation window
        cutoff_time = current_time - timedelta(minutes=threshold.evaluation_window_minutes)
        recent_values = [
            entry['value'] for entry in values
            if datetime.fromisoformat(entry['timestamp']) >= cutoff_time
        ]
        
        if len(recent_values) < threshold.min_samples_required:
            return []
        
        current_value = recent_values[-1]
        
        # Evaluate threshold conditions
        warning_triggered = self._evaluate_condition(
            current_value, threshold.warning_threshold, threshold.comparison_operator
        )
        
        critical_triggered = self._evaluate_condition(
            current_value, threshold.critical_threshold, threshold.comparison_operator
        )
        
        # Determine alert severity
        if critical_triggered:
            severity = AlertSeverity.CRITICAL
            threshold_value = threshold.critical_threshold
        elif warning_triggered:
            severity = AlertSeverity.HIGH
            threshold_value = threshold.warning_threshold
        else:
            return []  # No alert condition met
        
        # Check for existing alert (prevent duplicates)
        existing_alert_key = f"{metric_name}_{severity.value}"
        if existing_alert_key in self.active_alerts:
            existing_alert = self.active_alerts[existing_alert_key]
            
            # Apply hysteresis to prevent flapping
            hysteresis_threshold = threshold_value * (1 + threshold.hysteresis_factor)
            if threshold.comparison_operator == '>':
                if current_value < hysteresis_threshold:
                    self._resolve_alert(existing_alert.alert_id)
            elif threshold.comparison_operator == '<':
                hysteresis_threshold = threshold_value * (1 - threshold.hysteresis_factor)
                if current_value > hysteresis_threshold:
                    self._resolve_alert(existing_alert.alert_id)
            
            return []  # Don't create duplicate alert
        
        # Create new alert
        alert_id = f"alert_{metric_name}_{severity.value}_{int(time.time())}_{hash(str(current_value)) % 10000}"
        
        alert = Alert(
            alert_id=alert_id,
            severity=severity,
            status=AlertStatus.ACTIVE,
            title=f"{metric_name.replace('_', ' ').title()} {severity.value.title()} Alert",
            description=f"{metric_name} is {current_value:.2f}, exceeding {severity.value} threshold of {threshold_value:.2f}",
            source_system="monitoring_system",
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            quantum_correlation=None,
            tags={'metric': metric_name, 'threshold_type': severity.value},
            created_at=current_time,
            updated_at=current_time,
            acknowledged_at=None,
            resolved_at=None,
            acknowledged_by=None,
            escalation_level=0,
            suppression_rules=[]
        )
        
        alerts.append(alert)
        
        with self._lock:
            self.active_alerts[existing_alert_key] = alert
            self.alert_history.append(asdict(alert))
        
        return alerts
    
    def _evaluate_condition(self, value: float, threshold: float, operator: str) -> bool:
        """Evaluate threshold condition"""
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return abs(value - threshold) < 0.001
        elif operator == '!=':
            return abs(value - threshold) >= 0.001
        else:
            return False
    
    def process_anomaly_alerts(self, anomalies: List[AnomalyDetection]) -> List[Alert]:
        """Process anomalies and create alerts"""
        
        alerts = []
        current_time = datetime.now()
        
        for anomaly in anomalies:
            # Determine severity based on anomaly score and confidence
            if anomaly.confidence_level > 0.8 and anomaly.anomaly_score > 1.0:
                severity = AlertSeverity.CRITICAL
            elif anomaly.confidence_level > 0.6 or anomaly.anomaly_score > 0.7:
                severity = AlertSeverity.HIGH
            elif anomaly.confidence_level > 0.4 or anomaly.anomaly_score > 0.5:
                severity = AlertSeverity.MEDIUM
            else:
                severity = AlertSeverity.LOW
            
            # Create alert
            alert_id = f"anomaly_{anomaly.anomaly_id}"
            
            alert = Alert(
                alert_id=alert_id,
                severity=severity,
                status=AlertStatus.ACTIVE,
                title=f"Anomaly Detected: {anomaly.metric_name.replace('_', ' ').title()}",
                description=f"{anomaly.anomaly_type.title()} anomaly in {anomaly.metric_name}: "
                           f"current={anomaly.current_value:.2f}, baseline={anomaly.baseline_value:.2f}, "
                           f"confidence={anomaly.confidence_level:.1%}",
                source_system="anomaly_detector",
                metric_name=anomaly.metric_name,
                current_value=anomaly.current_value,
                threshold_value=anomaly.baseline_value,
                quantum_correlation=anomaly.quantum_influence,
                tags={
                    'metric': anomaly.metric_name,
                    'anomaly_type': anomaly.anomaly_type,
                    'detection_method': anomaly.contextual_factors.get('detection_method', 'unknown')
                },
                created_at=current_time,
                updated_at=current_time,
                acknowledged_at=None,
                resolved_at=None,
                acknowledged_by=None,
                escalation_level=0,
                suppression_rules=[]
            )
            
            alerts.append(alert)
            
            with self._lock:
                self.active_alerts[alert_id] = alert
                self.alert_history.append(asdict(alert))
        
        return alerts
    
    def _resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()
                alert.updated_at = datetime.now()
                del self.active_alerts[alert_id]
                logger.info(f"Resolved alert {alert_id}")
    
    def send_notifications(self, alerts: List[Alert]):
        """Send notifications for alerts"""
        for alert in alerts:
            self._send_alert_notification(alert)
    
    def _send_alert_notification(self, alert: Alert):
        """Send notification for a single alert"""
        
        # Console notification (always enabled)
        if self.notification_channels.get('console', {}).get('enabled', True):
            console_message = (
                f"\n🚨 ALERT: {alert.severity.value.upper()}\n"
                f"Title: {alert.title}\n"
                f"Description: {alert.description}\n"
                f"Metric: {alert.metric_name} = {alert.current_value:.2f}\n"
                f"Time: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Alert ID: {alert.alert_id}\n"
            )
            print(console_message)
        
        # Webhook notification
        webhook_config = self.notification_channels.get('webhook', {})
        if webhook_config.get('enabled', False) and webhook_config.get('url'):
            try:
                payload = {
                    'alert_id': alert.alert_id,
                    'severity': alert.severity.value,
                    'title': alert.title,
                    'description': alert.description,
                    'metric_name': alert.metric_name,
                    'current_value': alert.current_value,
                    'threshold_value': alert.threshold_value,
                    'created_at': alert.created_at.isoformat(),
                    'tags': alert.tags
                }
                
                response = requests.post(
                    webhook_config['url'],
                    json=payload,
                    timeout=10,
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code == 200:
                    logger.info(f"Webhook notification sent for alert {alert.alert_id}")
                else:
                    logger.warning(f"Webhook notification failed for alert {alert.alert_id}: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Error sending webhook notification: {e}")
        
        # Email notification (simplified)
        email_config = self.notification_channels.get('email', {})
        if email_config.get('enabled', False) and email_config.get('recipients'):
            logger.info(f"Email notification would be sent for alert {alert.alert_id}")
            # Actual email implementation would go here
    
    def get_active_alerts(self) -> List[Dict]:
        """Get all active alerts"""
        with self._lock:
            return [asdict(alert) for alert in self.active_alerts.values()]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        with self._lock:
            total_alerts = len(self.alert_history)
            active_count = len(self.active_alerts)
            
            # Severity distribution
            severity_counts = defaultdict(int)
            for alert_data in self.alert_history:
                severity_counts[alert_data['severity']] += 1
            
            # Recent alert rate
            recent_time = datetime.now() - timedelta(hours=1)
            recent_alerts = sum(
                1 for alert_data in self.alert_history
                if datetime.fromisoformat(alert_data['created_at']) >= recent_time
            )
            
            return {
                'total_alerts_generated': total_alerts,
                'active_alerts': active_count,
                'alerts_last_hour': recent_alerts,
                'severity_distribution': dict(severity_counts),
                'configured_thresholds': len(self.thresholds),
                'notification_channels': len([ch for ch in self.notification_channels.values() if ch.get('enabled')])
            }

class AdvancedMonitoringAlertingSystem:
    """Main advanced monitoring and alerting system"""
    
    def __init__(self, monitoring_interval: float = 30.0):
        self.metrics_collector = QuantumAwareMetricsCollector()
        self.anomaly_detector = AIAnomalyDetector()
        self.alerting_system = IntelligentAlertingSystem()
        
        self.monitoring_interval = monitoring_interval
        self.is_running = False
        self.monitor_thread = None
        self._lock = threading.Lock()
        
        self.system_stats = {
            'monitoring_cycles': 0,
            'alerts_generated': 0,
            'anomalies_detected': 0,
            'last_cycle_time': 0.0
        }
    
    def start_monitoring(self):
        """Start the monitoring system"""
        if self.is_running:
            logger.warning("Monitoring system already running")
            return
        
        self.is_running = True
        
        # Start metrics collection
        self.metrics_collector.start_collection()
        
        # Start monitoring loop
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Started advanced monitoring and alerting system")
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.is_running = False
        
        # Stop metrics collection
        self.metrics_collector.stop_collection()
        
        # Stop monitoring thread
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        logger.info("Stopped advanced monitoring and alerting system")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            cycle_start = time.time()
            
            try:
                # Get quantum correlations
                quantum_correlations = self.metrics_collector.get_quantum_correlations()
                
                # Evaluate all configured metrics
                all_alerts = []
                all_anomalies = []
                
                for metric_name in self.alerting_system.thresholds.keys():
                    # Get recent metric values
                    values = self.metrics_collector.get_metric_values(metric_name, time_window_minutes=60)
                    
                    if not values:
                        continue
                    
                    # Threshold-based alerting
                    threshold_alerts = self.alerting_system.evaluate_thresholds(metric_name, values)
                    all_alerts.extend(threshold_alerts)
                    
                    # Anomaly detection
                    anomalies = self.anomaly_detector.detect_anomalies(
                        metric_name, values, quantum_correlations
                    )
                    all_anomalies.extend(anomalies)
                    
                    # Anomaly-based alerting
                    if anomalies:
                        anomaly_alerts = self.alerting_system.process_anomaly_alerts(anomalies)
                        all_alerts.extend(anomaly_alerts)
                
                # Send notifications for new alerts
                if all_alerts:
                    self.alerting_system.send_notifications(all_alerts)
                
                # Update system statistics
                cycle_time = time.time() - cycle_start
                with self._lock:
                    self.system_stats['monitoring_cycles'] += 1
                    self.system_stats['alerts_generated'] += len(all_alerts)
                    self.system_stats['anomalies_detected'] += len(all_anomalies)
                    self.system_stats['last_cycle_time'] = cycle_time
                
                logger.debug(f"Monitoring cycle completed: {len(all_alerts)} alerts, "
                           f"{len(all_anomalies)} anomalies, {cycle_time:.3f}s")
                
                # Sleep until next cycle
                sleep_time = max(0, self.monitoring_interval - cycle_time)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        metrics_status = {
            'metrics_buffer_size': sum(len(buffer) for buffer in self.metrics_collector.metrics_buffer.values()),
            'quantum_correlations': len(self.metrics_collector.quantum_correlations),
            'collection_active': self.metrics_collector.is_collecting
        }
        
        anomaly_status = {
            'detection_models': len(self.anomaly_detector.detection_models),
            'anomaly_history_size': len(self.anomaly_detector.anomaly_history),
            'detection_sensitivity': self.anomaly_detector.detection_sensitivity
        }
        
        alert_stats = self.alerting_system.get_alert_statistics()
        
        return {
            'system_running': self.is_running,
            'monitoring_interval_seconds': self.monitoring_interval,
            'system_statistics': self.system_stats.copy(),
            'metrics_collector': metrics_status,
            'anomaly_detector': anomaly_status,
            'alerting_system': alert_stats,
            'active_alerts': len(self.alerting_system.active_alerts),
            'timestamp': datetime.now().isoformat()
        }

# Global monitoring system instance
_global_monitoring_system = None
_global_monitoring_lock = threading.Lock()

def get_monitoring_system() -> AdvancedMonitoringAlertingSystem:
    """Get global monitoring system instance"""
    global _global_monitoring_system
    with _global_monitoring_lock:
        if _global_monitoring_system is None:
            _global_monitoring_system = AdvancedMonitoringAlertingSystem()
        return _global_monitoring_system

def start_monitoring_service(monitoring_interval: float = 30.0):
    """Start the global monitoring service"""
    system = get_monitoring_system()
    system.monitoring_interval = monitoring_interval
    system.start_monitoring()
    return system

def stop_monitoring_service():
    """Stop the global monitoring service"""
    system = get_monitoring_system()
    system.stop_monitoring()

if __name__ == "__main__":
    # Demonstration
    print("Starting Advanced Monitoring and Alerting System...")
    
    system = start_monitoring_service(monitoring_interval=10.0)  # 10 second intervals for demo
    
    try:
        # Run for demonstration
        time.sleep(60)  # Run for 1 minute
        
        # Get system status
        status = system.get_system_status()
        print(f"\nMonitoring System Status:")
        print(f"System Running: {status['system_running']}")
        print(f"Monitoring Cycles: {status['system_statistics']['monitoring_cycles']}")
        print(f"Alerts Generated: {status['system_statistics']['alerts_generated']}")
        print(f"Anomalies Detected: {status['system_statistics']['anomalies_detected']}")
        print(f"Active Alerts: {status['active_alerts']}")
        print(f"Quantum Correlations: {status['metrics_collector']['quantum_correlations']}")
        
        # Show active alerts
        active_alerts = system.alerting_system.get_active_alerts()
        if active_alerts:
            print(f"\nActive Alerts ({len(active_alerts)}):")
            for alert in active_alerts[:5]:  # Show first 5
                print(f"  - {alert['severity'].upper()}: {alert['title']}")
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_monitoring_service()
        print("Advanced Monitoring and Alerting System stopped.")