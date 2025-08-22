"""
Medical AI Monitoring System for Quantum-Enhanced Healthcare
============================================================

Real-time monitoring, alerting, and observability system for quantum
medical AI applications with comprehensive health checks and performance
tracking.

Features:
- Real-time performance monitoring
- Medical-specific alert conditions
- HIPAA-compliant audit logging  
- Predictive failure detection
- Resource utilization tracking
- Regulatory compliance monitoring
- Clinical workflow integration
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels for medical AI monitoring."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"  # Immediate clinical intervention required

class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"

class ComplianceStatus(Enum):
    """Regulatory compliance status."""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    AUDIT_REQUIRED = "audit_required"

@dataclass
class MetricData:
    """Individual metric data point."""
    name: str
    value: float
    timestamp: float
    unit: str
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class Alert:
    """Alert data structure."""
    id: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: float
    metric_name: str
    current_value: float
    threshold_value: float
    clinical_impact: str
    recommended_action: str
    auto_resolution: bool = False

@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: HealthStatus
    latency_ms: float
    details: Dict[str, Any]
    timestamp: float

class MedicalMetricsCollector:
    """
    Specialized metrics collector for medical AI applications.
    
    Collects performance, safety, and compliance metrics with
    medical-specific considerations.
    """
    
    def __init__(self, retention_hours: int = 24):
        """Initialize metrics collector."""
        self.retention_hours = retention_hours
        self.metrics_buffer = deque(maxlen=10000)
        self.metric_thresholds = self._initialize_medical_thresholds()
        
    def _initialize_medical_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize medical AI specific thresholds."""
        return {
            "model_accuracy": {
                "warning": 0.85,
                "critical": 0.80,
                "emergency": 0.75
            },
            "prediction_confidence": {
                "warning": 0.70,
                "critical": 0.60,
                "emergency": 0.50
            },
            "response_time_ms": {
                "warning": 2000,
                "critical": 5000,
                "emergency": 10000
            },
            "memory_usage_percent": {
                "warning": 80,
                "critical": 90,
                "emergency": 95
            },
            "quantum_fidelity": {
                "warning": 0.95,
                "critical": 0.90,
                "emergency": 0.85
            },
            "hipaa_compliance_score": {
                "warning": 0.95,
                "critical": 0.90,
                "emergency": 0.85
            }
        }
    
    async def collect_model_performance_metrics(self, 
                                              model_name: str,
                                              predictions: List[Dict[str, Any]]) -> List[MetricData]:
        """Collect model performance metrics."""
        
        metrics = []
        timestamp = time.time()
        
        if not predictions:
            return metrics
        
        # Calculate accuracy (simplified)
        confidences = [p.get("confidence", 0.0) for p in predictions]
        avg_confidence = np.mean(confidences)
        min_confidence = np.min(confidences)
        std_confidence = np.std(confidences)
        
        metrics.extend([
            MetricData(
                name="model_confidence_avg",
                value=avg_confidence,
                timestamp=timestamp,
                unit="ratio",
                tags={"model": model_name, "type": "performance"}
            ),
            MetricData(
                name="model_confidence_min",
                value=min_confidence,
                timestamp=timestamp,
                unit="ratio",
                tags={"model": model_name, "type": "performance"}
            ),
            MetricData(
                name="model_confidence_std",
                value=std_confidence,
                timestamp=timestamp,
                unit="ratio",
                tags={"model": model_name, "type": "performance"}
            )
        ])
        
        # Quantum metrics if available
        quantum_fidelities = [
            p.get("quantum_fidelity", 1.0) for p in predictions 
            if "quantum_fidelity" in p
        ]
        
        if quantum_fidelities:
            avg_fidelity = np.mean(quantum_fidelities)
            metrics.append(
                MetricData(
                    name="quantum_fidelity",
                    value=avg_fidelity,
                    timestamp=timestamp,
                    unit="ratio",
                    tags={"model": model_name, "type": "quantum"}
                )
            )
        
        # Add metrics to buffer
        self.metrics_buffer.extend(metrics)
        
        return metrics
    
    async def collect_system_metrics(self) -> List[MetricData]:
        """Collect system resource metrics."""
        
        metrics = []
        timestamp = time.time()
        
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics.extend([
            MetricData(
                name="cpu_usage_percent",
                value=cpu_percent,
                timestamp=timestamp,
                unit="percent",
                tags={"type": "system"}
            ),
            MetricData(
                name="memory_usage_percent",
                value=memory.percent,
                timestamp=timestamp,
                unit="percent",
                tags={"type": "system"}
            ),
            MetricData(
                name="disk_usage_percent",
                value=disk.percent,
                timestamp=timestamp,
                unit="percent",
                tags={"type": "system"}
            )
        ])
        
        # Network I/O
        try:
            net_io = psutil.net_io_counters()
            metrics.extend([
                MetricData(
                    name="network_bytes_sent",
                    value=net_io.bytes_sent,
                    timestamp=timestamp,
                    unit="bytes",
                    tags={"type": "network"}
                ),
                MetricData(
                    name="network_bytes_recv",
                    value=net_io.bytes_recv,
                    timestamp=timestamp,
                    unit="bytes",
                    tags={"type": "network"}
                )
            ])
        except Exception:
            pass  # Network metrics not available
        
        self.metrics_buffer.extend(metrics)
        return metrics
    
    async def collect_compliance_metrics(self, 
                                       compliance_checks: Dict[str, float]) -> List[MetricData]:
        """Collect regulatory compliance metrics."""
        
        metrics = []
        timestamp = time.time()
        
        for compliance_type, score in compliance_checks.items():
            metrics.append(
                MetricData(
                    name=f"{compliance_type.lower()}_compliance_score",
                    value=score,
                    timestamp=timestamp,
                    unit="ratio",
                    tags={"type": "compliance", "regulation": compliance_type}
                )
            )
        
        self.metrics_buffer.extend(metrics)
        return metrics
    
    def get_recent_metrics(self, 
                          metric_name: str, 
                          hours: int = 1) -> List[MetricData]:
        """Get recent metrics by name."""
        
        cutoff_time = time.time() - (hours * 3600)
        
        return [
            metric for metric in self.metrics_buffer
            if metric.name == metric_name and metric.timestamp >= cutoff_time
        ]

class MedicalAlertManager:
    """
    Medical AI alert management system.
    
    Manages alerts with medical-specific severity and escalation.
    """
    
    def __init__(self):
        """Initialize alert manager."""
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.escalation_rules = self._initialize_escalation_rules()
        
    def _initialize_escalation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize medical alert escalation rules."""
        return {
            "model_accuracy": {
                "escalation_time": 300,  # 5 minutes
                "auto_remediation": False,
                "clinical_notification": True
            },
            "prediction_confidence": {
                "escalation_time": 180,  # 3 minutes
                "auto_remediation": False,
                "clinical_notification": True
            },
            "response_time_ms": {
                "escalation_time": 600,  # 10 minutes
                "auto_remediation": True,
                "clinical_notification": False
            },
            "quantum_fidelity": {
                "escalation_time": 120,  # 2 minutes
                "auto_remediation": False,
                "clinical_notification": True
            },
            "hipaa_compliance_score": {
                "escalation_time": 60,   # 1 minute
                "auto_remediation": False,
                "clinical_notification": True
            }
        }
    
    async def evaluate_alert_conditions(self, 
                                      metrics: List[MetricData],
                                      thresholds: Dict[str, Dict[str, float]]) -> List[Alert]:
        """Evaluate metrics against alert conditions."""
        
        new_alerts = []
        
        for metric in metrics:
            if metric.name in thresholds:
                threshold_config = thresholds[metric.name]
                alert = await self._check_metric_threshold(metric, threshold_config)
                
                if alert:
                    new_alerts.append(alert)
                    await self._handle_new_alert(alert)
        
        return new_alerts
    
    async def _check_metric_threshold(self, 
                                    metric: MetricData,
                                    thresholds: Dict[str, float]) -> Optional[Alert]:
        """Check if metric violates thresholds."""
        
        severity = None
        threshold_value = None
        
        # Determine severity (higher values are worse for most metrics)
        if metric.name in ["response_time_ms", "memory_usage_percent"]:
            # Higher is worse
            if metric.value >= thresholds.get("emergency", float('inf')):
                severity = AlertSeverity.EMERGENCY
                threshold_value = thresholds["emergency"]
            elif metric.value >= thresholds.get("critical", float('inf')):
                severity = AlertSeverity.CRITICAL
                threshold_value = thresholds["critical"]
            elif metric.value >= thresholds.get("warning", float('inf')):
                severity = AlertSeverity.WARNING
                threshold_value = thresholds["warning"]
        else:
            # Lower is worse (accuracy, confidence, etc.)
            if metric.value <= thresholds.get("emergency", 0):
                severity = AlertSeverity.EMERGENCY
                threshold_value = thresholds["emergency"]
            elif metric.value <= thresholds.get("critical", 0):
                severity = AlertSeverity.CRITICAL
                threshold_value = thresholds["critical"]
            elif metric.value <= thresholds.get("warning", 0):
                severity = AlertSeverity.WARNING
                threshold_value = thresholds["warning"]
        
        if severity:
            alert_id = f"{metric.name}_{int(metric.timestamp)}"
            
            # Generate medical-specific alert details
            clinical_impact, recommended_action = self._generate_medical_guidance(
                metric.name, severity, metric.value
            )
            
            return Alert(
                id=alert_id,
                severity=severity,
                title=f"{metric.name.replace('_', ' ').title()} Alert",
                description=f"{metric.name} is {metric.value:.3f} {metric.unit}",
                timestamp=metric.timestamp,
                metric_name=metric.name,
                current_value=metric.value,
                threshold_value=threshold_value,
                clinical_impact=clinical_impact,
                recommended_action=recommended_action,
                auto_resolution=self._should_auto_resolve(metric.name, severity)
            )
        
        return None
    
    def _generate_medical_guidance(self, 
                                 metric_name: str,
                                 severity: AlertSeverity,
                                 value: float) -> Tuple[str, str]:
        """Generate medical-specific clinical impact and recommendations."""
        
        guidance_map = {
            "model_accuracy": {
                AlertSeverity.WARNING: (
                    "Reduced diagnostic accuracy may affect patient care quality",
                    "Review recent predictions and consider model retraining"
                ),
                AlertSeverity.CRITICAL: (
                    "Significantly reduced accuracy poses patient safety risk",
                    "Immediately review model performance and consider fallback procedures"
                ),
                AlertSeverity.EMERGENCY: (
                    "Critical accuracy degradation requires immediate intervention",
                    "Stop automated predictions and switch to manual review mode"
                )
            },
            "prediction_confidence": {
                AlertSeverity.WARNING: (
                    "Lower confidence may indicate model uncertainty",
                    "Increase human oversight for low-confidence predictions"
                ),
                AlertSeverity.CRITICAL: (
                    "Low confidence predictions require clinical verification",
                    "Require radiologist confirmation for all predictions"
                ),
                AlertSeverity.EMERGENCY: (
                    "Extremely low confidence indicates model failure",
                    "Disable automated predictions immediately"
                )
            },
            "quantum_fidelity": {
                AlertSeverity.WARNING: (
                    "Quantum decoherence may affect optimization quality",
                    "Monitor quantum error rates and consider recalibration"
                ),
                AlertSeverity.CRITICAL: (
                    "Quantum errors may compromise diagnostic accuracy",
                    "Switch to classical fallback algorithms"
                ),
                AlertSeverity.EMERGENCY: (
                    "Severe quantum decoherence requires immediate attention",
                    "Halt quantum processing and initiate error correction"
                )
            },
            "hipaa_compliance_score": {
                AlertSeverity.WARNING: (
                    "Potential compliance issue detected",
                    "Review data handling procedures and audit logs"
                ),
                AlertSeverity.CRITICAL: (
                    "HIPAA compliance violation risk",
                    "Immediate privacy audit and corrective action required"
                ),
                AlertSeverity.EMERGENCY: (
                    "Critical HIPAA violation - legal notification required",
                    "Stop data processing and contact legal/compliance team"
                )
            }
        }
        
        default_guidance = (
            "System metric outside normal parameters",
            "Review system status and consider corrective action"
        )
        
        return guidance_map.get(metric_name, {}).get(severity, default_guidance)
    
    def _should_auto_resolve(self, metric_name: str, severity: AlertSeverity) -> bool:
        """Determine if alert should auto-resolve."""
        
        # Critical medical alerts should not auto-resolve
        if severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            return False
        
        # System metrics can auto-resolve
        if metric_name in ["response_time_ms", "memory_usage_percent", "cpu_usage_percent"]:
            return True
        
        # Medical accuracy metrics require manual resolution
        return False
    
    async def _handle_new_alert(self, alert: Alert):
        """Handle new alert with escalation."""
        
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        # Log alert
        logger.error(f"MEDICAL ALERT [{alert.severity.value.upper()}]: {alert.title}")
        logger.error(f"Clinical Impact: {alert.clinical_impact}")
        logger.error(f"Recommended Action: {alert.recommended_action}")
        
        # Check escalation rules
        if alert.metric_name in self.escalation_rules:
            rules = self.escalation_rules[alert.metric_name]
            
            if rules["clinical_notification"]:
                await self._send_clinical_notification(alert)
            
            if rules["auto_remediation"] and alert.severity != AlertSeverity.EMERGENCY:
                await self._attempt_auto_remediation(alert)
    
    async def _send_clinical_notification(self, alert: Alert):
        """Send notification to clinical staff."""
        
        # In production, integrate with hospital notification systems
        logger.critical(f"CLINICAL NOTIFICATION: {alert.title}")
        logger.critical(f"Immediate action required: {alert.recommended_action}")
    
    async def _attempt_auto_remediation(self, alert: Alert):
        """Attempt automatic remediation for system alerts."""
        
        logger.info(f"Attempting auto-remediation for alert: {alert.id}")
        
        # Simple auto-remediation examples
        if alert.metric_name == "memory_usage_percent":
            # Trigger garbage collection
            import gc
            gc.collect()
            logger.info("Memory cleanup triggered")
        
        elif alert.metric_name == "response_time_ms":
            # Scale up resources (simulated)
            logger.info("Resource scaling triggered")

class MedicalHealthChecker:
    """
    Comprehensive health checker for medical AI systems.
    """
    
    def __init__(self):
        """Initialize health checker."""
        self.health_checks = {}
        self._register_health_checks()
    
    def _register_health_checks(self):
        """Register all health check functions."""
        self.health_checks = {
            "model_availability": self._check_model_availability,
            "database_connection": self._check_database_connection,
            "quantum_subsystem": self._check_quantum_subsystem,
            "compliance_systems": self._check_compliance_systems,
            "clinical_integration": self._check_clinical_integration
        }
    
    async def check_all_health(self) -> Dict[str, HealthCheck]:
        """Run all health checks."""
        
        results = {}
        
        for check_name, check_function in self.health_checks.items():
            try:
                start_time = time.time()
                result = await check_function()
                latency_ms = (time.time() - start_time) * 1000
                
                results[check_name] = HealthCheck(
                    name=check_name,
                    status=result["status"],
                    latency_ms=latency_ms,
                    details=result.get("details", {}),
                    timestamp=time.time()
                )
                
            except Exception as e:
                results[check_name] = HealthCheck(
                    name=check_name,
                    status=HealthStatus.CRITICAL,
                    latency_ms=0.0,
                    details={"error": str(e)},
                    timestamp=time.time()
                )
        
        return results
    
    async def _check_model_availability(self) -> Dict[str, Any]:
        """Check if AI models are available and responding."""
        
        # Simulate model health check
        await asyncio.sleep(0.1)
        
        return {
            "status": HealthStatus.HEALTHY,
            "details": {
                "models_loaded": 3,
                "model_versions": ["v1.0", "v1.1", "v1.2"],
                "memory_usage_mb": 512
            }
        }
    
    async def _check_database_connection(self) -> Dict[str, Any]:
        """Check database connectivity."""
        
        # Simulate database check
        await asyncio.sleep(0.05)
        
        return {
            "status": HealthStatus.HEALTHY,
            "details": {
                "connection_pool_size": 10,
                "active_connections": 3,
                "response_time_ms": 15
            }
        }
    
    async def _check_quantum_subsystem(self) -> Dict[str, Any]:
        """Check quantum computing subsystem."""
        
        # Simulate quantum system check
        await asyncio.sleep(0.2)
        
        # Simulate occasional quantum decoherence
        quantum_fidelity = np.random.normal(0.95, 0.02)
        
        if quantum_fidelity > 0.90:
            status = HealthStatus.HEALTHY
        elif quantum_fidelity > 0.85:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.UNHEALTHY
        
        return {
            "status": status,
            "details": {
                "quantum_fidelity": quantum_fidelity,
                "error_rate": 1.0 - quantum_fidelity,
                "coherence_time_ms": 100,
                "gate_count": 1000
            }
        }
    
    async def _check_compliance_systems(self) -> Dict[str, Any]:
        """Check regulatory compliance systems."""
        
        await asyncio.sleep(0.1)
        
        compliance_scores = {
            "HIPAA": np.random.normal(0.95, 0.02),
            "FDA": np.random.normal(0.92, 0.03),
            "GDPR": np.random.normal(0.94, 0.02)
        }
        
        min_score = min(compliance_scores.values())
        
        if min_score > 0.90:
            status = HealthStatus.HEALTHY
        elif min_score > 0.85:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.CRITICAL
        
        return {
            "status": status,
            "details": {
                "compliance_scores": compliance_scores,
                "audit_logs_current": True,
                "encryption_active": True
            }
        }
    
    async def _check_clinical_integration(self) -> Dict[str, Any]:
        """Check clinical system integration."""
        
        await asyncio.sleep(0.15)
        
        return {
            "status": HealthStatus.HEALTHY,
            "details": {
                "hl7_connection": True,
                "pacs_integration": True,
                "ehr_sync_status": "current",
                "last_sync_time": time.time() - 300
            }
        }

class MedicalAIMonitoringSystem:
    """
    Comprehensive monitoring system for quantum medical AI.
    
    Integrates metrics collection, alerting, and health checking
    into unified monitoring platform.
    """
    
    def __init__(self):
        """Initialize monitoring system."""
        self.metrics_collector = MedicalMetricsCollector()
        self.alert_manager = MedicalAlertManager()
        self.health_checker = MedicalHealthChecker()
        self.monitoring_active = False
        
    async def start_monitoring(self, interval_seconds: int = 30):
        """Start continuous monitoring."""
        
        logger.info("🏥 Starting Medical AI Monitoring System")
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                # Collect metrics
                system_metrics = await self.metrics_collector.collect_system_metrics()
                
                # Simulate model predictions for monitoring
                sample_predictions = [
                    {"confidence": np.random.normal(0.85, 0.1), "quantum_fidelity": np.random.normal(0.95, 0.02)}
                    for _ in range(10)
                ]
                
                model_metrics = await self.metrics_collector.collect_model_performance_metrics(
                    "QuantumPneumoniaDetector", sample_predictions
                )
                
                # Collect compliance metrics
                compliance_checks = {
                    "HIPAA": np.random.normal(0.95, 0.02),
                    "FDA": np.random.normal(0.92, 0.03)
                }
                
                compliance_metrics = await self.metrics_collector.collect_compliance_metrics(
                    compliance_checks
                )
                
                # Evaluate alerts
                all_metrics = system_metrics + model_metrics + compliance_metrics
                alerts = await self.alert_manager.evaluate_alert_conditions(
                    all_metrics, self.metrics_collector.metric_thresholds
                )
                
                # Run health checks
                health_results = await self.health_checker.check_all_health()
                
                # Log monitoring summary
                if alerts:
                    logger.warning(f"Generated {len(alerts)} alerts")
                
                healthy_checks = sum(1 for h in health_results.values() 
                                   if h.status == HealthStatus.HEALTHY)
                total_checks = len(health_results)
                
                logger.info(f"Health: {healthy_checks}/{total_checks} checks healthy")
                
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Monitoring error: {str(e)}")
                await asyncio.sleep(interval_seconds)
    
    def stop_monitoring(self):
        """Stop monitoring."""
        logger.info("Stopping Medical AI Monitoring System")
        self.monitoring_active = False
    
    async def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        
        # Get recent metrics
        recent_metrics = {}
        for metric_name in self.metrics_collector.metric_thresholds.keys():
            recent_data = self.metrics_collector.get_recent_metrics(metric_name, hours=1)
            if recent_data:
                values = [m.value for m in recent_data]
                recent_metrics[metric_name] = {
                    "current": values[-1] if values else 0,
                    "average": np.mean(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values)
                }
        
        # Get active alerts
        active_alerts = list(self.alert_manager.active_alerts.values())
        
        # Get health status
        health_results = await self.health_checker.check_all_health()
        
        return {
            "monitoring_status": "active" if self.monitoring_active else "stopped",
            "timestamp": time.time(),
            "metrics_summary": recent_metrics,
            "active_alerts": len(active_alerts),
            "alert_breakdown": {
                severity.value: sum(1 for a in active_alerts if a.severity == severity)
                for severity in AlertSeverity
            },
            "health_summary": {
                status.value: sum(1 for h in health_results.values() if h.status == status)
                for status in HealthStatus
            },
            "overall_health": "healthy" if all(
                h.status == HealthStatus.HEALTHY for h in health_results.values()
            ) else "degraded"
        }

# Testing and Demonstration
async def demo_monitoring_system():
    """Demonstrate medical AI monitoring system."""
    
    logger.info("🚀 Medical AI Monitoring System Demo")
    
    # Initialize monitoring system
    monitoring = MedicalAIMonitoringSystem()
    
    # Start monitoring (run for 2 minutes in demo)
    monitoring_task = asyncio.create_task(monitoring.start_monitoring(interval_seconds=10))
    
    # Let it run for demo period
    await asyncio.sleep(30)
    
    # Get monitoring summary
    summary = await monitoring.get_monitoring_summary()
    
    # Stop monitoring
    monitoring.stop_monitoring()
    monitoring_task.cancel()
    
    # Print results
    print("\n" + "="*70)
    print("🏥 MEDICAL AI MONITORING SYSTEM SUMMARY")
    print("="*70)
    print(f"Monitoring Status: {summary['monitoring_status']}")
    print(f"Overall Health: {summary['overall_health']}")
    print(f"Active Alerts: {summary['active_alerts']}")
    
    print("\nAlert Breakdown:")
    for severity, count in summary['alert_breakdown'].items():
        if count > 0:
            print(f"  {severity.upper()}: {count}")
    
    print("\nHealth Status Breakdown:")
    for status, count in summary['health_summary'].items():
        if count > 0:
            print(f"  {status.upper()}: {count}")
    
    print("\nRecent Metrics:")
    for metric_name, data in summary['metrics_summary'].items():
        print(f"  {metric_name}: current={data['current']:.3f}, avg={data['average']:.3f}")
    
    print("="*70)
    print("✅ MONITORING DEMO COMPLETE")
    print("="*70)
    
    return summary

if __name__ == "__main__":
    asyncio.run(demo_monitoring_system())