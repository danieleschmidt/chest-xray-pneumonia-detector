#!/usr/bin/env python3
"""
Intelligent Error Recovery System - Generation 2: MAKE IT ROBUST
Self-healing system with predictive failure detection and automatic recovery.
"""

import asyncio
import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from collections import deque
import numpy as np

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryAction(Enum):
    """Types of recovery actions."""
    RETRY = "retry"
    FALLBACK = "fallback"
    RESTART_SERVICE = "restart_service"
    SCALE_UP = "scale_up"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    MANUAL_INTERVENTION = "manual_intervention"

class ErrorPattern(Enum):
    """Common error patterns."""
    TRANSIENT = "transient"
    CASCADING = "cascading"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    TIMEOUT = "timeout"
    DEPENDENCY_FAILURE = "dependency_failure"
    DATA_CORRUPTION = "data_corruption"

@dataclass
class ErrorEvent:
    """Error event data structure."""
    id: str
    timestamp: float
    service_name: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    pattern: Optional[ErrorPattern] = None
    resolved: bool = False
    recovery_actions: List[RecoveryAction] = field(default_factory=list)

@dataclass
class RecoveryStrategy:
    """Recovery strategy definition."""
    name: str
    condition: Callable[[ErrorEvent], bool]
    actions: List[RecoveryAction]
    max_retries: int = 3
    backoff_multiplier: float = 2.0
    timeout_seconds: int = 300
    success_threshold: float = 0.8

class PredictiveAnalyzer:
    """Predictive failure analysis using patterns."""
    
    def __init__(self, history_size: int = 1000):
        self.error_history = deque(maxlen=history_size)
        self.pattern_weights = {
            ErrorPattern.TRANSIENT: 0.2,
            ErrorPattern.CASCADING: 0.8,
            ErrorPattern.RESOURCE_EXHAUSTION: 0.7,
            ErrorPattern.TIMEOUT: 0.5,
            ErrorPattern.DEPENDENCY_FAILURE: 0.6,
            ErrorPattern.DATA_CORRUPTION: 0.9
        }
        
    def add_error_event(self, error: ErrorEvent):
        """Add error event to history."""
        self.error_history.append(error)
        
    def predict_failure_probability(self, service_name: str, 
                                  time_window_minutes: int = 30) -> Dict[str, float]:
        """Predict probability of failure in the next time window."""
        current_time = time.time()
        window_start = current_time - (time_window_minutes * 60)
        
        # Get recent errors for the service
        recent_errors = [
            error for error in self.error_history
            if error.service_name == service_name and error.timestamp > window_start
        ]
        
        if not recent_errors:
            return {'probability': 0.0, 'confidence': 0.5}
            
        # Calculate failure indicators
        error_rate = len(recent_errors) / time_window_minutes
        severity_score = sum(self._get_severity_weight(error.severity) for error in recent_errors)
        pattern_score = sum(self.pattern_weights.get(error.pattern, 0.5) for error in recent_errors)
        
        # Combine indicators
        base_probability = min(error_rate * 0.1, 1.0)
        severity_factor = min(severity_score / len(recent_errors), 1.0)
        pattern_factor = min(pattern_score / len(recent_errors), 1.0)
        
        probability = min(base_probability * severity_factor * pattern_factor, 1.0)
        confidence = min(len(recent_errors) / 10.0, 1.0)  # More data = higher confidence
        
        return {
            'probability': probability,
            'confidence': confidence,
            'error_rate': error_rate,
            'recent_errors': len(recent_errors),
            'primary_patterns': self._get_primary_patterns(recent_errors)
        }
        
    def _get_severity_weight(self, severity: ErrorSeverity) -> float:
        """Get numeric weight for error severity."""
        weights = {
            ErrorSeverity.LOW: 1.0,
            ErrorSeverity.MEDIUM: 2.0,
            ErrorSeverity.HIGH: 4.0,
            ErrorSeverity.CRITICAL: 8.0
        }
        return weights.get(severity, 1.0)
        
    def _get_primary_patterns(self, errors: List[ErrorEvent]) -> List[str]:
        """Get most common error patterns."""
        pattern_counts = {}
        for error in errors:
            if error.pattern:
                pattern_counts[error.pattern.value] = pattern_counts.get(error.pattern.value, 0) + 1
                
        # Return top 3 patterns
        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        return [pattern for pattern, count in sorted_patterns[:3]]
        
    def detect_anomalies(self, service_name: str) -> Dict[str, Any]:
        """Detect anomalous error patterns."""
        recent_errors = [
            error for error in self.error_history
            if error.service_name == service_name and 
            error.timestamp > time.time() - 3600  # Last hour
        ]
        
        if len(recent_errors) < 5:
            return {'anomalous': False}
            
        # Analyze error distribution
        error_intervals = []
        for i in range(1, len(recent_errors)):
            interval = recent_errors[i].timestamp - recent_errors[i-1].timestamp
            error_intervals.append(interval)
            
        if error_intervals:
            mean_interval = np.mean(error_intervals)
            std_interval = np.std(error_intervals)
            
            # Check for burst patterns (errors clustered in time)
            burst_threshold = mean_interval - (2 * std_interval)
            burst_count = sum(1 for interval in error_intervals if interval < burst_threshold)
            
            if burst_count > len(error_intervals) * 0.3:  # 30% of errors in bursts
                return {
                    'anomalous': True,
                    'pattern': 'error_burst',
                    'confidence': min(burst_count / len(error_intervals), 1.0)
                }
                
        return {'anomalous': False}

class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
        
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
                
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
                
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                
            raise e
            
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'failure_threshold': self.failure_threshold,
            'last_failure_time': self.last_failure_time,
            'time_until_retry': max(0, self.recovery_timeout - (time.time() - self.last_failure_time))
        }

class SelfHealingService:
    """Self-healing service wrapper."""
    
    def __init__(self, service_name: str, health_check_func: Callable,
                 restart_func: Optional[Callable] = None):
        self.service_name = service_name
        self.health_check_func = health_check_func
        self.restart_func = restart_func
        self.is_healthy = True
        self.last_health_check = time.time()
        self.restart_count = 0
        self.max_restarts = 3
        self.restart_window = 3600  # 1 hour
        self.restart_timestamps = deque(maxlen=self.max_restarts)
        
    async def check_health(self) -> bool:
        """Check service health."""
        try:
            if asyncio.iscoroutinefunction(self.health_check_func):
                self.is_healthy = await self.health_check_func()
            else:
                self.is_healthy = self.health_check_func()
                
            self.last_health_check = time.time()
            return self.is_healthy
            
        except Exception as e:
            self.is_healthy = False
            logging.error(f"Health check failed for {self.service_name}: {e}")
            return False
            
    async def attempt_self_heal(self) -> bool:
        """Attempt to heal the service."""
        if not self.restart_func:
            logging.warning(f"No restart function available for {self.service_name}")
            return False
            
        # Check restart rate limits
        current_time = time.time()
        recent_restarts = [
            timestamp for timestamp in self.restart_timestamps
            if current_time - timestamp < self.restart_window
        ]
        
        if len(recent_restarts) >= self.max_restarts:
            logging.error(f"Restart limit exceeded for {self.service_name}")
            return False
            
        try:
            logging.info(f"Attempting to restart {self.service_name}")
            
            if asyncio.iscoroutinefunction(self.restart_func):
                success = await self.restart_func()
            else:
                success = self.restart_func()
                
            if success:
                self.restart_timestamps.append(current_time)
                self.restart_count += 1
                
                # Wait for service to stabilize
                await asyncio.sleep(10)
                
                # Verify health after restart
                return await self.check_health()
            else:
                return False
                
        except Exception as e:
            logging.error(f"Self-healing failed for {self.service_name}: {e}")
            return False
            
    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            'service_name': self.service_name,
            'is_healthy': self.is_healthy,
            'last_health_check': self.last_health_check,
            'restart_count': self.restart_count,
            'recent_restarts': len(self.restart_timestamps)
        }

class IntelligentRecoveryOrchestrator:
    """Main error recovery orchestration system."""
    
    def __init__(self):
        self.predictive_analyzer = PredictiveAnalyzer()
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.self_healing_services: Dict[str, SelfHealingService] = {}
        self.error_events: List[ErrorEvent] = []
        self.recovery_history: List[Dict] = []
        self.setup_default_strategies()
        
    def setup_default_strategies(self):
        """Setup default recovery strategies."""
        strategies = [
            RecoveryStrategy(
                name="transient_error_retry",
                condition=lambda error: error.pattern == ErrorPattern.TRANSIENT,
                actions=[RecoveryAction.RETRY],
                max_retries=3,
                backoff_multiplier=2.0
            ),
            RecoveryStrategy(
                name="dependency_failure_fallback",
                condition=lambda error: error.pattern == ErrorPattern.DEPENDENCY_FAILURE,
                actions=[RecoveryAction.FALLBACK, RecoveryAction.CIRCUIT_BREAKER],
                max_retries=1
            ),
            RecoveryStrategy(
                name="resource_exhaustion_scale",
                condition=lambda error: error.pattern == ErrorPattern.RESOURCE_EXHAUSTION,
                actions=[RecoveryAction.SCALE_UP, RecoveryAction.GRACEFUL_DEGRADATION],
                max_retries=2
            ),
            RecoveryStrategy(
                name="critical_error_intervention",
                condition=lambda error: error.severity == ErrorSeverity.CRITICAL,
                actions=[RecoveryAction.RESTART_SERVICE, RecoveryAction.MANUAL_INTERVENTION],
                max_retries=1
            ),
            RecoveryStrategy(
                name="timeout_circuit_breaker",
                condition=lambda error: error.pattern == ErrorPattern.TIMEOUT,
                actions=[RecoveryAction.CIRCUIT_BREAKER, RecoveryAction.RETRY],
                max_retries=2
            )
        ]
        
        for strategy in strategies:
            self.recovery_strategies[strategy.name] = strategy
            
    def register_service(self, service_name: str, health_check_func: Callable,
                        restart_func: Optional[Callable] = None):
        """Register a service for self-healing."""
        self.self_healing_services[service_name] = SelfHealingService(
            service_name, health_check_func, restart_func
        )
        self.circuit_breakers[service_name] = CircuitBreaker()
        
    async def handle_error(self, error: ErrorEvent) -> Dict[str, Any]:
        """Handle error with intelligent recovery."""
        # Classify error pattern
        error.pattern = self._classify_error_pattern(error)
        
        # Add to history
        self.error_events.append(error)
        self.predictive_analyzer.add_error_event(error)
        
        # Find matching recovery strategy
        recovery_strategy = self._find_recovery_strategy(error)
        if not recovery_strategy:
            logging.warning(f"No recovery strategy found for error: {error.error_type}")
            return {'recovered': False, 'reason': 'no_strategy'}
            
        # Execute recovery actions
        recovery_result = await self._execute_recovery(error, recovery_strategy)
        
        # Log recovery attempt
        self.recovery_history.append({
            'error_id': error.id,
            'timestamp': time.time(),
            'strategy': recovery_strategy.name,
            'actions': [action.value for action in recovery_strategy.actions],
            'success': recovery_result['recovered']
        })
        
        return recovery_result
        
    def _classify_error_pattern(self, error: ErrorEvent) -> ErrorPattern:
        """Classify error into a pattern."""
        error_message = error.error_message.lower()
        
        # Pattern classification based on error message and context
        if any(keyword in error_message for keyword in ['timeout', 'timed out']):
            return ErrorPattern.TIMEOUT
        elif any(keyword in error_message for keyword in ['connection', 'network', 'unreachable']):
            return ErrorPattern.DEPENDENCY_FAILURE
        elif any(keyword in error_message for keyword in ['memory', 'cpu', 'disk', 'resource']):
            return ErrorPattern.RESOURCE_EXHAUSTION
        elif any(keyword in error_message for keyword in ['corrupt', 'invalid', 'malformed']):
            return ErrorPattern.DATA_CORRUPTION
        elif error.context.get('retry_count', 0) > 0:
            return ErrorPattern.TRANSIENT
        else:
            # Check for cascading failures
            recent_errors = [
                e for e in self.error_events
                if e.timestamp > time.time() - 300 and e.service_name == error.service_name
            ]
            if len(recent_errors) > 5:
                return ErrorPattern.CASCADING
                
        return ErrorPattern.TRANSIENT  # Default
        
    def _find_recovery_strategy(self, error: ErrorEvent) -> Optional[RecoveryStrategy]:
        """Find appropriate recovery strategy for error."""
        for strategy in self.recovery_strategies.values():
            if strategy.condition(error):
                return strategy
        return None
        
    async def _execute_recovery(self, error: ErrorEvent, 
                              strategy: RecoveryStrategy) -> Dict[str, Any]:
        """Execute recovery actions."""
        recovery_result = {
            'recovered': False,
            'actions_taken': [],
            'retry_count': 0,
            'final_action': None
        }
        
        for action in strategy.actions:
            try:
                success = await self._execute_action(error, action, strategy)
                recovery_result['actions_taken'].append(action.value)
                
                if success:
                    recovery_result['recovered'] = True
                    recovery_result['final_action'] = action.value
                    error.resolved = True
                    error.recovery_actions = strategy.actions
                    break
                    
            except Exception as e:
                logging.error(f"Recovery action {action.value} failed: {e}")
                
        return recovery_result
        
    async def _execute_action(self, error: ErrorEvent, action: RecoveryAction,
                            strategy: RecoveryStrategy) -> bool:
        """Execute specific recovery action."""
        service_name = error.service_name
        
        if action == RecoveryAction.RETRY:
            return await self._retry_operation(error, strategy)
        elif action == RecoveryAction.FALLBACK:
            return await self._execute_fallback(error)
        elif action == RecoveryAction.RESTART_SERVICE:
            return await self._restart_service(service_name)
        elif action == RecoveryAction.SCALE_UP:
            return await self._scale_up_service(service_name)
        elif action == RecoveryAction.CIRCUIT_BREAKER:
            return await self._activate_circuit_breaker(service_name)
        elif action == RecoveryAction.GRACEFUL_DEGRADATION:
            return await self._graceful_degradation(service_name)
        elif action == RecoveryAction.MANUAL_INTERVENTION:
            return await self._request_manual_intervention(error)
            
        return False
        
    async def _retry_operation(self, error: ErrorEvent, strategy: RecoveryStrategy) -> bool:
        """Retry the failed operation with exponential backoff."""
        retry_count = error.context.get('retry_count', 0)
        
        if retry_count >= strategy.max_retries:
            return False
            
        # Exponential backoff
        delay = min(strategy.backoff_multiplier ** retry_count, 60)
        await asyncio.sleep(delay)
        
        # Update retry count
        error.context['retry_count'] = retry_count + 1
        
        # In production, would re-execute the original operation
        logging.info(f"Retrying operation for error {error.id}, attempt {retry_count + 1}")
        
        # Simulate retry success/failure
        import random
        return random.random() > 0.3  # 70% success rate
        
    async def _execute_fallback(self, error: ErrorEvent) -> bool:
        """Execute fallback operation."""
        logging.info(f"Executing fallback for {error.service_name}")
        
        # In production, would switch to backup service or cached data
        await asyncio.sleep(1)  # Simulate fallback execution
        return True
        
    async def _restart_service(self, service_name: str) -> bool:
        """Restart service."""
        if service_name in self.self_healing_services:
            return await self.self_healing_services[service_name].attempt_self_heal()
        else:
            logging.info(f"Mock restart for service: {service_name}")
            await asyncio.sleep(2)
            return True
            
    async def _scale_up_service(self, service_name: str) -> bool:
        """Scale up service resources."""
        logging.info(f"Scaling up service: {service_name}")
        
        # In production, would integrate with Kubernetes HPA or similar
        await asyncio.sleep(3)  # Simulate scaling
        return True
        
    async def _activate_circuit_breaker(self, service_name: str) -> bool:
        """Activate circuit breaker for service."""
        if service_name in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[service_name]
            circuit_breaker.state = "open"
            logging.info(f"Circuit breaker activated for {service_name}")
            return True
        return False
        
    async def _graceful_degradation(self, service_name: str) -> bool:
        """Enable graceful degradation mode."""
        logging.info(f"Graceful degradation activated for {service_name}")
        
        # In production, would reduce service functionality
        await asyncio.sleep(1)
        return True
        
    async def _request_manual_intervention(self, error: ErrorEvent) -> bool:
        """Request manual intervention."""
        alert_data = {
            'error_id': error.id,
            'service': error.service_name,
            'severity': error.severity.value,
            'message': error.error_message,
            'timestamp': error.timestamp
        }
        
        logging.critical(f"Manual intervention required: {json.dumps(alert_data)}")
        
        # In production, would send alerts to on-call engineers
        return False  # Manual intervention required
        
    async def predictive_maintenance(self):
        """Proactive maintenance based on predictions."""
        while True:
            try:
                for service_name in self.self_healing_services.keys():
                    # Predict failure probability
                    prediction = self.predictive_analyzer.predict_failure_probability(service_name)
                    
                    if prediction['probability'] > 0.7 and prediction['confidence'] > 0.6:
                        logging.warning(f"High failure probability predicted for {service_name}: {prediction}")
                        
                        # Take preemptive action
                        await self._preemptive_action(service_name, prediction)
                        
                    # Check for anomalies
                    anomaly = self.predictive_analyzer.detect_anomalies(service_name)
                    if anomaly['anomalous']:
                        logging.warning(f"Anomaly detected in {service_name}: {anomaly}")
                        
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logging.error(f"Predictive maintenance error: {e}")
                await asyncio.sleep(600)  # Wait longer on error
                
    async def _preemptive_action(self, service_name: str, prediction: Dict):
        """Take preemptive action based on failure prediction."""
        primary_patterns = prediction.get('primary_patterns', [])
        
        if 'resource_exhaustion' in primary_patterns:
            await self._scale_up_service(service_name)
        elif 'dependency_failure' in primary_patterns:
            await self._activate_circuit_breaker(service_name)
        else:
            # Generic preemptive action
            await self._graceful_degradation(service_name)
            
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        service_status = {}
        for name, service in self.self_healing_services.items():
            service_status[name] = service.get_status()
            
        circuit_breaker_status = {}
        for name, cb in self.circuit_breakers.items():
            circuit_breaker_status[name] = cb.get_state()
            
        recent_errors = [
            error for error in self.error_events
            if error.timestamp > time.time() - 3600  # Last hour
        ]
        
        return {
            'timestamp': time.time(),
            'services': service_status,
            'circuit_breakers': circuit_breaker_status,
            'recent_errors': len(recent_errors),
            'unresolved_errors': len([e for e in recent_errors if not e.resolved]),
            'recovery_success_rate': self._calculate_recovery_success_rate(),
            'error_trends': self._analyze_error_trends()
        }
        
    def _calculate_recovery_success_rate(self) -> float:
        """Calculate recovery success rate."""
        if not self.recovery_history:
            return 1.0
            
        recent_recoveries = [
            r for r in self.recovery_history
            if r['timestamp'] > time.time() - 3600  # Last hour
        ]
        
        if not recent_recoveries:
            return 1.0
            
        success_count = sum(1 for r in recent_recoveries if r['success'])
        return success_count / len(recent_recoveries)
        
    def _analyze_error_trends(self) -> Dict[str, Any]:
        """Analyze error trends."""
        recent_errors = [
            error for error in self.error_events
            if error.timestamp > time.time() - 3600
        ]
        
        if not recent_errors:
            return {'trend': 'stable', 'error_rate': 0.0}
            
        # Calculate error rate trend
        hour_ago = time.time() - 3600
        half_hour_ago = time.time() - 1800
        
        first_half_errors = len([
            e for e in recent_errors 
            if hour_ago <= e.timestamp < half_hour_ago
        ])
        second_half_errors = len([
            e for e in recent_errors 
            if e.timestamp >= half_hour_ago
        ])
        
        if first_half_errors == 0:
            trend = 'increasing' if second_half_errors > 0 else 'stable'
        else:
            ratio = second_half_errors / first_half_errors
            if ratio > 1.5:
                trend = 'increasing'
            elif ratio < 0.5:
                trend = 'decreasing'
            else:
                trend = 'stable'
                
        return {
            'trend': trend,
            'error_rate': len(recent_errors) / 60,  # errors per minute
            'first_half_errors': first_half_errors,
            'second_half_errors': second_half_errors
        }

async def main():
    """Main entry point for testing."""
    recovery_system = IntelligentRecoveryOrchestrator()
    
    # Register a test service
    async def health_check():
        return True
        
    async def restart_service():
        return True
        
    recovery_system.register_service(
        "pneumonia-detector", 
        health_check, 
        restart_service
    )
    
    print("Intelligent Error Recovery System initialized")
    
    # Simulate an error
    import secrets
    test_error = ErrorEvent(
        id=secrets.token_urlsafe(16),
        timestamp=time.time(),
        service_name="pneumonia-detector",
        error_type="ConnectionError",
        error_message="Connection timeout to database",
        severity=ErrorSeverity.HIGH,
        context={'retry_count': 0}
    )
    
    recovery_result = await recovery_system.handle_error(test_error)
    print(f"Recovery result: {recovery_result}")
    
    # Start predictive maintenance
    await recovery_system.predictive_maintenance()

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Error recovery system stopped")