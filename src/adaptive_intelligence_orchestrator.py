"""
Adaptive Intelligence Orchestrator
Coordinates Gen4 Neural-Quantum systems with self-healing and auto-scaling
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import json

try:
    from .gen4_neural_quantum_fusion import (
        Gen4NeuralQuantumFusion, 
        MedicalIntelligenceContext,
        AIIntelligenceLevel
    )
except ImportError:
    from gen4_neural_quantum_fusion import (
        Gen4NeuralQuantumFusion, 
        MedicalIntelligenceContext,
        AIIntelligenceLevel
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemState(Enum):
    """System operational states."""
    OPTIMAL = "optimal"
    DEGRADED = "degraded" 
    CRITICAL = "critical"
    RECOVERY = "recovery"
    MAINTENANCE = "maintenance"

class ScalingTrigger(Enum):
    """Auto-scaling triggers."""
    LOAD_HIGH = "load_high"
    LOAD_LOW = "load_low"
    ERROR_RATE_HIGH = "error_rate_high"
    QUANTUM_COHERENCE_LOW = "quantum_coherence_low"
    PERFORMANCE_DEGRADED = "performance_degraded"

@dataclass
class SystemMetrics:
    """System performance and health metrics."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    request_rate: float = 0.0
    error_rate: float = 0.0
    avg_response_time: float = 0.0
    quantum_coherence_avg: float = 0.85
    active_instances: int = 1
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ScalingEvent:
    """Represents an auto-scaling event."""
    trigger: ScalingTrigger
    current_instances: int
    target_instances: int
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)

class AdaptiveIntelligenceOrchestrator:
    """
    Orchestrates multiple Gen4 Neural-Quantum Fusion instances with:
    - Auto-scaling based on load and performance
    - Self-healing capabilities
    - Load balancing and circuit breaking
    - Continuous performance optimization
    """
    
    def __init__(self, 
                 min_instances: int = 1,
                 max_instances: int = 5,
                 target_cpu_usage: float = 0.7,
                 error_rate_threshold: float = 0.05):
        """Initialize the Adaptive Intelligence Orchestrator."""
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.target_cpu_usage = target_cpu_usage
        self.error_rate_threshold = error_rate_threshold
        
        # Instance management
        self.fusion_instances: Dict[str, Gen4NeuralQuantumFusion] = {}
        self.instance_metrics: Dict[str, SystemMetrics] = {}
        self.load_balancer_state = {"current_index": 0, "weights": {}}
        
        # System state tracking
        self.system_state = SystemState.OPTIMAL
        self.scaling_history: List[ScalingEvent] = []
        self.performance_history: List[SystemMetrics] = []
        
        # Circuit breaker state
        self.circuit_breaker = {
            "state": "closed",  # closed, open, half-open
            "failure_count": 0,
            "last_failure_time": None,
            "timeout": 30.0  # seconds
        }
        
        # Auto-healing configuration
        self.healing_enabled = True
        self.healing_thresholds = {
            "quantum_coherence_min": 0.6,
            "error_rate_max": 0.1,
            "response_time_max": 5.0
        }
        
        logger.info("Adaptive Intelligence Orchestrator initialized")
    
    async def initialize(self):
        """Initialize the orchestrator with minimum instances."""
        logger.info(f"Initializing {self.min_instances} fusion instances...")
        
        for i in range(self.min_instances):
            await self._create_fusion_instance(f"fusion_{i:03d}")
        
        # Start background monitoring and scaling tasks
        asyncio.create_task(self._monitoring_loop())
        asyncio.create_task(self._auto_scaling_loop())
        asyncio.create_task(self._self_healing_loop())
        
        logger.info("Orchestrator initialization complete")
    
    async def _create_fusion_instance(self, instance_id: str) -> bool:
        """Create and initialize a new fusion instance."""
        try:
            fusion_instance = Gen4NeuralQuantumFusion(
                AIIntelligenceLevel.NEURAL_QUANTUM_FUSION
            )
            
            # Initialize quantum coherence
            await fusion_instance.initialize_quantum_coherence()
            
            self.fusion_instances[instance_id] = fusion_instance
            self.instance_metrics[instance_id] = SystemMetrics()
            
            # Initialize load balancer weight
            self.load_balancer_state["weights"][instance_id] = 1.0
            
            logger.info(f"Created fusion instance: {instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create fusion instance {instance_id}: {e}")
            return False
    
    async def _destroy_fusion_instance(self, instance_id: str) -> bool:
        """Safely destroy a fusion instance."""
        try:
            if instance_id in self.fusion_instances:
                del self.fusion_instances[instance_id]
                del self.instance_metrics[instance_id]
                del self.load_balancer_state["weights"][instance_id]
                
                logger.info(f"Destroyed fusion instance: {instance_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to destroy fusion instance {instance_id}: {e}")
            return False
    
    async def process_medical_request(self, 
                                    image_data: np.ndarray,
                                    context: MedicalIntelligenceContext) -> Dict[str, Any]:
        """
        Process medical request with intelligent load balancing and circuit breaking.
        """
        start_time = datetime.now()
        
        # Check circuit breaker
        if not self._check_circuit_breaker():
            raise Exception("Circuit breaker is open - service temporarily unavailable")
        
        try:
            # Select best instance using weighted load balancing
            instance_id = self._select_best_instance()
            
            if not instance_id:
                raise Exception("No healthy instances available")
            
            fusion_instance = self.fusion_instances[instance_id]
            
            # Process request
            result = await fusion_instance.enhance_medical_prediction(image_data, context)
            
            # Update instance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            await self._update_instance_metrics(instance_id, processing_time, success=True)
            
            # Reset circuit breaker on success
            self.circuit_breaker["failure_count"] = 0
            
            # Add orchestration metadata
            result["orchestration"] = {
                "instance_id": instance_id,
                "processing_time": processing_time,
                "total_instances": len(self.fusion_instances),
                "system_state": self.system_state.value
            }
            
            return result
            
        except Exception as e:
            # Handle failure
            await self._update_instance_metrics(instance_id if 'instance_id' in locals() else "unknown", 
                                              0.0, success=False)
            self._handle_circuit_breaker_failure()
            
            logger.error(f"Request processing failed: {e}")
            raise
    
    def _select_best_instance(self) -> Optional[str]:
        """Select the best instance using weighted load balancing."""
        if not self.fusion_instances:
            return None
        
        # Calculate weights based on performance metrics
        weights = {}
        for instance_id, metrics in self.instance_metrics.items():
            if instance_id not in self.fusion_instances:
                continue
                
            # Base weight = 1.0, adjusted by performance factors
            weight = 1.0
            
            # Adjust for quantum coherence
            weight *= (metrics.quantum_coherence_avg + 0.5)
            
            # Adjust for error rate (lower error = higher weight)
            weight *= max(0.1, 1.0 - metrics.error_rate)
            
            # Adjust for response time (faster = higher weight)
            if metrics.avg_response_time > 0:
                weight *= max(0.1, 1.0 / (metrics.avg_response_time + 0.1))
            
            weights[instance_id] = max(0.1, weight)
        
        # Weighted random selection
        if weights:
            total_weight = sum(weights.values())
            if total_weight > 0:
                import random
                r = random.random() * total_weight
                cumulative = 0.0
                
                for instance_id, weight in weights.items():
                    cumulative += weight
                    if r <= cumulative:
                        return instance_id
        
        # Fallback to round-robin
        instance_ids = list(self.fusion_instances.keys())
        if instance_ids:
            self.load_balancer_state["current_index"] = (
                (self.load_balancer_state["current_index"] + 1) % len(instance_ids)
            )
            return instance_ids[self.load_balancer_state["current_index"]]
        
        return None
    
    def _check_circuit_breaker(self) -> bool:
        """Check circuit breaker state."""
        if self.circuit_breaker["state"] == "closed":
            return True
        elif self.circuit_breaker["state"] == "open":
            # Check if timeout has passed
            if (self.circuit_breaker["last_failure_time"] and
                datetime.now() - self.circuit_breaker["last_failure_time"] > 
                timedelta(seconds=self.circuit_breaker["timeout"])):
                self.circuit_breaker["state"] = "half-open"
                logger.info("Circuit breaker moved to half-open state")
            return False
        else:  # half-open
            return True
    
    def _handle_circuit_breaker_failure(self):
        """Handle circuit breaker failure."""
        self.circuit_breaker["failure_count"] += 1
        self.circuit_breaker["last_failure_time"] = datetime.now()
        
        if (self.circuit_breaker["failure_count"] >= 5 and 
            self.circuit_breaker["state"] != "open"):
            self.circuit_breaker["state"] = "open"
            logger.warning("Circuit breaker opened due to high failure rate")
    
    async def _update_instance_metrics(self, instance_id: str, processing_time: float, success: bool):
        """Update metrics for a specific instance."""
        if instance_id not in self.instance_metrics:
            return
        
        metrics = self.instance_metrics[instance_id]
        
        # Update response time (exponential moving average)
        alpha = 0.1
        if metrics.avg_response_time == 0:
            metrics.avg_response_time = processing_time
        else:
            metrics.avg_response_time = (
                alpha * processing_time + (1 - alpha) * metrics.avg_response_time
            )
        
        # Update error rate
        if success:
            metrics.error_rate = max(0, metrics.error_rate - 0.01)
        else:
            metrics.error_rate = min(1.0, metrics.error_rate + 0.05)
        
        # Update quantum coherence from instance
        if instance_id in self.fusion_instances:
            status = await self.fusion_instances[instance_id].get_system_status()
            metrics.quantum_coherence_avg = status["quantum_state"]["coherence"]
        
        metrics.timestamp = datetime.now()
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while True:
            try:
                await self._collect_system_metrics()
                await self._update_system_state()
                await asyncio.sleep(10)  # Monitor every 10 seconds
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)
    
    async def _collect_system_metrics(self):
        """Collect and aggregate system metrics."""
        if not self.instance_metrics:
            return
        
        # Aggregate metrics across all instances
        total_instances = len(self.instance_metrics)
        avg_error_rate = np.mean([m.error_rate for m in self.instance_metrics.values()])
        avg_response_time = np.mean([m.avg_response_time for m in self.instance_metrics.values()])
        avg_quantum_coherence = np.mean([m.quantum_coherence_avg for m in self.instance_metrics.values()])
        
        system_metrics = SystemMetrics(
            request_rate=0.0,  # Would be calculated from actual request logs
            error_rate=avg_error_rate,
            avg_response_time=avg_response_time,
            quantum_coherence_avg=avg_quantum_coherence,
            active_instances=total_instances
        )
        
        self.performance_history.append(system_metrics)
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    async def _update_system_state(self):
        """Update overall system state based on metrics."""
        if not self.performance_history:
            return
        
        recent_metrics = self.performance_history[-1]
        
        # Determine system state
        if (recent_metrics.error_rate < 0.01 and 
            recent_metrics.avg_response_time < 1.0 and
            recent_metrics.quantum_coherence_avg > 0.8):
            new_state = SystemState.OPTIMAL
        elif (recent_metrics.error_rate < 0.05 and 
              recent_metrics.avg_response_time < 3.0 and
              recent_metrics.quantum_coherence_avg > 0.6):
            new_state = SystemState.DEGRADED
        else:
            new_state = SystemState.CRITICAL
        
        if new_state != self.system_state:
            logger.info(f"System state changed: {self.system_state.value} -> {new_state.value}")
            self.system_state = new_state
    
    async def _auto_scaling_loop(self):
        """Auto-scaling decision loop."""
        while True:
            try:
                await self._evaluate_scaling_needs()
                await asyncio.sleep(30)  # Evaluate scaling every 30 seconds
            except Exception as e:
                logger.error(f"Auto-scaling loop error: {e}")
                await asyncio.sleep(15)
    
    async def _evaluate_scaling_needs(self):
        """Evaluate if scaling is needed."""
        if not self.performance_history:
            return
        
        recent_metrics = self.performance_history[-5:]  # Last 5 measurements
        avg_error_rate = np.mean([m.error_rate for m in recent_metrics])
        avg_response_time = np.mean([m.avg_response_time for m in recent_metrics])
        avg_quantum_coherence = np.mean([m.quantum_coherence_avg for m in recent_metrics])
        
        current_instances = len(self.fusion_instances)
        target_instances = current_instances
        scaling_reason = None
        
        # Scale up conditions
        if (avg_error_rate > 0.05 and current_instances < self.max_instances):
            target_instances = min(self.max_instances, current_instances + 1)
            scaling_reason = "High error rate"
        elif (avg_response_time > 3.0 and current_instances < self.max_instances):
            target_instances = min(self.max_instances, current_instances + 1)
            scaling_reason = "High response time"
        elif (avg_quantum_coherence < 0.6 and current_instances < self.max_instances):
            target_instances = min(self.max_instances, current_instances + 1)
            scaling_reason = "Low quantum coherence"
        
        # Scale down conditions
        elif (avg_error_rate < 0.01 and avg_response_time < 0.5 and 
              avg_quantum_coherence > 0.9 and current_instances > self.min_instances):
            target_instances = max(self.min_instances, current_instances - 1)
            scaling_reason = "Optimal performance with excess capacity"
        
        # Execute scaling if needed
        if target_instances != current_instances and scaling_reason:
            await self._execute_scaling(current_instances, target_instances, scaling_reason)
    
    async def _execute_scaling(self, current: int, target: int, reason: str):
        """Execute scaling operation."""
        scaling_event = ScalingEvent(
            trigger=ScalingTrigger.PERFORMANCE_DEGRADED if target > current else ScalingTrigger.LOAD_LOW,
            current_instances=current,
            target_instances=target,
            reason=reason
        )
        
        self.scaling_history.append(scaling_event)
        
        if target > current:
            # Scale up
            for i in range(target - current):
                new_instance_id = f"fusion_{len(self.fusion_instances):03d}"
                success = await self._create_fusion_instance(new_instance_id)
                if not success:
                    logger.error(f"Failed to scale up - could not create instance {new_instance_id}")
                    break
        else:
            # Scale down
            instances_to_remove = list(self.fusion_instances.keys())[:current - target]
            for instance_id in instances_to_remove:
                await self._destroy_fusion_instance(instance_id)
        
        logger.info(f"Scaling executed: {current} -> {len(self.fusion_instances)} instances. Reason: {reason}")
    
    async def _self_healing_loop(self):
        """Self-healing monitoring and execution loop."""
        while True:
            try:
                if self.healing_enabled:
                    await self._check_and_heal_instances()
                await asyncio.sleep(60)  # Check for healing every minute
            except Exception as e:
                logger.error(f"Self-healing loop error: {e}")
                await asyncio.sleep(30)
    
    async def _check_and_heal_instances(self):
        """Check instance health and perform healing if needed."""
        unhealthy_instances = []
        
        for instance_id, metrics in self.instance_metrics.items():
            if (metrics.quantum_coherence_avg < self.healing_thresholds["quantum_coherence_min"] or
                metrics.error_rate > self.healing_thresholds["error_rate_max"] or
                metrics.avg_response_time > self.healing_thresholds["response_time_max"]):
                unhealthy_instances.append(instance_id)
        
        # Heal unhealthy instances
        for instance_id in unhealthy_instances:
            await self._heal_instance(instance_id)
    
    async def _heal_instance(self, instance_id: str):
        """Heal a specific unhealthy instance."""
        logger.info(f"Healing instance: {instance_id}")
        
        try:
            # Try to evolve the instance intelligence first
            if instance_id in self.fusion_instances:
                fusion_instance = self.fusion_instances[instance_id]
                evolution_result = await fusion_instance.evolve_intelligence()
                
                if evolution_result["status"] == "evolved":
                    logger.info(f"Instance {instance_id} healed through evolution")
                    return
            
            # If evolution doesn't help, recreate the instance
            await self._destroy_fusion_instance(instance_id)
            success = await self._create_fusion_instance(instance_id)
            
            if success:
                logger.info(f"Instance {instance_id} healed through recreation")
            else:
                logger.error(f"Failed to heal instance {instance_id}")
                
        except Exception as e:
            logger.error(f"Healing failed for instance {instance_id}: {e}")
    
    async def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status."""
        instance_statuses = {}
        for instance_id, fusion_instance in self.fusion_instances.items():
            instance_statuses[instance_id] = await fusion_instance.get_system_status()
        
        recent_scaling = self.scaling_history[-5:] if self.scaling_history else []
        
        return {
            "system_state": self.system_state.value,
            "total_instances": len(self.fusion_instances),
            "instance_range": {"min": self.min_instances, "max": self.max_instances},
            "circuit_breaker": self.circuit_breaker,
            "auto_healing_enabled": self.healing_enabled,
            "recent_scaling_events": len(recent_scaling),
            "performance_summary": {
                "avg_quantum_coherence": np.mean([
                    status["quantum_state"]["coherence"] 
                    for status in instance_statuses.values()
                ]) if instance_statuses else 0.0,
                "total_learning_instances": sum([
                    status["learning_stats"]["total_instances"]
                    for status in instance_statuses.values()
                ]) if instance_statuses else 0
            },
            "timestamp": datetime.now().isoformat()
        }


# Factory function
def create_adaptive_intelligence_orchestrator(min_instances: int = 1, 
                                            max_instances: int = 5) -> AdaptiveIntelligenceOrchestrator:
    """Create and initialize an Adaptive Intelligence Orchestrator."""
    return AdaptiveIntelligenceOrchestrator(min_instances=min_instances, max_instances=max_instances)


if __name__ == "__main__":
    async def demo():
        """Demonstration of Adaptive Intelligence Orchestrator."""
        print("=== Adaptive Intelligence Orchestrator Demo ===")
        
        # Create and initialize orchestrator
        orchestrator = create_adaptive_intelligence_orchestrator(min_instances=2, max_instances=4)
        await orchestrator.initialize()
        
        # Show initial status
        status = await orchestrator.get_orchestrator_status()
        print(f"System State: {status['system_state']}")
        print(f"Active Instances: {status['total_instances']}")
        
        # Simulate some requests
        for i in range(3):
            dummy_image = np.random.rand(256, 256, 1)
            context = MedicalIntelligenceContext(
                patient_id=f"DEMO_{i:03d}",
                urgency_level=np.random.randint(1, 4)
            )
            
            result = await orchestrator.process_medical_request(dummy_image, context)
            print(f"Request {i+1}: {result['prediction']} (confidence: {result['confidence']:.3f})")
            
            # Small delay between requests
            await asyncio.sleep(0.5)
        
        # Final status
        final_status = await orchestrator.get_orchestrator_status()
        print(f"\nFinal Status:")
        print(f"System State: {final_status['system_state']}")
        print(f"Avg Quantum Coherence: {final_status['performance_summary']['avg_quantum_coherence']:.3f}")
        
        print("\n=== Demo Complete ===")
    
    # Run demo
    asyncio.run(demo())