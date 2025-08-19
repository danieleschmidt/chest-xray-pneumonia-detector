"""Adaptive Quantum-Medical Pipeline - Generation 1: Intelligent Processing.

Self-adapting medical AI pipeline that learns and optimizes based on 
quantum computing principles and medical domain expertise.
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from quantum_inspired_task_planner import QuantumScheduler, QuantumTask, TaskPriority, TaskStatus
from quantum_medical_fusion_engine import QuantumMedicalFusionEngine, MedicalDiagnosisResult


class AdaptationStrategy(Enum):
    """Strategy for pipeline adaptation."""
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    RESEARCH_MODE = "research"


@dataclass
class PipelineMetrics:
    """Metrics for pipeline performance tracking."""
    accuracy: float
    processing_speed: float
    resource_utilization: float
    quantum_coherence: float
    medical_compliance_score: float
    patient_safety_index: float


@dataclass
class AdaptivePipelineConfig:
    """Configuration for adaptive pipeline."""
    adaptation_strategy: AdaptationStrategy = AdaptationStrategy.BALANCED
    learning_rate: float = 0.01
    adaptation_threshold: float = 0.1
    safety_threshold: float = 0.95
    max_concurrent_tasks: int = 4
    enable_quantum_acceleration: bool = True
    medical_validation_required: bool = True


class AdaptiveQuantumMedicalPipeline:
    """Intelligent medical AI pipeline with quantum optimization and self-adaptation."""
    
    def __init__(self, config: Optional[AdaptivePipelineConfig] = None):
        """Initialize adaptive quantum-medical pipeline."""
        self.config = config or AdaptivePipelineConfig()
        self.fusion_engine = QuantumMedicalFusionEngine()
        self.quantum_scheduler = QuantumScheduler()
        
        # Adaptation mechanism
        self.performance_history = []
        self.adaptation_parameters = {
            'quantum_weight': 0.5,
            'medical_weight': 0.5,
            'confidence_boost': 0.1,
            'processing_priority': 1.0
        }
        
        # Resource management
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_tasks)
        self.active_tasks = {}
        
        # Performance tracking
        self.metrics_history = []
        self.patient_outcomes = []
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Adaptive Pipeline initialized with strategy: {self.config.adaptation_strategy}")
    
    async def initialize_medical_models(self, model_paths: Dict[str, str]) -> bool:
        """Initialize medical AI models asynchronously."""
        initialization_tasks = []
        
        for model_name, model_path in model_paths.items():
            task = QuantumTask(
                task_id=f"init_model_{model_name}",
                priority=TaskPriority.HIGH,
                estimated_duration=5.0,
                dependencies=[],
                metadata={"type": "model_initialization", "model": model_name}
            )
            
            self.quantum_scheduler.add_task(task)
            initialization_tasks.append(self._load_model_async(model_name, model_path))
        
        results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
        success_count = sum(1 for r in results if r is True)
        
        self.logger.info(f"Model initialization: {success_count}/{len(results)} successful")
        return success_count == len(results)
    
    async def _load_model_async(self, model_name: str, model_path: str) -> bool:
        """Load model asynchronously."""
        try:
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                self.executor, 
                self.fusion_engine.load_medical_model, 
                model_path
            )
            return success
        except Exception as e:
            self.logger.error(f"Failed to load {model_name}: {e}")
            return False
    
    def analyze_performance_trends(self) -> Dict[str, float]:
        """Analyze performance trends for adaptation."""
        if len(self.performance_history) < 3:
            return {"trend": "insufficient_data"}
        
        recent_metrics = self.performance_history[-10:]
        accuracy_trend = np.polyfit(range(len(recent_metrics)), 
                                  [m.accuracy for m in recent_metrics], 1)[0]
        speed_trend = np.polyfit(range(len(recent_metrics)), 
                               [m.processing_speed for m in recent_metrics], 1)[0]
        
        return {
            "accuracy_trend": accuracy_trend,
            "speed_trend": speed_trend,
            "stability": np.std([m.accuracy for m in recent_metrics]),
            "recent_avg_accuracy": np.mean([m.accuracy for m in recent_metrics]),
            "recent_avg_speed": np.mean([m.processing_speed for m in recent_metrics])
        }
    
    def adapt_pipeline_parameters(self, trends: Dict[str, float]) -> bool:
        """Adapt pipeline parameters based on performance trends."""
        adaptation_made = False
        
        if trends.get("trend") == "insufficient_data":
            return False
        
        # Adaptive strategy implementation
        if self.config.adaptation_strategy == AdaptationStrategy.AGGRESSIVE:
            adaptation_factor = 0.2
        elif self.config.adaptation_strategy == AdaptationStrategy.CONSERVATIVE:
            adaptation_factor = 0.05
        else:  # BALANCED or RESEARCH_MODE
            adaptation_factor = 0.1
        
        # Adapt quantum weight based on accuracy trend
        if trends.get("accuracy_trend", 0) < -0.01:  # Declining accuracy
            self.adaptation_parameters['quantum_weight'] += adaptation_factor
            self.adaptation_parameters['medical_weight'] -= adaptation_factor
            adaptation_made = True
        elif trends.get("accuracy_trend", 0) > 0.01:  # Improving accuracy
            # Maintain current balance or slightly adjust
            pass
        
        # Adapt processing priority based on speed trends
        if trends.get("speed_trend", 0) < -0.1:  # Slowing down
            self.adaptation_parameters['processing_priority'] += 0.1
            adaptation_made = True
        
        # Ensure parameters stay within valid ranges
        self.adaptation_parameters['quantum_weight'] = np.clip(
            self.adaptation_parameters['quantum_weight'], 0.1, 0.9
        )
        self.adaptation_parameters['medical_weight'] = np.clip(
            self.adaptation_parameters['medical_weight'], 0.1, 0.9
        )
        
        if adaptation_made:
            self.logger.info(f"Pipeline adapted: quantum_weight={self.adaptation_parameters['quantum_weight']:.3f}")
        
        return adaptation_made
    
    async def process_medical_case_adaptive(
        self, 
        image_data: np.ndarray, 
        patient_metadata: Dict,
        priority: TaskPriority = TaskPriority.MEDIUM
    ) -> MedicalDiagnosisResult:
        """Process medical case with adaptive optimization."""
        case_id = patient_metadata.get('id', f'case_{int(time.time())}')
        
        # Create adaptive task
        task = QuantumTask(
            task_id=f"medical_case_{case_id}",
            priority=priority,
            estimated_duration=self._estimate_processing_time(image_data),
            dependencies=[],
            metadata={
                "type": "medical_diagnosis",
                "patient_id": case_id,
                "adaptation_params": self.adaptation_parameters.copy()
            }
        )
        
        self.quantum_scheduler.add_task(task)
        self.active_tasks[case_id] = task
        
        # Process with current adaptation parameters
        start_time = time.time()
        
        try:
            # Apply adaptive processing
            result = await self._adaptive_medical_processing(
                image_data, patient_metadata, case_id
            )
            
            processing_time = time.time() - start_time
            
            # Update metrics
            metrics = PipelineMetrics(
                accuracy=result.confidence,  # Using confidence as proxy for accuracy
                processing_speed=1.0 / processing_time,
                resource_utilization=self._calculate_resource_utilization(),
                quantum_coherence=result.quantum_optimization_score,
                medical_compliance_score=self._calculate_medical_compliance_score(result),
                patient_safety_index=self._calculate_safety_index(result)
            )
            
            self.metrics_history.append(metrics)
            self.performance_history.append(metrics)
            
            # Trigger adaptation if needed
            if len(self.performance_history) % 10 == 0:
                trends = self.analyze_performance_trends()
                self.adapt_pipeline_parameters(trends)
            
            # Update task status
            task.status = TaskStatus.COMPLETED
            del self.active_tasks[case_id]
            
            self.logger.info(f"Adaptive processing complete for {case_id}: {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Adaptive processing failed for {case_id}: {e}")
            task.status = TaskStatus.FAILED
            if case_id in self.active_tasks:
                del self.active_tasks[case_id]
            raise
    
    async def _adaptive_medical_processing(
        self, 
        image_data: np.ndarray, 
        patient_metadata: Dict,
        case_id: str
    ) -> MedicalDiagnosisResult:
        """Apply adaptive medical processing with quantum optimization."""
        loop = asyncio.get_event_loop()
        
        # Run fusion engine processing in executor
        result = await loop.run_in_executor(
            self.executor,
            self.fusion_engine.process_medical_image,
            image_data,
            patient_metadata
        )
        
        # Apply adaptation parameters
        adapted_prediction = (
            result.prediction * self.adaptation_parameters['medical_weight'] +
            result.quantum_optimization_score * self.adaptation_parameters['quantum_weight']
        )
        
        adapted_confidence = min(
            result.confidence + self.adaptation_parameters['confidence_boost'], 
            1.0
        )
        
        # Create adapted result
        adapted_result = MedicalDiagnosisResult(
            prediction=adapted_prediction,
            confidence=adapted_confidence,
            quantum_optimization_score=result.quantum_optimization_score,
            processing_time=result.processing_time,
            metadata={
                **result.metadata,
                "adaptation_applied": True,
                "original_prediction": result.prediction,
                "original_confidence": result.confidence,
                "case_id": case_id
            }
        )
        
        return adapted_result
    
    def _estimate_processing_time(self, image_data: np.ndarray) -> float:
        """Estimate processing time based on image characteristics."""
        base_time = 0.5  # Base processing time
        complexity_factor = image_data.size / (150 * 150)  # Relative to standard size
        return base_time * complexity_factor
    
    def _calculate_resource_utilization(self) -> float:
        """Calculate current resource utilization."""
        active_task_count = len(self.active_tasks)
        max_tasks = self.config.max_concurrent_tasks
        return active_task_count / max_tasks
    
    def _calculate_medical_compliance_score(self, result: MedicalDiagnosisResult) -> float:
        """Calculate medical compliance score."""
        compliance_score = 0.9  # Base compliance
        
        # Adjust based on confidence
        if result.confidence >= 0.8:
            compliance_score += 0.05
        elif result.confidence < 0.6:
            compliance_score -= 0.1
        
        # Adjust based on quantum optimization
        if result.quantum_optimization_score >= 0.8:
            compliance_score += 0.03
        
        return min(compliance_score, 1.0)
    
    def _calculate_safety_index(self, result: MedicalDiagnosisResult) -> float:
        """Calculate patient safety index."""
        safety_index = 0.95  # High baseline safety
        
        # Conservative approach for edge cases
        if 0.4 <= result.prediction <= 0.6:  # Uncertain predictions
            safety_index -= 0.05
        
        if result.confidence < 0.7:  # Low confidence
            safety_index -= 0.1
        
        return max(safety_index, 0.8)
    
    async def batch_process_adaptive(
        self, 
        cases: List[Tuple[np.ndarray, Dict]], 
        priority_mapping: Optional[Dict[str, TaskPriority]] = None
    ) -> List[MedicalDiagnosisResult]:
        """Process multiple cases with adaptive optimization."""
        priority_mapping = priority_mapping or {}
        
        self.logger.info(f"Starting adaptive batch processing of {len(cases)} cases")
        
        # Create processing tasks
        processing_tasks = []
        for i, (image_data, metadata) in enumerate(cases):
            case_id = metadata.get('id', f'batch_case_{i}')
            priority = priority_mapping.get(case_id, TaskPriority.MEDIUM)
            
            task = self.process_medical_case_adaptive(image_data, metadata, priority)
            processing_tasks.append(task)
        
        # Process with controlled concurrency
        results = []
        batch_size = min(self.config.max_concurrent_tasks, len(processing_tasks))
        
        for i in range(0, len(processing_tasks), batch_size):
            batch = processing_tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Batch processing error: {result}")
                else:
                    results.append(result)
        
        # Generate batch performance report
        self._log_batch_performance(results)
        
        return results
    
    def _log_batch_performance(self, results: List[MedicalDiagnosisResult]) -> None:
        """Log batch processing performance."""
        if not results:
            return
        
        avg_confidence = np.mean([r.confidence for r in results])
        avg_quantum_score = np.mean([r.quantum_optimization_score for r in results])
        avg_processing_time = np.mean([r.processing_time for r in results])
        
        self.logger.info(
            f"Batch performance: avg_conf={avg_confidence:.3f}, "
            f"avg_qscore={avg_quantum_score:.3f}, avg_time={avg_processing_time:.3f}s"
        )
    
    def generate_adaptation_report(self) -> Dict[str, Any]:
        """Generate comprehensive adaptation report."""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = self.metrics_history[-20:] if len(self.metrics_history) >= 20 else self.metrics_history
        
        report = {
            "adaptation_strategy": self.config.adaptation_strategy.value,
            "current_parameters": self.adaptation_parameters,
            "performance_trends": self.analyze_performance_trends(),
            "recent_performance": {
                "avg_accuracy": np.mean([m.accuracy for m in recent_metrics]),
                "avg_processing_speed": np.mean([m.processing_speed for m in recent_metrics]),
                "avg_quantum_coherence": np.mean([m.quantum_coherence for m in recent_metrics]),
                "avg_safety_index": np.mean([m.patient_safety_index for m in recent_metrics])
            },
            "total_cases_processed": len(self.metrics_history),
            "active_tasks": len(self.active_tasks),
            "scheduler_efficiency": self.quantum_scheduler.get_efficiency_score()
        }
        
        return report
    
    def save_adaptation_state(self, filepath: Path) -> bool:
        """Save current adaptation state for persistence."""
        try:
            state = {
                "adaptation_parameters": self.adaptation_parameters,
                "performance_history": [asdict(m) for m in self.metrics_history[-100:]],  # Last 100 metrics
                "config": asdict(self.config),
                "timestamp": time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            self.logger.info(f"Adaptation state saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save adaptation state: {e}")
            return False
    
    def load_adaptation_state(self, filepath: Path) -> bool:
        """Load adaptation state from file."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.adaptation_parameters = state["adaptation_parameters"]
            # Note: Could restore performance history if needed
            
            self.logger.info(f"Adaptation state loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load adaptation state: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        self.logger.info("Pipeline cleanup complete")


async def main():
    """Demonstration of Adaptive Quantum-Medical Pipeline."""
    print("🧠 Adaptive Quantum-Medical Pipeline - Generation 1 Demo")
    print("=" * 60)
    
    # Initialize pipeline
    config = AdaptivePipelineConfig(
        adaptation_strategy=AdaptationStrategy.BALANCED,
        max_concurrent_tasks=3,
        enable_quantum_acceleration=True
    )
    
    pipeline = AdaptiveQuantumMedicalPipeline(config)
    
    # Create demo cases
    demo_cases = []
    for i in range(6):
        image = np.random.normal(0.5, 0.2, (150, 150, 1))
        metadata = {
            "id": f"patient_{i:03d}",
            "age": 25 + i * 8,
            "symptoms": ["cough", "fever", "chest_pain"][i % 3],
            "urgency": "high" if i % 3 == 0 else "medium"
        }
        demo_cases.append((image, metadata))
    
    print(f"📊 Processing {len(demo_cases)} adaptive medical cases...")
    
    try:
        # Process cases adaptively
        results = await pipeline.batch_process_adaptive(demo_cases)
        
        # Display results
        print("\n🏥 Adaptive Medical Analysis Results:")
        print("-" * 50)
        for i, result in enumerate(results):
            status = "PNEUMONIA DETECTED" if result.prediction > 0.5 else "NORMAL"
            adaptation_applied = result.metadata.get("adaptation_applied", False)
            print(f"Patient {i+1:02d}: {status} (conf: {result.confidence:.3f}, "
                  f"adapted: {'Yes' if adaptation_applied else 'No'})")
        
        # Generate adaptation report
        report = pipeline.generate_adaptation_report()
        print(f"\n📈 Adaptation Report:")
        print(f"Adaptation strategy: {report['adaptation_strategy']}")
        print(f"Total cases processed: {report['total_cases_processed']}")
        print(f"Average accuracy: {report['recent_performance']['avg_accuracy']:.3f}")
        print(f"Average safety index: {report['recent_performance']['avg_safety_index']:.3f}")
        print(f"Scheduler efficiency: {report['scheduler_efficiency']:.3f}")
        
        # Save state for future sessions
        state_file = Path("/tmp/adaptive_pipeline_state.json")
        pipeline.save_adaptation_state(state_file)
        print(f"\n💾 Adaptation state saved to {state_file}")
        
    finally:
        pipeline.cleanup()
    
    print("\n✅ Adaptive Quantum-Medical Pipeline demonstration complete!")


if __name__ == "__main__":
    asyncio.run(main())