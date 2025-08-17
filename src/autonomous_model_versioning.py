"""Autonomous Model Versioning and A/B Testing Framework.

Implements automated model versioning, deployment, and A/B testing
for medical AI systems with quantum-enhanced decision making.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import uuid

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Represents a versioned model with metadata."""
    version_id: str
    model_path: str
    performance_metrics: Dict[str, float]
    creation_timestamp: datetime
    training_parameters: Dict[str, Any]
    validation_results: Dict[str, Any]
    medical_compliance_score: float
    quantum_optimization_applied: bool
    parent_version_id: Optional[str] = None
    deployment_status: str = "staged"  # staged, active, retired
    a_b_test_allocation: float = 0.0  # Percentage of traffic


@dataclass
class ABTestResult:
    """Results of A/B testing between model versions."""
    test_id: str
    version_a: str
    version_b: str
    start_time: datetime
    end_time: Optional[datetime]
    sample_size_a: int
    sample_size_b: int
    performance_a: Dict[str, float]
    performance_b: Dict[str, float]
    statistical_significance: Dict[str, float]
    winner: Optional[str]
    confidence_level: float


class QuantumVersioningStrategy:
    """Quantum-inspired strategy for model versioning decisions."""
    
    def __init__(self, coherence_threshold: float = 0.8):
        self.coherence_threshold = coherence_threshold
        
    def should_create_version(self, current_metrics: Dict[str, float],
                            new_metrics: Dict[str, float],
                            training_params: Dict[str, Any]) -> Tuple[bool, float]:
        """Determine if new model warrants a version using quantum decision theory."""
        
        # Calculate quantum coherence between models
        coherence = self._calculate_model_coherence(current_metrics, new_metrics)
        
        # Quantum superposition of decision states
        improvement_score = self._calculate_improvement_score(current_metrics, new_metrics)
        stability_score = self._calculate_stability_score(training_params)
        medical_safety_score = self._calculate_medical_safety_score(new_metrics)
        
        # Quantum interference pattern for decision making
        decision_amplitude = (
            improvement_score * 0.4 +
            stability_score * 0.3 +
            medical_safety_score * 0.3
        )
        
        # Quantum measurement collapse to binary decision
        probability_threshold = 0.7
        should_version = decision_amplitude > probability_threshold and coherence < self.coherence_threshold
        
        return should_version, decision_amplitude
    
    def _calculate_model_coherence(self, metrics_a: Dict[str, float], 
                                 metrics_b: Dict[str, float]) -> float:
        """Calculate quantum coherence between model performance states."""
        common_keys = set(metrics_a.keys()) & set(metrics_b.keys())
        if not common_keys:
            return 1.0
        
        coherence_sum = 0.0
        for key in common_keys:
            # Phase difference between performance metrics
            phase_diff = abs(metrics_a[key] - metrics_b[key])
            coherence_sum += np.exp(-phase_diff)
        
        return coherence_sum / len(common_keys)
    
    def _calculate_improvement_score(self, old_metrics: Dict[str, float],
                                   new_metrics: Dict[str, float]) -> float:
        """Calculate improvement score with medical weighting."""
        improvements = []
        
        # Key medical metrics with higher weights
        medical_weights = {
            'sensitivity': 3.0,
            'specificity': 2.5,
            'auc_score': 2.0,
            'f1_score': 1.5,
            'precision': 1.0,
            'recall': 1.0
        }
        
        for metric in old_metrics:
            if metric in new_metrics:
                improvement = (new_metrics[metric] - old_metrics[metric]) / old_metrics[metric]
                weight = medical_weights.get(metric, 1.0)
                improvements.append(improvement * weight)
        
        return np.mean(improvements) if improvements else 0.0
    
    def _calculate_stability_score(self, training_params: Dict[str, Any]) -> float:
        """Calculate model stability score from training parameters."""
        stability_indicators = {
            'validation_loss_std': lambda x: 1.0 / (1.0 + x),  # Lower std = higher stability
            'training_epochs': lambda x: min(1.0, x / 100),    # More epochs up to limit
            'cross_validation_folds': lambda x: min(1.0, x / 10),  # More folds = more stable
            'early_stopping_patience': lambda x: min(1.0, x / 20)  # Patience indicates stability
        }
        
        scores = []
        for param, func in stability_indicators.items():
            if param in training_params:
                scores.append(func(training_params[param]))
        
        return np.mean(scores) if scores else 0.5
    
    def _calculate_medical_safety_score(self, metrics: Dict[str, float]) -> float:
        """Calculate medical safety score based on critical thresholds."""
        safety_thresholds = {
            'sensitivity': 0.85,  # High sensitivity critical for medical screening
            'specificity': 0.80,  # Reasonable specificity to avoid false positives
            'precision': 0.75,    # Precision important for treatment decisions
            'auc_score': 0.80     # Overall discriminative ability
        }
        
        safety_scores = []
        for metric, threshold in safety_thresholds.items():
            if metric in metrics:
                if metrics[metric] >= threshold:
                    safety_scores.append(1.0)
                else:
                    # Exponential penalty for falling below threshold
                    penalty = np.exp(-(metrics[metric] / threshold))
                    safety_scores.append(penalty)
        
        return np.mean(safety_scores) if safety_scores else 0.0


class AutonomousModelVersionManager:
    """Manages autonomous model versioning and deployment."""
    
    def __init__(self, storage_path: Path, quantum_strategy: Optional[QuantumVersioningStrategy] = None):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.quantum_strategy = quantum_strategy or QuantumVersioningStrategy()
        
        self.versions: Dict[str, ModelVersion] = {}
        self.active_tests: Dict[str, ABTestResult] = {}
        self.load_existing_versions()
        
    def load_existing_versions(self):
        """Load existing model versions from storage."""
        versions_file = self.storage_path / "versions.json"
        if versions_file.exists():
            try:
                with open(versions_file, 'r') as f:
                    data = json.load(f)
                    for version_data in data:
                        version = ModelVersion(**version_data)
                        version.creation_timestamp = datetime.fromisoformat(version.creation_timestamp)
                        self.versions[version.version_id] = version
            except Exception as e:
                logger.warning(f"Failed to load existing versions: {e}")
    
    def save_versions(self):
        """Save model versions to storage."""
        versions_file = self.storage_path / "versions.json"
        try:
            data = []
            for version in self.versions.values():
                version_dict = asdict(version)
                version_dict['creation_timestamp'] = version.creation_timestamp.isoformat()
                data.append(version_dict)
            
            with open(versions_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save versions: {e}")
    
    async def register_new_model(self, model_path: str, 
                               performance_metrics: Dict[str, float],
                               training_parameters: Dict[str, Any],
                               validation_results: Dict[str, Any],
                               medical_compliance_score: float = 1.0,
                               quantum_optimization_applied: bool = False,
                               parent_version_id: Optional[str] = None) -> Optional[str]:
        """Register a new model version if it meets versioning criteria."""
        
        # Get current active version for comparison
        current_version = self.get_active_version()
        
        if current_version:
            should_version, decision_score = self.quantum_strategy.should_create_version(
                current_version.performance_metrics,
                performance_metrics,
                training_parameters
            )
            
            if not should_version:
                logger.info(f"New model does not meet versioning criteria (score: {decision_score:.3f})")
                return None
        
        # Create new version
        version_id = self._generate_version_id(model_path, training_parameters)
        
        new_version = ModelVersion(
            version_id=version_id,
            model_path=model_path,
            performance_metrics=performance_metrics,
            creation_timestamp=datetime.now(),
            training_parameters=training_parameters,
            validation_results=validation_results,
            medical_compliance_score=medical_compliance_score,
            quantum_optimization_applied=quantum_optimization_applied,
            parent_version_id=parent_version_id,
            deployment_status="staged"
        )
        
        self.versions[version_id] = new_version
        self.save_versions()
        
        logger.info(f"Created new model version: {version_id}")
        
        # Automatically initiate A/B test if there's an active version
        if current_version:
            await self.initiate_ab_test(current_version.version_id, version_id)
        
        return version_id
    
    def _generate_version_id(self, model_path: str, training_params: Dict[str, Any]) -> str:
        """Generate unique version ID based on model characteristics."""
        # Create hash from model path and key training parameters
        content = f"{model_path}_{json.dumps(training_params, sort_keys=True)}"
        hash_digest = hashlib.sha256(content.encode()).hexdigest()[:12]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"v_{timestamp}_{hash_digest}"
    
    async def initiate_ab_test(self, version_a_id: str, version_b_id: str,
                             test_duration_hours: int = 24,
                             traffic_split: Tuple[float, float] = (0.5, 0.5)) -> str:
        """Initiate A/B test between two model versions."""
        
        test_id = str(uuid.uuid4())[:8]
        
        # Update traffic allocation
        if version_a_id in self.versions:
            self.versions[version_a_id].a_b_test_allocation = traffic_split[0]
        if version_b_id in self.versions:
            self.versions[version_b_id].a_b_test_allocation = traffic_split[1]
        
        ab_test = ABTestResult(
            test_id=test_id,
            version_a=version_a_id,
            version_b=version_b_id,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=test_duration_hours),
            sample_size_a=0,
            sample_size_b=0,
            performance_a={},
            performance_b={},
            statistical_significance={},
            winner=None,
            confidence_level=0.95
        )
        
        self.active_tests[test_id] = ab_test
        
        logger.info(f"Started A/B test {test_id}: {version_a_id} vs {version_b_id}")
        
        # Schedule automatic test completion
        asyncio.create_task(self._auto_complete_ab_test(test_id, test_duration_hours))
        
        return test_id
    
    async def _auto_complete_ab_test(self, test_id: str, duration_hours: int):
        """Automatically complete A/B test after duration."""
        await asyncio.sleep(duration_hours * 3600)  # Convert hours to seconds
        
        if test_id in self.active_tests:
            await self.complete_ab_test(test_id)
    
    async def update_ab_test_metrics(self, test_id: str, version_id: str,
                                   prediction_result: Dict[str, Any]):
        """Update A/B test metrics with new prediction result."""
        
        if test_id not in self.active_tests:
            return
        
        test = self.active_tests[test_id]
        
        # Simulate performance metric extraction from prediction result
        # In practice, this would aggregate real prediction results
        performance_update = {
            'accuracy': prediction_result.get('confidence', 0.8),
            'response_time': prediction_result.get('processing_time', 0.1),
            'memory_usage': prediction_result.get('memory_mb', 100)
        }
        
        if version_id == test.version_a:
            test.sample_size_a += 1
            # Update running averages
            for metric, value in performance_update.items():
                if metric in test.performance_a:
                    # Running average update
                    n = test.sample_size_a
                    test.performance_a[metric] = ((n-1) * test.performance_a[metric] + value) / n
                else:
                    test.performance_a[metric] = value
        
        elif version_id == test.version_b:
            test.sample_size_b += 1
            for metric, value in performance_update.items():
                if metric in test.performance_b:
                    n = test.sample_size_b
                    test.performance_b[metric] = ((n-1) * test.performance_b[metric] + value) / n
                else:
                    test.performance_b[metric] = value
    
    async def complete_ab_test(self, test_id: str) -> Optional[ABTestResult]:
        """Complete A/B test and determine winner."""
        
        if test_id not in self.active_tests:
            return None
        
        test = self.active_tests[test_id]
        test.end_time = datetime.now()
        
        # Calculate statistical significance
        test.statistical_significance = self._calculate_statistical_significance(test)
        
        # Determine winner using quantum-enhanced decision making
        winner = self._determine_ab_test_winner(test)
        test.winner = winner
        
        # Promote winner to active status
        if winner:
            await self.promote_version_to_active(winner)
            
            # Retire the losing version
            losing_version = test.version_a if winner == test.version_b else test.version_b
            await self.retire_version(losing_version)
        
        # Clean up test
        del self.active_tests[test_id]
        
        logger.info(f"Completed A/B test {test_id}, winner: {winner}")
        
        return test
    
    def _calculate_statistical_significance(self, test: ABTestResult) -> Dict[str, float]:
        """Calculate statistical significance for A/B test metrics."""
        significance = {}
        
        for metric in test.performance_a:
            if metric in test.performance_b:
                # Simplified t-test calculation
                # In practice, would use proper statistical methods
                diff = abs(test.performance_a[metric] - test.performance_b[metric])
                pooled_std = 0.1  # Simplified assumption
                n_a, n_b = test.sample_size_a, test.sample_size_b
                
                if n_a > 1 and n_b > 1:
                    se_diff = pooled_std * np.sqrt(1/n_a + 1/n_b)
                    t_stat = diff / se_diff if se_diff > 0 else 0
                    # Approximate p-value calculation
                    p_value = max(0.001, 1 / (1 + t_stat**2))
                    significance[metric] = 1 - p_value  # Convert to significance level
                else:
                    significance[metric] = 0.0
        
        return significance
    
    def _determine_ab_test_winner(self, test: ABTestResult) -> Optional[str]:
        """Determine A/B test winner using quantum-enhanced decision theory."""
        
        # Medical-specific metric weights
        metric_weights = {
            'accuracy': 3.0,
            'sensitivity': 4.0,
            'specificity': 3.5,
            'response_time': -1.0,  # Lower is better
            'memory_usage': -0.5    # Lower is better
        }
        
        score_a = 0.0
        score_b = 0.0
        significance_threshold = 0.8
        
        for metric in test.performance_a:
            if metric in test.performance_b and metric in metric_weights:
                weight = metric_weights[metric]
                
                # Check statistical significance
                significance = test.statistical_significance.get(metric, 0.0)
                if significance < significance_threshold:
                    continue  # Skip non-significant differences
                
                value_a = test.performance_a[metric]
                value_b = test.performance_b[metric]
                
                if weight > 0:  # Higher is better
                    if value_a > value_b:
                        score_a += weight * significance
                    else:
                        score_b += weight * significance
                else:  # Lower is better
                    if value_a < value_b:
                        score_a += abs(weight) * significance
                    else:
                        score_b += abs(weight) * significance
        
        # Require minimum score difference for winner declaration
        min_difference = 1.0
        if abs(score_a - score_b) < min_difference:
            return None  # No clear winner
        
        return test.version_a if score_a > score_b else test.version_b
    
    async def promote_version_to_active(self, version_id: str):
        """Promote a model version to active deployment status."""
        
        # Retire current active version
        for vid, version in self.versions.items():
            if version.deployment_status == "active":
                version.deployment_status = "retired"
        
        # Activate new version
        if version_id in self.versions:
            self.versions[version_id].deployment_status = "active"
            self.versions[version_id].a_b_test_allocation = 1.0
            self.save_versions()
            
            logger.info(f"Promoted version {version_id} to active status")
    
    async def retire_version(self, version_id: str):
        """Retire a model version."""
        if version_id in self.versions:
            self.versions[version_id].deployment_status = "retired"
            self.versions[version_id].a_b_test_allocation = 0.0
            self.save_versions()
            
            logger.info(f"Retired version {version_id}")
    
    def get_active_version(self) -> Optional[ModelVersion]:
        """Get the currently active model version."""
        for version in self.versions.values():
            if version.deployment_status == "active":
                return version
        return None
    
    def get_version_for_prediction(self, request_context: Dict[str, Any] = None) -> Optional[str]:
        """Get the appropriate model version for a prediction request."""
        
        # Check if there are active A/B tests
        for test in self.active_tests.values():
            if test.end_time and datetime.now() < test.end_time:
                # Route traffic based on allocation
                if np.random.random() < self.versions[test.version_a].a_b_test_allocation:
                    return test.version_a
                else:
                    return test.version_b
        
        # Default to active version
        active_version = self.get_active_version()
        return active_version.version_id if active_version else None
    
    def get_version_performance_history(self, version_id: str) -> Dict[str, List[float]]:
        """Get performance history for a specific version."""
        # This would integrate with monitoring systems in practice
        # For now, return simulated data
        
        if version_id not in self.versions:
            return {}
        
        # Simulate performance data over time
        metrics = self.versions[version_id].performance_metrics
        history = {}
        
        for metric, value in metrics.items():
            # Generate simulated time series around the base value
            noise = np.random.normal(0, value * 0.05, 24)  # 24 hours of data
            history[metric] = [max(0, value + n) for n in noise]
        
        return history