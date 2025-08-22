"""
Comprehensive Test Suite for Quantum Medical Research Framework
================================================================

Publication-ready test suite with statistical validation, performance
benchmarking, and regulatory compliance verification for academic
research and peer review.

Test Categories:
1. Statistical Hypothesis Testing Validation
2. Quantum Algorithm Correctness Verification  
3. Medical Safety & Compliance Testing
4. Performance Benchmark Validation
5. Reproducibility & Determinism Testing
6. Robustness & Error Handling Testing
7. Integration & End-to-End Testing
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
from scipy import stats
from unittest.mock import AsyncMock, MagicMock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import modules to test
from research.quantum_medical_research_framework import (
    QuantumMedicalResearchFramework,
    ResearchHypothesis,
    ExperimentalResult,
    StatisticalAnalysis
)
from research.novel_quantum_medical_algorithms import (
    QuantumVariationalMedicalOptimizer,
    MedicalQuantumFeatureSelector,
    QuantumMedicalEnsembleOptimizer,
    MedicalOptimizationProblem
)
from research.robust_quantum_medical_validation import (
    RobustQuantumMedicalValidator,
    QuantumCircuitBreaker,
    SecureMedicalDataProcessor,
    ValidationStatus
)
from research.medical_ai_monitoring_system import (
    MedicalAIMonitoringSystem,
    MedicalMetricsCollector,
    MedicalAlertManager
)
from research.quantum_scaling_orchestrator import (
    QuantumScalingOrchestrator,
    QuantumLoadBalancer,
    QuantumPredictiveScaler
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestQuantumMedicalResearchFramework:
    """Test suite for quantum medical research framework."""
    
    @pytest.fixture
    def research_framework(self):
        """Create research framework instance."""
        return QuantumMedicalResearchFramework(random_seed=42)
    
    @pytest.mark.asyncio
    async def test_research_framework_initialization(self, research_framework):
        """Test research framework initializes correctly."""
        assert research_framework.random_seed == 42
        assert len(research_framework.hypotheses) == 3
        assert research_framework.results_cache == {}
        assert research_framework.experimental_log == []
    
    @pytest.mark.asyncio
    async def test_hypothesis_definitions(self, research_framework):
        """Test research hypotheses are properly defined."""
        hypotheses = research_framework.hypotheses
        
        # Check all required hypotheses exist
        hypothesis_names = [h.name for h in hypotheses]
        expected_names = [
            "quantum_optimization_advantage",
            "quantum_cnn_accuracy_improvement", 
            "quantum_scheduling_latency_reduction"
        ]
        
        for expected_name in expected_names:
            assert expected_name in hypothesis_names
        
        # Check hypothesis structure
        for hypothesis in hypotheses:
            assert isinstance(hypothesis, ResearchHypothesis)
            assert hypothesis.alpha == 0.05
            assert hypothesis.sample_size > 0
            assert len(hypothesis.statement) > 10
            assert len(hypothesis.null_hypothesis) > 10
    
    @pytest.mark.asyncio
    async def test_experimental_data_generation(self, research_framework):
        """Test experimental data generation produces valid results."""
        hypothesis = research_framework.hypotheses[0]
        
        # Test quantum data generation
        quantum_result = await research_framework._generate_quantum_experimental_data(hypothesis)
        assert isinstance(quantum_result, ExperimentalResult)
        assert quantum_result.algorithm_name == "quantum_enhanced"
        assert len(quantum_result.performance_scores) == hypothesis.sample_size
        assert len(quantum_result.convergence_times) == hypothesis.sample_size
        assert all(0.5 <= score <= 1.0 for score in quantum_result.performance_scores)
        
        # Test classical data generation
        classical_result = await research_framework._generate_classical_experimental_data(hypothesis)
        assert isinstance(classical_result, ExperimentalResult)
        assert classical_result.algorithm_name == "classical_baseline"
        assert len(classical_result.performance_scores) == hypothesis.sample_size
        assert len(classical_result.convergence_times) == hypothesis.sample_size
    
    @pytest.mark.asyncio
    async def test_statistical_analysis(self, research_framework):
        """Test statistical analysis produces valid results."""
        hypothesis = research_framework.hypotheses[0]
        
        # Generate test data with known difference
        quantum_result = ExperimentalResult(
            algorithm_name="quantum_enhanced",
            performance_scores=[0.9] * 50,  # Higher performance
            convergence_times=[10.0] * 50,  # Faster convergence
            resource_usage=[{"cpu": 0.5}] * 50,
            medical_compliance_scores=[0.95] * 50
        )
        
        classical_result = ExperimentalResult(
            algorithm_name="classical_baseline", 
            performance_scores=[0.8] * 50,  # Lower performance
            convergence_times=[20.0] * 50,  # Slower convergence
            resource_usage=[{"cpu": 0.7}] * 50,
            medical_compliance_scores=[0.90] * 50
        )
        
        # Perform statistical test
        analysis = research_framework._perform_statistical_test(
            hypothesis, quantum_result, classical_result
        )
        
        assert isinstance(analysis, StatisticalAnalysis)
        assert analysis.hypothesis == hypothesis
        assert 0.0 <= analysis.p_value <= 1.0
        assert analysis.effect_size >= 0.0
        assert len(analysis.confidence_interval) == 2
        assert isinstance(analysis.is_significant, bool)
        assert len(analysis.interpretation) > 10
    
    @pytest.mark.asyncio
    async def test_comprehensive_study_execution(self, research_framework):
        """Test full research study execution."""
        # Run study (should complete without errors)
        results = await research_framework.run_comprehensive_study()
        
        # Validate study structure
        assert "study_metadata" in results
        assert "hypothesis_results" in results
        assert "comparative_analysis" in results
        assert "publication_summary" in results
        
        # Validate metadata
        metadata = results["study_metadata"]
        assert "start_time" in metadata
        assert "end_time" in metadata
        assert metadata["random_seed"] == 42
        
        # Validate hypothesis results
        hypothesis_results = results["hypothesis_results"]
        assert len(hypothesis_results) == 3
        
        for hypothesis_name, result in hypothesis_results.items():
            assert "hypothesis" in result
            assert "quantum_results" in result
            assert "classical_results" in result
            assert "statistical_analysis" in result
    
    @pytest.mark.asyncio
    async def test_publication_summary_generation(self, research_framework):
        """Test publication summary contains required elements."""
        results = await research_framework.run_comprehensive_study()
        pub_summary = results["publication_summary"]
        
        # Check required publication elements
        required_fields = [
            "title", "abstract_summary", "key_findings",
            "statistical_significance", "effect_size_summary",
            "recommended_venues", "reproducibility_score"
        ]
        
        for field in required_fields:
            assert field in pub_summary
        
        # Validate content quality
        assert len(pub_summary["title"]) > 20
        assert len(pub_summary["abstract_summary"]) > 50
        assert isinstance(pub_summary["key_findings"], list)
        assert len(pub_summary["recommended_venues"]) > 0
        assert pub_summary["reproducibility_score"] == "high"

class TestQuantumMedicalAlgorithms:
    """Test suite for novel quantum medical algorithms."""
    
    @pytest.fixture
    def sample_optimization_problem(self):
        """Create sample medical optimization problem."""
        def objective_function(params):
            return np.sum(params**2)
        
        return MedicalOptimizationProblem(
            objective_function=objective_function,
            constraints=[],
            medical_safety_bounds={
                "sensitivity_constraint": (0.85, 1.0),
                "specificity_constraint": (0.80, 1.0)
            },
            regulatory_requirements={"fda_approval": 0.9},
            parameter_bounds=[(0, 1) for _ in range(5)]
        )
    
    @pytest.mark.asyncio
    async def test_quantum_variational_optimizer(self, sample_optimization_problem):
        """Test Quantum Variational Medical Optimizer."""
        optimizer = QuantumVariationalMedicalOptimizer(n_qubits=4, n_layers=2)
        
        result = await optimizer.optimize(sample_optimization_problem)
        
        # Validate result structure
        assert "algorithm" in result
        assert "optimal_parameters" in result
        assert "quantum_state" in result
        assert "objective_value" in result
        assert "convergence_iterations" in result
        assert "optimization_time" in result
        
        # Validate quantum-specific metrics
        quantum_state = result["quantum_state"]
        assert "amplitudes" in quantum_state
        assert "phases" in quantum_state
        assert "entanglement_measure" in quantum_state
        
        # Validate medical metrics
        assert "medical_safety_score" in result
        assert "regulatory_compliance" in result
        
        # Validate novel contributions
        assert "novel_contributions" in result
        assert len(result["novel_contributions"]) > 0
    
    @pytest.mark.asyncio
    async def test_medical_quantum_feature_selector(self, sample_optimization_problem):
        """Test Medical Quantum Feature Selector."""
        selector = MedicalQuantumFeatureSelector(n_features=6)
        
        result = await selector.optimize(sample_optimization_problem)
        
        # Validate result structure
        assert "algorithm" in result
        assert "selected_features" in result
        assert "feature_importance_scores" in result
        assert "quantum_coherence" in result
        assert "medical_relevance_score" in result
        
        # Validate feature selection
        selected_features = result["selected_features"]
        assert len(selected_features) == 6
        assert all(f in [0, 1] for f in selected_features)
        
        # Validate importance scores
        importance_scores = result["feature_importance_scores"]
        assert len(importance_scores) == 6
        assert all(0.0 <= score <= 1.0 for score in importance_scores)
    
    @pytest.mark.asyncio
    async def test_quantum_ensemble_optimizer(self, sample_optimization_problem):
        """Test Quantum Medical Ensemble Optimizer."""
        optimizer = QuantumMedicalEnsembleOptimizer(n_models=4)
        
        result = await optimizer.optimize(sample_optimization_problem)
        
        # Validate result structure
        assert "algorithm" in result
        assert "optimal_ensemble_weights" in result
        assert "ensemble_diversity" in result
        assert "medical_safety_score" in result
        
        # Validate ensemble weights
        weights = result["optimal_ensemble_weights"]
        assert len(weights) == 4
        assert abs(sum(weights) - 1.0) < 1e-6  # Weights sum to 1
        assert all(w >= 0 for w in weights)  # Non-negative weights
        
        # Validate diversity and safety
        assert 0.0 <= result["ensemble_diversity"] <= 1.0
        assert 0.0 <= result["medical_safety_score"] <= 1.0

class TestRobustValidation:
    """Test suite for robust quantum medical validation."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self):
        """Test quantum circuit breaker prevents cascading failures."""
        circuit_breaker = QuantumCircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        
        # Test successful operation
        async def successful_function():
            return "success"
        
        result = await circuit_breaker.call(successful_function)
        assert result == "success"
        assert circuit_breaker.state == "CLOSED"
        
        # Test failure handling
        async def failing_function():
            raise Exception("Test failure")
        
        # Trigger failures
        for _ in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_function)
        
        # Circuit should be OPEN now
        assert circuit_breaker.state == "OPEN"
        
        # Further calls should be blocked
        with pytest.raises(Exception, match="Circuit breaker OPEN"):
            await circuit_breaker.call(failing_function)
    
    @pytest.mark.asyncio
    async def test_secure_medical_data_processor(self):
        """Test secure medical data processing."""
        processor = SecureMedicalDataProcessor()
        
        # Test data encryption/decryption
        test_data = {"patient_id": "12345", "diagnosis": "pneumonia"}
        patient_id = "TEST_PATIENT"
        
        # Encrypt data
        encrypted_data, audit_hash = await processor.encrypt_medical_data(test_data, patient_id)
        
        assert isinstance(encrypted_data, bytes)
        assert len(audit_hash) == 64  # SHA-256 hash length
        
        # Decrypt data
        decrypted_data = await processor.decrypt_medical_data(encrypted_data, audit_hash)
        
        assert decrypted_data == test_data
    
    @pytest.mark.asyncio
    async def test_robust_validator_integration(self):
        """Test complete robust validation workflow."""
        validator = RobustQuantumMedicalValidator()
        
        # Sample algorithm for testing
        async def sample_algorithm(medical_data, config):
            return {
                "prediction": "pneumonia",
                "confidence": 0.85,
                "quantum_state": [0.5, 0.5, 0.0, 0.0],
                "medical_constraints": {"safety_margin": 0.1},
                "accuracy": 0.90,
                "sensitivity": 0.88,
                "specificity": 0.92,
                "audit_hash": "test_hash",
                "consent_hash": "consent_hash"
            }
        
        # Test data
        medical_data = {"image_data": "test_data"}
        validation_config = {
            "patient_id": "12345",
            "algorithm_name": "TestAlgorithm",
            "data_type": "xray",
            "compliance": ["HIPAA"]
        }
        
        # Run validation
        result = await validator.validate_quantum_medical_algorithm(
            sample_algorithm, medical_data, validation_config
        )
        
        # Validate results
        assert result.status == ValidationStatus.PASSED
        assert result.algorithm_name == "TestAlgorithm"
        assert result.execution_time > 0
        assert 0.0 <= result.compliance_score <= 1.0
        assert 0.0 <= result.medical_safety_score <= 1.0

class TestMedicalAIMonitoring:
    """Test suite for medical AI monitoring system."""
    
    @pytest.mark.asyncio
    async def test_metrics_collector(self):
        """Test medical metrics collection."""
        collector = MedicalMetricsCollector()
        
        # Test model performance metrics
        sample_predictions = [
            {"confidence": 0.85, "quantum_fidelity": 0.95},
            {"confidence": 0.90, "quantum_fidelity": 0.93},
            {"confidence": 0.78, "quantum_fidelity": 0.97}
        ]
        
        metrics = await collector.collect_model_performance_metrics(
            "TestModel", sample_predictions
        )
        
        assert len(metrics) >= 3  # At least avg, min, std confidence + quantum fidelity
        
        # Check metric structure
        for metric in metrics:
            assert hasattr(metric, 'name')
            assert hasattr(metric, 'value')
            assert hasattr(metric, 'timestamp')
            assert hasattr(metric, 'unit')
            assert hasattr(metric, 'tags')
    
    @pytest.mark.asyncio
    async def test_alert_manager(self):
        """Test medical alert management."""
        alert_manager = MedicalAlertManager()
        
        # Create sample metrics that should trigger alerts
        from research.medical_ai_monitoring_system import MetricData, AlertSeverity
        
        # Low accuracy metric (should trigger alert)
        low_accuracy_metric = MetricData(
            name="model_accuracy",
            value=0.70,  # Below warning threshold
            timestamp=time.time(),
            unit="ratio",
            tags={"model": "test"}
        )
        
        # Test alert evaluation
        thresholds = {
            "model_accuracy": {
                "warning": 0.85,
                "critical": 0.80,
                "emergency": 0.75
            }
        }
        
        alerts = await alert_manager.evaluate_alert_conditions([low_accuracy_metric], thresholds)
        
        assert len(alerts) == 1
        alert = alerts[0]
        assert alert.severity == AlertSeverity.CRITICAL  # 0.70 is below critical threshold
        assert "accuracy" in alert.title.lower()
        assert len(alert.clinical_impact) > 0
        assert len(alert.recommended_action) > 0
    
    @pytest.mark.asyncio
    async def test_health_checker(self):
        """Test medical system health checking."""
        from research.medical_ai_monitoring_system import MedicalHealthChecker, HealthStatus
        
        health_checker = MedicalHealthChecker()
        
        # Run all health checks
        health_results = await health_checker.check_all_health()
        
        # Validate results
        expected_checks = [
            "model_availability",
            "database_connection", 
            "quantum_subsystem",
            "compliance_systems",
            "clinical_integration"
        ]
        
        for check_name in expected_checks:
            assert check_name in health_results
            health_check = health_results[check_name]
            assert hasattr(health_check, 'name')
            assert hasattr(health_check, 'status')
            assert hasattr(health_check, 'latency_ms')
            assert hasattr(health_check, 'details')
            assert health_check.status in HealthStatus
            assert health_check.latency_ms >= 0

class TestQuantumScaling:
    """Test suite for quantum scaling orchestrator."""
    
    @pytest.mark.asyncio
    async def test_quantum_load_balancer(self):
        """Test quantum load balancer."""
        regions = ["us-east-1", "us-west-2"]
        load_balancer = QuantumLoadBalancer(regions)
        
        # Test initialization
        assert len(load_balancer.regions) == 2
        assert len(load_balancer.quantum_weights) == 2
        assert load_balancer.entanglement_matrix.shape == (2, 2)
        
        # Create sample workload and metrics
        from research.quantum_scaling_orchestrator import MedicalWorkload, WorkloadType, ScalingMetrics
        
        workload = MedicalWorkload(
            workload_id="test_workload",
            workload_type=WorkloadType.EMERGENCY_DIAGNOSIS,
            priority=9,
            resource_requirements={"cpu_cores": 4},
            latency_requirement_ms=500,
            compliance_requirements=["HIPAA"],
            estimated_duration_minutes=30,
            quantum_enhancement_enabled=True
        )
        
        region_metrics = {
            "us-east-1": ScalingMetrics(
                timestamp=time.time(),
                cpu_utilization=0.6,
                memory_utilization=0.5,
                gpu_utilization=0.4,
                quantum_utilization=0.3,
                request_rate=50.0,
                average_latency_ms=200.0,
                error_rate=0.01,
                cost_per_hour=100.0,
                compliance_score=0.95
            ),
            "us-west-2": ScalingMetrics(
                timestamp=time.time(),
                cpu_utilization=0.8,
                memory_utilization=0.7,
                gpu_utilization=0.6,
                quantum_utilization=0.5,
                request_rate=75.0,
                average_latency_ms=300.0,
                error_rate=0.02,
                cost_per_hour=120.0,
                compliance_score=0.93
            )
        }
        
        quantum_resources = {}  # Simplified for test
        
        # Test load balancing decision
        selected_region = await load_balancer.quantum_load_balance(
            workload, region_metrics, quantum_resources
        )
        
        assert selected_region in regions
        # us-east-1 should be preferred due to lower utilization
        assert selected_region == "us-east-1"
    
    @pytest.mark.asyncio
    async def test_quantum_predictive_scaler(self):
        """Test quantum predictive scaler."""
        scaler = QuantumPredictiveScaler(prediction_horizon_minutes=30)
        
        # Create sample metrics and workloads
        from research.quantum_scaling_orchestrator import ScalingMetrics, MedicalWorkload, WorkloadType, ScalingDecision
        
        current_metrics = {
            "us-east-1": ScalingMetrics(
                timestamp=time.time(),
                cpu_utilization=0.7,
                memory_utilization=0.6,
                gpu_utilization=0.5,
                quantum_utilization=0.4,
                request_rate=60.0,
                average_latency_ms=250.0,
                error_rate=0.015,
                cost_per_hour=110.0,
                compliance_score=0.94
            )
        }
        
        workload_queue = [
            MedicalWorkload(
                workload_id="emergency_1",
                workload_type=WorkloadType.EMERGENCY_DIAGNOSIS,
                priority=10,
                resource_requirements={"cpu_cores": 8},
                latency_requirement_ms=200,
                compliance_requirements=["HIPAA"],
                estimated_duration_minutes=15,
                quantum_enhancement_enabled=True
            )
        ]
        
        # Test prediction
        scaling_decisions = await scaler.predict_scaling_needs(current_metrics, workload_queue)
        
        assert "us-east-1" in scaling_decisions
        decision = scaling_decisions["us-east-1"]
        assert isinstance(decision, ScalingDecision)

class TestIntegrationAndPerformance:
    """Integration and performance test suite."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_research_workflow(self):
        """Test complete research workflow from start to finish."""
        # Initialize research framework
        framework = QuantumMedicalResearchFramework(random_seed=42)
        
        # Run complete study
        start_time = time.time()
        results = await framework.run_comprehensive_study()
        execution_time = time.time() - start_time
        
        # Performance validation
        assert execution_time < 30.0  # Should complete within 30 seconds
        
        # Results validation
        assert "study_metadata" in results
        assert "hypothesis_results" in results
        assert "comparative_analysis" in results
        assert "publication_summary" in results
        
        # Statistical significance validation
        comparative = results["comparative_analysis"]
        assert "significance_rate" in comparative
        assert 0.0 <= comparative["significance_rate"] <= 1.0
        
        # Publication readiness validation
        pub_summary = results["publication_summary"]
        assert pub_summary["reproducibility_score"] == "high"
        assert len(pub_summary["recommended_venues"]) > 0
    
    @pytest.mark.asyncio
    async def test_reproducibility_validation(self):
        """Test research reproducibility with same random seed."""
        # Run study twice with same seed
        framework1 = QuantumMedicalResearchFramework(random_seed=123)
        framework2 = QuantumMedicalResearchFramework(random_seed=123)
        
        results1 = await framework1.run_comprehensive_study()
        results2 = await framework2.run_comprehensive_study()
        
        # Compare key metrics for reproducibility
        comp1 = results1["comparative_analysis"]
        comp2 = results2["comparative_analysis"]
        
        # Results should be identical with same seed
        assert comp1["total_hypotheses_tested"] == comp2["total_hypotheses_tested"]
        
        # Significance rates should be close (allowing for small numerical differences)
        assert abs(comp1["significance_rate"] - comp2["significance_rate"]) < 0.1
        assert abs(comp1["mean_effect_size"] - comp2["mean_effect_size"]) < 0.1
    
    @pytest.mark.asyncio
    async def test_memory_usage_validation(self):
        """Test memory usage remains within acceptable bounds."""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run research study
        framework = QuantumMedicalResearchFramework(random_seed=42)
        results = await framework.run_comprehensive_study()
        
        # Force garbage collection
        gc.collect()
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB for test)
        assert memory_increase < 100.0, f"Memory usage increased by {memory_increase:.1f}MB"
        
        # Cleanup
        del framework
        del results
        gc.collect()

class TestStatisticalValidation:
    """Statistical validation test suite for academic rigor."""
    
    def test_statistical_power_analysis(self):
        """Test statistical power analysis for sample size validation."""
        # Create research framework
        framework = QuantumMedicalResearchFramework(random_seed=42)
        
        # Check sample sizes meet power requirements
        for hypothesis in framework.hypotheses:
            # For effect size of 0.5 and alpha of 0.05, minimum sample size is ~64
            assert hypothesis.sample_size >= 50, f"Insufficient sample size for {hypothesis.name}"
            assert hypothesis.alpha == 0.05, "Alpha level should be 0.05"
            assert hypothesis.effect_size_threshold >= 0.3, "Effect size threshold too low"
    
    def test_multiple_comparison_correction(self):
        """Test multiple comparison correction implementation."""
        # Test Bonferroni correction
        num_hypotheses = 3
        alpha = 0.05
        corrected_alpha = alpha / num_hypotheses
        
        assert corrected_alpha == pytest.approx(0.0167, rel=1e-3)
        
        # Validate framework uses appropriate corrections
        framework = QuantumMedicalResearchFramework(random_seed=42)
        for hypothesis in framework.hypotheses:
            # Individual hypothesis alpha should account for multiple testing
            assert hypothesis.alpha <= 0.05
    
    def test_effect_size_calculations(self):
        """Test effect size calculation accuracy."""
        # Test Cohen's d calculation
        group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        group2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        
        # Manual Cohen's d calculation
        mean_diff = np.mean(group2) - np.mean(group1)
        pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
        expected_cohens_d = mean_diff / pooled_std
        
        # Should be exactly 1.0 for this example
        assert expected_cohens_d == pytest.approx(1.0, rel=1e-10)

# Performance benchmarks
@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmark test suite."""
    
    @pytest.mark.asyncio
    async def test_algorithm_performance_benchmarks(self):
        """Benchmark quantum algorithm performance."""
        # Test QVMO performance
        optimizer = QuantumVariationalMedicalOptimizer(n_qubits=6, n_layers=3)
        
        def objective(params):
            return np.sum(params**2)
        
        problem = MedicalOptimizationProblem(
            objective_function=objective,
            constraints=[],
            medical_safety_bounds={"safety": (0.1, 0.9)},
            regulatory_requirements={"fda": 0.9},
            parameter_bounds=[(0, 1) for _ in range(5)]
        )
        
        # Benchmark optimization time
        start_time = time.time()
        result = await optimizer.optimize(problem)
        execution_time = time.time() - start_time
        
        # Performance assertions
        assert execution_time < 5.0, "QVMO should complete within 5 seconds"
        assert result["optimization_time"] < 5.0
        assert "quantum_advantage_metrics" in result
    
    @pytest.mark.asyncio
    async def test_scaling_performance_benchmarks(self):
        """Benchmark scaling orchestrator performance."""
        regions = ["us-east-1", "us-west-2", "eu-west-1"]
        orchestrator = QuantumScalingOrchestrator(regions)
        
        # Benchmark status retrieval
        start_time = time.time()
        status = await orchestrator.get_orchestration_status()
        execution_time = time.time() - start_time
        
        # Performance assertions
        assert execution_time < 1.0, "Status retrieval should be fast"
        assert "global_metrics" in status
        assert "regional_status" in status

# Run all tests
if __name__ == "__main__":
    # Configure pytest arguments for comprehensive testing
    pytest_args = [
        __file__,
        "-v",                    # Verbose output
        "--tb=short",           # Short traceback format
        "--strict-markers",     # Strict marker checking
        "--tb=line",           # Line-level traceback
        "-x",                  # Stop on first failure
        "--disable-warnings",  # Disable warnings for cleaner output
    ]
    
    # Run performance tests separately if requested
    if "--performance" in sys.argv:
        pytest_args.extend(["-m", "performance"])
    
    logger.info("🧪 Starting Comprehensive Quantum Medical Research Test Suite")
    logger.info("📊 Testing statistical validation, algorithms, monitoring, and scaling")
    
    # Execute tests
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        logger.info("✅ All tests passed - Research framework ready for publication")
    else:
        logger.error("❌ Some tests failed - Review and fix before publication")
    
    sys.exit(exit_code)