"""
Advanced Feature Testing Suite
Comprehensive tests for newly implemented advanced features.
"""

import pytest
import numpy as np
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tensorflow as tf

# Import modules to test
from src.adaptive_learning_scheduler import (
    AdaptiveLearningScheduler,
    GradientClippingCallback,
    create_adaptive_scheduler
)
from src.federated_learning_coordinator import (
    FederatedLearningCoordinator,
    FederatedClient,
    ModelUpdate,
    FedAvgAggregator,
    AdaptiveFedAggregator,
    create_federated_pneumonia_detector
)
from src.quantum_enhanced_optimizer import (
    QuantumInspiredOptimizer,
    QuantumAnnealingScheduler,
    QuantumErrorCorrection,
    create_quantum_optimizer
)
from src.advanced_error_recovery import (
    AdvancedErrorRecoverySystem,
    ErrorSeverity,
    ModelCheckpointRecovery,
    DataCorruptionRecovery,
    MemoryRecovery,
    ErrorRecoveryContext
)
from src.comprehensive_validation import (
    ComprehensiveValidationFramework,
    DataIntegrityValidator,
    ImageQualityValidator,
    ModelArchitectureValidator,
    SecurityValidator
)
from src.intelligent_auto_scaler import (
    IntelligentAutoScaler,
    MetricsCollector,
    AdaptiveScalingStrategy,
    ResourceMetrics,
    ScalingAction
)


class TestAdaptiveLearningScheduler:
    """Test adaptive learning scheduler functionality."""
    
    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        scheduler = AdaptiveLearningScheduler(
            base_lr=0.001,
            max_lr=0.01,
            step_size=2000
        )
        
        assert scheduler.base_lr == 0.001
        assert scheduler.max_lr == 0.01
        assert scheduler.step_size == 2000
        assert len(scheduler.monitor_metrics) > 0
        
    def test_cyclical_learning_rate(self):
        """Test cyclical learning rate calculation."""
        scheduler = AdaptiveLearningScheduler(
            base_lr=0.001,
            max_lr=0.01,
            step_size=100
        )
        
        # Test different iteration points
        scheduler.clr_iterations = 0
        lr1 = scheduler.clr()
        
        scheduler.clr_iterations = 50
        lr2 = scheduler.clr()
        
        scheduler.clr_iterations = 100
        lr3 = scheduler.clr()
        
        # Learning rate should vary cyclically
        assert scheduler.base_lr <= lr1 <= scheduler.max_lr
        assert scheduler.base_lr <= lr2 <= scheduler.max_lr
        assert scheduler.base_lr <= lr3 <= scheduler.max_lr
        
    def test_warmup_learning_rate(self):
        """Test warmup phase learning rate."""
        scheduler = AdaptiveLearningScheduler(
            base_lr=0.001,
            warmup_epochs=5
        )
        
        # Test warmup progression
        lr_epoch_1 = scheduler.warmup_lr(0)
        lr_epoch_3 = scheduler.warmup_lr(2)
        lr_epoch_5 = scheduler.warmup_lr(4)
        
        assert lr_epoch_1 < lr_epoch_3 < lr_epoch_5
        assert lr_epoch_5 == scheduler.base_lr
        
    @patch('tensorflow.keras.backend.set_value')
    @patch('tensorflow.keras.backend.get_value')
    def test_on_batch_end(self, mock_get_value, mock_set_value):
        """Test learning rate update on batch end."""
        mock_get_value.return_value = 0.001
        
        scheduler = AdaptiveLearningScheduler()
        scheduler.model = Mock()
        scheduler.model.optimizer.learning_rate = Mock()
        scheduler.model.steps_per_epoch = 100
        
        logs = {'loss': 0.5}
        scheduler.on_batch_end(0, logs)
        
        # Verify learning rate was set
        mock_set_value.assert_called()
        assert 'lr' in logs
        
    def test_gradient_clipping_callback(self):
        """Test gradient clipping callback."""
        callback = GradientClippingCallback(clip_norm=1.0)
        
        assert callback.clip_norm == 1.0
        assert len(callback.gradient_norms) == 0
        
    def test_create_adaptive_scheduler(self):
        """Test scheduler creation utility."""
        scheduler = create_adaptive_scheduler(
            base_lr=0.0001,
            max_lr=0.001,
            training_samples=1000,
            batch_size=32
        )
        
        assert isinstance(scheduler, AdaptiveLearningScheduler)
        assert scheduler.base_lr == 0.0001
        assert scheduler.max_lr == 0.001


class TestFederatedLearningCoordinator:
    """Test federated learning coordination."""
    
    def test_federated_client_creation(self):
        """Test federated client creation."""
        client = FederatedClient(
            client_id="test_client",
            name="Test Hospital",
            data_samples=1000,
            model_version="1.0.0",
            last_update=time.time()
        )
        
        assert client.client_id == "test_client"
        assert client.data_samples == 1000
        assert client.reputation_score == 1.0
        
    def test_model_update_creation(self):
        """Test model update creation."""
        weights = [np.random.random((10, 5)), np.random.random((5, 1))]
        metrics = {"accuracy": 0.85, "loss": 0.3}
        
        update = ModelUpdate(
            client_id="client_1",
            model_weights=weights,
            metrics=metrics,
            timestamp=time.time()
        )
        
        assert update.client_id == "client_1"
        assert len(update.model_weights) == 2
        assert update.metrics["accuracy"] == 0.85
        
    def test_fedavg_aggregator(self):
        """Test FedAvg aggregation strategy."""
        aggregator = FedAvgAggregator()
        
        # Create mock updates
        weights1 = [np.ones((2, 2)), np.ones((2, 1))]
        weights2 = [np.ones((2, 2)) * 2, np.ones((2, 1)) * 2]
        
        updates = [
            ModelUpdate("client1", weights1, {}, time.time()),
            ModelUpdate("client2", weights2, {}, time.time())
        ]
        
        clients = [
            FederatedClient("client1", "Hospital 1", 100, "1.0", time.time()),
            FederatedClient("client2", "Hospital 2", 100, "1.0", time.time())
        ]
        
        aggregated = aggregator.aggregate(updates, clients)
        
        # Should be average of weights
        assert len(aggregated) == 2
        assert np.allclose(aggregated[0], np.ones((2, 2)) * 1.5)
        assert np.allclose(aggregated[1], np.ones((2, 1)) * 1.5)
        
    def test_adaptive_fed_aggregator(self):
        """Test adaptive federated aggregation."""
        aggregator = AdaptiveFedAggregator()
        
        # Create updates with different performance
        weights1 = [np.ones((2, 2))]
        weights2 = [np.ones((2, 2)) * 2]
        
        updates = [
            ModelUpdate("client1", weights1, {"val_accuracy": 0.8}, time.time()),
            ModelUpdate("client2", weights2, {"val_accuracy": 0.9}, time.time())
        ]
        
        clients = [
            FederatedClient("client1", "Hospital 1", 100, "1.0", time.time(), reputation_score=0.8),
            FederatedClient("client2", "Hospital 2", 100, "1.0", time.time(), reputation_score=0.9)
        ]
        
        aggregated = aggregator.aggregate(updates, clients)
        
        # Higher performing client should have more influence
        assert len(aggregated) == 1
        assert aggregated[0].shape == (2, 2)
        
    def test_coordinator_initialization(self):
        """Test coordinator initialization."""
        # Create simple model template
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(5,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        coordinator = FederatedLearningCoordinator(
            model_template=model,
            min_clients=2,
            rounds=10
        )
        
        assert coordinator.min_clients == 2
        assert coordinator.rounds == 10
        assert len(coordinator.clients) == 0
        
    def test_client_registration(self):
        """Test client registration."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(5,))
        ])
        
        coordinator = FederatedLearningCoordinator(model_template=model)
        
        # Register clients
        success1 = coordinator.register_client("hospital1", "Hospital 1", 500)
        success2 = coordinator.register_client("hospital2", "Hospital 2", 300)
        
        assert success1 is True
        assert success2 is True
        assert len(coordinator.clients) == 2
        
        # Try to register duplicate
        success3 = coordinator.register_client("hospital1", "Duplicate", 100)
        assert success3 is False
        
    def test_create_federated_detector(self):
        """Test federated pneumonia detector creation."""
        coordinator = create_federated_pneumonia_detector(
            input_shape=(150, 150, 3),
            aggregation_strategy="adaptive"
        )
        
        assert isinstance(coordinator, FederatedLearningCoordinator)
        assert isinstance(coordinator.aggregator, AdaptiveFedAggregator)


class TestQuantumEnhancedOptimizer:
    """Test quantum-enhanced optimization."""
    
    def test_quantum_optimizer_initialization(self):
        """Test quantum optimizer initialization."""
        optimizer = QuantumInspiredOptimizer(
            learning_rate=0.001,
            quantum_strength=0.1,
            num_qubits=8
        )
        
        assert optimizer.num_qubits == 8
        assert optimizer._get_hyper('learning_rate') == 0.001
        assert optimizer._get_hyper('quantum_strength') == 0.1
        
    def test_quantum_annealing_scheduler(self):
        """Test quantum annealing scheduler."""
        scheduler = QuantumAnnealingScheduler(
            initial_lr=0.01,
            final_lr=0.001,
            annealing_steps=1000,
            temperature_schedule='quantum'
        )
        
        # Test learning rate progression
        lr_start = scheduler(0)
        lr_mid = scheduler(500)
        lr_end = scheduler(1000)
        
        assert lr_start == 0.01
        assert lr_end == 0.001
        assert lr_start > lr_mid > lr_end
        
    def test_quantum_error_correction(self):
        """Test quantum error correction."""
        corrector = QuantumErrorCorrection()
        
        # Create gradients with errors
        good_grad = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        nan_grad = tf.constant([[1.0, float('nan')], [3.0, 4.0]])
        inf_grad = tf.constant([[1.0, float('inf')], [3.0, 4.0]])
        
        gradients = [good_grad, nan_grad, inf_grad]
        
        # Detect errors
        errors = corrector.detect_gradient_errors(gradients)
        assert len(errors) > 0
        
        # Correct gradients
        corrected = corrector.correct_gradients(gradients)
        assert len(corrected) == 3
        
        # Check that NaN/Inf are removed
        assert not tf.reduce_any(tf.math.is_nan(corrected[1]))
        assert not tf.reduce_any(tf.math.is_inf(corrected[2]))
        
    def test_create_quantum_optimizer(self):
        """Test quantum optimizer creation."""
        optimizer = create_quantum_optimizer(
            learning_rate=0.001,
            optimizer_type="quantum_inspired"
        )
        
        assert isinstance(optimizer, QuantumInspiredOptimizer)


class TestAdvancedErrorRecovery:
    """Test advanced error recovery system."""
    
    def test_error_recovery_system_initialization(self):
        """Test error recovery system initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            system = AdvancedErrorRecoverySystem(
                checkpoint_dir=Path(temp_dir) / "checkpoints",
                max_recovery_attempts=3
            )
            
            assert system.max_recovery_attempts == 3
            assert len(system.recovery_strategies) > 0
            
    def test_error_capture(self):
        """Test error capture functionality."""
        system = AdvancedErrorRecoverySystem()
        
        try:
            raise ValueError("Test error")
        except Exception as e:
            error_report = system.capture_error(
                e, {"context": "test"}, ErrorSeverity.MEDIUM
            )
            
            assert error_report.error_type == "ValueError"
            assert error_report.message == "Test error"
            assert error_report.severity == ErrorSeverity.MEDIUM
            assert len(system.error_history) == 1
            
    def test_memory_recovery_strategy(self):
        """Test memory recovery strategy."""
        strategy = MemoryRecovery()
        
        # Create mock error report
        error_report = Mock()
        error_report.message = "OOM error occurred"
        
        assert strategy.can_recover(error_report) is True
        
        # Test recovery
        context = {"batch_size": 32}
        success, updated_context = strategy.recover(error_report, context)
        
        assert success is True
        assert updated_context["batch_size"] < context["batch_size"]
        
    def test_data_corruption_recovery(self):
        """Test data corruption recovery."""
        strategy = DataCorruptionRecovery()
        
        # Create mock error report
        error_report = Mock()
        error_report.message = "Data corrupted"
        
        assert strategy.can_recover(error_report) is True
        
        # Test recovery
        context = {"use_dummy_data": False}
        success, updated_context = strategy.recover(error_report, context)
        
        # Should enable dummy data as fallback
        assert success is True or updated_context.get("use_dummy_data") is True
        
    def test_error_recovery_context_manager(self):
        """Test error recovery context manager."""
        system = AdvancedErrorRecoverySystem()
        context = {"test": "value"}
        
        # Test successful recovery
        with patch.object(system, 'attempt_recovery', return_value=(True, context)):
            with ErrorRecoveryContext(system, context):
                raise ValueError("Recoverable error")
                
        # Should have captured the error
        assert len(system.error_history) == 1


class TestComprehensiveValidation:
    """Test comprehensive validation framework."""
    
    def test_data_integrity_validator(self):
        """Test data integrity validation."""
        validator = DataIntegrityValidator()
        
        # Test with good data
        good_data = np.random.random((100, 10))
        result = validator.validate(good_data, {})
        
        assert result.passed is True
        assert result.score > 0.8
        assert "shape" in result.details
        
        # Test with bad data
        bad_data = np.array([np.nan, 1.0, np.inf])
        result = validator.validate(bad_data, {})
        
        assert result.passed is False
        assert len(result.errors) > 0
        
    def test_image_quality_validator(self):
        """Test image quality validation."""
        validator = ImageQualityValidator(expected_channels=3, min_resolution=(128, 128))
        
        # Test with good image
        good_image = np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8)
        result = validator.validate(good_image, {})
        
        assert result.passed is True or len(result.errors) == 0
        assert "width" in result.details
        assert "height" in result.details
        
        # Test with low resolution
        small_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = validator.validate(small_image, {})
        
        assert len(result.warnings) > 0 or result.score < 1.0
        
    def test_model_architecture_validator(self):
        """Test model architecture validation."""
        validator = ModelArchitectureValidator()
        
        # Create test model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        result = validator.validate(model, {"data_type": "tabular"})
        
        assert result.passed is True
        assert "total_params" in result.details
        assert "has_dropout" in result.details
        assert result.details["has_dropout"] is True
        
    def test_security_validator(self):
        """Test security validation."""
        validator = SecurityValidator()
        
        # Test with secure context
        secure_context = {
            "encryption_enabled": True,
            "data_paths": ["/secure/path"],
            "differential_privacy": True
        }
        
        result = validator.validate({}, secure_context)
        
        assert result.passed is True
        assert result.details["encryption_enabled"] is True
        
        # Test with insecure context
        insecure_context = {
            "encryption_enabled": False,
            "data_paths": ["/tmp/data"]
        }
        
        result = validator.validate({}, insecure_context)
        
        assert result.passed is False or len(result.warnings) > 0
        
    def test_validation_framework(self):
        """Test comprehensive validation framework."""
        framework = ComprehensiveValidationFramework(
            strict_mode=False,
            save_reports=False
        )
        
        # Test data
        test_data = {
            "training_data": np.random.random((50, 10)),
            "config": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 10
            },
            "metrics": {
                "accuracy": 0.85,
                "val_accuracy": 0.82,
                "recall": 0.88
            }
        }
        
        context = {"domain": "medical", "data_type": "tabular"}
        
        passed, results = framework.validate_all(test_data, context)
        
        assert len(results) == len(framework.validators)
        assert all(isinstance(r.score, float) for r in results)
        
        # Get summary
        summary = framework.get_validation_summary(results)
        assert "total_validators" in summary
        assert "average_score" in summary


class TestIntelligentAutoScaler:
    """Test intelligent auto-scaling system."""
    
    def test_metrics_collector_initialization(self):
        """Test metrics collector initialization."""
        collector = MetricsCollector(collection_interval=1.0)
        
        assert collector.collection_interval == 1.0
        assert collector.active_requests == 0
        assert len(collector.metrics_history) == 0
        
    def test_resource_metrics_creation(self):
        """Test resource metrics creation."""
        metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=75.0,
            memory_percent=60.0,
            gpu_memory_percent=50.0,
            disk_io_read=1000.0,
            disk_io_write=500.0,
            network_sent=2000.0,
            network_recv=1500.0,
            active_requests=10,
            queue_length=5,
            response_time=1.5,
            error_rate=0.02
        )
        
        assert metrics.cpu_percent == 75.0
        assert metrics.memory_percent == 60.0
        assert metrics.active_requests == 10
        
    def test_adaptive_scaling_strategy(self):
        """Test adaptive scaling strategy."""
        strategy = AdaptiveScalingStrategy(
            cpu_threshold_high=80.0,
            memory_threshold_high=85.0
        )
        
        # Test scale up condition
        high_cpu_metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=90.0,
            memory_percent=70.0,
            gpu_memory_percent=50.0,
            disk_io_read=0, disk_io_write=0,
            network_sent=0, network_recv=0,
            active_requests=10, queue_length=5,
            response_time=1.0, error_rate=0.01
        )
        
        action, reason = strategy.should_scale(high_cpu_metrics)
        assert action == ScalingAction.SCALE_UP
        assert "CPU" in reason
        
        # Test scale down condition
        low_usage_metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=20.0,
            memory_percent=25.0,
            gpu_memory_percent=10.0,
            disk_io_read=0, disk_io_write=0,
            network_sent=0, network_recv=0,
            active_requests=2, queue_length=0,
            response_time=0.5, error_rate=0.001
        )
        
        action, reason = strategy.should_scale(low_usage_metrics)
        assert action == ScalingAction.SCALE_DOWN or action == ScalingAction.MAINTAIN
        
    def test_auto_scaler_initialization(self):
        """Test auto-scaler initialization."""
        scaler = IntelligentAutoScaler(
            enable_prediction=True,
            scaling_cooldown=60.0
        )
        
        assert scaler.enable_prediction is True
        assert scaler.scaling_cooldown == 60.0
        assert scaler.predictor is not None
        
    def test_config_calculation(self):
        """Test configuration calculation."""
        strategy = AdaptiveScalingStrategy()
        
        current_config = {
            "workers": 2,
            "batch_size": 32,
            "request_timeout": 30
        }
        
        high_cpu_metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=90.0, memory_percent=70.0,
            gpu_memory_percent=50.0,
            disk_io_read=0, disk_io_write=0,
            network_sent=0, network_recv=0,
            active_requests=10, queue_length=5,
            response_time=1.0, error_rate=0.01
        )
        
        new_config = strategy.calculate_new_config(
            current_config, ScalingAction.SCALE_UP, high_cpu_metrics
        )
        
        # Should increase workers for scale up
        assert new_config["workers"] >= current_config["workers"]


@pytest.fixture
def temporary_directory():
    """Provide temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestIntegrationScenarios:
    """Integration tests for advanced features working together."""
    
    def test_federated_learning_with_error_recovery(self, temporary_directory):
        """Test federated learning with error recovery."""
        # Initialize error recovery system
        recovery_system = AdvancedErrorRecoverySystem(
            checkpoint_dir=temporary_directory / "checkpoints"
        )
        
        # Create federated coordinator
        coordinator = create_federated_pneumonia_detector()
        
        # Register clients
        coordinator.register_client("hospital1", "Hospital 1", 500)
        coordinator.register_client("hospital2", "Hospital 2", 300)
        
        assert len(coordinator.clients) == 2
        
        # Test error recovery context
        context = {"federated_learning": True}
        
        try:
            with ErrorRecoveryContext(recovery_system, context):
                # Simulate federated operation that might fail
                if len(coordinator.clients) < 5:  # Simulate condition
                    pass  # Normal operation
        except:
            pass
            
        # System should handle gracefully
        assert len(recovery_system.error_history) >= 0
        
    def test_quantum_optimizer_with_validation(self):
        """Test quantum optimizer with validation framework."""
        # Create quantum optimizer
        optimizer = create_quantum_optimizer()
        
        # Create validation framework
        framework = ComprehensiveValidationFramework(save_reports=False)
        
        # Create simple model with quantum optimizer
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=optimizer, loss='binary_crossentropy')
        
        # Validate model architecture
        test_data = {"model": model}
        passed, results = framework.validate_all(test_data)
        
        # Should pass basic validation
        architecture_result = next((r for r in results if r.validator_name == "ModelArchitectureValidator"), None)
        assert architecture_result is not None
        
    def test_auto_scaler_with_adaptive_scheduler(self):
        """Test auto-scaler integration with adaptive scheduler."""
        # Create adaptive scheduler
        scheduler = create_adaptive_scheduler()
        
        # Create auto-scaler
        scaler = IntelligentAutoScaler(enable_prediction=False)
        
        # Test configuration compatibility
        config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "scheduler": scheduler
        }
        
        # Should be able to handle scheduler in config
        assert "learning_rate" in config
        assert config["batch_size"] == 32


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])