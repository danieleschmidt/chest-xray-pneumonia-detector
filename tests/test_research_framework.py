# Comprehensive Tests for Research Framework
# Tests novel architectures, experimental framework, and validation systems

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from research.novel_architectures import (
    DualPathCNN, HierarchicalAttentionCNN, UncertaintyAwareCNN,
    create_ensemble_model, build_research_baseline_models
)
from research.experimental_framework import (
    ExperimentConfig, ExperimentRunner, ResourceMonitor,
    create_synthetic_dataset
)
from validation.comprehensive_validators import (
    ImageQualityValidator, ModelArchitectureValidator,
    DatasetValidator, ModelPerformanceValidator,
    ComprehensiveValidationSuite
)
from security.medical_data_protection import (
    MedicalDataEncryption, AccessControlManager, AuditLogger,
    create_secure_medical_ai_system
)
from optimization.model_acceleration import (
    GPUOptimizer, ModelPruner, PredictionCache,
    DistributedInferenceEngine
)
from optimization.adaptive_scaling import (
    MetricsCollector, AutoScaler, AdaptiveScalingManager
)


class TestNovelArchitectures:
    """Test novel CNN architectures."""
    
    @pytest.fixture
    def dummy_input(self):
        return tf.random.normal((2, 224, 224, 3))
    
    def test_dual_path_cnn_creation(self):
        """Test DualPathCNN model creation."""
        model = DualPathCNN(input_shape=(224, 224, 3), num_classes=1)
        assert model is not None
        assert hasattr(model, 'spatial_conv1')
        assert hasattr(model, 'texture_conv1')
        assert hasattr(model, 'cross_attention')
    
    def test_dual_path_cnn_forward_pass(self, dummy_input):
        """Test DualPathCNN forward pass."""
        model = DualPathCNN()
        output = model(dummy_input)
        
        assert output.shape == (2, 1)  # Binary classification
        assert tf.reduce_all(output >= 0) and tf.reduce_all(output <= 1)  # Sigmoid output
    
    def test_hierarchical_attention_cnn(self, dummy_input):
        """Test HierarchicalAttentionCNN."""
        with patch('tensorflow.keras.applications.EfficientNetV2B0') as mock_backbone:
            # Mock the backbone to avoid loading pretrained weights
            mock_backbone.return_value = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.GlobalAveragePooling2D()
            ])
            
            model = HierarchicalAttentionCNN()
            # Test basic structure
            assert hasattr(model, 'fine_attention')
            assert hasattr(model, 'medium_attention')
            assert hasattr(model, 'coarse_attention')
    
    def test_uncertainty_aware_cnn(self, dummy_input):
        """Test UncertaintyAwareCNN."""
        with patch('tensorflow.keras.applications.EfficientNetV2B0') as mock_backbone:
            mock_backbone.return_value = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.GlobalAveragePooling2D()
            ])
            
            model = UncertaintyAwareCNN()
            output = model(dummy_input)
            
            assert isinstance(output, dict)
            assert 'prediction' in output
            assert 'uncertainty' in output
    
    def test_ensemble_model_creation(self):
        """Test ensemble model creation."""
        with patch.multiple(
            'tensorflow.keras.applications',
            EfficientNetV2B0=Mock(return_value=tf.keras.layers.Lambda(lambda x: x)),
            DenseNet121=Mock(return_value=tf.keras.layers.Lambda(lambda x: x)),
            ResNet50V2=Mock(return_value=tf.keras.layers.Lambda(lambda x: x))
        ):
            model = create_ensemble_model()
            assert model is not None
            assert model.name == 'EnsembleCNN'
    
    def test_build_research_baseline_models(self):
        """Test building research baseline models."""
        with patch.multiple(
            'tensorflow.keras.applications',
            EfficientNetV2B0=Mock(return_value=tf.keras.layers.Lambda(lambda x: x)),
            DenseNet121=Mock(return_value=tf.keras.layers.Lambda(lambda x: x)),
            ResNet50V2=Mock(return_value=tf.keras.layers.Lambda(lambda x: x))
        ):
            models = build_research_baseline_models()
            
            expected_models = [
                'efficientnet_baseline', 'resnet_baseline', 'densenet_baseline',
                'dual_path_cnn', 'hierarchical_attention', 'ensemble_cnn', 'uncertainty_aware'
            ]
            
            for model_name in expected_models:
                assert model_name in models
                assert models[model_name] is not None


class TestExperimentalFramework:
    """Test experimental framework components."""
    
    def test_experiment_config_creation(self):
        """Test ExperimentConfig creation."""
        config = ExperimentConfig(
            name="test_experiment",
            description="Test experiment",
            models=['model1', 'model2'],
            datasets=['dataset1']
        )
        
        assert config.name == "test_experiment"
        assert len(config.random_seeds) == 3  # Default
        assert config.num_runs == 3  # Default
        assert 'accuracy' in config.metrics
    
    def test_resource_monitor_context_manager(self):
        """Test ResourceMonitor context manager."""
        monitor = ResourceMonitor()
        
        with monitor.monitor():
            # Simulate some work
            time.sleep(0.1)
        
        assert hasattr(monitor, 'training_time')
        assert hasattr(monitor, 'memory_usage')
        assert monitor.training_time > 0
    
    def test_create_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        X, y = create_synthetic_dataset(num_samples=100, image_size=(64, 64))
        
        assert X.shape == (100, 64, 64, 3)
        assert y.shape == (100,)
        assert np.all(np.isin(y, [0, 1]))  # Binary labels
        
        # Check for class balance (approximately)
        class_balance = np.mean(y)
        assert 0.3 < class_balance < 0.7  # Reasonable balance
    
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metrics')
    def test_experiment_runner_single_experiment(self, mock_log_metrics, mock_log_params, mock_start_run):
        """Test running single experiment."""
        config = ExperimentConfig("test", "test", ['model1'], ['dataset1'], num_runs=1)
        runner = ExperimentRunner(config, output_dir=tempfile.mkdtemp())
        
        # Mock model factory
        def mock_model_factory():
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            return model
        
        # Create dummy data
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.randn(20, 10)
        y_val = np.random.randint(0, 2, 20)
        X_test = np.random.randn(20, 10)
        y_test = np.random.randint(0, 2, 20)
        
        # Mock MLflow context manager
        mock_start_run.return_value.__enter__ = Mock()
        mock_start_run.return_value.__exit__ = Mock()
        
        result = runner.run_single_experiment(
            mock_model_factory,
            (X_train, y_train),
            (X_val, y_val),
            (X_test, y_test),
            "test_model",
            "test_dataset",
            0,
            42
        )
        
        assert result is not None
        assert result.model_name == "test_model"
        assert result.dataset_name == "test_dataset"
        assert 'accuracy' in result.metrics


class TestComprehensiveValidators:
    """Test validation framework."""
    
    def test_image_quality_validator_creation(self):
        """Test ImageQualityValidator creation."""
        validator = ImageQualityValidator(
            min_resolution=(128, 128),
            max_resolution=(2048, 2048),
            required_channels=3
        )
        
        assert validator.min_resolution == (128, 128)
        assert validator.max_resolution == (2048, 2048)
        assert validator.required_channels == 3
    
    @patch('PIL.Image.open')
    def test_image_validation_success(self, mock_open):
        """Test successful image validation."""
        # Mock PIL Image
        mock_img = Mock()
        mock_img.size = (512, 512)
        mock_img.mode = 'RGB'
        mock_img.format = 'JPEG'
        mock_open.return_value.__enter__ = Mock(return_value=mock_img)
        mock_open.return_value.__exit__ = Mock()
        
        # Mock PIL ImageStat
        with patch('PIL.ImageStat.Stat') as mock_stat:
            mock_stat.return_value.mean = [128, 128, 128]
            mock_stat.return_value.stddev = [50, 50, 50]
            
            # Mock numpy array conversion
            with patch('numpy.array', return_value=np.random.randint(0, 255, (512, 512, 3))):
                with patch('pathlib.Path.stat') as mock_stat_path:
                    mock_stat_path.return_value.st_size = 1024 * 1024  # 1MB
                    
                    validator = ImageQualityValidator()
                    result = validator.validate_image('test_image.jpg')
                    
                    assert result is not None
                    assert hasattr(result, 'is_valid')
                    assert hasattr(result, 'score')
                    assert hasattr(result, 'issues')
    
    def test_model_architecture_validator(self):
        """Test ModelArchitectureValidator."""
        validator = ModelArchitectureValidator()
        
        # Create test model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        result = validator.validate_model(model)
        
        assert result is not None
        assert result.score > 0.5  # Should be reasonably good architecture
        assert 'total_layers' in result.metadata
        assert 'trainable_params' in result.metadata
    
    def test_dataset_validator(self):
        """Test DatasetValidator."""
        validator = DatasetValidator()
        
        # Create test dataset
        X = np.random.randn(500, 224, 224, 3)
        y = np.random.randint(0, 2, 500)
        
        result = validator.validate_dataset(X, y, "test_dataset")
        
        assert result is not None
        assert result.metadata['num_samples'] == 500
        assert result.metadata['num_classes'] == 2
        assert 'class_distribution' in result.metadata
    
    def test_model_performance_validator(self):
        """Test ModelPerformanceValidator."""
        validator = ModelPerformanceValidator()
        
        # Create test predictions
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 1, 1, 0, 0])  # Some errors
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.6, 0.7, 0.3, 0.4])
        
        result = validator.validate_performance(y_true, y_pred, y_prob)
        
        assert result is not None
        assert 'accuracy' in result.metadata
        assert 'precision' in result.metadata
        assert 'recall' in result.metadata
        assert 'f1_score' in result.metadata
    
    def test_comprehensive_validation_suite(self):
        """Test ComprehensiveValidationSuite."""
        suite = ComprehensiveValidationSuite()
        
        # Create test components
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(50, 10)
        y_test = np.random.randint(0, 2, 50)
        predictions = np.random.rand(50, 1)
        
        results = suite.run_full_validation(
            model=model,
            train_data=(X_train, y_train),
            test_data=(X_test, y_test),
            predictions=predictions
        )
        
        assert results is not None
        assert 'model_validation' in results
        assert 'dataset_validation' in results
        assert 'performance_validation' in results
        assert 'overall_score' in results
        assert isinstance(results['overall_score'], float)


class TestMedicalDataProtection:
    """Test medical data protection and security."""
    
    def test_medical_data_encryption(self):
        """Test MedicalDataEncryption."""
        password = b'test_password_123'
        encryption = MedicalDataEncryption(password)
        
        # Test string encryption
        test_data = "Patient has pneumonia"
        encrypted = encryption.encrypt_data(test_data)
        decrypted = encryption.decrypt_data(encrypted)
        
        assert decrypted.decode() == test_data
        assert encrypted != test_data.encode()
    
    def test_access_control_manager(self):
        """Test AccessControlManager."""
        acm = AccessControlManager()
        
        # Create user
        success = acm.create_user('dr_test', 'radiologist', 'Dr. Test')
        assert success
        
        # Authenticate user
        session_id = acm.authenticate_user('dr_test', 'password')
        assert session_id is not None
        
        # Check permissions
        can_view = acm.check_permission(session_id, 'view_images')
        assert can_view
        
        cannot_admin = acm.check_permission(session_id, 'manage_users')
        assert not cannot_admin
    
    def test_audit_logger_creation(self):
        """Test AuditLogger creation."""
        with tempfile.NamedTemporaryFile(suffix='.log') as tmp_file:
            logger = AuditLogger(tmp_file.name)
            assert logger is not None
            assert logger.log_file.exists()
    
    def test_secure_medical_ai_system_creation(self):
        """Test secure medical AI system factory."""
        system = create_secure_medical_ai_system()
        
        required_components = [
            'access_control', 'audit_logger', 'data_encryption',
            'data_anonymizer', 'data_validator'
        ]
        
        for component in required_components:
            assert component in system
            assert system[component] is not None


class TestModelOptimization:
    """Test model optimization components."""
    
    def test_gpu_optimizer_creation(self):
        """Test GPUOptimizer creation."""
        optimizer = GPUOptimizer()
        assert optimizer is not None
    
    def test_prediction_cache(self):
        """Test PredictionCache functionality."""
        cache = PredictionCache(max_size=10, ttl_seconds=1)
        
        # Test cache miss
        test_input = np.random.randn(5, 5)
        result = cache.get(test_input)
        assert result is None
        
        # Test cache put and hit
        test_prediction = np.array([0.7])
        cache.put(test_input, test_prediction)
        
        cached_result = cache.get(test_input)
        assert cached_result is not None
        np.testing.assert_array_equal(cached_result, test_prediction)
        
        # Test cache stats
        stats = cache.get_stats()
        assert stats['cache_size'] == 1
        assert stats['hits'] == 1
        assert stats['misses'] == 1
    
    def test_distributed_inference_engine(self):
        """Test DistributedInferenceEngine."""
        # Create simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        engine = DistributedInferenceEngine(model, num_workers=2)
        
        # Test batch prediction
        X_batch = np.random.randn(20, 5)
        predictions, metrics = engine.predict_batch_parallel(X_batch, use_cache=False)
        
        assert predictions.shape[0] == 20
        assert metrics.throughput > 0
        assert metrics.inference_time > 0


class TestAdaptiveScaling:
    """Test adaptive scaling components."""
    
    def test_metrics_collector_creation(self):
        """Test MetricsCollector creation."""
        collector = MetricsCollector(collection_interval=0.1)
        assert collector is not None
        assert collector.collection_interval == 0.1
    
    def test_auto_scaler_creation(self):
        """Test AutoScaler creation."""
        scaler = AutoScaler(
            min_workers=1,
            max_workers=5,
            target_cpu_usage=70.0,
            cooldown_period=10
        )
        
        assert scaler.min_workers == 1
        assert scaler.max_workers == 5
        assert scaler.current_workers == 1
    
    @patch('psutil.cpu_percent', return_value=85.0)
    @patch('psutil.virtual_memory')
    def test_scaling_decision_scale_up(self, mock_memory, mock_cpu):
        """Test scaling decision for scale up."""
        from optimization.adaptive_scaling import ResourceMetrics
        from datetime import datetime
        
        mock_memory.return_value.percent = 90.0
        
        scaler = AutoScaler(cooldown_period=0)  # No cooldown for test
        
        current_metrics = ResourceMetrics(
            timestamp=datetime.now(),
            cpu_usage=85.0,
            memory_usage=90.0,
            gpu_usage=80.0,
            gpu_memory=70.0,
            disk_io=10.0,
            network_io=5.0,
            active_requests=20,
            queue_depth=15,
            response_time=8.0
        )
        
        predicted_load = {'cpu': 88.0, 'memory': 85.0, 'requests': 25, 'confidence': 0.8}
        
        decision = scaler.make_scaling_decision(current_metrics, predicted_load)
        
        assert decision.action == "scale_up"
        assert decision.target_workers > scaler.current_workers
        assert decision.confidence > 0.5
    
    def test_adaptive_scaling_manager_creation(self):
        """Test AdaptiveScalingManager creation."""
        config = {
            'min_workers': 2,
            'max_workers': 8,
            'target_cpu_usage': 60.0
        }
        
        manager = AdaptiveScalingManager(config, monitoring_interval=1.0)
        
        assert manager.auto_scaler.min_workers == 2
        assert manager.auto_scaler.max_workers == 8
        assert manager.monitoring_interval == 1.0


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_end_to_end_model_pipeline(self):
        """Test end-to-end model pipeline."""
        # Create model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=(64, 64, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Create synthetic data
        X_train = np.random.randn(100, 64, 64, 3)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 64, 64, 3)
        y_test = np.random.randint(0, 2, 20)
        
        # Validate dataset
        dataset_validator = DatasetValidator()
        dataset_result = dataset_validator.validate_dataset(X_train, y_train)
        assert dataset_result.score > 0.5
        
        # Validate model architecture
        model_validator = ModelArchitectureValidator()
        model_result = model_validator.validate_model(model)
        assert model_result.score > 0.5
        
        # Test inference
        predictions = model.predict(X_test, verbose=0)
        assert predictions.shape == (20, 1)
        
        # Validate performance
        y_pred = (predictions > 0.5).astype(int).flatten()
        perf_validator = ModelPerformanceValidator()
        perf_result = perf_validator.validate_performance(y_test, y_pred, predictions.flatten())
        
        # Performance might be poor with random data, but should not crash
        assert perf_result is not None
        assert 'accuracy' in perf_result.metadata
    
    def test_security_and_validation_integration(self):
        """Test integration of security and validation systems."""
        # Create secure system
        secure_system = create_secure_medical_ai_system()
        
        # Authenticate user
        session_id = secure_system['access_control'].authenticate_user('dr_smith', 'password')
        assert session_id is not None
        
        # Test data validation
        test_data = {'patient_id': 'TEST123', 'diagnosis': 'pneumonia'}
        sanitized_data = secure_system['data_validator'].sanitize_input_data(test_data)
        
        assert 'patient_id' in sanitized_data
        assert sanitized_data['patient_id'] == 'TEST123'
    
    @patch('tensorflow.keras.applications.EfficientNetV2B0')
    def test_research_framework_integration(self, mock_backbone):
        """Test research framework integration."""
        # Mock backbone to avoid loading pretrained weights
        mock_backbone.return_value = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D()
        ])
        
        # Create experiment config
        config = ExperimentConfig(
            name="integration_test",
            description="Test integration",
            models=['simple_model'],
            datasets=['synthetic'],
            num_runs=1
        )
        
        # Create runner
        runner = ExperimentRunner(config, output_dir=tempfile.mkdtemp())
        
        # Test with minimal setup
        assert runner.config.name == "integration_test"
        assert len(runner.results) == 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])