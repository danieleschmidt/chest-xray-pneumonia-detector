"""Tests for refactored train_pipeline functions."""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import MagicMock, patch, call
from dataclasses import dataclass

pytest.importorskip("tensorflow")
pytest.importorskip("mlflow")

from src.train_engine import (
    TrainingArgs,
    _setup_training_environment,
    _setup_mlflow_tracking,
    _execute_training_workflow,
    _cleanup_training_resources,
    train_pipeline,
)


class TestSetupTrainingEnvironment:
    """Test training environment setup functionality."""

    @pytest.fixture
    def training_args(self):
        """Create test training arguments."""
        return TrainingArgs(
            seed=42,
            use_dummy_data=True,
            train_dir=None,
            val_dir=None,
        )

    @patch('src.train_engine.config.ensure_directories')
    @patch('src.train_engine.tf.random.set_seed')
    @patch('src.train_engine.np.random.seed')
    @patch('src.train_engine.random.seed')
    @patch('src.train_engine._load_generators')
    def test_setup_training_environment_success(
        self, mock_load_generators, mock_random_seed, mock_np_seed, 
        mock_tf_seed, mock_ensure_dirs, training_args
    ):
        """Test successful training environment setup."""
        # Mock return values
        mock_train_gen = MagicMock()
        mock_val_gen = MagicMock()
        mock_dummy_base = "dummy_data"
        mock_load_generators.return_value = (mock_train_gen, mock_val_gen, mock_dummy_base)
        
        train_gen, val_gen, dummy_base = _setup_training_environment(training_args)
        
        # Verify directory setup
        mock_ensure_dirs.assert_called_once()
        
        # Verify random seeds
        mock_random_seed.assert_called_once_with(42)
        mock_np_seed.assert_called_once_with(42)
        mock_tf_seed.assert_called_once_with(42)
        
        # Verify generator loading
        mock_load_generators.assert_called_once_with(training_args)
        
        # Verify return values
        assert train_gen == mock_train_gen
        assert val_gen == mock_val_gen
        assert dummy_base == mock_dummy_base

    @patch('src.train_engine.config.ensure_directories')
    @patch('src.train_engine._load_generators')
    def test_setup_training_environment_custom_seed(
        self, mock_load_generators, mock_ensure_dirs
    ):
        """Test training environment setup with custom seed."""
        args = TrainingArgs(seed=123)
        mock_load_generators.return_value = (MagicMock(), MagicMock(), None)
        
        with patch('src.train_engine.random.seed') as mock_random_seed:
            _setup_training_environment(args)
            mock_random_seed.assert_called_once_with(123)


class TestSetupMLflowTracking:
    """Test MLflow tracking setup functionality."""

    @pytest.fixture
    def training_args(self):
        """Create test training arguments."""
        return TrainingArgs(
            mlflow_tracking_uri="http://localhost:5000",
            mlflow_experiment="test_experiment",
            mlflow_run_name="test_run",
            batch_size=32,
            epochs=10,
            use_attention_model=False,
            use_transfer_learning=True,
            base_model_name="MobileNetV2",
            class_weights=[1.0, 2.0],
            resume_checkpoint="checkpoint.keras"
        )

    @patch('src.train_engine.mlflow')
    def test_setup_mlflow_tracking_full_config(self, mock_mlflow, training_args):
        """Test MLflow setup with full configuration."""
        mock_context = MagicMock()
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_context
        
        with _setup_mlflow_tracking(training_args) as context:
            pass
        
        # Verify MLflow configuration
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
        mock_mlflow.set_experiment.assert_called_once_with("test_experiment")
        mock_mlflow.start_run.assert_called_once_with(run_name="test_run")
        
        # Verify parameter logging
        expected_calls = [
            call({"batch_size": 32}),
            call({"epochs_stage1": 10}),
            call({"use_attention_model": False}),
            call({"use_transfer_learning": True}),
            call({"base_model": "MobileNetV2"}),
        ]
        mock_mlflow.log_params.assert_called()
        
        # Verify additional parameter logging
        mock_mlflow.log_param.assert_any_call("class_weights_manual", [1.0, 2.0])
        mock_mlflow.log_param.assert_any_call("resume_checkpoint", "checkpoint.keras")

    @patch('src.train_engine.mlflow')
    def test_setup_mlflow_tracking_minimal_config(self, mock_mlflow):
        """Test MLflow setup with minimal configuration."""
        args = TrainingArgs(
            mlflow_tracking_uri=None,
            mlflow_experiment="minimal_experiment",
            mlflow_run_name=None,
            class_weights=None,
            resume_checkpoint=None
        )
        
        with _setup_mlflow_tracking(args):
            pass
        
        # Verify no tracking URI set when None
        mock_mlflow.set_tracking_uri.assert_not_called()
        mock_mlflow.set_experiment.assert_called_once_with("minimal_experiment")
        mock_mlflow.start_run.assert_called_once_with(run_name=None)


class TestExecuteTrainingWorkflow:
    """Test training workflow execution functionality."""

    @pytest.fixture
    def mock_components(self):
        """Create mock components for training workflow."""
        model = MagicMock()
        train_gen = MagicMock()
        val_gen = MagicMock()
        class_weights = {0: 1.0, 1: 2.0}
        args = TrainingArgs(
            resume_checkpoint="checkpoint.keras",
            checkpoint_path="checkpoint.keras"
        )
        return model, train_gen, val_gen, class_weights, args

    @patch('src.train_engine.os.path.exists')
    @patch('src.train_engine._train')
    @patch('src.train_engine._evaluate')
    def test_execute_training_workflow_with_checkpoint(
        self, mock_evaluate, mock_train, mock_exists, mock_components
    ):
        """Test training workflow execution with checkpoint loading."""
        model, train_gen, val_gen, class_weights, args = mock_components
        mock_exists.return_value = True
        mock_history = MagicMock()
        mock_train.return_value = mock_history
        
        _execute_training_workflow(model, train_gen, val_gen, class_weights, args)
        
        # Verify checkpoint loading
        mock_exists.assert_called_once_with("checkpoint.keras")
        model.load_weights.assert_called_once_with("checkpoint.keras")
        
        # Verify training and evaluation
        mock_train.assert_called_once_with(model, train_gen, val_gen, class_weights, args)
        mock_evaluate.assert_called_once_with(model, val_gen, mock_history, args)

    @patch('src.train_engine.os.path.exists')
    @patch('src.train_engine._train')
    @patch('src.train_engine._evaluate')
    def test_execute_training_workflow_no_checkpoint(
        self, mock_evaluate, mock_train, mock_exists, mock_components
    ):
        """Test training workflow execution without checkpoint."""
        model, train_gen, val_gen, class_weights, args = mock_components
        args.resume_checkpoint = None
        mock_history = MagicMock()
        mock_train.return_value = mock_history
        
        _execute_training_workflow(model, train_gen, val_gen, class_weights, args)
        
        # Verify no checkpoint loading
        mock_exists.assert_not_called()
        model.load_weights.assert_not_called()
        
        # Verify training and evaluation
        mock_train.assert_called_once_with(model, train_gen, val_gen, class_weights, args)
        mock_evaluate.assert_called_once_with(model, val_gen, mock_history, args)

    @patch('src.train_engine.os.path.exists')
    @patch('src.train_engine._train')
    @patch('src.train_engine._evaluate')
    def test_execute_training_workflow_checkpoint_load_failure(
        self, mock_evaluate, mock_train, mock_exists, mock_components
    ):
        """Test graceful handling of checkpoint loading failure."""
        model, train_gen, val_gen, class_weights, args = mock_components
        mock_exists.return_value = True
        model.load_weights.side_effect = Exception("Load failed")
        mock_history = MagicMock()
        mock_train.return_value = mock_history
        
        # Should not raise exception
        _execute_training_workflow(model, train_gen, val_gen, class_weights, args)
        
        # Verify training continues despite checkpoint failure
        mock_train.assert_called_once_with(model, train_gen, val_gen, class_weights, args)
        mock_evaluate.assert_called_once_with(model, val_gen, mock_history, args)


class TestCleanupTrainingResources:
    """Test training resource cleanup functionality."""

    @patch('src.train_engine.cleanup_dummy_data')
    def test_cleanup_training_resources_with_dummy_data(self, mock_cleanup):
        """Test cleanup when dummy data was used."""
        dummy_base = "path/to/dummy"
        
        _cleanup_training_resources(dummy_base)
        
        mock_cleanup.assert_called_once_with(base_dir=dummy_base)

    @patch('src.train_engine.cleanup_dummy_data')
    def test_cleanup_training_resources_no_dummy_data(self, mock_cleanup):
        """Test cleanup when no dummy data was used."""
        dummy_base = None
        
        _cleanup_training_resources(dummy_base)
        
        mock_cleanup.assert_not_called()


class TestTrainPipelineIntegration:
    """Test train_pipeline integration with refactored functions."""

    @pytest.fixture
    def training_args(self):
        """Create test training arguments."""
        return TrainingArgs(
            use_dummy_data=True,
            epochs=1,
            batch_size=2
        )

    @patch('src.train_engine._cleanup_training_resources')
    @patch('src.train_engine._execute_training_workflow')
    @patch('src.train_engine._compute_class_weights')
    @patch('src.train_engine._create_model')
    @patch('src.train_engine._setup_mlflow_tracking')
    @patch('src.train_engine._setup_training_environment')
    def test_train_pipeline_integration(
        self, mock_setup_env, mock_setup_mlflow, mock_create_model,
        mock_compute_weights, mock_execute_workflow, mock_cleanup,
        training_args
    ):
        """Test full train_pipeline integration."""
        # Mock return values
        mock_train_gen = MagicMock()
        mock_val_gen = MagicMock()
        mock_dummy_base = "dummy_data"
        mock_setup_env.return_value = (mock_train_gen, mock_val_gen, mock_dummy_base)
        
        mock_mlflow_context = MagicMock()
        mock_setup_mlflow.return_value.__enter__.return_value = mock_mlflow_context
        mock_setup_mlflow.return_value.__exit__.return_value = None
        
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model
        
        mock_class_weights = {0: 1.0, 1: 2.0}
        mock_compute_weights.return_value = mock_class_weights
        
        train_pipeline(training_args)
        
        # Verify function call sequence
        mock_setup_env.assert_called_once_with(training_args)
        mock_setup_mlflow.assert_called_once_with(training_args)
        mock_create_model.assert_called_once()
        mock_compute_weights.assert_called_once_with(mock_train_gen, training_args)
        mock_execute_workflow.assert_called_once_with(
            mock_model, mock_train_gen, mock_val_gen, mock_class_weights, training_args
        )
        mock_cleanup.assert_called_once_with(mock_dummy_base)

    @patch('src.train_engine._setup_training_environment')
    def test_train_pipeline_error_handling(self, mock_setup_env, training_args):
        """Test train_pipeline error handling."""
        mock_setup_env.side_effect = Exception("Setup failed")
        
        # Should propagate exceptions for proper error handling
        with pytest.raises(Exception, match="Setup failed"):
            train_pipeline(training_args)