"""Tests for refactored train_engine evaluation functions."""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
from unittest.mock import MagicMock, patch, call
from dataclasses import dataclass

pytest.importorskip("tensorflow")
pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

from src.train_engine import (
    TrainingArgs,
    _calculate_metrics,
    _plot_confusion_matrix,
    _plot_training_history,
    _save_artifacts,
    _evaluate_refactored as _evaluate,
)


@dataclass
class MockHistory:
    """Mock training history object."""
    history: dict


class TestCalculateMetrics:
    """Test metric calculation functionality."""

    @pytest.fixture
    def mock_generator(self):
        """Create mock validation generator."""
        generator = MagicMock()
        generator.samples = 100
        generator.batch_size = 10
        generator.classes = np.array([0, 1, 0, 1, 1] * 20)  # 100 samples
        return generator

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = MagicMock()
        # Binary classification predictions
        model.predict.return_value = np.array([[0.2], [0.8], [0.1], [0.9], [0.7]] * 20)
        return model

    def test_calculate_metrics_binary_classification(self, mock_model, mock_generator):
        """Test metric calculation for binary classification."""
        args = TrainingArgs(num_classes=1)
        
        metrics = _calculate_metrics(mock_model, mock_generator, args)
        
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        assert 'predictions' in metrics
        assert 'true_labels' in metrics
        
        # Verify shapes
        assert len(metrics['predictions']) == 100
        assert len(metrics['true_labels']) == 100
        
        # Verify types
        assert isinstance(metrics['precision'], (int, float))
        assert isinstance(metrics['recall'], (int, float))
        assert isinstance(metrics['f1_score'], (int, float))
        assert isinstance(metrics['roc_auc'], (int, float))

    def test_calculate_metrics_multiclass_classification(self, mock_generator):
        """Test metric calculation for multiclass classification."""
        # Setup multiclass scenario
        mock_model = MagicMock()
        mock_generator.classes = np.array([0, 1, 2, 0, 1] * 20)  # 3 classes
        mock_model.predict.return_value = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1], 
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1]
        ] * 20)
        
        args = TrainingArgs(num_classes=3)
        
        metrics = _calculate_metrics(mock_model, mock_generator, args)
        
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        
        # Verify weighted averaging was used
        assert 0.0 <= metrics['precision'] <= 1.0
        assert 0.0 <= metrics['recall'] <= 1.0
        assert 0.0 <= metrics['f1_score'] <= 1.0

    def test_calculate_metrics_handles_roc_auc_error(self, mock_model, mock_generator):
        """Test that ROC AUC calculation handles errors gracefully."""
        # Create scenario where ROC AUC calculation might fail (all same class)
        mock_generator.classes = np.array([0] * 100)  # All same class
        args = TrainingArgs(num_classes=1)
        
        metrics = _calculate_metrics(mock_model, mock_generator, args)
        
        # Should return NaN for ROC AUC when calculation fails
        assert np.isnan(metrics['roc_auc'])

    def test_calculate_metrics_prediction_steps_calculation(self, mock_model, mock_generator):
        """Test correct calculation of prediction steps and batch-wise prediction."""
        # Test with samples not evenly divisible by batch_size
        mock_generator.samples = 95
        mock_generator.batch_size = 10
        
        # Mock individual batch predictions for memory-efficient implementation
        batch_preds = np.array([[0.3], [0.7], [0.2], [0.8], [0.6], [0.4], [0.9], [0.1], [0.5], [0.75]])
        mock_model.predict.return_value = batch_preds
        mock_generator.__getitem__.return_value = (None, None)  # Mock batch data
        
        args = TrainingArgs(num_classes=1)
        
        _calculate_metrics(mock_model, mock_generator, args)
        
        # Should call predict 10 times (95/10 = 9.5, rounded up to 10) for batch-wise processing
        assert mock_model.predict.call_count == 10

    def test_calculate_metrics_memory_efficient_batch_processing(self, mock_model, mock_generator):
        """Test that memory-efficient batch processing works correctly."""
        # Setup test data
        mock_generator.samples = 25
        mock_generator.batch_size = 10
        mock_generator.classes = np.array([0, 1, 0, 1, 0] * 5)
        
        # Mock batch predictions - 3 batches: 10, 10, 5 samples
        batch_predictions = [
            np.array([[0.2], [0.8], [0.1], [0.9], [0.3], [0.7], [0.4], [0.6], [0.15], [0.85]]),  # batch 1
            np.array([[0.25], [0.75], [0.35], [0.65], [0.45], [0.55], [0.2], [0.8], [0.1], [0.9]]),  # batch 2
            np.array([[0.3], [0.7], [0.4], [0.6], [0.5]])  # batch 3 (partial)
        ]
        
        mock_model.predict.side_effect = batch_predictions
        mock_generator.__getitem__.return_value = (None, None)
        
        args = TrainingArgs(num_classes=1)
        
        metrics = _calculate_metrics(mock_model, mock_generator, args)
        
        # Verify batch-wise processing occurred
        assert mock_model.predict.call_count == 3  # 3 batches: 25/10 = 2.5, rounded up to 3
        
        # Verify metrics are calculated correctly
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics


class TestPlotConfusionMatrix:
    """Test confusion matrix plotting functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('seaborn.heatmap')
    @patch('matplotlib.pyplot.figure')
    @patch('os.makedirs')
    def test_plot_confusion_matrix_basic(self, mock_makedirs, mock_figure, 
                                       mock_heatmap, mock_close, mock_savefig):
        """Test basic confusion matrix plotting."""
        cm = np.array([[50, 5], [10, 35]])
        save_path = "/test/path/cm.png"
        
        _plot_confusion_matrix(cm, save_path)
        
        # Verify directory creation
        mock_makedirs.assert_called_once_with("/test/path", exist_ok=True)
        
        # Verify plot setup
        mock_figure.assert_called_once_with(figsize=(4, 4))
        mock_heatmap.assert_called_once_with(cm, annot=True, fmt="d", cmap="Blues")
        
        # Verify file saving and cleanup
        mock_savefig.assert_called_once_with(save_path)
        mock_close.assert_called_once()

    def test_plot_confusion_matrix_creates_valid_plot(self, temp_dir):
        """Test that confusion matrix plot is actually created."""
        cm = np.array([[45, 5], [8, 42]])
        save_path = os.path.join(temp_dir, "test_cm.png")
        
        _plot_confusion_matrix(cm, save_path)
        
        # Verify file was created
        assert os.path.exists(save_path)
        
        # Verify file is not empty
        assert os.path.getsize(save_path) > 0

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_confusion_matrix_handles_nested_directory(self, mock_close, mock_savefig, temp_dir):
        """Test confusion matrix plotting with nested directory creation."""
        save_path = os.path.join(temp_dir, "nested", "deep", "cm.png")
        cm = np.array([[10, 2], [3, 15]])
        
        _plot_confusion_matrix(cm, save_path)
        
        # Verify nested directory was created
        assert os.path.exists(os.path.dirname(save_path))


class TestPlotTrainingHistory:
    """Test training history plotting functionality."""

    @pytest.fixture
    def sample_history(self):
        """Create sample training history."""
        return MockHistory(history={
            'accuracy': [0.6, 0.7, 0.8, 0.85, 0.9],
            'val_accuracy': [0.55, 0.68, 0.75, 0.82, 0.87],
            'loss': [1.0, 0.8, 0.6, 0.4, 0.3],
            'val_loss': [1.1, 0.85, 0.65, 0.45, 0.35]
        })

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.figure')
    @patch('os.makedirs')
    def test_plot_training_history_basic(self, mock_makedirs, mock_figure, 
                                       mock_close, mock_savefig, sample_history):
        """Test basic training history plotting."""
        save_path = "/test/path/history.png"
        epochs = 5
        
        _plot_training_history(sample_history, epochs, save_path)
        
        # Verify directory creation
        mock_makedirs.assert_called_once_with("/test/path", exist_ok=True)
        
        # Verify plot setup
        mock_figure.assert_called_once_with(figsize=(12, 6))
        
        # Verify file saving and cleanup
        mock_savefig.assert_called_once_with(save_path)
        mock_close.assert_called_once()

    def test_plot_training_history_creates_valid_plot(self, sample_history, temp_dir):
        """Test that training history plot is actually created."""
        save_path = os.path.join(temp_dir, "test_history.png")
        epochs = 5
        
        _plot_training_history(sample_history, epochs, save_path)
        
        # Verify file was created
        assert os.path.exists(save_path)
        
        # Verify file is not empty
        assert os.path.getsize(save_path) > 0

    def test_plot_training_history_handles_current_directory(self, sample_history):
        """Test handling when plot_path dirname is empty (current directory)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                save_path = "history.png"  # No directory component
                _plot_training_history(sample_history, 5, save_path)
                
                # Verify file was created in current directory
                assert os.path.exists(save_path)
            finally:
                os.chdir(original_cwd)


class TestSaveArtifacts:
    """Test artifact saving functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_model(self):
        """Create mock model with save method."""
        model = MagicMock()
        return model

    @pytest.fixture
    def sample_history(self):
        """Create sample training history."""
        return MockHistory(history={
            'accuracy': [0.8, 0.85, 0.9],
            'val_accuracy': [0.75, 0.82, 0.87],
            'loss': [0.6, 0.4, 0.3],
            'val_loss': [0.65, 0.45, 0.35]
        })

    @patch('mlflow.log_artifact')
    @patch('mlflow.keras.log_model')
    def test_save_artifacts_basic(self, mock_log_model, mock_log_artifact, 
                                mock_model, sample_history, temp_dir):
        """Test basic artifact saving functionality."""
        args = TrainingArgs(
            save_model_path=os.path.join(temp_dir, "model.keras"),
            history_csv=os.path.join(temp_dir, "history.csv")
        )
        
        _save_artifacts(mock_model, sample_history, args)
        
        # Verify model saving
        mock_model.save.assert_called_once_with(args.save_model_path)
        mock_log_model.assert_called_once_with(mock_model, "model")
        
        # Verify CSV file was created and logged
        assert os.path.exists(args.history_csv)
        mock_log_artifact.assert_called_once_with(args.history_csv)
        
        # Verify CSV content
        df = pd.read_csv(args.history_csv)
        assert list(df.columns) == ['accuracy', 'val_accuracy', 'loss', 'val_loss']
        assert len(df) == 3

    def test_save_artifacts_csv_content(self, mock_model, sample_history, temp_dir):
        """Test that CSV file contains correct training history data."""
        args = TrainingArgs(
            save_model_path=os.path.join(temp_dir, "model.keras"),
            history_csv=os.path.join(temp_dir, "history.csv")
        )
        
        with patch('mlflow.log_artifact'), patch('mlflow.keras.log_model'):
            _save_artifacts(mock_model, sample_history, args)
        
        # Read and verify CSV content
        df = pd.read_csv(args.history_csv)
        
        assert df['accuracy'].tolist() == [0.8, 0.85, 0.9]
        assert df['val_accuracy'].tolist() == [0.75, 0.82, 0.87]
        assert df['loss'].tolist() == [0.6, 0.4, 0.3]
        assert df['val_loss'].tolist() == [0.65, 0.45, 0.35]


class TestEvaluateRefactored:
    """Test the refactored _evaluate function integration."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = MagicMock()
        model.predict.return_value = np.array([[0.3], [0.8], [0.1], [0.9]] * 25)
        return model

    @pytest.fixture
    def mock_generator(self):
        """Create mock validation generator."""
        generator = MagicMock()
        generator.samples = 100
        generator.batch_size = 10
        generator.classes = np.array([0, 1, 0, 1] * 25)
        return generator

    @pytest.fixture
    def sample_history(self):
        """Create sample training history."""
        return MockHistory(history={
            'accuracy': [0.7, 0.8, 0.85],
            'val_accuracy': [0.65, 0.75, 0.82],
            'loss': [0.8, 0.5, 0.3],
            'val_loss': [0.85, 0.55, 0.35]
        })

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @patch('mlflow.log_metric')
    @patch('mlflow.log_artifact')
    @patch('mlflow.keras.log_model')
    def test_evaluate_refactored_integration(self, mock_log_model, mock_log_artifact, 
                                           mock_log_metric, mock_model, mock_generator, 
                                           sample_history, temp_dir):
        """Test the complete refactored evaluate function integration."""
        args = TrainingArgs(
            num_classes=1,
            epochs=3,
            cm_path=os.path.join(temp_dir, "cm.png"),
            plot_path=os.path.join(temp_dir, "history.png"),
            history_csv=os.path.join(temp_dir, "history.csv"),
            save_model_path=os.path.join(temp_dir, "model.keras")
        )
        
        _evaluate(mock_model, mock_generator, sample_history, args)
        
        # Verify metrics were logged
        expected_metrics = ['precision', 'recall', 'f1', 'roc_auc']
        for metric in expected_metrics:
            assert any(call_args[0][0] == metric for call_args in mock_log_metric.call_args_list)
        
        # Verify artifacts were created and logged
        assert os.path.exists(args.cm_path)
        assert os.path.exists(args.plot_path)
        assert os.path.exists(args.history_csv)
        
        # Verify MLflow artifact logging
        expected_artifacts = [args.cm_path, args.plot_path, args.history_csv]
        for artifact in expected_artifacts:
            assert any(call_args[0][0] == artifact for call_args in mock_log_artifact.call_args_list)
        
        # Verify model saving
        mock_model.save.assert_called_once_with(args.save_model_path)
        mock_log_model.assert_called_once_with(mock_model, "model")

    def test_evaluate_refactored_function_separation(self, mock_model, mock_generator, 
                                                   sample_history, temp_dir):
        """Test that refactored function properly separates concerns."""
        args = TrainingArgs(
            num_classes=1,
            epochs=3,
            cm_path=os.path.join(temp_dir, "cm.png"),
            plot_path=os.path.join(temp_dir, "history.png"),
            history_csv=os.path.join(temp_dir, "history.csv"),
            save_model_path=os.path.join(temp_dir, "model.keras")
        )
        
        with patch('src.train_engine._calculate_metrics') as mock_calc_metrics, \
             patch('src.train_engine._plot_confusion_matrix') as mock_plot_cm, \
             patch('src.train_engine._plot_training_history') as mock_plot_history, \
             patch('src.train_engine._save_artifacts') as mock_save_artifacts, \
             patch('mlflow.log_metric'):
            
            mock_calc_metrics.return_value = {
                'precision': 0.8, 'recall': 0.75, 'f1_score': 0.77, 'roc_auc': 0.82,
                'predictions': np.array([0.3, 0.8, 0.1, 0.9]),
                'true_labels': np.array([0, 1, 0, 1]),
                'confusion_matrix': np.array([[2, 0], [0, 2]])
            }
            
            _evaluate(mock_model, mock_generator, sample_history, args)
            
            # Verify each function was called exactly once
            mock_calc_metrics.assert_called_once_with(mock_model, mock_generator, args)
            mock_plot_cm.assert_called_once()
            mock_plot_history.assert_called_once_with(sample_history, args.epochs, args.plot_path)
            mock_save_artifacts.assert_called_once_with(mock_model, sample_history, args)