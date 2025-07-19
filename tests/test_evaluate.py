import subprocess
import sys
import pytest
import tempfile
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock
from pathlib import Path

pytest.importorskip("tensorflow")

from src.evaluate import evaluate_predictions  # noqa: E402


class TestEvaluatePredictions:
    """Test suite for evaluate_predictions function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_binary_data(self, temp_dir):
        """Create sample binary classification data."""
        # Predictions CSV
        pred_data = pd.DataFrame({
            'filepath': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg'],
            'prediction': [0.8, 0.3, 0.9, 0.2],
            'label': [1, 0, 1, 0]
        })
        pred_csv = os.path.join(temp_dir, 'predictions.csv')
        pred_data.to_csv(pred_csv, index=False)
        
        return pred_csv, pred_data

    @pytest.fixture
    def sample_multiclass_data(self, temp_dir):
        """Create sample multi-class classification data."""
        # Multi-class predictions CSV
        pred_data = pd.DataFrame({
            'filepath': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg'],
            'prediction': [1, 0, 2, 0],
            'prob_0': [0.2, 0.8, 0.1, 0.9],
            'prob_1': [0.7, 0.15, 0.1, 0.08],
            'prob_2': [0.1, 0.05, 0.8, 0.02],
            'label': [1, 0, 2, 0]
        })
        pred_csv = os.path.join(temp_dir, 'multiclass_predictions.csv')
        pred_data.to_csv(pred_csv, index=False)
        
        return pred_csv, pred_data

    @pytest.fixture
    def separate_label_data(self, temp_dir):
        """Create separate prediction and label files."""
        # Predictions CSV (no labels)
        pred_data = pd.DataFrame({
            'filepath': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg'],
            'prediction': [0.75, 0.25, 0.85, 0.15]
        })
        pred_csv = os.path.join(temp_dir, 'predictions_only.csv')
        pred_data.to_csv(pred_csv, index=False)
        
        # Labels CSV
        label_data = pd.DataFrame({
            'filepath': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg'],
            'label': [1, 0, 1, 0]
        })
        label_csv = os.path.join(temp_dir, 'labels.csv')
        label_data.to_csv(label_csv, index=False)
        
        return pred_csv, label_csv, pred_data, label_data

    def test_binary_classification_evaluation(self, sample_binary_data, temp_dir):
        """Test binary classification evaluation."""
        pred_csv, pred_data = sample_binary_data
        output_png = os.path.join(temp_dir, 'confusion_matrix.png')
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.close') as mock_close:
            
            metrics = evaluate_predictions(
                pred_csv=pred_csv,
                output_png=output_png,
                threshold=0.5,
                num_classes=1
            )
            
            # Verify metrics are computed
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1' in metrics
            assert 'roc_auc' in metrics
            
            # Verify all metrics are numbers
            for metric_name, metric_value in metrics.items():
                assert isinstance(metric_value, (int, float, np.floating))
                assert not np.isnan(metric_value)
            
            # Verify plot is saved
            mock_savefig.assert_called_once_with(output_png)
            mock_close.assert_called_once()

    def test_multiclass_evaluation(self, sample_multiclass_data, temp_dir):
        """Test multi-class classification evaluation."""
        pred_csv, pred_data = sample_multiclass_data
        output_png = os.path.join(temp_dir, 'multiclass_cm.png')
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.close') as mock_close:
            
            metrics = evaluate_predictions(
                pred_csv=pred_csv,
                output_png=output_png,
                num_classes=3
            )
            
            # Verify metrics are computed
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1' in metrics
            assert 'roc_auc' in metrics
            
            # Verify all metrics are valid numbers
            for metric_name, metric_value in metrics.items():
                assert isinstance(metric_value, (int, float, np.floating))
                assert not np.isnan(metric_value)
            
            # Verify plot is saved
            mock_savefig.assert_called_once_with(output_png)
            mock_close.assert_called_once()

    def test_separate_label_files(self, separate_label_data, temp_dir):
        """Test evaluation with separate prediction and label files."""
        pred_csv, label_csv, pred_data, label_data = separate_label_data
        output_png = os.path.join(temp_dir, 'separate_files_cm.png')
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.close') as mock_close:
            
            metrics = evaluate_predictions(
                pred_csv=pred_csv,
                label_csv=label_csv,
                output_png=output_png,
                threshold=0.5,
                num_classes=1
            )
            
            # Verify metrics are computed
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1' in metrics
            assert 'roc_auc' in metrics
            
            # Verify plot is saved
            mock_savefig.assert_called_once_with(output_png)
            mock_close.assert_called_once()

    def test_metrics_csv_export(self, sample_binary_data, temp_dir):
        """Test exporting metrics to CSV."""
        pred_csv, pred_data = sample_binary_data
        output_png = os.path.join(temp_dir, 'cm.png')
        metrics_csv = os.path.join(temp_dir, 'metrics.csv')
        
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'):
            
            metrics = evaluate_predictions(
                pred_csv=pred_csv,
                output_png=output_png,
                metrics_csv=metrics_csv,
                threshold=0.5,
                num_classes=1
            )
            
            # Verify metrics CSV is created
            assert os.path.exists(metrics_csv)
            
            # Read and verify CSV content
            saved_metrics = pd.read_csv(metrics_csv)
            assert len(saved_metrics) == 1
            assert 'precision' in saved_metrics.columns
            assert 'recall' in saved_metrics.columns
            assert 'f1' in saved_metrics.columns
            assert 'roc_auc' in saved_metrics.columns

    def test_normalized_confusion_matrix(self, sample_binary_data, temp_dir):
        """Test normalized confusion matrix generation."""
        pred_csv, pred_data = sample_binary_data
        output_png = os.path.join(temp_dir, 'normalized_cm.png')
        
        with patch('sklearn.metrics.confusion_matrix') as mock_cm, \
             patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'):
            
            mock_cm.return_value = np.array([[0.5, 0.5], [0.3, 0.7]])
            
            evaluate_predictions(
                pred_csv=pred_csv,
                output_png=output_png,
                normalize_cm=True,
                threshold=0.5,
                num_classes=1
            )
            
            # Verify confusion_matrix called with normalize="true"
            mock_cm.assert_called_once()
            args, kwargs = mock_cm.call_args
            assert kwargs.get('normalize') == 'true'

    def test_custom_threshold(self, sample_binary_data, temp_dir):
        """Test evaluation with custom threshold."""
        pred_csv, pred_data = sample_binary_data
        output_png = os.path.join(temp_dir, 'custom_threshold_cm.png')
        
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'):
            
            # Test with high threshold (0.9)
            metrics_high = evaluate_predictions(
                pred_csv=pred_csv,
                output_png=output_png,
                threshold=0.9,
                num_classes=1
            )
            
            # Test with low threshold (0.1)
            metrics_low = evaluate_predictions(
                pred_csv=pred_csv,
                output_png=output_png,
                threshold=0.1,
                num_classes=1
            )
            
            # Metrics should be different for different thresholds
            assert metrics_high['precision'] != metrics_low['precision'] or \
                   metrics_high['recall'] != metrics_low['recall']

    def test_missing_labels_error(self, temp_dir):
        """Test error when labels are missing."""
        # Create predictions CSV without labels
        pred_data = pd.DataFrame({
            'filepath': ['img1.jpg', 'img2.jpg'],
            'prediction': [0.8, 0.2]
        })
        pred_csv = os.path.join(temp_dir, 'no_labels.csv')
        pred_data.to_csv(pred_csv, index=False)
        
        with pytest.raises(ValueError, match="No ground truth labels provided"):
            evaluate_predictions(pred_csv=pred_csv, num_classes=1)

    def test_zero_division_handling(self, temp_dir):
        """Test handling of zero division in metrics."""
        # Create edge case data that might cause zero division
        pred_data = pd.DataFrame({
            'filepath': ['img1.jpg', 'img2.jpg'],
            'prediction': [0.0, 0.0],  # All predictions are 0
            'label': [0, 0]  # All labels are 0
        })
        pred_csv = os.path.join(temp_dir, 'zero_div.csv')
        pred_data.to_csv(pred_csv, index=False)
        output_png = os.path.join(temp_dir, 'zero_div_cm.png')
        
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'):
            
            metrics = evaluate_predictions(
                pred_csv=pred_csv,
                output_png=output_png,
                threshold=0.5,
                num_classes=1
            )
            
            # Should not raise error and return valid metrics
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1' in metrics

    def test_perfect_predictions(self, temp_dir):
        """Test evaluation with perfect predictions."""
        pred_data = pd.DataFrame({
            'filepath': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg'],
            'prediction': [1.0, 0.0, 1.0, 0.0],  # Perfect predictions
            'label': [1, 0, 1, 0]
        })
        pred_csv = os.path.join(temp_dir, 'perfect.csv')
        pred_data.to_csv(pred_csv, index=False)
        output_png = os.path.join(temp_dir, 'perfect_cm.png')
        
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'):
            
            metrics = evaluate_predictions(
                pred_csv=pred_csv,
                output_png=output_png,
                threshold=0.5,
                num_classes=1
            )
            
            # Should have perfect metrics
            assert metrics['precision'] == 1.0
            assert metrics['recall'] == 1.0
            assert metrics['f1'] == 1.0
            assert metrics['roc_auc'] == 1.0


def test_evaluate_cli_help():
    """Test CLI help output."""
    result = subprocess.run(
        [sys.executable, "-m", "src.evaluate", "--help"], capture_output=True
    )
    assert result.returncode == 0
    assert b"--normalize_cm" in result.stdout
    assert b"--threshold" in result.stdout
    assert b"--metrics_csv" in result.stdout
    assert b"--num_classes" in result.stdout


def test_main_function_integration(temp_dir_fixture=None):
    """Test the main function with mocked arguments."""
    if temp_dir_fixture is None:
        import tempfile
        temp_dir_fixture = tempfile.mkdtemp()
    
    # Create test data
    pred_data = pd.DataFrame({
        'filepath': ['img1.jpg', 'img2.jpg'],
        'prediction': [0.8, 0.2],
        'label': [1, 0]
    })
    pred_csv = os.path.join(temp_dir_fixture, 'test_main.csv')
    pred_data.to_csv(pred_csv, index=False)
    output_png = os.path.join(temp_dir_fixture, 'test_main_cm.png')
    
    # Mock command line arguments
    test_args = [
        'src.evaluate',
        '--pred_csv', pred_csv,
        '--output_png', output_png,
        '--threshold', '0.5',
        '--num_classes', '1'
    ]
    
    with patch('sys.argv', test_args), \
         patch('matplotlib.pyplot.savefig'), \
         patch('matplotlib.pyplot.close'), \
         patch('builtins.print') as mock_print:
        
        from src.evaluate import main
        main()
        
        # Verify metrics were printed
        assert mock_print.call_count > 0
        printed_output = [str(call) for call in mock_print.call_args_list]
        metrics_printed = any('precision:' in output for output in printed_output)
        assert metrics_printed
