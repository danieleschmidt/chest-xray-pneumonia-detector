"""Tests for configuration management."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest

from src.config import Config, config


class TestConfig:
    """Test configuration management functionality."""

    def test_default_values(self):
        """Test that default configuration values are set correctly."""
        # Test that basic paths are set
        assert Config.CHECKPOINT_PATH.endswith("best_pneumonia_cnn.keras")
        assert Config.SAVE_MODEL_PATH.endswith("pneumonia_cnn_v1.keras")
        assert Config.MLFLOW_EXPERIMENT == "pneumonia-detector"
        assert Config.DUMMY_DATA_IMAGES_PER_CLASS == 5
        assert Config.DUMMY_IMAGE_WIDTH == 60
        assert Config.DUMMY_IMAGE_HEIGHT == 30
        assert Config.RANDOM_SEED == 42

    def test_environment_variable_override(self):
        """Test that environment variables override default values."""
        with patch.dict(os.environ, {
            "CXR_CHECKPOINT_PATH": "/custom/checkpoint.keras",
            "CXR_MLFLOW_EXPERIMENT": "custom-experiment",
            "CXR_DUMMY_DATA_IMAGES_PER_CLASS": "10",
            "CXR_DUMMY_IMAGE_WIDTH": "120",
            "CXR_RANDOM_SEED": "123"
        }):
            # Create a new Config instance to pick up environment changes
            test_config = Config()
            
            assert test_config.CHECKPOINT_PATH == "/custom/checkpoint.keras"
            assert test_config.MLFLOW_EXPERIMENT == "custom-experiment"
            assert test_config.DUMMY_DATA_IMAGES_PER_CLASS == 10
            assert test_config.DUMMY_IMAGE_WIDTH == 120
            assert test_config.RANDOM_SEED == 123

    def test_optional_environment_variables(self):
        """Test optional environment variables (can be None)."""
        # By default, these should be None
        assert Config.MLFLOW_TRACKING_URI is None or isinstance(Config.MLFLOW_TRACKING_URI, str)
        assert Config.MLFLOW_RUN_NAME is None or isinstance(Config.MLFLOW_RUN_NAME, str)
        
        # Test setting them via environment
        with patch.dict(os.environ, {
            "CXR_MLFLOW_TRACKING_URI": "http://localhost:5000",
            "CXR_MLFLOW_RUN_NAME": "test-run"
        }):
            test_config = Config()
            assert test_config.MLFLOW_TRACKING_URI == "http://localhost:5000"
            assert test_config.MLFLOW_RUN_NAME == "test-run"

    def test_numeric_type_conversion(self):
        """Test that numeric environment variables are properly converted."""
        with patch.dict(os.environ, {
            "CXR_REDUCE_LR_FACTOR": "0.5",
            "CXR_REDUCE_LR_MIN_LR": "1e-6",
            "CXR_EARLY_STOPPING_PATIENCE": "15"
        }):
            test_config = Config()
            
            assert test_config.REDUCE_LR_FACTOR == 0.5
            assert test_config.REDUCE_LR_MIN_LR == 1e-6
            assert test_config.EARLY_STOPPING_PATIENCE == 15

    def test_ensure_directories(self):
        """Test directory creation functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_base = Path(tmpdir)
            
            with patch.dict(os.environ, {
                "CXR_CHECKPOINT_PATH": str(test_base / "models" / "checkpoint.keras"),
                "CXR_CONFUSION_MATRIX_PATH": str(test_base / "reports" / "cm.png"),
                "CXR_DUMMY_DATA_BASE_DIR": str(test_base / "data")
            }):
                test_config = Config()
                test_config.ensure_directories()
                
                # Check that directories were created
                assert (test_base / "models").exists()
                assert (test_base / "reports").exists()
                assert (test_base / "data").exists()

    def test_get_env_info(self):
        """Test environment information retrieval."""
        env_info = Config.get_env_info()
        
        # Check that all expected keys are present
        expected_keys = [
            "CXR_CHECKPOINT_PATH",
            "CXR_SAVE_MODEL_PATH",
            "CXR_MLFLOW_EXPERIMENT",
            "CXR_DUMMY_DATA_IMAGES_PER_CLASS",
            "CXR_RANDOM_SEED"
        ]
        
        for key in expected_keys:
            assert key in env_info
        
        # Check that values match current config
        assert env_info["CXR_CHECKPOINT_PATH"] == Config.CHECKPOINT_PATH
        assert env_info["CXR_MLFLOW_EXPERIMENT"] == Config.MLFLOW_EXPERIMENT
        assert env_info["CXR_RANDOM_SEED"] == Config.RANDOM_SEED

    def test_global_config_instance(self):
        """Test that global config instance is accessible."""
        assert config is not None
        assert hasattr(config, 'CHECKPOINT_PATH')
        assert hasattr(config, 'MLFLOW_EXPERIMENT')
        assert callable(config.ensure_directories)

    def test_path_handling(self):
        """Test that paths are handled correctly."""
        # Test that paths are strings
        assert isinstance(Config.CHECKPOINT_PATH, str)
        assert isinstance(Config.SAVE_MODEL_PATH, str)
        assert isinstance(Config.DUMMY_DATA_BASE_DIR, str)
        
        # Test that paths are absolute or properly formed
        checkpoint_path = Path(Config.CHECKPOINT_PATH)
        save_model_path = Path(Config.SAVE_MODEL_PATH)
        
        # Paths should be valid Path objects
        assert checkpoint_path.name == "best_pneumonia_cnn.keras"
        assert save_model_path.name == "pneumonia_cnn_v1.keras"
        
    def test_ensure_directories_permission_denied(self):
        """Test handling of permission denied errors when creating directories."""
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            # Simulate permission denied error
            mock_mkdir.side_effect = PermissionError("Permission denied: '/restricted/path'")
            
            with pytest.raises(PermissionError) as exc_info:
                Config.ensure_directories()
            
            assert "Permission denied to create directory" in str(exc_info.value)
            assert "Check permissions for path" in str(exc_info.value)

    def test_ensure_directories_other_os_error(self):
        """Test handling of other OS errors when creating directories.""" 
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            # Simulate disk full error
            mock_mkdir.side_effect = OSError("No space left on device")
            
            with pytest.raises(OSError) as exc_info:
                Config.ensure_directories()
            
            assert "Failed to create directory" in str(exc_info.value)
            assert "No space left on device" in str(exc_info.value)

    def test_invalid_numeric_environment_variables(self):
        """Test handling of invalid numeric environment variables."""
        with patch.dict(os.environ, {
            "CXR_DUMMY_DATA_IMAGES_PER_CLASS": "invalid_number"
        }):
            with pytest.raises(ValueError):
                Config()


class TestConfigurationBestPractices:
    """Test that configuration follows best practices."""

    def test_twelve_factor_app_compliance(self):
        """Test compliance with Twelve-Factor App principles."""
        # All configuration should be available via environment variables
        env_info = Config.get_env_info()
        
        # Check that we have environment variable names for all configuration
        assert len(env_info) > 10  # Should have many configurable items
        
        # All env var names should follow consistent naming convention
        for key in env_info.keys():
            if key.startswith("CXR_"):
                assert key == key.upper()  # Should be uppercase
                assert "_" in key  # Should use underscores

    def test_no_hardcoded_secrets(self):
        """Test that no secrets are hardcoded."""
        env_info = Config.get_env_info()
        
        # Check that sensitive values are None by default (must be set via env)
        assert Config.MLFLOW_TRACKING_URI is None or not Config.MLFLOW_TRACKING_URI.startswith("http")
        
        # No values should contain obvious secrets
        for key, value in env_info.items():
            if value:
                value_str = str(value).lower()
                assert "password" not in value_str
                assert "secret" not in value_str
                assert "token" not in value_str
                assert "key" not in value_str or "key" in key.lower()  # Allow in key names

    def test_defensive_defaults(self):
        """Test that default values are safe and reasonable."""
        # Numeric values should be reasonable
        assert 0 < Config.DUMMY_DATA_IMAGES_PER_CLASS <= 100
        assert 0 < Config.DUMMY_IMAGE_WIDTH <= 1000
        assert 0 < Config.DUMMY_IMAGE_HEIGHT <= 1000
        assert 0 < Config.REDUCE_LR_FACTOR < 1
        assert 0 < Config.REDUCE_LR_MIN_LR < 1
        assert Config.EARLY_STOPPING_PATIENCE > 0
        
        # Paths should not be absolute system paths
        assert not Config.CHECKPOINT_PATH.startswith("/usr")
        assert not Config.CHECKPOINT_PATH.startswith("/etc")
        assert not Config.SAVE_MODEL_PATH.startswith("/usr")