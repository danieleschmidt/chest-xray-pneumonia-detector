"""Configuration management for the chest X-ray pneumonia detector.

This module provides centralized configuration management using environment variables
with sensible defaults. It follows the Twelve-Factor App principle of storing
configuration in the environment.
"""

import os
from pathlib import Path
from typing import Optional


class Config:
    """Central configuration class that reads from environment variables."""
    
    # Base directories
    BASE_DIR = Path(__file__).parent.parent
    
    # Model and checkpoint paths
    CHECKPOINT_PATH: str = os.getenv(
        "CXR_CHECKPOINT_PATH", 
        str(BASE_DIR / "saved_models" / "best_pneumonia_cnn.keras")
    )
    
    SAVE_MODEL_PATH: str = os.getenv(
        "CXR_SAVE_MODEL_PATH",
        str(BASE_DIR / "saved_models" / "pneumonia_cnn_v1.keras")
    )
    
    # Report and output paths
    PLOT_PATH: str = os.getenv(
        "CXR_PLOT_PATH",
        str(BASE_DIR / "training_history.png")
    )
    
    CONFUSION_MATRIX_PATH: str = os.getenv(
        "CXR_CONFUSION_MATRIX_PATH",
        str(BASE_DIR / "reports" / "confusion_matrix_val.png")
    )
    
    HISTORY_CSV_PATH: str = os.getenv(
        "CXR_HISTORY_CSV_PATH",
        str(BASE_DIR / "training_history.csv")
    )
    
    # MLflow configuration
    MLFLOW_EXPERIMENT: str = os.getenv(
        "CXR_MLFLOW_EXPERIMENT",
        "pneumonia-detector"
    )
    
    MLFLOW_TRACKING_URI: Optional[str] = os.getenv("CXR_MLFLOW_TRACKING_URI")
    MLFLOW_RUN_NAME: Optional[str] = os.getenv("CXR_MLFLOW_RUN_NAME")
    
    # Dummy data configuration
    DUMMY_DATA_BASE_DIR: str = os.getenv(
        "CXR_DUMMY_DATA_BASE_DIR",
        str(BASE_DIR / "data_train_engine")
    )
    
    DUMMY_DATA_IMAGES_PER_CLASS: int = int(os.getenv(
        "CXR_DUMMY_DATA_IMAGES_PER_CLASS",
        "5"
    ))
    
    DUMMY_IMAGE_WIDTH: int = int(os.getenv("CXR_DUMMY_IMAGE_WIDTH", "60"))
    DUMMY_IMAGE_HEIGHT: int = int(os.getenv("CXR_DUMMY_IMAGE_HEIGHT", "30"))
    
    # Training configuration
    RANDOM_SEED: int = int(os.getenv("CXR_RANDOM_SEED", "42"))
    
    # Early stopping and learning rate configuration
    EARLY_STOPPING_PATIENCE: int = int(os.getenv("CXR_EARLY_STOPPING_PATIENCE", "10"))
    REDUCE_LR_FACTOR: float = float(os.getenv("CXR_REDUCE_LR_FACTOR", "0.2"))
    REDUCE_LR_PATIENCE: int = int(os.getenv("CXR_REDUCE_LR_PATIENCE", "5"))
    REDUCE_LR_MIN_LR: float = float(os.getenv("CXR_REDUCE_LR_MIN_LR", "1e-5"))
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure all required directories exist."""
        directories = [
            Path(cls.CHECKPOINT_PATH).parent,
            Path(cls.SAVE_MODEL_PATH).parent,
            Path(cls.PLOT_PATH).parent,
            Path(cls.CONFUSION_MATRIX_PATH).parent,
            Path(cls.HISTORY_CSV_PATH).parent,
            Path(cls.DUMMY_DATA_BASE_DIR),
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_env_info(cls) -> dict:
        """Get information about environment variable usage."""
        return {
            "CXR_CHECKPOINT_PATH": cls.CHECKPOINT_PATH,
            "CXR_SAVE_MODEL_PATH": cls.SAVE_MODEL_PATH,
            "CXR_PLOT_PATH": cls.PLOT_PATH,
            "CXR_CONFUSION_MATRIX_PATH": cls.CONFUSION_MATRIX_PATH,
            "CXR_HISTORY_CSV_PATH": cls.HISTORY_CSV_PATH,
            "CXR_MLFLOW_EXPERIMENT": cls.MLFLOW_EXPERIMENT,
            "CXR_MLFLOW_TRACKING_URI": cls.MLFLOW_TRACKING_URI,
            "CXR_MLFLOW_RUN_NAME": cls.MLFLOW_RUN_NAME,
            "CXR_DUMMY_DATA_BASE_DIR": cls.DUMMY_DATA_BASE_DIR,
            "CXR_DUMMY_DATA_IMAGES_PER_CLASS": cls.DUMMY_DATA_IMAGES_PER_CLASS,
            "CXR_DUMMY_IMAGE_WIDTH": cls.DUMMY_IMAGE_WIDTH,
            "CXR_DUMMY_IMAGE_HEIGHT": cls.DUMMY_IMAGE_HEIGHT,
            "CXR_RANDOM_SEED": cls.RANDOM_SEED,
            "CXR_EARLY_STOPPING_PATIENCE": cls.EARLY_STOPPING_PATIENCE,
            "CXR_REDUCE_LR_FACTOR": cls.REDUCE_LR_FACTOR,
            "CXR_REDUCE_LR_PATIENCE": cls.REDUCE_LR_PATIENCE,
            "CXR_REDUCE_LR_MIN_LR": cls.REDUCE_LR_MIN_LR,
        }


# Global configuration instance
config = Config()