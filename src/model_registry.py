"""Model versioning and A/B testing framework for production ML deployments.

This module provides a comprehensive model registry system designed for medical AI applications
where model versioning, A/B testing, and safe deployments are critical for regulatory compliance
and patient safety. It integrates with MLflow for experiment tracking and provides production-ready
features for model management.

Key Features:
- Semantic versioning for models with promotion workflows
- A/B testing framework with traffic splitting
- Model performance tracking and metrics collection
- Safe rollback capabilities
- Thread-safe operations for concurrent access
- Integration with existing MLflow tracking
"""

import json
import os
import shutil
import hashlib
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelValidationError(Exception):
    """Raised when model metadata fails validation."""
    pass


class ModelPromotionError(Exception):
    """Raised when model promotion fails."""
    pass


@dataclass
class ModelMetadata:
    """Comprehensive metadata for a registered model.
    
    Attributes
    ----------
    model_id : str
        Unique identifier for the model family (e.g., 'pneumonia_detector').
    version : str
        Semantic version string (e.g., '1.2.3').
    accuracy : float
        Model accuracy on validation set (0-1 range).
    f1_score : float
        F1 score on validation set (0-1 range).
    roc_auc : float
        ROC AUC score on validation set (0-1 range).
    training_date : datetime
        When the model was trained.
    dataset_version : str
        Version identifier for the training dataset.
    model_path : str
        File system path to the model file.
    training_config : Dict[str, Any], optional
        Training hyperparameters and configuration.
    description : str, optional
        Human-readable description of the model.
    tags : List[str], optional
        Tags for categorization and search.
    is_production : bool, default False
        Whether this model is currently in production.
    status : str, default 'staged'
        Model status: 'staged', 'production', 'archived', 'failed'.
    mlflow_run_id : str, optional
        Associated MLflow run ID for experiment tracking.
    checksum : str, optional
        SHA256 checksum of the model file for integrity verification.
    """
    model_id: str
    version: str
    accuracy: float
    f1_score: float
    roc_auc: float
    training_date: datetime
    dataset_version: str
    model_path: str
    training_config: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    is_production: bool = False
    status: str = "staged"
    mlflow_run_id: Optional[str] = None
    checksum: Optional[str] = None

    def __post_init__(self):
        """Validate metadata after initialization."""
        self._validate()
        if self.training_config is None:
            self.training_config = {}
        if self.tags is None:
            self.tags = []

    def _validate(self):
        """Validate model metadata for correctness and safety."""
        if not self.model_id or not self.model_id.strip():
            raise ModelValidationError("Model ID cannot be empty")
        
        if not self.version or not self.version.strip():
            raise ModelValidationError("Version cannot be empty")
        
        # Validate semantic version format
        try:
            ModelVersion(self.version)
        except ModelValidationError:
            raise ModelValidationError(f"Invalid version format: {self.version}")
        
        # Validate metric ranges
        if not 0 <= self.accuracy <= 1:
            raise ModelValidationError("Accuracy must be between 0 and 1")
        if not 0 <= self.f1_score <= 1:
            raise ModelValidationError("F1 score must be between 0 and 1")
        if not 0 <= self.roc_auc <= 1:
            raise ModelValidationError("ROC AUC must be between 0 and 1")
        
        # Validate status
        valid_statuses = ["staged", "production", "archived", "failed"]
        if self.status not in valid_statuses:
            raise ModelValidationError(f"Status must be one of: {valid_statuses}")

    def to_json(self) -> str:
        """Serialize metadata to JSON string."""
        data = asdict(self)
        # Convert datetime to ISO format
        data['training_date'] = self.training_date.isoformat()
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'ModelMetadata':
        """Deserialize metadata from JSON string."""
        data = json.loads(json_str)
        # Convert ISO format back to datetime
        data['training_date'] = datetime.fromisoformat(data['training_date'])
        return cls(**data)

    def calculate_checksum(self) -> str:
        """Calculate SHA256 checksum of the model file."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        sha256_hash = hashlib.sha256()
        with open(self.model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


class ModelVersion:
    """Semantic version handling for models."""
    
    def __init__(self, version: str):
        """Initialize semantic version.
        
        Parameters
        ----------
        version : str
            Semantic version string (e.g., '1.2.3').
        """
        self.version = version
        self._parse_version()

    def _parse_version(self):
        """Parse and validate semantic version string."""
        parts = self.version.split('.')
        if len(parts) != 3:
            raise ModelValidationError(f"Invalid version format: {self.version}. Expected format: major.minor.patch")
        
        try:
            self.major = int(parts[0])
            self.minor = int(parts[1])
            self.patch = int(parts[2])
        except ValueError:
            raise ModelValidationError(f"Invalid version format: {self.version}. All parts must be integers")

    def __lt__(self, other: 'ModelVersion') -> bool:
        """Less than comparison for version ordering."""
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __eq__(self, other: 'ModelVersion') -> bool:
        """Equality comparison for versions."""
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __str__(self) -> str:
        """String representation of version."""
        return self.version

    def increment_major(self) -> 'ModelVersion':
        """Create new version with incremented major version."""
        return ModelVersion(f"{self.major + 1}.0.0")

    def increment_minor(self) -> 'ModelVersion':
        """Create new version with incremented minor version."""
        return ModelVersion(f"{self.major}.{self.minor + 1}.0")

    def increment_patch(self) -> 'ModelVersion':
        """Create new version with incremented patch version."""
        return ModelVersion(f"{self.major}.{self.minor}.{self.patch + 1}")


@dataclass
class ABTestConfig:
    """Configuration for A/B testing experiments.
    
    Attributes
    ----------
    experiment_name : str
        Unique name for the A/B test experiment.
    control_model_version : str
        Version of the control (baseline) model.
    treatment_model_version : str
        Version of the treatment (new) model being tested.
    traffic_split : float
        Fraction of traffic routed to treatment (0-1 range).
    success_metrics : List[str]
        Metrics to track for experiment success.
    duration_days : int
        Duration of the experiment in days.
    start_date : datetime, optional
        When the experiment started. Defaults to current time.
    end_date : datetime, optional
        When the experiment should end. Calculated from duration if not provided.
    minimum_sample_size : int, default 1000
        Minimum number of samples needed for statistical significance.
    confidence_level : float, default 0.95
        Statistical confidence level for significance testing.
    """
    experiment_name: str
    control_model_version: str
    treatment_model_version: str
    traffic_split: float
    success_metrics: List[str]
    duration_days: int
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    minimum_sample_size: int = 1000
    confidence_level: float = 0.95

    def __post_init__(self):
        """Initialize calculated fields."""
        if self.start_date is None:
            self.start_date = datetime.now()
        if self.end_date is None:
            self.end_date = self.start_date + timedelta(days=self.duration_days)
        
        # Validate traffic split
        if not 0 <= self.traffic_split <= 1:
            raise ModelValidationError("Traffic split must be between 0 and 1")

    def is_active(self) -> bool:
        """Check if the A/B test is currently active."""
        now = datetime.now()
        return self.start_date <= now <= self.end_date

    def should_use_treatment(self, user_id: str) -> bool:
        """Determine if a user should receive the treatment model.
        
        Uses deterministic hashing to ensure consistent routing for the same user.
        
        Parameters
        ----------
        user_id : str
            Unique identifier for the user/request.
            
        Returns
        -------
        bool
            True if user should receive treatment model, False for control.
        """
        # Use SHA256 hash of user_id for deterministic routing (security fix)
        hash_value = int(hashlib.sha256(user_id.encode()).hexdigest(), 16)
        return (hash_value % 100) / 100 < self.traffic_split


class ModelRegistry:
    """Production-ready model registry with versioning and A/B testing.
    
    This class provides a comprehensive model management system designed for
    medical AI applications. It handles model registration, version tracking,
    promotion workflows, A/B testing, and performance monitoring.
    
    Features:
    - Thread-safe operations for concurrent access
    - Atomic file operations with rollback capabilities
    - Integration with MLflow for experiment tracking
    - Comprehensive audit logging for regulatory compliance
    - Performance metrics collection and analysis
    
    Parameters
    ----------
    registry_path : str
        Base directory for the model registry storage.
    mlflow_tracking_uri : str, optional
        MLflow tracking server URI for integration.
    """
    
    def __init__(self, registry_path: str, mlflow_tracking_uri: Optional[str] = None):
        """Initialize model registry."""
        self.registry_path = Path(registry_path)
        self.mlflow_tracking_uri = mlflow_tracking_uri
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize directory structure
        self._initialize_registry()
        
        # Load existing metadata
        self._metadata_cache = self._load_metadata()
        self._ab_tests = self._load_ab_tests()
        
        logger.info(f"Model registry initialized at {registry_path}")

    def _initialize_registry(self):
        """Create registry directory structure."""
        directories = [
            self.registry_path,
            self.registry_path / "models",
            self.registry_path / "metadata",
            self.registry_path / "ab_tests",
            self.registry_path / "performance_logs",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata file if it doesn't exist
        metadata_file = self.registry_path / "metadata.json"
        if not metadata_file.exists():
            with open(metadata_file, 'w') as f:
                json.dump({}, f)

    def _get_write_lock(self):
        """Get write lock for thread-safe operations."""
        return self._lock

    def _load_metadata(self) -> Dict[str, Dict[str, ModelMetadata]]:
        """Load model metadata from storage."""
        metadata_file = self.registry_path / "metadata.json"
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            
            # Convert to ModelMetadata objects
            result = {}
            for model_id, versions in data.items():
                result[model_id] = {}
                for version, metadata_dict in versions.items():
                    metadata_dict['training_date'] = datetime.fromisoformat(metadata_dict['training_date'])
                    result[model_id][version] = ModelMetadata(**metadata_dict)
            
            return result
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_metadata(self):
        """Save model metadata to storage."""
        metadata_file = self.registry_path / "metadata.json"
        
        # Convert to serializable format
        data = {}
        for model_id, versions in self._metadata_cache.items():
            data[model_id] = {}
            for version, metadata in versions.items():
                metadata_dict = asdict(metadata)
                metadata_dict['training_date'] = metadata.training_date.isoformat()
                data[model_id][version] = metadata_dict
        
        # Atomic write with backup
        temp_file = metadata_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Atomic rename
        temp_file.replace(metadata_file)

    def _load_ab_tests(self) -> List[ABTestConfig]:
        """Load active A/B tests from storage."""
        ab_tests_file = self.registry_path / "ab_tests" / "active_tests.json"
        try:
            with open(ab_tests_file, 'r') as f:
                data = json.load(f)
            
            # Convert to ABTestConfig objects
            result = []
            for test_data in data:
                test_data['start_date'] = datetime.fromisoformat(test_data['start_date'])
                test_data['end_date'] = datetime.fromisoformat(test_data['end_date'])
                result.append(ABTestConfig(**test_data))
            
            return result
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_ab_tests(self):
        """Save A/B tests to storage."""
        ab_tests_file = self.registry_path / "ab_tests" / "active_tests.json"
        
        # Convert to serializable format
        data = []
        for test in self._ab_tests:
            test_dict = asdict(test)
            test_dict['start_date'] = test.start_date.isoformat()
            test_dict['end_date'] = test.end_date.isoformat()
            data.append(test_dict)
        
        with open(ab_tests_file, 'w') as f:
            json.dump(data, f, indent=2)

    def register_model(self, metadata: ModelMetadata) -> str:
        """Register a new model version in the registry.
        
        Parameters
        ----------
        metadata : ModelMetadata
            Complete metadata for the model to register.
            
        Returns
        -------
        str
            Path to the registered model file in the registry.
            
        Raises
        ------
        ModelValidationError
            If metadata validation fails.
        FileNotFoundError
            If the source model file doesn't exist.
        """
        with self._get_write_lock():
            # Validate model file exists
            if not os.path.exists(metadata.model_path):
                raise FileNotFoundError(f"Model file not found: {metadata.model_path}")
            
            # Calculate checksum for integrity
            metadata.checksum = metadata.calculate_checksum()
            
            # Determine registry path for this model
            model_filename = f"{metadata.model_id}_v{metadata.version}.keras"
            registry_model_path = self.registry_path / "models" / model_filename
            
            # Copy model file to registry
            shutil.copy2(metadata.model_path, registry_model_path)
            
            # Update metadata with registry path
            metadata.model_path = str(registry_model_path)
            
            # Add to cache
            if metadata.model_id not in self._metadata_cache:
                self._metadata_cache[metadata.model_id] = {}
            self._metadata_cache[metadata.model_id][metadata.version] = metadata
            
            # Save metadata
            self._save_metadata()
            
            logger.info(f"Registered model {metadata.model_id} v{metadata.version}")
            return str(registry_model_path)

    def promote_to_production(self, model_id: str, version: str) -> None:
        """Promote a model version to production status.
        
        Parameters
        ----------
        model_id : str
            Model identifier.
        version : str
            Version to promote.
            
        Raises
        ------
        ModelPromotionError
            If promotion fails validation.
        """
        with self._get_write_lock():
            # Validate model exists
            if model_id not in self._metadata_cache or version not in self._metadata_cache[model_id]:
                raise ModelPromotionError(f"Model {model_id} v{version} not found in registry")
            
            metadata = self._metadata_cache[model_id][version]
            
            # Validate model file integrity
            current_checksum = metadata.calculate_checksum()
            if metadata.checksum != current_checksum:
                raise ModelPromotionError(f"Model file integrity check failed for {model_id} v{version}")
            
            # Demote any existing production models
            for v, m in self._metadata_cache[model_id].items():
                if m.is_production:
                    m.is_production = False
                    m.status = "archived"
            
            # Promote this version
            metadata.is_production = True
            metadata.status = "production"
            
            # Save changes
            self._save_metadata()
            
            logger.info(f"Promoted model {model_id} v{version} to production")

    def rollback_model(self, model_id: str, target_version: str) -> None:
        """Rollback production model to a previous version.
        
        Parameters
        ----------
        model_id : str
            Model identifier.
        target_version : str
            Version to rollback to.
        """
        logger.warning(f"Rolling back model {model_id} to version {target_version}")
        self.promote_to_production(model_id, target_version)

    def get_production_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get the current production model for a given model ID.
        
        Parameters
        ----------
        model_id : str
            Model identifier.
            
        Returns
        -------
        ModelMetadata or None
            Production model metadata, or None if no production model exists.
        """
        if model_id not in self._metadata_cache:
            return None
        
        for metadata in self._metadata_cache[model_id].values():
            if metadata.is_production:
                return metadata
        
        return None

    def list_models(self, model_id: Optional[str] = None) -> List[ModelMetadata]:
        """List all registered models or versions of a specific model.
        
        Parameters
        ----------
        model_id : str, optional
            If provided, only return versions of this model.
            
        Returns
        -------
        List[ModelMetadata]
            List of model metadata objects.
        """
        result = []
        
        if model_id:
            if model_id in self._metadata_cache:
                result.extend(self._metadata_cache[model_id].values())
        else:
            for versions in self._metadata_cache.values():
                result.extend(versions.values())
        
        # Sort by model_id and version
        return sorted(result, key=lambda m: (m.model_id, ModelVersion(m.version)))

    def start_ab_test(self, model_id: str, config: ABTestConfig) -> None:
        """Start an A/B test for the specified model.
        
        Parameters
        ----------
        model_id : str
            Model identifier for the A/B test.
        config : ABTestConfig
            A/B test configuration.
        """
        with self._get_write_lock():
            # Validate both model versions exist
            if (model_id not in self._metadata_cache or 
                config.control_model_version not in self._metadata_cache[model_id] or
                config.treatment_model_version not in self._metadata_cache[model_id]):
                raise ModelValidationError("Both control and treatment model versions must be registered")
            
            # Stop any existing A/B tests for this model
            self._ab_tests = [test for test in self._ab_tests 
                            if not (test.control_model_version.startswith(model_id) or 
                                   test.treatment_model_version.startswith(model_id))]
            
            # Add new test
            self._ab_tests.append(config)
            self._save_ab_tests()
            
            logger.info(f"Started A/B test {config.experiment_name} for model {model_id}")

    def list_active_ab_tests(self) -> List[ABTestConfig]:
        """List all active A/B tests.
        
        Returns
        -------
        List[ABTestConfig]
            List of active A/B test configurations.
        """
        return [test for test in self._ab_tests if test.is_active()]

    def get_model_for_inference(self, model_id: str, user_id: Optional[str] = None) -> ModelMetadata:
        """Get the appropriate model for inference, considering A/B tests.
        
        Parameters
        ----------
        model_id : str
            Model identifier.
        user_id : str, optional
            User ID for A/B test routing. If None, returns production model.
            
        Returns
        -------
        ModelMetadata
            Model metadata for inference.
        """
        # Check for active A/B tests
        if user_id:
            for test in self.list_active_ab_tests():
                if (model_id in test.control_model_version or 
                    model_id in test.treatment_model_version):
                    
                    if test.should_use_treatment(user_id):
                        version = test.treatment_model_version
                    else:
                        version = test.control_model_version
                    
                    return self._metadata_cache[model_id][version]
        
        # Return production model
        production_model = self.get_production_model(model_id)
        if production_model is None:
            raise ModelPromotionError(f"No production model found for {model_id}")
        
        return production_model

    def record_inference_metrics(self, model_id: str, version: str, user_id: str,
                               prediction_confidence: float, inference_time_ms: float,
                               correct_prediction: Optional[bool] = None) -> None:
        """Record inference metrics for performance tracking.
        
        Parameters
        ----------
        model_id : str
            Model identifier.
        version : str
            Model version.
        user_id : str
            User ID for the inference.
        prediction_confidence : float
            Confidence score of the prediction.
        inference_time_ms : float
            Inference time in milliseconds.
        correct_prediction : bool, optional
            Whether the prediction was correct (if ground truth is available).
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "model_id": model_id,
            "version": version,
            "user_id": user_id,
            "prediction_confidence": prediction_confidence,
            "inference_time_ms": inference_time_ms,
            "correct_prediction": correct_prediction,
        }
        
        # Log to performance file
        log_file = self.registry_path / "performance_logs" / f"{model_id}_v{version}.jsonl"
        with open(log_file, 'a') as f:
            f.write(f"{json.dumps(metrics)}\n")

    def get_model_performance_summary(self, model_id: str, version: str) -> Dict[str, Any]:
        """Get performance summary for a model version.
        
        Parameters
        ----------
        model_id : str
            Model identifier.
        version : str
            Model version.
            
        Returns
        -------
        Dict[str, Any]
            Performance summary metrics.
        """
        log_file = self.registry_path / "performance_logs" / f"{model_id}_v{version}.jsonl"
        
        if not log_file.exists():
            return {"total_inferences": 0}
        
        total_inferences = 0
        confidence_sum = 0
        time_sum = 0
        correct_predictions = 0
        total_with_ground_truth = 0
        
        with open(log_file, 'r') as f:
            for line in f:
                metrics = json.loads(line)
                total_inferences += 1
                confidence_sum += metrics["prediction_confidence"]
                time_sum += metrics["inference_time_ms"]
                
                if metrics["correct_prediction"] is not None:
                    total_with_ground_truth += 1
                    if metrics["correct_prediction"]:
                        correct_predictions += 1
        
        summary = {
            "total_inferences": total_inferences,
            "avg_confidence": confidence_sum / total_inferences if total_inferences > 0 else 0,
            "avg_inference_time_ms": time_sum / total_inferences if total_inferences > 0 else 0,
        }
        
        if total_with_ground_truth > 0:
            summary["accuracy"] = correct_predictions / total_with_ground_truth
        
        return summary