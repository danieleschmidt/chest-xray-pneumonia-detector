"""
Federated Learning Coordinator for Medical AI
Enables privacy-preserving distributed training across healthcare institutions.
"""

import json
import logging
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import hashlib
import time
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


@dataclass
class FederatedClient:
    """Represents a federated learning client (e.g., hospital, clinic)."""
    client_id: str
    name: str
    data_samples: int
    model_version: str
    last_update: float
    public_key: Optional[str] = None
    reputation_score: float = 1.0
    contribution_weight: float = 1.0


@dataclass 
class ModelUpdate:
    """Represents a model update from a federated client."""
    client_id: str
    model_weights: List[np.ndarray]
    metrics: Dict[str, float]
    timestamp: float
    signature: Optional[str] = None
    encrypted: bool = False


class FederationAggregator(ABC):
    """Abstract base class for federated learning aggregation strategies."""
    
    @abstractmethod
    def aggregate(self, updates: List[ModelUpdate], clients: List[FederatedClient]) -> List[np.ndarray]:
        """Aggregate model updates from multiple clients."""
        pass


class FedAvgAggregator(FederationAggregator):
    """Implements Federated Averaging (FedAvg) aggregation."""
    
    def aggregate(self, updates: List[ModelUpdate], clients: List[FederatedClient]) -> List[np.ndarray]:
        """Aggregate using weighted average based on client data samples."""
        if not updates:
            raise ValueError("No updates provided for aggregation")
            
        # Calculate total samples for weighting
        client_samples = {client.client_id: client.data_samples for client in clients}
        total_samples = sum(client_samples.get(update.client_id, 1) for update in updates)
        
        # Initialize aggregated weights
        aggregated_weights = None
        
        for update in updates:
            client_weight = client_samples.get(update.client_id, 1) / total_samples
            
            if aggregated_weights is None:
                aggregated_weights = [w * client_weight for w in update.model_weights]
            else:
                for i, layer_weights in enumerate(update.model_weights):
                    aggregated_weights[i] += layer_weights * client_weight
                    
        logger.info(f"Aggregated {len(updates)} model updates using FedAvg")
        return aggregated_weights


class AdaptiveFedAggregator(FederationAggregator):
    """Advanced aggregation with client reputation and adaptive weighting."""
    
    def __init__(self, reputation_weight: float = 0.3, performance_weight: float = 0.7):
        self.reputation_weight = reputation_weight  
        self.performance_weight = performance_weight
        
    def aggregate(self, updates: List[ModelUpdate], clients: List[FederatedClient]) -> List[np.ndarray]:
        """Aggregate with adaptive weighting based on reputation and performance."""
        if not updates:
            raise ValueError("No updates provided for aggregation")
            
        client_info = {client.client_id: client for client in clients}
        
        # Calculate adaptive weights
        total_weight = 0
        update_weights = []
        
        for update in updates:
            client = client_info.get(update.client_id)
            if client is None:
                continue
                
            # Combine reputation, performance, and data size
            reputation_factor = client.reputation_score
            performance_factor = update.metrics.get('val_accuracy', 0.5)  
            size_factor = client.data_samples
            
            adaptive_weight = (
                reputation_factor * self.reputation_weight + 
                performance_factor * self.performance_weight
            ) * np.sqrt(size_factor)  # Square root to prevent large clients from dominating
            
            update_weights.append(adaptive_weight)
            total_weight += adaptive_weight
            
        # Normalize weights
        if total_weight > 0:
            update_weights = [w / total_weight for w in update_weights]
        else:
            update_weights = [1.0 / len(updates)] * len(updates)
        
        # Aggregate with adaptive weights
        aggregated_weights = None
        
        for update, weight in zip(updates, update_weights):
            if aggregated_weights is None:
                aggregated_weights = [w * weight for w in update.model_weights]
            else:
                for i, layer_weights in enumerate(update.model_weights):
                    aggregated_weights[i] += layer_weights * weight
                    
        logger.info(f"Aggregated {len(updates)} updates using adaptive weighting")
        return aggregated_weights


class FederatedLearningCoordinator:
    """
    Coordinates federated learning across multiple healthcare institutions.
    
    Features:
    - Privacy-preserving model aggregation
    - Client reputation management  
    - Secure communication protocols
    - Byzantine fault tolerance
    - Differential privacy support
    """
    
    def __init__(
        self,
        model_template: tf.keras.Model,
        aggregator: FederationAggregator = None,
        encryption_key: Optional[bytes] = None,
        min_clients: int = 3,
        max_clients: int = 100,
        rounds: int = 100,
        client_fraction: float = 0.3,
        differential_privacy: bool = False,
        privacy_budget: float = 1.0
    ):
        """
        Initialize federated learning coordinator.
        
        Args:
            model_template: TensorFlow model template for federation
            aggregator: Strategy for aggregating client updates
            encryption_key: Key for encrypting model updates
            min_clients: Minimum clients required per round
            max_clients: Maximum clients allowed
            rounds: Number of federation rounds
            client_fraction: Fraction of clients to use per round
            differential_privacy: Enable differential privacy
            privacy_budget: Privacy budget for differential privacy
        """
        self.model_template = model_template
        self.aggregator = aggregator or FedAvgAggregator()
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        
        self.min_clients = min_clients
        self.max_clients = max_clients
        self.rounds = rounds
        self.client_fraction = client_fraction
        self.differential_privacy = differential_privacy
        self.privacy_budget = privacy_budget
        
        self.clients: List[FederatedClient] = []
        self.current_round = 0
        self.global_model_weights = model_template.get_weights()
        self.round_history = []
        
        logger.info(f"Initialized FederatedLearningCoordinator with {rounds} rounds")
        
    def register_client(
        self, 
        client_id: str, 
        name: str, 
        data_samples: int,
        public_key: Optional[str] = None
    ) -> bool:
        """Register a new federated learning client."""
        if len(self.clients) >= self.max_clients:
            logger.warning(f"Maximum clients ({self.max_clients}) reached")
            return False
            
        if any(client.client_id == client_id for client in self.clients):
            logger.warning(f"Client {client_id} already registered")
            return False
            
        client = FederatedClient(
            client_id=client_id,
            name=name,
            data_samples=data_samples,
            model_version="1.0.0",
            last_update=time.time(),
            public_key=public_key
        )
        
        self.clients.append(client)
        logger.info(f"Registered client {client_id} with {data_samples} samples")
        return True
        
    def select_clients(self) -> List[FederatedClient]:
        """Select clients for current federation round."""
        available_clients = [c for c in self.clients if c.reputation_score > 0.1]
        
        if len(available_clients) < self.min_clients:
            logger.error(f"Insufficient clients: {len(available_clients)} < {self.min_clients}")
            return []
            
        # Select fraction of clients, weighted by reputation and contribution
        num_selected = max(self.min_clients, int(len(available_clients) * self.client_fraction))
        num_selected = min(num_selected, len(available_clients))
        
        # Weighted selection based on reputation and data size
        weights = []
        for client in available_clients:
            weight = client.reputation_score * np.sqrt(client.data_samples)
            weights.append(weight)
            
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        selected_indices = np.random.choice(
            len(available_clients), 
            size=num_selected, 
            replace=False,
            p=weights
        )
        
        selected_clients = [available_clients[i] for i in selected_indices]
        logger.info(f"Selected {len(selected_clients)} clients for round {self.current_round}")
        
        return selected_clients
        
    def encrypt_weights(self, weights: List[np.ndarray]) -> bytes:
        """Encrypt model weights for secure transmission."""
        weights_bytes = self._serialize_weights(weights)
        encrypted_weights = self.cipher.encrypt(weights_bytes)
        return encrypted_weights
        
    def decrypt_weights(self, encrypted_weights: bytes) -> List[np.ndarray]:
        """Decrypt model weights from secure transmission.""" 
        weights_bytes = self.cipher.decrypt(encrypted_weights)
        weights = self._deserialize_weights(weights_bytes)
        return weights
        
    def _serialize_weights(self, weights: List[np.ndarray]) -> bytes:
        """Serialize model weights to bytes."""
        weights_dict = {f"layer_{i}": w.tolist() for i, w in enumerate(weights)}
        weights_json = json.dumps(weights_dict)
        return weights_json.encode('utf-8')
        
    def _deserialize_weights(self, weights_bytes: bytes) -> List[np.ndarray]:
        """Deserialize bytes to model weights."""
        weights_json = weights_bytes.decode('utf-8')
        weights_dict = json.loads(weights_json)
        weights = []
        
        for i in range(len(weights_dict)):
            layer_weights = np.array(weights_dict[f"layer_{i}"])
            weights.append(layer_weights)
            
        return weights
        
    def add_differential_privacy_noise(self, weights: List[np.ndarray], sensitivity: float = 1.0) -> List[np.ndarray]:
        """Add differential privacy noise to model weights."""
        if not self.differential_privacy:
            return weights
            
        # Calculate noise scale based on privacy budget
        noise_scale = sensitivity / self.privacy_budget
        
        noisy_weights = []
        for layer_weights in weights:
            noise = np.random.laplace(0, noise_scale, layer_weights.shape)
            noisy_weights.append(layer_weights + noise)
            
        logger.debug(f"Added DP noise with scale {noise_scale}")
        return noisy_weights
        
    def validate_update(self, update: ModelUpdate, client: FederatedClient) -> bool:
        """Validate client model update for Byzantine fault tolerance."""
        # Check timestamp freshness
        if time.time() - update.timestamp > 3600:  # 1 hour timeout
            logger.warning(f"Stale update from client {update.client_id}")
            return False
            
        # Validate metrics are reasonable
        if update.metrics.get('loss', float('inf')) > 100:
            logger.warning(f"Suspicious loss value from client {update.client_id}")
            return False
            
        if update.metrics.get('val_accuracy', 0) < 0 or update.metrics.get('val_accuracy', 0) > 1:
            logger.warning(f"Invalid accuracy from client {update.client_id}")
            return False
            
        # Check weight dimensions match global model
        if len(update.model_weights) != len(self.global_model_weights):
            logger.warning(f"Weight dimension mismatch from client {update.client_id}")
            return False
            
        return True
        
    def update_client_reputation(self, client_id: str, round_metrics: Dict[str, float]):
        """Update client reputation based on contribution quality."""
        client = next((c for c in self.clients if c.client_id == client_id), None)
        if client is None:
            return
            
        # Simple reputation update based on validation performance
        val_acc = round_metrics.get('val_accuracy', 0.5)
        performance_factor = min(2.0, max(0.1, val_acc / 0.8))  # Scale around 80% target
        
        # Exponential moving average for reputation
        alpha = 0.1
        client.reputation_score = (1 - alpha) * client.reputation_score + alpha * performance_factor
        client.last_update = time.time()
        
        logger.debug(f"Updated reputation for {client_id}: {client.reputation_score:.3f}")
        
    def run_federation_round(self, client_updates: List[ModelUpdate]) -> Dict[str, Any]:
        """Execute a single federation round."""
        self.current_round += 1
        
        # Validate and filter updates
        valid_updates = []
        for update in client_updates:
            client = next((c for c in self.clients if c.client_id == update.client_id), None)
            if client and self.validate_update(update, client):
                valid_updates.append(update)
            else:
                logger.warning(f"Rejected update from client {update.client_id}")
                
        if len(valid_updates) < self.min_clients:
            logger.error(f"Insufficient valid updates: {len(valid_updates)}")
            return {"success": False, "round": self.current_round}
            
        # Aggregate model updates
        try:
            aggregated_weights = self.aggregator.aggregate(valid_updates, self.clients)
            
            # Apply differential privacy
            if self.differential_privacy:
                aggregated_weights = self.add_differential_privacy_noise(aggregated_weights)
                
            # Update global model
            self.global_model_weights = aggregated_weights
            
            # Calculate round metrics
            round_metrics = self._calculate_round_metrics(valid_updates)
            
            # Update client reputations
            for update in valid_updates:
                self.update_client_reputation(update.client_id, update.metrics)
                
            # Record round history
            round_info = {
                "round": self.current_round,
                "participating_clients": len(valid_updates),
                "metrics": round_metrics,
                "timestamp": time.time()
            }
            self.round_history.append(round_info)
            
            logger.info(f"Completed federation round {self.current_round} "
                       f"with {len(valid_updates)} clients")
            
            return {
                "success": True, 
                "round": self.current_round,
                "metrics": round_metrics,
                "global_weights": self.global_model_weights
            }
            
        except Exception as e:
            logger.error(f"Federation round {self.current_round} failed: {e}")
            return {"success": False, "round": self.current_round, "error": str(e)}
            
    def _calculate_round_metrics(self, updates: List[ModelUpdate]) -> Dict[str, float]:
        """Calculate aggregated metrics for the federation round."""
        if not updates:
            return {}
            
        metrics = {}
        metric_names = set()
        
        # Collect all metric names
        for update in updates:
            metric_names.update(update.metrics.keys())
            
        # Calculate weighted average for each metric
        for metric_name in metric_names:
            total_weight = 0
            weighted_sum = 0
            
            for update in updates:
                if metric_name in update.metrics:
                    client = next((c for c in self.clients if c.client_id == update.client_id), None)
                    weight = client.data_samples if client else 1
                    
                    weighted_sum += update.metrics[metric_name] * weight
                    total_weight += weight
                    
            if total_weight > 0:
                metrics[metric_name] = weighted_sum / total_weight
                
        return metrics
        
    def get_global_model(self) -> tf.keras.Model:
        """Get the current global federated model.""" 
        model = tf.keras.models.clone_model(self.model_template)
        model.set_weights(self.global_model_weights)
        return model
        
    def save_checkpoint(self, filepath: Path):
        """Save federation state checkpoint."""
        checkpoint_data = {
            "current_round": self.current_round,
            "global_weights": [w.tolist() for w in self.global_model_weights],
            "clients": [asdict(client) for client in self.clients],
            "round_history": self.round_history,
            "config": {
                "min_clients": self.min_clients,
                "max_clients": self.max_clients,
                "rounds": self.rounds,
                "client_fraction": self.client_fraction,
                "differential_privacy": self.differential_privacy,
                "privacy_budget": self.privacy_budget
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
            
        logger.info(f"Saved federation checkpoint to {filepath}")
        
    def load_checkpoint(self, filepath: Path):
        """Load federation state from checkpoint."""
        with open(filepath, 'r') as f:
            checkpoint_data = json.load(f)
            
        self.current_round = checkpoint_data["current_round"]
        self.global_model_weights = [np.array(w) for w in checkpoint_data["global_weights"]]
        self.clients = [FederatedClient(**client_data) for client_data in checkpoint_data["clients"]]
        self.round_history = checkpoint_data["round_history"]
        
        logger.info(f"Loaded federation checkpoint from {filepath}")


# Example usage and utilities
def create_federated_pneumonia_detector(
    input_shape: Tuple[int, int, int] = (150, 150, 3),
    num_classes: int = 1,
    aggregation_strategy: str = "fedavg"
) -> FederatedLearningCoordinator:
    """Create a federated learning coordinator for pneumonia detection."""
    
    # Create a simple CNN model template
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Select aggregation strategy
    if aggregation_strategy == "adaptive":
        aggregator = AdaptiveFedAggregator()
    else:
        aggregator = FedAvgAggregator()
    
    coordinator = FederatedLearningCoordinator(
        model_template=model,
        aggregator=aggregator,
        min_clients=2,
        rounds=50,
        client_fraction=0.5,
        differential_privacy=True,
        privacy_budget=1.0
    )
    
    logger.info("Created federated pneumonia detection coordinator")
    return coordinator


if __name__ == "__main__":
    # Test federated learning coordinator
    coordinator = create_federated_pneumonia_detector()
    
    # Register some test clients
    coordinator.register_client("hospital_a", "General Hospital A", 1000)
    coordinator.register_client("clinic_b", "Rural Clinic B", 200) 
    coordinator.register_client("research_c", "Research Center C", 500)
    
    print(f"Federated Learning Coordinator initialized with {len(coordinator.clients)} clients")
    print(f"Configuration: min_clients={coordinator.min_clients}, rounds={coordinator.rounds}")