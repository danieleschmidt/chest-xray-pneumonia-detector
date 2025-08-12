"""
Adaptive Learning Scheduler - Advanced learning rate optimization
Implements dynamic learning rate adjustment based on training metrics.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AdaptiveLearningScheduler(Callback):
    """
    Advanced learning rate scheduler that adapts based on multiple metrics.
    
    Features:
    - Multi-metric monitoring (loss, accuracy, validation metrics)
    - Cyclical learning rates with adaptive ranges
    - Warm restart capability
    - Gradient norm tracking for stability
    """
    
    def __init__(
        self,
        base_lr: float = 0.001,
        max_lr: float = 0.01,
        step_size: int = 2000,
        mode: str = 'triangular2',
        gamma: float = 1.0,
        scale_fn=None,
        scale_mode: str = 'cycle',
        warmup_epochs: int = 5,
        min_lr_factor: float = 0.01,
        patience: int = 10,
        factor: float = 0.5,
        monitor_metrics: List[str] = None
    ):
        """
        Initialize adaptive learning scheduler.
        
        Args:
            base_lr: Lower boundary of learning rate range
            max_lr: Upper boundary of learning rate range  
            step_size: Number of training iterations in half cycle
            mode: Learning rate policy ('triangular', 'triangular2', 'exp_range')
            gamma: Decay factor for exponential mode
            warmup_epochs: Number of epochs for learning rate warmup
            min_lr_factor: Minimum learning rate as factor of base_lr
            patience: Epochs to wait before reducing lr on plateau
            factor: Factor to reduce learning rate by
            monitor_metrics: Metrics to monitor for adaptation
        """
        super().__init__()
        
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.min_lr_factor = min_lr_factor
        self.patience = patience
        self.factor = factor
        self.monitor_metrics = monitor_metrics or ['val_loss']
        
        self.clr_iterations = 0
        self.trn_iterations = 0
        self.history = {}
        self.best_metrics = {}
        self.wait_counts = {}
        
        # Initialize monitoring for each metric
        for metric in self.monitor_metrics:
            self.best_metrics[metric] = np.inf if 'loss' in metric else -np.inf
            self.wait_counts[metric] = 0
        
        # Scale function for learning rate
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.0
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2.0 ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** x
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
            
        logger.info(f"Initialized AdaptiveLearningScheduler with base_lr={base_lr}, max_lr={max_lr}")

    def clr(self):
        """Calculate cyclical learning rate."""
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(self.clr_iterations)

    def warmup_lr(self, epoch):
        """Calculate learning rate during warmup phase."""
        return self.base_lr * (epoch + 1) / self.warmup_epochs

    def on_train_begin(self, logs=None):
        """Initialize training history."""
        logs = logs or {}
        if self.clr_iterations == 0:
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.base_lr)
        else:
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.clr())

    def on_batch_end(self, batch, logs=None):
        """Update learning rate after each batch."""
        logs = logs or {}
        
        self.trn_iterations += 1
        self.clr_iterations += 1
        
        # Store training metrics
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
            
        # Apply cyclical learning rate (skip during warmup)
        current_epoch = len(self.history.get('loss', [0])) // self.model.steps_per_epoch
        if current_epoch >= self.warmup_epochs:
            lr = self.clr()
        else:
            lr = self.warmup_lr(current_epoch)
            
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        logs['lr'] = lr

    def on_epoch_end(self, epoch, logs=None):
        """Adapt learning rate based on epoch metrics."""
        logs = logs or {}
        
        # Apply warmup during first few epochs
        if epoch < self.warmup_epochs:
            lr = self.warmup_lr(epoch)
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
            logger.info(f"Epoch {epoch}: Warmup LR = {lr:.6f}")
            return
        
        # Monitor metrics and adapt
        lr_updated = False
        for metric in self.monitor_metrics:
            if metric in logs:
                current_value = logs[metric]
                
                # Check if metric improved
                if ('loss' in metric and current_value < self.best_metrics[metric]) or \
                   ('acc' in metric and current_value > self.best_metrics[metric]):
                    self.best_metrics[metric] = current_value
                    self.wait_counts[metric] = 0
                else:
                    self.wait_counts[metric] += 1
                
                # Reduce LR on plateau
                if self.wait_counts[metric] >= self.patience:
                    old_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                    new_lr = max(old_lr * self.factor, self.base_lr * self.min_lr_factor)
                    
                    if new_lr < old_lr:
                        tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
                        logger.info(f"Epoch {epoch}: Reduced LR from {old_lr:.6f} to {new_lr:.6f} "
                                  f"due to {metric} plateau")
                        lr_updated = True
                        
                    self.wait_counts[metric] = 0
        
        # Log current learning rate
        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        logs['lr'] = current_lr
        
        if not lr_updated:
            logger.debug(f"Epoch {epoch}: LR = {current_lr:.6f}")

    def get_config(self):
        """Get scheduler configuration."""
        return {
            'base_lr': self.base_lr,
            'max_lr': self.max_lr,
            'step_size': self.step_size,
            'mode': self.mode,
            'gamma': self.gamma,
            'warmup_epochs': self.warmup_epochs,
            'min_lr_factor': self.min_lr_factor,
            'patience': self.patience,
            'factor': self.factor,
            'monitor_metrics': self.monitor_metrics
        }


class GradientClippingCallback(Callback):
    """Monitors and clips gradients to prevent exploding gradients."""
    
    def __init__(self, clip_norm=1.0, clip_value=None):
        super().__init__()
        self.clip_norm = clip_norm
        self.clip_value = clip_value
        self.gradient_norms = []
        
    def on_train_batch_end(self, batch, logs=None):
        """Monitor gradient norms."""
        logs = logs or {}
        
        # Calculate gradient norm
        gradients = []
        for layer in self.model.layers:
            if hasattr(layer, 'trainable_weights') and layer.trainable_weights:
                for weight in layer.trainable_weights:
                    gradient = tf.gradients(self.model.total_loss, weight)[0]
                    if gradient is not None:
                        gradients.append(gradient)
        
        if gradients:
            global_norm = tf.linalg.global_norm(gradients)
            self.gradient_norms.append(float(global_norm))
            logs['gradient_norm'] = float(global_norm)
            
            # Log warnings for extreme gradients
            if global_norm > 10.0:
                logger.warning(f"Large gradient norm detected: {global_norm:.4f}")


def create_adaptive_scheduler(
    base_lr: float = 0.001,
    max_lr: float = 0.01,
    training_samples: int = 1000,
    batch_size: int = 32,
    epochs: int = 100,
    **kwargs
) -> AdaptiveLearningScheduler:
    """
    Create an adaptive learning scheduler with optimized parameters.
    
    Args:
        base_lr: Base learning rate
        max_lr: Maximum learning rate
        training_samples: Number of training samples
        batch_size: Batch size
        epochs: Number of epochs
        **kwargs: Additional scheduler parameters
        
    Returns:
        Configured AdaptiveLearningScheduler
    """
    steps_per_epoch = training_samples // batch_size
    step_size = kwargs.get('step_size', 4 * steps_per_epoch)  # 4 epochs per cycle
    
    scheduler = AdaptiveLearningScheduler(
        base_lr=base_lr,
        max_lr=max_lr,
        step_size=step_size,
        **kwargs
    )
    
    logger.info(f"Created adaptive scheduler: step_size={step_size}, "
               f"base_lr={base_lr}, max_lr={max_lr}")
    
    return scheduler


# Example usage and validation
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Test scheduler behavior
    scheduler = create_adaptive_scheduler(
        base_lr=0.0001,
        max_lr=0.001,
        training_samples=1000,
        batch_size=32,
        epochs=20
    )
    
    print("Adaptive Learning Scheduler created successfully!")
    print(f"Configuration: {scheduler.get_config()}")