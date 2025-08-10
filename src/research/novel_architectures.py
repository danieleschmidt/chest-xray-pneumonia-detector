# Novel CNN Architectures for Pneumonia Detection
# Research-grade implementations with comparative baselines

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2B0, ResNet50V2, DenseNet121
import numpy as np
from typing import Tuple, Optional, Dict, Any


class DualPathCNN(Model):
    """
    Novel dual-path CNN architecture that processes spatial and textural features separately.
    Research hypothesis: Separate pathways for texture analysis and spatial pattern recognition
    improve pneumonia detection accuracy compared to single-path CNNs.
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3), num_classes: int = 1):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Spatial pathway - focuses on anatomical structures
        self.spatial_conv1 = layers.Conv2D(32, (7, 7), activation='relu', padding='same')
        self.spatial_conv2 = layers.Conv2D(64, (5, 5), activation='relu', padding='same')
        self.spatial_pool1 = layers.MaxPooling2D((2, 2))
        self.spatial_conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.spatial_pool2 = layers.MaxPooling2D((2, 2))
        
        # Texture pathway - focuses on local patterns and opacity changes
        self.texture_conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.texture_conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.texture_pool1 = layers.MaxPooling2D((2, 2))
        self.texture_conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.texture_pool2 = layers.MaxPooling2D((2, 2))
        
        # Cross-attention mechanism between pathways
        self.cross_attention = layers.MultiHeadAttention(num_heads=4, key_dim=128)
        
        # Fusion and classification layers
        self.global_pool = layers.GlobalAveragePooling2D()
        self.fusion_dense = layers.Dense(256, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.classifier = layers.Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
        
    def call(self, inputs, training=None):
        # Spatial pathway
        spatial = self.spatial_conv1(inputs)
        spatial = self.spatial_conv2(spatial)
        spatial = self.spatial_pool1(spatial)
        spatial = self.spatial_conv3(spatial)
        spatial = self.spatial_pool2(spatial)
        
        # Texture pathway
        texture = self.texture_conv1(inputs)
        texture = self.texture_conv2(texture)
        texture = self.texture_pool1(texture)
        texture = self.texture_conv3(texture)
        texture = self.texture_pool2(texture)
        
        # Cross-attention between pathways
        spatial_flat = tf.reshape(spatial, [tf.shape(spatial)[0], -1, spatial.shape[-1]])
        texture_flat = tf.reshape(texture, [tf.shape(texture)[0], -1, texture.shape[-1]])
        
        attended_spatial = self.cross_attention(spatial_flat, texture_flat)
        attended_texture = self.cross_attention(texture_flat, spatial_flat)
        
        # Global pooling and fusion
        spatial_pooled = tf.reduce_mean(attended_spatial, axis=1)
        texture_pooled = tf.reduce_mean(attended_texture, axis=1)
        
        # Concatenate pathways
        fused = tf.concat([spatial_pooled, texture_pooled], axis=-1)
        fused = self.fusion_dense(fused)
        fused = self.dropout(fused, training=training)
        
        return self.classifier(fused)


class HierarchicalAttentionCNN(Model):
    """
    Novel hierarchical attention mechanism that focuses on different scales of pathology.
    Research hypothesis: Multi-scale attention improves detection of pneumonia patterns
    at different anatomical scales (alveolar, lobar, segmental).
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3), num_classes: int = 1):
        super().__init__()
        
        # Multi-scale feature extraction
        self.backbone = EfficientNetV2B0(include_top=False, weights='imagenet', input_shape=input_shape)
        
        # Attention modules at different scales
        self.fine_attention = layers.MultiHeadAttention(num_heads=8, key_dim=64)
        self.medium_attention = layers.MultiHeadAttention(num_heads=4, key_dim=128)
        self.coarse_attention = layers.MultiHeadAttention(num_heads=2, key_dim=256)
        
        # Scale-specific pooling
        self.fine_pool = layers.AveragePooling2D((2, 2))
        self.medium_pool = layers.AveragePooling2D((4, 4))
        self.coarse_pool = layers.AveragePooling2D((8, 8))
        
        # Classification head
        self.global_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(512, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.classifier = layers.Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
        
    def call(self, inputs, training=None):
        # Extract multi-scale features
        features = self.backbone(inputs, training=training)
        
        # Create multi-scale feature maps
        fine_features = self.fine_pool(features)
        medium_features = self.medium_pool(features)
        coarse_features = self.coarse_pool(features)
        
        # Flatten for attention
        fine_flat = tf.reshape(fine_features, [tf.shape(fine_features)[0], -1, fine_features.shape[-1]])
        medium_flat = tf.reshape(medium_features, [tf.shape(medium_features)[0], -1, medium_features.shape[-1]])
        coarse_flat = tf.reshape(coarse_features, [tf.shape(coarse_features)[0], -1, coarse_features.shape[-1]])
        
        # Apply hierarchical attention
        fine_attended = self.fine_attention(fine_flat, fine_flat)
        medium_attended = self.medium_attention(medium_flat, medium_flat)
        coarse_attended = self.coarse_attention(coarse_flat, coarse_flat)
        
        # Pool attention outputs
        fine_pooled = tf.reduce_mean(fine_attended, axis=1)
        medium_pooled = tf.reduce_mean(medium_attended, axis=1)
        coarse_pooled = tf.reduce_mean(coarse_attended, axis=1)
        
        # Combine multi-scale features
        combined = tf.concat([fine_pooled, medium_pooled, coarse_pooled], axis=-1)
        
        # Classification
        x = self.dense1(combined)
        x = self.dropout(x, training=training)
        return self.classifier(x)


def create_ensemble_model(input_shape: Tuple[int, int, int] = (224, 224, 3), 
                         num_classes: int = 1) -> Model:
    """
    Creates an ensemble of diverse architectures for robust predictions.
    Research hypothesis: Ensemble of complementary architectures reduces variance
    and improves generalization on diverse pneumonia presentations.
    """
    
    inputs = layers.Input(shape=input_shape)
    
    # Model 1: Traditional CNN with attention
    efficientnet = EfficientNetV2B0(include_top=False, weights='imagenet')(inputs)
    efficientnet_pooled = layers.GlobalAveragePooling2D()(efficientnet)
    
    # Model 2: DenseNet for feature reuse
    densenet = DenseNet121(include_top=False, weights='imagenet')(inputs)
    densenet_pooled = layers.GlobalAveragePooling2D()(densenet)
    
    # Model 3: ResNet for deep features
    resnet = ResNet50V2(include_top=False, weights='imagenet')(inputs)
    resnet_pooled = layers.GlobalAveragePooling2D()(resnet)
    
    # Ensemble fusion with learned weights
    ensemble_features = tf.stack([efficientnet_pooled, densenet_pooled, resnet_pooled], axis=1)
    
    # Attention-based fusion
    fusion_attention = layers.MultiHeadAttention(num_heads=1, key_dim=128)(
        ensemble_features, ensemble_features
    )
    fused_features = tf.reduce_mean(fusion_attention, axis=1)
    
    # Classification head
    x = layers.Dense(256, activation='relu')(fused_features)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs, name='EnsembleCNN')


class UncertaintyAwareCNN(Model):
    """
    CNN with built-in uncertainty quantification using Monte Carlo Dropout.
    Research hypothesis: Uncertainty estimates help identify difficult cases
    and improve clinical decision support.
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3), num_classes: int = 1):
        super().__init__()
        
        self.backbone = EfficientNetV2B0(include_top=False, weights='imagenet', input_shape=input_shape)
        self.global_pool = layers.GlobalAveragePooling2D()
        
        # Multiple dropout layers for uncertainty estimation
        self.dropout1 = layers.Dropout(0.3)
        self.dense1 = layers.Dense(512, activation='relu')
        self.dropout2 = layers.Dropout(0.4)
        self.dense2 = layers.Dense(256, activation='relu')
        self.dropout3 = layers.Dropout(0.5)
        
        # Output layers
        self.classifier = layers.Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
        self.uncertainty_head = layers.Dense(1, activation='relu', name='uncertainty')
        
    def call(self, inputs, training=None):
        x = self.backbone(inputs, training=training)
        x = self.global_pool(x)
        
        # Apply multiple dropout layers
        x = self.dropout1(x, training=True)  # Always apply dropout for uncertainty
        x = self.dense1(x)
        x = self.dropout2(x, training=True)
        x = self.dense2(x)
        x = self.dropout3(x, training=True)
        
        # Dual outputs: prediction and uncertainty
        prediction = self.classifier(x)
        uncertainty = self.uncertainty_head(x)
        
        return {'prediction': prediction, 'uncertainty': uncertainty}
    
    def predict_with_uncertainty(self, x, num_samples: int = 100):
        """
        Generate predictions with uncertainty estimates using Monte Carlo dropout.
        """
        predictions = []
        uncertainties = []
        
        for _ in range(num_samples):
            output = self(x, training=True)
            predictions.append(output['prediction'])
            uncertainties.append(output['uncertainty'])
        
        predictions = tf.stack(predictions)
        uncertainties = tf.stack(uncertainties)
        
        # Calculate statistics
        mean_pred = tf.reduce_mean(predictions, axis=0)
        std_pred = tf.math.reduce_std(predictions, axis=0)
        mean_uncertainty = tf.reduce_mean(uncertainties, axis=0)
        
        return {
            'prediction': mean_pred,
            'epistemic_uncertainty': std_pred,  # Model uncertainty
            'aleatoric_uncertainty': mean_uncertainty,  # Data uncertainty
            'total_uncertainty': std_pred + mean_uncertainty
        }


def build_research_baseline_models() -> Dict[str, Model]:
    """
    Build baseline models for comparative research studies.
    """
    models = {}
    
    # Standard baselines
    models['efficientnet_baseline'] = tf.keras.Sequential([
        EfficientNetV2B0(include_top=False, weights='imagenet', input_shape=(224, 224, 3)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    models['resnet_baseline'] = tf.keras.Sequential([
        ResNet50V2(include_top=False, weights='imagenet', input_shape=(224, 224, 3)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    models['densenet_baseline'] = tf.keras.Sequential([
        DenseNet121(include_top=False, weights='imagenet', input_shape=(224, 224, 3)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Novel architectures
    models['dual_path_cnn'] = DualPathCNN()
    models['hierarchical_attention'] = HierarchicalAttentionCNN()
    models['ensemble_cnn'] = create_ensemble_model()
    models['uncertainty_aware'] = UncertaintyAwareCNN()
    
    return models


if __name__ == "__main__":
    # Demonstration of novel architectures
    models = build_research_baseline_models()
    
    # Print model summaries
    dummy_input = tf.random.normal((1, 224, 224, 3))
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Model: {name}")
        print(f"{'='*50}")
        try:
            output = model(dummy_input)
            if isinstance(output, dict):
                for key, value in output.items():
                    print(f"{key}: {value.shape}")
            else:
                print(f"Output shape: {output.shape}")
        except Exception as e:
            print(f"Error testing model: {e}")