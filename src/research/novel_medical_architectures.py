"""Novel Medical AI Architectures for Chest X-Ray Analysis.

Implements cutting-edge neural network architectures specifically designed
for medical imaging applications with focus on interpretability and accuracy.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class AttentionGatedUNet(Model):
    """Attention-Gated U-Net for medical image segmentation and analysis."""
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 2,
                 base_filters: int = 64,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_shape_param = input_shape
        self.num_classes = num_classes
        self.base_filters = base_filters
        
        # Build the network
        self._build_network()
    
    def _build_network(self):
        """Build the attention-gated U-Net architecture."""
        inputs = layers.Input(shape=self.input_shape_param)
        
        # Encoder path
        conv1 = self._conv_block(inputs, self.base_filters)
        pool1 = layers.MaxPooling2D((2, 2))(conv1)
        
        conv2 = self._conv_block(pool1, self.base_filters * 2)
        pool2 = layers.MaxPooling2D((2, 2))(conv2)
        
        conv3 = self._conv_block(pool2, self.base_filters * 4)
        pool3 = layers.MaxPooling2D((2, 2))(conv3)
        
        conv4 = self._conv_block(pool3, self.base_filters * 8)
        pool4 = layers.MaxPooling2D((2, 2))(conv4)
        
        # Bottleneck
        bottleneck = self._conv_block(pool4, self.base_filters * 16)
        
        # Decoder path with attention gates
        up4 = self._upsampling_block(bottleneck, conv4, self.base_filters * 8)
        up3 = self._upsampling_block(up4, conv3, self.base_filters * 4)
        up2 = self._upsampling_block(up3, conv2, self.base_filters * 2)
        up1 = self._upsampling_block(up2, conv1, self.base_filters)
        
        # Global average pooling for classification
        gap = layers.GlobalAveragePooling2D()(up1)
        
        # Classification head
        dense1 = layers.Dense(256, activation='relu')(gap)
        dense1 = layers.Dropout(0.5)(dense1)
        dense2 = layers.Dense(128, activation='relu')(dense1)
        dense2 = layers.Dropout(0.3)(dense2)
        
        if self.num_classes == 1:
            outputs = layers.Dense(1, activation='sigmoid', name='classification')(dense2)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax', name='classification')(dense2)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='AttentionGatedUNet')
    
    def _conv_block(self, x, filters: int):
        """Convolutional block with batch normalization and activation."""
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x
    
    def _attention_gate(self, gate, skip_connection, inter_channels):
        """Attention gate mechanism."""
        gate_conv = layers.Conv2D(inter_channels, 1, strides=1, padding='same')(gate)
        gate_conv = layers.BatchNormalization()(gate_conv)
        
        skip_conv = layers.Conv2D(inter_channels, 1, strides=1, padding='same')(skip_connection)
        skip_conv = layers.BatchNormalization()(skip_conv)
        
        # Add and apply ReLU
        add = layers.Add()([gate_conv, skip_conv])
        add = layers.Activation('relu')(add)
        
        # Apply attention
        attention = layers.Conv2D(1, 1, strides=1, padding='same')(add)
        attention = layers.BatchNormalization()(attention)
        attention = layers.Activation('sigmoid')(attention)
        
        # Apply attention to skip connection
        attended = layers.Multiply()([skip_connection, attention])
        
        return attended
    
    def _upsampling_block(self, x, skip_connection, filters: int):
        """Upsampling block with attention gate."""
        # Upsampling
        up = layers.UpSampling2D((2, 2))(x)
        up = layers.Conv2D(filters, 2, padding='same')(up)
        up = layers.BatchNormalization()(up)
        up = layers.Activation('relu')(up)
        
        # Apply attention gate
        attended_skip = self._attention_gate(up, skip_connection, filters // 2)
        
        # Concatenate
        concat = layers.Concatenate()([up, attended_skip])
        
        # Convolutional block
        conv = self._conv_block(concat, filters)
        
        return conv
    
    def call(self, inputs, training=None):
        """Forward pass."""
        return self.model(inputs, training=training)


class DenseNetWithCAM(Model):
    """DenseNet with Class Activation Mapping for interpretability."""
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 2,
                 growth_rate: int = 32,
                 block_config: Tuple[int, ...] = (6, 12, 24, 16),
                 **kwargs):
        super().__init__(**kwargs)
        self.input_shape_param = input_shape
        self.num_classes = num_classes
        self.growth_rate = growth_rate
        self.block_config = block_config
        
        self._build_network()
    
    def _build_network(self):
        """Build DenseNet with CAM."""
        inputs = layers.Input(shape=self.input_shape_param)
        
        # Initial convolution
        x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
        # Dense blocks
        num_features = 64
        for i, num_layers in enumerate(self.block_config):
            x = self._dense_block(x, num_layers, num_features)
            num_features += num_layers * self.growth_rate
            
            if i != len(self.block_config) - 1:
                x = self._transition_layer(x, num_features // 2)
                num_features = num_features // 2
        
        # Final batch normalization
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Class Activation Mapping
        # Keep spatial dimensions for CAM
        self.feature_maps = x
        
        # Global Average Pooling
        gap = layers.GlobalAveragePooling2D()(x)
        
        # Final classification layer (important for CAM)
        if self.num_classes == 1:
            outputs = layers.Dense(1, activation='sigmoid', name='predictions')(gap)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax', name='predictions')(gap)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='DenseNetWithCAM')
    
    def _dense_block(self, x, num_layers: int, num_input_features: int):
        """Dense block with growth rate."""
        for i in range(num_layers):
            bn_function = self._bn_function(x)
            new_features = layers.Conv2D(self.growth_rate, 3, padding='same')(bn_function)
            x = layers.Concatenate()([x, new_features])
        return x
    
    def _bn_function(self, x):
        """Bottleneck function."""
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(4 * self.growth_rate, 1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x
    
    def _transition_layer(self, x, num_output_features: int):
        """Transition layer between dense blocks."""
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(num_output_features, 1)(x)
        x = layers.AveragePooling2D(2, strides=2)(x)
        return x
    
    def call(self, inputs, training=None):
        """Forward pass."""
        return self.model(inputs, training=training)
    
    def get_cam(self, image: np.ndarray, class_index: int = 0) -> np.ndarray:
        """Generate Class Activation Map."""
        # Get feature maps
        feature_extractor = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('batch_normalization').output  # Last conv features
        )
        
        features = feature_extractor(image)
        
        # Get weights of the final dense layer
        class_weights = self.model.get_layer('predictions').get_weights()[0]
        
        if self.num_classes == 1:
            weights = class_weights[:, 0]
        else:
            weights = class_weights[:, class_index]
        
        # Generate CAM
        cam = np.zeros(features.shape[1:3], dtype=np.float32)
        
        for i, weight in enumerate(weights):
            cam += weight * features[0, :, :, i]
        
        # Apply ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam) if np.max(cam) > 0 else cam
        
        return cam


class MultiScaleResidualNet(Model):
    """Multi-scale Residual Network for medical image analysis."""
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 2,
                 scales: List[int] = [1, 2, 4],
                 **kwargs):
        super().__init__(**kwargs)
        self.input_shape_param = input_shape
        self.num_classes = num_classes
        self.scales = scales
        
        self._build_network()
    
    def _build_network(self):
        """Build multi-scale residual network."""
        inputs = layers.Input(shape=self.input_shape_param)
        
        # Multi-scale feature extraction
        scale_features = []
        
        for scale in self.scales:
            # Resize input for different scales
            if scale == 1:
                scaled_input = inputs
            else:
                new_size = (self.input_shape_param[0] // scale, 
                           self.input_shape_param[1] // scale)
                scaled_input = layers.Lambda(
                    lambda x: tf.image.resize(x, new_size)
                )(inputs)
                # Resize back to original size
                scaled_input = layers.Lambda(
                    lambda x: tf.image.resize(x, self.input_shape_param[:2])
                )(scaled_input)
            
            # Extract features at this scale
            features = self._residual_feature_extractor(scaled_input, f'scale_{scale}')
            scale_features.append(features)
        
        # Fuse multi-scale features
        if len(scale_features) > 1:
            fused_features = layers.Add()(scale_features)
        else:
            fused_features = scale_features[0]
        
        # Additional residual blocks
        x = self._residual_block(fused_features, 256, 'fused_1')
        x = self._residual_block(x, 512, 'fused_2')
        
        # Global average pooling
        gap = layers.GlobalAveragePooling2D()(x)
        
        # Classification head
        x = layers.Dense(512, activation='relu')(gap)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        if self.num_classes == 1:
            outputs = layers.Dense(1, activation='sigmoid', name='classification')(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax', name='classification')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='MultiScaleResidualNet')
    
    def _residual_feature_extractor(self, x, name_prefix: str):
        """Residual feature extractor for a given scale."""
        # Initial convolution
        x = layers.Conv2D(64, 7, strides=2, padding='same', name=f'{name_prefix}_conv1')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu1')(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same', name=f'{name_prefix}_pool1')(x)
        
        # Residual blocks
        x = self._residual_block(x, 64, f'{name_prefix}_block1')
        x = self._residual_block(x, 128, f'{name_prefix}_block2', stride=2)
        x = self._residual_block(x, 256, f'{name_prefix}_block3', stride=2)
        
        return x
    
    def _residual_block(self, x, filters: int, name_prefix: str, stride: int = 1):
        """Residual block with shortcut connection."""
        shortcut = x
        
        # First convolution
        x = layers.Conv2D(filters, 3, strides=stride, padding='same', 
                         name=f'{name_prefix}_conv1')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu1')(x)
        
        # Second convolution
        x = layers.Conv2D(filters, 3, padding='same', name=f'{name_prefix}_conv2')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
        
        # Shortcut connection
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, strides=stride, 
                                   name=f'{name_prefix}_shortcut')(shortcut)
            shortcut = layers.BatchNormalization(name=f'{name_prefix}_shortcut_bn')(shortcut)
        
        # Add shortcut and apply activation
        x = layers.Add(name=f'{name_prefix}_add')([x, shortcut])
        x = layers.Activation('relu', name=f'{name_prefix}_relu2')(x)
        
        return x
    
    def call(self, inputs, training=None):
        """Forward pass."""
        return self.model(inputs, training=training)


class TransformerCNN(Model):
    """Hybrid CNN-Transformer for medical image analysis."""
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 2,
                 patch_size: int = 16,
                 num_transformer_layers: int = 6,
                 num_heads: int = 8,
                 hidden_dim: int = 512,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_shape_param = input_shape
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_transformer_layers = num_transformer_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        self._build_network()
    
    def _build_network(self):
        """Build CNN-Transformer hybrid network."""
        inputs = layers.Input(shape=self.input_shape_param)
        
        # CNN feature extraction
        x = layers.Conv2D(64, 3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(256, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(512, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Convert to patches for transformer
        # Reshape to patches
        batch_size = tf.shape(x)[0]
        height, width = x.shape[1], x.shape[2]
        channels = x.shape[3]
        
        # Create patches
        patch_dims = height * width
        x_patches = layers.Reshape((patch_dims, channels))(x)
        
        # Add positional encoding
        positions = tf.range(start=0, limit=patch_dims, delta=1)
        pos_encoding = layers.Embedding(
            input_dim=patch_dims, 
            output_dim=channels
        )(positions)
        x_patches = x_patches + pos_encoding
        
        # Transformer blocks
        for i in range(self.num_transformer_layers):
            x_patches = self._transformer_block(
                x_patches, 
                num_heads=self.num_heads, 
                hidden_dim=self.hidden_dim,
                name_prefix=f'transformer_{i}'
            )
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x_patches)
        
        # Classification head
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        if self.num_classes == 1:
            outputs = layers.Dense(1, activation='sigmoid', name='classification')(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax', name='classification')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='TransformerCNN')
    
    def _transformer_block(self, x, num_heads: int, hidden_dim: int, name_prefix: str):
        """Transformer block with multi-head attention."""
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=x.shape[-1] // num_heads,
            name=f'{name_prefix}_attention'
        )(x, x)
        
        # Add & Norm
        attention_output = layers.Dropout(0.1)(attention_output)
        x1 = layers.Add(name=f'{name_prefix}_add1')([x, attention_output])
        x1 = layers.LayerNormalization(name=f'{name_prefix}_norm1')(x1)
        
        # Feed Forward Network
        ffn_output = layers.Dense(hidden_dim, activation='relu', 
                                name=f'{name_prefix}_ffn1')(x1)
        ffn_output = layers.Dropout(0.1)(ffn_output)
        ffn_output = layers.Dense(x.shape[-1], name=f'{name_prefix}_ffn2')(ffn_output)
        
        # Add & Norm
        ffn_output = layers.Dropout(0.1)(ffn_output)
        x2 = layers.Add(name=f'{name_prefix}_add2')([x1, ffn_output])
        x2 = layers.LayerNormalization(name=f'{name_prefix}_norm2')(x2)
        
        return x2
    
    def call(self, inputs, training=None):
        """Forward pass."""
        return self.model(inputs, training=training)


def create_novel_architecture(architecture_type: str, 
                            input_shape: Tuple[int, int, int] = (224, 224, 3),
                            num_classes: int = 2,
                            **kwargs) -> Model:
    """Factory function to create novel architectures."""
    
    if architecture_type == "attention_unet":
        return AttentionGatedUNet(input_shape, num_classes, **kwargs)
    elif architecture_type == "densenet_cam":
        return DenseNetWithCAM(input_shape, num_classes, **kwargs)
    elif architecture_type == "multiscale_resnet":
        return MultiScaleResidualNet(input_shape, num_classes, **kwargs)
    elif architecture_type == "transformer_cnn":
        return TransformerCNN(input_shape, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown architecture type: {architecture_type}")


def benchmark_novel_architectures():
    """Benchmark different novel architectures."""
    architectures = [
        "attention_unet",
        "densenet_cam", 
        "multiscale_resnet",
        "transformer_cnn"
    ]
    
    input_shape = (224, 224, 3)
    num_classes = 2
    
    results = {}
    
    for arch_type in architectures:
        logger.info(f"Creating {arch_type} architecture...")
        
        try:
            model = create_novel_architecture(
                arch_type, 
                input_shape=input_shape,
                num_classes=num_classes
            )
            
            # Count parameters
            total_params = model.model.count_params()
            trainable_params = sum([tf.keras.backend.count_params(w) 
                                  for w in model.model.trainable_weights])
            
            results[arch_type] = {
                "total_params": total_params,
                "trainable_params": trainable_params,
                "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
            }
            
            logger.info(f"{arch_type}: {total_params:,} parameters, "
                       f"{results[arch_type]['model_size_mb']:.2f} MB")
            
        except Exception as e:
            logger.error(f"Failed to create {arch_type}: {e}")
            results[arch_type] = {"error": str(e)}
    
    return results


if __name__ == "__main__":
    # Run benchmark
    results = benchmark_novel_architectures()
    
    print("Novel Architecture Benchmark Results:")
    print("=" * 50)
    
    for arch, metrics in results.items():
        print(f"\n{arch.upper()}:")
        if "error" in metrics:
            print(f"  Error: {metrics['error']}")
        else:
            print(f"  Parameters: {metrics['total_params']:,}")
            print(f"  Model Size: {metrics['model_size_mb']:.2f} MB")