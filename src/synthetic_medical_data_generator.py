"""Synthetic medical data generation for realistic pipeline testing."""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import math

from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np


@dataclass
class MedicalImageConfiguration:
    """Configuration for synthetic medical image generation."""
    
    image_size: Tuple[int, int] = (224, 224)
    pathology_probability: float = 0.3
    noise_level: float = 0.1
    contrast_enhancement: bool = True
    add_anatomical_markers: bool = True
    brightness_variation: float = 0.2
    pathology_types: List[str] = None
    
    def __post_init__(self):
        if self.pathology_types is None:
            self.pathology_types = ["bacterial_pneumonia", "viral_pneumonia", "consolidation"]


@dataclass
class DatasetMetadata:
    """Metadata for synthetic medical dataset."""
    
    total_images: int
    normal_count: int
    pneumonia_count: int
    image_format: str
    image_size: Tuple[int, int]
    generation_timestamp: str
    pathology_types: List[str]
    quality_metrics: Dict[str, float]
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert metadata to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


def generate_synthetic_chest_xray(
    config: MedicalImageConfiguration,
    is_pathological: bool = False,
    pathology_type: str = "pneumonia"
) -> Image.Image:
    """Generate a synthetic chest X-ray image.
    
    Parameters
    ----------
    config : MedicalImageConfiguration
        Configuration for image generation
    is_pathological : bool, default=False
        Whether to generate pathological features
    pathology_type : str, default="pneumonia"
        Type of pathology to simulate
        
    Returns
    -------
    PIL.Image.Image
        Generated synthetic chest X-ray image
    """
    width, height = config.image_size
    
    # Create base grayscale image (typical for X-rays)
    img = Image.new('L', (width, height), color=50)  # Dark background
    draw = ImageDraw.Draw(img)
    
    # Generate anatomical structures
    _add_ribcage(draw, width, height, config.add_anatomical_markers)
    _add_lung_fields(draw, width, height, is_pathological, pathology_type)
    _add_heart_shadow(draw, width, height)
    _add_spine(draw, width, height)
    
    # Apply post-processing effects
    if config.contrast_enhancement:
        img = _enhance_contrast(img)
    
    if config.noise_level > 0:
        img = _add_medical_noise(img, config.noise_level)
    
    if config.brightness_variation > 0:
        img = _apply_brightness_variation(img, config.brightness_variation)
    
    # Convert to RGB for compatibility with training pipeline
    img = img.convert('RGB')
    
    return img


def _add_ribcage(draw: ImageDraw.Draw, width: int, height: int, add_markers: bool):
    """Add ribcage structure to chest X-ray."""
    if not add_markers:
        return
    
    # Draw simplified rib outlines
    center_x = width // 2
    rib_color = 180  # Lighter gray for bones
    
    for i in range(6):  # 6 visible rib pairs
        y_pos = height // 4 + i * (height // 12)
        
        # Left ribs
        left_start = center_x - width // 6
        left_end = center_x - width // 3
        draw.arc([left_end, y_pos - 10, left_start, y_pos + 10], 
                start=0, end=180, fill=rib_color, width=2)
        
        # Right ribs
        right_start = center_x + width // 6
        right_end = center_x + width // 3
        draw.arc([right_start, y_pos - 10, right_end, y_pos + 10], 
                start=0, end=180, fill=rib_color, width=2)


def _add_lung_fields(draw: ImageDraw.Draw, width: int, height: int, 
                    is_pathological: bool, pathology_type: str):
    """Add lung field areas to chest X-ray."""
    center_x = width // 2
    lung_color = 120  # Medium gray for healthy lung tissue
    
    # Left lung field
    left_lung_coords = [
        center_x - width // 3, height // 4,
        center_x - width // 10, height // 4,
        center_x - width // 10, height * 3 // 4,
        center_x - width // 3, height * 3 // 4
    ]
    draw.polygon(left_lung_coords, fill=lung_color)
    
    # Right lung field  
    right_lung_coords = [
        center_x + width // 10, height // 4,
        center_x + width // 3, height // 4,
        center_x + width // 3, height * 3 // 4,
        center_x + width // 10, height * 3 // 4
    ]
    draw.polygon(right_lung_coords, fill=lung_color)
    
    # Add pathological features if needed
    if is_pathological:
        _add_pathological_features(draw, width, height, pathology_type)


def _add_pathological_features(draw: ImageDraw.Draw, width: int, height: int, 
                              pathology_type: str):
    """Add pathological features to simulate pneumonia or other conditions."""
    center_x = width // 2
    
    if pathology_type in ["pneumonia", "bacterial_pneumonia", "viral_pneumonia"]:
        # Add consolidation/opacity patterns
        opacity_color = 90  # Darker gray for fluid/consolidation
        
        # Random consolidation areas
        num_areas = random.randint(1, 3)
        for _ in range(num_areas):
            # Random position in lung fields
            side = random.choice(['left', 'right'])
            if side == 'left':
                x_center = center_x - width // 5
            else:
                x_center = center_x + width // 5
            
            y_center = random.randint(height // 3, height * 2 // 3)
            area_size = random.randint(width // 20, width // 10)
            
            # Draw irregular consolidation pattern
            draw.ellipse([
                x_center - area_size, y_center - area_size,
                x_center + area_size, y_center + area_size
            ], fill=opacity_color)
            
            # Add some texture/irregularity
            for _ in range(5):
                offset_x = random.randint(-area_size//2, area_size//2)
                offset_y = random.randint(-area_size//2, area_size//2)
                small_size = area_size // 3
                draw.ellipse([
                    x_center + offset_x - small_size, y_center + offset_y - small_size,
                    x_center + offset_x + small_size, y_center + offset_y + small_size
                ], fill=opacity_color - 20)


def _add_heart_shadow(draw: ImageDraw.Draw, width: int, height: int):
    """Add cardiac silhouette to chest X-ray."""
    center_x = width // 2
    heart_color = 80  # Darker gray for heart shadow
    
    # Simplified heart shape
    heart_coords = [
        center_x - width // 8, height // 3,
        center_x + width // 6, height // 3,
        center_x + width // 6, height * 2 // 3,
        center_x, height * 3 // 4,
        center_x - width // 8, height * 2 // 3
    ]
    draw.polygon(heart_coords, fill=heart_color)


def _add_spine(draw: ImageDraw.Draw, width: int, height: int):
    """Add spinal column to chest X-ray."""
    center_x = width // 2
    spine_color = 200  # Light gray for spine
    
    # Vertical spine line
    draw.line([center_x, height // 6, center_x, height * 5 // 6], 
              fill=spine_color, width=max(2, width // 100))


def _enhance_contrast(img: Image.Image) -> Image.Image:
    """Enhance contrast of medical image."""
    enhancer = ImageEnhance.Contrast(img)
    # Typical medical X-ray contrast enhancement
    enhanced_img = enhancer.enhance(1.3)
    return enhanced_img


def _add_medical_noise(img: Image.Image, noise_level: float) -> Image.Image:
    """Add realistic medical imaging noise."""
    if noise_level <= 0:
        return img
    
    # Convert to numpy for noise addition
    img_array = np.array(img)
    
    # Add Gaussian noise (common in medical imaging)
    noise = np.random.normal(0, noise_level * 50, img_array.shape)
    noisy_array = img_array + noise
    
    # Clip values to valid range
    noisy_array = np.clip(noisy_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(noisy_array)


def _apply_brightness_variation(img: Image.Image, variation: float) -> Image.Image:
    """Apply brightness variation to simulate different exposure conditions."""
    if variation <= 0:
        return img
    
    # Random brightness adjustment
    brightness_factor = 1.0 + random.uniform(-variation, variation)
    enhancer = ImageEnhance.Brightness(img)
    adjusted_img = enhancer.enhance(brightness_factor)
    
    return adjusted_img


def create_synthetic_medical_dataset(
    output_dir: str,
    total_images: int,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    config: MedicalImageConfiguration = None
) -> str:
    """Create a complete synthetic medical dataset.
    
    Parameters
    ----------
    output_dir : str
        Directory to create dataset in
    total_images : int
        Total number of images to generate
    train_split : float, default=0.8
        Fraction for training set
    val_split : float, default=0.1
        Fraction for validation set
    test_split : float, default=0.1
        Fraction for test set
    config : MedicalImageConfiguration, optional
        Configuration for image generation
        
    Returns
    -------
    str
        Path to created dataset directory
    """
    if config is None:
        config = MedicalImageConfiguration()
    
    # Validate splits
    if abs(train_split + val_split + test_split - 1.0) > 0.01:
        raise ValueError("Dataset splits must sum to 1.0")
    
    # Create dataset directory structure
    dataset_dir = os.path.join(output_dir, "synthetic_medical_dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    splits = {
        "train": int(total_images * train_split),
        "val": int(total_images * val_split),
        "test": int(total_images * test_split)
    }
    
    # Adjust for rounding
    splits["train"] = total_images - splits["val"] - splits["test"]
    
    # Create directory structure
    for split in splits.keys():
        for class_name in ["NORMAL", "PNEUMONIA"]:
            class_dir = os.path.join(dataset_dir, split, class_name)
            os.makedirs(class_dir, exist_ok=True)
    
    # Generate images for each split
    total_normal = 0
    total_pneumonia = 0
    
    for split, num_images in splits.items():
        print(f"Generating {num_images} images for {split} split...")
        
        for i in range(num_images):
            # Determine if image should be pathological
            is_pathological = random.random() < config.pathology_probability
            
            if is_pathological:
                class_name = "PNEUMONIA"
                pathology_type = random.choice(config.pathology_types)
                total_pneumonia += 1
            else:
                class_name = "NORMAL"
                pathology_type = None
                total_normal += 1
            
            # Generate image
            img = generate_synthetic_chest_xray(
                config=config,
                is_pathological=is_pathological,
                pathology_type=pathology_type
            )
            
            # Save image
            img_filename = f"{class_name.lower()}_{i:04d}.png"
            img_path = os.path.join(dataset_dir, split, class_name, img_filename)
            img.save(img_path, "PNG")
    
    # Calculate quality metrics
    quality_metrics = {
        "avg_contrast": 0.75,  # Simulated metric
        "noise_level": config.noise_level,
        "pathology_distribution": total_pneumonia / total_images,
        "image_quality_score": 0.85  # Simulated metric
    }
    
    # Create metadata
    metadata = DatasetMetadata(
        total_images=total_images,
        normal_count=total_normal,
        pneumonia_count=total_pneumonia,
        image_format="PNG",
        image_size=config.image_size,
        generation_timestamp=datetime.now().isoformat(),
        pathology_types=config.pathology_types,
        quality_metrics=quality_metrics,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split
    )
    
    # Save metadata
    metadata_path = os.path.join(dataset_dir, "dataset_metadata.json")
    with open(metadata_path, 'w') as f:
        f.write(metadata.to_json())
    
    print(f"✅ Synthetic medical dataset created successfully!")
    print(f"📍 Location: {dataset_dir}")
    print(f"📊 Total Images: {total_images}")
    print(f"🫁 Normal: {total_normal}, 🦠 Pneumonia: {total_pneumonia}")
    print(f"📋 Metadata saved to: {metadata_path}")
    
    return dataset_dir


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic medical datasets for testing"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for synthetic dataset"
    )
    parser.add_argument(
        "--total-images",
        type=int,
        default=100,
        help="Total number of images to generate"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Size of generated images (square)"
    )
    parser.add_argument(
        "--pathology-probability",
        type=float,
        default=0.3,
        help="Probability of generating pathological images"
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.1,
        help="Level of noise to add (0.0 to 1.0)"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction for training set"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction for validation set"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Fraction for test set"
    )
    parser.add_argument(
        "--contrast-enhancement",
        action="store_true",
        help="Apply contrast enhancement"
    )
    parser.add_argument(
        "--anatomical-markers",
        action="store_true",
        help="Add anatomical markers (ribs, spine, etc.)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if args.train_split + args.val_split + args.test_split != 1.0:
        print("❌ Error: Dataset splits must sum to 1.0")
        sys.exit(1)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Create configuration
    config = MedicalImageConfiguration(
        image_size=(args.image_size, args.image_size),
        pathology_probability=args.pathology_probability,
        noise_level=args.noise_level,
        contrast_enhancement=args.contrast_enhancement,
        add_anatomical_markers=args.anatomical_markers
    )
    
    if args.verbose:
        print("🔧 Configuration:")
        print(f"  Image Size: {config.image_size}")
        print(f"  Pathology Probability: {config.pathology_probability}")
        print(f"  Noise Level: {config.noise_level}")
        print(f"  Contrast Enhancement: {config.contrast_enhancement}")
        print(f"  Anatomical Markers: {config.add_anatomical_markers}")
        print()
    
    # Generate dataset
    start_time = time.time()
    
    dataset_path = create_synthetic_medical_dataset(
        output_dir=args.output_dir,
        total_images=args.total_images,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        config=config
    )
    
    generation_time = time.time() - start_time
    
    print(f"⏱️  Generation completed in {generation_time:.2f} seconds")
    print(f"📈 Performance: {args.total_images / generation_time:.1f} images/second")
    
    if args.verbose:
        # Display some statistics
        metadata_path = os.path.join(dataset_path, "dataset_metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print("\n📊 Dataset Statistics:")
        print(f"  Normal Images: {metadata['normal_count']}")
        print(f"  Pneumonia Images: {metadata['pneumonia_count']}")
        print(f"  Pathology Distribution: {metadata['quality_metrics']['pathology_distribution']:.2%}")
        print(f"  Image Format: {metadata['image_format']}")
        print(f"  Image Size: {metadata['image_size']}")


if __name__ == "__main__":
    main()