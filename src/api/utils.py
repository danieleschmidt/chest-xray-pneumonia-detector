"""
Utility functions for API request processing and validation.
"""

import asyncio
import hashlib
import logging
from io import BytesIO
from typing import Dict, Any, Optional, Tuple
import mimetypes

import numpy as np
from PIL import Image, ImageOps
from fastapi import HTTPException, UploadFile, status
import cv2


logger = logging.getLogger(__name__)


async def validate_image(file: UploadFile) -> Dict[str, Any]:
    """
    Comprehensive image validation for medical imaging.
    
    Args:
        file: Uploaded file to validate
        
    Returns:
        Dict containing validation results
        
    Raises:
        HTTPException: If validation fails
    """
    # Check file size (max 50MB for medical images)
    max_size = 50 * 1024 * 1024  # 50MB
    file_size = 0
    
    # Read file content
    content = await file.read()
    file_size = len(content)
    
    # Reset file position
    await file.seek(0)
    
    if file_size > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {max_size // (1024*1024)}MB"
        )
    
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file provided"
        )
    
    # Validate MIME type
    allowed_types = {
        'image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 
        'image/tiff', 'image/tif', 'image/webp'
    }
    
    content_type = file.content_type
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {content_type}. Allowed: {', '.join(allowed_types)}"
        )
    
    # Validate file extension
    if file.filename:
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        file_extension = file.filename.lower().split('.')[-1]
        if f'.{file_extension}' not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file extension: {file_extension}"
            )
    
    try:
        # Validate image can be opened
        image = Image.open(BytesIO(content))
        
        # Check image properties
        width, height = image.size
        channels = len(image.getbands()) if hasattr(image, 'getbands') else 1
        
        # Minimum resolution check (for medical images)
        min_resolution = 64  # Minimum 64x64
        if width < min_resolution or height < min_resolution:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image resolution too small. Minimum: {min_resolution}x{min_resolution}"
            )
        
        # Maximum resolution check
        max_resolution = 4096  # Maximum 4096x4096
        if width > max_resolution or height > max_resolution:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image resolution too large. Maximum: {max_resolution}x{max_resolution}"
            )
        
        # Validate image integrity
        try:
            image.verify()
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Corrupted or invalid image file"
            )
        
        return {
            "valid": True,
            "format": image.format,
            "size": (width, height),
            "channels": channels,
            "file_size": file_size,
            "content_type": content_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file"
        )


async def preprocess_image(image_data: bytes, target_size: Tuple[int, int] = (150, 150)) -> np.ndarray:
    """
    Preprocess image for model inference.
    
    Args:
        image_data: Raw image bytes
        target_size: Target dimensions for resizing
        
    Returns:
        Preprocessed image as numpy array
    """
    try:
        # Load image
        image = Image.open(BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply CLAHE for better contrast (important for medical images)
        image_array = np.array(image)
        
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        
        # Merge channels back
        enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Convert back to PIL for resizing
        enhanced_image = Image.fromarray(enhanced_rgb)
        
        # Resize with high-quality resampling
        resized_image = enhanced_image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(resized_image)
        
        # Normalize pixel values to [0, 1]
        image_array = image_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
        
    except Exception as e:
        logger.error(f"Image preprocessing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to preprocess image"
        )


def calculate_image_hash(image_data: bytes) -> str:
    """Calculate SHA256 hash of image data for deduplication and audit."""
    return hashlib.sha256(image_data).hexdigest()


async def validate_model_version(version: str) -> bool:
    """Validate if the requested model version exists."""
    # This would integrate with model registry
    valid_versions = ["latest", "v1.0", "v1.1", "production"]
    return version in valid_versions


def extract_image_metadata(image: Image.Image) -> Dict[str, Any]:
    """Extract comprehensive metadata from image."""
    metadata = {
        "format": image.format,
        "mode": image.mode,
        "size": image.size,
        "has_transparency": image.mode in ("RGBA", "LA") or "transparency" in image.info,
    }
    
    # Extract EXIF data if available
    if hasattr(image, '_getexif') and image._getexif():
        exif_data = image._getexif()
        if exif_data:
            metadata["exif"] = {
                k: v for k, v in exif_data.items() 
                if isinstance(v, (str, int, float))
            }
    
    return metadata


async def sanitize_filename(filename: str) -> str:
    """Sanitize uploaded filename for security."""
    if not filename:
        return "unknown_file"
    
    # Remove path components
    filename = filename.split('/')[-1].split('\\')[-1]
    
    # Remove dangerous characters
    dangerous_chars = '<>:"/\\|?*'
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        name = name[:250 - len(ext)]
        filename = f"{name}.{ext}" if ext else name
    
    return filename


class ImageProcessor:
    """Advanced image processing utilities for medical images."""
    
    @staticmethod
    def apply_window_level(image: np.ndarray, window: float, level: float) -> np.ndarray:
        """Apply window/level adjustment for medical images."""
        min_val = level - window / 2
        max_val = level + window / 2
        
        windowed = np.clip(image, min_val, max_val)
        windowed = (windowed - min_val) / (max_val - min_val)
        
        return (windowed * 255).astype(np.uint8)
    
    @staticmethod
    def enhance_contrast(image: np.ndarray, alpha: float = 1.5, beta: int = 10) -> np.ndarray:
        """Enhance image contrast for better visibility."""
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return enhanced
    
    @staticmethod
    def reduce_noise(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply noise reduction to medical images."""
        denoised = cv2.bilateralFilter(image, kernel_size, 80, 80)
        return denoised
    
    @staticmethod
    async def batch_preprocess(image_list: list, target_size: Tuple[int, int] = (150, 150)) -> np.ndarray:
        """Preprocess multiple images efficiently."""
        processed_images = []
        
        for image_data in image_list:
            processed = await preprocess_image(image_data, target_size)
            processed_images.append(processed[0])  # Remove batch dimension
        
        return np.array(processed_images)


def validate_medical_image_quality(image: np.ndarray) -> Dict[str, Any]:
    """Validate medical image quality for diagnostic use."""
    quality_metrics = {}
    
    # Convert to grayscale for analysis
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Calculate image sharpness using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    quality_metrics['sharpness'] = float(laplacian_var)
    quality_metrics['is_sharp'] = laplacian_var > 100  # Threshold for medical images
    
    # Calculate contrast
    contrast = gray.std()
    quality_metrics['contrast'] = float(contrast)
    quality_metrics['has_good_contrast'] = contrast > 30
    
    # Calculate brightness
    brightness = gray.mean()
    quality_metrics['brightness'] = float(brightness)
    quality_metrics['brightness_adequate'] = 50 <= brightness <= 200
    
    # Overall quality assessment
    quality_metrics['quality_score'] = (
        quality_metrics['is_sharp'] * 0.4 +
        quality_metrics['has_good_contrast'] * 0.3 +
        quality_metrics['brightness_adequate'] * 0.3
    )
    
    quality_metrics['diagnostic_quality'] = quality_metrics['quality_score'] > 0.7
    
    return quality_metrics