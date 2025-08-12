"""
Advanced input sanitization for medical data processing.
Enhances security with comprehensive validation and sanitization.
"""

import re
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import hashlib
import mimetypes
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedInputSanitizer:
    """Advanced input sanitization with medical data focus."""
    
    # Allowed file extensions for medical images
    ALLOWED_IMAGE_EXTENSIONS = {
        '.jpg', '.jpeg', '.png', '.dicom', '.dcm', '.tiff', '.tif'
    }
    
    # Maximum file sizes (in bytes)
    MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_BATCH_SIZE = 1000
    
    # Regex patterns for validation
    PATIENT_ID_PATTERN = re.compile(r'^[A-Za-z0-9_-]{1,50}$')
    FILENAME_PATTERN = re.compile(r'^[A-Za-z0-9._-]+$')
    
    def __init__(self):
        self.blocked_hashes = set()
        self.suspicious_patterns = [
            b'<script', b'javascript:', b'eval(', b'exec(',
            b'import os', b'subprocess', b'__import__'
        ]
    
    def sanitize_file_path(self, file_path: str) -> str:
        """Sanitize and validate file paths."""
        try:
            # Convert to Path object and resolve
            path = Path(file_path).resolve()
            
            # Check for path traversal
            if '..' in str(path) or str(path).startswith('/'):
                raise ValueError("Invalid path: potential path traversal")
            
            # Validate filename
            if not self.FILENAME_PATTERN.match(path.name):
                raise ValueError(f"Invalid filename: {path.name}")
            
            # Check extension
            if path.suffix.lower() not in self.ALLOWED_IMAGE_EXTENSIONS:
                raise ValueError(f"Unsupported file extension: {path.suffix}")
            
            return str(path)
            
        except Exception as e:
            logger.error(f"Path sanitization failed: {e}")
            raise ValueError(f"Invalid file path: {file_path}")
    
    def validate_image_file(self, file_path: str) -> Dict[str, Any]:
        """Comprehensive image file validation."""
        try:
            path = Path(file_path)
            
            # Check file exists and size
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_size = path.stat().st_size
            if file_size > self.MAX_IMAGE_SIZE:
                raise ValueError(f"File too large: {file_size} bytes")
            
            # Validate MIME type
            mime_type, _ = mimetypes.guess_type(str(path))
            if not mime_type or not mime_type.startswith('image/'):
                raise ValueError(f"Invalid MIME type: {mime_type}")
            
            # Calculate file hash for duplicate detection
            file_hash = self._calculate_file_hash(path)
            if file_hash in self.blocked_hashes:
                raise ValueError("File blocked: known malicious content")
            
            # Validate image with PIL
            try:
                with Image.open(path) as img:
                    img.verify()  # Verify image integrity
                    
                # Re-open for metadata (verify() closes the image)
                with Image.open(path) as img:
                    width, height = img.size
                    mode = img.mode
                    format_name = img.format
                    
                    # Basic sanity checks
                    if width < 50 or height < 50:
                        logger.warning(f"Unusually small image: {width}x{height}")
                    
                    if width > 10000 or height > 10000:
                        raise ValueError(f"Image too large: {width}x{height}")
                    
                    return {
                        'path': str(path),
                        'size': file_size,
                        'hash': file_hash,
                        'mime_type': mime_type,
                        'dimensions': (width, height),
                        'mode': mode,
                        'format': format_name,
                        'validated': True
                    }
                    
            except Exception as img_error:
                raise ValueError(f"Invalid image file: {img_error}")
                
        except Exception as e:
            logger.error(f"Image validation failed for {file_path}: {e}")
            raise
    
    def sanitize_training_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize and validate training parameters."""
        sanitized = {}
        
        # Validate numeric parameters
        numeric_params = {
            'batch_size': (1, self.MAX_BATCH_SIZE),
            'epochs': (1, 1000),
            'learning_rate': (1e-6, 1.0),
            'dropout_rate': (0.0, 0.9),
            'rotation_range': (0, 180),
            'zoom_range': (0.0, 2.0)
        }
        
        for param, (min_val, max_val) in numeric_params.items():
            if param in params:
                value = float(params[param])
                if not (min_val <= value <= max_val):
                    raise ValueError(f"{param} must be between {min_val} and {max_val}")
                sanitized[param] = value
        
        # Validate string parameters
        if 'model_name' in params:
            model_name = str(params['model_name'])
            allowed_models = {
                'VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152',
                'InceptionV3', 'Xception', 'MobileNet', 'MobileNetV2',
                'DenseNet121', 'DenseNet169', 'DenseNet201'
            }
            if model_name not in allowed_models:
                raise ValueError(f"Unsupported model: {model_name}")
            sanitized['model_name'] = model_name
        
        # Validate image size
        if 'img_size' in params:
            img_size = params['img_size']
            if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
                width, height = map(int, img_size)
                if not (32 <= width <= 2048 and 32 <= height <= 2048):
                    raise ValueError("Image dimensions must be between 32 and 2048")
                sanitized['img_size'] = (width, height)
            else:
                raise ValueError("img_size must be a tuple of (width, height)")
        
        return sanitized
    
    def validate_patient_data(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize patient data for HIPAA compliance."""
        sanitized = {}
        
        # Validate patient ID
        if 'patient_id' in patient_data:
            patient_id = str(patient_data['patient_id'])
            if not self.PATIENT_ID_PATTERN.match(patient_id):
                raise ValueError("Invalid patient ID format")
            
            # Hash patient ID for anonymization
            sanitized['patient_id_hash'] = hashlib.sha256(
                patient_id.encode('utf-8')
            ).hexdigest()[:16]
        
        # Validate age (if provided)
        if 'age' in patient_data:
            age = int(patient_data['age'])
            if not (0 <= age <= 150):
                raise ValueError("Invalid age range")
            sanitized['age'] = age
        
        # Validate sex/gender
        if 'sex' in patient_data:
            sex = str(patient_data['sex']).upper()
            if sex not in {'M', 'F', 'O', 'U'}:  # Male, Female, Other, Unknown
                raise ValueError("Invalid sex value")
            sanitized['sex'] = sex
        
        return sanitized
    
    def scan_for_malicious_content(self, file_path: str) -> bool:
        """Scan file for potentially malicious content."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read(1024 * 1024)  # Read first 1MB
                
                for pattern in self.suspicious_patterns:
                    if pattern in content:
                        logger.warning(f"Suspicious pattern found in {file_path}: {pattern}")
                        return True
                        
            return False
            
        except Exception as e:
            logger.error(f"Malicious content scan failed: {e}")
            return True  # Err on the side of caution
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def add_blocked_hash(self, file_hash: str) -> None:
        """Add a file hash to the blocked list."""
        self.blocked_hashes.add(file_hash)
        logger.info(f"Added hash to blocked list: {file_hash[:16]}...")
    
    def validate_batch_operation(self, file_list: List[str]) -> List[Dict[str, Any]]:
        """Validate a batch of files for processing."""
        if len(file_list) > self.MAX_BATCH_SIZE:
            raise ValueError(f"Batch size {len(file_list)} exceeds maximum {self.MAX_BATCH_SIZE}")
        
        validated_files = []
        
        for file_path in file_list:
            try:
                # Sanitize path
                clean_path = self.sanitize_file_path(file_path)
                
                # Validate image
                validation_result = self.validate_image_file(clean_path)
                
                # Scan for malicious content
                if self.scan_for_malicious_content(clean_path):
                    logger.warning(f"Skipping potentially malicious file: {clean_path}")
                    continue
                
                validated_files.append(validation_result)
                
            except Exception as e:
                logger.error(f"Validation failed for {file_path}: {e}")
                continue
        
        logger.info(f"Validated {len(validated_files)}/{len(file_list)} files")
        return validated_files


def create_sanitizer() -> AdvancedInputSanitizer:
    """Factory function to create a configured sanitizer."""
    return AdvancedInputSanitizer()


if __name__ == '__main__':
    # Example usage
    sanitizer = create_sanitizer()
    
    # Test parameter sanitization
    test_params = {
        'batch_size': 32,
        'epochs': 10,
        'learning_rate': 0.001,
        'model_name': 'ResNet50',
        'img_size': (224, 224)
    }
    
    try:
        sanitized = sanitizer.sanitize_training_parameters(test_params)
        print(f"Sanitized parameters: {sanitized}")
    except ValueError as e:
        print(f"Validation error: {e}")
