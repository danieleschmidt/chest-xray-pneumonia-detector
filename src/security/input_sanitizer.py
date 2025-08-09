"""Input sanitization and validation for security."""

import re
import base64
from typing import Optional, Dict, Any
from PIL import Image
import io
import numpy as np


class InputSanitizer:
    """Sanitizes and validates inputs for security and safety."""
    
    # Maximum image size (10MB)
    MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024
    
    # Allowed image formats
    ALLOWED_FORMATS = {'JPEG', 'PNG', 'BMP', 'TIFF'}
    
    # Maximum image dimensions
    MAX_WIDTH = 2048
    MAX_HEIGHT = 2048
    
    @classmethod
    def sanitize_base64_image(cls, image_data: str) -> Optional[np.ndarray]:
        """Sanitize and validate base64 encoded image data.
        
        Args:
            image_data: Base64 encoded image string
            
        Returns:
            Sanitized image as numpy array or None if invalid
            
        Raises:
            ValueError: If image data is invalid or unsafe
        """
        try:
            # Remove data URL prefix if present
            if image_data.startswith('data:image/'):
                image_data = image_data.split(',')[1]
            
            # Decode base64
            try:
                decoded_data = base64.b64decode(image_data)
            except Exception:
                raise ValueError("Invalid base64 encoding")
            
            # Check size limits
            if len(decoded_data) > cls.MAX_IMAGE_SIZE_BYTES:
                raise ValueError(f"Image too large: {len(decoded_data)} bytes")
            
            # Validate image format and content
            try:
                image = Image.open(io.BytesIO(decoded_data))
                
                # Check format
                if image.format not in cls.ALLOWED_FORMATS:
                    raise ValueError(f"Unsupported format: {image.format}")
                
                # Check dimensions
                if image.width > cls.MAX_WIDTH or image.height > cls.MAX_HEIGHT:
                    raise ValueError(f"Image too large: {image.width}x{image.height}")
                
                # Convert to RGB if needed and resize for processing
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Resize to standard input size
                image = image.resize((150, 150))
                
                # Convert to numpy array
                image_array = np.array(image) / 255.0
                
                return image_array
                
            except Exception as e:
                raise ValueError(f"Invalid image data: {str(e)}")
                
        except Exception as e:
            raise ValueError(f"Image sanitization failed: {str(e)}")
    
    @classmethod
    def sanitize_text_input(cls, text: str, max_length: int = 1000) -> str:
        """Sanitize text input by removing harmful characters.
        
        Args:
            text: Input text to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized text string
            
        Raises:
            ValueError: If text is too long or contains harmful content
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        if len(text) > max_length:
            raise ValueError(f"Text too long: {len(text)} > {max_length}")
        
        # Remove potentially harmful characters
        # Keep alphanumeric, spaces, and basic punctuation
        sanitized = re.sub(r'[^\w\s\-\.\,\!\?\:\;]', '', text)
        
        # Remove excessive whitespace
        sanitized = ' '.join(sanitized.split())
        
        return sanitized
    
    @classmethod
    def validate_model_path(cls, path: str) -> bool:
        """Validate model file path for security.
        
        Args:
            path: File path to validate
            
        Returns:
            True if path is safe, False otherwise
        """
        if not isinstance(path, str):
            return False
        
        # Check for path traversal attempts
        if '..' in path or path.startswith('/'):
            return False
        
        # Must be in saved_models directory with .keras extension
        if not (path.startswith('saved_models/') and path.endswith('.keras')):
            return False
        
        # Additional validation for allowed characters
        if not re.match(r'^saved_models/[a-zA-Z0-9_\-\.]+\.keras$', path):
            return False
        
        return True
    
    @classmethod
    def sanitize_api_parameters(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize API parameters for security.
        
        Args:
            params: Dictionary of API parameters
            
        Returns:
            Sanitized parameters dictionary
        """
        sanitized = {}
        
        for key, value in params.items():
            # Sanitize key
            clean_key = cls.sanitize_text_input(key, max_length=100)
            
            # Sanitize value based on type
            if isinstance(value, str):
                clean_value = cls.sanitize_text_input(value, max_length=10000)
            elif isinstance(value, (int, float)):
                # Validate numeric ranges
                if abs(value) > 1e10:  # Prevent extreme values
                    raise ValueError(f"Numeric value out of range: {value}")
                clean_value = value
            elif isinstance(value, bool):
                clean_value = value
            elif isinstance(value, (list, dict)):
                # For complex types, convert to string and sanitize
                clean_value = cls.sanitize_text_input(str(value), max_length=10000)
            else:
                # Convert unknown types to string and sanitize
                clean_value = cls.sanitize_text_input(str(value), max_length=1000)
            
            sanitized[clean_key] = clean_value
        
        return sanitized