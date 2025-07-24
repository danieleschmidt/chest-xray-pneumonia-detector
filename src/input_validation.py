"""
Input validation utilities for CLI tools and file path handling.

This module provides secure input validation functions to prevent
path traversal attacks, injection vulnerabilities, and other security issues.
"""

import os
import pathlib
from typing import Optional, List


class ValidationError(Exception):
    """Custom exception for input validation failures."""
    pass


def validate_file_path(
    file_path: str,
    must_exist: bool = False,
    allowed_extensions: Optional[List[str]] = None,
    max_path_length: int = 4096
) -> str:
    """
    Validate and sanitize file path input.
    
    Parameters
    ----------
    file_path : str
        The file path to validate
    must_exist : bool, default=False
        Whether the file must already exist
    allowed_extensions : List[str], optional
        List of allowed file extensions (e.g., ['.keras', '.h5'])
    max_path_length : int, default=4096
        Maximum allowed path length
        
    Returns
    -------
    str
        The normalized, validated file path
        
    Raises
    ------
    ValidationError
        If the path fails validation
        
    Examples
    --------
    >>> validate_file_path("models/my_model.keras", allowed_extensions=['.keras'])
    'models/my_model.keras'
    """
    if not file_path or not file_path.strip():
        raise ValidationError("File path cannot be empty")
    
    file_path = file_path.strip()
    
    # Check path length
    if len(file_path) > max_path_length:
        raise ValidationError(f"Path too long: {len(file_path)} > {max_path_length}")
    
    # Prevent null bytes
    if '\x00' in file_path:
        raise ValidationError("Path contains null bytes")
    
    # Normalize path to prevent traversal attacks
    try:
        normalized_path = os.path.normpath(file_path)
        resolved_path = str(pathlib.Path(normalized_path).resolve())
    except (ValueError, OSError) as e:
        raise ValidationError(f"Invalid path format: {e}")
    
    # Check for path traversal attempts
    if '..' in normalized_path.split(os.sep):
        raise ValidationError("Path traversal detected")
    
    # Validate file extension if specified
    if allowed_extensions:
        file_ext = pathlib.Path(file_path).suffix.lower()
        if file_ext not in [ext.lower() for ext in allowed_extensions]:
            raise ValidationError(f"Invalid file extension '{file_ext}'. Allowed: {allowed_extensions}")
    
    # Check if file exists if required
    if must_exist and not os.path.exists(normalized_path):
        raise ValidationError(f"File does not exist: {normalized_path}")
    
    return normalized_path


def validate_directory_path(
    dir_path: str,
    must_exist: bool = False,
    create_if_missing: bool = False,
    max_path_length: int = 4096
) -> str:
    """
    Validate and sanitize directory path input.
    
    Parameters
    ---------- 
    dir_path : str
        The directory path to validate
    must_exist : bool, default=False
        Whether the directory must already exist
    create_if_missing : bool, default=False
        Whether to create the directory if it doesn't exist
    max_path_length : int, default=4096
        Maximum allowed path length
        
    Returns
    -------
    str
        The normalized, validated directory path
        
    Raises
    ------
    ValidationError
        If the path fails validation
    """
    if not dir_path or not dir_path.strip():
        raise ValidationError("Directory path cannot be empty")
    
    dir_path = dir_path.strip()
    
    # Check path length
    if len(dir_path) > max_path_length:
        raise ValidationError(f"Path too long: {len(dir_path)} > {max_path_length}")
    
    # Prevent null bytes
    if '\x00' in dir_path:
        raise ValidationError("Path contains null bytes")
    
    # Normalize path
    try:
        normalized_path = os.path.normpath(dir_path)
    except (ValueError, OSError) as e:
        raise ValidationError(f"Invalid path format: {e}")
    
    # Check for path traversal attempts
    if '..' in normalized_path.split(os.sep):
        raise ValidationError("Path traversal detected")
    
    # Handle directory creation/existence
    if create_if_missing and not os.path.exists(normalized_path):
        try:
            os.makedirs(normalized_path, exist_ok=True)
        except OSError as e:
            raise ValidationError(f"Cannot create directory: {e}")
    elif must_exist and not os.path.isdir(normalized_path):
        raise ValidationError(f"Directory does not exist: {normalized_path}")
    
    return normalized_path


def validate_model_path(model_path: str, must_exist: bool = True) -> str:
    """
    Validate model file path with ML-specific extensions.
    
    Parameters
    ----------
    model_path : str
        Path to model file
    must_exist : bool, default=True
        Whether the model file must exist
        
    Returns
    -------
    str
        Validated model path
    """
    allowed_extensions = ['.keras', '.h5', '.pb', '.pkl', '.joblib']
    return validate_file_path(
        model_path,
        must_exist=must_exist,
        allowed_extensions=allowed_extensions
    )


def validate_image_path(image_path: str, must_exist: bool = True) -> str:
    """
    Validate image file path with image-specific extensions.
    
    Parameters
    ----------
    image_path : str
        Path to image file
    must_exist : bool, default=True
        Whether the image file must exist
        
    Returns
    -------
    str
        Validated image path
    """
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    return validate_file_path(
        image_path,
        must_exist=must_exist,
        allowed_extensions=allowed_extensions
    )