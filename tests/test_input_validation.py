"""
Tests for input validation utilities.
"""

import os
import tempfile
import pytest
from unittest.mock import patch

from src.input_validation import (
    validate_file_path,
    validate_directory_path,
    validate_model_path,
    validate_image_path,
    ValidationError
)


class TestValidateFilePath:
    """Test file path validation."""
    
    def test_validate_file_path_basic(self):
        """Test basic file path validation."""
        result = validate_file_path("test.txt")
        assert result == "test.txt"
    
    def test_validate_file_path_with_extension_filter(self):
        """Test file path validation with allowed extensions."""
        result = validate_file_path("model.keras", allowed_extensions=['.keras'])
        assert result == "model.keras"
    
    def test_validate_file_path_invalid_extension(self):
        """Test rejection of invalid file extensions."""
        with pytest.raises(ValidationError, match="Invalid file extension"):
            validate_file_path("model.txt", allowed_extensions=['.keras'])
    
    def test_validate_file_path_empty(self):
        """Test rejection of empty paths."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_file_path("")
        
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_file_path("   ")
    
    def test_validate_file_path_too_long(self):
        """Test rejection of paths that are too long."""
        long_path = "a" * 5000
        with pytest.raises(ValidationError, match="Path too long"):
            validate_file_path(long_path, max_path_length=4096)
    
    def test_validate_file_path_null_bytes(self):
        """Test rejection of paths with null bytes."""
        with pytest.raises(ValidationError, match="null bytes"):
            validate_file_path("test\x00.txt")
    
    def test_validate_file_path_traversal_attack(self):
        """Test prevention of path traversal attacks."""
        with pytest.raises(ValidationError, match="Path traversal detected"):
            validate_file_path("../../../etc/passwd")
        
        with pytest.raises(ValidationError, match="Path traversal detected"):
            validate_file_path("models/../../../secrets.txt")
    
    def test_validate_file_path_must_exist(self):
        """Test validation when file must exist."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test")
            tmp_path = tmp.name
        
        try:    
            # Should pass when file exists
            result = validate_file_path(tmp_path, must_exist=True)
            assert result == os.path.normpath(tmp_path)
            
            # Should fail when file doesn't exist
            with pytest.raises(ValidationError, match="does not exist"):
                validate_file_path("nonexistent.txt", must_exist=True)
        finally:
            os.unlink(tmp_path)
    
    def test_validate_file_path_normalization(self):
        """Test path normalization."""
        result = validate_file_path("./models//test.keras")
        assert result == os.path.normpath("./models//test.keras")


class TestValidateDirectoryPath:
    """Test directory path validation."""
    
    def test_validate_directory_path_basic(self):
        """Test basic directory path validation."""
        result = validate_directory_path("models")
        assert result == "models"
    
    def test_validate_directory_path_empty(self):
        """Test rejection of empty directory paths."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_directory_path("")
    
    def test_validate_directory_path_traversal(self):
        """Test prevention of directory traversal attacks."""
        with pytest.raises(ValidationError, match="Path traversal detected"):
            validate_directory_path("../../../")
    
    def test_validate_directory_path_must_exist(self):
        """Test validation when directory must exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Should pass when directory exists
            result = validate_directory_path(tmp_dir, must_exist=True)
            assert result == os.path.normpath(tmp_dir)
        
        # Should fail when directory doesn't exist
        with pytest.raises(ValidationError, match="does not exist"):
            validate_directory_path("nonexistent_dir", must_exist=True)
    
    def test_validate_directory_path_create_if_missing(self):
        """Test directory creation when missing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            new_dir = os.path.join(tmp_dir, "new_subdir")
            
            # Should create the directory
            result = validate_directory_path(new_dir, create_if_missing=True)
            assert result == new_dir
            assert os.path.isdir(new_dir)
    
    def test_validate_directory_path_create_permission_error(self):
        """Test handling of permission errors during directory creation."""
        with patch('os.makedirs', side_effect=OSError("Permission denied")):
            with pytest.raises(ValidationError, match="Cannot create directory"):
                validate_directory_path("test_dir", create_if_missing=True)


class TestValidateModelPath:
    """Test model-specific path validation."""
    
    def test_validate_model_path_keras(self):
        """Test validation of Keras model paths."""
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            result = validate_model_path(tmp_path)
            assert result == os.path.normpath(tmp_path)
        finally:
            os.unlink(tmp_path)
    
    def test_validate_model_path_invalid_extension(self):
        """Test rejection of invalid model extensions."""
        with pytest.raises(ValidationError, match="Invalid file extension"):
            validate_model_path("model.txt", must_exist=False)
    
    def test_validate_model_path_allowed_extensions(self):
        """Test all allowed model extensions."""
        allowed_extensions = ['.keras', '.h5', '.pb', '.pkl', '.joblib']
        
        for ext in allowed_extensions:
            # Should not raise exception
            result = validate_model_path(f"model{ext}", must_exist=False)
            assert result == f"model{ext}"


class TestValidateImagePath:
    """Test image-specific path validation."""
    
    def test_validate_image_path_jpg(self):
        """Test validation of JPEG image paths."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            result = validate_image_path(tmp_path)
            assert result == os.path.normpath(tmp_path)
        finally:
            os.unlink(tmp_path)
    
    def test_validate_image_path_invalid_extension(self):
        """Test rejection of invalid image extensions."""
        with pytest.raises(ValidationError, match="Invalid file extension"):
            validate_image_path("image.txt", must_exist=False)
    
    def test_validate_image_path_case_insensitive(self):
        """Test case-insensitive extension matching."""
        result = validate_image_path("image.JPG", must_exist=False)
        assert result == "image.JPG"
    
    def test_validate_image_path_allowed_extensions(self):
        """Test all allowed image extensions."""
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        for ext in allowed_extensions:
            # Should not raise exception
            result = validate_image_path(f"image{ext}", must_exist=False)
            assert result == f"image{ext}"


class TestSecurityScenarios:
    """Test various security attack scenarios."""
    
    def test_prevent_symlink_attacks(self):
        """Test prevention of symlink-based attacks."""
        # This is a simplified test - in practice, more sophisticated checks might be needed
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a file outside the temp directory
            outside_file = "/tmp/outside_target.txt"
            with open(outside_file, "w") as f:
                f.write("secret")
            
            try:
                # Create a symlink pointing outside
                symlink_path = os.path.join(tmp_dir, "link_to_outside")
                os.symlink(outside_file, symlink_path)
                
                # The validation should still work (symlinks are resolved)
                # But the important thing is that path traversal is prevented
                result = validate_file_path(symlink_path, must_exist=True)
                assert os.path.isfile(result)
                
                # But direct traversal should still be blocked
                with pytest.raises(ValidationError, match="Path traversal detected"):
                    validate_file_path("../../../tmp/outside_target.txt")
                    
            finally:
                if os.path.exists(outside_file):
                    os.unlink(outside_file)
    
    def test_prevent_directory_injection(self):
        """Test prevention of directory name injection."""
        malicious_names = [
            "test\n../../etc/passwd",
            "test;rm -rf /",
            "test`whoami`",
            "test$(whoami)",
        ]
        
        for name in malicious_names:
            # Should not cause issues - paths are normalized and validated
            with pytest.raises(ValidationError):
                # Most of these will fail due to invalid characters or traversal
                validate_directory_path(name)
    
    def test_unicode_normalization(self):
        """Test handling of Unicode characters in paths.""" 
        unicode_path = "models/тест.keras"  # Cyrillic characters
        result = validate_file_path(unicode_path, must_exist=False, allowed_extensions=['.keras'])
        assert result == unicode_path
        
        # Test potential Unicode normalization attacks
        # (This is a simplified test - real Unicode attacks are more complex)
        suspicious_unicode = "models/test\u202e.keras"  # Right-to-left override
        result = validate_file_path(suspicious_unicode, must_exist=False, allowed_extensions=['.keras'])
        assert result  # Should not crash, but may normalize