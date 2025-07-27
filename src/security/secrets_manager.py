"""
Secrets management for secure configuration handling.
"""

import os
import base64
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import json

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

logger = logging.getLogger(__name__)


class SecretsManager:
    """Secure secrets management for the application."""
    
    def __init__(self, encryption_key: Optional[str] = None, secrets_file: Optional[str] = None):
        """Initialize secrets manager."""
        self.encryption_key = encryption_key or os.getenv('ENCRYPTION_KEY')
        self.secrets_file = secrets_file or os.getenv('SECRETS_FILE', '.secrets.enc')
        
        if CRYPTO_AVAILABLE and self.encryption_key:
            self._fernet = self._create_fernet_instance()
        else:
            self._fernet = None
            if not CRYPTO_AVAILABLE:
                logger.warning("Cryptography library not available. Secrets will not be encrypted.")
    
    def _create_fernet_instance(self) -> Optional[Fernet]:
        """Create Fernet encryption instance."""
        try:
            # Derive key from password
            password = self.encryption_key.encode()
            salt = b'chest_xray_salt'  # In production, use a random salt per secret
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            return Fernet(key)
        except Exception as e:
            logger.error(f"Failed to create encryption instance: {e}")
            return None
    
    def encrypt_secret(self, secret: str) -> str:
        """Encrypt a secret value."""
        if not self._fernet:
            logger.warning("Encryption not available, returning plain text")
            return secret
        
        try:
            encrypted = self._fernet.encrypt(secret.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Failed to encrypt secret: {e}")
            return secret
    
    def decrypt_secret(self, encrypted_secret: str) -> str:
        """Decrypt a secret value."""
        if not self._fernet:
            logger.warning("Encryption not available, returning as-is")
            return encrypted_secret
        
        try:
            decoded = base64.urlsafe_b64decode(encrypted_secret.encode())
            decrypted = self._fernet.decrypt(decoded)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt secret: {e}")
            return encrypted_secret
    
    def store_secret(self, key: str, value: str, encrypt: bool = True) -> bool:
        """Store a secret value."""
        try:
            # Load existing secrets
            secrets = self.load_secrets()
            
            # Encrypt if requested and available
            if encrypt and self._fernet:
                value = self.encrypt_secret(value)
                key = f"enc_{key}"
            
            secrets[key] = value
            
            # Save back to file
            return self.save_secrets(secrets)
            
        except Exception as e:
            logger.error(f"Failed to store secret {key}: {e}")
            return False
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve a secret value."""
        try:
            # Try environment variable first
            env_value = os.getenv(key.upper())
            if env_value:
                return env_value
            
            # Load from secrets file
            secrets = self.load_secrets()
            
            # Check for encrypted version
            enc_key = f"enc_{key}"
            if enc_key in secrets:
                return self.decrypt_secret(secrets[enc_key])
            
            # Check for plain version
            if key in secrets:
                return secrets[key]
            
            return default
            
        except Exception as e:
            logger.error(f"Failed to get secret {key}: {e}")
            return default
    
    def load_secrets(self) -> Dict[str, str]:
        """Load secrets from file."""
        secrets_path = Path(self.secrets_file)
        
        if not secrets_path.exists():
            return {}
        
        try:
            with open(secrets_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load secrets from {secrets_path}: {e}")
            return {}
    
    def save_secrets(self, secrets: Dict[str, str]) -> bool:
        """Save secrets to file."""
        try:
            secrets_path = Path(self.secrets_file)
            secrets_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(secrets_path, 'w') as f:
                json.dump(secrets, f, indent=2)
            
            # Set restrictive permissions
            os.chmod(secrets_path, 0o600)
            return True
            
        except Exception as e:
            logger.error(f"Failed to save secrets: {e}")
            return False
    
    def delete_secret(self, key: str) -> bool:
        """Delete a secret."""
        try:
            secrets = self.load_secrets()
            
            # Remove both encrypted and plain versions
            removed = False
            if key in secrets:
                del secrets[key]
                removed = True
            
            enc_key = f"enc_{key}"
            if enc_key in secrets:
                del secrets[enc_key]
                removed = True
            
            if removed:
                return self.save_secrets(secrets)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete secret {key}: {e}")
            return False
    
    def list_secrets(self, include_values: bool = False) -> Dict[str, Union[str, bool]]:
        """List all stored secrets."""
        try:
            secrets = self.load_secrets()
            
            if include_values:
                # Decrypt encrypted secrets for display
                result = {}
                for key, value in secrets.items():
                    if key.startswith('enc_'):
                        original_key = key[4:]  # Remove 'enc_' prefix
                        result[original_key] = self.decrypt_secret(value)
                    elif not any(k.startswith(f'enc_{key}') for k in secrets.keys()):
                        result[key] = value
                return result
            else:
                # Just return keys
                result = {}
                for key in secrets.keys():
                    if key.startswith('enc_'):
                        result[key[4:]] = True  # Encrypted
                    elif not any(k.startswith(f'enc_{key}') for k in secrets.keys()):
                        result[key] = False  # Plain text
                return result
                
        except Exception as e:
            logger.error(f"Failed to list secrets: {e}")
            return {}
    
    def rotate_encryption_key(self, new_key: str) -> bool:
        """Rotate the encryption key for all secrets."""
        if not CRYPTO_AVAILABLE:
            logger.error("Cannot rotate encryption key: cryptography library not available")
            return False
        
        try:
            # Load current secrets
            secrets = self.load_secrets()
            
            # Decrypt all encrypted secrets with old key
            decrypted_secrets = {}
            for key, value in secrets.items():
                if key.startswith('enc_'):
                    original_key = key[4:]
                    decrypted_value = self.decrypt_secret(value)
                    decrypted_secrets[original_key] = decrypted_value
                else:
                    decrypted_secrets[key] = value
            
            # Create new encryption instance
            old_key = self.encryption_key
            self.encryption_key = new_key
            self._fernet = self._create_fernet_instance()
            
            if not self._fernet:
                # Restore old key
                self.encryption_key = old_key
                self._fernet = self._create_fernet_instance()
                return False
            
            # Re-encrypt all secrets with new key
            new_secrets = {}
            for key, value in decrypted_secrets.items():
                if key in [k[4:] for k in secrets.keys() if k.startswith('enc_')]:
                    # Was encrypted, re-encrypt
                    encrypted_value = self.encrypt_secret(value)
                    new_secrets[f"enc_{key}"] = encrypted_value
                else:
                    # Was plain text, keep as is
                    new_secrets[key] = value
            
            # Save with new encryption
            return self.save_secrets(new_secrets)
            
        except Exception as e:
            logger.error(f"Failed to rotate encryption key: {e}")
            return False


class EnvironmentSecretsManager:
    """Manage secrets through environment variables with validation."""
    
    def __init__(self):
        self.required_secrets = {
            'DATABASE_URL': 'Database connection string',
            'SECRET_KEY': 'Application secret key',
            'JWT_SECRET': 'JWT signing secret',
            'ENCRYPTION_KEY': 'Data encryption key'
        }
        
        self.optional_secrets = {
            'REDIS_URL': 'Redis connection string',
            'SMTP_PASSWORD': 'Email SMTP password',
            'API_KEYS': 'External API keys',
            'OAUTH_SECRETS': 'OAuth client secrets'
        }
    
    def validate_secrets(self) -> Dict[str, Any]:
        """Validate that required secrets are present and properly formatted."""
        validation_result = {
            'valid': True,
            'missing_required': [],
            'missing_optional': [],
            'invalid_format': [],
            'warnings': []
        }
        
        # Check required secrets
        for secret_name, description in self.required_secrets.items():
            value = os.getenv(secret_name)
            if not value:
                validation_result['missing_required'].append({
                    'name': secret_name,
                    'description': description
                })
                validation_result['valid'] = False
            else:
                # Validate format
                format_issues = self._validate_secret_format(secret_name, value)
                if format_issues:
                    validation_result['invalid_format'].extend(format_issues)
                    validation_result['valid'] = False
        
        # Check optional secrets
        for secret_name, description in self.optional_secrets.items():
            value = os.getenv(secret_name)
            if not value:
                validation_result['missing_optional'].append({
                    'name': secret_name,
                    'description': description
                })
        
        # General security warnings
        self._add_security_warnings(validation_result)
        
        return validation_result
    
    def _validate_secret_format(self, name: str, value: str) -> List[Dict[str, str]]:
        """Validate secret format based on its type."""
        issues = []
        
        if name == 'DATABASE_URL':
            if not value.startswith(('postgresql://', 'mysql://', 'sqlite:///')):
                issues.append({
                    'name': name,
                    'issue': 'Invalid database URL format'
                })
        
        elif name == 'SECRET_KEY':
            if len(value) < 32:
                issues.append({
                    'name': name,
                    'issue': 'Secret key should be at least 32 characters'
                })
        
        elif name == 'JWT_SECRET':
            if len(value) < 32:
                issues.append({
                    'name': name,
                    'issue': 'JWT secret should be at least 32 characters'
                })
        
        elif name == 'ENCRYPTION_KEY':
            if len(value) < 16:
                issues.append({
                    'name': name,
                    'issue': 'Encryption key should be at least 16 characters'
                })
        
        return issues
    
    def _add_security_warnings(self, validation_result: Dict[str, Any]):
        """Add general security warnings."""
        # Check if running in development with weak secrets
        if os.getenv('ENVIRONMENT') in ['development', 'dev']:
            validation_result['warnings'].append(
                "Running in development mode - ensure production secrets are properly secured"
            )
        
        # Check for default/example values
        secret_key = os.getenv('SECRET_KEY', '')
        if secret_key in ['your-secret-key-here', 'changeme', 'secret']:
            validation_result['warnings'].append(
                "SECRET_KEY appears to be a default value - change in production"
            )
    
    def generate_secret_key(self, length: int = 32) -> str:
        """Generate a cryptographically secure secret key."""
        import secrets
        import string
        
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def mask_secret(self, secret: str, visible_chars: int = 4) -> str:
        """Mask a secret for display purposes."""
        if len(secret) <= visible_chars:
            return '*' * len(secret)
        
        return secret[:visible_chars] + '*' * (len(secret) - visible_chars)


# Global instances
secrets_manager = SecretsManager()
env_secrets_manager = EnvironmentSecretsManager()


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Convenience function to get a secret."""
    return secrets_manager.get_secret(key, default)


def validate_environment_secrets() -> Dict[str, Any]:
    """Convenience function to validate environment secrets."""
    return env_secrets_manager.validate_secrets()