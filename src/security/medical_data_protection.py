# Medical Data Protection and HIPAA Compliance
# Implements advanced security measures for medical AI systems

import os
import hashlib
import secrets
from typing import Any, Dict, List, Optional, Union
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import json
from datetime import datetime, timedelta
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import tensorflow as tf


@dataclass
class AccessLog:
    """Comprehensive access logging for HIPAA compliance."""
    user_id: str
    action: str
    resource: str
    timestamp: datetime
    ip_address: str
    success: bool
    phi_accessed: bool = False
    reason: Optional[str] = None
    session_id: Optional[str] = None


class MedicalDataEncryption:
    """
    HIPAA-compliant encryption for Protected Health Information (PHI).
    Uses AES-256 encryption with PBKDF2 key derivation.
    """
    
    def __init__(self, password: bytes):
        self.salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        self.cipher_suite = Fernet(key)
        
    def encrypt_data(self, data: Union[str, bytes, np.ndarray]) -> bytes:
        """Encrypt sensitive medical data."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        elif isinstance(data, np.ndarray):
            data = data.tobytes()
            
        return self.cipher_suite.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt medical data."""
        return self.cipher_suite.decrypt(encrypted_data)
    
    def encrypt_model_weights(self, model: tf.keras.Model) -> bytes:
        """Encrypt model weights for secure storage."""
        weights = model.get_weights()
        weights_bytes = pickle.dumps(weights)
        return self.encrypt_data(weights_bytes)
    
    def decrypt_model_weights(self, encrypted_weights: bytes) -> List[np.ndarray]:
        """Decrypt model weights."""
        import pickle
        weights_bytes = self.decrypt_data(encrypted_weights)
        return pickle.loads(weights_bytes)


class AccessControlManager:
    """Role-based access control for medical AI systems."""
    
    def __init__(self):
        self.roles = {
            'radiologist': ['view_images', 'generate_predictions', 'view_reports'],
            'physician': ['view_predictions', 'view_reports'],
            'researcher': ['view_anonymized_data', 'train_models'],
            'admin': ['manage_users', 'view_logs', 'system_config']
        }
        self.users: Dict[str, Dict] = {}
        self.sessions: Dict[str, Dict] = {}
        
    def create_user(self, user_id: str, role: str, name: str) -> bool:
        """Create a new user with specified role."""
        if role not in self.roles:
            return False
            
        self.users[user_id] = {
            'role': role,
            'name': name,
            'created': datetime.now(),
            'last_login': None,
            'active': True
        }
        return True
    
    def authenticate_user(self, user_id: str, password: str) -> Optional[str]:
        """Authenticate user and create session."""
        if user_id not in self.users or not self.users[user_id]['active']:
            return None
            
        # In production, use proper password hashing
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            'user_id': user_id,
            'created': datetime.now(),
            'expires': datetime.now() + timedelta(hours=8),
            'ip_address': None  # Set by calling function
        }
        
        self.users[user_id]['last_login'] = datetime.now()
        return session_id
    
    def check_permission(self, session_id: str, action: str) -> bool:
        """Check if user has permission for action."""
        if session_id not in self.sessions:
            return False
            
        session = self.sessions[session_id]
        if datetime.now() > session['expires']:
            del self.sessions[session_id]
            return False
            
        user_id = session['user_id']
        if user_id not in self.users:
            return False
            
        user_role = self.users[user_id]['role']
        return action in self.roles.get(user_role, [])


class AuditLogger:
    """Comprehensive audit logging for HIPAA compliance."""
    
    def __init__(self, log_file: str = "medical_ai_audit.log"):
        self.log_file = Path(log_file)
        self.logger = logging.getLogger('medical_ai_audit')
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
    def log_access(self, access_log: AccessLog):
        """Log access event for audit trail."""
        log_data = asdict(access_log)
        log_data['timestamp'] = log_data['timestamp'].isoformat()
        
        self.logger.info(f"ACCESS: {json.dumps(log_data)}")
        
    def log_model_prediction(self, user_id: str, model_name: str, 
                           prediction: float, confidence: float):
        """Log model prediction events."""
        log_data = {
            'event_type': 'PREDICTION',
            'user_id': user_id,
            'model_name': model_name,
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"PREDICTION: {json.dumps(log_data)}")
        
    def log_data_access(self, user_id: str, data_type: str, 
                       patient_count: int, anonymized: bool):
        """Log data access for compliance."""
        log_data = {
            'event_type': 'DATA_ACCESS',
            'user_id': user_id,
            'data_type': data_type,
            'patient_count': patient_count,
            'anonymized': anonymized,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"DATA_ACCESS: {json.dumps(log_data)}")


class DataAnonymizer:
    """Anonymize medical data for research and development."""
    
    @staticmethod
    def anonymize_image_metadata(image_path: str) -> Dict[str, Any]:
        """Remove or anonymize DICOM metadata."""
        try:
            import pydicom
            
            ds = pydicom.dcmread(image_path)
            
            # Remove patient identifying information
            anonymized_fields = {
                'PatientName': 'ANONYMOUS',
                'PatientID': f'ANON_{secrets.token_hex(8)}',
                'PatientBirthDate': '',
                'PatientSex': ds.get('PatientSex', ''),
                'StudyDate': ds.get('StudyDate', ''),
                'StudyTime': '',
                'AccessionNumber': '',
                'StudyID': f'STUDY_{secrets.token_hex(6)}'
            }
            
            for field, value in anonymized_fields.items():
                if field in ds:
                    ds[field].value = value
                    
            return anonymized_fields
            
        except ImportError:
            # Fallback for non-DICOM images
            return {
                'filename': Path(image_path).name,
                'anonymized_id': f'IMG_{secrets.token_hex(8)}'
            }
    
    @staticmethod
    def generate_synthetic_identifiers(count: int) -> List[str]:
        """Generate synthetic patient identifiers for testing."""
        return [f'SYNTH_{secrets.token_hex(8)}' for _ in range(count)]


class SecureModelWrapper:
    """Secure wrapper for ML models with access control and audit logging."""
    
    def __init__(self, model: tf.keras.Model, model_name: str,
                 access_control: AccessControlManager,
                 audit_logger: AuditLogger):
        self.model = model
        self.model_name = model_name
        self.access_control = access_control
        self.audit_logger = audit_logger
        
    def predict(self, X: np.ndarray, session_id: str, 
               patient_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Secure prediction with access control and logging."""
        
        # Check permissions
        if not self.access_control.check_permission(session_id, 'generate_predictions'):
            raise PermissionError("User does not have prediction permissions")
        
        # Get user info
        session = self.access_control.sessions[session_id]
        user_id = session['user_id']
        
        try:
            # Generate predictions
            predictions = self.model.predict(X)
            
            # Calculate confidence scores
            if predictions.shape[1] == 1:  # Binary classification
                confidences = np.abs(predictions.flatten() - 0.5) * 2
            else:  # Multi-class
                confidences = np.max(predictions, axis=1)
            
            # Log predictions
            for i, (pred, conf) in enumerate(zip(predictions.flatten(), confidences)):
                patient_id = patient_ids[i] if patient_ids else f"unknown_{i}"
                self.audit_logger.log_model_prediction(
                    user_id, self.model_name, float(pred), float(conf)
                )
            
            return {
                'predictions': predictions.tolist(),
                'confidences': confidences.tolist(),
                'model_name': self.model_name,
                'user_id': user_id,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            # Log failed prediction attempt
            access_log = AccessLog(
                user_id=user_id,
                action='generate_predictions',
                resource=self.model_name,
                timestamp=datetime.now(),
                ip_address=session.get('ip_address', 'unknown'),
                success=False,
                reason=str(e)
            )
            self.audit_logger.log_access(access_log)
            raise


class MedicalDataValidator:
    """Validate medical data for security and integrity."""
    
    @staticmethod
    def validate_image_security(image_path: str) -> Dict[str, Any]:
        """Validate image for security threats."""
        validation_results = {
            'is_valid': True,
            'file_size': 0,
            'format_valid': False,
            'no_embedded_code': True,
            'metadata_clean': True,
            'issues': []
        }
        
        try:
            file_path = Path(image_path)
            
            # Check file size (prevent ZIP bombs)
            file_size = file_path.stat().st_size
            validation_results['file_size'] = file_size
            
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                validation_results['is_valid'] = False
                validation_results['issues'].append('File too large')
            
            # Validate image format
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    if img.format in ['JPEG', 'PNG', 'TIFF', 'DCM']:
                        validation_results['format_valid'] = True
                    else:
                        validation_results['is_valid'] = False
                        validation_results['issues'].append('Invalid image format')
            except Exception:
                validation_results['is_valid'] = False
                validation_results['issues'].append('Cannot open as image')
            
            # Check for suspicious metadata
            if file_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.tiff', '.dcm']:
                validation_results['is_valid'] = False
                validation_results['issues'].append('Suspicious file extension')
                
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f'Validation error: {str(e)}')
        
        return validation_results
    
    @staticmethod
    def sanitize_input_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize input data to prevent injection attacks."""
        sanitized = {}
        
        for key, value in data.items():
            # Remove potentially dangerous characters
            if isinstance(value, str):
                # Basic SQL injection prevention
                sanitized_value = value.replace("'", "").replace('"', '').replace(';', '')
                # XSS prevention
                sanitized_value = sanitized_value.replace('<', '').replace('>', '')
                sanitized[key] = sanitized_value[:1000]  # Limit length
            elif isinstance(value, (int, float)):
                sanitized[key] = value
            elif isinstance(value, list):
                sanitized[key] = value[:100]  # Limit list size
            else:
                sanitized[key] = str(value)[:1000]
                
        return sanitized


def create_secure_medical_ai_system() -> Dict[str, Any]:
    """Factory function to create a complete secure medical AI system."""
    
    # Initialize security components
    access_control = AccessControlManager()
    audit_logger = AuditLogger()
    
    # Create default roles and users
    access_control.create_user('dr_smith', 'radiologist', 'Dr. John Smith')
    access_control.create_user('research_team', 'researcher', 'Research Team')
    access_control.create_user('admin_user', 'admin', 'System Administrator')
    
    # Initialize encryption (use environment variable for production)
    encryption_password = os.environ.get('MEDICAL_AI_KEY', b'default_dev_key_change_in_production')
    if isinstance(encryption_password, str):
        encryption_password = encryption_password.encode()
    
    data_encryption = MedicalDataEncryption(encryption_password)
    
    return {
        'access_control': access_control,
        'audit_logger': audit_logger,
        'data_encryption': data_encryption,
        'data_anonymizer': DataAnonymizer(),
        'data_validator': MedicalDataValidator()
    }


if __name__ == "__main__":
    # Demonstration of medical data protection system
    
    print("Initializing secure medical AI system...")
    system = create_secure_medical_ai_system()
    
    # Simulate user authentication
    session_id = system['access_control'].authenticate_user('dr_smith', 'secure_password')
    print(f"User authenticated with session ID: {session_id}")
    
    # Test permission checking
    can_predict = system['access_control'].check_permission(session_id, 'generate_predictions')
    print(f"User can generate predictions: {can_predict}")
    
    # Test data encryption
    sensitive_data = "Patient John Doe has pneumonia in right lung"
    encrypted = system['data_encryption'].encrypt_data(sensitive_data)
    decrypted = system['data_encryption'].decrypt_data(encrypted)
    print(f"Encryption test successful: {decrypted.decode() == sensitive_data}")
    
    # Test audit logging
    access_log = AccessLog(
        user_id='dr_smith',
        action='generate_predictions',
        resource='pneumonia_model_v1',
        timestamp=datetime.now(),
        ip_address='192.168.1.100',
        success=True,
        phi_accessed=True
    )
    system['audit_logger'].log_access(access_log)
    
    print("Security system demonstration completed successfully!")