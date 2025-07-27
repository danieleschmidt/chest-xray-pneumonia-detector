"""
HIPAA compliance and healthcare data protection utilities.
"""

import hashlib
import logging
import re
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class HIPAACompliance:
    """HIPAA compliance utilities for healthcare data protection."""
    
    # PHI (Protected Health Information) patterns
    PHI_PATTERNS = {
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'date_of_birth': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
        'medical_record_number': r'\bMRN\s*:?\s*\d+\b',
        'account_number': r'\bAccount\s*:?\s*\d+\b',
        'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
    }
    
    def __init__(self):
        self.anonymization_map = {}
        self.audit_trail = []
    
    def scan_for_phi(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Scan text for potential PHI."""
        findings = {}
        
        for phi_type, pattern in self.PHI_PATTERNS.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            phi_findings = []
            
            for match in matches:
                phi_findings.append({
                    'value': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'context': text[max(0, match.start()-20):match.end()+20]
                })
            
            if phi_findings:
                findings[phi_type] = phi_findings
        
        return findings
    
    def anonymize_text(self, text: str, preserve_format: bool = True) -> Tuple[str, Dict[str, str]]:
        """Anonymize PHI in text while preserving format."""
        anonymized_text = text
        anonymization_map = {}
        
        for phi_type, pattern in self.PHI_PATTERNS.items():
            matches = list(re.finditer(pattern, anonymized_text, re.IGNORECASE))
            
            # Process matches in reverse order to preserve indices
            for match in reversed(matches):
                original_value = match.group()
                anonymized_value = self._generate_anonymous_value(phi_type, original_value, preserve_format)
                
                # Replace in text
                anonymized_text = (
                    anonymized_text[:match.start()] + 
                    anonymized_value + 
                    anonymized_text[match.end():]
                )
                
                # Store mapping
                anonymization_map[original_value] = anonymized_value
        
        return anonymized_text, anonymization_map
    
    def _generate_anonymous_value(self, phi_type: str, original_value: str, preserve_format: bool) -> str:
        """Generate anonymous replacement value."""
        if not preserve_format:
            return f"[REDACTED_{phi_type.upper()}]"
        
        if phi_type == 'ssn':
            return 'XXX-XX-XXXX'
        elif phi_type == 'phone':
            # Preserve format of phone number
            if '-' in original_value:
                return 'XXX-XXX-XXXX'
            elif '.' in original_value:
                return 'XXX.XXX.XXXX'
            else:
                return 'XXXXXXXXXX'
        elif phi_type == 'email':
            return 'patient@example.com'
        elif phi_type == 'date_of_birth':
            return '01/01/1900'
        elif phi_type == 'medical_record_number':
            return 'MRN: XXXXXXX'
        elif phi_type == 'account_number':
            return 'Account: XXXXXXX'
        elif phi_type == 'ip_address':
            return 'XXX.XXX.XXX.XXX'
        else:
            return f'[ANON_{phi_type.upper()}]'
    
    def hash_identifier(self, identifier: str, salt: Optional[str] = None) -> str:
        """Create a consistent hash for identifiers."""
        if salt is None:
            salt = "chest_xray_detector_salt"  # Use consistent salt for matching
        
        # Create SHA-256 hash
        hash_input = f"{identifier}{salt}".encode('utf-8')
        return hashlib.sha256(hash_input).hexdigest()[:16]  # Use first 16 chars
    
    def generate_patient_id(self) -> str:
        """Generate a HIPAA-compliant anonymous patient ID."""
        return f"ANON_{uuid.uuid4().hex[:12].upper()}"
    
    def log_data_access(self, user_id: str, action: str, data_type: str, 
                       patient_id: Optional[str] = None, purpose: Optional[str] = None):
        """Log data access for HIPAA audit trail."""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'user_id': user_id,
            'action': action,
            'data_type': data_type,
            'patient_id': patient_id,
            'purpose': purpose,
            'session_id': getattr(self, '_current_session_id', None)
        }
        
        self.audit_trail.append(audit_entry)
        
        # Log to application logger as well
        logger.info(
            f"HIPAA audit: {action}",
            extra={
                'audit_type': 'hipaa_data_access',
                **audit_entry
            }
        )
    
    def validate_data_retention(self, creation_date: datetime, 
                              retention_period_days: int = 2555) -> Dict[str, Any]:
        """Validate data retention compliance (default 7 years for medical records)."""
        now = datetime.utcnow()
        age_days = (now - creation_date).days
        days_until_deletion = retention_period_days - age_days
        
        return {
            'age_days': age_days,
            'retention_period_days': retention_period_days,
            'days_until_deletion': days_until_deletion,
            'should_delete': days_until_deletion <= 0,
            'warning_threshold': days_until_deletion <= 30
        }
    
    def generate_consent_tracking(self, patient_id: str, consent_types: List[str]) -> Dict[str, Any]:
        """Generate consent tracking record."""
        consent_id = str(uuid.uuid4())
        
        consent_record = {
            'consent_id': consent_id,
            'patient_id': patient_id,
            'consent_types': consent_types,
            'granted_at': datetime.utcnow().isoformat() + 'Z',
            'expires_at': (datetime.utcnow() + timedelta(days=365)).isoformat() + 'Z',
            'status': 'active',
            'ip_address': '[LOGGED_SEPARATELY]',
            'user_agent': '[LOGGED_SEPARATELY]'
        }
        
        return consent_record
    
    def check_minimum_necessary(self, requested_data: List[str], 
                               purpose: str) -> Dict[str, Any]:
        """Check minimum necessary standard compliance."""
        # Define data categories and their purposes
        data_purposes = {
            'diagnosis': ['image_data', 'patient_demographics', 'medical_history'],
            'treatment': ['image_data', 'diagnosis_results', 'treatment_plan'],
            'research': ['anonymized_image_data', 'anonymized_demographics'],
            'quality_assurance': ['image_data', 'diagnosis_results'],
            'billing': ['patient_demographics', 'procedure_codes']
        }
        
        allowed_data = data_purposes.get(purpose.lower(), [])
        
        compliant_data = []
        non_compliant_data = []
        
        for data_item in requested_data:
            if data_item in allowed_data:
                compliant_data.append(data_item)
            else:
                non_compliant_data.append(data_item)
        
        return {
            'purpose': purpose,
            'requested_data': requested_data,
            'compliant_data': compliant_data,
            'non_compliant_data': non_compliant_data,
            'is_compliant': len(non_compliant_data) == 0,
            'allowed_for_purpose': allowed_data
        }
    
    def encrypt_phi_data(self, data: str, encryption_key: str) -> str:
        """Encrypt PHI data for storage."""
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            import base64
            
            # Derive key from password
            password = encryption_key.encode()
            salt = b'hipaa_phi_salt_2025'  # Use consistent salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            fernet = Fernet(key)
            
            # Encrypt data
            encrypted_data = fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
            
        except ImportError:
            logger.error("Cryptography library not available for PHI encryption")
            raise Exception("PHI encryption not available")
        except Exception as e:
            logger.error(f"Failed to encrypt PHI data: {e}")
            raise
    
    def decrypt_phi_data(self, encrypted_data: str, encryption_key: str) -> str:
        """Decrypt PHI data."""
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            import base64
            
            # Derive key from password
            password = encryption_key.encode()
            salt = b'hipaa_phi_salt_2025'  # Use consistent salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            fernet = Fernet(key)
            
            # Decrypt data
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = fernet.decrypt(decoded_data)
            return decrypted_data.decode()
            
        except ImportError:
            logger.error("Cryptography library not available for PHI decryption")
            raise Exception("PHI decryption not available")
        except Exception as e:
            logger.error(f"Failed to decrypt PHI data: {e}")
            raise
    
    def generate_breach_report(self, incident_details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate breach notification report template."""
        return {
            'incident_id': str(uuid.uuid4()),
            'discovery_date': datetime.utcnow().isoformat() + 'Z',
            'incident_date': incident_details.get('incident_date'),
            'description': incident_details.get('description', ''),
            'affected_individuals': incident_details.get('affected_count', 0),
            'data_types_involved': incident_details.get('data_types', []),
            'cause': incident_details.get('cause', ''),
            'mitigation_actions': incident_details.get('mitigation_actions', []),
            'notification_required': incident_details.get('affected_count', 0) >= 500,
            'notification_deadline': (datetime.utcnow() + timedelta(days=60)).isoformat() + 'Z',
            'status': 'under_investigation',
            'created_by': incident_details.get('created_by', ''),
            'last_updated': datetime.utcnow().isoformat() + 'Z'
        }
    
    def save_audit_trail(self, file_path: str):
        """Save audit trail to file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.audit_trail, f, indent=2, default=str)
            
            # Set restrictive permissions
            import os
            os.chmod(file_path, 0o600)
            
            logger.info(f"Audit trail saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save audit trail: {e}")
            raise


class DataClassification:
    """Data classification system for healthcare data."""
    
    CLASSIFICATION_LEVELS = {
        'public': {
            'level': 1,
            'description': 'Information that can be freely shared',
            'examples': ['General health education', 'Public research findings']
        },
        'internal': {
            'level': 2,
            'description': 'Information for internal organizational use',
            'examples': ['Internal policies', 'Training materials']
        },
        'confidential': {
            'level': 3,
            'description': 'Sensitive information requiring protection',
            'examples': ['Patient lists', 'Aggregate health statistics']
        },
        'restricted': {
            'level': 4,
            'description': 'Highly sensitive PHI requiring strict controls',
            'examples': ['Individual patient records', 'Medical images with identifiers']
        }
    }
    
    def __init__(self):
        self.classification_rules = self._init_classification_rules()
    
    def _init_classification_rules(self) -> Dict[str, str]:
        """Initialize data classification rules."""
        return {
            'patient_name': 'restricted',
            'patient_id': 'restricted',
            'medical_record_number': 'restricted',
            'date_of_birth': 'restricted',
            'ssn': 'restricted',
            'phone_number': 'restricted',
            'email_address': 'restricted',
            'address': 'restricted',
            'medical_image_with_metadata': 'restricted',
            'medical_image_anonymized': 'confidential',
            'diagnosis_code': 'confidential',
            'treatment_code': 'confidential',
            'aggregate_statistics': 'internal',
            'research_data_anonymized': 'internal',
            'system_logs': 'internal',
            'public_health_info': 'public'
        }
    
    def classify_data(self, data_type: str, contains_phi: bool = False) -> Dict[str, Any]:
        """Classify data based on type and PHI content."""
        base_classification = self.classification_rules.get(data_type.lower(), 'confidential')
        
        # Upgrade classification if PHI is detected
        if contains_phi and base_classification != 'restricted':
            classification = 'restricted'
        else:
            classification = base_classification
        
        return {
            'data_type': data_type,
            'classification': classification,
            'level': self.CLASSIFICATION_LEVELS[classification]['level'],
            'description': self.CLASSIFICATION_LEVELS[classification]['description'],
            'handling_requirements': self._get_handling_requirements(classification),
            'retention_period': self._get_retention_period(classification),
            'access_controls': self._get_access_controls(classification)
        }
    
    def _get_handling_requirements(self, classification: str) -> List[str]:
        """Get handling requirements for classification level."""
        requirements = {
            'public': ['No special handling required'],
            'internal': ['Internal use only', 'Authorized personnel access'],
            'confidential': [
                'Encrypted in transit and at rest',
                'Access logging required',
                'Need-to-know basis'
            ],
            'restricted': [
                'AES-256 encryption required',
                'Multi-factor authentication',
                'Comprehensive audit logging',
                'HIPAA compliance required',
                'Minimum necessary access only'
            ]
        }
        return requirements.get(classification, [])
    
    def _get_retention_period(self, classification: str) -> str:
        """Get retention period for classification level."""
        retention = {
            'public': 'Indefinite',
            'internal': '7 years',
            'confidential': '7 years', 
            'restricted': '7 years (medical records)'
        }
        return retention.get(classification, '7 years')
    
    def _get_access_controls(self, classification: str) -> List[str]:
        """Get access control requirements."""
        controls = {
            'public': ['No access controls required'],
            'internal': ['Organization member authentication'],
            'confidential': [
                'Role-based access control',
                'Regular access reviews'
            ],
            'restricted': [
                'Strict role-based access control',
                'Multi-factor authentication',
                'Real-time access monitoring',
                'Quarterly access reviews',
                'Break-glass procedures documented'
            ]
        }
        return controls.get(classification, [])


# Global instances
hipaa_compliance = HIPAACompliance()
data_classifier = DataClassification()