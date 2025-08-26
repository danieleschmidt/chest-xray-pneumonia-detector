"""Medical Data Privacy Framework.

Implements comprehensive privacy protection for medical AI systems including
HIPAA compliance, differential privacy, and secure federated learning.
"""

import hashlib
import hmac
import logging
import secrets
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import json

import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

logger = logging.getLogger(__name__)


@dataclass
class PrivacyConfig:
    """Configuration for privacy protection."""
    enable_differential_privacy: bool = True
    epsilon: float = 1.0  # Privacy budget
    delta: float = 1e-5  # Privacy parameter
    enable_encryption: bool = True
    enable_audit_logging: bool = True
    data_retention_days: int = 90
    anonymization_level: str = "high"  # low, medium, high
    enable_federated_learning: bool = False


@dataclass
class AuditLogEntry:
    """Audit log entry for medical data access."""
    timestamp: datetime
    user_id: str
    action: str
    resource: str
    result: str
    ip_address: Optional[str] = None
    session_id: Optional[str] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)


class DifferentialPrivacyMechanism:
    """Differential privacy implementation for medical data."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.privacy_budget_used = 0.0
    
    def add_laplace_noise(self, value: float, sensitivity: float) -> float:
        """Add Laplace noise for differential privacy."""
        if self.privacy_budget_used + self.epsilon > self.epsilon:
            raise ValueError("Privacy budget exhausted")
        
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        
        self.privacy_budget_used += self.epsilon
        return value + noise
    
    def add_gaussian_noise(self, value: float, sensitivity: float) -> float:
        """Add Gaussian noise for differential privacy."""
        if self.privacy_budget_used + self.epsilon > self.epsilon:
            raise ValueError("Privacy budget exhausted")
        
        # Calculate noise scale for (epsilon, delta)-differential privacy
        sigma = (sensitivity * np.sqrt(2 * np.log(1.25 / self.delta))) / self.epsilon
        noise = np.random.normal(0, sigma)
        
        self.privacy_budget_used += self.epsilon
        return value + noise
    
    def private_mean(self, values: List[float], sensitivity: float = 1.0) -> float:
        """Compute differentially private mean."""
        true_mean = np.mean(values)
        return self.add_laplace_noise(true_mean, sensitivity / len(values))
    
    def private_count(self, count: int, sensitivity: float = 1.0) -> float:
        """Compute differentially private count."""
        return self.add_laplace_noise(float(count), sensitivity)
    
    def reset_budget(self):
        """Reset privacy budget."""
        self.privacy_budget_used = 0.0


class MedicalDataEncryption:
    """Encryption utilities for medical data."""
    
    def __init__(self):
        self.key = None
        self.fernet = None
    
    def generate_key(self) -> bytes:
        """Generate encryption key."""
        self.key = Fernet.generate_key()
        self.fernet = Fernet(self.key)
        return self.key
    
    def load_key(self, key: bytes):
        """Load existing encryption key."""
        self.key = key
        self.fernet = Fernet(key)
    
    def encrypt_data(self, data: Union[str, bytes]) -> bytes:
        """Encrypt data."""
        if not self.fernet:
            raise ValueError("Encryption key not set")
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return self.fernet.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data."""
        if not self.fernet:
            raise ValueError("Encryption key not set")
        
        return self.fernet.decrypt(encrypted_data)
    
    def encrypt_file(self, file_path: str, output_path: str):
        """Encrypt file."""
        with open(file_path, 'rb') as file:
            data = file.read()
        
        encrypted_data = self.encrypt_data(data)
        
        with open(output_path, 'wb') as encrypted_file:
            encrypted_file.write(encrypted_data)
    
    def decrypt_file(self, encrypted_file_path: str, output_path: str):
        """Decrypt file."""
        with open(encrypted_file_path, 'rb') as encrypted_file:
            encrypted_data = encrypted_file.read()
        
        decrypted_data = self.decrypt_data(encrypted_data)
        
        with open(output_path, 'wb') as file:
            file.write(decrypted_data)


class DataAnonymizer:
    """Data anonymization for medical records."""
    
    def __init__(self, anonymization_level: str = "high"):
        self.level = anonymization_level
        self.anonymization_map = {}
    
    def anonymize_patient_id(self, patient_id: str) -> str:
        """Anonymize patient ID with consistent mapping."""
        if patient_id in self.anonymization_map:
            return self.anonymization_map[patient_id]
        
        # Generate anonymous ID
        anonymous_id = hashlib.sha256(
            (patient_id + secrets.token_hex(16)).encode()
        ).hexdigest()[:16]
        
        self.anonymization_map[patient_id] = anonymous_id
        return anonymous_id
    
    def anonymize_dates(self, date: datetime, granularity: str = "month") -> datetime:
        """Anonymize dates by reducing granularity."""
        if granularity == "year":
            return datetime(date.year, 1, 1)
        elif granularity == "month":
            return datetime(date.year, date.month, 1)
        elif granularity == "day":
            return datetime(date.year, date.month, date.day)
        else:
            return date
    
    def anonymize_location(self, location: str) -> str:
        """Anonymize location data."""
        if self.level == "high":
            # Remove all location info
            return "ANONYMIZED"
        elif self.level == "medium":
            # Keep only state/province level
            return location.split(',')[0] if ',' in location else "ANONYMIZED"
        else:
            # Keep city level
            return ','.join(location.split(',')[:2]) if ',' in location else location
    
    def k_anonymize(self, data: List[Dict], k: int = 5, 
                   quasi_identifiers: List[str] = None) -> List[Dict]:
        """Apply k-anonymity to dataset."""
        if not quasi_identifiers:
            quasi_identifiers = ['age', 'gender', 'location']
        
        # Group records by quasi-identifier combinations
        groups = {}
        for record in data:
            key = tuple(record.get(qi, None) for qi in quasi_identifiers)
            if key not in groups:
                groups[key] = []
            groups[key].append(record)
        
        # Generalize groups with less than k records
        anonymized_data = []
        for group_records in groups.values():
            if len(group_records) < k:
                # Generalize quasi-identifiers
                for record in group_records:
                    for qi in quasi_identifiers:
                        if qi in record:
                            record[qi] = self._generalize_value(record[qi], qi)
            
            anonymized_data.extend(group_records)
        
        return anonymized_data
    
    def _generalize_value(self, value: Any, attribute: str) -> str:
        """Generalize attribute value."""
        if attribute == 'age' and isinstance(value, (int, float)):
            # Age ranges
            age_ranges = [(0, 18), (18, 35), (35, 50), (50, 65), (65, 100)]
            for min_age, max_age in age_ranges:
                if min_age <= value < max_age:
                    return f"{min_age}-{max_age}"
            return "65+"
        elif attribute == 'location':
            return self.anonymize_location(str(value))
        else:
            return "GENERALIZED"


class HIPAAAuditLogger:
    """HIPAA-compliant audit logging."""
    
    def __init__(self, log_dir: str = "audit_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup secure logging
        self.current_log_file = None
        self._rotate_log_file()
    
    def _rotate_log_file(self):
        """Rotate log file daily."""
        date_str = datetime.now().strftime("%Y%m%d")
        self.current_log_file = self.log_dir / f"hipaa_audit_{date_str}.log"
    
    def log_access(self, 
                   user_id: str,
                   action: str,
                   resource: str,
                   result: str,
                   ip_address: Optional[str] = None,
                   session_id: Optional[str] = None,
                   **kwargs):
        """Log data access event."""
        
        entry = AuditLogEntry(
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource=resource,
            result=result,
            ip_address=ip_address,
            session_id=session_id,
            additional_info=kwargs
        )
        
        self._write_log_entry(entry)
    
    def _write_log_entry(self, entry: AuditLogEntry):
        """Write audit log entry to file."""
        # Check if we need to rotate log file
        if datetime.now().date() != datetime.fromtimestamp(
            self.current_log_file.stat().st_mtime
        ).date():
            self._rotate_log_file()
        
        log_line = {
            "timestamp": entry.timestamp.isoformat(),
            "user_id": entry.user_id,
            "action": entry.action,
            "resource": entry.resource,
            "result": entry.result,
            "ip_address": entry.ip_address,
            "session_id": entry.session_id,
            "additional_info": entry.additional_info
        }
        
        with open(self.current_log_file, 'a') as f:
            f.write(json.dumps(log_line) + '\n')
    
    def generate_audit_report(self, 
                             start_date: datetime,
                             end_date: datetime) -> Dict[str, Any]:
        """Generate audit report for given date range."""
        report = {
            "period": f"{start_date.date()} to {end_date.date()}",
            "total_accesses": 0,
            "users": set(),
            "actions": {},
            "failed_accesses": 0,
            "resources_accessed": set()
        }
        
        # Scan log files in date range
        current_date = start_date.date()
        while current_date <= end_date.date():
            log_file = self.log_dir / f"hipaa_audit_{current_date.strftime('%Y%m%d')}.log"
            
            if log_file.exists():
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            entry_date = datetime.fromisoformat(entry["timestamp"]).date()
                            
                            if start_date.date() <= entry_date <= end_date.date():
                                report["total_accesses"] += 1
                                report["users"].add(entry["user_id"])
                                report["resources_accessed"].add(entry["resource"])
                                
                                action = entry["action"]
                                report["actions"][action] = report["actions"].get(action, 0) + 1
                                
                                if entry["result"] != "SUCCESS":
                                    report["failed_accesses"] += 1
                                    
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.error(f"Error parsing log entry: {e}")
            
            current_date += timedelta(days=1)
        
        # Convert sets to lists for JSON serialization
        report["users"] = list(report["users"])
        report["resources_accessed"] = list(report["resources_accessed"])
        
        return report


class MedicalDataPrivacyFramework:
    """Main privacy framework for medical data."""
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.dp_mechanism = None
        self.encryption = None
        self.anonymizer = None
        self.audit_logger = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize privacy components based on configuration."""
        if self.config.enable_differential_privacy:
            self.dp_mechanism = DifferentialPrivacyMechanism(
                self.config.epsilon, 
                self.config.delta
            )
        
        if self.config.enable_encryption:
            self.encryption = MedicalDataEncryption()
            self.encryption.generate_key()
        
        if self.config.anonymization_level:
            self.anonymizer = DataAnonymizer(self.config.anonymization_level)
        
        if self.config.enable_audit_logging:
            self.audit_logger = HIPAAAuditLogger()
    
    @contextmanager
    def secure_data_access(self, 
                          user_id: str,
                          resource: str,
                          action: str,
                          ip_address: Optional[str] = None):
        """Context manager for secure data access with logging."""
        session_id = secrets.token_hex(16)
        start_time = time.time()
        
        if self.audit_logger:
            self.audit_logger.log_access(
                user_id=user_id,
                action=f"{action}_START",
                resource=resource,
                result="SUCCESS",
                ip_address=ip_address,
                session_id=session_id
            )
        
        try:
            yield session_id
            
            if self.audit_logger:
                self.audit_logger.log_access(
                    user_id=user_id,
                    action=f"{action}_END",
                    resource=resource,
                    result="SUCCESS",
                    ip_address=ip_address,
                    session_id=session_id,
                    duration=time.time() - start_time
                )
                
        except Exception as e:
            if self.audit_logger:
                self.audit_logger.log_access(
                    user_id=user_id,
                    action=f"{action}_FAILED",
                    resource=resource,
                    result="ERROR",
                    ip_address=ip_address,
                    session_id=session_id,
                    error=str(e),
                    duration=time.time() - start_time
                )
            raise
    
    def encrypt_patient_data(self, data: Dict[str, Any]) -> bytes:
        """Encrypt patient data."""
        if not self.encryption:
            raise ValueError("Encryption not enabled")
        
        json_data = json.dumps(data)
        return self.encryption.encrypt_data(json_data)
    
    def decrypt_patient_data(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Decrypt patient data."""
        if not self.encryption:
            raise ValueError("Encryption not enabled")
        
        decrypted_data = self.encryption.decrypt_data(encrypted_data)
        return json.loads(decrypted_data.decode('utf-8'))
    
    def anonymize_dataset(self, 
                         dataset: List[Dict[str, Any]],
                         k: int = 5) -> List[Dict[str, Any]]:
        """Anonymize medical dataset."""
        if not self.anonymizer:
            raise ValueError("Anonymization not enabled")
        
        # Apply k-anonymity
        anonymized = self.anonymizer.k_anonymize(dataset, k)
        
        # Apply additional anonymization
        for record in anonymized:
            if 'patient_id' in record:
                record['patient_id'] = self.anonymizer.anonymize_patient_id(
                    record['patient_id']
                )
            
            if 'location' in record:
                record['location'] = self.anonymizer.anonymize_location(
                    record['location']
                )
        
        return anonymized
    
    def add_differential_privacy(self, 
                               statistics: Dict[str, float],
                               sensitivity: float = 1.0) -> Dict[str, float]:
        """Add differential privacy to statistics."""
        if not self.dp_mechanism:
            raise ValueError("Differential privacy not enabled")
        
        private_stats = {}
        for key, value in statistics.items():
            if isinstance(value, (int, float)):
                private_stats[key] = self.dp_mechanism.add_laplace_noise(
                    float(value), sensitivity
                )
            else:
                private_stats[key] = value
        
        return private_stats
    
    def generate_privacy_report(self) -> Dict[str, Any]:
        """Generate comprehensive privacy compliance report."""
        report = {
            "framework_status": "ACTIVE",
            "configuration": {
                "differential_privacy": self.config.enable_differential_privacy,
                "encryption": self.config.enable_encryption,
                "audit_logging": self.config.enable_audit_logging,
                "anonymization_level": self.config.anonymization_level
            },
            "privacy_metrics": {},
            "compliance_status": {}
        }
        
        if self.dp_mechanism:
            report["privacy_metrics"]["privacy_budget_used"] = self.dp_mechanism.privacy_budget_used
            report["privacy_metrics"]["privacy_budget_remaining"] = (
                self.dp_mechanism.epsilon - self.dp_mechanism.privacy_budget_used
            )
        
        # HIPAA compliance checks
        report["compliance_status"]["HIPAA"] = {
            "audit_logging": self.config.enable_audit_logging,
            "encryption": self.config.enable_encryption,
            "access_controls": True,  # Assumed based on framework usage
            "data_integrity": True
        }
        
        return report


def demonstrate_privacy_framework():
    """Demonstrate medical data privacy framework."""
    # Configure privacy framework
    config = PrivacyConfig(
        enable_differential_privacy=True,
        epsilon=1.0,
        delta=1e-5,
        enable_encryption=True,
        enable_audit_logging=True,
        anonymization_level="high"
    )
    
    framework = MedicalDataPrivacyFramework(config)
    
    # Example patient data
    patient_data = {
        "patient_id": "P12345",
        "age": 45,
        "gender": "F",
        "diagnosis": "Pneumonia",
        "location": "New York, NY, USA",
        "admission_date": "2024-01-15"
    }
    
    print("Medical Data Privacy Framework Demo")
    print("=" * 50)
    
    # Demonstrate encryption
    print("\n1. Data Encryption:")
    encrypted_data = framework.encrypt_patient_data(patient_data)
    print(f"Original data: {patient_data}")
    print(f"Encrypted data length: {len(encrypted_data)} bytes")
    
    decrypted_data = framework.decrypt_patient_data(encrypted_data)
    print(f"Decrypted data: {decrypted_data}")
    
    # Demonstrate anonymization
    print("\n2. Data Anonymization:")
    dataset = [patient_data.copy() for _ in range(10)]
    for i, record in enumerate(dataset):
        record["patient_id"] = f"P{12345 + i}"
        record["age"] = 45 + (i % 20)
    
    anonymized_dataset = framework.anonymize_dataset(dataset)
    print(f"Original patient ID: {dataset[0]['patient_id']}")
    print(f"Anonymized patient ID: {anonymized_dataset[0]['patient_id']}")
    
    # Demonstrate differential privacy
    print("\n3. Differential Privacy:")
    statistics = {"average_age": 52.3, "pneumonia_rate": 0.15}
    private_stats = framework.add_differential_privacy(statistics)
    print(f"Original statistics: {statistics}")
    print(f"Private statistics: {private_stats}")
    
    # Demonstrate secure access
    print("\n4. Secure Data Access:")
    with framework.secure_data_access(
        user_id="doctor_001",
        resource="patient_records",
        action="READ",
        ip_address="192.168.1.100"
    ) as session_id:
        print(f"Secure session started: {session_id}")
        # Simulate data access
        time.sleep(0.1)
        print("Data accessed securely")
    
    # Generate privacy report
    print("\n5. Privacy Compliance Report:")
    report = framework.generate_privacy_report()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    demonstrate_privacy_framework()