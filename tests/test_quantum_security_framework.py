"""Comprehensive tests for the Quantum Security Framework.

This module tests quantum-resistant encryption, homomorphic computing,
and advanced threat detection for medical AI systems.
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import json
import base64

# Import the modules to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from quantum_security_framework import (
        SecurityLevel, ThreatLevel, SecurityEvent, QuantumResistantCrypto,
        HomomorphicComputing, AdvancedThreatDetector, SecurityAuditLogger,
        QuantumSecurityFramework, secure_medical_inference, SecurityError
    )
except ImportError:
    # Fallback for testing environment
    QuantumResistantCrypto = Mock
    HomomorphicComputing = Mock
    AdvancedThreatDetector = Mock
    SecurityAuditLogger = Mock
    QuantumSecurityFramework = Mock


class TestSecurityEvent:
    """Test security event functionality."""
    
    def test_security_event_initialization(self):
        """Test security event initialization with defaults."""
        event = SecurityEvent(
            event_type="login_attempt",
            user_id="user123",
            action="login"
        )
        
        assert event.event_type == "login_attempt"
        assert event.user_id == "user123"
        assert event.action == "login"
        assert event.severity == ThreatLevel.LOW
        assert event.success == True
        assert isinstance(event.details, dict)
        
    def test_security_event_to_dict(self):
        """Test security event conversion to dictionary."""
        event = SecurityEvent(
            event_type="access_denied",
            severity=ThreatLevel.HIGH,
            user_id="user456",
            action="unauthorized_access",
            success=False,
            details={"reason": "invalid_credentials"}
        )
        
        event_dict = event.to_dict()
        
        assert event_dict["event_type"] == "access_denied"
        assert event_dict["severity"] == "high"
        assert event_dict["user_id"] == "user456"
        assert event_dict["action"] == "unauthorized_access"
        assert event_dict["success"] == False
        assert event_dict["details"]["reason"] == "invalid_credentials"
        assert "timestamp" in event_dict


class TestQuantumResistantCrypto:
    """Test quantum-resistant cryptographic operations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.crypto = QuantumResistantCrypto()
        
    def test_generate_quantum_safe_key(self):
        """Test quantum-safe key generation."""
        key = self.crypto.generate_quantum_safe_key(256)
        
        assert len(key) == 32  # 256 bits = 32 bytes
        assert isinstance(key, bytes)
        
        # Generate another key and ensure they're different
        key2 = self.crypto.generate_quantum_safe_key(256)
        assert key != key2
        
    def test_generate_lattice_keypair(self):
        """Test lattice-based key pair generation."""
        public_key, private_key = self.crypto.generate_lattice_keypair()
        
        assert len(public_key) == 32  # SHA3-256 output
        assert len(private_key) == 32
        assert isinstance(public_key, bytes)
        assert isinstance(private_key, bytes)
        assert public_key != private_key
        
    def test_lattice_encryption_decryption(self):
        """Test lattice-based encryption and decryption."""
        public_key, private_key = self.crypto.generate_lattice_keypair()
        
        original_data = b"Sensitive medical data for patient #12345"
        encrypted_data = self.crypto.encrypt_with_lattice(original_data, public_key)
        decrypted_data = self.crypto.decrypt_with_lattice(encrypted_data, private_key)
        
        assert decrypted_data == original_data
        assert encrypted_data != original_data
        assert len(encrypted_data) > len(original_data)  # Encrypted data should be larger
        
    def test_aes_encryption_decryption(self):
        """Test AES encryption and decryption."""
        key = self.crypto.generate_quantum_safe_key(256)
        original_data = b"Medical record data that needs protection"
        
        encrypted_data = self.crypto.aes_encrypt(original_data, key)
        decrypted_data = self.crypto.aes_decrypt(encrypted_data, key)
        
        assert decrypted_data == original_data
        assert encrypted_data != original_data
        assert len(encrypted_data) > len(original_data)  # IV + tag + ciphertext
        
    def test_aes_encryption_with_wrong_key_fails(self):
        """Test that AES decryption fails with wrong key."""
        key1 = self.crypto.generate_quantum_safe_key(256)
        key2 = self.crypto.generate_quantum_safe_key(256)
        
        original_data = b"Secret medical data"
        encrypted_data = self.crypto.aes_encrypt(original_data, key1)
        
        # Decryption with wrong key should fail
        with pytest.raises(Exception):
            self.crypto.aes_decrypt(encrypted_data, key2)


class TestHomomorphicComputing:
    """Test homomorphic computing functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.homomorphic = HomomorphicComputing()
        
    def test_encrypt_data_for_inference(self):
        """Test data encryption for homomorphic inference."""
        # Create sample medical image data
        medical_data = np.random.rand(28, 28).astype(np.float32)
        
        encrypted_package = self.homomorphic.encrypt_data_for_inference(medical_data)
        
        assert "encrypted_data" in encrypted_package
        assert "shape" in encrypted_package
        assert "encryption_key" in encrypted_package
        assert "scale_factor" in encrypted_package
        
        assert encrypted_package["shape"] == medical_data.shape
        assert len(encrypted_package["encrypted_data"]) == medical_data.size
        assert encrypted_package["scale_factor"] == 1000000
        
    def test_homomorphic_inference(self):
        """Test homomorphic inference computation."""
        # Create sample data and model weights
        medical_data = np.random.rand(10, 5).astype(np.float32)
        model_weights = np.random.rand(5, 2).astype(np.float32)
        
        # Encrypt data
        encrypted_data = self.homomorphic.encrypt_data_for_inference(medical_data)
        
        # Perform homomorphic inference
        encrypted_result = self.homomorphic.homomorphic_inference(encrypted_data, model_weights)
        
        assert "encrypted_result" in encrypted_result
        assert "result_shape" in encrypted_result
        assert "scale_factor" in encrypted_result
        
        # Verify result shape
        expected_shape = (medical_data.shape[0], model_weights.shape[1])
        assert encrypted_result["result_shape"] == expected_shape
        
    def test_decrypt_inference_result(self):
        """Test decryption of homomorphic inference results."""
        # Create and process data
        medical_data = np.random.rand(5, 3).astype(np.float32)
        model_weights = np.random.rand(3, 2).astype(np.float32)
        
        encrypted_data = self.homomorphic.encrypt_data_for_inference(medical_data)
        encrypted_result = self.homomorphic.homomorphic_inference(encrypted_data, model_weights)
        
        # Decrypt result
        decrypted_result = self.homomorphic.decrypt_inference_result(
            encrypted_result, 
            encrypted_data["encryption_key"]
        )
        
        # Verify result properties
        expected_shape = (medical_data.shape[0], model_weights.shape[1])
        assert decrypted_result.shape == expected_shape
        assert isinstance(decrypted_result, np.ndarray)
        
    def test_additive_encryption_decryption(self):
        """Test additive homomorphic encryption operations."""
        key = b"test_key_32_bytes_long_for_testing"
        original_value = 12345
        
        encrypted_value = self.homomorphic._additive_encrypt(original_value, key)
        decrypted_value = self.homomorphic._additive_decrypt(encrypted_value, key)
        
        assert decrypted_value == original_value
        assert encrypted_value != original_value


class TestAdvancedThreatDetector:
    """Test advanced threat detection functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.detector = AdvancedThreatDetector()
        
    def test_detector_initialization(self):
        """Test threat detector initialization."""
        assert hasattr(self.detector, 'threat_patterns')
        assert hasattr(self.detector, 'anomaly_baseline')
        assert hasattr(self.detector, 'threat_scores')
        
    def test_detect_no_threats_clean_request(self):
        """Test threat detection with clean request."""
        clean_request = {
            "patient_id": "12345",
            "image_data": "base64_encoded_xray_image",
            "metadata": {"timestamp": "2024-01-01T10:00:00Z"}
        }
        
        threat_level, threats = self.detector.detect_threats(clean_request)
        
        assert threat_level == ThreatLevel.LOW
        assert len(threats) == 0
        
    def test_detect_sql_injection_threats(self):
        """Test SQL injection threat detection."""
        malicious_request = {
            "patient_id": "123'; DROP TABLE patients; --",
            "query": "SELECT * FROM records UNION SELECT * FROM admin_data"
        }
        
        threat_level, threats = self.detector.detect_threats(malicious_request)
        
        assert threat_level >= ThreatLevel.HIGH
        assert any("SQL injection" in threat for threat in threats)
        
    def test_detect_anomalous_behavior(self):
        """Test anomalous behavior detection."""
        anomalous_request = {
            "admin": True,
            "debug": "enable",
            "limit": "999999",  # Unusually large limit
            "data": "x" * 2000000  # Very large request
        }
        
        threat_level, threats = self.detector.detect_threats(anomalous_request)
        
        assert threat_level >= ThreatLevel.MEDIUM
        assert len(threats) > 0
        
    def test_detect_data_exfiltration_attempts(self):
        """Test data exfiltration attempt detection."""
        exfiltration_request = {
            "action": "export",
            "limit": 50000,  # Large data request
            "format": "dump",
            "include": "all_patient_data"
        }
        
        threat_level, threats = self.detector.detect_threats(exfiltration_request)
        
        assert threat_level >= ThreatLevel.CRITICAL
        assert any("exfiltration" in threat for threat in threats)
        
    def test_detect_model_extraction_attacks(self):
        """Test model extraction attack detection."""
        extraction_request = {
            "batch_size": 1000,  # Unusually large batch
            "query_type": "model_weights",
            "parameters": "all",
            "architecture": "extract"
        }
        
        threat_level, threats = self.detector.detect_threats(extraction_request)
        
        assert threat_level >= ThreatLevel.HIGH
        assert any("extraction" in threat for threat in threats)
        
    def test_detect_adversarial_inputs(self):
        """Test adversarial input detection."""
        adversarial_request = {
            "image_data": "modified_image",
            "noise_level": 0.1,
            "perturbation": "gradient_based"
        }
        
        threat_level, threats = self.detector.detect_threats(adversarial_request)
        
        assert threat_level >= ThreatLevel.MEDIUM
        assert any("adversarial" in threat for threat in threats)


class TestSecurityAuditLogger:
    """Test security audit logging functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_log_file = "/tmp/test_security_audit.log"
        self.logger = SecurityAuditLogger(self.test_log_file)
        
    def teardown_method(self):
        """Clean up test files."""
        import os
        if os.path.exists(self.test_log_file):
            os.remove(self.test_log_file)
            
    def test_logger_initialization(self):
        """Test audit logger initialization."""
        assert self.logger.log_file == self.test_log_file
        assert self.logger.logger is not None
        
    def test_log_security_event(self):
        """Test logging security events."""
        event = SecurityEvent(
            event_type="test_event",
            severity=ThreatLevel.MEDIUM,
            user_id="test_user",
            action="test_action",
            success=True
        )
        
        # Should not raise exceptions
        self.logger.log_security_event(event)
        
        # Verify log file was created
        import os
        assert os.path.exists(self.test_log_file)
        
    def test_log_access_attempt(self):
        """Test logging access attempts."""
        self.logger.log_access_attempt(
            user_id="user123",
            resource="patient_records",
            action="read",
            success=True,
            source_ip="192.168.1.100"
        )
        
        # Should create log entry
        import os
        assert os.path.exists(self.test_log_file)
        
    def test_log_threat_detection(self):
        """Test logging threat detection events."""
        threats = ["SQL injection detected", "Anomalous behavior pattern"]
        
        self.logger.log_threat_detection(
            threat_level=ThreatLevel.HIGH,
            threats=threats,
            source_ip="10.0.0.1",
            details={"request_id": "req_123"}
        )
        
        # Should create log entry
        import os
        assert os.path.exists(self.test_log_file)


class TestQuantumSecurityFramework:
    """Test the main quantum security framework."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.framework = QuantumSecurityFramework()
        
    def test_framework_initialization(self):
        """Test framework initialization."""
        assert self.framework.crypto is not None
        assert self.framework.homomorphic is not None
        assert self.framework.threat_detector is not None
        assert self.framework.audit_logger is not None
        assert len(self.framework.security_policies) > 0
        
    def test_secure_inference_request_clean(self):
        """Test securing clean inference requests."""
        clean_request = {
            "patient_id": "12345",
            "image_data": "base64_image",
            "metadata": {"timestamp": "2024-01-01"}
        }
        
        is_secure, secured_data, threats = self.framework.secure_inference_request(
            clean_request,
            SecurityLevel.PHI_PROTECTED,
            "user123",
            "192.168.1.1"
        )
        
        assert is_secure == True
        assert len(threats) == 0
        assert isinstance(secured_data, dict)
        
    def test_secure_inference_request_threats(self):
        """Test securing requests with threats."""
        malicious_request = {
            "query": "DROP TABLE patients",
            "admin": True,
            "limit": 100000
        }
        
        is_secure, secured_data, threats = self.framework.secure_inference_request(
            malicious_request,
            SecurityLevel.PHI_PROTECTED,
            "user456",
            "10.0.0.1"
        )
        
        assert is_secure == False
        assert len(threats) > 0
        
    def test_encrypt_medical_data_phi_protected(self):
        """Test medical data encryption with PHI protection."""
        medical_data = np.random.rand(100, 100).astype(np.float32)
        
        encrypted_package = self.framework.encrypt_medical_data(
            medical_data,
            SecurityLevel.PHI_PROTECTED
        )
        
        assert "encrypted_data" in encrypted_package
        assert "encryption_type" in encrypted_package
        assert "key" in encrypted_package
        assert encrypted_package["encryption_type"] == "aes_256"
        assert encrypted_package["security_level"] == "phi_protected"
        
    def test_encrypt_medical_data_quantum_safe(self):
        """Test medical data encryption with quantum-safe protection."""
        medical_data = b"Sensitive patient information"
        
        encrypted_package = self.framework.encrypt_medical_data(
            medical_data,
            SecurityLevel.QUANTUM_SAFE
        )
        
        assert "encrypted_data" in encrypted_package
        assert "encryption_type" in encrypted_package
        assert "private_key" in encrypted_package
        assert encrypted_package["encryption_type"] == "lattice_based"
        assert encrypted_package["security_level"] == "quantum_safe"
        
    def test_decrypt_medical_data_roundtrip(self):
        """Test medical data encryption/decryption roundtrip."""
        original_data = np.random.rand(50, 50).astype(np.float32)
        
        # Encrypt
        encrypted_package = self.framework.encrypt_medical_data(
            original_data,
            SecurityLevel.PHI_PROTECTED
        )
        
        # Decrypt
        decrypted_data = self.framework.decrypt_medical_data(encrypted_package)
        
        assert np.array_equal(original_data, decrypted_data)
        
    def test_privacy_preserving_inference(self):
        """Test privacy-preserving inference."""
        # Create sample data and model weights
        data = np.random.rand(10, 5).astype(np.float32)
        model_weights = np.random.rand(5, 3).astype(np.float32)
        
        result = self.framework.privacy_preserving_inference(
            data,
            model_weights,
            "user789"
        )
        
        # Verify result shape and type
        expected_shape = (data.shape[0], model_weights.shape[1])
        assert result.shape == expected_shape
        assert isinstance(result, np.ndarray)


class TestSecureMedicalInferenceDecorator:
    """Test the secure medical inference decorator."""
    
    def test_secure_decorator_clean_request(self):
        """Test decorator with clean request."""
        
        @secure_medical_inference(SecurityLevel.PHI_PROTECTED)
        def mock_inference(request_data=None, user_id="", source_ip=""):
            return {"prediction": "normal", "confidence": 0.95}
            
        result = mock_inference(
            request_data={"patient_id": "12345"},
            user_id="user123",
            source_ip="192.168.1.1"
        )
        
        assert result["prediction"] == "normal"
        assert result["confidence"] == 0.95
        
    def test_secure_decorator_threat_detection(self):
        """Test decorator with threat detection."""
        
        @secure_medical_inference(SecurityLevel.PHI_PROTECTED)
        def mock_inference(request_data=None, user_id="", source_ip=""):
            return {"prediction": "pneumonia", "confidence": 0.88}
            
        # Should raise SecurityError for malicious request
        with pytest.raises(SecurityError):
            mock_inference(
                request_data={"query": "DROP TABLE patients"},
                user_id="attacker",
                source_ip="10.0.0.1"
            )


class TestIntegrationScenarios:
    """Integration tests for complete security scenarios."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.framework = QuantumSecurityFramework()
        
    def test_end_to_end_secure_inference(self):
        """Test complete end-to-end secure inference workflow."""
        
        # Step 1: Prepare medical data
        xray_image = np.random.rand(256, 256).astype(np.float32)
        patient_data = {
            "patient_id": "P12345",
            "image_data": xray_image.tolist(),
            "metadata": {
                "scan_date": "2024-01-01",
                "scanner_id": "XR001"
            }
        }
        
        # Step 2: Secure the request
        is_secure, secured_data, threats = self.framework.secure_inference_request(
            patient_data,
            SecurityLevel.PHI_PROTECTED,
            "doctor_smith",
            "172.16.0.100"
        )
        
        assert is_secure == True
        assert len(threats) == 0
        
        # Step 3: Encrypt medical data
        encrypted_package = self.framework.encrypt_medical_data(
            xray_image,
            SecurityLevel.QUANTUM_SAFE
        )
        
        # Step 4: Verify encryption
        assert "encrypted_data" in encrypted_package
        assert encrypted_package["encryption_type"] == "lattice_based"
        
        # Step 5: Decrypt and verify
        decrypted_image = self.framework.decrypt_medical_data(encrypted_package)
        assert np.array_equal(xray_image, decrypted_image)
        
    def test_threat_detection_and_response(self):
        """Test threat detection and appropriate responses."""
        
        # Test various threat scenarios
        threat_scenarios = [
            {
                "name": "SQL Injection",
                "request": {"query": "'; DROP TABLE patients; --"},
                "expected_level": ThreatLevel.HIGH
            },
            {
                "name": "Data Exfiltration",
                "request": {"action": "export", "limit": 50000},
                "expected_level": ThreatLevel.CRITICAL
            },
            {
                "name": "Model Extraction",
                "request": {"batch_size": 1000, "extract": "weights"},
                "expected_level": ThreatLevel.HIGH
            },
            {
                "name": "Clean Request",
                "request": {"patient_id": "12345", "scan_type": "chest_xray"},
                "expected_level": ThreatLevel.LOW
            }
        ]
        
        for scenario in threat_scenarios:
            threat_level, threats = self.framework.threat_detector.detect_threats(scenario["request"])
            
            if scenario["expected_level"] == ThreatLevel.LOW:
                assert threat_level == ThreatLevel.LOW and len(threats) == 0
            else:
                assert threat_level >= scenario["expected_level"]
                assert len(threats) > 0
                
    def test_multi_level_security_policies(self):
        """Test different security levels and their policies."""
        
        test_data = np.random.rand(32, 32).astype(np.float32)
        
        # Test different security levels
        security_levels = [
            SecurityLevel.PHI_PROTECTED,
            SecurityLevel.QUANTUM_SAFE
        ]
        
        for level in security_levels:
            # Encrypt data
            encrypted_package = self.framework.encrypt_medical_data(test_data, level)
            
            # Verify encryption type matches security level
            if level == SecurityLevel.PHI_PROTECTED:
                assert encrypted_package["encryption_type"] == "aes_256"
            elif level == SecurityLevel.QUANTUM_SAFE:
                assert encrypted_package["encryption_type"] == "lattice_based"
                
            # Verify roundtrip
            decrypted_data = self.framework.decrypt_medical_data(encrypted_package)
            assert np.array_equal(test_data, decrypted_data)
            
    def test_concurrent_security_operations(self):
        """Test security framework under concurrent load."""
        
        import threading
        import time
        
        results = []
        errors = []
        
        def security_worker(worker_id):
            try:
                # Generate test data
                test_data = np.random.rand(16, 16).astype(np.float32)
                
                # Encrypt
                encrypted = self.framework.encrypt_medical_data(
                    test_data, 
                    SecurityLevel.PHI_PROTECTED
                )
                
                # Decrypt
                decrypted = self.framework.decrypt_medical_data(encrypted)
                
                # Verify
                if np.array_equal(test_data, decrypted):
                    results.append(f"worker_{worker_id}_success")
                else:
                    errors.append(f"worker_{worker_id}_data_mismatch")
                    
            except Exception as e:
                errors.append(f"worker_{worker_id}_error: {e}")
                
        # Start multiple worker threads
        threads = []
        for i in range(20):
            thread = threading.Thread(target=security_worker, args=(i,))
            threads.append(thread)
            thread.start()
            
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10.0)
            
        # Verify results
        assert len(results) == 20
        assert len(errors) == 0


@pytest.mark.performance
class TestSecurityPerformance:
    """Performance tests for security operations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.framework = QuantumSecurityFramework()
        
    def test_encryption_performance(self):
        """Test encryption performance with various data sizes."""
        
        data_sizes = [
            (32, 32),    # Small image
            (256, 256),  # Medium image
            (512, 512),  # Large image
        ]
        
        for height, width in data_sizes:
            test_data = np.random.rand(height, width).astype(np.float32)
            
            # Measure encryption time
            start_time = time.time()
            encrypted = self.framework.encrypt_medical_data(
                test_data, 
                SecurityLevel.PHI_PROTECTED
            )
            encryption_time = time.time() - start_time
            
            # Measure decryption time
            start_time = time.time()
            decrypted = self.framework.decrypt_medical_data(encrypted)
            decryption_time = time.time() - start_time
            
            # Performance assertions (should complete within reasonable time)
            assert encryption_time < 1.0  # Less than 1 second
            assert decryption_time < 1.0   # Less than 1 second
            assert np.array_equal(test_data, decrypted)
            
            print(f"Size {height}x{width}: Encrypt={encryption_time:.3f}s, Decrypt={decryption_time:.3f}s")
            
    def test_threat_detection_performance(self):
        """Test threat detection performance."""
        
        # Generate test requests of varying sizes
        test_requests = [
            {"simple": "request"},
            {"medium": "request", "data": "x" * 1000},
            {"large": "request", "data": "x" * 10000, "metadata": {"key": "value"}}
        ]
        
        for i, request in enumerate(test_requests):
            start_time = time.time()
            threat_level, threats = self.framework.threat_detector.detect_threats(request)
            detection_time = time.time() - start_time
            
            # Should complete quickly
            assert detection_time < 0.1  # Less than 100ms
            
            print(f"Request {i+1}: Detection time={detection_time:.4f}s, Threats={len(threats)}")


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v", "--tb=short", "-m", "not performance"])