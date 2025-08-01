"""
Tests for HIPAA-compliant audit logging functionality
Ensures comprehensive audit trail for medical AI operations
"""

import pytest
import json
import tempfile
from unittest.mock import patch, MagicMock
from src.monitoring.logging_config import HITPAAuditLogger, audit_phi_access


class TestHIPAAAuditLogging:
    """Test suite for HIPAA audit logging compliance."""
    
    def setup_method(self):
        """Setup test environment."""
        self.audit_logger = HITPAAuditLogger()
        
    def test_phi_access_logging(self):
        """Test PHI access is properly logged."""
        user_id = "test_clinician_001"
        action = "VIEW_PATIENT_XRAY"
        phi_data = {"patient_id": "P123", "study_date": "2025-01-15"}
        
        with patch.object(self.audit_logger.logger, 'info') as mock_info:
            self.audit_logger.log_phi_access(user_id, action, phi_data)
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args
            
            # Verify audit data structure
            assert call_args[1]['user_id'] == user_id
            assert call_args[1]['action'] == action
            assert call_args[1]['phi_involved'] == 'YES'
            assert call_args[1]['outcome'] == 'SUCCESS'
            
            # Verify audit data is JSON serializable
            audit_data = json.loads(call_args[1]['audit_data'])
            assert 'timestamp' in audit_data
            assert 'phi_data_hash' in audit_data
            assert audit_data['compliance_framework'] == 'HIPAA'
    
    def test_model_prediction_logging(self):
        """Test model predictions are properly audited."""
        user_id = "radiologist_001"
        model_version = "pneumonia_detector_v1.2"
        input_hash = "abc123def456"
        prediction = {"class": "pneumonia", "confidence": 0.92}
        
        with patch.object(self.audit_logger, 'log_phi_access') as mock_log:
            self.audit_logger.log_model_prediction(
                user_id, model_version, input_hash, prediction
            )
            
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            
            assert call_args[1]['user_id'] == user_id
            assert model_version in call_args[1]['action']
            assert call_args[1]['outcome'] == 'PREDICTION_GENERATED'
    
    def test_phi_data_hashing(self):
        """Test PHI data is properly hashed for audit logs."""
        phi_data = {"patient_id": "P123", "age": 45}
        
        hash1 = self.audit_logger._hash_phi_data(phi_data)
        hash2 = self.audit_logger._hash_phi_data(phi_data)
        
        # Same data should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 16  # Truncated SHA256
        
        # Different data should produce different hash
        phi_data2 = {"patient_id": "P124", "age": 45}
        hash3 = self.audit_logger._hash_phi_data(phi_data2)
        assert hash1 != hash3
    
    def test_audit_decorator(self):
        """Test audit decorator properly logs function calls."""
        @audit_phi_access("TEST_ACTION")
        def test_function(data, user_id="test_user"):
            return {"result": "success"}
        
        with patch.object(HITPAAuditLogger, 'log_phi_access') as mock_log:
            result = test_function({"test": "data"}, user_id="clinician_001")
            
            assert result == {"result": "success"}
            mock_log.assert_called()
    
    def test_audit_decorator_error_handling(self):
        """Test audit decorator logs errors properly."""
        @audit_phi_access("ERROR_TEST")
        def failing_function(user_id="test_user"):
            raise ValueError("Test error")
        
        with patch.object(HITPAAuditLogger, 'log_phi_access') as mock_log:
            with pytest.raises(ValueError):
                failing_function(user_id="clinician_001")
            
            # Should log the failure
            mock_log.assert_called()
            call_args = mock_log.call_args
            assert "FAILED" in call_args[1]['outcome']


class TestAuditCompliance:
    """Test HIPAA compliance requirements."""
    
    def test_audit_log_format_compliance(self):
        """Test audit log format meets HIPAA requirements."""
        # HIPAA requires: who, what, when, where accessed PHI
        audit_logger = HITPAAuditLogger()
        
        with patch.object(audit_logger.logger, 'info') as mock_info:
            audit_logger.log_phi_access(
                user_id="clinician_001",
                action="VIEW_PATIENT_DATA", 
                phi_data={"patient": "P123"},
                outcome="SUCCESS"
            )
            
            call_args = mock_info.call_args
            
            # WHO: user_id is present
            assert 'user_id' in call_args[1]
            
            # WHAT: action is present  
            assert 'action' in call_args[1]
            
            # WHEN: timestamp in audit_data
            audit_data = json.loads(call_args[1]['audit_data'])
            assert 'timestamp' in audit_data
            
            # OUTCOME: success/failure recorded
            assert 'outcome' in call_args[1]
    
    def test_minimum_retention_period(self):
        """Test audit logs are configured for HIPAA retention requirements."""
        # HIPAA requires minimum 6 years retention
        retention_policy_path = Path("logs/retention-policy.txt")
        
        if retention_policy_path.exists():
            content = retention_policy_path.read_text()
            assert "LOG_RETENTION_YEARS = 6" in content
