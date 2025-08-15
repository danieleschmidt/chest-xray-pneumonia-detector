#!/usr/bin/env python3
"""Tests for Medical Compliance Validator"""

import pytest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock, mock_open
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medical_compliance_validator import (
    MedicalComplianceValidator,
    ComplianceCheck,
    ErrorSeverity
)


class TestComplianceCheck:
    """Test ComplianceCheck dataclass"""
    
    def test_compliance_check_creation(self):
        """Test ComplianceCheck creation"""
        check = ComplianceCheck(
            name="Test Check",
            status="passed",
            score=0.85,
            details="Test details",
            recommendations=["Test recommendation"]
        )
        
        assert check.name == "Test Check"
        assert check.status == "passed"
        assert check.score == 0.85
        assert check.details == "Test details"
        assert check.recommendations == ["Test recommendation"]


class TestMedicalComplianceValidator:
    """Test suite for MedicalComplianceValidator"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.validator = MedicalComplianceValidator("src")
    
    @patch('pathlib.Path.rglob')
    @patch('pathlib.Path.read_text')
    def test_check_hipaa_compliance_high_score(self, mock_read_text, mock_rglob):
        """Test HIPAA compliance check with high compliance score"""
        # Mock file paths
        mock_files = [MagicMock() for _ in range(3)]
        mock_rglob.return_value = mock_files
        
        # Mock file content with HIPAA patterns
        mock_read_text.return_value = """
        This file contains encryption and authentication logic.
        We implement audit logging for all access.
        Data anonymization is performed before storage.
        SSL/TLS is used for secure transmission.
        We have hash verification for data integrity.
        """
        
        check = self.validator.check_hipaa_compliance()
        
        assert isinstance(check, ComplianceCheck)
        assert check.name == "HIPAA Compliance"
        assert check.score > 0.5  # Should find multiple patterns
        assert check.status in ["passed", "failed"]
        assert isinstance(check.recommendations, list)
    
    @patch('pathlib.Path.rglob')
    @patch('pathlib.Path.read_text')
    def test_check_hipaa_compliance_low_score(self, mock_read_text, mock_rglob):
        """Test HIPAA compliance check with low compliance score"""
        # Mock file paths
        mock_files = [MagicMock() for _ in range(3)]
        mock_rglob.return_value = mock_files
        
        # Mock file content without HIPAA patterns
        mock_read_text.return_value = "Basic application code without compliance features."
        
        check = self.validator.check_hipaa_compliance()
        
        assert check.name == "HIPAA Compliance"
        assert check.score < 0.7  # Should have low score
        assert check.status == "failed"
        assert len(check.recommendations) > 0
    
    @patch('pathlib.Path.rglob')
    @patch('pathlib.Path.read_text')
    @patch('pathlib.Path.exists')
    def test_check_fda_21cfr820_compliance(self, mock_exists, mock_read_text, mock_rglob):
        """Test FDA 21 CFR 820 compliance check"""
        # Mock docs directory exists
        mock_exists.return_value = True
        
        # Mock file paths
        mock_files = [MagicMock() for _ in range(5)]
        mock_rglob.return_value = mock_files
        
        # Mock file content with QSR patterns
        mock_read_text.return_value = """
        Design controls are implemented according to FDA guidelines.
        Risk management procedures are documented.
        Validation protocols are established.
        Version control and configuration management in place.
        """
        
        check = self.validator.check_fda_21cfr820_compliance()
        
        assert check.name == "FDA 21 CFR 820 Compliance"
        assert isinstance(check.score, float)
        assert check.status in ["passed", "warning", "failed"]
        assert isinstance(check.recommendations, list)
    
    @patch('pathlib.Path.rglob')
    @patch('pathlib.Path.read_text')
    def test_check_gdpr_compliance(self, mock_read_text, mock_rglob):
        """Test GDPR compliance check"""
        # Mock file paths
        mock_files = [MagicMock() for _ in range(3)]
        mock_rglob.return_value = mock_files
        
        # Mock file content with GDPR patterns
        mock_read_text.return_value = """
        Data protection and privacy by design implemented.
        User consent management system in place.
        Data deletion and right to be forgotten supported.
        Data portability features available.
        """
        
        check = self.validator.check_gdpr_compliance()
        
        assert check.name == "GDPR Compliance"
        assert isinstance(check.score, float)
        assert check.status in ["passed", "warning", "failed"]
        assert isinstance(check.recommendations, list)
    
    @patch('pathlib.Path.rglob')
    @patch('pathlib.Path.read_text')
    def test_check_ai_model_validation(self, mock_read_text, mock_rglob):
        """Test AI model validation check"""
        # Mock file paths
        mock_files = [MagicMock() for _ in range(3)]
        mock_rglob.return_value = mock_files
        
        # Mock file content with AI validation patterns
        mock_read_text.return_value = """
        Model testing and validation implemented.
        Accuracy, precision, recall metrics calculated.
        Bias testing for fairness.
        Robustness testing against adversarial inputs.
        Model explainability using grad-cam.
        Cross-validation implemented.
        """
        
        check = self.validator.check_ai_model_validation()
        
        assert check.name == "AI Model Validation"
        assert isinstance(check.score, float)
        assert check.status in ["passed", "warning", "failed"]
        assert isinstance(check.recommendations, list)
    
    @patch('pathlib.Path.rglob')
    @patch('pathlib.Path.read_text')
    @patch('pathlib.Path.exists')
    def test_check_medical_device_software(self, mock_exists, mock_read_text, mock_rglob):
        """Test medical device software compliance check"""
        # Mock docs directory exists
        mock_exists.return_value = True
        
        # Mock file paths from src and docs
        mock_files = [MagicMock() for _ in range(5)]
        mock_rglob.return_value = mock_files
        
        # Mock file content with medical device software patterns
        mock_read_text.return_value = """
        Software development lifecycle documented.
        Risk classification according to IEC 62304.
        Software verification and validation procedures.
        Configuration management implemented.
        """
        
        check = self.validator.check_medical_device_software()
        
        assert check.name == "Medical Device Software (IEC 62304)"
        assert isinstance(check.score, float)
        assert check.status in ["passed", "warning", "failed"]
        assert isinstance(check.recommendations, list)
    
    def test_get_compliance_grade(self):
        """Test compliance grade calculation"""
        validator = MedicalComplianceValidator()
        
        assert validator._get_compliance_grade(0.95) == "A"
        assert validator._get_compliance_grade(0.85) == "B"
        assert validator._get_compliance_grade(0.75) == "C"
        assert validator._get_compliance_grade(0.65) == "D"
        assert validator._get_compliance_grade(0.45) == "F"
    
    @patch.object(MedicalComplianceValidator, 'check_hipaa_compliance')
    @patch.object(MedicalComplianceValidator, 'check_fda_21cfr820_compliance')
    @patch.object(MedicalComplianceValidator, 'check_gdpr_compliance')
    @patch.object(MedicalComplianceValidator, 'check_ai_model_validation')
    @patch.object(MedicalComplianceValidator, 'check_medical_device_software')
    def test_generate_compliance_report_all_passed(self, mock_medical, mock_ai, mock_gdpr, mock_fda, mock_hipaa):
        """Test compliance report generation with all checks passed"""
        # Mock all checks to return passed status
        mock_checks = [
            ComplianceCheck("HIPAA", "passed", 0.9, "Details", []),
            ComplianceCheck("FDA", "passed", 0.8, "Details", []),
            ComplianceCheck("GDPR", "passed", 0.85, "Details", []),
            ComplianceCheck("AI Validation", "passed", 0.9, "Details", []),
            ComplianceCheck("Medical Device", "passed", 0.95, "Details", [])
        ]
        
        mock_hipaa.return_value = mock_checks[0]
        mock_fda.return_value = mock_checks[1]
        mock_gdpr.return_value = mock_checks[2]
        mock_ai.return_value = mock_checks[3]
        mock_medical.return_value = mock_checks[4]
        
        report = self.validator.generate_compliance_report()
        
        assert report["overall_status"] == "COMPLIANT"
        assert report["overall_score"] == 0.88  # Average of scores
        assert report["compliance_grade"] == "B"
        assert len(report["checks"]) == 5
        assert report["summary"]["passed"] == 5
        assert report["summary"]["failed"] == 0
        assert report["summary"]["warnings"] == 0
    
    @patch.object(MedicalComplianceValidator, 'check_hipaa_compliance')
    @patch.object(MedicalComplianceValidator, 'check_fda_21cfr820_compliance')
    @patch.object(MedicalComplianceValidator, 'check_gdpr_compliance')
    @patch.object(MedicalComplianceValidator, 'check_ai_model_validation')
    @patch.object(MedicalComplianceValidator, 'check_medical_device_software')
    def test_generate_compliance_report_with_failures(self, mock_medical, mock_ai, mock_gdpr, mock_fda, mock_hipaa):
        """Test compliance report generation with some failures"""
        # Mock checks with mixed results
        mock_checks = [
            ComplianceCheck("HIPAA", "failed", 0.4, "Details", ["Fix encryption"]),
            ComplianceCheck("FDA", "warning", 0.6, "Details", ["Add documentation"]),
            ComplianceCheck("GDPR", "failed", 0.3, "Details", ["Add consent management"]),
            ComplianceCheck("AI Validation", "passed", 0.9, "Details", []),
            ComplianceCheck("Medical Device", "passed", 0.8, "Details", [])
        ]
        
        mock_hipaa.return_value = mock_checks[0]
        mock_fda.return_value = mock_checks[1]
        mock_gdpr.return_value = mock_checks[2]
        mock_ai.return_value = mock_checks[3]
        mock_medical.return_value = mock_checks[4]
        
        report = self.validator.generate_compliance_report()
        
        assert report["overall_status"] == "CRITICAL"  # 2 failed checks
        assert report["summary"]["passed"] == 2
        assert report["summary"]["failed"] == 2
        assert report["summary"]["warnings"] == 1
        assert len(report["priority_recommendations"]) > 0
    
    def test_get_priority_recommendations(self):
        """Test priority recommendations generation"""
        validator = MedicalComplianceValidator()
        
        # Create mock checks with failures
        validator.checks = [
            ComplianceCheck("Test1", "failed", 0.3, "Details", ["Rec1", "Rec2", "Rec3"]),
            ComplianceCheck("Test2", "failed", 0.4, "Details", ["Rec4", "Rec5"]),
            ComplianceCheck("Test3", "passed", 0.9, "Details", [])
        ]
        
        recommendations = validator._get_priority_recommendations()
        
        # Should get top 2 from each failed check, max 5 total
        assert len(recommendations) <= 5
        assert "Rec1" in recommendations
        assert "Rec2" in recommendations
        assert "Rec4" in recommendations
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_report(self, mock_json_dump, mock_file):
        """Test saving compliance report to file"""
        validator = MedicalComplianceValidator()
        
        # Mock the generate_compliance_report method
        validator.generate_compliance_report = MagicMock(return_value={"test": "report"})
        
        validator.save_report("test_report.json")
        
        # Verify file operations
        mock_file.assert_called_once_with("test_report.json", 'w')
        mock_json_dump.assert_called_once_with({"test": "report"}, mock_file.return_value, indent=2)
    
    @patch('builtins.print')
    def test_print_summary(self, mock_print):
        """Test printing compliance summary"""
        validator = MedicalComplianceValidator()
        
        # Mock checks for summary
        validator.checks = [
            ComplianceCheck("Test1", "passed", 0.9, "Details", []),
            ComplianceCheck("Test2", "warning", 0.7, "Details", []),
            ComplianceCheck("Test3", "failed", 0.3, "Details", ["Fix this"])
        ]
        validator.overall_score = 0.8
        
        # Mock the generate_compliance_report method
        validator.generate_compliance_report = MagicMock(return_value={
            "overall_status": "REVIEW_REQUIRED",
            "compliance_grade": "B",
            "overall_score": 0.8,
            "priority_recommendations": ["Fix encryption", "Add documentation"],
            "summary": {"passed": 1, "warnings": 1, "failed": 1}
        })
        
        validator.print_summary()
        
        # Verify print was called (testing that it doesn't crash)
        assert mock_print.called


class TestComplianceValidatorErrorHandling:
    """Test error handling in compliance validator"""
    
    @patch('pathlib.Path.rglob')
    def test_file_read_error_handling(self, mock_rglob):
        """Test handling of file read errors"""
        # Mock file that raises exception on read
        mock_file = MagicMock()
        mock_file.read_text.side_effect = Exception("File read error")
        mock_rglob.return_value = [mock_file]
        
        validator = MedicalComplianceValidator()
        check = validator.check_hipaa_compliance()
        
        # Should not crash and return a valid check
        assert isinstance(check, ComplianceCheck)
        assert check.name == "HIPAA Compliance"
    
    def test_empty_source_directory(self):
        """Test handling of empty source directory"""
        validator = MedicalComplianceValidator("nonexistent_dir")
        check = validator.check_hipaa_compliance()
        
        # Should handle gracefully
        assert isinstance(check, ComplianceCheck)
        assert check.score == 0.0  # No files found


class TestComplianceValidatorIntegration:
    """Integration tests for compliance validator"""
    
    def test_full_compliance_report_generation(self):
        """Test full compliance report generation"""
        validator = MedicalComplianceValidator()
        report = validator.generate_compliance_report()
        
        # Verify report structure
        assert "timestamp" in report
        assert "overall_status" in report
        assert "overall_score" in report
        assert "compliance_grade" in report
        assert "checks" in report
        assert "summary" in report
        assert "priority_recommendations" in report
        
        # Verify all checks are present
        check_names = [check["name"] for check in report["checks"]]
        expected_checks = [
            "HIPAA Compliance",
            "FDA 21 CFR 820 Compliance", 
            "GDPR Compliance",
            "AI Model Validation",
            "Medical Device Software (IEC 62304)"
        ]
        
        for expected in expected_checks:
            assert expected in check_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])