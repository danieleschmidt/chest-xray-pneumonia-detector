#!/usr/bin/env python3
"""Tests for Enhanced Quality Gates System"""

import pytest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock
import sys
import subprocess

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_quality_gates import QualityGateRunner


class TestQualityGateRunner:
    """Test suite for QualityGateRunner"""
    
    def test_quality_gate_runner_initialization(self):
        """Test QualityGateRunner initialization"""
        runner = QualityGateRunner()
        
        assert runner.results == {}
        assert runner.passed_gates == []
        assert runner.failed_gates == []
    
    @patch('subprocess.run')
    def test_run_security_scan_success(self, mock_run):
        """Test successful security scan"""
        # Mock successful bandit run
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        
        runner = QualityGateRunner()
        success, result = runner.run_security_scan()
        
        assert success is True
        assert result["status"] == "passed"
        assert "security_scan" in runner.passed_gates
    
    @patch('subprocess.run')
    def test_run_security_scan_failure(self, mock_run):
        """Test failed security scan"""
        # Mock failed bandit run
        mock_run.return_value = MagicMock(returncode=1, stdout="Security issues found", stderr="")
        
        runner = QualityGateRunner()
        success, result = runner.run_security_scan()
        
        assert success is False
        assert result["status"] == "failed"
        assert "security_scan" in runner.failed_gates
    
    @patch('subprocess.run')
    def test_run_code_quality_check_success(self, mock_run):
        """Test successful code quality check"""
        # Mock successful ruff run
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        
        runner = QualityGateRunner()
        success, result = runner.run_code_quality_check()
        
        assert success is True
        assert result["status"] == "passed"
        assert "code_quality" in runner.passed_gates
    
    @patch('subprocess.run')
    def test_run_code_quality_check_failure(self, mock_run):
        """Test failed code quality check"""
        # Mock failed ruff run
        mock_run.return_value = MagicMock(returncode=1, stdout="Code quality issues", stderr="")
        
        runner = QualityGateRunner()
        success, result = runner.run_code_quality_check()
        
        assert success is False
        assert result["status"] == "failed"
        assert "code_quality" in runner.failed_gates
    
    @patch('subprocess.run')
    @patch('builtins.open')
    @patch('json.load')
    def test_run_test_suite_success_with_coverage(self, mock_json_load, mock_open, mock_run):
        """Test successful test suite with adequate coverage"""
        # Mock successful pytest run
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        
        # Mock coverage file
        mock_json_load.return_value = {"totals": {"percent_covered": 90.0}}
        
        runner = QualityGateRunner()
        success, result = runner.run_test_suite()
        
        assert success is True
        assert result["status"] == "passed"
        assert result["coverage"] == 90.0
        assert "test_suite" in runner.passed_gates
    
    @patch('subprocess.run')
    @patch('builtins.open')
    @patch('json.load')
    def test_run_test_suite_low_coverage(self, mock_json_load, mock_open, mock_run):
        """Test test suite with low coverage"""
        # Mock successful pytest run
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        
        # Mock coverage file with low coverage
        mock_json_load.return_value = {"totals": {"percent_covered": 70.0}}
        
        runner = QualityGateRunner()
        success, result = runner.run_test_suite()
        
        assert success is False
        assert result["status"] == "failed"
        assert result["coverage"] == 70.0
        assert "test_suite" in runner.failed_gates
    
    def test_run_medical_ai_validation_success(self):
        """Test successful medical AI validation"""
        runner = QualityGateRunner()
        success, result = runner.run_medical_ai_validation()
        
        # Should pass if enough patterns are found
        assert isinstance(success, bool)
        assert "status" in result
        assert "compliance_score" in result
        assert "found_patterns" in result
    
    def test_run_performance_benchmark(self):
        """Test performance benchmark"""
        runner = QualityGateRunner()
        success, result = runner.run_performance_benchmark()
        
        assert isinstance(success, bool)
        assert "status" in result
        assert "total_time" in result
        assert "import_times" in result
    
    @patch('subprocess.run')
    @patch('json.loads')
    def test_run_dependency_audit_success(self, mock_json_loads, mock_run):
        """Test successful dependency audit"""
        # Mock successful pip list run
        mock_run.return_value = MagicMock(returncode=0, stdout='[{"name": "tensorflow", "version": "2.17.0"}]', stderr="")
        mock_json_loads.return_value = [{"name": "tensorflow", "version": "2.17.0"}]
        
        runner = QualityGateRunner()
        success, result = runner.run_dependency_audit()
        
        assert success is True
        assert result["status"] == "passed"
        assert "dependency_audit" in runner.passed_gates
    
    def test_generate_report(self):
        """Test report generation"""
        runner = QualityGateRunner()
        
        # Simulate some passed and failed gates
        runner.passed_gates = ["security_scan", "code_quality"]
        runner.failed_gates = ["test_suite"]
        runner.results = {
            "security_scan": {"status": "passed"},
            "code_quality": {"status": "passed"},
            "test_suite": {"status": "failed"}
        }
        
        report = runner.generate_report()
        
        assert "timestamp" in report
        assert "overall_status" in report
        assert "success_rate" in report
        assert report["total_gates"] == 3
        assert report["passed_gates"] == ["security_scan", "code_quality"]
        assert report["failed_gates"] == ["test_suite"]
        assert "detailed_results" in report
        assert "recommendations" in report
    
    def test_generate_recommendations(self):
        """Test recommendation generation"""
        runner = QualityGateRunner()
        
        # Simulate failed gates
        runner.failed_gates = ["security_scan", "code_quality", "test_suite"]
        
        recommendations = runner._generate_recommendations()
        
        assert len(recommendations) > 0
        assert any("security" in rec.lower() for rec in recommendations)
        assert any("code quality" in rec.lower() for rec in recommendations)
        assert any("test coverage" in rec.lower() or "coverage" in rec.lower() for rec in recommendations)
    
    @patch('json.dump')
    @patch('builtins.open')
    def test_run_all_gates_saves_report(self, mock_open, mock_json_dump):
        """Test that run_all_gates saves a report"""
        runner = QualityGateRunner()
        
        # Mock all gate methods to return success
        runner.run_security_scan = MagicMock(return_value=(True, {"status": "passed"}))
        runner.run_code_quality_check = MagicMock(return_value=(True, {"status": "passed"}))
        runner.run_test_suite = MagicMock(return_value=(True, {"status": "passed"}))
        runner.run_medical_ai_validation = MagicMock(return_value=(True, {"status": "passed"}))
        runner.run_performance_benchmark = MagicMock(return_value=(True, {"status": "passed"}))
        runner.run_dependency_audit = MagicMock(return_value=(True, {"status": "passed"}))
        
        report = runner.run_all_gates()
        
        assert report["overall_status"] == "PASSED"
        assert report["success_rate"] == 1.0
        
        # Verify that the report was saved
        mock_open.assert_called_with("enhanced_quality_gate_report.json", "w")
        mock_json_dump.assert_called_once()


class TestQualityGateIntegration:
    """Integration tests for quality gates"""
    
    def test_medical_ai_patterns_detection(self):
        """Test that medical AI patterns are correctly detected"""
        runner = QualityGateRunner()
        success, result = runner.run_medical_ai_validation()
        
        # Should find patterns in the codebase
        assert "compliance_score" in result
        assert result["compliance_score"] >= 0
        assert "found_patterns" in result
        assert isinstance(result["found_patterns"], list)
    
    def test_import_performance_benchmark(self):
        """Test that performance benchmark measures import times"""
        runner = QualityGateRunner()
        success, result = runner.run_performance_benchmark()
        
        assert "total_time" in result
        assert "import_times" in result
        assert result["total_time"] > 0
        assert isinstance(result["import_times"], dict)


class TestQualityGateErrorHandling:
    """Test error handling in quality gates"""
    
    @patch('subprocess.run')
    def test_security_scan_timeout(self, mock_run):
        """Test security scan timeout handling"""
        # Mock timeout exception
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["bandit"], timeout=300)
        
        runner = QualityGateRunner()
        success, result = runner.run_security_scan()
        
        assert success is False
        assert result["status"] == "error"
        assert "security_scan" in runner.failed_gates
    
    @patch('subprocess.run')
    def test_code_quality_exception(self, mock_run):
        """Test code quality check exception handling"""
        # Mock exception
        mock_run.side_effect = Exception("Ruff execution failed")
        
        runner = QualityGateRunner()
        success, result = runner.run_code_quality_check()
        
        assert success is False
        assert result["status"] == "error"
        assert "code_quality" in runner.failed_gates
    
    def test_performance_benchmark_exception_handling(self):
        """Test performance benchmark handles import errors gracefully"""
        runner = QualityGateRunner()
        
        # This should not raise an exception even if some imports fail
        success, result = runner.run_performance_benchmark()
        
        assert isinstance(success, bool)
        assert "status" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])