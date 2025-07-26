"""Tests for dependency security scanning functionality."""

import pytest
import json
from unittest.mock import patch, MagicMock
import tempfile
import os

from src.dependency_security_scan import (
    ScanResult,
    VulnerabilityInfo,
    DependencySecurityScanner,
    scan_dependencies,
    check_tool_availability,
    parse_pip_audit_output,
    parse_safety_output,
    main,
)


class TestVulnerabilityInfo:
    """Test VulnerabilityInfo dataclass."""

    def test_vulnerability_info_creation(self):
        """Test creating VulnerabilityInfo with all fields."""
        vuln = VulnerabilityInfo(
            package="requests",
            version="2.25.0",
            vulnerability_id="CVE-2021-33503",
            severity="HIGH",
            description="Inefficient Regular Expression Complexity in requests",
            fixed_version="2.25.1"
        )
        
        assert vuln.package == "requests"
        assert vuln.version == "2.25.0"
        assert vuln.vulnerability_id == "CVE-2021-33503"
        assert vuln.severity == "HIGH"
        assert "Inefficient Regular Expression" in vuln.description
        assert vuln.fixed_version == "2.25.1"

    def test_vulnerability_info_to_dict(self):
        """Test converting VulnerabilityInfo to dictionary."""
        vuln = VulnerabilityInfo(
            package="numpy",
            version="1.19.0",
            vulnerability_id="PYSEC-2021-123",
            severity="MEDIUM",
            description="Buffer overflow in numpy",
            fixed_version="1.19.5"
        )
        
        vuln_dict = vuln.to_dict()
        assert vuln_dict["package"] == "numpy"
        assert vuln_dict["severity"] == "MEDIUM"


class TestScanResult:
    """Test ScanResult dataclass."""

    def test_scan_result_creation(self):
        """Test creating ScanResult with vulnerabilities."""
        vulns = [
            VulnerabilityInfo("requests", "2.25.0", "CVE-2021-33503", "HIGH", "Test", "2.25.1"),
            VulnerabilityInfo("numpy", "1.19.0", "PYSEC-2021-123", "MEDIUM", "Test", "1.19.5")
        ]
        
        result = ScanResult(
            scan_timestamp="2025-07-20T10:00:00Z",
            total_vulnerabilities=2,
            critical_count=0,
            high_count=1,
            medium_count=1,
            low_count=0,
            vulnerabilities=vulns,
            scan_tool="pip-audit",
            scan_duration=5.2
        )
        
        assert result.total_vulnerabilities == 2
        assert result.high_count == 1
        assert result.medium_count == 1
        assert len(result.vulnerabilities) == 2
        assert result.scan_tool == "pip-audit"

    def test_scan_result_no_vulnerabilities(self):
        """Test ScanResult with no vulnerabilities found."""
        result = ScanResult(
            scan_timestamp="2025-07-20T10:00:00Z",
            total_vulnerabilities=0,
            critical_count=0,
            high_count=0,
            medium_count=0,
            low_count=0,
            vulnerabilities=[],
            scan_tool="safety"
        )
        
        assert result.total_vulnerabilities == 0
        assert len(result.vulnerabilities) == 0
        assert result.is_clean

    def test_scan_result_to_json(self):
        """Test converting ScanResult to JSON."""
        vulns = [VulnerabilityInfo("test", "1.0.0", "TEST-001", "LOW", "Test vuln", "1.0.1")]
        result = ScanResult(
            scan_timestamp="2025-07-20T10:00:00Z",
            total_vulnerabilities=1,
            critical_count=0,
            high_count=0,
            medium_count=0,
            low_count=1,
            vulnerabilities=vulns,
            scan_tool="manual"
        )
        
        json_str = result.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["total_vulnerabilities"] == 1
        assert parsed["scan_tool"] == "manual"
        assert len(parsed["vulnerabilities"]) == 1


class TestCheckToolAvailability:
    """Test tool availability checking."""

    @patch('subprocess.run')
    def test_check_tool_available(self, mock_run):
        """Test checking when tool is available."""
        mock_run.return_value.returncode = 0
        
        result = check_tool_availability("pip-audit")
        
        assert result is True
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_check_tool_not_available(self, mock_run):
        """Test checking when tool is not available."""
        mock_run.return_value.returncode = 1
        
        result = check_tool_availability("nonexistent-tool")
        
        assert result is False

    @patch('subprocess.run')
    def test_check_tool_exception_handling(self, mock_run):
        """Test exception handling in tool availability check."""
        mock_run.side_effect = FileNotFoundError("Command not found")
        
        result = check_tool_availability("missing-tool")
        
        assert result is False


class TestParsePipAuditOutput:
    """Test pip-audit output parsing."""

    def test_parse_pip_audit_json_output(self):
        """Test parsing valid pip-audit JSON output."""
        pip_audit_output = """
        [
          {
            "package": "requests",
            "version": "2.25.0",
            "id": "PYSEC-2021-59",
            "fix_versions": ["2.25.1"],
            "description": "The package requests before 2.25.1 is vulnerable to Inefficient Regular Expression Complexity in requests/models.py."
          },
          {
            "package": "numpy",
            "version": "1.19.0", 
            "id": "PYSEC-2021-439",
            "fix_versions": ["1.19.5"],
            "description": "Buffer overflow in numpy"
          }
        ]
        """
        
        vulnerabilities = parse_pip_audit_output(pip_audit_output)
        
        assert len(vulnerabilities) == 2
        assert vulnerabilities[0].package == "requests"
        assert vulnerabilities[0].vulnerability_id == "PYSEC-2021-59"
        assert vulnerabilities[1].package == "numpy"

    def test_parse_pip_audit_empty_output(self):
        """Test parsing empty pip-audit output."""
        vulnerabilities = parse_pip_audit_output("[]")
        
        assert len(vulnerabilities) == 0

    def test_parse_pip_audit_invalid_json(self):
        """Test parsing invalid JSON from pip-audit."""
        vulnerabilities = parse_pip_audit_output("invalid json")
        
        assert len(vulnerabilities) == 0


class TestParseSafetyOutput:
    """Test safety output parsing."""

    def test_parse_safety_text_output(self):
        """Test parsing safety text output."""
        safety_output = """
        requests==2.25.0
        - 51457: The package requests before 2.25.1 is vulnerable to Inefficient Regular Expression
        
        numpy==1.19.0  
        - 44715: Buffer overflow vulnerability in numpy before 1.19.5
        """
        
        vulnerabilities = parse_safety_output(safety_output)
        
        assert len(vulnerabilities) == 2
        assert vulnerabilities[0].package == "requests"
        assert vulnerabilities[0].version == "2.25.0"
        assert vulnerabilities[1].package == "numpy"

    def test_parse_safety_no_vulnerabilities(self):
        """Test parsing safety output with no vulnerabilities."""
        safety_output = "No known security vulnerabilities found."
        
        vulnerabilities = parse_safety_output(safety_output)
        
        assert len(vulnerabilities) == 0

    def test_parse_safety_empty_output(self):
        """Test parsing empty safety output."""
        vulnerabilities = parse_safety_output("")
        
        assert len(vulnerabilities) == 0


class TestDependencySecurityScanner:
    """Test DependencySecurityScanner class."""

    def test_scanner_initialization(self):
        """Test scanner initialization."""
        scanner = DependencySecurityScanner()
        
        assert scanner.scan_results == []

    @patch('src.dependency_security_scan.check_tool_availability')
    @patch('subprocess.run')
    def test_scan_with_pip_audit(self, mock_run, mock_check_tool):
        """Test scanning with pip-audit tool."""
        mock_check_tool.side_effect = lambda tool: tool == "pip-audit"
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = """
        [
          {
            "package": "requests",
            "version": "2.25.0",
            "id": "PYSEC-2021-59",
            "fix_versions": ["2.25.1"],
            "description": "Test vulnerability"
          }
        ]
        """
        
        scanner = DependencySecurityScanner()
        result = scanner.scan()
        
        assert result.total_vulnerabilities == 1
        assert result.scan_tool == "pip-audit"
        assert len(result.vulnerabilities) == 1

    @patch('src.dependency_security_scan.check_tool_availability')
    @patch('subprocess.run')
    def test_scan_with_safety_fallback(self, mock_run, mock_check_tool):
        """Test scanning with safety as fallback."""
        mock_check_tool.side_effect = lambda tool: tool == "safety"
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = """
        requests==2.25.0
        - 51457: Test vulnerability description
        """
        
        scanner = DependencySecurityScanner()
        result = scanner.scan()
        
        assert result.total_vulnerabilities == 1
        assert result.scan_tool == "safety"

    @patch('src.dependency_security_scan.check_tool_availability')
    def test_scan_no_tools_available(self, mock_check_tool):
        """Test scanning when no security tools are available."""
        mock_check_tool.return_value = False
        
        scanner = DependencySecurityScanner()
        result = scanner.scan()
        
        assert result.total_vulnerabilities == 0
        assert result.scan_tool == "manual"
        assert "No security scanning tools available" in result.scan_notes

    @patch('src.dependency_security_scan.check_tool_availability')
    @patch('subprocess.run')
    def test_scan_tool_execution_failure(self, mock_run, mock_check_tool):
        """Test handling tool execution failure."""
        mock_check_tool.return_value = True
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "Tool execution failed"
        
        scanner = DependencySecurityScanner()
        result = scanner.scan()
        
        assert result.scan_tool in ["pip-audit", "safety"]
        assert result.total_vulnerabilities == 0

    def test_scan_with_custom_requirements_file(self):
        """Test scanning with custom requirements file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("requests==2.25.0\nnumpy==1.19.0\n")
            requirements_file = f.name
        
        try:
            scanner = DependencySecurityScanner()
            # This will use manual scanning since no tools are available
            result = scanner.scan(requirements_file=requirements_file)
            
            assert result.scan_tool == "manual"
        finally:
            os.unlink(requirements_file)


class TestScanDependencies:
    """Test high-level scan_dependencies function."""

    @patch('src.dependency_security_scan.DependencySecurityScanner')
    def test_scan_dependencies_default(self, mock_scanner_class):
        """Test scan_dependencies with default parameters."""
        mock_scanner = MagicMock()
        mock_result = ScanResult(
            scan_timestamp="2025-07-20T10:00:00Z",
            total_vulnerabilities=0,
            critical_count=0,
            high_count=0,
            medium_count=0,
            low_count=0,
            vulnerabilities=[],
            scan_tool="pip-audit"
        )
        mock_scanner.scan.return_value = mock_result
        mock_scanner_class.return_value = mock_scanner
        
        result = scan_dependencies()
        
        assert result.total_vulnerabilities == 0
        mock_scanner.scan.assert_called_once()

    @patch('src.dependency_security_scan.DependencySecurityScanner')
    def test_scan_dependencies_with_requirements_file(self, mock_scanner_class):
        """Test scan_dependencies with custom requirements file."""
        mock_scanner = MagicMock()
        mock_scanner.scan.return_value = MagicMock()
        mock_scanner_class.return_value = mock_scanner
        
        scan_dependencies(requirements_file="custom_requirements.txt")
        
        mock_scanner.scan.assert_called_once_with(
            requirements_file="custom_requirements.txt"
        )


class TestCLIInterface:
    """Test command-line interface for dependency security scanning."""

    @patch('src.dependency_security_scan.scan_dependencies')
    @patch('builtins.print')
    def test_cli_default_scan(self, mock_print, mock_scan):
        """Test CLI with default scan."""
        mock_result = ScanResult(
            scan_timestamp="2025-07-20T10:00:00Z",
            total_vulnerabilities=0,
            critical_count=0,
            high_count=0,
            medium_count=0,
            low_count=0,
            vulnerabilities=[],
            scan_tool="pip-audit"
        )
        mock_scan.return_value = mock_result
        
        
        with patch('sys.argv', ['security-scan']):
            main()
        
        mock_scan.assert_called_once()
        mock_print.assert_called()

    @patch('src.dependency_security_scan.scan_dependencies')
    def test_cli_with_json_output(self, mock_scan):
        """Test CLI with JSON output option."""
        mock_result = ScanResult(
            scan_timestamp="2025-07-20T10:00:00Z",
            total_vulnerabilities=1,
            critical_count=0,
            high_count=1,
            medium_count=0,
            low_count=0,
            vulnerabilities=[
                VulnerabilityInfo("test", "1.0.0", "TEST-001", "HIGH", "Test", "1.0.1")
            ],
            scan_tool="pip-audit"
        )
        mock_scan.return_value = mock_result
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = f.name
        
        try:
            from src.dependency_security_scan import main
            
            with patch('sys.argv', ['security-scan', '--output_json', output_file]):
                main()
            
            # Verify JSON file was created
            assert os.path.exists(output_file)
            with open(output_file, 'r') as f:
                data = json.load(f)
                assert data['total_vulnerabilities'] == 1
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)

    @patch('builtins.print')
    def test_cli_help(self, mock_print):
        """Test CLI help functionality."""
        
        with patch('sys.argv', ['security-scan', '--help']):
            with pytest.raises(SystemExit):
                main()


class TestSecurityConsiderations:
    """Test security aspects of the dependency scanner."""

    def test_no_shell_injection_in_tool_calls(self):
        """Test that tool calls are protected against shell injection."""
        scanner = DependencySecurityScanner()
        
        # Verify subprocess calls use list arguments, not shell=True
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1
            
            scanner._run_pip_audit()
            
            # Verify subprocess.run was called with list, not string
            call_args = mock_run.call_args[0][0]
            assert isinstance(call_args, list)
            assert call_args[0] in ['pip-audit', 'python3']

    def test_subprocess_calls_explicitly_disable_shell(self):
        """Test that all subprocess calls explicitly set shell=False for security."""
        scanner = DependencySecurityScanner()
        
        # Test _run_pip_audit
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "[]"
            
            scanner._run_pip_audit()
            
            # Verify shell=False is explicitly set
            call_kwargs = mock_run.call_args[1]
            assert 'shell' in call_kwargs
            assert call_kwargs['shell'] is False
        
        # Test check_tool_availability
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            
            check_tool_availability("pip-audit")
            
            # Verify shell=False is explicitly set
            call_kwargs = mock_run.call_args[1]
            assert 'shell' in call_kwargs
            assert call_kwargs['shell'] is False
        
        # Test pip-audit in scan_pip_audit
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "[]"
            
            scanner.scan_pip_audit()
            
            # Verify shell=False is explicitly set
            call_kwargs = mock_run.call_args[1]
            assert 'shell' in call_kwargs
            assert call_kwargs['shell'] is False
        
        # Test safety in scan_safety
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stdout = ""
            
            scanner.scan_safety()
            
            # Verify shell=False is explicitly set
            call_kwargs = mock_run.call_args[1]
            assert 'shell' in call_kwargs
            assert call_kwargs['shell'] is False

    def test_output_sanitization(self):
        """Test that scanner output is properly sanitized."""
        # Test with potentially malicious input
        malicious_output = """
        [
          {
            "package": "test; rm -rf /",
            "version": "1.0.0",
            "id": "$(malicious)",
            "fix_versions": ["`malicious`"],
            "description": "Malicious & dangerous <script>alert('xss')</script>"
          }
        ]
        """
        
        vulnerabilities = parse_pip_audit_output(malicious_output)
        
        assert len(vulnerabilities) == 1
        # Verify the malicious content is contained as data, not executed
        vuln = vulnerabilities[0]
        assert vuln.package == "test; rm -rf /"  # Stored as data
        assert vuln.vulnerability_id == "$(malicious)"  # Stored as data

    def test_file_path_validation(self):
        """Test that file paths are properly validated."""
        scanner = DependencySecurityScanner()
        
        # Test with potentially dangerous file paths
        dangerous_paths = [
            "../../../etc/passwd",
            "/etc/shadow",
            "file://dangerous",
            "http://malicious.com/requirements.txt"
        ]
        
        for path in dangerous_paths:
            result = scanner.scan(requirements_file=path)
            # Should handle gracefully without executing dangerous operations
            assert result.scan_tool == "manual"  # Falls back to safe mode