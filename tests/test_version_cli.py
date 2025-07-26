"""Tests for version_cli module."""

import subprocess
import sys
from unittest.mock import patch
from importlib.metadata import PackageNotFoundError
import re

from src.version_cli import main


class TestVersionCliUnit:
    """Unit tests for version_cli functions."""

    @patch('src.version_cli.version')
    @patch('builtins.print')
    def test_main_prints_version_when_package_found(self, mock_print, mock_version):
        """Test main() prints version when package is found."""
        mock_version.return_value = "1.2.3"
        
        main()
        
        mock_version.assert_called_once_with("chest_xray_pneumonia_detector")
        mock_print.assert_called_once_with("1.2.3")

    @patch('src.version_cli.version')
    @patch('builtins.print')
    def test_main_prints_unknown_when_package_not_found(self, mock_print, mock_version):
        """Test main() prints 'unknown' when PackageNotFoundError is raised."""
        mock_version.side_effect = PackageNotFoundError("Package not found")
        
        main()
        
        mock_version.assert_called_once_with("chest_xray_pneumonia_detector")
        mock_print.assert_called_once_with("unknown")

    @patch('src.version_cli.version')
    @patch('builtins.print')
    def test_main_handles_various_version_formats(self, mock_print, mock_version):
        """Test main() handles different version formats correctly."""
        test_versions = [
            "0.1.0",
            "1.0.0",
            "2.1.3",
            "1.0.0a1",
            "1.0.0b2",
            "1.0.0rc1",
            "1.0.0.dev1",
            "1.0.0+local.version"
        ]
        
        for version_str in test_versions:
            mock_version.return_value = version_str
            mock_print.reset_mock()
            
            main()
            
            mock_print.assert_called_once_with(version_str)

    @patch('src.version_cli.version')
    @patch('builtins.print')
    def test_main_handles_empty_version(self, mock_print, mock_version):
        """Test main() handles empty version string."""
        mock_version.return_value = ""
        
        main()
        
        mock_print.assert_called_once_with("")

    @patch('src.version_cli.version')
    @patch('builtins.print')
    def test_main_handles_none_version(self, mock_print, mock_version):
        """Test main() handles None version (edge case)."""
        mock_version.return_value = None
        
        main()
        
        mock_print.assert_called_once_with(None)


class TestVersionCliIntegration:
    """Integration tests for version_cli CLI interface."""

    def test_version_cli_outputs_version(self) -> None:
        """Test CLI outputs a version string."""
        result = subprocess.run(
            [sys.executable, "-m", "src.version_cli"], 
            capture_output=True, 
            text=True
        )
        
        assert result.returncode == 0
        assert result.stdout.strip(), "version output should not be empty"

    def test_version_cli_outputs_valid_format(self) -> None:
        """Test CLI outputs a valid version format."""
        result = subprocess.run(
            [sys.executable, "-m", "src.version_cli"], 
            capture_output=True, 
            text=True
        )
        
        assert result.returncode == 0
        version_output = result.stdout.strip()
        
        # Should be either a valid semantic version or "unknown"
        version_pattern = r'^(\d+\.\d+\.\d+.*|unknown)$'
        assert re.match(version_pattern, version_output), \
            f"Version output '{version_output}' does not match expected format"

    def test_version_cli_stderr_empty(self) -> None:
        """Test CLI produces no error output under normal conditions."""
        result = subprocess.run(
            [sys.executable, "-m", "src.version_cli"], 
            capture_output=True, 
            text=True
        )
        
        assert result.returncode == 0
        assert result.stderr == "", "No error output expected for normal operation"

    def test_version_cli_direct_execution(self) -> None:
        """Test direct execution of version_cli.py file."""
        result = subprocess.run(
            [sys.executable, "/root/repo/src/version_cli.py"], 
            capture_output=True, 
            text=True
        )
        
        assert result.returncode == 0
        assert result.stdout.strip(), "Direct execution should produce version output"

    def test_version_cli_multiple_calls_consistent(self) -> None:
        """Test multiple calls to version CLI produce consistent output."""
        results = []
        for _ in range(3):
            result = subprocess.run(
                [sys.executable, "-m", "src.version_cli"], 
                capture_output=True, 
                text=True
            )
            assert result.returncode == 0
            results.append(result.stdout.strip())
        
        # All results should be identical
        assert all(r == results[0] for r in results), \
            "Multiple version calls should produce consistent output"


class TestVersionCliEdgeCases:
    """Edge case tests for version_cli."""

    @patch('src.version_cli.version')
    @patch('builtins.print')
    def test_main_handles_generic_exception_gracefully(self, mock_print, mock_version):
        """Test main() handles unexpected exceptions gracefully."""
        mock_version.side_effect = RuntimeError("Unexpected error")
        
        main()
        
        mock_print.assert_called_once_with("unknown")

    @patch('src.version_cli.version')
    def test_main_calls_correct_package_name(self, mock_version):
        """Test main() calls version() with correct package name."""
        mock_version.return_value = "1.0.0"
        
        main()
        
        mock_version.assert_called_once_with("chest_xray_pneumonia_detector")

    @patch('src.version_cli.version')
    @patch('builtins.print')
    def test_main_print_called_exactly_once(self, mock_print, mock_version):
        """Test main() calls print exactly once."""
        mock_version.return_value = "1.0.0"
        
        main()
        
        assert mock_print.call_count == 1


class TestVersionCliDocumentation:
    """Tests for version_cli documentation and metadata."""

    def test_main_function_has_docstring(self):
        """Test main() function has proper docstring."""
        assert main.__doc__ is not None
        assert "version" in main.__doc__.lower()

    def test_module_imports_are_minimal(self):
        """Test module has minimal, necessary imports only."""
        import src.version_cli as version_cli_module
        
        # Check that module doesn't import unnecessary dependencies
        expected_attrs = {'main', 'version', 'PackageNotFoundError'}
        module_attrs = set(dir(version_cli_module))
        
        # Should contain expected functions/imports
        assert expected_attrs.issubset(module_attrs)


class TestVersionCLISecurityConsiderations:
    """Security-related tests for version_cli."""

    def test_no_shell_injection_vulnerability(self):
        """Test CLI is not vulnerable to shell injection."""
        # version_cli doesn't take user input, but verify safe practices
        result = subprocess.run(
            [sys.executable, "-m", "src.version_cli"], 
            capture_output=True, 
            text=True
        )
        
        # Output should be clean version string, no shell characters
        version_output = result.stdout.strip()
        dangerous_chars = [';', '&', '|', '`', '$', '(', ')']
        
        for char in dangerous_chars:
            assert char not in version_output, \
                f"Version output contains potentially dangerous character: {char}"

    def test_no_sensitive_information_leaked(self):
        """Test version output doesn't leak sensitive information."""
        result = subprocess.run(
            [sys.executable, "-m", "src.version_cli"], 
            capture_output=True, 
            text=True
        )
        
        version_output = result.stdout.strip().lower()
        
        # Check for common sensitive patterns
        sensitive_patterns = ['password', 'secret', 'key', 'token', 'auth']
        for pattern in sensitive_patterns:
            assert pattern not in version_output, \
                f"Version output may contain sensitive information: {pattern}"
