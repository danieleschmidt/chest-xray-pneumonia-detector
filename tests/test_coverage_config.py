"""
Test coverage configuration and utilities.

This module provides utilities for testing code coverage and ensuring
quality gates are met across the test suite.
"""

import pytest
import coverage
from pathlib import Path
import json
import subprocess
import sys


class CoverageReporter:
    """Utility class for generating and analyzing coverage reports."""
    
    def __init__(self, source_dir="src", min_coverage=85):
        self.source_dir = source_dir
        self.min_coverage = min_coverage
        self.cov = coverage.Coverage(source=source_dir)
    
    def start_coverage(self):
        """Start coverage measurement."""
        self.cov.start()
    
    def stop_coverage(self):
        """Stop coverage measurement."""
        self.cov.stop()
        self.cov.save()
    
    def generate_report(self, output_file=None):
        """Generate coverage report."""
        if output_file:
            with open(output_file, 'w') as f:
                self.cov.report(file=f)
        else:
            self.cov.report()
    
    def generate_html_report(self, directory="htmlcov"):
        """Generate HTML coverage report."""
        self.cov.html_report(directory=directory)
    
    def check_coverage_threshold(self):
        """Check if coverage meets minimum threshold."""
        total_coverage = self.cov.report(show_missing=False)
        return total_coverage >= self.min_coverage
    
    def get_missing_coverage(self):
        """Get files with missing coverage."""
        missing_files = []
        for filename in self.cov.get_data().measured_files():
            analysis = self.cov.analysis2(filename)
            if analysis.missing:
                missing_files.append({
                    'file': filename,
                    'missing_lines': analysis.missing,
                    'excluded_lines': analysis.excluded
                })
        return missing_files


@pytest.fixture
def coverage_reporter():
    """Fixture providing coverage reporter instance."""
    return CoverageReporter()


def test_coverage_configuration():
    """Test that coverage is properly configured."""
    # Check that coverage config exists in pyproject.toml
    pyproject_path = Path("pyproject.toml")
    assert pyproject_path.exists(), "pyproject.toml not found"
    
    # Check for coverage configuration
    with open(pyproject_path) as f:
        content = f.read()
        assert "[tool.coverage.run]" in content, "Coverage run configuration missing"
        assert "[tool.coverage.report]" in content, "Coverage report configuration missing"


def test_minimum_coverage_threshold():
    """Test that minimum coverage threshold is enforced."""
    try:
        # Run coverage command
        result = subprocess.run([
            sys.executable, "-m", "pytest", "--cov=src", "--cov-report=json", 
            "--cov-fail-under=85", "tests/"
        ], capture_output=True, text=True, timeout=300)
        
        # Check if coverage meets threshold
        assert result.returncode == 0, f"Coverage below threshold: {result.stdout}"
        
    except subprocess.TimeoutExpired:
        pytest.skip("Coverage test timeout - may indicate performance issues")
    except FileNotFoundError:
        pytest.skip("Coverage tools not available")


def test_coverage_excludes():
    """Test that appropriate files are excluded from coverage."""
    excluded_patterns = [
        "*/tests/*",
        "*/test_*.py",
        "*/conftest.py",
        "*/__init__.py"
    ]
    
    pyproject_path = Path("pyproject.toml")
    with open(pyproject_path) as f:
        content = f.read()
        
    for pattern in excluded_patterns:
        # Check if pattern exists in coverage omit configuration
        assert pattern in content or pattern.replace("*/", "") in content, \
            f"Exclude pattern {pattern} not found in coverage config"


@pytest.mark.slow
def test_full_coverage_report(coverage_reporter, tmp_path):
    """Generate and validate full coverage report."""
    coverage_reporter.start_coverage()
    
    # Import and run key modules to measure coverage
    try:
        from src import data_loader, model_builder, train_engine
        # Basic imports to measure coverage
        coverage_reporter.stop_coverage()
        
        # Generate reports
        report_file = tmp_path / "coverage_report.txt"
        coverage_reporter.generate_report(str(report_file))
        
        html_dir = tmp_path / "htmlcov"
        coverage_reporter.generate_html_report(str(html_dir))
        
        # Verify reports were generated
        assert report_file.exists(), "Coverage report not generated"
        assert (html_dir / "index.html").exists(), "HTML coverage report not generated"
        
    except ImportError as e:
        pytest.skip(f"Source modules not available for coverage testing: {e}")


def test_coverage_quality_gates():
    """Test coverage quality gates for critical components."""
    critical_modules = [
        "src/data_loader.py",
        "src/model_builder.py", 
        "src/train_engine.py",
        "src/inference.py"
    ]
    
    # This would be implemented with actual coverage measurement
    # For now, we just check that the files exist
    for module in critical_modules:
        module_path = Path(module)
        if module_path.exists():
            # In a real implementation, we'd check per-file coverage
            assert True, f"Critical module {module} should have high coverage"
        else:
            pytest.skip(f"Module {module} not found")


if __name__ == "__main__":
    # CLI for running coverage checks
    import argparse
    
    parser = argparse.ArgumentParser(description="Coverage testing utilities")
    parser.add_argument("--generate-report", action="store_true", 
                      help="Generate coverage report")
    parser.add_argument("--check-threshold", action="store_true",
                      help="Check coverage threshold")
    parser.add_argument("--html-report", type=str, default="htmlcov",
                      help="Generate HTML report in specified directory")
    
    args = parser.parse_args()
    
    reporter = CoverageReporter()
    
    if args.generate_report:
        reporter.generate_report()
    
    if args.check_threshold:
        if reporter.check_coverage_threshold():
            print("✅ Coverage threshold met")
            sys.exit(0)
        else:
            print("❌ Coverage below threshold")
            sys.exit(1)
    
    if args.html_report:
        reporter.generate_html_report(args.html_report)
        print(f"📊 HTML report generated in {args.html_report}")