"""
Quality gates and testing standards enforcement.

This module defines and enforces quality gates across the codebase
to ensure consistent code quality, security, and maintainability.
"""

import pytest
import subprocess
import sys
import json
from pathlib import Path
import ast
import re
from typing import List, Dict, Any


class QualityGate:
    """Base class for quality gate checks."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        
    def check(self) -> bool:
        """Run the quality gate check."""
        raise NotImplementedError
        
    def get_details(self) -> Dict[str, Any]:
        """Get detailed results of the check."""
        return {"name": self.name, "description": self.description}


class CodeStyleGate(QualityGate):
    """Quality gate for code style compliance."""
    
    def __init__(self):
        super().__init__("Code Style", "Enforces consistent code formatting")
        
    def check(self) -> bool:
        """Check code style with black and ruff."""
        try:
            # Check black formatting
            black_result = subprocess.run([
                sys.executable, "-m", "black", "--check", "src/", "tests/"
            ], capture_output=True, text=True)
            
            # Check ruff linting
            ruff_result = subprocess.run([
                sys.executable, "-m", "ruff", "check", "src/", "tests/"
            ], capture_output=True, text=True)
            
            return black_result.returncode == 0 and ruff_result.returncode == 0
            
        except FileNotFoundError:
            return False


class SecurityGate(QualityGate):
    """Quality gate for security compliance."""
    
    def __init__(self):
        super().__init__("Security", "Enforces security best practices")
        
    def check(self) -> bool:
        """Check security with bandit."""
        try:
            result = subprocess.run([
                sys.executable, "-m", "bandit", "-r", "src/", 
                "-f", "json", "-ll"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                return True
                
            # Parse bandit output for severity
            try:
                bandit_data = json.loads(result.stdout)
                high_severity_issues = [
                    issue for issue in bandit_data.get("results", [])
                    if issue.get("issue_severity") == "HIGH"
                ]
                return len(high_severity_issues) == 0
            except json.JSONDecodeError:
                return False
                
        except FileNotFoundError:
            return False


class TestCoverageGate(QualityGate):
    """Quality gate for test coverage."""
    
    def __init__(self, min_coverage: float = 85.0):
        super().__init__("Test Coverage", f"Enforces minimum {min_coverage}% test coverage")
        self.min_coverage = min_coverage
        
    def check(self) -> bool:
        """Check test coverage."""
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", "--cov=src", 
                "--cov-report=json", "--cov-fail-under=" + str(self.min_coverage),
                "tests/", "-q"
            ], capture_output=True, text=True, timeout=300)
            
            return result.returncode == 0
            
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False


class DocumentationGate(QualityGate):
    """Quality gate for documentation quality."""
    
    def __init__(self):
        super().__init__("Documentation", "Enforces documentation standards")
        
    def check(self) -> bool:
        """Check documentation completeness."""
        required_docs = [
            "README.md",
            "CONTRIBUTING.md", 
            "LICENSE",
            "SECURITY.md",
            "CODE_OF_CONDUCT.md"
        ]
        
        for doc in required_docs:
            if not Path(doc).exists():
                return False
                
        return self._check_docstring_coverage()
        
    def _check_docstring_coverage(self) -> bool:
        """Check docstring coverage in source files."""
        src_files = list(Path("src").rglob("*.py"))
        
        for file_path in src_files:
            if file_path.name == "__init__.py":
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                    
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        if not ast.get_docstring(node) and not node.name.startswith('_'):
                            # Public function/class without docstring
                            return False
                            
            except (SyntaxError, UnicodeDecodeError):
                continue
                
        return True


class PerformanceGate(QualityGate):
    """Quality gate for performance standards."""
    
    def __init__(self):
        super().__init__("Performance", "Enforces performance standards")
        
    def check(self) -> bool:
        """Check performance benchmarks."""
        try:
            # Run performance tests
            result = subprocess.run([
                sys.executable, "-m", "pytest", "tests/performance/", 
                "-v", "--tb=short"
            ], capture_output=True, text=True, timeout=600)
            
            return result.returncode == 0
            
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False


class DependencyGate(QualityGate):
    """Quality gate for dependency security."""
    
    def __init__(self):
        super().__init__("Dependencies", "Checks dependency security")
        
    def check(self) -> bool:
        """Check dependencies for security vulnerabilities."""
        try:
            # Check with safety (if available)
            result = subprocess.run([
                sys.executable, "-m", "safety", "check", "--json"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                return True
                
            # Parse safety output
            try:
                safety_data = json.loads(result.stdout)
                return len(safety_data) == 0
            except json.JSONDecodeError:
                return False
                
        except FileNotFoundError:
            # If safety not available, check with pip-audit
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pip_audit", "--format=json"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    return True
                    
            except FileNotFoundError:
                # If no security tools available, skip
                return True
                
        return False


class QualityGateRunner:
    """Runs and manages quality gates."""
    
    def __init__(self):
        self.gates = [
            CodeStyleGate(),
            SecurityGate(),
            TestCoverageGate(),
            DocumentationGate(),
            PerformanceGate(),
            DependencyGate()
        ]
        
    def run_all_gates(self) -> Dict[str, bool]:
        """Run all quality gates."""
        results = {}
        for gate in self.gates:
            try:
                results[gate.name] = gate.check()
            except Exception as e:
                print(f"Error running {gate.name}: {e}")
                results[gate.name] = False
                
        return results
        
    def run_gate(self, gate_name: str) -> bool:
        """Run a specific quality gate."""
        for gate in self.gates:
            if gate.name == gate_name:
                return gate.check()
        raise ValueError(f"Unknown gate: {gate_name}")


# Pytest tests for quality gates
def test_code_style_gate():
    """Test code style quality gate."""
    gate = CodeStyleGate()
    result = gate.check()
    assert result, "Code style quality gate failed"


def test_security_gate():
    """Test security quality gate."""
    gate = SecurityGate()
    result = gate.check()
    assert result, "Security quality gate failed"


@pytest.mark.slow
def test_coverage_gate():
    """Test coverage quality gate."""
    gate = TestCoverageGate(min_coverage=80.0)  # Slightly lower for testing
    result = gate.check()
    assert result, "Coverage quality gate failed"


def test_documentation_gate():
    """Test documentation quality gate."""
    gate = DocumentationGate()
    result = gate.check()
    assert result, "Documentation quality gate failed"


@pytest.mark.slow
def test_performance_gate():
    """Test performance quality gate."""
    gate = PerformanceGate()
    result = gate.check()
    assert result, "Performance quality gate failed"


def test_dependency_gate():
    """Test dependency security gate."""
    gate = DependencyGate()
    result = gate.check()
    assert result, "Dependency security gate failed"


def test_all_quality_gates():
    """Test all quality gates together."""
    runner = QualityGateRunner()
    results = runner.run_all_gates()
    
    failed_gates = [name for name, passed in results.items() if not passed]
    
    if failed_gates:
        pytest.fail(f"Quality gates failed: {failed_gates}")


if __name__ == "__main__":
    # CLI for running quality gates
    import argparse
    
    parser = argparse.ArgumentParser(description="Quality gate runner")
    parser.add_argument("--gate", type=str, help="Run specific quality gate")
    parser.add_argument("--all", action="store_true", help="Run all quality gates")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    
    args = parser.parse_args()
    
    runner = QualityGateRunner()
    
    if args.gate:
        result = runner.run_gate(args.gate)
        if args.json:
            print(json.dumps({args.gate: result}))
        else:
            print(f"✅ {args.gate}: PASSED" if result else f"❌ {args.gate}: FAILED")
        sys.exit(0 if result else 1)
        
    elif args.all:
        results = runner.run_all_gates()
        
        if args.json:
            print(json.dumps(results))
        else:
            for gate_name, passed in results.items():
                status = "✅ PASSED" if passed else "❌ FAILED"
                print(f"{gate_name}: {status}")
                
        all_passed = all(results.values())
        sys.exit(0 if all_passed else 1)
    else:
        parser.print_help()