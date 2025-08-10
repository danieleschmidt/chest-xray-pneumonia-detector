#!/usr/bin/env python3
# Quality Gates Runner for Medical AI System
# Comprehensive quality assurance and security validation

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import hashlib


@dataclass
class QualityCheck:
    """Individual quality check result."""
    name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    execution_time: float
    error_message: Optional[str] = None


@dataclass
class QualityGateResult:
    """Overall quality gate assessment."""
    timestamp: datetime
    overall_passed: bool
    overall_score: float
    individual_checks: List[QualityCheck]
    critical_issues: List[str]
    recommendations: List[str]
    execution_summary: Dict[str, Any]


class CodeQualityChecker:
    """Performs code quality analysis."""
    
    def __init__(self, source_dir: str = "src"):
        self.source_dir = Path(source_dir)
        self.logger = logging.getLogger(__name__)
    
    def check_python_syntax(self) -> QualityCheck:
        """Check Python syntax validity."""
        start_time = time.time()
        issues = []
        
        try:
            python_files = list(self.source_dir.rglob("*.py"))
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        code = f.read()
                    
                    # Compile to check syntax
                    compile(code, str(py_file), 'exec')
                    
                except SyntaxError as e:
                    issues.append(f"{py_file}: {e}")
                except Exception as e:
                    issues.append(f"{py_file}: {e}")
            
            passed = len(issues) == 0
            score = 1.0 if passed else max(0.0, 1.0 - len(issues) / len(python_files))
            
            return QualityCheck(
                name="Python Syntax Check",
                passed=passed,
                score=score,
                details={
                    'files_checked': len(python_files),
                    'syntax_errors': len(issues),
                    'error_details': issues[:10]  # Limit output
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QualityCheck(
                name="Python Syntax Check",
                passed=False,
                score=0.0,
                details={},
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def check_import_structure(self) -> QualityCheck:
        """Check import structure and dependencies."""
        start_time = time.time()
        
        try:
            python_files = list(self.source_dir.rglob("*.py"))
            import_issues = []
            circular_imports = []
            
            # Basic import analysis
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Check for relative imports outside package
                    for i, line in enumerate(lines, 1):
                        line = line.strip()
                        if line.startswith('from .') and not py_file.name == '__init__.py':
                            # Check if parent has __init__.py
                            parent_init = py_file.parent / '__init__.py'
                            if not parent_init.exists():
                                import_issues.append(f"{py_file}:{i}: Relative import without package structure")
                        
                        # Check for star imports (potential issue)
                        if 'import *' in line:
                            import_issues.append(f"{py_file}:{i}: Star import detected (may cause issues)")
                
                except Exception as e:
                    import_issues.append(f"{py_file}: Error reading file - {e}")
            
            passed = len(import_issues) == 0
            score = max(0.0, 1.0 - len(import_issues) / max(1, len(python_files)))
            
            return QualityCheck(
                name="Import Structure Check",
                passed=passed,
                score=score,
                details={
                    'files_analyzed': len(python_files),
                    'import_issues': len(import_issues),
                    'issue_details': import_issues[:10]
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QualityCheck(
                name="Import Structure Check",
                passed=False,
                score=0.0,
                details={},
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def check_code_complexity(self) -> QualityCheck:
        """Analyze code complexity metrics."""
        start_time = time.time()
        
        try:
            python_files = list(self.source_dir.rglob("*.py"))
            complexity_issues = []
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Simple complexity metrics
                    total_lines = len(lines)
                    code_lines = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
                    
                    # Count functions and classes
                    functions = len([l for l in lines if l.strip().startswith('def ')])
                    classes = len([l for l in lines if l.strip().startswith('class ')])
                    
                    # Check for very long functions (simple heuristic)
                    in_function = False
                    function_length = 0
                    current_function = ""
                    
                    for line in lines:
                        if line.strip().startswith('def '):
                            if in_function and function_length > 50:
                                complexity_issues.append(f"{py_file}: Function '{current_function}' is very long ({function_length} lines)")
                            in_function = True
                            function_length = 0
                            current_function = line.strip().split('(')[0].replace('def ', '')
                        elif in_function and line.strip():
                            function_length += 1
                    
                    # Check final function
                    if in_function and function_length > 50:
                        complexity_issues.append(f"{py_file}: Function '{current_function}' is very long ({function_length} lines)")
                    
                    # Check file length
                    if total_lines > 1000:
                        complexity_issues.append(f"{py_file}: File is very long ({total_lines} lines)")
                
                except Exception as e:
                    complexity_issues.append(f"{py_file}: Error analyzing complexity - {e}")
            
            passed = len(complexity_issues) == 0
            score = max(0.0, 1.0 - len(complexity_issues) / max(1, len(python_files)))
            
            return QualityCheck(
                name="Code Complexity Check",
                passed=passed,
                score=score,
                details={
                    'files_analyzed': len(python_files),
                    'complexity_issues': len(complexity_issues),
                    'issue_details': complexity_issues[:10]
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QualityCheck(
                name="Code Complexity Check",
                passed=False,
                score=0.0,
                details={},
                execution_time=time.time() - start_time,
                error_message=str(e)
            )


class SecurityChecker:
    """Performs security analysis."""
    
    def __init__(self, source_dir: str = "src"):
        self.source_dir = Path(source_dir)
        self.logger = logging.getLogger(__name__)
    
    def check_hardcoded_secrets(self) -> QualityCheck:
        """Check for hardcoded secrets and credentials."""
        start_time = time.time()
        
        # Common patterns for secrets
        secret_patterns = [
            r'password\s*=\s*["\'](?!.*\{.*\})[^"\']{8,}["\']',
            r'api_key\s*=\s*["\'][^"\']{20,}["\']',
            r'secret\s*=\s*["\'][^"\']{16,}["\']',
            r'token\s*=\s*["\'][^"\']{20,}["\']',
            r'key\s*=\s*["\'][^"\']{16,}["\']',
        ]
        
        try:
            import re
            
            python_files = list(self.source_dir.rglob("*.py"))
            security_issues = []
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for potential secrets
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        # Skip comments and test files
                        if line.strip().startswith('#') or 'test' in py_file.name.lower():
                            continue
                        
                        for pattern in secret_patterns:
                            if re.search(pattern, line, re.IGNORECASE):
                                # Allow certain safe patterns
                                if any(safe in line.lower() for safe in ['default', 'example', 'test', 'placeholder', 'demo']):
                                    continue
                                security_issues.append(f"{py_file}:{i}: Potential hardcoded secret detected")
                
                except Exception as e:
                    security_issues.append(f"{py_file}: Error scanning for secrets - {e}")
            
            passed = len(security_issues) == 0
            score = 1.0 if passed else max(0.0, 1.0 - len(security_issues) / max(1, len(python_files)))
            
            return QualityCheck(
                name="Hardcoded Secrets Check",
                passed=passed,
                score=score,
                details={
                    'files_scanned': len(python_files),
                    'potential_secrets': len(security_issues),
                    'issue_details': security_issues[:10]
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QualityCheck(
                name="Hardcoded Secrets Check",
                passed=False,
                score=0.0,
                details={},
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def check_sql_injection_patterns(self) -> QualityCheck:
        """Check for potential SQL injection vulnerabilities."""
        start_time = time.time()
        
        try:
            import re
            
            # Patterns that might indicate SQL injection vulnerabilities
            sql_patterns = [
                r'execute\s*\([^)]*%[^)]*\)',  # String formatting in execute
                r'query\s*=\s*["\'][^"\']*%[^"\']*["\']',  # String formatting in queries
                r'\.format\s*\([^)]*\)\s*(?:FROM|WHERE|SELECT)',  # .format() with SQL keywords
            ]
            
            python_files = list(self.source_dir.rglob("*.py"))
            sql_issues = []
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if line.strip().startswith('#'):
                            continue
                        
                        for pattern in sql_patterns:
                            if re.search(pattern, line, re.IGNORECASE):
                                sql_issues.append(f"{py_file}:{i}: Potential SQL injection pattern detected")
                
                except Exception as e:
                    sql_issues.append(f"{py_file}: Error scanning for SQL patterns - {e}")
            
            passed = len(sql_issues) == 0
            score = 1.0 if passed else max(0.0, 1.0 - len(sql_issues) / max(1, len(python_files)))
            
            return QualityCheck(
                name="SQL Injection Check",
                passed=passed,
                score=score,
                details={
                    'files_scanned': len(python_files),
                    'potential_vulnerabilities': len(sql_issues),
                    'issue_details': sql_issues[:10]
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QualityCheck(
                name="SQL Injection Check",
                passed=False,
                score=0.0,
                details={},
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def check_file_permissions(self) -> QualityCheck:
        """Check file permissions for security issues."""
        start_time = time.time()
        
        try:
            permission_issues = []
            
            # Check Python files for execute permissions (usually not needed)
            python_files = list(self.source_dir.rglob("*.py"))
            
            for py_file in python_files:
                stat_info = py_file.stat()
                mode = stat_info.st_mode
                
                # Check if file is executable by others (potential issue)
                if mode & 0o001:  # Execute by others
                    permission_issues.append(f"{py_file}: File executable by others")
                
                # Check if file is writable by others (security issue)
                if mode & 0o002:  # Write by others
                    permission_issues.append(f"{py_file}: File writable by others")
            
            passed = len(permission_issues) == 0
            score = 1.0 if passed else max(0.0, 1.0 - len(permission_issues) / max(1, len(python_files)))
            
            return QualityCheck(
                name="File Permissions Check",
                passed=passed,
                score=score,
                details={
                    'files_checked': len(python_files),
                    'permission_issues': len(permission_issues),
                    'issue_details': permission_issues[:10]
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QualityCheck(
                name="File Permissions Check",
                passed=False,
                score=0.0,
                details={},
                execution_time=time.time() - start_time,
                error_message=str(e)
            )


class DocumentationChecker:
    """Checks documentation quality and completeness."""
    
    def __init__(self, source_dir: str = "src"):
        self.source_dir = Path(source_dir)
        self.logger = logging.getLogger(__name__)
    
    def check_docstring_coverage(self) -> QualityCheck:
        """Check docstring coverage for functions and classes."""
        start_time = time.time()
        
        try:
            import ast
            
            python_files = list(self.source_dir.rglob("*.py"))
            total_functions = 0
            documented_functions = 0
            total_classes = 0
            documented_classes = 0
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            total_functions += 1
                            if ast.get_docstring(node):
                                documented_functions += 1
                        elif isinstance(node, ast.ClassDef):
                            total_classes += 1
                            if ast.get_docstring(node):
                                documented_classes += 1
                
                except Exception as e:
                    self.logger.warning(f"Error analyzing {py_file}: {e}")
            
            function_coverage = documented_functions / max(1, total_functions)
            class_coverage = documented_classes / max(1, total_classes)
            overall_coverage = (function_coverage + class_coverage) / 2 if total_classes > 0 else function_coverage
            
            passed = overall_coverage >= 0.7  # 70% coverage threshold
            
            return QualityCheck(
                name="Docstring Coverage Check",
                passed=passed,
                score=overall_coverage,
                details={
                    'total_functions': total_functions,
                    'documented_functions': documented_functions,
                    'function_coverage': function_coverage,
                    'total_classes': total_classes,
                    'documented_classes': documented_classes,
                    'class_coverage': class_coverage,
                    'overall_coverage': overall_coverage
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QualityCheck(
                name="Docstring Coverage Check",
                passed=False,
                score=0.0,
                details={},
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def check_readme_quality(self) -> QualityCheck:
        """Check README file quality and completeness."""
        start_time = time.time()
        
        try:
            readme_files = list(Path('.').glob('README*'))
            
            if not readme_files:
                return QualityCheck(
                    name="README Quality Check",
                    passed=False,
                    score=0.0,
                    details={'error': 'No README file found'},
                    execution_time=time.time() - start_time
                )
            
            readme_file = readme_files[0]
            with open(readme_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for essential sections
            required_sections = [
                'installation', 'usage', 'example', 'license',
                'getting started', 'quick start', 'how to'
            ]
            
            content_lower = content.lower()
            found_sections = [section for section in required_sections 
                            if section in content_lower]
            
            # Check content quality
            quality_indicators = {
                'has_code_examples': '```' in content or '    ' in content,
                'has_links': 'http' in content or '[' in content,
                'sufficient_length': len(content) > 500,
                'has_headers': '#' in content,
                'has_installation_info': any(word in content_lower 
                                           for word in ['install', 'pip', 'setup', 'requirements'])
            }
            
            quality_score = sum(quality_indicators.values()) / len(quality_indicators)
            section_score = len(found_sections) / len(required_sections)
            overall_score = (quality_score + section_score) / 2
            
            passed = overall_score >= 0.6
            
            return QualityCheck(
                name="README Quality Check",
                passed=passed,
                score=overall_score,
                details={
                    'readme_file': str(readme_file),
                    'content_length': len(content),
                    'required_sections_found': found_sections,
                    'quality_indicators': quality_indicators,
                    'section_score': section_score,
                    'quality_score': quality_score
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QualityCheck(
                name="README Quality Check",
                passed=False,
                score=0.0,
                details={},
                execution_time=time.time() - start_time,
                error_message=str(e)
            )


class QualityGateRunner:
    """Main quality gate runner that coordinates all checks."""
    
    def __init__(self, source_dir: str = "src"):
        self.source_dir = source_dir
        self.logger = logging.getLogger(__name__)
        
        # Initialize checkers
        self.code_checker = CodeQualityChecker(source_dir)
        self.security_checker = SecurityChecker(source_dir)
        self.doc_checker = DocumentationChecker(source_dir)
    
    def run_all_checks(self) -> QualityGateResult:
        """Run all quality checks and return comprehensive result."""
        
        start_time = time.time()
        self.logger.info("Starting comprehensive quality gate analysis...")
        
        checks = []
        critical_issues = []
        recommendations = []
        
        # Code Quality Checks
        self.logger.info("Running code quality checks...")
        checks.extend([
            self.code_checker.check_python_syntax(),
            self.code_checker.check_import_structure(),
            self.code_checker.check_code_complexity()
        ])
        
        # Security Checks
        self.logger.info("Running security checks...")
        checks.extend([
            self.security_checker.check_hardcoded_secrets(),
            self.security_checker.check_sql_injection_patterns(),
            self.security_checker.check_file_permissions()
        ])
        
        # Documentation Checks
        self.logger.info("Running documentation checks...")
        checks.extend([
            self.doc_checker.check_docstring_coverage(),
            self.doc_checker.check_readme_quality()
        ])
        
        # Analyze results
        passed_checks = [c for c in checks if c.passed]
        failed_checks = [c for c in checks if not c.passed]
        
        # Calculate overall score
        overall_score = sum(c.score for c in checks) / len(checks) if checks else 0.0
        overall_passed = len(failed_checks) == 0 and overall_score >= 0.7
        
        # Identify critical issues
        for check in failed_checks:
            if check.score < 0.5:  # Critical threshold
                critical_issues.append(f"{check.name}: {check.error_message or 'Failed with low score'}")
        
        # Generate recommendations
        if not overall_passed:
            if any('Syntax' in c.name for c in failed_checks):
                recommendations.append("Fix Python syntax errors before proceeding")
            
            if any('Security' in c.name for c in failed_checks):
                recommendations.append("Address security vulnerabilities immediately")
            
            if any('Docstring' in c.name for c in failed_checks):
                recommendations.append("Improve code documentation coverage")
            
            if overall_score < 0.6:
                recommendations.append("Overall quality score too low - comprehensive review needed")
        
        execution_summary = {
            'total_checks': len(checks),
            'passed_checks': len(passed_checks),
            'failed_checks': len(failed_checks),
            'total_execution_time': time.time() - start_time,
            'checks_per_second': len(checks) / (time.time() - start_time)
        }
        
        result = QualityGateResult(
            timestamp=datetime.now(),
            overall_passed=overall_passed,
            overall_score=overall_score,
            individual_checks=checks,
            critical_issues=critical_issues,
            recommendations=recommendations,
            execution_summary=execution_summary
        )
        
        self.logger.info(f"Quality gate analysis completed in {execution_summary['total_execution_time']:.2f}s")
        self.logger.info(f"Overall result: {'PASSED' if overall_passed else 'FAILED'} (Score: {overall_score:.3f})")
        
        return result
    
    def save_results(self, result: QualityGateResult, output_file: str = "quality_gate_results.json"):
        """Save results to JSON file."""
        
        # Convert to JSON-serializable format
        result_dict = asdict(result)
        result_dict['timestamp'] = result.timestamp.isoformat()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to {output_file}")
    
    def generate_report(self, result: QualityGateResult) -> str:
        """Generate human-readable quality gate report."""
        
        report_lines = [
            "=" * 80,
            "QUALITY GATE ASSESSMENT REPORT",
            "=" * 80,
            f"Timestamp: {result.timestamp.isoformat()}",
            f"Overall Result: {'✅ PASSED' if result.overall_passed else '❌ FAILED'}",
            f"Overall Score: {result.overall_score:.3f}/1.000",
            "",
            "EXECUTION SUMMARY",
            "-" * 40,
            f"Total Checks: {result.execution_summary['total_checks']}",
            f"Passed: {result.execution_summary['passed_checks']}",
            f"Failed: {result.execution_summary['failed_checks']}",
            f"Execution Time: {result.execution_summary['total_execution_time']:.2f}s",
            "",
            "INDIVIDUAL CHECK RESULTS",
            "-" * 40
        ]
        
        for check in result.individual_checks:
            status = "✅ PASS" if check.passed else "❌ FAIL"
            report_lines.extend([
                f"{check.name}: {status} (Score: {check.score:.3f})",
                f"  Execution Time: {check.execution_time:.3f}s"
            ])
            
            if check.error_message:
                report_lines.append(f"  Error: {check.error_message}")
            
            if check.details and isinstance(check.details, dict):
                for key, value in list(check.details.items())[:3]:  # Limit output
                    report_lines.append(f"  {key}: {value}")
            
            report_lines.append("")
        
        if result.critical_issues:
            report_lines.extend([
                "CRITICAL ISSUES",
                "-" * 40
            ])
            for issue in result.critical_issues:
                report_lines.append(f"⚠️ {issue}")
            report_lines.append("")
        
        if result.recommendations:
            report_lines.extend([
                "RECOMMENDATIONS",
                "-" * 40
            ])
            for rec in result.recommendations:
                report_lines.append(f"💡 {rec}")
            report_lines.append("")
        
        report_lines.extend([
            "=" * 80,
            "END OF REPORT",
            "=" * 80
        ])
        
        return "\n".join(report_lines)


def main():
    """Main entry point for quality gate runner."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run quality gates
    runner = QualityGateRunner()
    result = runner.run_all_checks()
    
    # Save results
    runner.save_results(result)
    
    # Generate and display report
    report = runner.generate_report(result)
    print(report)
    
    # Write report to file
    with open("quality_gate_report.txt", "w", encoding='utf-8') as f:
        f.write(report)
    
    # Exit with appropriate code
    sys.exit(0 if result.overall_passed else 1)


if __name__ == "__main__":
    main()