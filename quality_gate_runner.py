#!/usr/bin/env python3
"""Quality Gate Runner for Medical AI Project.

Runs comprehensive quality checks without external dependencies.
"""

import ast
import os
import re
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import time


class QualityGateRunner:
    """Runs quality gates for the medical AI project."""
    
    def __init__(self, project_root: str = "/root/repo"):
        self.project_root = Path(project_root)
        self.results = {}
        
    def run_all_quality_gates(self) -> Dict[str, Any]:
        """Run all quality gate checks."""
        print("🚀 Running Comprehensive Quality Gates")
        print("=" * 50)
        
        gates = [
            ("Code Structure", self.check_code_structure),
            ("Import Analysis", self.check_imports),
            ("Security Analysis", self.check_security_patterns),
            ("Documentation", self.check_documentation),
            ("Performance Patterns", self.check_performance_patterns),
            ("Test Coverage", self.estimate_test_coverage),
            ("Configuration", self.check_configuration),
            ("Dependencies", self.check_dependencies)
        ]
        
        for gate_name, gate_func in gates:
            print(f"\n📊 Running {gate_name}...")
            try:
                result = gate_func()
                self.results[gate_name] = {
                    'status': 'PASS' if result.get('passed', True) else 'FAIL',
                    'details': result
                }
                status = "✅ PASS" if result.get('passed', True) else "❌ FAIL"
                print(f"   {status}: {result.get('summary', 'Completed')}")
                
            except Exception as e:
                self.results[gate_name] = {
                    'status': 'ERROR',
                    'details': {'error': str(e)}
                }
                print(f"   ⚠️  ERROR: {str(e)}")
        
        # Generate overall summary
        self.generate_summary()
        return self.results
    
    def check_code_structure(self) -> Dict[str, Any]:
        """Check code structure and organization."""
        python_files = list(self.project_root.rglob("*.py"))
        
        structure_analysis = {
            'total_files': len(python_files),
            'total_lines': 0,
            'avg_lines_per_file': 0,
            'files_by_size': {'small': 0, 'medium': 0, 'large': 0, 'xlarge': 0},
            'has_init_files': 0,
            'src_structure_good': False
        }
        
        line_counts = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())
                    line_counts.append(lines)
                    structure_analysis['total_lines'] += lines
                    
                    # Categorize by size
                    if lines < 100:
                        structure_analysis['files_by_size']['small'] += 1
                    elif lines < 300:
                        structure_analysis['files_by_size']['medium'] += 1
                    elif lines < 800:
                        structure_analysis['files_by_size']['large'] += 1
                    else:
                        structure_analysis['files_by_size']['xlarge'] += 1
                        
                if py_file.name == '__init__.py':
                    structure_analysis['has_init_files'] += 1
                    
            except Exception as e:
                continue
        
        if line_counts:
            structure_analysis['avg_lines_per_file'] = sum(line_counts) / len(line_counts)
        
        # Check if src/ directory structure exists and is well organized
        src_dir = self.project_root / "src"
        if src_dir.exists():
            src_subdirs = [d for d in src_dir.iterdir() if d.is_dir()]
            structure_analysis['src_structure_good'] = len(src_subdirs) >= 3
        
        passed = (
            structure_analysis['total_files'] > 10 and
            structure_analysis['avg_lines_per_file'] < 500 and
            structure_analysis['has_init_files'] > 5
        )
        
        return {
            'passed': passed,
            'summary': f"Found {structure_analysis['total_files']} Python files, avg {structure_analysis['avg_lines_per_file']:.1f} lines/file",
            **structure_analysis
        }
    
    def check_imports(self) -> Dict[str, Any]:
        """Analyze import statements and dependencies."""
        python_files = list(self.project_root.rglob("*.py"))
        
        import_analysis = {
            'total_imports': 0,
            'stdlib_imports': 0,
            'third_party_imports': 0,
            'local_imports': 0,
            'common_third_party': {},
            'potential_issues': []
        }
        
        # Common standard library modules
        stdlib_modules = {
            'os', 'sys', 'time', 'datetime', 'json', 'logging', 'threading',
            'asyncio', 'pathlib', 'collections', 'dataclasses', 'typing',
            'enum', 'random', 'math', 'statistics', 're', 'contextlib',
            'functools', 'itertools', 'traceback', 'uuid', 'queue',
            'concurrent', 'multiprocessing'
        }
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST to find imports
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                import_analysis['total_imports'] += 1
                                module_name = alias.name.split('.')[0]
                                
                                if module_name in stdlib_modules:
                                    import_analysis['stdlib_imports'] += 1
                                elif module_name.startswith('.') or 'src.' in alias.name:
                                    import_analysis['local_imports'] += 1
                                else:
                                    import_analysis['third_party_imports'] += 1
                                    import_analysis['common_third_party'][module_name] = \
                                        import_analysis['common_third_party'].get(module_name, 0) + 1
                        
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                import_analysis['total_imports'] += 1
                                module_name = node.module.split('.')[0]
                                
                                if module_name in stdlib_modules:
                                    import_analysis['stdlib_imports'] += 1
                                elif node.module.startswith('.') or 'src.' in node.module:
                                    import_analysis['local_imports'] += 1
                                else:
                                    import_analysis['third_party_imports'] += 1
                                    import_analysis['common_third_party'][module_name] = \
                                        import_analysis['common_third_party'].get(module_name, 0) + 1
                
                except SyntaxError:
                    import_analysis['potential_issues'].append(f"Syntax error in {py_file}")
                    
            except Exception:
                continue
        
        # Check for heavy dependencies
        heavy_deps = ['tensorflow', 'torch', 'sklearn', 'numpy', 'pandas']
        has_ai_deps = any(dep in import_analysis['common_third_party'] for dep in heavy_deps)
        
        passed = (
            import_analysis['total_imports'] > 20 and
            import_analysis['local_imports'] > 5 and
            has_ai_deps
        )
        
        return {
            'passed': passed,
            'summary': f"Analyzed {import_analysis['total_imports']} imports, {import_analysis['local_imports']} local",
            **import_analysis
        }
    
    def check_security_patterns(self) -> Dict[str, Any]:
        """Check for security patterns and potential vulnerabilities."""
        python_files = list(self.project_root.rglob("*.py"))
        
        security_analysis = {
            'files_checked': 0,
            'security_patterns': {
                'encryption_usage': 0,
                'hashing_usage': 0,
                'authentication_patterns': 0,
                'input_validation': 0,
                'logging_security': 0
            },
            'potential_issues': {
                'hardcoded_secrets': 0,
                'unsafe_eval': 0,
                'sql_injection_risk': 0,
                'path_traversal_risk': 0
            },
            'security_files': []
        }
        
        # Security-related patterns to look for
        security_patterns = {
            'encryption_usage': [r'encrypt', r'decrypt', r'Fernet', r'AES', r'cryptography'],
            'hashing_usage': [r'hashlib', r'sha256', r'md5', r'bcrypt', r'scrypt'],
            'authentication_patterns': [r'auth', r'login', r'token', r'jwt', r'session'],
            'input_validation': [r'validate', r'sanitiz', r'clean', r'escape'],
            'logging_security': [r'audit', r'log.*secur', r'security.*log']
        }
        
        # Potential security issues
        issue_patterns = {
            'hardcoded_secrets': [r'password\s*=\s*[\'"][^\'"]+[\'"]', r'secret\s*=\s*[\'"][^\'"]+[\'"]', r'key\s*=\s*[\'"][^\'"]+[\'"]'],
            'unsafe_eval': [r'\beval\s*\(', r'\bexec\s*\('],
            'sql_injection_risk': [r'execute\s*\([^)]*%', r'query.*\+.*'],
            'path_traversal_risk': [r'open\s*\([^)]*\+', r'file.*\.\./']
        }
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                security_analysis['files_checked'] += 1
                
                # Check for security patterns
                for pattern_type, patterns in security_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            security_analysis['security_patterns'][pattern_type] += 1
                            if pattern_type == 'encryption_usage' and py_file.name not in security_analysis['security_files']:
                                security_analysis['security_files'].append(str(py_file.relative_to(self.project_root)))
                
                # Check for potential issues
                for issue_type, patterns in issue_patterns.items():
                    for pattern in patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        security_analysis['potential_issues'][issue_type] += len(matches)
                        
            except Exception:
                continue
        
        # Evaluate security posture
        total_security_patterns = sum(security_analysis['security_patterns'].values())
        total_potential_issues = sum(security_analysis['potential_issues'].values())
        
        passed = (
            total_security_patterns >= 5 and  # Good security patterns
            total_potential_issues <= 2 and   # Few potential issues
            len(security_analysis['security_files']) > 0  # Has security-focused files
        )
        
        return {
            'passed': passed,
            'summary': f"Found {total_security_patterns} security patterns, {total_potential_issues} potential issues",
            **security_analysis
        }
    
    def check_documentation(self) -> Dict[str, Any]:
        """Check documentation quality."""
        python_files = list(self.project_root.rglob("*.py"))
        md_files = list(self.project_root.rglob("*.md"))
        
        doc_analysis = {
            'python_files': len(python_files),
            'files_with_docstrings': 0,
            'total_docstrings': 0,
            'markdown_files': len(md_files),
            'readme_exists': False,
            'has_api_docs': False,
            'documentation_score': 0
        }
        
        # Check for README
        readme_files = ['README.md', 'README.rst', 'README.txt', 'readme.md']
        for readme_name in readme_files:
            if (self.project_root / readme_name).exists():
                doc_analysis['readme_exists'] = True
                break
        
        # Check for API documentation
        api_doc_patterns = ['API', 'api', 'usage', 'guide', 'GUIDE']
        for md_file in md_files:
            if any(pattern in md_file.name for pattern in api_doc_patterns):
                doc_analysis['has_api_docs'] = True
                break
        
        # Analyze Python docstrings
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST to find docstrings
                try:
                    tree = ast.parse(content)
                    file_has_docstring = False
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                            if (ast.get_docstring(node) is not None):
                                doc_analysis['total_docstrings'] += 1
                                file_has_docstring = True
                    
                    if file_has_docstring:
                        doc_analysis['files_with_docstrings'] += 1
                        
                except SyntaxError:
                    continue
                    
            except Exception:
                continue
        
        # Calculate documentation score
        docstring_ratio = doc_analysis['files_with_docstrings'] / doc_analysis['python_files'] if doc_analysis['python_files'] > 0 else 0
        doc_analysis['documentation_score'] = (
            (docstring_ratio * 40) +  # 40% for docstring coverage
            (20 if doc_analysis['readme_exists'] else 0) +  # 20% for README
            (20 if doc_analysis['has_api_docs'] else 0) +  # 20% for API docs
            (min(doc_analysis['markdown_files'], 5) * 4)  # Up to 20% for additional docs
        )
        
        passed = (
            doc_analysis['documentation_score'] >= 60 and
            doc_analysis['readme_exists'] and
            docstring_ratio >= 0.3
        )
        
        return {
            'passed': passed,
            'summary': f"Documentation score: {doc_analysis['documentation_score']:.1f}/100",
            **doc_analysis
        }
    
    def check_performance_patterns(self) -> Dict[str, Any]:
        """Check for performance optimization patterns."""
        python_files = list(self.project_root.rglob("*.py"))
        
        perf_analysis = {
            'files_checked': 0,
            'performance_patterns': {
                'async_usage': 0,
                'caching_patterns': 0,
                'optimization_imports': 0,
                'multiprocessing_usage': 0,
                'lazy_loading': 0,
                'vectorization': 0
            },
            'potential_bottlenecks': {
                'nested_loops': 0,
                'inefficient_string_concat': 0,
                'unoptimized_file_io': 0
            }
        }
        
        # Performance patterns to look for
        perf_patterns = {
            'async_usage': [r'\basync\s+def', r'\bawait\b', r'asyncio'],
            'caching_patterns': [r'@cache', r'@lru_cache', r'cache', r'memoiz'],
            'optimization_imports': [r'numpy', r'pandas', r'numba', r'cython'],
            'multiprocessing_usage': [r'multiprocessing', r'ThreadPool', r'ProcessPool', r'concurrent'],
            'lazy_loading': [r'lazy', r'@property', r'generator'],
            'vectorization': [r'vectoriz', r'apply', r'map\(', r'np\.']
        }
        
        # Potential bottleneck patterns
        bottleneck_patterns = {
            'nested_loops': [r'for.*:\s*for.*:'],
            'inefficient_string_concat': [r'\+=.*[\'"]'],
            'unoptimized_file_io': [r'open\([^)]*\)[^.]*\.read\(\)']
        }
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                perf_analysis['files_checked'] += 1
                
                # Check for performance patterns
                for pattern_type, patterns in perf_patterns.items():
                    for pattern in patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        perf_analysis['performance_patterns'][pattern_type] += len(matches)
                
                # Check for potential bottlenecks
                for bottleneck_type, patterns in bottleneck_patterns.items():
                    for pattern in patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                        perf_analysis['potential_bottlenecks'][bottleneck_type] += len(matches)
                        
            except Exception:
                continue
        
        # Evaluate performance optimization
        total_perf_patterns = sum(perf_analysis['performance_patterns'].values())
        total_bottlenecks = sum(perf_analysis['potential_bottlenecks'].values())
        
        passed = (
            total_perf_patterns >= 10 and  # Good performance patterns
            perf_analysis['performance_patterns']['async_usage'] > 0 and  # Uses async
            perf_analysis['performance_patterns']['optimization_imports'] > 0  # Uses optimized libraries
        )
        
        return {
            'passed': passed,
            'summary': f"Found {total_perf_patterns} performance patterns, {total_bottlenecks} potential bottlenecks",
            **perf_analysis
        }
    
    def estimate_test_coverage(self) -> Dict[str, Any]:
        """Estimate test coverage by analyzing test files."""
        test_dirs = ['tests', 'test']
        test_files = []
        
        for test_dir in test_dirs:
            test_path = self.project_root / test_dir
            if test_path.exists():
                test_files.extend(list(test_path.rglob("test_*.py")))
                test_files.extend(list(test_path.rglob("*_test.py")))
        
        src_files = list((self.project_root / "src").rglob("*.py"))
        
        coverage_analysis = {
            'test_files': len(test_files),
            'src_files': len(src_files),
            'estimated_coverage': 0,
            'test_functions': 0,
            'test_patterns': {
                'unit_tests': 0,
                'integration_tests': 0,
                'fixtures_usage': 0,
                'mocking_usage': 0
            }
        }
        
        # Analyze test files
        test_patterns = {
            'unit_tests': [r'def test_', r'class Test'],
            'integration_tests': [r'integration', r'e2e', r'end.to.end'],
            'fixtures_usage': [r'@fixture', r'@pytest\.fixture'],
            'mocking_usage': [r'mock', r'Mock', r'patch', r'@patch']
        }
        
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Count test functions
                coverage_analysis['test_functions'] += len(re.findall(r'def test_', content))
                
                # Check for test patterns
                for pattern_type, patterns in test_patterns.items():
                    for pattern in patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        coverage_analysis['test_patterns'][pattern_type] += len(matches)
                        
            except Exception:
                continue
        
        # Estimate coverage based on test-to-source ratio
        if coverage_analysis['src_files'] > 0:
            test_ratio = coverage_analysis['test_files'] / coverage_analysis['src_files']
            coverage_analysis['estimated_coverage'] = min(test_ratio * 100, 95)
        
        passed = (
            coverage_analysis['test_files'] >= 5 and
            coverage_analysis['test_functions'] >= 20 and
            coverage_analysis['estimated_coverage'] >= 30
        )
        
        return {
            'passed': passed,
            'summary': f"Found {coverage_analysis['test_files']} test files, ~{coverage_analysis['estimated_coverage']:.1f}% coverage",
            **coverage_analysis
        }
    
    def check_configuration(self) -> Dict[str, Any]:
        """Check configuration files and setup."""
        config_files = {
            'pyproject.toml': False,
            'requirements.txt': False,
            'pytest.ini': False,
            'Dockerfile': False,
            'docker-compose.yml': False,
            '.gitignore': False,
            'Makefile': False
        }
        
        for config_file in config_files.keys():
            if (self.project_root / config_file).exists():
                config_files[config_file] = True
        
        # Check for environment-specific configs
        env_configs = list(self.project_root.glob("requirements-*.txt"))
        docker_configs = list(self.project_root.glob("docker-compose.*.yml"))
        
        config_analysis = {
            'config_files': config_files,
            'total_config_files': sum(config_files.values()),
            'env_specific_configs': len(env_configs),
            'docker_configs': len(docker_configs),
            'has_ci_config': False
        }
        
        # Check for CI/CD configuration
        ci_paths = ['.github/workflows', '.gitlab-ci.yml', 'Jenkinsfile', '.travis.yml']
        for ci_path in ci_paths:
            if (self.project_root / ci_path).exists():
                config_analysis['has_ci_config'] = True
                break
        
        passed = (
            config_analysis['total_config_files'] >= 4 and
            config_files['pyproject.toml'] and
            config_files['requirements.txt'] and
            config_analysis['env_specific_configs'] >= 2
        )
        
        return {
            'passed': passed,
            'summary': f"Found {config_analysis['total_config_files']}/7 config files",
            **config_analysis
        }
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Analyze project dependencies."""
        requirements_files = list(self.project_root.glob("requirements*.txt"))
        
        dep_analysis = {
            'requirements_files': len(requirements_files),
            'total_dependencies': 0,
            'security_dependencies': 0,
            'ml_dependencies': 0,
            'web_dependencies': 0,
            'dev_dependencies': 0,
            'pinned_versions': 0,
            'dependency_categories': {}
        }
        
        # Dependency categories
        security_deps = ['cryptography', 'bcrypt', 'passlib', 'pyjwt', 'authlib']
        ml_deps = ['tensorflow', 'torch', 'sklearn', 'numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn']
        web_deps = ['flask', 'django', 'fastapi', 'requests', 'httpx', 'aiohttp']
        dev_deps = ['pytest', 'black', 'ruff', 'mypy', 'coverage', 'bandit']
        
        for req_file in requirements_files:
            try:
                with open(req_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        dep_analysis['total_dependencies'] += 1
                        
                        # Check if version is pinned
                        if '==' in line or '>=' in line or '<=' in line:
                            dep_analysis['pinned_versions'] += 1
                        
                        # Categorize dependency
                        dep_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0].strip()
                        
                        if dep_name.lower() in security_deps:
                            dep_analysis['security_dependencies'] += 1
                        elif dep_name.lower() in ml_deps:
                            dep_analysis['ml_dependencies'] += 1
                        elif dep_name.lower() in web_deps:
                            dep_analysis['web_dependencies'] += 1
                        elif dep_name.lower() in dev_deps:
                            dep_analysis['dev_dependencies'] += 1
                            
            except Exception:
                continue
        
        # Calculate version pinning ratio
        pinning_ratio = dep_analysis['pinned_versions'] / dep_analysis['total_dependencies'] if dep_analysis['total_dependencies'] > 0 else 0
        
        passed = (
            dep_analysis['total_dependencies'] >= 10 and
            dep_analysis['ml_dependencies'] >= 3 and  # ML project should have ML deps
            dep_analysis['security_dependencies'] >= 1 and  # Should have security deps
            pinning_ratio >= 0.7  # Most deps should have pinned versions
        )
        
        return {
            'passed': passed,
            'summary': f"Found {dep_analysis['total_dependencies']} dependencies, {pinning_ratio:.1%} pinned",
            **dep_analysis
        }
    
    def generate_summary(self):
        """Generate overall quality gate summary."""
        total_gates = len(self.results)
        passed_gates = sum(1 for result in self.results.values() if result['status'] == 'PASS')
        failed_gates = sum(1 for result in self.results.values() if result['status'] == 'FAIL')
        error_gates = sum(1 for result in self.results.values() if result['status'] == 'ERROR')
        
        pass_rate = (passed_gates / total_gates) * 100 if total_gates > 0 else 0
        
        print(f"\n" + "=" * 50)
        print("📋 QUALITY GATE SUMMARY")
        print("=" * 50)
        print(f"Total Gates: {total_gates}")
        print(f"✅ Passed: {passed_gates}")
        print(f"❌ Failed: {failed_gates}")
        print(f"⚠️  Errors: {error_gates}")
        print(f"📊 Pass Rate: {pass_rate:.1f}%")
        
        if pass_rate >= 80:
            print("🎉 EXCELLENT: High quality codebase!")
        elif pass_rate >= 60:
            print("👍 GOOD: Solid codebase with room for improvement")
        elif pass_rate >= 40:
            print("⚠️  FAIR: Some quality issues need attention")
        else:
            print("🚨 POOR: Significant quality improvements needed")
        
        # Save results to file
        results_file = self.project_root / "quality_gate_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'summary': {
                    'total_gates': total_gates,
                    'passed_gates': passed_gates,
                    'failed_gates': failed_gates,
                    'error_gates': error_gates,
                    'pass_rate': pass_rate
                },
                'results': self.results
            }, f, indent=2)
        
        print(f"📄 Detailed results saved to: {results_file}")


def main():
    """Main entry point."""
    runner = QualityGateRunner()
    results = runner.run_all_quality_gates()
    
    # Exit with appropriate code
    failed_gates = sum(1 for result in results.values() if result['status'] in ['FAIL', 'ERROR'])
    sys.exit(0 if failed_gates == 0 else 1)


if __name__ == "__main__":
    main()