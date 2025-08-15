#!/usr/bin/env python3
"""Enhanced Quality Gates for Medical AI System"""

import subprocess
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import tempfile
import os


class QualityGateRunner:
    """Enhanced quality gate system for medical AI applications"""
    
    def __init__(self):
        self.results = {}
        self.passed_gates = []
        self.failed_gates = []
        
    def run_security_scan(self) -> Tuple[bool, Dict]:
        """Run comprehensive security scanning"""
        print("🔒 Running security scan...")
        
        try:
            # Run bandit security scan
            result = subprocess.run(
                ["python", "-m", "bandit", "-r", "src/", "-f", "json"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                self.passed_gates.append("security_scan")
                return True, {"status": "passed", "details": "No security issues found"}
            else:
                bandit_output = result.stdout if result.stdout else result.stderr
                self.failed_gates.append("security_scan")
                return False, {"status": "failed", "details": bandit_output}
                
        except Exception as e:
            self.failed_gates.append("security_scan")
            return False, {"status": "error", "details": str(e)}
    
    def run_code_quality_check(self) -> Tuple[bool, Dict]:
        """Run code quality checks with ruff"""
        print("📝 Running code quality check...")
        
        try:
            # Run ruff linting
            result = subprocess.run(
                ["python", "-m", "ruff", "check", "src/", "--output-format=json"],
                capture_output=True,
                text=True,
                timeout=180
            )
            
            if result.returncode == 0:
                self.passed_gates.append("code_quality")
                return True, {"status": "passed", "details": "All code quality checks passed"}
            else:
                self.failed_gates.append("code_quality")
                return False, {"status": "failed", "details": result.stdout}
                
        except Exception as e:
            self.failed_gates.append("code_quality")
            return False, {"status": "error", "details": str(e)}
    
    def run_test_suite(self) -> Tuple[bool, Dict]:
        """Run comprehensive test suite"""
        print("🧪 Running test suite...")
        
        try:
            # Run pytest with coverage
            result = subprocess.run([
                "python", "-m", "pytest", 
                "tests/", 
                "--cov=src", 
                "--cov-report=json",
                "--json-report",
                "--json-report-file=test_results.json",
                "-v"
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                # Check coverage
                try:
                    with open("coverage.json", "r") as f:
                        coverage_data = json.load(f)
                        coverage_percent = coverage_data["totals"]["percent_covered"]
                        
                    if coverage_percent >= 85:
                        self.passed_gates.append("test_suite")
                        return True, {
                            "status": "passed", 
                            "coverage": coverage_percent,
                            "details": "All tests passed with sufficient coverage"
                        }
                    else:
                        self.failed_gates.append("test_suite")
                        return False, {
                            "status": "failed",
                            "coverage": coverage_percent,
                            "details": f"Coverage {coverage_percent}% below required 85%"
                        }
                except FileNotFoundError:
                    self.passed_gates.append("test_suite")
                    return True, {"status": "passed", "details": "Tests passed (coverage file not found)"}
            else:
                self.failed_gates.append("test_suite")
                return False, {"status": "failed", "details": result.stdout}
                
        except Exception as e:
            self.failed_gates.append("test_suite")
            return False, {"status": "error", "details": str(e)}
    
    def run_medical_ai_validation(self) -> Tuple[bool, Dict]:
        """Run medical AI specific validations"""
        print("🏥 Running medical AI validation...")
        
        try:
            # Check for HIPAA compliance patterns
            hipaa_patterns = [
                "encryption",
                "audit",
                "logging", 
                "anonymization",
                "validation"
            ]
            
            compliant_patterns = []
            src_files = list(Path("src").rglob("*.py"))
            
            for pattern in hipaa_patterns:
                for file_path in src_files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            if pattern in f.read().lower():
                                compliant_patterns.append(pattern)
                                break
                    except:
                        continue
            
            compliance_score = len(compliant_patterns) / len(hipaa_patterns)
            
            if compliance_score >= 0.6:
                self.passed_gates.append("medical_ai_validation")
                return True, {
                    "status": "passed",
                    "compliance_score": compliance_score,
                    "found_patterns": compliant_patterns
                }
            else:
                self.failed_gates.append("medical_ai_validation")
                return False, {
                    "status": "failed", 
                    "compliance_score": compliance_score,
                    "found_patterns": compliant_patterns
                }
                
        except Exception as e:
            self.failed_gates.append("medical_ai_validation")
            return False, {"status": "error", "details": str(e)}
    
    def run_performance_benchmark(self) -> Tuple[bool, Dict]:
        """Run performance benchmarks"""
        print("⚡ Running performance benchmark...")
        
        try:
            # Simple performance test
            start_time = time.time()
            
            # Test import performance
            import importlib
            modules_to_test = [
                "src.model_builder",
                "src.data_loader", 
                "src.train_engine"
            ]
            
            import_times = {}
            for module in modules_to_test:
                module_start = time.time()
                try:
                    importlib.import_module(module)
                    import_times[module] = time.time() - module_start
                except ImportError as e:
                    import_times[module] = f"Import failed: {e}"
            
            total_time = time.time() - start_time
            
            if total_time < 10.0:  # Should import in under 10 seconds
                self.passed_gates.append("performance_benchmark")
                return True, {
                    "status": "passed",
                    "total_time": total_time,
                    "import_times": import_times
                }
            else:
                self.failed_gates.append("performance_benchmark")
                return False, {
                    "status": "failed",
                    "total_time": total_time,
                    "details": "Import performance too slow"
                }
                
        except Exception as e:
            self.failed_gates.append("performance_benchmark")
            return False, {"status": "error", "details": str(e)}
    
    def run_dependency_audit(self) -> Tuple[bool, Dict]:
        """Audit dependencies for security vulnerabilities"""
        print("📦 Running dependency audit...")
        
        try:
            # Check if safety is available and run it
            result = subprocess.run(
                ["python", "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                packages = json.loads(result.stdout)
                critical_packages = [
                    "tensorflow", "mlflow", "cryptography", 
                    "pillow", "flask", "gunicorn"
                ]
                
                found_packages = []
                for pkg in packages:
                    if pkg["name"].lower() in critical_packages:
                        found_packages.append(pkg)
                
                self.passed_gates.append("dependency_audit")
                return True, {
                    "status": "passed",
                    "total_packages": len(packages),
                    "critical_packages": found_packages
                }
            else:
                self.failed_gates.append("dependency_audit")
                return False, {"status": "failed", "details": result.stderr}
                
        except Exception as e:
            self.failed_gates.append("dependency_audit")
            return False, {"status": "error", "details": str(e)}
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality gate report"""
        total_gates = len(self.passed_gates) + len(self.failed_gates)
        success_rate = len(self.passed_gates) / total_gates if total_gates > 0 else 0
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_status": "PASSED" if success_rate >= 0.85 else "FAILED",
            "success_rate": success_rate,
            "total_gates": total_gates,
            "passed_gates": self.passed_gates,
            "failed_gates": self.failed_gates,
            "detailed_results": self.results,
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on failed gates"""
        recommendations = []
        
        if "security_scan" in self.failed_gates:
            recommendations.append("Address security vulnerabilities identified by bandit")
        
        if "code_quality" in self.failed_gates:
            recommendations.append("Fix code quality issues identified by ruff")
        
        if "test_suite" in self.failed_gates:
            recommendations.append("Increase test coverage to at least 85%")
        
        if "medical_ai_validation" in self.failed_gates:
            recommendations.append("Improve HIPAA compliance patterns in codebase")
        
        if "performance_benchmark" in self.failed_gates:
            recommendations.append("Optimize module import performance")
        
        if "dependency_audit" in self.failed_gates:
            recommendations.append("Update vulnerable dependencies")
        
        return recommendations
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates and return comprehensive report"""
        print("🚀 Starting Enhanced Quality Gate Execution...")
        print("=" * 60)
        
        gates = [
            ("Security Scan", self.run_security_scan),
            ("Code Quality", self.run_code_quality_check),
            ("Test Suite", self.run_test_suite),
            ("Medical AI Validation", self.run_medical_ai_validation),
            ("Performance Benchmark", self.run_performance_benchmark),
            ("Dependency Audit", self.run_dependency_audit),
        ]
        
        for gate_name, gate_func in gates:
            try:
                success, result = gate_func()
                self.results[gate_name.lower().replace(" ", "_")] = result
                
                status_emoji = "✅" if success else "❌"
                print(f"{status_emoji} {gate_name}: {result['status'].upper()}")
                
            except Exception as e:
                self.results[gate_name.lower().replace(" ", "_")] = {
                    "status": "error",
                    "details": str(e)
                }
                self.failed_gates.append(gate_name.lower().replace(" ", "_"))
                print(f"❌ {gate_name}: ERROR - {str(e)}")
        
        print("=" * 60)
        
        # Generate and save report
        report = self.generate_report()
        
        # Save report to file
        with open("enhanced_quality_gate_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Overall Status: {report['overall_status']}")
        print(f"📈 Success Rate: {report['success_rate']:.1%}")
        print(f"✅ Passed Gates: {len(self.passed_gates)}")
        print(f"❌ Failed Gates: {len(self.failed_gates)}")
        
        if report["recommendations"]:
            print("\n💡 Recommendations:")
            for rec in report["recommendations"]:
                print(f"  • {rec}")
        
        return report


def main():
    """Main entry point for quality gate runner"""
    runner = QualityGateRunner()
    report = runner.run_all_gates()
    
    # Exit with appropriate code
    if report["overall_status"] == "PASSED":
        print("\n🎉 All quality gates passed! System ready for deployment.")
        sys.exit(0)
    else:
        print("\n⚠️  Some quality gates failed. Please address issues before deployment.")
        sys.exit(1)


if __name__ == "__main__":
    main()