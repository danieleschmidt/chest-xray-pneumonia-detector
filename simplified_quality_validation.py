#!/usr/bin/env python3
"""Simplified Quality Validation for Quantum-Medical AI System.

Performs essential code quality checks without external dependencies.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_file_structure() -> Tuple[bool, Dict[str, Any]]:
    """Validate project file structure."""
    required_files = [
        "src/quantum_inspired_task_planner/__init__.py",
        "src/api/main.py",
        "src/autonomous_model_versioning.py",
        "src/real_time_quantum_performance_optimizer.py",
        "src/research/novel_quantum_medical_fusion.py",
        "src/research/quantum_vs_classical_comparative_study.py",
        "requirements.txt",
        "pyproject.toml"
    ]
    
    missing_files = []
    present_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            present_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    passed = len(missing_files) == 0
    
    details = {
        "required_files": len(required_files),
        "present_files": len(present_files),
        "missing_files": missing_files,
        "file_structure_score": len(present_files) / len(required_files)
    }
    
    return passed, details


def check_python_syntax() -> Tuple[bool, Dict[str, Any]]:
    """Check Python syntax for all source files."""
    src_dir = Path("src")
    python_files = list(src_dir.rglob("*.py"))
    
    syntax_errors = []
    valid_files = []
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Compile to check syntax
            compile(content, str(py_file), 'exec')
            valid_files.append(str(py_file))
            
        except SyntaxError as e:
            syntax_errors.append({
                "file": str(py_file),
                "error": str(e),
                "line": getattr(e, 'lineno', 'unknown')
            })
        except UnicodeDecodeError:
            syntax_errors.append({
                "file": str(py_file),
                "error": "Unicode decode error",
                "line": "unknown"
            })
        except Exception as e:
            syntax_errors.append({
                "file": str(py_file),
                "error": str(e),
                "line": "unknown"
            })
    
    passed = len(syntax_errors) == 0
    
    details = {
        "total_python_files": len(python_files),
        "valid_files": len(valid_files),
        "syntax_errors": syntax_errors,
        "syntax_score": len(valid_files) / len(python_files) if python_files else 1.0
    }
    
    return passed, details


def check_imports() -> Tuple[bool, Dict[str, Any]]:
    """Check critical imports can be resolved."""
    src_dir = Path("src")
    sys.path.insert(0, str(src_dir))
    
    import_tests = [
        ("quantum_inspired_task_planner.quantum_optimization", "QuantumAnnealer"),
        ("autonomous_model_versioning", "AutonomousModelVersionManager"),
        ("real_time_quantum_performance_optimizer", "RealTimeQuantumOptimizer"),
        ("research.novel_quantum_medical_fusion", "QuantumMedicalFusionNetwork"),
        ("research.quantum_vs_classical_comparative_study", "ComparativeStudyFramework")
    ]
    
    successful_imports = []
    failed_imports = []
    
    for module_name, class_name in import_tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            successful_imports.append(f"{module_name}.{class_name}")
        except ImportError as e:
            failed_imports.append({
                "module": module_name,
                "class": class_name,
                "error": str(e)
            })
        except AttributeError as e:
            failed_imports.append({
                "module": module_name,
                "class": class_name,
                "error": f"Class {class_name} not found: {str(e)}"
            })
        except Exception as e:
            failed_imports.append({
                "module": module_name,
                "class": class_name,
                "error": str(e)
            })
    
    passed = len(failed_imports) == 0
    
    details = {
        "total_import_tests": len(import_tests),
        "successful_imports": successful_imports,
        "failed_imports": failed_imports,
        "import_score": len(successful_imports) / len(import_tests)
    }
    
    return passed, details


def check_documentation() -> Tuple[bool, Dict[str, Any]]:
    """Check documentation quality."""
    required_docs = [
        "README.md",
        "ARCHITECTURE.md",
        "API_USAGE_GUIDE.md",
        "DEPLOYMENT_GUIDE.md"
    ]
    
    doc_files = []
    missing_docs = []
    
    for doc_file in required_docs:
        if Path(doc_file).exists():
            doc_files.append(doc_file)
            # Check if file has content
            try:
                with open(doc_file, 'r') as f:
                    content = f.read().strip()
                    if len(content) < 100:  # Minimum content requirement
                        missing_docs.append(f"{doc_file} (too short)")
            except Exception:
                missing_docs.append(f"{doc_file} (read error)")
        else:
            missing_docs.append(doc_file)
    
    passed = len(missing_docs) == 0
    
    details = {
        "required_docs": len(required_docs),
        "present_docs": len(doc_files),
        "missing_docs": missing_docs,
        "documentation_score": len(doc_files) / len(required_docs)
    }
    
    return passed, details


def check_security_patterns() -> Tuple[bool, Dict[str, Any]]:
    """Check for basic security patterns."""
    src_dir = Path("src")
    python_files = list(src_dir.rglob("*.py"))
    
    security_issues = []
    security_patterns = [
        ("eval(", "Use of eval() function"),
        ("exec(", "Use of exec() function"),
        ("__import__", "Dynamic imports"),
        ("subprocess.call", "Subprocess calls"),
        ("os.system", "OS system calls"),
        ("shell=True", "Shell execution"),
        ("pickle.load", "Pickle deserialization"),
        ("yaml.load", "YAML unsafe loading")
    ]
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for pattern, description in security_patterns:
                if pattern in content:
                    # Count lines to find approximate location
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if pattern in line:
                            security_issues.append({
                                "file": str(py_file),
                                "pattern": pattern,
                                "description": description,
                                "line": i,
                                "line_content": line.strip()
                            })
                            break
        except Exception as e:
            security_issues.append({
                "file": str(py_file),
                "pattern": "file_read_error",
                "description": f"Could not read file: {str(e)}",
                "line": 0,
                "line_content": ""
            })
    
    # Filter out acceptable uses (like subprocess.run with explicit args)
    filtered_issues = []
    for issue in security_issues:
        line_content = issue["line_content"].lower()
        
        # Allow certain patterns in specific contexts
        if issue["pattern"] == "subprocess.call" and ("capture_output=true" in line_content or "shell=false" in line_content):
            continue
        if issue["pattern"] == "subprocess.run" and "shell=false" in line_content:
            continue
        if issue["pattern"] == "__import__" and ("fromlist=" in line_content):
            continue
            
        filtered_issues.append(issue)
    
    passed = len(filtered_issues) == 0
    
    details = {
        "total_files_scanned": len(python_files),
        "security_issues": filtered_issues,
        "security_score": 1.0 - (len(filtered_issues) / max(1, len(python_files)))
    }
    
    return passed, details


def check_medical_compliance() -> Tuple[bool, Dict[str, Any]]:
    """Check for medical AI compliance indicators."""
    src_dir = Path("src")
    python_files = list(src_dir.rglob("*.py"))
    
    medical_indicators = [
        "hipaa",
        "medical_compliance",
        "sensitivity",
        "specificity",
        "medical_safety",
        "patient_privacy",
        "healthcare",
        "clinical"
    ]
    
    compliance_scores = {}
    total_indicators = 0
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read().lower()
            
            file_indicators = 0
            for indicator in medical_indicators:
                if indicator in content:
                    file_indicators += 1
                    total_indicators += 1
            
            if file_indicators > 0:
                compliance_scores[str(py_file)] = file_indicators
                
        except Exception:
            continue
    
    passed = total_indicators >= 10  # Minimum medical compliance indicators
    
    details = {
        "total_files_scanned": len(python_files),
        "medical_indicators_found": total_indicators,
        "files_with_medical_content": len(compliance_scores),
        "compliance_score": min(1.0, total_indicators / 20),  # Scale to 20 indicators
        "compliance_files": compliance_scores
    }
    
    return passed, details


def check_quantum_algorithms() -> Tuple[bool, Dict[str, Any]]:
    """Check for quantum algorithm implementations."""
    src_dir = Path("src")
    python_files = list(src_dir.rglob("*.py"))
    
    quantum_keywords = [
        "quantum",
        "coherence",
        "superposition",
        "entanglement",
        "annealing",
        "quantum_state",
        "quantum_phase",
        "interference"
    ]
    
    quantum_files = {}
    total_quantum_indicators = 0
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read().lower()
            
            file_quantum_count = 0
            for keyword in quantum_keywords:
                file_quantum_count += content.count(keyword)
            
            if file_quantum_count > 0:
                quantum_files[str(py_file)] = file_quantum_count
                total_quantum_indicators += file_quantum_count
                
        except Exception:
            continue
    
    passed = total_quantum_indicators >= 50  # Minimum quantum algorithm indicators
    
    details = {
        "total_files_scanned": len(python_files),
        "quantum_indicators_found": total_quantum_indicators,
        "files_with_quantum_content": len(quantum_files),
        "quantum_score": min(1.0, total_quantum_indicators / 100),  # Scale to 100 indicators
        "quantum_files": dict(list(quantum_files.items())[:10])  # Top 10 files
    }
    
    return passed, details


def run_all_checks() -> Dict[str, Any]:
    """Run all quality validation checks."""
    checks = [
        ("File Structure", check_file_structure),
        ("Python Syntax", check_python_syntax),
        ("Import Resolution", check_imports),
        ("Documentation", check_documentation),
        ("Security Patterns", check_security_patterns),
        ("Medical Compliance", check_medical_compliance),
        ("Quantum Algorithms", check_quantum_algorithms)
    ]
    
    results = {}
    overall_passed = True
    total_time = 0
    
    logger.info("🚀 Starting Quantum-Medical AI Quality Validation")
    
    for check_name, check_func in checks:
        logger.info(f"Running {check_name}...")
        start_time = time.time()
        
        try:
            passed, details = check_func()
            execution_time = time.time() - start_time
            total_time += execution_time
            
            status = "✅ PASSED" if passed else "⚠️ ISSUES"
            logger.info(f"{status} {check_name} ({execution_time:.2f}s)")
            
            results[check_name.lower().replace(" ", "_")] = {
                "name": check_name,
                "passed": passed,
                "execution_time": execution_time,
                "details": details
            }
            
            if not passed:
                overall_passed = False
                
        except Exception as e:
            execution_time = time.time() - start_time
            total_time += execution_time
            
            logger.error(f"💥 {check_name} CRASHED ({execution_time:.2f}s): {str(e)}")
            
            results[check_name.lower().replace(" ", "_")] = {
                "name": check_name,
                "passed": False,
                "execution_time": execution_time,
                "details": {"error": str(e)}
            }
            
            overall_passed = False
    
    # Calculate overall scores
    scores = {}
    for check_name, result in results.items():
        if "score" in str(result.get("details", {})):
            details = result["details"]
            for key, value in details.items():
                if key.endswith("_score") and isinstance(value, (int, float)):
                    scores[key] = value
    
    summary = {
        "overall_passed": overall_passed,
        "total_checks": len(checks),
        "passed_checks": sum(1 for r in results.values() if r["passed"]),
        "failed_checks": sum(1 for r in results.values() if not r["passed"]),
        "total_execution_time": total_time,
        "quality_scores": scores,
        "overall_quality_score": sum(scores.values()) / len(scores) if scores else 0.0,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    logger.info(f"🏁 Quality validation completed: {summary['passed_checks']}/{summary['total_checks']} checks passed")
    
    if overall_passed:
        logger.info("🎉 QUANTUM-MEDICAL AI SYSTEM VALIDATED SUCCESSFULLY!")
    else:
        logger.warning("⚠️ Some quality issues detected - review results")
    
    return {
        "summary": summary,
        "results": results
    }


def main():
    """Main execution function."""
    # Run all quality checks
    validation_results = run_all_checks()
    
    # Save results
    output_file = "quantum_medical_quality_validation.json"
    try:
        with open(output_file, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        logger.info(f"📄 Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
    
    # Print summary
    summary = validation_results["summary"]
    print("\n" + "="*80)
    print("🧬 QUANTUM-MEDICAL AI SYSTEM QUALITY VALIDATION SUMMARY")
    print("="*80)
    print(f"Overall Status: {'✅ VALIDATED' if summary['overall_passed'] else '⚠️ ISSUES DETECTED'}")
    print(f"Checks Passed: {summary['passed_checks']}/{summary['total_checks']}")
    print(f"Overall Quality Score: {summary['overall_quality_score']:.2%}")
    print(f"Execution Time: {summary['total_execution_time']:.2f} seconds")
    print(f"Validation Date: {summary['timestamp']}")
    
    if summary["quality_scores"]:
        print("\n📊 Quality Scores:")
        for score_name, score_value in summary["quality_scores"].items():
            score_display = score_name.replace("_", " ").title()
            print(f"  {score_display}: {score_value:.2%}")
    
    print("="*80)
    
    # Exit with appropriate code
    sys.exit(0 if summary["overall_passed"] else 1)


if __name__ == "__main__":
    main()