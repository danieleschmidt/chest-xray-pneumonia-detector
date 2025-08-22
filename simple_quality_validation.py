"""
Simple Quality Validation for Quantum Medical Research
========================================================

Lightweight quality validation without external dependencies.
Validates core framework structure and functionality.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

class SimpleQualityValidator:
    """Simple quality validator for research framework."""
    
    def __init__(self):
        """Initialize validator."""
        self.results = {}
        self.errors = []
    
    async def validate_all(self):
        """Run all validation checks."""
        
        print("🔍 Starting Simple Quality Validation")
        
        # Check 1: File Structure
        print("📁 Checking file structure...")
        structure_ok = self._check_file_structure()
        self.results["file_structure"] = structure_ok
        
        # Check 2: Import Validation
        print("📦 Validating imports...")
        imports_ok = await self._validate_imports()
        self.results["imports"] = imports_ok
        
        # Check 3: Basic Functionality
        print("⚡ Testing basic functionality...")
        functionality_ok = await self._test_basic_functionality()
        self.results["functionality"] = functionality_ok
        
        # Generate report
        self._generate_report()
        
        return all(self.results.values())
    
    def _check_file_structure(self):
        """Check if all required files exist."""
        
        required_files = [
            "src/research/quantum_medical_research_framework.py",
            "src/research/novel_quantum_medical_algorithms.py", 
            "src/research/robust_quantum_medical_validation.py",
            "src/research/medical_ai_monitoring_system.py",
            "src/research/quantum_scaling_orchestrator.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            self.errors.extend([f"Missing file: {f}" for f in missing_files])
            return False
        
        print("  ✅ All required files present")
        return True
    
    async def _validate_imports(self):
        """Validate that modules can be imported."""
        
        import_tests = [
            ("research.quantum_medical_research_framework", "QuantumMedicalResearchFramework"),
            ("research.novel_quantum_medical_algorithms", "QuantumVariationalMedicalOptimizer"),
            ("research.robust_quantum_medical_validation", "RobustQuantumMedicalValidator"),
            ("research.medical_ai_monitoring_system", "MedicalAIMonitoringSystem"),
            ("research.quantum_scaling_orchestrator", "QuantumScalingOrchestrator")
        ]
        
        success_count = 0
        for module_name, class_name in import_tests:
            try:
                module = __import__(module_name, fromlist=[class_name])
                getattr(module, class_name)
                print(f"  ✅ {module_name}")
                success_count += 1
            except Exception as e:
                print(f"  ❌ {module_name}: {str(e)}")
                self.errors.append(f"Import failed: {module_name} - {str(e)}")
        
        return success_count == len(import_tests)
    
    async def _test_basic_functionality(self):
        """Test basic functionality of key components."""
        
        try:
            # Test 1: Research Framework Creation
            from research.quantum_medical_research_framework import QuantumMedicalResearchFramework
            
            framework = QuantumMedicalResearchFramework(random_seed=42)
            if len(framework.hypotheses) != 3:
                self.errors.append("Incorrect number of hypotheses")
                return False
            
            print("  ✅ Research framework creation")
            
            # Test 2: Basic Algorithm Creation
            from research.novel_quantum_medical_algorithms import QuantumVariationalMedicalOptimizer
            
            optimizer = QuantumVariationalMedicalOptimizer(n_qubits=4, n_layers=2)
            if optimizer.n_qubits != 4 or optimizer.n_layers != 2:
                self.errors.append("QVMO initialization failed")
                return False
            
            print("  ✅ Algorithm creation")
            
            # Test 3: Validator Creation
            from research.robust_quantum_medical_validation import RobustQuantumMedicalValidator
            
            validator = RobustQuantumMedicalValidator()
            if not hasattr(validator, 'circuit_breaker'):
                self.errors.append("Validator initialization failed")
                return False
            
            print("  ✅ Validator creation")
            
            return True
            
        except Exception as e:
            self.errors.append(f"Functionality test failed: {str(e)}")
            return False
    
    def _generate_report(self):
        """Generate validation report."""
        
        total_checks = len(self.results)
        passed_checks = sum(self.results.values())
        
        print("\n" + "="*60)
        print("📊 SIMPLE QUALITY VALIDATION REPORT")
        print("="*60)
        print(f"Total Checks: {total_checks}")
        print(f"Passed: {passed_checks}")
        print(f"Failed: {total_checks - passed_checks}")
        print(f"Success Rate: {passed_checks/total_checks:.1%}")
        
        print("\nCheck Results:")
        for check_name, result in self.results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {check_name}: {status}")
        
        if self.errors:
            print("\nErrors Found:")
            for error in self.errors:
                print(f"  • {error}")
        
        if all(self.results.values()):
            print("\n🎉 ALL CHECKS PASSED!")
            print("Framework structure is valid and ready for development.")
        else:
            print("\n⚠️  SOME CHECKS FAILED")
            print("Please address errors before proceeding.")
        
        print("="*60)

async def main():
    """Run simple quality validation."""
    
    validator = SimpleQualityValidator()
    success = await validator.validate_all()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)