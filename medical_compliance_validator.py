#!/usr/bin/env python3
"""Medical Compliance Validator for Healthcare AI Systems"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import hashlib
import datetime


@dataclass
class ComplianceCheck:
    """Represents a compliance check result"""
    name: str
    status: str  # "passed", "failed", "warning"
    score: float
    details: str
    recommendations: List[str]


class MedicalComplianceValidator:
    """Comprehensive medical compliance validation for AI systems"""
    
    def __init__(self, src_dir: str = "src"):
        self.src_dir = Path(src_dir)
        self.checks = []
        self.overall_score = 0.0
        
    def check_hipaa_compliance(self) -> ComplianceCheck:
        """Check HIPAA compliance patterns"""
        hipaa_patterns = {
            "encryption": r"encrypt|crypt|cipher|aes|rsa",
            "access_control": r"auth|login|permission|role|access",
            "audit_logging": r"audit|log|track|monitor",
            "data_minimization": r"anonymiz|pseudonym|redact|mask",
            "secure_transmission": r"ssl|tls|https|secure",
            "data_integrity": r"hash|checksum|verify|validate",
            "business_associate": r"agreement|contract|compliance",
            "breach_notification": r"breach|incident|alert|notify"
        }
        
        found_patterns = {}
        total_files = 0
        
        for file_path in self.src_dir.rglob("*.py"):
            total_files += 1
            try:
                content = file_path.read_text(encoding='utf-8').lower()
                for pattern_name, pattern in hipaa_patterns.items():
                    if re.search(pattern, content):
                        if pattern_name not in found_patterns:
                            found_patterns[pattern_name] = []
                        found_patterns[pattern_name].append(str(file_path))
            except Exception:
                continue
        
        compliance_score = len(found_patterns) / len(hipaa_patterns)
        
        recommendations = []
        missing_patterns = set(hipaa_patterns.keys()) - set(found_patterns.keys())
        
        for missing in missing_patterns:
            if missing == "encryption":
                recommendations.append("Implement PHI encryption at rest and in transit")
            elif missing == "access_control":
                recommendations.append("Add role-based access control for medical data")
            elif missing == "audit_logging":
                recommendations.append("Implement comprehensive audit logging")
            elif missing == "data_minimization":
                recommendations.append("Add data anonymization/pseudonymization")
        
        status = "passed" if compliance_score >= 0.7 else "failed"
        
        return ComplianceCheck(
            name="HIPAA Compliance",
            status=status,
            score=compliance_score,
            details=f"Found {len(found_patterns)}/{len(hipaa_patterns)} compliance patterns in {total_files} files",
            recommendations=recommendations
        )
    
    def check_fda_21cfr820_compliance(self) -> ComplianceCheck:
        """Check FDA 21 CFR 820 (Quality System Regulation) compliance"""
        qsr_patterns = {
            "design_controls": r"design.*control|specification|requirement",
            "risk_management": r"risk.*manag|hazard|mitigation",
            "validation": r"validat|verif|test.*protocol",
            "configuration_management": r"version|config|change.*control",
            "document_control": r"document.*control|record.*keeping",
            "training": r"training|competency|qualification",
            "corrective_action": r"corrective.*action|capa|non.*conformance",
            "management_responsibility": r"management.*review|quality.*policy"
        }
        
        found_patterns = {}
        
        # Check both source code and documentation
        search_paths = [
            self.src_dir,
            Path("docs") if Path("docs").exists() else None,
            Path(".") if Path("README.md").exists() else None
        ]
        
        for search_path in filter(None, search_paths):
            for file_path in search_path.rglob("*"):
                if file_path.suffix in [".py", ".md", ".rst", ".txt"]:
                    try:
                        content = file_path.read_text(encoding='utf-8').lower()
                        for pattern_name, pattern in qsr_patterns.items():
                            if re.search(pattern, content):
                                if pattern_name not in found_patterns:
                                    found_patterns[pattern_name] = []
                                found_patterns[pattern_name].append(str(file_path))
                    except Exception:
                        continue
        
        compliance_score = len(found_patterns) / len(qsr_patterns)
        
        recommendations = []
        missing_patterns = set(qsr_patterns.keys()) - set(found_patterns.keys())
        
        for missing in missing_patterns:
            if missing == "design_controls":
                recommendations.append("Implement design controls documentation")
            elif missing == "risk_management":
                recommendations.append("Add risk management procedures")
            elif missing == "validation":
                recommendations.append("Enhance validation and verification protocols")
        
        status = "passed" if compliance_score >= 0.6 else "warning" if compliance_score >= 0.4 else "failed"
        
        return ComplianceCheck(
            name="FDA 21 CFR 820 Compliance",
            status=status,
            score=compliance_score,
            details=f"Found {len(found_patterns)}/{len(qsr_patterns)} QSR patterns",
            recommendations=recommendations
        )
    
    def check_gdpr_compliance(self) -> ComplianceCheck:
        """Check GDPR compliance for AI systems"""
        gdpr_patterns = {
            "data_protection": r"data.*protection|privacy|gdpr",
            "consent_management": r"consent|agree|opt.*in|opt.*out",
            "right_to_deletion": r"delete|remove|forget|erasure",
            "data_portability": r"export|download|portability",
            "privacy_by_design": r"privacy.*design|data.*minimization",
            "dpo": r"data.*protection.*officer|dpo",
            "lawful_basis": r"lawful.*basis|legitimate.*interest",
            "data_subject_rights": r"subject.*rights|access.*request"
        }
        
        found_patterns = {}
        
        for file_path in self.src_dir.rglob("*.py"):
            try:
                content = file_path.read_text(encoding='utf-8').lower()
                for pattern_name, pattern in gdpr_patterns.items():
                    if re.search(pattern, content):
                        if pattern_name not in found_patterns:
                            found_patterns[pattern_name] = []
                        found_patterns[pattern_name].append(str(file_path))
            except Exception:
                continue
        
        compliance_score = len(found_patterns) / len(gdpr_patterns)
        
        recommendations = []
        missing_patterns = set(gdpr_patterns.keys()) - set(found_patterns.keys())
        
        for missing in missing_patterns:
            if missing == "consent_management":
                recommendations.append("Implement consent management system")
            elif missing == "right_to_deletion":
                recommendations.append("Add data deletion capabilities")
            elif missing == "data_portability":
                recommendations.append("Implement data export functionality")
        
        status = "passed" if compliance_score >= 0.5 else "warning" if compliance_score >= 0.3 else "failed"
        
        return ComplianceCheck(
            name="GDPR Compliance",
            status=status,
            score=compliance_score,
            details=f"Found {len(found_patterns)}/{len(gdpr_patterns)} GDPR patterns",
            recommendations=recommendations
        )
    
    def check_ai_model_validation(self) -> ComplianceCheck:
        """Check AI model validation and testing patterns"""
        validation_patterns = {
            "model_testing": r"test.*model|model.*test|unit.*test",
            "performance_metrics": r"accuracy|precision|recall|f1.*score|auc",
            "bias_testing": r"bias|fairness|equity|demographic",
            "robustness_testing": r"robust|adversarial|noise|perturbation",
            "explainability": r"explain|interpret|grad.*cam|shap|lime",
            "cross_validation": r"cross.*validation|k.*fold|stratified",
            "data_quality": r"data.*quality|data.*validation|outlier",
            "model_monitoring": r"drift|monitoring|performance.*tracking"
        }
        
        found_patterns = {}
        
        for file_path in self.src_dir.rglob("*.py"):
            try:
                content = file_path.read_text(encoding='utf-8').lower()
                for pattern_name, pattern in validation_patterns.items():
                    if re.search(pattern, content):
                        if pattern_name not in found_patterns:
                            found_patterns[pattern_name] = []
                        found_patterns[pattern_name].append(str(file_path))
            except Exception:
                continue
        
        compliance_score = len(found_patterns) / len(validation_patterns)
        
        recommendations = []
        missing_patterns = set(validation_patterns.keys()) - set(found_patterns.keys())
        
        for missing in missing_patterns:
            if missing == "bias_testing":
                recommendations.append("Implement bias and fairness testing")
            elif missing == "explainability":
                recommendations.append("Add model explainability features")
            elif missing == "robustness_testing":
                recommendations.append("Add robustness and adversarial testing")
        
        status = "passed" if compliance_score >= 0.7 else "warning" if compliance_score >= 0.5 else "failed"
        
        return ComplianceCheck(
            name="AI Model Validation",
            status=status,
            score=compliance_score,
            details=f"Found {len(found_patterns)}/{len(validation_patterns)} validation patterns",
            recommendations=recommendations
        )
    
    def check_medical_device_software(self) -> ComplianceCheck:
        """Check medical device software standards (IEC 62304)"""
        medical_sw_patterns = {
            "software_lifecycle": r"lifecycle|development.*process|sdlc",
            "risk_classification": r"risk.*class|safety.*class|medical.*device",
            "software_verification": r"verification|v.*v|software.*testing",
            "software_validation": r"validation|clinical.*validation|user.*acceptance",
            "configuration_management": r"configuration|version.*control|change.*management",
            "problem_resolution": r"problem.*resolution|incident|defect",
            "software_maintenance": r"maintenance|update|patch|support",
            "documentation": r"documentation|specification|design.*document"
        }
        
        found_patterns = {}
        
        # Check source code and documentation
        search_paths = [self.src_dir]
        if Path("docs").exists():
            search_paths.append(Path("docs"))
        
        for search_path in search_paths:
            for file_path in search_path.rglob("*"):
                if file_path.suffix in [".py", ".md", ".rst", ".txt"]:
                    try:
                        content = file_path.read_text(encoding='utf-8').lower()
                        for pattern_name, pattern in medical_sw_patterns.items():
                            if re.search(pattern, content):
                                if pattern_name not in found_patterns:
                                    found_patterns[pattern_name] = []
                                found_patterns[pattern_name].append(str(file_path))
                    except Exception:
                        continue
        
        compliance_score = len(found_patterns) / len(medical_sw_patterns)
        
        recommendations = []
        missing_patterns = set(medical_sw_patterns.keys()) - set(found_patterns.keys())
        
        for missing in missing_patterns:
            if missing == "software_lifecycle":
                recommendations.append("Document software development lifecycle")
            elif missing == "risk_classification":
                recommendations.append("Classify software risk according to IEC 62304")
            elif missing == "clinical_validation":
                recommendations.append("Add clinical validation procedures")
        
        status = "passed" if compliance_score >= 0.6 else "warning" if compliance_score >= 0.4 else "failed"
        
        return ComplianceCheck(
            name="Medical Device Software (IEC 62304)",
            status=status,
            score=compliance_score,
            details=f"Found {len(found_patterns)}/{len(medical_sw_patterns)} medical software patterns",
            recommendations=recommendations
        )
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        # Run all compliance checks
        self.checks = [
            self.check_hipaa_compliance(),
            self.check_fda_21cfr820_compliance(),
            self.check_gdpr_compliance(),
            self.check_ai_model_validation(),
            self.check_medical_device_software()
        ]
        
        # Calculate overall score
        self.overall_score = sum(check.score for check in self.checks) / len(self.checks)
        
        # Determine overall status
        failed_checks = [check for check in self.checks if check.status == "failed"]
        warning_checks = [check for check in self.checks if check.status == "warning"]
        
        if len(failed_checks) > 2:
            overall_status = "CRITICAL"
        elif len(failed_checks) > 0:
            overall_status = "NEEDS_ATTENTION"
        elif len(warning_checks) > 2:
            overall_status = "REVIEW_REQUIRED"
        else:
            overall_status = "COMPLIANT"
        
        # Generate report
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "overall_status": overall_status,
            "overall_score": self.overall_score,
            "compliance_grade": self._get_compliance_grade(self.overall_score),
            "checks": [
                {
                    "name": check.name,
                    "status": check.status,
                    "score": check.score,
                    "details": check.details,
                    "recommendations": check.recommendations
                }
                for check in self.checks
            ],
            "summary": {
                "total_checks": len(self.checks),
                "passed": len([c for c in self.checks if c.status == "passed"]),
                "warnings": len([c for c in self.checks if c.status == "warning"]),
                "failed": len([c for c in self.checks if c.status == "failed"])
            },
            "priority_recommendations": self._get_priority_recommendations()
        }
        
        return report
    
    def _get_compliance_grade(self, score: float) -> str:
        """Convert numerical score to letter grade"""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"
    
    def _get_priority_recommendations(self) -> List[str]:
        """Get priority recommendations based on failed checks"""
        priority_recs = []
        
        for check in self.checks:
            if check.status == "failed" and check.recommendations:
                priority_recs.extend(check.recommendations[:2])  # Top 2 per failed check
        
        return priority_recs[:5]  # Top 5 overall
    
    def save_report(self, filename: str = "medical_compliance_report.json") -> None:
        """Save compliance report to file"""
        report = self.generate_compliance_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📋 Compliance report saved to {filename}")
    
    def print_summary(self) -> None:
        """Print compliance summary to console"""
        report = self.generate_compliance_report()
        
        print("🏥 Medical Compliance Validation Report")
        print("=" * 50)
        print(f"Overall Status: {report['overall_status']}")
        print(f"Compliance Grade: {report['compliance_grade']} ({report['overall_score']:.1%})")
        print()
        
        for check in self.checks:
            status_emoji = "✅" if check.status == "passed" else "⚠️" if check.status == "warning" else "❌"
            print(f"{status_emoji} {check.name}: {check.status.upper()} ({check.score:.1%})")
        
        if report["priority_recommendations"]:
            print("\n💡 Priority Recommendations:")
            for i, rec in enumerate(report["priority_recommendations"], 1):
                print(f"  {i}. {rec}")
        
        print(f"\n📊 Summary: {report['summary']['passed']} passed, {report['summary']['warnings']} warnings, {report['summary']['failed']} failed")


def main():
    """Main entry point for compliance validator"""
    validator = MedicalComplianceValidator()
    
    try:
        validator.print_summary()
        validator.save_report()
        
        report = validator.generate_compliance_report()
        
        if report["overall_status"] in ["CRITICAL", "NEEDS_ATTENTION"]:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"❌ Error running compliance validation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()