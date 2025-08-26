#!/usr/bin/env python3
"""Security Audit Report Generator.

Analyzes security patterns and generates detailed security audit report.
"""

import ast
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class SecurityAuditor:
    """Security audit analyzer for medical AI systems."""
    
    def __init__(self, project_root: str = "/root/repo"):
        self.project_root = Path(project_root)
        
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security audit report."""
        print("🔒 Generating Security Audit Report")
        print("=" * 50)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'summary': {},
            'findings': {
                'critical': [],
                'high': [],
                'medium': [],
                'low': [],
                'info': []
            },
            'compliance': {},
            'recommendations': []
        }
        
        # Run security analysis
        python_files = list(self.project_root.rglob("*.py"))
        
        for py_file in python_files:
            findings = self.analyze_file_security(py_file)
            for finding in findings:
                severity = finding['severity']
                report['findings'][severity].append(finding)
        
        # Generate summary
        report['summary'] = {
            'total_files_scanned': len(python_files),
            'total_findings': sum(len(findings) for findings in report['findings'].values()),
            'critical_count': len(report['findings']['critical']),
            'high_count': len(report['findings']['high']),
            'medium_count': len(report['findings']['medium']),
            'low_count': len(report['findings']['low']),
            'info_count': len(report['findings']['info'])
        }
        
        # Compliance analysis
        report['compliance'] = self.analyze_compliance()
        
        # Generate recommendations
        report['recommendations'] = self.generate_recommendations(report)
        
        # Calculate security score
        report['security_score'] = self.calculate_security_score(report)
        
        return report
    
    def analyze_file_security(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze security issues in a single file."""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return [{
                'severity': 'medium',
                'type': 'file_access_error',
                'description': f"Could not read file: {str(e)}",
                'file': str(file_path.relative_to(self.project_root)),
                'line': 0
            }]
        
        # Check for hardcoded secrets
        secret_patterns = [
            (r'password\s*=\s*[\'"][^\'"]{8,}[\'"]', 'Potential hardcoded password'),
            (r'secret\s*=\s*[\'"][^\'"]{16,}[\'"]', 'Potential hardcoded secret'),
            (r'key\s*=\s*[\'"][^\'"]{16,}[\'"]', 'Potential hardcoded key'),
            (r'token\s*=\s*[\'"][^\'"]{20,}[\'"]', 'Potential hardcoded token'),
            (r'api[_-]?key\s*=\s*[\'"][^\'"]+[\'"]', 'Potential hardcoded API key')
        ]
        
        for pattern, description in secret_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                findings.append({
                    'severity': 'critical',
                    'type': 'hardcoded_secret',
                    'description': description,
                    'file': str(file_path.relative_to(self.project_root)),
                    'line': line_num,
                    'code': match.group().strip()
                })
        
        # Check for dangerous functions
        dangerous_patterns = [
            (r'\beval\s*\(', 'Use of eval() function - potential code injection', 'high'),
            (r'\bexec\s*\(', 'Use of exec() function - potential code injection', 'high'),
            (r'subprocess\.call\([^)]*shell\s*=\s*True', 'Shell injection vulnerability', 'critical'),
            (r'os\.system\s*\(', 'Use of os.system() - command injection risk', 'high'),
            (r'pickle\.loads?\s*\(', 'Unsafe deserialization with pickle', 'medium'),
            (r'yaml\.load\s*\([^)]*Loader\s*=\s*yaml\.Loader', 'Unsafe YAML loading', 'medium')
        ]
        
        for pattern, description, severity in dangerous_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                findings.append({
                    'severity': severity,
                    'type': 'dangerous_function',
                    'description': description,
                    'file': str(file_path.relative_to(self.project_root)),
                    'line': line_num,
                    'code': match.group().strip()
                })
        
        # Check for SQL injection patterns
        sql_patterns = [
            (r'execute\s*\([^)]*%\s*[^)]*\)', 'Potential SQL injection - string formatting in query'),
            (r'query.*\+.*', 'Potential SQL injection - string concatenation in query'),
            (r'SELECT.*\+.*FROM', 'Potential SQL injection in SELECT statement')
        ]
        
        for pattern, description in sql_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                findings.append({
                    'severity': 'high',
                    'type': 'sql_injection',
                    'description': description,
                    'file': str(file_path.relative_to(self.project_root)),
                    'line': line_num,
                    'code': match.group().strip()
                })
        
        # Check for path traversal vulnerabilities
        path_patterns = [
            (r'open\s*\([^)]*\.\./[^)]*\)', 'Path traversal vulnerability - relative path in file operation'),
            (r'file.*\.\./.*', 'Potential path traversal in file handling')
        ]
        
        for pattern, description in path_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                findings.append({
                    'severity': 'medium',
                    'type': 'path_traversal',
                    'description': description,
                    'file': str(file_path.relative_to(self.project_root)),
                    'line': line_num,
                    'code': match.group().strip()
                })
        
        # Check for weak cryptography
        crypto_patterns = [
            (r'md5\s*\(', 'Use of weak MD5 hash function', 'medium'),
            (r'sha1\s*\(', 'Use of weak SHA1 hash function', 'low'),
            (r'DES\s*\(', 'Use of weak DES encryption', 'high'),
            (r'random\s*\(', 'Use of weak random number generator for security', 'medium')
        ]
        
        for pattern, description, severity in crypto_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                # Only flag as security issue if in security-related context
                context = content[max(0, match.start()-100):match.end()+100]
                if any(keyword in context.lower() for keyword in ['password', 'key', 'encrypt', 'hash', 'auth', 'token']):
                    findings.append({
                        'severity': severity,
                        'type': 'weak_crypto',
                        'description': description,
                        'file': str(file_path.relative_to(self.project_root)),
                        'line': line_num,
                        'code': match.group().strip()
                    })
        
        # Check for positive security patterns
        security_patterns = [
            (r'bcrypt', 'Good: Using bcrypt for password hashing', 'info'),
            (r'scrypt', 'Good: Using scrypt for key derivation', 'info'),
            (r'Fernet', 'Good: Using Fernet for symmetric encryption', 'info'),
            (r'sha256', 'Good: Using SHA-256 hash function', 'info'),
            (r'secrets\.', 'Good: Using secrets module for secure random generation', 'info'),
            (r'cryptography\.', 'Good: Using cryptography library', 'info'),
            (r'validate.*input', 'Good: Input validation detected', 'info'),
            (r'sanitiz', 'Good: Input sanitization detected', 'info')
        ]
        
        for pattern, description, severity in security_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                findings.append({
                    'severity': severity,
                    'type': 'security_pattern',
                    'description': description,
                    'file': str(file_path.relative_to(self.project_root)),
                    'line': line_num,
                    'code': match.group().strip()
                })
        
        return findings
    
    def analyze_compliance(self) -> Dict[str, Any]:
        """Analyze compliance with security standards."""
        compliance = {
            'HIPAA': self.check_hipaa_compliance(),
            'GDPR': self.check_gdpr_compliance(),
            'OWASP': self.check_owasp_compliance()
        }
        
        return compliance
    
    def check_hipaa_compliance(self) -> Dict[str, Any]:
        """Check HIPAA compliance indicators."""
        indicators = {
            'encryption_at_rest': False,
            'encryption_in_transit': False,
            'audit_logging': False,
            'access_controls': False,
            'data_integrity': False,
            'score': 0
        }
        
        # Look for encryption patterns
        crypto_files = list(self.project_root.rglob("*encryption*.py")) + \
                      list(self.project_root.rglob("*crypto*.py")) + \
                      list(self.project_root.rglob("*security*.py"))
        
        if crypto_files:
            indicators['encryption_at_rest'] = True
            indicators['encryption_in_transit'] = True
        
        # Look for audit logging
        audit_files = list(self.project_root.rglob("*audit*.py")) + \
                     list(self.project_root.rglob("*log*.py"))
        
        if audit_files:
            indicators['audit_logging'] = True
        
        # Look for access control patterns
        auth_files = list(self.project_root.rglob("*auth*.py")) + \
                    list(self.project_root.rglob("*permission*.py"))
        
        if auth_files:
            indicators['access_controls'] = True
        
        # Data integrity (look for validation patterns)
        validation_files = list(self.project_root.rglob("*validation*.py")) + \
                          list(self.project_root.rglob("*validator*.py"))
        
        if validation_files:
            indicators['data_integrity'] = True
        
        # Calculate score
        indicators['score'] = sum(1 for v in indicators.values() if v is True) * 20
        
        return indicators
    
    def check_gdpr_compliance(self) -> Dict[str, Any]:
        """Check GDPR compliance indicators."""
        indicators = {
            'data_anonymization': False,
            'consent_management': False,
            'data_deletion': False,
            'privacy_by_design': False,
            'score': 0
        }
        
        # Look for anonymization patterns
        anonym_files = list(self.project_root.rglob("*anonym*.py")) + \
                      list(self.project_root.rglob("*privacy*.py"))
        
        if anonym_files:
            indicators['data_anonymization'] = True
            indicators['privacy_by_design'] = True
        
        # Look for consent patterns
        consent_patterns = ['consent', 'gdpr', 'privacy', 'agreement']
        for pattern in consent_patterns:
            files = list(self.project_root.rglob(f"*{pattern}*.py"))
            if files:
                indicators['consent_management'] = True
                break
        
        # Look for data deletion patterns
        deletion_patterns = ['delete', 'remove', 'purge', 'cleanup']
        for pattern in deletion_patterns:
            files = list(self.project_root.rglob(f"*{pattern}*.py"))
            if files:
                indicators['data_deletion'] = True
                break
        
        indicators['score'] = sum(1 for v in indicators.values() if v is True) * 25
        
        return indicators
    
    def check_owasp_compliance(self) -> Dict[str, Any]:
        """Check OWASP Top 10 compliance."""
        indicators = {
            'injection_prevention': False,
            'broken_authentication': False,
            'sensitive_data_exposure': False,
            'xml_external_entities': False,
            'broken_access_control': False,
            'security_misconfiguration': False,
            'xss_prevention': False,
            'insecure_deserialization': False,
            'vulnerable_components': False,
            'insufficient_logging': False,
            'score': 0
        }
        
        # Check for various OWASP compliance indicators
        # This is a simplified check - real implementation would be more thorough
        
        # Look for input validation (injection prevention)
        validation_files = list(self.project_root.rglob("*validation*.py"))
        if validation_files:
            indicators['injection_prevention'] = True
        
        # Look for authentication mechanisms
        auth_files = list(self.project_root.rglob("*auth*.py"))
        if auth_files:
            indicators['broken_authentication'] = True
        
        # Look for encryption (sensitive data protection)
        crypto_files = list(self.project_root.rglob("*encrypt*.py"))
        if crypto_files:
            indicators['sensitive_data_exposure'] = True
        
        # Look for logging mechanisms
        log_files = list(self.project_root.rglob("*log*.py"))
        if log_files:
            indicators['insufficient_logging'] = True
        
        indicators['score'] = sum(1 for v in indicators.values() if v is True and v is not 0) * 10
        
        return indicators
    
    def generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate security recommendations based on findings."""
        recommendations = []
        
        # Critical findings recommendations
        if report['summary']['critical_count'] > 0:
            recommendations.append("URGENT: Address all critical security findings immediately, especially hardcoded secrets")
        
        # High severity recommendations
        if report['summary']['high_count'] > 0:
            recommendations.append("HIGH PRIORITY: Fix dangerous function usage and potential injection vulnerabilities")
        
        # Compliance recommendations
        hipaa_score = report['compliance']['HIPAA']['score']
        if hipaa_score < 80:
            recommendations.append(f"HIPAA Compliance: Improve security controls (current score: {hipaa_score}/100)")
        
        gdpr_score = report['compliance']['GDPR']['score']
        if gdpr_score < 80:
            recommendations.append(f"GDPR Compliance: Implement privacy protection measures (current score: {gdpr_score}/100)")
        
        # General recommendations
        if report['summary']['medium_count'] > 5:
            recommendations.append("Consider implementing automated security scanning in CI/CD pipeline")
        
        recommendations.extend([
            "Implement regular security code reviews",
            "Use static analysis security testing (SAST) tools",
            "Implement dynamic application security testing (DAST)",
            "Conduct regular penetration testing",
            "Establish security incident response procedures",
            "Implement security awareness training for developers"
        ])
        
        return recommendations
    
    def calculate_security_score(self, report: Dict[str, Any]) -> int:
        """Calculate overall security score."""
        # Start with perfect score
        score = 100
        
        # Deduct points for findings
        score -= report['summary']['critical_count'] * 20  # Critical: -20 points each
        score -= report['summary']['high_count'] * 10      # High: -10 points each
        score -= report['summary']['medium_count'] * 5     # Medium: -5 points each
        score -= report['summary']['low_count'] * 2        # Low: -2 points each
        
        # Add points for positive security patterns (info findings)
        score += min(report['summary']['info_count'] * 2, 20)  # Max 20 points for good patterns
        
        # Adjust based on compliance scores
        compliance_avg = (
            report['compliance']['HIPAA']['score'] +
            report['compliance']['GDPR']['score'] +
            report['compliance']['OWASP']['score']
        ) / 3
        
        # Weighted average with compliance score
        final_score = int((score * 0.7) + (compliance_avg * 0.3))
        
        return max(0, min(100, final_score))
    
    def save_report(self, report: Dict[str, Any], filename: str = "security_audit_report.json"):
        """Save security report to file."""
        output_file = self.project_root / filename
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return output_file


def main():
    """Generate and display security audit report."""
    auditor = SecurityAuditor()
    report = auditor.generate_security_report()
    
    # Display summary
    print(f"\n🔒 SECURITY AUDIT SUMMARY")
    print("=" * 50)
    print(f"Files Scanned: {report['summary']['total_files_scanned']}")
    print(f"Total Findings: {report['summary']['total_findings']}")
    print(f"🚨 Critical: {report['summary']['critical_count']}")
    print(f"⚠️  High: {report['summary']['high_count']}")
    print(f"📋 Medium: {report['summary']['medium_count']}")
    print(f"📝 Low: {report['summary']['low_count']}")
    print(f"ℹ️  Info: {report['summary']['info_count']}")
    print(f"🛡️  Security Score: {report['security_score']}/100")
    
    print(f"\n📊 COMPLIANCE SCORES:")
    print(f"HIPAA: {report['compliance']['HIPAA']['score']}/100")
    print(f"GDPR: {report['compliance']['GDPR']['score']}/100")
    print(f"OWASP: {report['compliance']['OWASP']['score']}/100")
    
    if report['summary']['critical_count'] > 0 or report['summary']['high_count'] > 0:
        print(f"\n🚨 URGENT SECURITY ISSUES DETECTED!")
        print("Review critical and high severity findings immediately.")
    
    # Save report
    output_file = auditor.save_report(report)
    print(f"\n📄 Detailed security report saved to: {output_file}")


if __name__ == "__main__":
    main()