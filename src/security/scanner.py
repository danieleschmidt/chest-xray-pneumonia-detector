"""
Comprehensive security scanning and SBOM generation.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib
import logging

logger = logging.getLogger(__name__)


class SecurityScanner:
    """Comprehensive security scanner for the application."""
    
    def __init__(self, output_dir: str = ".artifacts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
    
    def run_all_scans(self) -> Dict[str, Any]:
        """Run all security scans and return consolidated results."""
        logger.info("Starting comprehensive security scan")
        
        # Run individual scans
        self.results['bandit'] = self.run_bandit_scan()
        self.results['safety'] = self.run_safety_scan()
        self.results['pip_audit'] = self.run_pip_audit()
        self.results['secrets'] = self.scan_for_secrets()
        self.results['dependencies'] = self.analyze_dependencies()
        self.results['sbom'] = self.generate_sbom()
        
        # Generate summary
        self.results['summary'] = self.generate_summary()
        
        # Save consolidated report
        self.save_report()
        
        logger.info("Security scan completed")
        return self.results
    
    def run_bandit_scan(self) -> Dict[str, Any]:
        """Run Bandit security scanner."""
        logger.info("Running Bandit security scan")
        
        try:
            cmd = [
                sys.executable, "-m", "bandit",
                "-r", "src/",
                "-f", "json",
                "-o", str(self.output_dir / "bandit-report.json")
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Bandit returns non-zero exit code when issues are found
            if result.returncode in [0, 1]:
                # Load the JSON report
                report_file = self.output_dir / "bandit-report.json"
                if report_file.exists():
                    with open(report_file, 'r') as f:
                        bandit_data = json.load(f)
                    
                    return {
                        'status': 'success',
                        'issues_found': len(bandit_data.get('results', [])),
                        'severity_breakdown': self._analyze_bandit_severity(bandit_data),
                        'report_file': str(report_file)
                    }
            
            return {
                'status': 'error',
                'error': result.stderr,
                'issues_found': 0
            }
            
        except Exception as e:
            logger.error(f"Bandit scan failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'issues_found': 0
            }
    
    def run_safety_scan(self) -> Dict[str, Any]:
        """Run Safety vulnerability scanner."""
        logger.info("Running Safety vulnerability scan")
        
        try:
            cmd = [
                sys.executable, "-m", "safety", "check",
                "--json",
                "--output", str(self.output_dir / "safety-report.json")
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return {
                    'status': 'success',
                    'vulnerabilities_found': 0,
                    'report_file': str(self.output_dir / "safety-report.json")
                }
            else:
                # Parse JSON output for vulnerability details
                try:
                    vulnerabilities = json.loads(result.stdout)
                    return {
                        'status': 'issues_found',
                        'vulnerabilities_found': len(vulnerabilities),
                        'vulnerabilities': vulnerabilities[:5],  # First 5 for summary
                        'report_file': str(self.output_dir / "safety-report.json")
                    }
                except json.JSONDecodeError:
                    return {
                        'status': 'error',
                        'error': result.stderr,
                        'vulnerabilities_found': 0
                    }
                    
        except Exception as e:
            logger.error(f"Safety scan failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'vulnerabilities_found': 0
            }
    
    def run_pip_audit(self) -> Dict[str, Any]:
        """Run pip-audit for dependency vulnerabilities."""
        logger.info("Running pip-audit dependency scan")
        
        try:
            cmd = [
                sys.executable, "-m", "pip_audit",
                "--format=json",
                "--output=" + str(self.output_dir / "pip-audit-report.json")
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Load and analyze results
            report_file = self.output_dir / "pip-audit-report.json"
            if report_file.exists():
                with open(report_file, 'r') as f:
                    audit_data = json.load(f)
                
                vulnerabilities = audit_data.get('vulnerabilities', [])
                return {
                    'status': 'success',
                    'vulnerabilities_found': len(vulnerabilities),
                    'critical_count': sum(1 for v in vulnerabilities if v.get('severity') == 'CRITICAL'),
                    'high_count': sum(1 for v in vulnerabilities if v.get('severity') == 'HIGH'),
                    'report_file': str(report_file)
                }
            
            return {
                'status': 'error',
                'error': 'Report file not generated',
                'vulnerabilities_found': 0
            }
            
        except Exception as e:
            logger.error(f"pip-audit scan failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'vulnerabilities_found': 0
            }
    
    def scan_for_secrets(self) -> Dict[str, Any]:
        """Scan for hardcoded secrets and sensitive information."""
        logger.info("Scanning for hardcoded secrets")
        
        secret_patterns = {
            'aws_access_key': r'AKIA[0-9A-Z]{16}',
            'aws_secret_key': r'[0-9a-zA-Z/+]{40}',
            'api_key': r'[a-zA-Z0-9_-]{32,}',
            'password': r'password\s*=\s*["\'][^"\']+["\']',
            'private_key': r'-----BEGIN.*PRIVATE KEY-----',
            'jwt_token': r'eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+',
        }
        
        found_secrets = []
        
        try:
            # Scan Python files
            for py_file in Path('src').rglob('*.py'):
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                
                for secret_type, pattern in secret_patterns.items():
                    import re
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    
                    for match in matches:
                        found_secrets.append({
                            'type': secret_type,
                            'file': str(py_file),
                            'line': content[:match.start()].count('\n') + 1,
                            'pattern': pattern[:20] + '...' if len(pattern) > 20 else pattern
                        })
            
            # Save results
            secrets_file = self.output_dir / "secrets-scan.json"
            with open(secrets_file, 'w') as f:
                json.dump(found_secrets, f, indent=2)
            
            return {
                'status': 'success',
                'secrets_found': len(found_secrets),
                'secrets': found_secrets,
                'report_file': str(secrets_file)
            }
            
        except Exception as e:
            logger.error(f"Secrets scan failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'secrets_found': 0
            }
    
    def analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze project dependencies."""
        logger.info("Analyzing project dependencies")
        
        try:
            # Read requirements files
            req_files = ['requirements.txt', 'requirements-dev.txt', 'requirements-security.txt']
            dependencies = {}
            
            for req_file in req_files:
                if Path(req_file).exists():
                    with open(req_file, 'r') as f:
                        deps = []
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                deps.append(line)
                        dependencies[req_file] = deps
            
            # Generate dependency tree
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "list", "--format=json"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    installed_packages = json.loads(result.stdout)
                else:
                    installed_packages = []
            except Exception:
                installed_packages = []
            
            # Save analysis
            dep_analysis = {
                'requirements_files': dependencies,
                'installed_packages': installed_packages,
                'total_dependencies': len(installed_packages),
                'analysis_date': datetime.now().isoformat()
            }
            
            analysis_file = self.output_dir / "dependency-analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(dep_analysis, f, indent=2)
            
            return {
                'status': 'success',
                'total_dependencies': len(installed_packages),
                'requirements_files': list(dependencies.keys()),
                'report_file': str(analysis_file)
            }
            
        except Exception as e:
            logger.error(f"Dependency analysis failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'total_dependencies': 0
            }
    
    def generate_sbom(self) -> Dict[str, Any]:
        """Generate Software Bill of Materials (SBOM)."""
        logger.info("Generating Software Bill of Materials (SBOM)")
        
        try:
            # Get installed packages
            result = subprocess.run([
                sys.executable, "-m", "pip", "list", "--format=json"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception("Failed to get package list")
            
            packages = json.loads(result.stdout)
            
            # Create SBOM structure
            sbom = {
                'bomFormat': 'CycloneDX',
                'specVersion': '1.4',
                'serialNumber': f'urn:uuid:{self._generate_uuid()}',
                'version': 1,
                'metadata': {
                    'timestamp': datetime.now().isoformat() + 'Z',
                    'tools': [{
                        'vendor': 'chest-xray-pneumonia-detector',
                        'name': 'security-scanner',
                        'version': '0.2.0'
                    }],
                    'component': {
                        'type': 'application',
                        'bom-ref': 'chest-xray-pneumonia-detector',
                        'name': 'chest-xray-pneumonia-detector',
                        'version': '0.2.0',
                        'description': 'AI-powered pneumonia detection from chest X-ray images'
                    }
                },
                'components': []
            }
            
            # Add package components
            for package in packages:
                component = {
                    'type': 'library',
                    'bom-ref': f"{package['name']}@{package['version']}",
                    'name': package['name'],
                    'version': package['version'],
                    'purl': f"pkg:pypi/{package['name']}@{package['version']}"
                }
                sbom['components'].append(component)
            
            # Save SBOM
            sbom_file = self.output_dir / "sbom.json"
            with open(sbom_file, 'w') as f:
                json.dump(sbom, f, indent=2)
            
            return {
                'status': 'success',
                'components_count': len(packages),
                'sbom_format': 'CycloneDX',
                'report_file': str(sbom_file)
            }
            
        except Exception as e:
            logger.error(f"SBOM generation failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'components_count': 0
            }
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate security scan summary."""
        total_issues = 0
        critical_issues = 0
        
        # Count issues from different scans
        if self.results.get('bandit', {}).get('status') == 'success':
            total_issues += self.results['bandit'].get('issues_found', 0)
        
        if self.results.get('safety', {}).get('status') == 'issues_found':
            safety_issues = self.results['safety'].get('vulnerabilities_found', 0)
            total_issues += safety_issues
            critical_issues += safety_issues  # Treat all safety issues as critical
        
        if self.results.get('pip_audit', {}).get('status') == 'success':
            total_issues += self.results['pip_audit'].get('vulnerabilities_found', 0)
            critical_issues += self.results['pip_audit'].get('critical_count', 0)
        
        if self.results.get('secrets', {}).get('status') == 'success':
            secrets_found = self.results['secrets'].get('secrets_found', 0)
            total_issues += secrets_found
            critical_issues += secrets_found  # Treat all secrets as critical
        
        # Determine overall risk level
        if critical_issues > 0:
            risk_level = 'HIGH'
        elif total_issues > 5:
            risk_level = 'MEDIUM'
        elif total_issues > 0:
            risk_level = 'LOW'
        else:
            risk_level = 'MINIMAL'
        
        return {
            'scan_date': datetime.now().isoformat(),
            'total_issues': total_issues,
            'critical_issues': critical_issues,
            'risk_level': risk_level,
            'scans_completed': len([r for r in self.results.values() if isinstance(r, dict) and r.get('status') in ['success', 'issues_found']]),
            'recommendations': self._generate_recommendations()
        }
    
    def save_report(self):
        """Save consolidated security report."""
        report_file = self.output_dir / "security-report.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Also create a human-readable summary
        summary_file = self.output_dir / "security-summary.txt"
        with open(summary_file, 'w') as f:
            f.write("SECURITY SCAN SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            summary = self.results.get('summary', {})
            f.write(f"Scan Date: {summary.get('scan_date', 'Unknown')}\n")
            f.write(f"Risk Level: {summary.get('risk_level', 'Unknown')}\n")
            f.write(f"Total Issues: {summary.get('total_issues', 0)}\n")
            f.write(f"Critical Issues: {summary.get('critical_issues', 0)}\n\n")
            
            # Add details for each scan
            for scan_type, result in self.results.items():
                if scan_type != 'summary' and isinstance(result, dict):
                    f.write(f"{scan_type.upper()} SCAN:\n")
                    f.write(f"Status: {result.get('status', 'Unknown')}\n")
                    
                    if scan_type == 'bandit':
                        f.write(f"Issues Found: {result.get('issues_found', 0)}\n")
                    elif scan_type == 'safety':
                        f.write(f"Vulnerabilities: {result.get('vulnerabilities_found', 0)}\n")
                    elif scan_type == 'pip_audit':
                        f.write(f"Vulnerabilities: {result.get('vulnerabilities_found', 0)}\n")
                    elif scan_type == 'secrets':
                        f.write(f"Secrets Found: {result.get('secrets_found', 0)}\n")
                    
                    f.write("\n")
        
        logger.info(f"Security report saved to {report_file}")
        logger.info(f"Security summary saved to {summary_file}")
    
    def _analyze_bandit_severity(self, bandit_data: Dict) -> Dict[str, int]:
        """Analyze Bandit results by severity."""
        severity_count = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0}
        
        for result in bandit_data.get('results', []):
            severity = result.get('issue_severity', 'MEDIUM')
            if severity in severity_count:
                severity_count[severity] += 1
        
        return severity_count
    
    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations based on scan results."""
        recommendations = []
        
        if self.results.get('secrets', {}).get('secrets_found', 0) > 0:
            recommendations.append("Remove hardcoded secrets and use environment variables or secret management systems")
        
        if self.results.get('safety', {}).get('vulnerabilities_found', 0) > 0:
            recommendations.append("Update dependencies with known security vulnerabilities")
        
        if self.results.get('bandit', {}).get('issues_found', 0) > 0:
            recommendations.append("Review and fix code security issues identified by Bandit")
        
        if self.results.get('pip_audit', {}).get('critical_count', 0) > 0:
            recommendations.append("Immediately update packages with critical vulnerabilities")
        
        if not recommendations:
            recommendations.append("Continue regular security scanning and monitoring")
        
        return recommendations
    
    def _generate_uuid(self) -> str:
        """Generate a UUID for SBOM."""
        import uuid
        return str(uuid.uuid4())


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive security scan')
    parser.add_argument('--output-dir', default='.artifacts', help='Output directory for reports')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run security scan
    scanner = SecurityScanner(args.output_dir)
    results = scanner.run_all_scans()
    
    # Print summary
    summary = results.get('summary', {})
    print(f"\nSecurity Scan Complete!")
    print(f"Risk Level: {summary.get('risk_level', 'Unknown')}")
    print(f"Total Issues: {summary.get('total_issues', 0)}")
    print(f"Critical Issues: {summary.get('critical_issues', 0)}")
    print(f"Reports saved to: {args.output_dir}")
    
    # Exit with non-zero code if critical issues found
    if summary.get('critical_issues', 0) > 0:
        sys.exit(1)
    elif summary.get('total_issues', 0) > 0:
        sys.exit(2)  # Warning level
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()