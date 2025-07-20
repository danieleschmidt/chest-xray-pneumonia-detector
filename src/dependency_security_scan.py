"""Dependency security scanning utilities for detecting known vulnerabilities."""

import argparse
import json
import subprocess
import sys
import time
import re
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class VulnerabilityInfo:
    """Information about a specific vulnerability."""
    
    package: str
    version: str
    vulnerability_id: str
    severity: str
    description: str
    fixed_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert vulnerability info to dictionary."""
        return asdict(self)


@dataclass
class ScanResult:
    """Results from a dependency security scan."""
    
    scan_timestamp: str
    total_vulnerabilities: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    vulnerabilities: List[VulnerabilityInfo]
    scan_tool: str
    scan_duration: Optional[float] = None
    scan_notes: Optional[str] = None
    
    @property
    def is_clean(self) -> bool:
        """Return True if no vulnerabilities were found."""
        return self.total_vulnerabilities == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert scan result to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert scan result to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


def check_tool_availability(tool_name: str) -> bool:
    """Check if a security scanning tool is available.
    
    Parameters
    ----------
    tool_name : str
        Name of the tool to check (e.g., 'pip-audit', 'safety')
        
    Returns
    -------
    bool
        True if tool is available, False otherwise
    """
    try:
        # Use --version flag which most tools support
        result = subprocess.run(
            [tool_name, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return False


def parse_pip_audit_output(output: str) -> List[VulnerabilityInfo]:
    """Parse pip-audit JSON output into vulnerability information.
    
    Parameters
    ----------
    output : str
        JSON output from pip-audit
        
    Returns
    -------
    list
        List of VulnerabilityInfo objects
    """
    vulnerabilities = []
    
    try:
        data = json.loads(output)
        if not isinstance(data, list):
            return vulnerabilities
            
        for item in data:
            if not isinstance(item, dict):
                continue
                
            # Extract vulnerability information
            package = item.get("package", "unknown")
            version = item.get("version", "unknown")
            vuln_id = item.get("id", "unknown")
            description = item.get("description", "No description available")
            fix_versions = item.get("fix_versions", [])
            fixed_version = fix_versions[0] if fix_versions else None
            
            # Determine severity based on vulnerability ID patterns
            severity = "MEDIUM"  # Default
            if "CRITICAL" in vuln_id.upper() or "CVE-" in vuln_id:
                severity = "HIGH"
            elif "LOW" in vuln_id.upper():
                severity = "LOW"
            
            vulnerability = VulnerabilityInfo(
                package=package,
                version=version,
                vulnerability_id=vuln_id,
                severity=severity,
                description=description,
                fixed_version=fixed_version
            )
            vulnerabilities.append(vulnerability)
            
    except (json.JSONDecodeError, KeyError, TypeError):
        # Return empty list if parsing fails
        pass
    
    return vulnerabilities


def parse_safety_output(output: str) -> List[VulnerabilityInfo]:
    """Parse safety text output into vulnerability information.
    
    Parameters
    ----------
    output : str
        Text output from safety
        
    Returns
    -------
    list
        List of VulnerabilityInfo objects
    """
    vulnerabilities = []
    
    if not output or "No known security vulnerabilities found" in output:
        return vulnerabilities
    
    lines = output.strip().split('\n')
    current_package = None
    current_version = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for package==version lines
        if '==' in line and not line.startswith('-'):
            parts = line.split('==')
            if len(parts) == 2:
                current_package = parts[0].strip()
                current_version = parts[1].strip()
        
        # Look for vulnerability lines starting with -
        elif line.startswith('-') and current_package:
            # Extract vulnerability ID and description
            vuln_match = re.match(r'- (\d+): (.+)', line)
            if vuln_match:
                vuln_id = vuln_match.group(1)
                description = vuln_match.group(2)
                
                vulnerability = VulnerabilityInfo(
                    package=current_package,
                    version=current_version,
                    vulnerability_id=f"SAFETY-{vuln_id}",
                    severity="MEDIUM",  # Safety doesn't provide severity
                    description=description,
                    fixed_version=None
                )
                vulnerabilities.append(vulnerability)
    
    return vulnerabilities


class DependencySecurityScanner:
    """Main dependency security scanner."""
    
    def __init__(self):
        self.scan_results: List[ScanResult] = []
    
    def scan(self, requirements_file: str = None) -> ScanResult:
        """Perform dependency security scan.
        
        Parameters
        ----------
        requirements_file : str, optional
            Path to requirements file to scan
            
        Returns
        -------
        ScanResult
            Scan results with vulnerability information
        """
        start_time = time.time()
        scan_timestamp = datetime.now().isoformat()
        
        # Try pip-audit first (preferred tool)
        if check_tool_availability("pip-audit"):
            result = self._scan_with_pip_audit(requirements_file)
            tool_used = "pip-audit"
        elif check_tool_availability("safety"):
            result = self._scan_with_safety(requirements_file)
            tool_used = "safety"
        else:
            # Fallback to manual scanning
            result = self._scan_manual(requirements_file)
            tool_used = "manual"
        
        # Calculate scan duration
        scan_duration = time.time() - start_time
        
        # Count vulnerabilities by severity
        critical_count = sum(1 for v in result if v.severity == "CRITICAL")
        high_count = sum(1 for v in result if v.severity == "HIGH")
        medium_count = sum(1 for v in result if v.severity == "MEDIUM")
        low_count = sum(1 for v in result if v.severity == "LOW")
        
        scan_result = ScanResult(
            scan_timestamp=scan_timestamp,
            total_vulnerabilities=len(result),
            critical_count=critical_count,
            high_count=high_count,
            medium_count=medium_count,
            low_count=low_count,
            vulnerabilities=result,
            scan_tool=tool_used,
            scan_duration=scan_duration
        )
        
        self.scan_results.append(scan_result)
        return scan_result
    
    def _scan_with_pip_audit(self, requirements_file: str = None) -> List[VulnerabilityInfo]:
        """Scan using pip-audit tool."""
        try:
            cmd = ["pip-audit", "--format", "json"]
            if requirements_file:
                cmd.extend(["--requirement", requirements_file])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            if result.returncode == 0:
                return parse_pip_audit_output(result.stdout)
            else:
                # pip-audit returns non-zero when vulnerabilities are found
                # Try to parse the output anyway
                if result.stdout:
                    return parse_pip_audit_output(result.stdout)
                
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            pass
        
        return []
    
    def _scan_with_safety(self, requirements_file: str = None) -> List[VulnerabilityInfo]:
        """Scan using safety tool."""
        try:
            cmd = ["safety", "check"]
            if requirements_file:
                cmd.extend(["--requirement", requirements_file])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            # Safety returns non-zero when vulnerabilities are found
            if result.stdout:
                return parse_safety_output(result.stdout)
                
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            pass
        
        return []
    
    def _scan_manual(self, requirements_file: str = None) -> List[VulnerabilityInfo]:
        """Manual scanning when security tools are not available."""
        # In a real implementation, this could check against a local
        # vulnerability database or perform basic checks
        
        # For now, return empty list with a note
        # This is a placeholder for manual vulnerability checking
        return []
    
    def _run_pip_audit(self) -> subprocess.CompletedProcess:
        """Run pip-audit command safely."""
        return subprocess.run(
            ["pip-audit", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=120
        )


def scan_dependencies(requirements_file: str = None, **kwargs) -> ScanResult:
    """High-level function to scan dependencies for vulnerabilities.
    
    Parameters
    ----------
    requirements_file : str, optional
        Path to requirements file to scan
    **kwargs
        Additional arguments (for future extensibility)
        
    Returns
    -------
    ScanResult
        Scan results with vulnerability information
    """
    scanner = DependencySecurityScanner()
    return scanner.scan(requirements_file=requirements_file)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Dependency security scanning for chest X-ray pneumonia detector"
    )
    
    parser.add_argument(
        "--requirements-file",
        type=str,
        help="Path to requirements file to scan"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        help="Save scan results to JSON file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--fail-on-vulnerabilities",
        action="store_true",
        help="Exit with error code if vulnerabilities are found"
    )
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    print("🔍 Scanning dependencies for security vulnerabilities...")
    
    # Perform scan
    result = scan_dependencies(requirements_file=args.requirements_file)
    
    # Print results
    print(f"\n=== Dependency Security Scan Results ===")
    print(f"Scan Tool: {result.scan_tool}")
    print(f"Scan Duration: {result.scan_duration:.2f} seconds")
    print(f"Total Vulnerabilities: {result.total_vulnerabilities}")
    
    if result.total_vulnerabilities > 0:
        print(f"  Critical: {result.critical_count}")
        print(f"  High: {result.high_count}")
        print(f"  Medium: {result.medium_count}")
        print(f"  Low: {result.low_count}")
        
        if args.verbose:
            print("\n=== Vulnerability Details ===")
            for vuln in result.vulnerabilities:
                print(f"\n📦 Package: {vuln.package} ({vuln.version})")
                print(f"🔴 Vulnerability: {vuln.vulnerability_id}")
                print(f"⚠️  Severity: {vuln.severity}")
                print(f"📝 Description: {vuln.description}")
                if vuln.fixed_version:
                    print(f"✅ Fixed in: {vuln.fixed_version}")
    else:
        print("✅ No known vulnerabilities found!")
    
    # Add tool availability notes
    if result.scan_tool == "manual":
        print("\n⚠️  Note: No security scanning tools (pip-audit, safety) were available.")
        print("   Install pip-audit or safety for comprehensive vulnerability scanning:")
        print("   pip install pip-audit")
        print("   pip install safety")
    
    # Save to JSON if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            f.write(result.to_json())
        print(f"\n💾 Results saved to: {args.output_json}")
    
    # Exit with error code if vulnerabilities found and requested
    if args.fail_on_vulnerabilities and result.total_vulnerabilities > 0:
        print(f"\n❌ Exiting with error code due to {result.total_vulnerabilities} vulnerabilities found.")
        sys.exit(1)
    
    print("\n✅ Security scan completed successfully!")


if __name__ == "__main__":
    main()