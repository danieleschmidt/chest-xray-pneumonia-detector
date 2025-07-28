#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# 🔒 Comprehensive Security Scanning Script
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-./security-reports}"
FAIL_ON_HIGH="${FAIL_ON_HIGH:-true}"
FAIL_ON_MEDIUM="${FAIL_ON_MEDIUM:-false}"
EXPORT_FORMATS="${EXPORT_FORMATS:-json,table}"
DOCKER_IMAGE="${DOCKER_IMAGE:-}"

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if required tools are installed
check_security_tools() {
    echo -e "${BLUE}🔍 Checking security tools...${NC}"
    
    local tools_status=()
    
    # Check bandit (Python security)
    if command -v bandit &> /dev/null; then
        tools_status+=("bandit:✅")
    else
        tools_status+=("bandit:❌")
    fi
    
    # Check safety (Python dependencies)
    if command -v safety &> /dev/null; then
        tools_status+=("safety:✅")
    else
        tools_status+=("safety:❌")
    fi
    
    # Check pip-audit (Python dependencies alternative)
    if command -v pip-audit &> /dev/null; then
        tools_status+=("pip-audit:✅")
    else
        tools_status+=("pip-audit:❌")
    fi
    
    # Check semgrep (static analysis)
    if command -v semgrep &> /dev/null; then
        tools_status+=("semgrep:✅")
    else
        tools_status+=("semgrep:❌")
    fi
    
    # Check trivy (container/filesystem scanning)
    if command -v trivy &> /dev/null; then
        tools_status+=("trivy:✅")
    else
        tools_status+=("trivy:❌")
    fi
    
    # Check ruff (linting with security rules)
    if command -v ruff &> /dev/null; then
        tools_status+=("ruff:✅")
    else
        tools_status+=("ruff:❌")
    fi
    
    # Print tool status
    for status in "${tools_status[@]}"; do
        echo -e "  ${status//:✅/:$GREEN✅$NC}"
        echo -e "  ${status//:❌/:$RED❌$NC}"
    done
    
    # Check if at least bandit is available (minimum requirement)
    if ! command -v bandit &> /dev/null; then
        print_error "Bandit is required but not found. Install with: pip install bandit"
        exit 1
    fi
    
    print_status "Security tools check completed"
}

# Setup output directory
setup_output_dir() {
    echo -e "${BLUE}📁 Setting up output directory...${NC}"
    
    mkdir -p "$OUTPUT_DIR"
    print_status "Output directory created: $OUTPUT_DIR"
}

# Run Bandit security scan
run_bandit_scan() {
    echo -e "${BLUE}🐍 Running Bandit security scan...${NC}"
    
    local bandit_json="$OUTPUT_DIR/bandit-report.json"
    local bandit_txt="$OUTPUT_DIR/bandit-report.txt"
    local exit_code=0
    
    # Run bandit with JSON output
    bandit -r src/ -f json -o "$bandit_json" -ll || exit_code=$?
    
    # Run bandit with human-readable output
    bandit -r src/ -f txt -o "$bandit_txt" -ll || true
    
    # Check results
    if [ $exit_code -eq 0 ]; then
        print_status "Bandit scan completed - no high severity issues found"
    else
        local high_count=$(jq -r '.results | map(select(.issue_severity == "HIGH")) | length' "$bandit_json" 2>/dev/null || echo "0")
        local medium_count=$(jq -r '.results | map(select(.issue_severity == "MEDIUM")) | length' "$bandit_json" 2>/dev/null || echo "0")
        
        print_warning "Bandit found security issues:"
        print_warning "  High severity: $high_count"
        print_warning "  Medium severity: $medium_count"
        
        if [ "$FAIL_ON_HIGH" = "true" ] && [ "$high_count" -gt 0 ]; then
            print_error "High severity security issues found. Build should fail."
            return 1
        fi
        
        if [ "$FAIL_ON_MEDIUM" = "true" ] && [ "$medium_count" -gt 0 ]; then
            print_error "Medium severity security issues found. Build should fail."
            return 1
        fi
    fi
    
    return 0
}

# Run dependency security scan
run_dependency_scan() {
    echo -e "${BLUE}📦 Running dependency security scan...${NC}"
    
    local safety_json="$OUTPUT_DIR/safety-report.json"
    local safety_txt="$OUTPUT_DIR/safety-report.txt"
    local pip_audit_json="$OUTPUT_DIR/pip-audit-report.json"
    
    # Try safety first
    if command -v safety &> /dev/null; then
        print_info "Using Safety for dependency scanning..."
        
        safety check --json --output "$safety_json" || {
            print_warning "Safety found vulnerabilities in dependencies"
            
            # Generate human-readable report
            safety check --output "$safety_txt" || true
            
            # Check if we should fail
            if [ "$FAIL_ON_HIGH" = "true" ]; then
                local vuln_count=$(jq '. | length' "$safety_json" 2>/dev/null || echo "1")
                if [ "$vuln_count" -gt 0 ]; then
                    print_error "Vulnerabilities found in dependencies"
                    return 1
                fi
            fi
        }
        
        print_status "Safety dependency scan completed"
    elif command -v pip-audit &> /dev/null; then
        print_info "Using pip-audit for dependency scanning..."
        
        pip-audit --format=json --output "$pip_audit_json" || {
            print_warning "pip-audit found vulnerabilities in dependencies"
            
            # Check if we should fail
            if [ "$FAIL_ON_HIGH" = "true" ]; then
                print_error "Vulnerabilities found in dependencies"
                return 1
            fi
        }
        
        print_status "pip-audit dependency scan completed"
    else
        print_warning "No dependency security scanner available (safety or pip-audit)"
    fi
    
    return 0
}

# Run Semgrep static analysis
run_semgrep_scan() {
    if ! command -v semgrep &> /dev/null; then
        print_warning "Semgrep not available - skipping static analysis"
        return 0
    fi
    
    echo -e "${BLUE}🔍 Running Semgrep static analysis...${NC}"
    
    local semgrep_json="$OUTPUT_DIR/semgrep-report.json"
    local semgrep_txt="$OUTPUT_DIR/semgrep-report.txt"
    
    # Run semgrep with security rules
    semgrep --config=auto --json --output="$semgrep_json" src/ || {
        print_warning "Semgrep found potential security issues"
        
        # Generate human-readable report
        semgrep --config=auto --output="$semgrep_txt" src/ || true
        
        # Parse results
        local error_count=$(jq -r '.results | map(select(.extra.severity == "ERROR")) | length' "$semgrep_json" 2>/dev/null || echo "0")
        
        if [ "$FAIL_ON_HIGH" = "true" ] && [ "$error_count" -gt 0 ]; then
            print_error "High severity issues found by Semgrep"
            return 1
        fi
    }
    
    print_status "Semgrep static analysis completed"
    return 0
}

# Run Trivy filesystem scan
run_trivy_scan() {
    if ! command -v trivy &> /dev/null; then
        print_warning "Trivy not available - skipping filesystem scan"
        return 0
    fi
    
    echo -e "${BLUE}🗂️  Running Trivy filesystem scan...${NC}"
    
    local trivy_json="$OUTPUT_DIR/trivy-fs-report.json"
    local trivy_txt="$OUTPUT_DIR/trivy-fs-report.txt"
    
    # Scan filesystem
    trivy fs --format json --output "$trivy_json" --severity HIGH,CRITICAL . || {
        print_warning "Trivy found filesystem vulnerabilities"
        
        # Generate human-readable report
        trivy fs --format table --output "$trivy_txt" --severity HIGH,CRITICAL . || true
        
        if [ "$FAIL_ON_HIGH" = "true" ]; then
            print_error "High/Critical vulnerabilities found by Trivy"
            return 1
        fi
    }
    
    print_status "Trivy filesystem scan completed"
    return 0
}

# Run Docker image scan
run_docker_scan() {
    if [ -z "$DOCKER_IMAGE" ]; then
        print_info "No Docker image specified - skipping container scan"
        return 0
    fi
    
    if ! command -v trivy &> /dev/null; then
        print_warning "Trivy not available - skipping Docker scan"
        return 0
    fi
    
    echo -e "${BLUE}🐳 Running Docker image security scan...${NC}"
    
    local docker_json="$OUTPUT_DIR/docker-scan-report.json"
    local docker_txt="$OUTPUT_DIR/docker-scan-report.txt"
    
    # Check if image exists
    if ! docker image inspect "$DOCKER_IMAGE" &> /dev/null; then
        print_warning "Docker image $DOCKER_IMAGE not found locally"
        return 0
    fi
    
    # Scan Docker image
    trivy image --format json --output "$docker_json" --severity HIGH,CRITICAL "$DOCKER_IMAGE" || {
        print_warning "Docker image vulnerabilities found"
        
        # Generate human-readable report
        trivy image --format table --output "$docker_txt" --severity HIGH,CRITICAL "$DOCKER_IMAGE" || true
        
        if [ "$FAIL_ON_HIGH" = "true" ]; then
            print_error "High/Critical vulnerabilities found in Docker image"
            return 1
        fi
    }
    
    print_status "Docker image scan completed"
    return 0
}

# Run code quality security checks
run_code_quality_scan() {
    echo -e "${BLUE}📝 Running code quality security checks...${NC}"
    
    local ruff_output="$OUTPUT_DIR/ruff-security.txt"
    
    # Run ruff with security-focused rules
    if command -v ruff &> /dev/null; then
        ruff check --select S src/ > "$ruff_output" 2>&1 || {
            print_warning "Ruff found security-related code quality issues"
            
            # Check if there are actual security issues (not just style)
            if grep -E "(S[0-9]+|security)" "$ruff_output" &> /dev/null; then
                if [ "$FAIL_ON_MEDIUM" = "true" ]; then
                    print_error "Security-related code quality issues found"
                    return 1
                fi
            fi
        }
        
        print_status "Code quality security checks completed"
    else
        print_warning "Ruff not available - skipping code quality security checks"
    fi
    
    return 0
}

# Generate comprehensive security report
generate_security_report() {
    echo -e "${BLUE}📊 Generating comprehensive security report...${NC}"
    
    local report_file="$OUTPUT_DIR/security-summary.md"
    local timestamp=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    local git_commit=$(git rev-parse HEAD 2>/dev/null || echo 'unknown')
    
    cat > "$report_file" << EOF
# Security Scan Report

**Generated:** $timestamp  
**Git Commit:** $git_commit  
**Configuration:**
- Fail on High: $FAIL_ON_HIGH
- Fail on Medium: $FAIL_ON_MEDIUM
- Docker Image: ${DOCKER_IMAGE:-"None"}

## Scan Results

### Static Analysis (Bandit)
$(if [ -f "$OUTPUT_DIR/bandit-report.json" ]; then
    local high_issues=$(jq -r '.results | map(select(.issue_severity == "HIGH")) | length' "$OUTPUT_DIR/bandit-report.json" 2>/dev/null || echo "0")
    local medium_issues=$(jq -r '.results | map(select(.issue_severity == "MEDIUM")) | length' "$OUTPUT_DIR/bandit-report.json" 2>/dev/null || echo "0")
    local low_issues=$(jq -r '.results | map(select(.issue_severity == "LOW")) | length' "$OUTPUT_DIR/bandit-report.json" 2>/dev/null || echo "0")
    echo "- High severity issues: $high_issues"
    echo "- Medium severity issues: $medium_issues"
    echo "- Low severity issues: $low_issues"
    echo "- Status: $([ "$high_issues" -eq 0 ] && echo "✅ PASS" || echo "❌ FAIL")"
else
    echo "- Status: ⚠️ NOT RUN"
fi)

### Dependency Scan
$(if [ -f "$OUTPUT_DIR/safety-report.json" ]; then
    local vuln_count=$(jq '. | length' "$OUTPUT_DIR/safety-report.json" 2>/dev/null || echo "0")
    echo "- Vulnerabilities found: $vuln_count"
    echo "- Status: $([ "$vuln_count" -eq 0 ] && echo "✅ PASS" || echo "❌ FAIL")"
elif [ -f "$OUTPUT_DIR/pip-audit-report.json" ]; then
    echo "- Scanned with pip-audit"
    echo "- Status: ✅ COMPLETED"
else
    echo "- Status: ⚠️ NOT RUN"
fi)

### Static Analysis (Semgrep)
$(if [ -f "$OUTPUT_DIR/semgrep-report.json" ]; then
    local error_count=$(jq -r '.results | map(select(.extra.severity == "ERROR")) | length' "$OUTPUT_DIR/semgrep-report.json" 2>/dev/null || echo "0")
    local warning_count=$(jq -r '.results | map(select(.extra.severity == "WARNING")) | length' "$OUTPUT_DIR/semgrep-report.json" 2>/dev/null || echo "0")
    echo "- Errors: $error_count"
    echo "- Warnings: $warning_count"
    echo "- Status: $([ "$error_count" -eq 0 ] && echo "✅ PASS" || echo "❌ FAIL")"
else
    echo "- Status: ⚠️ NOT RUN"
fi)

### Filesystem Scan (Trivy)
$(if [ -f "$OUTPUT_DIR/trivy-fs-report.json" ]; then
    echo "- Filesystem vulnerabilities scanned"
    echo "- Status: ✅ COMPLETED"
else
    echo "- Status: ⚠️ NOT RUN"
fi)

### Docker Image Scan
$(if [ -f "$OUTPUT_DIR/docker-scan-report.json" ]; then
    echo "- Docker image: $DOCKER_IMAGE"
    echo "- Status: ✅ COMPLETED"
else
    echo "- Status: ⚠️ NOT RUN"
fi)

## Recommendations

1. **Review all HIGH severity issues immediately**
2. **Address MEDIUM severity issues based on risk assessment**
3. **Keep dependencies updated regularly**
4. **Run security scans in CI/CD pipeline**
5. **Consider implementing security policies**

## Report Files

The following detailed reports are available:

EOF
    
    # List all generated report files
    for file in "$OUTPUT_DIR"/*.{json,txt} 2>/dev/null; do
        if [ -f "$file" ] && [ "$(basename "$file")" != "security-summary.md" ]; then
            local filename=$(basename "$file")
            echo "- **$filename**" >> "$report_file"
        fi
    done
    
    print_status "Security summary report generated: $report_file"
}

# Main function
main() {
    echo -e "${BLUE}🔒 Starting comprehensive security scan${NC}"
    echo "Output directory: $OUTPUT_DIR"
    echo "Fail on high: $FAIL_ON_HIGH"
    echo "Fail on medium: $FAIL_ON_MEDIUM"
    if [ -n "$DOCKER_IMAGE" ]; then
        echo "Docker image: $DOCKER_IMAGE"
    fi
    echo
    
    local overall_status=0
    
    # Setup
    check_security_tools
    setup_output_dir
    
    # Run security scans
    run_bandit_scan || overall_status=1
    run_dependency_scan || overall_status=1
    run_semgrep_scan || overall_status=1
    run_trivy_scan || overall_status=1
    run_docker_scan || overall_status=1
    run_code_quality_scan || overall_status=1
    
    # Generate report
    generate_security_report
    
    echo
    if [ $overall_status -eq 0 ]; then
        print_status "All security scans completed successfully!"
        echo -e "${GREEN}🎉 No critical security issues found${NC}"
    else
        print_error "Security scans found issues that require attention"
        echo -e "${RED}⚠️  Check the detailed reports in: $OUTPUT_DIR${NC}"
    fi
    
    # List generated files
    echo -e "${BLUE}Generated reports:${NC}"
    ls -la "$OUTPUT_DIR/"
    
    exit $overall_status
}

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run comprehensive security scanning for the project.

Environment Variables:
  OUTPUT_DIR          Output directory for reports (default: ./security-reports)
  FAIL_ON_HIGH        Fail build on high severity issues (default: true)
  FAIL_ON_MEDIUM      Fail build on medium severity issues (default: false)
  EXPORT_FORMATS      Report formats to generate (default: json,table)
  DOCKER_IMAGE        Docker image to scan (optional)

Examples:
  $0                                                    # Basic security scan
  DOCKER_IMAGE=my-app:latest $0                        # Include Docker scan
  FAIL_ON_MEDIUM=true OUTPUT_DIR=/tmp/security $0      # Strict mode with custom output

Required Tools (install as needed):
  - bandit: pip install bandit
  - safety: pip install safety
  - pip-audit: pip install pip-audit
  - semgrep: pip install semgrep
  - trivy: https://aquasecurity.github.io/trivy/
  - ruff: pip install ruff
EOF
}

# Handle help flag
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

# Run main function
main "$@"