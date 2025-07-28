#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# 📋 Software Bill of Materials (SBOM) Generation Script
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-./sbom}"
FORMAT="${FORMAT:-json}"
INCLUDE_LICENSES="${INCLUDE_LICENSES:-true}"
INCLUDE_VULNERABILITIES="${INCLUDE_VULNERABILITIES:-true}"

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if required tools are installed
check_tools() {
    echo -e "${BLUE}🔍 Checking required tools...${NC}"
    
    local missing_tools=()
    
    # Check for syft (SBOM generation)
    if ! command -v syft &> /dev/null; then
        missing_tools+=("syft")
    fi
    
    # Check for grype (vulnerability scanning)
    if [ "$INCLUDE_VULNERABILITIES" = "true" ] && ! command -v grype &> /dev/null; then
        print_warning "grype not found - vulnerability scanning will be skipped"
        INCLUDE_VULNERABILITIES="false"
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_error "Missing required tools: ${missing_tools[*]}"
        echo "Install with:"
        echo "  # Syft (SBOM generation)"
        echo "  curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin"
        echo "  # Grype (vulnerability scanning)"
        echo "  curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin"
        exit 1
    fi
    
    print_status "All required tools are available"
}

# Create output directory
setup_output_dir() {
    echo -e "${BLUE}📁 Setting up output directory...${NC}"
    
    mkdir -p "$OUTPUT_DIR"
    print_status "Output directory created: $OUTPUT_DIR"
}

# Generate SBOM for source code
generate_source_sbom() {
    echo -e "${BLUE}📋 Generating SBOM for source code...${NC}"
    
    local output_file="$OUTPUT_DIR/source-sbom.$FORMAT"
    
    # Generate SBOM for current directory
    syft . -o "$FORMAT" > "$output_file"
    
    # Generate additional formats
    if [ "$FORMAT" = "json" ]; then
        syft . -o table > "$OUTPUT_DIR/source-sbom.txt"
        syft . -o spdx-json > "$OUTPUT_DIR/source-sbom-spdx.json"
    fi
    
    print_status "Source SBOM generated: $output_file"
}

# Generate SBOM for Docker image
generate_docker_sbom() {
    local image_name="$1"
    
    echo -e "${BLUE}📋 Generating SBOM for Docker image: $image_name${NC}"
    
    local output_file="$OUTPUT_DIR/docker-sbom.$FORMAT"
    
    # Check if image exists locally
    if ! docker image inspect "$image_name" &> /dev/null; then
        print_warning "Docker image $image_name not found locally"
        return 1
    fi
    
    # Generate SBOM for Docker image
    syft "$image_name" -o "$FORMAT" > "$output_file"
    
    # Generate additional formats
    if [ "$FORMAT" = "json" ]; then
        syft "$image_name" -o table > "$OUTPUT_DIR/docker-sbom.txt"
        syft "$image_name" -o spdx-json > "$OUTPUT_DIR/docker-sbom-spdx.json"
    fi
    
    print_status "Docker SBOM generated: $output_file"
}

# Generate Python requirements SBOM
generate_python_sbom() {
    echo -e "${BLUE}🐍 Generating Python dependencies SBOM...${NC}"
    
    local output_file="$OUTPUT_DIR/python-deps.$FORMAT"
    
    # Create temporary requirements file with all dependencies
    local temp_req_file=$(mktemp)
    
    # Combine all requirements files
    for req_file in requirements*.txt; do
        if [ -f "$req_file" ]; then
            echo "# From $req_file" >> "$temp_req_file"
            cat "$req_file" >> "$temp_req_file"
            echo "" >> "$temp_req_file"
        fi
    done
    
    if [ -s "$temp_req_file" ]; then
        # Generate SBOM from requirements
        syft "$temp_req_file" -o "$FORMAT" > "$output_file"
        
        # Generate pip freeze output for comparison
        if command -v pip &> /dev/null; then
            pip freeze > "$OUTPUT_DIR/pip-freeze.txt"
        fi
        
        print_status "Python dependencies SBOM generated: $output_file"
    else
        print_warning "No requirements files found"
    fi
    
    # Cleanup
    rm -f "$temp_req_file"
}

# Scan for vulnerabilities
scan_vulnerabilities() {
    if [ "$INCLUDE_VULNERABILITIES" != "true" ]; then
        print_info "Vulnerability scanning disabled"
        return 0
    fi
    
    echo -e "${BLUE}🔍 Scanning for vulnerabilities...${NC}"
    
    local vuln_output="$OUTPUT_DIR/vulnerabilities.json"
    local vuln_report="$OUTPUT_DIR/vulnerability-report.txt"
    
    # Scan source code
    if [ -f "$OUTPUT_DIR/source-sbom.json" ]; then
        grype sbom:"$OUTPUT_DIR/source-sbom.json" -o json > "$vuln_output" 2>/dev/null || true
        grype sbom:"$OUTPUT_DIR/source-sbom.json" -o table > "$vuln_report" 2>/dev/null || true
        print_status "Vulnerability scan completed for source code"
    fi
    
    # Scan Docker image if provided
    if [ -n "${1:-}" ] && [ -f "$OUTPUT_DIR/docker-sbom.json" ]; then
        local docker_vuln_output="$OUTPUT_DIR/docker-vulnerabilities.json"
        local docker_vuln_report="$OUTPUT_DIR/docker-vulnerability-report.txt"
        
        grype sbom:"$OUTPUT_DIR/docker-sbom.json" -o json > "$docker_vuln_output" 2>/dev/null || true
        grype sbom:"$OUTPUT_DIR/docker-sbom.json" -o table > "$docker_vuln_report" 2>/dev/null || true
        print_status "Vulnerability scan completed for Docker image"
    fi
}

# Generate license report
generate_license_report() {
    if [ "$INCLUDE_LICENSES" != "true" ]; then
        print_info "License reporting disabled"
        return 0
    fi
    
    echo -e "${BLUE}📄 Generating license report...${NC}"
    
    local license_output="$OUTPUT_DIR/licenses.json"
    
    # Extract license information from SBOM
    if [ -f "$OUTPUT_DIR/source-sbom.json" ]; then
        # Use jq to extract license information if available
        if command -v jq &> /dev/null; then
            jq -r '.artifacts[] | select(.licenses != null) | {name: .name, version: .version, licenses: .licenses}' \
                "$OUTPUT_DIR/source-sbom.json" > "$license_output" 2>/dev/null || true
        fi
        
        # Generate license summary
        if [ -s "$license_output" ]; then
            print_status "License report generated: $license_output"
        else
            print_warning "No license information found in SBOM"
        fi
    fi
}

# Generate comprehensive report
generate_report() {
    echo -e "${BLUE}📊 Generating comprehensive SBOM report...${NC}"
    
    local report_file="$OUTPUT_DIR/sbom-report.md"
    local timestamp=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    local git_commit=$(git rev-parse HEAD 2>/dev/null || echo 'unknown')
    
    cat > "$report_file" << EOF
# Software Bill of Materials Report

**Generated:** $timestamp  
**Git Commit:** $git_commit  
**Tool:** Syft + Grype  

## Summary

This report contains the Software Bill of Materials (SBOM) for the Chest X-Ray Pneumonia Detector project.

## Files Generated

EOF
    
    # List generated files
    for file in "$OUTPUT_DIR"/*; do
        if [ -f "$file" ] && [ "$(basename "$file")" != "sbom-report.md" ]; then
            local filename=$(basename "$file")
            local filesize=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "unknown")
            echo "- **$filename** (${filesize} bytes)" >> "$report_file"
        fi
    done
    
    cat >> "$report_file" << EOF

## SBOM Formats

- **JSON**: Machine-readable format for automated processing
- **Table**: Human-readable tabular format
- **SPDX**: Industry-standard SBOM format

## Vulnerability Scanning

$(if [ "$INCLUDE_VULNERABILITIES" = "true" ]; then
    echo "Vulnerability scanning was performed using Grype."
    if [ -f "$OUTPUT_DIR/vulnerabilities.json" ]; then
        echo "See vulnerability-report.txt for detailed findings."
    fi
else
    echo "Vulnerability scanning was disabled."
fi)

## License Information

$(if [ "$INCLUDE_LICENSES" = "true" ]; then
    echo "License information was extracted from package metadata."
    if [ -f "$OUTPUT_DIR/licenses.json" ]; then
        echo "See licenses.json for detailed license information."
    fi
else
    echo "License reporting was disabled."
fi)

## Usage

These SBOM files can be used for:

- Supply chain security analysis
- Compliance reporting
- Vulnerability management
- License compliance
- Software composition analysis

## Tools Information

- **Syft**: $(syft version 2>/dev/null | head -1 || echo "version unknown")
$(if [ "$INCLUDE_VULNERABILITIES" = "true" ]; then
    echo "- **Grype**: $(grype version 2>/dev/null | head -1 || echo "version unknown")"
fi)
EOF
    
    print_status "Comprehensive report generated: $report_file"
}

# Main function
main() {
    local image_name="${1:-}"
    
    echo -e "${BLUE}📋 Starting SBOM generation${NC}"
    echo "Output directory: $OUTPUT_DIR"
    echo "Format: $FORMAT"
    echo "Include licenses: $INCLUDE_LICENSES"
    echo "Include vulnerabilities: $INCLUDE_VULNERABILITIES"
    if [ -n "$image_name" ]; then
        echo "Docker image: $image_name"
    fi
    echo
    
    # Execute SBOM generation pipeline
    check_tools
    setup_output_dir
    
    # Generate SBOMs
    generate_source_sbom
    generate_python_sbom
    
    if [ -n "$image_name" ]; then
        generate_docker_sbom "$image_name" || print_warning "Docker SBOM generation failed"
    fi
    
    # Additional analysis
    scan_vulnerabilities "$image_name"
    generate_license_report
    generate_report
    
    echo
    print_status "SBOM generation completed successfully!"
    echo -e "${GREEN}📋 SBOM files available in: $OUTPUT_DIR${NC}"
    
    # List generated files
    echo -e "${BLUE}Generated files:${NC}"
    ls -la "$OUTPUT_DIR/"
}

# Usage function
usage() {
    cat << EOF
Usage: $0 [DOCKER_IMAGE]

Generate Software Bill of Materials (SBOM) for the project.

Arguments:
  DOCKER_IMAGE    Optional Docker image name to generate SBOM for

Environment Variables:
  OUTPUT_DIR              Output directory (default: ./sbom)
  FORMAT                  SBOM format: json|table|spdx (default: json)
  INCLUDE_LICENSES        Include license information (default: true)
  INCLUDE_VULNERABILITIES Include vulnerability scanning (default: true)

Examples:
  $0                                          # Generate SBOM for source code only
  $0 my-image:latest                         # Include Docker image SBOM
  OUTPUT_DIR=/tmp/sbom $0                    # Custom output directory
  FORMAT=spdx INCLUDE_VULNERABILITIES=false $0  # SPDX format, no vulnerability scan

Required Tools:
  - syft: https://github.com/anchore/syft
  - grype: https://github.com/anchore/grype (optional, for vulnerability scanning)
EOF
}

# Handle help flag
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

# Run main function
main "${1:-}"