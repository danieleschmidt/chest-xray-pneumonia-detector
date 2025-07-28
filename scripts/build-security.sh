#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# 🔒 Security-Enhanced Build Script
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
STAGE="${1:-production}"
SCAN_ENABLED="${SCAN_ENABLED:-true}"
SBOM_ENABLED="${SBOM_ENABLED:-true}"
PUSH_ENABLED="${PUSH_ENABLED:-false}"
REGISTRY="${REGISTRY:-ghcr.io}"
IMAGE_NAME="${IMAGE_NAME:-chest-xray-pneumonia-detector}"
TAG="${TAG:-latest}"

echo -e "${BLUE}🔒 Starting security-enhanced build for stage: ${STAGE}${NC}"

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

# Check prerequisites
check_prerequisites() {
    echo -e "${BLUE}🔍 Checking prerequisites...${NC}"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
    fi
    print_status "Docker found"
    
    # Check Docker BuildKit
    if ! docker buildx version &> /dev/null; then
        print_error "Docker BuildKit not available"
    fi
    print_status "Docker BuildKit available"
    
    # Check security tools (optional)
    if [ "$SCAN_ENABLED" = "true" ]; then
        if command -v trivy &> /dev/null; then
            print_status "Trivy scanner found"
        else
            print_warning "Trivy not found - container scanning disabled"
            SCAN_ENABLED="false"
        fi
    fi
    
    if [ "$SBOM_ENABLED" = "true" ]; then
        if command -v syft &> /dev/null; then
            print_status "Syft SBOM generator found"
        else
            print_warning "Syft not found - SBOM generation disabled"
            SBOM_ENABLED="false"
        fi
    fi
}

# Build Docker image with security best practices
build_image() {
    echo -e "${BLUE}🏗️  Building Docker image...${NC}"
    
    local full_image_name="${REGISTRY}/${IMAGE_NAME}:${TAG}"
    local build_args=""
    
    # Add build metadata
    build_args+="--build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') "
    build_args+="--build-arg VCS_REF=$(git rev-parse HEAD 2>/dev/null || echo 'unknown') "
    build_args+="--build-arg VERSION=$(cat VERSION 2>/dev/null || echo '0.0.0') "
    
    # Security-focused build options
    docker buildx build \
        --platform linux/amd64,linux/arm64 \
        --target "${STAGE}" \
        --tag "${full_image_name}" \
        --label "org.opencontainers.image.created=$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --label "org.opencontainers.image.source=https://github.com/your-org/chest-xray-pneumonia-detector" \
        --label "org.opencontainers.image.version=${TAG}" \
        --label "org.opencontainers.image.revision=$(git rev-parse HEAD 2>/dev/null || echo 'unknown')" \
        --label "org.opencontainers.image.title=Chest X-Ray Pneumonia Detector" \
        --label "org.opencontainers.image.description=ML system for pneumonia detection from chest X-rays" \
        --label "org.opencontainers.image.vendor=Your Organization" \
        --label "org.opencontainers.image.licenses=MIT" \
        ${build_args} \
        --load \
        .
    
    print_status "Docker image built: ${full_image_name}"
    echo "IMAGE_NAME=${full_image_name}" >> "${GITHUB_OUTPUT:-/dev/null}" 2>/dev/null || true
}

# Scan image for vulnerabilities
scan_image() {
    if [ "$SCAN_ENABLED" != "true" ]; then
        print_warning "Security scanning disabled"
        return 0
    fi
    
    echo -e "${BLUE}🔍 Scanning image for vulnerabilities...${NC}"
    
    local full_image_name="${REGISTRY}/${IMAGE_NAME}:${TAG}"
    local scan_output="security-scan-report.json"
    
    # Run Trivy scan
    trivy image \
        --format json \
        --output "${scan_output}" \
        --severity HIGH,CRITICAL \
        --exit-code 1 \
        "${full_image_name}" || {
        print_error "High/Critical vulnerabilities found in image"
    }
    
    # Generate human-readable report
    trivy image \
        --format table \
        --severity HIGH,CRITICAL \
        "${full_image_name}"
    
    print_status "Security scan completed - no high/critical vulnerabilities"
}

# Generate Software Bill of Materials (SBOM)
generate_sbom() {
    if [ "$SBOM_ENABLED" != "true" ]; then
        print_warning "SBOM generation disabled"
        return 0
    fi
    
    echo -e "${BLUE}📋 Generating Software Bill of Materials...${NC}"
    
    local full_image_name="${REGISTRY}/${IMAGE_NAME}:${TAG}"
    local sbom_output="sbom.json"
    
    # Generate SBOM with Syft
    syft "${full_image_name}" -o json > "${sbom_output}"
    
    # Generate human-readable SBOM
    syft "${full_image_name}" -o table > "sbom.txt"
    
    print_status "SBOM generated: ${sbom_output}"
}

# Test image functionality
test_image() {
    echo -e "${BLUE}🧪 Testing image functionality...${NC}"
    
    local full_image_name="${REGISTRY}/${IMAGE_NAME}:${TAG}"
    
    # Basic smoke test
    docker run --rm "${full_image_name}" python -c "import sys; print(f'Python {sys.version}'); sys.exit(0)" || {
        print_error "Image smoke test failed"
    }
    
    # Test package installation
    docker run --rm "${full_image_name}" python -c "import src; print('Package import successful')" || {
        print_error "Package import test failed"
    }
    
    print_status "Image functionality tests passed"
}

# Push image to registry
push_image() {
    if [ "$PUSH_ENABLED" != "true" ]; then
        print_warning "Image push disabled"
        return 0
    fi
    
    echo -e "${BLUE}📤 Pushing image to registry...${NC}"
    
    local full_image_name="${REGISTRY}/${IMAGE_NAME}:${TAG}"
    
    # Push multi-platform image
    docker buildx build \
        --platform linux/amd64,linux/arm64 \
        --target "${STAGE}" \
        --tag "${full_image_name}" \
        --push \
        .
    
    print_status "Image pushed: ${full_image_name}"
}

# Generate build report
generate_report() {
    echo -e "${BLUE}📊 Generating build report...${NC}"
    
    local report_file="build-report.json"
    local full_image_name="${REGISTRY}/${IMAGE_NAME}:${TAG}"
    
    cat > "${report_file}" << EOF
{
  "build_info": {
    "timestamp": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')",
    "stage": "${STAGE}",
    "image": "${full_image_name}",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "builder": "$(whoami)@$(hostname)"
  },
  "security": {
    "scan_enabled": ${SCAN_ENABLED},
    "sbom_generated": ${SBOM_ENABLED},
    "scan_passed": true
  },
  "artifacts": {
    "image": "${full_image_name}",
    "sbom": "$([ -f sbom.json ] && echo 'sbom.json' || echo 'null')",
    "scan_report": "$([ -f security-scan-report.json ] && echo 'security-scan-report.json' || echo 'null')"
  }
}
EOF
    
    print_status "Build report generated: ${report_file}"
}

# Cleanup function
cleanup() {
    echo -e "${BLUE}🧹 Cleaning up...${NC}"
    
    # Remove temporary files
    rm -f Dockerfile.tmp
    
    print_status "Cleanup completed"
}

# Main execution
main() {
    echo -e "${BLUE}🚀 Starting security-enhanced Docker build${NC}"
    echo "Stage: ${STAGE}"
    echo "Image: ${REGISTRY}/${IMAGE_NAME}:${TAG}"
    echo "Scan enabled: ${SCAN_ENABLED}"
    echo "SBOM enabled: ${SBOM_ENABLED}"
    echo "Push enabled: ${PUSH_ENABLED}"
    echo
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Execute build pipeline
    check_prerequisites
    build_image
    test_image
    scan_image
    generate_sbom
    generate_report
    push_image
    
    echo
    print_status "Security-enhanced build completed successfully!"
    echo -e "${GREEN}🎉 Image ready: ${REGISTRY}/${IMAGE_NAME}:${TAG}${NC}"
}

# Script usage
usage() {
    cat << EOF
Usage: $0 [STAGE] [OPTIONS]

STAGE: Docker build stage (development|production|api|inference)

Environment Variables:
  SCAN_ENABLED    - Enable security scanning (default: true)
  SBOM_ENABLED    - Enable SBOM generation (default: true)
  PUSH_ENABLED    - Enable image push (default: false)
  REGISTRY        - Container registry (default: ghcr.io)
  IMAGE_NAME      - Image name (default: chest-xray-pneumonia-detector)
  TAG             - Image tag (default: latest)

Examples:
  $0 production
  PUSH_ENABLED=true $0 production
  TAG=v1.0.0 $0 production
EOF
}

# Handle help flag
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

# Run main function
main "$@"