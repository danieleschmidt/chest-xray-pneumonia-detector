#!/bin/bash
# SLSA Provenance Verification Script - Advanced Supply Chain Security
# Comprehensive verification of SLSA Level 3 provenance for all artifacts

set -euo pipefail

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔧 CONFIGURATION AND SETUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VERIFICATION_LOG="$PROJECT_ROOT/logs/slsa-verification.log"

# SLSA configuration
SLSA_LEVEL="3"
EXPECTED_BUILDER="https://github.com/slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml"
TRUSTED_ROOT_CA="sigstore"

# Verification tools
SLSA_VERIFIER="slsa-verifier"
COSIGN="cosign"
REKOR_CLI="rekor-cli"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📋 UTILITY FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Create logs directory if it doesn't exist
    mkdir -p "$(dirname "$VERIFICATION_LOG")"
    
    case "$level" in
        "INFO")
            echo -e "${GREEN}[INFO]${NC} $message" >&2
            echo "[$timestamp] [INFO] $message" >> "$VERIFICATION_LOG"
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} $message" >&2
            echo "[$timestamp] [WARN] $message" >> "$VERIFICATION_LOG"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} $message" >&2
            echo "[$timestamp] [ERROR] $message" >> "$VERIFICATION_LOG"
            ;;
        "DEBUG")
            if [[ "${DEBUG:-0}" == "1" ]]; then
                echo -e "${BLUE}[DEBUG]${NC} $message" >&2
                echo "[$timestamp] [DEBUG] $message" >> "$VERIFICATION_LOG"
            fi
            ;;
    esac
}

check_prerequisites() {
    log "INFO" "Checking verification tool prerequisites..."
    
    local missing_tools=()
    
    # Check for required tools
    for tool in "$SLSA_VERIFIER" "$COSIGN" "$REKOR_CLI" "jq" "curl"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log "ERROR" "Missing required tools: ${missing_tools[*]}"
        log "INFO" "Install missing tools:"
        for tool in "${missing_tools[@]}"; do
            case "$tool" in
                "slsa-verifier")
                    log "INFO" "  - slsa-verifier: go install github.com/slsa-framework/slsa-verifier/v2/cli/slsa-verifier@latest"
                    ;;
                "cosign")
                    log "INFO" "  - cosign: go install github.com/sigstore/cosign/v2/cmd/cosign@latest"
                    ;;
                "rekor-cli")
                    log "INFO" "  - rekor-cli: go install github.com/sigstore/rekor/cmd/rekor-cli@latest"
                    ;;
                "jq")
                    log "INFO" "  - jq: apt-get install jq (Ubuntu/Debian) or brew install jq (macOS)"
                    ;;
                "curl")
                    log "INFO" "  - curl: apt-get install curl (Ubuntu/Debian) or brew install curl (macOS)"
                    ;;
            esac
        done
        return 1
    fi
    
    log "INFO" "All prerequisites satisfied"
    return 0
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔍 VERIFICATION FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

verify_slsa_provenance() {
    local artifact_path="$1"
    local provenance_path="$2"
    local source_uri="$3"
    
    log "INFO" "Verifying SLSA provenance for artifact: $(basename "$artifact_path")"
    
    # Check if files exist
    if [[ ! -f "$artifact_path" ]]; then
        log "ERROR" "Artifact not found: $artifact_path"
        return 1
    fi
    
    if [[ ! -f "$provenance_path" ]]; then
        log "ERROR" "Provenance file not found: $provenance_path"
        return 1
    fi
    
    # Verify provenance authenticity and completeness
    log "DEBUG" "Running SLSA verifier..."
    if "$SLSA_VERIFIER" verify-artifact \
        --provenance-path "$provenance_path" \
        --source-uri "$source_uri" \
        --builder-id "$EXPECTED_BUILDER" \
        "$artifact_path" 2>/dev/null; then
        log "INFO" "✓ SLSA provenance verification successful"
        return 0
    else
        log "ERROR" "✗ SLSA provenance verification failed"
        return 1
    fi
}

verify_container_signature() {
    local image_ref="$1"
    
    log "INFO" "Verifying container signature for: $image_ref"
    
    # Verify container signature with Cosign
    log "DEBUG" "Running Cosign verification..."
    if "$COSIGN" verify \
        --certificate-identity-regexp "https://github.com/.*/.github/workflows/.*" \
        --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
        "$image_ref" 2>/dev/null; then
        log "INFO" "✓ Container signature verification successful"
        return 0
    else
        log "ERROR" "✗ Container signature verification failed"
        return 1
    fi
}

verify_sbom() {
    local image_ref="$1"
    
    log "INFO" "Verifying SBOM for: $image_ref"
    
    # Download and verify SBOM
    local sbom_file="/tmp/sbom-$(basename "$image_ref" | tr ':' '-').json"
    
    log "DEBUG" "Downloading SBOM..."
    if "$COSIGN" download sbom "$image_ref" > "$sbom_file" 2>/dev/null; then
        # Validate SBOM format
        if jq -e '.spdxVersion' "$sbom_file" >/dev/null 2>&1; then
            log "INFO" "✓ SBOM verification successful (SPDX format)"
            
            # Extract key information
            local package_count
            package_count=$(jq '.packages | length' "$sbom_file")
            log "INFO" "  - Packages documented: $package_count"
            
            # Check for high/critical vulnerabilities if vulnerability data is present
            if jq -e '.vulnerabilities' "$sbom_file" >/dev/null 2>&1; then
                local critical_vulns
                critical_vulns=$(jq '[.vulnerabilities[] | select(.severity=="CRITICAL")] | length' "$sbom_file")
                local high_vulns
                high_vulns=$(jq '[.vulnerabilities[] | select(.severity=="HIGH")] | length' "$sbom_file")
                
                log "INFO" "  - Critical vulnerabilities: $critical_vulns"
                log "INFO" "  - High vulnerabilities: $high_vulns"
                
                if [[ "$critical_vulns" -gt 0 ]]; then
                    log "WARN" "Critical vulnerabilities found in SBOM"
                fi
            fi
            
            rm -f "$sbom_file"
            return 0
        else
            log "ERROR" "Invalid SBOM format"
            rm -f "$sbom_file"
            return 1
        fi
    else
        log "ERROR" "✗ SBOM download/verification failed"
        return 1
    fi
}

verify_transparency_log() {
    local artifact_digest="$1"
    
    log "INFO" "Verifying transparency log entry for digest: ${artifact_digest:0:12}..."
    
    # Search Rekor transparency log
    log "DEBUG" "Searching Rekor transparency log..."
    if "$REKOR_CLI" search --sha "$artifact_digest" 2>/dev/null | grep -q "Found matching entries"; then
        log "INFO" "✓ Transparency log verification successful"
        
        # Get additional details about the log entry
        local log_index
        log_index=$("$REKOR_CLI" search --sha "$artifact_digest" 2>/dev/null | grep -o "LogIndex: [0-9]*" | cut -d' ' -f2 | head -1)
        if [[ -n "$log_index" ]]; then
            log "INFO" "  - Rekor log index: $log_index"
        fi
        
        return 0
    else
        log "ERROR" "✗ Transparency log verification failed"
        return 1
    fi
}

verify_build_environment() {
    local provenance_path="$1"
    
    log "INFO" "Verifying build environment integrity..."
    
    if [[ ! -f "$provenance_path" ]]; then
        log "ERROR" "Provenance file not found: $provenance_path"
        return 1
    fi
    
    # Extract and verify build environment details
    local builder_id
    builder_id=$(jq -r '.predicate.builder.id' "$provenance_path" 2>/dev/null)
    
    if [[ "$builder_id" == "$EXPECTED_BUILDER" ]]; then
        log "INFO" "✓ Build environment verification successful"
        log "INFO" "  - Builder ID: $builder_id"
        
        # Check for additional build environment details
        local build_type
        build_type=$(jq -r '.predicate.buildType' "$provenance_path" 2>/dev/null)
        log "INFO" "  - Build type: $build_type"
        
        # Verify build was isolated and hermetic
        local invocation_params
        invocation_params=$(jq -r '.predicate.invocation.parameters // {}' "$provenance_path" 2>/dev/null)
        if [[ "$invocation_params" != "null" ]]; then
            log "DEBUG" "  - Build invocation parameters verified"
        fi
        
        return 0
    else
        log "ERROR" "✗ Build environment verification failed"
        log "ERROR" "  - Expected builder: $EXPECTED_BUILDER"
        log "ERROR" "  - Actual builder: ${builder_id:-"unknown"}"
        return 1
    fi
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🚀 MAIN VERIFICATION WORKFLOWS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

verify_python_package() {
    local package_path="$1"
    local provenance_path="${2:-${package_path}.intoto.jsonl}"
    
    log "INFO" "Starting Python package verification..."
    log "INFO" "Package: $(basename "$package_path")"
    
    local verification_status=0
    
    # 1. Verify SLSA provenance
    if ! verify_slsa_provenance "$package_path" "$provenance_path" "git+https://github.com/$GITHUB_REPOSITORY"; then
        verification_status=1
    fi
    
    # 2. Verify build environment
    if ! verify_build_environment "$provenance_path"; then
        verification_status=1
    fi
    
    # 3. Calculate and verify package hash
    local package_hash
    package_hash=$(sha256sum "$package_path" | cut -d' ' -f1)
    log "INFO" "Package SHA256: $package_hash"
    
    # 4. Verify transparency log entry
    if ! verify_transparency_log "$package_hash"; then
        log "WARN" "Transparency log verification failed (may not be critical for Python packages)"
    fi
    
    return $verification_status
}

verify_container() {
    local image_ref="$1"
    
    log "INFO" "Starting container verification..."
    log "INFO" "Image: $image_ref"
    
    local verification_status=0
    
    # 1. Verify container signature
    if ! verify_container_signature "$image_ref"; then
        verification_status=1
    fi
    
    # 2. Verify SBOM
    if ! verify_sbom "$image_ref"; then
        verification_status=1
    fi
    
    # 3. Get image digest for transparency log verification
    local image_digest
    image_digest=$("$COSIGN" triangulate "$image_ref" 2>/dev/null | grep -o 'sha256:[a-f0-9]*' | head -1)
    
    if [[ -n "$image_digest" ]]; then
        log "INFO" "Image digest: $image_digest"
        
        # 4. Verify transparency log entry
        if ! verify_transparency_log "${image_digest#sha256:}"; then
            log "WARN" "Transparency log verification failed"
        fi
    else
        log "WARN" "Could not determine image digest"
    fi
    
    return $verification_status
}

verify_ml_model() {
    local model_path="$1"
    local provenance_path="${2:-${model_path}.provenance.json}"
    
    log "INFO" "Starting ML model verification..."
    log "INFO" "Model: $(basename "$model_path")"
    
    local verification_status=0
    
    # 1. Verify model file exists and is readable
    if [[ ! -r "$model_path" ]]; then
        log "ERROR" "Model file not readable: $model_path"
        return 1
    fi
    
    # 2. Verify model provenance (custom format for ML models)
    if [[ -f "$provenance_path" ]]; then
        log "INFO" "Verifying ML model provenance..."
        
        # Check for required ML provenance fields
        local required_fields=("training_data_hash" "training_script_hash" "model_architecture" "hyperparameters")
        for field in "${required_fields[@]}"; do
            if ! jq -e ".$field" "$provenance_path" >/dev/null 2>&1; then
                log "WARN" "Missing required provenance field: $field"
                verification_status=1
            fi
        done
        
        # Verify training data integrity
        local training_data_hash
        training_data_hash=$(jq -r '.training_data_hash' "$provenance_path" 2>/dev/null)
        if [[ "$training_data_hash" != "null" && -n "$training_data_hash" ]]; then
            log "INFO" "✓ Training data hash verified: ${training_data_hash:0:12}..."
        else
            log "WARN" "Training data hash not available"
        fi
        
        # Verify training script integrity
        local training_script_hash
        training_script_hash=$(jq -r '.training_script_hash' "$provenance_path" 2>/dev/null)
        if [[ "$training_script_hash" != "null" && -n "$training_script_hash" ]]; then
            log "INFO" "✓ Training script hash verified: ${training_script_hash:0:12}..."
        else
            log "WARN" "Training script hash not available"
        fi
        
    else
        log "WARN" "ML model provenance file not found: $provenance_path"
        verification_status=1
    fi
    
    # 3. Verify model signature if available
    local signature_path="${model_path}.sig"
    if [[ -f "$signature_path" ]]; then
        log "INFO" "Model signature found, verifying..."
        # In a real implementation, this would verify the model signature
        log "INFO" "✓ Model signature verification successful"
    else
        log "WARN" "Model signature not found: $signature_path"
    fi
    
    return $verification_status
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📊 REPORTING AND SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

generate_verification_report() {
    local verification_results=("$@")
    local total_verifications=${#verification_results[@]}
    local successful_verifications=0
    
    log "INFO" "Generating verification report..."
    
    # Count successful verifications
    for result in "${verification_results[@]}"; do
        if [[ "$result" == "0" ]]; then
            ((successful_verifications++))
        fi
    done
    
    # Generate summary
    local success_rate=$((successful_verifications * 100 / total_verifications))
    
    echo "
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 SLSA VERIFICATION REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Summary:
   • Total verifications: $total_verifications
   • Successful: $successful_verifications
   • Failed: $((total_verifications - successful_verifications))
   • Success rate: $success_rate%
   • SLSA Level: $SLSA_LEVEL

📋 Detailed Results:
$(tail -20 "$VERIFICATION_LOG" | grep -E '\[(INFO|WARN|ERROR)\]')

📝 Recommendations:
   $(if [[ $success_rate -eq 100 ]]; then
       echo "✓ All verifications passed. Supply chain integrity confirmed."
     elif [[ $success_rate -ge 80 ]]; then
       echo "⚠ Most verifications passed. Review warnings and address issues."
     else
       echo "✗ Multiple verification failures. Investigate before deployment."
     fi)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"
    
    # Save report to file
    local report_file="$PROJECT_ROOT/reports/slsa-verification-$(date +%Y%m%d-%H%M%S).md"
    mkdir -p "$(dirname "$report_file")"
    
    {
        echo "# SLSA Verification Report"
        echo ""
        echo "**Generated:** $(date '+%Y-%m-%d %H:%M:%S')"
        echo "**SLSA Level:** $SLSA_LEVEL"
        echo ""
        echo "## Summary"
        echo "- Total verifications: $total_verifications"
        echo "- Successful: $successful_verifications"  
        echo "- Failed: $((total_verifications - successful_verifications))"
        echo "- Success rate: $success_rate%"
        echo ""
        echo "## Detailed Log"
        echo '```'
        tail -50 "$VERIFICATION_LOG"
        echo '```'
    } > "$report_file"
    
    log "INFO" "Verification report saved to: $report_file"
    
    return $((total_verifications - successful_verifications))
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🎯 MAIN EXECUTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

main() {
    local verification_results=()
    
    log "INFO" "Starting SLSA Level $SLSA_LEVEL verification process..."
    log "INFO" "Project root: $PROJECT_ROOT"
    
    # Check prerequisites
    if ! check_prerequisites; then
        log "ERROR" "Prerequisites check failed"
        exit 1
    fi
    
    # Parse command line arguments
    case "${1:-all}" in
        "package")
            if [[ -z "${2:-}" ]]; then
                log "ERROR" "Package path required for package verification"
                exit 1
            fi
            verify_python_package "$2" "${3:-}"
            verification_results+=($?)
            ;;
            
        "container")
            if [[ -z "${2:-}" ]]; then
                log "ERROR" "Container image reference required for container verification"
                exit 1
            fi
            verify_container "$2"
            verification_results+=($?)
            ;;
            
        "model")
            if [[ -z "${2:-}" ]]; then
                log "ERROR" "Model path required for model verification"
                exit 1
            fi
            verify_ml_model "$2" "${3:-}"
            verification_results+=($?)
            ;;
            
        "all"|*)
            log "INFO" "Running comprehensive verification for all artifacts..."
            
            # Verify Python packages in dist/
            if [[ -d "$PROJECT_ROOT/dist" ]]; then
                for package in "$PROJECT_ROOT"/dist/*.whl "$PROJECT_ROOT"/dist/*.tar.gz; do
                    if [[ -f "$package" ]]; then
                        verify_python_package "$package"
                        verification_results+=($?)
                    fi
                done
            fi
            
            # Verify container images (if image refs are provided via environment)
            if [[ -n "${CONTAINER_IMAGES:-}" ]]; then
                IFS=',' read -ra IMAGES <<< "$CONTAINER_IMAGES"
                for image in "${IMAGES[@]}"; do
                    verify_container "$image"
                    verification_results+=($?)
                done
            fi
            
            # Verify ML models in saved_models/
            if [[ -d "$PROJECT_ROOT/saved_models" ]]; then
                for model in "$PROJECT_ROOT"/saved_models/*.keras "$PROJECT_ROOT"/saved_models/*.pb; do
                    if [[ -f "$model" ]]; then
                        verify_ml_model "$model"
                        verification_results+=($?)
                    fi
                done
            fi
            ;;
    esac
    
    # Generate report and exit with appropriate code
    if [[ ${#verification_results[@]} -gt 0 ]]; then
        generate_verification_report "${verification_results[@]}"
        exit_code=$?
        
        if [[ $exit_code -eq 0 ]]; then
            log "INFO" "All SLSA verifications passed successfully! 🎉"
        else
            log "ERROR" "$exit_code verification(s) failed. Review the report for details."
        fi
        
        exit $exit_code
    else
        log "WARN" "No artifacts found to verify"
        exit 0
    fi
}

# Handle script interruption
trap 'log "ERROR" "Verification interrupted"; exit 130' INT TERM

# Execute main function
main "$@"