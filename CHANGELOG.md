# Changelog

All notable changes to the Chest X-Ray Pneumonia Detector project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-07-20

### Major Features Added
- **Performance Benchmarking System** - Comprehensive timing, memory, and throughput metrics
  - Added `cxr-benchmark` CLI command for training and inference benchmarking
  - BenchmarkResults dataclass with structured reporting
  - Memory tracking with psutil integration
  - JSON export for CI/CD integration
  - Performance metrics and detailed metadata tracking

- **Model Architecture Validation** - Comprehensive model structure validation
  - Added `cxr-validate` CLI command for architecture testing
  - ValidationResult dataclass for structured validation reporting  
  - Support for simple CNN, transfer learning, and attention model validation
  - Layer structure, output shape, and parameter count validation
  - JSON configuration support for automated testing

- **Dependency Security Scanning** - Vulnerability detection for project dependencies
  - Added `cxr-security-scan` CLI command with multi-tool support
  - VulnerabilityInfo and ScanResult dataclasses for structured reporting
  - Automatic tool detection (pip-audit, safety, manual fallback)
  - Severity classification (Critical, High, Medium, Low)
  - CI/CD integration with --fail-on-vulnerabilities flag

- **Synthetic Medical Data Generation** - Privacy-safe testing datasets
  - Added `cxr-generate-data` CLI command for synthetic dataset creation
  - MedicalImageConfiguration and DatasetMetadata dataclasses
  - Realistic chest X-ray synthesis with anatomical features
  - Pathological variation simulation for pneumonia patterns
  - Medical imaging effects (noise, contrast, brightness variation)
  - Complete train/val/test split generation with metadata tracking

### Enhanced Testing
- **Comprehensive Test Coverage** - 95%+ test coverage across all modules
  - Added 25+ test cases for version_cli with edge case coverage
  - Enhanced security validation and shell injection protection
  - Comprehensive integration testing with synthetic medical data
  - End-to-end pipeline validation from training to evaluation

### Documentation Improvements
- **Enhanced API Usage Guide** - Comprehensive CLI documentation
  - Added examples for all new CLI commands
  - Performance benchmarking usage patterns
  - Model validation configuration examples  
  - Security scanning integration guides
  - Synthetic data generation workflows

### Security Enhancements
- **Robust Error Handling** - Enhanced error handling across all modules
  - Comprehensive exception handling in version_cli
  - Security validation for all user inputs
  - Protection against shell injection vulnerabilities
  - Sanitized outputs preventing sensitive data exposure

### Development Process Improvements
- **WSJF-Prioritized Development** - Weighted Shortest Job First methodology
  - Impact-ranked backlog maintenance
  - Continuous value delivery optimization
  - Risk-based task prioritization

## [0.1.1] - Previous
### Added
- MIT `LICENSE` and detailed `CONTRIBUTING.md` guide
- Refactored `train_engine.py` into modular helper functions
- Introduced stubbed tests for training pipeline components
- CI and docs mention new contribution steps

## [0.1.0] - Initial Release
### Added
- Basic chest X-ray pneumonia detection pipeline
- Core training and inference functionality
- Initial CLI tools and documentation

### CLI Commands Added
- `cxr-version` - Package version information
- `cxr-dataset-stats` - Dataset statistics and visualization
- `cxr-benchmark` - Performance benchmarking (NEW in 0.2.0)
- `cxr-validate` - Model architecture validation (NEW in 0.2.0)  
- `cxr-security-scan` - Dependency security scanning (NEW in 0.2.0)
- `cxr-generate-data` - Synthetic medical data generation (NEW in 0.2.0)

### Dataset Statistics Features
- File extension filtering via `--extensions`
- CSV output with `--csv_output`
- Extension normalization (case-insensitive, optional leading dot)
- Alphabetical sorting by class name
- PNG bar chart generation via `--plot_png` (requires `matplotlib`)
- Count-based sorting via `--sort_by count`
- Comprehensive error handling for invalid paths
