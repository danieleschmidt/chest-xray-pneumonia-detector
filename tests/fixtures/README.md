# Test Fixtures

This directory contains test fixtures and sample data used across the test suite.

## Directory Structure

```
fixtures/
├── README.md           # This file
├── images/            # Sample medical images for testing
│   ├── normal/        # Normal chest X-ray samples
│   └── pneumonia/     # Pneumonia chest X-ray samples
├── models/            # Pre-trained model artifacts for testing
├── configs/           # Configuration files for testing
└── data/             # Structured test datasets
    ├── small/        # Small datasets for unit tests
    ├── medium/       # Medium datasets for integration tests
    └── large/        # Large datasets for performance tests
```

## Usage Guidelines

### Image Fixtures
- Use synthetic or anonymized medical images only
- Images should be representative but not contain actual patient data
- Include various image sizes and formats for comprehensive testing

### Model Fixtures
- Small, fast-training models for testing purposes
- Pre-trained weights for consistent test results
- Model configurations matching production architectures

### Configuration Fixtures
- Various configuration scenarios (valid/invalid)
- Edge cases for configuration validation
- Performance benchmarking configurations

## Adding New Fixtures

1. **Create appropriately sized fixtures**: Keep fixtures minimal but representative
2. **Document fixture purpose**: Add comments explaining what each fixture tests
3. **Use synthetic data**: Never include real patient data or sensitive information
4. **Version control friendly**: Keep binary files small and use Git LFS if needed

## Security Notes

- All fixtures are reviewed for privacy compliance
- No real medical data is included in this repository
- Synthetic data generation scripts are available in `src/synthetic_medical_data_generator.py`