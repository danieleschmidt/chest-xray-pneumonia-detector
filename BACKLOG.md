# Development Backlog - WSJF Prioritized

## Scoring Methodology
- **Business Value (1-10)**: Impact on users/stakeholders
- **Time Criticality (1-10)**: Urgency of completion
- **Risk Reduction (1-10)**: Risk mitigation value
- **Job Size (1-20)**: Effort/complexity estimate
- **WSJF Score**: (Business Value + Time Criticality + Risk Reduction) / Job Size

## Recently Completed (2025-07-19)

### ✅ 1. Add Comprehensive Tests for grad_cam.py
- **WSJF Score**: 2.0 
- **Status**: COMPLETED
- **Description**: Added comprehensive test suite covering generate_grad_cam function with mocking for TensorFlow dependencies. Tests include binary/multiclass classification, edge cases, and error handling.

### ✅ 2. Enhance evaluate.py Test Coverage  
- **WSJF Score**: 1.8
- **Status**: COMPLETED
- **Description**: Expanded test coverage from basic CLI help to comprehensive testing of evaluate_predictions function including binary/multiclass metrics, CSV export, normalization, thresholds, and error handling.

### ✅ 3. Enhance inference.py Test Coverage
- **WSJF Score**: 1.6
- **Status**: COMPLETED  
- **Description**: Enhanced test coverage for predict_directory function including binary/multiclass prediction, model loading, data generation, file path handling, and edge cases.

### ✅ 4. Enhance predict_utils.py Test Coverage
- **WSJF Score**: 1.3
- **Status**: COMPLETED
- **Description**: Added comprehensive test coverage for load_image and display_grad_cam functions with 13 test cases covering image loading, normalization, Grad-CAM generation, heatmap processing, and error handling.

### ✅ 5. Add Integration Tests for Full Pipeline
- **WSJF Score**: 1.4
- **Status**: COMPLETED
- **Description**: Implemented comprehensive end-to-end pipeline tests with 8 test cases covering train→predict→evaluate workflow, data preprocessing, error handling, and CLI interface validation using minimal dummy data and proper mocking.

## High Priority Items (WSJF > 1.0)

### 1. Add Performance Benchmarking
- **WSJF Score**: 1.2
- **Business Value**: 7 (Important for production deployment)
- **Time Criticality**: 2 (Not immediately critical)
- **Risk Reduction**: 3 (Nice to have metrics)
- **Job Size**: 10 (Medium complexity - timing/memory tests)
- **Description**: Add benchmarks for training and inference performance.

## Medium Priority Items (WSJF 0.5-1.0)

### 7. Add Model Architecture Validation Tests
- **WSJF Score**: 0.9
- **Business Value**: 6 (Ensures model correctness)
- **Time Criticality**: 2 (Not urgent)
- **Risk Reduction**: 4 (Model structure validation)
- **Job Size**: 13 (High complexity - architecture testing)
- **Description**: Validate model layer configurations and shapes.

### 8. Enhance version_cli.py Test Coverage
- **WSJF Score**: 0.8
- **Business Value**: 3 (Simple utility function)
- **Time Criticality**: 2 (Low priority)
- **Risk Reduction**: 3 (Currently minimal testing)
- **Job Size**: 5 (Low complexity - simple CLI testing)
- **Description**: Add version format validation and error handling tests.

### 9. Add Dependency Security Scanning
- **WSJF Score**: 0.7
- **Business Value**: 5 (Security assurance)
- **Time Criticality**: 3 (Moderate security concern)
- **Risk Reduction**: 6 (Important for production)
- **Job Size**: 8 (Medium-low complexity - tool integration)
- **Description**: Integrate safety/pip-audit for dependency vulnerability scanning.

### 10. Add Real Data Integration Tests
- **WSJF Score**: 0.6
- **Business Value**: 6 (Validation with real datasets)
- **Time Criticality**: 2 (Not urgent)
- **Risk Reduction**: 4 (Ensures real-world compatibility)
- **Job Size**: 15 (High complexity - requires real data setup)
- **Description**: Test pipeline with actual medical image datasets.

## Low Priority Items (WSJF < 0.5)

### 11. Documentation Enhancements
- **WSJF Score**: 0.4
- **Business Value**: 4 (Improves developer experience)
- **Time Criticality**: 1 (Can be done anytime)
- **Risk Reduction**: 2 (Low risk impact)
- **Job Size**: 8 (Medium-low complexity)
- **Description**: Add more detailed docstrings and API documentation.

### 12. Code Quality Improvements
- **WSJF Score**: 0.3
- **Business Value**: 3 (Code maintainability)
- **Time Criticality**: 1 (Can be done anytime)
- **Risk Reduction**: 2 (Minor risk reduction)
- **Job Size**: 12 (Medium-high complexity - refactoring)
- **Description**: Address code style and reduce technical debt.

## Backlog Maintenance
- **Last Updated**: 2025-07-19
- **Next Review**: After completing top 3 items
- **Methodology**: WSJF scoring with 1-week job size normalization