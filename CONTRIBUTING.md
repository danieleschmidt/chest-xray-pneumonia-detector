# Contributing Guidelines

Thank you for considering contributing to the Chest X-Ray Pneumonia Detector project! We welcome contributions of all kinds including bug fixes, documentation, new features, and improvements.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Docker (optional, for containerized development)

### Development Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/your-username/chest-xray-pneumonia-detector.git
   cd chest-xray-pneumonia-detector
   ```

2. **Set Up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements-dev.txt
   pip install -e .
   
   # Install pre-commit hooks
   pre-commit install
   ```

3. **Verify Setup**
   ```bash
   # Run tests
   pytest
   
   # Run linting
   ruff check src/ tests/
   black --check src/ tests/
   
   # Run security scan
   bandit -r src/
   ```

## Development Workflow

### Creating a Contribution

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

2. **Make Your Changes**
   - Write clean, readable code following the project's style guidelines
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure all existing tests pass

3. **Test Your Changes**
   ```bash
   # Run the full test suite
   pytest
   
   # Run with coverage
   pytest --cov=src --cov-report=html
   
   # Run performance tests
   pytest tests/performance/
   ```

4. **Code Quality Checks**
   ```bash
   # Lint and format code
   ruff check src/ tests/ --fix
   black src/ tests/
   
   # Security scan
   bandit -r src/
   
   # Type checking (if applicable)
   mypy src/
   ```

### Commit Guidelines

We follow conventional commit format for consistency:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(model): add attention mechanism to CNN architecture
fix(data): handle edge case in image preprocessing
docs(api): update inference API documentation
test(train): add unit tests for training callbacks
```

### Pull Request Process

1. **Ensure Quality Standards**
   - All tests pass (`pytest`)
   - Code coverage maintained (>85%)
   - No linting errors (`ruff`, `black`)
   - No security vulnerabilities (`bandit`)
   - Documentation updated

2. **Create Pull Request**
   - Use a descriptive title
   - Fill out the PR template completely
   - Link any related issues
   - Add appropriate labels

3. **Code Review Process**
   - All PRs require at least one reviewer approval
   - Address reviewer feedback promptly
   - Keep PRs focused and reasonably sized
   - Rebase if necessary to maintain clean history

## Code Style Guidelines

### Python Style
- Follow PEP 8 conventions
- Use Black for formatting (line length: 88)
- Use type hints where appropriate
- Write docstrings for all public functions/classes

### Documentation
- Update README.md for user-facing changes
- Add docstrings for new functions/classes
- Update API documentation as needed
- Include code examples in documentation

### Testing
- Write unit tests for all new functionality
- Aim for >85% code coverage
- Use descriptive test names
- Include integration tests for complex workflows
- Add performance tests for optimization changes

## Types of Contributions

### Bug Reports
When filing bug reports, please include:
- Python version and OS
- Steps to reproduce the issue
- Expected vs. actual behavior
- Error messages and stack traces
- Minimal code example if applicable

### Feature Requests
For new features, please provide:
- Clear description of the proposed feature
- Use case and motivation
- Possible implementation approach
- Any breaking changes considerations

### Documentation Improvements
- Fix typos and grammatical errors
- Improve clarity and completeness
- Add examples and tutorials
- Update outdated information

### Performance Improvements
- Include benchmarks showing improvement
- Ensure changes don't break existing functionality
- Consider backward compatibility
- Document any new dependencies

## Project Structure

```
chest-xray-pneumonia-detector/
├── src/                    # Source code
│   ├── chest_xray_pneumonia_detector/
│   ├── data_loader.py
│   ├── model_builder.py
│   └── ...
├── tests/                  # Test suite
│   ├── unit/
│   ├── integration/
│   └── performance/
├── docs/                   # Documentation
├── scripts/                # Build and utility scripts
├── monitoring/             # Monitoring configuration
└── requirements*.txt       # Dependencies
```

## Security Considerations

- Never commit sensitive data (API keys, passwords, etc.)
- Use environment variables for configuration
- Follow secure coding practices
- Report security vulnerabilities privately
- Run security scans before submitting PRs

## Community Guidelines

- Be respectful and inclusive
- Follow our [Code of Conduct](CODE_OF_CONDUCT.md)
- Help newcomers get started
- Provide constructive feedback
- Celebrate contributions from all skill levels

## Getting Help

- **Documentation**: Check the project documentation first
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use GitHub Discussions for questions
- **Code Review**: Request reviews from maintainers

## Recognition

Contributors are recognized in:
- CHANGELOG.md for significant contributions
- GitHub contributor statistics
- Special mentions in release notes

## Legal

By contributing to this project, you agree that your contributions will be licensed under the same [MIT License](LICENSE) that covers the project. You also certify that you have the right to submit the work under this license and agree to the [Developer Certificate of Origin](https://developercertificate.org/).

---

Thank you for contributing to making healthcare AI more accessible and reliable!
