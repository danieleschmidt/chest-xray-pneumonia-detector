# GitHub Workflows Documentation

## Overview

This directory contains comprehensive documentation and templates for GitHub Actions workflows. Due to GitHub App permission limitations, these workflows must be manually created by repository maintainers.

## Required Workflows

The following workflows are essential for the Chest X-Ray Pneumonia Detector project:

### 1. Continuous Integration (CI) - `ci.yml`
**Purpose**: Automated testing, linting, and quality gates for pull requests  
**Triggers**: Pull requests, pushes to main/develop branches  
**Key Features**:
- Multi-Python version testing (3.8, 3.9, 3.10)
- Code quality checks (ruff, black, bandit)
- Test execution with coverage reporting
- Security scanning
- Performance benchmarking

### 2. Continuous Deployment (CD) - `cd.yml`
**Purpose**: Automated deployment to staging and production environments  
**Triggers**: Pushes to main branch, manual dispatch  
**Key Features**:
- Multi-stage deployment (staging → production)
- Docker image building and scanning
- SBOM generation
- Rollback capabilities
- Deployment notifications

### 3. Security Scanning - `security.yml`
**Purpose**: Comprehensive security analysis and vulnerability management  
**Triggers**: Daily schedule, pull requests  
**Key Features**:
- Dependency vulnerability scanning
- Container image security analysis
- SAST (Static Application Security Testing)
- DAST (Dynamic Application Security Testing)
- Security report generation

### 4. Dependency Updates - `dependency-update.yml`
**Purpose**: Automated dependency updates and security patches  
**Triggers**: Weekly schedule, manual dispatch  
**Key Features**:
- Dependabot configuration
- Automated testing of updates
- Security-focused updates
- PR creation with test results

### 5. Release Management - `release.yml`
**Purpose**: Automated release creation and artifact publishing  
**Triggers**: Tag creation, manual dispatch  
**Key Features**:
- Semantic versioning
- Release notes generation
- Artifact publishing
- Docker image tagging and pushing

## Quick Setup Guide

### 1. Manual Workflow Creation

Repository maintainers must manually create workflow files in `.github/workflows/` using the templates provided in `examples/`.

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy templates (manual process)
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/cd.yml .github/workflows/
cp docs/workflows/examples/security.yml .github/workflows/
cp docs/workflows/examples/dependency-update.yml .github/workflows/
cp docs/workflows/examples/release.yml .github/workflows/
```

### 2. Required Secrets Configuration

Add the following secrets in GitHub repository settings:

#### Docker Registry
- `DOCKER_REGISTRY_URL`: Container registry URL
- `DOCKER_REGISTRY_USERNAME`: Registry username
- `DOCKER_REGISTRY_PASSWORD`: Registry password/token

#### Security Scanning
- `SNYK_TOKEN`: Snyk vulnerability scanning token
- `SONAR_TOKEN`: SonarCloud analysis token

#### Notifications
- `SLACK_WEBHOOK`: Slack webhook for deployment notifications
- `TEAMS_WEBHOOK`: Microsoft Teams webhook

#### Cloud Deployment (if applicable)
- `AWS_ACCESS_KEY_ID`: AWS deployment credentials
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `AZURE_CREDENTIALS`: Azure service principal
- `GCP_SERVICE_ACCOUNT_KEY`: Google Cloud service account

### 3. Environment Configuration

Create environment files for different deployment stages:

#### Staging Environment (`staging.env`)
```bash
API_HOST=staging-api.example.com
DATABASE_URL=postgresql://staging-db
LOG_LEVEL=DEBUG
MLFLOW_TRACKING_URI=http://staging-mlflow:5000
```

#### Production Environment (`production.env`)
```bash
API_HOST=api.example.com
DATABASE_URL=postgresql://prod-db
LOG_LEVEL=INFO
MLFLOW_TRACKING_URI=http://prod-mlflow:5000
```

## Workflow Features

### Security Integration
- **Vulnerability Scanning**: Automated dependency and container scanning
- **SAST Analysis**: Static code analysis for security issues
- **Secret Detection**: Prevention of secret leakage
- **Compliance Checks**: SLSA framework compliance

### Quality Gates
- **Test Coverage**: Minimum 85% coverage requirement
- **Code Quality**: Linting and formatting checks
- **Performance**: Benchmark validation
- **Security**: No high/critical vulnerabilities

### Deployment Strategy
- **Blue-Green Deployment**: Zero-downtime deployments
- **Canary Releases**: Gradual rollout with monitoring
- **Rollback Capability**: Automatic rollback on failure
- **Health Checks**: Comprehensive post-deployment validation

### Monitoring Integration
- **Deployment Metrics**: Track deployment success/failure
- **Performance Monitoring**: Monitor deployment impact
- **Alert Integration**: Notify on deployment issues
- **Dashboard Updates**: Automatic metric collection

## Best Practices

### Workflow Security
1. **Use OIDC authentication** where possible instead of long-lived tokens
2. **Limit permissions** to minimum required scope
3. **Pin action versions** to specific commits or tags
4. **Validate inputs** and sanitize user-provided data
5. **Use environment protection rules** for production deployments

### Performance Optimization
1. **Cache dependencies** to reduce build times
2. **Use matrix builds** for parallel execution
3. **Optimize Docker layers** for faster builds
4. **Skip unnecessary jobs** based on file changes

### Reliability
1. **Implement retry logic** for flaky operations
2. **Use timeout settings** to prevent hanging jobs
3. **Add comprehensive error handling**
4. **Monitor workflow performance** and optimize

## Troubleshooting

### Common Issues

#### Workflow Not Triggering
- Check trigger conditions and branch filters
- Verify repository permissions and secrets
- Review workflow syntax and indentation

#### Test Failures
- Check test environment setup
- Verify dependency installation
- Review test data and fixtures

#### Deployment Failures
- Validate environment configuration
- Check service health endpoints
- Review deployment logs and metrics

### Debug Commands

```bash
# Validate workflow syntax locally
act --list

# Dry run workflow
act push --dry-run

# Check workflow runs
gh run list --workflow=ci.yml

# View workflow logs
gh run view <run-id> --log
```

## Integration with External Tools

### Code Quality
- **SonarCloud**: Code quality and security analysis
- **CodeClimate**: Technical debt and maintainability
- **Snyk**: Vulnerability scanning and monitoring

### Deployment Platforms
- **Docker Hub/GHCR**: Container registry
- **AWS/Azure/GCP**: Cloud deployment platforms
- **Kubernetes**: Container orchestration

### Monitoring
- **DataDog**: Application performance monitoring
- **New Relic**: Full-stack observability
- **Prometheus/Grafana**: Metrics and visualization

## Compliance and Governance

### Audit Requirements
- All workflows must log deployment activities
- Security scans must be performed on every release
- SBOM must be generated for all container images
- Compliance reports must be archived

### Approval Process
- Production deployments require manual approval
- Security exceptions need security team review
- Infrastructure changes require architecture review

### Documentation Requirements
- All workflows must be documented
- Runbooks must be updated for new procedures
- Emergency procedures must be tested quarterly

## Support and Maintenance

### Workflow Updates
- Review and update workflows monthly
- Test workflow changes in feature branches
- Document all workflow modifications
- Monitor workflow performance metrics

### Training and Knowledge Sharing
- Team training on workflow modifications
- Documentation of workflow troubleshooting
- Knowledge sharing sessions on CI/CD best practices

---

**Document Version**: 1.0  
**Last Updated**: 2025-07-28  
**Next Review**: 2025-08-28  
**Owner**: DevOps Team  
**Manual Setup Required**: ⚠️ Yes - Workflows must be created manually due to GitHub App permissions