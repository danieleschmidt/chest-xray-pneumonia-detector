# Security Policy - Chest X-Ray Pneumonia Detector

## Dependency Management

1. **Version Pinning**: All production dependencies MUST be pinned to specific versions
2. **Security Scanning**: Run `pip-audit` before each deployment
3. **Update Schedule**: Security patches applied within 48 hours of disclosure
4. **Approval Process**: All dependency updates require security team review

## Known Security Considerations

### MLflow
- Multiple CVEs patched in version 2.19.0
- Ensure proper authentication is configured
- Disable public access to tracking server

### TensorFlow/Keras
- Safe mode limitations in Keras - verify model sources
- Do not load untrusted H5 model files
- Use SavedModel format when possible

### scikit-learn
- Never use joblib.load() on untrusted files
- Validate all model inputs

## HIPAA Compliance

- All PHI must be encrypted using AES-256
- Implement comprehensive audit logging
- Follow minimum necessary standard for data access
- Regular security assessments required

## Security Contacts

- Security issues: security@your-organization.com
- HIPAA Officer: hipaa@your-organization.com
