# Pull Request

## =Ë Description

<!-- Provide a brief description of the changes in this PR -->

### = Related Issues
<!-- Link any related issues using #issue_number -->
- Closes #
- Related to #

### <¯ Type of Change
<!-- Mark the relevant option with an "x" -->
- [ ] = Bug fix (non-breaking change which fixes an issue)
- [ ] ( New feature (non-breaking change which adds functionality)
- [ ] =¥ Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] =Ú Documentation update
- [ ] =' Code refactoring (no functional changes)
- [ ] ¡ Performance improvement
- [ ] >ê Test addition or improvement
- [ ] <× Build system or external dependency update

## >ê Testing

### Test Coverage
<!-- Describe the tests you added or modified -->
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Performance tests added/updated
- [ ] Manual testing completed

### Test Results
<!-- Paste relevant test output or describe manual testing -->
```bash
# Test command and output
pytest tests/ -v
```

## =Ê Performance Impact
<!-- If applicable, describe any performance implications -->
- [ ] No performance impact
- [ ] Performance improvement (describe below)
- [ ] Performance regression (explain and justify below)

**Performance Details:**
<!-- Provide benchmarks, profiling results, or timing comparisons -->

## = Security Considerations
<!-- Address any security implications -->
- [ ] No security impact
- [ ] Security improvement
- [ ] Potential security impact (explain below)

**Security Details:**
<!-- Describe any security considerations or changes -->

## =Ý Documentation
<!-- Check all that apply -->
- [ ] Code is self-documenting with clear variable/function names
- [ ] Docstrings added/updated for public functions
- [ ] README.md updated (if needed)
- [ ] API documentation updated (if needed)
- [ ] Changelog updated

##  Checklist

### Code Quality
- [ ] Code follows the project's style guidelines
- [ ] Self-review of code completed
- [ ] Code is properly commented, particularly in hard-to-understand areas
- [ ] No debugging prints or commented-out code left behind

### Testing & Validation
- [ ] All tests pass locally (`pytest`)
- [ ] Code coverage maintained or improved
- [ ] Linting passes (`ruff check .`)
- [ ] Formatting applied (`black .`)
- [ ] Security scan passes (`bandit -r src/`)
- [ ] Pre-commit hooks pass

### Dependencies & Compatibility
- [ ] No new dependencies added, or dependencies are justified
- [ ] Changes are compatible with supported Python versions (3.8+)
- [ ] No breaking changes to public APIs (or breaking changes are documented)

### Deployment & Operations
- [ ] Changes work in containerized environment
- [ ] Environment variables documented (if new ones added)
- [ ] Migration steps documented (if needed)

## <¯ Review Focus Areas
<!-- Guide reviewers on what to focus on -->
Please pay special attention to:
- [ ] Algorithm correctness
- [ ] Error handling
- [ ] Resource usage
- [ ] API design
- [ ] Security implications
- [ ] Performance impact

## =ø Screenshots/Videos
<!-- If applicable, add screenshots or videos showing the changes -->

## > Questions for Reviewers
<!-- Any specific questions or concerns you'd like reviewers to address -->

## <‰ Post-Merge Tasks
<!-- Any tasks that need to be completed after this PR is merged -->
- [ ] Update deployment documentation
- [ ] Notify stakeholders
- [ ] Monitor performance metrics
- [ ] Update related issues

---

**Additional Notes:**
<!-- Any other information that would be helpful for reviewers -->

By submitting this pull request, I confirm that:
- [ ] I have read and agree to the project's [Code of Conduct](CODE_OF_CONDUCT.md)
- [ ] My contributions are licensed under the same license as this project
- [ ] I have followed the [Contributing Guidelines](CONTRIBUTING.md)