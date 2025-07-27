# ADR-003: Modular CLI Architecture

**Status**: Accepted  
**Date**: 2025-07-27  
**Deciders**: Development Team  

## Context

The project needs a user-friendly interface that supports various workflows including training, inference, evaluation, and data processing. The interface should be scriptable, testable, and suitable for both development and production environments.

## Decision

We will implement a modular CLI architecture where each major component exposes its functionality through command-line interfaces, organized as Python modules that can be invoked via `python -m` or dedicated entry points.

## Implementation Details

- Each component (train_engine, inference, evaluate, etc.) provides CLI functionality
- Entry points defined in `pyproject.toml` for convenient access (e.g., `cxr-version`, `cxr-dataset-stats`)
- Consistent argument parsing using argparse
- Comprehensive help documentation for each command

## Consequences

### Positive
- Clear separation of concerns between components
- Easy to test individual components in isolation
- Scriptable for automation and CI/CD pipelines
- User-friendly for both developers and end users
- Supports both module invocation and direct commands

### Negative
- Additional complexity in maintaining CLI interfaces
- Need to ensure consistency across different command interfaces
- Potential for API drift between components