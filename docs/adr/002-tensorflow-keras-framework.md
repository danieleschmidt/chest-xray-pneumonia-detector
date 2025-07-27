# ADR-002: TensorFlow/Keras as Primary ML Framework

**Status**: Accepted  
**Date**: 2025-07-27  
**Deciders**: Development Team  

## Context

The project requires a robust machine learning framework for building and training convolutional neural networks for medical image classification. Key considerations include ease of use, performance, community support, and integration capabilities.

## Decision

We will use TensorFlow/Keras as the primary machine learning framework for the Chest X-Ray Pneumonia Detector project.

## Rationale

1. **High-level API**: Keras provides intuitive, user-friendly APIs for rapid prototyping
2. **Transfer Learning Support**: Excellent pre-trained model ecosystem (VGG, ResNet, MobileNet)
3. **Medical AI Ecosystem**: Strong adoption in medical imaging applications
4. **Production Readiness**: TensorFlow Serving for scalable model deployment
5. **Documentation**: Comprehensive documentation and tutorials
6. **MLflow Integration**: Native support for experiment tracking

## Consequences

### Positive
- Faster development with high-level APIs
- Access to state-of-the-art pre-trained models
- Strong community and ecosystem support
- Production deployment capabilities
- Excellent visualization tools (TensorBoard)

### Negative
- Larger memory footprint compared to lightweight frameworks
- Potential overkill for simple models
- Learning curve for team members unfamiliar with TensorFlow