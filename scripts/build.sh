#!/bin/bash
set -e

# Build script for Chest X-Ray Pneumonia Detector
# Usage: ./scripts/build.sh [target] [options]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Default values
TARGET="${1:-production}"
PUSH=false
TAG="latest"
REGISTRY=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH=true
            shift
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [target] [options]"
            echo ""
            echo "Targets:"
            echo "  development    Build development image"
            echo "  production     Build production image (default)"
            echo "  api           Build API image"
            echo "  inference     Build inference image"
            echo "  all           Build all images"
            echo ""
            echo "Options:"
            echo "  --push        Push images to registry after building"
            echo "  --tag TAG     Tag for the images (default: latest)"
            echo "  --registry    Registry prefix (e.g., docker.io/username)"
            echo "  --help        Show this help message"
            exit 0
            ;;
        *)
            TARGET="$1"
            shift
            ;;
    esac
done

# Set image name
IMAGE_NAME="pneumonia-detector"
if [[ -n "$REGISTRY" ]]; then
    FULL_IMAGE_NAME="$REGISTRY/$IMAGE_NAME"
else
    FULL_IMAGE_NAME="$IMAGE_NAME"
fi

echo "🏗️  Building Chest X-Ray Pneumonia Detector"
echo "Target: $TARGET"
echo "Tag: $TAG"
echo "Image: $FULL_IMAGE_NAME:$TAG"

# Function to build a specific target
build_target() {
    local target=$1
    local image_tag="$FULL_IMAGE_NAME-$target:$TAG"
    
    echo "📦 Building $target image..."
    docker build \
        --target "$target" \
        --tag "$image_tag" \
        --tag "$FULL_IMAGE_NAME-$target:latest" \
        .
    
    if [[ "$PUSH" == "true" ]]; then
        echo "🚀 Pushing $target image..."
        docker push "$image_tag"
        docker push "$FULL_IMAGE_NAME-$target:latest"
    fi
    
    echo "✅ Built $target image: $image_tag"
}

# Build based on target
case $TARGET in
    development)
        build_target "development"
        ;;
    production)
        build_target "production"
        # Also tag as main image
        docker tag "$FULL_IMAGE_NAME-production:$TAG" "$FULL_IMAGE_NAME:$TAG"
        docker tag "$FULL_IMAGE_NAME-production:latest" "$FULL_IMAGE_NAME:latest"
        if [[ "$PUSH" == "true" ]]; then
            docker push "$FULL_IMAGE_NAME:$TAG"
            docker push "$FULL_IMAGE_NAME:latest"
        fi
        ;;
    api)
        build_target "api"
        ;;
    inference)
        build_target "inference"
        ;;
    all)
        build_target "development"
        build_target "production"
        build_target "api"
        build_target "inference"
        # Tag production as main
        docker tag "$FULL_IMAGE_NAME-production:$TAG" "$FULL_IMAGE_NAME:$TAG"
        docker tag "$FULL_IMAGE_NAME-production:latest" "$FULL_IMAGE_NAME:latest"
        if [[ "$PUSH" == "true" ]]; then
            docker push "$FULL_IMAGE_NAME:$TAG"
            docker push "$FULL_IMAGE_NAME:latest"
        fi
        ;;
    *)
        echo "❌ Unknown target: $TARGET"
        echo "Valid targets: development, production, api, inference, all"
        exit 1
        ;;
esac

echo ""
echo "🎉 Build completed!"
echo ""
echo "Available images:"
docker images | grep "$IMAGE_NAME" | head -10

echo ""
echo "💡 Usage examples:"
echo "  # Run development container:"
echo "  docker run -it --rm $FULL_IMAGE_NAME-development:$TAG bash"
echo ""
echo "  # Run inference:"
echo "  docker run --rm -v \$(pwd)/data:/app/data $FULL_IMAGE_NAME-inference:$TAG"
echo ""
echo "  # Start full development environment:"
echo "  docker-compose up -d"