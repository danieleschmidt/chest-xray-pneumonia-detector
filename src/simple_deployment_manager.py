#!/usr/bin/env python3
"""
Simple Deployment Manager
Generation 1: Basic deployment automation for medical AI models
"""

import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import argparse


class SimpleDeploymentManager:
    """Basic deployment manager for medical AI models."""
    
    def __init__(self, deployment_dir: str = "deployments"):
        self.deployment_dir = Path(deployment_dir)
        self.deployment_dir.mkdir(exist_ok=True)
        self.config_file = self.deployment_dir / "deployment_config.json"
    
    def create_deployment_package(self, model_path: str, version: str, 
                                description: str = "") -> str:
        """Create a deployment package for a trained model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        deployment_id = f"deploy_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        deployment_path = self.deployment_dir / deployment_id
        deployment_path.mkdir(exist_ok=True)
        
        # Copy model file
        model_name = os.path.basename(model_path)
        shutil.copy2(model_path, deployment_path / model_name)
        
        # Create deployment manifest
        manifest = {
            "deployment_id": deployment_id,
            "model_version": version,
            "model_file": model_name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "status": "created",
            "requirements": self._get_basic_requirements()
        }
        
        with open(deployment_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        
        # Create simple Docker deployment files
        self._create_docker_files(deployment_path, model_name, version)
        
        # Update deployment config
        self._update_deployment_config(deployment_id, manifest)
        
        return str(deployment_path)
    
    def _get_basic_requirements(self) -> List[str]:
        """Get basic requirements for model deployment."""
        return [
            "tensorflow>=2.17.0",
            "numpy>=1.26.0",
            "Pillow>=10.0.0",
            "flask>=3.0.0",
            "gunicorn>=23.0.0"
        ]
    
    def _create_docker_files(self, deployment_path: Path, model_name: str, version: str):
        """Create basic Docker deployment files."""
        
        # Simple Dockerfile
        dockerfile_content = f"""FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and application files
COPY {model_name} ./model/
COPY app.py .
COPY manifest.json .

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120", "app:app"]
"""
        
        with open(deployment_path / "Dockerfile", "w") as f:
            f.write(dockerfile_content)
        
        # Simple Flask app for model serving
        app_content = f"""#!/usr/bin/env python3
import json
import os
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load model (placeholder - would load actual TensorFlow model)
MODEL_PATH = "model/{model_name}"
model = None  # Would load with tf.keras.models.load_model(MODEL_PATH)

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({{"status": "healthy", "model_loaded": model is not None}})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Basic prediction endpoint
        data = request.get_json()
        
        if "image" not in data:
            return jsonify({{"error": "No image provided"}}), 400
        
        # Placeholder prediction logic
        # In real implementation, would decode image and run model inference
        prediction = 0.5  # Dummy prediction
        confidence = 0.85  # Dummy confidence
        
        return jsonify({{
            "prediction": prediction,
            "confidence": confidence,
            "predicted_class": "PNEUMONIA" if prediction > 0.5 else "NORMAL",
            "model_version": "{version}"
        }})
    
    except Exception as e:
        return jsonify({{"error": str(e)}}), 500

@app.route("/info", methods=["GET"])
def model_info():
    with open("manifest.json", "r") as f:
        manifest = json.load(f)
    return jsonify(manifest)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
"""
        
        with open(deployment_path / "app.py", "w") as f:
            f.write(app_content)
        
        # Requirements file
        requirements = "\\n".join(self._get_basic_requirements())
        with open(deployment_path / "requirements.txt", "w") as f:
            f.write(requirements)
    
    def _update_deployment_config(self, deployment_id: str, manifest: Dict):
        """Update the global deployment configuration."""
        config = {}
        if self.config_file.exists():
            with open(self.config_file, "r") as f:
                config = json.load(f)
        
        if "deployments" not in config:
            config["deployments"] = {}
        
        config["deployments"][deployment_id] = {
            "created_at": manifest["created_at"],
            "model_version": manifest["model_version"],
            "status": manifest["status"],
            "description": manifest["description"]
        }
        
        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)
    
    def list_deployments(self) -> List[Dict]:
        """List all available deployments."""
        if not self.config_file.exists():
            return []
        
        with open(self.config_file, "r") as f:
            config = json.load(f)
        
        deployments = []
        for deployment_id, info in config.get("deployments", {}).items():
            deployment_path = self.deployment_dir / deployment_id
            deployments.append({
                "deployment_id": deployment_id,
                "path": str(deployment_path),
                "exists": deployment_path.exists(),
                **info
            })
        
        return deployments
    
    def build_deployment(self, deployment_id: str) -> bool:
        """Build Docker image for deployment."""
        deployment_path = self.deployment_dir / deployment_id
        
        if not deployment_path.exists():
            raise FileNotFoundError(f"Deployment not found: {deployment_id}")
        
        try:
            # Build Docker image
            cmd = [
                "docker", "build", 
                "-t", f"medical-ai:{deployment_id}",
                str(deployment_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Successfully built Docker image: medical-ai:{deployment_id}")
                return True
            else:
                print(f"Docker build failed: {result.stderr}")
                return False
        
        except FileNotFoundError:
            print("Docker not found. Please install Docker to build deployments.")
            return False
    
    def generate_deployment_script(self, deployment_id: str) -> str:
        """Generate a deployment script for the given deployment."""
        deployment_path = self.deployment_dir / deployment_id
        
        script_content = f"""#!/bin/bash
# Deployment script for {deployment_id}
# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

set -e

echo "Starting deployment of {deployment_id}..."

# Build Docker image
echo "Building Docker image..."
docker build -t medical-ai:{deployment_id} {deployment_path}

# Stop existing container if running
echo "Stopping existing container..."
docker stop medical-ai-{deployment_id} 2>/dev/null || true
docker rm medical-ai-{deployment_id} 2>/dev/null || true

# Run new container
echo "Starting new container..."
docker run -d \\
    --name medical-ai-{deployment_id} \\
    --restart unless-stopped \\
    -p 8080:8080 \\
    medical-ai:{deployment_id}

echo "Deployment completed successfully!"
echo "Service available at: http://localhost:8080"
echo "Health check: curl http://localhost:8080/health"
"""
        
        script_path = deployment_path / "deploy.sh"
        with open(script_path, "w") as f:
            f.write(script_content)
        
        # Make script executable
        script_path.chmod(0o755)
        
        return str(script_path)


def main():
    """CLI interface for deployment manager."""
    parser = argparse.ArgumentParser(description="Simple Deployment Manager")
    parser.add_argument("--deployment-dir", default="deployments", 
                       help="Deployment directory")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create deployment command
    create_parser = subparsers.add_parser("create", help="Create deployment package")
    create_parser.add_argument("--model-path", required=True, 
                              help="Path to trained model file")
    create_parser.add_argument("--version", required=True, 
                              help="Model version")
    create_parser.add_argument("--description", default="", 
                              help="Deployment description")
    
    # List deployments command
    list_parser = subparsers.add_parser("list", help="List deployments")
    
    # Build deployment command
    build_parser = subparsers.add_parser("build", help="Build deployment")
    build_parser.add_argument("deployment_id", help="Deployment ID to build")
    
    # Generate script command
    script_parser = subparsers.add_parser("script", help="Generate deployment script")
    script_parser.add_argument("deployment_id", help="Deployment ID")
    
    args = parser.parse_args()
    
    manager = SimpleDeploymentManager(args.deployment_dir)
    
    if args.command == "create":
        try:
            deployment_path = manager.create_deployment_package(
                args.model_path, args.version, args.description
            )
            print(f"Deployment package created: {deployment_path}")
            
            # Generate deployment script
            deployment_id = os.path.basename(deployment_path)
            script_path = manager.generate_deployment_script(deployment_id)
            print(f"Deployment script created: {script_path}")
            
        except Exception as e:
            print(f"Error creating deployment: {e}")
    
    elif args.command == "list":
        deployments = manager.list_deployments()
        if not deployments:
            print("No deployments found.")
        else:
            print("Available deployments:")
            for dep in deployments:
                status = "✓" if dep["exists"] else "✗"
                print(f"  {status} {dep['deployment_id']} (v{dep['model_version']}) - {dep['description']}")
    
    elif args.command == "build":
        try:
            success = manager.build_deployment(args.deployment_id)
            if success:
                print(f"Deployment {args.deployment_id} built successfully!")
            else:
                print(f"Failed to build deployment {args.deployment_id}")
        except Exception as e:
            print(f"Error building deployment: {e}")
    
    elif args.command == "script":
        try:
            script_path = manager.generate_deployment_script(args.deployment_id)
            print(f"Deployment script generated: {script_path}")
        except Exception as e:
            print(f"Error generating script: {e}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()