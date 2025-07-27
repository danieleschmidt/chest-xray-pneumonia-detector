#!/usr/bin/env python3
"""
Script to update version numbers across project files.
Used by semantic-release for automated version management.
"""

import sys
import re
from pathlib import Path
import tomllib
import tomli_w


def update_pyproject_toml(version: str):
    """Update version in pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    
    if not pyproject_path.exists():
        print(f"Warning: {pyproject_path} not found")
        return
    
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    
    data["project"]["version"] = version
    
    with open(pyproject_path, "wb") as f:
        tomli_w.dump(data, f)
    
    print(f"Updated {pyproject_path} to version {version}")


def update_version_cli(version: str):
    """Update version in version_cli.py."""
    version_file = Path("src/version_cli.py")
    
    if not version_file.exists():
        print(f"Warning: {version_file} not found")
        return
    
    content = version_file.read_text()
    
    # Update __version__ variable
    new_content = re.sub(
        r'__version__\s*=\s*["\'][^"\']*["\']',
        f'__version__ = "{version}"',
        content
    )
    
    version_file.write_text(new_content)
    print(f"Updated {version_file} to version {version}")


def update_dockerfile(version: str):
    """Update version labels in Dockerfile."""
    dockerfile = Path("Dockerfile")
    
    if not dockerfile.exists():
        print(f"Warning: {dockerfile} not found")
        return
    
    content = dockerfile.read_text()
    
    # Update version label
    new_content = re.sub(
        r'LABEL version="[^"]*"',
        f'LABEL version="{version}"',
        content
    )
    
    # If no version label exists, don't add one
    if new_content == content:
        print(f"No version label found in {dockerfile}")
        return
    
    dockerfile.write_text(new_content)
    print(f"Updated {dockerfile} to version {version}")


def update_docker_compose(version: str):
    """Update image tags in docker-compose files."""
    for compose_file in ["docker-compose.yml", "docker-compose.prod.yml"]:
        compose_path = Path(compose_file)
        
        if not compose_path.exists():
            continue
        
        content = compose_path.read_text()
        
        # Update image tags
        new_content = re.sub(
            r'image:\s*([^:\s]+):latest',
            rf'image: \1:{version}',
            content
        )
        
        # Update image tags with version numbers
        new_content = re.sub(
            r'image:\s*([^:\s]+):v?\d+\.\d+\.\d+',
            rf'image: \1:{version}',
            new_content
        )
        
        if new_content != content:
            compose_path.write_text(new_content)
            print(f"Updated {compose_path} to version {version}")


def update_kubernetes_manifests(version: str):
    """Update image tags in Kubernetes manifests."""
    k8s_dir = Path("k8s")
    
    if not k8s_dir.exists():
        return
    
    for manifest_file in k8s_dir.glob("*.yaml"):
        content = manifest_file.read_text()
        
        # Update image tags
        new_content = re.sub(
            r'image:\s*([^:\s]+):latest',
            rf'image: \1:{version}',
            content
        )
        
        new_content = re.sub(
            r'image:\s*([^:\s]+):v?\d+\.\d+\.\d+',
            rf'image: \1:{version}',
            new_content
        )
        
        if new_content != content:
            manifest_file.write_text(new_content)
            print(f"Updated {manifest_file} to version {version}")


def update_documentation(version: str):
    """Update version references in documentation."""
    docs_dir = Path("docs")
    
    if not docs_dir.exists():
        return
    
    # Update version in documentation files
    for doc_file in docs_dir.rglob("*.md"):
        content = doc_file.read_text()
        
        # Update version references
        new_content = re.sub(
            r'version-\d+\.\d+\.\d+',
            f'version-{version}',
            content
        )
        
        new_content = re.sub(
            r'v\d+\.\d+\.\d+',
            f'v{version}',
            new_content
        )
        
        if new_content != content:
            doc_file.write_text(new_content)
            print(f"Updated {doc_file} to version {version}")


def main():
    """Main function to update all version references."""
    if len(sys.argv) != 2:
        print("Usage: python update_version.py <version>")
        sys.exit(1)
    
    version = sys.argv[1]
    
    # Validate version format
    if not re.match(r'^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$', version):
        print(f"Error: Invalid version format: {version}")
        sys.exit(1)
    
    print(f"Updating project to version {version}")
    
    # Update all version references
    update_pyproject_toml(version)
    update_version_cli(version)
    update_dockerfile(version)
    update_docker_compose(version)
    update_kubernetes_manifests(version)
    update_documentation(version)
    
    print(f"Version update to {version} completed successfully")


if __name__ == "__main__":
    main()