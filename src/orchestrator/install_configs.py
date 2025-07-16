"""Install default configuration files to user's home directory."""

import os
import shutil
from pathlib import Path


def install_default_configs():
    """Copy default configuration files to ~/.orchestrator/ if they don't exist."""
    # Create ~/.orchestrator directory
    config_dir = Path.home() / ".orchestrator"
    config_dir.mkdir(exist_ok=True)
    
    # Get config source directory (package root/config)
    package_root = Path(__file__).parent.parent.parent
    src_dir = package_root / "config"
    
    # Configuration files to install
    config_files = {
        "orchestrator.yaml": "Default orchestrator configuration",
        "models.yaml": "Model configuration and registry"
    }
    
    for filename, description in config_files.items():
        src_file = src_dir / filename
        dst_file = config_dir / filename
        
        # Only copy if destination doesn't exist (don't overwrite user configs)
        if not dst_file.exists() and src_file.exists():
            shutil.copy2(src_file, dst_file)
            print(f"Installed {filename} to {dst_file}")
        elif dst_file.exists():
            print(f"Keeping existing {filename} at {dst_file}")
        
    # Create a README in the config directory
    readme_path = config_dir / "README.md"
    if not readme_path.exists():
        readme_content = """# Orchestrator Configuration Directory

This directory contains configuration files for the Orchestrator framework.

## Configuration Files

### models.yaml
Defines available AI models and their properties. You can:
- Add new models from Ollama, HuggingFace, or cloud providers
- Set model expertise areas and size information
- Configure default model preferences
- Define fallback chains for model selection

### orchestrator.yaml
Main configuration for the Orchestrator framework. You can:
- Set default execution parameters
- Configure resource limits
- Customize error handling behavior
- Set up monitoring and logging preferences

## Customization

Feel free to edit these files to customize Orchestrator's behavior. The framework
will automatically pick up changes on the next run.

For more information, see: https://orc.readthedocs.io/en/latest/user_guide/configuration.html
"""
        readme_path.write_text(readme_content)
        print(f"Created README at {readme_path}")
    
    print(f"\nConfiguration files installed to: {config_dir}")
    print("You can customize these files to change Orchestrator's behavior.")


if __name__ == "__main__":
    install_default_configs()