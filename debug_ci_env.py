#!/usr/bin/env python3
"""Debug CI Environment
Quick debugging script to check CI environment configuration.
This script is referenced by .github/workflows/tests.yml
"""
import os
import sys
import platform
from pathlib import Path

def main():
    """Print CI environment debug information."""
    print("=== CI Environment Debug Information ===")
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    
    # Check for required environment variables
    required_vars = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_AI_API_KEY', 'HF_TOKEN']
    print("\n=== API Keys Status ===")
    for var in required_vars:
        value = os.environ.get(var)
        if value:
            print(f"[OK] {var}: {'*' * 20} (set)")
        else:
            print(f"[MISSING] {var}: Not set")
    
    # Check for key files
    print("\n=== Key Files Check ===")
    key_files = [
        'requirements.txt',
        'pyproject.toml', 
        'src/orchestrator/__init__.py',
        'tests/',
        'config/',
        'scripts/',
    ]
    
    for file_path in key_files:
        path = Path(file_path)
        if path.exists():
            print(f"[OK] {file_path}: exists")
        else:
            print(f"[MISSING] {file_path}: missing")
    
    # Check Python package installation
    print("\n=== Package Installation Check ===")
    try:
        import orchestrator
        print("[OK] orchestrator package: importable")
        print(f"   Version: {getattr(orchestrator, '__version__', 'unknown')}")
        print(f"   Location: {orchestrator.__file__}")
    except ImportError as e:
        print(f"[ERROR] orchestrator package: {e}")
    
    # Test basic functionality
    print("\n=== Basic Functionality Test ===")
    try:
        # Try importing core components
        from orchestrator.api import compile
        from orchestrator.core.pipeline import Pipeline
        print("[OK] Core components: importable")
    except ImportError as e:
        print(f"[ERROR] Core components: {e}")
    
    print("\n=== CI Environment Debug Complete ===")
    return 0

if __name__ == '__main__':
    sys.exit(main())