#!/usr/bin/env python3
"""Simple test to check basic functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    print("Testing imports...")
    print("✓ Orchestrator imported")
    
    from src.orchestrator import init_models
    print("✓ init_models imported")
    
    print("\nChecking for API keys...")
    import os
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY")) 
    has_google = bool(os.environ.get("GOOGLE_API_KEY"))
    
    print(f"OPENAI_API_KEY: {'✓ Set' if has_openai else '✗ Not set'}")
    print(f"ANTHROPIC_API_KEY: {'✓ Set' if has_anthropic else '✗ Not set'}")
    print(f"GOOGLE_API_KEY: {'✓ Set' if has_google else '✗ Not set'}")
    
    print("\nTrying to initialize models...")
    registry = init_models()
    print(f"✓ Model registry created with {len(registry.models)} models")
    
    print("\nRegistered models:")
    for model_key in registry.list_models():
        print(f"  - {model_key}")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()