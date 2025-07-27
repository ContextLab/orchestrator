#!/usr/bin/env python3
"""Debug script to check environment in CI."""

import os
import sys

print("=== CI Environment Debug ===")
print(f"Python version: {sys.version}")
print(f"CI: {os.environ.get('CI', 'not set')}")
print(f"GITHUB_ACTIONS: {os.environ.get('GITHUB_ACTIONS', 'not set')}")
print(f"GITHUB_WORKFLOW: {os.environ.get('GITHUB_WORKFLOW', 'not set')}")

# Check for API keys (don't print values)
api_keys = {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
    "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
    "GOOGLE_AI_API_KEY": os.environ.get("GOOGLE_AI_API_KEY"),
    "HF_TOKEN": os.environ.get("HF_TOKEN"),
}

print("\n=== API Keys Status ===")
for key, value in api_keys.items():
    if value:
        print(f"{key}: SET (length: {len(value)})")
    else:
        print(f"{key}: NOT SET")

# Try loading API keys through our function
print("\n=== Testing load_api_keys_optional ===")
try:
    from orchestrator.utils.api_keys_flexible import load_api_keys_optional
    loaded = load_api_keys_optional()
    print(f"Loaded {len(loaded)} API keys: {list(loaded.keys())}")
except Exception as e:
    print(f"Error loading API keys: {e}")

# Try initializing models
print("\n=== Testing init_models ===")
try:
    from orchestrator import init_models
    registry = init_models()
    models = registry.list_models()
    print(f"Initialized {len(models)} models")
    if models:
        print(f"First 5 models: {models[:5]}")
except Exception as e:
    print(f"Error initializing models: {e}")
    import traceback
    traceback.print_exc()