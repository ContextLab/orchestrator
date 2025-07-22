#!/usr/bin/env python3
"""Test if example pipelines can be compiled and run."""

import orchestrator as orc
import os
import sys

# Test the working web search example
example_file = "examples/working_web_search.yaml"

print(f"Testing example: {example_file}")
print("=" * 50)

try:
    # Initialize models
    print("Initializing models...")
    orc.init_models()
    print("✓ Models initialized")
except Exception as e:
    print(f"✗ Failed to initialize models: {e}")
    sys.exit(1)

try:
    # Compile pipeline
    print(f"\nCompiling pipeline from {example_file}...")
    pipeline = orc.compile(example_file)
    print("✓ Pipeline compiled successfully")
    
    # Attempt to run
    print("\nAttempting to run pipeline...")
    result = pipeline.run()
    print("✓ Pipeline executed")
    
    print("\nResults:")
    print(result)
    
except FileNotFoundError as e:
    print(f"✗ File not found: {e}")
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()