#!/usr/bin/env python3
"""Simple test to verify CI environment."""

import os
import sys

# Just print environment info
print("=== CI Environment Test ===")
print(f"CI={os.environ.get('CI', 'not set')}")
print(f"GITHUB_ACTIONS={os.environ.get('GITHUB_ACTIONS', 'not set')}")

# Check API keys (don't print values)
for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_AI_API_KEY", "HF_TOKEN"]:
    value = os.environ.get(key)
    if value:
        print(f"{key}: SET (len={len(value)})")
    else:
        print(f"{key}: NOT SET")

# Exit with error if no keys found
keys_found = sum(1 for k in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_AI_API_KEY", "HF_TOKEN"] if os.environ.get(k))
print(f"\nTotal keys found: {keys_found}")

if keys_found == 0:
    print("ERROR: No API keys found!")
    sys.exit(1)
else:
    print("SUCCESS: API keys detected")
    sys.exit(0)