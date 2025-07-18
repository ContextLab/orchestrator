#!/usr/bin/env python3
"""Test model initialization"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import init_models

print("Testing model initialization...")
try:
    registry = init_models()
    print(f"Models available: {len(registry.list_models())}")
    for model in registry.list_models()[:5]:
        print(f"  - {model}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()