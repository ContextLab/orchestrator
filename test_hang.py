#!/usr/bin/env python3
"""Find what's hanging"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("1. Testing AmbiguityResolver...")
from orchestrator.compiler.ambiguity_resolver import AmbiguityResolver

try:
    print("2. Creating resolver with fallback_to_mock=False...")
    # This should fail if no models are available
    resolver = AmbiguityResolver(model=None, fallback_to_mock=False)
    print("3. Resolver created - this shouldn't happen!")
    
except Exception as e:
    print(f"3. Expected error: {e}")
    
print("\n4. Testing with init_models...")
from orchestrator import init_models

# Temporarily disable all output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable TensorFlow logging if used

print("5. Initializing models...")
import time
start = time.time()

# Add timeout using signal
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Model initialization timed out")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(5)  # 5 second timeout

try:
    registry = init_models()
    signal.alarm(0)  # Cancel timeout
    print(f"6. Models initialized in {time.time() - start:.2f}s: {len(registry.list_models())} models")
except TimeoutError:
    print("6. ERROR: Model initialization timed out!")
except Exception as e:
    print(f"6. Error during init: {e}")
    import traceback
    traceback.print_exc()