#!/usr/bin/env python3
"""Test HuggingFace model instantiation"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing HuggingFace model instantiation...")

# Set timeout
import signal
def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

signal.signal(signal.SIGALRM, timeout_handler)

try:
    print("1. Importing HuggingFaceModel...")
    from orchestrator.integrations.huggingface_model import HuggingFaceModel
    
    print("2. Creating TinyLlama model...")
    signal.alarm(5)  # 5 second timeout
    
    model = HuggingFaceModel(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    signal.alarm(0)  # Cancel timeout
    print("3. Model created successfully!")
    
except TimeoutError:
    print("3. ERROR: Model creation timed out - likely trying to download")
except ImportError as e:
    print(f"3. Import error: {e}")
except Exception as e:
    print(f"3. Error: {e}")
    import traceback
    traceback.print_exc()