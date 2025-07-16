#!/usr/bin/env python3
"""Run all snippet tests with proper reporting."""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run snippet tests."""
    root = Path(__file__).parent
    test_dir = root / "tests" / "snippet_tests"
    
    # Check if we're in CI or local mode
    is_ci = os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS')
    
    if is_ci:
        print("Running in CI mode - using mock models")
    else:
        print("Running in local mode - using real models where API keys are available")
        print("Set these environment variables for full testing:")
        print("  - OPENAI_API_KEY")
        print("  - ANTHROPIC_API_KEY") 
        print("  - GOOGLE_AI_API_KEY")
        print()
    
    # Run pytest on snippet tests
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_dir),
        "-v",
        "--tb=short",
        "--no-header",
        "-x"  # Stop on first failure
    ]
    
    if not is_ci:
        # In local mode, show warnings
        cmd.append("-W default")
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
