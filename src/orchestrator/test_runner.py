"""
Test runner entry point for py-orc-test command.

This module provides a convenient entry point for running the comprehensive
pipeline test suite from the command line via the py-orc-test command.
"""

import asyncio
import sys
from pathlib import Path


def main():
    """Main entry point for py-orc-test command."""
    try:
        # Import and run the test runner from tests directory
        test_runner_path = Path(__file__).parent.parent.parent.parent / "tests" / "pipeline_tests" / "run_all.py"
        
        if not test_runner_path.exists():
            print("❌ Test runner not found. Please ensure tests/pipeline_tests/run_all.py exists.")
            return 1
        
        # Import the test runner module dynamically
        import importlib.util
        import importlib
        
        spec = importlib.util.spec_from_file_location("run_all", test_runner_path)
        run_all_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_all_module)
        
        # Execute the main function
        return asyncio.run(run_all_module.main())
        
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())