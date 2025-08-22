#!/usr/bin/env python
"""Simple test script for while loop iteration variables."""

import asyncio
import tempfile
import shutil
from pathlib import Path

async def test_iteration_variables():
    """Test that iteration variables work in while loops."""
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="test_iteration_")
    print(f"Testing in: {temp_dir}")
    
    try:
        # Create simple test pipeline
        pipeline_yaml = f"""
name: test_iteration_variables
description: Test iteration variables in while loop templates

steps:
  - id: test_loop
    while: "{{{{ iteration < 3 }}}}"
    max_iterations: 3
    steps:
      - id: save_file
        tool: filesystem
        action: write
        parameters:
          path: "{temp_dir}/iteration_{{{{ iteration }}}}.txt"
          content: "This is iteration {{{{ iteration }}}}"
"""
        
        # Save pipeline
        pipeline_file = Path(temp_dir) / "pipeline.yaml"
        pipeline_file.write_text(pipeline_yaml)
        
        # Run using scripts/run_pipeline.py
        import subprocess
        result = subprocess.run(
            ["python", "scripts/run_pipeline.py", str(pipeline_file), "-o", temp_dir],
            capture_output=True,
            text=True,
            cwd="/Users/jmanning/orchestrator"
        )
        
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        # Check if files were created
        print("\nChecking created files:")
        for i in range(3):
            file_path = Path(temp_dir) / f"iteration_{i}.txt"
            if file_path.exists():
                content = file_path.read_text()
                print(f"✓ iteration_{i}.txt exists with content: '{content}'")
            else:
                print(f"✗ iteration_{i}.txt NOT FOUND")
        
        # List all files in temp dir
        print(f"\nAll files in {temp_dir}:")
        for f in Path(temp_dir).iterdir():
            print(f"  - {f.name}")
        
        return result.returncode == 0
        
    finally:
        # Cleanup
        print(f"\nCleaning up {temp_dir}")
        # Don't clean up immediately for debugging
        # shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    success = asyncio.run(test_iteration_variables())
    if success:
        print("\n✓ Test PASSED")
    else:
        print("\n✗ Test FAILED")