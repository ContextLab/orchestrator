#!/usr/bin/env python
"""Debug script to see exactly what's happening with iteration variables."""

import asyncio
import tempfile
from pathlib import Path

async def debug_iteration_variables():
    """Debug the iteration variable issue."""
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="debug_iteration_")
    print(f"Debugging in: {temp_dir}")
    
    try:
        # Create simple test pipeline with debug output
        pipeline_yaml = f"""
name: debug_iteration_variables
description: Debug iteration variables in while loop templates

steps:
  - id: test_loop
    while: "{{{{ iteration < 2 }}}}"
    max_iterations: 2
    steps:
      - id: debug_output
        tool: filesystem
        action: write
        parameters:
          path: "{temp_dir}/debug_{{{{ iteration }}}}.txt"
          content: "Iteration: {{{{ iteration }}}}, Index: {{{{ $index if $index is defined else 'UNDEFINED' }}}}"
"""
        
        # Save pipeline
        pipeline_file = Path(temp_dir) / "pipeline.yaml"
        pipeline_file.write_text(pipeline_yaml)
        
        # Run with more debug output
        import subprocess
        import os
        
        # Use the same environment but extend it
        env = os.environ.copy()
        env["PYTHONPATH"] = "/Users/jmanning/orchestrator"
        
        result = subprocess.run(
            ["python3", "scripts/run_pipeline.py", str(pipeline_file), "-o", temp_dir],
            capture_output=True,
            text=True,
            cwd="/Users/jmanning/orchestrator",
            env=env
        )
        
        print("=== STDOUT ===")
        print(result.stdout)
        print("\n=== STDERR ===") 
        print(result.stderr)
        
        # Check created files
        print("\n=== FILES CREATED ===")
        for file_path in Path(temp_dir).glob("*.txt"):
            if file_path.name != "pipeline.yaml":
                content = file_path.read_text()
                print(f"{file_path.name}: '{content}'")
        
        return result.returncode == 0
        
    finally:
        print(f"\nDebugging completed. Files remain in: {temp_dir}")

if __name__ == "__main__":
    success = asyncio.run(debug_iteration_variables())
    if success:
        print("\n✓ Debug COMPLETED")
    else:
        print("\n✗ Debug FAILED")