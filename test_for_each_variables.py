#!/usr/bin/env python
"""Test script for for-each loop variables to ensure they still work."""

import asyncio
import tempfile
import shutil
from pathlib import Path

async def test_for_each_variables():
    """Test that for-each variables work correctly."""
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="test_foreach_")
    print(f"Testing in: {temp_dir}")
    
    try:
        # Create simple test pipeline
        pipeline_yaml = f"""
name: test_for_each_variables
description: Test for-each variables work correctly

steps:
  - id: test_foreach
    for_each: "['apple', 'banana', 'cherry']"
    steps:
      - id: save_item
        tool: filesystem
        action: write
        parameters:
          path: "{temp_dir}/item_{{{{ $index }}}}_{{{{ $item }}}}.txt"
          content: "Index: {{{{ $index }}}}, Item: {{{{ $item }}}}, First: {{{{ $is_first }}}}, Last: {{{{ $is_last }}}}"
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
        expected_files = [
            ("item_0_apple.txt", "Index: 0, Item: apple, First: True, Last: False"),
            ("item_1_banana.txt", "Index: 1, Item: banana, First: False, Last: False"),
            ("item_2_cherry.txt", "Index: 2, Item: cherry, First: False, Last: True"),
        ]
        
        success = True
        for filename, expected_content in expected_files:
            file_path = Path(temp_dir) / filename
            if file_path.exists():
                content = file_path.read_text()
                if content == expected_content:
                    print(f"✓ {filename} exists with correct content: '{content}'")
                else:
                    print(f"✗ {filename} has wrong content: '{content}' (expected: '{expected_content}')")
                    success = False
            else:
                print(f"✗ {filename} NOT FOUND")
                success = False
        
        # List all files in temp dir
        print(f"\nAll files in {temp_dir}:")
        for f in Path(temp_dir).iterdir():
            print(f"  - {f.name}")
        
        return result.returncode == 0 and success
        
    finally:
        print(f"\nCleaning up {temp_dir}")
        # Don't clean up immediately for debugging
        # shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    success = asyncio.run(test_for_each_variables())
    if success:
        print("\n✓ For-each test PASSED")
    else:
        print("\n✗ For-each test FAILED")