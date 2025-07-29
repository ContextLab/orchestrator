#!/usr/bin/env python3
"""Test example using the run_pipeline script."""

import subprocess
import json
from pathlib import Path

# Test simple_research.yaml
example_path = Path("examples/pipelines/simple_research.yaml")
print(f"Testing: {example_path}")

# Create inputs file
inputs = {"topic": "artificial intelligence"}
with open("test_inputs.json", "w") as f:
    json.dump(inputs, f)

# Run the pipeline
result = subprocess.run(
    ["python", "scripts/run_pipeline.py", str(example_path), "-f", "test_inputs.json", "-o", "test_output"],
    capture_output=True,
    text=True
)

print(f"Return code: {result.returncode}")
print(f"STDOUT:\n{result.stdout}")
if result.stderr:
    print(f"STDERR:\n{result.stderr}")

# Check for output files
output_files = list(Path("test_output").glob("**/*.md"))
if output_files:
    print(f"\nOutput files created: {[str(f) for f in output_files]}")
    # Read first file
    content = output_files[0].read_text()
    print(f"\nFile content preview:\n{content[:500]}...")
else:
    print("\nNo output files found in test_output/")

# Cleanup
Path("test_inputs.json").unlink(missing_ok=True)