#!/usr/bin/env python
"""Debug test for iteration variables in templates."""

import tempfile
from pathlib import Path
import subprocess
import json
import os

temp_dir = tempfile.mkdtemp(prefix="debug_iteration_")
print(f"Testing in: {temp_dir}")

# Simple pipeline that just saves iteration numbers
pipeline_yaml = f"""
name: debug_iteration
description: Debug iteration variable rendering

steps:
  - id: test_loop
    while: "{{{{ iteration < 3 }}}}"
    max_iterations: 3
    steps:
      - id: debug_vars
        action: python_executor
        parameters:
          code: |
            import json
            vars_dict = {{
                "iteration_from_context": {{{{ iteration | default(-1) }}}},
                "dollar_iteration_from_context": {{{{ $iteration | default(-1) }}}}
            }}
            print(json.dumps(vars_dict))
      
      - id: save_file
        tool: filesystem
        action: write
        parameters:
          path: "{temp_dir}/iter_{{{{ iteration }}}}.txt"
          content: "Iteration={{{{ iteration }}}}, Result={{{{ debug_vars.result | default('none') }}}}"
"""

# Save and run pipeline
pipeline_file = Path(temp_dir) / "pipeline.yaml"
pipeline_file.write_text(pipeline_yaml)

result = subprocess.run(
    ["python", "scripts/run_pipeline.py", str(pipeline_file), "-o", temp_dir],
    capture_output=True,
    text=True,
    cwd=os.getcwd()
)

print("\n=== RESULTS ===")
# Parse results from stdout
if "Results:" in result.stdout:
    results_section = result.stdout.split("Results:")[1].split("\n\n")[0]
    for line in results_section.strip().split("\n"):
        if "debug_vars:" in line or "save_file:" in line:
            print(line.strip())

print("\n=== FILES CREATED ===")
for f in sorted(Path(temp_dir).glob("iter_*.txt")):
    content = f.read_text()
    print(f"{f.name}: {content}")

print(f"\nTemp dir: {temp_dir}")