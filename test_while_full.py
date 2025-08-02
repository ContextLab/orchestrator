import asyncio
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import Orchestrator, init_models

# Set up logging to see details
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Only show relevant logs
for logger_name in ["orchestrator.utils", "orchestrator.compiler", "orchestrator.core.control_system"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

async def test():
    model_registry = init_models()
    orchestrator = Orchestrator(model_registry=model_registry)
    
    yaml_content = '''
name: test_while_full
description: Test while loop with max_attempts

inputs:
  max_attempts:
    type: integer
    default: 1

steps:
  - id: guessing_loop
    while: 'true'
    max_iterations: "{{ max_attempts }}"
    steps:
      - id: log_attempt
        tool: filesystem
        action: write
        parameters:
          path: "/tmp/guessing_loop_{{ guessing_loop.iteration }}.txt"
          content: "Iteration {{ guessing_loop.iteration }}"
          
  - id: done
    action: echo All done
    dependencies: [guessing_loop]
'''
    
    result = await orchestrator.execute_yaml(yaml_content, {"max_attempts": 1})
    
    print("\n=== RESULTS ===")
    iteration_count = 0
    for task_id, task_result in result.items():
        if 'log_attempt' in task_id:
            print(f"{task_id}: {task_result}")
            iteration_count += 1
    
    print(f"\nTotal iterations executed: {iteration_count}")
    print(f"Expected iterations: 1")
    
    # Check created files
    print("\n=== FILES CREATED ===")
    import glob
    files = glob.glob("/tmp/guessing_loop_*.txt")
    for f in sorted(files):
        print(f)

asyncio.run(test())