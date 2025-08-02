import asyncio
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import Orchestrator, init_models

# Set up logging to see details
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test():
    model_registry = init_models()
    orchestrator = Orchestrator(model_registry=model_registry)
    
    yaml_content = '''
name: test_while_simple
description: Simple while loop test

steps:
  - id: counter_loop
    while: "true"
    max_iterations: 1  # Should only run once (iteration 0)
    steps:
      - id: print_iteration
        action: echo Hello from iteration {{ counter_loop.iteration }}
        
  - id: done
    action: echo All done
    dependencies: [counter_loop]
'''
    
    result = await orchestrator.execute_yaml(yaml_content, {})
    
    print("\n=== RESULTS ===")
    for task_id, task_result in result.items():
        print(f"{task_id}: {task_result.get('result', 'ERROR')}")
    
    # Check iteration count
    iteration_count = sum(1 for k in result.keys() if k.startswith('counter_loop_') and k.endswith('_print_iteration'))
    print(f"\nTotal iterations executed: {iteration_count}")
    print(f"Expected iterations: 1")
    
    if iteration_count != 1:
        print(f"ERROR: Wrong number of iterations!")

asyncio.run(test())