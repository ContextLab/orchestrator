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

# Only show relevant logs
logging.getLogger("orchestrator.utils.model_config_loader").setLevel(logging.WARNING)
logging.getLogger("orchestrator.compiler.yaml_compiler").setLevel(logging.WARNING)

async def test():
    model_registry = init_models()
    orchestrator = Orchestrator(model_registry=model_registry)
    
    yaml_content = '''
name: test_while_debug
description: Debug while loop with file writes

steps:
  - id: test_loop
    while: 'true'
    max_iterations: 1  # Should only run once (iteration 0)
    steps:
      - id: write_test
        action: write
        parameters:
          path: "/tmp/test_loop_{{ test_loop.iteration }}.txt"
          content: "This is iteration {{ test_loop.iteration }}"
          
  - id: done
    action: echo All done
    dependencies: [test_loop]
'''
    
    result = await orchestrator.execute_yaml(yaml_content, {})
    
    print("\n=== RESULTS ===")
    for task_id, task_result in result.items():
        if 'write' in task_id:
            if isinstance(task_result, dict):
                print(f"{task_id}: {task_result.get('result', 'ERROR')}")
            else:
                print(f"{task_id}: {task_result}")
    
    # Check created files
    print("\n=== FILES CREATED ===")
    import glob
    files = glob.glob("/tmp/test_loop_*.txt")
    for f in sorted(files):
        print(f)

asyncio.run(test())