import asyncio
import logging
import sys
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import Orchestrator, init_models

# Enable debug logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

async def test():
    # Initialize models first
    print("Initializing models...")
    model_registry = init_models()
    
    print("Creating orchestrator...")
    orchestrator = Orchestrator(model_registry=model_registry)
    
    yaml_content = '''
name: test_simple_while
description: Simple test

steps:
  - id: init
    action: echo Starting
    
  - id: loop
    while: "true"
    max_iterations: 1
    steps:
      - id: task1
        action: echo Hello
        
  - id: done
    action: echo Done
    dependencies: [loop]
'''
    
    print("Executing pipeline...")
    try:
        # Add timeout
        result = await asyncio.wait_for(
            orchestrator.execute_yaml(yaml_content, {}),
            timeout=10.0
        )
        print('SUCCESS - Steps completed:', list(result['steps'].keys()))
        for step_id, step_result in result['steps'].items():
            print(f"  {step_id}: {step_result}")
    except asyncio.TimeoutError:
        print("ERROR: Execution timed out after 10 seconds")
    except Exception as e:
        print('ERROR:', str(e))
        import traceback
        traceback.print_exc()

asyncio.run(test())