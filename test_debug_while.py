import asyncio
import logging
from src.orchestrator import Orchestrator
from src.orchestrator.models import init_models

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

async def test():
    # Initialize models first
    await init_models()
    
    orchestrator = Orchestrator()
    yaml_content = '''
name: test_simple_while
description: Simple test

steps:
  - id: init
    action: echo Starting
    
  - id: loop
    while: "true"
    max_iterations: 2
    steps:
      - id: task1
        action: echo Hello
        
  - id: done
    action: echo Done
    dependencies: [loop]
'''
    
    try:
        result = await orchestrator.execute_yaml(yaml_content, {})
        print('SUCCESS - Steps completed:', list(result['steps'].keys()))
        for step_id, step_result in result['steps'].items():
            print(f"  {step_id}: {step_result}")
    except Exception as e:
        print('ERROR:', str(e))
        import traceback
        traceback.print_exc()

asyncio.run(test())