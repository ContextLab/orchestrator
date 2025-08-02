import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import Orchestrator, init_models

async def test():
    model_registry = init_models()
    orchestrator = Orchestrator(model_registry=model_registry)
    
    yaml_content = '''
name: test_goto_metadata
steps:
  - id: task1
    action: echo Test
    goto: target_task
    
  - id: task2_auto
    action: echo Test2
    goto: "<AUTO>Choose target: target_task</AUTO>"
    
  - id: target_task
    action: echo Target
'''
    
    # Compile but don't execute
    from orchestrator.compiler.control_flow_compiler import ControlFlowCompiler
    compiler = ControlFlowCompiler(model_registry=model_registry)
    
    pipeline_def = await compiler.compile_yaml(yaml_content, {})
    
    print("Compiled pipeline steps:")
    for step in pipeline_def.get("steps", []):
        print(f"\nStep {step['id']}:")
        print(f"  goto: {step.get('goto', 'None')}")
        print(f"  metadata: {step.get('metadata', {})}")

asyncio.run(test())