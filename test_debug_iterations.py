import asyncio
import logging
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import Orchestrator, init_models

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Focus on orchestrator and control flow logs
for logger_name in ["orchestrator.orchestrator", "orchestrator.control_flow.loops"]:
    logging.getLogger(logger_name).setLevel(logging.DEBUG)

# Suppress other logs
for logger_name in ["orchestrator.utils", "orchestrator.compiler", "orchestrator.core.control_system", "orchestrator.tools"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

async def test():
    model_registry = init_models()
    orchestrator = Orchestrator(model_registry=model_registry)
    
    # Test with the actual control_flow_while_loop.yaml
    with open('examples/control_flow_while_loop.yaml', 'r') as f:
        yaml_content = f.read()
    
    print("\n=== STARTING TEST WITH max_attempts=1 ===\n")
    
    result = await orchestrator.execute_yaml(yaml_content, {"target_number": 42, "max_attempts": 1})
    
    print("\n=== EXECUTION COMPLETE ===")
    print(f"Result type: {type(result)}")
    print(f"Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
    
    # If result contains 'steps', it might be the parsed YAML
    if isinstance(result, dict) and 'steps' in result:
        print("WARNING: Result appears to be the parsed YAML, not execution results!")
        # Try to get actual results from a different attribute
        if hasattr(orchestrator, 'last_results'):
            result = orchestrator.last_results
            print(f"Using orchestrator.last_results instead")
    
    print(f"Total tasks executed: {len(result) if isinstance(result, dict) else 0}")
    print(f"All task IDs: {sorted(result.keys()) if isinstance(result, dict) else []}")
    
    # Count iterations
    iteration_tasks = []
    for task_id in result.keys():
        if 'guessing_loop_' in task_id:
            print(f"  Loop task: {task_id}")
            if '_log_attempt' in task_id:
                iteration_tasks.append(task_id)
    
    print(f"\nIteration tasks found: {sorted(iteration_tasks)}")
    print(f"Total loop iterations: {len(iteration_tasks)}")
    
    # Check output files
    import glob
    log_files = glob.glob("examples/outputs/control_flow_while_loop/logs/attempt_*.txt")
    print(f"\nLog files created: {sorted(log_files)}")

asyncio.run(test())