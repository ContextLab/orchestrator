import asyncio
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import Orchestrator, init_models

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('goto_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

async def test():
    logger.info("Starting goto test...")
    
    # Initialize models first
    logger.info("Initializing models...")
    model_registry = init_models()
    
    logger.info("Creating orchestrator...")
    orchestrator = Orchestrator(model_registry=model_registry)
    
    yaml_content = '''
name: test_goto_flow
description: Test goto functionality

steps:
  - id: start
    action: echo Starting
    
  - id: check_condition
    action: echo Checking condition
    goto: skip_to_end  # Simple goto without AUTO
    dependencies: [start]
    
  - id: middle_task
    action: echo This should be skipped
    dependencies: [check_condition]
    
  - id: another_task
    action: echo This should also be skipped
    dependencies: [middle_task]
    
  - id: skip_to_end
    action: echo Jumped here via goto
    dependencies: [another_task]
    
  - id: final
    action: echo Done
    dependencies: [skip_to_end]
'''
    
    logger.info("Executing pipeline...")
    try:
        result = await asyncio.wait_for(
            orchestrator.execute_yaml(yaml_content, {}),
            timeout=10.0
        )
        logger.info(f'SUCCESS - Result: {result}')
        
        # Check which tasks were executed
        executed_tasks = list(result.keys())
        logger.info(f"Executed tasks: {executed_tasks}")
        
        # Verify middle_task and another_task were skipped
        if "middle_task" not in executed_tasks and "another_task" not in executed_tasks:
            logger.info("✓ Middle tasks were correctly skipped")
        else:
            logger.error("✗ Middle tasks were not skipped as expected")
            
        if "skip_to_end" in executed_tasks:
            logger.info("✓ Goto target was executed")
        else:
            logger.error("✗ Goto target was not executed")
            
    except asyncio.TimeoutError:
        logger.error("ERROR: Execution timed out")
    except Exception as e:
        logger.error(f'ERROR: {str(e)}')
        import traceback
        traceback.print_exc()

asyncio.run(test())