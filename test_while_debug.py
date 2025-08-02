import asyncio
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import Orchestrator, init_models

# Set up file logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('while_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

async def test():
    logger.info("Starting test...")
    
    # Initialize models first
    logger.info("Initializing models...")
    model_registry = init_models()
    
    logger.info("Creating orchestrator...")
    orchestrator = Orchestrator(model_registry=model_registry)
    
    yaml_content = '''
name: test_simple_while
description: Simple test

steps:
  - id: init
    action: echo Starting
    
  - id: loop
    while: "true"
    max_iterations: 3
    steps:
      - id: task1
        action: echo Hello
        
  - id: done
    action: echo Done
    dependencies: [loop]
'''
    
    logger.info("Executing pipeline...")
    try:
        # Add timeout
        result = await asyncio.wait_for(
            orchestrator.execute_yaml(yaml_content, {}),
            timeout=5.0
        )
        logger.info(f'SUCCESS - Result: {result}')
        if isinstance(result, dict):
            for key, value in result.items():
                logger.info(f"  {key}: {value}")
    except asyncio.TimeoutError:
        logger.error("ERROR: Execution timed out after 5 seconds")
    except Exception as e:
        logger.error(f'ERROR: {str(e)}')
        import traceback
        traceback.print_exc()

asyncio.run(test())