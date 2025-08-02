import asyncio
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import Orchestrator, init_models

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def test():
    logger.info("Starting AUTO goto test...")
    
    # Initialize models first
    logger.info("Initializing models...")
    model_registry = init_models()
    
    logger.info("Creating orchestrator...")
    orchestrator = Orchestrator(model_registry=model_registry)
    
    yaml_content = '''
name: test_goto_auto
description: Test goto with AUTO resolution

steps:
  - id: start
    action: echo Starting
    
  - id: calculate_value
    action: generate_text
    parameters:
      prompt: "Generate a random number between 1 and 10. Reply with just the number."
      model: gpt-4o-mini
      max_tokens: 5
    dependencies: [start]
    
  - id: check_value
    action: analyze_text
    parameters:
      text: "{{ calculate_value.result }}"
      analysis_type: "extract_number"
    goto: "<AUTO>Based on the number {{ calculate_value.result }}, select the target task ID: if the number is greater than 5, return 'high_path', otherwise return 'low_path'. Reply with just the task ID.</AUTO>"
    dependencies: [calculate_value]
    
  - id: middle_task
    action: echo This should be skipped
    dependencies: [check_value]
    
  - id: low_path
    action: echo Low number path
    dependencies: [middle_task]
    
  - id: high_path
    action: echo High number path  
    dependencies: [middle_task]
    
  - id: final
    action: echo Done
    dependencies: [low_path, high_path]
'''
    
    logger.info("Executing pipeline...")
    try:
        result = await orchestrator.execute_yaml(yaml_content, {})
        logger.info(f'SUCCESS - Result keys: {list(result.keys())}')
        
        # Check which path was taken
        if "low_path" in result:
            logger.info("✓ Low path was taken")
            number = result.get("calculate_value", {}).get("result", "unknown")
            logger.info(f"  Generated number: {number}")
        elif "high_path" in result:
            logger.info("✓ High path was taken") 
            number = result.get("calculate_value", {}).get("result", "unknown")
            logger.info(f"  Generated number: {number}")
            
        # Verify middle_task was skipped
        if "middle_task" not in result:
            logger.info("✓ Middle task was correctly skipped")
        else:
            logger.error("✗ Middle task was not skipped as expected")
            
    except Exception as e:
        logger.error(f'ERROR: {str(e)}')
        import traceback
        traceback.print_exc()

asyncio.run(test())