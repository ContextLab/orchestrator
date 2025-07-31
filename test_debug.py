import logging
import asyncio
from scripts.run_pipeline import run_pipeline

# Set logging to DEBUG
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s'
)

# Run the pipeline
asyncio.run(run_pipeline(
    'examples/research_advanced_tools.yaml', 
    {'topic': 'test debug'}, 
    'examples/outputs/test/'
))