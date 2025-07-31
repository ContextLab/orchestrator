#!/usr/bin/env python3
"""Debug template rendering issue in research_minimal pipeline."""

import asyncio
import logging
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.orchestrator import Orchestrator, init_models

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug_template_rendering.log'),
        logging.StreamHandler()
    ]
)

# Focus on template manager logs
logging.getLogger('orchestrator.core.template_manager').setLevel(logging.DEBUG)


async def main():
    """Debug template rendering."""
    print("Initializing models...")
    model_registry = init_models()
    
    # Create orchestrator with debug templates enabled
    orchestrator = Orchestrator(model_registry=model_registry, debug_templates=True)
    
    # Load pipeline
    pipeline_path = Path("examples/research_minimal.yaml")
    yaml_content = pipeline_path.read_text()
    
    # Test inputs
    inputs = {
        "topic": "debugging template rendering"
    }
    
    print(f"\nRunning pipeline with topic: {inputs['topic']}")
    print("Check debug_template_rendering.log for detailed logs")
    
    try:
        result = await orchestrator.execute_yaml(yaml_content, inputs)
        print("\nPipeline completed!")
        
        # Check output file
        output_files = list(Path("examples/outputs/research_minimal").glob("*debugging*"))
        if output_files:
            for f in output_files:
                content = f.read_text()
                if "{{topic}}" in content:
                    print(f"\n❌ ISSUE: Unrendered {{{{topic}}}} found in {f}")
                    print(f"First line: {content.splitlines()[0]}")
                else:
                    print(f"\n✅ SUCCESS: Templates properly rendered in {f}")
                    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())