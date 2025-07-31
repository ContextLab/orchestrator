#!/usr/bin/env python3
import asyncio
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.orchestrator import Orchestrator, init_models

logging.basicConfig(level=logging.WARNING, format='%(name)s - %(levelname)s - %(message)s')

async def main():
    model_registry = init_models()
    orchestrator = Orchestrator(model_registry=model_registry)
    
    yaml_content = Path('examples/research_minimal.yaml').read_text()
    inputs = {'topic': 'filesystem debug test'}
    
    result = await orchestrator.execute_yaml(yaml_content, inputs)
    print(f"\nCompleted: {result.get('outputs', {}).get('output_file', 'No output file')}")
    
    # Check the file
    files = list(Path('examples/outputs/research_minimal').glob('*filesystem*'))
    if files:
        content = files[0].read_text()
        print(f"\nFile: {files[0]}")
        print(f"First line: {content.splitlines()[0]}")
        if '{{topic}}' in content:
            print("❌ ISSUE: {{topic}} not rendered")
        else:
            print("✅ SUCCESS: Templates rendered")

if __name__ == "__main__":
    asyncio.run(main())