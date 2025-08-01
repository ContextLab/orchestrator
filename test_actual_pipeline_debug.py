#!/usr/bin/env python3
"""Run the actual pipeline with extra debugging."""

import asyncio
import logging
from pathlib import Path
import sys

# Enable ALL debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import Orchestrator, init_models

async def run_with_debug():
    """Run pipeline with debugging."""
    
    # Initialize models
    print("Initializing models...")
    model_registry = init_models()
    
    # Initialize orchestrator
    orchestrator = Orchestrator(model_registry=model_registry)
    
    # Read pipeline
    with open("examples/research_advanced_tools.yaml", "r") as f:
        yaml_content = f.read()
    
    # Run with minimal topic to speed up
    inputs = {
        "topic": "test templates",
        "max_results": 2  # Reduce for faster testing
    }
    
    print("Running pipeline with debug logging...")
    results = await orchestrator.execute_yaml(yaml_content, inputs)
    
    print("\n=== RESULTS ===")
    print(f"Success: {results.get('success', False)}")
    
    # Check the saved file
    output_file = Path("examples/outputs/research_advanced_tools/research_test-templates.md")
    if output_file.exists():
        content = output_file.read_text()
        print(f"\nSaved file exists: {len(content)} chars")
        print("First 200 chars:")
        print("-" * 60)
        print(content[:200])
        print("-" * 60)
        
        if "{{" in content:
            print("\n❌ Templates NOT rendered in output!")
        else:
            print("\n✅ Templates rendered in output!")
    else:
        print(f"\n❌ Output file not found: {output_file}")

if __name__ == "__main__":
    asyncio.run(run_with_debug())