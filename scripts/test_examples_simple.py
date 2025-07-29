#!/usr/bin/env python3
"""Simple test for example pipelines - one at a time."""

import asyncio
import sys
from pathlib import Path

# Add orchestrator to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestrator import Orchestrator, init_models


async def test_simple_example():
    """Test a simple example pipeline."""
    # Initialize models
    print("Initializing models...")
    init_models()
    
    # Create orchestrator
    orchestrator = Orchestrator()
    
    # Test simple_research.yaml
    example_path = Path(__file__).parent.parent / "examples/pipelines/simple_research.yaml"
    print(f"\nTesting: {example_path.name}")
    
    # Read pipeline YAML
    yaml_content = example_path.read_text()
    
    # Prepare inputs
    inputs = {"topic": "artificial intelligence"}
    
    try:
        # Run pipeline
        results = await orchestrator.execute_yaml(yaml_content, inputs)
        print("✅ Success!")
        print(f"Results: {results}")
        
        # Check for output files
        output_files = list(Path(".").glob("research/*.md"))
        if output_files:
            print(f"Output files created: {[str(f) for f in output_files]}")
            # Read first file
            content = output_files[0].read_text()
            print(f"\nFile content preview:\n{content[:500]}...")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_simple_example())