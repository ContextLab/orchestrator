#!/usr/bin/env python3
"""Debug context issue in pipeline execution."""

import asyncio
import json
import sys
from pathlib import Path

# Add orchestrator to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestrator import Orchestrator, init_models
from orchestrator.control_systems.hybrid_control_system import HybridControlSystem


# Monkey patch to debug context
original_build_template_context = HybridControlSystem._build_template_context

def debug_build_template_context(self, context):
    print(f"\n=== DEBUG: Building template context ===")
    print(f"Input context keys: {list(context.keys())}")
    if "previous_results" in context:
        print(f"Previous results: {list(context['previous_results'].keys())}")
    
    result = original_build_template_context(self, context)
    
    print(f"Output template context keys: {list(result.keys())}")
    if "search_web" in result:
        print(f"search_web keys: {list(result['search_web'].keys()) if isinstance(result['search_web'], dict) else 'not a dict'}")
    
    return result

HybridControlSystem._build_template_context = debug_build_template_context


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
        print("\n✅ Success!")
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_simple_example())