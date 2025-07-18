#!/usr/bin/env python3
"""Test using ModelBasedControlSystem directly."""

import asyncio
import os
import sys
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from orchestrator.models.model_registry import ModelRegistry
from orchestrator.integrations.openai_model import OpenAIModel
from orchestrator.control_systems.model_based_control_system import ModelBasedControlSystem
from orchestrator.compiler.yaml_compiler import YAMLCompiler


async def test_control_system_direct():
    """Test using ModelBasedControlSystem directly."""
    
    # Set up models
    model_registry = ModelRegistry()
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("No OpenAI API key found")
        return False
        
    openai_model = OpenAIModel(model_name="gpt-4o-mini")
    model_registry.register_model(openai_model)
    
    # Create control system and compiler
    control_system = ModelBasedControlSystem(model_registry)
    compiler = YAMLCompiler()
    
    # Test simple YAML
    with open("simple_test.yaml", 'r') as f:
        yaml_content = f.read()
    
    inputs = {
        "message": "Hello world! How are you today?"
    }
    
    try:
        print("Testing ModelBasedControlSystem directly...")
        
        # Compile YAML to pipeline
        pipeline = await compiler.compile(yaml_content, inputs)
        print(f"Compiled pipeline with {len(pipeline.tasks)} tasks")
        
        # Execute pipeline directly using control system
        result = await asyncio.wait_for(
            control_system.execute_pipeline(pipeline),
            timeout=30.0
        )
        
        step_count = len(result) if isinstance(result, dict) else 0
        print(f"✓ SUCCESS - completed ({step_count} steps)")
        print(f"Result: {result}")
        
        return True
        
    except asyncio.TimeoutError:
        print("✗ TIMEOUT - exceeded 30 seconds")
        return False
        
    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    asyncio.run(test_control_system_direct())