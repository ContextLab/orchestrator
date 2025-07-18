#!/usr/bin/env python3
"""Test simple YAML without loops."""

import asyncio
import os
import sys
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from orchestrator import Orchestrator
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.integrations.openai_model import OpenAIModel
from orchestrator.control_systems.model_based_control_system import ModelBasedControlSystem


async def test_simple():
    """Test simple YAML without loops."""
    
    # Set up models
    model_registry = ModelRegistry()
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("No OpenAI API key found")
        return False
        
    openai_model = OpenAIModel(model_name="gpt-4o-mini")
    model_registry.register_model(openai_model)
    
    # Create control system and orchestrator
    control_system = ModelBasedControlSystem(model_registry)
    orchestrator = Orchestrator(
        model_registry=model_registry,
        control_system=control_system
    )
    
    # Test simple YAML
    with open("simple_test.yaml", 'r') as f:
        yaml_content = f.read()
    
    inputs = {
        "message": "Hello world! How are you today?"
    }
    
    try:
        print("Testing simple YAML without loops...")
        
        # Set a timeout
        result = await asyncio.wait_for(
            orchestrator.execute_yaml(yaml_content, inputs),
            timeout=30.0
        )
        
        step_count = len(result.get("step_results", {})) if isinstance(result, dict) else 0
        print(f"✓ SUCCESS - completed ({step_count} steps)")
        
        # Print full result for debugging
        print(f"Full result: {result}")
        
        # Print outputs
        if isinstance(result, dict) and "outputs" in result:
            print(f"Outputs: {result['outputs']}")
        
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
    asyncio.run(test_simple())