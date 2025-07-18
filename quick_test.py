#!/usr/bin/env python3
"""Quick test to verify the fixes work."""

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


async def quick_test():
    """Quick test with one simple example."""
    # Set up models
    model_registry = ModelRegistry()
    
    if os.environ.get("OPENAI_API_KEY"):
        openai_model = OpenAIModel(model_name="gpt-4o-mini")
        model_registry.register_model(openai_model)
    else:
        print("No OpenAI API key found")
        return False
    
    # Create control system and orchestrator
    control_system = ModelBasedControlSystem(model_registry)
    orchestrator = Orchestrator(
        model_registry=model_registry,
        control_system=control_system
    )
    
    # Test research assistant with simple query
    yaml_file = Path("examples") / "research_assistant.yaml"
    if not yaml_file.exists():
        print(f"File not found: {yaml_file}")
        return False
        
    with open(yaml_file, 'r') as f:
        yaml_content = f.read()
    
    inputs = {
        "query": "What is machine learning?",
        "context": "Simple explanation for beginners",
        "max_sources": 3,
        "quality_threshold": 0.8
    }
    
    try:
        print("Testing research_assistant.yaml...")
        result = await orchestrator.execute_yaml(yaml_content, inputs)
        print(f"✓ SUCCESS - executed {len(result.get('step_results', {}))} steps")
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {str(e)}")
        return False


if __name__ == "__main__":
    asyncio.run(quick_test())