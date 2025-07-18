#!/usr/bin/env python3
"""Test examples one by one to identify issues."""

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


async def test_one_example(example_name, inputs):
    """Test a single example with timeout."""
    
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
    
    # Test the example
    yaml_file = Path("examples") / example_name
    if not yaml_file.exists():
        print(f"File not found: {yaml_file}")
        return False
        
    with open(yaml_file, 'r') as f:
        yaml_content = f.read()
    
    try:
        print(f"Testing {example_name}...")
        
        # Set a timeout using asyncio.wait_for
        result = await asyncio.wait_for(
            orchestrator.execute_yaml(yaml_content, inputs),
            timeout=30.0  # 30 second timeout
        )
        
        step_count = len(result.get("step_results", {})) if isinstance(result, dict) else 0
        print(f"✓ SUCCESS - {example_name} completed ({step_count} steps)")
        return True
        
    except asyncio.TimeoutError:
        print(f"✗ TIMEOUT - {example_name} exceeded 30 seconds")
        return False
        
    except Exception as e:
        print(f"✗ FAILED - {example_name}: {type(e).__name__}: {str(e)}")
        return False


async def test_all_examples():
    """Test all examples one by one."""
    
    examples = [
        ("research_assistant.yaml", {
            "query": "What is machine learning?",
            "context": "Simple explanation for beginners",
            "max_sources": 3,
            "quality_threshold": 0.8
        }),
        ("creative_writing_assistant.yaml", {
            "genre": "science fiction",
            "length": "short_story",
            "writing_style": "literary",
            "target_audience": "adult readers",
            "initial_premise": "A scientist discovers parallel universes",
            "include_worldbuilding": True,
            "chapter_count": 3
        }),
        ("interactive_chat_bot.yaml", {
            "message": "Hello, how are you?",
            "conversation_id": "conv_001",
            "persona": "helpful-assistant",
            "enable_streaming": False,
            "available_tools": ["web_search"]
        }),
    ]
    
    results = []
    for example_name, inputs in examples:
        success = await test_one_example(example_name, inputs)
        results.append((example_name, success))
    
    print("\n=== RESULTS ===")
    successful = sum(1 for _, success in results if success)
    print(f"Successful: {successful}/{len(results)}")
    
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {name}")


if __name__ == "__main__":
    asyncio.run(test_all_examples())