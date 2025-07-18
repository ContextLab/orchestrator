#!/usr/bin/env python3

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from orchestrator import Orchestrator
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.integrations.openai_model import OpenAIModel
from orchestrator.control_systems.model_based_control_system import ModelBasedControlSystem


async def test_one_example():
    """Test one example to verify fixes."""
    # Set up models
    model_registry = ModelRegistry()
    
    if os.environ.get("OPENAI_API_KEY"):
        openai_model = OpenAIModel(model_name="gpt-4o-mini")
        model_registry.register_model(openai_model)
    
    # Create control system and orchestrator
    control_system = ModelBasedControlSystem(model_registry)
    orchestrator = Orchestrator(
        model_registry=model_registry,
        control_system=control_system
    )
    
    # Test the creative writing assistant with fixed inputs
    yaml_file = Path("examples") / "creative_writing_assistant.yaml"
    with open(yaml_file, 'r') as f:
        yaml_content = f.read()
    
    inputs = {
        "genre": "science fiction",
        "length": "short_story",
        "writing_style": "literary",
        "target_audience": "adult readers",
        "initial_premise": "A scientist discovers parallel universes through quantum computing",
        "include_worldbuilding": True,
        "chapter_count": 3
    }
    
    try:
        print("Testing creative_writing_assistant.yaml...")
        result = await orchestrator.execute_yaml(yaml_content, inputs)
        print(f"✓ SUCCESS - {len(result.get('step_results', {}))} steps executed")
        
        # Show first few steps
        if isinstance(result, dict) and "step_results" in result:
            step_results = result["step_results"]
            for i, (step_id, step_result) in enumerate(step_results.items()):
                if i < 3:  # Show first 3 steps
                    result_str = str(step_result)[:100] + "..." if len(str(step_result)) > 100 else str(step_result)
                    print(f"  {step_id}: {result_str}")
        
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {str(e)}")
        return False


if __name__ == "__main__":
    asyncio.run(test_one_example())