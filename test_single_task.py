#!/usr/bin/env python3
"""Test execution of a single task."""

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
from orchestrator.core.task import Task


async def test_single_task():
    """Test executing a single task."""
    
    # Set up models
    model_registry = ModelRegistry()
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("No OpenAI API key found")
        return False
        
    openai_model = OpenAIModel(model_name="gpt-4o-mini")
    model_registry.register_model(openai_model)
    
    # Create control system
    control_system = ModelBasedControlSystem(model_registry)
    
    # Create a simple task
    task = Task(
        id="test_task",
        name="Test Task",
        action="<AUTO>What is machine learning? Explain in simple terms.</AUTO>",
        parameters={}
    )
    
    context = {
        "pipeline_id": "test_pipeline",
        "previous_results": {},
    }
    
    try:
        print("Executing single task...")
        result = await control_system.execute_task(task, context)
        print(f"✓ SUCCESS: {result[:100]}...")
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    asyncio.run(test_single_task())