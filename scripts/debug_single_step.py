#!/usr/bin/env python3

import asyncio
import os
import sys
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from orchestrator import Orchestrator
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.integrations.openai_model import OpenAIModel
from orchestrator.control_systems.model_based_control_system import ModelBasedControlSystem

async def test_single_step():
    """Test just one step to debug context propagation."""
    
    # Set up models
    model_registry = ModelRegistry()
    
    # Try OpenAI first since Anthropic has low credits
    if os.environ.get("OPENAI_API_KEY"):
        openai_model = OpenAIModel(model_name="gpt-4o-mini")
        model_registry.register_model(openai_model)
        print(f"Registered OpenAI model: {openai_model.name}")
    else:
        print("No OpenAI API key found")
    
    # Check available models
    available_models = await model_registry.get_available_models()
    print(f"Available models: {available_models}")
    
    # Debug health check
    if available_models:
        print(f"Found {len(available_models)} healthy models")
    else:
        print("No models available - checking health of registered models directly")
        for model in model_registry.models.values():
            try:
                health = await model.health_check()
                print(f"Model {model.name} health: {health}")
            except Exception as e:
                print(f"Model {model.name} health check failed: {e}")
    
    # Create control system
    control_system = ModelBasedControlSystem(model_registry)
    
    # Initialize orchestrator
    orchestrator = Orchestrator(
        model_registry=model_registry,
        control_system=control_system
    )
    
    # Simple YAML with just two steps
    simple_yaml = """
name: "Debug Test"
description: "Test context propagation"

inputs:
  query:
    type: string
    description: "Test query"
    required: true

steps:
  - id: step1
    action: <AUTO>Analyze this query: "{{query}}" and return a 50-word summary of what this query is asking for.</AUTO>
    timeout: 10.0
    
  - id: step2
    action: <AUTO>Using the analysis from step1: {{step1.result}}, create a refined version of the original query.</AUTO>
    depends_on: [step1]
    timeout: 10.0

outputs:
  original_query: "{{query}}"
  analysis: "{{step1.result}}"
  refined_query: "{{step2.result}}"
"""
    
    inputs = {
        "query": "What are the latest quantum computing breakthroughs in 2024?"
    }
    
    try:
        # First compile the YAML to see the pipeline structure
        pipeline = await orchestrator.yaml_compiler.compile(simple_yaml, inputs)
        print("Pipeline tasks:", list(pipeline.tasks.keys()))
        
        # Check execution levels
        execution_levels = pipeline.get_execution_levels()
        print("Execution levels:", execution_levels)
        
        # Check dependencies
        for task_id, task in pipeline.tasks.items():
            print(f"Task {task_id} depends on: {task.dependencies}")
        
        result = await orchestrator.execute_yaml(simple_yaml, inputs)
        print("Result:", result)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_single_step())