#!/usr/bin/env python3
"""Debug script to understand JIT template rendering issue."""

import asyncio
import sys
sys.path.insert(0, '.')
from src.orchestrator import init_models
from src.orchestrator.orchestrator import Orchestrator

async def test_conditional_templates():
    """Test conditional task with templates."""
    # Initialize models
    model_pool = await init_models()
    
    # Create orchestrator
    orchestrator = Orchestrator(debug_templates=True)
    
    # Define a simple pipeline with conditional task
    yaml_content = """
name: test_conditional_templates
id: test_conditional_templates

inputs:
  topic: 
    type: string
    description: Test topic

steps:
  - id: step1
    action: generate_text
    parameters:
      prompt: "Generate text about {{ topic }}"
      max_tokens: 50
      
  - id: step2
    action: generate_text
    parameters:
      prompt: "Also generate about {{ topic }}"
      max_tokens: 50
      
  - id: conditional_step
    action: generate_text
    parameters:
      prompt: "Use result: {{ step1.result }}"
      max_tokens: 50
    dependencies:
      - step1
      - step2
    condition: "{{ step1.result|length > 10 }}"
"""
    
    # Run pipeline
    try:
        result = await orchestrator.execute_yaml(
            yaml_content,
            context={"topic": "JIT test"}
        )
        print("Pipeline succeeded!")
        print(f"Results: {list(result['steps'].keys())}")
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_conditional_templates())