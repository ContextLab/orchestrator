#!/usr/bin/env python3

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add the src directory to the path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from orchestrator import Orchestrator
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.integrations.anthropic_model import AnthropicModel
from orchestrator.integrations.openai_model import OpenAIModel
from orchestrator.integrations.google_model import GoogleModel
from orchestrator.control_systems.model_based_control_system import ModelBasedControlSystem

async def test_few_examples():
    """Test just a few examples to verify fixes."""
    # Set up models
    model_registry = ModelRegistry()
    
    # Add Anthropic models
    if os.environ.get("ANTHROPIC_API_KEY"):
        claude_sonnet = AnthropicModel(model_name="claude-3-5-sonnet-20241022")
        model_registry.register_model(claude_sonnet)
    
    # Add OpenAI models
    if os.environ.get("OPENAI_API_KEY"):
        gpt4o_mini = OpenAIModel(model_name="gpt-4o-mini")
        model_registry.register_model(gpt4o_mini)
    
    # Add Google models
    if os.environ.get("GOOGLE_API_KEY"):
        gemini_flash = GoogleModel(model_name="gemini-1.5-flash")
        model_registry.register_model(gemini_flash)
    
    # Create control system
    control_system = ModelBasedControlSystem(model_registry)
    
    # Initialize orchestrator with the control system
    orchestrator = Orchestrator(
        model_registry=model_registry,
        control_system=control_system
    )
    
    # Test a few examples that were previously failing
    test_examples = [
        "creative_writing_assistant.yaml",
        "customer_support_automation.yaml", 
        "research_assistant.yaml"
    ]
    
    results = {}
    
    for example in test_examples:
        print(f"\n=== Testing {example} ===")
        
        # Load YAML file
        yaml_file = Path("examples") / example
        with open(yaml_file, 'r') as f:
            yaml_content = f.read()
        
        # Set up inputs based on example
        inputs = get_example_inputs(example)
        
        try:
            # Execute pipeline
            result = await orchestrator.execute_yaml(yaml_content, inputs)
            results[example] = {
                "status": "success",
                "result": result,
                "inputs": inputs
            }
            print(f"✓ {example} executed successfully")
            
            # Show first few steps for inspection
            if isinstance(result, dict) and "step_results" in result:
                step_results = result["step_results"]
                for i, (step_id, step_result) in enumerate(step_results.items()):
                    if i < 3:  # Show first 3 steps
                        result_str = str(step_result)[:200] + "..." if len(str(step_result)) > 200 else str(step_result)
                        print(f"  Step {step_id}: {result_str}")
            
        except Exception as e:
            results[example] = {
                "status": "error",
                "error": str(e),
                "inputs": inputs
            }
            print(f"✗ {example} failed: {str(e)}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("example_outputs") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "test_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir}")
    
    # Summary
    success_count = sum(1 for r in results.values() if r["status"] == "success")
    print(f"\nSummary: {success_count}/{len(test_examples)} examples successful")
    
    return results

def get_example_inputs(example_name: str) -> dict:
    """Get appropriate inputs for each example."""
    inputs = {
        "creative_writing_assistant.yaml": {
            "genre": "science fiction",
            "length": "short_story",
            "writing_style": "literary",
            "target_audience": "adult readers",
            "initial_premise": "A scientist discovers that memories can be transferred between parallel universes",
            "include_worldbuilding": True,
            "chapter_count": 3
        },
        "customer_support_automation.yaml": {
            "ticket_id": "TICKET-12345",
            "ticketing_system": "zendesk",
            "auto_respond": True,
            "languages": ["en", "es"],
            "escalation_threshold": -0.5,
            "kb_confidence_threshold": 0.75
        },
        "research_assistant.yaml": {
            "query": "quantum computing breakthroughs 2024",
            "context": "Focus on recent hardware advances and algorithmic improvements",
            "max_sources": 15,
            "quality_threshold": 0.7
        }
    }
    
    return inputs.get(example_name, {})

if __name__ == "__main__":
    asyncio.run(test_few_examples())