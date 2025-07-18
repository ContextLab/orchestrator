#!/usr/bin/env python3
"""Test specific fixed examples."""

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


async def test_fixed_examples():
    """Test a few of the fixed examples."""
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
    
    # Test specific examples that I just fixed
    test_cases = [
        {
            "name": "scalable_customer_service_agent.yaml",
            "inputs": {
                "interaction_id": "INT-2024-001",
                "customer_id": "CUST-12345",
                "channel": "chat",
                "content": "I'm having trouble with my recent order and need a refund",
                "metadata": {"session_id": "sess_123", "user_agent": "web_chat"},
                "languages": ["en"],
                "sla_targets": {"first_response": 60, "resolution": 3600}
            }
        },
        {
            "name": "creative_writing_assistant.yaml",
            "inputs": {
                "genre": "science fiction",
                "length": "short_story",
                "writing_style": "literary",
                "target_audience": "adult readers",
                "initial_premise": "A scientist discovers parallel universes through quantum computing",
                "include_worldbuilding": True,
                "chapter_count": 3
            }
        },
        {
            "name": "data_processing_workflow.yaml",
            "inputs": {
                "source": "sample_dataset.csv",
                "output_path": "/tmp/processed_data",
                "output_format": "json",
                "quality_threshold": 0.9
            }
        }
    ]
    
    results = {}
    
    for test_case in test_cases:
        example_name = test_case["name"]
        inputs = test_case["inputs"]
        
        print(f"\n=== Testing {example_name} ===")
        
        try:
            # Load YAML content
            yaml_file = Path("examples") / example_name
            with open(yaml_file, 'r') as f:
                yaml_content = f.read()
            
            # Execute pipeline
            result = await orchestrator.execute_yaml(yaml_content, inputs)
            
            results[example_name] = {
                "status": "success",
                "step_count": len(result.get("step_results", {})) if isinstance(result, dict) else 0
            }
            
            print(f"✓ {example_name} - SUCCESS ({results[example_name]['step_count']} steps)")
            
        except Exception as e:
            results[example_name] = {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
            
            print(f"✗ {example_name} - FAILED: {type(e).__name__}")
    
    # Summary
    successful = sum(1 for r in results.values() if r["status"] == "success")
    total = len(results)
    
    print(f"\n=== SUMMARY ===")
    print(f"Successful: {successful}/{total} ({successful/total*100:.1f}%)")
    
    for name, result in results.items():
        if result["status"] == "success":
            print(f"✓ {name}")
        else:
            print(f"✗ {name} ({result['error_type']})")
    
    return results


if __name__ == "__main__":
    asyncio.run(test_fixed_examples())