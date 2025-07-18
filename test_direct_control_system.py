#!/usr/bin/env python3
"""Test examples using direct control system approach."""

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


async def test_direct_control_system():
    """Test examples using direct control system approach."""
    
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
    
    # Test examples that should work
    test_cases = [
        ("simple_test.yaml", {
            "message": "Hello world! How are you today?"
        }),
        ("simple_pipeline.yaml", {
            "user_input": "What is machine learning?"
        }),
        ("model_requirements_pipeline.yaml", {
            "task": "Translate English to French",
            "input_data": "Hello world"
        }),
        ("multi_model_pipeline.yaml", {
            "query": "Explain quantum computing",
            "data": "quantum physics concepts"
        }),
    ]
    
    results = []
    
    for yaml_file, inputs in test_cases:
        print(f"\n=== Testing {yaml_file} ===")
        
        yaml_path = Path(yaml_file)
        if not yaml_path.exists():
            yaml_path = Path("examples") / yaml_file
        
        if not yaml_path.exists():
            print(f"  ✗ File not found: {yaml_file}")
            results.append((yaml_file, False, "File not found"))
            continue
        
        try:
            # Load YAML content
            with open(yaml_path, 'r') as f:
                yaml_content = f.read()
            
            # Compile YAML to pipeline
            pipeline = await compiler.compile(yaml_content, inputs)
            
            # Execute pipeline directly using control system
            result = await asyncio.wait_for(
                control_system.execute_pipeline(pipeline),
                timeout=60.0
            )
            
            step_count = len(result) if isinstance(result, dict) else 0
            print(f"  ✓ SUCCESS - {yaml_file} completed ({step_count} steps)")
            
            # Show first result
            if isinstance(result, dict) and result:
                first_key = next(iter(result.keys()))
                first_result = str(result[first_key])[:100] + "..." if len(str(result[first_key])) > 100 else str(result[first_key])
                print(f"    First result: {first_result}")
            
            results.append((yaml_file, True, f"{step_count} steps"))
            
        except asyncio.TimeoutError:
            print(f"  ✗ TIMEOUT - {yaml_file} exceeded 60 seconds")
            results.append((yaml_file, False, "Timeout"))
            
        except Exception as e:
            print(f"  ✗ FAILED - {yaml_file}: {type(e).__name__}: {str(e)}")
            results.append((yaml_file, False, f"{type(e).__name__}: {str(e)}"))
    
    # Summary
    successful = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"\n=== SUMMARY ===")
    print(f"Successful: {successful}/{total} ({successful/total*100:.1f}%)")
    
    for name, success, details in results:
        status = "✓" if success else "✗"
        print(f"{status} {name} - {details}")
    
    return successful > 0


if __name__ == "__main__":
    success = asyncio.run(test_direct_control_system())
    if success:
        print("\n🎉 Direct control system approach works!")
    else:
        print("\n❌ Direct control system approach failed")