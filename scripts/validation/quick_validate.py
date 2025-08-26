#!/usr/bin/env python3
"""Quick validation of pipelines to check if they're working after our fixes."""

import os
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import Orchestrator, init_models
from orchestrator.compiler.yaml_compiler import YAMLCompiler
from orchestrator.control_systems.hybrid_control_system import HybridControlSystem

async def quick_check():
    """Quick check of a few pipelines to see if our fixes worked."""
    
    # Initialize models first
    print("Initializing models...")
    model_registry = init_models()
    
    if not model_registry or not model_registry.models:
        print("❌ No models available. Please check your API keys and models.yaml")
        return
    
    # Create control system with models
    control_system = HybridControlSystem(model_registry)
    
    test_pipelines = [
        ("examples/research_minimal.yaml", {"topic": "AI safety"}),
        ("examples/simple_data_processing.yaml", {"data": [1, 2, 3, 4, 5]}),
        ("examples/control_flow_advanced.yaml", {"input_text": "Test text"}),
    ]
    
    for pipeline_path, inputs in test_pipelines:
        print(f"\nTesting: {pipeline_path}")
        try:
            with open(pipeline_path) as f:
                yaml_content = f.read()
            
            compiler = YAMLCompiler(development_mode=True)
            pipeline = await compiler.compile(yaml_content)
            
            orchestrator = Orchestrator(
                model_registry=model_registry,
                control_system=control_system
            )
            
            # Add output_path to inputs if not present
            if 'output_path' not in inputs:
                inputs['output_path'] = f"examples/outputs/{Path(pipeline_path).stem}_test"
            
            results = await orchestrator.execute_yaml(yaml_content, inputs)
            
            # Quick quality check
            result_str = str(results)
            if "{{" in result_str or "}}" in result_str:
                print("  ⚠️ Unrendered templates found")
            elif "error" in result_str.lower():
                print("  ⚠️ Error in output")
            else:
                print("  ✅ Pipeline executed successfully")
                
        except Exception as e:
            print(f"  ❌ Failed: {e}")

if __name__ == "__main__":
    asyncio.run(quick_check())