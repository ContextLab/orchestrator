#!/usr/bin/env python3
"""Quick validation of pipelines to check if they're working after our fixes."""

import os
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import Orchestrator
from orchestrator.compiler.yaml_compiler import YAMLCompiler

async def quick_check():
    """Quick check of a few pipelines to see if our fixes worked."""
    
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
            
            orchestrator = Orchestrator()
            results = await orchestrator.run(
                pipeline,
                inputs=inputs,
                output_dir=f"examples/outputs/{Path(pipeline_path).stem}_test"
            )
            
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