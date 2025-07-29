#!/usr/bin/env python3
"""Test a batch of example pipelines."""

import asyncio
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import orchestrator

# Create output directory
output_dir = Path("example_outputs")
output_dir.mkdir(exist_ok=True)

async def test_example(pipeline_path: str, inputs: dict = None):
    """Test a single example pipeline."""
    pipeline_name = Path(pipeline_path).stem
    print(f"\n{'='*60}")
    print(f"Testing: {pipeline_name}")
    print(f"{'='*60}")
    
    try:
        # Initialize models if not already done
        if not hasattr(test_example, '_models_initialized'):
            orchestrator.init_models()
            test_example._models_initialized = True
        
        # Compile pipeline
        pipeline = await orchestrator.compile_async(pipeline_path)
        
        # Run with inputs
        inputs = inputs or {}
        print(f"Inputs: {inputs}")
        
        result = await pipeline.run_async(**inputs)
        
        # Save output
        output_file = output_dir / f"{pipeline_name}_output.json"
        with open(output_file, 'w') as f:
            if isinstance(result, dict):
                json.dump(result, f, indent=2, default=str)
            else:
                json.dump({"result": str(result)}, f, indent=2)
        
        print(f"✅ Success! Output saved to: {output_file}")
        
        # Check for generated files
        generated_files = []
        for ext in ['.pdf', '.md', '.txt', '.html', '.png', '.jpg', '.csv']:
            for file in Path('.').glob(f"*{ext}"):
                if file.stat().st_mtime > pipeline._start_time:
                    generated_files.append(str(file))
        
        if generated_files:
            print(f"Generated files: {', '.join(generated_files)}")
        
        return {"status": "success", "output": str(output_file), "generated_files": generated_files}
        
    except Exception as e:
        error_msg = f"❌ Failed: {str(e)}"
        print(error_msg)
        return {"status": "failed", "error": str(e)}

async def test_batch():
    """Test a batch of examples."""
    
    # Define test cases
    test_cases = [
        # Simple examples that should work
        {
            "path": "examples/model_routing_demo.yaml",
            "inputs": {
                "task": "Write a creative story",
                "requirements": "Must be engaging and original"
            }
        },
        {
            "path": "examples/control_flow_conditional.yaml", 
            "inputs": {
                "value": 75,
                "threshold": 50
            }
        },
        {
            "path": "examples/control_flow_for_loop.yaml",
            "inputs": {
                "items": ["apple", "banana", "orange"]
            }
        },
        {
            "path": "examples/validation_pipeline.yaml",
            "inputs": {
                "data": {"name": "test", "age": 25},
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer", "minimum": 0}
                    },
                    "required": ["name", "age"]
                }
            }
        },
        {
            "path": "examples/web_research_pipeline.yaml",
            "inputs": {
                "topic": "sustainable energy solutions",
                "max_results": 3
            }
        }
    ]
    
    # Run tests
    results = {}
    for test_case in test_cases:
        result = await test_example(test_case["path"], test_case.get("inputs", {}))
        results[test_case["path"]] = result
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    total = len(results)
    success = sum(1 for r in results.values() if r["status"] == "success")
    failed = sum(1 for r in results.values() if r["status"] == "failed")
    
    print(f"Total: {total}")
    print(f"Success: {success}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed examples:")
        for path, result in results.items():
            if result["status"] == "failed":
                print(f"  - {path}: {result['error']}")
    
    # Save full results
    with open(output_dir / "test_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFull results saved to: {output_dir / 'test_results.json'}")

if __name__ == "__main__":
    # Add start time tracking
    import time
    orchestrator.compile_async._start_time = time.time()
    
    asyncio.run(test_batch())