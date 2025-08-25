#!/usr/bin/env python3
"""Quick runner for representative pipelines to verify they work."""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def run_pipeline(pipeline_name: str, inputs: dict):
    """Run a single pipeline using the run_pipeline.py script."""
    from scripts.run_pipeline import run_pipeline as run_pipeline_func
    
    pipeline_path = f"examples/{pipeline_name}"
    output_dir = f"examples/outputs/{pipeline_name.replace('.yaml', '')}"
    
    # Ensure output directory in inputs
    if 'output_path' not in inputs:
        inputs['output_path'] = output_dir
    
    print(f"\n{'='*60}")
    print(f"Running: {pipeline_name}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    try:
        result = await run_pipeline_func(pipeline_path, inputs, output_dir)
        if result == 0:
            print(f"✅ Success: {pipeline_name}")
            
            # Check outputs
            output_path = Path(output_dir)
            if output_path.exists():
                files = list(output_path.glob("**/*"))
                files = [f for f in files if f.is_file()]
                print(f"   Generated {len(files)} output files")
                for f in files[:3]:
                    print(f"   - {f.relative_to(output_path)}")
        else:
            print(f"❌ Failed: {pipeline_name}")
        return result
    except Exception as e:
        print(f"❌ Error running {pipeline_name}: {e}")
        return 1

async def main():
    """Run representative pipelines."""
    
    # Select key pipelines that represent different features
    test_pipelines = [
        # Basic pipelines
        ("research_minimal.yaml", {
            "topic": "renewable energy"
        }),
        
        ("simple_data_processing.yaml", {
            "output_path": "examples/outputs/simple_data_processing"
        }),
        
        # Control flow pipelines
        ("control_flow_conditional.yaml", {
            "input_file": "test.txt",
            "size_threshold": 1000,
            "output_path": "examples/outputs/control_flow_conditional"
        }),
        
        ("control_flow_for_loop.yaml", {
            "items": ["apple", "banana", "orange"],
            "processing_type": "analyze",
            "output_path": "examples/outputs/control_flow_for_loop"
        }),
        
        # Advanced pipelines
        ("web_research_pipeline.yaml", {
            "research_topic": "artificial intelligence",
            "max_sources": 3,
            "output_format": "markdown",
            "research_depth": "standard",
            "output_path": "examples/outputs/web_research_pipeline"
        }),
        
        ("data_processing.yaml", {
            "data_source": "csv",
            "output_format": "json",
            "output_path": "examples/outputs/data_processing"
        }),
        
        ("validation_pipeline.yaml", {
            "data": {"name": "Test User", "email": "test@example.com", "age": 25},
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "age": {"type": "number"}
                }
            },
            "output_path": "examples/outputs/validation_pipeline"
        })
    ]
    
    print("🚀 Running representative pipelines...")
    print(f"📋 Testing {len(test_pipelines)} pipelines")
    
    results = []
    for pipeline_name, inputs in test_pipelines:
        result = await run_pipeline(pipeline_name, inputs)
        results.append((pipeline_name, result))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    successful = sum(1 for _, r in results if r == 0)
    failed = sum(1 for _, r in results if r != 0)
    
    print(f"✅ Successful: {successful}/{len(results)}")
    print(f"❌ Failed: {failed}/{len(results)}")
    
    print("\nResults:")
    for pipeline_name, result in results:
        status = "✅" if result == 0 else "❌"
        print(f"  {status} {pipeline_name}")
    
    # Check output directories
    print("\n📁 Output Directories Created:")
    outputs_dir = Path("examples/outputs")
    for pipeline_name, _ in test_pipelines:
        output_dir = outputs_dir / pipeline_name.replace('.yaml', '')
        if output_dir.exists():
            files = list(output_dir.glob("**/*"))
            files = [f for f in files if f.is_file()]
            print(f"  {output_dir.name}: {len(files)} files")

if __name__ == "__main__":
    os.environ["LOG_LEVEL"] = "WARNING"
    asyncio.run(main())