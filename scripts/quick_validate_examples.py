#!/usr/bin/env python3
"""Quick validation of key example pipelines."""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import orchestrator
from orchestrator.utils.api_keys_flexible import load_api_keys_optional


async def validate_example(yaml_path: Path, inputs: dict) -> dict:
    """Validate a single example."""
    print(f"\n{'='*70}")
    print(f"Testing: {yaml_path.name}")
    print(f"{'='*70}")
    
    result = {
        "name": yaml_path.stem,
        "status": "pending",
        "error": None,
        "outputs": []
    }
    
    try:
        # Compile pipeline
        print("Compiling pipeline...")
        pipeline = await orchestrator.compile_async(str(yaml_path))
        
        # Run pipeline
        print("Running pipeline...")
        start = datetime.now()
        outputs = await pipeline.run_async(**inputs)
        duration = (datetime.now() - start).total_seconds()
        
        result["status"] = "success"
        result["duration"] = duration
        result["outputs"] = outputs
        
        print(f"✅ Success! Duration: {duration:.2f}s")
        
    except Exception as e:
        result["status"] = "failed"
        result["error"] = f"{type(e).__name__}: {str(e)}"
        print(f"❌ Failed: {e}")
        
    return result


async def main():
    """Run quick validation on key examples."""
    
    # Initialize models
    print("Initializing models...")
    orchestrator.init_models()
    
    # Define test cases
    test_cases = [
        ("examples/control_flow_conditional.yaml", {
            "value": 85,
            "threshold": 70
        }),
        ("examples/control_flow_for_loop.yaml", {
            "items": ["apple", "banana", "cherry"]
        }),
        ("examples/simple_data_processing.yaml", {
            "data_file": "test_data.csv"
        }),
        ("examples/auto_tags_demo.yaml", {
            "data_file": "sales_data.csv",
            "analysis_goal": "identify trends",
            "output_format_preference": "summary"
        })
    ]
    
    # Create test data
    test_data_dir = Path("examples/test_data")
    test_data_dir.mkdir(exist_ok=True)
    
    # Create test CSV
    csv_file = test_data_dir / "test_data.csv"
    with open(csv_file, 'w') as f:
        f.write("id,name,value\n")
        f.write("1,Item A,100\n")
        f.write("2,Item B,200\n")
        f.write("3,Item C,150\n")
    
    # Run tests
    results = []
    for yaml_path, inputs in test_cases:
        result = await validate_example(Path(yaml_path), inputs)
        results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    
    print(f"Total: {len(results)}")
    print(f"✅ Success: {success}")
    print(f"❌ Failed: {failed}")
    
    if failed > 0:
        print("\nFailed examples:")
        for r in results:
            if r["status"] == "failed":
                print(f"  - {r['name']}: {r['error']}")
    
    # Check API keys
    keys = load_api_keys_optional()
    if keys:
        print(f"\nAPI keys available: {', '.join(keys.keys())}")
    else:
        print("\n⚠️  No API keys found - only local models available")


if __name__ == "__main__":
    asyncio.run(main())