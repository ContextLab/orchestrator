#!/usr/bin/env python3
"""Run example pipelines for testing."""

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Reduce logging
os.environ['ORCHESTRATOR_LOG_LEVEL'] = 'WARNING'

import orchestrator

async def test_examples():
    """Test a few key examples."""
    
    print("Initializing models...")
    orchestrator.init_models()
    
    # Output directory
    output_dir = Path("test_outputs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test cases
    test_cases = [
        ("examples/terminal_automation.yaml", {}, ["system_report.md"]),
        ("examples/simple_data_processing.yaml", {
            "data_file": "examples/test_data/test_data.csv"
        }, ["processed_data.csv", "data_analysis_report.md"]),
        ("examples/validation_pipeline.yaml", {
            "data": {"name": "Test", "email": "test@example.com"},
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string", "format": "email"}
                },
                "required": ["name", "email"]
            }
        }, ["reports/validation_report.json"])
    ]
    
    results = []
    
    for yaml_path, inputs, expected_files in test_cases:
        example_name = Path(yaml_path).stem
        print(f"\n{'='*70}")
        print(f"Testing: {example_name}")
        print(f"{'='*70}")
        
        # Clean up any existing files
        for f in expected_files:
            if Path(f).exists():
                os.remove(f)
        
        try:
            # Compile and run pipeline
            pipeline = await orchestrator.compile_async(yaml_path)
            outputs = await pipeline.run_async(**inputs)
            
            # Check for expected files
            example_dir = output_dir / example_name
            example_dir.mkdir(exist_ok=True)
            
            files_found = []
            for expected_file in expected_files:
                if Path(expected_file).exists():
                    # Move file to output directory
                    dest = example_dir / Path(expected_file).name
                    shutil.move(expected_file, dest)
                    files_found.append(dest.name)
                    print(f"✅ Found output file: {expected_file}")
                else:
                    print(f"❌ Missing expected file: {expected_file}")
            
            # Save pipeline output
            with open(example_dir / "pipeline_output.json", 'w') as f:
                json.dump(outputs if isinstance(outputs, dict) else {"result": str(outputs)}, 
                         f, indent=2, default=str)
            
            results.append({
                "example": example_name,
                "status": "success",
                "files": files_found,
                "outputs": outputs if isinstance(outputs, dict) else {"result": str(outputs)}
            })
            
        except Exception as e:
            print(f"❌ Failed: {type(e).__name__}: {str(e)}")
            results.append({
                "example": example_name,
                "status": "failed",
                "error": str(e)
            })
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    success = sum(1 for r in results if r["status"] == "success")
    total = len(results)
    print(f"Success rate: {success}/{total} ({success/total*100:.0f}%)")
    
    # Save summary
    with open(output_dir / "test_summary.json", 'w') as f:
        json.dump({
            "test_time": datetime.now().isoformat(),
            "results": results
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    
    # Inspect output quality
    for result in results:
        if result["status"] == "success" and result.get("files"):
            print(f"\n{result['example']} output files:")
            example_dir = output_dir / result['example']
            for file_name in result['files']:
                file_path = example_dir / file_name
                if file_path.exists():
                    print(f"  - {file_name} ({file_path.stat().st_size} bytes)")
                    if file_path.suffix in ['.md', '.txt', '.json', '.csv']:
                        with open(file_path, 'r') as f:
                            content = f.read()[:200]
                            print(f"    Preview: {content}...")

if __name__ == "__main__":
    asyncio.run(test_examples())