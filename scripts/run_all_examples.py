#!/usr/bin/env python3
"""Run all example pipelines and verify outputs."""

import asyncio
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_pipeline import run_pipeline


async def run_all_examples():
    """Run all example pipelines."""
    examples_dir = Path("examples")
    output_dir = Path("examples/output")
    output_dir.mkdir(exist_ok=True)
    
    # Get all YAML files
    yaml_files = sorted(examples_dir.glob("*.yaml"))
    
    print(f"Found {len(yaml_files)} example pipelines")
    print("="*60)
    
    results = []
    
    # Define inputs for different pipeline types
    pipeline_inputs = {
        "research": {"topic": "Artificial Intelligence Safety", "depth": "comprehensive"},
        "content": {"topic": "Best Practices for API Design", "audience": "developers", "tone": "professional"},
        "creative": {"genre": "science fiction", "theme": "AI consciousness", "style": "thoughtful"},
        "analysis": {"code_path": "examples/test_data/sample_code.py"},
        "data": {"data_file": "examples/test_data/sample_data.csv"},
        "chat": {"conversation_topic": "Future of Technology", "num_exchanges": 3},
        "financial": {"ticker": "AAPL", "period": "1 year"},
        "customer": {"issue": "Cannot login to account", "customer_name": "John Doe"},
        "document": {"document_path": "README.md"},
        "multi_agent": {"topic": "Climate Change Solutions", "num_agents": 3},
        "test": {"test_suite": "integration", "coverage_threshold": 80},
    }
    
    for yaml_file in yaml_files:
        print(f"\n{'='*60}")
        print(f"Running: {yaml_file.name}")
        
        # Determine inputs based on pipeline type
        inputs = {}
        for key, value in pipeline_inputs.items():
            if key in yaml_file.stem.lower():
                inputs = value
                break
        
        # Add common inputs
        if not inputs:
            inputs = {"topic": "Technology Trends 2025"}
        
        try:
            start_time = datetime.now()
            exit_code = await run_pipeline(
                str(yaml_file),
                inputs,
                str(output_dir)
            )
            duration = (datetime.now() - start_time).total_seconds()
            
            status = "✅ SUCCESS" if exit_code == 0 else "❌ FAILED"
            results.append({
                "pipeline": yaml_file.name,
                "status": status,
                "duration": duration,
                "exit_code": exit_code
            })
            
        except Exception as e:
            results.append({
                "pipeline": yaml_file.name,
                "status": "❌ ERROR",
                "duration": 0,
                "error": str(e)
            })
            print(f"❌ Error: {str(e)}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    success_count = sum(1 for r in results if r["status"] == "✅ SUCCESS")
    total_count = len(results)
    
    print(f"\nTotal pipelines: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_count - success_count}")
    print(f"\nSuccess rate: {success_count/total_count*100:.1f}%")
    
    # Show failed pipelines
    failed = [r for r in results if r["status"] != "✅ SUCCESS"]
    if failed:
        print("\nFailed pipelines:")
        for r in failed:
            print(f"  - {r['pipeline']}: {r.get('error', 'Exit code ' + str(r.get('exit_code')))}")
    
    # List output files
    print(f"\n{'='*60}")
    print("OUTPUT FILES")
    print(f"{'='*60}")
    
    output_files = sorted(output_dir.glob("*.md"))
    for f in output_files[-20:]:  # Show last 20 files
        size = f.stat().st_size
        print(f"  {f.name:<50} ({size:>7,} bytes)")
    
    return 0 if success_count == total_count else 1


async def main():
    """Main entry point."""
    exit_code = await run_all_examples()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())