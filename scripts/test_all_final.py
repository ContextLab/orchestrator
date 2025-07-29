#!/usr/bin/env python3
"""Final test of all example pipelines."""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add orchestrator to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestrator import Orchestrator, init_models


async def test_example(example_path: Path, output_dir: Path) -> Dict:
    """Test a single example pipeline."""
    result = {
        "example": example_path.name,
        "path": str(example_path),
        "status": "pending",
        "error": None,
        "outputs": {},
        "files_created": [],
        "duration": 0
    }
    
    try:
        # Create example-specific output directory
        example_output_dir = output_dir / example_path.stem
        example_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Change to output directory to contain file outputs
        original_cwd = os.getcwd()
        os.chdir(example_output_dir)
        
        # Create data directory for examples that need it
        Path("data").mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Testing: {example_path.name}")
        print(f"Output directory: {example_output_dir}")
        print(f"{'='*60}")
        
        # Read pipeline YAML
        yaml_content = example_path.read_text()
        
        # Prepare inputs based on example
        inputs = {}
        if "research" in example_path.name:
            inputs["topic"] = "artificial intelligence"
        elif "code_review" in example_path.name or "code_optimization" in example_path.name:
            # Create sample code file
            Path("sample_code.py").write_text("""def hello(name):
    print(f'Hello {name}')
    
def calculate(x, y):
    return x + y
""")
            inputs["code"] = "def hello(name):\\n    print(f'Hello {name}')"
            inputs["code_file"] = str(Path("sample_code.py").absolute())
        elif "data_processing" in example_path.name:
            # Create sample data file
            Path("data/input.csv").write_text("name,value\\ntest1,10\\ntest2,20")
            inputs["data_source"] = str(Path("data/input.csv").absolute())
        elif "statistical" in example_path.name:
            inputs["data"] = {"values": [1, 2, 3, 4, 5]}
        elif "text_processing" in example_path.name:
            inputs["text"] = "This is a sample text for processing."
        elif "multimodal" in example_path.name:
            inputs["media_url"] = "https://example.com/image.jpg"
        elif "control_flow" in example_path.name:
            if "conditional" in example_path.name:
                # Create a small test file
                Path("data/sample.txt").write_text("This is a test file.")
                inputs["input_file"] = "data/sample.txt"
            elif "for_loop" in example_path.name:
                # Create test files for batch processing
                for i in range(1, 4):
                    Path(f"data/file{i}.txt").write_text(f"This is test file {i}.")
            inputs["input_text"] = "Process this text"
        
        # Create orchestrator for this test
        orchestrator = Orchestrator()
        
        # Run pipeline
        start_time = datetime.now()
        try:
            results = await orchestrator.execute_yaml(yaml_content, inputs)
            duration = (datetime.now() - start_time).total_seconds()
            
            result["status"] = "success"
            result["duration"] = duration
            result["outputs"] = results.get("outputs", {})
            
            # Find created files
            for root, dirs, files in os.walk("."):
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for file in files:
                    if not file.startswith("."):
                        result["files_created"].append(os.path.join(root, file))
            
            print(f"✅ Success! Duration: {duration:.2f}s")
            print(f"Files created: {len(result['files_created'])}")
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            print(f"❌ Pipeline execution failed: {e}")
            
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Setup failed: {str(e)}"
        print(f"❌ Setup failed: {e}")
    
    finally:
        # Restore original directory
        os.chdir(original_cwd)
    
    return result


async def test_all_examples():
    """Test all example pipelines."""
    # Initialize models once
    print("Initializing models...")
    init_models()
    
    # Find all example YAML files
    examples_dir = Path(__file__).parent.parent / "examples"
    example_files = []
    
    # Collect all YAML files (avoid duplicates)
    seen_names = set()
    for pattern in ["*.yaml", "pipelines/*.yaml", "sub_pipelines/*.yaml"]:
        for file in examples_dir.glob(pattern):
            if file.name not in seen_names:
                example_files.append(file)
                seen_names.add(file.name)
    
    # Sort for consistent ordering
    example_files.sort()
    
    # Filter to test specific examples if needed
    test_subset = [
        "simple_research.yaml",
        "control_flow_for_loop.yaml",
        "statistical_analysis.yaml",
        "research-report-template.yaml",
        "test_validation_pipeline.yaml"
    ]
    
    example_files = [f for f in example_files if f.name in test_subset]
    
    print(f"\nFound {len(example_files)} example pipelines to test")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"test_outputs/final_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test each example
    results = []
    for example_file in example_files:
        result = await test_example(example_file, output_dir)
        results.append(result)
    
    # Generate summary
    summary = {
        "test_time": datetime.now().isoformat(),
        "total_examples": len(results),
        "successful": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] == "error"]),
        "results": results
    }
    
    # Save summary
    summary_path = output_dir / "test_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total examples tested: {summary['total_examples']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"\nResults saved to: {output_dir}")
    
    return summary


if __name__ == "__main__":
    asyncio.run(test_all_examples())