#!/usr/bin/env python3
"""Test all research pipelines with multiple inputs."""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.run_pipeline import run_pipeline

# Test configurations
PIPELINES = [
    {
        "name": "research_minimal.yaml",
        "test_inputs": [
            {"topic": "quantum computing basics"},
            {"topic": "machine learning fundamentals"},
            {"topic": "climate change mitigation strategies"}
        ]
    },
    {
        "name": "research_basic.yaml", 
        "test_inputs": [
            {"topic": "artificial intelligence ethics"},
            {"topic": "renewable energy technologies"},
            {"topic": "space exploration advances"}
        ]
    },
    {
        "name": "research_advanced_tools.yaml",
        "test_inputs": [
            {"topic": "CRISPR gene editing"},
            {"topic": "blockchain applications"},
            {"topic": "neuroplasticity research"}
        ]
    }
]

async def test_pipeline(pipeline_name: str, inputs: dict, output_base: str):
    """Test a single pipeline with given inputs."""
    print(f"\n{'='*60}")
    print(f"Testing: {pipeline_name}")
    print(f"Inputs: {inputs}")
    print(f"{'='*60}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic_slug = inputs['topic'].replace(' ', '_').lower()[:30]
    output_dir = f"{output_base}/{pipeline_name.replace('.yaml', '')}/{topic_slug}_{timestamp}"
    
    try:
        # Run pipeline
        results = await run_pipeline(
            f'examples/{pipeline_name}',
            inputs,
            output_dir
        )
        
        # Check for output files
        output_path = Path(output_dir)
        files_created = list(output_path.rglob("*"))
        
        print(f"✅ SUCCESS: Pipeline completed")
        print(f"   Files created: {len(files_created)}")
        
        # Check for template rendering issues
        template_issues = []
        for file_path in files_created:
            if file_path.is_file() and file_path.suffix in ['.md', '.txt']:
                content = file_path.read_text()
                if '{{' in content or '{%' in content:
                    template_issues.append(str(file_path))
        
        if template_issues:
            print(f"❌ TEMPLATE ISSUES found in {len(template_issues)} files:")
            for issue in template_issues[:3]:  # Show first 3
                print(f"   - {issue}")
        else:
            print(f"✅ No template rendering issues found")
            
        # Show sample output
        md_files = [f for f in files_created if f.suffix == '.md']
        if md_files:
            sample = md_files[0]
            content = sample.read_text()
            print(f"\n📄 Sample output ({sample.name}):")
            print(content[:500] + "..." if len(content) > 500 else content)
            
        return True, None
        
    except Exception as e:
        print(f"❌ FAILURE: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, str(e)

async def main():
    """Run all pipeline tests."""
    output_base = "examples/outputs/pipeline_tests"
    Path(output_base).mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for pipeline_config in PIPELINES:
        pipeline_name = pipeline_config["name"]
        
        for inputs in pipeline_config["test_inputs"]:
            success, error = await test_pipeline(pipeline_name, inputs, output_base)
            all_results.append({
                "pipeline": pipeline_name,
                "inputs": inputs,
                "success": success,
                "error": error
            })
            
            # Small delay between tests
            await asyncio.sleep(2)
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    success_count = sum(1 for r in all_results if r["success"])
    total_count = len(all_results)
    
    print(f"Total tests: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_count - success_count}")
    print(f"Success rate: {success_count/total_count*100:.1f}%")
    
    # Show failures
    failures = [r for r in all_results if not r["success"]]
    if failures:
        print(f"\nFailed tests:")
        for failure in failures:
            print(f"- {failure['pipeline']} with {failure['inputs']}: {failure['error']}")
    
    return all_results

if __name__ == "__main__":
    results = asyncio.run(main())