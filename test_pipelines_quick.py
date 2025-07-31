#!/usr/bin/env python3
"""Quick test of all pipelines."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scripts.run_pipeline import run_pipeline

async def test_pipeline(name, topic):
    """Test a single pipeline."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Topic: {topic}")
    print(f"{'='*60}")
    
    output_dir = f"examples/outputs/quick_test/{name.replace('.yaml', '')}_{topic.replace(' ', '_')}"
    
    try:
        results = await run_pipeline(
            f'examples/{name}',
            {'topic': topic},
            output_dir
        )
        
        # Check output files
        output_path = Path(output_dir)
        output_files = list(output_path.rglob("*.md")) + list(output_path.rglob("*.txt"))
        
        # Check for template issues
        template_issues = False
        for file in output_files:
            content = file.read_text()
            if '{{' in content or '{%' in content:
                print(f"❌ TEMPLATE ISSUE in {file.name}")
                # Show problematic lines
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if '{{' in line or '{%' in line:
                        print(f"   Line {i+1}: {line.strip()}")
                        template_issues = True
                        break
                break
        
        if not template_issues:
            print(f"✅ SUCCESS - No template issues found")
            # Show first file content preview
            if output_files:
                content = output_files[0].read_text()
                print(f"\n📄 Output preview ({output_files[0].name}):")
                print(content[:300] + "..." if len(content) > 300 else content)
        
        return not template_issues
    
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

async def main():
    """Run all tests."""
    tests = [
        ("research_minimal.yaml", "quantum computing"),
        ("research_basic.yaml", "machine learning"),
        ("research_advanced_tools.yaml", "CRISPR gene editing"),
        ("control_flow_advanced.yaml", "test template rendering")
    ]
    
    results = []
    for pipeline, topic in tests:
        success = await test_pipeline(pipeline, topic)
        results.append((pipeline, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for pipeline, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{pipeline:30} {status}")

if __name__ == "__main__":
    asyncio.run(main())