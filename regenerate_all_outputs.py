#!/usr/bin/env python3
"""
Regenerate all output files for control_flow_conditional pipeline.
"""

import asyncio
import sys
from pathlib import Path
import shutil

sys.path.insert(0, '/Users/jmanning/orchestrator/src')

from orchestrator.orchestrator import Orchestrator
from orchestrator import init_models


async def regenerate_all_outputs():
    """Regenerate all output files with various test cases."""
    print("=" * 80)
    print("REGENERATING ALL CONTROL_FLOW_CONDITIONAL OUTPUTS")
    print("=" * 80)
    
    # Clean existing outputs
    output_dir = Path('examples/outputs/control_flow_conditional')
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test directory
    test_dir = Path('data/test_regen')
    test_dir.mkdir(exist_ok=True, parents=True)
    
    # Define all test cases
    test_cases = [
        # Original sample file
        {
            'name': 'sample.txt',
            'path': 'data/sample.txt',
            'use_existing': True,
            'size': 150,
            'expected_type': 'Expanded'
        },
        # Empty file
        {
            'name': 'empty.txt',
            'content': '',
            'size': 0,
            'expected_type': 'Empty file'
        },
        # Tiny file
        {
            'name': 'tiny.txt',
            'content': 'Small test content for expansion.',
            'size': 33,
            'expected_type': 'Expanded'
        },
        # Small file with repetition
        {
            'name': 'small.txt',
            'content': 'This is a medium-sized test file that should still be expanded. ' * 7,
            'size': 448,
            'expected_type': 'Expanded'
        },
        # Exact threshold
        {
            'name': 'exact_threshold.txt',
            'content': 'X' * 1000,
            'size': 1000,
            'expected_type': 'Expanded'
        },
        # Just over threshold
        {
            'name': 'just_over.txt',
            'content': 'Y' * 1001,
            'size': 1001,
            'expected_type': 'Compressed'
        },
        # Large file
        {
            'name': 'large.txt',
            'content': 'This is a large file for testing compression. The content needs to be substantial enough to trigger the compression logic. ' * 40,
            'size': 4920,
            'expected_type': 'Compressed'
        },
        # Special characters
        {
            'name': 'special_chars.txt',
            'content': 'Test with émojis 🎉 and spëcial çharacters: @#$%^&*()',
            'size': 52,
            'expected_type': 'Expanded'
        },
        # Multiline content
        {
            'name': 'multiline.txt',
            'content': '\n'.join([f'Line {i}: This is test content for line number {i}.' for i in range(1, 11)]),
            'size': 331,
            'expected_type': 'Expanded'
        },
        # Long single line
        {
            'name': 'long_line.txt',
            'content': 'A' * 2000,
            'size': 2000,
            'expected_type': 'Compressed'
        },
        # Repeated A pattern
        {
            'name': 'repeated_a.txt',
            'content': 'A' * 2000,
            'size': 2000,
            'expected_type': 'Compressed'
        },
        # Repeated X pattern
        {
            'name': 'repeated_x.txt',
            'content': 'X' * 1000,
            'size': 1000,
            'expected_type': 'Expanded'
        },
        # Medium repetitive content
        {
            'name': 'medium_repetitive.txt',
            'content': 'This is a test file. ' * 20,
            'size': 420,
            'expected_type': 'Expanded'
        }
    ]
    
    # Initialize orchestrator once
    print("\nInitializing Orchestrator...")
    orchestrator = Orchestrator(model_registry=init_models())
    
    # Load pipeline YAML once
    with open('examples/control_flow_conditional.yaml', 'r') as f:
        yaml_content = f.read()
    
    print(f"\nRegenerating {len(test_cases)} test cases...")
    print("-" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] {test_case['name']}")
        print(f"  Size: {test_case['size']} bytes")
        print(f"  Expected: {test_case['expected_type']}")
        
        try:
            # Prepare test file
            if test_case.get('use_existing'):
                test_file = Path(test_case['path'])
            else:
                test_file = test_dir / test_case['name']
                test_file.write_text(test_case['content'])
            
            # Run pipeline
            context = {
                'input_file': str(test_file),
                'size_threshold': 1000
            }
            
            print(f"  Running pipeline...")
            result = await orchestrator.execute_yaml(yaml_content, context=context)
            
            # Check if output was created
            output_file = output_dir / f"processed_{test_case['name']}"
            if output_file.exists():
                print(f"  ✅ Output created: processed_{test_case['name']}")
            else:
                print(f"  ❌ Output NOT created")
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)[:100]}")
    
    # Clean up test directory
    shutil.rmtree(test_dir, ignore_errors=True)
    
    print("\n" + "=" * 80)
    print("REGENERATION COMPLETE")
    print("=" * 80)
    
    # List all generated files
    generated_files = sorted(output_dir.glob("processed_*.txt"))
    print(f"\nGenerated {len(generated_files)} output files:")
    for f in generated_files:
        size = f.stat().st_size
        print(f"  - {f.name} ({size} bytes)")


if __name__ == "__main__":
    asyncio.run(regenerate_all_outputs())