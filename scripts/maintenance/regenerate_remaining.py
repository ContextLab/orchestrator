#!/usr/bin/env python3
"""
Regenerate remaining output files for control_flow_conditional pipeline.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, '/Users/jmanning/orchestrator/src')

from orchestrator.orchestrator import Orchestrator
from orchestrator import init_models


async def regenerate_remaining():
    """Regenerate the remaining output files."""
    print("=" * 80)
    print("REGENERATING REMAINING CONTROL_FLOW_CONDITIONAL OUTPUTS")
    print("=" * 80)
    
    output_dir = Path('examples/outputs/control_flow_conditional')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test directory
    test_dir = Path('data/test_regen')
    test_dir.mkdir(exist_ok=True, parents=True)
    
    # Check which files already exist
    existing_files = {f.name.replace('processed_', '') for f in output_dir.glob("processed_*.txt")}
    print(f"\nExisting files: {existing_files}")
    
    # Define test cases for missing files
    test_cases = []
    
    # Add missing test cases
    if 'multiline.txt' not in existing_files:
        test_cases.append({
            'name': 'multiline.txt',
            'content': '\n'.join([f'Line {i}: This is test content for line number {i}.' for i in range(1, 11)]),
            'size': 331,
            'expected_type': 'Expanded'
        })
    
    if 'long_line.txt' not in existing_files:
        test_cases.append({
            'name': 'long_line.txt',
            'content': 'A' * 2000,
            'size': 2000,
            'expected_type': 'Compressed'
        })
    
    if 'repeated_a.txt' not in existing_files:
        test_cases.append({
            'name': 'repeated_a.txt',
            'content': 'A' * 2000,
            'size': 2000,
            'expected_type': 'Compressed'
        })
    
    if 'repeated_x.txt' not in existing_files:
        test_cases.append({
            'name': 'repeated_x.txt',
            'content': 'X' * 1000,
            'size': 1000,
            'expected_type': 'Expanded'
        })
    
    if 'medium_repetitive.txt' not in existing_files:
        test_cases.append({
            'name': 'medium_repetitive.txt',
            'content': 'This is a test file. ' * 20,
            'size': 420,
            'expected_type': 'Expanded'
        })
    
    if not test_cases:
        print("\nAll files already exist!")
        return
    
    # Initialize orchestrator once
    print(f"\nRegenerating {len(test_cases)} missing files...")
    orchestrator = Orchestrator(model_registry=init_models())
    
    # Load pipeline YAML once
    with open('examples/control_flow_conditional.yaml', 'r') as f:
        yaml_content = f.read()
    
    print("-" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] {test_case['name']}")
        print(f"  Size: {test_case['size']} bytes")
        print(f"  Expected: {test_case['expected_type']}")
        
        try:
            # Prepare test file
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
                # Read and show first few lines
                content = output_file.read_text()
                lines = content.split('\n')[:10]
                print(f"  Preview:")
                for line in lines:
                    print(f"    {line[:100]}")
            else:
                print(f"  ❌ Output NOT created")
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)[:200]}")
    
    # Clean up test directory
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)
    
    print("\n" + "=" * 80)
    print("REGENERATION COMPLETE")
    print("=" * 80)
    
    # List all generated files
    generated_files = sorted(output_dir.glob("processed_*.txt"))
    print(f"\nTotal {len(generated_files)} output files now exist:")
    for f in generated_files:
        size = f.stat().st_size
        print(f"  - {f.name} ({size} bytes)")


if __name__ == "__main__":
    asyncio.run(regenerate_remaining())