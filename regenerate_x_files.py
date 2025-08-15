#!/usr/bin/env python3
"""
Regenerate the X-pattern files that need fixing.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, '/Users/jmanning/orchestrator/src')

from orchestrator.orchestrator import Orchestrator
from orchestrator import init_models


async def regenerate_x_files():
    """Regenerate the X pattern files."""
    print("=" * 80)
    print("REGENERATING X-PATTERN FILES")
    print("=" * 80)
    
    output_dir = Path('examples/outputs/control_flow_conditional')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test directory
    test_dir = Path('data/test_regen')
    test_dir.mkdir(exist_ok=True, parents=True)
    
    # Define test cases for X files
    test_cases = [
        {
            'name': 'exact_threshold.txt',
            'content': 'X' * 1000,
            'size': 1000,
            'expected_type': 'Expanded'
        },
        {
            'name': 'repeated_x.txt',
            'content': 'X' * 1000,
            'size': 1000,
            'expected_type': 'Expanded'
        }
    ]
    
    # Initialize orchestrator
    print(f"\nRegenerating {len(test_cases)} files...")
    orchestrator = Orchestrator(model_registry=init_models())
    
    # Load pipeline YAML
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
                lines = content.split('\n')[:15]
                print(f"  Preview:")
                for line in lines:
                    if line.strip():
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


if __name__ == "__main__":
    asyncio.run(regenerate_x_files())