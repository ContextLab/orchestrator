#!/usr/bin/env python3
"""
Final test to verify Issue #160 is fully resolved.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, '/Users/jmanning/orchestrator/src')

from orchestrator.orchestrator import Orchestrator
from orchestrator import init_models


async def test_issue_160():
    """Test that control_flow_conditional pipeline works correctly."""
    print("=" * 80)
    print("TESTING ISSUE #160 - control_flow_conditional pipeline")
    print("=" * 80)
    
    orchestrator = Orchestrator(model_registry=init_models())
    
    # Load pipeline
    with open('examples/control_flow_conditional.yaml', 'r') as f:
        yaml_content = f.read()
    
    # Test cases
    test_cases = [
        {'name': 'empty', 'content': '', 'expected_type': 'Empty file'},
        {'name': 'small', 'content': 'Small test', 'expected_type': 'Expanded'},
        {'name': 'large', 'content': 'X' * 2000, 'expected_type': 'Compressed'},
    ]
    
    all_passed = True
    
    # Create test directory
    test_dir = Path('data/test_final')
    test_dir.mkdir(exist_ok=True, parents=True)
    
    for test in test_cases:
        print(f"\nTesting {test['name']} file...")
        
        # Create test file
        test_file = test_dir / f"{test['name']}.txt"
        test_file.write_text(test['content'])
        
        # Run pipeline
        context = {
            'input_file': str(test_file),
            'size_threshold': 1000
        }
        
        try:
            result = await orchestrator.execute_yaml(yaml_content, context=context)
            
            # Check output
            output_file = Path(f"examples/outputs/control_flow_conditional/processed_{test['name']}.md")
            
            if output_file.exists():
                content = output_file.read_text()
                
                # Verify it's a markdown file
                if not content.startswith("# Processed File"):
                    print(f"  ❌ Not a proper markdown file")
                    all_passed = False
                    continue
                
                # Verify processing type
                if test['expected_type'] in content:
                    print(f"  ✅ Correct processing type: {test['expected_type']}")
                else:
                    print(f"  ❌ Wrong processing type")
                    all_passed = False
                    continue
                
                # Verify no template placeholders
                if "{{" in content or "}}" in content:
                    print(f"  ❌ Template placeholders found")
                    all_passed = False
                else:
                    print(f"  ✅ No template placeholders")
                    
                # Verify no conversational output
                conversational = ["let's", "let me", "okay", "here's", "i'll"]
                if any(word in content.lower() for word in conversational):
                    print(f"  ❌ Conversational output detected")
                    all_passed = False
                else:
                    print(f"  ✅ No conversational output")
                    
            else:
                print(f"  ❌ Output file not created")
                all_passed = False
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)[:100]}")
            all_passed = False
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED - Issue #160 is RESOLVED!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_issue_160())
    sys.exit(0 if success else 1)