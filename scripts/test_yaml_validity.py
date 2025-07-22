#!/usr/bin/env python3
"""Test that all YAML examples can be loaded and have valid structure."""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser

def test_yaml_file(filepath):
    """Test a single YAML file."""
    print(f"\n{'='*60}")
    print(f"Testing: {filepath.name}")
    print('='*60)
    
    try:
        # Test 1: Can it be parsed as YAML?
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Test with AUTO tag parser
        parser = AutoTagYAMLParser()
        parsed = parser.parse(content)
        
        print("✅ YAML parsing: Success")
        
        # Test 2: Has required fields?
        required_fields = ['name', 'description', 'inputs', 'steps']
        for field in required_fields:
            if field in parsed:
                print(f"✅ Has '{field}' field")
            else:
                print(f"❌ Missing '{field}' field")
        
        # Test 3: Check AUTO tags
        auto_tags = parser.find_auto_tags(parsed)
        if auto_tags:
            print(f"✅ Found {len(auto_tags)} AUTO tags")
            for path, content in auto_tags[:3]:  # Show first 3
                print(f"   - {path}: {content[:50]}...")
        
        # Test 4: Check inputs
        if 'inputs' in parsed:
            print(f"✅ Inputs defined: {list(parsed['inputs'].keys())}")
        
        # Test 5: Check steps
        if 'steps' in parsed:
            step_ids = [step.get('id', 'unnamed') for step in parsed['steps']]
            print(f"✅ Steps defined ({len(step_ids)}): {', '.join(step_ids[:5])}...")
        
        # Test 6: Check outputs
        if 'outputs' in parsed:
            print(f"✅ Outputs defined: {list(parsed['outputs'].keys())[:5]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        return False

def main():
    """Test all documented YAML examples."""
    examples_dir = Path("examples")
    
    documented_examples = [
        "research_assistant.yaml",
        "code_analysis_suite.yaml", 
        "content_creation_pipeline.yaml",
        "data_processing_workflow.yaml",
        "multi_agent_collaboration.yaml",
        "automated_testing_system.yaml",
        "document_intelligence.yaml",
        "creative_writing_assistant.yaml",
        "financial_analysis_bot.yaml",
        "interactive_chat_bot.yaml",
        "scalable_customer_service_agent.yaml",
        "customer_support_automation.yaml",
    ]
    
    success_count = 0
    
    for example in documented_examples:
        filepath = examples_dir / example
        if filepath.exists():
            if test_yaml_file(filepath):
                success_count += 1
        else:
            print(f"\n❌ File not found: {example}")
    
    print(f"\n{'='*60}")
    print(f"Summary: {success_count}/{len(documented_examples)} examples passed all tests")
    print('='*60)

if __name__ == "__main__":
    main()