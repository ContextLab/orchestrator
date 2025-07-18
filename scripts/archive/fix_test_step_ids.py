#!/usr/bin/env python3
"""Fix step IDs in test files to match actual YAML files."""

import os
import re
from pathlib import Path
from orchestrator.compiler.auto_tag_yaml_parser import parse_yaml_with_auto_tags

# Map of test files to their corresponding YAML files
TEST_TO_YAML = {
    'test_research_assistant_yaml.py': 'research_assistant.yaml',
    'test_data_processing_workflow_yaml.py': 'data_processing_workflow.yaml',
    'test_multi_agent_collaboration_yaml.py': 'multi_agent_collaboration.yaml',
    'test_content_creation_pipeline_yaml.py': 'content_creation_pipeline.yaml',
    'test_code_analysis_suite_yaml.py': 'code_analysis_suite.yaml',
    'test_customer_support_automation_yaml.py': 'customer_support_automation.yaml',
    'test_automated_testing_system_yaml.py': 'automated_testing_system.yaml',
    'test_creative_writing_assistant_yaml.py': 'creative_writing_assistant.yaml',
    'test_interactive_chat_bot_yaml.py': 'interactive_chat_bot.yaml',
    'test_scalable_customer_service_agent_yaml.py': 'scalable_customer_service_agent.yaml',
    'test_document_intelligence_yaml.py': 'document_intelligence.yaml',
    'test_financial_analysis_bot_yaml.py': 'financial_analysis_bot.yaml',
}

def get_step_ids_from_yaml(yaml_file):
    """Extract step IDs from YAML file."""
    yaml_path = Path(__file__).parent / 'examples' / yaml_file
    with open(yaml_path, 'r') as f:
        content = f.read()
    
    config = parse_yaml_with_auto_tags(content)
    step_ids = [step['id'] for step in config.get('steps', [])]
    return step_ids

def fix_test_file(test_file, yaml_file):
    """Fix step IDs in test file to match YAML."""
    test_path = Path(__file__).parent / 'tests' / 'examples' / test_file
    
    if not test_path.exists():
        print(f"Test file not found: {test_path}")
        return
    
    # Get actual step IDs from YAML
    try:
        actual_ids = get_step_ids_from_yaml(yaml_file)
        print(f"\n{yaml_file} has steps: {actual_ids}")
    except Exception as e:
        print(f"Error reading {yaml_file}: {e}")
        return
    
    # Read test file
    with open(test_path, 'r') as f:
        content = f.read()
    
    # Find all assert statements checking step IDs
    # Pattern: assert 'some_id' in step_ids
    pattern = r"assert\s+'([^']+)'\s+in\s+step_ids"
    
    matches = list(re.finditer(pattern, content))
    if matches:
        print(f"Found {len(matches)} step ID assertions in {test_file}")
        
        # Check which ones need fixing
        for match in matches:
            old_id = match.group(1)
            if old_id not in actual_ids:
                print(f"  - '{old_id}' not found in YAML")
                # Try to find a similar ID
                similar = [id for id in actual_ids if old_id.replace('_', '') in id.replace('_', '') or id.replace('_', '') in old_id.replace('_', '')]
                if similar:
                    print(f"    Possible match: {similar[0]}")
    
    # Also find step IDs in mock responses and other places
    # Pattern: 'step_id' == 'some_id'
    pattern2 = r"(?:step_id|step\.get\('id'\)|step\['id'\])\s*==\s*'([^']+)'"
    matches2 = list(re.finditer(pattern2, content))
    
    # Pattern: if step_id == 'some_id':
    pattern3 = r"if\s+step_id\s*==\s*'([^']+)':"
    matches3 = list(re.finditer(pattern3, content))
    
    # Pattern: 'some_id' in auto_tags
    pattern4 = r"'([^']+)'\s+in\s+auto_tags"
    matches4 = list(re.finditer(pattern4, content))
    
    all_test_ids = set()
    for match in matches + matches2 + matches3 + matches4:
        if hasattr(match, 'group'):
            all_test_ids.add(match.group(1))
    
    if all_test_ids:
        print(f"All step IDs referenced in test: {sorted(all_test_ids)}")
        missing = [id for id in all_test_ids if id not in actual_ids]
        if missing:
            print(f"Missing from YAML: {missing}")

def main():
    """Fix all test files."""
    for test_file, yaml_file in TEST_TO_YAML.items():
        fix_test_file(test_file, yaml_file)

if __name__ == '__main__':
    main()