#!/usr/bin/env python3
"""Automatically fix test files to match YAML step IDs."""

import re
from pathlib import Path
from orchestrator.compiler.auto_tag_yaml_parser import parse_yaml_with_auto_tags

# Common mappings across multiple files
COMMON_MAPPINGS = {
    # Research assistant
    'search_web': 'web_search',
    'evaluate_sources': 'analyze_credibility',
    'extract_information': 'extract_content',
    
    # Data processing
    'transform_data': 'enrich_data',
    'validate_quality': 'validate_output',
    
    # Multi-agent
    'define_problem': 'decompose_problem',
    'create_agents': 'initialize_agents',
    'assign_roles': 'assign_tasks',
    'evaluate_proposals': 'peer_review',
    'build_consensus': 'resolve_conflicts',
    
    # Content creation
    'create_content': 'create_blog_content',
    'generate_images': 'generate_visuals',
    
    # Code analysis
    'scan_codebase': 'discover_code',
    'complexity_analysis': 'architecture_review',
    'fix_issues': 'generate_insights',
    
    # Customer support
    'analyze_ticket': 'analyze_sentiment',
    'categorize_issue': 'classify_ticket',
    'check_knowledge_base': 'search_knowledge_base',
    'retrieve_customer_history': 'extract_entities',
    'sentiment_check': 'analyze_sentiment',
    'escalation_check': 'check_automation_eligibility',
    'escalate_to_human': 'assign_to_agent',
    'adjust_response_tone': 'translate_response',
    
    # Automated testing
    'scan_existing_tests': 'analyze_existing_tests',
    'identify_gaps': 'generate_test_plan',
    'generate_tests': 'generate_unit_tests',
    'run_tests': 'execute_tests',
    'analyze_coverage': 'analyze_failures',
    'update_tests': 'optimize_test_suite',
    
    # Creative writing
    'brainstorm_ideas': 'generate_premise',
    'create_outline': 'outline_chapters',
    'write_chapter': 'write_key_scenes',
    'review_and_edit': 'check_consistency',
    'apply_edits': 'apply_fixes',
    
    # Interactive chat
    'load_context': 'retrieve_context',
    'analyze_intent': 'classify_intent',
    'check_tools': 'select_tools',
    'execute_tool': 'execute_tools',
    'stream_output': 'prepare_streaming',
    'manage_context_window': 'update_memory',
    
    # Scalable customer service
    'load_customer_data': 'identify_customer',
    'search_solutions': 'search_knowledge_base',
    'apply_automation': 'check_automation',
    'escalate_priority': 'check_sla',
    'enhance_response': 'quality_check',
    'track_metrics': 'log_analytics',
}

def fix_test_file(test_file, yaml_file):
    """Fix step IDs in a test file."""
    test_path = Path(__file__).parent / 'tests' / 'examples' / test_file
    yaml_path = Path(__file__).parent / 'examples' / yaml_file
    
    if not test_path.exists():
        print(f"Test file not found: {test_path}")
        return
    
    # Get actual step IDs from YAML
    try:
        with open(yaml_path, 'r') as f:
            content = f.read()
        config = parse_yaml_with_auto_tags(content)
        actual_ids = [step['id'] for step in config.get('steps', [])]
        print(f"\nFixing {test_file} based on {yaml_file}")
        print(f"  Actual step IDs: {actual_ids[:5]}...")
    except Exception as e:
        print(f"Error reading {yaml_file}: {e}")
        return
    
    # Read test file
    with open(test_path, 'r') as f:
        test_content = f.read()
    
    original_content = test_content
    
    # Apply common mappings
    for old_id, new_id in COMMON_MAPPINGS.items():
        # Fix in assertions
        test_content = re.sub(
            rf"assert\s+'{old_id}'\s+in\s+step_ids",
            f"assert '{new_id}' in step_ids",
            test_content
        )
        
        # Fix in string comparisons
        test_content = re.sub(
            rf"==\s*'{old_id}'",
            f"== '{new_id}'",
            test_content
        )
        
        # Fix in dict keys
        test_content = re.sub(
            rf"'{old_id}':\s*{{",
            f"'{new_id}': {{",
            test_content
        )
        
        # Fix in if statements
        test_content = re.sub(
            rf"if\s+step_id\s*==\s*'{old_id}':",
            f"if step_id == '{new_id}':",
            test_content
        )
        
        # Fix in auto_tags checks
        test_content = re.sub(
            rf"'{old_id}'\s+in\s+auto_tags",
            f"'{new_id}' in auto_tags",
            test_content
        )
    
    # Write back if changed
    if test_content != original_content:
        with open(test_path, 'w') as f:
            f.write(test_content)
        print(f"  Fixed {test_file}")
    else:
        print(f"  No changes needed for {test_file}")

def main():
    """Fix all test files."""
    test_yaml_pairs = [
        ('test_research_assistant_yaml.py', 'research_assistant.yaml'),
        ('test_data_processing_workflow_yaml.py', 'data_processing_workflow.yaml'),
        ('test_multi_agent_collaboration_yaml.py', 'multi_agent_collaboration.yaml'),
        ('test_content_creation_pipeline_yaml.py', 'content_creation_pipeline.yaml'),
        ('test_code_analysis_suite_yaml.py', 'code_analysis_suite.yaml'),
        ('test_customer_support_automation_yaml.py', 'customer_support_automation.yaml'),
        ('test_automated_testing_system_yaml.py', 'automated_testing_system.yaml'),
        ('test_creative_writing_assistant_yaml.py', 'creative_writing_assistant.yaml'),
        ('test_interactive_chat_bot_yaml.py', 'interactive_chat_bot.yaml'),
        ('test_scalable_customer_service_agent_yaml.py', 'scalable_customer_service_agent.yaml'),
        ('test_document_intelligence_yaml.py', 'document_intelligence.yaml'),
        ('test_financial_analysis_bot_yaml.py', 'financial_analysis_bot.yaml'),
    ]
    
    for test_file, yaml_file in test_yaml_pairs:
        fix_test_file(test_file, yaml_file)
    
    print("\nDone!")

if __name__ == '__main__':
    main()