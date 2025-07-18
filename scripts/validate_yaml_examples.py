#!/usr/bin/env python3
"""Validate YAML syntax for all documented examples."""

import yaml
from pathlib import Path

DOCUMENTED_EXAMPLES = [
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

def validate_yaml_file(filepath):
    """Validate a single YAML file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            yaml.safe_load(content)
        return True, None
    except yaml.YAMLError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def main():
    """Validate all documented examples."""
    print("🔍 Validating YAML syntax for documented examples...")
    
    examples_dir = Path("examples")
    errors = []
    
    for example in DOCUMENTED_EXAMPLES:
        filepath = examples_dir / example
        if filepath.exists():
            valid, error = validate_yaml_file(filepath)
            if valid:
                print(f"✅ {example}")
            else:
                print(f"❌ {example}")
                print(f"   Error: {error}")
                errors.append((example, error))
        else:
            print(f"⚠️  {example} - File not found")
            errors.append((example, "File not found"))
    
    if errors:
        print(f"\n❌ Found {len(errors)} errors")
        return 1
    else:
        print("\n✅ All YAML files are valid!")
        return 0

if __name__ == "__main__":
    exit(main())