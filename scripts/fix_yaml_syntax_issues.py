#!/usr/bin/env python3
"""Fix specific YAML syntax issues in example files."""

import re
from pathlib import Path

def fix_template_in_depends_on(content):
    """Fix template expressions in depends_on that break YAML parsing."""
    # Fix patterns like: depends_on: [""""{{ previous_steps | join('", "') }}""""]
    # Should be: depends_on: ["{{ previous_steps | join(', ') }}"]
    
    # Pattern 1: Multiple quotes around template expressions
    content = re.sub(
        r'depends_on:\s*\["""+"?\{\{([^}]+)\}\}"""+"?\]',
        r'depends_on: ["{{ \1 }}"]',
        content
    )
    
    # Pattern 2: Fix join expressions with problematic quotes
    content = re.sub(
        r"\{\{\s*previous_steps\s*\|\s*join\s*\(['\"]\"*,\s*\"*['\"]\)\s*\}\}",
        r"{{ previous_steps | join(', ') }}",
        content
    )
    
    return content

def fix_indentation_errors(content):
    """Fix common indentation errors."""
    lines = content.split('\n')
    fixed_lines = []
    in_action_block = False
    action_indent = 0
    
    for i, line in enumerate(lines):
        # Check for action: | pattern
        if re.match(r'^\s*action:\s*\|', line):
            in_action_block = True
            action_indent = len(line) - len(line.lstrip())
            fixed_lines.append(line)
            continue
        
        # Check if we've exited the action block
        if in_action_block and line.strip():
            # Look for fields that should be at the same level as 'action'
            if re.match(r'^\s*(depends_on|condition|timeout|cache_results|on_error|tags|model):', line):
                # Make sure it's at the right indentation level
                field_match = re.match(r'^(\s*)(depends_on|condition|timeout|cache_results|on_error|tags|model):(.*)', line)
                if field_match:
                    # Set to same indent as action
                    fixed_line = ' ' * action_indent + field_match.group(2) + ':' + field_match.group(3)
                    fixed_lines.append(fixed_line)
                    in_action_block = False
                    continue
            
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_malformed_model_references(content):
    """Fix malformed model references."""
    # Fix patterns like: model: "anthropic/claude-sonnet-4-20250514".2"
    content = re.sub(
        r'model:\s*"([^"]+)"\.(\d+)"',
        r'model: "\1"',
        content
    )
    
    return content

def fix_yaml_file(filepath):
    """Fix all issues in a YAML file."""
    print(f"Fixing {filepath.name}...")
    
    try:
        content = filepath.read_text()
        original_content = content
        
        # Apply fixes
        content = fix_template_in_depends_on(content)
        content = fix_indentation_errors(content)
        content = fix_malformed_model_references(content)
        
        # Write back if changed
        if content != original_content:
            filepath.write_text(content)
            print(f"  ✅ Fixed {filepath.name}")
            return True
        else:
            print(f"  ⏭️  No changes needed for {filepath.name}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error fixing {filepath.name}: {e}")
        return False

def main():
    """Fix all documented YAML files."""
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
    
    fixed_count = 0
    for example in documented_examples:
        filepath = examples_dir / example
        if filepath.exists():
            if fix_yaml_file(filepath):
                fixed_count += 1
    
    print(f"\n✨ Fixed {fixed_count} files")

if __name__ == "__main__":
    main()