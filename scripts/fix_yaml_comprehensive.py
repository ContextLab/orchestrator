#!/usr/bin/env python3
"""Comprehensive YAML fix for all syntax issues in example files."""

import re
from pathlib import Path
import yaml

def fix_multiline_action_blocks(content):
    """Fix action blocks that have improper structure."""
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check for action: | pattern
        if re.match(r'^(\s*)action:\s*\|', line):
            fixed_lines.append(line)
            base_indent = len(line) - len(line.lstrip())
            i += 1
            
            # Collect all action content
            action_content = []
            while i < len(lines):
                next_line = lines[i]
                
                # Check if we've hit a field that should be outside the action
                if next_line.strip() and re.match(r'^\s*(depends_on|condition|timeout|cache_results|on_error|tags|model|requires_model):', next_line):
                    # Found a field - stop collecting action content
                    break
                
                # If it's part of the action content
                if next_line.strip() or not next_line:  # Include content and blank lines
                    # Ensure proper indentation for action content
                    if next_line.strip():
                        # Add 2 spaces to base indent for action content
                        action_content.append(' ' * (base_indent + 2) + next_line.strip())
                    else:
                        action_content.append('')  # Preserve blank lines
                i += 1
            
            # Add the action content
            fixed_lines.extend(action_content)
            
            # Now handle the field that broke us out of the loop
            if i < len(lines):
                field_line = lines[i]
                # Ensure the field is at the same level as 'action'
                field_match = re.match(r'^\s*(depends_on|condition|timeout|cache_results|on_error|tags|model|requires_model):(.*)', field_line)
                if field_match:
                    fixed_lines.append(' ' * base_indent + field_match.group(1) + ':' + field_match.group(2))
                else:
                    fixed_lines.append(field_line)
        else:
            fixed_lines.append(line)
        
        i += 1
    
    return '\n'.join(fixed_lines)

def fix_broken_depends_on_with_content(content):
    """Fix cases where depends_on has action content after it."""
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Look for depends_on followed by numbered content
        if 'depends_on:' in line and i + 1 < len(lines):
            next_line = lines[i + 1]
            # Check if next line is numbered content that should be in action
            if re.match(r'^\s+\d+\.', next_line):
                # We need to move this content before depends_on
                # First, find all the numbered content
                numbered_content = []
                j = i + 1
                while j < len(lines) and (re.match(r'^\s+\d+\.', lines[j]) or 
                                         (lines[j].strip() and not re.match(r'^\s*-', lines[j]) and ':' not in lines[j])):
                    numbered_content.append(lines[j])
                    j += 1
                
                # Find the action block this belongs to (should be above)
                # Look backwards for action: |
                action_idx = i - 1
                while action_idx >= 0:
                    if re.match(r'^\s*action:\s*\|', lines[action_idx]):
                        break
                    action_idx -= 1
                
                if action_idx >= 0:
                    # We found the action block
                    # Insert the numbered content after the action line
                    # Skip this depends_on for now, we'll add it after the content
                    i = j
                    continue
                
        fixed_lines.append(line)
        i += 1
    
    return '\n'.join(fixed_lines)

def fix_nested_on_error_blocks(content):
    """Fix improperly nested on_error blocks."""
    # Fix patterns where on_error action content isn't properly indented
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check for on_error:
        if re.match(r'^(\s*)on_error:', line):
            fixed_lines.append(line)
            base_indent = len(line) - len(line.lstrip())
            i += 1
            
            # Check if next line is action: |
            if i < len(lines) and re.match(r'^\s*action:\s*\|', lines[i]):
                # Ensure proper indentation
                fixed_lines.append(' ' * (base_indent + 2) + 'action: |')
                i += 1
                
                # Process the action content
                while i < len(lines):
                    next_line = lines[i]
                    if next_line.strip() and not next_line.startswith(' ' * (base_indent + 4)):
                        # Check if it's a field like continue_on_error
                        if re.match(r'^\s*(continue_on_error|retry_count|fallback_value):', next_line):
                            # Indent it properly under on_error
                            field_match = re.match(r'^\s*(\w+):(.*)', next_line)
                            if field_match:
                                fixed_lines.append(' ' * (base_indent + 2) + field_match.group(1) + ':' + field_match.group(2))
                            i += 1
                        else:
                            break
                    else:
                        if next_line.strip():
                            # Add proper indentation for action content
                            fixed_lines.append(' ' * (base_indent + 4) + next_line.strip())
                        else:
                            fixed_lines.append('')
                        i += 1
                i -= 1  # Back up one
        else:
            fixed_lines.append(line)
        
        i += 1
    
    return '\n'.join(fixed_lines)

def fix_template_expressions(content):
    """Fix template expressions in depends_on and other fields."""
    # Fix multiple quotes around template expressions
    content = re.sub(
        r'depends_on:\s*\[""""?\{\{([^}]+)\}\}""""?\]',
        r'depends_on: ["{{ \1 }}"]',
        content
    )
    
    # Fix join expressions
    content = re.sub(
        r'\{\{\s*previous_steps\s*\|\s*join\s*\([\'"][\s,\'\"]*[\'\"]\)\s*\}\}',
        r"{{ previous_steps | join(', ') }}",
        content
    )
    
    return content

def fix_yaml_file(filepath):
    """Apply all fixes to a YAML file."""
    print(f"\nProcessing {filepath.name}...")
    
    try:
        content = filepath.read_text()
        original_content = content
        
        # Apply fixes in order
        content = fix_template_expressions(content)
        content = fix_broken_depends_on_with_content(content)
        content = fix_multiline_action_blocks(content)
        content = fix_nested_on_error_blocks(content)
        
        # Try to parse to validate
        try:
            yaml.safe_load(content)
            print(f"  ✅ Valid YAML structure")
        except yaml.YAMLError as e:
            print(f"  ⚠️  YAML validation warning: {str(e)[:100]}...")
        
        # Write back if changed
        if content != original_content:
            filepath.write_text(content)
            print(f"  ✅ Fixed and saved {filepath.name}")
            return True
        else:
            print(f"  ⏭️  No changes needed")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    """Fix all documented example YAML files."""
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
    
    print("🔧 Comprehensive YAML Fix Tool")
    print("=" * 50)
    
    fixed_count = 0
    for example in documented_examples:
        filepath = examples_dir / example
        if filepath.exists():
            if fix_yaml_file(filepath):
                fixed_count += 1
        else:
            print(f"\n❌ File not found: {example}")
    
    print(f"\n{'=' * 50}")
    print(f"✨ Fixed {fixed_count} files")
    
    # Validate all files
    print(f"\n🔍 Validating all files...")
    validation_errors = []
    
    for example in documented_examples:
        filepath = examples_dir / example
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    yaml.safe_load(f)
                print(f"  ✅ {example}")
            except yaml.YAMLError as e:
                validation_errors.append((example, str(e)))
                print(f"  ❌ {example}: {str(e)[:80]}...")
    
    if validation_errors:
        print(f"\n⚠️  {len(validation_errors)} files still have validation issues")
        for filename, error in validation_errors:
            print(f"\n{filename}:")
            print(f"  {error[:200]}...")
    else:
        print(f"\n✅ All files are valid YAML!")

if __name__ == "__main__":
    main()