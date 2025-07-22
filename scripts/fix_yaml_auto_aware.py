#!/usr/bin/env python3
"""Fix YAML syntax issues while preserving <AUTO> tag functionality."""

import re
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser

def fix_duplicate_depends_on(content):
    """Fix duplicate depends_on entries in YAML files."""
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a depends_on line
        if re.match(r'^\s*depends_on:', line):
            # Look ahead to see if there's another depends_on coming soon
            j = i + 1
            found_duplicate = False
            action_content_between = []
            
            while j < len(lines) and j - i < 5:  # Look at next few lines
                next_line = lines[j]
                if re.match(r'^\s*depends_on:', next_line):
                    # Found duplicate depends_on
                    found_duplicate = True
                    break
                elif next_line.strip() and not next_line.strip().startswith('#'):
                    # Found non-comment content between depends_on lines
                    if not re.match(r'^\s*(condition|timeout|cache_results|on_error|tags|model|requires_model):', next_line):
                        action_content_between.append(next_line)
                j += 1
            
            if found_duplicate and action_content_between:
                # Skip the first depends_on and the content between
                # We'll add the second depends_on which should be correct
                i = j
                continue
                
        fixed_lines.append(line)
        i += 1
    
    return '\n'.join(fixed_lines)

def fix_on_error_structure(content):
    """Fix on_error blocks that have incorrect structure."""
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check for on_error:
        if re.match(r'^(\s*)on_error:', line):
            base_indent = len(line) - len(line.lstrip())
            fixed_lines.append(line)
            i += 1
            
            # Collect the on_error block content
            while i < len(lines):
                next_line = lines[i]
                
                # Check if we've exited the on_error block
                if next_line.strip() and not next_line.startswith(' ' * (base_indent + 2)):
                    # Check if it's a field at the wrong level
                    if re.match(r'^\s*(continue_on_error|retry_count|fallback_value):', next_line):
                        # Fix indentation
                        field_match = re.match(r'^\s*(\w+):(.*)', next_line)
                        if field_match:
                            fixed_lines.append(' ' * (base_indent + 2) + field_match.group(1) + ':' + field_match.group(2))
                            i += 1
                            continue
                    break
                
                # Check for action: | at wrong indentation
                if re.match(r'^\s*action:\s*\|', next_line):
                    # Ensure it's properly indented
                    fixed_lines.append(' ' * (base_indent + 2) + 'action: |')
                    i += 1
                    
                    # Process action content
                    while i < len(lines):
                        action_line = lines[i]
                        if action_line.strip() and not action_line.startswith(' ' * (base_indent + 4)):
                            # Check if it's a field that should be under on_error
                            if re.match(r'^\s*(continue_on_error|retry_count|fallback_value):', action_line):
                                # This should be at on_error level, not in action
                                field_match = re.match(r'^\s*(\w+):(.*)', action_line)
                                if field_match:
                                    fixed_lines.append(' ' * (base_indent + 2) + field_match.group(1) + ':' + field_match.group(2))
                                i += 1
                                continue
                            break
                        if action_line.strip():
                            fixed_lines.append(' ' * (base_indent + 4) + action_line.strip())
                        else:
                            fixed_lines.append('')
                        i += 1
                    i -= 1  # Back up one
                else:
                    fixed_lines.append(next_line)
                
                i += 1
            i -= 1  # Back up one
        else:
            fixed_lines.append(line)
        
        i += 1
    
    return '\n'.join(fixed_lines)

def validate_yaml_with_auto_tags(filepath):
    """Validate YAML file using AUTO-aware parser."""
    try:
        parser = AutoTagYAMLParser()
        with open(filepath, 'r') as f:
            content = f.read()
        
        parser.parse(content)
        return True, None
    except Exception as e:
        return False, str(e)

def fix_yaml_file(filepath):
    """Fix YAML file preserving AUTO tags."""
    print(f"\nProcessing {filepath.name}...")
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Apply fixes
        content = fix_duplicate_depends_on(content)
        content = fix_on_error_structure(content)
        
        # Validate with AUTO-aware parser
        is_valid, error = validate_yaml_with_auto_tags(filepath)
        
        if content != original_content:
            # Write back the fixed content
            with open(filepath, 'w') as f:
                f.write(content)
            
            # Re-validate after fix
            is_valid, error = validate_yaml_with_auto_tags(filepath)
            
            if is_valid:
                print(f"  ✅ Fixed and validated {filepath.name}")
                return True
            else:
                print(f"  ⚠️  Fixed but still has issues: {error[:100]}...")
                return True  # We made progress
        else:
            if is_valid:
                print("  ✅ Already valid")
            else:
                print(f"  ⚠️  No automatic fix available: {error[:100]}...")
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
    
    print("🔧 AUTO-Aware YAML Fix Tool")
    print("=" * 50)
    
    fixed_count = 0
    valid_count = 0
    
    for example in documented_examples:
        filepath = examples_dir / example
        if filepath.exists():
            if fix_yaml_file(filepath):
                fixed_count += 1
            
            # Check final validation status
            is_valid, _ = validate_yaml_with_auto_tags(filepath)
            if is_valid:
                valid_count += 1
        else:
            print(f"\n❌ File not found: {example}")
    
    print(f"\n{'=' * 50}")
    print(f"✨ Fixed {fixed_count} files")
    print(f"✅ {valid_count}/{len(documented_examples)} files are valid")
    
    # Final validation report
    print("\n📊 Final Validation Report:")
    for example in documented_examples:
        filepath = examples_dir / example
        if filepath.exists():
            is_valid, error = validate_yaml_with_auto_tags(filepath)
            if is_valid:
                print(f"  ✅ {example}")
            else:
                print(f"  ❌ {example}")
                if error:
                    print(f"     → {error[:150]}...")

if __name__ == "__main__":
    main()