#!/usr/bin/env python3
"""Fix action block indentation issues in YAML files."""

import re
from pathlib import Path

def fix_action_block_indentation(content):
    """Fix indentation in action blocks to ensure proper YAML structure."""
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
            
            # Process action block content
            action_content = []
            while i < len(lines):
                next_line = lines[i]
                
                # Check if we've exited the action block
                if next_line.strip() and not next_line.startswith(' ' * (base_indent + 2)):
                    # Check if this is a field that should be outside the action
                    if re.match(r'^\s*(depends_on|condition|timeout|cache_results|on_error|tags|model|requires_model):', next_line):
                        break
                    # If it's numbered list or other content, it should be indented more
                    if re.match(r'^\s*\d+\.', next_line.strip()) or re.match(r'^\s*-\s', next_line.strip()):
                        # Add extra indentation for list items in action blocks
                        fixed_lines.append(' ' * (base_indent + 6) + next_line.strip())
                    else:
                        # Regular content gets standard indentation
                        fixed_lines.append(' ' * (base_indent + 6) + next_line.strip())
                else:
                    fixed_lines.append(next_line)
                i += 1
            i -= 1  # Back up one since we'll increment at the end
        else:
            fixed_lines.append(line)
        i += 1
    
    return '\n'.join(fixed_lines)

def fix_yaml_file(filepath):
    """Fix action block indentation in a YAML file."""
    print(f"Fixing {filepath.name}...")
    
    try:
        content = filepath.read_text()
        original_content = content
        
        # Apply fix
        content = fix_action_block_indentation(content)
        
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
    """Fix action blocks in problematic YAML files."""
    examples_dir = Path("examples")
    
    problem_files = [
        "code_analysis_suite.yaml",
        "content_creation_pipeline.yaml",
        "data_processing_workflow.yaml",
        "multi_agent_collaboration.yaml",
        "automated_testing_system.yaml",
        "document_intelligence.yaml",
        "creative_writing_assistant.yaml",
        "financial_analysis_bot.yaml",
    ]
    
    fixed_count = 0
    for filename in problem_files:
        filepath = examples_dir / filename
        if filepath.exists():
            if fix_yaml_file(filepath):
                fixed_count += 1
    
    print(f"\n✨ Fixed {fixed_count} files")

if __name__ == "__main__":
    main()