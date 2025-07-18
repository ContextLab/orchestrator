#!/usr/bin/env python3
"""Fix remaining YAML issues in example files."""

from pathlib import Path
import re

def fix_yaml_auto_tags(filepath):
    """Fix YAML files by properly handling AUTO tags."""
    print(f"\nProcessing {filepath.name}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Look for patterns where content appears after depends_on that should be in an AUTO tag
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a depends_on line followed by action content
        if 'depends_on:' in line and i + 1 < len(lines):
            # Add the depends_on line
            fixed_lines.append(line)
            i += 1
            
            # Check if next lines look like action content (numbered lists, etc)
            action_content = []
            start_collecting = False
            
            while i < len(lines):
                next_line = lines[i]
                
                # Check if this looks like action content that's misplaced
                if (re.match(r'^\s+(Create tests for:|Extract:|For each conversation:|Calculate metrics:)', next_line) or
                    (start_collecting and re.match(r'^\s+\d+\.', next_line)) or
                    (start_collecting and next_line.strip() and not re.match(r'^\s*(depends_on|condition|timeout|cache_results|on_error|tags|model|requires_model|-):', next_line))):
                    action_content.append(next_line)
                    start_collecting = True
                    i += 1
                elif start_collecting and not next_line.strip():
                    # Empty line while collecting
                    action_content.append(next_line)
                    i += 1
                else:
                    # We've hit something that's not action content
                    break
            
            # If we collected action content, we need to find where it belongs
            if action_content:
                # Look backwards for the action: | line
                j = len(fixed_lines) - 1
                while j >= 0:
                    if re.match(r'^\s*action:\s*\|', fixed_lines[j]):
                        # Found the action line
                        action_indent = len(fixed_lines[j]) - len(fixed_lines[j].lstrip())
                        
                        # Find where to insert the content
                        insert_pos = j + 1
                        while insert_pos < len(fixed_lines) and fixed_lines[insert_pos].strip() and not re.match(r'^\s*(depends_on|condition|timeout|cache_results|on_error|tags|model|requires_model):', fixed_lines[insert_pos]):
                            insert_pos += 1
                        
                        # Check if the action content should be wrapped in AUTO tags
                        if action_content and not any('<AUTO>' in line for line in action_content):
                            # Check if there's already content and we need to add to it
                            existing_content_end = insert_pos - 1
                            if existing_content_end > j and '</AUTO>' in fixed_lines[existing_content_end]:
                                # Insert before the closing AUTO tag
                                fixed_lines[existing_content_end:existing_content_end] = action_content
                            else:
                                # Add the content with proper indentation
                                for content_line in action_content:
                                    if content_line.strip():
                                        fixed_lines.insert(insert_pos, ' ' * (action_indent + 2) + content_line.strip())
                                    else:
                                        fixed_lines.insert(insert_pos, '')
                                    insert_pos += 1
                        break
                    j -= 1
                
                # Continue processing from where we left off
                i -= 1  # Back up one since we'll increment at the end
        else:
            fixed_lines.append(line)
        
        i += 1
    
    # Write the fixed content
    fixed_content = '\n'.join(fixed_lines)
    with open(filepath, 'w') as f:
        f.write(fixed_content)
    
    print(f"  ✅ Fixed {filepath.name}")

def main():
    """Fix remaining problematic YAML files."""
    examples_dir = Path("examples")
    
    problem_files = [
        "automated_testing_system.yaml",
        "document_intelligence.yaml", 
        "creative_writing_assistant.yaml",
        "financial_analysis_bot.yaml",
    ]
    
    for filename in problem_files:
        filepath = examples_dir / filename
        if filepath.exists():
            fix_yaml_auto_tags(filepath)

if __name__ == "__main__":
    main()