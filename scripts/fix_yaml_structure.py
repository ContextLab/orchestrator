#!/usr/bin/env python3
"""Fix specific YAML structure issues by examining and correcting action blocks."""

from pathlib import Path
import re

def fix_broken_action_blocks(filepath):
    """Fix action blocks where depends_on or other fields interrupt the action content."""
    print(f"Fixing {filepath.name}...")
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        fixed_lines = []
        i = 0
        changes_made = False
        
        while i < len(lines):
            line = lines[i]
            
            # Check if this is a depends_on line that might have content after it
            if 'depends_on:' in line and i + 1 < len(lines):
                # Check if the next line looks like it should be part of an action block
                next_line = lines[i + 1]
                if re.match(r'^\s+\d+\.', next_line) or (next_line.strip() and ':' not in next_line and not next_line.strip().startswith('-')):
                    # This content should come before depends_on
                    # Find where the action block should end
                    action_lines = []
                    j = i + 1
                    while j < len(lines) and (re.match(r'^\s+\d+\.', lines[j]) or 
                                              (lines[j].strip() and ':' not in lines[j] and not lines[j].strip().startswith('-'))):
                        action_lines.append(lines[j])
                        j += 1
                    
                    # Insert the action lines before depends_on
                    if action_lines:
                        # Don't add the current line yet
                        for action_line in action_lines:
                            fixed_lines.append(action_line)
                        # Now add the depends_on line
                        fixed_lines.append(line)
                        # Skip the lines we just moved
                        i = j
                        changes_made = True
                        continue
            
            fixed_lines.append(line)
            i += 1
        
        if changes_made:
            with open(filepath, 'w') as f:
                f.writelines(fixed_lines)
            print(f"  ✅ Fixed {filepath.name}")
            return True
        else:
            print("  ⏭️  No changes needed")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    """Fix specific problematic files."""
    examples_dir = Path("examples")
    
    # Files that need fixing based on validation errors
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
            if fix_broken_action_blocks(filepath):
                fixed_count += 1
    
    print(f"\n✨ Fixed {fixed_count} files")

if __name__ == "__main__":
    main()