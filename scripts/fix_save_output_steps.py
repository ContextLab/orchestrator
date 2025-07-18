#!/usr/bin/env python3
"""Fix save_output steps that use {{ previous_steps }}."""

from pathlib import Path
import re

def fix_save_output_step(content):
    """Fix save_output steps to not use {{ previous_steps }}."""
    # Pattern to find save_output steps with depends_on using previous_steps
    pattern = r'depends_on:\s*\["{{ previous_steps \| join\(\', \'\) }}"\]'
    
    # Find all step IDs in the file (except save_output)
    step_ids = []
    for match in re.finditer(r'^\s*-\s+id:\s+(\w+)', content, re.MULTILINE):
        step_id = match.group(1)
        if step_id != 'save_output':
            step_ids.append(step_id)
    
    # Replace the pattern with actual step IDs
    if step_ids:
        # Get the last meaningful step (not save_output)
        last_step = step_ids[-1] if step_ids else "generate_report"
        replacement = f'depends_on: ["{last_step}"]'
        content = re.sub(pattern, replacement, content)
    
    return content

def main():
    """Fix save_output steps in all YAML files."""
    examples_dir = Path("examples")
    
    yaml_files = [
        "code_analysis_suite.yaml",
        "multi_agent_collaboration.yaml", 
        "automated_testing_system.yaml",
    ]
    
    for filename in yaml_files:
        filepath = examples_dir / filename
        if filepath.exists():
            print(f"Processing {filename}...")
            
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Apply fix
            fixed_content = fix_save_output_step(content)
            
            if content != fixed_content:
                with open(filepath, 'w') as f:
                    f.write(fixed_content)
                print(f"  ✅ Fixed {filename}")
            else:
                print(f"  ⏭️  No changes needed")

if __name__ == "__main__":
    main()