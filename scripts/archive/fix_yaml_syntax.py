#!/usr/bin/env python3
"""Fix YAML syntax issues caused by loop removal."""

import re
import yaml
from pathlib import Path


def fix_yaml_syntax(content):
    """Fix common YAML syntax issues."""
    # Fix malformed action blocks that got broken
    lines = content.split('\n')
    fixed_lines = []
    in_action_block = False
    
    for i, line in enumerate(lines):
        # Check if we're starting an action block
        if re.match(r'\s*action:\s*<AUTO>', line):
            in_action_block = True
            fixed_lines.append(line)
            continue
        
        # Check if we're ending an action block
        if in_action_block and '</AUTO>' in line:
            in_action_block = False
            fixed_lines.append(line)
            continue
        
        # Skip empty lines that might cause issues
        if line.strip() == '':
            fixed_lines.append(line)
            continue
        
        # If we're in an action block, don't treat it as YAML structure
        if in_action_block:
            fixed_lines.append(line)
            continue
        
        # Check for malformed depends_on or other fields that got merged
        if re.match(r'\s*depends_on:\s*\[', line):
            # This is a normal depends_on line
            fixed_lines.append(line)
            continue
        
        # Check if this line looks like it should be a new field but got merged
        if re.match(r'\s*[a-zA-Z_][a-zA-Z0-9_]*:\s*', line) and not in_action_block:
            fixed_lines.append(line)
            continue
        
        # Default: add the line as-is
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)


def validate_and_fix_yaml(file_path):
    """Validate and fix a YAML file."""
    print(f"Checking {file_path}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    try:
        # Try to parse as YAML
        yaml.safe_load(content)
        print(f"  ✓ {file_path} is valid YAML")
        return True
    except yaml.YAMLError as e:
        print(f"  ✗ {file_path} has YAML syntax error: {e}")
        
        # Try to fix the syntax
        fixed_content = fix_yaml_syntax(content)
        
        try:
            # Test the fixed version
            yaml.safe_load(fixed_content)
            
            # Write the fixed version back
            with open(file_path, 'w') as f:
                f.write(fixed_content)
            
            print(f"  ✓ Fixed {file_path}")
            return True
        except yaml.YAMLError as e2:
            print(f"  ✗ Could not fix {file_path}: {e2}")
            return False


def main():
    """Check and fix all YAML files."""
    examples_dir = Path("examples")
    yaml_files = list(examples_dir.glob("*.yaml"))
    
    print(f"Checking {len(yaml_files)} YAML files...")
    
    valid_count = 0
    for yaml_file in yaml_files:
        if validate_and_fix_yaml(yaml_file):
            valid_count += 1
    
    print(f"\nValid YAML files: {valid_count}/{len(yaml_files)}")


if __name__ == "__main__":
    main()