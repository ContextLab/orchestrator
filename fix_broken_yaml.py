#!/usr/bin/env python3
"""Fix broken YAML files by properly handling AUTO tags."""

import re
import yaml
from pathlib import Path


def fix_yaml_file(file_path):
    """Fix a single YAML file."""
    print(f"Fixing {file_path}...")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this line has an action with AUTO tag
        if re.match(r'\s*action:\s*<AUTO>', line):
            # Find the end of the AUTO tag
            auto_content = []
            auto_line = line
            
            # Keep collecting lines until we find the closing </AUTO> tag
            while i < len(lines) and '</AUTO>' not in lines[i]:
                auto_content.append(lines[i])
                i += 1
            
            # Add the closing line
            if i < len(lines):
                auto_content.append(lines[i])
            
            # Now fix the AUTO block by ensuring proper YAML structure
            fixed_auto = fix_auto_block(auto_content)
            fixed_lines.extend(fixed_auto)
            
        else:
            fixed_lines.append(line)
        
        i += 1
    
    # Write the fixed content back
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)
    
    # Test if it's valid YAML now
    try:
        with open(file_path, 'r') as f:
            yaml.safe_load(f.read())
        print(f"  ✓ Fixed {file_path}")
        return True
    except yaml.YAMLError as e:
        print(f"  ✗ Still broken {file_path}: {e}")
        return False


def fix_auto_block(lines):
    """Fix an AUTO block to ensure proper YAML formatting."""
    if not lines:
        return lines
    
    # Join all lines and then split again to work with the full content
    content = ''.join(lines)
    
    # Use a pipe literal block to preserve the content structure
    # This avoids YAML parsing issues with colons in the content
    if '<AUTO>' in content and '</AUTO>' in content:
        # Extract the action prefix and AUTO content
        parts = content.split('<AUTO>', 1)
        prefix = parts[0]  # "    action: "
        
        rest = parts[1]
        auto_parts = rest.split('</AUTO>', 1)
        auto_content = auto_parts[0]
        suffix = auto_parts[1] if len(auto_parts) > 1 else ""
        
        # Clean up the auto content - remove excessive whitespace but preserve structure
        auto_content = auto_content.strip()
        
        # Use YAML literal block syntax with proper indentation
        # Calculate base indentation from the prefix
        base_indent = len(prefix) - len(prefix.lstrip())
        content_indent = ' ' * (base_indent + 2)
        
        # Create the literal block
        result = f"{prefix}|\n"
        for line in auto_content.split('\n'):
            result += f"{content_indent}{line}\n"
        result += suffix
        
        return result.split('\n')
    
    return lines


def main():
    """Fix all broken YAML files."""
    examples_dir = Path("examples")
    yaml_files = list(examples_dir.glob("*.yaml"))
    
    print(f"Fixing {len(yaml_files)} YAML files...")
    
    fixed_count = 0
    for yaml_file in yaml_files:
        if fix_yaml_file(yaml_file):
            fixed_count += 1
    
    print(f"\nFixed YAML files: {fixed_count}/{len(yaml_files)}")


if __name__ == "__main__":
    main()