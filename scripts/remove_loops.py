#!/usr/bin/env python3
"""Remove loop constructs from YAML files."""

import re
from pathlib import Path

def remove_loop_constructs(yaml_content):
    """Remove loop constructs from YAML content."""
    # Pattern to match loop constructs
    loop_pattern = r'\n\s*loop:\s*\n(?:\s*[^:\n]+:[^\n]+\n)*'
    
    # Remove loop constructs
    cleaned_content = re.sub(loop_pattern, '\n', yaml_content)
    
    # Also remove any remaining foreach references in action text
    foreach_pattern = r'{{item\.([^}]+)}}'
    cleaned_content = re.sub(foreach_pattern, r'{{loop_item.\1}}', cleaned_content)
    
    # Remove standalone {{item}} references
    item_pattern = r'{{item}}'
    cleaned_content = re.sub(item_pattern, r'{{loop_item}}', cleaned_content)
    
    return cleaned_content

def process_yaml_file(file_path):
    """Process a single YAML file."""
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if file has loop constructs
    if 'loop:' not in content:
        print(f"  No loop constructs found in {file_path}")
        return
    
    # Remove loop constructs
    cleaned_content = remove_loop_constructs(content)
    
    # Write back to file
    with open(file_path, 'w') as f:
        f.write(cleaned_content)
    
    print(f"  Removed loop constructs from {file_path}")

def main():
    """Process all YAML files in examples directory."""
    examples_dir = Path("examples")
    
    if not examples_dir.exists():
        print("Examples directory not found!")
        return
    
    yaml_files = list(examples_dir.glob("*.yaml"))
    
    if not yaml_files:
        print("No YAML files found!")
        return
    
    print(f"Found {len(yaml_files)} YAML files to process")
    
    for yaml_file in yaml_files:
        process_yaml_file(yaml_file)
    
    print("Done!")

if __name__ == "__main__":
    main()