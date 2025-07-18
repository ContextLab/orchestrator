#!/usr/bin/env python3
"""Fix malformed action blocks in YAML files."""

import re
from pathlib import Path


def fix_malformed_action(content: str) -> str:
    """Fix action blocks that are collapsed into single lines."""
    
    # Pattern to find action: | followed by text on the same line
    # This regex captures the action content that needs to be reformatted
    pattern = r'(action:\s*\|)([^\n]+)((?:\s+depends_on:|\s+condition:|\s+timeout:|\s+tags:|\s+on_error:|\s+-\s+id:))'
    
    def reformat_action(match):
        prefix = match.group(1)  # "action: |"
        action_content = match.group(2).strip()  # The collapsed action text
        suffix = match.group(3)  # The next field
        
        # Split the action content by multiple spaces (likely collapsed newlines)
        # First, replace common patterns
        formatted_content = action_content
        
        # Replace multiple spaces with newlines where appropriate
        formatted_content = re.sub(r'\s{4,}', '\n      ', formatted_content)
        
        # Fix numbered lists
        formatted_content = re.sub(r'(\d+\.)\s+', r'\n      \1 ', formatted_content)
        
        # Ensure proper indentation after colons
        formatted_content = re.sub(r':\s+(?=[A-Z])', r':\n      ', formatted_content)
        
        # Clean up any double newlines
        formatted_content = re.sub(r'\n\s*\n', '\n', formatted_content)
        
        # Ensure first line is properly indented
        if not formatted_content.startswith('\n'):
            formatted_content = '\n      ' + formatted_content
        
        # Add proper final newline with correct indentation
        formatted_content = formatted_content.rstrip() + '\n    '
        
        return prefix + formatted_content + suffix
    
    # Apply the fix
    fixed_content = re.sub(pattern, reformat_action, content, flags=re.MULTILINE | re.DOTALL)
    
    return fixed_content


def main():
    """Fix all YAML files in the examples directory."""
    examples_dir = Path("examples")
    
    if not examples_dir.exists():
        print("Examples directory not found!")
        return
    
    for yaml_file in examples_dir.glob("*.yaml"):
        print(f"Processing {yaml_file.name}...")
        
        with open(yaml_file, 'r') as f:
            content = f.read()
        
        # Check if file needs fixing
        if re.search(r'action:\s*\|[^\n]+[a-zA-Z]', content):
            print(f"  - Fixing malformed actions in {yaml_file.name}")
            fixed_content = fix_malformed_action(content)
            
            with open(yaml_file, 'w') as f:
                f.write(fixed_content)
        else:
            print(f"  - No issues found in {yaml_file.name}")


if __name__ == "__main__":
    main()