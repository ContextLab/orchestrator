#!/usr/bin/env python3
"""Remove timeout decorators and parameters from tests."""

import re
import os
from pathlib import Path

def remove_timeout_decorators(content):
    """Remove @pytest.mark.timeout(...) decorators."""
    # Pattern to match @pytest.mark.timeout(number) on its own line
    pattern = r'^\s*@pytest\.mark\.timeout\(\d+\)\s*\n'
    return re.sub(pattern, '', content, flags=re.MULTILINE)

def remove_timeout_parameters(content):
    """Remove timeout= parameters from function calls."""
    # Pattern to match timeout=number as a parameter
    # This handles both timeout=30 and timeout=30.0
    pattern = r',?\s*timeout\s*=\s*[\d.]+\s*(?=[,)])'
    
    # First pass: remove timeout parameters
    content = re.sub(pattern, '', content)
    
    # Clean up any double commas or leading commas that might be left
    content = re.sub(r'\(\s*,', '(', content)
    content = re.sub(r',\s*,', ',', content)
    content = re.sub(r',\s*\)', ')', content)
    
    return content

def process_file(filepath):
    """Process a single file to remove timeouts."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Remove timeout decorators
    content = remove_timeout_decorators(content)
    
    # Remove timeout parameters
    content = remove_timeout_parameters(content)
    
    # Only write if content changed
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    """Process all test files."""
    test_dir = Path('/home/jmanning/orchestrator/tests')
    
    # Find all Python test files
    test_files = list(test_dir.rglob('test_*.py'))
    test_files.extend(list(test_dir.rglob('*_test_*.py')))
    
    modified_count = 0
    for filepath in sorted(test_files):
        if process_file(filepath):
            print(f"Modified: {filepath}")
            modified_count += 1
    
    print(f"\nTotal files modified: {modified_count}")

if __name__ == '__main__':
    main()