#!/usr/bin/env python3
"""Fix snippet tests by removing code fence markers from content."""

import re
from pathlib import Path
from typing import List

def fix_test_file(file_path: Path):
    """Fix a single test file by removing code fence markers."""
    content = file_path.read_text()
    
    # Pattern to find snippet assignments with code fences
    pattern = r'(snippet_(?:bash|yaml|python|code|content)\s*=\s*""")(```[a-z]*\n)(.*?)(```?)(""")'
    
    def replacer(match):
        # Keep groups 1 (variable assignment), 3 (actual code), and 5 (closing quotes)
        # Skip groups 2 (opening fence) and 4 (closing fence)
        prefix = match.group(1)
        code = match.group(3)
        suffix = match.group(5)
        
        # Remove trailing newline if it exists
        code = code.rstrip('\n')
        
        return prefix + code + suffix
    
    # Replace all occurrences
    fixed_content = re.sub(pattern, replacer, content, flags=re.DOTALL)
    
    # Write back if changed
    if fixed_content != content:
        file_path.write_text(fixed_content)
        return True
    return False

def main():
    """Fix all snippet test files."""
    test_dir = Path('/Users/jmanning/orchestrator/tests/snippet_tests')
    
    if not test_dir.exists():
        print(f"Test directory {test_dir} does not exist")
        return
    
    fixed_count = 0
    for test_file in test_dir.glob('test_snippets_batch_*.py'):
        print(f"Processing {test_file.name}...", end=' ')
        if fix_test_file(test_file):
            print("fixed")
            fixed_count += 1
        else:
            print("no changes")
    
    print(f"\nFixed {fixed_count} test files")

if __name__ == '__main__':
    main()