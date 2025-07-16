#!/usr/bin/env python3
"""Fix quote issues in snippet test files by using raw strings."""

import re
from pathlib import Path

def fix_test_file(file_path: Path):
    """Fix quote issues in a test file."""
    content = file_path.read_text()
    
    # Find all triple-quoted strings that might have issues
    # This regex finds content between triple quotes
    pattern = r'(""")(.*?)(""")'
    
    def fix_quotes(match):
        prefix = match.group(1)
        content = match.group(2)
        suffix = match.group(3)
        
        # If content ends with quotes, we need to escape or adjust
        if content.rstrip().endswith('"'):
            # Add a newline to separate the quotes
            if not content.endswith('\n'):
                content = content.rstrip() + '\n'
        
        return prefix + content + suffix
    
    # Apply fix with DOTALL flag to match across newlines
    fixed_content = re.sub(pattern, fix_quotes, content, flags=re.DOTALL)
    
    # Also check for specific problematic patterns
    # Fix cases where we have 4 or more quotes in a row
    fixed_content = re.sub(r'""""+', '"""\n', fixed_content)
    
    # Write back if changed
    if fixed_content != content:
        file_path.write_text(fixed_content)
        return True
    return False

def main():
    """Fix all snippet test files."""
    test_dir = Path('/Users/jmanning/orchestrator/tests/snippet_tests')
    
    print("Fixing quote issues in test files...")
    fixed_count = 0
    
    for test_file in sorted(test_dir.glob('test_snippets_batch_*.py')):
        if fix_test_file(test_file):
            print(f"  Fixed: {test_file.name}")
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")

if __name__ == '__main__':
    main()