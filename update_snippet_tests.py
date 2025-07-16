#!/usr/bin/env python3
"""Update snippet tests to fix string escaping issues."""

import re
from pathlib import Path
from typing import List

def fix_snippet_tests(file_path: Path):
    """Fix string escaping in snippet test files."""
    content = file_path.read_text()
    
    # Find all snippet assignments and fix escaping
    lines = content.split('\n')
    fixed_lines = []
    in_snippet = False
    snippet_start_line = -1
    snippet_var = None
    snippet_content = []
    
    for i, line in enumerate(lines):
        # Check for start of snippet assignment
        match = re.match(r'^\s*(snippet_\w+)\s*=\s*"""(.*)$', line)
        if match:
            in_snippet = True
            snippet_start_line = i
            snippet_var = match.group(1)
            # Start content with what's after the """
            snippet_content = [match.group(2)] if match.group(2) else []
            fixed_lines.append(line)  # Keep the original line for now
            continue
        
        # Check for end of snippet
        if in_snippet and line.strip().endswith('"""'):
            in_snippet = False
            # Get the content before the closing """
            content_before_quotes = line[:line.rfind('"""')]
            if content_before_quotes.strip():
                snippet_content.append(content_before_quotes)
            
            # Now reconstruct the snippet with proper escaping
            full_content = '\n'.join(snippet_content)
            
            # Escape backslashes first, then quotes
            escaped_content = full_content.replace('\\', '\\\\').replace('"', '\\"')
            
            # Replace the lines from snippet_start_line to current line
            # Remove the old lines
            del fixed_lines[snippet_start_line:]
            
            # Add the new escaped version
            fixed_lines.append(f'    {snippet_var} = "{escaped_content}"')
            
            snippet_content = []
            continue
        
        # If we're in a snippet, collect the content
        if in_snippet:
            snippet_content.append(line)
            fixed_lines.append(line)  # Keep for now, will be removed later
        else:
            fixed_lines.append(line)
    
    # Write back
    file_path.write_text('\n'.join(fixed_lines))

def main():
    """Fix all snippet test files."""
    test_dir = Path('/Users/jmanning/orchestrator/tests/snippet_tests')
    
    if not test_dir.exists():
        print(f"Test directory {test_dir} does not exist")
        return
    
    for test_file in test_dir.glob('test_snippets_batch_*.py'):
        print(f"Fixing {test_file.name}...")
        fix_snippet_tests(test_file)
    
    print("Done!")

if __name__ == '__main__':
    main()