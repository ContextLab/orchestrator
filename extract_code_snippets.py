#!/usr/bin/env python3
"""Extract all code snippets from documentation files."""

import os
import re
import csv
from pathlib import Path
from typing import List, Dict, Tuple

def extract_markdown_snippets(file_path: Path) -> List[Dict]:
    """Extract code snippets from Markdown files."""
    snippets = []
    content = file_path.read_text()
    lines = content.split('\n')
    
    # Pattern for code blocks with language
    code_pattern = re.compile(r'^```(\w+)?$')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        match = code_pattern.match(line)
        
        if match:
            # Found start of code block
            lang = match.group(1) or 'unknown'
            i += 1  # Skip the ``` line
            start_line = i + 1  # Line numbers are 1-based
            
            # Find end of code block
            code_lines = []
            while i < len(lines) and not lines[i].strip() == '```':
                code_lines.append(lines[i])
                i += 1
            
            if i < len(lines):  # Found closing ```
                end_line = i  # This is the line with closing ```
                code_content = '\n'.join(code_lines)
                
                # Try to find description from previous line
                desc_line = start_line - 2
                description = ""
                if desc_line >= 0:
                    # Look for nearby text that might describe the snippet
                    for j in range(max(0, start_line - 5), start_line - 1):
                        if lines[j].strip() and not lines[j].startswith('#'):
                            description = lines[j].strip()
                            break
                
                if not description:
                    description = f"{lang} code block at line {start_line}"
                
                snippets.append({
                    'file': str(file_path),
                    'line_start': start_line,
                    'line_end': end_line,
                    'type': lang,
                    'description': description[:100],  # Truncate long descriptions
                    'content': code_content
                })
        i += 1
    
    return snippets

def extract_rst_snippets(file_path: Path) -> List[Dict]:
    """Extract code snippets from RST files."""
    snippets = []
    content = file_path.read_text()
    lines = content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for .. code-block:: directive
        if line.strip().startswith('.. code-block::'):
            lang = line.strip().split('::')[1].strip() if '::' in line else 'unknown'
            start_line = i + 1
            
            # Skip blank lines after directive
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            
            if i < len(lines):
                # Determine indentation level
                indent_match = re.match(r'^(\s+)', lines[i])
                indent_level = len(indent_match.group(1)) if indent_match else 0
                
                # Collect code lines
                code_lines = []
                actual_start = i + 1  # Line number in file (1-based)
                while i < len(lines):
                    if lines[i].strip():  # Non-empty line
                        current_indent = len(lines[i]) - len(lines[i].lstrip())
                        if current_indent < indent_level:
                            break
                    code_lines.append(lines[i])
                    i += 1
                
                end_line = i
                code_content = '\n'.join(code_lines)
                
                # Find description
                description = ""
                for j in range(max(0, start_line - 5), start_line):
                    if lines[j].strip() and not lines[j].startswith('..'):
                        description = lines[j].strip()
                        break
                
                if not description:
                    description = f"{lang} code block at line {actual_start}"
                
                snippets.append({
                    'file': str(file_path),
                    'line_start': actual_start,
                    'line_end': end_line,
                    'type': lang,
                    'description': description[:100],
                    'content': code_content.strip()
                })
        
        # Check for inline code literals
        elif '``' in line:
            matches = re.finditer(r'``([^`]+)``', line)
            for match in matches:
                code = match.group(1)
                # Only track substantial inline code (not single words)
                if ' ' in code or any(c in code for c in ['(', ')', '=', '.', '[', ']']):
                    snippets.append({
                        'file': str(file_path),
                        'line_start': i + 1,
                        'line_end': i + 1,
                        'type': 'inline',
                        'description': f"Inline code: {code[:50]}",
                        'content': code
                    })
        
        i += 1
    
    return snippets

def extract_python_doctests(file_path: Path) -> List[Dict]:
    """Extract doctest examples from Python files."""
    snippets = []
    content = file_path.read_text()
    lines = content.split('\n')
    
    in_docstring = False
    docstring_start = 0
    current_example = []
    example_start = 0
    
    for i, line in enumerate(lines):
        # Track docstring boundaries
        if '"""' in line or "'''" in line:
            if not in_docstring:
                in_docstring = True
                docstring_start = i
            else:
                in_docstring = False
                # Process any pending example
                if current_example:
                    snippets.append({
                        'file': str(file_path),
                        'line_start': example_start + 1,
                        'line_end': i + 1,
                        'type': 'doctest',
                        'description': f"Doctest example in {file_path.name}",
                        'content': '\n'.join(current_example)
                    })
                    current_example = []
        
        # Look for doctest examples
        if in_docstring and line.strip().startswith('>>>'):
            if not current_example:
                example_start = i
            current_example.append(line.strip())
        elif in_docstring and current_example and (line.strip().startswith('...') or 
                                                   (line.strip() and not line.strip().startswith('>>>'))):
            current_example.append(line.strip())
    
    return snippets

def find_all_docs_files(root_dir: Path) -> Tuple[List[Path], List[Path], List[Path]]:
    """Find all documentation files."""
    md_files = []
    rst_files = []
    py_files = []
    
    # Documentation directories to search
    doc_dirs = ['docs', '.', 'examples', 'tutorials']
    
    for doc_dir in doc_dirs:
        dir_path = root_dir / doc_dir
        if dir_path.exists():
            # Find all relevant files
            for file_path in dir_path.rglob('*'):
                if file_path.is_file():
                    if file_path.suffix == '.md':
                        md_files.append(file_path)
                    elif file_path.suffix == '.rst':
                        rst_files.append(file_path)
                    elif file_path.suffix == '.py' and 'test' not in file_path.name:
                        # Include Python files that might have doctest examples
                        py_files.append(file_path)
    
    return md_files, rst_files, py_files

def main():
    """Extract all code snippets from documentation."""
    root = Path('/Users/jmanning/orchestrator')
    
    # Find all documentation files
    print("Finding documentation files...")
    md_files, rst_files, py_files = find_all_docs_files(root)
    
    print(f"Found {len(md_files)} Markdown files")
    print(f"Found {len(rst_files)} RST files")
    print(f"Found {len(py_files)} Python files")
    
    all_snippets = []
    
    # Extract from Markdown files
    print("\nExtracting from Markdown files...")
    for md_file in md_files:
        print(f"  Processing {md_file}")
        snippets = extract_markdown_snippets(md_file)
        all_snippets.extend(snippets)
    
    # Extract from RST files
    print("\nExtracting from RST files...")
    for rst_file in rst_files:
        print(f"  Processing {rst_file}")
        snippets = extract_rst_snippets(rst_file)
        all_snippets.extend(snippets)
    
    # Extract from Python doctests
    print("\nExtracting from Python doctests...")
    for py_file in py_files:
        if 'src/orchestrator' in str(py_file):  # Only process source files
            print(f"  Processing {py_file}")
            snippets = extract_python_doctests(py_file)
            all_snippets.extend(snippets)
    
    # Write to CSV
    output_file = root / 'code_snippets_verification.csv'
    print(f"\nWriting {len(all_snippets)} snippets to {output_file}")
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'line_start', 'line_end', 'type', 'description', 'test_status', 'test_file', 'notes'])
        
        for snippet in all_snippets:
            # Make file path relative to root
            file_path = Path(snippet['file'])
            try:
                rel_path = file_path.relative_to(root)
            except ValueError:
                rel_path = file_path
            
            writer.writerow([
                str(rel_path),
                snippet['line_start'],
                snippet['line_end'],
                snippet['type'],
                snippet['description'],
                'unverified',
                '',
                ''
            ])
    
    print(f"\nTotal snippets found: {len(all_snippets)}")
    
    # Print summary by type
    type_counts = {}
    for snippet in all_snippets:
        type_counts[snippet['type']] = type_counts.get(snippet['type'], 0) + 1
    
    print("\nSnippets by type:")
    for lang, count in sorted(type_counts.items()):
        print(f"  {lang}: {count}")

if __name__ == '__main__':
    main()