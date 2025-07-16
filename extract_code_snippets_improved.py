#!/usr/bin/env python3
"""Extract all code snippets from documentation files with improved handling."""

import os
import re
import csv
import json
from pathlib import Path
from typing import List, Dict, Tuple

def extract_markdown_snippets(file_path: Path) -> List[Dict]:
    """Extract code snippets from Markdown files with improved content handling."""
    snippets = []
    content = file_path.read_text()
    lines = content.split('\n')
    
    # Pattern for code blocks with optional language
    code_pattern = re.compile(r'^```(\w+)?$')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        match = code_pattern.match(line)
        
        if match:
            # Found start of code block
            lang = match.group(1) or 'text'  # Use 'text' instead of 'unknown'
            start_line = i + 1  # Line number in file (1-based)
            
            # Find end of code block
            code_lines = []
            i += 1  # Move to first line of code
            code_start = i + 1  # 1-based line number where code starts
            
            while i < len(lines) and not lines[i].strip() == '```':
                code_lines.append(lines[i])
                i += 1
            
            if i < len(lines):  # Found closing ```
                end_line = i + 1  # 1-based line number of closing ```
                code_content = '\n'.join(code_lines)
                
                # Try to find description from previous lines
                description = ""
                # Look backwards from the opening ``` for descriptive text
                for j in range(start_line - 2, max(0, start_line - 10), -1):
                    line_text = lines[j].strip()
                    if line_text and not line_text.startswith('#'):
                        # Found potential description
                        description = line_text
                        # Remove markdown formatting
                        description = re.sub(r'\*\*(.*?)\*\*', r'\1', description)
                        description = re.sub(r'`(.*?)`', r'\1', description)
                        description = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', description)
                        break
                
                if not description:
                    description = f"{lang} code block at line {start_line}"
                
                # Clean up the code content - remove trailing empty lines
                code_lines_clean = []
                for line in code_lines:
                    code_lines_clean.append(line.rstrip())
                while code_lines_clean and not code_lines_clean[-1]:
                    code_lines_clean.pop()
                code_content_clean = '\n'.join(code_lines_clean)
                
                snippets.append({
                    'file': str(file_path),
                    'line_start': code_start,
                    'line_end': end_line - 1,  # Don't include the closing ```
                    'type': lang,
                    'description': description[:200],  # Longer descriptions
                    'content': code_content_clean
                })
        i += 1
    
    return snippets

def extract_rst_snippets(file_path: Path) -> List[Dict]:
    """Extract code snippets from RST files with improved content handling."""
    snippets = []
    content = file_path.read_text()
    lines = content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for .. code-block:: directive
        if line.strip().startswith('.. code-block::'):
            lang = line.strip().split('::')[1].strip() if '::' in line else 'text'
            directive_line = i + 1  # 1-based
            
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
                code_start = i + 1  # 1-based line number where code starts
                
                while i < len(lines):
                    if lines[i].strip():  # Non-empty line
                        current_indent = len(lines[i]) - len(lines[i].lstrip())
                        if current_indent < indent_level:
                            break
                        # Remove the base indentation
                        code_lines.append(lines[i][indent_level:])
                    else:
                        code_lines.append('')  # Preserve empty lines
                    i += 1
                
                end_line = i  # 1-based
                
                # Clean up the code content
                code_lines_clean = []
                for line in code_lines:
                    code_lines_clean.append(line.rstrip())
                while code_lines_clean and not code_lines_clean[-1]:
                    code_lines_clean.pop()
                code_content = '\n'.join(code_lines_clean)
                
                # Find description
                description = ""
                for j in range(directive_line - 2, max(0, directive_line - 10), -1):
                    line_text = lines[j].strip()
                    if line_text and not line_text.startswith('..') and not line_text.startswith('#'):
                        description = line_text
                        break
                
                if not description:
                    description = f"{lang} code block at line {code_start}"
                
                snippets.append({
                    'file': str(file_path),
                    'line_start': code_start,
                    'line_end': end_line,
                    'type': lang,
                    'description': description[:200],
                    'content': code_content
                })
        
        i += 1
    
    return snippets

def extract_python_doctests(file_path: Path) -> List[Dict]:
    """Extract doctest examples from Python files."""
    snippets = []
    content = file_path.read_text()
    lines = content.split('\n')
    
    in_docstring = False
    docstring_indent = 0
    current_example = []
    example_start = 0
    example_indent = 0
    
    for i, line in enumerate(lines):
        # Track docstring boundaries
        if '"""' in line or "'''" in line:
            if not in_docstring:
                in_docstring = True
                # Determine docstring indentation
                docstring_indent = len(line) - len(line.lstrip())
            else:
                # End of docstring - process any pending example
                if current_example:
                    # Clean up the example
                    clean_lines = []
                    for ex_line in current_example:
                        # Remove docstring indentation
                        if ex_line.startswith(' ' * (docstring_indent + 4)):
                            clean_lines.append(ex_line[docstring_indent + 4:])
                        else:
                            clean_lines.append(ex_line.strip())
                    
                    snippets.append({
                        'file': str(file_path),
                        'line_start': example_start + 1,
                        'line_end': i,
                        'type': 'doctest',
                        'description': f"Doctest example in {file_path.name}",
                        'content': '\n'.join(clean_lines)
                    })
                    current_example = []
                in_docstring = False
        
        # Look for doctest examples within docstrings
        elif in_docstring:
            stripped = line.strip()
            if stripped.startswith('>>>'):
                if not current_example:
                    example_start = i
                    example_indent = len(line) - len(line.lstrip())
                current_example.append(line)
            elif current_example and (stripped.startswith('...') or 
                                     (line.startswith(' ' * example_indent) and line.strip())):
                current_example.append(line)
            elif current_example and not line.strip():
                # Empty line might be part of output
                current_example.append(line)
            elif current_example:
                # Non-example line, save current example
                if any('>>>' in l for l in current_example):
                    clean_lines = []
                    for ex_line in current_example:
                        if ex_line.startswith(' ' * (docstring_indent + 4)):
                            clean_lines.append(ex_line[docstring_indent + 4:])
                        else:
                            clean_lines.append(ex_line.strip())
                    
                    snippets.append({
                        'file': str(file_path),
                        'line_start': example_start + 1,
                        'line_end': i,
                        'type': 'doctest',
                        'description': f"Doctest example in {file_path.name}",
                        'content': '\n'.join(clean_lines)
                    })
                current_example = []
    
    return snippets

def find_all_docs_files(root_dir: Path) -> Tuple[List[Path], List[Path], List[Path]]:
    """Find all documentation files."""
    md_files = []
    rst_files = []
    py_files = []
    
    # Directories to search
    search_dirs = [
        'docs',
        'docs_sphinx', 
        '.',  # Root directory for README.md, CLAUDE.md, design.md
        'examples',
        'notes',
        'src/orchestrator'  # For docstrings
    ]
    
    # Files to skip
    skip_patterns = ['test_', '_test', 'htmlcov', '__pycache__', '.git', 'build', '_build']
    
    for search_dir in search_dirs:
        dir_path = root_dir / search_dir
        if dir_path.exists():
            if dir_path.is_file():
                # Handle case where search_dir is actually a file
                if dir_path.suffix == '.md':
                    md_files.append(dir_path)
                elif dir_path.suffix == '.rst':
                    rst_files.append(dir_path)
                elif dir_path.suffix == '.py':
                    py_files.append(dir_path)
            else:
                # Search directory
                for file_path in dir_path.rglob('*'):
                    # Skip if path contains any skip patterns
                    if any(skip in str(file_path) for skip in skip_patterns):
                        continue
                        
                    if file_path.is_file():
                        if file_path.suffix == '.md':
                            md_files.append(file_path)
                        elif file_path.suffix == '.rst':
                            rst_files.append(file_path)
                        elif file_path.suffix == '.py' and search_dir == 'src/orchestrator':
                            # Only include Python files from source directory
                            py_files.append(file_path)
    
    # Remove duplicates while preserving order
    md_files = list(dict.fromkeys(md_files))
    rst_files = list(dict.fromkeys(rst_files))
    py_files = list(dict.fromkeys(py_files))
    
    return md_files, rst_files, py_files

def main():
    """Extract all code snippets from documentation."""
    root = Path('/Users/jmanning/orchestrator')
    
    # Find all documentation files
    print("Finding documentation files...")
    md_files, rst_files, py_files = find_all_docs_files(root)
    
    print(f"Found {len(md_files)} Markdown files")
    print(f"Found {len(rst_files)} RST files")
    print(f"Found {len(py_files)} Python files with potential doctests")
    
    all_snippets = []
    
    # Extract from Markdown files
    print("\nExtracting from Markdown files...")
    for md_file in sorted(md_files):
        print(f"  Processing {md_file.relative_to(root)}")
        try:
            snippets = extract_markdown_snippets(md_file)
            all_snippets.extend(snippets)
            print(f"    Found {len(snippets)} snippets")
        except Exception as e:
            print(f"    Error: {e}")
    
    # Extract from RST files
    print("\nExtracting from RST files...")
    for rst_file in sorted(rst_files):
        print(f"  Processing {rst_file.relative_to(root)}")
        try:
            snippets = extract_rst_snippets(rst_file)
            all_snippets.extend(snippets)
            print(f"    Found {len(snippets)} snippets")
        except Exception as e:
            print(f"    Error: {e}")
    
    # Extract from Python doctests
    print("\nExtracting from Python doctests...")
    for py_file in sorted(py_files):
        print(f"  Processing {py_file.relative_to(root)}")
        try:
            snippets = extract_python_doctests(py_file)
            if snippets:
                all_snippets.extend(snippets)
                print(f"    Found {len(snippets)} doctests")
        except Exception as e:
            print(f"    Error: {e}")
    
    # Write to CSV with proper escaping
    output_file = root / 'code_snippets_extracted.csv'
    print(f"\nWriting {len(all_snippets)} snippets to {output_file}")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['file', 'line_start', 'line_end', 'type', 'description', 
                     'content', 'test_status', 'test_file', 'notes']
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        
        for snippet in all_snippets:
            # Make file path relative to root
            file_path = Path(snippet['file'])
            try:
                rel_path = file_path.relative_to(root)
            except ValueError:
                rel_path = file_path
            
            writer.writerow({
                'file': str(rel_path),
                'line_start': snippet['line_start'],
                'line_end': snippet['line_end'],
                'type': snippet['type'],
                'description': snippet['description'],
                'content': snippet['content'],
                'test_status': 'unverified',
                'test_file': '',
                'notes': ''
            })
    
    # Also write a JSON version for easier processing
    json_file = root / 'code_snippets_extracted.json'
    print(f"Also writing JSON version to {json_file}")
    
    json_snippets = []
    for snippet in all_snippets:
        file_path = Path(snippet['file'])
        try:
            rel_path = file_path.relative_to(root)
        except ValueError:
            rel_path = file_path
            
        json_snippets.append({
            'file': str(rel_path),
            'line_start': snippet['line_start'],
            'line_end': snippet['line_end'],
            'type': snippet['type'],
            'description': snippet['description'],
            'content': snippet['content'],
            'test_status': 'unverified',
            'test_file': '',
            'notes': ''
        })
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_snippets, f, indent=2, ensure_ascii=False)
    
    print(f"\nTotal snippets found: {len(all_snippets)}")
    
    # Print summary by type
    type_counts = {}
    for snippet in all_snippets:
        type_counts[snippet['type']] = type_counts.get(snippet['type'], 0) + 1
    
    print("\nSnippets by type:")
    for lang, count in sorted(type_counts.items()):
        print(f"  {lang}: {count}")
    
    # Print summary by file
    file_counts = {}
    for snippet in all_snippets:
        file_path = Path(snippet['file'])
        try:
            rel_path = str(file_path.relative_to(root))
        except ValueError:
            rel_path = str(file_path)
        file_counts[rel_path] = file_counts.get(rel_path, 0) + 1
    
    print("\nTop files with most snippets:")
    for file_path, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {file_path}: {count}")

if __name__ == '__main__':
    main()