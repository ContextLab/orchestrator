#!/usr/bin/env python3
"""Find all mentions of 'mock' in the codebase for issue #10 checklist."""

import os
import re
from pathlib import Path
from collections import defaultdict

def find_mock_mentions(root_dir):
    """Find all files that mention 'mock' in any way."""
    mock_files = defaultdict(list)
    
    # Patterns to search for (case insensitive)
    patterns = [
        r'\bmock\b',
        r'\bMock\b',
        r'\bMOCK\b',
        r'mock_',
        r'_mock',
        r'Mock[A-Z]',
        r'mock[A-Z]',
    ]
    
    combined_pattern = re.compile('|'.join(patterns), re.IGNORECASE)
    
    for root, dirs, files in os.walk(root_dir):
        # Skip .git and __pycache__ directories
        dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache', 'node_modules'}]
        
        for file in files:
            # Only check Python files and markdown files
            if not (file.endswith('.py') or file.endswith('.md')):
                continue
                
            filepath = Path(root) / file
            relative_path = filepath.relative_to(root_dir)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Find all mock mentions with line numbers
                for line_num, line in enumerate(content.splitlines(), 1):
                    if combined_pattern.search(line):
                        # Get context of what type of mock usage
                        context = "unknown"
                        if 'MockModel' in line:
                            context = "MockModel"
                        elif 'MockControlSystem' in line:
                            context = "MockControlSystem"
                        elif 'mock_mode' in line:
                            context = "mock_mode"
                        elif 'from unittest.mock' in line or 'import mock' in line:
                            context = "unittest.mock import"
                        elif 'Mock(' in line or 'AsyncMock(' in line or 'MagicMock(' in line:
                            context = "mock instantiation"
                        elif '@mock' in line or '@patch' in line:
                            context = "mock decorator"
                        elif 'mock' in line.lower() and '#' in line:
                            context = "comment"
                        elif 'mock' in line.lower():
                            context = "other mention"
                            
                        mock_files[str(relative_path)].append({
                            'line': line_num,
                            'content': line.strip()[:100] + ('...' if len(line.strip()) > 100 else ''),
                            'context': context
                        })
                        
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
    
    return mock_files

def categorize_files(mock_files):
    """Categorize files by type."""
    categories = {
        'production_core': [],
        'production_other': [],
        'tests': [],
        'examples': [],
        'docs': [],
        'scripts': [],
    }
    
    for filepath in sorted(mock_files.keys()):
        if filepath.startswith('tests/'):
            categories['tests'].append(filepath)
        elif filepath.startswith('examples/'):
            categories['examples'].append(filepath)
        elif filepath.startswith('docs/') or filepath.endswith('.md'):
            categories['docs'].append(filepath)
        elif filepath.startswith('scripts/') or filepath in ['run_example_minimal.py', 'find_all_mocks.py']:
            categories['scripts'].append(filepath)
        elif filepath.startswith('src/'):
            # Separate core vs other production code
            if any(core in filepath for core in ['core/', 'compiler/', 'engine/', 'orchestrator.py', '__init__.py']):
                categories['production_core'].append(filepath)
            else:
                categories['production_other'].append(filepath)
        else:
            categories['scripts'].append(filepath)
    
    return categories

def generate_checklist(mock_files, categories):
    """Generate markdown checklist for issue #10."""
    checklist = ["## Complete Mock Removal Checklist\n"]
    checklist.append("Every file that mentions 'mock' in any way (including comments, class names, imports, etc.):\n")
    
    # Production code first (highest priority)
    if categories['production_core']:
        checklist.append("\n### 🚨 Production Code - Core")
        for filepath in categories['production_core']:
            mentions = mock_files[filepath]
            checklist.append(f"- [ ] `{filepath}` ({len(mentions)} mentions)")
            for m in mentions[:3]:  # Show first 3 mentions
                checklist.append(f"  - Line {m['line']}: {m['context']} - `{m['content']}`")
            if len(mentions) > 3:
                checklist.append(f"  - ... and {len(mentions) - 3} more mentions")
    
    if categories['production_other']:
        checklist.append("\n### 🚨 Production Code - Other")
        for filepath in categories['production_other']:
            mentions = mock_files[filepath]
            checklist.append(f"- [ ] `{filepath}` ({len(mentions)} mentions)")
            for m in mentions[:2]:  # Show first 2 mentions
                checklist.append(f"  - Line {m['line']}: {m['context']}")
    
    # Examples and scripts
    if categories['examples']:
        checklist.append("\n### 📘 Examples")
        for filepath in categories['examples']:
            checklist.append(f"- [ ] `{filepath}` ({len(mock_files[filepath])} mentions)")
    
    if categories['scripts']:
        checklist.append("\n### 📜 Scripts")
        for filepath in categories['scripts']:
            checklist.append(f"- [ ] `{filepath}` ({len(mock_files[filepath])} mentions)")
    
    # Documentation
    if categories['docs']:
        checklist.append("\n### 📚 Documentation")
        for filepath in categories['docs']:
            checklist.append(f"- [ ] `{filepath}` ({len(mock_files[filepath])} mentions)")
    
    # Tests (lowest priority)
    if categories['tests']:
        checklist.append(f"\n### 🧪 Test Files ({len(categories['tests'])} files)")
        checklist.append("*Note: Test files may need to be converted to use real API calls instead of mocks*")
        for filepath in categories['tests'][:10]:  # Show first 10
            checklist.append(f"- [ ] `{filepath}` ({len(mock_files[filepath])} mentions)")
        if len(categories['tests']) > 10:
            checklist.append(f"- [ ] ... and {len(categories['tests']) - 10} more test files")
    
    # Summary
    total_files = sum(len(files) for files in categories.values())
    total_mentions = sum(len(mentions) for mentions in mock_files.values())
    checklist.append("\n### 📊 Summary")
    checklist.append(f"- Total files with mock mentions: {total_files}")
    checklist.append(f"- Total mock mentions: {total_mentions}")
    checklist.append(f"- Production code files: {len(categories['production_core']) + len(categories['production_other'])}")
    checklist.append(f"- Test files: {len(categories['tests'])}")
    
    return '\n'.join(checklist)

def main():
    root_dir = Path(__file__).parent
    print(f"Searching for all mock mentions in {root_dir}...")
    
    mock_files = find_mock_mentions(root_dir)
    categories = categorize_files(mock_files)
    checklist = generate_checklist(mock_files, categories)
    
    # Save to file
    output_file = root_dir / 'mock_removal_checklist.md'
    with open(output_file, 'w') as f:
        f.write(checklist)
    
    print(f"\nChecklist saved to: {output_file}")
    print(f"\nFound {len(mock_files)} files with mock mentions")
    
    # Also print to console for immediate use
    print("\n" + "="*80)
    print(checklist)

if __name__ == '__main__':
    main()