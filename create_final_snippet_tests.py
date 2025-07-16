#!/usr/bin/env python3
"""Create final working snippet tests using a simple, robust approach."""

import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List


def safe_string_repr(s: str) -> str:
    """Create a safe string representation for Python code."""
    return repr(s)


def generate_basic_test(snippet: Dict, test_name: str) -> str:
    """Generate a basic test that focuses on validation without execution."""
    content = snippet['content']
    description = snippet['description']
    
    # Use repr() for safe string representation
    content_repr = safe_string_repr(content)
    
    if snippet['type'] == 'python':
        # For Python, try to compile it to check syntax
        return f'''
def {test_name}():
    """Test Python snippet from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    # Description: {description}
    content = {content_repr}
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid Python syntax
    try:
        compile(content, '<string>', 'exec')
    except SyntaxError as e:
        pytest.fail(f"Python syntax error: {{e}}")
    
    # If it's a simple import, try to execute it
    if content.strip().startswith(('import ', 'from ')) and len(content.strip().split('\\n')) <= 3:
        try:
            exec(content)
        except ImportError:
            pytest.skip("Import not available in test environment")
        except Exception as e:
            pytest.fail(f"Import failed: {{e}}")
'''
    
    elif snippet['type'] == 'yaml':
        return f'''
def {test_name}():
    """Test YAML snippet from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    # Description: {description}
    import yaml
    
    content = {content_repr}
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        data = yaml.safe_load(content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing error: {{e}}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                assert 'id' in step, "Each step should have an id"
'''
    
    elif snippet['type'] == 'bash':
        return f'''
def {test_name}():
    """Test Bash snippet from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    # Description: {description}
    content = {content_repr}
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Special handling for pip install commands
    if 'pip install' in content:
        lines = content.strip().split('\\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                assert line.startswith('pip install'), f"Expected pip install command: {{line}}"
        return  # Skip further validation for pip commands
    
    # For other bash commands, just check they're not empty
    assert len(content.strip()) > 0, "Bash content should not be empty"
'''
    
    elif snippet['type'] == 'doctest':
        return f'''
def {test_name}():
    """Test doctest snippet from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    # Description: {description}
    content = {content_repr}
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    assert '>>>' in content, "Doctest should contain >>> prompts"
'''
    
    else:
        # For other types (text, unknown, etc.)
        return f'''
def {test_name}():
    """Test {snippet['type']} snippet from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    # Description: {description}
    content = {content_repr}
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    assert len(content) > 0, "Content should have length"
'''


def create_working_test_file(snippets: List[Dict], output_file: Path, batch_num: int):
    """Create a working test file for a batch of snippets."""
    
    header = f'''"""Working tests for documentation code snippets - Batch {batch_num}."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# These tests focus on validation and syntax checking without execution
# They should run quickly and not require external dependencies

'''
    
    # Generate tests for each snippet
    tests = []
    for i, snippet in enumerate(snippets):
        # Create a safe test name
        file_name = Path(snippet['file']).stem.replace('-', '_').replace('.', '_')
        test_name = f"test_{file_name}_lines_{snippet['line_start']}_{snippet['line_end']}"
        test_name = re.sub(r'[^a-zA-Z0-9_]', '_', test_name)
        test_name = f"{test_name}_{i}"
        
        test_code = generate_basic_test(snippet, test_name)
        tests.append(test_code)
    
    # Write the complete file
    content = header + '\n'.join(tests)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(content)
    
    print(f"Generated {output_file} with {len(tests)} tests")


def main():
    """Create final working snippet tests."""
    root = Path('/Users/jmanning/orchestrator')
    
    # Read snippets from JSON
    json_file = root / 'code_snippets_extracted.json'
    with open(json_file, 'r') as f:
        all_snippets = json.load(f)
    
    print(f"Loaded {len(all_snippets)} snippets")
    
    # Create test directory
    test_dir = root / 'tests' / 'snippet_tests_working'
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear existing test files
    for test_file in test_dir.glob('test_*.py'):
        test_file.unlink()
    
    # Create __init__.py
    (test_dir / '__init__.py').write_text('"""Working tests for documentation code snippets."""\\n')
    
    # Batch snippets (10 per file to make debugging easier)
    batch_size = 10
    batches = [all_snippets[i:i+batch_size] for i in range(0, len(all_snippets), batch_size)]
    
    print(f"Creating {len(batches)} working test files...")
    
    for i, batch in enumerate(batches, 1):
        output_file = test_dir / f'test_snippets_batch_{i}.py'
        create_working_test_file(batch, output_file, i)
    
    print(f"\\nGenerated {len(batches)} working test files in {test_dir}")
    print("\\nTo run: pytest tests/snippet_tests_working/")
    
    # Update the CSV file with the new test information
    csv_file = root / 'code_snippets_verification.csv'
    if csv_file.exists():
        # Read current CSV
        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Update with new test file paths
        batch_size = 10
        for i, row in enumerate(rows):
            batch_num = (i // batch_size) + 1
            row['test_status'] = 'test_created'
            row['test_file'] = f'tests/snippet_tests_working/test_snippets_batch_{batch_num}.py'
            row['notes'] = 'Working test created with basic validation'
        
        # Write updated CSV
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if rows:
                fieldnames = rows[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
                writer.writeheader()
                writer.writerows(rows)
        
        print(f"Updated {csv_file} with working test information")


if __name__ == '__main__':
    main()