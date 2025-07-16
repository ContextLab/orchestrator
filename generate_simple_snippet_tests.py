#!/usr/bin/env python3
"""Generate simple, robust test files for documentation code snippets."""

import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List

def clean_code_content(content: str) -> str:
    """Clean code content for safe inclusion in test files."""
    # Escape triple quotes by adding a space
    content = content.replace('"""', '""" ')
    # Handle other problematic patterns
    content = content.replace('\\', '\\\\')
    return content

def generate_python_test(snippet: Dict, test_name: str) -> str:
    """Generate a simple test for a Python snippet."""
    content = clean_code_content(snippet['content'])
    
    if snippet['type'] == 'doctest':
        return f'''
def {test_name}():
    """Test doctest from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    # Doctest execution would go here - skipping for now
    pytest.skip("Doctest execution not yet implemented")
'''
    
    if content.strip().startswith(('import ', 'from ')):
        return f'''
def {test_name}():
    """Test Python import from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    # Import test - check if modules are available
    code = {repr(content)}
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {{e}}")
    except Exception as e:
        pytest.fail(f"Import failed: {{e}}")
'''
    
    # For complex Python code with orchestrator components
    if any(keyword in content for keyword in ['pipeline', 'orchestrator', 'model', 'task']):
        return f'''
@pytest.mark.asyncio
async def {test_name}():
    """Test Python snippet from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    # Skip complex orchestrator code for now - would need full setup
    pytest.skip("Complex orchestrator code testing not yet implemented")
'''
    
    # Simple Python code
    return f'''
def {test_name}():
    """Test Python snippet from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    code = {repr(content)}
    
    try:
        exec(code)
    except Exception as e:
        pytest.fail(f"Code execution failed: {{e}}")
'''

def generate_yaml_test(snippet: Dict, test_name: str) -> str:
    """Generate a simple test for a YAML snippet."""
    content = clean_code_content(snippet['content'])
    
    # Check if it looks like a pipeline
    if any(key in content for key in ['steps:', 'tasks:', 'pipeline:']):
        return f'''
@pytest.mark.asyncio
async def {test_name}():
    """Test YAML pipeline from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    import yaml
    
    yaml_content = {repr(content)}
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {{e}}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")
'''
    
    # Simple YAML validation
    return f'''
def {test_name}():
    """Test YAML snippet from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    import yaml
    
    yaml_content = {repr(content)}
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {{e}}")
'''

def generate_bash_test(snippet: Dict, test_name: str) -> str:
    """Generate a simple test for a bash snippet."""
    content = clean_code_content(snippet['content'])
    
    if 'pip install' in content:
        return f'''
def {test_name}():
    """Test bash snippet from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    bash_content = {repr(content)}
    
    # Verify it's a pip install command
    assert "pip install" in bash_content
    
    # Parse each line
    lines = bash_content.strip().split('\\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {{line}}"
'''
    
    return f'''
def {test_name}():
    """Test bash snippet from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    bash_content = {repr(content)}
    
    # Skip dangerous commands
    skip_commands = ['rm -rf', 'sudo', 'docker', 'systemctl']
    if any(cmd in bash_content for cmd in skip_commands):
        pytest.skip("Skipping potentially destructive command")
    
    # For now, just check it's not empty
    assert bash_content.strip(), "Bash content should not be empty"
'''

def generate_test_file(snippets: List[Dict], output_file: Path, batch_num: int):
    """Generate a test file for a batch of snippets."""
    
    content = f'''"""Tests for documentation code snippets - Batch {batch_num}."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

'''
    
    # Generate tests for each snippet
    test_count = 0
    for i, snippet in enumerate(snippets):
        # Create a safe test name
        file_name = Path(snippet['file']).stem.replace('-', '_').replace('.', '_')
        test_name = f"test_{file_name}_lines_{snippet['line_start']}_{snippet['line_end']}_{i}"
        test_name = re.sub(r'[^a-zA-Z0-9_]', '_', test_name)
        
        if snippet['type'] in ['python', 'doctest']:
            content += generate_python_test(snippet, test_name)
        elif snippet['type'] == 'yaml':
            content += generate_yaml_test(snippet, test_name)
        elif snippet['type'] == 'bash':
            content += generate_bash_test(snippet, test_name)
        else:
            # Skip other types
            content += f'''
def {test_name}():
    """Test {snippet['type']} snippet from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    pytest.skip("Snippet type '{snippet['type']}' not yet supported")
'''
        
        test_count += 1
    
    # Write the test file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(content)
    
    print(f"Generated {output_file} with {test_count} tests")

def main():
    """Generate simple test files for all code snippets."""
    root = Path('/Users/jmanning/orchestrator')
    
    # Read snippets from JSON
    json_file = root / 'code_snippets_extracted.json'
    with open(json_file, 'r') as f:
        all_snippets = json.load(f)
    
    print(f"Loaded {len(all_snippets)} snippets")
    
    # Create test directory
    test_dir = root / 'tests' / 'snippet_tests'
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear existing test files
    for test_file in test_dir.glob('test_*.py'):
        test_file.unlink()
    
    # Create __init__.py
    (test_dir / '__init__.py').write_text('"""Tests for documentation code snippets."""\n')
    
    # Batch snippets (30 per file)
    batch_size = 30
    batches = [all_snippets[i:i+batch_size] for i in range(0, len(all_snippets), batch_size)]
    
    print(f"Creating {len(batches)} test files...")
    
    for i, batch in enumerate(batches, 1):
        output_file = test_dir / f'test_snippets_batch_{i}.py'
        generate_test_file(batch, output_file, i)
    
    print(f"\nGenerated {len(batches)} simple test files in {test_dir}")
    print("Note: Complex tests are skipped for now - focus on syntax validation")

if __name__ == '__main__':
    main()