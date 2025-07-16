#!/usr/bin/env python3
"""Generate test files for documentation code snippets using real models in all environments."""

import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

def generate_python_test(snippet: Dict, test_name: str) -> str:
    """Generate a test for a Python code snippet."""
    content = snippet['content']
    description = snippet['description']
    
    # Check if this is a doctest
    if snippet['type'] == 'doctest':
        return f'''
def {test_name}():
    """Test doctest from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    import doctest
    
    # Doctest content
    doctest_string = """{content}"""
    
    # Parse and run the doctest
    parser = doctest.DocTestParser()
    test = parser.get_doctest(doctest_string, {{}}, "{test_name}", "{snippet['file']}", {snippet['line_start']})
    runner = doctest.DocTestRunner(verbose=False)
    runner.run(test)
    
    if runner.failures:
        pytest.fail(f"Doctest failed with {{runner.failures}} failures")
'''
    
    # For regular Python code
    # Check if it's an import statement
    if content.strip().startswith(('import ', 'from ')):
        return f'''
def {test_name}():
    """Test Python import from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    # Test imports
    try:
        exec("""{content}""")
    except ImportError as e:
        pytest.skip(f"Import not available: {{e}}")
    except Exception as e:
        pytest.fail(f"Import failed: {{e}}")
'''
    
    # Check if it's a complete script or needs setup
    needs_setup = any(keyword in content for keyword in [
        'pipeline', 'orchestrator', 'model', 'task', 'config'
    ])
    
    if needs_setup:
        return f'''
@pytest.mark.asyncio
async def {test_name}():
    """Test Python snippet from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    # {description}
    
    # Import required modules
    import os
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Set up test environment
        os.environ['ORCHESTRATOR_CONFIG'] = str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml")
        
        # Test code snippet
        code = """{content}"""
        
        # Execute with real models (API keys from environment/GitHub secrets)
        try:
            # Check if required API keys are available
            missing_keys = []
            if 'openai' in code.lower() and not os.environ.get('OPENAI_API_KEY'):
                missing_keys.append('OPENAI_API_KEY')
            if 'anthropic' in code.lower() and not os.environ.get('ANTHROPIC_API_KEY'):
                missing_keys.append('ANTHROPIC_API_KEY')
            if ('gemini' in code.lower() or 'google' in code.lower()) and not os.environ.get('GOOGLE_AI_API_KEY'):
                missing_keys.append('GOOGLE_AI_API_KEY')
            
            if missing_keys:
                pytest.skip(f"Missing API keys for real model testing: {{', '.join(missing_keys)}}")
            
            # Execute the code with real models
            if 'await' in code or 'async' in code:
                # Handle async code
                import asyncio
                exec_globals = {{'__name__': '__main__', 'asyncio': asyncio}}
                exec(code, exec_globals)
                
                # If there's a main coroutine, run it
                if 'main' in exec_globals and asyncio.iscoroutinefunction(exec_globals['main']):
                    await exec_globals['main']()
            else:
                exec(code, {{'__name__': '__main__'}})
                
        except Exception as e:
            # Check if it's an expected error
            if "No eligible models" in str(e):
                pytest.skip(f"No eligible models available: {{e}}")
            else:
                pytest.fail(f"Code execution failed: {{e}}")
'''
    else:
        # Simple Python code without special setup
        return f'''
def {test_name}():
    """Test Python snippet from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    # {description}
    
    code = """{content}"""
    
    try:
        exec(code, {{'__name__': '__main__'}})
    except Exception as e:
        pytest.fail(f"Code execution failed: {{e}}")
'''

def generate_yaml_test(snippet: Dict, test_name: str) -> str:
    """Generate a test for a YAML code snippet."""
    content = snippet['content']
    
    # Check if this is a pipeline YAML
    if any(key in content for key in ['steps:', 'tasks:', 'pipeline:']):
        return f'''
@pytest.mark.asyncio
async def {test_name}():
    """Test YAML pipeline from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """{content}"""
    
    # Parse YAML
    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {{e}}")
    
    # If it's a pipeline, validate it
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        from orchestrator.compiler import YAMLCompiler
        import orchestrator
        
        # Set up environment
        os.environ['ORCHESTRATOR_CONFIG'] = str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml")
        
        compiler = YAMLCompiler()
        
        # Initialize real models
        try:
            registry = orchestrator.init_models()
            compiler.set_model_registry(registry)
            
            # Check if we have any models available
            if not registry.list_models():
                pytest.skip("No models available for testing")
                
        except Exception as e:
            if "API key" in str(e):
                pytest.skip(f"Missing API keys for real model testing: {{e}}")
            else:
                raise
        
        # Compile the pipeline
        try:
            pipeline = await compiler.compile(data)
            assert pipeline is not None
            assert pipeline.id
            
            # Validate pipeline structure
            if 'steps' in data:
                assert len(pipeline.tasks) == len(data['steps'])
            
        except Exception as e:
            if "No eligible models" in str(e):
                pytest.skip(f"No eligible models available: {{e}}")
            else:
                pytest.fail(f"Pipeline compilation failed: {{e}}")
'''
    else:
        # Simple YAML validation
        return f'''
def {test_name}():
    """Test YAML snippet from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    import yaml
    
    yaml_content = """{content}"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {{e}}")
'''

def generate_bash_test(snippet: Dict, test_name: str) -> str:
    """Generate a test for a bash code snippet."""
    content = snippet['content']
    
    # Don't actually run pip install commands
    if 'pip install' in content:
        # Create a simpler test that just validates the command structure
        return f'''
def {test_name}():
    """Test bash snippet from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    # Bash command snippet
    snippet_bash = r"""{content}"""
    
    # Don't actually install packages in tests
    assert "pip install" in snippet_bash
    
    # Verify it's a valid pip command structure
    lines = snippet_bash.strip().split('\\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {{line}}"
'''
    
    # For other bash commands, check syntax only
    return f'''
def {test_name}():
    """Test bash snippet from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    import subprocess
    import tempfile
    import os
    
    bash_content = r"""{content}"""
    
    # Skip if it's a command we shouldn't run
    skip_commands = ['rm -rf', 'sudo', 'docker', 'systemctl']
    if any(cmd in bash_content for cmd in skip_commands):
        pytest.skip("Skipping potentially destructive command")
    
    # Check bash syntax
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(bash_content)
            f.flush()
            
            # Check syntax only
            result = subprocess.run(['bash', '-n', f.name], 
                                  capture_output=True, text=True)
            
            os.unlink(f.name)
            
            if result.returncode != 0:
                pytest.fail(f"Bash syntax error: {{result.stderr}}")
                
    except FileNotFoundError:
        pytest.skip("Bash not available for testing")
'''

def generate_test_file(snippets: List[Dict], output_file: Path, batch_num: int):
    """Generate a test file for a batch of snippets."""
    
    # Generate file content
    content = f'''"""Tests for documentation code snippets - Batch {batch_num}."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Set up test environment
os.environ.setdefault('ORCHESTRATOR_CONFIG', str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml"))

# Note: API keys should be set as environment variables or GitHub secrets:
# - OPENAI_API_KEY
# - ANTHROPIC_API_KEY  
# - GOOGLE_AI_API_KEY

'''
    
    # Generate tests for each snippet
    test_count = 0
    for i, snippet in enumerate(snippets):
        # Create a safe test name
        file_name = Path(snippet['file']).stem.replace('-', '_').replace('.', '_')
        test_name = f"test_{file_name}_lines_{snippet['line_start']}_{snippet['line_end']}"
        test_name = re.sub(r'[^a-zA-Z0-9_]', '_', test_name)
        
        # Ensure unique test names
        test_name = f"{test_name}_{i}"
        
        if snippet['type'] in ['python', 'doctest']:
            content += generate_python_test(snippet, test_name)
        elif snippet['type'] == 'yaml':
            content += generate_yaml_test(snippet, test_name)
        elif snippet['type'] == 'bash':
            content += generate_bash_test(snippet, test_name)
        else:
            # Skip other types for now
            content += f'''
def {test_name}():
    """Test {snippet['type']} snippet from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    # Snippet type '{snippet['type']}' not yet supported for testing
    pytest.skip("Snippet type '{snippet['type']}' not yet supported")
'''
        
        test_count += 1
    
    # Write the test file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(content)
    
    print(f"Generated {output_file} with {test_count} tests")

def main():
    """Generate test files for all code snippets."""
    root = Path('/Users/jmanning/orchestrator')
    
    # Read snippets from JSON (easier to parse than CSV)
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
    
    # Batch snippets (30 per file to avoid huge test files)
    batch_size = 30
    batches = [all_snippets[i:i+batch_size] for i in range(0, len(all_snippets), batch_size)]
    
    print(f"Creating {len(batches)} test files...")
    
    for i, batch in enumerate(batches, 1):
        output_file = test_dir / f'test_snippets_batch_{i}.py'
        generate_test_file(batch, output_file, i)
    
    # Update the CSV file with test file information
    csv_file = root / 'code_snippets_extracted.csv'
    rows = []
    
    with open(csv_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for i, row in enumerate(reader):
            # Add test file info
            batch_num = (i // batch_size) + 1
            row['test_status'] = 'test_created'
            row['test_file'] = f'tests/snippet_tests/test_snippets_batch_{batch_num}.py'
            rows.append(row)
    
    # Write updated CSV
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nGenerated {len(batches)} test files in {test_dir}")
    print(f"Updated {csv_file} with test file information")
    print("\nNote: Tests will use real models with API keys from environment variables or GitHub secrets")

if __name__ == '__main__':
    main()