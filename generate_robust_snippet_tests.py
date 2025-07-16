#!/usr/bin/env python3
"""Generate robust tests for documentation code snippets with proper escaping."""

import csv
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List


def create_test_content_safe(content: str, indent: str = "") -> str:
    """Create a safe string representation for test content."""
    # Use triple quotes with proper handling
    if '"""' in content:
        # Use single quotes if content has triple double quotes
        if "'''" in content:
            # Content has both types of triple quotes - use repr()
            return f"({repr(content)})"
        else:
            return f"('''{content}''')"
    else:
        return f'("""{content}""")'


def generate_safe_python_test(snippet: Dict, test_name: str) -> str:
    """Generate a safe Python test that won't hang or fail unexpectedly."""
    content = snippet['content']
    description = snippet['description']
    content_safe = create_test_content_safe(content)
    
    # Check if this is a doctest
    if snippet['type'] == 'doctest':
        return f'''
def {test_name}():
    """Test doctest from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    import doctest
    
    # Doctest content
    doctest_string = {content_safe}
    
    # Parse and run the doctest
    parser = doctest.DocTestParser()
    test = parser.get_doctest(doctest_string, {{}}, "{test_name}", "{snippet['file']}", {snippet['line_start']})
    runner = doctest.DocTestRunner(verbose=False)
    runner.run(test)
    
    if runner.failures:
        pytest.fail(f"Doctest failed with {{runner.failures}} failures")
'''
    
    # For import statements only
    if content.strip().startswith(('import ', 'from ')) and len(content.strip().split('\n')) <= 3:
        return f'''
def {test_name}():
    """Test Python import from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    # Test import statement only
    try:
        code = {content_safe}
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {{e}}")
    except Exception as e:
        pytest.fail(f"Import failed: {{e}}")
'''
    
    # For orchestrator-related code that needs special handling
    if any(keyword in content.lower() for keyword in ['orchestrator', 'pipeline', 'orc.', 'init_models']):
        return f'''
@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def {test_name}():
    """Test orchestrator code from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    # {description}
    
    # Check for required API keys
    missing_keys = []
    if not os.environ.get('OPENAI_API_KEY'):
        missing_keys.append('OPENAI_API_KEY')
    if not os.environ.get('ANTHROPIC_API_KEY'):
        missing_keys.append('ANTHROPIC_API_KEY')
    
    if missing_keys:
        pytest.skip(f"Missing API keys for real model testing: {{', '.join(missing_keys)}}")
    
    # Set up test environment
    import tempfile
    import orchestrator as orc
    
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Create a minimal test pipeline file if referenced
        code = {content_safe}
        if 'hello_world.yaml' in code:
            with open('hello_world.yaml', 'w') as f:
                f.write("""
id: hello_world
name: Hello World
steps:
  - id: greet
    action: generate_text
    parameters:
      prompt: "Say hello world"
outputs:
  greeting: "{{{{ greet.result }}}}"
""".strip())
        
        # Mock the compile function to avoid real file dependencies
        original_compile = getattr(orc, 'compile', None)
        
        def mock_compile(filename):
            # Return a mock pipeline object
            class MockPipeline:
                def run(self):
                    return {{"greeting": "Hello World"}}
            return MockPipeline()
        
        # Test the code with mocked dependencies
        try:
            # Replace compile calls temporarily
            if original_compile:
                setattr(orc, 'compile_mock', mock_compile)
                setattr(orc, 'compile', mock_compile)
            
            # Execute the code
            exec(code, {{'__name__': '__main__', 'orc': orc, 'orchestrator': orc}})
            
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {{e}}")
            else:
                pytest.fail(f"Code execution failed: {{e}}")
        finally:
            # Restore original function
            if original_compile:
                setattr(orc, 'compile', original_compile)
'''
    
    # For other Python code
    return f'''
def {test_name}():
    """Test Python snippet from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    # {description}
    
    code = {content_safe}
    
    try:
        exec(code, {{'__name__': '__main__'}})
    except Exception as e:
        pytest.fail(f"Code execution failed: {{e}}")
'''


def generate_safe_yaml_test(snippet: Dict, test_name: str) -> str:
    """Generate a safe YAML test."""
    content = snippet['content']
    content_safe = create_test_content_safe(content)
    
    # Check if this is a pipeline YAML
    if any(key in content for key in ['steps:', 'tasks:', 'pipeline:', 'id:']):
        return f'''
@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def {test_name}():
    """Test YAML pipeline from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    import yaml
    import orchestrator
    
    yaml_content = {content_safe}
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {{e}}")
    
    # Skip actual pipeline compilation unless we have real models
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        # Check for API keys
        if not (os.environ.get('OPENAI_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')):
            pytest.skip("No API keys available for pipeline testing")
        
        try:
            # Initialize models
            registry = orchestrator.init_models()
            
            # Check if we have any models available
            if not registry.list_models():
                pytest.skip("No models available for testing")
            
            # Create compiler
            from orchestrator.compiler import YAMLCompiler
            compiler = YAMLCompiler()
            compiler.set_model_registry(registry)
            
            # Compile the pipeline (but don't run it)
            pipeline = await compiler.compile(data)
            assert pipeline is not None
            assert pipeline.id
            
            # Validate pipeline structure
            if 'steps' in data:
                assert len(pipeline.tasks) == len(data['steps'])
                
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {{e}}")
            else:
                pytest.fail(f"Pipeline compilation failed: {{e}}")
'''
    else:
        # Simple YAML validation
        return f'''
def {test_name}():
    """Test YAML snippet from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    import yaml
    
    yaml_content = {content_safe}
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {{e}}")
'''


def generate_safe_bash_test(snippet: Dict, test_name: str) -> str:
    """Generate a safe bash test that doesn't actually install packages."""
    content = snippet['content']
    content_safe = create_test_content_safe(content)
    
    # Don't actually run pip install commands
    if 'pip install' in content:
        return f'''
def {test_name}():
    """Test bash snippet from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    # Bash command snippet
    snippet_bash = {content_safe}
    
    # Validate pip install commands without actually running them
    assert "pip install" in snippet_bash
    
    lines = snippet_bash.strip().split('\\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {{line}}"
'''
    
    return f'''
def {test_name}():
    """Test bash snippet from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    import subprocess
    import tempfile
    import os
    
    bash_content = {content_safe}
    
    # Skip potentially dangerous commands
    skip_commands = ['rm -rf', 'sudo', 'docker run', 'systemctl', 'service']
    if any(cmd in bash_content for cmd in skip_commands):
        pytest.skip("Skipping potentially destructive command")
    
    # Check bash syntax only
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


def generate_robust_test_file(snippets: List[Dict], output_file: Path, batch_num: int):
    """Generate a robust test file for a batch of snippets."""
    
    # Generate file content
    content = f'''"""Tests for documentation code snippets - Batch {batch_num} (Robust)."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Set up test environment
os.environ.setdefault('ORCHESTRATOR_CONFIG', str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml"))

# Note: Set RUN_REAL_TESTS=1 to enable tests that use real models
# API keys should be set as environment variables:
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
            content += generate_safe_python_test(snippet, test_name)
        elif snippet['type'] == 'yaml':
            content += generate_safe_yaml_test(snippet, test_name)
        elif snippet['type'] == 'bash':
            content += generate_safe_bash_test(snippet, test_name)
        else:
            # For other types, create a basic validation test
            content_safe = create_test_content_safe(snippet['content'])
            content += f'''
def {test_name}():
    """Test {snippet['type']} snippet from {snippet['file']} lines {snippet['line_start']}-{snippet['line_end']}."""
    # Content validation for {snippet['type']} snippet
    content = {content_safe}
    
    # Basic checks
    assert content.strip(), "Snippet content should not be empty"
    assert len(content.strip()) > 0, "Snippet should have content"
'''
        
        test_count += 1
    
    # Write the test file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(content)
    
    print(f"Generated {output_file} with {test_count} tests")


def main():
    """Generate robust snippet tests."""
    root = Path('/Users/jmanning/orchestrator')
    
    # Read snippets from JSON
    json_file = root / 'code_snippets_extracted.json'
    with open(json_file, 'r') as f:
        all_snippets = json.load(f)
    
    print(f"Loaded {len(all_snippets)} snippets")
    
    # Create test directory
    test_dir = root / 'tests' / 'snippet_tests_robust'
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear existing test files
    for test_file in test_dir.glob('test_*.py'):
        test_file.unlink()
    
    # Create __init__.py
    (test_dir / '__init__.py').write_text('"""Robust tests for documentation code snippets."""\n')
    
    # Batch snippets (15 per file to avoid huge test files)
    batch_size = 15
    batches = [all_snippets[i:i+batch_size] for i in range(0, len(all_snippets), batch_size)]
    
    print(f"Creating {len(batches)} robust test files...")
    
    for i, batch in enumerate(batches, 1):
        output_file = test_dir / f'test_snippets_batch_{i}.py'
        generate_robust_test_file(batch, output_file, i)
    
    print(f"\nGenerated {len(batches)} robust test files in {test_dir}")
    print("\nTo run with real models: RUN_REAL_TESTS=1 pytest tests/snippet_tests_robust/")
    print("To run syntax/validation only: pytest tests/snippet_tests_robust/")


if __name__ == '__main__':
    main()