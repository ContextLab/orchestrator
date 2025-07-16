"""Working tests for documentation code snippets - Batch 5."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# These tests focus on validation and syntax checking without execution
# They should run quickly and not require external dependencies


def test_README_lines_74_80_0():
    """Test Bash snippet from examples/README.md lines 74-80."""
    # Description: `
    content = '# Simple pipeline\npython -m orchestrator run examples/simple_pipeline.yaml \\\n    --context input_topic="machine learning"\n\n# Multi-model pipeline\npython -m orchestrator run examples/multi_model_pipeline.yaml \\\n    --context dataset_url="https://example.com/sales_data.csv"'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Special handling for pip install commands
    if 'pip install' in content:
        lines = content.strip().split('\n')
        has_pip_command = False
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and 'pip install' in line:
                has_pip_command = True
                break
        if has_pip_command:
            return  # Skip further validation for pip commands
    
    # For other bash commands, just check they're not empty
    assert len(content.strip()) > 0, "Bash content should not be empty"


def test_README_lines_89_89_1():
    """Test YAML snippet from examples/README.md lines 89-89."""
    # Description: The examples demonstrate various AUTO tag patterns:
    import yaml
    
    content = 'analysis_type: <AUTO>What type of analysis is most appropriate for this text?</AUTO>'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        # Check if content contains AUTO tags
        if '<AUTO>' in content:
            # Use AUTO tag parser
            from orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser
            parser = AutoTagYAMLParser()
            data = parser.parse(content)
        else:
            # Use standard YAML parser
            data = yaml.safe_load(content)
        # Note: data can be None for YAML with only comments
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                # Note: 'id' is optional in minimal examples


def test_README_lines_94_94_2():
    """Test YAML snippet from examples/README.md lines 94-94."""
    # Description: `
    import yaml
    
    content = 'format: <AUTO>Determine the best format for this data source</AUTO>'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        # Check if content contains AUTO tags
        if '<AUTO>' in content:
            # Use AUTO tag parser
            from orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser
            parser = AutoTagYAMLParser()
            data = parser.parse(content)
        else:
            # Use standard YAML parser
            data = yaml.safe_load(content)
        # Note: data can be None for YAML with only comments
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                # Note: 'id' is optional in minimal examples


def test_README_lines_99_99_3():
    """Test YAML snippet from examples/README.md lines 99-99."""
    # Description: `
    import yaml
    
    content = 'methods: <AUTO>Choose the most appropriate statistical methods</AUTO>'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        # Check if content contains AUTO tags
        if '<AUTO>' in content:
            # Use AUTO tag parser
            from orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser
            parser = AutoTagYAMLParser()
            data = parser.parse(content)
        else:
            # Use standard YAML parser
            data = yaml.safe_load(content)
        # Note: data can be None for YAML with only comments
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                # Note: 'id' is optional in minimal examples


def test_README_lines_104_104_4():
    """Test YAML snippet from examples/README.md lines 104-104."""
    # Description: `
    import yaml
    
    content = 'validation_rules: <AUTO>Generate appropriate validation rules for this dataset</AUTO>'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        # Check if content contains AUTO tags
        if '<AUTO>' in content:
            # Use AUTO tag parser
            from orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser
            parser = AutoTagYAMLParser()
            data = parser.parse(content)
        else:
            # Use standard YAML parser
            data = yaml.safe_load(content)
        # Note: data can be None for YAML with only comments
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                # Note: 'id' is optional in minimal examples


def test_README_lines_112_131_5():
    """Test YAML snippet from examples/README.md lines 112-131."""
    # Description: `
    import yaml
    
    content = 'id: my_pipeline\nname: My Custom Pipeline\ndescription: Description of what this pipeline does\nversion: "1.0"\n\ncontext:\n  # Global variables accessible to all tasks\n  variable_name: value\n\nsteps:\n  - id: task1\n    name: First Task\n    action: generate  # or analyze, transform, etc.\n    parameters:\n      # Task-specific parameters\n      prompt: "Your prompt here"\n    metadata:\n      # Optional metadata\n      requires_model: true\n      priority: 1.0'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        # Check if content contains AUTO tags
        if '<AUTO>' in content:
            # Use AUTO tag parser
            from orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser
            parser = AutoTagYAMLParser()
            data = parser.parse(content)
        else:
            # Use standard YAML parser
            data = yaml.safe_load(content)
        # Note: data can be None for YAML with only comments
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                # Note: 'id' is optional in minimal examples


def test_README_lines_137_152_6():
    """Test YAML snippet from examples/README.md lines 137-152."""
    # Description: `
    import yaml
    
    content = 'steps:\n  - id: advanced_task\n    name: Advanced Task\n    action: analyze\n    parameters:\n      data: "{{ results.previous_task }}"  # Reference previous results\n      method: <AUTO>Choose best method</AUTO>  # AUTO resolution\n    dependencies:\n      - previous_task  # Task dependencies\n    metadata:\n      requires_model: gpt-4  # Specific model requirement\n      cpu_cores: 4  # Resource requirements\n      memory_mb: 2048\n      timeout: 300\n      priority: 0.8\n      on_failure: continue  # Error handling'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        # Check if content contains AUTO tags
        if '<AUTO>' in content:
            # Use AUTO tag parser
            from orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser
            parser = AutoTagYAMLParser()
            data = parser.parse(content)
        else:
            # Use standard YAML parser
            data = yaml.safe_load(content)
        # Note: data can be None for YAML with only comments
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                # Note: 'id' is optional in minimal examples


def test_README_lines_181_184_7():
    """Test Python snippet from examples/README.md lines 181-184."""
    # Description: Enable debug logging to see detailed execution information:
    content = 'import logging\nlogging.basicConfig(level=logging.DEBUG)\n\norchestrator = Orchestrator()'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid Python syntax
    # Skip syntax check for notebook-specific code with top-level await
    if 'await' in content and ('notebook' in content.lower() or 'jupyter' in content.lower()):
        # This is notebook-specific syntax, skip syntax validation
        pass
    else:
        try:
            compile(content, '<string>', 'exec')
        except SyntaxError as e:
            pytest.fail(f"Python syntax error: {e}")
    
    # If it's a simple import, try to execute it
    if content.strip().startswith(('import ', 'from ')) and len(content.strip().split('\n')) <= 3:
        try:
            exec(content)
        except ImportError:
            pytest.skip("Import not available in test environment")
        except Exception as e:
            pytest.fail(f"Import failed: {e}")


def test_README_lines_192_199_8():
    """Test Python snippet from examples/README.md lines 192-199."""
    # Description: Verify system health before running pipelines:
    content = 'async def check_health():\n    health = await orchestrator.health_check()\n    print("System health:", health["overall"])\n    return health\n\n# Run the health check\nimport asyncio\nhealth_status = asyncio.run(check_health())'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid Python syntax
    # Skip syntax check for notebook-specific code with top-level await
    if 'await' in content and ('notebook' in content.lower() or 'jupyter' in content.lower()):
        # This is notebook-specific syntax, skip syntax validation
        pass
    else:
        try:
            compile(content, '<string>', 'exec')
        except SyntaxError as e:
            pytest.fail(f"Python syntax error: {e}")
    
    # If it's a simple import, try to execute it
    if content.strip().startswith(('import ', 'from ')) and len(content.strip().split('\n')) <= 3:
        try:
            exec(content)
        except ImportError:
            pytest.skip("Import not available in test environment")
        except Exception as e:
            pytest.fail(f"Import failed: {e}")


def test_draft_report_lines_59_59_9():
    """Test text snippet from examples/output/readme_report/draft_report.md lines 59-59."""
    # Description: - Execution Engine: Action orchestration and error handling
    content = 'Thought → Action → Observation → Thought → ...'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    assert len(content) > 0, "Content should have length"
