"""Working tests for documentation code snippets - Batch 24."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# These tests focus on validation and syntax checking without execution
# They should run quickly and not require external dependencies


def test_getting_started_lines_141_148_0():
    """Test Python snippet from docs_sphinx/getting_started.rst lines 141-148."""
    # Description: The same pipeline works with different inputs:
    content = '# One pipeline, many uses\npipeline = orc.compile("report-template.yaml")\n\n# Generate different reports\nai_report = pipeline.run(topic="AI", style="technical")\nbio_report = pipeline.run(topic="Biology", style="educational")\neco_report = pipeline.run(topic="Economics", style="executive")'
    
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


def test_getting_started_lines_153_160_1():
    """Test YAML snippet from docs_sphinx/getting_started.rst lines 153-160."""
    # Description: eco_report = pipeline.run(topic="Economics", style="executive")
    import yaml
    
    content = 'inputs:\n  topic:\n    type: string\n    required: true\n  style:\n    type: string\n    default: "technical"'
    
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


def test_getting_started_lines_168_177_2():
    """Test YAML snippet from docs_sphinx/getting_started.rst lines 168-177."""
    # Description: Tools are automatically detected and made available:
    import yaml
    
    content = 'steps:\n  - id: fetch_data\n    action: search_web        # Auto-detects web tool\n\n  - id: save_results\n    action: write_file        # Auto-detects filesystem tool\n\n  - id: run_analysis\n    action: "!python analyze.py"  # Auto-detects terminal tool'
    
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


def test_getting_started_lines_185_193_3():
    """Test Python snippet from docs_sphinx/getting_started.rst lines 185-193."""
    # Description: The framework intelligently selects the best model for each task:
    content = "# Models are selected based on:\n# - Task requirements (reasoning, coding, etc.)\n# - Available resources\n# - Performance history\n\nregistry = orc.init_models()\nprint(registry.list_models())\n# Output: ['ollama:gemma2:27b', 'ollama:llama3.2:1b', 'huggingface:gpt2']"
    
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


def test_index_lines_43_56_4():
    """Test Python snippet from docs_sphinx/index.rst lines 43-56."""
    # Description: -------------
    content = 'import orchestrator as orc\n\n# Initialize models\norc.init_models()\n\n# Compile a pipeline\npipeline = orc.compile("pipelines/research-report.yaml")\n\n# Execute with different inputs\nresult = pipeline.run(\n    topic="quantum_computing",\n    instructions="Focus on error correction"\n)'
    
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


def test_installation_lines_34_41_5():
    """Test Bash snippet from docs_sphinx/installation.rst lines 34-41."""
    # Description: -----------------------
    content = '# Install from PyPI (when available)\npip install py-orc\n\n# Or install from source\ngit clone https://github.com/ContextLab/orchestrator.git\ncd orchestrator\npip install -e .'
    
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


def test_installation_lines_47_53_6():
    """Test Bash snippet from docs_sphinx/installation.rst lines 47-53."""
    # Description: -----------
    content = '# Create conda environment\nconda create -n py-orc python=3.11\nconda activate py-orc\n\n# Install orchestrator\npip install py-orc'
    
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


def test_installation_lines_59_64_7():
    """Test Bash snippet from docs_sphinx/installation.rst lines 59-64."""
    # Description: ------------
    content = '# Pull the official image\ndocker pull contextlab/py-orc:latest\n\n# Run with volume mount\ndocker run -v $(pwd):/workspace contextlab/py-orc'
    
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


def test_installation_lines_72_85_8():
    """Test Bash snippet from docs_sphinx/installation.rst lines 72-85."""
    # Description: For contributors and developers:
    content = '# Clone the repository\ngit clone https://github.com/ContextLab/orchestrator.git\ncd orchestrator\n\n# Create virtual environment\npython -m venv venv\nsource venv/bin/activate  # On Windows: venv\\Scripts\\activate\n\n# Install in development mode with extras\npip install -e ".[dev,test,docs]"\n\n# Install pre-commit hooks\npre-commit install'
    
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


def test_installation_lines_96_101_9():
    """Test Bash snippet from docs_sphinx/installation.rst lines 96-101."""
    # Description: 1. **Install Ollama**:
    content = '# macOS\nbrew install ollama\n\n# Linux\ncurl -fsSL https://ollama.ai/install.sh | sh'
    
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
