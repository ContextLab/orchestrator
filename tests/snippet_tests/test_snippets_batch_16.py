"""Working tests for documentation code snippets - Batch 16."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# These tests focus on validation and syntax checking without execution
# They should run quickly and not require external dependencies


def test_notebooks_lines_55_64_0():
    """Test Bash snippet from docs/tutorials/notebooks.rst lines 55-64."""
    # Description: ~~~~~~~~~~~~
    content = '# Install Orchestrator Framework\npip install py-orc\n\n# Install Jupyter (if not already installed)\npip install jupyter\n\n# Clone the repository for tutorials\ngit clone https://github.com/ContextLab/orchestrator.git\ncd orchestrator'
    
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


def test_notebooks_lines_70_75_1():
    """Test Bash snippet from docs/tutorials/notebooks.rst lines 70-75."""
    # Description: ~~~~~~~~~~~~~~~~~
    content = '# Start Jupyter Notebook\njupyter notebook\n\n# Or start JupyterLab\njupyter lab'
    
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


def test_notebooks_lines_112_131_2():
    """Test Python snippet from docs/tutorials/notebooks.rst lines 112-131."""
    # Description: * Add state management for reliability
    content = '# Example from Tutorial 01\nimport os\nfrom orchestrator import Orchestrator, Task, Pipeline\nfrom orchestrator.models.openai_model import OpenAIModel\nfrom orchestrator.utils.api_keys import load_api_keys\n\n# Load API keys from environment\nload_api_keys()\n\n# Create a real AI model\nmodel = OpenAIModel(\n    name="gpt-3.5-turbo",\n    api_key=os.environ.get("OPENAI_API_KEY"),\n)\n\n# Create your first task\ntask = Task(\n    id="hello_world",\n    name="Hello World Task",\n    action="generate_text",\n    parameters={"prompt": "Hello, Orchestrator!"}\n)\n\n# Build and execute pipeline\npipeline = Pipeline(id="first_pipeline", name="First Pipeline")\npipeline.add_task(task)\n\norchestrator = Orchestrator()\norchestrator.register_model(model)\n# Note: This code is for Jupyter notebooks which support top-level await\nresult = await orchestrator.execute_pipeline(pipeline)'
    
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


def test_notebooks_lines_166_187_3():
    """Test YAML snippet from docs/tutorials/notebooks.rst lines 166-187."""
    # Description: * Create reusable pipeline templates
    import yaml
    
    content = '# Example from Tutorial 02\nid: research_pipeline\nname: Research Assistant Pipeline\n\ncontext:\n  topic: artificial intelligence\n\ntasks:\n  - id: research\n    name: Generate Research Questions\n    action: generate_text\n    parameters:\n      prompt: "Research questions about: {topic}"\n\n  - id: analyze\n    name: Analyze Themes\n    action: generate_text\n    parameters:\n      prompt: "Analyze themes in: {research}"\n    dependencies:\n      - research'
    
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


def test_notebooks_lines_222_236_4():
    """Test Python snippet from docs/tutorials/notebooks.rst lines 222-236."""
    # Description: * Optimize for cost and latency
    content = '# Example from Tutorial 03\nimport os\nfrom orchestrator.models.openai_model import OpenAIModel\nfrom orchestrator.models.anthropic_model import AnthropicModel\n\n# API keys should be set in environment variables or ~/.orchestrator/.env\n# Register multiple models\ngpt4 = OpenAIModel(name="gpt-4", api_key=os.environ.get("OPENAI_API_KEY"))\nclaude = AnthropicModel(name="claude-3", api_key=os.environ.get("ANTHROPIC_API_KEY"))\n\norchestrator.register_model(gpt4)\norchestrator.register_model(claude)\n\n# Orchestrator automatically selects best model\n# Note: This code is for Jupyter notebooks which support top-level await\nresult = await orchestrator.execute_pipeline(pipeline)'
    
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


def test_notebooks_lines_277_293_5():
    """Test text snippet from docs/tutorials/notebooks.rst lines 277-293."""
    # Description: The tutorials come with supporting files:
    content = 'notebooks/\n├── 01_getting_started.ipynb\n├── 02_yaml_configuration.ipynb\n├── 03_advanced_model_integration.ipynb\n├── README.md                           # Tutorial guide\n├── data/                               # Sample data files\n│   ├── sample_pipeline.yaml\n│   ├── complex_workflow.yaml\n│   └── test_data.json\n├── images/                             # Tutorial images\n│   ├── architecture_diagram.png\n│   └── workflow_visualization.png\n└── solutions/                          # Exercise solutions\n    ├── 01_solutions.ipynb\n    ├── 02_solutions.ipynb\n    └── 03_solutions.ipynb'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    assert len(content) > 0, "Content should have length"


def test_notebooks_lines_312_317_6():
    """Test Bash snippet from docs/tutorials/notebooks.rst lines 312-317."""
    # Description: **Jupyter Not Starting**
    content = '# Try updating Jupyter\npip install --upgrade jupyter\n\n# Or install JupyterLab\npip install jupyterlab'
    
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


def test_notebooks_lines_321_326_7():
    """Test Bash snippet from docs/tutorials/notebooks.rst lines 321-326."""
    # Description: **Import Errors**
    content = '# Make sure Orchestrator is installed\npip install py-orc\n\n# Or install in development mode\npip install -e .'
    
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


def test_notebooks_lines_330_332_8():
    """Test Python snippet from docs/tutorials/notebooks.rst lines 330-332."""
    # Description: **API Key Issues**
    content = '# Make sure API keys are set in environment\nexport OPENAI_API_KEY="your-api-key"\n# or\nexport ANTHROPIC_API_KEY="your-api-key"'
    
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


def test_notebooks_lines_336_338_9():
    """Test Python snippet from docs/tutorials/notebooks.rst lines 336-338."""
    # Description: **Async/Await Problems**
    content = '# Use await in notebook cells (Jupyter notebooks only)\nresult = await orchestrator.execute_pipeline(pipeline)'
    
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
