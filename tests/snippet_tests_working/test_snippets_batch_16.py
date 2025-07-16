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
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                assert line.startswith('pip install'), f"Expected pip install command: {line}"
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
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                assert line.startswith('pip install'), f"Expected pip install command: {line}"
        return  # Skip further validation for pip commands
    
    # For other bash commands, just check they're not empty
    assert len(content.strip()) > 0, "Bash content should not be empty"


def test_notebooks_lines_112_130_2():
    """Test Python snippet from docs/tutorials/notebooks.rst lines 112-130."""
    # Description: * Add state management for reliability
    content = '# Example from Tutorial 01\nfrom orchestrator import Orchestrator, Task, Pipeline\nfrom orchestrator.models.mock_model import MockModel\n\n# Create your first task\ntask = Task(\n    id="hello_world",\n    name="Hello World Task",\n    action="generate_text",\n    parameters={"prompt": "Hello, Orchestrator!"}\n)\n\n# Build and execute pipeline\npipeline = Pipeline(id="first_pipeline", name="First Pipeline")\npipeline.add_task(task)\n\norchestrator = Orchestrator()\nresult = await orchestrator.execute_pipeline(pipeline)'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid Python syntax
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


def test_notebooks_lines_165_186_3():
    """Test YAML snippet from docs/tutorials/notebooks.rst lines 165-186."""
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
        assert data is not None
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                assert 'id' in step, "Each step should have an id"


def test_notebooks_lines_221_234_4():
    """Test Python snippet from docs/tutorials/notebooks.rst lines 221-234."""
    # Description: * Optimize for cost and latency
    content = '# Example from Tutorial 03\nfrom orchestrator.models.openai_model import OpenAIModel\nfrom orchestrator.models.anthropic_model import AnthropicModel\n\n# Register multiple models\ngpt4 = OpenAIModel(name="gpt-4", api_key="your-key")\nclaude = AnthropicModel(name="claude-3", api_key="your-key")\n\norchestrator.register_model(gpt4)\norchestrator.register_model(claude)\n\n# Orchestrator automatically selects best model\nresult = await orchestrator.execute_pipeline(pipeline)'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid Python syntax
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


def test_notebooks_lines_275_291_5():
    """Test text snippet from docs/tutorials/notebooks.rst lines 275-291."""
    # Description: The tutorials come with supporting files:
    content = 'notebooks/\n├── 01_getting_started.ipynb\n├── 02_yaml_configuration.ipynb\n├── 03_advanced_model_integration.ipynb\n├── README.md                           # Tutorial guide\n├── data/                               # Sample data files\n│   ├── sample_pipeline.yaml\n│   ├── complex_workflow.yaml\n│   └── test_data.json\n├── images/                             # Tutorial images\n│   ├── architecture_diagram.png\n│   └── workflow_visualization.png\n└── solutions/                          # Exercise solutions\n    ├── 01_solutions.ipynb\n    ├── 02_solutions.ipynb\n    └── 03_solutions.ipynb'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    assert len(content) > 0, "Content should have length"


def test_notebooks_lines_310_315_6():
    """Test Bash snippet from docs/tutorials/notebooks.rst lines 310-315."""
    # Description: **Jupyter Not Starting**
    content = '# Try updating Jupyter\npip install --upgrade jupyter\n\n# Or install JupyterLab\npip install jupyterlab'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Special handling for pip install commands
    if 'pip install' in content:
        lines = content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                assert line.startswith('pip install'), f"Expected pip install command: {line}"
        return  # Skip further validation for pip commands
    
    # For other bash commands, just check they're not empty
    assert len(content.strip()) > 0, "Bash content should not be empty"


def test_notebooks_lines_319_324_7():
    """Test Python snippet from docs/tutorials/notebooks.rst lines 319-324."""
    # Description: **Import Errors**
    content = '# Make sure Orchestrator is installed\npip install py-orc\n\n# Or install in development mode\npip install -e .'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid Python syntax
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


def test_notebooks_lines_328_330_8():
    """Test Python snippet from docs/tutorials/notebooks.rst lines 328-330."""
    # Description: **Mock Model Issues**
    content = '# Mock models need explicit responses\nmodel.set_response("your prompt", "expected response")'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid Python syntax
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


def test_notebooks_lines_334_336_9():
    """Test Python snippet from docs/tutorials/notebooks.rst lines 334-336."""
    # Description: **Async/Await Problems**
    content = '# Use await in notebook cells\nresult = await orchestrator.execute_pipeline(pipeline)'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid Python syntax
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
