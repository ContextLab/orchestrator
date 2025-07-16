"""Working tests for documentation code snippets - Batch 20."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# These tests focus on validation and syntax checking without execution
# They should run quickly and not require external dependencies


def test_yaml_configuration_lines_26_44_0():
    """Test YAML snippet from docs/user_guide/yaml_configuration.rst lines 26-44."""
    # Description: A basic pipeline YAML file contains:
    import yaml
    
    content = 'id: my_pipeline\nname: My Pipeline\ndescription: A sample pipeline\n\ntasks:\n  - id: task1\n    name: First Task\n    action: generate_text\n    parameters:\n      prompt: "Hello, world!"\n\n  - id: task2\n    name: Second Task\n    action: generate_text\n    parameters:\n      prompt: "Process this: {task1}"\n    dependencies:\n      - task1'
    
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


def test_yaml_configuration_lines_52_65_1():
    """Test YAML snippet from docs/user_guide/yaml_configuration.rst lines 52-65."""
    # Description: Use template variables for dynamic content:
    import yaml
    
    content = 'id: research_pipeline\nname: Research Pipeline\n\ncontext:\n  topic: artificial intelligence\n  depth: detailed\n\ntasks:\n  - id: research\n    name: Research Task\n    action: generate_text\n    parameters:\n      prompt: "Research {topic} with {depth} analysis"'
    
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


def test_yaml_configuration_lines_73_80_2():
    """Test YAML snippet from docs/user_guide/yaml_configuration.rst lines 73-80."""
    # Description: The AUTO tag automatically resolves ambiguous parameters:
    import yaml
    
    content = 'tasks:\n  - id: analysis\n    name: Analysis Task\n    action: <AUTO>\n    parameters:\n      data: {previous_task}\n      model: <AUTO>'
    
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


def test_index_lines_118_128_3():
    """Test Python snippet from docs_sphinx/api/index.rst lines 118-128."""
    # Description: -----------
    content = 'import orchestrator as orc\n\n# Initialize models\nregistry = orc.init_models()\n\n# Compile pipeline\npipeline = orc.compile("my_pipeline.yaml")\n\n# Execute\nresult = pipeline.run(input_param="value")'
    
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


def test_index_lines_134_151_4():
    """Test Python snippet from docs_sphinx/api/index.rst lines 134-151."""
    # Description: --------------
    content = 'from orchestrator import Orchestrator\nfrom orchestrator.core.control_system import MockControlSystem\nfrom orchestrator.models.model_registry import ModelRegistry\n\n# Create custom orchestrator\ncontrol_system = MockControlSystem()\norchestrator = Orchestrator(control_system=control_system)\n\n# Use custom model registry\nregistry = ModelRegistry()\n# ... configure models\n\n# Compile with custom settings\npipeline = orchestrator.compile(\n    yaml_content,\n    config={"timeout": 3600}\n)'
    
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


def test_index_lines_159_168_5():
    """Test Python snippet from docs_sphinx/api/index.rst lines 159-168."""
    # Description: The Orchestrator framework uses comprehensive type annotations for better IDE support and type checking:
    content = 'from typing import Dict, Any, List, Optional\nfrom orchestrator import Pipeline, Task\n\ndef process_pipeline(\n    pipeline: Pipeline,\n    inputs: Dict[str, Any],\n    timeout: Optional[int] = None\n) -> Dict[str, Any]:\n    return pipeline.run(**inputs)'
    
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


def test_index_lines_198_215_6():
    """Test YAML snippet from docs_sphinx/api/index.rst lines 198-215."""
    # Description: Default configuration can be overridden using a config file at ``~/.orchestrator/config.yaml``:
    import yaml
    
    content = 'models:\n  default: "ollama:gemma2:27b"\n  fallback: "ollama:llama3.2:1b"\n  timeout: 300\n\ntools:\n  mcp_port: 8000\n  auto_start: true\n\nexecution:\n  parallel: true\n  checkpoint: true\n  timeout: 3600\n\nlogging:\n  level: "INFO"\n  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"'
    
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


def test_index_lines_226_235_7():
    """Test Python snippet from docs_sphinx/api/index.rst lines 226-235."""
    # Description: Models are loaded lazily and cached. For better performance:
    content = '# Initialize models once at startup\norc.init_models()\n\n# Reuse compiled pipelines\npipeline = orc.compile("pipeline.yaml")\n\n# Multiple executions reuse the same pipeline\nfor inputs in input_batches:\n    result = pipeline.run(**inputs)'
    
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


def test_index_lines_243_253_8():
    """Test Python snippet from docs_sphinx/api/index.rst lines 243-253."""
    # Description: Large pipelines and datasets can consume significant memory:
    content = '# Enable checkpointing for long pipelines\npipeline = orc.compile("pipeline.yaml", config={\n    "checkpoint": True,\n    "memory_limit": "8GB"\n})\n\n# Process data in batches\nfor batch in data_batches:\n    result = pipeline.run(data=batch)\n    # Results are automatically checkpointed'
    
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


def test_index_lines_261_273_9():
    """Test Python snippet from docs_sphinx/api/index.rst lines 261-273."""
    # Description: The framework provides structured error handling:
    content = 'from orchestrator import CompilationError, ExecutionError\n\ntry:\n    pipeline = orc.compile("pipeline.yaml")\n    result = pipeline.run(input="value")\nexcept CompilationError as e:\n    print(f"Pipeline compilation failed: {e}")\n    print(f"Error details: {e.details}")\nexcept ExecutionError as e:\n    print(f"Pipeline execution failed: {e}")\n    print(f"Failed step: {e.step_id}")\n    print(f"Error context: {e.context}")'
    
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
