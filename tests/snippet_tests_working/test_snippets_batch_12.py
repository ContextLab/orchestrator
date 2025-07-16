"""Working tests for documentation code snippets - Batch 12."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# These tests focus on validation and syntax checking without execution
# They should run quickly and not require external dependencies


def test_basic_concepts_lines_226_232_0():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 226-232."""
    # Description: ~~~~~~~~~~~~~~~
    content = '# Register models in order of preference\norchestrator.register_model(primary_model)\norchestrator.register_model(fallback_model)\n\n# Will use fallback if primary fails\nresult = await orchestrator.execute_pipeline(pipeline)'
    
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


def test_basic_concepts_lines_243_250_1():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 243-250."""
    # Description: ~~~~~~~~~~~~~
    content = 'from orchestrator.state import StateManager\n\nstate_manager = StateManager(storage_path="./checkpoints")\norchestrator = Orchestrator(state_manager=state_manager)\n\n# Automatically saves checkpoints during execution\nresult = await orchestrator.execute_pipeline(pipeline)'
    
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


def test_basic_concepts_lines_256_258_2():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 256-258."""
    # Description: ~~~~~~~~
    content = '# Resume from last checkpoint\nresult = await orchestrator.resume_pipeline("pipeline_id")'
    
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


def test_basic_concepts_lines_266_283_3():
    """Test YAML snippet from docs/getting_started/basic_concepts.rst lines 266-283."""
    # Description: Define pipelines declaratively in YAML:
    import yaml
    
    content = 'id: document_pipeline\nname: Document Processing Pipeline\n\ntasks:\n  - id: extract\n    name: Extract Information\n    action: generate_text\n    parameters:\n      prompt: "Extract key facts from: {document}"\n\n  - id: summarize\n    name: Summarize Facts\n    action: generate_text\n    parameters:\n      prompt: "Summarize these facts: {extract}"\n    dependencies:\n      - extract'
    
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


def test_basic_concepts_lines_288_294_4():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 288-294."""
    # Description: Load and execute:
    content = 'from orchestrator.compiler import YAMLCompiler\n\ncompiler = YAMLCompiler()\npipeline = compiler.compile_file("document_pipeline.yaml")\n\nresult = await orchestrator.execute_pipeline(pipeline)'
    
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


def test_basic_concepts_lines_303_310_5():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 303-310."""
    # Description: ~~~~~~~~~~~~~~~~~~
    content = 'from orchestrator.core.resource_allocator import ResourceAllocator\n\nallocator = ResourceAllocator()\norchestrator = Orchestrator(resource_allocator=allocator)\n\n# Automatically manages CPU, memory, and API quotas\nresult = await orchestrator.execute_pipeline(pipeline)'
    
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


def test_basic_concepts_lines_316_323_6():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 316-323."""
    # Description: ~~~~~~~~~~~~~~~~~~
    content = 'from orchestrator.executor import ParallelExecutor\n\nexecutor = ParallelExecutor(max_workers=4)\norchestrator = Orchestrator(executor=executor)\n\n# Independent tasks run in parallel\nresult = await orchestrator.execute_pipeline(pipeline)'
    
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


def test_basic_concepts_lines_329_336_7():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 329-336."""
    # Description: ~~~~~~~
    content = 'from orchestrator.core.cache import MultiLevelCache\n\ncache = MultiLevelCache()\norchestrator = Orchestrator(cache=cache)\n\n# Results are cached for faster subsequent runs\nresult = await orchestrator.execute_pipeline(pipeline)'
    
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


def test_installation_lines_26_27_8():
    """Test Bash snippet from docs/getting_started/installation.rst lines 26-27."""
    # Description: Install Orchestrator using pip:
    content = 'pip install py-orc'
    
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


def test_installation_lines_37_40_9():
    """Test Bash snippet from docs/getting_started/installation.rst lines 37-40."""
    # Description: To install from source for development:
    content = 'git clone https://github.com/ContextLab/orchestrator.git\ncd orchestrator\npip install -e .'
    
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
