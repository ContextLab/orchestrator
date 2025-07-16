"""Working tests for documentation code snippets - Batch 12."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# These tests focus on validation and syntax checking without execution
# They should run quickly and not require external dependencies


def test_basic_concepts_lines_259_272_0():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 259-272."""
    # Description: ~~~~~~~~~~~~~~~
    content = 'import asyncio\n\nasync def run_with_fallback():\n    # Register models in order of preference\n    orchestrator.register_model(primary_model)\n    orchestrator.register_model(fallback_model)\n\n    # Will use fallback if primary fails\n    result = await orchestrator.execute_pipeline(pipeline)\n    return result\n\n# Run with fallback support\nresult = asyncio.run(run_with_fallback())'
    
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


def test_basic_concepts_lines_283_296_1():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 283-296."""
    # Description: ~~~~~~~~~~~~~
    content = 'import asyncio\nfrom orchestrator.state import StateManager\n\nasync def run_with_checkpointing():\n    state_manager = StateManager(storage_path="./checkpoints")\n    orchestrator = Orchestrator(state_manager=state_manager)\n\n    # Automatically saves checkpoints during execution\n    result = await orchestrator.execute_pipeline(pipeline)\n    return result\n\n# Run with checkpointing\nresult = asyncio.run(run_with_checkpointing())'
    
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


def test_basic_concepts_lines_302_311_2():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 302-311."""
    # Description: ~~~~~~~~
    content = 'import asyncio\n\nasync def resume_from_checkpoint():\n    # Resume from last checkpoint\n    result = await orchestrator.resume_pipeline("pipeline_id")\n    return result\n\n# Resume execution\nresult = asyncio.run(resume_from_checkpoint())'
    
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


def test_basic_concepts_lines_319_336_3():
    """Test YAML snippet from docs/getting_started/basic_concepts.rst lines 319-336."""
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


def test_basic_concepts_lines_341_353_4():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 341-353."""
    # Description: Load and execute:
    content = 'import asyncio\nfrom orchestrator.compiler import YAMLCompiler\n\nasync def run_yaml_pipeline():\n    compiler = YAMLCompiler()\n    pipeline = compiler.compile_file("document_pipeline.yaml")\n\n    result = await orchestrator.execute_pipeline(pipeline)\n    return result\n\n# Run YAML pipeline\nresult = asyncio.run(run_yaml_pipeline())'
    
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


def test_basic_concepts_lines_362_375_5():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 362-375."""
    # Description: ~~~~~~~~~~~~~~~~~~
    content = 'import asyncio\nfrom orchestrator.core.resource_allocator import ResourceAllocator\n\nasync def run_with_resource_management():\n    allocator = ResourceAllocator()\n    orchestrator = Orchestrator(resource_allocator=allocator)\n\n    # Automatically manages CPU, memory, and API quotas\n    result = await orchestrator.execute_pipeline(pipeline)\n    return result\n\n# Run with resource management\nresult = asyncio.run(run_with_resource_management())'
    
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


def test_basic_concepts_lines_381_394_6():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 381-394."""
    # Description: ~~~~~~~~~~~~~~~~~~
    content = 'import asyncio\nfrom orchestrator.executor import ParallelExecutor\n\nasync def run_parallel_execution():\n    executor = ParallelExecutor(max_workers=4)\n    orchestrator = Orchestrator(executor=executor)\n\n    # Independent tasks run in parallel\n    result = await orchestrator.execute_pipeline(pipeline)\n    return result\n\n# Run with parallel execution\nresult = asyncio.run(run_parallel_execution())'
    
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


def test_basic_concepts_lines_400_413_7():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 400-413."""
    # Description: ~~~~~~~
    content = 'import asyncio\nfrom orchestrator.core.cache import MultiLevelCache\n\nasync def run_with_caching():\n    cache = MultiLevelCache()\n    orchestrator = Orchestrator(cache=cache)\n\n    # Results are cached for faster subsequent runs\n    result = await orchestrator.execute_pipeline(pipeline)\n    return result\n\n# Run with caching\nresult = asyncio.run(run_with_caching())'
    
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


def test_installation_lines_26_27_8():
    """Test Bash snippet from docs/getting_started/installation.rst lines 26-27."""
    # Description: Install Orchestrator using pip:
    content = 'pip install py-orc'
    
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


def test_installation_lines_37_40_9():
    """Test Bash snippet from docs/getting_started/installation.rst lines 37-40."""
    # Description: To install from source for development:
    content = 'git clone https://github.com/ContextLab/orchestrator.git\ncd orchestrator\npip install -e .'
    
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
