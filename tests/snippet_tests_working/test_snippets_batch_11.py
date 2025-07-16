"""Working tests for documentation code snippets - Batch 11."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# These tests focus on validation and syntax checking without execution
# They should run quickly and not require external dependencies


def test_basic_concepts_lines_48_59_0():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 48-59."""
    # Description: A **Pipeline** is a collection of tasks with defined dependencies. It represents your complete workflow:
    content = 'from orchestrator import Pipeline\n\npipeline = Pipeline(\n    id="document_processing",\n    name="Document Processing Pipeline"\n)\n\n# Add tasks to pipeline\npipeline.add_task(extract_task)\npipeline.add_task(summarize_task)\npipeline.add_task(classify_task)'
    
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


def test_basic_concepts_lines_71_78_1():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 71-78."""
    # Description: * **Custom models** (your own implementations)
    content = 'from orchestrator.models import OpenAIModel\n\nmodel = OpenAIModel(\n    name="gpt-4",\n    api_key="your-api-key",\n    model="gpt-4"\n)'
    
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


def test_basic_concepts_lines_91_103_2():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 91-103."""
    # Description: * Manages state and checkpointing
    content = 'import asyncio\nfrom orchestrator import Orchestrator\n\nasync def run_pipeline():\n    orchestrator = Orchestrator()\n    orchestrator.register_model(model)\n\n    result = await orchestrator.execute_pipeline(pipeline)\n    return result\n\n# Run the pipeline\nresult = asyncio.run(run_pipeline())'
    
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


def test_basic_concepts_lines_111_121_3():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 111-121."""
    # Description: Tasks can depend on other tasks, creating a directed acyclic graph (DAG):
    content = '# Task A (no dependencies)\ntask_a = Task(id="a", name="Task A", action="generate_text")\n\n# Task B depends on A\ntask_b = Task(id="b", name="Task B", action="generate_text",\n              dependencies=["a"])\n\n# Task C depends on A and B\ntask_c = Task(id="c", name="Task C", action="generate_text",\n              dependencies=["a", "b"])'
    
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


def test_basic_concepts_lines_129_132_4():
    """Test text snippet from docs/getting_started/basic_concepts.rst lines 129-132."""
    # Description: The orchestrator automatically determines execution order based on dependencies:
    content = 'Level 0: [Task A]           # No dependencies\nLevel 1: [Task B]           # Depends on A\nLevel 2: [Task C]           # Depends on A and B'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    assert len(content) > 0, "Content should have length"


def test_basic_concepts_lines_142_156_5():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 142-156."""
    # Description: Tasks can reference outputs from other tasks using template syntax:
    content = 'task_a = Task(\n    id="extract",\n    name="Extract Information",\n    action="generate_text",\n    parameters={"prompt": "Extract key facts from: {document}"}\n)\n\ntask_b = Task(\n    id="summarize",\n    name="Summarize Facts",\n    action="generate_text",\n    parameters={"prompt": "Summarize these facts: {extract}"},\n    dependencies=["extract"]\n)'
    
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


def test_basic_concepts_lines_171_184_6():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 171-184."""
    # Description: 6. **Returns** results from all tasks
    content = 'import asyncio\n\nasync def execute_and_process():\n    # Execute pipeline\n    result = await orchestrator.execute_pipeline(pipeline)\n\n    # Access individual task results\n    print(result["extract"])    # Output from extract task\n    print(result["summarize"])  # Output from summarize task\n    return result\n\n# Run the execution\nresult = asyncio.run(execute_and_process())'
    
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


def test_basic_concepts_lines_197_211_7():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 197-211."""
    # Description: * **Cost** - Resource usage and API costs
    content = 'import asyncio\n\nasync def run_with_model_selection():\n    # Register multiple models\n    orchestrator.register_model(gpt4_model)\n    orchestrator.register_model(claude_model)\n    orchestrator.register_model(local_model)\n\n    # Orchestrator will select best model for each task\n    result = await orchestrator.execute_pipeline(pipeline)\n    return result\n\n# Run with model selection\nresult = asyncio.run(run_with_model_selection())'
    
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


def test_basic_concepts_lines_222_235_8():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 222-235."""
    # Description: ~~~~~~~~~~~~~~~~
    content = 'import asyncio\nfrom orchestrator.core.error_handler import ErrorHandler\n\nasync def run_with_retry():\n    error_handler = ErrorHandler()\n    orchestrator = Orchestrator(error_handler=error_handler)\n\n    # Tasks will automatically retry on failure\n    result = await orchestrator.execute_pipeline(pipeline)\n    return result\n\n# Run with retry handling\nresult = asyncio.run(run_with_retry())'
    
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


def test_basic_concepts_lines_241_253_9():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 241-253."""
    # Description: ~~~~~~~~~~~~~~~~
    content = 'import asyncio\n\nasync def run_with_circuit_breaker():\n    # Circuit breaker prevents cascading failures\n    breaker = error_handler.get_circuit_breaker("openai_api")\n\n    # Executes with circuit breaker protection\n    result = await orchestrator.execute_pipeline(pipeline)\n    return result\n\n# Run with circuit breaker\nresult = asyncio.run(run_with_circuit_breaker())'
    
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
