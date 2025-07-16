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


def test_basic_concepts_lines_91_97_2():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 91-97."""
    # Description: * Manages state and checkpointing
    content = 'from orchestrator import Orchestrator\n\norchestrator = Orchestrator()\norchestrator.register_model(model)\n\nresult = await orchestrator.execute_pipeline(pipeline)'
    
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


def test_basic_concepts_lines_105_115_3():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 105-115."""
    # Description: Tasks can depend on other tasks, creating a directed acyclic graph (DAG):
    content = '# Task A (no dependencies)\ntask_a = Task(id="a", name="Task A", action="generate_text")\n\n# Task B depends on A\ntask_b = Task(id="b", name="Task B", action="generate_text",\n              dependencies=["a"])\n\n# Task C depends on A and B\ntask_c = Task(id="c", name="Task C", action="generate_text",\n              dependencies=["a", "b"])'
    
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


def test_basic_concepts_lines_123_126_4():
    """Test text snippet from docs/getting_started/basic_concepts.rst lines 123-126."""
    # Description: The orchestrator automatically determines execution order based on dependencies:
    content = 'Level 0: [Task A]           # No dependencies\nLevel 1: [Task B]           # Depends on A\nLevel 2: [Task C]           # Depends on A and B'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    assert len(content) > 0, "Content should have length"


def test_basic_concepts_lines_136_150_5():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 136-150."""
    # Description: Tasks can reference outputs from other tasks using template syntax:
    content = 'task_a = Task(\n    id="extract",\n    name="Extract Information",\n    action="generate_text",\n    parameters={"prompt": "Extract key facts from: {document}"}\n)\n\ntask_b = Task(\n    id="summarize",\n    name="Summarize Facts",\n    action="generate_text",\n    parameters={"prompt": "Summarize these facts: {extract}"},\n    dependencies=["extract"]\n)'
    
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


def test_basic_concepts_lines_165_171_6():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 165-171."""
    # Description: 6. **Returns** results from all tasks
    content = '# Execute pipeline\nresult = await orchestrator.execute_pipeline(pipeline)\n\n# Access individual task results\nprint(result["extract"])    # Output from extract task\nprint(result["summarize"])  # Output from summarize task'
    
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


def test_basic_concepts_lines_184_191_7():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 184-191."""
    # Description: * **Cost** - Resource usage and API costs
    content = '# Register multiple models\norchestrator.register_model(gpt4_model)\norchestrator.register_model(claude_model)\norchestrator.register_model(local_model)\n\n# Orchestrator will select best model for each task\nresult = await orchestrator.execute_pipeline(pipeline)'
    
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


def test_basic_concepts_lines_202_209_8():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 202-209."""
    # Description: ~~~~~~~~~~~~~~~~
    content = 'from orchestrator.core.error_handler import ErrorHandler\n\nerror_handler = ErrorHandler()\norchestrator = Orchestrator(error_handler=error_handler)\n\n# Tasks will automatically retry on failure\nresult = await orchestrator.execute_pipeline(pipeline)'
    
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


def test_basic_concepts_lines_215_220_9():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 215-220."""
    # Description: ~~~~~~~~~~~~~~~~
    content = '# Circuit breaker prevents cascading failures\nbreaker = error_handler.get_circuit_breaker("openai_api")\n\n# Executes with circuit breaker protection\nresult = await orchestrator.execute_pipeline(pipeline)'
    
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
