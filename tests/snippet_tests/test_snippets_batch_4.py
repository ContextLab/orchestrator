"""Tests for documentation code snippets - Batch 4."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Set up test environment
os.environ.setdefault('ORCHESTRATOR_CONFIG', str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml"))

# Note: API keys should be set as environment variables or GitHub secrets:
# - OPENAI_API_KEY
# - ANTHROPIC_API_KEY  
# - GOOGLE_AI_API_KEY


@pytest.mark.asyncio
async def test_session_summary_context_limit_lines_45_49_0():
    """Test Python snippet from notes/session_summary_context_limit.md lines 45-49."""
    # Core Framework Structure:
    
    # Import required modules
    import os
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Set up test environment
        os.environ['ORCHESTRATOR_CONFIG'] = str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml")
        
        # Test code snippet
        code = """# Main abstractions in src/orchestrator/core/
- task.py:Task, TaskStatus (lines 1-200+)
- pipeline.py:Pipeline (lines 1-300+)
- model.py:Model, ModelCapabilities (lines 1-250+)
- control_system.py:ControlSystem, ControlAction (lines 1-150+)"""
        
        # Execute with real models (API keys from environment/GitHub secrets)
        try:
            # Check if required API keys are available
            missing_keys = []
            if 'openai' in code.lower() and not os.environ.get('OPENAI_API_KEY'):
                missing_keys.append('OPENAI_API_KEY')
            if 'anthropic' in code.lower() and not os.environ.get('ANTHROPIC_API_KEY'):
                missing_keys.append('ANTHROPIC_API_KEY')
            if ('gemini' in code.lower() or 'google' in code.lower()) and not os.environ.get('GOOGLE_AI_API_KEY'):
                missing_keys.append('GOOGLE_AI_API_KEY')
            
            if missing_keys:
                pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
            
            # Execute the code with real models
            if 'await' in code or 'async' in code:
                # Handle async code
                import asyncio
                exec_globals = {'__name__': '__main__', 'asyncio': asyncio}
                exec(code, exec_globals)
                
                # If there's a main coroutine, run it
                if 'main' in exec_globals and asyncio.iscoroutinefunction(exec_globals['main']):
                    await exec_globals['main']()
            else:
                exec(code, {'__name__': '__main__'})
                
        except Exception as e:
            # Check if it's an expected error
            if "No eligible models" in str(e):
                pytest.skip(f"No eligible models available: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")

@pytest.mark.asyncio
async def test_session_summary_context_limit_lines_54_60_1():
    """Test Python snippet from notes/session_summary_context_limit.md lines 54-60."""
    # Advanced Components:
    
    # Import required modules
    import os
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Set up test environment
        os.environ['ORCHESTRATOR_CONFIG'] = str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml")
        
        # Test code snippet
        code = """# Advanced features in src/orchestrator/
- core/error_handler.py:ErrorHandler, CircuitBreaker (lines 1-400+)
- core/cache.py:MultiLevelCache (lines 1-550+)
- core/resource_allocator.py:ResourceAllocator (lines 1-450+)
- executor/parallel_executor.py:ParallelExecutor (lines 1-425+)
- executor/sandboxed_executor.py:SandboxManager (lines 1-345+)
- state/adaptive_checkpoint.py:AdaptiveCheckpointManager (lines 1-400+)"""
        
        # Execute with real models (API keys from environment/GitHub secrets)
        try:
            # Check if required API keys are available
            missing_keys = []
            if 'openai' in code.lower() and not os.environ.get('OPENAI_API_KEY'):
                missing_keys.append('OPENAI_API_KEY')
            if 'anthropic' in code.lower() and not os.environ.get('ANTHROPIC_API_KEY'):
                missing_keys.append('ANTHROPIC_API_KEY')
            if ('gemini' in code.lower() or 'google' in code.lower()) and not os.environ.get('GOOGLE_AI_API_KEY'):
                missing_keys.append('GOOGLE_AI_API_KEY')
            
            if missing_keys:
                pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
            
            # Execute the code with real models
            if 'await' in code or 'async' in code:
                # Handle async code
                import asyncio
                exec_globals = {'__name__': '__main__', 'asyncio': asyncio}
                exec(code, exec_globals)
                
                # If there's a main coroutine, run it
                if 'main' in exec_globals and asyncio.iscoroutinefunction(exec_globals['main']):
                    await exec_globals['main']()
            else:
                exec(code, {'__name__': '__main__'})
                
        except Exception as e:
            # Check if it's an expected error
            if "No eligible models" in str(e):
                pytest.skip(f"No eligible models available: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")

@pytest.mark.asyncio
async def test_session_summary_context_limit_lines_65_67_2():
    """Test Python snippet from notes/session_summary_context_limit.md lines 65-67."""
    # Control System Adapters:
    
    # Import required modules
    import os
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Set up test environment
        os.environ['ORCHESTRATOR_CONFIG'] = str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml")
        
        # Test code snippet
        code = """# Adapters in src/orchestrator/adapters/
- langgraph_adapter.py:LangGraphAdapter (lines 1-350+)
- mcp_adapter.py:MCPAdapter (lines 1-450+)"""
        
        # Execute with real models (API keys from environment/GitHub secrets)
        try:
            # Check if required API keys are available
            missing_keys = []
            if 'openai' in code.lower() and not os.environ.get('OPENAI_API_KEY'):
                missing_keys.append('OPENAI_API_KEY')
            if 'anthropic' in code.lower() and not os.environ.get('ANTHROPIC_API_KEY'):
                missing_keys.append('ANTHROPIC_API_KEY')
            if ('gemini' in code.lower() or 'google' in code.lower()) and not os.environ.get('GOOGLE_AI_API_KEY'):
                missing_keys.append('GOOGLE_AI_API_KEY')
            
            if missing_keys:
                pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
            
            # Execute the code with real models
            if 'await' in code or 'async' in code:
                # Handle async code
                import asyncio
                exec_globals = {'__name__': '__main__', 'asyncio': asyncio}
                exec(code, exec_globals)
                
                # If there's a main coroutine, run it
                if 'main' in exec_globals and asyncio.iscoroutinefunction(exec_globals['main']):
                    await exec_globals['main']()
            else:
                exec(code, {'__name__': '__main__'})
                
        except Exception as e:
            # Check if it's an expected error
            if "No eligible models" in str(e):
                pytest.skip(f"No eligible models available: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")

def test_compiler_lines_22_26_3():
    """Test Python import from docs/api/compiler.rst lines 22-26."""
    # Test imports
    try:
        exec("""from orchestrator.compiler import YAMLCompiler

compiler = YAMLCompiler()
pipeline = compiler.compile_file("my_pipeline.yaml")""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_core_lines_108_125_4():
    """Test Python import from docs/api/core.rst lines 108-125."""
    # Test imports
    try:
        exec("""from orchestrator import Task, Pipeline, Orchestrator

# Create a task
task = Task(
    id="hello",
    name="Hello Task",
    action="generate_text",
    parameters={"prompt": "Hello, world!"}
)

# Create a pipeline
pipeline = Pipeline(id="demo", name="Demo Pipeline")
pipeline.add_task(task)

# Execute with orchestrator
orchestrator = Orchestrator()
result = await orchestrator.execute_pipeline(pipeline)""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

@pytest.mark.asyncio
async def test_core_lines_131_140_5():
    """Test YAML pipeline from docs/api/core.rst lines 131-140."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """id: demo_pipeline
name: Demo Pipeline

tasks:
  - id: hello
    name: Hello Task
    action: generate_text
    parameters:
      prompt: "Hello, world!""""
    
    # Parse YAML
    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # If it's a pipeline, validate it
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        from orchestrator.compiler import YAMLCompiler
        import orchestrator
        
        # Set up environment
        os.environ['ORCHESTRATOR_CONFIG'] = str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml")
        
        compiler = YAMLCompiler()
        
        # Initialize real models
        try:
            registry = orchestrator.init_models()
            compiler.set_model_registry(registry)
            
            # Check if we have any models available
            if not registry.list_models():
                pytest.skip("No models available for testing")
                
        except Exception as e:
            if "API key" in str(e):
                pytest.skip(f"Missing API keys for real model testing: {e}")
            else:
                raise
        
        # Compile the pipeline
        try:
            pipeline = await compiler.compile(data)
            assert pipeline is not None
            assert pipeline.id
            
            # Validate pipeline structure
            if 'steps' in data:
                assert len(pipeline.tasks) == len(data['steps'])
            
        except Exception as e:
            if "No eligible models" in str(e):
                pytest.skip(f"No eligible models available: {e}")
            else:
                pytest.fail(f"Pipeline compilation failed: {e}")

def test_core_lines_146_153_6():
    """Test Python import from docs/api/core.rst lines 146-153."""
    # Test imports
    try:
        exec("""from orchestrator.core.error_handler import ErrorHandler

error_handler = ErrorHandler()
orchestrator = Orchestrator(error_handler=error_handler)

# Tasks will automatically retry on failure
result = await orchestrator.execute_pipeline(pipeline)""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_github_actions_lines_67_72_7():
    """Test bash snippet from docs/development/github_actions.rst lines 67-72."""
    import subprocess
    import tempfile
    import os
    
    bash_content = r"""# Run tests with coverage
pytest --cov=src/orchestrator --cov-report=term

# The output will show coverage percentage
# Update the README badge URL with the percentage"""
    
    # Skip if it's a command we shouldn't run
    skip_commands = ['rm -rf', 'sudo', 'docker', 'systemctl']
    if any(cmd in bash_content for cmd in skip_commands):
        pytest.skip("Skipping potentially destructive command")
    
    # Check bash syntax
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(bash_content)
            f.flush()
            
            # Check syntax only
            result = subprocess.run(['bash', '-n', f.name], 
                                  capture_output=True, text=True)
            
            os.unlink(f.name)
            
            if result.returncode != 0:
                pytest.fail(f"Bash syntax error: {result.stderr}")
                
    except FileNotFoundError:
        pytest.skip("Bash not available for testing")

def test_github_actions_lines_93_100_8():
    """Test markdown snippet from docs/development/github_actions.rst lines 93-100."""
    # Snippet type 'markdown' not yet supported for testing
    pytest.skip("Snippet type 'markdown' not yet supported")

def test_basic_concepts_lines_28_40_9():
    """Test Python import from docs/getting_started/basic_concepts.rst lines 28-40."""
    # Test imports
    try:
        exec("""from orchestrator import Task

task = Task(
    id="summarize",
    name="Summarize Document",
    action="generate_text",
    parameters={
        "prompt": "Summarize this document: {document}",
        "max_tokens": 150
    },
    dependencies=["extract_document"]
)""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_basic_concepts_lines_48_59_10():
    """Test Python import from docs/getting_started/basic_concepts.rst lines 48-59."""
    # Test imports
    try:
        exec("""from orchestrator import Pipeline

pipeline = Pipeline(
    id="document_processing",
    name="Document Processing Pipeline"
)

# Add tasks to pipeline
pipeline.add_task(extract_task)
pipeline.add_task(summarize_task)
pipeline.add_task(classify_task)""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_basic_concepts_lines_71_78_11():
    """Test Python import from docs/getting_started/basic_concepts.rst lines 71-78."""
    # Test imports
    try:
        exec("""from orchestrator.models import OpenAIModel

model = OpenAIModel(
    name="gpt-4",
    api_key="your-api-key",
    model="gpt-4"
)""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_basic_concepts_lines_91_97_12():
    """Test Python import from docs/getting_started/basic_concepts.rst lines 91-97."""
    # Test imports
    try:
        exec("""from orchestrator import Orchestrator

orchestrator = Orchestrator()
orchestrator.register_model(model)

result = await orchestrator.execute_pipeline(pipeline)""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

@pytest.mark.asyncio
async def test_basic_concepts_lines_105_115_13():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 105-115."""
    # Tasks can depend on other tasks, creating a directed acyclic graph (DAG):
    
    # Import required modules
    import os
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Set up test environment
        os.environ['ORCHESTRATOR_CONFIG'] = str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml")
        
        # Test code snippet
        code = """# Task A (no dependencies)
task_a = Task(id="a", name="Task A", action="generate_text")

# Task B depends on A
task_b = Task(id="b", name="Task B", action="generate_text",
              dependencies=["a"])

# Task C depends on A and B
task_c = Task(id="c", name="Task C", action="generate_text",
              dependencies=["a", "b"])"""
        
        # Execute with real models (API keys from environment/GitHub secrets)
        try:
            # Check if required API keys are available
            missing_keys = []
            if 'openai' in code.lower() and not os.environ.get('OPENAI_API_KEY'):
                missing_keys.append('OPENAI_API_KEY')
            if 'anthropic' in code.lower() and not os.environ.get('ANTHROPIC_API_KEY'):
                missing_keys.append('ANTHROPIC_API_KEY')
            if ('gemini' in code.lower() or 'google' in code.lower()) and not os.environ.get('GOOGLE_AI_API_KEY'):
                missing_keys.append('GOOGLE_AI_API_KEY')
            
            if missing_keys:
                pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
            
            # Execute the code with real models
            if 'await' in code or 'async' in code:
                # Handle async code
                import asyncio
                exec_globals = {'__name__': '__main__', 'asyncio': asyncio}
                exec(code, exec_globals)
                
                # If there's a main coroutine, run it
                if 'main' in exec_globals and asyncio.iscoroutinefunction(exec_globals['main']):
                    await exec_globals['main']()
            else:
                exec(code, {'__name__': '__main__'})
                
        except Exception as e:
            # Check if it's an expected error
            if "No eligible models" in str(e):
                pytest.skip(f"No eligible models available: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")

def test_basic_concepts_lines_123_126_14():
    """Test text snippet from docs/getting_started/basic_concepts.rst lines 123-126."""
    # Snippet type 'text' not yet supported for testing
    pytest.skip("Snippet type 'text' not yet supported")

@pytest.mark.asyncio
async def test_basic_concepts_lines_136_150_15():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 136-150."""
    # Tasks can reference outputs from other tasks using template syntax:
    
    # Import required modules
    import os
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Set up test environment
        os.environ['ORCHESTRATOR_CONFIG'] = str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml")
        
        # Test code snippet
        code = """task_a = Task(
    id="extract",
    name="Extract Information",
    action="generate_text",
    parameters={"prompt": "Extract key facts from: {document}"}
)

task_b = Task(
    id="summarize",
    name="Summarize Facts",
    action="generate_text",
    parameters={"prompt": "Summarize these facts: {extract}"},
    dependencies=["extract"]
)"""
        
        # Execute with real models (API keys from environment/GitHub secrets)
        try:
            # Check if required API keys are available
            missing_keys = []
            if 'openai' in code.lower() and not os.environ.get('OPENAI_API_KEY'):
                missing_keys.append('OPENAI_API_KEY')
            if 'anthropic' in code.lower() and not os.environ.get('ANTHROPIC_API_KEY'):
                missing_keys.append('ANTHROPIC_API_KEY')
            if ('gemini' in code.lower() or 'google' in code.lower()) and not os.environ.get('GOOGLE_AI_API_KEY'):
                missing_keys.append('GOOGLE_AI_API_KEY')
            
            if missing_keys:
                pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
            
            # Execute the code with real models
            if 'await' in code or 'async' in code:
                # Handle async code
                import asyncio
                exec_globals = {'__name__': '__main__', 'asyncio': asyncio}
                exec(code, exec_globals)
                
                # If there's a main coroutine, run it
                if 'main' in exec_globals and asyncio.iscoroutinefunction(exec_globals['main']):
                    await exec_globals['main']()
            else:
                exec(code, {'__name__': '__main__'})
                
        except Exception as e:
            # Check if it's an expected error
            if "No eligible models" in str(e):
                pytest.skip(f"No eligible models available: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")

@pytest.mark.asyncio
async def test_basic_concepts_lines_165_171_16():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 165-171."""
    # 6. **Returns** results from all tasks
    
    # Import required modules
    import os
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Set up test environment
        os.environ['ORCHESTRATOR_CONFIG'] = str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml")
        
        # Test code snippet
        code = """# Execute pipeline
result = await orchestrator.execute_pipeline(pipeline)

# Access individual task results
print(result["extract"])    # Output from extract task
print(result["summarize"])  # Output from summarize task"""
        
        # Execute with real models (API keys from environment/GitHub secrets)
        try:
            # Check if required API keys are available
            missing_keys = []
            if 'openai' in code.lower() and not os.environ.get('OPENAI_API_KEY'):
                missing_keys.append('OPENAI_API_KEY')
            if 'anthropic' in code.lower() and not os.environ.get('ANTHROPIC_API_KEY'):
                missing_keys.append('ANTHROPIC_API_KEY')
            if ('gemini' in code.lower() or 'google' in code.lower()) and not os.environ.get('GOOGLE_AI_API_KEY'):
                missing_keys.append('GOOGLE_AI_API_KEY')
            
            if missing_keys:
                pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
            
            # Execute the code with real models
            if 'await' in code or 'async' in code:
                # Handle async code
                import asyncio
                exec_globals = {'__name__': '__main__', 'asyncio': asyncio}
                exec(code, exec_globals)
                
                # If there's a main coroutine, run it
                if 'main' in exec_globals and asyncio.iscoroutinefunction(exec_globals['main']):
                    await exec_globals['main']()
            else:
                exec(code, {'__name__': '__main__'})
                
        except Exception as e:
            # Check if it's an expected error
            if "No eligible models" in str(e):
                pytest.skip(f"No eligible models available: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")

@pytest.mark.asyncio
async def test_basic_concepts_lines_184_191_17():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 184-191."""
    # * **Cost** - Resource usage and API costs
    
    # Import required modules
    import os
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Set up test environment
        os.environ['ORCHESTRATOR_CONFIG'] = str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml")
        
        # Test code snippet
        code = """# Register multiple models
orchestrator.register_model(gpt4_model)
orchestrator.register_model(claude_model)
orchestrator.register_model(local_model)

# Orchestrator will select best model for each task
result = await orchestrator.execute_pipeline(pipeline)"""
        
        # Execute with real models (API keys from environment/GitHub secrets)
        try:
            # Check if required API keys are available
            missing_keys = []
            if 'openai' in code.lower() and not os.environ.get('OPENAI_API_KEY'):
                missing_keys.append('OPENAI_API_KEY')
            if 'anthropic' in code.lower() and not os.environ.get('ANTHROPIC_API_KEY'):
                missing_keys.append('ANTHROPIC_API_KEY')
            if ('gemini' in code.lower() or 'google' in code.lower()) and not os.environ.get('GOOGLE_AI_API_KEY'):
                missing_keys.append('GOOGLE_AI_API_KEY')
            
            if missing_keys:
                pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
            
            # Execute the code with real models
            if 'await' in code or 'async' in code:
                # Handle async code
                import asyncio
                exec_globals = {'__name__': '__main__', 'asyncio': asyncio}
                exec(code, exec_globals)
                
                # If there's a main coroutine, run it
                if 'main' in exec_globals and asyncio.iscoroutinefunction(exec_globals['main']):
                    await exec_globals['main']()
            else:
                exec(code, {'__name__': '__main__'})
                
        except Exception as e:
            # Check if it's an expected error
            if "No eligible models" in str(e):
                pytest.skip(f"No eligible models available: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")

def test_basic_concepts_lines_202_209_18():
    """Test Python import from docs/getting_started/basic_concepts.rst lines 202-209."""
    # Test imports
    try:
        exec("""from orchestrator.core.error_handler import ErrorHandler

error_handler = ErrorHandler()
orchestrator = Orchestrator(error_handler=error_handler)

# Tasks will automatically retry on failure
result = await orchestrator.execute_pipeline(pipeline)""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

@pytest.mark.asyncio
async def test_basic_concepts_lines_215_220_19():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 215-220."""
    # ~~~~~~~~~~~~~~~~
    
    # Import required modules
    import os
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Set up test environment
        os.environ['ORCHESTRATOR_CONFIG'] = str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml")
        
        # Test code snippet
        code = """# Circuit breaker prevents cascading failures
breaker = error_handler.get_circuit_breaker("openai_api")

# Executes with circuit breaker protection
result = await orchestrator.execute_pipeline(pipeline)"""
        
        # Execute with real models (API keys from environment/GitHub secrets)
        try:
            # Check if required API keys are available
            missing_keys = []
            if 'openai' in code.lower() and not os.environ.get('OPENAI_API_KEY'):
                missing_keys.append('OPENAI_API_KEY')
            if 'anthropic' in code.lower() and not os.environ.get('ANTHROPIC_API_KEY'):
                missing_keys.append('ANTHROPIC_API_KEY')
            if ('gemini' in code.lower() or 'google' in code.lower()) and not os.environ.get('GOOGLE_AI_API_KEY'):
                missing_keys.append('GOOGLE_AI_API_KEY')
            
            if missing_keys:
                pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
            
            # Execute the code with real models
            if 'await' in code or 'async' in code:
                # Handle async code
                import asyncio
                exec_globals = {'__name__': '__main__', 'asyncio': asyncio}
                exec(code, exec_globals)
                
                # If there's a main coroutine, run it
                if 'main' in exec_globals and asyncio.iscoroutinefunction(exec_globals['main']):
                    await exec_globals['main']()
            else:
                exec(code, {'__name__': '__main__'})
                
        except Exception as e:
            # Check if it's an expected error
            if "No eligible models" in str(e):
                pytest.skip(f"No eligible models available: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")

@pytest.mark.asyncio
async def test_basic_concepts_lines_226_232_20():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 226-232."""
    # ~~~~~~~~~~~~~~~
    
    # Import required modules
    import os
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Set up test environment
        os.environ['ORCHESTRATOR_CONFIG'] = str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml")
        
        # Test code snippet
        code = """# Register models in order of preference
orchestrator.register_model(primary_model)
orchestrator.register_model(fallback_model)

# Will use fallback if primary fails
result = await orchestrator.execute_pipeline(pipeline)"""
        
        # Execute with real models (API keys from environment/GitHub secrets)
        try:
            # Check if required API keys are available
            missing_keys = []
            if 'openai' in code.lower() and not os.environ.get('OPENAI_API_KEY'):
                missing_keys.append('OPENAI_API_KEY')
            if 'anthropic' in code.lower() and not os.environ.get('ANTHROPIC_API_KEY'):
                missing_keys.append('ANTHROPIC_API_KEY')
            if ('gemini' in code.lower() or 'google' in code.lower()) and not os.environ.get('GOOGLE_AI_API_KEY'):
                missing_keys.append('GOOGLE_AI_API_KEY')
            
            if missing_keys:
                pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
            
            # Execute the code with real models
            if 'await' in code or 'async' in code:
                # Handle async code
                import asyncio
                exec_globals = {'__name__': '__main__', 'asyncio': asyncio}
                exec(code, exec_globals)
                
                # If there's a main coroutine, run it
                if 'main' in exec_globals and asyncio.iscoroutinefunction(exec_globals['main']):
                    await exec_globals['main']()
            else:
                exec(code, {'__name__': '__main__'})
                
        except Exception as e:
            # Check if it's an expected error
            if "No eligible models" in str(e):
                pytest.skip(f"No eligible models available: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")

def test_basic_concepts_lines_243_250_21():
    """Test Python import from docs/getting_started/basic_concepts.rst lines 243-250."""
    # Test imports
    try:
        exec("""from orchestrator.state import StateManager

state_manager = StateManager(storage_path="./checkpoints")
orchestrator = Orchestrator(state_manager=state_manager)

# Automatically saves checkpoints during execution
result = await orchestrator.execute_pipeline(pipeline)""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

@pytest.mark.asyncio
async def test_basic_concepts_lines_256_258_22():
    """Test Python snippet from docs/getting_started/basic_concepts.rst lines 256-258."""
    # ~~~~~~~~
    
    # Import required modules
    import os
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Set up test environment
        os.environ['ORCHESTRATOR_CONFIG'] = str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml")
        
        # Test code snippet
        code = """# Resume from last checkpoint
result = await orchestrator.resume_pipeline("pipeline_id")"""
        
        # Execute with real models (API keys from environment/GitHub secrets)
        try:
            # Check if required API keys are available
            missing_keys = []
            if 'openai' in code.lower() and not os.environ.get('OPENAI_API_KEY'):
                missing_keys.append('OPENAI_API_KEY')
            if 'anthropic' in code.lower() and not os.environ.get('ANTHROPIC_API_KEY'):
                missing_keys.append('ANTHROPIC_API_KEY')
            if ('gemini' in code.lower() or 'google' in code.lower()) and not os.environ.get('GOOGLE_AI_API_KEY'):
                missing_keys.append('GOOGLE_AI_API_KEY')
            
            if missing_keys:
                pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
            
            # Execute the code with real models
            if 'await' in code or 'async' in code:
                # Handle async code
                import asyncio
                exec_globals = {'__name__': '__main__', 'asyncio': asyncio}
                exec(code, exec_globals)
                
                # If there's a main coroutine, run it
                if 'main' in exec_globals and asyncio.iscoroutinefunction(exec_globals['main']):
                    await exec_globals['main']()
            else:
                exec(code, {'__name__': '__main__'})
                
        except Exception as e:
            # Check if it's an expected error
            if "No eligible models" in str(e):
                pytest.skip(f"No eligible models available: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")

@pytest.mark.asyncio
async def test_basic_concepts_lines_266_283_23():
    """Test YAML pipeline from docs/getting_started/basic_concepts.rst lines 266-283."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """id: document_pipeline
name: Document Processing Pipeline

tasks:
  - id: extract
    name: Extract Information
    action: generate_text
    parameters:
      prompt: "Extract key facts from: {document}"

  - id: summarize
    name: Summarize Facts
    action: generate_text
    parameters:
      prompt: "Summarize these facts: {extract}"
    dependencies:
      - extract"""
    
    # Parse YAML
    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # If it's a pipeline, validate it
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        from orchestrator.compiler import YAMLCompiler
        import orchestrator
        
        # Set up environment
        os.environ['ORCHESTRATOR_CONFIG'] = str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml")
        
        compiler = YAMLCompiler()
        
        # Initialize real models
        try:
            registry = orchestrator.init_models()
            compiler.set_model_registry(registry)
            
            # Check if we have any models available
            if not registry.list_models():
                pytest.skip("No models available for testing")
                
        except Exception as e:
            if "API key" in str(e):
                pytest.skip(f"Missing API keys for real model testing: {e}")
            else:
                raise
        
        # Compile the pipeline
        try:
            pipeline = await compiler.compile(data)
            assert pipeline is not None
            assert pipeline.id
            
            # Validate pipeline structure
            if 'steps' in data:
                assert len(pipeline.tasks) == len(data['steps'])
            
        except Exception as e:
            if "No eligible models" in str(e):
                pytest.skip(f"No eligible models available: {e}")
            else:
                pytest.fail(f"Pipeline compilation failed: {e}")

def test_basic_concepts_lines_288_294_24():
    """Test Python import from docs/getting_started/basic_concepts.rst lines 288-294."""
    # Test imports
    try:
        exec("""from orchestrator.compiler import YAMLCompiler

compiler = YAMLCompiler()
pipeline = compiler.compile_file("document_pipeline.yaml")

result = await orchestrator.execute_pipeline(pipeline)""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_basic_concepts_lines_303_310_25():
    """Test Python import from docs/getting_started/basic_concepts.rst lines 303-310."""
    # Test imports
    try:
        exec("""from orchestrator.core.resource_allocator import ResourceAllocator

allocator = ResourceAllocator()
orchestrator = Orchestrator(resource_allocator=allocator)

# Automatically manages CPU, memory, and API quotas
result = await orchestrator.execute_pipeline(pipeline)""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_basic_concepts_lines_316_323_26():
    """Test Python import from docs/getting_started/basic_concepts.rst lines 316-323."""
    # Test imports
    try:
        exec("""from orchestrator.executor import ParallelExecutor

executor = ParallelExecutor(max_workers=4)
orchestrator = Orchestrator(executor=executor)

# Independent tasks run in parallel
result = await orchestrator.execute_pipeline(pipeline)""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_basic_concepts_lines_329_336_27():
    """Test Python import from docs/getting_started/basic_concepts.rst lines 329-336."""
    # Test imports
    try:
        exec("""from orchestrator.core.cache import MultiLevelCache

cache = MultiLevelCache()
orchestrator = Orchestrator(cache=cache)

# Results are cached for faster subsequent runs
result = await orchestrator.execute_pipeline(pipeline)""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_installation_lines_26_27_28():
    """Test bash snippet from docs/getting_started/installation.rst lines 26-27."""
    # Bash command snippet
    snippet_bash = r"""pip install py-orc"""
    
    # Don't actually install packages in tests
    assert "pip install" in snippet_bash
    
    # Verify it's a valid pip command structure
    lines = snippet_bash.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {line}"

def test_installation_lines_37_40_29():
    """Test bash snippet from docs/getting_started/installation.rst lines 37-40."""
    # Bash command snippet
    snippet_bash = r"""git clone https://github.com/ContextLab/orchestrator.git
cd orchestrator
pip install -e ."""
    
    # Don't actually install packages in tests
    assert "pip install" in snippet_bash
    
    # Verify it's a valid pip command structure
    lines = snippet_bash.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {line}"
