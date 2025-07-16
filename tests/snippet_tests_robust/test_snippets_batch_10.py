"""Tests for documentation code snippets - Batch 10 (Robust)."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Set up test environment
os.environ.setdefault('ORCHESTRATOR_CONFIG', str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml"))

# Note: Set RUN_REAL_TESTS=1 to enable tests that use real models
# API keys should be set as environment variables:
# - OPENAI_API_KEY
# - ANTHROPIC_API_KEY  
# - GOOGLE_AI_API_KEY


@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_quickstart_lines_184_194_0():
    """Test orchestrator code from docs/getting_started/quickstart.rst lines 184-194."""
    # Enable checkpointing for long-running pipelines:
    
    # Check for required API keys
    missing_keys = []
    if not os.environ.get('OPENAI_API_KEY'):
        missing_keys.append('OPENAI_API_KEY')
    if not os.environ.get('ANTHROPIC_API_KEY'):
        missing_keys.append('ANTHROPIC_API_KEY')
    
    if missing_keys:
        pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
    
    # Set up test environment
    import tempfile
    import orchestrator as orc
    
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Create a minimal test pipeline file if referenced
        code = ("""from orchestrator.state import StateManager

# Create state manager
state_manager = StateManager(storage_path="./checkpoints")

# Configure orchestrator with state management
orchestrator = Orchestrator(state_manager=state_manager)

# Execute pipeline with automatic checkpointing
result = await orchestrator.execute_pipeline(pipeline)""")
        if 'hello_world.yaml' in code:
            with open('hello_world.yaml', 'w') as f:
                f.write("""
id: hello_world
name: Hello World
steps:
  - id: greet
    action: generate_text
    parameters:
      prompt: "Say hello world"
outputs:
  greeting: "{{ greet.result }}"
""".strip())
        
        # Mock the compile function to avoid real file dependencies
        original_compile = getattr(orc, 'compile', None)
        
        def mock_compile(filename):
            # Return a mock pipeline object
            class MockPipeline:
                def run(self):
                    return {"greeting": "Hello World"}
            return MockPipeline()
        
        # Test the code with mocked dependencies
        try:
            # Replace compile calls temporarily
            if original_compile:
                setattr(orc, 'compile_mock', mock_compile)
                setattr(orc, 'compile', mock_compile)
            
            # Execute the code
            exec(code, {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")
        finally:
            # Restore original function
            if original_compile:
                setattr(orc, 'compile', original_compile)

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_quickstart_lines_202_214_1():
    """Test orchestrator code from docs/getting_started/quickstart.rst lines 202-214."""
    # Enable monitoring to track pipeline execution:
    
    # Check for required API keys
    missing_keys = []
    if not os.environ.get('OPENAI_API_KEY'):
        missing_keys.append('OPENAI_API_KEY')
    if not os.environ.get('ANTHROPIC_API_KEY'):
        missing_keys.append('ANTHROPIC_API_KEY')
    
    if missing_keys:
        pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
    
    # Set up test environment
    import tempfile
    import orchestrator as orc
    
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Create a minimal test pipeline file if referenced
        code = ("""import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Execute pipeline with logging
result = await orchestrator.execute_pipeline(pipeline)

# Get execution statistics
stats = orchestrator.get_execution_stats()
print(f"Execution time: {stats['total_time']:.2f}s")
print(f"Tasks completed: {stats['completed_tasks']}")""")
        if 'hello_world.yaml' in code:
            with open('hello_world.yaml', 'w') as f:
                f.write("""
id: hello_world
name: Hello World
steps:
  - id: greet
    action: generate_text
    parameters:
      prompt: "Say hello world"
outputs:
  greeting: "{{ greet.result }}"
""".strip())
        
        # Mock the compile function to avoid real file dependencies
        original_compile = getattr(orc, 'compile', None)
        
        def mock_compile(filename):
            # Return a mock pipeline object
            class MockPipeline:
                def run(self):
                    return {"greeting": "Hello World"}
            return MockPipeline()
        
        # Test the code with mocked dependencies
        try:
            # Replace compile calls temporarily
            if original_compile:
                setattr(orc, 'compile_mock', mock_compile)
                setattr(orc, 'compile', mock_compile)
            
            # Execute the code
            exec(code, {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")
        finally:
            # Restore original function
            if original_compile:
                setattr(orc, 'compile', original_compile)

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_your_first_pipeline_lines_24_46_2():
    """Test orchestrator code from docs/getting_started/your_first_pipeline.rst lines 24-46."""
    # First, let's set up our environment:
    
    # Check for required API keys
    missing_keys = []
    if not os.environ.get('OPENAI_API_KEY'):
        missing_keys.append('OPENAI_API_KEY')
    if not os.environ.get('ANTHROPIC_API_KEY'):
        missing_keys.append('ANTHROPIC_API_KEY')
    
    if missing_keys:
        pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
    
    # Set up test environment
    import tempfile
    import orchestrator as orc
    
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Create a minimal test pipeline file if referenced
        code = ("""import asyncio
from orchestrator import Orchestrator, Task, Pipeline
from orchestrator.models.mock_model import MockModel

# Create a mock model for testing
model = MockModel("research_assistant")

# Set up responses for our mock model
model.set_response(
    "Generate 3 research questions about: artificial intelligence",
    "1. How does AI impact job markets?\n2. What are the ethical implications of AI?\n3. How can AI be made more accessible?"
)

model.set_response(
    "Analyze these questions and identify key themes: 1. How does AI impact job markets?\n2. What are the ethical implications of AI?\n3. How can AI be made more accessible?",
    "Key themes identified: Economic Impact, Ethics and Responsibility, Accessibility and Democratization"
)

model.set_response(
    "Write a comprehensive report on artificial intelligence covering these themes: Economic Impact, Ethics and Responsibility, Accessibility and Democratization",
    "# AI Research Report\n\n## Economic Impact\nAI is reshaping job markets...\n\n## Ethics and Responsibility\nAI systems must be developed responsibly...\n\n## Accessibility and Democratization\nMaking AI tools accessible to all..."
)""")
        if 'hello_world.yaml' in code:
            with open('hello_world.yaml', 'w') as f:
                f.write("""
id: hello_world
name: Hello World
steps:
  - id: greet
    action: generate_text
    parameters:
      prompt: "Say hello world"
outputs:
  greeting: "{{ greet.result }}"
""".strip())
        
        # Mock the compile function to avoid real file dependencies
        original_compile = getattr(orc, 'compile', None)
        
        def mock_compile(filename):
            # Return a mock pipeline object
            class MockPipeline:
                def run(self):
                    return {"greeting": "Hello World"}
            return MockPipeline()
        
        # Test the code with mocked dependencies
        try:
            # Replace compile calls temporarily
            if original_compile:
                setattr(orc, 'compile_mock', mock_compile)
                setattr(orc, 'compile', mock_compile)
            
            # Execute the code
            exec(code, {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")
        finally:
            # Restore original function
            if original_compile:
                setattr(orc, 'compile', original_compile)

def test_your_first_pipeline_lines_54_88_3():
    """Test Python snippet from docs/getting_started/your_first_pipeline.rst lines 54-88."""
    # Now let's create our three tasks:
    
    code = ("""# Task 1: Generate research questions
research_task = Task(
    id="research_questions",
    name="Generate Research Questions",
    action="generate_text",
    parameters={
        "prompt": "Generate 3 research questions about: {topic}",
        "max_tokens": 200
    }
)

# Task 2: Analyze questions for themes
analysis_task = Task(
    id="analyze_themes",
    name="Analyze Key Themes",
    action="generate_text",
    parameters={
        "prompt": "Analyze these questions and identify key themes: {research_questions}",
        "max_tokens": 150
    },
    dependencies=["research_questions"]  # Depends on research task
)

# Task 3: Write comprehensive report
report_task = Task(
    id="write_report",
    name="Write Research Report",
    action="generate_text",
    parameters={
        "prompt": "Write a comprehensive report on {topic} covering these themes: {analyze_themes}",
        "max_tokens": 500
    },
    dependencies=["analyze_themes"]  # Depends on analysis task
)""")
    
    try:
        exec(code, {'__name__': '__main__'})
    except Exception as e:
        pytest.fail(f"Code execution failed: {e}")

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_your_first_pipeline_lines_96_114_4():
    """Test orchestrator code from docs/getting_started/your_first_pipeline.rst lines 96-114."""
    # Combine tasks into a pipeline:
    
    # Check for required API keys
    missing_keys = []
    if not os.environ.get('OPENAI_API_KEY'):
        missing_keys.append('OPENAI_API_KEY')
    if not os.environ.get('ANTHROPIC_API_KEY'):
        missing_keys.append('ANTHROPIC_API_KEY')
    
    if missing_keys:
        pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
    
    # Set up test environment
    import tempfile
    import orchestrator as orc
    
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Create a minimal test pipeline file if referenced
        code = ("""# Create pipeline
pipeline = Pipeline(
    id="research_assistant",
    name="Research Assistant Pipeline",
    description="Generates research questions, analyzes themes, and writes a report"
)

# Add tasks to pipeline
pipeline.add_task(research_task)
pipeline.add_task(analysis_task)
pipeline.add_task(report_task)

# Set initial context
pipeline.set_context("topic", "artificial intelligence")

print("Pipeline created successfully!")
print(f"Tasks: {list(pipeline.tasks.keys())}")
print(f"Execution order: {pipeline.get_execution_order()}")""")
        if 'hello_world.yaml' in code:
            with open('hello_world.yaml', 'w') as f:
                f.write("""
id: hello_world
name: Hello World
steps:
  - id: greet
    action: generate_text
    parameters:
      prompt: "Say hello world"
outputs:
  greeting: "{{ greet.result }}"
""".strip())
        
        # Mock the compile function to avoid real file dependencies
        original_compile = getattr(orc, 'compile', None)
        
        def mock_compile(filename):
            # Return a mock pipeline object
            class MockPipeline:
                def run(self):
                    return {"greeting": "Hello World"}
            return MockPipeline()
        
        # Test the code with mocked dependencies
        try:
            # Replace compile calls temporarily
            if original_compile:
                setattr(orc, 'compile_mock', mock_compile)
                setattr(orc, 'compile', mock_compile)
            
            # Execute the code
            exec(code, {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")
        finally:
            # Restore original function
            if original_compile:
                setattr(orc, 'compile', original_compile)

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_your_first_pipeline_lines_122_143_5():
    """Test orchestrator code from docs/getting_started/your_first_pipeline.rst lines 122-143."""
    # Now let's execute our pipeline:
    
    # Check for required API keys
    missing_keys = []
    if not os.environ.get('OPENAI_API_KEY'):
        missing_keys.append('OPENAI_API_KEY')
    if not os.environ.get('ANTHROPIC_API_KEY'):
        missing_keys.append('ANTHROPIC_API_KEY')
    
    if missing_keys:
        pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
    
    # Set up test environment
    import tempfile
    import orchestrator as orc
    
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Create a minimal test pipeline file if referenced
        code = ("""async def run_pipeline():
    # Create orchestrator
    orchestrator = Orchestrator()

    # Register our model
    orchestrator.register_model(model)

    print("Starting pipeline execution...")

    # Execute pipeline
    result = await orchestrator.execute_pipeline(pipeline)

    print("\n=== Pipeline Results ===")
    print(f"Research Questions:\n{result['research_questions']}\n")
    print(f"Key Themes:\n{result['analyze_themes']}\n")
    print(f"Final Report:\n{result['write_report']}\n")

    return result

# Run the pipeline
result = await run_pipeline()""")
        if 'hello_world.yaml' in code:
            with open('hello_world.yaml', 'w') as f:
                f.write("""
id: hello_world
name: Hello World
steps:
  - id: greet
    action: generate_text
    parameters:
      prompt: "Say hello world"
outputs:
  greeting: "{{ greet.result }}"
""".strip())
        
        # Mock the compile function to avoid real file dependencies
        original_compile = getattr(orc, 'compile', None)
        
        def mock_compile(filename):
            # Return a mock pipeline object
            class MockPipeline:
                def run(self):
                    return {"greeting": "Hello World"}
            return MockPipeline()
        
        # Test the code with mocked dependencies
        try:
            # Replace compile calls temporarily
            if original_compile:
                setattr(orc, 'compile_mock', mock_compile)
                setattr(orc, 'compile', mock_compile)
            
            # Execute the code
            exec(code, {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")
        finally:
            # Restore original function
            if original_compile:
                setattr(orc, 'compile', original_compile)

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_your_first_pipeline_lines_151_181_6():
    """Test orchestrator code from docs/getting_started/your_first_pipeline.rst lines 151-181."""
    # Let's make our pipeline more robust:
    
    # Check for required API keys
    missing_keys = []
    if not os.environ.get('OPENAI_API_KEY'):
        missing_keys.append('OPENAI_API_KEY')
    if not os.environ.get('ANTHROPIC_API_KEY'):
        missing_keys.append('ANTHROPIC_API_KEY')
    
    if missing_keys:
        pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
    
    # Set up test environment
    import tempfile
    import orchestrator as orc
    
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Create a minimal test pipeline file if referenced
        code = ("""from orchestrator.core.error_handler import ErrorHandler
from orchestrator.core.error_handler import ExponentialBackoffRetry

async def run_robust_pipeline():
    # Create error handler with retry strategy
    error_handler = ErrorHandler()
    error_handler.register_retry_strategy(
        "research_retry",
        ExponentialBackoffRetry(max_retries=3, base_delay=1.0)
    )

    # Create orchestrator with error handling
    orchestrator = Orchestrator(error_handler=error_handler)
    orchestrator.register_model(model)

    try:
        print("Starting robust pipeline execution...")
        result = await orchestrator.execute_pipeline(pipeline)
        print("✅ Pipeline completed successfully!")
        return result

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        # Get execution statistics
        stats = error_handler.get_error_statistics()
        print(f"Errors encountered: {stats['total_errors']}")
        return None

# Run robust pipeline
result = await run_robust_pipeline()""")
        if 'hello_world.yaml' in code:
            with open('hello_world.yaml', 'w') as f:
                f.write("""
id: hello_world
name: Hello World
steps:
  - id: greet
    action: generate_text
    parameters:
      prompt: "Say hello world"
outputs:
  greeting: "{{ greet.result }}"
""".strip())
        
        # Mock the compile function to avoid real file dependencies
        original_compile = getattr(orc, 'compile', None)
        
        def mock_compile(filename):
            # Return a mock pipeline object
            class MockPipeline:
                def run(self):
                    return {"greeting": "Hello World"}
            return MockPipeline()
        
        # Test the code with mocked dependencies
        try:
            # Replace compile calls temporarily
            if original_compile:
                setattr(orc, 'compile_mock', mock_compile)
                setattr(orc, 'compile', mock_compile)
            
            # Execute the code
            exec(code, {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")
        finally:
            # Restore original function
            if original_compile:
                setattr(orc, 'compile', original_compile)

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_your_first_pipeline_lines_189_214_7():
    """Test orchestrator code from docs/getting_started/your_first_pipeline.rst lines 189-214."""
    # For longer pipelines, add checkpointing:
    
    # Check for required API keys
    missing_keys = []
    if not os.environ.get('OPENAI_API_KEY'):
        missing_keys.append('OPENAI_API_KEY')
    if not os.environ.get('ANTHROPIC_API_KEY'):
        missing_keys.append('ANTHROPIC_API_KEY')
    
    if missing_keys:
        pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
    
    # Set up test environment
    import tempfile
    import orchestrator as orc
    
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Create a minimal test pipeline file if referenced
        code = ("""from orchestrator.state import StateManager

async def run_stateful_pipeline():
    # Create state manager
    state_manager = StateManager(storage_path="./checkpoints")

    # Create orchestrator with state management
    orchestrator = Orchestrator(state_manager=state_manager)
    orchestrator.register_model(model)

    print("Starting stateful pipeline execution...")

    # Execute with automatic checkpointing
    result = await orchestrator.execute_pipeline(pipeline)

    print("✅ Pipeline completed with checkpointing!")

    # List checkpoints created
    checkpoints = await state_manager.list_checkpoints("research_assistant")
    print(f"Checkpoints created: {len(checkpoints)}")

    return result

# Run stateful pipeline
result = await run_stateful_pipeline()""")
        if 'hello_world.yaml' in code:
            with open('hello_world.yaml', 'w') as f:
                f.write("""
id: hello_world
name: Hello World
steps:
  - id: greet
    action: generate_text
    parameters:
      prompt: "Say hello world"
outputs:
  greeting: "{{ greet.result }}"
""".strip())
        
        # Mock the compile function to avoid real file dependencies
        original_compile = getattr(orc, 'compile', None)
        
        def mock_compile(filename):
            # Return a mock pipeline object
            class MockPipeline:
                def run(self):
                    return {"greeting": "Hello World"}
            return MockPipeline()
        
        # Test the code with mocked dependencies
        try:
            # Replace compile calls temporarily
            if original_compile:
                setattr(orc, 'compile_mock', mock_compile)
                setattr(orc, 'compile', mock_compile)
            
            # Execute the code
            exec(code, {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")
        finally:
            # Restore original function
            if original_compile:
                setattr(orc, 'compile', original_compile)

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_your_first_pipeline_lines_222_255_8():
    """Test YAML pipeline from docs/getting_started/your_first_pipeline.rst lines 222-255."""
    import yaml
    import orchestrator
    
    yaml_content = ("""# research_pipeline.yaml
id: research_assistant
name: Research Assistant Pipeline
description: Generates research questions, analyzes themes, and writes a report

context:
  topic: artificial intelligence

tasks:
  - id: research_questions
    name: Generate Research Questions
    action: generate_text
    parameters:
      prompt: "Generate 3 research questions about: {topic}"
      max_tokens: 200

  - id: analyze_themes
    name: Analyze Key Themes
    action: generate_text
    parameters:
      prompt: "Analyze these questions and identify key themes: {research_questions}"
      max_tokens: 150
    dependencies:
      - research_questions

  - id: write_report
    name: Write Research Report
    action: generate_text
    parameters:
      prompt: "Write a comprehensive report on {topic} covering these themes: {analyze_themes}"
      max_tokens: 500
    dependencies:
      - analyze_themes""")
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip actual pipeline compilation unless we have real models
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        # Check for API keys
        if not (os.environ.get('OPENAI_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')):
            pytest.skip("No API keys available for pipeline testing")
        
        try:
            # Initialize models
            registry = orchestrator.init_models()
            
            # Check if we have any models available
            if not registry.list_models():
                pytest.skip("No models available for testing")
            
            # Create compiler
            from orchestrator.compiler import YAMLCompiler
            compiler = YAMLCompiler()
            compiler.set_model_registry(registry)
            
            # Compile the pipeline (but don't run it)
            pipeline = await compiler.compile(data)
            assert pipeline is not None
            assert pipeline.id
            
            # Validate pipeline structure
            if 'steps' in data:
                assert len(pipeline.tasks) == len(data['steps'])
                
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Pipeline compilation failed: {e}")

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_your_first_pipeline_lines_260_281_9():
    """Test orchestrator code from docs/getting_started/your_first_pipeline.rst lines 260-281."""
    # Load and execute the YAML pipeline:
    
    # Check for required API keys
    missing_keys = []
    if not os.environ.get('OPENAI_API_KEY'):
        missing_keys.append('OPENAI_API_KEY')
    if not os.environ.get('ANTHROPIC_API_KEY'):
        missing_keys.append('ANTHROPIC_API_KEY')
    
    if missing_keys:
        pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
    
    # Set up test environment
    import tempfile
    import orchestrator as orc
    
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Create a minimal test pipeline file if referenced
        code = ("""from orchestrator.compiler import YAMLCompiler

async def run_yaml_pipeline():
    # Create compiler and load pipeline
    compiler = YAMLCompiler()
    pipeline = compiler.compile_file("research_pipeline.yaml")

    # Create orchestrator
    orchestrator = Orchestrator()
    orchestrator.register_model(model)

    print("Starting YAML pipeline execution...")

    # Execute pipeline
    result = await orchestrator.execute_pipeline(pipeline)

    print("✅ YAML pipeline completed!")
    return result

# Run YAML pipeline
result = await run_yaml_pipeline()""")
        if 'hello_world.yaml' in code:
            with open('hello_world.yaml', 'w') as f:
                f.write("""
id: hello_world
name: Hello World
steps:
  - id: greet
    action: generate_text
    parameters:
      prompt: "Say hello world"
outputs:
  greeting: "{{ greet.result }}"
""".strip())
        
        # Mock the compile function to avoid real file dependencies
        original_compile = getattr(orc, 'compile', None)
        
        def mock_compile(filename):
            # Return a mock pipeline object
            class MockPipeline:
                def run(self):
                    return {"greeting": "Hello World"}
            return MockPipeline()
        
        # Test the code with mocked dependencies
        try:
            # Replace compile calls temporarily
            if original_compile:
                setattr(orc, 'compile_mock', mock_compile)
                setattr(orc, 'compile', mock_compile)
            
            # Execute the code
            exec(code, {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")
        finally:
            # Restore original function
            if original_compile:
                setattr(orc, 'compile', original_compile)

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_your_first_pipeline_lines_289_313_10():
    """Test orchestrator code from docs/getting_started/your_first_pipeline.rst lines 289-313."""
    # Replace mock model with real AI:
    
    # Check for required API keys
    missing_keys = []
    if not os.environ.get('OPENAI_API_KEY'):
        missing_keys.append('OPENAI_API_KEY')
    if not os.environ.get('ANTHROPIC_API_KEY'):
        missing_keys.append('ANTHROPIC_API_KEY')
    
    if missing_keys:
        pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
    
    # Set up test environment
    import tempfile
    import orchestrator as orc
    
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Create a minimal test pipeline file if referenced
        code = ("""from orchestrator.models.openai_model import OpenAIModel

async def run_with_real_ai():
    # Create OpenAI model
    openai_model = OpenAIModel(
        name="gpt-4",
        api_key="your-openai-api-key",
        model="gpt-4"
    )

    # Create orchestrator with real AI
    orchestrator = Orchestrator()
    orchestrator.register_model(openai_model)

    print("Starting pipeline with real AI...")

    # Execute pipeline with real AI
    result = await orchestrator.execute_pipeline(pipeline)

    print("✅ Real AI pipeline completed!")
    return result

# Run with real AI (uncomment when you have API keys)
# result = await run_with_real_ai()""")
        if 'hello_world.yaml' in code:
            with open('hello_world.yaml', 'w') as f:
                f.write("""
id: hello_world
name: Hello World
steps:
  - id: greet
    action: generate_text
    parameters:
      prompt: "Say hello world"
outputs:
  greeting: "{{ greet.result }}"
""".strip())
        
        # Mock the compile function to avoid real file dependencies
        original_compile = getattr(orc, 'compile', None)
        
        def mock_compile(filename):
            # Return a mock pipeline object
            class MockPipeline:
                def run(self):
                    return {"greeting": "Hello World"}
            return MockPipeline()
        
        # Test the code with mocked dependencies
        try:
            # Replace compile calls temporarily
            if original_compile:
                setattr(orc, 'compile_mock', mock_compile)
                setattr(orc, 'compile', mock_compile)
            
            # Execute the code
            exec(code, {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")
        finally:
            # Restore original function
            if original_compile:
                setattr(orc, 'compile', original_compile)

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_your_first_pipeline_lines_321_351_11():
    """Test orchestrator code from docs/getting_started/your_first_pipeline.rst lines 321-351."""
    # Add monitoring to track performance:
    
    # Check for required API keys
    missing_keys = []
    if not os.environ.get('OPENAI_API_KEY'):
        missing_keys.append('OPENAI_API_KEY')
    if not os.environ.get('ANTHROPIC_API_KEY'):
        missing_keys.append('ANTHROPIC_API_KEY')
    
    if missing_keys:
        pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
    
    # Set up test environment
    import tempfile
    import orchestrator as orc
    
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Create a minimal test pipeline file if referenced
        code = ("""import time
from orchestrator.core.resource_allocator import ResourceAllocator

async def run_monitored_pipeline():
    # Create resource allocator for monitoring
    allocator = ResourceAllocator()

    # Create orchestrator with monitoring
    orchestrator = Orchestrator(resource_allocator=allocator)
    orchestrator.register_model(model)

    print("Starting monitored pipeline execution...")
    start_time = time.time()

    # Execute pipeline
    result = await orchestrator.execute_pipeline(pipeline)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"✅ Pipeline completed in {execution_time:.2f} seconds")

    # Get resource statistics
    stats = allocator.get_overall_statistics()
    print(f"Resource utilization: {stats['overall_utilization']:.2f}")

    return result

# Run monitored pipeline
result = await run_monitored_pipeline()""")
        if 'hello_world.yaml' in code:
            with open('hello_world.yaml', 'w') as f:
                f.write("""
id: hello_world
name: Hello World
steps:
  - id: greet
    action: generate_text
    parameters:
      prompt: "Say hello world"
outputs:
  greeting: "{{ greet.result }}"
""".strip())
        
        # Mock the compile function to avoid real file dependencies
        original_compile = getattr(orc, 'compile', None)
        
        def mock_compile(filename):
            # Return a mock pipeline object
            class MockPipeline:
                def run(self):
                    return {"greeting": "Hello World"}
            return MockPipeline()
        
        # Test the code with mocked dependencies
        try:
            # Replace compile calls temporarily
            if original_compile:
                setattr(orc, 'compile_mock', mock_compile)
                setattr(orc, 'compile', mock_compile)
            
            # Execute the code
            exec(code, {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")
        finally:
            # Restore original function
            if original_compile:
                setattr(orc, 'compile', original_compile)

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_your_first_pipeline_lines_359_474_12():
    """Test orchestrator code from docs/getting_started/your_first_pipeline.rst lines 359-474."""
    # Here's the complete, production-ready pipeline:
    
    # Check for required API keys
    missing_keys = []
    if not os.environ.get('OPENAI_API_KEY'):
        missing_keys.append('OPENAI_API_KEY')
    if not os.environ.get('ANTHROPIC_API_KEY'):
        missing_keys.append('ANTHROPIC_API_KEY')
    
    if missing_keys:
        pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
    
    # Set up test environment
    import tempfile
    import orchestrator as orc
    
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Create a minimal test pipeline file if referenced
        code = ('''import asyncio
import logging
from orchestrator import Orchestrator, Task, Pipeline
from orchestrator.models.mock_model import MockModel
from orchestrator.core.error_handler import ErrorHandler
from orchestrator.state import StateManager
from orchestrator.core.resource_allocator import ResourceAllocator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_research_pipeline():
    """Create a production-ready research assistant pipeline."""

    # Create mock model with responses
    model = MockModel("research_assistant")
    model.set_response(
        "Generate 3 research questions about: artificial intelligence",
        "1. How does AI impact job markets?\n2. What are the ethical implications of AI?\n3. How can AI be made more accessible?"
    )
    model.set_response(
        "Analyze these questions and identify key themes: 1. How does AI impact job markets?\n2. What are the ethical implications of AI?\n3. How can AI be made more accessible?",
        "Key themes: Economic Impact, Ethics and Responsibility, Accessibility"
    )
    model.set_response(
        "Write a comprehensive report on artificial intelligence covering these themes: Economic Impact, Ethics and Responsibility, Accessibility",
        "# AI Research Report\n\n## Economic Impact\nAI is transforming industries...\n\n## Ethics\nResponsible AI development...\n\n## Accessibility\nDemocratizing AI tools..."
    )

    # Create tasks
    tasks = [
        Task(
            id="research_questions",
            name="Generate Research Questions",
            action="generate_text",
            parameters={
                "prompt": "Generate 3 research questions about: {topic}",
                "max_tokens": 200
            }
        ),
        Task(
            id="analyze_themes",
            name="Analyze Key Themes",
            action="generate_text",
            parameters={
                "prompt": "Analyze these questions and identify key themes: {research_questions}",
                "max_tokens": 150
            },
            dependencies=["research_questions"]
        ),
        Task(
            id="write_report",
            name="Write Research Report",
            action="generate_text",
            parameters={
                "prompt": "Write a comprehensive report on {topic} covering these themes: {analyze_themes}",
                "max_tokens": 500
            },
            dependencies=["analyze_themes"]
        )
    ]

    # Create pipeline
    pipeline = Pipeline(
        id="research_assistant",
        name="Research Assistant Pipeline"
    )

    for task in tasks:
        pipeline.add_task(task)

    pipeline.set_context("topic", "artificial intelligence")

    # Create components
    error_handler = ErrorHandler()
    state_manager = StateManager(storage_path="./checkpoints")
    resource_allocator = ResourceAllocator()

    # Create orchestrator
    orchestrator = Orchestrator(
        error_handler=error_handler,
        state_manager=state_manager,
        resource_allocator=resource_allocator
    )

    orchestrator.register_model(model)

    return orchestrator, pipeline

async def main():
    """Main execution function."""
    logger.info("Creating research assistant pipeline...")

    orchestrator, pipeline = await create_research_pipeline()

    logger.info("Executing pipeline...")

    try:
        result = await orchestrator.execute_pipeline(pipeline)

        logger.info("Pipeline completed successfully!")

        print("\n=== Results ===")
        for task_id, output in result.items():
            print(f"\n{task_id}:")
            print(f"{output}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

# Run the complete example
if __name__ == "__main__":
    asyncio.run(main())''')
        if 'hello_world.yaml' in code:
            with open('hello_world.yaml', 'w') as f:
                f.write("""
id: hello_world
name: Hello World
steps:
  - id: greet
    action: generate_text
    parameters:
      prompt: "Say hello world"
outputs:
  greeting: "{{ greet.result }}"
""".strip())
        
        # Mock the compile function to avoid real file dependencies
        original_compile = getattr(orc, 'compile', None)
        
        def mock_compile(filename):
            # Return a mock pipeline object
            class MockPipeline:
                def run(self):
                    return {"greeting": "Hello World"}
            return MockPipeline()
        
        # Test the code with mocked dependencies
        try:
            # Replace compile calls temporarily
            if original_compile:
                setattr(orc, 'compile_mock', mock_compile)
                setattr(orc, 'compile', mock_compile)
            
            # Execute the code
            exec(code, {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")
        finally:
            # Restore original function
            if original_compile:
                setattr(orc, 'compile', original_compile)

def test_index_lines_27_28_13():
    """Test bash snippet from docs/index.rst lines 27-28."""
    # Bash command snippet
    snippet_bash = ("""pip install py-orc""")
    
    # Validate pip install commands without actually running them
    assert "pip install" in snippet_bash
    
    lines = snippet_bash.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {line}"

def test_index_lines_107_124_14():
    """Test text snippet from docs/index.rst lines 107-124."""
    # Content validation for text snippet
    content = ("""┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator Engine                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  YAML Compiler  │  │ Model Registry  │  │ State Manager   │
│  - Parser       │  │ - Selection     │  │ - Checkpoints   │
│  - Validation   │  │ - Load Balance  │  │ - Recovery      │
│  - Templates    │  │ - Health Check  │  │ - Persistence   │
└─────────────────┘  └─────────────────┘  └─────────────────┘

┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Execution Layer │  │ Error Handler   │  │ Resource Mgmt   │
│ - Parallel      │  │ - Circuit Break │  │ - Allocation    │
│ - Sandboxed     │  │ - Retry Logic   │  │ - Monitoring    │
│ - Distributed   │  │ - Recovery      │  │ - Optimization  │
└─────────────────┘  └─────────────────┘  └─────────────────┘""")
    
    # Basic checks
    assert content.strip(), "Snippet content should not be empty"
    assert len(content.strip()) > 0, "Snippet should have content"
