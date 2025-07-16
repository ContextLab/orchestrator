"""Tests for documentation code snippets - Batch 7 (Fixed)."""
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


def test_installation_lines_56_57_0():
    """Test bash snippet from docs/getting_started/installation.rst lines 56-57."""
    # Bash command snippet
    snippet_bash = r"""pip install py-orc[docker]"""
    
    # Validate pip install commands without actually running them
    assert "pip install" in snippet_bash
    
    lines = snippet_bash.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {line}"

def test_installation_lines_64_65_1():
    """Test bash snippet from docs/getting_started/installation.rst lines 64-65."""
    # Bash command snippet
    snippet_bash = r"""pip install py-orc[database]"""
    
    # Validate pip install commands without actually running them
    assert "pip install" in snippet_bash
    
    lines = snippet_bash.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {line}"

def test_installation_lines_72_73_2():
    """Test bash snippet from docs/getting_started/installation.rst lines 72-73."""
    # Bash command snippet
    snippet_bash = r"""pip install py-orc[all]"""
    
    # Validate pip install commands without actually running them
    assert "pip install" in snippet_bash
    
    lines = snippet_bash.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {line}"

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_installation_lines_81_92_3():
    """Test orchestrator code from docs/getting_started/installation.rst lines 81-92."""
    # Verify your installation by running:
    
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
        if 'hello_world.yaml' in r"""import orchestrator as orc
print(f"Orchestrator version: {orchestrator.__version__}")

# Test basic functionality
from orchestrator import Task, Pipeline

task = Task(id="test", name="Test Task", action="echo", parameters={"message": "Hello!"})
pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
pipeline.add_task(task)

print("✅ Installation successful!")""":
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
            exec(r"""import orchestrator as orc
print(f"Orchestrator version: {orchestrator.__version__}")

# Test basic functionality
from orchestrator import Task, Pipeline

task = Task(id="test", name="Test Task", action="echo", parameters={"message": "Hello!"})
pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
pipeline.add_task(task)

print("✅ Installation successful!")""", {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")
        finally:
            # Restore original function
            if original_compile:
                setattr(orc, 'compile', original_compile)

def test_installation_lines_105_113_4():
    """Test bash snippet from docs/getting_started/installation.rst lines 105-113."""
    import subprocess
    import tempfile
    import os
    
    bash_content = r"""# Optional: Set cache directory
export ORCHESTRATOR_CACHE_DIR=/path/to/cache

# Optional: Set checkpoint directory
export ORCHESTRATOR_CHECKPOINT_DIR=/path/to/checkpoints

# Optional: Set log level
export ORCHESTRATOR_LOG_LEVEL=INFO"""
    
    # Skip potentially dangerous commands
    skip_commands = ['rm -rf', 'sudo', 'docker run', 'systemctl', 'service']
    if any(cmd in bash_content for cmd in skip_commands):
        pytest.skip("Skipping potentially destructive command")
    
    # Check bash syntax only
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

def test_installation_lines_121_129_5():
    """Test bash snippet from docs/getting_started/installation.rst lines 121-129."""
    import subprocess
    import tempfile
    import os
    
    bash_content = r"""# OpenAI
export OPENAI_API_KEY=your_openai_key

# Anthropic
export ANTHROPIC_API_KEY=your_anthropic_key

# Google
export GOOGLE_API_KEY=your_google_key"""
    
    # Skip potentially dangerous commands
    skip_commands = ['rm -rf', 'sudo', 'docker run', 'systemctl', 'service']
    if any(cmd in bash_content for cmd in skip_commands):
        pytest.skip("Skipping potentially destructive command")
    
    # Check bash syntax only
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

def test_installation_lines_137_139_6():
    """Test bash snippet from docs/getting_started/installation.rst lines 137-139."""
    import subprocess
    import tempfile
    import os
    
    bash_content = r"""docker --version
docker run hello-world"""
    
    # Skip potentially dangerous commands
    skip_commands = ['rm -rf', 'sudo', 'docker run', 'systemctl', 'service']
    if any(cmd in bash_content for cmd in skip_commands):
        pytest.skip("Skipping potentially destructive command")
    
    # Check bash syntax only
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

def test_installation_lines_157_166_7():
    """Test bash snippet from docs/getting_started/installation.rst lines 157-166."""
    import subprocess
    import tempfile
    import os
    
    bash_content = r"""# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev build-essential

# macOS
brew install python

# Windows
# Use Python from python.org"""
    
    # Skip potentially dangerous commands
    skip_commands = ['rm -rf', 'sudo', 'docker run', 'systemctl', 'service']
    if any(cmd in bash_content for cmd in skip_commands):
        pytest.skip("Skipping potentially destructive command")
    
    # Check bash syntax only
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

def test_installation_lines_172_174_8():
    """Test bash snippet from docs/getting_started/installation.rst lines 172-174."""
    import subprocess
    import tempfile
    import os
    
    bash_content = r"""docker --version
docker info"""
    
    # Skip potentially dangerous commands
    skip_commands = ['rm -rf', 'sudo', 'docker run', 'systemctl', 'service']
    if any(cmd in bash_content for cmd in skip_commands):
        pytest.skip("Skipping potentially destructive command")
    
    # Check bash syntax only
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

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_quickstart_lines_13_39_9():
    """Test orchestrator code from docs/getting_started/quickstart.rst lines 13-39."""
    # Let's create a simple text generation pipeline:
    
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
        if 'hello_world.yaml' in r"""from orchestrator import Orchestrator, Task, Pipeline
from orchestrator.models.mock_model import MockModel

# Create a mock model for testing
model = MockModel("gpt-test")
model.set_response("Hello, world!", "Hello! How can I help you today?")

# Create a task
task = Task(
    id="greeting",
    name="Generate Greeting",
    action="generate_text",
    parameters={"prompt": "Hello, world!"}
)

# Create a pipeline
pipeline = Pipeline(id="hello_pipeline", name="Hello Pipeline")
pipeline.add_task(task)

# Create orchestrator and register model
orchestrator = Orchestrator()
orchestrator.register_model(model)

# Execute pipeline
result = await orchestrator.execute_pipeline(pipeline)
print(f"Result: {result['greeting']}")""":
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
            exec(r"""from orchestrator import Orchestrator, Task, Pipeline
from orchestrator.models.mock_model import MockModel

# Create a mock model for testing
model = MockModel("gpt-test")
model.set_response("Hello, world!", "Hello! How can I help you today?")

# Create a task
task = Task(
    id="greeting",
    name="Generate Greeting",
    action="generate_text",
    parameters={"prompt": "Hello, world!"}
)

# Create a pipeline
pipeline = Pipeline(id="hello_pipeline", name="Hello Pipeline")
pipeline.add_task(task)

# Create orchestrator and register model
orchestrator = Orchestrator()
orchestrator.register_model(model)

# Execute pipeline
result = await orchestrator.execute_pipeline(pipeline)
print(f"Result: {result['greeting']}")""", {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
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
async def test_quickstart_lines_47_86_10():
    """Test orchestrator code from docs/getting_started/quickstart.rst lines 47-86."""
    # Let's create a more complex pipeline with multiple tasks:
    
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
        if 'hello_world.yaml' in r"""from orchestrator import Task, Pipeline

# Task 1: Generate story outline
outline_task = Task(
    id="outline",
    name="Generate Story Outline",
    action="generate_text",
    parameters={"prompt": "Create a story outline about space exploration"}
)

# Task 2: Write story (depends on outline)
story_task = Task(
    id="story",
    name="Write Story",
    action="generate_text",
    parameters={"prompt": "Write a story based on: {outline}"},
    dependencies=["outline"]
)

# Task 3: Summarize story (depends on story)
summary_task = Task(
    id="summary",
    name="Summarize Story",
    action="generate_text",
    parameters={"prompt": "Summarize this story: {story}"},
    dependencies=["story"]
)

# Create pipeline with all tasks
pipeline = Pipeline(id="story_pipeline", name="Story Creation Pipeline")
pipeline.add_task(outline_task)
pipeline.add_task(story_task)
pipeline.add_task(summary_task)

# Execute pipeline
result = await orchestrator.execute_pipeline(pipeline)
print(f"Outline: {result['outline']}")
print(f"Story: {result['story']}")
print(f"Summary: {result['summary']}")""":
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
            exec(r"""from orchestrator import Task, Pipeline

# Task 1: Generate story outline
outline_task = Task(
    id="outline",
    name="Generate Story Outline",
    action="generate_text",
    parameters={"prompt": "Create a story outline about space exploration"}
)

# Task 2: Write story (depends on outline)
story_task = Task(
    id="story",
    name="Write Story",
    action="generate_text",
    parameters={"prompt": "Write a story based on: {outline}"},
    dependencies=["outline"]
)

# Task 3: Summarize story (depends on story)
summary_task = Task(
    id="summary",
    name="Summarize Story",
    action="generate_text",
    parameters={"prompt": "Summarize this story: {story}"},
    dependencies=["story"]
)

# Create pipeline with all tasks
pipeline = Pipeline(id="story_pipeline", name="Story Creation Pipeline")
pipeline.add_task(outline_task)
pipeline.add_task(story_task)
pipeline.add_task(summary_task)

# Execute pipeline
result = await orchestrator.execute_pipeline(pipeline)
print(f"Outline: {result['outline']}")
print(f"Story: {result['story']}")
print(f"Summary: {result['summary']}")""", {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
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
async def test_quickstart_lines_94_120_11():
    """Test YAML pipeline from docs/getting_started/quickstart.rst lines 94-120."""
    import yaml
    import orchestrator
    
    yaml_content = r"""# story_pipeline.yaml
id: story_pipeline
name: Story Creation Pipeline

tasks:
  - id: outline
    name: Generate Story Outline
    action: generate_text
    parameters:
      prompt: "Create a story outline about space exploration"

  - id: story
    name: Write Story
    action: generate_text
    parameters:
      prompt: "Write a story based on: {outline}"
    dependencies:
      - outline

  - id: summary
    name: Summarize Story
    action: generate_text
    parameters:
      prompt: "Summarize this story: {story}"
    dependencies:
      - story"""
    
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
async def test_quickstart_lines_125_133_12():
    """Test orchestrator code from docs/getting_started/quickstart.rst lines 125-133."""
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
        if 'hello_world.yaml' in r"""from orchestrator.compiler import YAMLCompiler

# Load pipeline from YAML
compiler = YAMLCompiler()
pipeline = compiler.compile_file("story_pipeline.yaml")

# Execute pipeline
result = await orchestrator.execute_pipeline(pipeline)""":
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
            exec(r"""from orchestrator.compiler import YAMLCompiler

# Load pipeline from YAML
compiler = YAMLCompiler()
pipeline = compiler.compile_file("story_pipeline.yaml")

# Execute pipeline
result = await orchestrator.execute_pipeline(pipeline)""", {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
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
async def test_quickstart_lines_141_155_13():
    """Test orchestrator code from docs/getting_started/quickstart.rst lines 141-155."""
    # Let's use a real AI model instead of the mock:
    
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
        if 'hello_world.yaml' in r"""from orchestrator.models.openai_model import OpenAIModel

# Create OpenAI model
openai_model = OpenAIModel(
    name="gpt-4",
    api_key="your-api-key-here",
    model="gpt-4"
)

# Register model
orchestrator.register_model(openai_model)

# Execute pipeline (will use OpenAI)
result = await orchestrator.execute_pipeline(pipeline)""":
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
            exec(r"""from orchestrator.models.openai_model import OpenAIModel

# Create OpenAI model
openai_model = OpenAIModel(
    name="gpt-4",
    api_key="your-api-key-here",
    model="gpt-4"
)

# Register model
orchestrator.register_model(openai_model)

# Execute pipeline (will use OpenAI)
result = await orchestrator.execute_pipeline(pipeline)""", {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
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
async def test_quickstart_lines_163_176_14():
    """Test orchestrator code from docs/getting_started/quickstart.rst lines 163-176."""
    # Orchestrator provides built-in error handling:
    
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
        if 'hello_world.yaml' in r"""from orchestrator.core.error_handler import ErrorHandler

# Create error handler with retry strategy
error_handler = ErrorHandler()

# Configure orchestrator with error handling
orchestrator = Orchestrator(error_handler=error_handler)

# Execute pipeline with automatic retry on failures
try:
    result = await orchestrator.execute_pipeline(pipeline)
except Exception as e:
    print(f"Pipeline failed: {e}")""":
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
            exec(r"""from orchestrator.core.error_handler import ErrorHandler

# Create error handler with retry strategy
error_handler = ErrorHandler()

# Configure orchestrator with error handling
orchestrator = Orchestrator(error_handler=error_handler)

# Execute pipeline with automatic retry on failures
try:
    result = await orchestrator.execute_pipeline(pipeline)
except Exception as e:
    print(f"Pipeline failed: {e}")""", {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
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
async def test_quickstart_lines_184_194_15():
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
        if 'hello_world.yaml' in r"""from orchestrator.state import StateManager

# Create state manager
state_manager = StateManager(storage_path="./checkpoints")

# Configure orchestrator with state management
orchestrator = Orchestrator(state_manager=state_manager)

# Execute pipeline with automatic checkpointing
result = await orchestrator.execute_pipeline(pipeline)""":
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
            exec(r"""from orchestrator.state import StateManager

# Create state manager
state_manager = StateManager(storage_path="./checkpoints")

# Configure orchestrator with state management
orchestrator = Orchestrator(state_manager=state_manager)

# Execute pipeline with automatic checkpointing
result = await orchestrator.execute_pipeline(pipeline)""", {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
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
async def test_quickstart_lines_202_214_16():
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
        if 'hello_world.yaml' in r"""import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Execute pipeline with logging
result = await orchestrator.execute_pipeline(pipeline)

# Get execution statistics
stats = orchestrator.get_execution_stats()
print(f"Execution time: {stats['total_time']:.2f}s")
print(f"Tasks completed: {stats['completed_tasks']}")""":
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
            exec(r"""import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Execute pipeline with logging
result = await orchestrator.execute_pipeline(pipeline)

# Get execution statistics
stats = orchestrator.get_execution_stats()
print(f"Execution time: {stats['total_time']:.2f}s")
print(f"Tasks completed: {stats['completed_tasks']}")""", {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
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
async def test_your_first_pipeline_lines_24_46_17():
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
        if 'hello_world.yaml' in r"""import asyncio
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
)""":
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
            exec(r"""import asyncio
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
)""", {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")
        finally:
            # Restore original function
            if original_compile:
                setattr(orc, 'compile', original_compile)

def test_your_first_pipeline_lines_54_88_18():
    """Test Python snippet from docs/getting_started/your_first_pipeline.rst lines 54-88."""
    # Now let's create our three tasks:
    
    code = r"""# Task 1: Generate research questions
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
)"""
    
    try:
        exec(code, {'__name__': '__main__'})
    except Exception as e:
        pytest.fail(f"Code execution failed: {e}")

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_your_first_pipeline_lines_96_114_19():
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
        if 'hello_world.yaml' in r"""# Create pipeline
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
print(f"Execution order: {pipeline.get_execution_order()}")""":
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
            exec(r"""# Create pipeline
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
print(f"Execution order: {pipeline.get_execution_order()}")""", {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")
        finally:
            # Restore original function
            if original_compile:
                setattr(orc, 'compile', original_compile)
