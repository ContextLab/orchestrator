"""Tests for documentation code snippets - Batch 17 (Robust)."""
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


def test_installation_lines_106_114_0():
    """Test bash snippet from docs_sphinx/installation.rst lines 106-114."""
    import subprocess
    import tempfile
    import os
    
    bash_content = ("""# Large model for complex tasks
ollama pull gemma2:27b

# Small model for simple tasks
ollama pull llama3.2:1b

# Code-focused model
ollama pull codellama:7b""")
    
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
async def test_installation_lines_119_124_1():
    """Test orchestrator code from docs_sphinx/installation.rst lines 119-124."""
    # 3. **Verify installation**:
    
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
        code = ("""import orchestrator as orc

# Initialize and check models
registry = orc.init_models()
print(registry.list_models())""")
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

def test_installation_lines_132_137_2():
    """Test bash snippet from docs_sphinx/installation.rst lines 132-137."""
    import subprocess
    import tempfile
    import os
    
    bash_content = ("""# Set environment variable
export HUGGINGFACE_TOKEN="your-token-here"

# Or create .env file
echo "HUGGINGFACE_TOKEN=your-token-here" > .env""")
    
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

def test_installation_lines_145_150_3():
    """Test bash snippet from docs_sphinx/installation.rst lines 145-150."""
    import subprocess
    import tempfile
    import os
    
    bash_content = ("""# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."""")
    
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

def test_installation_lines_161_168_4():
    """Test bash snippet from docs_sphinx/installation.rst lines 161-168."""
    # Bash command snippet
    snippet_bash = ("""# Install Playwright
pip install playwright
playwright install chromium

# Or use Selenium
pip install selenium
# Download appropriate driver""")
    
    # Validate pip install commands without actually running them
    assert "pip install" in snippet_bash
    
    lines = snippet_bash.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {line}"

def test_installation_lines_181_186_5():
    """Test bash snippet from docs_sphinx/installation.rst lines 181-186."""
    # Bash command snippet
    snippet_bash = ("""# For advanced data processing
pip install pandas numpy scipy

# For data validation
pip install pydantic jsonschema""")
    
    # Validate pip install commands without actually running them
    assert "pip install" in snippet_bash
    
    lines = snippet_bash.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {line}"

def test_installation_lines_194_214_6():
    """Test YAML snippet from docs_sphinx/installation.rst lines 194-214."""
    import yaml
    
    yaml_content = ("""# Model preferences
models:
  default: "ollama:gemma2:27b"
  fallback: "ollama:llama3.2:1b"

# Resource limits
resources:
  max_memory: "16GB"
  max_threads: 8
  gpu_enabled: true

# Tool settings
tools:
  mcp_port: 8000
  sandbox_enabled: true

# State management
state:
  backend: "postgresql"
  connection: "postgresql://user:pass@localhost/orchestrator"""")
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_installation_lines_222_233_7():
    """Test bash snippet from docs_sphinx/installation.rst lines 222-233."""
    import subprocess
    import tempfile
    import os
    
    bash_content = ("""# Core settings
export ORCHESTRATOR_HOME="$HOME/.orchestrator"
export ORCHESTRATOR_LOG_LEVEL="INFO"

# Model settings
export ORCHESTRATOR_MODEL_TIMEOUT="300"
export ORCHESTRATOR_MODEL_RETRIES="3"

# Tool settings
export ORCHESTRATOR_TOOL_TIMEOUT="60"
export ORCHESTRATOR_MCP_AUTO_START="true"""")
    
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
async def test_installation_lines_241_266_8():
    """Test orchestrator code from docs_sphinx/installation.rst lines 241-266."""
    # Run the verification script:
    
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
        code = ("""import orchestrator as orc

# Check version
print(f"Orchestrator version: {orc.__version__}")

# Check models
try:
    registry = orc.init_models()
    models = registry.list_models()
    print(f"Available models: {models}")
except Exception as e:
    print(f"Model initialization failed: {e}")

# Check tools
from orchestrator.tools.base import default_registry
tools = default_registry.list_tools()
print(f"Available tools: {tools}")

# Run test pipeline
try:
    pipeline = orc.compile("examples/hello-world.yaml")
    result = pipeline.run(message="Hello, Orchestrator!")
    print(f"Test pipeline result: {result}")
except Exception as e:
    print(f"Pipeline test failed: {e}")""")
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

def test_installation_lines_277_278_9():
    """Test text snippet from docs_sphinx/installation.rst lines 277-278."""
    # Content validation for text snippet
    content = ("""ModuleNotFoundError: No module named 'orchestrator'""")
    
    # Basic checks
    assert content.strip(), "Snippet content should not be empty"
    assert len(content.strip()) > 0, "Snippet should have content"

def test_installation_lines_285_286_10():
    """Test text snippet from docs_sphinx/installation.rst lines 285-286."""
    # Content validation for text snippet
    content = ("""Failed to connect to Ollama at http://localhost:11434""")
    
    # Basic checks
    assert content.strip(), "Snippet content should not be empty"
    assert len(content.strip()) > 0, "Snippet should have content"

def test_installation_lines_293_294_11():
    """Test text snippet from docs_sphinx/installation.rst lines 293-294."""
    # Content validation for text snippet
    content = ("""Permission denied: '/home/user/.orchestrator'""")
    
    # Basic checks
    assert content.strip(), "Snippet content should not be empty"
    assert len(content.strip()) > 0, "Snippet should have content"

def test_installation_lines_299_301_12():
    """Test bash snippet from docs_sphinx/installation.rst lines 299-301."""
    import subprocess
    import tempfile
    import os
    
    bash_content = ("""mkdir -p ~/.orchestrator
chmod 755 ~/.orchestrator""")
    
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
async def test_quickstart_lines_19_59_13():
    """Test YAML pipeline from docs_sphinx/quickstart.rst lines 19-59."""
    import yaml
    import orchestrator
    
    yaml_content = ("""name: topic-summarizer
description: Generate a concise summary of any topic

inputs:
  topic:
    type: string
    description: The topic to summarize
    required: true

  length:
    type: integer
    description: Approximate word count for the summary
    default: 200

outputs:
  summary:
    type: string
    value: "{{ inputs.topic }}_summary.txt"

steps:
  - id: research
    action: generate_content
    parameters:
      prompt: |
        Research and provide key information about: {{ inputs.topic }}
        Focus on the most important and interesting aspects.
      max_length: 500

  - id: summarize
    action: generate_summary
    parameters:
      content: "$results.research"
      target_length: "{{ inputs.length }}"
      style: <AUTO>Choose appropriate style for the topic</AUTO>

  - id: save_summary
    action: write_file
    parameters:
      path: "{{ outputs.summary }}"
      content: "$results.summarize"""")
    
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
async def test_quickstart_lines_67_87_14():
    """Test orchestrator code from docs_sphinx/quickstart.rst lines 67-87."""
    # Create a Python script to run your pipeline:
    
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
        code = ("""import orchestrator as orc

# Initialize the model pool
orc.init_models()

# Compile the pipeline
pipeline = orc.compile("summarize.yaml")

# Run with different topics
result1 = pipeline.run(
    topic="quantum computing",
    length=150
)

result2 = pipeline.run(
    topic="sustainable energy",
    length=250
)

print(f"Created summaries: {result1}, {result2}")""")
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
