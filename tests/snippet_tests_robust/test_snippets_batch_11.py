"""Tests for documentation code snippets - Batch 11 (Robust)."""
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


def test_notebooks_lines_55_64_0():
    """Test bash snippet from docs/tutorials/notebooks.rst lines 55-64."""
    # Bash command snippet
    snippet_bash = ("""# Install Orchestrator Framework
pip install py-orc

# Install Jupyter (if not already installed)
pip install jupyter

# Clone the repository for tutorials
git clone https://github.com/ContextLab/orchestrator.git
cd orchestrator""")
    
    # Validate pip install commands without actually running them
    assert "pip install" in snippet_bash
    
    lines = snippet_bash.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {line}"

def test_notebooks_lines_70_75_1():
    """Test bash snippet from docs/tutorials/notebooks.rst lines 70-75."""
    import subprocess
    import tempfile
    import os
    
    bash_content = ("""# Start Jupyter Notebook
jupyter notebook

# Or start JupyterLab
jupyter lab""")
    
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
async def test_notebooks_lines_112_130_2():
    """Test orchestrator code from docs/tutorials/notebooks.rst lines 112-130."""
    # * Add state management for reliability
    
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
        code = ("""# Example from Tutorial 01
from orchestrator import Orchestrator, Task, Pipeline
from orchestrator.models.mock_model import MockModel

# Create your first task
task = Task(
    id="hello_world",
    name="Hello World Task",
    action="generate_text",
    parameters={"prompt": "Hello, Orchestrator!"}
)

# Build and execute pipeline
pipeline = Pipeline(id="first_pipeline", name="First Pipeline")
pipeline.add_task(task)

orchestrator = Orchestrator()
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
async def test_notebooks_lines_165_186_3():
    """Test YAML pipeline from docs/tutorials/notebooks.rst lines 165-186."""
    import yaml
    import orchestrator
    
    yaml_content = ("""# Example from Tutorial 02
id: research_pipeline
name: Research Assistant Pipeline

context:
  topic: artificial intelligence

tasks:
  - id: research
    name: Generate Research Questions
    action: generate_text
    parameters:
      prompt: "Research questions about: {topic}"

  - id: analyze
    name: Analyze Themes
    action: generate_text
    parameters:
      prompt: "Analyze themes in: {research}"
    dependencies:
      - research""")
    
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
async def test_notebooks_lines_221_234_4():
    """Test orchestrator code from docs/tutorials/notebooks.rst lines 221-234."""
    # * Optimize for cost and latency
    
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
        code = ("""# Example from Tutorial 03
from orchestrator.models.openai_model import OpenAIModel
from orchestrator.models.anthropic_model import AnthropicModel

# Register multiple models
gpt4 = OpenAIModel(name="gpt-4", api_key="your-key")
claude = AnthropicModel(name="claude-3", api_key="your-key")

orchestrator.register_model(gpt4)
orchestrator.register_model(claude)

# Orchestrator automatically selects best model
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

def test_notebooks_lines_275_291_5():
    """Test text snippet from docs/tutorials/notebooks.rst lines 275-291."""
    # Content validation for text snippet
    content = ("""notebooks/
├── 01_getting_started.ipynb
├── 02_yaml_configuration.ipynb
├── 03_advanced_model_integration.ipynb
├── README.md                           # Tutorial guide
├── data/                               # Sample data files
│   ├── sample_pipeline.yaml
│   ├── complex_workflow.yaml
│   └── test_data.json
├── images/                             # Tutorial images
│   ├── architecture_diagram.png
│   └── workflow_visualization.png
└── solutions/                          # Exercise solutions
    ├── 01_solutions.ipynb
    ├── 02_solutions.ipynb
    └── 03_solutions.ipynb""")
    
    # Basic checks
    assert content.strip(), "Snippet content should not be empty"
    assert len(content.strip()) > 0, "Snippet should have content"

def test_notebooks_lines_310_315_6():
    """Test bash snippet from docs/tutorials/notebooks.rst lines 310-315."""
    # Bash command snippet
    snippet_bash = ("""# Try updating Jupyter
pip install --upgrade jupyter

# Or install JupyterLab
pip install jupyterlab""")
    
    # Validate pip install commands without actually running them
    assert "pip install" in snippet_bash
    
    lines = snippet_bash.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {line}"

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_notebooks_lines_319_324_7():
    """Test orchestrator code from docs/tutorials/notebooks.rst lines 319-324."""
    # **Import Errors**
    
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
        code = ("""# Make sure Orchestrator is installed
pip install py-orc

# Or install in development mode
pip install -e .""")
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

def test_notebooks_lines_328_330_8():
    """Test Python snippet from docs/tutorials/notebooks.rst lines 328-330."""
    # **Mock Model Issues**
    
    code = ("""# Mock models need explicit responses
model.set_response("your prompt", "expected response")""")
    
    try:
        exec(code, {'__name__': '__main__'})
    except Exception as e:
        pytest.fail(f"Code execution failed: {e}")

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_notebooks_lines_334_336_9():
    """Test orchestrator code from docs/tutorials/notebooks.rst lines 334-336."""
    # **Async/Await Problems**
    
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
        code = ("""# Use await in notebook cells
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

def test_configuration_lines_33_35_10():
    """Test bash snippet from docs/user_guide/configuration.rst lines 33-35."""
    import subprocess
    import tempfile
    import os
    
    bash_content = ("""# Install default configs to ~/.orchestrator/
orchestrator-install-configs""")
    
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

def test_configuration_lines_53_81_11():
    """Test YAML snippet from docs/user_guide/configuration.rst lines 53-81."""
    import yaml
    
    yaml_content = ("""models:
  # Local models (via Ollama)
  - source: ollama
    name: llama3.1:8b
    expertise: [general, reasoning, multilingual]
    size: 8b

  # Cloud models
  - source: openai
    name: gpt-4o
    expertise: [general, reasoning, code, analysis, vision]
    size: 1760b  # Estimated

  # HuggingFace models
  - source: huggingface
    name: microsoft/Phi-3.5-mini-instruct
    expertise: [reasoning, code, compact]
    size: 3.8b

defaults:
  expertise_preferences:
    code: qwen2.5-coder:7b
    reasoning: deepseek-r1:8b
    fast: llama3.2:1b
  fallback_chain:
    - llama3.1:8b
    - mistral:7b
    - llama3.2:1b""")
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_configuration_lines_86_91_12():
    """Test YAML snippet from docs/user_guide/configuration.rst lines 86-91."""
    import yaml
    
    yaml_content = ("""# Add a new Ollama model
- source: ollama
  name: my-custom-model:13b
  expertise: [domain-specific, analysis]
  size: 13b""")
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_configuration_lines_99_129_13():
    """Test YAML pipeline from docs/user_guide/configuration.rst lines 99-129."""
    import yaml
    import orchestrator
    
    yaml_content = ("""# Execution settings
execution:
  parallel_tasks: 10
  timeout_seconds: 300
  retry_attempts: 3
  retry_delay: 1.0

# Resource limits
resources:
  max_memory_mb: 8192
  max_cpu_percent: 80
  gpu_enabled: true

# Caching
cache:
  enabled: true
  ttl_seconds: 3600
  max_size_mb: 1024

# Monitoring
monitoring:
  log_level: INFO
  metrics_enabled: true
  trace_enabled: false

# Error handling
error_handling:
  circuit_breaker_threshold: 5
  circuit_breaker_timeout: 60
  fallback_enabled: true""")
    
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

def test_configuration_lines_137_144_14():
    """Test bash snippet from docs/user_guide/configuration.rst lines 137-144."""
    import subprocess
    import tempfile
    import os
    
    bash_content = ("""# Set custom config location
export ORCHESTRATOR_HOME=/path/to/configs

# Override specific settings
export ORCHESTRATOR_LOG_LEVEL=DEBUG
export ORCHESTRATOR_PARALLEL_TASKS=20
export ORCHESTRATOR_CACHE_ENABLED=false""")
    
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
