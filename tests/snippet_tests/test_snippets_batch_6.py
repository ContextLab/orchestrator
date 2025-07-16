"""Tests for documentation code snippets - Batch 6."""
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


def test_notebooks_lines_55_64_0():
    """Test bash snippet from docs/tutorials/notebooks.rst lines 55-64."""
    # Bash command snippet
    snippet_bash = r"""# Install Orchestrator Framework
pip install py-orc

# Install Jupyter (if not already installed)
pip install jupyter

# Clone the repository for tutorials
git clone https://github.com/ContextLab/orchestrator.git
cd orchestrator"""
    
    # Don't actually install packages in tests
    assert "pip install" in snippet_bash
    
    # Verify it's a valid pip command structure
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
    
    bash_content = r"""# Start Jupyter Notebook
jupyter notebook

# Or start JupyterLab
jupyter lab"""
    
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

@pytest.mark.asyncio
async def test_notebooks_lines_112_130_2():
    """Test Python snippet from docs/tutorials/notebooks.rst lines 112-130."""
    # * Add state management for reliability
    
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
        code = """# Example from Tutorial 01
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
async def test_notebooks_lines_165_186_3():
    """Test YAML pipeline from docs/tutorials/notebooks.rst lines 165-186."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """# Example from Tutorial 02
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
      - research"""
    
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

@pytest.mark.asyncio
async def test_notebooks_lines_221_234_4():
    """Test Python snippet from docs/tutorials/notebooks.rst lines 221-234."""
    # * Optimize for cost and latency
    
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
        code = """# Example from Tutorial 03
from orchestrator.models.openai_model import OpenAIModel
from orchestrator.models.anthropic_model import AnthropicModel

# Register multiple models
gpt4 = OpenAIModel(name="gpt-4", api_key="your-key")
claude = AnthropicModel(name="claude-3", api_key="your-key")

orchestrator.register_model(gpt4)
orchestrator.register_model(claude)

# Orchestrator automatically selects best model
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

def test_notebooks_lines_275_291_5():
    """Test text snippet from docs/tutorials/notebooks.rst lines 275-291."""
    # Snippet type 'text' not yet supported for testing
    pytest.skip("Snippet type 'text' not yet supported")

def test_notebooks_lines_310_315_6():
    """Test bash snippet from docs/tutorials/notebooks.rst lines 310-315."""
    # Bash command snippet
    snippet_bash = r"""# Try updating Jupyter
pip install --upgrade jupyter

# Or install JupyterLab
pip install jupyterlab"""
    
    # Don't actually install packages in tests
    assert "pip install" in snippet_bash
    
    # Verify it's a valid pip command structure
    lines = snippet_bash.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {line}"

def test_notebooks_lines_319_324_7():
    """Test Python snippet from docs/tutorials/notebooks.rst lines 319-324."""
    # **Import Errors**
    
    code = """# Make sure Orchestrator is installed
pip install py-orc

# Or install in development mode
pip install -e ."""
    
    try:
        exec(code, {'__name__': '__main__'})
    except Exception as e:
        pytest.fail(f"Code execution failed: {e}")

@pytest.mark.asyncio
async def test_notebooks_lines_328_330_8():
    """Test Python snippet from docs/tutorials/notebooks.rst lines 328-330."""
    # **Mock Model Issues**
    
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
        code = """# Mock models need explicit responses
model.set_response("your prompt", "expected response")"""
        
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
async def test_notebooks_lines_334_336_9():
    """Test Python snippet from docs/tutorials/notebooks.rst lines 334-336."""
    # **Async/Await Problems**
    
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
        code = """# Use await in notebook cells
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

def test_configuration_lines_33_35_10():
    """Test bash snippet from docs/user_guide/configuration.rst lines 33-35."""
    import subprocess
    import tempfile
    import os
    
    bash_content = r"""# Install default configs to ~/.orchestrator/
orchestrator-install-configs"""
    
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

def test_configuration_lines_53_81_11():
    """Test YAML snippet from docs/user_guide/configuration.rst lines 53-81."""
    import yaml
    
    yaml_content = """models:
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
    - llama3.2:1b"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_configuration_lines_86_91_12():
    """Test YAML snippet from docs/user_guide/configuration.rst lines 86-91."""
    import yaml
    
    yaml_content = """# Add a new Ollama model
- source: ollama
  name: my-custom-model:13b
  expertise: [domain-specific, analysis]
  size: 13b"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.asyncio
async def test_configuration_lines_99_129_13():
    """Test YAML pipeline from docs/user_guide/configuration.rst lines 99-129."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """# Execution settings
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
  fallback_enabled: true"""
    
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

def test_configuration_lines_137_144_14():
    """Test bash snippet from docs/user_guide/configuration.rst lines 137-144."""
    import subprocess
    import tempfile
    import os
    
    bash_content = r"""# Set custom config location
export ORCHESTRATOR_HOME=/path/to/configs

# Override specific settings
export ORCHESTRATOR_LOG_LEVEL=DEBUG
export ORCHESTRATOR_PARALLEL_TASKS=20
export ORCHESTRATOR_CACHE_ENABLED=false"""
    
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

def test_configuration_lines_161_167_15():
    """Test Python import from docs/user_guide/configuration.rst lines 161-167."""
    # Test imports
    try:
        exec("""import orchestrator as orc

# Validate configuration files
config_valid, errors = orc.validate_config()
if not config_valid:
    print("Configuration errors:", errors)""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

@pytest.mark.asyncio
async def test_configuration_lines_176_187_16():
    """Test YAML pipeline from docs/user_guide/configuration.rst lines 176-187."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """# orchestrator.yaml for development
execution:
  parallel_tasks: 2
  timeout_seconds: 60

monitoring:
  log_level: DEBUG
  trace_enabled: true

cache:
  enabled: false  # Disable cache for testing"""
    
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

@pytest.mark.asyncio
async def test_configuration_lines_193_206_17():
    """Test YAML pipeline from docs/user_guide/configuration.rst lines 193-206."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """# orchestrator.yaml for production
execution:
  parallel_tasks: 50
  timeout_seconds: 600
  retry_attempts: 5

monitoring:
  log_level: WARNING
  metrics_enabled: true

error_handling:
  circuit_breaker_threshold: 10
  fallback_enabled: true"""
    
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

def test_configuration_lines_212_224_18():
    """Test YAML snippet from docs/user_guide/configuration.rst lines 212-224."""
    import yaml
    
    yaml_content = """# models.yaml for limited resources
models:
  # Only small, efficient models
  - source: ollama
    name: llama3.2:1b
    expertise: [general, fast]
    size: 1b

  - source: ollama
    name: phi-3-mini:3.8b
    expertise: [reasoning, compact]
    size: 3.8b"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.asyncio
async def test_configuration_lines_230_242_19():
    """Test YAML pipeline from docs/user_guide/configuration.rst lines 230-242."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """# orchestrator.yaml for high performance
execution:
  parallel_tasks: 100
  use_gpu: true

resources:
  max_memory_mb: 65536
  gpu_memory_fraction: 0.9

cache:
  backend: redis
  redis_url: redis://localhost:6379"""
    
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

def test_model_configuration_lines_16_61_20():
    """Test YAML snippet from docs/user_guide/model_configuration.rst lines 16-61."""
    import yaml
    
    yaml_content = """models:
  # Ollama models (automatically installed if not present)
  - source: ollama
    name: gemma2:27b
    expertise:
      - general
      - reasoning
      - analysis
    size: 27b

  - source: ollama
    name: codellama:7b
    expertise:
      - code
      - programming
    size: 7b

  # HuggingFace models (automatically downloaded)
  - source: huggingface
    name: microsoft/phi-2
    expertise:
      - reasoning
      - code
    size: 2.7b

  # Cloud models (require API keys)
  - source: openai
    name: gpt-4o
    expertise:
      - general
      - reasoning
      - code
      - analysis
      - vision
    size: 1760b

defaults:
  expertise_preferences:
    code: codellama:7b
    reasoning: gemma2:27b
    fast: llama3.2:1b
  fallback_chain:
    - gemma2:27b
    - llama3.2:1b
    - TinyLlama/TinyLlama-1.1B-Chat-v1.0"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_model_configuration_lines_104_112_21():
    """Test Python import from docs/user_guide/model_configuration.rst lines 104-112."""
    # Test imports
    try:
        exec("""import orchestrator as orc

# This registers models but doesn't download them yet
registry = orc.init_models()

# Models are downloaded only when first used by a pipeline
pipeline = orc.compile("my_pipeline.yaml")
result = pipeline.run()  # Model downloads happen here if needed""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_model_configuration_lines_130_133_22():
    """Test YAML snippet from docs/user_guide/model_configuration.rst lines 130-133."""
    import yaml
    
    yaml_content = """- source: huggingface
  name: microsoft/Phi-3.5-mini-instruct
  expertise: [reasoning, code]"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.asyncio
async def test_model_configuration_lines_156_162_23():
    """Test YAML pipeline from docs/user_guide/model_configuration.rst lines 156-162."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """steps:
  - id: summarize
    action: generate_text
    parameters:
      prompt: "Summarize this text..."
    requires_model: gemma2:27b  # Use specific model"""
    
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

@pytest.mark.asyncio
async def test_model_configuration_lines_170_178_24():
    """Test YAML pipeline from docs/user_guide/model_configuration.rst lines 170-178."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """steps:
  - id: generate_code
    action: generate_text
    parameters:
      prompt: "Write a Python function..."
    requires_model:
      expertise: code
      min_size: 7b  # At least 7B parameters"""
    
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

@pytest.mark.asyncio
async def test_model_configuration_lines_186_196_25():
    """Test YAML pipeline from docs/user_guide/model_configuration.rst lines 186-196."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """steps:
  - id: analyze
    action: analyze
    parameters:
      content: "{input_data}"
    requires_model:
      expertise:
        - reasoning
        - analysis
      min_size: 20b"""
    
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

@pytest.mark.asyncio
async def test_model_configuration_lines_204_240_26():
    """Test YAML pipeline from docs/user_guide/model_configuration.rst lines 204-240."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """id: multi_model_pipeline
name: Multi-Model Processing Pipeline

inputs:
  - name: topic
    type: string

steps:
  # Fast task with small model
  - id: quick_check
    action: generate_text
    parameters:
      prompt: "Is this topic related to programming: {topic}?"
    requires_model:
      expertise: fast
      min_size: 0  # Any size

  # Code generation with specialized model
  - id: code_example
    action: generate_text
    parameters:
      prompt: "Generate example code for: {topic}"
    requires_model:
      expertise: code
      min_size: 7b
    dependencies: [quick_check]

  # Complex reasoning with large model
  - id: deep_analysis
    action: analyze
    parameters:
      content: "{topic} with code: {code_example.result}"
    requires_model:
      expertise: [reasoning, analysis]
      min_size: 27b
    dependencies: [code_example]"""
    
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

def test_model_configuration_lines_280_291_27():
    """Test Python import from docs/user_guide/model_configuration.rst lines 280-291."""
    # Test imports
    try:
        exec("""import orchestrator as orc

# Initialize and list available models
registry = orc.init_models()
print("Available models:")
for model_key in registry.list_models():
    print(f"  - {model_key}")

# Run pipeline and check model selection
pipeline = orc.compile("pipeline.yaml")
result = pipeline.run(topic="AI agents")""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_model_configuration_lines_296_299_28():
    """Test text snippet from docs/user_guide/model_configuration.rst lines 296-299."""
    # Snippet type 'text' not yet supported for testing
    pytest.skip("Snippet type 'text' not yet supported")

def test_model_configuration_lines_337_338_29():
    """Test text snippet from docs/user_guide/model_configuration.rst lines 337-338."""
    # Snippet type 'text' not yet supported for testing
    pytest.skip("Snippet type 'text' not yet supported")
