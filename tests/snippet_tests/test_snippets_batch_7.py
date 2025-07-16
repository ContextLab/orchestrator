"""Tests for documentation code snippets - Batch 7."""
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


def test_model_configuration_lines_348_349_0():
    """Test text snippet from docs/user_guide/model_configuration.rst lines 348-349."""
    # Snippet type 'text' not yet supported for testing
    pytest.skip("Snippet type 'text' not yet supported")

def test_model_configuration_lines_359_360_1():
    """Test text snippet from docs/user_guide/model_configuration.rst lines 359-360."""
    # Snippet type 'text' not yet supported for testing
    pytest.skip("Snippet type 'text' not yet supported")

def test_models_and_adapters_lines_34_50_2():
    """Test Python import from docs/user_guide/models_and_adapters.rst lines 34-50."""
    # Test imports
    try:
        exec("""import orchestrator as orc

# Initialize and discover available models
registry = orc.init_models()

# List all detected models
available_models = registry.list_models()
print("Available models:", available_models)

# Check specific model availability
if any("gemma2:27b" in model for model in available_models):
    print("Large Ollama model available")
elif any("llama3.2:1b" in model for model in available_models):
    print("Lightweight Ollama model available")
else:
    print("Using fallback models")""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_models_and_adapters_lines_60_67_3():
    """Test bash snippet from docs/user_guide/models_and_adapters.rst lines 60-67."""
    import subprocess
    import tempfile
    import os
    
    bash_content = r"""# Install Ollama
brew install ollama  # macOS
# or visit https://ollama.ai for other platforms

# Pull recommended models
ollama pull gemma2:27b    # Large model for complex tasks
ollama pull llama3.2:1b   # Lightweight fallback"""
    
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

def test_models_and_adapters_lines_74_75_4():
    """Test bash snippet from docs/user_guide/models_and_adapters.rst lines 74-75."""
    # Bash command snippet
    snippet_bash = r"""pip install transformers torch"""
    
    # Don't actually install packages in tests
    assert "pip install" in snippet_bash
    
    # Verify it's a valid pip command structure
    lines = snippet_bash.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {line}"

def test_models_and_adapters_lines_82_85_5():
    """Test bash snippet from docs/user_guide/models_and_adapters.rst lines 82-85."""
    import subprocess
    import tempfile
    import os
    
    bash_content = r"""export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="...""""
    
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

def test_models_and_adapters_lines_96_103_6():
    """Test Python import from docs/user_guide/models_and_adapters.rst lines 96-103."""
    # Test imports
    try:
        exec("""from orchestrator.models.openai_model import OpenAIModel

model = OpenAIModel(
    name="gpt-4o",
    api_key="your-api-key",
    model="gpt-4o"
)""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_models_and_adapters_lines_109_116_7():
    """Test Python import from docs/user_guide/models_and_adapters.rst lines 109-116."""
    # Test imports
    try:
        exec("""from orchestrator.models.anthropic_model import AnthropicModel

model = AnthropicModel(
    name="claude-3.5-sonnet",
    api_key="your-api-key",
    model="claude-3.5-sonnet"
)""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_models_and_adapters_lines_122_128_8():
    """Test Python import from docs/user_guide/models_and_adapters.rst lines 122-128."""
    # Test imports
    try:
        exec("""from orchestrator.models.huggingface_model import HuggingFaceModel

model = HuggingFaceModel(
    name="llama-3.2-3b",
    model_path="meta-llama/Llama-3.2-3B-Instruct"
)""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_models_and_adapters_lines_136_144_9():
    """Test Python import from docs/user_guide/models_and_adapters.rst lines 136-144."""
    # Test imports
    try:
        exec("""from orchestrator.models.model_registry import ModelRegistry

registry = ModelRegistry()
registry.register_model(gpt4_model)
registry.register_model(claude_model)

# Automatic selection based on task requirements
selected_model = registry.select_model(task)""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

@pytest.mark.asyncio
async def test_yaml_configuration_lines_26_44_10():
    """Test YAML pipeline from docs/user_guide/yaml_configuration.rst lines 26-44."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """id: my_pipeline
name: My Pipeline
description: A sample pipeline

tasks:
  - id: task1
    name: First Task
    action: generate_text
    parameters:
      prompt: "Hello, world!"

  - id: task2
    name: Second Task
    action: generate_text
    parameters:
      prompt: "Process this: {task1}"
    dependencies:
      - task1"""
    
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
async def test_yaml_configuration_lines_52_65_11():
    """Test YAML pipeline from docs/user_guide/yaml_configuration.rst lines 52-65."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """id: research_pipeline
name: Research Pipeline

context:
  topic: artificial intelligence
  depth: detailed

tasks:
  - id: research
    name: Research Task
    action: generate_text
    parameters:
      prompt: "Research {topic} with {depth} analysis""""
    
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
async def test_yaml_configuration_lines_73_80_12():
    """Test YAML pipeline from docs/user_guide/yaml_configuration.rst lines 73-80."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """tasks:
  - id: analysis
    name: Analysis Task
    action: <AUTO>
    parameters:
      data: {previous_task}
      model: <AUTO>"""
    
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

def test_index_lines_118_128_13():
    """Test Python import from docs_sphinx/api/index.rst lines 118-128."""
    # Test imports
    try:
        exec("""import orchestrator as orc

# Initialize models
registry = orc.init_models()

# Compile pipeline
pipeline = orc.compile("my_pipeline.yaml")

# Execute
result = pipeline.run(input_param="value")""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_index_lines_134_151_14():
    """Test Python import from docs_sphinx/api/index.rst lines 134-151."""
    # Test imports
    try:
        exec("""from orchestrator import Orchestrator
from orchestrator.core.control_system import MockControlSystem
from orchestrator.models.model_registry import ModelRegistry

# Create custom orchestrator
control_system = MockControlSystem()
orchestrator = Orchestrator(control_system=control_system)

# Use custom model registry
registry = ModelRegistry()
# ... configure models

# Compile with custom settings
pipeline = orchestrator.compile(
    yaml_content,
    config={"timeout": 3600}
)""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_index_lines_159_168_15():
    """Test Python import from docs_sphinx/api/index.rst lines 159-168."""
    # Test imports
    try:
        exec("""from typing import Dict, Any, List, Optional
from orchestrator import Pipeline, Task

def process_pipeline(
    pipeline: Pipeline,
    inputs: Dict[str, Any],
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    return pipeline.run(**inputs)""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_index_lines_198_215_16():
    """Test YAML snippet from docs_sphinx/api/index.rst lines 198-215."""
    import yaml
    
    yaml_content = """models:
  default: "ollama:gemma2:27b"
  fallback: "ollama:llama3.2:1b"
  timeout: 300

tools:
  mcp_port: 8000
  auto_start: true

execution:
  parallel: true
  checkpoint: true
  timeout: 3600

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s""""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.asyncio
async def test_index_lines_226_235_17():
    """Test Python snippet from docs_sphinx/api/index.rst lines 226-235."""
    # Models are loaded lazily and cached. For better performance:
    
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
        code = """# Initialize models once at startup
orc.init_models()

# Reuse compiled pipelines
pipeline = orc.compile("pipeline.yaml")

# Multiple executions reuse the same pipeline
for inputs in input_batches:
    result = pipeline.run(**inputs)"""
        
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
async def test_index_lines_243_253_18():
    """Test Python snippet from docs_sphinx/api/index.rst lines 243-253."""
    # Large pipelines and datasets can consume significant memory:
    
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
        code = """# Enable checkpointing for long pipelines
pipeline = orc.compile("pipeline.yaml", config={
    "checkpoint": True,
    "memory_limit": "8GB"
})

# Process data in batches
for batch in data_batches:
    result = pipeline.run(data=batch)
    # Results are automatically checkpointed"""
        
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

def test_index_lines_261_273_19():
    """Test Python import from docs_sphinx/api/index.rst lines 261-273."""
    # Test imports
    try:
        exec("""from orchestrator import CompilationError, ExecutionError

try:
    pipeline = orc.compile("pipeline.yaml")
    result = pipeline.run(input="value")
except CompilationError as e:
    print(f"Pipeline compilation failed: {e}")
    print(f"Error details: {e.details}")
except ExecutionError as e:
    print(f"Pipeline execution failed: {e}")
    print(f"Failed step: {e.step_id}")
    print(f"Error context: {e.context}")""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_index_lines_281_291_20():
    """Test Python import from docs_sphinx/api/index.rst lines 281-291."""
    # Test imports
    try:
        exec("""import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Compile with debug information
pipeline = orc.compile("pipeline.yaml", debug=True)

# Execute with verbose output
result = pipeline.run(input="value", _verbose=True)""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_index_lines_302_308_21():
    """Test Python import from docs_sphinx/api/index.rst lines 302-308."""
    # Test imports
    try:
        exec("""from orchestrator.core.control_system import ControlSystem

class MyControlSystem(ControlSystem):
    async def execute_task(self, task: Task, context: dict) -> dict:
        # Custom execution logic
        pass""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_index_lines_314_323_22():
    """Test Python import from docs_sphinx/api/index.rst lines 314-323."""
    # Test imports
    try:
        exec("""from orchestrator.tools.base import Tool

class MyTool(Tool):
    def __init__(self):
        super().__init__("my-tool", "Description")

    async def execute(self, **kwargs) -> dict:
        # Tool implementation
        pass""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_index_lines_329_335_23():
    """Test Python import from docs_sphinx/api/index.rst lines 329-335."""
    # Test imports
    try:
        exec("""from orchestrator.core.model import Model

class MyModel(Model):
    async def generate(self, prompt: str, **kwargs) -> str:
        # Model implementation
        pass""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_index_lines_343_356_24():
    """Test Python import from docs_sphinx/api/index.rst lines 343-356."""
    # Test imports
    try:
        exec("""import concurrent.futures

# Safe to use across threads
pipeline = orc.compile("pipeline.yaml")

def process_input(input_data):
    return pipeline.run(**input_data)

# Parallel execution
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_input, data)
              for data in input_datasets]
    results = [f.result() for f in futures]""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_index_lines_364_381_25():
    """Test Python import from docs_sphinx/api/index.rst lines 364-381."""
    # Test imports
    try:
        exec("""from orchestrator.testing import MockModel, TestRunner

def test_my_pipeline():
    # Use mock model for testing
    with MockModel() as mock:
        mock.set_response("test response")

        pipeline = orc.compile("test_pipeline.yaml")
        result = pipeline.run(input="test")

        assert result == "expected"

# Test runner for pipeline validation
runner = TestRunner("pipelines/")
runner.validate_all()  # Validates all YAML files
runner.test_compilation()  # Tests compilation
runner.run_smoke_tests()  # Basic execution tests""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

@pytest.mark.asyncio
async def test_concepts_lines_19_31_26():
    """Test YAML pipeline from docs_sphinx/concepts.rst lines 19-31."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """# One pipeline definition
name: research-pipeline

inputs:
  topic: { type: string, required: true }
  depth: { type: string, default: "medium" }

steps:
  - id: research
    action: search_web
    parameters:
      query: "{{ inputs.topic }}""""
    
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

def test_concepts_lines_55_62_27():
    """Test YAML snippet from docs_sphinx/concepts.rst lines 55-62."""
    import yaml
    
    yaml_content = """- id: unique_identifier        # Required: Unique name
  action: what_to_do           # Required: Action to perform
  description: "What it does"  # Optional: Human description
  parameters:                  # Optional: Input parameters
    key: value
  depends_on: [other_task]     # Optional: Dependencies
  condition: "when_to_run"     # Optional: Conditional execution"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.asyncio
async def test_concepts_lines_70_87_28():
    """Test YAML pipeline from docs_sphinx/concepts.rst lines 70-87."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """steps:
  - id: fetch_data
    action: download_file
    parameters:
      url: "{{ inputs.data_url }}"

  - id: process_data
    depends_on: [fetch_data]   # Runs after fetch_data
    action: transform_data
    parameters:
      data: "$results.fetch_data"

  - id: save_results
    depends_on: [process_data] # Runs after process_data
    action: write_file
    parameters:
      content: "$results.process_data""""
    
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

def test_concepts_lines_98_109_29():
    """Test YAML snippet from docs_sphinx/concepts.rst lines 98-109."""
    import yaml
    
    yaml_content = """# Access input values
query: "{{ inputs.search_term }}"

# Reference results from other tasks
data: "$results.previous_task"

# Use filters and functions
filename: "{{ inputs.name | slugify }}.pdf"

# Conditional expressions
mode: "{{ 'advanced' if inputs.premium else 'basic' }}""""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
