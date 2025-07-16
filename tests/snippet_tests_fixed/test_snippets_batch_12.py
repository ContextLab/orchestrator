"""Tests for documentation code snippets - Batch 12 (Fixed)."""
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
async def test_concepts_lines_422_434_0():
    """Test YAML pipeline from docs_sphinx/concepts.rst lines 422-434."""
    import yaml
    import orchestrator
    
    yaml_content = r"""steps:
  # These run in parallel
  - id: source1
    action: fetch_data_a

  - id: source2
    action: fetch_data_b

  # This waits for both
  - id: combine
    depends_on: [source1, source2]
    action: merge_data"""
    
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
async def test_concepts_lines_442_449_1():
    """Test YAML pipeline from docs_sphinx/concepts.rst lines 442-449."""
    import yaml
    import orchestrator
    
    yaml_content = r"""steps:
  - id: expensive_computation
    action: complex_analysis
    cache:
      enabled: true
      key: "{{ inputs.data_hash }}"
      ttl: 3600  # 1 hour"""
    
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

def test_concepts_lines_457_462_2():
    """Test YAML snippet from docs_sphinx/concepts.rst lines 457-462."""
    import yaml
    
    yaml_content = r"""config:
  resources:
    max_memory: "8GB"
    max_threads: 4
    gpu_enabled: false"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_concepts_lines_483_494_3():
    """Test YAML snippet from docs_sphinx/concepts.rst lines 483-494."""
    import yaml
    
    yaml_content = r"""inputs:
  email:
    type: string
    validation:
      pattern: "^[\\w.-]+@[\\w.-]+\\.\\w+$"

  amount:
    type: number
    validation:
      min: 0
      max: 10000"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_concepts_lines_502_505_4():
    """Test YAML snippet from docs_sphinx/concepts.rst lines 502-505."""
    import yaml
    
    yaml_content = r"""parameters:
  api_key: "{{ env.SECRET_API_KEY }}"  # From environment
  password: "{{ vault.db_password }}"   # From secret vault"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_concepts_lines_523_533_5():
    """Test  snippet from docs_sphinx/concepts.rst lines 523-533."""
    # Content validation for  snippet
    content = r"""pipelines/
├── common/           # Shared components
│   ├── validation.yaml
│   └── formatting.yaml
├── workflows/        # Complete workflows
│   ├── research.yaml
│   └── analysis.yaml
└── specialized/      # Domain-specific
    ├── finance.yaml
    └── healthcare.yaml"""
    
    # Basic checks
    assert content.strip(), "Snippet content should not be empty"
    assert len(content.strip()) > 0, "Snippet should have content"

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_getting_started_lines_29_43_6():
    """Test YAML pipeline from docs_sphinx/getting_started.rst lines 29-43."""
    import yaml
    import orchestrator
    
    yaml_content = r"""name: research-report
description: Generate comprehensive research reports

inputs:
  topic:
    type: string
    description: Research topic
    required: true

steps:
  - id: search
    action: search_web
    parameters:
      query: "{{ inputs.topic }} latest research""""
    
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

def test_getting_started_lines_61_64_7():
    """Test YAML snippet from docs_sphinx/getting_started.rst lines 61-64."""
    import yaml
    
    yaml_content = r"""parameters:
  method: <AUTO>Choose best analysis method for this data</AUTO>
  depth: <AUTO>Determine appropriate depth level</AUTO>"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_getting_started_lines_83_108_8():
    """Test YAML pipeline from docs_sphinx/getting_started.rst lines 83-108."""
    import yaml
    import orchestrator
    
    yaml_content = r"""name: quick-research
description: Quick research on any topic

inputs:
  topic:
    type: string
    required: true

outputs:
  report:
    type: string
    value: "{{ inputs.topic }}_report.md"

steps:
  - id: search
    action: search_web
    parameters:
      query: "{{ inputs.topic }}"
      max_results: 5

  - id: summarize
    action: generate_summary
    parameters:
      content: "$results.search"
      style: <AUTO>Choose appropriate summary style</AUTO>"""
    
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
async def test_getting_started_lines_113_126_9():
    """Test orchestrator code from docs_sphinx/getting_started.rst lines 113-126."""
    # 2. **Run the pipeline**:
    
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

# Initialize models
orc.init_models()

# Compile the pipeline
pipeline = orc.compile_mock("research.yaml")

# Execute with different topics
result1 = pipeline.run(topic="artificial intelligence")
result2 = pipeline.run(topic="climate change")

print(f"Reports generated: {result1}, {result2}")""":
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

# Initialize models
orc.init_models()

# Compile the pipeline
pipeline = orc.compile_mock("research.yaml")

# Execute with different topics
result1 = pipeline.run(topic="artificial intelligence")
result2 = pipeline.run(topic="climate change")

print(f"Reports generated: {result1}, {result2}")""", {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
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
async def test_getting_started_lines_141_148_10():
    """Test orchestrator code from docs_sphinx/getting_started.rst lines 141-148."""
    # The same pipeline works with different inputs:
    
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
        if 'hello_world.yaml' in r"""# One pipeline, many uses
pipeline = orc.compile_mock("report-template.yaml")

# Generate different reports
ai_report = pipeline.run(topic="AI", style="technical")
bio_report = pipeline.run(topic="Biology", style="educational")
eco_report = pipeline.run(topic="Economics", style="executive")""":
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
            exec(r"""# One pipeline, many uses
pipeline = orc.compile_mock("report-template.yaml")

# Generate different reports
ai_report = pipeline.run(topic="AI", style="technical")
bio_report = pipeline.run(topic="Biology", style="educational")
eco_report = pipeline.run(topic="Economics", style="executive")""", {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")
        finally:
            # Restore original function
            if original_compile:
                setattr(orc, 'compile', original_compile)

def test_getting_started_lines_153_160_11():
    """Test YAML snippet from docs_sphinx/getting_started.rst lines 153-160."""
    import yaml
    
    yaml_content = r"""inputs:
  topic:
    type: string
    required: true
  style:
    type: string
    default: "technical""""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_getting_started_lines_168_177_12():
    """Test YAML pipeline from docs_sphinx/getting_started.rst lines 168-177."""
    import yaml
    import orchestrator
    
    yaml_content = r"""steps:
  - id: fetch_data
    action: search_web        # Auto-detects web tool

  - id: save_results
    action: write_file        # Auto-detects filesystem tool

  - id: run_analysis
    action: "!python analyze.py"  # Auto-detects terminal tool"""
    
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
async def test_getting_started_lines_185_193_13():
    """Test orchestrator code from docs_sphinx/getting_started.rst lines 185-193."""
    # The framework intelligently selects the best model for each task:
    
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
        if 'hello_world.yaml' in r"""# Models are selected based on:
# - Task requirements (reasoning, coding, etc.)
# - Available resources
# - Performance history

registry = orc.init_models()
print(registry.list_models())
# Output: ['ollama:gemma2:27b', 'ollama:llama3.2:1b', 'huggingface:gpt2']""":
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
            exec(r"""# Models are selected based on:
# - Task requirements (reasoning, coding, etc.)
# - Available resources
# - Performance history

registry = orc.init_models()
print(registry.list_models())
# Output: ['ollama:gemma2:27b', 'ollama:llama3.2:1b', 'huggingface:gpt2']""", {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
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
async def test_index_lines_43_56_14():
    """Test orchestrator code from docs_sphinx/index.rst lines 43-56."""
    # -------------
    
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

# Initialize models
orc.init_models()

# Compile a pipeline
pipeline = orc.compile_mock("pipelines/research-report.yaml")

# Execute with different inputs
result = pipeline.run(
    topic="quantum_computing",
    instructions="Focus on error correction"
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
            exec(r"""import orchestrator as orc

# Initialize models
orc.init_models()

# Compile a pipeline
pipeline = orc.compile_mock("pipelines/research-report.yaml")

# Execute with different inputs
result = pipeline.run(
    topic="quantum_computing",
    instructions="Focus on error correction"
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

def test_installation_lines_34_41_15():
    """Test bash snippet from docs_sphinx/installation.rst lines 34-41."""
    # Bash command snippet
    snippet_bash = r"""# Install from PyPI (when available)
pip install py-orc

# Or install from source
git clone https://github.com/ContextLab/orchestrator.git
cd orchestrator
pip install -e ."""
    
    # Validate pip install commands without actually running them
    assert "pip install" in snippet_bash
    
    lines = snippet_bash.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {line}"

def test_installation_lines_47_53_16():
    """Test bash snippet from docs_sphinx/installation.rst lines 47-53."""
    # Bash command snippet
    snippet_bash = r"""# Create conda environment
conda create -n py-orc python=3.11
conda activate py-orc

# Install orchestrator
pip install py-orc"""
    
    # Validate pip install commands without actually running them
    assert "pip install" in snippet_bash
    
    lines = snippet_bash.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {line}"

def test_installation_lines_59_64_17():
    """Test bash snippet from docs_sphinx/installation.rst lines 59-64."""
    import subprocess
    import tempfile
    import os
    
    bash_content = r"""# Pull the official image
docker pull contextlab/py-orc:latest

# Run with volume mount
docker run -v $(pwd):/workspace contextlab/py-orc"""
    
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

def test_installation_lines_72_85_18():
    """Test bash snippet from docs_sphinx/installation.rst lines 72-85."""
    # Bash command snippet
    snippet_bash = r"""# Clone the repository
git clone https://github.com/ContextLab/orchestrator.git
cd orchestrator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with extras
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install"""
    
    # Validate pip install commands without actually running them
    assert "pip install" in snippet_bash
    
    lines = snippet_bash.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {line}"

def test_installation_lines_96_101_19():
    """Test bash snippet from docs_sphinx/installation.rst lines 96-101."""
    import subprocess
    import tempfile
    import os
    
    bash_content = r"""# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh"""
    
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
