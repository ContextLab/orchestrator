"""Tests for documentation code snippets - Batch 9."""
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


def test_installation_lines_106_114_0():
    """Test bash snippet from docs_sphinx/installation.rst lines 106-114."""
    import subprocess
    import tempfile
    import os
    
    bash_content = r"""# Large model for complex tasks
ollama pull gemma2:27b

# Small model for simple tasks
ollama pull llama3.2:1b

# Code-focused model
ollama pull codellama:7b"""
    
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

def test_installation_lines_119_124_1():
    """Test Python import from docs_sphinx/installation.rst lines 119-124."""
    # Test imports
    try:
        exec("""import orchestrator as orc

# Initialize and check models
registry = orc.init_models()
print(registry.list_models())""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_installation_lines_132_137_2():
    """Test bash snippet from docs_sphinx/installation.rst lines 132-137."""
    import subprocess
    import tempfile
    import os
    
    bash_content = r"""# Set environment variable
export HUGGINGFACE_TOKEN="your-token-here"

# Or create .env file
echo "HUGGINGFACE_TOKEN=your-token-here" > .env"""
    
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

def test_installation_lines_145_150_3():
    """Test bash snippet from docs_sphinx/installation.rst lines 145-150."""
    import subprocess
    import tempfile
    import os
    
    bash_content = r"""# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-...""""
    
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

def test_installation_lines_161_168_4():
    """Test bash snippet from docs_sphinx/installation.rst lines 161-168."""
    # Bash command snippet
    snippet_bash = r"""# Install Playwright
pip install playwright
playwright install chromium

# Or use Selenium
pip install selenium
# Download appropriate driver"""
    
    # Don't actually install packages in tests
    assert "pip install" in snippet_bash
    
    # Verify it's a valid pip command structure
    lines = snippet_bash.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {line}"

def test_installation_lines_181_186_5():
    """Test bash snippet from docs_sphinx/installation.rst lines 181-186."""
    # Bash command snippet
    snippet_bash = r"""# For advanced data processing
pip install pandas numpy scipy

# For data validation
pip install pydantic jsonschema"""
    
    # Don't actually install packages in tests
    assert "pip install" in snippet_bash
    
    # Verify it's a valid pip command structure
    lines = snippet_bash.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {line}"

def test_installation_lines_194_214_6():
    """Test YAML snippet from docs_sphinx/installation.rst lines 194-214."""
    import yaml
    
    yaml_content = """# Model preferences
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
  connection: "postgresql://user:pass@localhost/orchestrator""""
    
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
    
    bash_content = r"""# Core settings
export ORCHESTRATOR_HOME="$HOME/.orchestrator"
export ORCHESTRATOR_LOG_LEVEL="INFO"

# Model settings
export ORCHESTRATOR_MODEL_TIMEOUT="300"
export ORCHESTRATOR_MODEL_RETRIES="3"

# Tool settings
export ORCHESTRATOR_TOOL_TIMEOUT="60"
export ORCHESTRATOR_MCP_AUTO_START="true""""
    
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

def test_installation_lines_241_266_8():
    """Test Python import from docs_sphinx/installation.rst lines 241-266."""
    # Test imports
    try:
        exec("""import orchestrator as orc

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
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_installation_lines_277_278_9():
    """Test text snippet from docs_sphinx/installation.rst lines 277-278."""
    # Snippet type 'text' not yet supported for testing
    pytest.skip("Snippet type 'text' not yet supported")

def test_installation_lines_285_286_10():
    """Test text snippet from docs_sphinx/installation.rst lines 285-286."""
    # Snippet type 'text' not yet supported for testing
    pytest.skip("Snippet type 'text' not yet supported")

def test_installation_lines_293_294_11():
    """Test text snippet from docs_sphinx/installation.rst lines 293-294."""
    # Snippet type 'text' not yet supported for testing
    pytest.skip("Snippet type 'text' not yet supported")

def test_installation_lines_299_301_12():
    """Test bash snippet from docs_sphinx/installation.rst lines 299-301."""
    import subprocess
    import tempfile
    import os
    
    bash_content = r"""mkdir -p ~/.orchestrator
chmod 755 ~/.orchestrator"""
    
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
async def test_quickstart_lines_19_59_13():
    """Test YAML pipeline from docs_sphinx/quickstart.rst lines 19-59."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """name: topic-summarizer
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
      content: "$results.summarize""""
    
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

def test_quickstart_lines_67_87_14():
    """Test Python import from docs_sphinx/quickstart.rst lines 67-87."""
    # Test imports
    try:
        exec("""import orchestrator as orc

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
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

@pytest.mark.asyncio
async def test_quickstart_lines_107_179_15():
    """Test YAML pipeline from docs_sphinx/quickstart.rst lines 107-179."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """name: research-report-generator
description: Generate comprehensive research reports with citations

inputs:
  topic:
    type: string
    required: true
  focus_areas:
    type: array
    description: Specific areas to focus on
    default: []

outputs:
  report_pdf:
    type: string
    value: "reports/{{ inputs.topic }}_report.pdf"

steps:
  # Web search for recent information
  - id: search_recent
    action: search_web
    parameters:
      query: "{{ inputs.topic }} 2024 latest developments"
      max_results: 10

  # Search academic sources
  - id: search_academic
    action: search_web
    parameters:
      query: "{{ inputs.topic }} research papers scholarly"
      max_results: 5

  # Compile all sources
  - id: compile_sources
    action: compile_markdown
    parameters:
      sources:
        - "$results.search_recent"
        - "$results.search_academic"
      include_citations: true

  # Generate the report
  - id: write_report
    action: generate_report
    parameters:
      research: "$results.compile_sources"
      topic: "{{ inputs.topic }}"
      focus_areas: "{{ inputs.focus_areas }}"
      style: "academic"
      sections:
        - "Executive Summary"
        - "Introduction"
        - "Current State"
        - "Recent Developments"
        - "Future Outlook"
        - "Conclusions"

  # Quality check
  - id: validate
    action: validate_report
    parameters:
      report: "$results.write_report"
      checks:
        - completeness
        - citation_accuracy
        - readability

  # Generate PDF
  - id: create_pdf
    action: "!pandoc -o {{ outputs.report_pdf }} --pdf-engine=xelatex"
    parameters:
      input: "$results.write_report""""
    
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

def test_quickstart_lines_192_201_16():
    """Test YAML snippet from docs_sphinx/quickstart.rst lines 192-201."""
    import yaml
    
    yaml_content = """# Web search
- action: search_web
  parameters:
    query: "your search query"

# Scrape webpage
- action: scrape_page
  parameters:
    url: "https://example.com""""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_quickstart_lines_206_218_17():
    """Test YAML snippet from docs_sphinx/quickstart.rst lines 206-218."""
    import yaml
    
    yaml_content = """# Run shell commands (prefix with !)
- action: "!ls -la"

# File operations
- action: read_file
  parameters:
    path: "data.txt"

- action: write_file
  parameters:
    path: "output.txt"
    content: "Your content""""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_quickstart_lines_223_238_18():
    """Test YAML snippet from docs_sphinx/quickstart.rst lines 223-238."""
    import yaml
    
    yaml_content = """# Process data
- action: transform_data
  parameters:
    input: "$results.previous_step"
    operations:
      - type: filter
        condition: "value > 100"

# Validate data
- action: validate_data
  parameters:
    data: "$results.data"
    schema:
      type: object
      required: ["name", "value"]"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.asyncio
async def test_quickstart_lines_246_254_19():
    """Test YAML pipeline from docs_sphinx/quickstart.rst lines 246-254."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """steps:
  - id: analyze
    action: analyze_data
    parameters:
      data: "$results.fetch"
      method: <AUTO>Choose best analysis method based on data type</AUTO>
      visualization: <AUTO>Determine if visualization would be helpful</AUTO>
      depth: <AUTO>Set analysis depth (shallow/medium/deep)</AUTO>"""
    
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
async def test_quickstart_lines_264_288_20():
    """Test YAML pipeline from docs_sphinx/quickstart.rst lines 264-288."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """name: composite-pipeline

imports:
  - common/data_fetcher.yaml as fetcher
  - common/validator.yaml as validator

steps:
  # Use imported pipeline
  - id: fetch_data
    pipeline: fetcher
    parameters:
      source: "api"

  # Local step
  - id: process
    action: process_data
    parameters:
      data: "$results.fetch_data"

  # Use another import
  - id: validate
    pipeline: validator
    parameters:
      data: "$results.process""""
    
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
async def test_quickstart_lines_296_309_21():
    """Test YAML pipeline from docs_sphinx/quickstart.rst lines 296-309."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """steps:
  - id: risky_operation
    action: fetch_external_data
    parameters:
      url: "{{ inputs.data_source }}"
    error_handling:
      retry:
        max_attempts: 3
        backoff: exponential
      fallback:
        action: use_cached_data
        parameters:
          cache_key: "{{ inputs.topic }}""""
    
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

def test_quickstart_lines_317_332_22():
    """Test Python import from docs_sphinx/quickstart.rst lines 317-332."""
    # Test imports
    try:
        exec("""import logging
import orchestrator as orc

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Compile with debug flag
pipeline = orc.compile("pipeline.yaml", debug=True)

# Run with verbose output
result = pipeline.run(
    topic="test",
    _verbose=True,
    _step_callback=lambda step: print(f"Executing: {step.id}")
)""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_tool_reference_lines_48_100_23():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 48-100."""
    import yaml
    
    yaml_content = """# Web search
- id: search
  action: search_web
  parameters:
    query: "orchestrator framework tutorial"    # Required: Search query
    max_results: 10                            # Optional: Number of results (default: 10)
    search_engine: "google"                    # Optional: google|bing|duckduckgo (default: google)
    include_snippets: true                     # Optional: Include text snippets (default: true)
    region: "us"                              # Optional: Region code (default: us)
    language: "en"                            # Optional: Language code (default: en)
    safe_search: "moderate"                   # Optional: off|moderate|strict (default: moderate)

# Scrape webpage
- id: scrape
  action: scrape_page
  parameters:
    url: "https://example.com/article"        # Required: URL to scrape
    selectors:                                # Optional: CSS selectors to extract
      title: "h1.main-title"
      content: "div.article-body"
      author: "span.author-name"
    wait_for: "div.content-loaded"            # Optional: Wait for element
    timeout: 30                               # Optional: Timeout in seconds (default: 30)
    javascript: true                          # Optional: Execute JavaScript (default: true)
    clean_html: true                          # Optional: Clean extracted HTML (default: true)

# Take screenshot
- id: screenshot
  action: screenshot_page
  parameters:
    url: "https://example.com"                # Required: URL to screenshot
    full_page: true                           # Optional: Capture full page (default: false)
    width: 1920                               # Optional: Viewport width (default: 1920)
    height: 1080                              # Optional: Viewport height (default: 1080)
    wait_for: "img"                           # Optional: Wait for element
    output_path: "screenshots/page.png"       # Optional: Save path

# Interact with page
- id: interact
  action: interact_with_page
  parameters:
    url: "https://example.com/form"           # Required: URL to interact with
    actions:                                  # Required: List of interactions
      - type: "fill"
        selector: "#username"
        value: "testuser"
      - type: "click"
        selector: "#submit-button"
      - type: "wait"
        duration: 2000
      - type: "extract"
        selector: ".result""""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.asyncio
async def test_tool_reference_lines_105_138_24():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 105-138."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """name: web-research-pipeline
description: Comprehensive web research with validation

steps:
  # Search for information
  - id: search_topic
    action: search_web
    parameters:
      query: "{{ inputs.topic }} latest news 2024"
      max_results: 20
      search_engine: "google"

  # Scrape top results
  - id: scrape_articles
    for_each: "{{ results.search_topic.results[:5] }}"
    as: result
    action: scrape_page
    parameters:
      url: "{{ result.url }}"
      selectors:
        title: "h1, h2.article-title"
        content: "main, article, div.content"
        date: "time, .date, .published"
      clean_html: true

  # Take screenshots for reference
  - id: capture_pages
    for_each: "{{ results.search_topic.results[:3] }}"
    as: result
    action: screenshot_page
    parameters:
      url: "{{ result.url }}"
      output_path: "research/{{ inputs.topic }}/{{ loop.index }}.png""""
    
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

def test_tool_reference_lines_154_189_25():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 154-189."""
    import yaml
    
    yaml_content = """# Quick search
- id: search
  action: quick_search
  parameters:
    query: "machine learning basics"          # Required: Search query
    max_results: 5                           # Optional: Result count (default: 10)
    format: "json"                           # Optional: json|text (default: json)

# News search
- id: news
  action: search_news
  parameters:
    query: "AI breakthroughs"                # Required: Search query
    date_range: "last_week"                  # Optional: last_day|last_week|last_month|last_year
    sources: ["reuters", "techcrunch"]       # Optional: Preferred sources
    sort_by: "relevance"                     # Optional: relevance|date (default: relevance)

# Academic search
- id: academic
  action: search_academic
  parameters:
    query: "quantum computing"               # Required: Search query
    databases: ["arxiv", "pubmed"]          # Optional: Databases to search
    year_range: "2020-2024"                 # Optional: Year range
    peer_reviewed: true                      # Optional: Only peer-reviewed (default: false)

# Image search
- id: images
  action: search_images
  parameters:
    query: "data visualization examples"     # Required: Search query
    max_results: 10                         # Optional: Number of images
    size: "large"                           # Optional: small|medium|large|any
    type: "photo"                           # Optional: photo|clipart|lineart|any
    license: "creative_commons"             # Optional: License filter"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_tool_reference_lines_207_233_26():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 207-233."""
    import yaml
    
    yaml_content = """# Direct command execution
- id: list_files
  action: "!ls -la /data"

# Command with parameters
- id: run_command
  action: execute_command
  parameters:
    command: "python analyze.py"              # Required: Command to execute
    arguments: ["--input", "data.csv"]       # Optional: Command arguments
    working_dir: "/project"                  # Optional: Working directory
    environment:                             # Optional: Environment variables
      PYTHONPATH: "/project/lib"
      DEBUG: "true"
    timeout: 300                             # Optional: Timeout in seconds (default: 60)
    capture_output: true                     # Optional: Capture stdout/stderr (default: true)
    shell: true                              # Optional: Use shell execution (default: true)

# Run script file
- id: run_analysis
  action: run_script
  parameters:
    script_path: "scripts/analyze.sh"        # Required: Path to script
    arguments: ["{{ inputs.data_file }}"]    # Optional: Script arguments
    interpreter: "bash"                      # Optional: bash|python|node (default: auto-detect)
    working_dir: "{{ execution.temp_dir }}"  # Optional: Working directory"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.asyncio
async def test_tool_reference_lines_238_282_27():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 238-282."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """name: data-processing-automation
description: Automated data processing with shell commands

steps:
  # Setup environment
  - id: setup
    action: "!mkdir -p output/{{ inputs.project_name }}"

  # Download data
  - id: download
    action: execute_command
    parameters:
      command: "wget"
      arguments:
        - "-O"
        - "data/raw_data.csv"
        - "{{ inputs.data_url }}"
      timeout: 600

  # Process with Python
  - id: process
    action: execute_command
    parameters:
      command: "python"
      arguments:
        - "scripts/process_data.py"
        - "--input"
        - "data/raw_data.csv"
        - "--output"
        - "output/{{ inputs.project_name }}/processed.csv"
      environment:
        DATA_QUALITY: "high"
        PROCESSING_MODE: "{{ inputs.mode }}"

  # Generate report with R
  - id: report
    action: "!Rscript reports/generate_report.R output/{{ inputs.project_name }}/processed.csv"

  # Package results
  - id: package
    action: execute_command
    parameters:
      command: "tar"
      arguments: ["-czf", "{{ outputs.package }}", "output/{{ inputs.project_name }}"]"""
    
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

def test_tool_reference_lines_302_367_28():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 302-367."""
    import yaml
    
    yaml_content = """# Read file
- id: read_config
  action: read_file
  parameters:
    path: "config/settings.json"             # Required: File path
    encoding: "utf-8"                        # Optional: File encoding (default: utf-8)
    parse: true                              # Optional: Parse JSON/YAML (default: false)

# Write file
- id: save_results
  action: write_file
  parameters:
    path: "output/results.json"              # Required: File path
    content: "{{ results.analysis | json }}" # Required: Content to write
    encoding: "utf-8"                        # Optional: File encoding (default: utf-8)
    create_dirs: true                        # Optional: Create parent dirs (default: true)
    overwrite: true                          # Optional: Overwrite existing (default: false)

# Copy file
- id: backup
  action: copy_file
  parameters:
    source: "data/important.db"              # Required: Source path
    destination: "backup/important_{{ execution.timestamp }}.db"  # Required: Destination
    overwrite: false                         # Optional: Overwrite existing (default: false)

# Move file
- id: archive
  action: move_file
  parameters:
    source: "temp/processed.csv"             # Required: Source path
    destination: "archive/2024/processed.csv" # Required: Destination
    create_dirs: true                        # Optional: Create parent dirs (default: true)

# Delete file
- id: cleanup
  action: delete_file
  parameters:
    path: "temp/*"                           # Required: Path or pattern
    recursive: true                          # Optional: Delete recursively (default: false)
    force: false                             # Optional: Force deletion (default: false)

# List directory
- id: scan_files
  action: list_directory
  parameters:
    path: "data/"                            # Required: Directory path
    pattern: "*.csv"                         # Optional: File pattern
    recursive: true                          # Optional: Search subdirs (default: false)
    include_hidden: false                    # Optional: Include hidden files (default: false)
    details: true                            # Optional: Include file details (default: false)

# Create directory
- id: setup_dirs
  action: create_directory
  parameters:
    path: "output/{{ inputs.project }}/data" # Required: Directory path
    parents: true                            # Optional: Create parents (default: true)
    exist_ok: true                           # Optional: Ok if exists (default: true)

# Check existence
- id: check_file
  action: file_exists
  parameters:
    path: "config/custom.yaml"               # Required: Path to check"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.asyncio
async def test_tool_reference_lines_372_416_29():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 372-416."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """name: file-organization-pipeline
description: Organize and process files automatically

steps:
  # Check for existing data
  - id: check_existing
    action: file_exists
    parameters:
      path: "data/current_dataset.csv"

  # Backup if exists
  - id: backup
    condition: "{{ results.check_existing }}"
    action: copy_file
    parameters:
      source: "data/current_dataset.csv"
      destination: "backups/dataset_{{ execution.timestamp }}.csv"

  # Read configuration
  - id: read_config
    action: read_file
    parameters:
      path: "config/processing.yaml"
      parse: true

  # Process files based on config
  - id: process_files
    for_each: "{{ results.read_config.file_patterns }}"
    as: pattern
    action: list_directory
    parameters:
      path: "{{ pattern.directory }}"
      pattern: "{{ pattern.glob }}"
      recursive: true

  # Organize by type
  - id: organize
    for_each: "{{ results.process_files | flatten }}"
    as: file
    action: move_file
    parameters:
      source: "{{ file.path }}"
      destination: "organized/{{ file.extension }}/{{ file.name }}"
      create_dirs: true"""
    
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
