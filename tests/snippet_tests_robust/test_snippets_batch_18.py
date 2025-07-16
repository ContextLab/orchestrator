"""Tests for documentation code snippets - Batch 18 (Robust)."""
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
async def test_quickstart_lines_107_179_0():
    """Test YAML pipeline from docs_sphinx/quickstart.rst lines 107-179."""
    import yaml
    import orchestrator
    
    yaml_content = ("""name: research-report-generator
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
      input: "$results.write_report"""")
    
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

def test_quickstart_lines_192_201_1():
    """Test YAML snippet from docs_sphinx/quickstart.rst lines 192-201."""
    import yaml
    
    yaml_content = ("""# Web search
- action: search_web
  parameters:
    query: "your search query"

# Scrape webpage
- action: scrape_page
  parameters:
    url: "https://example.com"""")
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_quickstart_lines_206_218_2():
    """Test YAML snippet from docs_sphinx/quickstart.rst lines 206-218."""
    import yaml
    
    yaml_content = ("""# Run shell commands (prefix with !)
- action: "!ls -la"

# File operations
- action: read_file
  parameters:
    path: "data.txt"

- action: write_file
  parameters:
    path: "output.txt"
    content: "Your content"""")
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_quickstart_lines_223_238_3():
    """Test YAML snippet from docs_sphinx/quickstart.rst lines 223-238."""
    import yaml
    
    yaml_content = ("""# Process data
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
      required: ["name", "value"]""")
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_quickstart_lines_246_254_4():
    """Test YAML pipeline from docs_sphinx/quickstart.rst lines 246-254."""
    import yaml
    import orchestrator
    
    yaml_content = ("""steps:
  - id: analyze
    action: analyze_data
    parameters:
      data: "$results.fetch"
      method: <AUTO>Choose best analysis method based on data type</AUTO>
      visualization: <AUTO>Determine if visualization would be helpful</AUTO>
      depth: <AUTO>Set analysis depth (shallow/medium/deep)</AUTO>""")
    
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
async def test_quickstart_lines_264_288_5():
    """Test YAML pipeline from docs_sphinx/quickstart.rst lines 264-288."""
    import yaml
    import orchestrator
    
    yaml_content = ("""name: composite-pipeline

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
      data: "$results.process"""")
    
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
async def test_quickstart_lines_296_309_6():
    """Test YAML pipeline from docs_sphinx/quickstart.rst lines 296-309."""
    import yaml
    import orchestrator
    
    yaml_content = ("""steps:
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
          cache_key: "{{ inputs.topic }}"""")
    
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
async def test_quickstart_lines_317_332_7():
    """Test orchestrator code from docs_sphinx/quickstart.rst lines 317-332."""
    # Enable debug mode for detailed execution logs:
    
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
async def test_tool_reference_lines_48_100_8():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 48-100."""
    import yaml
    import orchestrator
    
    yaml_content = ("""# Web search
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
        selector: ".result"""")
    
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
async def test_tool_reference_lines_105_138_9():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 105-138."""
    import yaml
    import orchestrator
    
    yaml_content = ("""name: web-research-pipeline
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
      output_path: "research/{{ inputs.topic }}/{{ loop.index }}.png"""")
    
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
async def test_tool_reference_lines_154_189_10():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 154-189."""
    import yaml
    import orchestrator
    
    yaml_content = ("""# Quick search
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
    license: "creative_commons"             # Optional: License filter""")
    
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
async def test_tool_reference_lines_207_233_11():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 207-233."""
    import yaml
    import orchestrator
    
    yaml_content = ("""# Direct command execution
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
    working_dir: "{{ execution.temp_dir }}"  # Optional: Working directory""")
    
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
async def test_tool_reference_lines_238_282_12():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 238-282."""
    import yaml
    import orchestrator
    
    yaml_content = ("""name: data-processing-automation
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
      arguments: ["-czf", "{{ outputs.package }}", "output/{{ inputs.project_name }}"]""")
    
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
async def test_tool_reference_lines_302_367_13():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 302-367."""
    import yaml
    import orchestrator
    
    yaml_content = ("""# Read file
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
    path: "config/custom.yaml"               # Required: Path to check""")
    
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
async def test_tool_reference_lines_372_416_14():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 372-416."""
    import yaml
    import orchestrator
    
    yaml_content = ("""name: file-organization-pipeline
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
      create_dirs: true""")
    
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
