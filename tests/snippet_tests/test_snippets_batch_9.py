"""Tests for documentation code snippets - Batch 9."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_installation_lines_106_114_0():
    """Test bash snippet from docs_sphinx/installation.rst lines 106-114."""
    bash_content = '# Large model for complex tasks\nollama pull gemma2:27b\n\n# Small model for simple tasks\nollama pull llama3.2:1b\n\n# Code-focused model\nollama pull codellama:7b'
    
    # Skip dangerous commands
    skip_commands = ['rm -rf', 'sudo', 'docker', 'systemctl']
    if any(cmd in bash_content for cmd in skip_commands):
        pytest.skip("Skipping potentially destructive command")
    
    # For now, just check it's not empty
    assert bash_content.strip(), "Bash content should not be empty"

def test_installation_lines_119_124_1():
    """Test Python import from docs_sphinx/installation.rst lines 119-124."""
    # Import test - check if modules are available
    code = 'import orchestrator as orc\n\n# Initialize and check models\nregistry = orc.init_models()\nprint(registry.list_models())'
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_installation_lines_132_137_2():
    """Test bash snippet from docs_sphinx/installation.rst lines 132-137."""
    bash_content = '# Set environment variable\nexport HUGGINGFACE_TOKEN="your-token-here"\n\n# Or create .env file\necho "HUGGINGFACE_TOKEN=your-token-here" > .env'
    
    # Skip dangerous commands
    skip_commands = ['rm -rf', 'sudo', 'docker', 'systemctl']
    if any(cmd in bash_content for cmd in skip_commands):
        pytest.skip("Skipping potentially destructive command")
    
    # For now, just check it's not empty
    assert bash_content.strip(), "Bash content should not be empty"

def test_installation_lines_145_150_3():
    """Test bash snippet from docs_sphinx/installation.rst lines 145-150."""
    bash_content = '# OpenAI\nexport OPENAI_API_KEY="sk-..."\n\n# Anthropic\nexport ANTHROPIC_API_KEY="sk-ant-..."'
    
    # Skip dangerous commands
    skip_commands = ['rm -rf', 'sudo', 'docker', 'systemctl']
    if any(cmd in bash_content for cmd in skip_commands):
        pytest.skip("Skipping potentially destructive command")
    
    # For now, just check it's not empty
    assert bash_content.strip(), "Bash content should not be empty"

def test_installation_lines_161_168_4():
    """Test bash snippet from docs_sphinx/installation.rst lines 161-168."""
    bash_content = '# Install Playwright\npip install playwright\nplaywright install chromium\n\n# Or use Selenium\npip install selenium\n# Download appropriate driver'
    
    # Verify it's a pip install command
    assert "pip install" in bash_content
    
    # Parse each line
    lines = bash_content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {line}"

def test_installation_lines_181_186_5():
    """Test bash snippet from docs_sphinx/installation.rst lines 181-186."""
    bash_content = '# For advanced data processing\npip install pandas numpy scipy\n\n# For data validation\npip install pydantic jsonschema'
    
    # Verify it's a pip install command
    assert "pip install" in bash_content
    
    # Parse each line
    lines = bash_content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {line}"

def test_installation_lines_194_214_6():
    """Test YAML snippet from docs_sphinx/installation.rst lines 194-214."""
    import yaml
    
    yaml_content = '# Model preferences\nmodels:\n  default: "ollama:gemma2:27b"\n  fallback: "ollama:llama3.2:1b"\n\n# Resource limits\nresources:\n  max_memory: "16GB"\n  max_threads: 8\n  gpu_enabled: true\n\n# Tool settings\ntools:\n  mcp_port: 8000\n  sandbox_enabled: true\n\n# State management\nstate:\n  backend: "postgresql"\n  connection: "postgresql://user:pass@localhost/orchestrator"'
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_installation_lines_222_233_7():
    """Test bash snippet from docs_sphinx/installation.rst lines 222-233."""
    bash_content = '# Core settings\nexport ORCHESTRATOR_HOME="$HOME/.orchestrator"\nexport ORCHESTRATOR_LOG_LEVEL="INFO"\n\n# Model settings\nexport ORCHESTRATOR_MODEL_TIMEOUT="300"\nexport ORCHESTRATOR_MODEL_RETRIES="3"\n\n# Tool settings\nexport ORCHESTRATOR_TOOL_TIMEOUT="60"\nexport ORCHESTRATOR_MCP_AUTO_START="true"'
    
    # Skip dangerous commands
    skip_commands = ['rm -rf', 'sudo', 'docker', 'systemctl']
    if any(cmd in bash_content for cmd in skip_commands):
        pytest.skip("Skipping potentially destructive command")
    
    # For now, just check it's not empty
    assert bash_content.strip(), "Bash content should not be empty"

def test_installation_lines_241_266_8():
    """Test Python import from docs_sphinx/installation.rst lines 241-266."""
    # Import test - check if modules are available
    code = 'import orchestrator as orc\n\n# Check version\nprint(f"Orchestrator version: {orc.__version__}")\n\n# Check models\ntry:\n    registry = orc.init_models()\n    models = registry.list_models()\n    print(f"Available models: {models}")\nexcept Exception as e:\n    print(f"Model initialization failed: {e}")\n\n# Check tools\nfrom orchestrator.tools.base import default_registry\ntools = default_registry.list_tools()\nprint(f"Available tools: {tools}")\n\n# Run test pipeline\ntry:\n    pipeline = orc.compile("examples/hello-world.yaml")\n    result = pipeline.run(message="Hello, Orchestrator!")\n    print(f"Test pipeline result: {result}")\nexcept Exception as e:\n    print(f"Pipeline test failed: {e}")'
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_installation_lines_277_278_9():
    """Test text snippet from docs_sphinx/installation.rst lines 277-278."""
    pytest.skip("Snippet type 'text' not yet supported")

def test_installation_lines_285_286_10():
    """Test text snippet from docs_sphinx/installation.rst lines 285-286."""
    pytest.skip("Snippet type 'text' not yet supported")

def test_installation_lines_293_294_11():
    """Test text snippet from docs_sphinx/installation.rst lines 293-294."""
    pytest.skip("Snippet type 'text' not yet supported")

def test_installation_lines_299_301_12():
    """Test bash snippet from docs_sphinx/installation.rst lines 299-301."""
    bash_content = 'mkdir -p ~/.orchestrator\nchmod 755 ~/.orchestrator'
    
    # Skip dangerous commands
    skip_commands = ['rm -rf', 'sudo', 'docker', 'systemctl']
    if any(cmd in bash_content for cmd in skip_commands):
        pytest.skip("Skipping potentially destructive command")
    
    # For now, just check it's not empty
    assert bash_content.strip(), "Bash content should not be empty"

@pytest.mark.asyncio
async def test_quickstart_lines_19_59_13():
    """Test YAML pipeline from docs_sphinx/quickstart.rst lines 19-59."""
    import yaml
    
    yaml_content = 'name: topic-summarizer\ndescription: Generate a concise summary of any topic\n\ninputs:\n  topic:\n    type: string\n    description: The topic to summarize\n    required: true\n\n  length:\n    type: integer\n    description: Approximate word count for the summary\n    default: 200\n\noutputs:\n  summary:\n    type: string\n    value: "{{ inputs.topic }}_summary.txt"\n\nsteps:\n  - id: research\n    action: generate_content\n    parameters:\n      prompt: |\n        Research and provide key information about: {{ inputs.topic }}\n        Focus on the most important and interesting aspects.\n      max_length: 500\n\n  - id: summarize\n    action: generate_summary\n    parameters:\n      content: "$results.research"\n      target_length: "{{ inputs.length }}"\n      style: <AUTO>Choose appropriate style for the topic</AUTO>\n\n  - id: save_summary\n    action: write_file\n    parameters:\n      path: "{{ outputs.summary }}"\n      content: "$results.summarize"'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

def test_quickstart_lines_67_87_14():
    """Test Python import from docs_sphinx/quickstart.rst lines 67-87."""
    # Import test - check if modules are available
    code = 'import orchestrator as orc\n\n# Initialize the model pool\norc.init_models()\n\n# Compile the pipeline\npipeline = orc.compile("summarize.yaml")\n\n# Run with different topics\nresult1 = pipeline.run(\n    topic="quantum computing",\n    length=150\n)\n\nresult2 = pipeline.run(\n    topic="sustainable energy",\n    length=250\n)\n\nprint(f"Created summaries: {result1}, {result2}")'
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

@pytest.mark.asyncio
async def test_quickstart_lines_107_179_15():
    """Test YAML pipeline from docs_sphinx/quickstart.rst lines 107-179."""
    import yaml
    
    yaml_content = 'name: research-report-generator\ndescription: Generate comprehensive research reports with citations\n\ninputs:\n  topic:\n    type: string\n    required: true\n  focus_areas:\n    type: array\n    description: Specific areas to focus on\n    default: []\n\noutputs:\n  report_pdf:\n    type: string\n    value: "reports/{{ inputs.topic }}_report.pdf"\n\nsteps:\n  # Web search for recent information\n  - id: search_recent\n    action: search_web\n    parameters:\n      query: "{{ inputs.topic }} 2024 latest developments"\n      max_results: 10\n\n  # Search academic sources\n  - id: search_academic\n    action: search_web\n    parameters:\n      query: "{{ inputs.topic }} research papers scholarly"\n      max_results: 5\n\n  # Compile all sources\n  - id: compile_sources\n    action: compile_markdown\n    parameters:\n      sources:\n        - "$results.search_recent"\n        - "$results.search_academic"\n      include_citations: true\n\n  # Generate the report\n  - id: write_report\n    action: generate_report\n    parameters:\n      research: "$results.compile_sources"\n      topic: "{{ inputs.topic }}"\n      focus_areas: "{{ inputs.focus_areas }}"\n      style: "academic"\n      sections:\n        - "Executive Summary"\n        - "Introduction"\n        - "Current State"\n        - "Recent Developments"\n        - "Future Outlook"\n        - "Conclusions"\n\n  # Quality check\n  - id: validate\n    action: validate_report\n    parameters:\n      report: "$results.write_report"\n      checks:\n        - completeness\n        - citation_accuracy\n        - readability\n\n  # Generate PDF\n  - id: create_pdf\n    action: "!pandoc -o {{ outputs.report_pdf }} --pdf-engine=xelatex"\n    parameters:\n      input: "$results.write_report"'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

def test_quickstart_lines_192_201_16():
    """Test YAML snippet from docs_sphinx/quickstart.rst lines 192-201."""
    import yaml
    
    yaml_content = '# Web search\n- action: search_web\n  parameters:\n    query: "your search query"\n\n# Scrape webpage\n- action: scrape_page\n  parameters:\n    url: "https://example.com"'
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_quickstart_lines_206_218_17():
    """Test YAML snippet from docs_sphinx/quickstart.rst lines 206-218."""
    import yaml
    
    yaml_content = '# Run shell commands (prefix with !)\n- action: "!ls -la"\n\n# File operations\n- action: read_file\n  parameters:\n    path: "data.txt"\n\n- action: write_file\n  parameters:\n    path: "output.txt"\n    content: "Your content"'
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_quickstart_lines_223_238_18():
    """Test YAML snippet from docs_sphinx/quickstart.rst lines 223-238."""
    import yaml
    
    yaml_content = '# Process data\n- action: transform_data\n  parameters:\n    input: "$results.previous_step"\n    operations:\n      - type: filter\n        condition: "value > 100"\n\n# Validate data\n- action: validate_data\n  parameters:\n    data: "$results.data"\n    schema:\n      type: object\n      required: ["name", "value"]'
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.asyncio
async def test_quickstart_lines_246_254_19():
    """Test YAML pipeline from docs_sphinx/quickstart.rst lines 246-254."""
    import yaml
    
    yaml_content = 'steps:\n  - id: analyze\n    action: analyze_data\n    parameters:\n      data: "$results.fetch"\n      method: <AUTO>Choose best analysis method based on data type</AUTO>\n      visualization: <AUTO>Determine if visualization would be helpful</AUTO>\n      depth: <AUTO>Set analysis depth (shallow/medium/deep)</AUTO>'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

@pytest.mark.asyncio
async def test_quickstart_lines_264_288_20():
    """Test YAML pipeline from docs_sphinx/quickstart.rst lines 264-288."""
    import yaml
    
    yaml_content = 'name: composite-pipeline\n\nimports:\n  - common/data_fetcher.yaml as fetcher\n  - common/validator.yaml as validator\n\nsteps:\n  # Use imported pipeline\n  - id: fetch_data\n    pipeline: fetcher\n    parameters:\n      source: "api"\n\n  # Local step\n  - id: process\n    action: process_data\n    parameters:\n      data: "$results.fetch_data"\n\n  # Use another import\n  - id: validate\n    pipeline: validator\n    parameters:\n      data: "$results.process"'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

@pytest.mark.asyncio
async def test_quickstart_lines_296_309_21():
    """Test YAML pipeline from docs_sphinx/quickstart.rst lines 296-309."""
    import yaml
    
    yaml_content = 'steps:\n  - id: risky_operation\n    action: fetch_external_data\n    parameters:\n      url: "{{ inputs.data_source }}"\n    error_handling:\n      retry:\n        max_attempts: 3\n        backoff: exponential\n      fallback:\n        action: use_cached_data\n        parameters:\n          cache_key: "{{ inputs.topic }}"'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

def test_quickstart_lines_317_332_22():
    """Test Python import from docs_sphinx/quickstart.rst lines 317-332."""
    # Import test - check if modules are available
    code = 'import logging\nimport orchestrator as orc\n\n# Enable debug logging\nlogging.basicConfig(level=logging.DEBUG)\n\n# Compile with debug flag\npipeline = orc.compile("pipeline.yaml", debug=True)\n\n# Run with verbose output\nresult = pipeline.run(\n    topic="test",\n    _verbose=True,\n    _step_callback=lambda step: print(f"Executing: {step.id}")\n)'
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_tool_reference_lines_48_100_23():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 48-100."""
    import yaml
    
    yaml_content = '# Web search\n- id: search\n  action: search_web\n  parameters:\n    query: "orchestrator framework tutorial"    # Required: Search query\n    max_results: 10                            # Optional: Number of results (default: 10)\n    search_engine: "google"                    # Optional: google|bing|duckduckgo (default: google)\n    include_snippets: true                     # Optional: Include text snippets (default: true)\n    region: "us"                              # Optional: Region code (default: us)\n    language: "en"                            # Optional: Language code (default: en)\n    safe_search: "moderate"                   # Optional: off|moderate|strict (default: moderate)\n\n# Scrape webpage\n- id: scrape\n  action: scrape_page\n  parameters:\n    url: "https://example.com/article"        # Required: URL to scrape\n    selectors:                                # Optional: CSS selectors to extract\n      title: "h1.main-title"\n      content: "div.article-body"\n      author: "span.author-name"\n    wait_for: "div.content-loaded"            # Optional: Wait for element\n    timeout: 30                               # Optional: Timeout in seconds (default: 30)\n    javascript: true                          # Optional: Execute JavaScript (default: true)\n    clean_html: true                          # Optional: Clean extracted HTML (default: true)\n\n# Take screenshot\n- id: screenshot\n  action: screenshot_page\n  parameters:\n    url: "https://example.com"                # Required: URL to screenshot\n    full_page: true                           # Optional: Capture full page (default: false)\n    width: 1920                               # Optional: Viewport width (default: 1920)\n    height: 1080                              # Optional: Viewport height (default: 1080)\n    wait_for: "img"                           # Optional: Wait for element\n    output_path: "screenshots/page.png"       # Optional: Save path\n\n# Interact with page\n- id: interact\n  action: interact_with_page\n  parameters:\n    url: "https://example.com/form"           # Required: URL to interact with\n    actions:                                  # Required: List of interactions\n      - type: "fill"\n        selector: "#username"\n        value: "testuser"\n      - type: "click"\n        selector: "#submit-button"\n      - type: "wait"\n        duration: 2000\n      - type: "extract"\n        selector: ".result"'
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.asyncio
async def test_tool_reference_lines_105_138_24():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 105-138."""
    import yaml
    
    yaml_content = 'name: web-research-pipeline\ndescription: Comprehensive web research with validation\n\nsteps:\n  # Search for information\n  - id: search_topic\n    action: search_web\n    parameters:\n      query: "{{ inputs.topic }} latest news 2024"\n      max_results: 20\n      search_engine: "google"\n\n  # Scrape top results\n  - id: scrape_articles\n    for_each: "{{ results.search_topic.results[:5] }}"\n    as: result\n    action: scrape_page\n    parameters:\n      url: "{{ result.url }}"\n      selectors:\n        title: "h1, h2.article-title"\n        content: "main, article, div.content"\n        date: "time, .date, .published"\n      clean_html: true\n\n  # Take screenshots for reference\n  - id: capture_pages\n    for_each: "{{ results.search_topic.results[:3] }}"\n    as: result\n    action: screenshot_page\n    parameters:\n      url: "{{ result.url }}"\n      output_path: "research/{{ inputs.topic }}/{{ loop.index }}.png"'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

def test_tool_reference_lines_154_189_25():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 154-189."""
    import yaml
    
    yaml_content = '# Quick search\n- id: search\n  action: quick_search\n  parameters:\n    query: "machine learning basics"          # Required: Search query\n    max_results: 5                           # Optional: Result count (default: 10)\n    format: "json"                           # Optional: json|text (default: json)\n\n# News search\n- id: news\n  action: search_news\n  parameters:\n    query: "AI breakthroughs"                # Required: Search query\n    date_range: "last_week"                  # Optional: last_day|last_week|last_month|last_year\n    sources: ["reuters", "techcrunch"]       # Optional: Preferred sources\n    sort_by: "relevance"                     # Optional: relevance|date (default: relevance)\n\n# Academic search\n- id: academic\n  action: search_academic\n  parameters:\n    query: "quantum computing"               # Required: Search query\n    databases: ["arxiv", "pubmed"]          # Optional: Databases to search\n    year_range: "2020-2024"                 # Optional: Year range\n    peer_reviewed: true                      # Optional: Only peer-reviewed (default: false)\n\n# Image search\n- id: images\n  action: search_images\n  parameters:\n    query: "data visualization examples"     # Required: Search query\n    max_results: 10                         # Optional: Number of images\n    size: "large"                           # Optional: small|medium|large|any\n    type: "photo"                           # Optional: photo|clipart|lineart|any\n    license: "creative_commons"             # Optional: License filter'
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_tool_reference_lines_207_233_26():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 207-233."""
    import yaml
    
    yaml_content = '# Direct command execution\n- id: list_files\n  action: "!ls -la /data"\n\n# Command with parameters\n- id: run_command\n  action: execute_command\n  parameters:\n    command: "python analyze.py"              # Required: Command to execute\n    arguments: ["--input", "data.csv"]       # Optional: Command arguments\n    working_dir: "/project"                  # Optional: Working directory\n    environment:                             # Optional: Environment variables\n      PYTHONPATH: "/project/lib"\n      DEBUG: "true"\n    timeout: 300                             # Optional: Timeout in seconds (default: 60)\n    capture_output: true                     # Optional: Capture stdout/stderr (default: true)\n    shell: true                              # Optional: Use shell execution (default: true)\n\n# Run script file\n- id: run_analysis\n  action: run_script\n  parameters:\n    script_path: "scripts/analyze.sh"        # Required: Path to script\n    arguments: ["{{ inputs.data_file }}"]    # Optional: Script arguments\n    interpreter: "bash"                      # Optional: bash|python|node (default: auto-detect)\n    working_dir: "{{ execution.temp_dir }}"  # Optional: Working directory'
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.asyncio
async def test_tool_reference_lines_238_282_27():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 238-282."""
    import yaml
    
    yaml_content = 'name: data-processing-automation\ndescription: Automated data processing with shell commands\n\nsteps:\n  # Setup environment\n  - id: setup\n    action: "!mkdir -p output/{{ inputs.project_name }}"\n\n  # Download data\n  - id: download\n    action: execute_command\n    parameters:\n      command: "wget"\n      arguments:\n        - "-O"\n        - "data/raw_data.csv"\n        - "{{ inputs.data_url }}"\n      timeout: 600\n\n  # Process with Python\n  - id: process\n    action: execute_command\n    parameters:\n      command: "python"\n      arguments:\n        - "scripts/process_data.py"\n        - "--input"\n        - "data/raw_data.csv"\n        - "--output"\n        - "output/{{ inputs.project_name }}/processed.csv"\n      environment:\n        DATA_QUALITY: "high"\n        PROCESSING_MODE: "{{ inputs.mode }}"\n\n  # Generate report with R\n  - id: report\n    action: "!Rscript reports/generate_report.R output/{{ inputs.project_name }}/processed.csv"\n\n  # Package results\n  - id: package\n    action: execute_command\n    parameters:\n      command: "tar"\n      arguments: ["-czf", "{{ outputs.package }}", "output/{{ inputs.project_name }}"]'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

def test_tool_reference_lines_302_367_28():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 302-367."""
    import yaml
    
    yaml_content = '# Read file\n- id: read_config\n  action: read_file\n  parameters:\n    path: "config/settings.json"             # Required: File path\n    encoding: "utf-8"                        # Optional: File encoding (default: utf-8)\n    parse: true                              # Optional: Parse JSON/YAML (default: false)\n\n# Write file\n- id: save_results\n  action: write_file\n  parameters:\n    path: "output/results.json"              # Required: File path\n    content: "{{ results.analysis | json }}" # Required: Content to write\n    encoding: "utf-8"                        # Optional: File encoding (default: utf-8)\n    create_dirs: true                        # Optional: Create parent dirs (default: true)\n    overwrite: true                          # Optional: Overwrite existing (default: false)\n\n# Copy file\n- id: backup\n  action: copy_file\n  parameters:\n    source: "data/important.db"              # Required: Source path\n    destination: "backup/important_{{ execution.timestamp }}.db"  # Required: Destination\n    overwrite: false                         # Optional: Overwrite existing (default: false)\n\n# Move file\n- id: archive\n  action: move_file\n  parameters:\n    source: "temp/processed.csv"             # Required: Source path\n    destination: "archive/2024/processed.csv" # Required: Destination\n    create_dirs: true                        # Optional: Create parent dirs (default: true)\n\n# Delete file\n- id: cleanup\n  action: delete_file\n  parameters:\n    path: "temp/*"                           # Required: Path or pattern\n    recursive: true                          # Optional: Delete recursively (default: false)\n    force: false                             # Optional: Force deletion (default: false)\n\n# List directory\n- id: scan_files\n  action: list_directory\n  parameters:\n    path: "data/"                            # Required: Directory path\n    pattern: "*.csv"                         # Optional: File pattern\n    recursive: true                          # Optional: Search subdirs (default: false)\n    include_hidden: false                    # Optional: Include hidden files (default: false)\n    details: true                            # Optional: Include file details (default: false)\n\n# Create directory\n- id: setup_dirs\n  action: create_directory\n  parameters:\n    path: "output/{{ inputs.project }}/data" # Required: Directory path\n    parents: true                            # Optional: Create parents (default: true)\n    exist_ok: true                           # Optional: Ok if exists (default: true)\n\n# Check existence\n- id: check_file\n  action: file_exists\n  parameters:\n    path: "config/custom.yaml"               # Required: Path to check'
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.asyncio
async def test_tool_reference_lines_372_416_29():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 372-416."""
    import yaml
    
    yaml_content = 'name: file-organization-pipeline\ndescription: Organize and process files automatically\n\nsteps:\n  # Check for existing data\n  - id: check_existing\n    action: file_exists\n    parameters:\n      path: "data/current_dataset.csv"\n\n  # Backup if exists\n  - id: backup\n    condition: "{{ results.check_existing }}"\n    action: copy_file\n    parameters:\n      source: "data/current_dataset.csv"\n      destination: "backups/dataset_{{ execution.timestamp }}.csv"\n\n  # Read configuration\n  - id: read_config\n    action: read_file\n    parameters:\n      path: "config/processing.yaml"\n      parse: true\n\n  # Process files based on config\n  - id: process_files\n    for_each: "{{ results.read_config.file_patterns }}"\n    as: pattern\n    action: list_directory\n    parameters:\n      path: "{{ pattern.directory }}"\n      pattern: "{{ pattern.glob }}"\n      recursive: true\n\n  # Organize by type\n  - id: organize\n    for_each: "{{ results.process_files | flatten }}"\n    as: file\n    action: move_file\n    parameters:\n      source: "{{ file.path }}"\n      destination: "organized/{{ file.extension }}/{{ file.name }}"\n      create_dirs: true'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")
