"""Working tests for documentation code snippets - Batch 27."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# These tests focus on validation and syntax checking without execution
# They should run quickly and not require external dependencies


def test_quickstart_lines_264_288_0():
    """Test YAML snippet from docs_sphinx/quickstart.rst lines 264-288."""
    # Description: You can compose pipelines from smaller, reusable components:
    import yaml
    
    content = 'name: composite-pipeline\n\nimports:\n  - common/data_fetcher.yaml as fetcher\n  - common/validator.yaml as validator\n\nsteps:\n  # Use imported pipeline\n  - id: fetch_data\n    pipeline: fetcher\n    parameters:\n      source: "api"\n\n  # Local step\n  - id: process\n    action: process_data\n    parameters:\n      data: "$results.fetch_data"\n\n  # Use another import\n  - id: validate\n    pipeline: validator\n    parameters:\n      data: "$results.process"'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        data = yaml.safe_load(content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                assert 'id' in step, "Each step should have an id"


def test_quickstart_lines_296_309_1():
    """Test YAML snippet from docs_sphinx/quickstart.rst lines 296-309."""
    # Description: Add error handling to make pipelines robust:
    import yaml
    
    content = 'steps:\n  - id: risky_operation\n    action: fetch_external_data\n    parameters:\n      url: "{{ inputs.data_source }}"\n    error_handling:\n      retry:\n        max_attempts: 3\n        backoff: exponential\n      fallback:\n        action: use_cached_data\n        parameters:\n          cache_key: "{{ inputs.topic }}"'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        data = yaml.safe_load(content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                assert 'id' in step, "Each step should have an id"


def test_quickstart_lines_317_332_2():
    """Test Python snippet from docs_sphinx/quickstart.rst lines 317-332."""
    # Description: Enable debug mode for detailed execution logs:
    content = 'import logging\nimport orchestrator as orc\n\n# Enable debug logging\nlogging.basicConfig(level=logging.DEBUG)\n\n# Compile with debug flag\npipeline = orc.compile("pipeline.yaml", debug=True)\n\n# Run with verbose output\nresult = pipeline.run(\n    topic="test",\n    _verbose=True,\n    _step_callback=lambda step: print(f"Executing: {step.id}")\n)'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid Python syntax
    try:
        compile(content, '<string>', 'exec')
    except SyntaxError as e:
        pytest.fail(f"Python syntax error: {e}")
    
    # If it's a simple import, try to execute it
    if content.strip().startswith(('import ', 'from ')) and len(content.strip().split('\n')) <= 3:
        try:
            exec(content)
        except ImportError:
            pytest.skip("Import not available in test environment")
        except Exception as e:
            pytest.fail(f"Import failed: {e}")


def test_tool_reference_lines_48_100_3():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 48-100."""
    # Description: **Parameters**:
    import yaml
    
    content = '# Web search\n- id: search\n  action: search_web\n  parameters:\n    query: "orchestrator framework tutorial"    # Required: Search query\n    max_results: 10                            # Optional: Number of results (default: 10)\n    search_engine: "google"                    # Optional: google|bing|duckduckgo (default: google)\n    include_snippets: true                     # Optional: Include text snippets (default: true)\n    region: "us"                              # Optional: Region code (default: us)\n    language: "en"                            # Optional: Language code (default: en)\n    safe_search: "moderate"                   # Optional: off|moderate|strict (default: moderate)\n\n# Scrape webpage\n- id: scrape\n  action: scrape_page\n  parameters:\n    url: "https://example.com/article"        # Required: URL to scrape\n    selectors:                                # Optional: CSS selectors to extract\n      title: "h1.main-title"\n      content: "div.article-body"\n      author: "span.author-name"\n    wait_for: "div.content-loaded"            # Optional: Wait for element\n    timeout: 30                               # Optional: Timeout in seconds (default: 30)\n    javascript: true                          # Optional: Execute JavaScript (default: true)\n    clean_html: true                          # Optional: Clean extracted HTML (default: true)\n\n# Take screenshot\n- id: screenshot\n  action: screenshot_page\n  parameters:\n    url: "https://example.com"                # Required: URL to screenshot\n    full_page: true                           # Optional: Capture full page (default: false)\n    width: 1920                               # Optional: Viewport width (default: 1920)\n    height: 1080                              # Optional: Viewport height (default: 1080)\n    wait_for: "img"                           # Optional: Wait for element\n    output_path: "screenshots/page.png"       # Optional: Save path\n\n# Interact with page\n- id: interact\n  action: interact_with_page\n  parameters:\n    url: "https://example.com/form"           # Required: URL to interact with\n    actions:                                  # Required: List of interactions\n      - type: "fill"\n        selector: "#username"\n        value: "testuser"\n      - type: "click"\n        selector: "#submit-button"\n      - type: "wait"\n        duration: 2000\n      - type: "extract"\n        selector: ".result"'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        data = yaml.safe_load(content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                assert 'id' in step, "Each step should have an id"


def test_tool_reference_lines_105_138_4():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 105-138."""
    # Description: **Example Pipeline**:
    import yaml
    
    content = 'name: web-research-pipeline\ndescription: Comprehensive web research with validation\n\nsteps:\n  # Search for information\n  - id: search_topic\n    action: search_web\n    parameters:\n      query: "{{ inputs.topic }} latest news 2024"\n      max_results: 20\n      search_engine: "google"\n\n  # Scrape top results\n  - id: scrape_articles\n    for_each: "{{ results.search_topic.results[:5] }}"\n    as: result\n    action: scrape_page\n    parameters:\n      url: "{{ result.url }}"\n      selectors:\n        title: "h1, h2.article-title"\n        content: "main, article, div.content"\n        date: "time, .date, .published"\n      clean_html: true\n\n  # Take screenshots for reference\n  - id: capture_pages\n    for_each: "{{ results.search_topic.results[:3] }}"\n    as: result\n    action: screenshot_page\n    parameters:\n      url: "{{ result.url }}"\n      output_path: "research/{{ inputs.topic }}/{{ loop.index }}.png"'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        data = yaml.safe_load(content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                assert 'id' in step, "Each step should have an id"


def test_tool_reference_lines_154_189_5():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 154-189."""
    # Description: **Parameters**:
    import yaml
    
    content = '# Quick search\n- id: search\n  action: quick_search\n  parameters:\n    query: "machine learning basics"          # Required: Search query\n    max_results: 5                           # Optional: Result count (default: 10)\n    format: "json"                           # Optional: json|text (default: json)\n\n# News search\n- id: news\n  action: search_news\n  parameters:\n    query: "AI breakthroughs"                # Required: Search query\n    date_range: "last_week"                  # Optional: last_day|last_week|last_month|last_year\n    sources: ["reuters", "techcrunch"]       # Optional: Preferred sources\n    sort_by: "relevance"                     # Optional: relevance|date (default: relevance)\n\n# Academic search\n- id: academic\n  action: search_academic\n  parameters:\n    query: "quantum computing"               # Required: Search query\n    databases: ["arxiv", "pubmed"]          # Optional: Databases to search\n    year_range: "2020-2024"                 # Optional: Year range\n    peer_reviewed: true                      # Optional: Only peer-reviewed (default: false)\n\n# Image search\n- id: images\n  action: search_images\n  parameters:\n    query: "data visualization examples"     # Required: Search query\n    max_results: 10                         # Optional: Number of images\n    size: "large"                           # Optional: small|medium|large|any\n    type: "photo"                           # Optional: photo|clipart|lineart|any\n    license: "creative_commons"             # Optional: License filter'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        data = yaml.safe_load(content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                assert 'id' in step, "Each step should have an id"


def test_tool_reference_lines_207_233_6():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 207-233."""
    # Description: **Parameters**:
    import yaml
    
    content = '# Direct command execution\n- id: list_files\n  action: "!ls -la /data"\n\n# Command with parameters\n- id: run_command\n  action: execute_command\n  parameters:\n    command: "python analyze.py"              # Required: Command to execute\n    arguments: ["--input", "data.csv"]       # Optional: Command arguments\n    working_dir: "/project"                  # Optional: Working directory\n    environment:                             # Optional: Environment variables\n      PYTHONPATH: "/project/lib"\n      DEBUG: "true"\n    timeout: 300                             # Optional: Timeout in seconds (default: 60)\n    capture_output: true                     # Optional: Capture stdout/stderr (default: true)\n    shell: true                              # Optional: Use shell execution (default: true)\n\n# Run script file\n- id: run_analysis\n  action: run_script\n  parameters:\n    script_path: "scripts/analyze.sh"        # Required: Path to script\n    arguments: ["{{ inputs.data_file }}"]    # Optional: Script arguments\n    interpreter: "bash"                      # Optional: bash|python|node (default: auto-detect)\n    working_dir: "{{ execution.temp_dir }}"  # Optional: Working directory'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        data = yaml.safe_load(content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                assert 'id' in step, "Each step should have an id"


def test_tool_reference_lines_238_282_7():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 238-282."""
    # Description: **Example Pipeline**:
    import yaml
    
    content = 'name: data-processing-automation\ndescription: Automated data processing with shell commands\n\nsteps:\n  # Setup environment\n  - id: setup\n    action: "!mkdir -p output/{{ inputs.project_name }}"\n\n  # Download data\n  - id: download\n    action: execute_command\n    parameters:\n      command: "wget"\n      arguments:\n        - "-O"\n        - "data/raw_data.csv"\n        - "{{ inputs.data_url }}"\n      timeout: 600\n\n  # Process with Python\n  - id: process\n    action: execute_command\n    parameters:\n      command: "python"\n      arguments:\n        - "scripts/process_data.py"\n        - "--input"\n        - "data/raw_data.csv"\n        - "--output"\n        - "output/{{ inputs.project_name }}/processed.csv"\n      environment:\n        DATA_QUALITY: "high"\n        PROCESSING_MODE: "{{ inputs.mode }}"\n\n  # Generate report with R\n  - id: report\n    action: "!Rscript reports/generate_report.R output/{{ inputs.project_name }}/processed.csv"\n\n  # Package results\n  - id: package\n    action: execute_command\n    parameters:\n      command: "tar"\n      arguments: ["-czf", "{{ outputs.package }}", "output/{{ inputs.project_name }}"]'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        data = yaml.safe_load(content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                assert 'id' in step, "Each step should have an id"


def test_tool_reference_lines_302_367_8():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 302-367."""
    # Description: **Parameters**:
    import yaml
    
    content = '# Read file\n- id: read_config\n  action: read_file\n  parameters:\n    path: "config/settings.json"             # Required: File path\n    encoding: "utf-8"                        # Optional: File encoding (default: utf-8)\n    parse: true                              # Optional: Parse JSON/YAML (default: false)\n\n# Write file\n- id: save_results\n  action: write_file\n  parameters:\n    path: "output/results.json"              # Required: File path\n    content: "{{ results.analysis | json }}" # Required: Content to write\n    encoding: "utf-8"                        # Optional: File encoding (default: utf-8)\n    create_dirs: true                        # Optional: Create parent dirs (default: true)\n    overwrite: true                          # Optional: Overwrite existing (default: false)\n\n# Copy file\n- id: backup\n  action: copy_file\n  parameters:\n    source: "data/important.db"              # Required: Source path\n    destination: "backup/important_{{ execution.timestamp }}.db"  # Required: Destination\n    overwrite: false                         # Optional: Overwrite existing (default: false)\n\n# Move file\n- id: archive\n  action: move_file\n  parameters:\n    source: "temp/processed.csv"             # Required: Source path\n    destination: "archive/2024/processed.csv" # Required: Destination\n    create_dirs: true                        # Optional: Create parent dirs (default: true)\n\n# Delete file\n- id: cleanup\n  action: delete_file\n  parameters:\n    path: "temp/*"                           # Required: Path or pattern\n    recursive: true                          # Optional: Delete recursively (default: false)\n    force: false                             # Optional: Force deletion (default: false)\n\n# List directory\n- id: scan_files\n  action: list_directory\n  parameters:\n    path: "data/"                            # Required: Directory path\n    pattern: "*.csv"                         # Optional: File pattern\n    recursive: true                          # Optional: Search subdirs (default: false)\n    include_hidden: false                    # Optional: Include hidden files (default: false)\n    details: true                            # Optional: Include file details (default: false)\n\n# Create directory\n- id: setup_dirs\n  action: create_directory\n  parameters:\n    path: "output/{{ inputs.project }}/data" # Required: Directory path\n    parents: true                            # Optional: Create parents (default: true)\n    exist_ok: true                           # Optional: Ok if exists (default: true)\n\n# Check existence\n- id: check_file\n  action: file_exists\n  parameters:\n    path: "config/custom.yaml"               # Required: Path to check'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        data = yaml.safe_load(content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                assert 'id' in step, "Each step should have an id"


def test_tool_reference_lines_372_416_9():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 372-416."""
    # Description: **Example Pipeline**:
    import yaml
    
    content = 'name: file-organization-pipeline\ndescription: Organize and process files automatically\n\nsteps:\n  # Check for existing data\n  - id: check_existing\n    action: file_exists\n    parameters:\n      path: "data/current_dataset.csv"\n\n  # Backup if exists\n  - id: backup\n    condition: "{{ results.check_existing }}"\n    action: copy_file\n    parameters:\n      source: "data/current_dataset.csv"\n      destination: "backups/dataset_{{ execution.timestamp }}.csv"\n\n  # Read configuration\n  - id: read_config\n    action: read_file\n    parameters:\n      path: "config/processing.yaml"\n      parse: true\n\n  # Process files based on config\n  - id: process_files\n    for_each: "{{ results.read_config.file_patterns }}"\n    as: pattern\n    action: list_directory\n    parameters:\n      path: "{{ pattern.directory }}"\n      pattern: "{{ pattern.glob }}"\n      recursive: true\n\n  # Organize by type\n  - id: organize\n    for_each: "{{ results.process_files | flatten }}"\n    as: file\n    action: move_file\n    parameters:\n      source: "{{ file.path }}"\n      destination: "organized/{{ file.extension }}/{{ file.name }}"\n      create_dirs: true'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        data = yaml.safe_load(content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                assert 'id' in step, "Each step should have an id"
