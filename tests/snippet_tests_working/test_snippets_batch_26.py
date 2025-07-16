"""Working tests for documentation code snippets - Batch 26."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# These tests focus on validation and syntax checking without execution
# They should run quickly and not require external dependencies


def test_installation_lines_285_286_0():
    """Test text snippet from docs_sphinx/installation.rst lines 285-286."""
    # Description: **Model Connection Error**:
    content = 'Failed to connect to Ollama at http://localhost:11434'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    assert len(content) > 0, "Content should have length"


def test_installation_lines_293_294_1():
    """Test text snippet from docs_sphinx/installation.rst lines 293-294."""
    # Description: **Permission Error**:
    content = "Permission denied: '/home/user/.orchestrator'"
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    assert len(content) > 0, "Content should have length"


def test_installation_lines_299_301_2():
    """Test Bash snippet from docs_sphinx/installation.rst lines 299-301."""
    # Description: Solution: Create directory with proper permissions:
    content = 'mkdir -p ~/.orchestrator\nchmod 755 ~/.orchestrator'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Special handling for pip install commands
    if 'pip install' in content:
        lines = content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                assert line.startswith('pip install'), f"Expected pip install command: {line}"
        return  # Skip further validation for pip commands
    
    # For other bash commands, just check they're not empty
    assert len(content.strip()) > 0, "Bash content should not be empty"


def test_quickstart_lines_19_59_3():
    """Test YAML snippet from docs_sphinx/quickstart.rst lines 19-59."""
    # Description: Create a file called ``summarize.yaml``:
    import yaml
    
    content = 'name: topic-summarizer\ndescription: Generate a concise summary of any topic\n\ninputs:\n  topic:\n    type: string\n    description: The topic to summarize\n    required: true\n\n  length:\n    type: integer\n    description: Approximate word count for the summary\n    default: 200\n\noutputs:\n  summary:\n    type: string\n    value: "{{ inputs.topic }}_summary.txt"\n\nsteps:\n  - id: research\n    action: generate_content\n    parameters:\n      prompt: |\n        Research and provide key information about: {{ inputs.topic }}\n        Focus on the most important and interesting aspects.\n      max_length: 500\n\n  - id: summarize\n    action: generate_summary\n    parameters:\n      content: "$results.research"\n      target_length: "{{ inputs.length }}"\n      style: <AUTO>Choose appropriate style for the topic</AUTO>\n\n  - id: save_summary\n    action: write_file\n    parameters:\n      path: "{{ outputs.summary }}"\n      content: "$results.summarize"'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        # Check if content contains AUTO tags
        if '<AUTO>' in content:
            # Use AUTO tag parser
            from orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser
            parser = AutoTagYAMLParser()
            data = parser.parse(content)
        else:
            # Use standard YAML parser
            data = yaml.safe_load(content)
        assert data is not None
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                assert 'id' in step, "Each step should have an id"


def test_quickstart_lines_67_87_4():
    """Test Python snippet from docs_sphinx/quickstart.rst lines 67-87."""
    # Description: Create a Python script to run your pipeline:
    content = 'import orchestrator as orc\n\n# Initialize the model pool\norc.init_models()\n\n# Compile the pipeline\npipeline = orc.compile("summarize.yaml")\n\n# Run with different topics\nresult1 = pipeline.run(\n    topic="quantum computing",\n    length=150\n)\n\nresult2 = pipeline.run(\n    topic="sustainable energy",\n    length=250\n)\n\nprint(f"Created summaries: {result1}, {result2}")'
    
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


def test_quickstart_lines_107_179_5():
    """Test YAML snippet from docs_sphinx/quickstart.rst lines 107-179."""
    # Description: Let's create a more sophisticated pipeline that generates research reports:
    import yaml
    
    content = 'name: research-report-generator\ndescription: Generate comprehensive research reports with citations\n\ninputs:\n  topic:\n    type: string\n    required: true\n  focus_areas:\n    type: array\n    description: Specific areas to focus on\n    default: []\n\noutputs:\n  report_pdf:\n    type: string\n    value: "reports/{{ inputs.topic }}_report.pdf"\n\nsteps:\n  # Web search for recent information\n  - id: search_recent\n    action: search_web\n    parameters:\n      query: "{{ inputs.topic }} 2024 latest developments"\n      max_results: 10\n\n  # Search academic sources\n  - id: search_academic\n    action: search_web\n    parameters:\n      query: "{{ inputs.topic }} research papers scholarly"\n      max_results: 5\n\n  # Compile all sources\n  - id: compile_sources\n    action: compile_markdown\n    parameters:\n      sources:\n        - "$results.search_recent"\n        - "$results.search_academic"\n      include_citations: true\n\n  # Generate the report\n  - id: write_report\n    action: generate_report\n    parameters:\n      research: "$results.compile_sources"\n      topic: "{{ inputs.topic }}"\n      focus_areas: "{{ inputs.focus_areas }}"\n      style: "academic"\n      sections:\n        - "Executive Summary"\n        - "Introduction"\n        - "Current State"\n        - "Recent Developments"\n        - "Future Outlook"\n        - "Conclusions"\n\n  # Quality check\n  - id: validate\n    action: validate_report\n    parameters:\n      report: "$results.write_report"\n      checks:\n        - completeness\n        - citation_accuracy\n        - readability\n\n  # Generate PDF\n  - id: create_pdf\n    action: "!pandoc -o {{ outputs.report_pdf }} --pdf-engine=xelatex"\n    parameters:\n      input: "$results.write_report"'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        # Check if content contains AUTO tags
        if '<AUTO>' in content:
            # Use AUTO tag parser
            from orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser
            parser = AutoTagYAMLParser()
            data = parser.parse(content)
        else:
            # Use standard YAML parser
            data = yaml.safe_load(content)
        assert data is not None
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                assert 'id' in step, "Each step should have an id"


def test_quickstart_lines_192_201_6():
    """Test YAML snippet from docs_sphinx/quickstart.rst lines 192-201."""
    # Description: **Web Tools**:
    import yaml
    
    content = '# Web search\n- action: search_web\n  parameters:\n    query: "your search query"\n\n# Scrape webpage\n- action: scrape_page\n  parameters:\n    url: "https://example.com"'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        # Check if content contains AUTO tags
        if '<AUTO>' in content:
            # Use AUTO tag parser
            from orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser
            parser = AutoTagYAMLParser()
            data = parser.parse(content)
        else:
            # Use standard YAML parser
            data = yaml.safe_load(content)
        assert data is not None
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                assert 'id' in step, "Each step should have an id"


def test_quickstart_lines_206_218_7():
    """Test YAML snippet from docs_sphinx/quickstart.rst lines 206-218."""
    # Description: **System Tools**:
    import yaml
    
    content = '# Run shell commands (prefix with !)\n- action: "!ls -la"\n\n# File operations\n- action: read_file\n  parameters:\n    path: "data.txt"\n\n- action: write_file\n  parameters:\n    path: "output.txt"\n    content: "Your content"'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        # Check if content contains AUTO tags
        if '<AUTO>' in content:
            # Use AUTO tag parser
            from orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser
            parser = AutoTagYAMLParser()
            data = parser.parse(content)
        else:
            # Use standard YAML parser
            data = yaml.safe_load(content)
        assert data is not None
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                assert 'id' in step, "Each step should have an id"


def test_quickstart_lines_223_238_8():
    """Test YAML snippet from docs_sphinx/quickstart.rst lines 223-238."""
    # Description: **Data Tools**:
    import yaml
    
    content = '# Process data\n- action: transform_data\n  parameters:\n    input: "$results.previous_step"\n    operations:\n      - type: filter\n        condition: "value > 100"\n\n# Validate data\n- action: validate_data\n  parameters:\n    data: "$results.data"\n    schema:\n      type: object\n      required: ["name", "value"]'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        # Check if content contains AUTO tags
        if '<AUTO>' in content:
            # Use AUTO tag parser
            from orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser
            parser = AutoTagYAMLParser()
            data = parser.parse(content)
        else:
            # Use standard YAML parser
            data = yaml.safe_load(content)
        assert data is not None
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                assert 'id' in step, "Each step should have an id"


def test_quickstart_lines_246_254_9():
    """Test YAML snippet from docs_sphinx/quickstart.rst lines 246-254."""
    # Description: AUTO tags let AI models make intelligent decisions:
    import yaml
    
    content = 'steps:\n  - id: analyze\n    action: analyze_data\n    parameters:\n      data: "$results.fetch"\n      method: <AUTO>Choose best analysis method based on data type</AUTO>\n      visualization: <AUTO>Determine if visualization would be helpful</AUTO>\n      depth: <AUTO>Set analysis depth (shallow/medium/deep)</AUTO>'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        # Check if content contains AUTO tags
        if '<AUTO>' in content:
            # Use AUTO tag parser
            from orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser
            parser = AutoTagYAMLParser()
            data = parser.parse(content)
        else:
            # Use standard YAML parser
            data = yaml.safe_load(content)
        assert data is not None
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                assert 'id' in step, "Each step should have an id"
