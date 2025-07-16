"""Working tests for documentation code snippets - Batch 32."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# These tests focus on validation and syntax checking without execution
# They should run quickly and not require external dependencies


def test_yaml_pipelines_lines_262_284_0():
    """Test YAML snippet from docs_sphinx/yaml_pipelines.rst lines 262-284."""
    # Description: **Error Handling**:
    import yaml
    
    content = 'steps:\n  - id: risky_operation\n    action: external_api_call\n    error_handling:\n      # Retry configuration\n      retry:\n        max_attempts: 3\n        backoff: exponential  # or: constant, linear\n        initial_delay: 1000   # milliseconds\n        max_delay: 30000\n\n      # Fallback action\n      fallback:\n        action: use_cached_data\n        parameters:\n          cache_key: "{{ inputs.topic }}"\n\n      # Continue on error\n      continue_on_error: true\n\n      # Custom error message\n      error_message: "Failed to fetch external data, using cache"'
    
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
        # Note: data can be None for YAML with only comments
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                # Note: 'id' is optional in minimal examples


def test_yaml_pipelines_lines_296_335_1():
    """Test YAML snippet from docs_sphinx/yaml_pipelines.rst lines 296-335."""
    # Description: Variables can be referenced throughout your pipeline using Jinja2-style template expressions:
    import yaml
    
    content = 'id: variable_demo\nname: Variable Access Demo\n\ninputs:\n  - name: user_topic\n    type: string\n    description: Topic to research\n\nsteps:\n  - id: initial_search\n    action: search_web\n    parameters:\n      # Reference input variables\n      query: "{{ user_topic }} latest news"\n\n  - id: analyze_results\n    action: analyze\n    parameters:\n      # Reference results from previous steps\n      data: "{{ initial_search.results }}"\n      # Can access nested fields\n      first_result: "{{ initial_search.results[0].title }}"\n    dependencies: [initial_search]\n\n  - id: final_report\n    action: generate_text\n    parameters:\n      # Combine multiple references\n      prompt: |\n        Create a report about {{ user_topic }}\n        Based on: {{ analyze_results.summary }}\n        Total results found: {{ initial_search.count }}\n    dependencies: [analyze_results]\n\noutputs:\n  # Define output variables\n  report: "{{ final_report.result }}"\n  summary: "{{ analyze_results.summary }}"\n  search_count: "{{ initial_search.count }}"'
    
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
        # Note: data can be None for YAML with only comments
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                # Note: 'id' is optional in minimal examples


def test_yaml_pipelines_lines_340_359_2():
    """Test text snippet from docs_sphinx/yaml_pipelines.rst lines 340-359."""
    # Description: **Filters and Functions**:
    content = '# String manipulation\n"{{ inputs.topic | lower }}"\n"{{ inputs.topic | upper }}"\n"{{ inputs.topic | slugify }}"\n"{{ inputs.topic | replace(\' \', \'_\') }}"\n\n# Date formatting\n"{{ execution.timestamp | strftime(\'%Y-%m-%d\') }}"\n\n# Math operations\n"{{ inputs.count * 2 }}"\n"{{ inputs.value | round(2) }}"\n\n# Conditionals\n"{{ \'premium\' if inputs.tier == \'gold\' else \'standard\' }}"\n\n# Lists and loops\n"{{ inputs.items | join(\', \') }}"\n"{{ inputs.sources | length }}"'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    assert len(content) > 0, "Content should have length"


def test_yaml_pipelines_lines_369_380_3():
    """Test YAML snippet from docs_sphinx/yaml_pipelines.rst lines 369-380."""
    # Description: **Basic AUTO Tags**:
    import yaml
    
    content = "parameters:\n  # Simple decision\n  style: <AUTO>Choose appropriate writing style</AUTO>\n\n  # Context-aware decision\n  method: <AUTO>Based on the data type {{ results.fetch.type }}, choose the best analysis method</AUTO>\n\n  # Multiple choices\n  options:\n    visualization: <AUTO>Should we create visualizations for this data?</AUTO>\n    format: <AUTO>What's the best output format: json, csv, or parquet?</AUTO>"
    
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
        # Note: data can be None for YAML with only comments
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                # Note: 'id' is optional in minimal examples


def test_yaml_pipelines_lines_385_406_4():
    """Test YAML snippet from docs_sphinx/yaml_pipelines.rst lines 385-406."""
    # Description: **Advanced AUTO Patterns**:
    import yaml
    
    content = '# Conditional AUTO\nanalysis_depth: |\n  <AUTO>\n  Given:\n  - Data size: {{ results.fetch.size }}\n  - Time constraint: {{ inputs.deadline }}\n  - Importance: {{ inputs.priority }}\n\n  Determine the appropriate analysis depth (1-10)\n  </AUTO>\n\n# Structured AUTO\nreport_sections: |\n  <AUTO>\n  For a report about {{ inputs.topic }}, determine which sections to include:\n  - Executive Summary: yes/no\n  - Technical Details: yes/no\n  - Future Outlook: yes/no\n  - Recommendations: yes/no\n  Return as JSON object\n  </AUTO>'
    
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
        # Note: data can be None for YAML with only comments
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                # Note: 'id' is optional in minimal examples


def test_yaml_pipelines_lines_425_446_5():
    """Test Python snippet from docs_sphinx/yaml_pipelines.rst lines 425-446."""
    # Description: **User Control Points**:
    content = 'import orchestrator as orc\n\n# Control compilation options\npipeline = orc.compile(\n    "pipeline.yaml",\n    # Override config values\n    config={\n        "timeout": 7200,\n        "checkpoint": True\n    },\n    # Set compilation flags\n    strict=True,           # Strict validation\n    optimize=True,         # Enable optimizations\n    dry_run=False,         # Actually compile (not just validate)\n    debug=True            # Include debug information\n)\n\n# Inspect compilation result\nprint(pipeline.get_required_tools())\nprint(pipeline.get_task_graph())\nprint(pipeline.get_estimated_cost())'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid Python syntax
    # Skip syntax check for notebook-specific code with top-level await
    if 'await' in content and ('notebook' in content.lower() or 'jupyter' in content.lower()):
        # This is notebook-specific syntax, skip syntax validation
        pass
    else:
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


def test_yaml_pipelines_lines_451_461_6():
    """Test YAML snippet from docs_sphinx/yaml_pipelines.rst lines 451-461."""
    # Description: **Runtime vs Compile-Time Resolution**:
    import yaml
    
    content = '# Compile-time (resolved during compilation)\nconfig:\n  timestamp: "{{ compile_time.timestamp }}"\n\n# Runtime (resolved during execution)\nsteps:\n  - id: dynamic\n    parameters:\n      query: "{{ inputs.topic }}"  # Runtime\n      results: "$results.previous"  # Runtime'
    
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
        # Note: data can be None for YAML with only comments
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                # Note: 'id' is optional in minimal examples


def test_yaml_pipelines_lines_472_491_7():
    """Test YAML snippet from docs_sphinx/yaml_pipelines.rst lines 472-491."""
    # Description: Reuse common patterns:
    import yaml
    
    content = 'imports:\n  # Import specific steps\n  - common/data_validation.yaml#validate_step as validate\n\n  # Import entire pipeline\n  - workflows/standard_analysis.yaml as analysis\n\nsteps:\n  # Use imported step\n  - id: validation\n    extends: validate\n    parameters:\n      data: "$results.fetch"\n\n  # Use imported pipeline\n  - id: analyze\n    pipeline: analysis\n    inputs:\n      data: "$results.validation"'
    
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
        # Note: data can be None for YAML with only comments
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                # Note: 'id' is optional in minimal examples


def test_yaml_pipelines_lines_497_525_8():
    """Test YAML snippet from docs_sphinx/yaml_pipelines.rst lines 497-525."""
    # Description: -------------------------
    import yaml
    
    content = 'steps:\n  # Define parallel group\n  - id: parallel_fetch\n    parallel:\n      - id: fetch_api\n        action: fetch_url\n        parameters:\n          url: "{{ inputs.api_url }}"\n\n      - id: fetch_db\n        action: query_database\n        parameters:\n          query: "{{ inputs.db_query }}"\n\n      - id: fetch_file\n        action: read_file\n        parameters:\n          path: "{{ inputs.file_path }}"\n\n  # Use results from parallel group\n  - id: merge\n    action: combine_data\n    depends_on: [parallel_fetch]\n    parameters:\n      sources:\n        - "$results.parallel_fetch.fetch_api"\n        - "$results.parallel_fetch.fetch_db"\n        - "$results.parallel_fetch.fetch_file"'
    
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
        # Note: data can be None for YAML with only comments
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                # Note: 'id' is optional in minimal examples


def test_yaml_pipelines_lines_531_548_9():
    """Test YAML snippet from docs_sphinx/yaml_pipelines.rst lines 531-548."""
    # Description: -------------------
    import yaml
    
    content = 'steps:\n  # For-each loop\n  - id: process_items\n    for_each: "{{ inputs.items }}"\n    as: item\n    action: process_single_item\n    parameters:\n      data: "{{ item }}"\n      index: "{{ loop.index }}"\n\n  # While loop\n  - id: iterative_refinement\n    while: "{{ results.quality_check.score < 0.95 }}"\n    max_iterations: 10\n    action: refine_result\n    parameters:\n      current: "$results.previous_iteration"'
    
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
        # Note: data can be None for YAML with only comments
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                # Note: 'id' is optional in minimal examples
