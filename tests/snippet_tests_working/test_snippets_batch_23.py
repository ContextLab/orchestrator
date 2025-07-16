"""Working tests for documentation code snippets - Batch 23."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# These tests focus on validation and syntax checking without execution
# They should run quickly and not require external dependencies


def test_concepts_lines_422_434_0():
    """Test YAML snippet from docs_sphinx/concepts.rst lines 422-434."""
    # Description: Tasks without dependencies can run in parallel:
    import yaml
    
    content = 'steps:\n  # These run in parallel\n  - id: source1\n    action: fetch_data_a\n\n  - id: source2\n    action: fetch_data_b\n\n  # This waits for both\n  - id: combine\n    depends_on: [source1, source2]\n    action: merge_data'
    
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


def test_concepts_lines_442_449_1():
    """Test YAML snippet from docs_sphinx/concepts.rst lines 442-449."""
    # Description: Expensive operations can be cached:
    import yaml
    
    content = 'steps:\n  - id: expensive_computation\n    action: complex_analysis\n    cache:\n      enabled: true\n      key: "{{ inputs.data_hash }}"\n      ttl: 3600  # 1 hour'
    
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


def test_concepts_lines_457_462_2():
    """Test YAML snippet from docs_sphinx/concepts.rst lines 457-462."""
    # Description: Control resource usage:
    import yaml
    
    content = 'config:\n  resources:\n    max_memory: "8GB"\n    max_threads: 4\n    gpu_enabled: false'
    
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


def test_concepts_lines_483_494_3():
    """Test YAML snippet from docs_sphinx/concepts.rst lines 483-494."""
    # Description: All inputs are validated:
    import yaml
    
    content = 'inputs:\n  email:\n    type: string\n    validation:\n      pattern: "^[\\\\w.-]+@[\\\\w.-]+\\\\.\\\\w+$"\n\n  amount:\n    type: number\n    validation:\n      min: 0\n      max: 10000'
    
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


def test_concepts_lines_502_505_4():
    """Test YAML snippet from docs_sphinx/concepts.rst lines 502-505."""
    # Description: Sensitive data is handled securely:
    import yaml
    
    content = 'parameters:\n  api_key: "{{ env.SECRET_API_KEY }}"  # From environment\n  password: "{{ vault.db_password }}"   # From secret vault'
    
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


def test_concepts_lines_523_533_5():
    """Test  snippet from docs_sphinx/concepts.rst lines 523-533."""
    # Description: --------------------
    content = 'pipelines/\n├── common/           # Shared components\n│   ├── validation.yaml\n│   └── formatting.yaml\n├── workflows/        # Complete workflows\n│   ├── research.yaml\n│   └── analysis.yaml\n└── specialized/      # Domain-specific\n    ├── finance.yaml\n    └── healthcare.yaml'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    assert len(content) > 0, "Content should have length"


def test_getting_started_lines_29_43_6():
    """Test YAML snippet from docs_sphinx/getting_started.rst lines 29-43."""
    # Description: A pipeline is a collection of tasks that work together to achieve a goal. Pipelines are defined in YAML and can include:
    import yaml
    
    content = 'name: research-report\ndescription: Generate comprehensive research reports\n\ninputs:\n  topic:\n    type: string\n    description: Research topic\n    required: true\n\nsteps:\n  - id: search\n    action: search_web\n    parameters:\n      query: "{{ inputs.topic }} latest research"'
    
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


def test_getting_started_lines_61_64_7():
    """Test YAML snippet from docs_sphinx/getting_started.rst lines 61-64."""
    # Description: When you're unsure about a value, use ``<AUTO>`` tags to let AI models decide:
    import yaml
    
    content = 'parameters:\n  method: <AUTO>Choose best analysis method for this data</AUTO>\n  depth: <AUTO>Determine appropriate depth level</AUTO>'
    
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


def test_getting_started_lines_83_108_8():
    """Test YAML snippet from docs_sphinx/getting_started.rst lines 83-108."""
    # Description: 1. **Create a pipeline definition** (``research.yaml``):
    import yaml
    
    content = 'name: quick-research\ndescription: Quick research on any topic\n\ninputs:\n  topic:\n    type: string\n    required: true\n\noutputs:\n  report:\n    type: string\n    value: "{{ inputs.topic }}_report.md"\n\nsteps:\n  - id: search\n    action: search_web\n    parameters:\n      query: "{{ inputs.topic }}"\n      max_results: 5\n\n  - id: summarize\n    action: generate_summary\n    parameters:\n      content: "$results.search"\n      style: <AUTO>Choose appropriate summary style</AUTO>'
    
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


def test_getting_started_lines_113_126_9():
    """Test Python snippet from docs_sphinx/getting_started.rst lines 113-126."""
    # Description: 2. **Run the pipeline**:
    content = 'import orchestrator as orc\n\n# Initialize models\norc.init_models()\n\n# Compile the pipeline\npipeline = orc.compile("research.yaml")\n\n# Execute with different topics\nresult1 = pipeline.run(topic="artificial intelligence")\nresult2 = pipeline.run(topic="climate change")\n\nprint(f"Reports generated: {result1}, {result2}")'
    
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
