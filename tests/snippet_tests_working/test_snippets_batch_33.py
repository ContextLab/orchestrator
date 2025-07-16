"""Working tests for documentation code snippets - Batch 33."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# These tests focus on validation and syntax checking without execution
# They should run quickly and not require external dependencies


def test_yaml_pipelines_lines_527_541_0():
    """Test YAML snippet from docs_sphinx/yaml_pipelines.rst lines 527-541."""
    # Description: ----------------
    import yaml
    
    content = '# Enable checkpointing\nconfig:\n  checkpoint:\n    enabled: true\n    frequency: "after_each_step"  # or: "every_n_steps: 5"\n    storage: "postgresql"         # or: "redis", "filesystem"\n\nsteps:\n  - id: long_running\n    action: expensive_computation\n    checkpoint: true  # Force checkpoint after this step\n    recovery:\n      strategy: "retry"  # or: "skip", "use_cached"\n      max_attempts: 3'
    
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


def test_yaml_pipelines_lines_578_638_1():
    """Test YAML snippet from docs_sphinx/yaml_pipelines.rst lines 578-638."""
    # Description: ------------------------
    import yaml
    
    content = 'name: data-processing-pipeline\ndescription: ETL pipeline with validation\n\ninputs:\n  source_url:\n    type: string\n    required: true\n\n  output_format:\n    type: string\n    default: "parquet"\n    validation:\n      enum: ["csv", "json", "parquet"]\n\nsteps:\n  # Extract\n  - id: extract\n    action: fetch_data\n    parameters:\n      url: "{{ inputs.source_url }}"\n      format: <AUTO>Detect format from URL</AUTO>\n\n  # Transform\n  - id: clean\n    action: clean_data\n    parameters:\n      data: "$results.extract"\n      rules:\n        - remove_duplicates: true\n        - handle_missing: "interpolate"\n        - standardize_dates: true\n\n  - id: transform\n    action: transform_data\n    parameters:\n      data: "$results.clean"\n      operations:\n        - type: "aggregate"\n          group_by: ["category"]\n          metrics: ["sum", "avg"]\n\n  # Load\n  - id: validate\n    action: validate_data\n    parameters:\n      data: "$results.transform"\n      schema:\n        type: "dataframe"\n        columns:\n          - name: "category"\n            type: "string"\n          - name: "total"\n            type: "float"\n\n  - id: save\n    action: save_data\n    parameters:\n      data: "$results.validate"\n      path: "output/processed_data.{{ inputs.output_format }}"\n      format: "{{ inputs.output_format }}"'
    
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


def test_yaml_pipelines_lines_644_703_2():
    """Test YAML snippet from docs_sphinx/yaml_pipelines.rst lines 644-703."""
    # Description: ------------------------------
    import yaml
    
    content = 'name: comprehensive-research\ndescription: Research from multiple sources with cross-validation\n\ninputs:\n  topic:\n    type: string\n    required: true\n\n  sources:\n    type: array\n    default: ["web", "academic", "news"]\n\nsteps:\n  # Parallel source fetching\n  - id: fetch_sources\n    parallel:\n      - id: web_search\n        condition: "\'web\' in inputs.sources"\n        action: search_web\n        parameters:\n          query: "{{ inputs.topic }}"\n          max_results: 20\n\n      - id: academic_search\n        condition: "\'academic\' in inputs.sources"\n        action: search_academic\n        parameters:\n          query: "{{ inputs.topic }}"\n          databases: ["arxiv", "pubmed", "scholar"]\n\n      - id: news_search\n        condition: "\'news\' in inputs.sources"\n        action: search_news\n        parameters:\n          query: "{{ inputs.topic }}"\n          date_range: "last_30_days"\n\n  # Process and validate\n  - id: extract_facts\n    action: extract_information\n    parameters:\n      sources: "$results.fetch_sources"\n      extract:\n        - facts\n        - claims\n        - statistics\n\n  - id: cross_validate\n    action: validate_claims\n    parameters:\n      claims: "$results.extract_facts.claims"\n      require_sources: 2  # Need 2+ sources to confirm\n\n  # Generate report\n  - id: synthesize\n    action: generate_synthesis\n    parameters:\n      validated_facts: "$results.cross_validate"\n      style: "analytical"\n      include_confidence: true'
    
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
