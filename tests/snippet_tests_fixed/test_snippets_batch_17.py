"""Tests for documentation code snippets - Batch 17 (Fixed)."""
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
async def test_yaml_pipelines_lines_527_541_0():
    """Test YAML pipeline from docs_sphinx/yaml_pipelines.rst lines 527-541."""
    import yaml
    import orchestrator
    
    yaml_content = r"""# Enable checkpointing
config:
  checkpoint:
    enabled: true
    frequency: "after_each_step"  # or: "every_n_steps: 5"
    storage: "postgresql"         # or: "redis", "filesystem"

steps:
  - id: long_running
    action: expensive_computation
    checkpoint: true  # Force checkpoint after this step
    recovery:
      strategy: "retry"  # or: "skip", "use_cached"
      max_attempts: 3"""
    
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
async def test_yaml_pipelines_lines_578_638_1():
    """Test YAML pipeline from docs_sphinx/yaml_pipelines.rst lines 578-638."""
    import yaml
    import orchestrator
    
    yaml_content = r"""name: data-processing-pipeline
description: ETL pipeline with validation

inputs:
  source_url:
    type: string
    required: true

  output_format:
    type: string
    default: "parquet"
    validation:
      enum: ["csv", "json", "parquet"]

steps:
  # Extract
  - id: extract
    action: fetch_data
    parameters:
      url: "{{ inputs.source_url }}"
      format: <AUTO>Detect format from URL</AUTO>

  # Transform
  - id: clean
    action: clean_data
    parameters:
      data: "$results.extract"
      rules:
        - remove_duplicates: true
        - handle_missing: "interpolate"
        - standardize_dates: true

  - id: transform
    action: transform_data
    parameters:
      data: "$results.clean"
      operations:
        - type: "aggregate"
          group_by: ["category"]
          metrics: ["sum", "avg"]

  # Load
  - id: validate
    action: validate_data
    parameters:
      data: "$results.transform"
      schema:
        type: "dataframe"
        columns:
          - name: "category"
            type: "string"
          - name: "total"
            type: "float"

  - id: save
    action: save_data
    parameters:
      data: "$results.validate"
      path: "output/processed_data.{{ inputs.output_format }}"
      format: "{{ inputs.output_format }}""""
    
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
async def test_yaml_pipelines_lines_644_703_2():
    """Test YAML pipeline from docs_sphinx/yaml_pipelines.rst lines 644-703."""
    import yaml
    import orchestrator
    
    yaml_content = r"""name: comprehensive-research
description: Research from multiple sources with cross-validation

inputs:
  topic:
    type: string
    required: true

  sources:
    type: array
    default: ["web", "academic", "news"]

steps:
  # Parallel source fetching
  - id: fetch_sources
    parallel:
      - id: web_search
        condition: "'web' in inputs.sources"
        action: search_web
        parameters:
          query: "{{ inputs.topic }}"
          max_results: 20

      - id: academic_search
        condition: "'academic' in inputs.sources"
        action: search_academic
        parameters:
          query: "{{ inputs.topic }}"
          databases: ["arxiv", "pubmed", "scholar"]

      - id: news_search
        condition: "'news' in inputs.sources"
        action: search_news
        parameters:
          query: "{{ inputs.topic }}"
          date_range: "last_30_days"

  # Process and validate
  - id: extract_facts
    action: extract_information
    parameters:
      sources: "$results.fetch_sources"
      extract:
        - facts
        - claims
        - statistics

  - id: cross_validate
    action: validate_claims
    parameters:
      claims: "$results.extract_facts.claims"
      require_sources: 2  # Need 2+ sources to confirm

  # Generate report
  - id: synthesize
    action: generate_synthesis
    parameters:
      validated_facts: "$results.cross_validate"
      style: "analytical"
      include_confidence: true"""
    
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
