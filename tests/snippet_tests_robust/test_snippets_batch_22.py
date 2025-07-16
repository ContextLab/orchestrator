"""Tests for documentation code snippets - Batch 22 (Robust)."""
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
async def test_yaml_pipelines_lines_398_419_0():
    """Test orchestrator code from docs_sphinx/yaml_pipelines.rst lines 398-419."""
    # **User Control Points**:
    
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
        code = ("""import orchestrator as orc

# Control compilation options
pipeline = orc.compile(
    "pipeline.yaml",
    # Override config values
    config={
        "timeout": 7200,
        "checkpoint": True
    },
    # Set compilation flags
    strict=True,           # Strict validation
    optimize=True,         # Enable optimizations
    dry_run=False,         # Actually compile (not just validate)
    debug=True            # Include debug information
)

# Inspect compilation result
print(pipeline.get_required_tools())
print(pipeline.get_task_graph())
print(pipeline.get_estimated_cost())""")
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
async def test_yaml_pipelines_lines_424_434_1():
    """Test YAML pipeline from docs_sphinx/yaml_pipelines.rst lines 424-434."""
    import yaml
    import orchestrator
    
    yaml_content = ("""# Compile-time (resolved during compilation)
config:
  timestamp: "{{ compile_time.timestamp }}"

# Runtime (resolved during execution)
steps:
  - id: dynamic
    parameters:
      query: "{{ inputs.topic }}"  # Runtime
      results: "$results.previous"  # Runtime""")
    
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
async def test_yaml_pipelines_lines_445_464_2():
    """Test YAML pipeline from docs_sphinx/yaml_pipelines.rst lines 445-464."""
    import yaml
    import orchestrator
    
    yaml_content = ("""imports:
  # Import specific steps
  - common/data_validation.yaml#validate_step as validate

  # Import entire pipeline
  - workflows/standard_analysis.yaml as analysis

steps:
  # Use imported step
  - id: validation
    extends: validate
    parameters:
      data: "$results.fetch"

  # Use imported pipeline
  - id: analyze
    pipeline: analysis
    inputs:
      data: "$results.validation"""")
    
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
async def test_yaml_pipelines_lines_470_498_3():
    """Test YAML pipeline from docs_sphinx/yaml_pipelines.rst lines 470-498."""
    import yaml
    import orchestrator
    
    yaml_content = ("""steps:
  # Define parallel group
  - id: parallel_fetch
    parallel:
      - id: fetch_api
        action: fetch_url
        parameters:
          url: "{{ inputs.api_url }}"

      - id: fetch_db
        action: query_database
        parameters:
          query: "{{ inputs.db_query }}"

      - id: fetch_file
        action: read_file
        parameters:
          path: "{{ inputs.file_path }}"

  # Use results from parallel group
  - id: merge
    action: combine_data
    depends_on: [parallel_fetch]
    parameters:
      sources:
        - "$results.parallel_fetch.fetch_api"
        - "$results.parallel_fetch.fetch_db"
        - "$results.parallel_fetch.fetch_file"""")
    
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
async def test_yaml_pipelines_lines_504_521_4():
    """Test YAML pipeline from docs_sphinx/yaml_pipelines.rst lines 504-521."""
    import yaml
    import orchestrator
    
    yaml_content = ("""steps:
  # For-each loop
  - id: process_items
    for_each: "{{ inputs.items }}"
    as: item
    action: process_single_item
    parameters:
      data: "{{ item }}"
      index: "{{ loop.index }}"

  # While loop
  - id: iterative_refinement
    while: "{{ results.quality_check.score < 0.95 }}"
    max_iterations: 10
    action: refine_result
    parameters:
      current: "$results.previous_iteration"""")
    
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
async def test_yaml_pipelines_lines_527_541_5():
    """Test YAML pipeline from docs_sphinx/yaml_pipelines.rst lines 527-541."""
    import yaml
    import orchestrator
    
    yaml_content = ("""# Enable checkpointing
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
      max_attempts: 3""")
    
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
async def test_yaml_pipelines_lines_578_638_6():
    """Test YAML pipeline from docs_sphinx/yaml_pipelines.rst lines 578-638."""
    import yaml
    import orchestrator
    
    yaml_content = ("""name: data-processing-pipeline
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
      format: "{{ inputs.output_format }}"""")
    
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
async def test_yaml_pipelines_lines_644_703_7():
    """Test YAML pipeline from docs_sphinx/yaml_pipelines.rst lines 644-703."""
    import yaml
    import orchestrator
    
    yaml_content = ("""name: comprehensive-research
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
      include_confidence: true""")
    
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
