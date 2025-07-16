"""Tests for documentation code snippets - Batch 11."""
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


def test_tutorial_web_research_lines_811_843_0():
    """Test Python import from docs_sphinx/tutorials/tutorial_web_research.rst lines 811-843."""
    # Test imports
    try:
        exec("""import orchestrator as orc

# Initialize
orc.init_models()

# Compile report generator
generator = orc.compile("report_generator.yaml")

# Generate executive report
exec_report = generator.run(
    topic="artificial intelligence in healthcare",
    report_type="executive",
    target_audience="executives",
    sections=["executive_summary", "key_findings", "recommendations"]
)

# Generate technical report
tech_report = generator.run(
    topic="blockchain scalability solutions",
    report_type="technical",
    target_audience="technical",
    sections=["introduction", "technical_analysis", "methodology", "results"]
)

# Generate standard briefing
briefing = generator.run(
    topic="cybersecurity threats 2024",
    report_type="briefing",
    target_audience="general"
)

print(f"Generated reports: {exec_report}, {tech_report}, {briefing}")""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

@pytest.mark.asyncio
async def test_tutorial_web_research_lines_854_865_1():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_web_research.rst lines 854-865."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """# Hints for your solution:
inputs:
  industry: # e.g., "fintech", "biotech", "cleantech"
  monitoring_period: # "daily", "weekly", "monthly"
  alert_keywords: # Important terms to watch for

steps:
  # Multiple search strategies
  # Trend analysis
  # Alert generation
  # Automated summaries"""
    
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

def test_tutorial_web_research_lines_873_878_2():
    """Test YAML snippet from docs_sphinx/tutorials/tutorial_web_research.rst lines 873-878."""
    import yaml
    
    yaml_content = """# Structure your pipeline to:
# 1. Research multiple companies
# 2. Compare features and positioning
# 3. Analyze market trends
# 4. Generate competitive analysis"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.asyncio
async def test_tutorial_web_research_lines_886_893_3():
    """Test Python snippet from docs_sphinx/tutorials/tutorial_web_research.rst lines 886-893."""
    # Create a pipeline that combines multiple research pipelines for comprehensive analysis:
    
    # Import required modules
    import os
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Set up test environment
        os.environ['ORCHESTRATOR_CONFIG'] = str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml")
        
        # Test code snippet
        code = """# Combine:
# - Basic web search
# - Multi-source research
# - Fact-checking
# - Report generation

# Into a single meta-pipeline"""
        
        # Execute with real models (API keys from environment/GitHub secrets)
        try:
            # Check if required API keys are available
            missing_keys = []
            if 'openai' in code.lower() and not os.environ.get('OPENAI_API_KEY'):
                missing_keys.append('OPENAI_API_KEY')
            if 'anthropic' in code.lower() and not os.environ.get('ANTHROPIC_API_KEY'):
                missing_keys.append('ANTHROPIC_API_KEY')
            if ('gemini' in code.lower() or 'google' in code.lower()) and not os.environ.get('GOOGLE_AI_API_KEY'):
                missing_keys.append('GOOGLE_AI_API_KEY')
            
            if missing_keys:
                pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
            
            # Execute the code with real models
            if 'await' in code or 'async' in code:
                # Handle async code
                import asyncio
                exec_globals = {'__name__': '__main__', 'asyncio': asyncio}
                exec(code, exec_globals)
                
                # If there's a main coroutine, run it
                if 'main' in exec_globals and asyncio.iscoroutinefunction(exec_globals['main']):
                    await exec_globals['main']()
            else:
                exec(code, {'__name__': '__main__'})
                
        except Exception as e:
            # Check if it's an expected error
            if "No eligible models" in str(e):
                pytest.skip(f"No eligible models available: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")

@pytest.mark.asyncio
async def test_yaml_pipelines_lines_14_67_4():
    """Test YAML pipeline from docs_sphinx/yaml_pipelines.rst lines 14-67."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """# Pipeline metadata
name: pipeline-name           # Required: Unique identifier
description: Pipeline purpose # Required: Human-readable description
version: "1.0.0"             # Optional: Version tracking

# Input definitions
inputs:
  parameter_name:
    type: string             # Required: string, integer, float, boolean, array, object
    description: Purpose     # Required: What this input does
    required: true          # Optional: Default is false
    default: "value"        # Optional: Default value if not provided
    validation:             # Optional: Input validation rules
      pattern: "^[a-z]+$"   # Regex for strings
      min: 0                # Minimum for numbers
      max: 100              # Maximum for numbers
      enum: ["a", "b"]      # Allowed values

# Output definitions
outputs:
  result_name:
    type: string            # Required: Output data type
    value: "expression"     # Required: How to generate the output
    description: Purpose    # Optional: What this output represents

# Configuration
config:
  timeout: 3600             # Optional: Global timeout in seconds
  parallel: true            # Optional: Enable parallel execution
  checkpoint: true          # Optional: Enable checkpointing
  error_mode: "continue"    # Optional: stop|continue|retry

# Resource requirements
resources:
  gpu: false                # Optional: Require GPU
  memory: "8GB"             # Optional: Memory requirement
  model_size: "large"       # Optional: Preferred model size

# Pipeline steps
steps:
  - id: step_identifier     # Required: Unique step ID
    action: action_name     # Required: What to do
    description: Purpose    # Optional: Step description
    parameters:             # Optional: Step parameters
      key: value
    depends_on: [step_id]   # Optional: Dependencies
    condition: expression   # Optional: Conditional execution
    error_handling:         # Optional: Error handling
      retry:
        max_attempts: 3
        backoff: exponential
      fallback:
        action: alternate_action"""
    
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

def test_yaml_pipelines_lines_78_87_5():
    """Test YAML snippet from docs_sphinx/yaml_pipelines.rst lines 78-87."""
    import yaml
    
    yaml_content = """name: advanced-research-pipeline
description: |
  Multi-stage research pipeline that:
  - Searches multiple sources
  - Validates information
  - Generates comprehensive reports
version: "2.1.0"
author: "Your Name"
tags: ["research", "automation", "reporting"]"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_yaml_pipelines_lines_97_148_6():
    """Test YAML snippet from docs_sphinx/yaml_pipelines.rst lines 97-148."""
    import yaml
    
    yaml_content = """inputs:
  # String input with validation
  topic:
    type: string
    description: "Research topic to investigate"
    required: true
    validation:
      pattern: "^[A-Za-z0-9 ]+$"
      min_length: 3
      max_length: 100

  # Integer with range
  depth:
    type: integer
    description: "Research depth (1-5)"
    default: 3
    validation:
      min: 1
      max: 5

  # Boolean flag
  include_images:
    type: boolean
    description: "Include images in report"
    default: false

  # Array of strings
  sources:
    type: array
    description: "Preferred information sources"
    default: ["web", "academic"]
    validation:
      min_items: 1
      max_items: 10
      item_type: string

  # Complex object
  config:
    type: object
    description: "Advanced configuration"
    default:
      language: "en"
      format: "pdf"
    validation:
      properties:
        language:
          type: string
          enum: ["en", "es", "fr", "de"]
        format:
          type: string
          enum: ["pdf", "html", "markdown"]"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_yaml_pipelines_lines_156_184_7():
    """Test YAML snippet from docs_sphinx/yaml_pipelines.rst lines 156-184."""
    import yaml
    
    yaml_content = """outputs:
  # Simple file output
  report:
    type: string
    value: "reports/{{ inputs.topic | slugify }}_report.pdf"
    description: "Generated PDF report"

  # Dynamic output using AUTO
  summary:
    type: string
    value: <AUTO>Generate filename based on content</AUTO>
    description: "Executive summary document"

  # Computed output
  metrics:
    type: object
    value:
      word_count: "{{ results.final_report.word_count }}"
      sources_used: "{{ results.compile_sources.count }}"
      generation_time: "{{ execution.duration }}"

  # Multiple file outputs
  artifacts:
    type: array
    value:
      - "{{ outputs.report }}"
      - "data/{{ inputs.topic }}_data.json"
      - "images/{{ inputs.topic }}_charts.png""""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.asyncio
async def test_yaml_pipelines_lines_194_226_8():
    """Test YAML pipeline from docs_sphinx/yaml_pipelines.rst lines 194-226."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """steps:
  # Simple action
  - id: fetch_data
    action: fetch_url
    parameters:
      url: "https://api.example.com/data"

  # Using input values
  - id: search
    action: search_web
    parameters:
      query: "{{ inputs.topic }} {{ inputs.year }}"
      max_results: "{{ inputs.depth * 5 }}"

  # Using previous results
  - id: analyze
    action: analyze_data
    parameters:
      data: "$results.fetch_data"
      method: "statistical"

  # Shell command (prefix with !)
  - id: convert
    action: "!pandoc -f markdown -t pdf -o output.pdf input.md"

  # Using AUTO tags
  - id: summarize
    action: generate_summary
    parameters:
      content: "$results.analyze"
      style: <AUTO>Choose style based on audience</AUTO>
      length: <AUTO>Determine optimal length</AUTO>"""
    
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
async def test_yaml_pipelines_lines_231_257_9():
    """Test YAML pipeline from docs_sphinx/yaml_pipelines.rst lines 231-257."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """steps:
  # Parallel execution (no dependencies)
  - id: source1
    action: fetch_source_a

  - id: source2
    action: fetch_source_b

  # Sequential execution
  - id: combine
    action: merge_data
    depends_on: [source1, source2]
    parameters:
      data1: "$results.source1"
      data2: "$results.source2"

  # Conditional execution
  - id: premium_analysis
    action: advanced_analysis
    condition: "{{ inputs.tier == 'premium' }}"
    parameters:
      data: "$results.combine"

  # Dynamic dependencies
  - id: final_step
    depends_on: "{{ ['combine', 'premium_analysis'] if inputs.tier == 'premium' else ['combine'] }}""""
    
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
async def test_yaml_pipelines_lines_262_284_10():
    """Test YAML pipeline from docs_sphinx/yaml_pipelines.rst lines 262-284."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """steps:
  - id: risky_operation
    action: external_api_call
    error_handling:
      # Retry configuration
      retry:
        max_attempts: 3
        backoff: exponential  # or: constant, linear
        initial_delay: 1000   # milliseconds
        max_delay: 30000

      # Fallback action
      fallback:
        action: use_cached_data
        parameters:
          cache_key: "{{ inputs.topic }}"

      # Continue on error
      continue_on_error: true

      # Custom error message
      error_message: "Failed to fetch external data, using cache""""
    
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

def test_yaml_pipelines_lines_294_308_11():
    """Test YAML snippet from docs_sphinx/yaml_pipelines.rst lines 294-308."""
    import yaml
    
    yaml_content = """# Input variables
"{{ inputs.parameter_name }}"

# Results from previous steps
"$results.step_id"
"$results.step_id.specific_field"

# Output references
"{{ outputs.output_name }}"

# Execution context
"{{ execution.timestamp }}"
"{{ execution.pipeline_id }}"
"{{ execution.run_id }}""""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_yaml_pipelines_lines_313_332_12():
    """Test YAML snippet from docs_sphinx/yaml_pipelines.rst lines 313-332."""
    import yaml
    
    yaml_content = """# String manipulation
"{{ inputs.topic | lower }}"
"{{ inputs.topic | upper }}"
"{{ inputs.topic | slugify }}"
"{{ inputs.topic | replace(' ', '_') }}"

# Date formatting
"{{ execution.timestamp | strftime('%Y-%m-%d') }}"

# Math operations
"{{ inputs.count * 2 }}"
"{{ inputs.value | round(2) }}"

# Conditionals
"{{ 'premium' if inputs.tier == 'gold' else 'standard' }}"

# Lists and loops
"{{ inputs.items | join(', ') }}"
"{{ inputs.sources | length }}""""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_yaml_pipelines_lines_342_353_13():
    """Test YAML snippet from docs_sphinx/yaml_pipelines.rst lines 342-353."""
    import yaml
    
    yaml_content = """parameters:
  # Simple decision
  style: <AUTO>Choose appropriate writing style</AUTO>

  # Context-aware decision
  method: <AUTO>Based on the data type {{ results.fetch.type }}, choose the best analysis method</AUTO>

  # Multiple choices
  options:
    visualization: <AUTO>Should we create visualizations for this data?</AUTO>
    format: <AUTO>What's the best output format: json, csv, or parquet?</AUTO>"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_yaml_pipelines_lines_358_379_14():
    """Test YAML snippet from docs_sphinx/yaml_pipelines.rst lines 358-379."""
    import yaml
    
    yaml_content = """# Conditional AUTO
analysis_depth: |
  <AUTO>
  Given:
  - Data size: {{ results.fetch.size }}
  - Time constraint: {{ inputs.deadline }}
  - Importance: {{ inputs.priority }}

  Determine the appropriate analysis depth (1-10)
  </AUTO>

# Structured AUTO
report_sections: |
  <AUTO>
  For a report about {{ inputs.topic }}, determine which sections to include:
  - Executive Summary: yes/no
  - Technical Details: yes/no
  - Future Outlook: yes/no
  - Recommendations: yes/no
  Return as JSON object
  </AUTO>"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_yaml_pipelines_lines_398_419_15():
    """Test Python import from docs_sphinx/yaml_pipelines.rst lines 398-419."""
    # Test imports
    try:
        exec("""import orchestrator as orc

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
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

@pytest.mark.asyncio
async def test_yaml_pipelines_lines_424_434_16():
    """Test YAML pipeline from docs_sphinx/yaml_pipelines.rst lines 424-434."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """# Compile-time (resolved during compilation)
config:
  timestamp: "{{ compile_time.timestamp }}"

# Runtime (resolved during execution)
steps:
  - id: dynamic
    parameters:
      query: "{{ inputs.topic }}"  # Runtime
      results: "$results.previous"  # Runtime"""
    
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
async def test_yaml_pipelines_lines_445_464_17():
    """Test YAML pipeline from docs_sphinx/yaml_pipelines.rst lines 445-464."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """imports:
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
      data: "$results.validation""""
    
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
async def test_yaml_pipelines_lines_470_498_18():
    """Test YAML pipeline from docs_sphinx/yaml_pipelines.rst lines 470-498."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """steps:
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
        - "$results.parallel_fetch.fetch_file""""
    
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
async def test_yaml_pipelines_lines_504_521_19():
    """Test YAML pipeline from docs_sphinx/yaml_pipelines.rst lines 504-521."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """steps:
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
      current: "$results.previous_iteration""""
    
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
async def test_yaml_pipelines_lines_527_541_20():
    """Test YAML pipeline from docs_sphinx/yaml_pipelines.rst lines 527-541."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """# Enable checkpointing
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
async def test_yaml_pipelines_lines_578_638_21():
    """Test YAML pipeline from docs_sphinx/yaml_pipelines.rst lines 578-638."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """name: data-processing-pipeline
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
async def test_yaml_pipelines_lines_644_703_22():
    """Test YAML pipeline from docs_sphinx/yaml_pipelines.rst lines 644-703."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """name: comprehensive-research
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
