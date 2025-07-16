"""Tests for documentation code snippets - Batch 21 (Robust)."""
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
async def test_tutorial_web_research_lines_811_843_0():
    """Test orchestrator code from docs_sphinx/tutorials/tutorial_web_research.rst lines 811-843."""
    # ------------------------------------
    
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
async def test_tutorial_web_research_lines_854_865_1():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_web_research.rst lines 854-865."""
    import yaml
    import orchestrator
    
    yaml_content = ("""# Hints for your solution:
inputs:
  industry: # e.g., "fintech", "biotech", "cleantech"
  monitoring_period: # "daily", "weekly", "monthly"
  alert_keywords: # Important terms to watch for

steps:
  # Multiple search strategies
  # Trend analysis
  # Alert generation
  # Automated summaries""")
    
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

def test_tutorial_web_research_lines_873_878_2():
    """Test YAML snippet from docs_sphinx/tutorials/tutorial_web_research.rst lines 873-878."""
    import yaml
    
    yaml_content = ("""# Structure your pipeline to:
# 1. Research multiple companies
# 2. Compare features and positioning
# 3. Analyze market trends
# 4. Generate competitive analysis""")
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_tutorial_web_research_lines_886_893_3():
    """Test orchestrator code from docs_sphinx/tutorials/tutorial_web_research.rst lines 886-893."""
    # Create a pipeline that combines multiple research pipelines for comprehensive analysis:
    
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
        code = ("""# Combine:
# - Basic web search
# - Multi-source research
# - Fact-checking
# - Report generation

# Into a single meta-pipeline""")
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
async def test_yaml_pipelines_lines_14_67_4():
    """Test YAML pipeline from docs_sphinx/yaml_pipelines.rst lines 14-67."""
    import yaml
    import orchestrator
    
    yaml_content = ("""# Pipeline metadata
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
        action: alternate_action""")
    
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

def test_yaml_pipelines_lines_78_87_5():
    """Test YAML snippet from docs_sphinx/yaml_pipelines.rst lines 78-87."""
    import yaml
    
    yaml_content = ("""name: advanced-research-pipeline
description: |
  Multi-stage research pipeline that:
  - Searches multiple sources
  - Validates information
  - Generates comprehensive reports
version: "2.1.0"
author: "Your Name"
tags: ["research", "automation", "reporting"]""")
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_yaml_pipelines_lines_97_148_6():
    """Test YAML snippet from docs_sphinx/yaml_pipelines.rst lines 97-148."""
    import yaml
    
    yaml_content = ("""inputs:
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
          enum: ["pdf", "html", "markdown"]""")
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_yaml_pipelines_lines_156_184_7():
    """Test YAML snippet from docs_sphinx/yaml_pipelines.rst lines 156-184."""
    import yaml
    
    yaml_content = ("""outputs:
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
      - "images/{{ inputs.topic }}_charts.png"""")
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_yaml_pipelines_lines_194_226_8():
    """Test YAML pipeline from docs_sphinx/yaml_pipelines.rst lines 194-226."""
    import yaml
    import orchestrator
    
    yaml_content = ("""steps:
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
      length: <AUTO>Determine optimal length</AUTO>""")
    
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
async def test_yaml_pipelines_lines_231_257_9():
    """Test YAML pipeline from docs_sphinx/yaml_pipelines.rst lines 231-257."""
    import yaml
    import orchestrator
    
    yaml_content = ("""steps:
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
    depends_on: "{{ ['combine', 'premium_analysis'] if inputs.tier == 'premium' else ['combine'] }}"""")
    
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
async def test_yaml_pipelines_lines_262_284_10():
    """Test YAML pipeline from docs_sphinx/yaml_pipelines.rst lines 262-284."""
    import yaml
    import orchestrator
    
    yaml_content = ("""steps:
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
      error_message: "Failed to fetch external data, using cache"""")
    
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

def test_yaml_pipelines_lines_294_308_11():
    """Test YAML snippet from docs_sphinx/yaml_pipelines.rst lines 294-308."""
    import yaml
    
    yaml_content = ("""# Input variables
"{{ inputs.parameter_name }}"

# Results from previous steps
"$results.step_id"
"$results.step_id.specific_field"

# Output references
"{{ outputs.output_name }}"

# Execution context
"{{ execution.timestamp }}"
"{{ execution.pipeline_id }}"
"{{ execution.run_id }}"""")
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_yaml_pipelines_lines_313_332_12():
    """Test YAML snippet from docs_sphinx/yaml_pipelines.rst lines 313-332."""
    import yaml
    
    yaml_content = ("""# String manipulation
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
"{{ inputs.sources | length }}"""")
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_yaml_pipelines_lines_342_353_13():
    """Test YAML snippet from docs_sphinx/yaml_pipelines.rst lines 342-353."""
    import yaml
    
    yaml_content = ("""parameters:
  # Simple decision
  style: <AUTO>Choose appropriate writing style</AUTO>

  # Context-aware decision
  method: <AUTO>Based on the data type {{ results.fetch.type }}, choose the best analysis method</AUTO>

  # Multiple choices
  options:
    visualization: <AUTO>Should we create visualizations for this data?</AUTO>
    format: <AUTO>What's the best output format: json, csv, or parquet?</AUTO>""")
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_yaml_pipelines_lines_358_379_14():
    """Test YAML snippet from docs_sphinx/yaml_pipelines.rst lines 358-379."""
    import yaml
    
    yaml_content = ("""# Conditional AUTO
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
  </AUTO>""")
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
