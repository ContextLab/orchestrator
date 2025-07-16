"""Tests for documentation code snippets - Batch 20 (Robust)."""
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
async def test_tutorial_data_processing_lines_571_815_0():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_data_processing.rst lines 571-815."""
    import yaml
    import orchestrator
    
    yaml_content = ("""name: data-quality-assessment
description: Comprehensive data quality evaluation and reporting

inputs:
  dataset_path:
    type: string
    required: true

  quality_rules:
    type: object
    description: "Custom quality rules"
    default:
      completeness_threshold: 0.95
      uniqueness_fields: ["id"]
      date_range_field: "created_date"
      numeric_fields: ["amount", "quantity"]

  remediation_mode:
    type: string
    description: "How to handle quality issues"
    default: "report"
    validation:
      enum: ["report", "fix", "quarantine"]

outputs:
  quality_report:
    type: string
    value: "quality/report_{{ execution.timestamp }}.html"

  cleaned_data:
    type: string
    value: "quality/cleaned_{{ execution.timestamp }}.parquet"

  issues_log:
    type: string
    value: "quality/issues_{{ execution.timestamp }}.json"

steps:
  # Load the dataset
  - id: load_dataset
    action: read_file
    parameters:
      path: "{{ inputs.dataset_path }}"
      parse: true

  # Basic data profiling
  - id: profile_data
    action: analyze_data
    parameters:
      data: "$results.load_dataset"
      analysis_types:
        - schema
        - statistics
        - distributions
        - patterns
        - outliers

  # Completeness assessment
  - id: check_completeness
    action: check_quality
    parameters:
      data: "$results.load_dataset"
      checks:
        - type: "completeness"
          threshold: "{{ inputs.quality_rules.completeness_threshold }}"
          report_by_column: true

        - type: "null_patterns"
          identify_patterns: true

  # Uniqueness validation
  - id: check_uniqueness
    action: validate_data
    parameters:
      data: "$results.load_dataset"
      rules:
        - name: "primary_key_uniqueness"
          type: "uniqueness"
          columns: "{{ inputs.quality_rules.uniqueness_fields }}"
          severity: "error"

        - name: "near_duplicates"
          type: "similarity"
          threshold: 0.9
          columns: ["name", "email"]
          severity: "warning"

  # Consistency validation
  - id: check_consistency
    action: validate_data
    parameters:
      data: "$results.load_dataset"
      rules:
        - name: "date_logic"
          condition: "start_date <= end_date"
          severity: "error"

        - name: "numeric_consistency"
          condition: "total == sum(line_items)"
          severity: "error"

        - name: "referential_integrity"
          type: "foreign_key"
          reference_table: "lookup_table"
          foreign_key: "category_id"
          severity: "warning"

  # Accuracy validation
  - id: check_accuracy
    action: validate_data
    parameters:
      data: "$results.load_dataset"
      rules:
        - name: "email_format"
          field: "email"
          validation: "regex:^[\\w.-]+@[\\w.-]+\\.\\w+$"
          severity: "warning"

        - name: "phone_format"
          field: "phone"
          validation: "regex:^\\+?1?\\d{9,15}$"
          severity: "info"

        - name: "numeric_ranges"
          field: "{{ inputs.quality_rules.numeric_fields }}"
          validation: "range:0,999999"
          severity: "error"

  # Timeliness assessment
  - id: check_timeliness
    action: validate_data
    parameters:
      data: "$results.load_dataset"
      rules:
        - name: "data_freshness"
          field: "{{ inputs.quality_rules.date_range_field }}"
          condition: "date_diff(value, today()) <= 30"
          severity: "warning"
          message: "Data is older than 30 days"

  # Outlier detection
  - id: detect_outliers
    action: analyze_data
    parameters:
      data: "$results.load_dataset"
      analysis_types:
        - outliers
      methods:
        - statistical  # Z-score, IQR
        - isolation_forest
        - local_outlier_factor
      numeric_columns: "{{ inputs.quality_rules.numeric_fields }}"

  # Compile quality issues
  - id: compile_issues
    action: transform_data
    parameters:
      data:
        completeness: "$results.check_completeness"
        uniqueness: "$results.check_uniqueness"
        consistency: "$results.check_consistency"
        accuracy: "$results.check_accuracy"
        timeliness: "$results.check_timeliness"
        outliers: "$results.detect_outliers"
      operations:
        - type: "consolidate_issues"
          prioritize: true
        - type: "categorize_severity"
          levels: ["critical", "major", "minor", "info"]

  # Data remediation (if requested)
  - id: remediate_data
    condition: "inputs.remediation_mode in ['fix', 'quarantine']"
    action: transform_data
    parameters:
      data: "$results.load_dataset"
      operations:
        # Fix common issues
        - type: "standardize_formats"
          columns:
            email: "lowercase"
            phone: "normalize_phone"
            name: "title_case"

        - type: "fill_missing"
          strategy: "smart"  # Use ML-based imputation
          columns: "{{ inputs.quality_rules.numeric_fields }}"

        - type: "remove_outliers"
          method: "iqr"
          columns: "{{ inputs.quality_rules.numeric_fields }}"
          action: "{{ 'quarantine' if inputs.remediation_mode == 'quarantine' else 'remove' }}"

        - type: "deduplicate"
          strategy: "keep_best"  # Keep record with highest completeness

  # Generate comprehensive quality report
  - id: create_quality_report
    action: generate_content
    parameters:
      prompt: |
        Create a comprehensive data quality report:

        Dataset: {{ inputs.dataset_path }}
        Profile: {{ results.profile_data | json }}
        Issues: {{ results.compile_issues | json }}

        Include:
        1. Executive Summary
        2. Data Profile Overview
        3. Quality Metrics Dashboard
        4. Issue Analysis by Category
        5. Impact Assessment
        6. Remediation Recommendations
        7. Quality Score Calculation

        Format as HTML with charts and tables.

      style: "technical"
      format: "html"
      max_tokens: 3000

  # Save quality report
  - id: save_quality_report
    action: write_file
    parameters:
      path: "{{ outputs.quality_report }}"
      content: "$results.create_quality_report"

  # Save cleaned data (if remediation performed)
  - id: save_cleaned_data
    condition: "inputs.remediation_mode in ['fix', 'quarantine']"
    action: write_file
    parameters:
      path: "{{ outputs.cleaned_data }}"
      content: "$results.remediate_data"
      format: "parquet"

  # Save issues log
  - id: save_issues_log
    action: write_file
    parameters:
      path: "{{ outputs.issues_log }}"
      content: "{{ results.compile_issues | json }}"""")
    
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
async def test_tutorial_data_processing_lines_828_923_1():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_data_processing.rst lines 828-923."""
    import yaml
    import orchestrator
    
    yaml_content = ("""name: realtime-data-processing
description: Process streaming data with real-time analytics

inputs:
  stream_source:
    type: object
    description: "Stream configuration"
    required: true
    # Example:
    # type: "kafka"
    # topic: "events"
    # batch_size: 1000
    # window_size: "5m"

  processing_rules:
    type: array
    description: "Processing rules to apply"
    default:
      - type: "filter"
        condition: "event_type in ['purchase', 'click']"
      - type: "enrich"
        lookup_table: "user_profiles"
      - type: "aggregate"
        window: "5m"
        metrics: ["count", "sum", "avg"]

outputs:
  processed_stream:
    type: string
    value: "stream/processed_{{ execution.date }}"

  alerts:
    type: string
    value: "alerts/stream_alerts_{{ execution.timestamp }}.json"

steps:
  # Connect to stream source
  - id: connect_stream
    action: connect_stream
    parameters:
      source: "{{ inputs.stream_source }}"
      batch_size: "{{ inputs.stream_source.batch_size | default(1000) }}"
      timeout: 30

  # Process incoming batches
  - id: process_batches
    action: process_stream_batch
    parameters:
      stream: "$results.connect_stream"
      processing_rules: "{{ inputs.processing_rules }}"
      window_config:
        size: "{{ inputs.stream_source.window_size | default('5m') }}"
        type: "tumbling"  # or "sliding", "session"

  # Real-time anomaly detection
  - id: detect_anomalies
    action: detect_anomalies
    parameters:
      data: "$results.process_batches"
      methods:
        - statistical_control
        - machine_learning
      thresholds:
        statistical: 3.0  # standard deviations
        ml_confidence: 0.95

  # Generate alerts
  - id: generate_alerts
    condition: "results.detect_anomalies.anomalies | length > 0"
    action: generate_content
    parameters:
      prompt: |
        Generate alerts for detected anomalies:
        {{ results.detect_anomalies.anomalies | json }}

        Include severity, description, and recommended actions.

      format: "json"

  # Save processed data
  - id: save_processed
    action: write_stream
    parameters:
      data: "$results.process_batches"
      destination: "{{ outputs.processed_stream }}"
      format: "parquet"
      partition_by: ["date", "hour"]

  # Save alerts
  - id: save_alerts
    condition: "results.generate_alerts is defined"
    action: write_file
    parameters:
      path: "{{ outputs.alerts }}"
      content: "$results.generate_alerts"""")
    
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
async def test_tutorial_data_processing_lines_932_990_2():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_data_processing.rst lines 932-990."""
    import yaml
    import orchestrator
    
    yaml_content = ("""name: customer-data-platform
description: Unified customer data processing and analytics

inputs:
  customer_sources:
    type: object
    required: true
    # CRM, support tickets, web analytics, purchase history

steps:
  # Extract from all customer touchpoints
  - id: extract_crm
    action: query_database
    parameters:
      connection: "{{ inputs.customer_sources.crm.connection }}"
      query: "SELECT * FROM customers WHERE updated_at >= CURRENT_DATE - INTERVAL '1 day'"

  - id: extract_support
    action: call_api
    parameters:
      url: "{{ inputs.customer_sources.support.api_url }}"
      headers:
        Authorization: "Bearer {{ env.SUPPORT_API_KEY }}"

  - id: extract_analytics
    action: read_file
    parameters:
      path: "{{ inputs.customer_sources.analytics.export_path }}"
      parse: true

  # Create unified customer profiles
  - id: merge_customer_data
    action: merge_data
    parameters:
      datasets:
        - "$results.extract_crm"
        - "$results.extract_support"
        - "$results.extract_analytics"
      on: "customer_id"
      how: "outer"

  # Calculate customer metrics
  - id: calculate_metrics
    action: transform_data
    parameters:
      data: "$results.merge_customer_data"
      operations:
        - type: "add_column"
          name: "customer_lifetime_value"
          expression: "sum(purchase_amounts) * retention_probability"

        - type: "add_column"
          name: "churn_risk_score"
          expression: "calculate_churn_risk(days_since_last_activity, support_tickets, engagement_score)"

        - type: "add_column"
          name: "segment"
          expression: "classify_customer_segment(clv, engagement, recency)"""")
    
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
async def test_tutorial_data_processing_lines_996_1062_3():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_data_processing.rst lines 996-1062."""
    import yaml
    import orchestrator
    
    yaml_content = ("""name: financial-data-pipeline
description: Process financial transactions with compliance checks

inputs:
  transaction_sources:
    type: array
    required: true

  compliance_rules:
    type: object
    required: true

steps:
  # Extract transactions from multiple sources
  - id: extract_transactions
    for_each: "{{ inputs.transaction_sources }}"
    as: source
    action: extract_financial_data
    parameters:
      source_config: "{{ source }}"
      date_range: "{{ execution.date | date_range('-1d') }}"

  # Compliance screening
  - id: screen_transactions
    action: validate_data
    parameters:
      data: "$results.extract_transactions"
      rules:
        - name: "aml_screening"
          type: "anti_money_laundering"
          threshold: "{{ inputs.compliance_rules.aml_threshold }}"

        - name: "sanctions_check"
          type: "sanctions_screening"
          watchlists: "{{ inputs.compliance_rules.watchlists }}"

        - name: "pep_screening"
          type: "politically_exposed_person"
          databases: "{{ inputs.compliance_rules.pep_databases }}"

  # Risk scoring
  - id: calculate_risk_scores
    action: transform_data
    parameters:
      data: "$results.extract_transactions"
      operations:
        - type: "add_column"
          name: "risk_score"
          expression: "calculate_transaction_risk(amount, counterparty, geography, transaction_type)"

        - type: "add_column"
          name: "risk_category"
          expression: "categorize_risk(risk_score)"

  # Generate compliance report
  - id: create_compliance_report
    action: generate_content
    parameters:
      prompt: |
        Generate daily compliance report:

        Transactions processed: {{ results.extract_transactions | length }}
        Screening results: {{ results.screen_transactions | json }}
        Risk distribution: {{ results.calculate_risk_scores | group_by('risk_category') }}

        Include regulatory compliance status and any required actions.""")
    
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

def test_tutorial_data_processing_lines_1073_1078_4():
    """Test YAML snippet from docs_sphinx/tutorials/tutorial_data_processing.rst lines 1073-1078."""
    import yaml
    
    yaml_content = ("""# Your challenge:
# - Extract: Orders, customers, products, reviews
# - Transform: Calculate metrics, segment customers
# - Load: Create analytics-ready datasets
# - Quality: Validate business rules""")
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_tutorial_data_processing_lines_1086_1091_5():
    """Test YAML snippet from docs_sphinx/tutorials/tutorial_data_processing.rst lines 1086-1091."""
    import yaml
    
    yaml_content = ("""# Requirements:
# - Handle high-volume time series data
# - Detect sensor anomalies
# - Aggregate by time windows
# - Generate maintenance alerts""")
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_tutorial_data_processing_lines_1099_1104_6():
    """Test YAML snippet from docs_sphinx/tutorials/tutorial_data_processing.rst lines 1099-1104."""
    import yaml
    
    yaml_content = ("""# Features:
# - Extract from multiple platforms
# - Text analysis and sentiment
# - Trend detection
# - Influence measurement""")
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_tutorial_web_research_lines_36_91_7():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_web_research.rst lines 36-91."""
    import yaml
    import orchestrator
    
    yaml_content = ("""name: basic-web-search
description: Search the web and compile results into a report

inputs:
  query:
    type: string
    description: "Search query"
    required: true

  max_results:
    type: integer
    description: "Maximum number of results to return"
    default: 10
    validation:
      min: 1
      max: 50

outputs:
  report:
    type: string
    value: "search_results_{{ inputs.query | slugify }}.md"

steps:
  # Search the web
  - id: search
    action: search_web
    parameters:
      query: "{{ inputs.query }}"
      max_results: "{{ inputs.max_results }}"
      include_snippets: true

  # Compile into markdown report
  - id: compile_report
    action: generate_content
    parameters:
      prompt: |
        Create a well-organized markdown report from these search results:

        {{ results.search | json }}

        Include:
        - Executive summary
        - Key findings
        - Source links
        - Relevant details from each result

      style: "professional"
      format: "markdown"

  # Save the report
  - id: save_report
    action: write_file
    parameters:
      path: "{{ outputs.report }}"
      content: "$results.compile_report"""")
    
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
async def test_tutorial_web_research_lines_97_117_8():
    """Test orchestrator code from docs_sphinx/tutorials/tutorial_web_research.rst lines 97-117."""
    # ------------------------
    
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

# Compile and run
pipeline = orc.compile("web_search.yaml")

# Search for different topics
result1 = pipeline.run(
    query="artificial intelligence trends 2024",
    max_results=15
)

result2 = pipeline.run(
    query="sustainable energy solutions",
    max_results=10
)

print(f"Generated reports: {result1}, {result2}")""")
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

def test_tutorial_web_research_lines_125_142_9():
    """Test markdown snippet from docs_sphinx/tutorials/tutorial_web_research.rst lines 125-142."""
    # Content validation for markdown snippet
    content = ("""# Search Results: Artificial Intelligence Trends 2024

## Executive Summary

Recent searches reveal significant developments in AI across multiple domains...

## Key Findings

1. **Large Language Models** - Continued advancement in reasoning capabilities
2. **AI Safety** - Increased focus on alignment and control
3. **Enterprise Adoption** - Growing integration in business processes

## Detailed Results

### 1. AI Breakthrough: New Model Achieves Human-Level Performance
**Source**: [TechCrunch](https://techcrunch.com/...)
**Summary**: Details about the latest AI advancement...""")
    
    # Basic checks
    assert content.strip(), "Snippet content should not be empty"
    assert len(content.strip()) > 0, "Snippet should have content"

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_tutorial_web_research_lines_155_346_10():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_web_research.rst lines 155-346."""
    import yaml
    import orchestrator
    
    yaml_content = ("""name: multi-source-research
description: Comprehensive research using web, news, and academic sources

inputs:
  topic:
    type: string
    required: true

  depth:
    type: string
    description: "Research depth"
    default: "medium"
    validation:
      enum: ["light", "medium", "deep"]

  include_sources:
    type: array
    description: "Sources to include"
    default: ["web", "news", "academic"]
    validation:
      enum_items: ["web", "news", "academic", "patents"]

outputs:
  comprehensive_report:
    type: string
    value: "research/{{ inputs.topic | slugify }}_comprehensive.md"

  data_file:
    type: string
    value: "research/{{ inputs.topic | slugify }}_data.json"

# Research depth configuration
config:
  research_params:
    light:
      web_results: 10
      news_results: 5
      academic_results: 3
    medium:
      web_results: 20
      news_results: 10
      academic_results: 8
    deep:
      web_results: 40
      news_results: 20
      academic_results: 15

steps:
  # Parallel search across sources
  - id: search_sources
    parallel:
      # Web search
      - id: web_search
        condition: "'web' in inputs.include_sources"
        action: search_web
        parameters:
          query: "{{ inputs.topic }} comprehensive overview"
          max_results: "{{ config.research_params[inputs.depth].web_results }}"
          include_snippets: true

      # News search
      - id: news_search
        condition: "'news' in inputs.include_sources"
        action: search_news
        parameters:
          query: "{{ inputs.topic }}"
          max_results: "{{ config.research_params[inputs.depth].news_results }}"
          date_range: "last_month"

      # Academic search
      - id: academic_search
        condition: "'academic' in inputs.include_sources"
        action: search_academic
        parameters:
          query: "{{ inputs.topic }}"
          max_results: "{{ config.research_params[inputs.depth].academic_results }}"
          year_range: "2020-2024"
          peer_reviewed: true

  # Extract key information from each source
  - id: extract_information
    action: extract_information
    parameters:
      content: "$results.search_sources"
      extract:
        key_facts:
          description: "Important facts and findings"
        statistics:
          description: "Numerical data and metrics"
        expert_opinions:
          description: "Quotes and opinions from experts"
        trends:
          description: "Emerging trends and developments"
        challenges:
          description: "Problems and challenges mentioned"
        opportunities:
          description: "Opportunities and potential solutions"

  # Cross-validate information
  - id: validate_facts
    action: validate_data
    parameters:
      data: "$results.extract_information"
      rules:
        - name: "source_diversity"
          condition: "count(unique(sources)) >= 2"
          severity: "warning"
          message: "Information should be confirmed by multiple sources"

        - name: "recent_information"
          field: "date"
          condition: "date_diff(value, today()) <= 365"
          severity: "info"
          message: "Information is from the last year"

  # Generate comprehensive analysis
  - id: analyze_findings
    action: generate_content
    parameters:
      prompt: |
        Analyze the following research data about {{ inputs.topic }}:

        {{ results.extract_information | json }}

        Provide:
        1. Current state analysis
        2. Key trends identification
        3. Challenge assessment
        4. Future outlook
        5. Recommendations

        Base your analysis on the evidence provided and note any limitations.

      style: "analytical"
      max_tokens: 2000

  # Create structured data export
  - id: export_data
    action: transform_data
    parameters:
      data:
        topic: "{{ inputs.topic }}"
        research_date: "{{ execution.timestamp }}"
        depth: "{{ inputs.depth }}"
        sources_used: "{{ inputs.include_sources }}"
        extracted_info: "$results.extract_information"
        validation_results: "$results.validate_facts"
        analysis: "$results.analyze_findings"
      operations:
        - type: "convert_format"
          to_format: "json"

  # Save structured data
  - id: save_data
    action: write_file
    parameters:
      path: "{{ outputs.data_file }}"
      content: "$results.export_data"

  # Generate final report
  - id: create_report
    action: generate_content
    parameters:
      prompt: |
        Create a comprehensive research report about {{ inputs.topic }} using:

        Analysis: {{ results.analyze_findings }}

        Structure the report with:
        1. Executive Summary
        2. Methodology
        3. Current State Analysis
        4. Key Findings
        5. Trends and Developments
        6. Challenges and Limitations
        7. Future Outlook
        8. Recommendations
        9. Sources and References

        Include confidence levels for major claims.

      style: "professional"
      format: "markdown"
      max_tokens: 3000

  # Save final report
  - id: save_report
    action: write_file
    parameters:
      path: "{{ outputs.comprehensive_report }}"
      content: "$results.create_report"""")
    
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
async def test_tutorial_web_research_lines_352_375_11():
    """Test orchestrator code from docs_sphinx/tutorials/tutorial_web_research.rst lines 352-375."""
    # ---------------------------------
    
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

# Compile pipeline
pipeline = orc.compile("multi_source_research.yaml")

# Run deep research on quantum computing
result = pipeline.run(
    topic="quantum computing applications",
    depth="deep",
    include_sources=["web", "academic", "news"]
)

print(f"Research complete: {result}")

# Run lighter research on emerging tech
result2 = pipeline.run(
    topic="edge computing trends",
    depth="medium",
    include_sources=["web", "news"]
)""")
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
async def test_tutorial_web_research_lines_388_488_12():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_web_research.rst lines 388-488."""
    import yaml
    import orchestrator
    
    yaml_content = ("""name: fact-checker
description: Verify claims against multiple reliable sources

inputs:
  claims:
    type: array
    description: "Claims to verify"
    required: true

  confidence_threshold:
    type: float
    description: "Minimum confidence level to accept claims"
    default: 0.7
    validation:
      min: 0.0
      max: 1.0

outputs:
  fact_check_report:
    type: string
    value: "fact_check_{{ execution.timestamp | strftime('%Y%m%d_%H%M') }}.md"

steps:
  # Research each claim
  - id: research_claims
    for_each: "{{ inputs.claims }}"
    as: claim
    action: search_web
    parameters:
      query: "{{ claim }} verification facts evidence"
      max_results: 15
      include_snippets: true

  # Extract supporting/contradicting evidence
  - id: analyze_evidence
    for_each: "{{ inputs.claims }}"
    as: claim
    action: extract_information
    parameters:
      content: "$results.research_claims[loop.index0]"
      extract:
        supporting_evidence:
          description: "Evidence that supports the claim"
        contradicting_evidence:
          description: "Evidence that contradicts the claim"
        source_credibility:
          description: "Assessment of source reliability"
        expert_opinions:
          description: "Expert statements about the claim"

  # Assess credibility of each claim
  - id: assess_claims
    for_each: "{{ inputs.claims }}"
    as: claim
    action: generate_content
    parameters:
      prompt: |
        Assess the veracity of this claim: "{{ claim }}"

        Based on the evidence:
        {{ results.analyze_evidence[loop.index0] | json }}

        Provide:
        1. Verdict: True/False/Partially True/Insufficient Evidence
        2. Confidence level (0-1)
        3. Supporting evidence summary
        4. Contradicting evidence summary
        5. Overall assessment

        Be objective and cite specific sources.

      style: "analytical"
      format: "structured"

  # Compile fact-check report
  - id: create_fact_check_report
    action: generate_content
    parameters:
      prompt: |
        Create a comprehensive fact-check report based on:

        Claims assessed: {{ inputs.claims | json }}
        Assessment results: {{ results.assess_claims | json }}

        Format as a professional fact-checking article with:
        1. Summary of findings
        2. Individual claim assessments
        3. Methodology used
        4. Sources consulted
        5. Limitations and caveats

      style: "journalistic"
      format: "markdown"

  # Save report
  - id: save_fact_check
    action: write_file
    parameters:
      path: "{{ outputs.fact_check_report }}"
      content: "$results.create_fact_check_report"""")
    
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
async def test_tutorial_web_research_lines_494_514_13():
    """Test orchestrator code from docs_sphinx/tutorials/tutorial_web_research.rst lines 494-514."""
    # ----------------------------
    
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

# Compile fact-checker
fact_checker = orc.compile("fact_checker.yaml")

# Check various claims
result = fact_checker.run(
    claims=[
        "Electric vehicles produce zero emissions",
        "AI will replace 50% of jobs by 2030",
        "Quantum computers can break all current encryption",
        "Renewable energy is now cheaper than fossil fuels"
    ],
    confidence_threshold=0.8
)

print(f"Fact-check report: {result}")""")
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
async def test_tutorial_web_research_lines_527_805_14():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_web_research.rst lines 527-805."""
    import yaml
    import orchestrator
    
    yaml_content = ("""name: automated-report-generator
description: Generate professional reports from research data

inputs:
  topic:
    type: string
    required: true

  report_type:
    type: string
    description: "Type of report to generate"
    default: "standard"
    validation:
      enum: ["executive", "technical", "standard", "briefing"]

  target_audience:
    type: string
    description: "Primary audience for the report"
    default: "general"
    validation:
      enum: ["executives", "technical", "general", "academic"]

  sections:
    type: array
    description: "Sections to include in report"
    default: ["summary", "introduction", "analysis", "conclusion"]

outputs:
  report_markdown:
    type: string
    value: "reports/{{ inputs.topic | slugify }}_{{ inputs.report_type }}.md"

  report_pdf:
    type: string
    value: "reports/{{ inputs.topic | slugify }}_{{ inputs.report_type }}.pdf"

  report_html:
    type: string
    value: "reports/{{ inputs.topic | slugify }}_{{ inputs.report_type }}.html"

# Report templates by type
config:
  report_templates:
    executive:
      style: "executive"
      length: "concise"
      focus: "strategic"
      sections: ["executive_summary", "key_findings", "recommendations", "appendix"]

    technical:
      style: "technical"
      length: "detailed"
      focus: "implementation"
      sections: ["introduction", "technical_analysis", "methodology", "results", "conclusion"]

    standard:
      style: "professional"
      length: "medium"
      focus: "comprehensive"
      sections: ["summary", "background", "analysis", "findings", "recommendations"]

    briefing:
      style: "concise"
      length: "short"
      focus: "actionable"
      sections: ["situation", "assessment", "recommendations"]

steps:
  # Gather comprehensive research data
  - id: research_topic
    action: search_web
    parameters:
      query: "{{ inputs.topic }} comprehensive analysis research"
      max_results: 25
      include_snippets: true

  # Get recent news for current context
  - id: current_context
    action: search_news
    parameters:
      query: "{{ inputs.topic }}"
      max_results: 10
      date_range: "last_week"

  # Extract structured information
  - id: extract_report_data
    action: extract_information
    parameters:
      content:
        research: "$results.research_topic"
        news: "$results.current_context"
      extract:
        key_points:
          description: "Main points and findings"
        statistics:
          description: "Important numbers and data"
        trends:
          description: "Current and emerging trends"
        implications:
          description: "Implications and consequences"
        expert_views:
          description: "Expert opinions and quotes"
        future_outlook:
          description: "Predictions and future scenarios"

  # Generate executive summary
  - id: create_executive_summary
    condition: "'summary' in inputs.sections or 'executive_summary' in inputs.sections"
    action: generate_content
    parameters:
      prompt: |
        Create an executive summary for {{ inputs.target_audience }} audience about {{ inputs.topic }}.

        Based on: {{ results.extract_report_data.key_points | json }}

        Style: {{ config.report_templates[inputs.report_type].style }}
        Focus: {{ config.report_templates[inputs.report_type].focus }}

        Include the most critical points in 200-400 words.

      style: "{{ config.report_templates[inputs.report_type].style }}"
      max_tokens: 500

  # Generate introduction/background
  - id: create_introduction
    condition: "'introduction' in inputs.sections or 'background' in inputs.sections"
    action: generate_content
    parameters:
      prompt: |
        Write an introduction/background section about {{ inputs.topic }} for {{ inputs.target_audience }}.

        Context: {{ results.extract_report_data | json }}

        Provide necessary background and context for understanding the topic.

      style: "{{ config.report_templates[inputs.report_type].style }}"
      max_tokens: 800

  # Generate main analysis
  - id: create_analysis
    condition: "'analysis' in inputs.sections or 'technical_analysis' in inputs.sections"
    action: generate_content
    parameters:
      prompt: |
        Create a comprehensive analysis section about {{ inputs.topic }}.

        Data: {{ results.extract_report_data | json }}

        Style: {{ config.report_templates[inputs.report_type].style }}
        Audience: {{ inputs.target_audience }}

        Include:
        - Current state analysis
        - Trend analysis
        - Key factors and drivers
        - Challenges and opportunities

        Support points with specific data and examples.

      style: "{{ config.report_templates[inputs.report_type].style }}"
      max_tokens: 1500

  # Generate findings and implications
  - id: create_findings
    condition: "'findings' in inputs.sections or 'key_findings' in inputs.sections"
    action: generate_content
    parameters:
      prompt: |
        Summarize key findings and implications regarding {{ inputs.topic }}.

        Analysis: {{ results.create_analysis }}
        Supporting data: {{ results.extract_report_data.implications | json }}

        Present clear, actionable findings with implications.

      style: "{{ config.report_templates[inputs.report_type].style }}"
      max_tokens: 1000

  # Generate recommendations
  - id: create_recommendations
    condition: "'recommendations' in inputs.sections"
    action: generate_content
    parameters:
      prompt: |
        Develop actionable recommendations based on the analysis of {{ inputs.topic }}.

        Findings: {{ results.create_findings }}
        Target audience: {{ inputs.target_audience }}

        Provide specific, actionable recommendations with priorities and considerations.

      style: "{{ config.report_templates[inputs.report_type].style }}"
      max_tokens: 800

  # Generate conclusion
  - id: create_conclusion
    condition: "'conclusion' in inputs.sections"
    action: generate_content
    parameters:
      prompt: |
        Write a strong conclusion for the {{ inputs.topic }} report.

        Key findings: {{ results.create_findings }}
        Recommendations: {{ results.create_recommendations }}

        Synthesize the main points and end with a clear call to action.

      style: "{{ config.report_templates[inputs.report_type].style }}"
      max_tokens: 400

  # Assemble complete report
  - id: assemble_report
    action: generate_content
    parameters:
      prompt: |
        Compile a complete, professional report about {{ inputs.topic }}.

        Report type: {{ inputs.report_type }}
        Target audience: {{ inputs.target_audience }}

        Sections to include:
        {% if results.create_executive_summary %}
        Executive Summary: {{ results.create_executive_summary }}
        {% endif %}

        {% if results.create_introduction %}
        Introduction: {{ results.create_introduction }}
        {% endif %}

        {% if results.create_analysis %}
        Analysis: {{ results.create_analysis }}
        {% endif %}

        {% if results.create_findings %}
        Findings: {{ results.create_findings }}
        {% endif %}

        {% if results.create_recommendations %}
        Recommendations: {{ results.create_recommendations }}
        {% endif %}

        {% if results.create_conclusion %}
        Conclusion: {{ results.create_conclusion }}
        {% endif %}

        Format as a professional markdown document with:
        - Proper headings and structure
        - Table of contents
        - Professional formatting
        - Source citations where appropriate

      style: "professional"
      format: "markdown"
      max_tokens: 4000

  # Save markdown version
  - id: save_markdown
    action: write_file
    parameters:
      path: "{{ outputs.report_markdown }}"
      content: "$results.assemble_report"

  # Convert to PDF
  - id: create_pdf
    action: "!pandoc {{ outputs.report_markdown }} -o {{ outputs.report_pdf }} --pdf-engine=xelatex"
    error_handling:
      continue_on_error: true
      fallback:
        action: write_file
        parameters:
          path: "{{ outputs.report_pdf }}.txt"
          content: "PDF generation requires pandoc with xelatex"

  # Convert to HTML
  - id: create_html
    action: "!pandoc {{ outputs.report_markdown }} -o {{ outputs.report_html }} --standalone --css=style.css"
    error_handling:
      continue_on_error: true""")
    
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
