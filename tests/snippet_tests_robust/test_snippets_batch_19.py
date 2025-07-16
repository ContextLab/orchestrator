"""Tests for documentation code snippets - Batch 19 (Robust)."""
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
async def test_tool_reference_lines_436_507_0():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 436-507."""
    import yaml
    import orchestrator
    
    yaml_content = ("""# Transform data
- id: transform
  action: transform_data
  parameters:
    data: "$results.load_data"               # Required: Input data or path
    operations:                              # Required: List of operations
      - type: "rename_columns"
        mapping:
          old_name: "new_name"
          price: "cost"
      - type: "add_column"
        name: "total"
        expression: "quantity * cost"
      - type: "drop_columns"
        columns: ["unnecessary_field"]
      - type: "convert_types"
        conversions:
          date: "datetime"
          amount: "float"

# Filter data
- id: filter
  action: filter_data
  parameters:
    data: "$results.transform"               # Required: Input data
    conditions:                              # Required: Filter conditions
      - field: "status"
        operator: "equals"                   # equals|not_equals|contains|gt|lt|gte|lte
        value: "active"
      - field: "amount"
        operator: "gt"
        value: 1000
    mode: "and"                              # Optional: and|or (default: and)

# Aggregate data
- id: aggregate
  action: aggregate_data
  parameters:
    data: "$results.filter"                  # Required: Input data
    group_by: ["category", "region"]        # Optional: Grouping columns
    aggregations:                            # Required: Aggregation rules
      total_amount:
        column: "amount"
        function: "sum"                      # sum|mean|median|min|max|count|std
      average_price:
        column: "price"
        function: "mean"
      item_count:
        column: "*"
        function: "count"

# Merge data
- id: merge
  action: merge_data
  parameters:
    left: "$results.main_data"               # Required: Left dataset
    right: "$results.lookup_data"            # Required: Right dataset
    on: "customer_id"                        # Required: Join column(s)
    how: "left"                              # Optional: left|right|inner|outer (default: left)
    suffixes: ["_main", "_lookup"]          # Optional: Column suffixes

# Convert format
- id: convert
  action: convert_format
  parameters:
    data: "$results.final_data"              # Required: Input data
    from_format: "json"                      # Optional: Auto-detect if not specified
    to_format: "parquet"                     # Required: Target format
    options:                                 # Optional: Format-specific options
      compression: "snappy"
      index: false""")
    
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
async def test_tool_reference_lines_512_582_1():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 512-582."""
    import yaml
    import orchestrator
    
    yaml_content = ("""name: sales-data-analysis
description: Process and analyze sales data

steps:
  # Load raw data
  - id: load_sales
    action: read_file
    parameters:
      path: "data/sales_2024.csv"
      parse: true

  # Clean and transform
  - id: clean_data
    action: transform_data
    parameters:
      data: "$results.load_sales"
      operations:
        - type: "rename_columns"
          mapping:
            "Sale Date": "sale_date"
            "Customer Name": "customer_name"
            "Product ID": "product_id"
            "Sale Amount": "amount"
        - type: "convert_types"
          conversions:
            sale_date: "datetime"
            amount: "float"
        - type: "add_column"
          name: "quarter"
          expression: "sale_date.quarter"

  # Filter valid sales
  - id: filter_valid
    action: filter_data
    parameters:
      data: "$results.clean_data"
      conditions:
        - field: "amount"
          operator: "gt"
          value: 0
        - field: "product_id"
          operator: "not_equals"
          value: null

  # Aggregate by quarter
  - id: quarterly_summary
    action: aggregate_data
    parameters:
      data: "$results.filter_valid"
      group_by: ["quarter", "product_id"]
      aggregations:
        total_sales:
          column: "amount"
          function: "sum"
        avg_sale:
          column: "amount"
          function: "mean"
        num_transactions:
          column: "*"
          function: "count"

  # Save results
  - id: save_summary
    action: convert_format
    parameters:
      data: "$results.quarterly_summary"
      to_format: "excel"
      options:
        sheet_name: "Quarterly Sales"
        index: false""")
    
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
async def test_tool_reference_lines_598_692_2():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 598-692."""
    import yaml
    import orchestrator
    
    yaml_content = ("""# Validate against schema
- id: validate_structure
  action: validate_schema
  parameters:
    data: "$results.processed_data"          # Required: Data to validate
    schema:                                  # Required: Validation schema
      type: "object"
      required: ["id", "name", "email"]
      properties:
        id:
          type: "integer"
          minimum: 1
        name:
          type: "string"
          minLength: 2
          maxLength: 100
        email:
          type: "string"
          format: "email"
        age:
          type: "integer"
          minimum: 0
          maximum: 150
    strict: false                            # Optional: Strict mode (default: false)

# Business rule validation
- id: validate_rules
  action: validate_data
  parameters:
    data: "$results.transactions"            # Required: Data to validate
    rules:                                   # Required: Validation rules
      - name: "positive_amounts"
        field: "amount"
        condition: "value > 0"
        severity: "error"                    # error|warning|info
        message: "Transaction amounts must be positive"

      - name: "valid_date_range"
        field: "transaction_date"
        condition: "value >= '2024-01-01' and value <= today()"
        severity: "error"

      - name: "customer_exists"
        field: "customer_id"
        condition: "value in valid_customers"
        severity: "warning"
        context:
          valid_customers: "$results.customer_list"

    stop_on_error: false                     # Optional: Stop on first error (default: false)

# Data quality checks
- id: quality_check
  action: check_quality
  parameters:
    data: "$results.dataset"                 # Required: Data to check
    checks:                                  # Required: Quality checks
      - type: "completeness"
        threshold: 0.95                      # 95% non-null required
        columns: ["id", "name", "email"]

      - type: "uniqueness"
        columns: ["id", "email"]

      - type: "consistency"
        rules:
          - "start_date <= end_date"
          - "total == sum(line_items)"

      - type: "accuracy"
        validations:
          email: "regex:^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
          phone: "regex:^\\+?1?\\d{9,15}$"

      - type: "timeliness"
        field: "last_updated"
        max_age_days: 30

# Report validation
- id: validate_report
  action: validate_report
  parameters:
    report: "$results.generated_report"      # Required: Report to validate
    checks:                                  # Required: Report checks
      - "completeness"                       # All sections present
      - "accuracy"                           # Facts are accurate
      - "consistency"                        # No contradictions
      - "readability"                        # Appropriate reading level
      - "citations"                          # Sources properly cited
    requirements:                            # Optional: Specific requirements
      min_words: 1000
      max_words: 5000
      required_sections: ["intro", "analysis", "conclusion"]
      citation_style: "APA"""")
    
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
async def test_tool_reference_lines_697_784_3():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 697-784."""
    import yaml
    import orchestrator
    
    yaml_content = ("""name: data-quality-pipeline
description: Comprehensive data validation and quality assurance

steps:
  # Load data
  - id: load
    action: read_file
    parameters:
      path: "{{ inputs.data_file }}"
      parse: true

  # Schema validation
  - id: validate_schema
    action: validate_schema
    parameters:
      data: "$results.load"
      schema:
        type: "array"
        items:
          type: "object"
          required: ["order_id", "customer_id", "amount", "date"]
          properties:
            order_id:
              type: "string"
              pattern: "^ORD-[0-9]{6}$"
            customer_id:
              type: "integer"
              minimum: 1
            amount:
              type: "number"
              minimum: 0
            date:
              type: "string"
              format: "date"

  # Business rules
  - id: validate_business
    action: validate_data
    parameters:
      data: "$results.load"
      rules:
        - name: "valid_amounts"
          field: "amount"
          condition: "value > 0 and value < 10000"
          severity: "error"

        - name: "recent_orders"
          field: "date"
          condition: "days_between(value, today()) <= 365"
          severity: "warning"
          message: "Order is older than 1 year"

  # Quality assessment
  - id: quality_report
    action: check_quality
    parameters:
      data: "$results.load"
      checks:
        - type: "completeness"
          threshold: 0.98
        - type: "uniqueness"
          columns: ["order_id"]
        - type: "consistency"
          rules:
            - "item_total == quantity * unit_price"
        - type: "accuracy"
          validations:
            email: "regex:^[\\w.-]+@[\\w.-]+\\.\\w+$"

  # Generate validation report
  - id: create_report
    action: generate_content
    parameters:
      template: |
        # Data Validation Report

        ## Schema Validation
        {{ results.validate_schema.summary }}

        ## Business Rules
        {{ results.validate_business.summary }}

        ## Quality Metrics
        {{ results.quality_report | format_quality_metrics }}

        ## Recommendations
        <AUTO>Based on the validation results, provide recommendations</AUTO>""")
    
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
async def test_tool_reference_lines_804_870_4():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 804-870."""
    import yaml
    import orchestrator
    
    yaml_content = ("""# Generate content
- id: generate
  action: generate_content
  parameters:
    prompt: "{{ inputs.prompt }}"            # Required: Generation prompt
    model: <AUTO>Select best model</AUTO>    # Optional: Model selection
    max_tokens: 1000                         # Optional: Maximum tokens
    temperature: 0.7                         # Optional: Creativity (0-2)
    system_prompt: "You are a helpful AI"    # Optional: System message
    format: "markdown"                       # Optional: Output format
    style: "professional"                    # Optional: Writing style

# Analyze text
- id: analyze
  action: analyze_text
  parameters:
    text: "$results.document"                # Required: Text to analyze
    analysis_types:                          # Required: Types of analysis
      - sentiment                            # Positive/negative/neutral
      - entities                             # Named entities
      - topics                               # Main topics
      - summary                              # Brief summary
      - key_points                           # Bullet points
      - language                             # Detect language
    output_format: "structured"              # Optional: structured|narrative

# Extract information
- id: extract
  action: extract_information
  parameters:
    content: "$results.raw_text"             # Required: Source content
    extract:                                 # Required: What to extract
      dates:
        description: "All mentioned dates"
        format: "YYYY-MM-DD"
      people:
        description: "Person names with roles"
        include_context: true
      organizations:
        description: "Company and organization names"
      numbers:
        description: "Numerical values with units"
        categories: ["financial", "metrics"]
    output_format: "json"                    # Optional: json|table|text

# Generate code
- id: code_gen
  action: generate_code
  parameters:
    description: "{{ inputs.feature_request }}" # Required: What to build
    language: "python"                       # Required: Programming language
    framework: "fastapi"                     # Optional: Framework/library
    include_tests: true                      # Optional: Generate tests
    include_docs: true                       # Optional: Generate docs
    style_guide: "PEP8"                     # Optional: Code style
    example_usage: true                      # Optional: Include examples

# Reasoning task
- id: reason
  action: reason_about
  parameters:
    question: "{{ inputs.problem }}"         # Required: Problem/question
    context: "$results.research"             # Optional: Additional context
    approach: "step_by_step"                 # Optional: Reasoning approach
    show_work: true                          # Optional: Show reasoning
    confidence_level: true                   # Optional: Include confidence""")
    
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
async def test_tool_reference_lines_889_898_5():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 889-898."""
    import yaml
    import orchestrator
    
    yaml_content = ("""# Query database
- id: fetch_data
  action: query_database
  parameters:
    connection: "postgresql://localhost/mydb" # Required: Connection string
    query: "SELECT * FROM users WHERE active = true" # Required: SQL query
    parameters: []                           # Optional: Query parameters
    fetch_size: 1000                         # Optional: Batch size
    timeout: 30                              # Optional: Query timeout""")
    
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
async def test_tool_reference_lines_913_927_6():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 913-927."""
    import yaml
    import orchestrator
    
    yaml_content = ("""# REST API call
- id: api_call
  action: call_api
  parameters:
    url: "https://api.example.com/data"     # Required: API endpoint
    method: "POST"                           # Required: HTTP method
    headers:                                 # Optional: Headers
      Authorization: "Bearer {{ env.API_TOKEN }}"
      Content-Type: "application/json"
    body:                                    # Optional: Request body
      query: "{{ inputs.search_term }}"
      limit: 100
    timeout: 60                              # Optional: Request timeout
    retry: 3                                 # Optional: Retry attempts""")
    
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
async def test_tool_reference_lines_936_1002_7():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 936-1002."""
    import yaml
    import orchestrator
    
    yaml_content = ("""name: comprehensive-research-tool-chain
description: Chain multiple tools for research and reporting

steps:
  # 1. Search multiple sources
  - id: web_search
    action: search_web
    parameters:
      query: "{{ inputs.topic }} latest research 2024"
      max_results: 20

  # 2. Scrape promising articles
  - id: scrape_articles
    for_each: "{{ results.web_search.results[:5] }}"
    as: article
    action: scrape_page
    parameters:
      url: "{{ article.url }}"
      selectors:
        content: "article, main, .content"

  # 3. Extract key information
  - id: extract_facts
    action: extract_information
    parameters:
      content: "$results.scrape_articles"
      extract:
        facts:
          description: "Key facts and findings"
        statistics:
          description: "Numerical data with context"
        quotes:
          description: "Notable quotes with attribution"

  # 4. Validate information
  - id: cross_validate
    action: validate_data
    parameters:
      data: "$results.extract_facts"
      rules:
        - name: "source_diversity"
          condition: "count(unique(sources)) >= 3"
          severity: "warning"

  # 5. Generate report
  - id: create_report
    action: generate_content
    parameters:
      prompt: |
        Create a comprehensive report about {{ inputs.topic }}
        using the following validated information:
        {{ results.extract_facts | json }}
      style: "academic"
      format: "markdown"
      max_tokens: 2000

  # 6. Save report
  - id: save_report
    action: write_file
    parameters:
      path: "reports/{{ inputs.topic }}_{{ execution.date }}.md"
      content: "$results.create_report"

  # 7. Generate PDF
  - id: create_pdf
    action: "!pandoc -f markdown -t pdf -o reports/{{ inputs.topic }}.pdf reports/{{ inputs.topic }}_{{ execution.date }}.md"""")
    
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
async def test_tool_reference_lines_1008_1099_8():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 1008-1099."""
    import yaml
    import orchestrator
    
    yaml_content = ("""name: etl-tool-chain
description: Extract, transform, and load data using tool chain

steps:
  # Extract from multiple sources
  - id: extract_database
    action: query_database
    parameters:
      connection: "{{ env.DB_CONNECTION }}"
      query: "SELECT * FROM sales WHERE date >= '2024-01-01'"

  - id: extract_api
    action: call_api
    parameters:
      url: "https://api.company.com/v2/transactions"
      method: "GET"
      headers:
        Authorization: "Bearer {{ env.API_KEY }}"
      params:
        start_date: "2024-01-01"
        page_size: 1000

  - id: extract_files
    action: list_directory
    parameters:
      path: "data/uploads/"
      pattern: "sales_*.csv"
      recursive: true

  # Load file data
  - id: load_files
    for_each: "{{ results.extract_files }}"
    as: file
    action: read_file
    parameters:
      path: "{{ file.path }}"
      parse: true

  # Transform all data
  - id: merge_all
    action: merge_data
    parameters:
      datasets:
        - "$results.extract_database"
        - "$results.extract_api.data"
        - "$results.load_files"
      key: "transaction_id"

  - id: clean_data
    action: transform_data
    parameters:
      data: "$results.merge_all"
      operations:
        - type: "remove_duplicates"
          columns: ["transaction_id"]
        - type: "fill_missing"
          strategy: "forward"
        - type: "standardize_formats"
          columns:
            date: "YYYY-MM-DD"
            amount: "decimal(10,2)"

  # Validate
  - id: validate_quality
    action: check_quality
    parameters:
      data: "$results.clean_data"
      checks:
        - type: "completeness"
          threshold: 0.99
        - type: "accuracy"
          validations:
            amount: "range:0,1000000"
            date: "date_range:2024-01-01,today"

  # Load to destination
  - id: save_processed
    action: write_file
    parameters:
      path: "processed/sales_cleaned_{{ execution.date }}.parquet"
      content: "$results.clean_data"
      format: "parquet"

  - id: update_database
    condition: "{{ results.validate_quality.passed }}"
    action: insert_data
    parameters:
      connection: "{{ env.DW_CONNECTION }}"
      table: "sales_fact"
      data: "$results.clean_data"
      mode: "append"""")
    
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
async def test_tool_reference_lines_1110_1152_9():
    """Test orchestrator code from docs_sphinx/tool_reference.rst lines 1110-1152."""
    # To create your own tools:
    
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
        code = ('''from orchestrator.tools.base import Tool

class MyCustomTool(Tool):
    def __init__(self):
        super().__init__(
            name="my-custom-tool",
            description="Does something special"
        )

        # Define parameters
        self.add_parameter(
            name="input_data",
            type="string",
            description="Data to process",
            required=True
        )

        self.add_parameter(
            name="mode",
            type="string",
            description="Processing mode",
            required=False,
            default="standard",
            enum=["standard", "advanced", "expert"]
        )

    async def execute(self, **kwargs):
        """Execute the tool action."""
        input_data = kwargs["input_data"]
        mode = kwargs.get("mode", "standard")

        # Your tool logic here
        result = process_data(input_data, mode)

        return {
            "status": "success",
            "result": result,
            "metadata": {
                "mode": mode,
                "timestamp": datetime.now()
            }
        }''')
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
async def test_tool_reference_lines_1160_1175_10():
    """Test orchestrator code from docs_sphinx/tool_reference.rst lines 1160-1175."""
    # Register your tool to make it available:
    
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
        code = ('''from orchestrator.tools.base import default_registry

# Register tool
tool = MyCustomTool()
default_registry.register(tool)

# Use in pipeline
pipeline_yaml = """
steps:
  - id: custom_step
    action: my-custom-tool
    parameters:
      input_data: "{{ inputs.data }}"
      mode: "advanced"
"""''')
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
async def test_tutorial_data_processing_lines_35_239_11():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_data_processing.rst lines 35-239."""
    import yaml
    import orchestrator
    
    yaml_content = ("""name: sales-etl-pipeline
description: Extract, transform, and load sales data

inputs:
  data_source:
    type: string
    description: "Path to source data file"
    required: true

  output_format:
    type: string
    description: "Output format"
    default: "parquet"
    validation:
      enum: ["csv", "json", "parquet", "excel"]

  date_range:
    type: object
    description: "Date range for filtering"
    default:
      start: "2024-01-01"
      end: "2024-12-31"

outputs:
  processed_data:
    type: string
    value: "processed/sales_{{ execution.date }}.{{ inputs.output_format }}"

  quality_report:
    type: string
    value: "reports/quality_{{ execution.date }}.json"

  summary_stats:
    type: string
    value: "reports/summary_{{ execution.date }}.md"

steps:
  # Extract: Load raw data
  - id: extract_data
    action: read_file
    parameters:
      path: "{{ inputs.data_source }}"
      parse: true
    error_handling:
      retry:
        max_attempts: 3
      fallback:
        action: generate_content
        parameters:
          prompt: "Generate sample sales data for testing"

  # Transform: Clean and process data
  - id: clean_data
    action: transform_data
    parameters:
      data: "$results.extract_data"
      operations:
        # Standardize column names
        - type: "rename_columns"
          mapping:
            "Sale Date": "sale_date"
            "Customer Name": "customer_name"
            "Product ID": "product_id"
            "Sale Amount": "amount"
            "Quantity": "quantity"
            "Sales Rep": "sales_rep"

        # Convert data types
        - type: "convert_types"
          conversions:
            sale_date: "datetime"
            amount: "float"
            quantity: "integer"
            product_id: "string"

        # Remove duplicates
        - type: "remove_duplicates"
          columns: ["product_id", "sale_date", "customer_name"]

        # Handle missing values
        - type: "fill_missing"
          strategy: "forward"
          columns: ["sales_rep"]

        # Add calculated fields
        - type: "add_column"
          name: "total_value"
          expression: "amount * quantity"

        - type: "add_column"
          name: "quarter"
          expression: "quarter(sale_date)"

        - type: "add_column"
          name: "year"
          expression: "year(sale_date)"

  # Filter data by date range
  - id: filter_data
    action: filter_data
    parameters:
      data: "$results.clean_data"
      conditions:
        - field: "sale_date"
          operator: "gte"
          value: "{{ inputs.date_range.start }}"
        - field: "sale_date"
          operator: "lte"
          value: "{{ inputs.date_range.end }}"
        - field: "amount"
          operator: "gt"
          value: 0

  # Data quality validation
  - id: validate_quality
    action: check_quality
    parameters:
      data: "$results.filter_data"
      checks:
        - type: "completeness"
          threshold: 0.95
          columns: ["product_id", "amount", "sale_date"]

        - type: "uniqueness"
          columns: ["product_id", "sale_date", "customer_name"]

        - type: "consistency"
          rules:
            - "total_value == amount * quantity"
            - "amount > 0"
            - "quantity > 0"

        - type: "accuracy"
          validations:
            product_id: "regex:^PROD-[0-9]{6}$"
            amount: "range:1,50000"
            quantity: "range:1,1000"

  # Generate summary statistics
  - id: calculate_summary
    action: aggregate_data
    parameters:
      data: "$results.filter_data"
      group_by: ["year", "quarter"]
      aggregations:
        total_sales:
          column: "total_value"
          function: "sum"
        avg_sale:
          column: "amount"
          function: "mean"
        num_transactions:
          column: "*"
          function: "count"
        unique_customers:
          column: "customer_name"
          function: "nunique"
        top_product:
          column: "product_id"
          function: "mode"

  # Load: Save processed data
  - id: save_processed_data
    action: convert_format
    parameters:
      data: "$results.filter_data"
      to_format: "{{ inputs.output_format }}"
      output_path: "{{ outputs.processed_data }}"
      options:
        compression: "snappy"
        index: false

  # Save quality report
  - id: save_quality_report
    action: write_file
    parameters:
      path: "{{ outputs.quality_report }}"
      content: "{{ results.validate_quality | json }}"

  # Generate readable summary
  - id: create_summary_report
    action: generate_content
    parameters:
      prompt: |
        Create a summary report for sales data processing:

        Quality Results: {{ results.validate_quality | json }}
        Summary Statistics: {{ results.calculate_summary | json }}

        Include:
        - Data quality assessment
        - Key metrics and trends
        - Any issues or recommendations
        - Processing summary

      style: "professional"
      format: "markdown"

  # Save summary report
  - id: save_summary
    action: write_file
    parameters:
      path: "{{ outputs.summary_stats }}"
      content: "$results.create_summary_report"""")
    
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
async def test_tutorial_data_processing_lines_245_264_12():
    """Test orchestrator code from docs_sphinx/tutorials/tutorial_data_processing.rst lines 245-264."""
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

# Compile pipeline
etl_pipeline = orc.compile("sales_etl.yaml")

# Process sales data
result = etl_pipeline.run(
    data_source="data/raw/sales_2024.csv",
    output_format="parquet",
    date_range={
        "start": "2024-01-01",
        "end": "2024-06-30"
    }
)

print(f"ETL completed: {result}")""")
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
async def test_tutorial_data_processing_lines_277_521_13():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_data_processing.rst lines 277-521."""
    import yaml
    import orchestrator
    
    yaml_content = ("""name: multi-source-integration
description: Integrate data from multiple sources with validation

inputs:
  sources:
    type: object
    description: "Data source configurations"
    required: true
    # Example:
    # database:
    #   type: "postgresql"
    #   connection: "postgresql://..."
    #   query: "SELECT * FROM sales"
    # api:
    #   type: "rest"
    #   url: "https://api.company.com/data"
    #   headers: {...}
    # files:
    #   type: "file"
    #   paths: ["data1.csv", "data2.json"]

  merge_strategy:
    type: string
    description: "How to merge data sources"
    default: "outer"
    validation:
      enum: ["inner", "outer", "left", "right"]

  deduplication_fields:
    type: array
    description: "Fields to use for deduplication"
    default: ["id", "timestamp"]

outputs:
  integrated_data:
    type: string
    value: "integrated/master_data_{{ execution.timestamp }}.parquet"

  integration_report:
    type: string
    value: "reports/integration_{{ execution.timestamp }}.md"

steps:
  # Extract from database sources
  - id: extract_database
    condition: "'database' in inputs.sources"
    action: query_database
    parameters:
      connection: "{{ inputs.sources.database.connection }}"
      query: "{{ inputs.sources.database.query }}"
      fetch_size: 10000
    error_handling:
      continue_on_error: true

  # Extract from API sources
  - id: extract_api
    condition: "'api' in inputs.sources"
    action: call_api
    parameters:
      url: "{{ inputs.sources.api.url }}"
      method: "GET"
      headers: "{{ inputs.sources.api.headers | default({}) }}"
      params: "{{ inputs.sources.api.params | default({}) }}"
      timeout: 300
    error_handling:
      retry:
        max_attempts: 3
        backoff: "exponential"

  # Extract from file sources
  - id: extract_files
    condition: "'files' in inputs.sources"
    for_each: "{{ inputs.sources.files.paths }}"
    as: file_path
    action: read_file
    parameters:
      path: "{{ file_path }}"
      parse: true

  # Standardize data schemas
  - id: standardize_database
    condition: "results.extract_database is defined"
    action: transform_data
    parameters:
      data: "$results.extract_database"
      operations:
        - type: "add_column"
          name: "source"
          value: "database"
        - type: "standardize_schema"
          target_schema:
            id: "string"
            timestamp: "datetime"
            value: "float"
            category: "string"

  - id: standardize_api
    condition: "results.extract_api is defined"
    action: transform_data
    parameters:
      data: "$results.extract_api.data"
      operations:
        - type: "add_column"
          name: "source"
          value: "api"
        - type: "flatten_nested"
          columns: ["metadata", "attributes"]
        - type: "standardize_schema"
          target_schema:
            id: "string"
            timestamp: "datetime"
            value: "float"
            category: "string"

  - id: standardize_files
    condition: "results.extract_files is defined"
    action: transform_data
    parameters:
      data: "$results.extract_files"
      operations:
        - type: "add_column"
          name: "source"
          value: "files"
        - type: "combine_files"
          strategy: "union"
        - type: "standardize_schema"
          target_schema:
            id: "string"
            timestamp: "datetime"
            value: "float"
            category: "string"

  # Merge all data sources
  - id: merge_sources
    action: merge_data
    parameters:
      datasets:
        - "$results.standardize_database"
        - "$results.standardize_api"
        - "$results.standardize_files"
      how: "{{ inputs.merge_strategy }}"
      on: ["id"]
      suffixes: ["_db", "_api", "_file"]

  # Remove duplicates
  - id: deduplicate
    action: transform_data
    parameters:
      data: "$results.merge_sources"
      operations:
        - type: "remove_duplicates"
          columns: "{{ inputs.deduplication_fields }}"
          keep: "last"  # Keep most recent

  # Data quality assessment
  - id: assess_integration_quality
    action: check_quality
    parameters:
      data: "$results.deduplicate"
      checks:
        - type: "completeness"
          threshold: 0.90
          critical_columns: ["id", "timestamp"]

        - type: "consistency"
          rules:
            - "value_db == value_api OR value_db IS NULL OR value_api IS NULL"
            - "timestamp >= '2020-01-01'"

        - type: "accuracy"
          validations:
            id: "not_null"
            timestamp: "datetime_format"
            value: "numeric_range:-1000000,1000000"

  # Resolve conflicts between sources
  - id: resolve_conflicts
    action: transform_data
    parameters:
      data: "$results.deduplicate"
      operations:
        - type: "resolve_conflicts"
          strategy: "priority"
          priority_order: ["database", "api", "files"]
          conflict_columns: ["value", "category"]

        - type: "add_column"
          name: "confidence_score"
          expression: "calculate_confidence(source_count, data_age, validation_status)"

  # Create final integrated dataset
  - id: finalize_integration
    action: transform_data
    parameters:
      data: "$results.resolve_conflicts"
      operations:
        - type: "select_columns"
          columns: ["id", "timestamp", "value", "category", "source", "confidence_score"]

        - type: "sort"
          columns: ["timestamp"]
          ascending: [false]

  # Save integrated data
  - id: save_integrated
    action: convert_format
    parameters:
      data: "$results.finalize_integration"
      to_format: "parquet"
      output_path: "{{ outputs.integrated_data }}"
      options:
        compression: "snappy"
        partition_cols: ["category"]

  # Generate integration report
  - id: create_integration_report
    action: generate_content
    parameters:
      prompt: |
        Create an integration report for multi-source data merge:

        Sources processed:
        {% for source in inputs.sources.keys() %}
        - {{ source }}
        {% endfor %}

        Quality assessment: {{ results.assess_integration_quality | json }}
        Final record count: {{ results.finalize_integration | length }}

        Include:
        - Source summary and statistics
        - Data quality metrics
        - Conflict resolution summary
        - Recommendations for data improvement

      style: "technical"
      format: "markdown"

  # Save integration report
  - id: save_report
    action: write_file
    parameters:
      path: "{{ outputs.integration_report }}"
      content: "$results.create_integration_report"""")
    
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
async def test_tutorial_data_processing_lines_527_558_14():
    """Test orchestrator code from docs_sphinx/tutorials/tutorial_data_processing.rst lines 527-558."""
    # -----------------------------------
    
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

# Compile integration pipeline
integration = orc.compile("data_integration.yaml")

# Integrate data from multiple sources
result = integration.run(
    sources={
        "database": {
            "type": "postgresql",
            "connection": "postgresql://user:pass@localhost/mydb",
            "query": "SELECT * FROM transactions WHERE date >= '2024-01-01'"
        },
        "api": {
            "type": "rest",
            "url": "https://api.external.com/v1/data",
            "headers": {"Authorization": "Bearer token123"}
        },
        "files": {
            "type": "file",
            "paths": ["data/file1.csv", "data/file2.json"]
        }
    },
    merge_strategy="outer",
    deduplication_fields=["transaction_id", "timestamp"]
)

print(f"Integration completed: {result}")""")
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
