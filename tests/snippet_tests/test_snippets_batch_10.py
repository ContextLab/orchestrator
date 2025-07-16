"""Tests for documentation code snippets - Batch 10."""
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


def test_tool_reference_lines_436_507_0():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 436-507."""
    import yaml
    
    yaml_content = """# Transform data
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
      index: false"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.asyncio
async def test_tool_reference_lines_512_582_1():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 512-582."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """name: sales-data-analysis
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
        index: false"""
    
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

def test_tool_reference_lines_598_692_2():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 598-692."""
    import yaml
    
    yaml_content = """# Validate against schema
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
      citation_style: "APA""""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.asyncio
async def test_tool_reference_lines_697_784_3():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 697-784."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """name: data-quality-pipeline
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
        <AUTO>Based on the validation results, provide recommendations</AUTO>"""
    
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

def test_tool_reference_lines_804_870_4():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 804-870."""
    import yaml
    
    yaml_content = """# Generate content
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
    confidence_level: true                   # Optional: Include confidence"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_tool_reference_lines_889_898_5():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 889-898."""
    import yaml
    
    yaml_content = """# Query database
- id: fetch_data
  action: query_database
  parameters:
    connection: "postgresql://localhost/mydb" # Required: Connection string
    query: "SELECT * FROM users WHERE active = true" # Required: SQL query
    parameters: []                           # Optional: Query parameters
    fetch_size: 1000                         # Optional: Batch size
    timeout: 30                              # Optional: Query timeout"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_tool_reference_lines_913_927_6():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 913-927."""
    import yaml
    
    yaml_content = """# REST API call
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
    retry: 3                                 # Optional: Retry attempts"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.asyncio
async def test_tool_reference_lines_936_1002_7():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 936-1002."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """name: comprehensive-research-tool-chain
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
    action: "!pandoc -f markdown -t pdf -o reports/{{ inputs.topic }}.pdf reports/{{ inputs.topic }}_{{ execution.date }}.md""""
    
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
async def test_tool_reference_lines_1008_1099_8():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 1008-1099."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """name: etl-tool-chain
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
      mode: "append""""
    
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

def test_tool_reference_lines_1110_1152_9():
    """Test Python import from docs_sphinx/tool_reference.rst lines 1110-1152."""
    # Test imports
    try:
        exec("""from orchestrator.tools.base import Tool

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
        }""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_tool_reference_lines_1160_1175_10():
    """Test Python import from docs_sphinx/tool_reference.rst lines 1160-1175."""
    # Test imports
    try:
        exec("""from orchestrator.tools.base import default_registry

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
"""""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

@pytest.mark.asyncio
async def test_tutorial_data_processing_lines_35_239_11():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_data_processing.rst lines 35-239."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """name: sales-etl-pipeline
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
      content: "$results.create_summary_report""""
    
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

def test_tutorial_data_processing_lines_245_264_12():
    """Test Python import from docs_sphinx/tutorials/tutorial_data_processing.rst lines 245-264."""
    # Test imports
    try:
        exec("""import orchestrator as orc

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
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

@pytest.mark.asyncio
async def test_tutorial_data_processing_lines_277_521_13():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_data_processing.rst lines 277-521."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """name: multi-source-integration
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
      content: "$results.create_integration_report""""
    
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

def test_tutorial_data_processing_lines_527_558_14():
    """Test Python import from docs_sphinx/tutorials/tutorial_data_processing.rst lines 527-558."""
    # Test imports
    try:
        exec("""import orchestrator as orc

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
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

@pytest.mark.asyncio
async def test_tutorial_data_processing_lines_571_815_15():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_data_processing.rst lines 571-815."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """name: data-quality-assessment
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
      content: "{{ results.compile_issues | json }}""""
    
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
async def test_tutorial_data_processing_lines_828_923_16():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_data_processing.rst lines 828-923."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """name: realtime-data-processing
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
      content: "$results.generate_alerts""""
    
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
async def test_tutorial_data_processing_lines_932_990_17():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_data_processing.rst lines 932-990."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """name: customer-data-platform
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
          expression: "classify_customer_segment(clv, engagement, recency)""""
    
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
async def test_tutorial_data_processing_lines_996_1062_18():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_data_processing.rst lines 996-1062."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """name: financial-data-pipeline
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

        Include regulatory compliance status and any required actions."""
    
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

def test_tutorial_data_processing_lines_1073_1078_19():
    """Test YAML snippet from docs_sphinx/tutorials/tutorial_data_processing.rst lines 1073-1078."""
    import yaml
    
    yaml_content = """# Your challenge:
# - Extract: Orders, customers, products, reviews
# - Transform: Calculate metrics, segment customers
# - Load: Create analytics-ready datasets
# - Quality: Validate business rules"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_tutorial_data_processing_lines_1086_1091_20():
    """Test YAML snippet from docs_sphinx/tutorials/tutorial_data_processing.rst lines 1086-1091."""
    import yaml
    
    yaml_content = """# Requirements:
# - Handle high-volume time series data
# - Detect sensor anomalies
# - Aggregate by time windows
# - Generate maintenance alerts"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_tutorial_data_processing_lines_1099_1104_21():
    """Test YAML snippet from docs_sphinx/tutorials/tutorial_data_processing.rst lines 1099-1104."""
    import yaml
    
    yaml_content = """# Features:
# - Extract from multiple platforms
# - Text analysis and sentiment
# - Trend detection
# - Influence measurement"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.asyncio
async def test_tutorial_web_research_lines_36_91_22():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_web_research.rst lines 36-91."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """name: basic-web-search
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
      content: "$results.compile_report""""
    
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

def test_tutorial_web_research_lines_97_117_23():
    """Test Python import from docs_sphinx/tutorials/tutorial_web_research.rst lines 97-117."""
    # Test imports
    try:
        exec("""import orchestrator as orc

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
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_tutorial_web_research_lines_125_142_24():
    """Test markdown snippet from docs_sphinx/tutorials/tutorial_web_research.rst lines 125-142."""
    # Snippet type 'markdown' not yet supported for testing
    pytest.skip("Snippet type 'markdown' not yet supported")

@pytest.mark.asyncio
async def test_tutorial_web_research_lines_155_346_25():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_web_research.rst lines 155-346."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """name: multi-source-research
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
      content: "$results.create_report""""
    
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

def test_tutorial_web_research_lines_352_375_26():
    """Test Python import from docs_sphinx/tutorials/tutorial_web_research.rst lines 352-375."""
    # Test imports
    try:
        exec("""import orchestrator as orc

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
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

@pytest.mark.asyncio
async def test_tutorial_web_research_lines_388_488_27():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_web_research.rst lines 388-488."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """name: fact-checker
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
      content: "$results.create_fact_check_report""""
    
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

def test_tutorial_web_research_lines_494_514_28():
    """Test Python import from docs_sphinx/tutorials/tutorial_web_research.rst lines 494-514."""
    # Test imports
    try:
        exec("""import orchestrator as orc

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
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

@pytest.mark.asyncio
async def test_tutorial_web_research_lines_527_805_29():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_web_research.rst lines 527-805."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """name: automated-report-generator
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
      continue_on_error: true"""
    
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
