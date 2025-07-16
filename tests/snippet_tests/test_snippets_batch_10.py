"""Tests for documentation code snippets - Batch 10."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_tool_reference_lines_436_507_0():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 436-507."""
    import yaml
    
    yaml_content = '# Transform data\n- id: transform\n  action: transform_data\n  parameters:\n    data: "$results.load_data"               # Required: Input data or path\n    operations:                              # Required: List of operations\n      - type: "rename_columns"\n        mapping:\n          old_name: "new_name"\n          price: "cost"\n      - type: "add_column"\n        name: "total"\n        expression: "quantity * cost"\n      - type: "drop_columns"\n        columns: ["unnecessary_field"]\n      - type: "convert_types"\n        conversions:\n          date: "datetime"\n          amount: "float"\n\n# Filter data\n- id: filter\n  action: filter_data\n  parameters:\n    data: "$results.transform"               # Required: Input data\n    conditions:                              # Required: Filter conditions\n      - field: "status"\n        operator: "equals"                   # equals|not_equals|contains|gt|lt|gte|lte\n        value: "active"\n      - field: "amount"\n        operator: "gt"\n        value: 1000\n    mode: "and"                              # Optional: and|or (default: and)\n\n# Aggregate data\n- id: aggregate\n  action: aggregate_data\n  parameters:\n    data: "$results.filter"                  # Required: Input data\n    group_by: ["category", "region"]        # Optional: Grouping columns\n    aggregations:                            # Required: Aggregation rules\n      total_amount:\n        column: "amount"\n        function: "sum"                      # sum|mean|median|min|max|count|std\n      average_price:\n        column: "price"\n        function: "mean"\n      item_count:\n        column: "*"\n        function: "count"\n\n# Merge data\n- id: merge\n  action: merge_data\n  parameters:\n    left: "$results.main_data"               # Required: Left dataset\n    right: "$results.lookup_data"            # Required: Right dataset\n    on: "customer_id"                        # Required: Join column(s)\n    how: "left"                              # Optional: left|right|inner|outer (default: left)\n    suffixes: ["_main", "_lookup"]          # Optional: Column suffixes\n\n# Convert format\n- id: convert\n  action: convert_format\n  parameters:\n    data: "$results.final_data"              # Required: Input data\n    from_format: "json"                      # Optional: Auto-detect if not specified\n    to_format: "parquet"                     # Required: Target format\n    options:                                 # Optional: Format-specific options\n      compression: "snappy"\n      index: false'
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.asyncio
async def test_tool_reference_lines_512_582_1():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 512-582."""
    import yaml
    
    yaml_content = 'name: sales-data-analysis\ndescription: Process and analyze sales data\n\nsteps:\n  # Load raw data\n  - id: load_sales\n    action: read_file\n    parameters:\n      path: "data/sales_2024.csv"\n      parse: true\n\n  # Clean and transform\n  - id: clean_data\n    action: transform_data\n    parameters:\n      data: "$results.load_sales"\n      operations:\n        - type: "rename_columns"\n          mapping:\n            "Sale Date": "sale_date"\n            "Customer Name": "customer_name"\n            "Product ID": "product_id"\n            "Sale Amount": "amount"\n        - type: "convert_types"\n          conversions:\n            sale_date: "datetime"\n            amount: "float"\n        - type: "add_column"\n          name: "quarter"\n          expression: "sale_date.quarter"\n\n  # Filter valid sales\n  - id: filter_valid\n    action: filter_data\n    parameters:\n      data: "$results.clean_data"\n      conditions:\n        - field: "amount"\n          operator: "gt"\n          value: 0\n        - field: "product_id"\n          operator: "not_equals"\n          value: null\n\n  # Aggregate by quarter\n  - id: quarterly_summary\n    action: aggregate_data\n    parameters:\n      data: "$results.filter_valid"\n      group_by: ["quarter", "product_id"]\n      aggregations:\n        total_sales:\n          column: "amount"\n          function: "sum"\n        avg_sale:\n          column: "amount"\n          function: "mean"\n        num_transactions:\n          column: "*"\n          function: "count"\n\n  # Save results\n  - id: save_summary\n    action: convert_format\n    parameters:\n      data: "$results.quarterly_summary"\n      to_format: "excel"\n      options:\n        sheet_name: "Quarterly Sales"\n        index: false'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

def test_tool_reference_lines_598_692_2():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 598-692."""
    import yaml
    
    yaml_content = '# Validate against schema\n- id: validate_structure\n  action: validate_schema\n  parameters:\n    data: "$results.processed_data"          # Required: Data to validate\n    schema:                                  # Required: Validation schema\n      type: "object"\n      required: ["id", "name", "email"]\n      properties:\n        id:\n          type: "integer"\n          minimum: 1\n        name:\n          type: "string"\n          minLength: 2\n          maxLength: 100\n        email:\n          type: "string"\n          format: "email"\n        age:\n          type: "integer"\n          minimum: 0\n          maximum: 150\n    strict: false                            # Optional: Strict mode (default: false)\n\n# Business rule validation\n- id: validate_rules\n  action: validate_data\n  parameters:\n    data: "$results.transactions"            # Required: Data to validate\n    rules:                                   # Required: Validation rules\n      - name: "positive_amounts"\n        field: "amount"\n        condition: "value > 0"\n        severity: "error"                    # error|warning|info\n        message: "Transaction amounts must be positive"\n\n      - name: "valid_date_range"\n        field: "transaction_date"\n        condition: "value >= \'2024-01-01\' and value <= today()"\n        severity: "error"\n\n      - name: "customer_exists"\n        field: "customer_id"\n        condition: "value in valid_customers"\n        severity: "warning"\n        context:\n          valid_customers: "$results.customer_list"\n\n    stop_on_error: false                     # Optional: Stop on first error (default: false)\n\n# Data quality checks\n- id: quality_check\n  action: check_quality\n  parameters:\n    data: "$results.dataset"                 # Required: Data to check\n    checks:                                  # Required: Quality checks\n      - type: "completeness"\n        threshold: 0.95                      # 95% non-null required\n        columns: ["id", "name", "email"]\n\n      - type: "uniqueness"\n        columns: ["id", "email"]\n\n      - type: "consistency"\n        rules:\n          - "start_date <= end_date"\n          - "total == sum(line_items)"\n\n      - type: "accuracy"\n        validations:\n          email: "regex:^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\\\\\\\.[a-zA-Z]{2,}$"\n          phone: "regex:^\\\\\\\\+?1?\\\\\\\\d{9,15}$"\n\n      - type: "timeliness"\n        field: "last_updated"\n        max_age_days: 30\n\n# Report validation\n- id: validate_report\n  action: validate_report\n  parameters:\n    report: "$results.generated_report"      # Required: Report to validate\n    checks:                                  # Required: Report checks\n      - "completeness"                       # All sections present\n      - "accuracy"                           # Facts are accurate\n      - "consistency"                        # No contradictions\n      - "readability"                        # Appropriate reading level\n      - "citations"                          # Sources properly cited\n    requirements:                            # Optional: Specific requirements\n      min_words: 1000\n      max_words: 5000\n      required_sections: ["intro", "analysis", "conclusion"]\n      citation_style: "APA"'
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.asyncio
async def test_tool_reference_lines_697_784_3():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 697-784."""
    import yaml
    
    yaml_content = 'name: data-quality-pipeline\ndescription: Comprehensive data validation and quality assurance\n\nsteps:\n  # Load data\n  - id: load\n    action: read_file\n    parameters:\n      path: "{{ inputs.data_file }}"\n      parse: true\n\n  # Schema validation\n  - id: validate_schema\n    action: validate_schema\n    parameters:\n      data: "$results.load"\n      schema:\n        type: "array"\n        items:\n          type: "object"\n          required: ["order_id", "customer_id", "amount", "date"]\n          properties:\n            order_id:\n              type: "string"\n              pattern: "^ORD-[0-9]{6}$"\n            customer_id:\n              type: "integer"\n              minimum: 1\n            amount:\n              type: "number"\n              minimum: 0\n            date:\n              type: "string"\n              format: "date"\n\n  # Business rules\n  - id: validate_business\n    action: validate_data\n    parameters:\n      data: "$results.load"\n      rules:\n        - name: "valid_amounts"\n          field: "amount"\n          condition: "value > 0 and value < 10000"\n          severity: "error"\n\n        - name: "recent_orders"\n          field: "date"\n          condition: "days_between(value, today()) <= 365"\n          severity: "warning"\n          message: "Order is older than 1 year"\n\n  # Quality assessment\n  - id: quality_report\n    action: check_quality\n    parameters:\n      data: "$results.load"\n      checks:\n        - type: "completeness"\n          threshold: 0.98\n        - type: "uniqueness"\n          columns: ["order_id"]\n        - type: "consistency"\n          rules:\n            - "item_total == quantity * unit_price"\n        - type: "accuracy"\n          validations:\n            email: "regex:^[\\\\\\\\w.-]+@[\\\\\\\\w.-]+\\\\\\\\.\\\\\\\\w+$"\n\n  # Generate validation report\n  - id: create_report\n    action: generate_content\n    parameters:\n      template: |\n        # Data Validation Report\n\n        ## Schema Validation\n        {{ results.validate_schema.summary }}\n\n        ## Business Rules\n        {{ results.validate_business.summary }}\n\n        ## Quality Metrics\n        {{ results.quality_report | format_quality_metrics }}\n\n        ## Recommendations\n        <AUTO>Based on the validation results, provide recommendations</AUTO>'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

def test_tool_reference_lines_804_870_4():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 804-870."""
    import yaml
    
    yaml_content = '# Generate content\n- id: generate\n  action: generate_content\n  parameters:\n    prompt: "{{ inputs.prompt }}"            # Required: Generation prompt\n    model: <AUTO>Select best model</AUTO>    # Optional: Model selection\n    max_tokens: 1000                         # Optional: Maximum tokens\n    temperature: 0.7                         # Optional: Creativity (0-2)\n    system_prompt: "You are a helpful AI"    # Optional: System message\n    format: "markdown"                       # Optional: Output format\n    style: "professional"                    # Optional: Writing style\n\n# Analyze text\n- id: analyze\n  action: analyze_text\n  parameters:\n    text: "$results.document"                # Required: Text to analyze\n    analysis_types:                          # Required: Types of analysis\n      - sentiment                            # Positive/negative/neutral\n      - entities                             # Named entities\n      - topics                               # Main topics\n      - summary                              # Brief summary\n      - key_points                           # Bullet points\n      - language                             # Detect language\n    output_format: "structured"              # Optional: structured|narrative\n\n# Extract information\n- id: extract\n  action: extract_information\n  parameters:\n    content: "$results.raw_text"             # Required: Source content\n    extract:                                 # Required: What to extract\n      dates:\n        description: "All mentioned dates"\n        format: "YYYY-MM-DD"\n      people:\n        description: "Person names with roles"\n        include_context: true\n      organizations:\n        description: "Company and organization names"\n      numbers:\n        description: "Numerical values with units"\n        categories: ["financial", "metrics"]\n    output_format: "json"                    # Optional: json|table|text\n\n# Generate code\n- id: code_gen\n  action: generate_code\n  parameters:\n    description: "{{ inputs.feature_request }}" # Required: What to build\n    language: "python"                       # Required: Programming language\n    framework: "fastapi"                     # Optional: Framework/library\n    include_tests: true                      # Optional: Generate tests\n    include_docs: true                       # Optional: Generate docs\n    style_guide: "PEP8"                     # Optional: Code style\n    example_usage: true                      # Optional: Include examples\n\n# Reasoning task\n- id: reason\n  action: reason_about\n  parameters:\n    question: "{{ inputs.problem }}"         # Required: Problem/question\n    context: "$results.research"             # Optional: Additional context\n    approach: "step_by_step"                 # Optional: Reasoning approach\n    show_work: true                          # Optional: Show reasoning\n    confidence_level: true                   # Optional: Include confidence'
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_tool_reference_lines_889_898_5():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 889-898."""
    import yaml
    
    yaml_content = '# Query database\n- id: fetch_data\n  action: query_database\n  parameters:\n    connection: "postgresql://localhost/mydb" # Required: Connection string\n    query: "SELECT * FROM users WHERE active = true" # Required: SQL query\n    parameters: []                           # Optional: Query parameters\n    fetch_size: 1000                         # Optional: Batch size\n    timeout: 30                              # Optional: Query timeout'
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_tool_reference_lines_913_927_6():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 913-927."""
    import yaml
    
    yaml_content = '# REST API call\n- id: api_call\n  action: call_api\n  parameters:\n    url: "https://api.example.com/data"     # Required: API endpoint\n    method: "POST"                           # Required: HTTP method\n    headers:                                 # Optional: Headers\n      Authorization: "Bearer {{ env.API_TOKEN }}"\n      Content-Type: "application/json"\n    body:                                    # Optional: Request body\n      query: "{{ inputs.search_term }}"\n      limit: 100\n    timeout: 60                              # Optional: Request timeout\n    retry: 3                                 # Optional: Retry attempts'
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.asyncio
async def test_tool_reference_lines_936_1002_7():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 936-1002."""
    import yaml
    
    yaml_content = 'name: comprehensive-research-tool-chain\ndescription: Chain multiple tools for research and reporting\n\nsteps:\n  # 1. Search multiple sources\n  - id: web_search\n    action: search_web\n    parameters:\n      query: "{{ inputs.topic }} latest research 2024"\n      max_results: 20\n\n  # 2. Scrape promising articles\n  - id: scrape_articles\n    for_each: "{{ results.web_search.results[:5] }}"\n    as: article\n    action: scrape_page\n    parameters:\n      url: "{{ article.url }}"\n      selectors:\n        content: "article, main, .content"\n\n  # 3. Extract key information\n  - id: extract_facts\n    action: extract_information\n    parameters:\n      content: "$results.scrape_articles"\n      extract:\n        facts:\n          description: "Key facts and findings"\n        statistics:\n          description: "Numerical data with context"\n        quotes:\n          description: "Notable quotes with attribution"\n\n  # 4. Validate information\n  - id: cross_validate\n    action: validate_data\n    parameters:\n      data: "$results.extract_facts"\n      rules:\n        - name: "source_diversity"\n          condition: "count(unique(sources)) >= 3"\n          severity: "warning"\n\n  # 5. Generate report\n  - id: create_report\n    action: generate_content\n    parameters:\n      prompt: |\n        Create a comprehensive report about {{ inputs.topic }}\n        using the following validated information:\n        {{ results.extract_facts | json }}\n      style: "academic"\n      format: "markdown"\n      max_tokens: 2000\n\n  # 6. Save report\n  - id: save_report\n    action: write_file\n    parameters:\n      path: "reports/{{ inputs.topic }}_{{ execution.date }}.md"\n      content: "$results.create_report"\n\n  # 7. Generate PDF\n  - id: create_pdf\n    action: "!pandoc -f markdown -t pdf -o reports/{{ inputs.topic }}.pdf reports/{{ inputs.topic }}_{{ execution.date }}.md"'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

@pytest.mark.asyncio
async def test_tool_reference_lines_1008_1099_8():
    """Test YAML pipeline from docs_sphinx/tool_reference.rst lines 1008-1099."""
    import yaml
    
    yaml_content = 'name: etl-tool-chain\ndescription: Extract, transform, and load data using tool chain\n\nsteps:\n  # Extract from multiple sources\n  - id: extract_database\n    action: query_database\n    parameters:\n      connection: "{{ env.DB_CONNECTION }}"\n      query: "SELECT * FROM sales WHERE date >= \'2024-01-01\'"\n\n  - id: extract_api\n    action: call_api\n    parameters:\n      url: "https://api.company.com/v2/transactions"\n      method: "GET"\n      headers:\n        Authorization: "Bearer {{ env.API_KEY }}"\n      params:\n        start_date: "2024-01-01"\n        page_size: 1000\n\n  - id: extract_files\n    action: list_directory\n    parameters:\n      path: "data/uploads/"\n      pattern: "sales_*.csv"\n      recursive: true\n\n  # Load file data\n  - id: load_files\n    for_each: "{{ results.extract_files }}"\n    as: file\n    action: read_file\n    parameters:\n      path: "{{ file.path }}"\n      parse: true\n\n  # Transform all data\n  - id: merge_all\n    action: merge_data\n    parameters:\n      datasets:\n        - "$results.extract_database"\n        - "$results.extract_api.data"\n        - "$results.load_files"\n      key: "transaction_id"\n\n  - id: clean_data\n    action: transform_data\n    parameters:\n      data: "$results.merge_all"\n      operations:\n        - type: "remove_duplicates"\n          columns: ["transaction_id"]\n        - type: "fill_missing"\n          strategy: "forward"\n        - type: "standardize_formats"\n          columns:\n            date: "YYYY-MM-DD"\n            amount: "decimal(10,2)"\n\n  # Validate\n  - id: validate_quality\n    action: check_quality\n    parameters:\n      data: "$results.clean_data"\n      checks:\n        - type: "completeness"\n          threshold: 0.99\n        - type: "accuracy"\n          validations:\n            amount: "range:0,1000000"\n            date: "date_range:2024-01-01,today"\n\n  # Load to destination\n  - id: save_processed\n    action: write_file\n    parameters:\n      path: "processed/sales_cleaned_{{ execution.date }}.parquet"\n      content: "$results.clean_data"\n      format: "parquet"\n\n  - id: update_database\n    condition: "{{ results.validate_quality.passed }}"\n    action: insert_data\n    parameters:\n      connection: "{{ env.DW_CONNECTION }}"\n      table: "sales_fact"\n      data: "$results.clean_data"\n      mode: "append"'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

def test_tool_reference_lines_1110_1152_9():
    """Test Python import from docs_sphinx/tool_reference.rst lines 1110-1152."""
    # Import test - check if modules are available
    code = 'from orchestrator.tools.base import Tool\n\nclass MyCustomTool(Tool):\n    def __init__(self):\n        super().__init__(\n            name="my-custom-tool",\n            description="Does something special"\n        )\n\n        # Define parameters\n        self.add_parameter(\n            name="input_data",\n            type="string",\n            description="Data to process",\n            required=True\n        )\n\n        self.add_parameter(\n            name="mode",\n            type="string",\n            description="Processing mode",\n            required=False,\n            default="standard",\n            enum=["standard", "advanced", "expert"]\n        )\n\n    async def execute(self, **kwargs):\n        """ Execute the tool action.""" \n        input_data = kwargs["input_data"]\n        mode = kwargs.get("mode", "standard")\n\n        # Your tool logic here\n        result = process_data(input_data, mode)\n\n        return {\n            "status": "success",\n            "result": result,\n            "metadata": {\n                "mode": mode,\n                "timestamp": datetime.now()\n            }\n        }'
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_tool_reference_lines_1160_1175_10():
    """Test Python import from docs_sphinx/tool_reference.rst lines 1160-1175."""
    # Import test - check if modules are available
    code = 'from orchestrator.tools.base import default_registry\n\n# Register tool\ntool = MyCustomTool()\ndefault_registry.register(tool)\n\n# Use in pipeline\npipeline_yaml = """ \nsteps:\n  - id: custom_step\n    action: my-custom-tool\n    parameters:\n      input_data: "{{ inputs.data }}"\n      mode: "advanced"\n""" '
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

@pytest.mark.asyncio
async def test_tutorial_data_processing_lines_35_239_11():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_data_processing.rst lines 35-239."""
    import yaml
    
    yaml_content = 'name: sales-etl-pipeline\ndescription: Extract, transform, and load sales data\n\ninputs:\n  data_source:\n    type: string\n    description: "Path to source data file"\n    required: true\n\n  output_format:\n    type: string\n    description: "Output format"\n    default: "parquet"\n    validation:\n      enum: ["csv", "json", "parquet", "excel"]\n\n  date_range:\n    type: object\n    description: "Date range for filtering"\n    default:\n      start: "2024-01-01"\n      end: "2024-12-31"\n\noutputs:\n  processed_data:\n    type: string\n    value: "processed/sales_{{ execution.date }}.{{ inputs.output_format }}"\n\n  quality_report:\n    type: string\n    value: "reports/quality_{{ execution.date }}.json"\n\n  summary_stats:\n    type: string\n    value: "reports/summary_{{ execution.date }}.md"\n\nsteps:\n  # Extract: Load raw data\n  - id: extract_data\n    action: read_file\n    parameters:\n      path: "{{ inputs.data_source }}"\n      parse: true\n    error_handling:\n      retry:\n        max_attempts: 3\n      fallback:\n        action: generate_content\n        parameters:\n          prompt: "Generate sample sales data for testing"\n\n  # Transform: Clean and process data\n  - id: clean_data\n    action: transform_data\n    parameters:\n      data: "$results.extract_data"\n      operations:\n        # Standardize column names\n        - type: "rename_columns"\n          mapping:\n            "Sale Date": "sale_date"\n            "Customer Name": "customer_name"\n            "Product ID": "product_id"\n            "Sale Amount": "amount"\n            "Quantity": "quantity"\n            "Sales Rep": "sales_rep"\n\n        # Convert data types\n        - type: "convert_types"\n          conversions:\n            sale_date: "datetime"\n            amount: "float"\n            quantity: "integer"\n            product_id: "string"\n\n        # Remove duplicates\n        - type: "remove_duplicates"\n          columns: ["product_id", "sale_date", "customer_name"]\n\n        # Handle missing values\n        - type: "fill_missing"\n          strategy: "forward"\n          columns: ["sales_rep"]\n\n        # Add calculated fields\n        - type: "add_column"\n          name: "total_value"\n          expression: "amount * quantity"\n\n        - type: "add_column"\n          name: "quarter"\n          expression: "quarter(sale_date)"\n\n        - type: "add_column"\n          name: "year"\n          expression: "year(sale_date)"\n\n  # Filter data by date range\n  - id: filter_data\n    action: filter_data\n    parameters:\n      data: "$results.clean_data"\n      conditions:\n        - field: "sale_date"\n          operator: "gte"\n          value: "{{ inputs.date_range.start }}"\n        - field: "sale_date"\n          operator: "lte"\n          value: "{{ inputs.date_range.end }}"\n        - field: "amount"\n          operator: "gt"\n          value: 0\n\n  # Data quality validation\n  - id: validate_quality\n    action: check_quality\n    parameters:\n      data: "$results.filter_data"\n      checks:\n        - type: "completeness"\n          threshold: 0.95\n          columns: ["product_id", "amount", "sale_date"]\n\n        - type: "uniqueness"\n          columns: ["product_id", "sale_date", "customer_name"]\n\n        - type: "consistency"\n          rules:\n            - "total_value == amount * quantity"\n            - "amount > 0"\n            - "quantity > 0"\n\n        - type: "accuracy"\n          validations:\n            product_id: "regex:^PROD-[0-9]{6}$"\n            amount: "range:1,50000"\n            quantity: "range:1,1000"\n\n  # Generate summary statistics\n  - id: calculate_summary\n    action: aggregate_data\n    parameters:\n      data: "$results.filter_data"\n      group_by: ["year", "quarter"]\n      aggregations:\n        total_sales:\n          column: "total_value"\n          function: "sum"\n        avg_sale:\n          column: "amount"\n          function: "mean"\n        num_transactions:\n          column: "*"\n          function: "count"\n        unique_customers:\n          column: "customer_name"\n          function: "nunique"\n        top_product:\n          column: "product_id"\n          function: "mode"\n\n  # Load: Save processed data\n  - id: save_processed_data\n    action: convert_format\n    parameters:\n      data: "$results.filter_data"\n      to_format: "{{ inputs.output_format }}"\n      output_path: "{{ outputs.processed_data }}"\n      options:\n        compression: "snappy"\n        index: false\n\n  # Save quality report\n  - id: save_quality_report\n    action: write_file\n    parameters:\n      path: "{{ outputs.quality_report }}"\n      content: "{{ results.validate_quality | json }}"\n\n  # Generate readable summary\n  - id: create_summary_report\n    action: generate_content\n    parameters:\n      prompt: |\n        Create a summary report for sales data processing:\n\n        Quality Results: {{ results.validate_quality | json }}\n        Summary Statistics: {{ results.calculate_summary | json }}\n\n        Include:\n        - Data quality assessment\n        - Key metrics and trends\n        - Any issues or recommendations\n        - Processing summary\n\n      style: "professional"\n      format: "markdown"\n\n  # Save summary report\n  - id: save_summary\n    action: write_file\n    parameters:\n      path: "{{ outputs.summary_stats }}"\n      content: "$results.create_summary_report"'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

def test_tutorial_data_processing_lines_245_264_12():
    """Test Python import from docs_sphinx/tutorials/tutorial_data_processing.rst lines 245-264."""
    # Import test - check if modules are available
    code = 'import orchestrator as orc\n\n# Initialize\norc.init_models()\n\n# Compile pipeline\netl_pipeline = orc.compile("sales_etl.yaml")\n\n# Process sales data\nresult = etl_pipeline.run(\n    data_source="data/raw/sales_2024.csv",\n    output_format="parquet",\n    date_range={\n        "start": "2024-01-01",\n        "end": "2024-06-30"\n    }\n)\n\nprint(f"ETL completed: {result}")'
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

@pytest.mark.asyncio
async def test_tutorial_data_processing_lines_277_521_13():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_data_processing.rst lines 277-521."""
    import yaml
    
    yaml_content = 'name: multi-source-integration\ndescription: Integrate data from multiple sources with validation\n\ninputs:\n  sources:\n    type: object\n    description: "Data source configurations"\n    required: true\n    # Example:\n    # database:\n    #   type: "postgresql"\n    #   connection: "postgresql://..."\n    #   query: "SELECT * FROM sales"\n    # api:\n    #   type: "rest"\n    #   url: "https://api.company.com/data"\n    #   headers: {...}\n    # files:\n    #   type: "file"\n    #   paths: ["data1.csv", "data2.json"]\n\n  merge_strategy:\n    type: string\n    description: "How to merge data sources"\n    default: "outer"\n    validation:\n      enum: ["inner", "outer", "left", "right"]\n\n  deduplication_fields:\n    type: array\n    description: "Fields to use for deduplication"\n    default: ["id", "timestamp"]\n\noutputs:\n  integrated_data:\n    type: string\n    value: "integrated/master_data_{{ execution.timestamp }}.parquet"\n\n  integration_report:\n    type: string\n    value: "reports/integration_{{ execution.timestamp }}.md"\n\nsteps:\n  # Extract from database sources\n  - id: extract_database\n    condition: "\'database\' in inputs.sources"\n    action: query_database\n    parameters:\n      connection: "{{ inputs.sources.database.connection }}"\n      query: "{{ inputs.sources.database.query }}"\n      fetch_size: 10000\n    error_handling:\n      continue_on_error: true\n\n  # Extract from API sources\n  - id: extract_api\n    condition: "\'api\' in inputs.sources"\n    action: call_api\n    parameters:\n      url: "{{ inputs.sources.api.url }}"\n      method: "GET"\n      headers: "{{ inputs.sources.api.headers | default({}) }}"\n      params: "{{ inputs.sources.api.params | default({}) }}"\n      timeout: 300\n    error_handling:\n      retry:\n        max_attempts: 3\n        backoff: "exponential"\n\n  # Extract from file sources\n  - id: extract_files\n    condition: "\'files\' in inputs.sources"\n    for_each: "{{ inputs.sources.files.paths }}"\n    as: file_path\n    action: read_file\n    parameters:\n      path: "{{ file_path }}"\n      parse: true\n\n  # Standardize data schemas\n  - id: standardize_database\n    condition: "results.extract_database is defined"\n    action: transform_data\n    parameters:\n      data: "$results.extract_database"\n      operations:\n        - type: "add_column"\n          name: "source"\n          value: "database"\n        - type: "standardize_schema"\n          target_schema:\n            id: "string"\n            timestamp: "datetime"\n            value: "float"\n            category: "string"\n\n  - id: standardize_api\n    condition: "results.extract_api is defined"\n    action: transform_data\n    parameters:\n      data: "$results.extract_api.data"\n      operations:\n        - type: "add_column"\n          name: "source"\n          value: "api"\n        - type: "flatten_nested"\n          columns: ["metadata", "attributes"]\n        - type: "standardize_schema"\n          target_schema:\n            id: "string"\n            timestamp: "datetime"\n            value: "float"\n            category: "string"\n\n  - id: standardize_files\n    condition: "results.extract_files is defined"\n    action: transform_data\n    parameters:\n      data: "$results.extract_files"\n      operations:\n        - type: "add_column"\n          name: "source"\n          value: "files"\n        - type: "combine_files"\n          strategy: "union"\n        - type: "standardize_schema"\n          target_schema:\n            id: "string"\n            timestamp: "datetime"\n            value: "float"\n            category: "string"\n\n  # Merge all data sources\n  - id: merge_sources\n    action: merge_data\n    parameters:\n      datasets:\n        - "$results.standardize_database"\n        - "$results.standardize_api"\n        - "$results.standardize_files"\n      how: "{{ inputs.merge_strategy }}"\n      on: ["id"]\n      suffixes: ["_db", "_api", "_file"]\n\n  # Remove duplicates\n  - id: deduplicate\n    action: transform_data\n    parameters:\n      data: "$results.merge_sources"\n      operations:\n        - type: "remove_duplicates"\n          columns: "{{ inputs.deduplication_fields }}"\n          keep: "last"  # Keep most recent\n\n  # Data quality assessment\n  - id: assess_integration_quality\n    action: check_quality\n    parameters:\n      data: "$results.deduplicate"\n      checks:\n        - type: "completeness"\n          threshold: 0.90\n          critical_columns: ["id", "timestamp"]\n\n        - type: "consistency"\n          rules:\n            - "value_db == value_api OR value_db IS NULL OR value_api IS NULL"\n            - "timestamp >= \'2020-01-01\'"\n\n        - type: "accuracy"\n          validations:\n            id: "not_null"\n            timestamp: "datetime_format"\n            value: "numeric_range:-1000000,1000000"\n\n  # Resolve conflicts between sources\n  - id: resolve_conflicts\n    action: transform_data\n    parameters:\n      data: "$results.deduplicate"\n      operations:\n        - type: "resolve_conflicts"\n          strategy: "priority"\n          priority_order: ["database", "api", "files"]\n          conflict_columns: ["value", "category"]\n\n        - type: "add_column"\n          name: "confidence_score"\n          expression: "calculate_confidence(source_count, data_age, validation_status)"\n\n  # Create final integrated dataset\n  - id: finalize_integration\n    action: transform_data\n    parameters:\n      data: "$results.resolve_conflicts"\n      operations:\n        - type: "select_columns"\n          columns: ["id", "timestamp", "value", "category", "source", "confidence_score"]\n\n        - type: "sort"\n          columns: ["timestamp"]\n          ascending: [false]\n\n  # Save integrated data\n  - id: save_integrated\n    action: convert_format\n    parameters:\n      data: "$results.finalize_integration"\n      to_format: "parquet"\n      output_path: "{{ outputs.integrated_data }}"\n      options:\n        compression: "snappy"\n        partition_cols: ["category"]\n\n  # Generate integration report\n  - id: create_integration_report\n    action: generate_content\n    parameters:\n      prompt: |\n        Create an integration report for multi-source data merge:\n\n        Sources processed:\n        {% for source in inputs.sources.keys() %}\n        - {{ source }}\n        {% endfor %}\n\n        Quality assessment: {{ results.assess_integration_quality | json }}\n        Final record count: {{ results.finalize_integration | length }}\n\n        Include:\n        - Source summary and statistics\n        - Data quality metrics\n        - Conflict resolution summary\n        - Recommendations for data improvement\n\n      style: "technical"\n      format: "markdown"\n\n  # Save integration report\n  - id: save_report\n    action: write_file\n    parameters:\n      path: "{{ outputs.integration_report }}"\n      content: "$results.create_integration_report"'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

def test_tutorial_data_processing_lines_527_558_14():
    """Test Python import from docs_sphinx/tutorials/tutorial_data_processing.rst lines 527-558."""
    # Import test - check if modules are available
    code = 'import orchestrator as orc\n\n# Initialize\norc.init_models()\n\n# Compile integration pipeline\nintegration = orc.compile("data_integration.yaml")\n\n# Integrate data from multiple sources\nresult = integration.run(\n    sources={\n        "database": {\n            "type": "postgresql",\n            "connection": "postgresql://user:pass@localhost/mydb",\n            "query": "SELECT * FROM transactions WHERE date >= \'2024-01-01\'"\n        },\n        "api": {\n            "type": "rest",\n            "url": "https://api.external.com/v1/data",\n            "headers": {"Authorization": "Bearer token123"}\n        },\n        "files": {\n            "type": "file",\n            "paths": ["data/file1.csv", "data/file2.json"]\n        }\n    },\n    merge_strategy="outer",\n    deduplication_fields=["transaction_id", "timestamp"]\n)\n\nprint(f"Integration completed: {result}")'
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

@pytest.mark.asyncio
async def test_tutorial_data_processing_lines_571_815_15():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_data_processing.rst lines 571-815."""
    import yaml
    
    yaml_content = 'name: data-quality-assessment\ndescription: Comprehensive data quality evaluation and reporting\n\ninputs:\n  dataset_path:\n    type: string\n    required: true\n\n  quality_rules:\n    type: object\n    description: "Custom quality rules"\n    default:\n      completeness_threshold: 0.95\n      uniqueness_fields: ["id"]\n      date_range_field: "created_date"\n      numeric_fields: ["amount", "quantity"]\n\n  remediation_mode:\n    type: string\n    description: "How to handle quality issues"\n    default: "report"\n    validation:\n      enum: ["report", "fix", "quarantine"]\n\noutputs:\n  quality_report:\n    type: string\n    value: "quality/report_{{ execution.timestamp }}.html"\n\n  cleaned_data:\n    type: string\n    value: "quality/cleaned_{{ execution.timestamp }}.parquet"\n\n  issues_log:\n    type: string\n    value: "quality/issues_{{ execution.timestamp }}.json"\n\nsteps:\n  # Load the dataset\n  - id: load_dataset\n    action: read_file\n    parameters:\n      path: "{{ inputs.dataset_path }}"\n      parse: true\n\n  # Basic data profiling\n  - id: profile_data\n    action: analyze_data\n    parameters:\n      data: "$results.load_dataset"\n      analysis_types:\n        - schema\n        - statistics\n        - distributions\n        - patterns\n        - outliers\n\n  # Completeness assessment\n  - id: check_completeness\n    action: check_quality\n    parameters:\n      data: "$results.load_dataset"\n      checks:\n        - type: "completeness"\n          threshold: "{{ inputs.quality_rules.completeness_threshold }}"\n          report_by_column: true\n\n        - type: "null_patterns"\n          identify_patterns: true\n\n  # Uniqueness validation\n  - id: check_uniqueness\n    action: validate_data\n    parameters:\n      data: "$results.load_dataset"\n      rules:\n        - name: "primary_key_uniqueness"\n          type: "uniqueness"\n          columns: "{{ inputs.quality_rules.uniqueness_fields }}"\n          severity: "error"\n\n        - name: "near_duplicates"\n          type: "similarity"\n          threshold: 0.9\n          columns: ["name", "email"]\n          severity: "warning"\n\n  # Consistency validation\n  - id: check_consistency\n    action: validate_data\n    parameters:\n      data: "$results.load_dataset"\n      rules:\n        - name: "date_logic"\n          condition: "start_date <= end_date"\n          severity: "error"\n\n        - name: "numeric_consistency"\n          condition: "total == sum(line_items)"\n          severity: "error"\n\n        - name: "referential_integrity"\n          type: "foreign_key"\n          reference_table: "lookup_table"\n          foreign_key: "category_id"\n          severity: "warning"\n\n  # Accuracy validation\n  - id: check_accuracy\n    action: validate_data\n    parameters:\n      data: "$results.load_dataset"\n      rules:\n        - name: "email_format"\n          field: "email"\n          validation: "regex:^[\\\\\\\\w.-]+@[\\\\\\\\w.-]+\\\\\\\\.\\\\\\\\w+$"\n          severity: "warning"\n\n        - name: "phone_format"\n          field: "phone"\n          validation: "regex:^\\\\\\\\+?1?\\\\\\\\d{9,15}$"\n          severity: "info"\n\n        - name: "numeric_ranges"\n          field: "{{ inputs.quality_rules.numeric_fields }}"\n          validation: "range:0,999999"\n          severity: "error"\n\n  # Timeliness assessment\n  - id: check_timeliness\n    action: validate_data\n    parameters:\n      data: "$results.load_dataset"\n      rules:\n        - name: "data_freshness"\n          field: "{{ inputs.quality_rules.date_range_field }}"\n          condition: "date_diff(value, today()) <= 30"\n          severity: "warning"\n          message: "Data is older than 30 days"\n\n  # Outlier detection\n  - id: detect_outliers\n    action: analyze_data\n    parameters:\n      data: "$results.load_dataset"\n      analysis_types:\n        - outliers\n      methods:\n        - statistical  # Z-score, IQR\n        - isolation_forest\n        - local_outlier_factor\n      numeric_columns: "{{ inputs.quality_rules.numeric_fields }}"\n\n  # Compile quality issues\n  - id: compile_issues\n    action: transform_data\n    parameters:\n      data:\n        completeness: "$results.check_completeness"\n        uniqueness: "$results.check_uniqueness"\n        consistency: "$results.check_consistency"\n        accuracy: "$results.check_accuracy"\n        timeliness: "$results.check_timeliness"\n        outliers: "$results.detect_outliers"\n      operations:\n        - type: "consolidate_issues"\n          prioritize: true\n        - type: "categorize_severity"\n          levels: ["critical", "major", "minor", "info"]\n\n  # Data remediation (if requested)\n  - id: remediate_data\n    condition: "inputs.remediation_mode in [\'fix\', \'quarantine\']"\n    action: transform_data\n    parameters:\n      data: "$results.load_dataset"\n      operations:\n        # Fix common issues\n        - type: "standardize_formats"\n          columns:\n            email: "lowercase"\n            phone: "normalize_phone"\n            name: "title_case"\n\n        - type: "fill_missing"\n          strategy: "smart"  # Use ML-based imputation\n          columns: "{{ inputs.quality_rules.numeric_fields }}"\n\n        - type: "remove_outliers"\n          method: "iqr"\n          columns: "{{ inputs.quality_rules.numeric_fields }}"\n          action: "{{ \'quarantine\' if inputs.remediation_mode == \'quarantine\' else \'remove\' }}"\n\n        - type: "deduplicate"\n          strategy: "keep_best"  # Keep record with highest completeness\n\n  # Generate comprehensive quality report\n  - id: create_quality_report\n    action: generate_content\n    parameters:\n      prompt: |\n        Create a comprehensive data quality report:\n\n        Dataset: {{ inputs.dataset_path }}\n        Profile: {{ results.profile_data | json }}\n        Issues: {{ results.compile_issues | json }}\n\n        Include:\n        1. Executive Summary\n        2. Data Profile Overview\n        3. Quality Metrics Dashboard\n        4. Issue Analysis by Category\n        5. Impact Assessment\n        6. Remediation Recommendations\n        7. Quality Score Calculation\n\n        Format as HTML with charts and tables.\n\n      style: "technical"\n      format: "html"\n      max_tokens: 3000\n\n  # Save quality report\n  - id: save_quality_report\n    action: write_file\n    parameters:\n      path: "{{ outputs.quality_report }}"\n      content: "$results.create_quality_report"\n\n  # Save cleaned data (if remediation performed)\n  - id: save_cleaned_data\n    condition: "inputs.remediation_mode in [\'fix\', \'quarantine\']"\n    action: write_file\n    parameters:\n      path: "{{ outputs.cleaned_data }}"\n      content: "$results.remediate_data"\n      format: "parquet"\n\n  # Save issues log\n  - id: save_issues_log\n    action: write_file\n    parameters:\n      path: "{{ outputs.issues_log }}"\n      content: "{{ results.compile_issues | json }}"'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

@pytest.mark.asyncio
async def test_tutorial_data_processing_lines_828_923_16():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_data_processing.rst lines 828-923."""
    import yaml
    
    yaml_content = 'name: realtime-data-processing\ndescription: Process streaming data with real-time analytics\n\ninputs:\n  stream_source:\n    type: object\n    description: "Stream configuration"\n    required: true\n    # Example:\n    # type: "kafka"\n    # topic: "events"\n    # batch_size: 1000\n    # window_size: "5m"\n\n  processing_rules:\n    type: array\n    description: "Processing rules to apply"\n    default:\n      - type: "filter"\n        condition: "event_type in [\'purchase\', \'click\']"\n      - type: "enrich"\n        lookup_table: "user_profiles"\n      - type: "aggregate"\n        window: "5m"\n        metrics: ["count", "sum", "avg"]\n\noutputs:\n  processed_stream:\n    type: string\n    value: "stream/processed_{{ execution.date }}"\n\n  alerts:\n    type: string\n    value: "alerts/stream_alerts_{{ execution.timestamp }}.json"\n\nsteps:\n  # Connect to stream source\n  - id: connect_stream\n    action: connect_stream\n    parameters:\n      source: "{{ inputs.stream_source }}"\n      batch_size: "{{ inputs.stream_source.batch_size | default(1000) }}"\n      timeout: 30\n\n  # Process incoming batches\n  - id: process_batches\n    action: process_stream_batch\n    parameters:\n      stream: "$results.connect_stream"\n      processing_rules: "{{ inputs.processing_rules }}"\n      window_config:\n        size: "{{ inputs.stream_source.window_size | default(\'5m\') }}"\n        type: "tumbling"  # or "sliding", "session"\n\n  # Real-time anomaly detection\n  - id: detect_anomalies\n    action: detect_anomalies\n    parameters:\n      data: "$results.process_batches"\n      methods:\n        - statistical_control\n        - machine_learning\n      thresholds:\n        statistical: 3.0  # standard deviations\n        ml_confidence: 0.95\n\n  # Generate alerts\n  - id: generate_alerts\n    condition: "results.detect_anomalies.anomalies | length > 0"\n    action: generate_content\n    parameters:\n      prompt: |\n        Generate alerts for detected anomalies:\n        {{ results.detect_anomalies.anomalies | json }}\n\n        Include severity, description, and recommended actions.\n\n      format: "json"\n\n  # Save processed data\n  - id: save_processed\n    action: write_stream\n    parameters:\n      data: "$results.process_batches"\n      destination: "{{ outputs.processed_stream }}"\n      format: "parquet"\n      partition_by: ["date", "hour"]\n\n  # Save alerts\n  - id: save_alerts\n    condition: "results.generate_alerts is defined"\n    action: write_file\n    parameters:\n      path: "{{ outputs.alerts }}"\n      content: "$results.generate_alerts"'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

@pytest.mark.asyncio
async def test_tutorial_data_processing_lines_932_990_17():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_data_processing.rst lines 932-990."""
    import yaml
    
    yaml_content = 'name: customer-data-platform\ndescription: Unified customer data processing and analytics\n\ninputs:\n  customer_sources:\n    type: object\n    required: true\n    # CRM, support tickets, web analytics, purchase history\n\nsteps:\n  # Extract from all customer touchpoints\n  - id: extract_crm\n    action: query_database\n    parameters:\n      connection: "{{ inputs.customer_sources.crm.connection }}"\n      query: "SELECT * FROM customers WHERE updated_at >= CURRENT_DATE - INTERVAL \'1 day\'"\n\n  - id: extract_support\n    action: call_api\n    parameters:\n      url: "{{ inputs.customer_sources.support.api_url }}"\n      headers:\n        Authorization: "Bearer {{ env.SUPPORT_API_KEY }}"\n\n  - id: extract_analytics\n    action: read_file\n    parameters:\n      path: "{{ inputs.customer_sources.analytics.export_path }}"\n      parse: true\n\n  # Create unified customer profiles\n  - id: merge_customer_data\n    action: merge_data\n    parameters:\n      datasets:\n        - "$results.extract_crm"\n        - "$results.extract_support"\n        - "$results.extract_analytics"\n      on: "customer_id"\n      how: "outer"\n\n  # Calculate customer metrics\n  - id: calculate_metrics\n    action: transform_data\n    parameters:\n      data: "$results.merge_customer_data"\n      operations:\n        - type: "add_column"\n          name: "customer_lifetime_value"\n          expression: "sum(purchase_amounts) * retention_probability"\n\n        - type: "add_column"\n          name: "churn_risk_score"\n          expression: "calculate_churn_risk(days_since_last_activity, support_tickets, engagement_score)"\n\n        - type: "add_column"\n          name: "segment"\n          expression: "classify_customer_segment(clv, engagement, recency)"'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

@pytest.mark.asyncio
async def test_tutorial_data_processing_lines_996_1062_18():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_data_processing.rst lines 996-1062."""
    import yaml
    
    yaml_content = 'name: financial-data-pipeline\ndescription: Process financial transactions with compliance checks\n\ninputs:\n  transaction_sources:\n    type: array\n    required: true\n\n  compliance_rules:\n    type: object\n    required: true\n\nsteps:\n  # Extract transactions from multiple sources\n  - id: extract_transactions\n    for_each: "{{ inputs.transaction_sources }}"\n    as: source\n    action: extract_financial_data\n    parameters:\n      source_config: "{{ source }}"\n      date_range: "{{ execution.date | date_range(\'-1d\') }}"\n\n  # Compliance screening\n  - id: screen_transactions\n    action: validate_data\n    parameters:\n      data: "$results.extract_transactions"\n      rules:\n        - name: "aml_screening"\n          type: "anti_money_laundering"\n          threshold: "{{ inputs.compliance_rules.aml_threshold }}"\n\n        - name: "sanctions_check"\n          type: "sanctions_screening"\n          watchlists: "{{ inputs.compliance_rules.watchlists }}"\n\n        - name: "pep_screening"\n          type: "politically_exposed_person"\n          databases: "{{ inputs.compliance_rules.pep_databases }}"\n\n  # Risk scoring\n  - id: calculate_risk_scores\n    action: transform_data\n    parameters:\n      data: "$results.extract_transactions"\n      operations:\n        - type: "add_column"\n          name: "risk_score"\n          expression: "calculate_transaction_risk(amount, counterparty, geography, transaction_type)"\n\n        - type: "add_column"\n          name: "risk_category"\n          expression: "categorize_risk(risk_score)"\n\n  # Generate compliance report\n  - id: create_compliance_report\n    action: generate_content\n    parameters:\n      prompt: |\n        Generate daily compliance report:\n\n        Transactions processed: {{ results.extract_transactions | length }}\n        Screening results: {{ results.screen_transactions | json }}\n        Risk distribution: {{ results.calculate_risk_scores | group_by(\'risk_category\') }}\n\n        Include regulatory compliance status and any required actions.'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

def test_tutorial_data_processing_lines_1073_1078_19():
    """Test YAML snippet from docs_sphinx/tutorials/tutorial_data_processing.rst lines 1073-1078."""
    import yaml
    
    yaml_content = '# Your challenge:\n# - Extract: Orders, customers, products, reviews\n# - Transform: Calculate metrics, segment customers\n# - Load: Create analytics-ready datasets\n# - Quality: Validate business rules'
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_tutorial_data_processing_lines_1086_1091_20():
    """Test YAML snippet from docs_sphinx/tutorials/tutorial_data_processing.rst lines 1086-1091."""
    import yaml
    
    yaml_content = '# Requirements:\n# - Handle high-volume time series data\n# - Detect sensor anomalies\n# - Aggregate by time windows\n# - Generate maintenance alerts'
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_tutorial_data_processing_lines_1099_1104_21():
    """Test YAML snippet from docs_sphinx/tutorials/tutorial_data_processing.rst lines 1099-1104."""
    import yaml
    
    yaml_content = '# Features:\n# - Extract from multiple platforms\n# - Text analysis and sentiment\n# - Trend detection\n# - Influence measurement'
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.asyncio
async def test_tutorial_web_research_lines_36_91_22():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_web_research.rst lines 36-91."""
    import yaml
    
    yaml_content = 'name: basic-web-search\ndescription: Search the web and compile results into a report\n\ninputs:\n  query:\n    type: string\n    description: "Search query"\n    required: true\n\n  max_results:\n    type: integer\n    description: "Maximum number of results to return"\n    default: 10\n    validation:\n      min: 1\n      max: 50\n\noutputs:\n  report:\n    type: string\n    value: "search_results_{{ inputs.query | slugify }}.md"\n\nsteps:\n  # Search the web\n  - id: search\n    action: search_web\n    parameters:\n      query: "{{ inputs.query }}"\n      max_results: "{{ inputs.max_results }}"\n      include_snippets: true\n\n  # Compile into markdown report\n  - id: compile_report\n    action: generate_content\n    parameters:\n      prompt: |\n        Create a well-organized markdown report from these search results:\n\n        {{ results.search | json }}\n\n        Include:\n        - Executive summary\n        - Key findings\n        - Source links\n        - Relevant details from each result\n\n      style: "professional"\n      format: "markdown"\n\n  # Save the report\n  - id: save_report\n    action: write_file\n    parameters:\n      path: "{{ outputs.report }}"\n      content: "$results.compile_report"'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

def test_tutorial_web_research_lines_97_117_23():
    """Test Python import from docs_sphinx/tutorials/tutorial_web_research.rst lines 97-117."""
    # Import test - check if modules are available
    code = 'import orchestrator as orc\n\n# Initialize\norc.init_models()\n\n# Compile and run\npipeline = orc.compile("web_search.yaml")\n\n# Search for different topics\nresult1 = pipeline.run(\n    query="artificial intelligence trends 2024",\n    max_results=15\n)\n\nresult2 = pipeline.run(\n    query="sustainable energy solutions",\n    max_results=10\n)\n\nprint(f"Generated reports: {result1}, {result2}")'
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_tutorial_web_research_lines_125_142_24():
    """Test markdown snippet from docs_sphinx/tutorials/tutorial_web_research.rst lines 125-142."""
    pytest.skip("Snippet type 'markdown' not yet supported")

@pytest.mark.asyncio
async def test_tutorial_web_research_lines_155_346_25():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_web_research.rst lines 155-346."""
    import yaml
    
    yaml_content = 'name: multi-source-research\ndescription: Comprehensive research using web, news, and academic sources\n\ninputs:\n  topic:\n    type: string\n    required: true\n\n  depth:\n    type: string\n    description: "Research depth"\n    default: "medium"\n    validation:\n      enum: ["light", "medium", "deep"]\n\n  include_sources:\n    type: array\n    description: "Sources to include"\n    default: ["web", "news", "academic"]\n    validation:\n      enum_items: ["web", "news", "academic", "patents"]\n\noutputs:\n  comprehensive_report:\n    type: string\n    value: "research/{{ inputs.topic | slugify }}_comprehensive.md"\n\n  data_file:\n    type: string\n    value: "research/{{ inputs.topic | slugify }}_data.json"\n\n# Research depth configuration\nconfig:\n  research_params:\n    light:\n      web_results: 10\n      news_results: 5\n      academic_results: 3\n    medium:\n      web_results: 20\n      news_results: 10\n      academic_results: 8\n    deep:\n      web_results: 40\n      news_results: 20\n      academic_results: 15\n\nsteps:\n  # Parallel search across sources\n  - id: search_sources\n    parallel:\n      # Web search\n      - id: web_search\n        condition: "\'web\' in inputs.include_sources"\n        action: search_web\n        parameters:\n          query: "{{ inputs.topic }} comprehensive overview"\n          max_results: "{{ config.research_params[inputs.depth].web_results }}"\n          include_snippets: true\n\n      # News search\n      - id: news_search\n        condition: "\'news\' in inputs.include_sources"\n        action: search_news\n        parameters:\n          query: "{{ inputs.topic }}"\n          max_results: "{{ config.research_params[inputs.depth].news_results }}"\n          date_range: "last_month"\n\n      # Academic search\n      - id: academic_search\n        condition: "\'academic\' in inputs.include_sources"\n        action: search_academic\n        parameters:\n          query: "{{ inputs.topic }}"\n          max_results: "{{ config.research_params[inputs.depth].academic_results }}"\n          year_range: "2020-2024"\n          peer_reviewed: true\n\n  # Extract key information from each source\n  - id: extract_information\n    action: extract_information\n    parameters:\n      content: "$results.search_sources"\n      extract:\n        key_facts:\n          description: "Important facts and findings"\n        statistics:\n          description: "Numerical data and metrics"\n        expert_opinions:\n          description: "Quotes and opinions from experts"\n        trends:\n          description: "Emerging trends and developments"\n        challenges:\n          description: "Problems and challenges mentioned"\n        opportunities:\n          description: "Opportunities and potential solutions"\n\n  # Cross-validate information\n  - id: validate_facts\n    action: validate_data\n    parameters:\n      data: "$results.extract_information"\n      rules:\n        - name: "source_diversity"\n          condition: "count(unique(sources)) >= 2"\n          severity: "warning"\n          message: "Information should be confirmed by multiple sources"\n\n        - name: "recent_information"\n          field: "date"\n          condition: "date_diff(value, today()) <= 365"\n          severity: "info"\n          message: "Information is from the last year"\n\n  # Generate comprehensive analysis\n  - id: analyze_findings\n    action: generate_content\n    parameters:\n      prompt: |\n        Analyze the following research data about {{ inputs.topic }}:\n\n        {{ results.extract_information | json }}\n\n        Provide:\n        1. Current state analysis\n        2. Key trends identification\n        3. Challenge assessment\n        4. Future outlook\n        5. Recommendations\n\n        Base your analysis on the evidence provided and note any limitations.\n\n      style: "analytical"\n      max_tokens: 2000\n\n  # Create structured data export\n  - id: export_data\n    action: transform_data\n    parameters:\n      data:\n        topic: "{{ inputs.topic }}"\n        research_date: "{{ execution.timestamp }}"\n        depth: "{{ inputs.depth }}"\n        sources_used: "{{ inputs.include_sources }}"\n        extracted_info: "$results.extract_information"\n        validation_results: "$results.validate_facts"\n        analysis: "$results.analyze_findings"\n      operations:\n        - type: "convert_format"\n          to_format: "json"\n\n  # Save structured data\n  - id: save_data\n    action: write_file\n    parameters:\n      path: "{{ outputs.data_file }}"\n      content: "$results.export_data"\n\n  # Generate final report\n  - id: create_report\n    action: generate_content\n    parameters:\n      prompt: |\n        Create a comprehensive research report about {{ inputs.topic }} using:\n\n        Analysis: {{ results.analyze_findings }}\n\n        Structure the report with:\n        1. Executive Summary\n        2. Methodology\n        3. Current State Analysis\n        4. Key Findings\n        5. Trends and Developments\n        6. Challenges and Limitations\n        7. Future Outlook\n        8. Recommendations\n        9. Sources and References\n\n        Include confidence levels for major claims.\n\n      style: "professional"\n      format: "markdown"\n      max_tokens: 3000\n\n  # Save final report\n  - id: save_report\n    action: write_file\n    parameters:\n      path: "{{ outputs.comprehensive_report }}"\n      content: "$results.create_report"'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

def test_tutorial_web_research_lines_352_375_26():
    """Test Python import from docs_sphinx/tutorials/tutorial_web_research.rst lines 352-375."""
    # Import test - check if modules are available
    code = 'import orchestrator as orc\n\n# Initialize\norc.init_models()\n\n# Compile pipeline\npipeline = orc.compile("multi_source_research.yaml")\n\n# Run deep research on quantum computing\nresult = pipeline.run(\n    topic="quantum computing applications",\n    depth="deep",\n    include_sources=["web", "academic", "news"]\n)\n\nprint(f"Research complete: {result}")\n\n# Run lighter research on emerging tech\nresult2 = pipeline.run(\n    topic="edge computing trends",\n    depth="medium",\n    include_sources=["web", "news"]\n)'
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

@pytest.mark.asyncio
async def test_tutorial_web_research_lines_388_488_27():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_web_research.rst lines 388-488."""
    import yaml
    
    yaml_content = 'name: fact-checker\ndescription: Verify claims against multiple reliable sources\n\ninputs:\n  claims:\n    type: array\n    description: "Claims to verify"\n    required: true\n\n  confidence_threshold:\n    type: float\n    description: "Minimum confidence level to accept claims"\n    default: 0.7\n    validation:\n      min: 0.0\n      max: 1.0\n\noutputs:\n  fact_check_report:\n    type: string\n    value: "fact_check_{{ execution.timestamp | strftime(\'%Y%m%d_%H%M\') }}.md"\n\nsteps:\n  # Research each claim\n  - id: research_claims\n    for_each: "{{ inputs.claims }}"\n    as: claim\n    action: search_web\n    parameters:\n      query: "{{ claim }} verification facts evidence"\n      max_results: 15\n      include_snippets: true\n\n  # Extract supporting/contradicting evidence\n  - id: analyze_evidence\n    for_each: "{{ inputs.claims }}"\n    as: claim\n    action: extract_information\n    parameters:\n      content: "$results.research_claims[loop.index0]"\n      extract:\n        supporting_evidence:\n          description: "Evidence that supports the claim"\n        contradicting_evidence:\n          description: "Evidence that contradicts the claim"\n        source_credibility:\n          description: "Assessment of source reliability"\n        expert_opinions:\n          description: "Expert statements about the claim"\n\n  # Assess credibility of each claim\n  - id: assess_claims\n    for_each: "{{ inputs.claims }}"\n    as: claim\n    action: generate_content\n    parameters:\n      prompt: |\n        Assess the veracity of this claim: "{{ claim }}"\n\n        Based on the evidence:\n        {{ results.analyze_evidence[loop.index0] | json }}\n\n        Provide:\n        1. Verdict: True/False/Partially True/Insufficient Evidence\n        2. Confidence level (0-1)\n        3. Supporting evidence summary\n        4. Contradicting evidence summary\n        5. Overall assessment\n\n        Be objective and cite specific sources.\n\n      style: "analytical"\n      format: "structured"\n\n  # Compile fact-check report\n  - id: create_fact_check_report\n    action: generate_content\n    parameters:\n      prompt: |\n        Create a comprehensive fact-check report based on:\n\n        Claims assessed: {{ inputs.claims | json }}\n        Assessment results: {{ results.assess_claims | json }}\n\n        Format as a professional fact-checking article with:\n        1. Summary of findings\n        2. Individual claim assessments\n        3. Methodology used\n        4. Sources consulted\n        5. Limitations and caveats\n\n      style: "journalistic"\n      format: "markdown"\n\n  # Save report\n  - id: save_fact_check\n    action: write_file\n    parameters:\n      path: "{{ outputs.fact_check_report }}"\n      content: "$results.create_fact_check_report"'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

def test_tutorial_web_research_lines_494_514_28():
    """Test Python import from docs_sphinx/tutorials/tutorial_web_research.rst lines 494-514."""
    # Import test - check if modules are available
    code = 'import orchestrator as orc\n\n# Initialize\norc.init_models()\n\n# Compile fact-checker\nfact_checker = orc.compile("fact_checker.yaml")\n\n# Check various claims\nresult = fact_checker.run(\n    claims=[\n        "Electric vehicles produce zero emissions",\n        "AI will replace 50% of jobs by 2030",\n        "Quantum computers can break all current encryption",\n        "Renewable energy is now cheaper than fossil fuels"\n    ],\n    confidence_threshold=0.8\n)\n\nprint(f"Fact-check report: {result}")'
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

@pytest.mark.asyncio
async def test_tutorial_web_research_lines_527_805_29():
    """Test YAML pipeline from docs_sphinx/tutorials/tutorial_web_research.rst lines 527-805."""
    import yaml
    
    yaml_content = 'name: automated-report-generator\ndescription: Generate professional reports from research data\n\ninputs:\n  topic:\n    type: string\n    required: true\n\n  report_type:\n    type: string\n    description: "Type of report to generate"\n    default: "standard"\n    validation:\n      enum: ["executive", "technical", "standard", "briefing"]\n\n  target_audience:\n    type: string\n    description: "Primary audience for the report"\n    default: "general"\n    validation:\n      enum: ["executives", "technical", "general", "academic"]\n\n  sections:\n    type: array\n    description: "Sections to include in report"\n    default: ["summary", "introduction", "analysis", "conclusion"]\n\noutputs:\n  report_markdown:\n    type: string\n    value: "reports/{{ inputs.topic | slugify }}_{{ inputs.report_type }}.md"\n\n  report_pdf:\n    type: string\n    value: "reports/{{ inputs.topic | slugify }}_{{ inputs.report_type }}.pdf"\n\n  report_html:\n    type: string\n    value: "reports/{{ inputs.topic | slugify }}_{{ inputs.report_type }}.html"\n\n# Report templates by type\nconfig:\n  report_templates:\n    executive:\n      style: "executive"\n      length: "concise"\n      focus: "strategic"\n      sections: ["executive_summary", "key_findings", "recommendations", "appendix"]\n\n    technical:\n      style: "technical"\n      length: "detailed"\n      focus: "implementation"\n      sections: ["introduction", "technical_analysis", "methodology", "results", "conclusion"]\n\n    standard:\n      style: "professional"\n      length: "medium"\n      focus: "comprehensive"\n      sections: ["summary", "background", "analysis", "findings", "recommendations"]\n\n    briefing:\n      style: "concise"\n      length: "short"\n      focus: "actionable"\n      sections: ["situation", "assessment", "recommendations"]\n\nsteps:\n  # Gather comprehensive research data\n  - id: research_topic\n    action: search_web\n    parameters:\n      query: "{{ inputs.topic }} comprehensive analysis research"\n      max_results: 25\n      include_snippets: true\n\n  # Get recent news for current context\n  - id: current_context\n    action: search_news\n    parameters:\n      query: "{{ inputs.topic }}"\n      max_results: 10\n      date_range: "last_week"\n\n  # Extract structured information\n  - id: extract_report_data\n    action: extract_information\n    parameters:\n      content:\n        research: "$results.research_topic"\n        news: "$results.current_context"\n      extract:\n        key_points:\n          description: "Main points and findings"\n        statistics:\n          description: "Important numbers and data"\n        trends:\n          description: "Current and emerging trends"\n        implications:\n          description: "Implications and consequences"\n        expert_views:\n          description: "Expert opinions and quotes"\n        future_outlook:\n          description: "Predictions and future scenarios"\n\n  # Generate executive summary\n  - id: create_executive_summary\n    condition: "\'summary\' in inputs.sections or \'executive_summary\' in inputs.sections"\n    action: generate_content\n    parameters:\n      prompt: |\n        Create an executive summary for {{ inputs.target_audience }} audience about {{ inputs.topic }}.\n\n        Based on: {{ results.extract_report_data.key_points | json }}\n\n        Style: {{ config.report_templates[inputs.report_type].style }}\n        Focus: {{ config.report_templates[inputs.report_type].focus }}\n\n        Include the most critical points in 200-400 words.\n\n      style: "{{ config.report_templates[inputs.report_type].style }}"\n      max_tokens: 500\n\n  # Generate introduction/background\n  - id: create_introduction\n    condition: "\'introduction\' in inputs.sections or \'background\' in inputs.sections"\n    action: generate_content\n    parameters:\n      prompt: |\n        Write an introduction/background section about {{ inputs.topic }} for {{ inputs.target_audience }}.\n\n        Context: {{ results.extract_report_data | json }}\n\n        Provide necessary background and context for understanding the topic.\n\n      style: "{{ config.report_templates[inputs.report_type].style }}"\n      max_tokens: 800\n\n  # Generate main analysis\n  - id: create_analysis\n    condition: "\'analysis\' in inputs.sections or \'technical_analysis\' in inputs.sections"\n    action: generate_content\n    parameters:\n      prompt: |\n        Create a comprehensive analysis section about {{ inputs.topic }}.\n\n        Data: {{ results.extract_report_data | json }}\n\n        Style: {{ config.report_templates[inputs.report_type].style }}\n        Audience: {{ inputs.target_audience }}\n\n        Include:\n        - Current state analysis\n        - Trend analysis\n        - Key factors and drivers\n        - Challenges and opportunities\n\n        Support points with specific data and examples.\n\n      style: "{{ config.report_templates[inputs.report_type].style }}"\n      max_tokens: 1500\n\n  # Generate findings and implications\n  - id: create_findings\n    condition: "\'findings\' in inputs.sections or \'key_findings\' in inputs.sections"\n    action: generate_content\n    parameters:\n      prompt: |\n        Summarize key findings and implications regarding {{ inputs.topic }}.\n\n        Analysis: {{ results.create_analysis }}\n        Supporting data: {{ results.extract_report_data.implications | json }}\n\n        Present clear, actionable findings with implications.\n\n      style: "{{ config.report_templates[inputs.report_type].style }}"\n      max_tokens: 1000\n\n  # Generate recommendations\n  - id: create_recommendations\n    condition: "\'recommendations\' in inputs.sections"\n    action: generate_content\n    parameters:\n      prompt: |\n        Develop actionable recommendations based on the analysis of {{ inputs.topic }}.\n\n        Findings: {{ results.create_findings }}\n        Target audience: {{ inputs.target_audience }}\n\n        Provide specific, actionable recommendations with priorities and considerations.\n\n      style: "{{ config.report_templates[inputs.report_type].style }}"\n      max_tokens: 800\n\n  # Generate conclusion\n  - id: create_conclusion\n    condition: "\'conclusion\' in inputs.sections"\n    action: generate_content\n    parameters:\n      prompt: |\n        Write a strong conclusion for the {{ inputs.topic }} report.\n\n        Key findings: {{ results.create_findings }}\n        Recommendations: {{ results.create_recommendations }}\n\n        Synthesize the main points and end with a clear call to action.\n\n      style: "{{ config.report_templates[inputs.report_type].style }}"\n      max_tokens: 400\n\n  # Assemble complete report\n  - id: assemble_report\n    action: generate_content\n    parameters:\n      prompt: |\n        Compile a complete, professional report about {{ inputs.topic }}.\n\n        Report type: {{ inputs.report_type }}\n        Target audience: {{ inputs.target_audience }}\n\n        Sections to include:\n        {% if results.create_executive_summary %}\n        Executive Summary: {{ results.create_executive_summary }}\n        {% endif %}\n\n        {% if results.create_introduction %}\n        Introduction: {{ results.create_introduction }}\n        {% endif %}\n\n        {% if results.create_analysis %}\n        Analysis: {{ results.create_analysis }}\n        {% endif %}\n\n        {% if results.create_findings %}\n        Findings: {{ results.create_findings }}\n        {% endif %}\n\n        {% if results.create_recommendations %}\n        Recommendations: {{ results.create_recommendations }}\n        {% endif %}\n\n        {% if results.create_conclusion %}\n        Conclusion: {{ results.create_conclusion }}\n        {% endif %}\n\n        Format as a professional markdown document with:\n        - Proper headings and structure\n        - Table of contents\n        - Professional formatting\n        - Source citations where appropriate\n\n      style: "professional"\n      format: "markdown"\n      max_tokens: 4000\n\n  # Save markdown version\n  - id: save_markdown\n    action: write_file\n    parameters:\n      path: "{{ outputs.report_markdown }}"\n      content: "$results.assemble_report"\n\n  # Convert to PDF\n  - id: create_pdf\n    action: "!pandoc {{ outputs.report_markdown }} -o {{ outputs.report_pdf }} --pdf-engine=xelatex"\n    error_handling:\n      continue_on_error: true\n      fallback:\n        action: write_file\n        parameters:\n          path: "{{ outputs.report_pdf }}.txt"\n          content: "PDF generation requires pandoc with xelatex"\n\n  # Convert to HTML\n  - id: create_html\n    action: "!pandoc {{ outputs.report_markdown }} -o {{ outputs.report_html }} --standalone --css=style.css"\n    error_handling:\n      continue_on_error: true'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")
