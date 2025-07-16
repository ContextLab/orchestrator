"""Working tests for documentation code snippets - Batch 28."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# These tests focus on validation and syntax checking without execution
# They should run quickly and not require external dependencies


def test_tool_reference_lines_436_507_0():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 436-507."""
    # Description: **Parameters**:
    import yaml
    
    content = '# Transform data\n- id: transform\n  action: transform_data\n  parameters:\n    data: "$results.load_data"               # Required: Input data or path\n    operations:                              # Required: List of operations\n      - type: "rename_columns"\n        mapping:\n          old_name: "new_name"\n          price: "cost"\n      - type: "add_column"\n        name: "total"\n        expression: "quantity * cost"\n      - type: "drop_columns"\n        columns: ["unnecessary_field"]\n      - type: "convert_types"\n        conversions:\n          date: "datetime"\n          amount: "float"\n\n# Filter data\n- id: filter\n  action: filter_data\n  parameters:\n    data: "$results.transform"               # Required: Input data\n    conditions:                              # Required: Filter conditions\n      - field: "status"\n        operator: "equals"                   # equals|not_equals|contains|gt|lt|gte|lte\n        value: "active"\n      - field: "amount"\n        operator: "gt"\n        value: 1000\n    mode: "and"                              # Optional: and|or (default: and)\n\n# Aggregate data\n- id: aggregate\n  action: aggregate_data\n  parameters:\n    data: "$results.filter"                  # Required: Input data\n    group_by: ["category", "region"]        # Optional: Grouping columns\n    aggregations:                            # Required: Aggregation rules\n      total_amount:\n        column: "amount"\n        function: "sum"                      # sum|mean|median|min|max|count|std\n      average_price:\n        column: "price"\n        function: "mean"\n      item_count:\n        column: "*"\n        function: "count"\n\n# Merge data\n- id: merge\n  action: merge_data\n  parameters:\n    left: "$results.main_data"               # Required: Left dataset\n    right: "$results.lookup_data"            # Required: Right dataset\n    on: "customer_id"                        # Required: Join column(s)\n    how: "left"                              # Optional: left|right|inner|outer (default: left)\n    suffixes: ["_main", "_lookup"]          # Optional: Column suffixes\n\n# Convert format\n- id: convert\n  action: convert_format\n  parameters:\n    data: "$results.final_data"              # Required: Input data\n    from_format: "json"                      # Optional: Auto-detect if not specified\n    to_format: "parquet"                     # Required: Target format\n    options:                                 # Optional: Format-specific options\n      compression: "snappy"\n      index: false'
    
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


def test_tool_reference_lines_512_582_1():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 512-582."""
    # Description: **Example Pipeline**:
    import yaml
    
    content = 'name: sales-data-analysis\ndescription: Process and analyze sales data\n\nsteps:\n  # Load raw data\n  - id: load_sales\n    action: read_file\n    parameters:\n      path: "data/sales_2024.csv"\n      parse: true\n\n  # Clean and transform\n  - id: clean_data\n    action: transform_data\n    parameters:\n      data: "$results.load_sales"\n      operations:\n        - type: "rename_columns"\n          mapping:\n            "Sale Date": "sale_date"\n            "Customer Name": "customer_name"\n            "Product ID": "product_id"\n            "Sale Amount": "amount"\n        - type: "convert_types"\n          conversions:\n            sale_date: "datetime"\n            amount: "float"\n        - type: "add_column"\n          name: "quarter"\n          expression: "sale_date.quarter"\n\n  # Filter valid sales\n  - id: filter_valid\n    action: filter_data\n    parameters:\n      data: "$results.clean_data"\n      conditions:\n        - field: "amount"\n          operator: "gt"\n          value: 0\n        - field: "product_id"\n          operator: "not_equals"\n          value: null\n\n  # Aggregate by quarter\n  - id: quarterly_summary\n    action: aggregate_data\n    parameters:\n      data: "$results.filter_valid"\n      group_by: ["quarter", "product_id"]\n      aggregations:\n        total_sales:\n          column: "amount"\n          function: "sum"\n        avg_sale:\n          column: "amount"\n          function: "mean"\n        num_transactions:\n          column: "*"\n          function: "count"\n\n  # Save results\n  - id: save_summary\n    action: convert_format\n    parameters:\n      data: "$results.quarterly_summary"\n      to_format: "excel"\n      options:\n        sheet_name: "Quarterly Sales"\n        index: false'
    
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


def test_tool_reference_lines_598_692_2():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 598-692."""
    # Description: **Parameters**:
    import yaml
    
    content = '# Validate against schema\n- id: validate_structure\n  action: validate_schema\n  parameters:\n    data: "$results.processed_data"          # Required: Data to validate\n    schema:                                  # Required: Validation schema\n      type: "object"\n      required: ["id", "name", "email"]\n      properties:\n        id:\n          type: "integer"\n          minimum: 1\n        name:\n          type: "string"\n          minLength: 2\n          maxLength: 100\n        email:\n          type: "string"\n          format: "email"\n        age:\n          type: "integer"\n          minimum: 0\n          maximum: 150\n    strict: false                            # Optional: Strict mode (default: false)\n\n# Business rule validation\n- id: validate_rules\n  action: validate_data\n  parameters:\n    data: "$results.transactions"            # Required: Data to validate\n    rules:                                   # Required: Validation rules\n      - name: "positive_amounts"\n        field: "amount"\n        condition: "value > 0"\n        severity: "error"                    # error|warning|info\n        message: "Transaction amounts must be positive"\n\n      - name: "valid_date_range"\n        field: "transaction_date"\n        condition: "value >= \'2024-01-01\' and value <= today()"\n        severity: "error"\n\n      - name: "customer_exists"\n        field: "customer_id"\n        condition: "value in valid_customers"\n        severity: "warning"\n        context:\n          valid_customers: "$results.customer_list"\n\n    stop_on_error: false                     # Optional: Stop on first error (default: false)\n\n# Data quality checks\n- id: quality_check\n  action: check_quality\n  parameters:\n    data: "$results.dataset"                 # Required: Data to check\n    checks:                                  # Required: Quality checks\n      - type: "completeness"\n        threshold: 0.95                      # 95% non-null required\n        columns: ["id", "name", "email"]\n\n      - type: "uniqueness"\n        columns: ["id", "email"]\n\n      - type: "consistency"\n        rules:\n          - "start_date <= end_date"\n          - "total == sum(line_items)"\n\n      - type: "accuracy"\n        validations:\n          email: "regex:^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\\\.[a-zA-Z]{2,}$"\n          phone: "regex:^\\\\+?1?\\\\d{9,15}$"\n\n      - type: "timeliness"\n        field: "last_updated"\n        max_age_days: 30\n\n# Report validation\n- id: validate_report\n  action: validate_report\n  parameters:\n    report: "$results.generated_report"      # Required: Report to validate\n    checks:                                  # Required: Report checks\n      - "completeness"                       # All sections present\n      - "accuracy"                           # Facts are accurate\n      - "consistency"                        # No contradictions\n      - "readability"                        # Appropriate reading level\n      - "citations"                          # Sources properly cited\n    requirements:                            # Optional: Specific requirements\n      min_words: 1000\n      max_words: 5000\n      required_sections: ["intro", "analysis", "conclusion"]\n      citation_style: "APA"'
    
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


def test_tool_reference_lines_697_784_3():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 697-784."""
    # Description: **Example Pipeline**:
    import yaml
    
    content = 'name: data-quality-pipeline\ndescription: Comprehensive data validation and quality assurance\n\nsteps:\n  # Load data\n  - id: load\n    action: read_file\n    parameters:\n      path: "{{ inputs.data_file }}"\n      parse: true\n\n  # Schema validation\n  - id: validate_schema\n    action: validate_schema\n    parameters:\n      data: "$results.load"\n      schema:\n        type: "array"\n        items:\n          type: "object"\n          required: ["order_id", "customer_id", "amount", "date"]\n          properties:\n            order_id:\n              type: "string"\n              pattern: "^ORD-[0-9]{6}$"\n            customer_id:\n              type: "integer"\n              minimum: 1\n            amount:\n              type: "number"\n              minimum: 0\n            date:\n              type: "string"\n              format: "date"\n\n  # Business rules\n  - id: validate_business\n    action: validate_data\n    parameters:\n      data: "$results.load"\n      rules:\n        - name: "valid_amounts"\n          field: "amount"\n          condition: "value > 0 and value < 10000"\n          severity: "error"\n\n        - name: "recent_orders"\n          field: "date"\n          condition: "days_between(value, today()) <= 365"\n          severity: "warning"\n          message: "Order is older than 1 year"\n\n  # Quality assessment\n  - id: quality_report\n    action: check_quality\n    parameters:\n      data: "$results.load"\n      checks:\n        - type: "completeness"\n          threshold: 0.98\n        - type: "uniqueness"\n          columns: ["order_id"]\n        - type: "consistency"\n          rules:\n            - "item_total == quantity * unit_price"\n        - type: "accuracy"\n          validations:\n            email: "regex:^[\\\\w.-]+@[\\\\w.-]+\\\\.\\\\w+$"\n\n  # Generate validation report\n  - id: create_report\n    action: generate_content\n    parameters:\n      template: |\n        # Data Validation Report\n\n        ## Schema Validation\n        {{ results.validate_schema.summary }}\n\n        ## Business Rules\n        {{ results.validate_business.summary }}\n\n        ## Quality Metrics\n        {{ results.quality_report | format_quality_metrics }}\n\n        ## Recommendations\n        <AUTO>Based on the validation results, provide recommendations</AUTO>'
    
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


def test_tool_reference_lines_804_870_4():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 804-870."""
    # Description: **Parameters**:
    import yaml
    
    content = '# Generate content\n- id: generate\n  action: generate_content\n  parameters:\n    prompt: "{{ inputs.prompt }}"            # Required: Generation prompt\n    model: <AUTO>Select best model</AUTO>    # Optional: Model selection\n    max_tokens: 1000                         # Optional: Maximum tokens\n    temperature: 0.7                         # Optional: Creativity (0-2)\n    system_prompt: "You are a helpful AI"    # Optional: System message\n    format: "markdown"                       # Optional: Output format\n    style: "professional"                    # Optional: Writing style\n\n# Analyze text\n- id: analyze\n  action: analyze_text\n  parameters:\n    text: "$results.document"                # Required: Text to analyze\n    analysis_types:                          # Required: Types of analysis\n      - sentiment                            # Positive/negative/neutral\n      - entities                             # Named entities\n      - topics                               # Main topics\n      - summary                              # Brief summary\n      - key_points                           # Bullet points\n      - language                             # Detect language\n    output_format: "structured"              # Optional: structured|narrative\n\n# Extract information\n- id: extract\n  action: extract_information\n  parameters:\n    content: "$results.raw_text"             # Required: Source content\n    extract:                                 # Required: What to extract\n      dates:\n        description: "All mentioned dates"\n        format: "YYYY-MM-DD"\n      people:\n        description: "Person names with roles"\n        include_context: true\n      organizations:\n        description: "Company and organization names"\n      numbers:\n        description: "Numerical values with units"\n        categories: ["financial", "metrics"]\n    output_format: "json"                    # Optional: json|table|text\n\n# Generate code\n- id: code_gen\n  action: generate_code\n  parameters:\n    description: "{{ inputs.feature_request }}" # Required: What to build\n    language: "python"                       # Required: Programming language\n    framework: "fastapi"                     # Optional: Framework/library\n    include_tests: true                      # Optional: Generate tests\n    include_docs: true                       # Optional: Generate docs\n    style_guide: "PEP8"                     # Optional: Code style\n    example_usage: true                      # Optional: Include examples\n\n# Reasoning task\n- id: reason\n  action: reason_about\n  parameters:\n    question: "{{ inputs.problem }}"         # Required: Problem/question\n    context: "$results.research"             # Optional: Additional context\n    approach: "step_by_step"                 # Optional: Reasoning approach\n    show_work: true                          # Optional: Show reasoning\n    confidence_level: true                   # Optional: Include confidence'
    
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


def test_tool_reference_lines_889_898_5():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 889-898."""
    # Description: **Parameters**:
    import yaml
    
    content = '# Query database\n- id: fetch_data\n  action: query_database\n  parameters:\n    connection: "postgresql://localhost/mydb" # Required: Connection string\n    query: "SELECT * FROM users WHERE active = true" # Required: SQL query\n    parameters: []                           # Optional: Query parameters\n    fetch_size: 1000                         # Optional: Batch size\n    timeout: 30                              # Optional: Query timeout'
    
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


def test_tool_reference_lines_913_927_6():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 913-927."""
    # Description: **Parameters**:
    import yaml
    
    content = '# REST API call\n- id: api_call\n  action: call_api\n  parameters:\n    url: "https://api.example.com/data"     # Required: API endpoint\n    method: "POST"                           # Required: HTTP method\n    headers:                                 # Optional: Headers\n      Authorization: "Bearer {{ env.API_TOKEN }}"\n      Content-Type: "application/json"\n    body:                                    # Optional: Request body\n      query: "{{ inputs.search_term }}"\n      limit: 100\n    timeout: 60                              # Optional: Request timeout\n    retry: 3                                 # Optional: Retry attempts'
    
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


def test_tool_reference_lines_936_1002_7():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 936-1002."""
    # Description: ----------------------------
    import yaml
    
    content = 'name: comprehensive-research-tool-chain\ndescription: Chain multiple tools for research and reporting\n\nsteps:\n  # 1. Search multiple sources\n  - id: web_search\n    action: search_web\n    parameters:\n      query: "{{ inputs.topic }} latest research 2024"\n      max_results: 20\n\n  # 2. Scrape promising articles\n  - id: scrape_articles\n    for_each: "{{ results.web_search.results[:5] }}"\n    as: article\n    action: scrape_page\n    parameters:\n      url: "{{ article.url }}"\n      selectors:\n        content: "article, main, .content"\n\n  # 3. Extract key information\n  - id: extract_facts\n    action: extract_information\n    parameters:\n      content: "$results.scrape_articles"\n      extract:\n        facts:\n          description: "Key facts and findings"\n        statistics:\n          description: "Numerical data with context"\n        quotes:\n          description: "Notable quotes with attribution"\n\n  # 4. Validate information\n  - id: cross_validate\n    action: validate_data\n    parameters:\n      data: "$results.extract_facts"\n      rules:\n        - name: "source_diversity"\n          condition: "count(unique(sources)) >= 3"\n          severity: "warning"\n\n  # 5. Generate report\n  - id: create_report\n    action: generate_content\n    parameters:\n      prompt: |\n        Create a comprehensive report about {{ inputs.topic }}\n        using the following validated information:\n        {{ results.extract_facts | json }}\n      style: "academic"\n      format: "markdown"\n      max_tokens: 2000\n\n  # 6. Save report\n  - id: save_report\n    action: write_file\n    parameters:\n      path: "reports/{{ inputs.topic }}_{{ execution.date }}.md"\n      content: "$results.create_report"\n\n  # 7. Generate PDF\n  - id: create_pdf\n    action: "!pandoc -f markdown -t pdf -o reports/{{ inputs.topic }}.pdf reports/{{ inputs.topic }}_{{ execution.date }}.md"'
    
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


def test_tool_reference_lines_1008_1099_8():
    """Test YAML snippet from docs_sphinx/tool_reference.rst lines 1008-1099."""
    # Description: ------------------------
    import yaml
    
    content = 'name: etl-tool-chain\ndescription: Extract, transform, and load data using tool chain\n\nsteps:\n  # Extract from multiple sources\n  - id: extract_database\n    action: query_database\n    parameters:\n      connection: "{{ env.DB_CONNECTION }}"\n      query: "SELECT * FROM sales WHERE date >= \'2024-01-01\'"\n\n  - id: extract_api\n    action: call_api\n    parameters:\n      url: "https://api.company.com/v2/transactions"\n      method: "GET"\n      headers:\n        Authorization: "Bearer {{ env.API_KEY }}"\n      params:\n        start_date: "2024-01-01"\n        page_size: 1000\n\n  - id: extract_files\n    action: list_directory\n    parameters:\n      path: "data/uploads/"\n      pattern: "sales_*.csv"\n      recursive: true\n\n  # Load file data\n  - id: load_files\n    for_each: "{{ results.extract_files }}"\n    as: file\n    action: read_file\n    parameters:\n      path: "{{ file.path }}"\n      parse: true\n\n  # Transform all data\n  - id: merge_all\n    action: merge_data\n    parameters:\n      datasets:\n        - "$results.extract_database"\n        - "$results.extract_api.data"\n        - "$results.load_files"\n      key: "transaction_id"\n\n  - id: clean_data\n    action: transform_data\n    parameters:\n      data: "$results.merge_all"\n      operations:\n        - type: "remove_duplicates"\n          columns: ["transaction_id"]\n        - type: "fill_missing"\n          strategy: "forward"\n        - type: "standardize_formats"\n          columns:\n            date: "YYYY-MM-DD"\n            amount: "decimal(10,2)"\n\n  # Validate\n  - id: validate_quality\n    action: check_quality\n    parameters:\n      data: "$results.clean_data"\n      checks:\n        - type: "completeness"\n          threshold: 0.99\n        - type: "accuracy"\n          validations:\n            amount: "range:0,1000000"\n            date: "date_range:2024-01-01,today"\n\n  # Load to destination\n  - id: save_processed\n    action: write_file\n    parameters:\n      path: "processed/sales_cleaned_{{ execution.date }}.parquet"\n      content: "$results.clean_data"\n      format: "parquet"\n\n  - id: update_database\n    condition: "{{ results.validate_quality.passed }}"\n    action: insert_data\n    parameters:\n      connection: "{{ env.DW_CONNECTION }}"\n      table: "sales_fact"\n      data: "$results.clean_data"\n      mode: "append"'
    
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


def test_tool_reference_lines_1110_1152_9():
    """Test Python snippet from docs_sphinx/tool_reference.rst lines 1110-1152."""
    # Description: To create your own tools:
    content = 'from orchestrator.tools.base import Tool\n\nclass MyCustomTool(Tool):\n    def __init__(self):\n        super().__init__(\n            name="my-custom-tool",\n            description="Does something special"\n        )\n\n        # Define parameters\n        self.add_parameter(\n            name="input_data",\n            type="string",\n            description="Data to process",\n            required=True\n        )\n\n        self.add_parameter(\n            name="mode",\n            type="string",\n            description="Processing mode",\n            required=False,\n            default="standard",\n            enum=["standard", "advanced", "expert"]\n        )\n\n    async def execute(self, **kwargs):\n        """Execute the tool action."""\n        input_data = kwargs["input_data"]\n        mode = kwargs.get("mode", "standard")\n\n        # Your tool logic here\n        result = process_data(input_data, mode)\n\n        return {\n            "status": "success",\n            "result": result,\n            "metadata": {\n                "mode": mode,\n                "timestamp": datetime.now()\n            }\n        }'
    
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
