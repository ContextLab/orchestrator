"""Working tests for documentation code snippets - Batch 29."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# These tests focus on validation and syntax checking without execution
# They should run quickly and not require external dependencies


def test_tool_reference_lines_1160_1175_0():
    """Test Python snippet from docs_sphinx/tool_reference.rst lines 1160-1175."""
    # Description: Register your tool to make it available:
    content = 'from orchestrator.tools.base import default_registry\n\n# Register tool\ntool = MyCustomTool()\ndefault_registry.register(tool)\n\n# Use in pipeline\npipeline_yaml = """\nsteps:\n  - id: custom_step\n    action: my-custom-tool\n    parameters:\n      input_data: "{{ inputs.data }}"\n      mode: "advanced"\n"""'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid Python syntax
    # Skip syntax check for notebook-specific code with top-level await
    if 'await' in content and ('notebook' in content.lower() or 'jupyter' in content.lower()):
        # This is notebook-specific syntax, skip syntax validation
        pass
    else:
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


def test_tutorial_data_processing_lines_35_239_1():
    """Test YAML snippet from docs_sphinx/tutorials/tutorial_data_processing.rst lines 35-239."""
    # Description: Create ``sales_etl.yaml``:
    import yaml
    
    content = 'name: sales-etl-pipeline\ndescription: Extract, transform, and load sales data\n\ninputs:\n  data_source:\n    type: string\n    description: "Path to source data file"\n    required: true\n\n  output_format:\n    type: string\n    description: "Output format"\n    default: "parquet"\n    validation:\n      enum: ["csv", "json", "parquet", "excel"]\n\n  date_range:\n    type: object\n    description: "Date range for filtering"\n    default:\n      start: "2024-01-01"\n      end: "2024-12-31"\n\noutputs:\n  processed_data:\n    type: string\n    value: "processed/sales_{{ execution.date }}.{{ inputs.output_format }}"\n\n  quality_report:\n    type: string\n    value: "reports/quality_{{ execution.date }}.json"\n\n  summary_stats:\n    type: string\n    value: "reports/summary_{{ execution.date }}.md"\n\nsteps:\n  # Extract: Load raw data\n  - id: extract_data\n    action: read_file\n    parameters:\n      path: "{{ inputs.data_source }}"\n      parse: true\n    error_handling:\n      retry:\n        max_attempts: 3\n      fallback:\n        action: generate_content\n        parameters:\n          prompt: "Generate sample sales data for testing"\n\n  # Transform: Clean and process data\n  - id: clean_data\n    action: transform_data\n    parameters:\n      data: "$results.extract_data"\n      operations:\n        # Standardize column names\n        - type: "rename_columns"\n          mapping:\n            "Sale Date": "sale_date"\n            "Customer Name": "customer_name"\n            "Product ID": "product_id"\n            "Sale Amount": "amount"\n            "Quantity": "quantity"\n            "Sales Rep": "sales_rep"\n\n        # Convert data types\n        - type: "convert_types"\n          conversions:\n            sale_date: "datetime"\n            amount: "float"\n            quantity: "integer"\n            product_id: "string"\n\n        # Remove duplicates\n        - type: "remove_duplicates"\n          columns: ["product_id", "sale_date", "customer_name"]\n\n        # Handle missing values\n        - type: "fill_missing"\n          strategy: "forward"\n          columns: ["sales_rep"]\n\n        # Add calculated fields\n        - type: "add_column"\n          name: "total_value"\n          expression: "amount * quantity"\n\n        - type: "add_column"\n          name: "quarter"\n          expression: "quarter(sale_date)"\n\n        - type: "add_column"\n          name: "year"\n          expression: "year(sale_date)"\n\n  # Filter data by date range\n  - id: filter_data\n    action: filter_data\n    parameters:\n      data: "$results.clean_data"\n      conditions:\n        - field: "sale_date"\n          operator: "gte"\n          value: "{{ inputs.date_range.start }}"\n        - field: "sale_date"\n          operator: "lte"\n          value: "{{ inputs.date_range.end }}"\n        - field: "amount"\n          operator: "gt"\n          value: 0\n\n  # Data quality validation\n  - id: validate_quality\n    action: check_quality\n    parameters:\n      data: "$results.filter_data"\n      checks:\n        - type: "completeness"\n          threshold: 0.95\n          columns: ["product_id", "amount", "sale_date"]\n\n        - type: "uniqueness"\n          columns: ["product_id", "sale_date", "customer_name"]\n\n        - type: "consistency"\n          rules:\n            - "total_value == amount * quantity"\n            - "amount > 0"\n            - "quantity > 0"\n\n        - type: "accuracy"\n          validations:\n            product_id: "regex:^PROD-[0-9]{6}$"\n            amount: "range:1,50000"\n            quantity: "range:1,1000"\n\n  # Generate summary statistics\n  - id: calculate_summary\n    action: aggregate_data\n    parameters:\n      data: "$results.filter_data"\n      group_by: ["year", "quarter"]\n      aggregations:\n        total_sales:\n          column: "total_value"\n          function: "sum"\n        avg_sale:\n          column: "amount"\n          function: "mean"\n        num_transactions:\n          column: "*"\n          function: "count"\n        unique_customers:\n          column: "customer_name"\n          function: "nunique"\n        top_product:\n          column: "product_id"\n          function: "mode"\n\n  # Load: Save processed data\n  - id: save_processed_data\n    action: convert_format\n    parameters:\n      data: "$results.filter_data"\n      to_format: "{{ inputs.output_format }}"\n      output_path: "{{ outputs.processed_data }}"\n      options:\n        compression: "snappy"\n        index: false\n\n  # Save quality report\n  - id: save_quality_report\n    action: write_file\n    parameters:\n      path: "{{ outputs.quality_report }}"\n      content: "{{ results.validate_quality | json }}"\n\n  # Generate readable summary\n  - id: create_summary_report\n    action: generate_content\n    parameters:\n      prompt: |\n        Create a summary report for sales data processing:\n\n        Quality Results: {{ results.validate_quality | json }}\n        Summary Statistics: {{ results.calculate_summary | json }}\n\n        Include:\n        - Data quality assessment\n        - Key metrics and trends\n        - Any issues or recommendations\n        - Processing summary\n\n      style: "professional"\n      format: "markdown"\n\n  # Save summary report\n  - id: save_summary\n    action: write_file\n    parameters:\n      path: "{{ outputs.summary_stats }}"\n      content: "$results.create_summary_report"'
    
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
        # Note: data can be None for YAML with only comments
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                # Note: 'id' is optional in minimal examples


def test_tutorial_data_processing_lines_245_264_2():
    """Test Python snippet from docs_sphinx/tutorials/tutorial_data_processing.rst lines 245-264."""
    # Description: ----------------------------
    content = 'import orchestrator as orc\n\n# Initialize\norc.init_models()\n\n# Compile pipeline\netl_pipeline = orc.compile("sales_etl.yaml")\n\n# Process sales data\nresult = etl_pipeline.run(\n    data_source="data/raw/sales_2024.csv",\n    output_format="parquet",\n    date_range={\n        "start": "2024-01-01",\n        "end": "2024-06-30"\n    }\n)\n\nprint(f"ETL completed: {result}")'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid Python syntax
    # Skip syntax check for notebook-specific code with top-level await
    if 'await' in content and ('notebook' in content.lower() or 'jupyter' in content.lower()):
        # This is notebook-specific syntax, skip syntax validation
        pass
    else:
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


def test_tutorial_data_processing_lines_277_521_3():
    """Test YAML snippet from docs_sphinx/tutorials/tutorial_data_processing.rst lines 277-521."""
    # Description: Create ``data_integration.yaml``:
    import yaml
    
    content = 'name: multi-source-integration\ndescription: Integrate data from multiple sources with validation\n\ninputs:\n  sources:\n    type: object\n    description: "Data source configurations"\n    required: true\n    # Example:\n    # database:\n    #   type: "postgresql"\n    #   connection: "postgresql://..."\n    #   query: "SELECT * FROM sales"\n    # api:\n    #   type: "rest"\n    #   url: "https://api.company.com/data"\n    #   headers: {...}\n    # files:\n    #   type: "file"\n    #   paths: ["data1.csv", "data2.json"]\n\n  merge_strategy:\n    type: string\n    description: "How to merge data sources"\n    default: "outer"\n    validation:\n      enum: ["inner", "outer", "left", "right"]\n\n  deduplication_fields:\n    type: array\n    description: "Fields to use for deduplication"\n    default: ["id", "timestamp"]\n\noutputs:\n  integrated_data:\n    type: string\n    value: "integrated/master_data_{{ execution.timestamp }}.parquet"\n\n  integration_report:\n    type: string\n    value: "reports/integration_{{ execution.timestamp }}.md"\n\nsteps:\n  # Extract from database sources\n  - id: extract_database\n    condition: "\'database\' in inputs.sources"\n    action: query_database\n    parameters:\n      connection: "{{ inputs.sources.database.connection }}"\n      query: "{{ inputs.sources.database.query }}"\n      fetch_size: 10000\n    error_handling:\n      continue_on_error: true\n\n  # Extract from API sources\n  - id: extract_api\n    condition: "\'api\' in inputs.sources"\n    action: call_api\n    parameters:\n      url: "{{ inputs.sources.api.url }}"\n      method: "GET"\n      headers: "{{ inputs.sources.api.headers | default({}) }}"\n      params: "{{ inputs.sources.api.params | default({}) }}"\n      timeout: 300\n    error_handling:\n      retry:\n        max_attempts: 3\n        backoff: "exponential"\n\n  # Extract from file sources\n  - id: extract_files\n    condition: "\'files\' in inputs.sources"\n    for_each: "{{ inputs.sources.files.paths }}"\n    as: file_path\n    action: read_file\n    parameters:\n      path: "{{ file_path }}"\n      parse: true\n\n  # Standardize data schemas\n  - id: standardize_database\n    condition: "results.extract_database is defined"\n    action: transform_data\n    parameters:\n      data: "$results.extract_database"\n      operations:\n        - type: "add_column"\n          name: "source"\n          value: "database"\n        - type: "standardize_schema"\n          target_schema:\n            id: "string"\n            timestamp: "datetime"\n            value: "float"\n            category: "string"\n\n  - id: standardize_api\n    condition: "results.extract_api is defined"\n    action: transform_data\n    parameters:\n      data: "$results.extract_api.data"\n      operations:\n        - type: "add_column"\n          name: "source"\n          value: "api"\n        - type: "flatten_nested"\n          columns: ["metadata", "attributes"]\n        - type: "standardize_schema"\n          target_schema:\n            id: "string"\n            timestamp: "datetime"\n            value: "float"\n            category: "string"\n\n  - id: standardize_files\n    condition: "results.extract_files is defined"\n    action: transform_data\n    parameters:\n      data: "$results.extract_files"\n      operations:\n        - type: "add_column"\n          name: "source"\n          value: "files"\n        - type: "combine_files"\n          strategy: "union"\n        - type: "standardize_schema"\n          target_schema:\n            id: "string"\n            timestamp: "datetime"\n            value: "float"\n            category: "string"\n\n  # Merge all data sources\n  - id: merge_sources\n    action: merge_data\n    parameters:\n      datasets:\n        - "$results.standardize_database"\n        - "$results.standardize_api"\n        - "$results.standardize_files"\n      how: "{{ inputs.merge_strategy }}"\n      on: ["id"]\n      suffixes: ["_db", "_api", "_file"]\n\n  # Remove duplicates\n  - id: deduplicate\n    action: transform_data\n    parameters:\n      data: "$results.merge_sources"\n      operations:\n        - type: "remove_duplicates"\n          columns: "{{ inputs.deduplication_fields }}"\n          keep: "last"  # Keep most recent\n\n  # Data quality assessment\n  - id: assess_integration_quality\n    action: check_quality\n    parameters:\n      data: "$results.deduplicate"\n      checks:\n        - type: "completeness"\n          threshold: 0.90\n          critical_columns: ["id", "timestamp"]\n\n        - type: "consistency"\n          rules:\n            - "value_db == value_api OR value_db IS NULL OR value_api IS NULL"\n            - "timestamp >= \'2020-01-01\'"\n\n        - type: "accuracy"\n          validations:\n            id: "not_null"\n            timestamp: "datetime_format"\n            value: "numeric_range:-1000000,1000000"\n\n  # Resolve conflicts between sources\n  - id: resolve_conflicts\n    action: transform_data\n    parameters:\n      data: "$results.deduplicate"\n      operations:\n        - type: "resolve_conflicts"\n          strategy: "priority"\n          priority_order: ["database", "api", "files"]\n          conflict_columns: ["value", "category"]\n\n        - type: "add_column"\n          name: "confidence_score"\n          expression: "calculate_confidence(source_count, data_age, validation_status)"\n\n  # Create final integrated dataset\n  - id: finalize_integration\n    action: transform_data\n    parameters:\n      data: "$results.resolve_conflicts"\n      operations:\n        - type: "select_columns"\n          columns: ["id", "timestamp", "value", "category", "source", "confidence_score"]\n\n        - type: "sort"\n          columns: ["timestamp"]\n          ascending: [false]\n\n  # Save integrated data\n  - id: save_integrated\n    action: convert_format\n    parameters:\n      data: "$results.finalize_integration"\n      to_format: "parquet"\n      output_path: "{{ outputs.integrated_data }}"\n      options:\n        compression: "snappy"\n        partition_cols: ["category"]\n\n  # Generate integration report\n  - id: create_integration_report\n    action: generate_content\n    parameters:\n      prompt: |\n        Create an integration report for multi-source data merge:\n\n        Sources processed:\n        {% for source in inputs.sources.keys() %}\n        - {{ source }}\n        {% endfor %}\n\n        Quality assessment: {{ results.assess_integration_quality | json }}\n        Final record count: {{ results.finalize_integration | length }}\n\n        Include:\n        - Source summary and statistics\n        - Data quality metrics\n        - Conflict resolution summary\n        - Recommendations for data improvement\n\n      style: "technical"\n      format: "markdown"\n\n  # Save integration report\n  - id: save_report\n    action: write_file\n    parameters:\n      path: "{{ outputs.integration_report }}"\n      content: "$results.create_integration_report"'
    
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
        # Note: data can be None for YAML with only comments
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                # Note: 'id' is optional in minimal examples


def test_tutorial_data_processing_lines_527_558_4():
    """Test Python snippet from docs_sphinx/tutorials/tutorial_data_processing.rst lines 527-558."""
    # Description: -----------------------------------
    content = 'import orchestrator as orc\n\n# Initialize\norc.init_models()\n\n# Compile integration pipeline\nintegration = orc.compile("data_integration.yaml")\n\n# Integrate data from multiple sources\nresult = integration.run(\n    sources={\n        "database": {\n            "type": "postgresql",\n            "connection": "postgresql://user:pass@localhost/mydb",\n            "query": "SELECT * FROM transactions WHERE date >= \'2024-01-01\'"\n        },\n        "api": {\n            "type": "rest",\n            "url": "https://api.external.com/v1/data",\n            "headers": {"Authorization": "Bearer token123"}\n        },\n        "files": {\n            "type": "file",\n            "paths": ["data/file1.csv", "data/file2.json"]\n        }\n    },\n    merge_strategy="outer",\n    deduplication_fields=["transaction_id", "timestamp"]\n)\n\nprint(f"Integration completed: {result}")'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid Python syntax
    # Skip syntax check for notebook-specific code with top-level await
    if 'await' in content and ('notebook' in content.lower() or 'jupyter' in content.lower()):
        # This is notebook-specific syntax, skip syntax validation
        pass
    else:
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


def test_tutorial_data_processing_lines_571_815_5():
    """Test YAML snippet from docs_sphinx/tutorials/tutorial_data_processing.rst lines 571-815."""
    # Description: Create ``data_quality.yaml``:
    import yaml
    
    content = 'name: data-quality-assessment\ndescription: Comprehensive data quality evaluation and reporting\n\ninputs:\n  dataset_path:\n    type: string\n    required: true\n\n  quality_rules:\n    type: object\n    description: "Custom quality rules"\n    default:\n      completeness_threshold: 0.95\n      uniqueness_fields: ["id"]\n      date_range_field: "created_date"\n      numeric_fields: ["amount", "quantity"]\n\n  remediation_mode:\n    type: string\n    description: "How to handle quality issues"\n    default: "report"\n    validation:\n      enum: ["report", "fix", "quarantine"]\n\noutputs:\n  quality_report:\n    type: string\n    value: "quality/report_{{ execution.timestamp }}.html"\n\n  cleaned_data:\n    type: string\n    value: "quality/cleaned_{{ execution.timestamp }}.parquet"\n\n  issues_log:\n    type: string\n    value: "quality/issues_{{ execution.timestamp }}.json"\n\nsteps:\n  # Load the dataset\n  - id: load_dataset\n    action: read_file\n    parameters:\n      path: "{{ inputs.dataset_path }}"\n      parse: true\n\n  # Basic data profiling\n  - id: profile_data\n    action: analyze_data\n    parameters:\n      data: "$results.load_dataset"\n      analysis_types:\n        - schema\n        - statistics\n        - distributions\n        - patterns\n        - outliers\n\n  # Completeness assessment\n  - id: check_completeness\n    action: check_quality\n    parameters:\n      data: "$results.load_dataset"\n      checks:\n        - type: "completeness"\n          threshold: "{{ inputs.quality_rules.completeness_threshold }}"\n          report_by_column: true\n\n        - type: "null_patterns"\n          identify_patterns: true\n\n  # Uniqueness validation\n  - id: check_uniqueness\n    action: validate_data\n    parameters:\n      data: "$results.load_dataset"\n      rules:\n        - name: "primary_key_uniqueness"\n          type: "uniqueness"\n          columns: "{{ inputs.quality_rules.uniqueness_fields }}"\n          severity: "error"\n\n        - name: "near_duplicates"\n          type: "similarity"\n          threshold: 0.9\n          columns: ["name", "email"]\n          severity: "warning"\n\n  # Consistency validation\n  - id: check_consistency\n    action: validate_data\n    parameters:\n      data: "$results.load_dataset"\n      rules:\n        - name: "date_logic"\n          condition: "start_date <= end_date"\n          severity: "error"\n\n        - name: "numeric_consistency"\n          condition: "total == sum(line_items)"\n          severity: "error"\n\n        - name: "referential_integrity"\n          type: "foreign_key"\n          reference_table: "lookup_table"\n          foreign_key: "category_id"\n          severity: "warning"\n\n  # Accuracy validation\n  - id: check_accuracy\n    action: validate_data\n    parameters:\n      data: "$results.load_dataset"\n      rules:\n        - name: "email_format"\n          field: "email"\n          validation: "regex:^[\\\\w.-]+@[\\\\w.-]+\\\\.\\\\w+$"\n          severity: "warning"\n\n        - name: "phone_format"\n          field: "phone"\n          validation: "regex:^\\\\+?1?\\\\d{9,15}$"\n          severity: "info"\n\n        - name: "numeric_ranges"\n          field: "{{ inputs.quality_rules.numeric_fields }}"\n          validation: "range:0,999999"\n          severity: "error"\n\n  # Timeliness assessment\n  - id: check_timeliness\n    action: validate_data\n    parameters:\n      data: "$results.load_dataset"\n      rules:\n        - name: "data_freshness"\n          field: "{{ inputs.quality_rules.date_range_field }}"\n          condition: "date_diff(value, today()) <= 30"\n          severity: "warning"\n          message: "Data is older than 30 days"\n\n  # Outlier detection\n  - id: detect_outliers\n    action: analyze_data\n    parameters:\n      data: "$results.load_dataset"\n      analysis_types:\n        - outliers\n      methods:\n        - statistical  # Z-score, IQR\n        - isolation_forest\n        - local_outlier_factor\n      numeric_columns: "{{ inputs.quality_rules.numeric_fields }}"\n\n  # Compile quality issues\n  - id: compile_issues\n    action: transform_data\n    parameters:\n      data:\n        completeness: "$results.check_completeness"\n        uniqueness: "$results.check_uniqueness"\n        consistency: "$results.check_consistency"\n        accuracy: "$results.check_accuracy"\n        timeliness: "$results.check_timeliness"\n        outliers: "$results.detect_outliers"\n      operations:\n        - type: "consolidate_issues"\n          prioritize: true\n        - type: "categorize_severity"\n          levels: ["critical", "major", "minor", "info"]\n\n  # Data remediation (if requested)\n  - id: remediate_data\n    condition: "inputs.remediation_mode in [\'fix\', \'quarantine\']"\n    action: transform_data\n    parameters:\n      data: "$results.load_dataset"\n      operations:\n        # Fix common issues\n        - type: "standardize_formats"\n          columns:\n            email: "lowercase"\n            phone: "normalize_phone"\n            name: "title_case"\n\n        - type: "fill_missing"\n          strategy: "smart"  # Use ML-based imputation\n          columns: "{{ inputs.quality_rules.numeric_fields }}"\n\n        - type: "remove_outliers"\n          method: "iqr"\n          columns: "{{ inputs.quality_rules.numeric_fields }}"\n          action: "{{ \'quarantine\' if inputs.remediation_mode == \'quarantine\' else \'remove\' }}"\n\n        - type: "deduplicate"\n          strategy: "keep_best"  # Keep record with highest completeness\n\n  # Generate comprehensive quality report\n  - id: create_quality_report\n    action: generate_content\n    parameters:\n      prompt: |\n        Create a comprehensive data quality report:\n\n        Dataset: {{ inputs.dataset_path }}\n        Profile: {{ results.profile_data | json }}\n        Issues: {{ results.compile_issues | json }}\n\n        Include:\n        1. Executive Summary\n        2. Data Profile Overview\n        3. Quality Metrics Dashboard\n        4. Issue Analysis by Category\n        5. Impact Assessment\n        6. Remediation Recommendations\n        7. Quality Score Calculation\n\n        Format as HTML with charts and tables.\n\n      style: "technical"\n      format: "html"\n      max_tokens: 3000\n\n  # Save quality report\n  - id: save_quality_report\n    action: write_file\n    parameters:\n      path: "{{ outputs.quality_report }}"\n      content: "$results.create_quality_report"\n\n  # Save cleaned data (if remediation performed)\n  - id: save_cleaned_data\n    condition: "inputs.remediation_mode in [\'fix\', \'quarantine\']"\n    action: write_file\n    parameters:\n      path: "{{ outputs.cleaned_data }}"\n      content: "$results.remediate_data"\n      format: "parquet"\n\n  # Save issues log\n  - id: save_issues_log\n    action: write_file\n    parameters:\n      path: "{{ outputs.issues_log }}"\n      content: "{{ results.compile_issues | json }}"'
    
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
        # Note: data can be None for YAML with only comments
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                # Note: 'id' is optional in minimal examples


def test_tutorial_data_processing_lines_828_923_6():
    """Test YAML snippet from docs_sphinx/tutorials/tutorial_data_processing.rst lines 828-923."""
    # Description: Create ``realtime_processing.yaml``:
    import yaml
    
    content = 'name: realtime-data-processing\ndescription: Process streaming data with real-time analytics\n\ninputs:\n  stream_source:\n    type: object\n    description: "Stream configuration"\n    required: true\n    # Example:\n    # type: "kafka"\n    # topic: "events"\n    # batch_size: 1000\n    # window_size: "5m"\n\n  processing_rules:\n    type: array\n    description: "Processing rules to apply"\n    default:\n      - type: "filter"\n        condition: "event_type in [\'purchase\', \'click\']"\n      - type: "enrich"\n        lookup_table: "user_profiles"\n      - type: "aggregate"\n        window: "5m"\n        metrics: ["count", "sum", "avg"]\n\noutputs:\n  processed_stream:\n    type: string\n    value: "stream/processed_{{ execution.date }}"\n\n  alerts:\n    type: string\n    value: "alerts/stream_alerts_{{ execution.timestamp }}.json"\n\nsteps:\n  # Connect to stream source\n  - id: connect_stream\n    action: connect_stream\n    parameters:\n      source: "{{ inputs.stream_source }}"\n      batch_size: "{{ inputs.stream_source.batch_size | default(1000) }}"\n      timeout: 30\n\n  # Process incoming batches\n  - id: process_batches\n    action: process_stream_batch\n    parameters:\n      stream: "$results.connect_stream"\n      processing_rules: "{{ inputs.processing_rules }}"\n      window_config:\n        size: "{{ inputs.stream_source.window_size | default(\'5m\') }}"\n        type: "tumbling"  # or "sliding", "session"\n\n  # Real-time anomaly detection\n  - id: detect_anomalies\n    action: detect_anomalies\n    parameters:\n      data: "$results.process_batches"\n      methods:\n        - statistical_control\n        - machine_learning\n      thresholds:\n        statistical: 3.0  # standard deviations\n        ml_confidence: 0.95\n\n  # Generate alerts\n  - id: generate_alerts\n    condition: "results.detect_anomalies.anomalies | length > 0"\n    action: generate_content\n    parameters:\n      prompt: |\n        Generate alerts for detected anomalies:\n        {{ results.detect_anomalies.anomalies | json }}\n\n        Include severity, description, and recommended actions.\n\n      format: "json"\n\n  # Save processed data\n  - id: save_processed\n    action: write_stream\n    parameters:\n      data: "$results.process_batches"\n      destination: "{{ outputs.processed_stream }}"\n      format: "parquet"\n      partition_by: ["date", "hour"]\n\n  # Save alerts\n  - id: save_alerts\n    condition: "results.generate_alerts is defined"\n    action: write_file\n    parameters:\n      path: "{{ outputs.alerts }}"\n      content: "$results.generate_alerts"'
    
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
        # Note: data can be None for YAML with only comments
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                # Note: 'id' is optional in minimal examples


def test_tutorial_data_processing_lines_932_990_7():
    """Test YAML snippet from docs_sphinx/tutorials/tutorial_data_processing.rst lines 932-990."""
    # Description: ---------------------------------
    import yaml
    
    content = 'name: customer-data-platform\ndescription: Unified customer data processing and analytics\n\ninputs:\n  customer_sources:\n    type: object\n    required: true\n    # CRM, support tickets, web analytics, purchase history\n\nsteps:\n  # Extract from all customer touchpoints\n  - id: extract_crm\n    action: query_database\n    parameters:\n      connection: "{{ inputs.customer_sources.crm.connection }}"\n      query: "SELECT * FROM customers WHERE updated_at >= CURRENT_DATE - INTERVAL \'1 day\'"\n\n  - id: extract_support\n    action: call_api\n    parameters:\n      url: "{{ inputs.customer_sources.support.api_url }}"\n      headers:\n        Authorization: "Bearer {{ env.SUPPORT_API_KEY }}"\n\n  - id: extract_analytics\n    action: read_file\n    parameters:\n      path: "{{ inputs.customer_sources.analytics.export_path }}"\n      parse: true\n\n  # Create unified customer profiles\n  - id: merge_customer_data\n    action: merge_data\n    parameters:\n      datasets:\n        - "$results.extract_crm"\n        - "$results.extract_support"\n        - "$results.extract_analytics"\n      on: "customer_id"\n      how: "outer"\n\n  # Calculate customer metrics\n  - id: calculate_metrics\n    action: transform_data\n    parameters:\n      data: "$results.merge_customer_data"\n      operations:\n        - type: "add_column"\n          name: "customer_lifetime_value"\n          expression: "sum(purchase_amounts) * retention_probability"\n\n        - type: "add_column"\n          name: "churn_risk_score"\n          expression: "calculate_churn_risk(days_since_last_activity, support_tickets, engagement_score)"\n\n        - type: "add_column"\n          name: "segment"\n          expression: "classify_customer_segment(clv, engagement, recency)"'
    
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
        # Note: data can be None for YAML with only comments
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                # Note: 'id' is optional in minimal examples


def test_tutorial_data_processing_lines_996_1062_8():
    """Test YAML snippet from docs_sphinx/tutorials/tutorial_data_processing.rst lines 996-1062."""
    # Description: ----------------------------------
    import yaml
    
    content = 'name: financial-data-pipeline\ndescription: Process financial transactions with compliance checks\n\ninputs:\n  transaction_sources:\n    type: array\n    required: true\n\n  compliance_rules:\n    type: object\n    required: true\n\nsteps:\n  # Extract transactions from multiple sources\n  - id: extract_transactions\n    for_each: "{{ inputs.transaction_sources }}"\n    as: source\n    action: extract_financial_data\n    parameters:\n      source_config: "{{ source }}"\n      date_range: "{{ execution.date | date_range(\'-1d\') }}"\n\n  # Compliance screening\n  - id: screen_transactions\n    action: validate_data\n    parameters:\n      data: "$results.extract_transactions"\n      rules:\n        - name: "aml_screening"\n          type: "anti_money_laundering"\n          threshold: "{{ inputs.compliance_rules.aml_threshold }}"\n\n        - name: "sanctions_check"\n          type: "sanctions_screening"\n          watchlists: "{{ inputs.compliance_rules.watchlists }}"\n\n        - name: "pep_screening"\n          type: "politically_exposed_person"\n          databases: "{{ inputs.compliance_rules.pep_databases }}"\n\n  # Risk scoring\n  - id: calculate_risk_scores\n    action: transform_data\n    parameters:\n      data: "$results.extract_transactions"\n      operations:\n        - type: "add_column"\n          name: "risk_score"\n          expression: "calculate_transaction_risk(amount, counterparty, geography, transaction_type)"\n\n        - type: "add_column"\n          name: "risk_category"\n          expression: "categorize_risk(risk_score)"\n\n  # Generate compliance report\n  - id: create_compliance_report\n    action: generate_content\n    parameters:\n      prompt: |\n        Generate daily compliance report:\n\n        Transactions processed: {{ results.extract_transactions | length }}\n        Screening results: {{ results.screen_transactions | json }}\n        Risk distribution: {{ results.calculate_risk_scores | group_by(\'risk_category\') }}\n\n        Include regulatory compliance status and any required actions.'
    
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
        # Note: data can be None for YAML with only comments
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                # Note: 'id' is optional in minimal examples


def test_tutorial_data_processing_lines_1073_1078_9():
    """Test text snippet from docs_sphinx/tutorials/tutorial_data_processing.rst lines 1073-1078."""
    # Description: Build a pipeline that processes e-commerce data:
    content = 'Your challenge:\n- Extract: Orders, customers, products, reviews\n- Transform: Calculate metrics, segment customers\n- Load: Create analytics-ready datasets\n- Quality: Validate business rules'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    assert len(content) > 0, "Content should have length"
