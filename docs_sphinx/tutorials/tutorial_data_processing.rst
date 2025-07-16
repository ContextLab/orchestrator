=======================
Data Processing Pipelines
=======================

This tutorial teaches you to build robust data processing pipelines that can handle ETL (Extract, Transform, Load) operations, data validation, and complex transformations.

What You'll Build
=================

1. **Basic ETL Pipeline** - Extract, transform, and load data
2. **Multi-Source Data Integration** - Combine data from various sources
3. **Data Quality Assessment** - Validate and clean data automatically
4. **Real-Time Data Processing** - Handle streaming data scenarios
5. **Data Pipeline Orchestration** - Coordinate complex data workflows

Prerequisites
=============

- Completed :doc:`tutorial_basics`
- Basic understanding of data formats (JSON, CSV, SQL)
- Familiarity with data concepts

Tutorial 1: Basic ETL Pipeline
==============================

Let's start with a fundamental ETL pipeline that processes sales data.

Step 1: Create the ETL Pipeline
-------------------------------

Create ``sales_etl.yaml``:

.. code-block:: yaml

   name: sales-etl-pipeline
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
         content: "$results.create_summary_report"

Step 2: Run the ETL Pipeline
----------------------------

.. code-block:: python

   import orchestrator as orc
   
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
   
   print(f"ETL completed: {result}")

Tutorial 2: Multi-Source Data Integration
=========================================

Now let's build a pipeline that integrates data from multiple sources.

Step 1: Multi-Source Integration Pipeline
-----------------------------------------

Create ``data_integration.yaml``:

.. code-block:: yaml

   name: multi-source-integration
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
         content: "$results.create_integration_report"

Step 2: Run Multi-Source Integration
-----------------------------------

.. code-block:: python

   import orchestrator as orc
   
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
   
   print(f"Integration completed: {result}")

Tutorial 3: Data Quality Assessment Pipeline
============================================

Create a comprehensive data quality assessment system.

Step 1: Data Quality Pipeline
----------------------------

Create ``data_quality.yaml``:

.. code-block:: yaml

   name: data-quality-assessment
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
         content: "{{ results.compile_issues | json }}"

Tutorial 4: Real-Time Data Processing
=====================================

Build a pipeline for handling streaming data scenarios.

Step 1: Real-Time Processing Pipeline
------------------------------------

Create ``realtime_processing.yaml``:

.. code-block:: yaml

   name: realtime-data-processing
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
         content: "$results.generate_alerts"

Advanced Examples
================

Example 1: Customer Data Platform
---------------------------------

.. code-block:: yaml

   name: customer-data-platform
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
             expression: "classify_customer_segment(clv, engagement, recency)"

Example 2: Financial Data Pipeline
----------------------------------

.. code-block:: yaml

   name: financial-data-pipeline
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
           
           Include regulatory compliance status and any required actions.

Exercises
=========

Exercise 1: E-commerce Analytics Pipeline
-----------------------------------------

Build a pipeline that processes e-commerce data:

.. code-block:: text

   Your challenge:
   - Extract: Orders, customers, products, reviews
   - Transform: Calculate metrics, segment customers
   - Load: Create analytics-ready datasets
   - Quality: Validate business rules

Exercise 2: IoT Data Processing
------------------------------

Create a pipeline for IoT sensor data:

.. code-block:: text

   Requirements:
   - Handle high-volume time series data
   - Detect sensor anomalies
   - Aggregate by time windows
   - Generate maintenance alerts

Exercise 3: Social Media Analytics
---------------------------------

Build a social media data processing pipeline:

.. code-block:: yaml

   # Features:
   # - Extract from multiple platforms
   # - Text analysis and sentiment
   # - Trend detection
   # - Influence measurement

Solutions and Next Steps
========================

Complete solutions for all exercises are available in ``examples/tutorials/data_processing/``.

**Next Steps:**

1. **Try** :doc:`tutorial_content_generation` for AI-powered content creation
2. **Explore** :doc:`tutorial_automation` for workflow automation
3. **Combine** data processing with web research for comprehensive analytics
4. **Scale** your pipelines for production workloads

Best Practices for Production
=============================

1. **Data Validation**: Always validate data at ingestion and transformation steps
2. **Error Handling**: Plan for data quality issues and processing failures
3. **Monitoring**: Track data lineage and processing metrics
4. **Performance**: Optimize for your data volumes and latency requirements
5. **Security**: Protect sensitive data and comply with regulations
6. **Testing**: Test pipelines with representative data samples
7. **Documentation**: Document data schemas and business logic