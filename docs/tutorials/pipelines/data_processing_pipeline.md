# Pipeline Tutorial: data_processing_pipeline

## Overview

**Complexity Level**: Intermediate  
**Difficulty Score**: 55/100  
**Estimated Runtime**: 5-15 minutes  

### Purpose
This pipeline shows how to process and analyze data using orchestrator's data processing capabilities. It demonstrates conditional_execution, csv_processing, data_flow for building robust data workflows.

### Use Cases
- AI-powered content generation
- Automated data processing workflows
- Business data analysis and reporting
- Data quality assessment and cleaning

### Prerequisites
- Basic understanding of YAML syntax
- Familiarity with template variables and data flow
- Understanding of basic control flow concepts

### Key Concepts
- Conditional logic and branching
- Data flow between pipeline steps
- Large language model integration
- Template variable substitution

## Pipeline Breakdown

### Configuration Analysis
- **input_section**: Defines the data inputs and parameters for the pipeline
- **steps_section**: Contains the sequence of operations to be executed
- **output_section**: Specifies how results are formatted and stored
- **template_usage**: Uses 15 template patterns for dynamic content
- **feature_highlights**: Demonstrates 7 key orchestrator features

### Data Flow
This pipeline processes input parameters through 7 steps to generate the specified outputs.

### Control Flow
Follows linear execution flow from first step to last step.

### Pipeline Configuration
```yaml
id: data-processing-pipeline
# Data Processing Pipeline
# Demonstrates data transformation, validation, and analysis

name: data-processing-pipeline
description: Comprehensive data processing workflow
version: "1.0.0"

inputs:
  input_file: "sales_data.csv"
  output_path: "examples/outputs/data_processing_pipeline"
  quality_threshold: 0.95
  enable_profiling: true

steps:
  # Step 1: Read input data
  - id: read_data
    tool: filesystem
    action: read
    parameters:
      path: "{{ output_path }}/{{ input_file }}"
      
  # Step 2: Profile data quality
  - id: profile_data
    tool: data-processing
    action: profile
    condition: "{{ enable_profiling }}"
    parameters:
      data: "{{ read_data.content }}"
      format: "csv"
      profiling_options:
        - missing_values
        - data_types
        - statistical_summary
        - outlier_detection
        - duplicate_detection
    dependencies:
      - read_data
        
  # Step 3: Validate data schema
  - id: validate_schema
    tool: validation
    action: validate
    parameters:
      data: "{{ read_data.content }}"
      mode: "CSV"  # Use CSV validation mode
      schema:
        type: object
        properties:
          order_id:
            type: string
            pattern: "^ORD-[0-9]{6}$"
          customer_id:
            type: string
          product_name:
            type: string
          quantity:
            type: integer
            minimum: 1
          unit_price:
            type: number
            minimum: 0
          order_date:
            type: string
            format: date
          status:
            type: string
            enum: ["pending", "processing", "shipped", "delivered", "cancelled"]
        required: ["order_id", "customer_id", "product_name", "quantity", "unit_price"]
      mode: "LENIENT"  # Try to fix minor issues
    dependencies:
      - read_data
      
  # Step 4: Clean and transform data
  - id: clean_data
    tool: data-processing
    action: transform
    parameters:
      data: "{{ validate_schema.data if validate_schema.valid else read_data.content }}"
      format: "json"  # The validated data is already parsed as JSON
      output_format: "json"
      operations:
        # Remove duplicates
        - type: deduplicate
          columns: ["order_id"]
          keep: "first"
          
        # Fix data types
        - type: cast
          columns:
            quantity: "integer"
            unit_price: "float"
            order_date: "datetime"
            
        # Handle missing values
        - type: fill_missing
          strategy:
            status: "pending"  # Default status
            quantity: 1        # Default quantity
            
        # Create calculated fields
        - type: calculate
          expressions:
            total_amount: "quantity * unit_price"
            order_month: "DATE_FORMAT(order_date, '%Y-%m')"
            
        # Filter out cancelled orders for analysis
        - type: filter
          condition: "status != 'cancelled'"
    dependencies:
      - validate_schema
          
  # Step 5: Aggregate data for analysis
  - id: aggregate_monthly
    tool: data-processing
    action: aggregate
    parameters:
      data: "{{ clean_data.processed_data }}"
      format: "json"
      output_format: "json"
      group_by: ["order_month", "product_name"]
      aggregations:
        total_quantity:
          column: "quantity"
          function: "sum"
        total_revenue:
          column: "total_amount"
          function: "sum"
        average_price:
          column: "unit_price"
          function: "mean"
        order_count:
          column: "order_id"
          function: "count"
        unique_customers:
          column: "customer_id"
          function: "count_distinct"
    dependencies:
      - clean_data
          
  # Step 6: Statistical analysis  
  - id: analyze_trends
    action: analyze_text
    parameters:
      analysis_type: "statistical_trends"
      prompt: |
        Analyze the following sales data and create a JSON response.
        
        Monthly sales data ({{ aggregate_monthly.processed_data | length }} products):
        {% for row in aggregate_monthly.processed_data %}
        - Product: {{ row.product_name }}, Revenue: ${{ row.total_revenue | round(2) }}, Quantity: {{ row.total_quantity }}, Month: {{ row.order_month }}
        {% endfor %}
        
        Based on this data, create a JSON response with the following:
        
        1. Calculate the growth_rate: If there are multiple months, calculate month-over-month growth percentage. If only one month, set to 0.
        
        2. Identify top_products: List all products sorted by total revenue (highest to lowest). Include product name and actual revenue from the data.
        
        3. Identify seasonal_patterns: Look for patterns across months if multiple months exist, otherwise note the time period covered.
        
        4. Detect anomalies: Identify any unusual patterns, sudden changes, or outliers in the data.
        
        Return ONLY a valid JSON object with this structure:
        {
          "growth_rate": <calculated_growth_rate>,
          "top_products": [
            {"product": "<product_name>", "revenue": <actual_revenue>},
            ...
          ],
          "seasonal_patterns": ["<pattern_description>"],
          "anomalies": [<any_anomalies_found>]
        }
      model: "<AUTO>"
    dependencies:
      - aggregate_monthly
                  
  # Step 7: Create pivot table
  - id: pivot_analysis
    tool: data-processing
    action: pivot
    parameters:
      data: "{{ clean_data.processed_data }}"
      format: "json"
      output_format: "json"
      index: ["product_name"]
      columns: ["status"]
      values: ["quantity"]
      aggfunc: "sum"
      fill_value: 0
    dependencies:
      - clean_data
      
  # Step 8: Quality check
  - id: quality_check
    action: analyze_text
    parameters:
      text: |
        Data profile summary:
        - Total rows: {{ profile_data.processed_data.row_count }}
        - Total columns: {{ profile_data.processed_data.column_count }}
        - Duplicate rows: {{ profile_data.processed_data.duplicate_rows }}
        - Columns with missing data: {% for col_name, col_data in profile_data.processed_data.columns.items() %}{% if col_data.missing_count > 0 %}{{ col_name }} ({{ col_data.missing_percentage }}%), {% endif %}{% endfor %}None
        - Data types: {% for col_name, col_data in profile_data.processed_data.columns.items() %}{{ col_name }}:{{ col_data.data_type }}, {% endfor %}
        - Validation passed: {{ validate_schema.valid | default(false) }}
        - Rows after cleaning: {{ clean_data.processed_data | from_json | length if clean_data.processed_data else 0 }}
      analysis_type: "quality_assessment"
      prompt: |
        Based on this data profile summary, calculate a data quality score.
        
        Score calculation:
        - Start with 1.0 (perfect score)
        - Deduct 0.1 for duplicate rows (we have {{ profile_data.processed_data.duplicate_rows }} duplicates)
        - Deduct 0.05 for each column with >10% missing data
        - Deduct 0.1 if validation failed (validation passed: {{ validate_schema.valid | default(false) }})
        - Deduct 0.05 for outliers if >10% (quantity column has {{ profile_data.processed_data.columns.quantity.outlier_percentage }}% outliers)
        
        Return as JSON with fields:
        - quality_score: calculated score between 0 and 1
        - issues_found: list specific issues found in THIS data
        - recommendations: list specific recommendations for THIS data
      model: "<AUTO>"
    dependencies:
      - profile_data
      - validate_schema
      - clean_data
      
  # Step 3b: Save validation report (always save for audit trail)
  - id: save_validation_report
    tool: filesystem
    action: write
    parameters:
      path: "{{ output_path }}/validation_report.json"
      content: |
        {
          "validation_success": {{ validate_schema.valid | default(false) | to_json }},
          "errors": {{ validate_schema.errors | default([]) | to_json }},
          "warnings": {{ validate_schema.warnings | default([]) | to_json }},
          "rows_validated": {{ validate_schema.rows_validated | default(0) }},
          "timestamp": "{{ now() }}",
          "input_file": "{{ input_file }}",
          "validation_mode": "{{ validate_schema.mode | default('strict') }}"
        }
    dependencies:
      - validate_schema
              
  # Step 9: Export processed data
  - id: export_data
    tool: data-processing
    action: convert
    parameters:
      data: "{{ clean_data.processed_data }}"
      format: "json"
      output_format: "csv"
    dependencies:
      - clean_data
      
  - id: save_processed
    tool: filesystem
    action: write
    parameters:
      path: "processed_data.csv"
      content: "{{ export_data.processed_data }}"
    dependencies:
      - export_data
      
  # Step 10: Generate data report
  - id: generate_report
    tool: filesystem
    action: write
    parameters:
      path: "data_processing_report.md"
      content: |
        # Data Processing Report
        
        ## Processing Summary
        
        - **Input File**: {{ input_file }}
        - **Output Path**: {{ output_path }}
        - **Rows Processed**: {{ clean_data.processed_data | from_json | length if clean_data.processed_data else 0 }}
        - **Data Profile**: 
          - Total Rows: {{ profile_data.processed_data.row_count | default(0) }}
          - Total Columns: {{ profile_data.processed_data.column_count | default(0) }}
          - Duplicate Rows: {{ profile_data.processed_data.duplicate_rows | default(0) }}
        
        ## Data Validation Results
        
        {% if validate_schema.valid %}
        ✅ **Validation Passed**: All data conforms to schema requirements
        {% else %}
        ❌ **Validation Failed**: {{ validate_schema.error | default("Schema validation errors detected") }}
        - Validation report saved to: `{{ output_path }}/validation_report.json`
        {% endif %}
        
        ## Data Quality Assessment
        
        ### Quality Score: {% if quality_check.result.quality_score %}{{ quality_check.result.quality_score | round(2) }}{% else %}0.00{% endif %}/1.0
        
        ### Issues Found
        {% if quality_check.result.issues_found %}
        {% for issue in quality_check.result.issues_found %}
        - ⚠️ {{ issue }}
        {% endfor %}
        {% else %}
        - ✅ No major issues detected
        {% endif %}
        
        ## Column Statistics
        
        | Column | Type | Missing % | Unique Values | Min | Max | Mean |
        |--------|------|-----------|---------------|-----|-----|------|
        {% for col_name, col_data in profile_data.processed_data.columns.items() %}
        | {{ col_name }} | {{ col_data.data_type }} | {{ col_data.missing_percentage | round(1) }}% | {{ col_data.unique_count }} | {{ col_data.min | default('N/A') }} | {{ col_data.max | default('N/A') }} | {% if col_data.data_type == 'numeric' and col_data.mean is defined %}{{ col_data.mean | round(2) }}{% else %}N/A{% endif %} |
        {% endfor %}
        
        ## Monthly Aggregations
        
        {% if aggregate_monthly.processed_data %}
        | Month | Total Quantity | Total Revenue | Avg Price | Order Count | Unique Customers |
        |-------|----------------|---------------|-----------|-------------|------------------|
        {% for row in aggregate_monthly.processed_data %}
        | {{ row.order_month }} | {{ row.total_quantity | default(0) }} | ${% if row.total_revenue %}{{ row.total_revenue | round(2) }}{% else %}0.00{% endif %} | ${% if row.average_price %}{{ row.average_price | round(2) }}{% else %}0.00{% endif %} | {{ row.order_count | default(0) }} | {{ row.unique_customers | default(0) }} |
        {% endfor %}
        {% else %}
        *No monthly aggregation data available*
        {% endif %}
        
        ## Product Status Distribution (Pivot Table)
        
        {% if pivot_analysis.processed_data %}
        | Product | Pending | Processing | Shipped | Delivered | Total |
        |---------|---------|------------|---------|-----------|-------|
        {% for row in pivot_analysis.processed_data %}
        | {{ row.product_name }} | {{ row.pending | default(0) }} | {{ row.processing | default(0) }} | {{ row.shipped | default(0) }} | {{ row.delivered | default(0) }} | {{ (row.pending | default(0)) + (row.processing | default(0)) + (row.shipped | default(0)) + (row.delivered | default(0)) }} |
        {% endfor %}
        {% else %}
        *No pivot table data available*
        {% endif %}
        
        ## Statistical Analysis
        
        {% if analyze_trends.result %}
        ### Growth Rate: {{ analyze_trends.result.growth_rate | default('N/A') }}%
        
        ### Top Products by Revenue
        {% for product in analyze_trends.result.top_products | default([]) %}
        {{ loop.index }}. **{{ product.product }}** - ${% if product.revenue %}{{ product.revenue | round(2) }}{% else %}0.00{% endif %}
        {% endfor %}
        
        ### Seasonal Patterns
        {% for pattern in analyze_trends.result.seasonal_patterns | default([]) %}
        - {{ pattern }}
        {% endfor %}
        
        ### Anomalies Detected
        {% for anomaly in analyze_trends.result.anomalies | default([]) %}
        - **{{ anomaly.month }}**: {{ anomaly.description }} ({{ anomaly.metric }})
        {% endfor %}
        {% else %}
        *Statistical analysis pending*
        {% endif %}
        
        ## Recommendations
        
        {% if quality_check.result.recommendations %}
        {% for rec in quality_check.result.recommendations %}
        - 📌 {{ rec }}
        {% endfor %}
        {% endif %}
        
        ---
        
        *Report generated on: {{ now() }}*
        *Pipeline ID: {{ pipeline_id }}*
    dependencies:
      - clean_data
      - quality_check
      - aggregate_monthly
      - analyze_trends
      - pivot_analysis
        
  # Step 11: Save report to outputs folder
  - id: save_report
    tool: filesystem
    action: copy
    parameters:
      path: "data_processing_report.md"  # source file
      destination: "{{ output_path }}/data_processing_report.md"
    dependencies:
      - generate_report
      
  # Step 12: Save processed data to outputs folder
  - id: save_processed_output
    tool: filesystem
    action: write
    parameters:
      path: "{{ output_path }}/processed_data.csv"
      content: "{{ export_data.processed_data }}"
    dependencies:
      - export_data

outputs:
  processed_file: "{{ output_path }}/processed_data.csv"
  quality_score: "{{ quality_check.result.quality_score | default(0) }}"
  rows_processed: "{{ profile_data.processed_data.row_count | default(0) }}"
  report_path: "{{ output_path }}/data_processing_report.md"
  validation_report: "{{ output_path }}/validation_report.json"
  validation_passed: "{{ validate_schema.valid | default(false) }}"
  growth_rate: "{{ analyze_trends.result.growth_rate | default('N/A') }}"
```

## Customization Guide

### Input Modifications
- Modify input parameters to match your specific data sources
- Adjust file paths and data formats as needed for your environment

### Parameter Tuning
- Adjust model parameters (temperature, max_tokens) for different output styles
- Modify prompts to change the tone and focus of generated content

### Step Modifications
- Add new steps by following the same pattern as existing ones
- Remove steps that aren't needed for your specific use case
- Reorder steps if your workflow requires different sequencing
- Replace tool actions with alternatives that provide similar functionality

### Output Customization
- Change output file paths and formats to match your requirements
- Modify output templates to customize the structure and content
- This pipeline produces Analysis results, CSV data, JSON data, Markdown documents, Reports - adjust output configuration accordingly

## Remixing Instructions

### Compatible Patterns
- fact_checker.yaml - for content verification
- research workflows - for information gathering

### Extension Ideas
- Add iterative processing for continuous improvement
- Implement parallel processing for better performance
- Include advanced error recovery mechanisms

### Combination Examples
- Combine with research workflows to gather additional data
- Use with statistical analysis for comprehensive insights
- Integrate with visualization tools for data presentation

### Advanced Variations
- Scale to handle larger datasets and more complex processing
- Add real-time processing capabilities for streaming data
- Implement distributed processing across multiple systems
- Use multiple AI models for comparison and validation

## Hands-On Exercise

### Execution Instructions
- 1. Navigate to your orchestrator project directory
- 2. Run: python scripts/run_pipeline.py examples/data_processing_pipeline.yaml
- 3. Monitor the output for progress and any error messages
- 4. Check the output directory for generated results

### Expected Outputs
- Generated Analysis results in the specified output directory
- Generated CSV data in the specified output directory
- Generated JSON data in the specified output directory
- Generated Markdown documents in the specified output directory
- Generated Reports in the specified output directory
- Execution logs showing step-by-step progress
- Completion message with runtime statistics
- No error messages or warnings (successful execution)

### Troubleshooting
- **Template Resolution Errors**: Check that all input parameters are provided and template syntax is correct
- **General Execution Errors**: Check the logs for specific error messages and verify your orchestrator installation

### Verification Steps
- Check that the pipeline completed without errors
- Verify all expected output files were created
- Review the output content for quality and accuracy
- Review generated content for relevance and quality

---

*Tutorial generated on 2025-08-27T23:40:24.396021*
