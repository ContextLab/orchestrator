# Pipeline Tutorial: interactive_pipeline

## Overview

**Complexity Level**: Intermediate  
**Difficulty Score**: 55/100  
**Estimated Runtime**: 5-15 minutes  

### Purpose
This pipeline demonstrates conditional_execution, csv_processing, interactive_workflows and provides a practical example of orchestrator's capabilities for intermediate-level workflows.

### Use Cases
- AI-powered content generation

### Prerequisites
- Basic understanding of YAML syntax
- Familiarity with template variables and data flow
- Understanding of basic control flow concepts

### Key Concepts
- Conditional logic and branching
- Large language model integration
- Template variable substitution

## Pipeline Breakdown

### Configuration Analysis
- **input_section**: Defines the data inputs and parameters for the pipeline
- **steps_section**: Contains the sequence of operations to be executed
- **output_section**: Specifies how results are formatted and stored
- **template_usage**: Uses 9 template patterns for dynamic content
- **feature_highlights**: Demonstrates 7 key orchestrator features

### Data Flow
This pipeline processes input parameters through 7 steps to generate the specified outputs.

### Control Flow
Follows linear execution flow from first step to last step.

### Pipeline Configuration
```yaml
# Interactive Pipeline with User Input and Approval
# Demonstrates human-in-the-loop workflows with real data processing
id: interactive_pipeline
name: Interactive Sales Data Processing Pipeline
description: Process sales data with user-selected transformations, approval gates, and feedback
version: "2.0.0"

parameters:
  input_file:
    type: string
    default: "data/sales_data.csv"
  output_dir:
    type: string
    default: "examples/outputs/interactive_pipeline"

steps:
  - id: get_processing_options
    tool: user-prompt
    action: execute
    parameters:
      prompt: "Select data processing method"
      input_type: "choice"
      choices: ["aggregate", "filter", "transform", "analyze"]
      default: "aggregate"
      context: "cli"
    
  - id: get_specific_operation
    tool: user-prompt
    action: execute
    parameters:
      prompt: |
        {% if get_processing_options.value == 'aggregate' %}
        Select aggregation type
        {% elif get_processing_options.value == 'filter' %}
        Select filter criteria
        {% elif get_processing_options.value == 'transform' %}
        Select transformation type
        {% else %}
        Select analysis type
        {% endif %}
      input_type: "choice"
      choices: |
        {% if get_processing_options.value == 'aggregate' %}
        ["by_category", "by_region", "by_date", "top_products"]
        {% elif get_processing_options.value == 'filter' %}
        ["high_value", "electronics_only", "recent_orders", "top_customers"]
        {% elif get_processing_options.value == 'transform' %}
        ["add_totals", "calculate_margins", "normalize_prices", "pivot_data"]
        {% else %}
        ["summary_stats", "sales_trends", "customer_analysis", "product_performance"]
        {% endif %}
      default: |
        {% if get_processing_options.value == 'aggregate' %}by_category{% elif get_processing_options.value == 'filter' %}high_value{% elif get_processing_options.value == 'transform' %}add_totals{% else %}summary_stats{% endif %}
      context: "cli"
    dependencies:
      - get_processing_options
    
  - id: get_output_format
    tool: user-prompt
    action: execute
    parameters:
      prompt: "Select output format"
      input_type: "choice"
      choices: ["csv", "json", "markdown"]
      default: "csv"
      context: "cli"
    dependencies:
      - get_specific_operation
    
  - id: read_data
    tool: filesystem
    action: read
    parameters:
      path: "{{ output_dir }}/{{ input_file }}"
    dependencies:
      - get_output_format
    
  - id: process_data
    action: generate_text
    parameters:
      prompt: |
        Process this CSV sales data according to the following instructions:
        
        Data:
        {{ read_data.content }}
        
        Processing Method: {{ get_processing_options.value }}
        Specific Operation: {{ get_specific_operation.value }}
        
        Instructions:
        {% if get_processing_options.value == 'aggregate' %}
          {% if get_specific_operation.value == 'by_category' %}
          Group by category and calculate:
          - Total quantity sold
          - Total revenue (quantity * unit_price)
          - Average unit price
          - Number of transactions
          {% elif get_specific_operation.value == 'by_region' %}
          Group by region and calculate:
          - Total sales revenue
          - Number of orders
          - Average order value
          - Most popular product
          {% elif get_specific_operation.value == 'by_date' %}
          Group by date and show:
          - Daily revenue
          - Number of transactions
          - Best selling product of the day
          {% else %}
          Find the top 5 products by total revenue with columns:
          - Product name
          - Total quantity sold
          - Total revenue
          - Average price
          {% endif %}
        {% elif get_processing_options.value == 'filter' %}
          {% if get_specific_operation.value == 'high_value' %}
          Filter to show only orders where (quantity * unit_price) > 500.
          Include all original columns plus a 'total_value' column.
          {% elif get_specific_operation.value == 'electronics_only' %}
          Filter to show only Electronics category items.
          Sort by total value (quantity * unit_price) descending.
          {% elif get_specific_operation.value == 'recent_orders' %}
          Filter to show only orders from January 20, 2024 onwards.
          Sort by date descending.
          {% else %}
          Show top 3 customers by total purchase amount with:
          - Customer ID
          - Number of orders
          - Total amount spent
          - Favorite category
          {% endif %}
        {% elif get_processing_options.value == 'transform' %}
          {% if get_specific_operation.value == 'add_totals' %}
          Add these new columns to each row:
          - total_value = quantity * unit_price
          - tax_amount = total_value * 0.08
          - final_amount = total_value + tax_amount
          {% elif get_specific_operation.value == 'calculate_margins' %}
          Add profit margin columns:
          - cost_basis = unit_price * 0.7 for Electronics, unit_price * 0.6 for Furniture
          - profit = (unit_price - cost_basis) * quantity
          - margin_percent = (profit / (unit_price * quantity)) * 100
          {% elif get_specific_operation.value == 'normalize_prices' %}
          Normalize all unit prices to an index where the highest price = 100.
          Add columns: original_price, normalized_price, price_index
          {% else %}
          Create a pivot table with:
          - Dates as rows
          - Categories as columns  
          - Sum of quantities as values
          {% endif %}
        {% else %}
          {% if get_specific_operation.value == 'summary_stats' %}
          Provide summary statistics:
          - Total revenue
          - Total orders
          - Average order value
          - Best selling product
          - Best sales day
          - Top customer
          {% elif get_specific_operation.value == 'sales_trends' %}
          Analyze sales trends:
          - Daily average sales
          - Growth rate from first to last day
          - Best and worst days
          - Weekend vs weekday performance
          {% elif get_specific_operation.value == 'customer_analysis' %}
          Analyze customers:
          - Total unique customers
          - Repeat customer rate
          - Average purchase per customer
          - Top 3 spenders
          - Most popular category per customer segment
          {% else %}
          Product performance analysis:
          - Best seller by quantity
          - Top revenue generator
          - Category breakdown (% of total sales)
          - Price point analysis
          {% endif %}
        {% endif %}
        
        Output format: {{ get_output_format.value }}
        {% if get_output_format.value == 'csv' %}
        Format as CSV with headers. Use comma separators.
        {% elif get_output_format.value == 'json' %}
        Format as valid JSON array of objects.
        {% else %}
        Format as a markdown table with a title and summary.
        {% endif %}
        
        IMPORTANT: Output ONLY the processed data in the requested format, no explanations or markdown code blocks.
      model: <AUTO>
      max_tokens: 2000
    dependencies:
      - read_data
    
  - id: create_summary
    action: generate_text
    parameters:
      prompt: |
        Create a concise business summary based on this data processing:
        
        Original data: {{ read_data.content.split('\n') | length - 1 }} rows of sales data
        Processing type: {{ get_processing_options.value }} - {{ get_specific_operation.value }}
        
        Processed results:
        {{ process_data | truncate(500) }}
        
        Write exactly 2-3 sentences summarizing:
        1. What processing was done
        2. Key findings with specific numbers from the results above
        
        Do NOT ask for more data or say you need more information. Use only the data shown above.
        Write in past tense. Be direct and specific.
      model: <AUTO>
      max_tokens: 200
    dependencies:
      - process_data
    
  - id: approve_results
    tool: approval-gate
    action: execute
    parameters:
      title: "Review Processed Data"
      content: |
        ## Processing Summary
        {{ create_summary }}
        
        ## Processed Data (first 1000 chars)
        ```
        {{ process_data | truncate(1000) }}
        ```
        
        Approve to save the results?
      format: "text"
      allow_modifications: true
      require_reason: true
      context: "cli"
    dependencies:
      - create_summary
    
  - id: save_if_approved
    tool: filesystem
    action: write
    parameters:
      path: "{{ output_dir }}/data/processed_{{ get_specific_operation.value }}.{{ get_output_format.value }}"
      content: "{{ approve_results.modified_content | default(process_data) }}"
    dependencies:
      - approve_results
    condition: "{{ approve_results.approved }}"
    
  - id: collect_feedback
    tool: feedback-collection
    action: execute
    parameters:
      title: "Pipeline Experience Feedback"
      questions:
        - id: "data_quality"
          text: "Rate the quality of data processing"
          type: "rating"
          scale: 5
        - id: "ease_of_use"
          text: "How easy was the pipeline to use?"
          type: "rating"
          scale: 5
        - id: "processing_useful"
          text: "Was the processing useful for your needs?"
          type: "boolean"
        - id: "would_use_again"
          text: "Would you use this pipeline again?"
          type: "boolean"
        - id: "suggestions"
          text: "Any suggestions for improvement?"
          type: "text"
      required_questions: ["data_quality", "ease_of_use", "would_use_again"]
      anonymous: false
      save_to_file: "{{ output_dir }}/feedback/pipeline_feedback.json"
      context: "cli"
    dependencies:
      - save_if_approved
    
  - id: generate_summary
    tool: filesystem
    action: write
    parameters:
      path: "{{ output_dir }}/summary.md"
      content: |
        # Pipeline Execution Summary
        
        ## Processing Configuration
        - **Method**: {{ get_processing_options.value }}
        - **Operation**: {{ get_specific_operation.value }}
        - **Output Format**: {{ get_output_format.value }}
        
        ## Data Summary
        - **Input File**: {{ input_file }}
        - **Rows Processed**: {{ read_data.content.split('\n') | length - 1 }}
        
        ## Business Insights
        {{ create_summary }}
        
        ## Approval Status
        - **Status**: {{ 'Approved' if approve_results.approved else 'Rejected' }}
        {% if approve_results.approved %}
        - **Output File**: data/processed_{{ get_specific_operation.value }}.{{ get_output_format.value }}
        {% else %}
        - **Rejection Reason**: {{ approve_results.rejection_reason }}
        {% endif %}
        
        ## User Feedback
        - **Data Quality**: {{ collect_feedback.summary.rating_average | round(1) }}/5
        - **Would Use Again**: {{ 'Yes' if collect_feedback.summary.boolean_summary.would_use_again else 'No' }}
        - **Processing Useful**: {{ 'Yes' if collect_feedback.summary.boolean_summary.processing_useful else 'No' }}
        
        ## Timestamp
        Generated at: {{ now() }}
    dependencies:
      - collect_feedback
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
- This pipeline produces Analysis results, CSV data, JSON data, Markdown documents - adjust output configuration accordingly

## Remixing Instructions

### Compatible Patterns
- fact_checker.yaml - for content verification
- research workflows - for information gathering

### Extension Ideas
- Add iterative processing for continuous improvement
- Implement parallel processing for better performance
- Include advanced error recovery mechanisms

### Combination Examples
- Can be combined with most other pipeline patterns

### Advanced Variations
- Scale to handle larger datasets and more complex processing
- Add real-time processing capabilities for streaming data
- Implement distributed processing across multiple systems
- Use multiple AI models for comparison and validation

## Hands-On Exercise

### Execution Instructions
- 1. Navigate to your orchestrator project directory
- 2. Run: python scripts/run_pipeline.py examples/interactive_pipeline.yaml
- 3. Monitor the output for progress and any error messages
- 4. Check the output directory for generated results

### Expected Outputs
- Generated Analysis results in the specified output directory
- Generated CSV data in the specified output directory
- Generated JSON data in the specified output directory
- Generated Markdown documents in the specified output directory
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

*Tutorial generated on 2025-08-27T23:40:24.396172*
