# Pipeline Development Best Practices

## Overview

This guide presents best practices for developing robust, maintainable, and scalable pipelines in the Orchestrator framework. These practices are based on lessons learned from 40+ example pipelines, common issues identified in validation testing, and production deployment experience.

## Core Design Principles

### 1. Pipeline Structure and Organization

#### Use Descriptive IDs and Names
```yaml
# Good - Clear, descriptive identifiers
- id: analyze_customer_data
  tool: data-processing
  
# Bad - Vague or abbreviated
- id: step1
  tool: dp
```

#### Organize Steps Logically
```yaml
# Group related operations
# 1. Setup and initialization
- id: create_output_dir
- id: validate_inputs

# 2. Data processing
- id: load_customer_data
- id: clean_data
- id: analyze_patterns

# 3. Output generation
- id: generate_report
- id: save_results
```

#### Document Pipeline Purpose
```yaml
# Always include a clear description
name: "Customer Behavior Analysis Pipeline"
description: |
  Analyzes customer transaction data to identify behavioral patterns
  and generate actionable insights for marketing campaigns.
  
parameters:
  customer_data_file:
    type: string
    description: "Path to CSV file containing customer transaction data"
    required: true
```

### 2. Parameter Management

#### Define Clear Parameter Schemas
```yaml
parameters:
  # Required parameters with validation
  input_file:
    type: string
    required: true
    description: "Input data file path"
    
  # Optional parameters with sensible defaults
  output_format:
    type: string
    default: "json"
    enum: ["json", "csv", "yaml"]
    description: "Output format for processed data"
    
  # Numeric parameters with bounds
  batch_size:
    type: integer
    default: 100
    minimum: 1
    maximum: 1000
    description: "Number of records to process per batch"
```

#### Use Parameter Validation
```yaml
steps:
  - id: validate_parameters
    tool: validation
    action: validate
    parameters:
      data: "{{ parameters }}"
      schema:
        type: object
        properties:
          input_file:
            type: string
            pattern: "^.*\\.(csv|json|yaml)$"
        required: ["input_file"]
```

### 3. Model Configuration Best Practices

#### Use AUTO Tags Effectively
```yaml
# Specific task-based AUTO tags
model: <AUTO task="analyze">Select best model for data analysis</AUTO>

# Context-aware AUTO tags
model: <AUTO task="generate" context="technical">
  Select a model capable of generating technical documentation
  based on code analysis results
</AUTO>

# Fallback models for reliability
model: <AUTO task="summarize" fallback="gpt-3.5-turbo">
  Select best summarization model, fallback to GPT-3.5-turbo if unavailable
</AUTO>
```

#### Model Selection Patterns
```yaml
# For analysis tasks - prefer analytical models
model: <AUTO task="analyze">Models good at structured analysis</AUTO>

# For creative tasks - prefer creative models  
model: <AUTO task="generate" style="creative">Creative content generation</AUTO>

# For coding tasks - prefer code-focused models
model: <AUTO task="code">Models specialized in code generation and review</AUTO>

# For factual tasks - prefer factual accuracy
model: <AUTO task="fact-check">Models with strong factual accuracy</AUTO>
```

### 4. Template Rendering Best Practices

#### Safe Template Variable Access
```yaml
# Always use default filters for optional variables
content: "{{ optional_data | default('No data available') }}"

# Check for existence before accessing nested properties
content: "{% if analysis_results and analysis_results.summary %}{{ analysis_results.summary }}{% else %}Analysis pending{% endif %}"

# Use safe attribute access
content: "{{ (step_result.data.field) | default('N/A') }}"
```

#### Conditional Step References
```yaml
steps:
  - id: optional_enhancement
    if: "{{ enable_enhancement }}"
    action: enhance_text
    
  - id: process_text
    action: process
    parameters:
      # Safe reference to conditional step
      text: "{{ optional_enhancement.result | default(original_text) }}"
    dependencies:
      - optional_enhancement  # Ensures skip decision is made
```

#### Template Organization
```yaml
# Keep complex templates in separate files
steps:
  - id: generate_report
    action: generate_text
    parameters:
      prompt: "{% include 'templates/analysis_prompt.j2' %}"
      
# Use template inheritance for consistency
# base_report_template.j2:
# {% block title %}Default Title{% endblock %}
# {% block content %}{% endblock %}

# specific_report.j2:
# {% extends "base_report_template.j2" %}
# {% block title %}Customer Analysis Report{% endblock %}
```

## Error Handling and Resilience

### 1. Defensive Programming

#### Input Validation
```yaml
steps:
  - id: validate_input_data
    tool: validation
    action: validate
    parameters:
      data: "{{ load_data.content }}"
      schema:
        type: object
        properties:
          records:
            type: array
            minItems: 1
        required: ["records"]
    on_error: "stop"  # Fail fast for critical validation
```

#### Graceful Degradation
```yaml
steps:
  - id: try_advanced_analysis
    action: analyze_advanced
    on_error: "continue"  # Don't fail the entire pipeline
    
  - id: fallback_basic_analysis
    if: "{{ not try_advanced_analysis.success }}"
    action: analyze_basic
    parameters:
      data: "{{ input_data }}"
```

#### Error Recovery Patterns
```yaml
steps:
  - id: primary_processing
    action: process_data
    retry:
      attempts: 3
      delay: 5
    on_error: "continue"
    
  - id: recovery_processing
    if: "{{ not primary_processing.success }}"
    action: fallback_process
    parameters:
      data: "{{ input_data }}"
      mode: "safe"
```

### 2. Resource Management

#### Timeout Configuration
```yaml
steps:
  - id: long_running_analysis
    action: deep_analysis
    timeout: 300  # 5 minutes
    parameters:
      data: "{{ large_dataset }}"
      
  - id: quick_fallback
    if: "{{ not long_running_analysis.success }}"
    action: summary_analysis
    timeout: 30  # 30 seconds
```

#### Memory Management for Large Data
```yaml
steps:
  - id: process_large_dataset
    action: batch_process
    parameters:
      data: "{{ input_data }}"
      batch_size: 1000  # Process in chunks
      
  - id: streaming_analysis
    tool: data-processing
    action: stream_process
    parameters:
      input_stream: "{{ data_file }}"
      chunk_size: 500
```

## Control Flow Patterns

### 1. Loop Design

#### Effective For-Each Loops
```yaml
# Good - Clear iteration with proper parallel limits
- id: process_documents
  for_each: "{{ document_list }}"
  max_parallel: 3  # Prevent resource exhaustion
  steps:
    - id: analyze_document
      action: analyze_text
      parameters:
        text: "{{ $item.content }}"
        document_id: "{{ $item.id }}"
        
    - id: save_analysis
      dependencies: [analyze_document]
      tool: filesystem
      action: write
      parameters:
        path: "{{ output_dir }}/analysis_{{ $index }}.json"
        content: "{{ analyze_document.result }}"
```

#### Loop Variable Usage
```yaml
# Available loop variables:
# $item - Current iteration item
# $index - Zero-based index (0, 1, 2, ...)  
# $is_first - Boolean, true for first item
# $is_last - Boolean, true for last item

parameters:
  filename: "result_{{ $index }}_{{ $item.name }}.txt"
  metadata: |
    Processing item {{ $index + 1 }} of {{ total_items }}
    First item: {{ $is_first }}
    Last item: {{ $is_last }}
```

#### While Loop Patterns
```yaml
# Good - Clear termination condition
- id: iterative_improvement
  while: "{{ quality_score < 0.8 and iterations < 5 }}"
  steps:
    - id: improve_content
      action: enhance_text
      
    - id: evaluate_quality
      action: evaluate_quality
      parameters:
        content: "{{ improve_content.result }}"
        
  loop_vars:
    quality_score: "{{ evaluate_quality.score }}"
    iterations: "{{ iterations + 1 }}"
```

### 2. Conditional Logic

#### Clear Conditional Steps
```yaml
# Good - Descriptive conditions
- id: enhance_if_needed
  if: "{{ content_quality < quality_threshold }}"
  action: enhance_text
  
- id: validate_if_production
  if: "{{ environment == 'production' }}"
  action: validate_output
  
# Handle conditional dependencies properly
- id: use_enhancement_or_original
  action: format_text
  parameters:
    text: "{{ enhance_if_needed.result | default(original_text) }}"
  dependencies:
    - enhance_if_needed  # Wait for condition evaluation
```

## Performance Optimization

### 1. Parallel Processing

#### Smart Parallelization
```yaml
# Process independent tasks in parallel
steps:
  - id: analyze_sentiment
    action: analyze_sentiment
    
  - id: extract_entities
    action: extract_entities
    
  - id: classify_topic
    action: classify_text
    
  # Combine results (depends on all parallel tasks)
  - id: combine_analysis
    dependencies: [analyze_sentiment, extract_entities, classify_topic]
    action: combine_results
```

#### Batch Processing with Limits
```yaml
- id: process_large_batch
  for_each: "{{ large_item_list }}"
  max_parallel: 5  # Prevent API rate limiting
  steps:
    - id: process_item
      action: process_individual
      parameters:
        item: "{{ $item }}"
        batch_id: "{{ $index // 100 }}"  # Group into batches of 100
```

### 2. Caching Strategies

#### Result Caching
```yaml
steps:
  - id: expensive_computation
    action: complex_analysis
    cache_key: "analysis_{{ input_hash }}"  # Cache based on input
    cache_ttl: 3600  # Cache for 1 hour
    
  - id: use_cached_result
    action: format_results
    parameters:
      data: "{{ expensive_computation.result }}"
```

#### Model Response Caching
```yaml
steps:
  - id: categorize_content
    action: generate_text
    parameters:
      prompt: "Categorize this content: {{ content }}"
      # Same content will reuse cached response
      cache_response: true
      cache_key: "categorize_{{ content_hash }}"
```

## Data Processing Best Practices

### 1. Schema Validation

#### Comprehensive Schema Definition
```yaml
steps:
  - id: validate_customer_data
    tool: validation
    action: validate
    parameters:
      data: "{{ raw_data }}"
      schema:
        type: object
        properties:
          customers:
            type: array
            items:
              type: object
              properties:
                id: 
                  type: integer
                  minimum: 1
                email:
                  type: string
                  format: email
                purchase_history:
                  type: array
                  items:
                    type: object
                    properties:
                      amount:
                        type: number
                        minimum: 0
                      date:
                        type: string
                        format: date
              required: ["id", "email"]
        required: ["customers"]
      mode: strict  # Enforce strict compliance
```

### 2. Data Transformation

#### Safe Transformations
```yaml
steps:
  - id: transform_data
    tool: data-processing
    action: transform
    parameters:
      data: "{{ validated_data }}"
      operations:
        - type: filter
          field: active
          value: true
        - type: sort
          field: created_date
          order: desc
        - type: limit
          count: 1000  # Prevent memory issues
```

### 3. Output Generation

#### Clean Output Formatting
```yaml
steps:
  - id: generate_clean_json
    action: generate_text
    parameters:
      prompt: |
        Convert this data to clean JSON format:
        {{ processed_data }}
        
        Requirements:
        - Return ONLY valid JSON
        - No markdown formatting
        - No code fences (```json)
        - No explanatory text
        - Start with { and end with }
```

## Testing and Validation

### 1. Pipeline Testing

#### Unit Testing Steps
```yaml
# Test individual steps in isolation
steps:
  - id: test_data_loader
    tool: filesystem
    action: read
    parameters:
      path: "test_data/sample.json"
      
  - id: validate_loader_output
    tool: validation
    action: validate
    parameters:
      data: "{{ test_data_loader.content }}"
      schema:
        type: object
        properties:
          test_field: {type: string}
```

#### Integration Testing
```yaml
# Test complete pipeline with sample data
parameters:
  test_mode:
    type: boolean
    default: false
    description: "Run pipeline in test mode with sample data"
    
steps:
  - id: load_test_data
    if: "{{ test_mode }}"
    tool: filesystem
    action: read
    parameters:
      path: "test_data/integration_test.json"
      
  - id: load_production_data
    if: "{{ not test_mode }}"
    tool: filesystem
    action: read
    parameters:
      path: "{{ production_data_path }}"
```

### 2. Output Validation

#### Validate Generated Content
```yaml
steps:
  - id: generate_report
    action: generate_text
    
  - id: validate_report_structure
    tool: validation
    action: validate
    parameters:
      data: "{{ generate_report.result }}"
      schema:
        type: object
        properties:
          title: {type: string}
          summary: {type: string}
          sections:
            type: array
            minItems: 1
        required: ["title", "summary", "sections"]
```

## Documentation and Maintenance

### 1. Pipeline Documentation

#### Comprehensive Documentation
```yaml
# Always include complete metadata
name: "Customer Analysis Pipeline"
version: "1.2.0"
description: |
  Analyzes customer transaction data to identify behavioral patterns,
  generate customer segments, and produce actionable marketing insights.
  
  Input: Customer transaction CSV with columns: customer_id, transaction_date, 
         amount, category, payment_method
  Output: Analysis report with customer segments and recommendations

author: "Data Science Team"
tags: ["analytics", "customer-data", "segmentation"]
requirements:
  - "Customer data in CSV format"
  - "Write access to output directory"
  - "Internet connection for model API calls"
```

#### Usage Examples
```yaml
# Include example usage in comments
# Example usage:
# python scripts/run_pipeline.py customer_analysis.yaml \
#   -i customer_data="data/customers_2024.csv" \
#   -i output_dir="results/customer_analysis" \
#   -i min_transactions=5
```

### 2. Version Control

#### Pipeline Versioning
```yaml
# Track pipeline versions and changes
version: "2.1.0"
changelog:
  - version: "2.1.0"
    date: "2024-08-22"
    changes:
      - "Added support for multiple data formats"
      - "Improved error handling for malformed data"
  - version: "2.0.0"
    date: "2024-08-15"
    changes:
      - "Complete rewrite using new validation framework"
      - "Breaking change: Updated parameter schema"
```

## Common Anti-Patterns to Avoid

### 1. Poor Error Handling
```yaml
# Bad - No error handling
- id: risky_operation
  action: external_api_call

# Good - Proper error handling
- id: safe_operation
  action: external_api_call
  retry:
    attempts: 3
    delay: 5
  on_error: "continue"
  timeout: 30
```

### 2. Inefficient Resource Usage
```yaml
# Bad - No limits on parallel processing
- id: process_all
  for_each: "{{ huge_list }}"  # Could be thousands of items
  steps:
    - id: process_item  # All items processed simultaneously

# Good - Controlled resource usage
- id: process_batched
  for_each: "{{ huge_list }}"
  max_parallel: 5  # Limit concurrent processing
  steps:
    - id: process_item
```

### 3. Poor Template Design
```yaml
# Bad - Unsafe template access
content: "{{ risky_step.nested.field.value }}"

# Good - Safe template access
content: "{{ (risky_step.nested.field.value) | default('N/A') }}"
```

### 4. Unclear Dependencies
```yaml
# Bad - Implicit dependencies
steps:
  - id: step1
  - id: step2  # Depends on step1 but not declared
  - id: step3

# Good - Explicit dependencies
steps:
  - id: step1
  - id: step2
    dependencies: [step1]
  - id: step3
    dependencies: [step1, step2]
```

## Performance Monitoring

### 1. Execution Metrics

#### Track Pipeline Performance
```yaml
steps:
  - id: start_timer
    tool: system
    action: timestamp
    
  - id: main_processing
    # ... processing steps
    
  - id: end_timer
    tool: system
    action: timestamp
    
  - id: calculate_duration
    action: generate_text
    parameters:
      prompt: |
        Calculate execution time:
        Start: {{ start_timer.timestamp }}
        End: {{ end_timer.timestamp }}
        Duration: {{ end_timer.timestamp - start_timer.timestamp }} seconds
```

### 2. Resource Monitoring

#### Memory and Resource Tracking
```yaml
steps:
  - id: check_system_resources
    tool: system
    action: resources
    
  - id: adjust_batch_size
    if: "{{ check_system_resources.memory_usage > 0.8 }}"
    action: reduce_batch_size
    parameters:
      current_size: "{{ batch_size }}"
      reduction_factor: 0.5
```

## Deployment Considerations

### 1. Environment Configuration

#### Environment-Specific Settings
```yaml
parameters:
  environment:
    type: string
    default: "development"
    enum: ["development", "staging", "production"]
    
  api_timeout:
    type: integer
    default: "{% if environment == 'production' %}60{% else %}30{% endif %}"
    
  max_retries:
    type: integer
    default: "{% if environment == 'production' %}5{% else %}2{% endif %}"
```

### 2. Security Best Practices

#### Secure API Key Management
```yaml
# Don't hardcode API keys
# Bad:
# api_key: "sk-1234567890abcdef"

# Good - Use environment variables
parameters:
  api_key:
    type: string
    required: true
    description: "API key from environment variable OPENAI_API_KEY"
    
# Or use secure configuration
steps:
  - id: load_api_config
    tool: config
    action: load_secure
    parameters:
      config_path: "{{ secure_config_path }}"
```

## Conclusion

Following these best practices will help you create robust, maintainable, and scalable pipelines. Key principles to remember:

1. **Design for Failure**: Always include error handling and fallback strategies
2. **Document Everything**: Clear documentation saves time and prevents errors  
3. **Test Thoroughly**: Unit test steps and integration test complete pipelines
4. **Monitor Performance**: Track execution metrics and resource usage
5. **Secure by Default**: Never hardcode sensitive information
6. **Start Simple**: Begin with basic implementations and add complexity gradually

For more specific guidance, refer to:
- [Troubleshooting Guide](troubleshooting.md) - Common issues and solutions
- [Migration Guide](migration.md) - Upgrading from older versions
- [Common Issues](common-issues.md) - Known limitations and workarounds

## Additional Resources

- [Example Pipelines Documentation](/docs/examples/README.md)
- [API Reference](/docs/api_reference.md) 
- [Tool Catalog](/docs/reference/tool_catalog.md)
- [YAML Configuration Guide](/docs/user_guide/yaml_configuration.rst)