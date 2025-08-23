# Troubleshooting Guide

## Overview

This guide provides solutions to common issues encountered when developing and running pipelines in the Orchestrator framework. Issues are organized by category with specific error messages, root causes, and step-by-step solutions.

Based on analysis of 40+ example pipelines and validation testing, this guide addresses the most frequent problems encountered in production environments.

## Quick Diagnostic Checklist

When encountering pipeline issues, check these common causes first:

1. **Model Configuration**: Are the specified models available and properly configured?
2. **Template Rendering**: Are all template variables defined and accessible?
3. **Tool Registration**: Are all required tools properly registered and available?
4. **File Paths**: Do all input files exist and are output directories writable?
5. **Dependencies**: Are step dependencies correctly specified?
6. **API Keys**: Are all required API keys set in environment variables?

## Model Configuration Issues

### Issue: Model Not Found

#### Symptoms
```
Error: Model 'gpt-3.5-turbo' not found in registry
Error: openai/gpt-3.5-turbo is not available
```

#### Root Cause
- Model is not registered in the model registry
- Model identifier format is incorrect
- API credentials are missing or invalid

#### Solutions

**Check Model Registry**
```bash
# List available models
python -c "from orchestrator.models import get_model_registry; print(get_model_registry().list_models())"
```

**Fix Model Names**
```yaml
# Incorrect format
model: "gpt-3.5-turbo"

# Correct format
model: "openai/gpt-3.5-turbo"
```

**Use AUTO Tags for Reliability**
```yaml
# Instead of hardcoded model names
model: <AUTO task="analyze">Select best model for text analysis</AUTO>
```

**Register Custom Models**
```python
from orchestrator.models import get_model_registry

registry = get_model_registry()
registry.register_model(
    name="custom/my-model",
    provider="custom",
    config={
        "api_key": os.getenv("CUSTOM_API_KEY"),
        "endpoint": "https://api.custom.com/v1"
    }
)
```

### Issue: AUTO Tag Resolution Failures

#### Symptoms
```
Error: Failed to resolve AUTO tag for task 'analyze'
Warning: AUTO tag fallback to default model
```

#### Root Cause
- No suitable models available for the specified task
- Model capabilities don't match task requirements
- API connectivity issues during resolution

#### Solutions

**Provide Fallback Models**
```yaml
model: <AUTO task="analyze" fallback="openai/gpt-3.5-turbo">
  Select best analytical model with reliable fallback
</AUTO>
```

**Use Specific Task Descriptions**
```yaml
# Vague - may fail resolution
model: <AUTO task="process">Process data</AUTO>

# Specific - better resolution
model: <AUTO task="analyze" context="structured-data">
  Analyze structured JSON data and extract insights
</AUTO>
```

**Check Available Providers**
```yaml
# Ensure multiple providers are available
model: <AUTO task="generate" providers="openai,anthropic">
  Select from OpenAI or Anthropic models for text generation
</AUTO>
```

### Issue: Structured Output Problems

#### Symptoms
```
Error: Expected JSON output but received plain text
Error: Model does not support structured output format
```

#### Root Cause
- Model doesn't support JSON mode
- Prompt doesn't specify output format clearly
- Model returns conversational text instead of structured data

#### Solutions

**Use JSON-Capable Models**
```yaml
model: <AUTO task="extract" format="json">
  Select a model capable of generating valid JSON output
</AUTO>
```

**Clear Output Instructions**
```yaml
prompt: |
  Extract information and return ONLY valid JSON.
  
  Requirements:
  - Start with { and end with }
  - No markdown formatting
  - No code fences (```json)
  - No explanatory text
  
  Data: {{ input_data }}
```

**Validate and Clean Output**
```yaml
steps:
  - id: generate_structured_data
    action: generate_text
    
  - id: clean_json_output
    action: generate_text
    parameters:
      prompt: |
        Clean this response to be valid JSON only:
        {{ generate_structured_data.result }}
        
        Return ONLY the JSON data, no other text.
```

## Template Rendering Issues

### Issue: Undefined Variable Errors

#### Symptoms
```
Error: 'analyze_results' is undefined
jinja2.exceptions.UndefinedError: 'step_result' is undefined
Template rendering failed: {{ missing_variable }}
```

#### Root Cause
- Referenced step was skipped due to conditions
- Variable name typo or incorrect reference
- Step hasn't executed yet due to dependency issues

#### Solutions

**Use Safe Variable Access**
```yaml
# Unsafe - will fail if variable is undefined
content: "{{ analyze_results.summary }}"

# Safe - provides fallback
content: "{{ (analyze_results.summary) | default('No analysis available') }}"
```

**Check Variable Existence**
```yaml
content: |
  {% if analyze_results and analyze_results.summary %}
    Summary: {{ analyze_results.summary }}
  {% else %}
    Analysis not available or incomplete
  {% endif %}
```

**Handle Conditional Steps**
```yaml
steps:
  - id: optional_analysis
    if: "{{ enable_analysis }}"
    action: analyze_text
    
  - id: use_analysis_result
    action: generate_text
    parameters:
      prompt: |
        {% if optional_analysis and optional_analysis.result %}
          Based on analysis: {{ optional_analysis.result }}
        {% else %}
          Based on original data: {{ original_data }}
        {% endif %}
    dependencies: [optional_analysis]  # Wait for condition evaluation
```

### Issue: Loop Variable Errors

#### Symptoms
```
Error: '$item' is undefined
Error: Loop variable '$index' not available outside loop
```

#### Root Cause
- Loop variables used outside of loop context
- Incorrect loop variable syntax
- Loop not properly configured

#### Solutions

**Correct Loop Variable Usage**
```yaml
# Loop variables only available inside for_each steps
- id: process_items
  for_each: "{{ item_list }}"
  steps:
    - id: process_single_item
      action: process
      parameters:
        item: "{{ $item }}"           # Current item
        position: "{{ $index }}"     # Zero-based index
        is_first: "{{ $is_first }}"  # Boolean flags
        is_last: "{{ $is_last }}"
```

**Pass Loop Data to Other Steps**
```yaml
- id: collect_loop_results
  for_each: "{{ items }}"
  steps:
    - id: process_item
      action: process
      
  # Access collected results outside loop
- id: summarize_all_results
  action: summarize
  parameters:
    data: "{{ collect_loop_results.results }}"  # Array of all results
```

### Issue: Complex Template Logic

#### Symptoms
```
Error: Template too complex to render
Error: Recursive template inclusion
```

#### Root Cause
- Templates with complex nested logic
- Circular template references
- Large template files causing memory issues

#### Solutions

**Simplify Template Logic**
```yaml
# Complex template - move logic to separate step
content: |
  {% set processed_items = [] %}
  {% for item in complex_list %}
    {% if item.condition and item.value > threshold %}
      {% set _ = processed_items.append(transform_item(item)) %}
    {% endif %}
  {% endfor %}
  
# Better - use separate processing step
steps:
  - id: filter_and_process
    action: data_processing
    parameters:
      items: "{{ complex_list }}"
      filter: "value > {{ threshold }}"
      
  - id: simple_template
    parameters:
      content: "Processed {{ filter_and_process.results | length }} items"
```

**Use Template Inheritance**
```yaml
# base_template.j2
<!DOCTYPE html>
<html>
<head>
    <title>{% block title %}Default Title{% endblock %}</title>
</head>
<body>
    {% block content %}{% endblock %}
</body>
</html>

# specific_template.j2
{% extends "base_template.j2" %}
{% block title %}Pipeline Results{% endblock %}
{% block content %}
    <h1>Results</h1>
    {{ results_content }}
{% endblock %}
```

## Tool and API Issues

### Issue: Tool Not Registered

#### Symptoms
```
Error: Tool 'custom_tool' is not registered
Error: Action 'special_action' not found in tool registry
```

#### Root Cause
- Tool is not imported or registered at startup
- Tool name mismatch between pipeline and registration
- Tool module failed to load

#### Solutions

**Check Tool Registration**
```python
# In your initialization code
from orchestrator.tools import ToolRegistry
from my_custom_tools import CustomDataTool

tool_registry = ToolRegistry()
tool_registry.register(CustomDataTool())
```

**Verify Tool Names**
```bash
# List available tools
python -c "from orchestrator.tools import get_tool_registry; print(get_tool_registry().list_tools())"
```

**Debug Tool Loading**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show tool registration debug info
from orchestrator.tools import get_tool_registry
registry = get_tool_registry()
```

### Issue: API Rate Limiting

#### Symptoms
```
Error: Rate limit exceeded for API
Error: Too many requests (429)
Warning: API quota exhausted
```

#### Root Cause
- Too many concurrent API calls
- High frequency of requests to same endpoint
- API quota limits reached

#### Solutions

**Add Rate Limiting**
```yaml
# Limit concurrent API calls
- id: parallel_processing
  for_each: "{{ large_item_list }}"
  max_parallel: 3  # Reduce from default
  steps:
    - id: api_call
      action: external_api
      retry:
        attempts: 5
        delay: 10  # Wait between retries
```

**Implement Backoff Strategy**
```yaml
steps:
  - id: api_with_backoff
    action: api_call
    retry:
      attempts: 5
      delay: 2
      backoff_factor: 2  # Exponential backoff: 2, 4, 8, 16 seconds
    timeout: 120
```

**Use API Response Caching**
```yaml
steps:
  - id: cached_api_call
    action: external_api
    parameters:
      endpoint: "{{ api_endpoint }}"
      data: "{{ request_data }}"
    cache_key: "api_{{ request_hash }}"
    cache_ttl: 3600  # Cache for 1 hour
```

### Issue: File System Access Problems

#### Symptoms
```
Error: Permission denied writing to output directory
Error: File not found: 'input_data.json'
Error: Disk space insufficient
```

#### Root Cause
- Insufficient file system permissions
- Incorrect file paths (relative vs absolute)
- Disk space issues

#### Solutions

**Check File Permissions**
```bash
# Check directory permissions
ls -la /path/to/output/directory

# Fix permissions if needed
chmod 755 /path/to/output/directory
```

**Use Absolute Paths**
```yaml
# Problematic - relative paths
parameters:
  input_file: "data/input.json"
  output_dir: "../results"
  
# Better - absolute paths or use working directory
parameters:
  input_file: "{{ working_dir }}/data/input.json"
  output_dir: "{{ working_dir }}/results"
```

**Create Directories If Missing**
```yaml
steps:
  - id: ensure_output_directory
    tool: filesystem
    action: mkdir
    parameters:
      path: "{{ output_dir }}"
      create_parents: true
      
  - id: write_results
    dependencies: [ensure_output_directory]
    tool: filesystem
    action: write
    parameters:
      path: "{{ output_dir }}/results.json"
      content: "{{ processed_data }}"
```

## Control Flow Issues

### Issue: Infinite Loops

#### Symptoms
```
Warning: While loop exceeded maximum iterations (100)
Error: Pipeline timeout after 600 seconds
```

#### Root Cause
- While loop condition never becomes false
- Loop variables not properly updated
- Logic errors in loop termination conditions

#### Solutions

**Set Maximum Iterations**
```yaml
- id: safe_while_loop
  while: "{{ quality < threshold }}"
  max_iterations: 10  # Safety limit
  steps:
    - id: improve_quality
      action: enhance
      
  loop_vars:
    quality: "{{ improve_quality.quality_score }}"
```

**Debug Loop Conditions**
```yaml
- id: debug_while_loop
  while: "{{ quality < threshold and iterations < max_iterations }}"
  steps:
    - id: log_iteration
      action: generate_text
      parameters:
        prompt: |
          Iteration {{ iterations }}: 
          Quality: {{ quality }}
          Threshold: {{ threshold }}
          Continue: {{ quality < threshold }}
          
  loop_vars:
    iterations: "{{ iterations + 1 }}"
    quality: "{{ improve_quality.quality_score }}"
```

### Issue: Dependency Cycles

#### Symptoms
```
Error: Circular dependency detected: step1 -> step2 -> step1
Error: Cannot resolve step execution order
```

#### Root Cause
- Steps depend on each other in a cycle
- Complex dependency chains with circular references

#### Solutions

**Analyze Dependencies**
```bash
# Use pipeline analyzer to detect cycles
python scripts/analyze_pipeline.py my_pipeline.yaml --check-dependencies
```

**Refactor Dependencies**
```yaml
# Problematic - circular dependency
steps:
  - id: step1
    dependencies: [step2]
  - id: step2  
    dependencies: [step1]
    
# Fixed - linear dependency
steps:
  - id: prepare_data
  - id: process_data
    dependencies: [prepare_data]
  - id: finalize_data
    dependencies: [process_data]
```

## Data Processing Issues

### Issue: Empty or Invalid Data Results

#### Symptoms
```
Error: Data processing tool returned empty results
Error: Invalid data format for processing
Warning: No records found matching filter criteria
```

#### Root Cause
- Input data is empty or malformed
- Filters are too restrictive
- Data processing tool configuration issues

#### Solutions

**Validate Input Data**
```yaml
steps:
  - id: validate_input
    tool: validation
    action: validate
    parameters:
      data: "{{ input_data }}"
      schema:
        type: object
        properties:
          records:
            type: array
            minItems: 1  # Ensure at least one record
        required: ["records"]
    on_error: "stop"  # Fail fast if no valid data
```

**Handle Empty Results**
```yaml
steps:
  - id: process_data
    tool: data-processing
    action: filter
    parameters:
      data: "{{ input_data }}"
      criteria: "{{ filter_criteria }}"
      
  - id: check_results
    if: "{{ process_data.results | length == 0 }}"
    action: generate_text
    parameters:
      prompt: |
        No data found matching criteria: {{ filter_criteria }}
        Original data had {{ input_data.records | length }} records.
        Consider relaxing filter criteria.
```

**Debug Data Processing**
```yaml
steps:
  - id: debug_data_structure
    action: generate_text
    parameters:
      prompt: |
        Analyze data structure:
        Type: {{ input_data | type }}
        Length: {% if input_data is iterable %}{{ input_data | length }}{% else %}N/A{% endif %}
        Sample: {{ input_data | truncate(500) }}
        
        Is it a list? {{ input_data is iterable }}
        Is it empty? {{ input_data | length == 0 }}
```

### Issue: Memory Issues with Large Datasets

#### Symptoms
```
Error: MemoryError: Unable to allocate array
Warning: High memory usage detected
Error: Process killed due to memory limit
```

#### Root Cause
- Processing large datasets in memory
- No chunking or streaming for large files
- Memory leaks in processing loops

#### Solutions

**Use Streaming Processing**
```yaml
steps:
  - id: stream_large_file
    tool: data-processing
    action: stream_process
    parameters:
      file_path: "{{ large_data_file }}"
      chunk_size: 1000  # Process 1000 records at a time
      
  - id: batch_process
    for_each: "{{ stream_large_file.chunks }}"
    max_parallel: 2  # Limit memory usage
    steps:
      - id: process_chunk
        tool: data-processing
        action: process_chunk
        parameters:
          data: "{{ $item }}"
```

**Monitor Memory Usage**
```yaml
steps:
  - id: check_memory
    tool: system
    action: memory_info
    
  - id: adjust_batch_size
    if: "{{ check_memory.memory_usage > 0.8 }}"
    action: reduce_batch_size
    parameters:
      current_size: "{{ batch_size }}"
      target_memory: 0.6
```

## Performance Issues

### Issue: Slow Pipeline Execution

#### Symptoms
```
Warning: Pipeline execution time exceeded 10 minutes
Warning: Step 'large_processing' running for 300 seconds
```

#### Root Cause
- Inefficient processing algorithms
- No parallel processing for independent tasks
- Large data sets processed sequentially

#### Solutions

**Profile Pipeline Performance**
```yaml
steps:
  - id: start_profiling
    tool: system
    action: start_timer
    
  - id: expensive_operation
    action: complex_processing
    timeout: 300  # Set reasonable timeout
    
  - id: end_profiling
    tool: system
    action: end_timer
    parameters:
      start_time: "{{ start_profiling.timestamp }}"
```

**Optimize with Parallelization**
```yaml
# Sequential processing - slow
steps:
  - id: process_file1
    action: process_file
  - id: process_file2  
    action: process_file
  - id: process_file3
    action: process_file
    
# Parallel processing - faster
steps:
  - id: parallel_processing
    for_each: "{{ file_list }}"
    max_parallel: 3
    steps:
      - id: process_file
        action: process_file
        parameters:
          file: "{{ $item }}"
```

**Use Appropriate Timeouts**
```yaml
steps:
  - id: quick_operation
    timeout: 30  # 30 seconds for quick tasks
    
  - id: long_operation
    timeout: 600  # 10 minutes for complex tasks
    
  - id: critical_operation
    timeout: 1800  # 30 minutes for critical but slow tasks
    retry:
      attempts: 2
```

### Issue: Resource Exhaustion

#### Symptoms
```
Error: Too many open files
Error: CPU usage at 100% for extended period
Warning: Network connection pool exhausted
```

#### Root Cause
- No resource limits on parallel processing
- File handles not properly closed
- Connection pools not managed

#### Solutions

**Limit Concurrent Operations**
```yaml
# Control resource usage
- id: resource_intensive_processing
  for_each: "{{ large_item_list }}"
  max_parallel: 5  # Limit concurrent processing
  steps:
    - id: process_with_limits
      action: heavy_processing
      timeout: 120
      retry:
        attempts: 2
        delay: 30
```

**Implement Resource Monitoring**
```yaml
steps:
  - id: monitor_resources
    tool: system
    action: resource_check
    parameters:
      cpu_threshold: 0.8
      memory_threshold: 0.8
      
  - id: throttle_if_needed
    if: "{{ monitor_resources.cpu_usage > 0.8 }}"
    action: sleep
    parameters:
      seconds: 10
```

## Network and Connectivity Issues

### Issue: API Connection Failures

#### Symptoms
```
Error: Connection timeout to api.openai.com
Error: SSL certificate verification failed
Error: Network unreachable
```

#### Root Cause
- Network connectivity issues
- Firewall blocking API endpoints
- SSL/TLS configuration problems
- API endpoint changes

#### Solutions

**Implement Robust Error Handling**
```yaml
steps:
  - id: api_call_with_retry
    action: external_api_call
    retry:
      attempts: 5
      delay: 10
      backoff_factor: 1.5
    timeout: 60
    on_error: "continue"
    
  - id: fallback_processing
    if: "{{ not api_call_with_retry.success }}"
    action: local_processing
    parameters:
      data: "{{ input_data }}"
```

**Test Connectivity**
```yaml
steps:
  - id: test_api_connectivity
    tool: network
    action: ping
    parameters:
      host: "api.openai.com"
      timeout: 10
      
  - id: proceed_if_connected
    if: "{{ test_api_connectivity.success }}"
    action: api_processing
    
  - id: offline_mode
    if: "{{ not test_api_connectivity.success }}"
    action: local_processing
```

## Debugging Techniques

### Enable Debug Logging

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Use Debug Steps

```yaml
steps:
  - id: debug_variables
    action: generate_text
    parameters:
      prompt: |
        Debug information:
        - Variable1: {{ variable1 }}
        - Variable2 type: {{ variable2 | type }}
        - Step results: {{ previous_step | tojsonpretty }}
```

### Test Individual Steps

```yaml
# Create minimal test pipelines for debugging
name: "Debug Single Step"
steps:
  - id: isolated_test
    action: problematic_action
    parameters:
      # Minimal test data
      test_data: "{{ simple_test_input }}"
```

### Validate Pipeline Configuration

```bash
# Use pipeline validation tools
python scripts/validate_pipeline.py my_pipeline.yaml
```

## Getting Additional Help

### Documentation Resources
- [Best Practices Guide](best-practices.md) - Development best practices
- [Migration Guide](migration.md) - Upgrading from older versions  
- [Common Issues](common-issues.md) - Known limitations and workarounds
- [API Documentation](/docs/api_reference.md) - Complete API reference

### Debugging Tools
- `scripts/validate_pipeline.py` - Pipeline configuration validation
- `scripts/analyze_pipeline.py` - Dependency analysis and optimization
- `scripts/run_pipeline.py --debug` - Run with detailed debug output

### Community Support
- GitHub Issues: Report bugs and feature requests
- Documentation: Check example pipelines for similar use cases
- Validation Reports: Review validation results for common patterns

## Summary

Most pipeline issues fall into these categories:

1. **Configuration Issues** (40%) - Model setup, tool registration, API keys
2. **Template Problems** (25%) - Variable access, loop contexts, conditional logic
3. **Data Issues** (20%) - Empty results, format problems, memory issues  
4. **Network/API Issues** (10%) - Connectivity, rate limits, timeouts
5. **Control Flow Issues** (5%) - Dependencies, loops, conditionals

When troubleshooting:
1. Start with the diagnostic checklist
2. Check logs for specific error messages
3. Use debug steps to inspect variables and data flow
4. Test individual components in isolation
5. Refer to working example pipelines for patterns

Remember: Most issues can be prevented by following the best practices in pipeline design, error handling, and testing.