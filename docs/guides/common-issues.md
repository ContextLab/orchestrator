# Common Issues and Workarounds

## Overview

This guide documents known issues, limitations, and workarounds in the Orchestrator framework. Based on analysis of 40+ example pipelines and production deployments, it provides practical solutions for common problems that don't have permanent fixes yet.

**Note**: This guide complements the [Troubleshooting Guide](troubleshooting.md) by focusing on **known limitations** rather than configuration errors.

## Model-Related Issues

### Issue: AUTO Tag Resolution Limitations

#### Problem Description
AUTO tags sometimes fail to resolve to optimal models, especially for complex tasks or when specific model capabilities are required.

**Affected Scenarios**:
- Complex multi-step reasoning tasks
- Structured output generation (JSON/YAML)
- Domain-specific analysis (legal, medical, technical)
- Large context requirements

#### Current Limitations
```yaml
# May not select optimal model for complex analysis
model: <AUTO task="analyze">Analyze complex financial data</AUTO>

# May fail with models that don't support structured output
model: <AUTO task="extract" format="json">Extract to JSON</AUTO>
```

#### Workarounds

**1. Use Specific Model Fallbacks**
```yaml
# Provide reliable fallback models
model: <AUTO task="analyze" fallback="openai/gpt-4">
  Complex financial analysis requiring advanced reasoning
</AUTO>
```

**2. Task-Specific Model Pools**
```yaml
# Restrict to known-good models for specific tasks
model: <AUTO task="code-review" providers="openai,anthropic">
  Code review requiring high accuracy
</AUTO>
```

**3. Manual Override for Critical Tasks**
```yaml
# Use manual selection for mission-critical operations
model: "openai/gpt-4"  # Direct model specification
# Alternative: Use conditional model selection
model: |
  {% if task_complexity == "high" %}
    openai/gpt-4
  {% else %}
    <AUTO task="analyze">Standard analysis task</AUTO>
  {% endif %}
```

**4. Multi-Stage Processing**
```yaml
steps:
  # Stage 1: Use AUTO for initial processing
  - id: initial_analysis
    model: <AUTO task="analyze">Initial analysis</AUTO>
    
  # Stage 2: Use specific model for complex parts
  - id: detailed_analysis
    model: "openai/gpt-4"
    if: "{{ initial_analysis.complexity_score > 0.8 }}"
```

#### Monitoring AUTO Tag Performance

```yaml
steps:
  - id: log_model_selection
    action: generate_text
    parameters:
      prompt: |
        Model Selection Debug:
        - AUTO Tag: {{ auto_tag_request }}
        - Selected Model: {{ selected_model }}
        - Selection Criteria: {{ selection_reason }}
        - Available Models: {{ available_models }}
```

---

### Issue: JSON Mode Compatibility

#### Problem Description
Not all models support JSON mode or structured output, leading to parsing errors or malformed responses.

**Affected Models**:
- Smaller open-source models (Llama 7B, etc.)
- Older model versions
- Some local/offline models

#### Current Limitations
```yaml
# May fail with models that don't support JSON mode
- id: extract_data
  action: generate_text
  parameters:
    prompt: "Extract data as JSON: {{ input_text }}"
    format: "json"  # Not supported by all models
```

#### Workarounds

**1. Fallback Chain Processing**
```yaml
steps:
  # Try JSON-capable model first
  - id: try_structured_extraction
    model: <AUTO task="extract" format="json">JSON extraction</AUTO>
    action: generate_text
    parameters:
      prompt: |
        Extract information as valid JSON:
        {{ input_text }}
        
        Return ONLY JSON, no other text.
    on_error: "continue"
    
  # Fallback to text extraction + parsing
  - id: text_extraction_fallback
    if: "{{ not try_structured_extraction.success }}"
    model: <AUTO task="extract">Text extraction</AUTO>
    action: generate_text
    parameters:
      prompt: |
        Extract information from this text:
        {{ input_text }}
        
        Format as:
        Name: [value]
        Age: [value]
        Location: [value]
        
  - id: parse_text_to_json
    if: "{{ text_extraction_fallback.success }}"
    action: generate_text
    parameters:
      prompt: |
        Convert this structured text to valid JSON:
        {{ text_extraction_fallback.result }}
        
        Return only the JSON object.
```

**2. Progressive Enhancement**
```yaml
steps:
  - id: test_json_capability
    model: "{{ target_model }}"
    action: generate_text
    parameters:
      prompt: 'Return this as JSON: {"test": "value"}'
    on_error: "continue"
    
  - id: use_json_mode
    if: "{{ test_json_capability.success }}"
    model: "{{ target_model }}"
    parameters:
      format: "json"
      
  - id: use_text_mode
    if: "{{ not test_json_capability.success }}"
    model: "{{ target_model }}"
    parameters:
      prompt: "Return as plain text, we'll parse it later"
```

---

## Template Rendering Issues

### Issue: Complex Nested Object Access

#### Problem Description
Deep nested object access in templates can fail silently or produce unexpected results, especially with optional or dynamic data structures.

**Problematic Patterns**:
```yaml
# May fail if any level is missing
content: "{{ analysis.results.categories[0].subcategories.items.data.value }}"

# Complex conditionals that are hard to debug
content: |
  {% if analysis and analysis.results and analysis.results.data %}
    {% if analysis.results.data.items|length > 0 %}
      {{ analysis.results.data.items[0].value }}
    {% endif %}
  {% endif %}
```

#### Workarounds

**1. Safe Navigation with Default Filters**
```yaml
# Use default filters at each level
content: |
  {{
    (((analysis.results | default({}))
      .categories | default([{}]))
      [0].subcategories | default({})
    ).items.data.value | default('N/A')
  }}
```

**2. Pre-Processing Steps**
```yaml
steps:
  # Extract nested data in separate step
  - id: extract_nested_value
    action: generate_text
    parameters:
      prompt: |
        Extract the nested value from this data structure:
        {{ analysis.results | tojsonpretty }}
        
        Path: categories[0].subcategories.items.data.value
        
        If path doesn't exist, return "N/A"
        Return only the value, no explanation.
        
  - id: use_extracted_value
    parameters:
      content: "Value: {{ extract_nested_value.result }}"
```

**3. Flattening Complex Data**
```yaml
steps:
  - id: flatten_analysis_data
    action: generate_text
    parameters:
      prompt: |
        Flatten this nested data structure:
        {{ complex_analysis_result }}
        
        Create a simple key-value structure with these fields:
        - primary_value
        - secondary_value  
        - status
        - confidence_score
        
        Return as JSON object with only these fields.
```

---

### Issue: Loop Variable Scope

#### Problem Description
Loop variables ($item, $index, etc.) are not available outside of their loop context, making it difficult to use loop results in subsequent steps.

#### Current Limitations
```yaml
- id: process_items
  for_each: "{{ item_list }}"
  steps:
    - id: process_item
      parameters:
        item_id: "{{ $item.id }}"  # Available here
        
- id: use_loop_results
  parameters:
    # These are NOT available here:
    # last_item: "{{ $item }}"     # Error: undefined
    # total_items: "{{ $index }}"  # Error: undefined
    results: "{{ process_items.results }}"  # This works
```

#### Workarounds

**1. Collect Loop Metadata**
```yaml
- id: process_with_metadata
  for_each: "{{ item_list }}"
  steps:
    - id: process_item
      action: process
      
    - id: collect_metadata
      action: generate_text
      parameters:
        prompt: |
          Item {{ $index }} of {{ item_list | length }}:
          ID: {{ $item.id }}
          Is first: {{ $is_first }}
          Is last: {{ $is_last }}
          
- id: access_collected_metadata
  parameters:
    metadata: "{{ process_with_metadata.results }}"
    # Access collected metadata from all iterations
```

**2. Store Loop State in Variables**
```yaml
# Initialize counters
- id: init_counters
  action: generate_text
  parameters:
    prompt: |
      Initialize processing counters:
      total_items: {{ item_list | length }}
      processed_count: 0
      failed_count: 0
      
- id: process_with_state_tracking
  for_each: "{{ item_list }}"
  steps:
    - id: process_item
      # Process individual item
      
    - id: update_state
      action: generate_text  
      parameters:
        prompt: |
          Update state after processing item {{ $index }}:
          Current progress: {{ ($index + 1) / (item_list | length) * 100 }}%
          Items remaining: {{ (item_list | length) - ($index + 1) }}
```

---

## Control Flow Issues

### Issue: While Loop Performance Degradation

#### Problem Description
While loops can become progressively slower as they execute more iterations, especially with complex template rendering or large data sets.

**Performance Impact**:
- Template rendering time increases with each iteration
- Memory usage grows with accumulated state
- API calls may slow due to increased context size

#### Current Limitations
```yaml
# Performance degrades after many iterations
- id: iterative_improvement
  while: "{{ quality_score < target_score }}"
  steps:
    - id: improve_content
      parameters:
        # Context grows with each iteration
        full_history: "{{ all_previous_results | join('\n') }}"
        current_content: "{{ content }}"
```

#### Workarounds

**1. Limit Context Growth**
```yaml
- id: iterative_improvement
  while: "{{ quality_score < target_score and iterations < 10 }}"
  steps:
    - id: improve_content
      parameters:
        # Only use recent history
        recent_history: "{{ previous_results | last(3) | join('\n') }}"
        current_content: "{{ content }}"
        
    - id: cleanup_old_data
      if: "{{ iterations > 5 }}"
      action: generate_text
      parameters:
        prompt: "Keep only the most recent improvements"
```

**2. Checkpoint and Reset**
```yaml
- id: iterative_with_checkpoints
  while: "{{ continue_processing }}"
  steps:
    - id: process_batch
      # Process in small batches
      
    - id: checkpoint_progress
      if: "{{ iterations % 5 == 0 }}"  # Every 5 iterations
      tool: filesystem
      action: write
      parameters:
        path: "checkpoint_{{ iterations }}.json"
        content: "{{ current_state }}"
        
    - id: reset_context
      if: "{{ iterations % 5 == 0 }}"
      action: generate_text
      parameters:
        prompt: "Reset context, load from checkpoint"
```

**3. Alternative: Bounded Iteration with Continuation**
```yaml
steps:
  # Phase 1: Limited iterations
  - id: initial_processing
    while: "{{ quality < target and iterations < 5 }}"
    steps:
      - id: improve
        
  # Phase 2: Continue if needed
  - id: continued_processing
    if: "{{ initial_processing.final_quality < target }}"
    while: "{{ quality < target and iterations < 5 }}"
    steps:
      - id: advanced_improve
        parameters:
          # Start fresh with best result so far
          base_content: "{{ initial_processing.best_result }}"
```

---

## Data Processing Issues

### Issue: Large Dataset Memory Issues

#### Problem Description
Processing large datasets (>10MB) can cause memory exhaustion, especially with complex transformations or multiple processing steps.

**Memory Pressure Scenarios**:
- Large JSON/CSV files loaded entirely into memory
- Complex data transformations creating multiple copies
- For-each loops over thousands of items

#### Current Limitations
```yaml
# Loads entire file into memory
- id: load_large_data
  tool: filesystem
  action: read
  parameters:
    path: "{{ large_data_file }}"  # 100MB+ file
    
# Creates multiple copies in memory
- id: transform_all_data
  for_each: "{{ load_large_data.records }}"  # 100k+ records
  steps:
    - id: complex_transform
      # Each iteration keeps data in memory
```

#### Workarounds

**1. Streaming/Chunked Processing**
```yaml
# Process in manageable chunks
- id: get_file_info
  tool: filesystem
  action: info
  parameters:
    path: "{{ large_data_file }}"
    
- id: process_in_chunks
  if: "{{ get_file_info.size > 10000000 }}"  # 10MB threshold
  tool: data-processing
  action: stream_process
  parameters:
    file_path: "{{ large_data_file }}"
    chunk_size: 1000
    
- id: process_normally
  if: "{{ get_file_info.size <= 10000000 }}"
  tool: filesystem
  action: read
```

**2. Pagination Pattern**
```yaml
- id: process_paginated
  while: "{{ has_more_data }}"
  steps:
    - id: load_page
      tool: data-processing
      action: read_page
      parameters:
        file: "{{ data_file }}"
        offset: "{{ current_offset }}"
        limit: 1000
        
    - id: process_page
      for_each: "{{ load_page.records }}"
      max_parallel: 3  # Limit memory usage
      steps:
        - id: transform_record
          
  loop_vars:
    current_offset: "{{ current_offset + 1000 }}"
    has_more_data: "{{ load_page.records | length == 1000 }}"
```

**3. Temporary File Strategy**
```yaml
- id: create_temp_processing_dir
  tool: filesystem
  action: mkdir
  parameters:
    path: "{{ temp_dir }}/processing_{{ timestamp }}"
    
- id: split_large_file
  tool: data-processing
  action: split
  parameters:
    input_file: "{{ large_data_file }}"
    output_dir: "{{ temp_dir }}/processing_{{ timestamp }}"
    chunk_size: 5000
    
- id: process_chunks
  for_each: "{{ split_large_file.chunk_files }}"
  max_parallel: 2
  steps:
    - id: process_chunk
      tool: data-processing
      action: transform
      parameters:
        file: "{{ $item }}"
        
- id: combine_results
  tool: data-processing
  action: combine
  parameters:
    input_files: "{{ process_chunks.results }}"
    output_file: "{{ final_output_file }}"
    
- id: cleanup_temp_files
  tool: filesystem
  action: delete
  parameters:
    path: "{{ temp_dir }}/processing_{{ timestamp }}"
    recursive: true
```

---

## API and Network Issues

### Issue: Inconsistent API Response Formats

#### Problem Description
Different models and providers return responses in slightly different formats, making it difficult to write robust parsing logic.

**Variation Examples**:
- Some models include metadata, others don't
- Response wrapping varies (plain text vs JSON objects)
- Error format inconsistencies
- Content-Type header differences

#### Current Limitations
```yaml
# Parsing may fail depending on provider
- id: parse_api_response
  action: extract_data
  parameters:
    # Response format varies:
    # OpenAI: {"choices": [{"message": {"content": "..."}}]}
    # Anthropic: {"content": "..."}
    # Local: "plain text response"
    response: "{{ api_call.result }}"
```

#### Workarounds

**1. Normalized Response Parsing**
```yaml
steps:
  - id: normalize_api_response
    action: generate_text
    parameters:
      prompt: |
        Normalize this API response to extract just the content:
        {{ api_call.result }}
        
        Return only the main content/message text, no metadata.
        If it's JSON, extract the content field.
        If it's plain text, return as-is.
        
  - id: process_normalized_content
    parameters:
      content: "{{ normalize_api_response.result }}"
```

**2. Provider-Specific Handlers**
```yaml
steps:
  - id: detect_response_format
    action: generate_text
    parameters:
      prompt: |
        Identify the format of this response:
        {{ api_call.result | truncate(200) }}
        
        Is it:
        A) OpenAI format (JSON with choices array)
        B) Anthropic format (JSON with content field)  
        C) Plain text
        D) Other JSON format
        
        Return only the letter (A, B, C, or D).
        
  - id: parse_openai_format
    if: "{{ detect_response_format.result == 'A' }}"
    action: extract_data
    parameters:
      data: "{{ api_call.result }}"
      path: "choices[0].message.content"
      
  - id: parse_anthropic_format
    if: "{{ detect_response_format.result == 'B' }}"
    action: extract_data
    parameters:
      data: "{{ api_call.result }}"
      path: "content"
      
  - id: use_plain_text
    if: "{{ detect_response_format.result == 'C' }}"
    parameters:
      content: "{{ api_call.result }}"
```

**3. Robust Response Extraction**
```yaml
- id: extract_content_safely
  action: generate_text
  parameters:
    prompt: |
      Extract the main text content from this API response:
      
      ```
      {{ api_call.result }}
      ```
      
      Rules:
      1. If it's JSON, find the actual text content (not metadata)
      2. If it's plain text, return it directly
      3. Remove any system messages or formatting
      4. Return only the core response content
      5. If extraction fails, return "EXTRACTION_FAILED"
      
      Content:
```

---

### Issue: Rate Limit Handling Inconsistencies

#### Problem Description
Different API providers have different rate limiting behaviors, error codes, and recovery mechanisms.

**Provider Differences**:
- OpenAI: 429 status with retry-after header
- Anthropic: 429 status with different error format
- Local models: No rate limiting but may have resource constraints
- Custom APIs: Varied implementations

#### Workarounds

**1. Universal Rate Limit Handler**
```yaml
steps:
  - id: api_call_with_universal_retry
    action: external_api_call
    retry:
      attempts: 5
      delay: 30  # Conservative initial delay
      backoff_factor: 2
      retry_conditions:
        - "status_code == 429"  # Rate limit
        - "status_code == 503"  # Service unavailable  
        - "status_code >= 500"  # Server errors
    timeout: 300
    
  - id: handle_rate_limit_failure
    if: "{{ not api_call_with_universal_retry.success }}"
    action: generate_text
    parameters:
      prompt: |
        API call failed after retries.
        Error: {{ api_call_with_universal_retry.error }}
        
        Suggest alternative approaches:
        1. Use cached response if available
        2. Use alternative API provider
        3. Simplify the request
        4. Process in smaller batches
```

**2. Adaptive Delay Strategy**
```yaml
- id: track_api_performance
  # Keep track of API call success rates
  
- id: calculate_adaptive_delay
  action: generate_text
  parameters:
    prompt: |
      Calculate delay based on recent API performance:
      Success rate: {{ api_success_rate }}%
      Recent failures: {{ recent_failures }}
      Provider: {{ api_provider }}
      
      Suggest delay in seconds (1-300):
      - High success rate (>90%): 1-5 seconds
      - Medium success rate (70-90%): 10-30 seconds  
      - Low success rate (<70%): 60-300 seconds
      
- id: api_call_with_adaptive_delay
  action: external_api_call
  retry:
    delay: "{{ calculate_adaptive_delay.result }}"
```

---

## Performance Issues

### Issue: Template Rendering Performance

#### Problem Description
Complex templates with many variables, loops, and conditionals can become slow to render, especially in loops or repeated operations.

**Performance Bottlenecks**:
- Large template files with complex logic
- Repeated template compilation
- Deep nested object access
- Complex Jinja2 filters and functions

#### Workarounds

**1. Template Caching**
```yaml
# Pre-compile frequently used templates
- id: prepare_report_template
  action: compile_template
  parameters:
    template: "{{ report_template_path }}"
  cache_key: "report_template_v1"
  
- id: use_cached_template
  action: render_template
  parameters:
    compiled_template: "{{ prepare_report_template.compiled }}"
    data: "{{ report_data }}"
```

**2. Template Simplification**
```yaml
# Instead of complex template logic
template_old: |
  {% for item in items %}
    {% if item.category == 'important' %}
      {% for detail in item.details %}
        {% if detail.value > threshold %}
          Processing: {{ item.name }} - {{ detail.name }}
          Value: {{ detail.value | format_currency }}
          Status: {{ detail.status | title }}
        {% endif %}
      {% endfor %}
    {% endif %}
  {% endfor %}

# Use preprocessing steps
steps:
  - id: filter_important_items
    action: filter_data
    parameters:
      data: "{{ items }}"
      criteria:
        category: "important"
        detail_value_gt: "{{ threshold }}"
        
  - id: simple_template
    parameters:
      template: |
        {% for item in filtered_items %}
          Processing: {{ item.name }} - {{ item.detail_name }}
          Value: {{ item.formatted_value }}
          Status: {{ item.formatted_status }}
        {% endfor %}
      data: "{{ filter_important_items.result }}"
```

**3. Batch Template Rendering**
```yaml
# Instead of rendering templates in loops
- id: render_individual_reports
  for_each: "{{ customers }}"
  steps:
    - id: render_customer_report  # Slow - renders template each time
      parameters:
        template: "{{ complex_report_template }}"
        customer: "{{ $item }}"

# Batch render all reports at once        
- id: prepare_all_report_data
  action: generate_text
  parameters:
    prompt: |
      Prepare report data for all customers:
      {{ customers | tojsonpretty }}
      
      Return as array of report data objects.
      
- id: batch_render_reports
  action: render_template_batch
  parameters:
    template: "{{ complex_report_template }}"
    data_array: "{{ prepare_all_report_data.result }}"
```

---

## Known Limitations

### Tool Availability

**Issue**: Some tools may not be available in all environments or configurations.

**Affected Tools**:
- `validation` tool - May not be registered by default
- Custom tools - Require manual registration
- System-specific tools - OS dependencies

**Workaround**:
```yaml
steps:
  - id: check_tool_availability
    action: list_tools
    
  - id: use_preferred_tool
    if: "{{ 'validation' in check_tool_availability.tools }}"
    tool: validation
    action: validate
    
  - id: use_fallback_validation
    if: "{{ 'validation' not in check_tool_availability.tools }}"
    action: generate_text
    parameters:
      prompt: |
        Validate this data manually:
        {{ data_to_validate }}
        
        Check for required fields and data types.
        Return "VALID" or "INVALID" with explanation.
```

### Model Context Limits

**Issue**: Long pipelines or large data sets may exceed model context windows.

**Current Limits**:
- GPT-3.5-turbo: ~4k tokens
- GPT-4: ~8k tokens (varies by version)
- Claude: ~100k tokens
- Local models: Varies widely

**Workaround**:
```yaml
- id: check_context_size
  action: estimate_tokens
  parameters:
    text: "{{ full_context }}"
    
- id: use_full_context
  if: "{{ check_context_size.tokens < 7000 }}"
  model: "openai/gpt-4"
  
- id: use_summarized_context
  if: "{{ check_context_size.tokens >= 7000 }}"
  steps:
    - id: summarize_context
      model: "openai/gpt-4"
      parameters:
        prompt: |
          Summarize this context to fit in 6000 tokens:
          {{ full_context }}
          
    - id: process_with_summary
      model: "openai/gpt-4"
      parameters:
        context: "{{ summarize_context.result }}"
```

## Monitoring and Diagnostics

### Performance Monitoring

```yaml
# Add performance monitoring to pipelines
- id: start_performance_tracking
  tool: system
  action: start_timer
  
# ... pipeline steps ...

- id: end_performance_tracking
  tool: system
  action: end_timer
  parameters:
    start_time: "{{ start_performance_tracking.timestamp }}"
    
- id: log_performance_metrics
  if: "{{ end_performance_tracking.duration > 300 }}"  # Log if > 5 minutes
  action: generate_text
  parameters:
    prompt: |
      Performance Warning - Pipeline took {{ end_performance_tracking.duration }} seconds
      
      Consider optimizations:
      - Reduce parallel processing
      - Simplify templates
      - Cache intermediate results
      - Use smaller data chunks
```

### Error Pattern Detection

```yaml
- id: detect_common_error_patterns
  if: "{{ pipeline_errors | length > 0 }}"
  action: generate_text
  parameters:
    prompt: |
      Analyze these pipeline errors for common patterns:
      {{ pipeline_errors | tojsonpretty }}
      
      Common issues to check:
      1. Model availability
      2. Template variable access
      3. Tool registration
      4. Data format problems
      5. Network connectivity
      
      Suggest specific fixes for detected patterns.
```

## Future Improvements

The following items are being tracked for future releases:

1. **Enhanced AUTO Tag Resolution**: Better model selection algorithms
2. **Streaming Data Processing**: Native support for large datasets
3. **Template Performance**: Compiled template caching
4. **Universal API Adapters**: Normalized response handling
5. **Resource Management**: Automatic memory and API rate limit handling
6. **Advanced Error Recovery**: Intelligent fallback strategies

## Getting Help

### When to Use This Guide
- You're encountering a known limitation
- Standard troubleshooting didn't resolve the issue
- You need a workaround for a specific scenario
- You want to understand framework limitations before designing

### Other Resources
- [Troubleshooting Guide](troubleshooting.md) - Configuration and usage errors
- [Best Practices Guide](best-practices.md) - Development guidelines
- [Migration Guide](migration.md) - Version upgrade instructions

### Reporting Issues
If you encounter an issue not covered here:
1. Check if it's a configuration problem (use troubleshooting guide)
2. Search existing GitHub issues
3. Create a new issue with:
   - Pipeline YAML that reproduces the issue
   - Error messages and logs
   - Environment information
   - Expected vs actual behavior

## Summary

This guide covers the most common limitations and workarounds in the Orchestrator framework. Key takeaways:

1. **AUTO Tags**: Use fallback models for critical operations
2. **Templates**: Simplify complex logic, use safe navigation
3. **Large Data**: Implement chunking and streaming patterns
4. **API Calls**: Use universal retry logic and response normalization
5. **Performance**: Monitor execution time and optimize bottlenecks

Most limitations can be worked around with the patterns described here. As the framework evolves, many of these issues will be addressed in future versions.