# AUTO Tags: Dynamic Intelligence in Your Pipelines

AUTO tags are one of Orchestrator's most powerful features, allowing you to delegate decisions to AI models at runtime. This guide covers everything you need to know about using AUTO tags effectively.

## What are AUTO Tags?

AUTO tags are special markers in your pipeline YAML that get resolved by an AI model during execution. They allow you to create dynamic, intelligent pipelines that adapt based on context.

### Basic Syntax

```yaml
parameter: <AUTO>Your prompt here</AUTO>
```

## Real-World Examples

### Example 1: Dynamic Data Analysis

```yaml
name: dynamic-data-analyzer
description: Analyze data with AI-determined methods
version: "1.0.0"

inputs:
  data_file:
    type: string
    description: Path to data file
    required: true

steps:
  - id: read_data
    action: read-file
    parameters:
      path: "{{ data_file }}"
      
  - id: analyze_data
    action: llm-generate
    parameters:
      prompt: |
        Analyze this data and determine the best analysis approach:
        {{ read_data.content }}
      analysis_type: <AUTO>Based on the data, should we use 'statistical', 'qualitative', or 'mixed' analysis?</AUTO>
      
  - id: generate_report
    action: report-generator
    parameters:
      title: "Data Analysis Report"
      format: <AUTO>Choose the best format for this data: 'markdown', 'html', or 'pdf'</AUTO>
      content: "{{ analyze_data.result }}"
```

### Example 2: Intelligent Error Handling

```yaml
name: smart-error-handler
description: Handle errors intelligently based on context
version: "1.0.0"

steps:
  - id: risky_operation
    action: web-search
    parameters:
      query: "latest AI news"
    error_handling:
      retry:
        max_attempts: <AUTO>Based on the importance of web search for AI news, how many retry attempts should we make? Choose between 1-5.</AUTO>
        
  - id: handle_failure
    action: llm-generate
    condition: "{{ risky_operation.status == 'failed' }}"
    parameters:
      prompt: "The web search failed. What should we do?"
      fallback_action: <AUTO>Should we 'skip', 'use_cached_data', or 'try_alternative_source'?</AUTO>
```

### Example 3: Dynamic Tool Selection

```yaml
name: smart-researcher
description: Research with dynamically selected tools
version: "1.0.0"

inputs:
  topic:
    type: string
    description: Research topic
    required: true

steps:
  - id: determine_approach
    action: llm-analyze
    parameters:
      prompt: "Research topic: {{ topic }}"
      questions:
        - <AUTO>Is this topic better researched using 'web_search', 'academic_sources', or 'both'?</AUTO>
        - <AUTO>Should we include visual data? Answer 'yes' or 'no'.</AUTO>
        
  - id: web_research
    action: web-search
    condition: "'web_search' in determine_approach.result or 'both' in determine_approach.result"
    parameters:
      query: "{{ topic }}"
      num_results: <AUTO>How many search results would be appropriate for the topic '{{ topic }}'? Choose between 3-10.</AUTO>
```

## AUTO Tags in Control Flow

AUTO tags are particularly powerful when used with control flow structures:

### Conditional Execution

```yaml
steps:
  - id: check_data_quality
    action: validation
    parameters:
      data: "{{ input_data }}"
      
  - id: process_data
    action: data-processing
    condition: <AUTO>Based on the validation results {{ check_data_quality.result }}, should we proceed with processing? Answer 'true' or 'false'.</AUTO>
    parameters:
      method: <AUTO>What processing method is best for this data quality level?</AUTO>
```

### Dynamic Loops

```yaml
steps:
  - id: iterative_improvement
    action: for-each
    parameters:
      items: <AUTO>Based on the task complexity, how many iterations should we perform? Return a list like ['iteration_1', 'iteration_2', ...]</AUTO>
      steps:
        - id: improve
          action: llm-generate
          parameters:
            prompt: "Improve the result from iteration {{ item }}"
```

## Best Practices

### 1. Be Specific and Constrained

**Good:**
```yaml
format: <AUTO>Choose output format: 'json', 'yaml', or 'xml'</AUTO>
```

**Bad:**
```yaml
format: <AUTO>Choose a format</AUTO>
```

### 2. Provide Context

**Good:**
```yaml
analysis_depth: <AUTO>Given that this is a {{ data_size }}MB dataset with {{ num_columns }} columns, choose analysis depth: 'quick' (5 min), 'standard' (15 min), or 'comprehensive' (1 hour)</AUTO>
```

**Bad:**
```yaml
analysis_depth: <AUTO>Choose analysis depth</AUTO>
```

### 3. Use Type Hints

```yaml
num_retries: <AUTO type="integer" min="1" max="5">How many retries for this critical operation?</AUTO>
include_images: <AUTO type="boolean">Should we include images in the report?</AUTO>
categories: <AUTO type="array">List relevant categories for this content</AUTO>
```

### 4. Handle AUTO Tag Failures

```yaml
steps:
  - id: smart_decision
    action: llm-analyze
    parameters:
      decision: <AUTO>Choose processing strategy</AUTO>
    error_handling:
      fallback:
        decision: "standard"  # Default if AUTO resolution fails
```

## Advanced AUTO Tag Features

### Nested AUTO Tags

```yaml
steps:
  - id: dynamic_pipeline
    action: sub-pipeline
    parameters:
      pipeline_id: <AUTO>Which pipeline should we run: 'simple_analysis' or 'complex_analysis'?</AUTO>
      config:
        model: <AUTO>Based on the chosen pipeline {{ pipeline_id }}, which model is best?</AUTO>
```

### AUTO Tags with Templates

```yaml
steps:
  - id: template_decision
    action: llm-generate
    parameters:
      template: |
        {% if <AUTO>Is this content technical? Answer 'true' or 'false'</AUTO> == 'true' %}
          Use technical language
        {% else %}
          Use simple language
        {% endif %}
```

### Contextual AUTO Tags

```yaml
steps:
  - id: contextual_decision
    action: llm-analyze
    parameters:
      context:
        previous_results: "{{ previous_step.result }}"
        user_preferences: "{{ inputs.preferences }}"
        system_resources: "{{ system.available_memory }}"
      decision: <AUTO>Considering all the context provided, what's the optimal batch size?</AUTO>
```

## Performance Considerations

1. **Caching**: AUTO tag resolutions are cached within a pipeline run
2. **Model Selection**: Use smaller models for simple decisions
3. **Batching**: Group related AUTO tags to reduce API calls

```yaml
steps:
  - id: batch_decisions
    action: llm-analyze
    parameters:
      decisions:
        - output_format: <AUTO>Best format for the output?</AUTO>
        - include_summary: <AUTO>Should we include a summary?</AUTO>
        - detail_level: <AUTO>Level of detail: 'low', 'medium', or 'high'?</AUTO>
```

## Debugging AUTO Tags

Enable debug logging to see AUTO tag resolutions:

```yaml
config:
  debug: true
  log_auto_resolutions: true
```

This will log:
- The prompt sent to the model
- The model's response
- The parsed value used in the pipeline

## Testing AUTO Tags

When testing pipelines with AUTO tags, you can use the `--auto-responses` flag:

```bash
orchestrator run pipeline.yaml --auto-responses responses.json
```

Where `responses.json` contains predetermined responses:

```json
{
  "step_id.parameter_name": "predetermined_value",
  "analyze_data.analysis_type": "statistical"
}
```

## Common Patterns

### 1. Progressive Enhancement

```yaml
steps:
  - id: basic_analysis
    action: data-analysis
    parameters:
      level: "basic"
      
  - id: enhance_if_needed
    action: data-analysis
    condition: <AUTO>Based on basic results {{ basic_analysis.result }}, do we need deeper analysis? 'true' or 'false'</AUTO>
    parameters:
      level: "advanced"
```

### 2. Resource Optimization

```yaml
steps:
  - id: check_resources
    action: system-info
    
  - id: choose_model
    action: llm-select
    parameters:
      model: <AUTO>Given {{ check_resources.available_memory }}GB RAM, choose model: 'small', 'medium', or 'large'</AUTO>
```

### 3. Error Recovery

```yaml
steps:
  - id: primary_source
    action: web-fetch
    parameters:
      url: "{{ primary_url }}"
    error_handling:
      on_error:
        - id: choose_alternative
          action: llm-decide
          parameters:
            alternative: <AUTO>Primary source failed. Choose alternative: 'cache', 'mirror', or 'skip'</AUTO>
```

## Integration with Model Routing

AUTO tags work seamlessly with Orchestrator's intelligent model routing:

```yaml
config:
  model_selection:
    auto_tags:
      simple_decisions: "gpt-3.5-turbo"  # Fast, cheap model
      complex_analysis: "gpt-4"          # Powerful model
      creative_tasks: "claude-3-opus"    # Creative model

steps:
  - id: simple_choice
    action: decide
    parameters:
      choice: <AUTO model_hint="simple_decisions">Choose 'yes' or 'no'</AUTO>
      
  - id: complex_reasoning
    action: analyze
    parameters:
      analysis: <AUTO model_hint="complex_analysis">Perform deep analysis of this data...</AUTO>
```

## Limitations and Considerations

1. **Determinism**: AUTO tags introduce non-determinism. Same input might produce different outputs.
2. **Cost**: Each AUTO tag resolution is an API call. Budget accordingly.
3. **Latency**: AUTO tag resolution adds latency. Consider caching for repeated runs.
4. **Error Handling**: Always have fallbacks for critical AUTO tags.

## Next Steps

- See [Pipeline Examples](../examples/) for more AUTO tag usage
- Learn about [Model Configuration](./model_configuration.md) for AUTO tag optimization
- Explore [Advanced Control Flow](./control_flow.md) with AUTO tags