# YAML Pipeline Definition Guide

## Overview

The Orchestrator framework uses YAML files to define AI pipelines declaratively. This guide explains how to create and structure YAML pipeline definitions.

## Basic Structure

A pipeline YAML file consists of these main sections:

```yaml
name: "Pipeline Name"
description: "What this pipeline does"

inputs:
  # Input parameter definitions
  
steps:
  # Pipeline steps (tasks)
  
outputs:
  # Output definitions
```

## Inputs Section

Define the parameters your pipeline accepts:

```yaml
inputs:
  query:
    type: string
    description: "Search query to process"
    required: true
    
  max_results:
    type: integer
    description: "Maximum number of results"
    default: 10
    
  options:
    type: object
    description: "Additional options"
    default: {}
```

### Supported Types
- `string` - Text input
- `integer` - Whole numbers
- `float` - Decimal numbers
- `boolean` - True/false values
- `list` - Arrays of values
- `object` - Key-value pairs

## Steps Section

Define the tasks that make up your pipeline:

```yaml
steps:
  - id: analyze_data
    action: |
      Analyze the input data: {{query}}
      
      Perform the following:
      1. Extract key information
      2. Identify patterns
      3. Generate insights
      
      Return structured analysis
    timeout: 30.0
    tags: ["analysis", "processing"]
```

### Step Properties

- **id** (required): Unique identifier for the step
- **action** (required): The task to perform (can use template variables)
- **depends_on**: List of step IDs this step depends on
- **condition**: Conditional execution (Jinja2 expression)
- **timeout**: Maximum execution time in seconds
- **tags**: List of tags for categorization
- **cache_results**: Whether to cache results
- **on_error**: Error handling configuration

### Using Template Variables

Template variables use Jinja2 syntax:

```yaml
# Input variables
{{variable_name}}

# Step results
{{step_id.result}}

# Default values
{{variable | default('default value')}}

# Conditional rendering
{% if condition %}
  Content when true
{% else %}
  Content when false
{% endif %}
```

## Dependencies and Execution Order

Control execution order with `depends_on`:

```yaml
steps:
  - id: fetch_data
    action: "Fetch data from source"
    
  - id: process_data
    action: "Process the data: {{fetch_data.result}}"
    depends_on: [fetch_data]
    
  - id: analyze_results
    action: "Analyze processed data: {{process_data.result}}"
    depends_on: [process_data]
```

## Conditional Execution

Execute steps conditionally:

```yaml
steps:
  - id: check_data
    action: "Check if data is valid"
    
  - id: process_valid_data
    action: "Process the valid data"
    depends_on: [check_data]
    condition: "{{check_data.result.is_valid}} == true"
    
  - id: handle_invalid_data
    action: "Handle invalid data case"
    depends_on: [check_data]
    condition: "{{check_data.result.is_valid}} == false"
```

## Error Handling

Configure error handling behavior:

```yaml
steps:
  - id: risky_operation
    action: "Perform operation that might fail"
    on_error:
      action: "Log error and use fallback"
      continue_on_error: true
      retry_count: 3
      fallback_value: "default result"
```

## Outputs Section

Define pipeline outputs:

```yaml
outputs:
  summary: "{{generate_summary.result}}"
  processed_data: "{{process_data.result}}"
  metadata:
    total_items: "{{analyze_results.result.count}}"
    status: "completed"
```

## Complete Example

Here's a complete example of a research pipeline:

```yaml
name: "Research Assistant"
description: "Research and analyze information on a topic"

inputs:
  topic:
    type: string
    description: "Topic to research"
    required: true
    
  depth:
    type: string
    description: "Research depth"
    default: "comprehensive"
    
  max_sources:
    type: integer
    description: "Maximum sources to analyze"
    default: 5

steps:
  - id: plan_research
    action: |
      Create a research plan for "{{topic}}":
      1. Identify key aspects to investigate
      2. Determine information sources needed
      3. Plan research approach for {{depth}} analysis
      Return structured research plan
      
  - id: gather_information
    action: |
      Gather information about "{{topic}}" following the plan:
      Research Plan: {{plan_research.result}}
      
      Collect:
      1. Key facts and data
      2. Recent developments
      3. Expert opinions
      4. Statistical information
      
      Limit to {{max_sources}} sources
      Return organized information
    depends_on: [plan_research]
    timeout: 60.0
    
  - id: analyze_findings
    action: |
      Analyze the gathered information:
      Information: {{gather_information.result}}
      
      Provide:
      1. Key insights and patterns
      2. Important findings
      3. Trends and implications
      4. Knowledge gaps
      
      Return comprehensive analysis
    depends_on: [gather_information]
    
  - id: generate_report
    action: |
      Create a research report on "{{topic}}":
      Analysis: {{analyze_findings.result}}
      
      Include:
      1. Executive summary
      2. Detailed findings
      3. Conclusions
      4. Recommendations
      
      Format as professional report
    depends_on: [analyze_findings]

outputs:
  report: "{{generate_report.result}}"
  key_findings: "{{analyze_findings.result.findings}}"
  research_plan: "{{plan_research.result}}"
```

## Best Practices

### 1. Clear Action Descriptions
Write clear, specific action descriptions that guide the AI model:

```yaml
# Good
action: |
  Analyze customer feedback data and identify:
  1. Common themes and issues
  2. Sentiment trends over time
  3. Product improvement suggestions
  Return categorized findings with priority scores

# Too vague
action: "Analyze feedback"
```

### 2. Proper Dependency Management
Always declare dependencies to ensure correct execution order:

```yaml
steps:
  - id: step_a
    action: "First task"
    
  - id: step_b
    action: "Use result from A: {{step_a.result}}"
    depends_on: [step_a]
    
  - id: step_c
    action: "Combine A and B: {{step_a.result}} + {{step_b.result}}"
    depends_on: [step_a, step_b]
```

### 3. Meaningful Step IDs
Use descriptive IDs that indicate the step's purpose:

```yaml
# Good
- id: extract_customer_data
- id: validate_email_format
- id: generate_summary_report

# Bad
- id: step1
- id: process
- id: task_a
```

### 4. Error Handling
Always consider potential failures:

```yaml
- id: external_api_call
  action: "Call external API"
  timeout: 30.0
  on_error:
    action: "Use cached data as fallback"
    continue_on_error: true
```

### 5. Use Tags for Organization
Tag steps for better organization and filtering:

```yaml
- id: analyze_security
  action: "Security analysis"
  tags: ["security", "analysis", "critical"]
```

## Advanced Features

### Multi-line Actions with Formatting
Use YAML literal blocks for complex actions:

```yaml
action: |
  Perform comprehensive analysis:
  
  Phase 1: Data Collection
  - Gather metrics from database
  - Collect user feedback
  - Review system logs
  
  Phase 2: Analysis
  - Statistical analysis
  - Trend identification
  - Anomaly detection
  
  Phase 3: Reporting
  - Generate visualizations
  - Create summary
  - Prepare recommendations
  
  Return complete analysis package
```

### Complex Conditions
Use Jinja2 expressions for complex conditions:

```yaml
condition: |
  {{step_a.result.score > 0.8 and 
   step_b.result.status == 'success' and
   input_data.length > 0}}
```

### Dynamic Timeouts
Set timeouts based on input:

```yaml
timeout: "{{data_size > 1000 ? 120.0 : 60.0}}"
```

## Limitations

Currently, the following features are **not** supported:

1. **Loops**: The `loop` construct is not implemented
2. **Parallel execution**: Steps execute sequentially
3. **Dynamic step generation**: All steps must be defined statically
4. **External tool calls**: No direct file system or API access

## Testing Your Pipeline

Test your pipeline with the direct control system:

```python
from orchestrator.compiler.yaml_compiler import YAMLCompiler
from orchestrator.control_systems.model_based_control_system import ModelBasedControlSystem

# Load and compile YAML
with open('my_pipeline.yaml', 'r') as f:
    yaml_content = f.read()

compiler = YAMLCompiler()
pipeline = await compiler.compile(yaml_content, inputs)

# Execute
control_system = ModelBasedControlSystem(model_registry)
results = await control_system.execute_pipeline(pipeline)
```

## Troubleshooting

### Common Issues

1. **Template Syntax Errors**
   - Ensure all variables use `{{variable}}` syntax
   - Check for typos in variable names
   - Verify step IDs match exactly

2. **Dependency Cycles**
   - Ensure no circular dependencies exist
   - Use a directed acyclic graph structure

3. **Missing Required Inputs**
   - Provide all required inputs when executing
   - Check input types match expected types

4. **Step Result References**
   - Ensure referenced steps have completed
   - Use correct syntax: `{{step_id.result}}`

## Next Steps

- See example pipelines in the `examples/` directory
- Read the API documentation for programmatic usage
- Check the architecture guide for system internals