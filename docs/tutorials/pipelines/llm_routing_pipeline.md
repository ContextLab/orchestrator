# Pipeline Tutorial: llm_routing_pipeline

## Overview

**Complexity Level**: Intermediate  
**Difficulty Score**: 35/100  
**Estimated Runtime**: 5-15 minutes  

### Purpose
This pipeline demonstrates interactive_workflows, llm_integration, template_variables and provides a practical example of orchestrator's capabilities for intermediate-level workflows.

### Use Cases
- AI-powered content generation

### Prerequisites
- Basic understanding of YAML syntax
- Familiarity with template variables and data flow
- Understanding of basic control flow concepts

### Key Concepts
- Large language model integration
- Template variable substitution

## Pipeline Breakdown

### Configuration Analysis
- **input_section**: Defines the data inputs and parameters for the pipeline
- **steps_section**: Contains the sequence of operations to be executed
- **output_section**: Specifies how results are formatted and stored
- **template_usage**: Uses 6 template patterns for dynamic content
- **feature_highlights**: Demonstrates 3 key orchestrator features

### Data Flow
This pipeline processes input parameters through 3 steps to generate the specified outputs.

### Control Flow
Follows linear execution flow from first step to last step.

### Pipeline Configuration
```yaml
# LLM Routing and Optimization Pipeline
# Demonstrates intelligent model selection and prompt optimization
id: llm_routing_pipeline
name: Smart LLM Routing Pipeline
description: Automatically selects the best model and optimizes prompts for tasks
version: "1.0.0"

parameters:
  task:
    type: string
    default: "Write a comprehensive analysis of renewable energy trends"
  optimization_goals:
    type: array
    default: ["clarity", "brevity", "model_specific"]
  routing_strategy:
    type: string
    default: "capability_based"

steps:
  - id: analyze_task
    tool: task-delegation
    action: execute
    parameters:
      task: "{{ parameters.task }}"
      cost_weight: 0.3
      quality_weight: 0.7
    
  - id: optimize_prompt
    tool: prompt-optimization
    action: execute
    parameters:
      prompt: "{{ parameters.task }}"
      model: "{{ analyze_task.selected_model }}"
      optimization_goals: "{{ parameters.optimization_goals }}"
      preserve_intent: true
    dependencies:
      - analyze_task
    
  - id: route_request
    tool: multi-model-routing
    action: execute
    parameters:
      request: "{{ optimize_prompt.optimized_prompt }}"
      models: "{{ analyze_task.fallback_models | default([analyze_task.selected_model]) }}"
      strategy: "{{ parameters.routing_strategy }}"
      max_concurrent: 10
    dependencies:
      - optimize_prompt
    
  - id: save_report
    tool: filesystem
    action: write
    parameters:
      path: "examples/outputs/llm_routing_pipeline/{{ parameters.task[:50] | slugify }}_report.md"
      content: |
        # LLM Task Routing and Optimization Report
        
        ## Task Analysis
        - **Original Task**: {{ parameters.task }}
        - **Task Type**: {{ analyze_task.task_analysis.task_type }}
        - **Complexity**: {{ analyze_task.task_analysis.complexity }}
        
        ## Model Selection
        - **Selected Model**: {{ analyze_task.selected_model }}
        - **Score**: {{ analyze_task.score }}
        - **Reasons**: {{ analyze_task.reasons | join(', ') }}
        - **Estimated Cost**: ${{ analyze_task.estimated_cost | round(3) }}
        - **Estimated Latency**: {{ analyze_task.estimated_latency }}s
        
        ## Prompt Optimization
        - **Original Length**: {{ optimize_prompt.metrics.original_tokens }} tokens
        - **Optimized Length**: {{ optimize_prompt.metrics.optimized_tokens }} tokens
        - **Reduction**: {{ optimize_prompt.metrics.reduction_percentage | round(1) }}%
        - **Applied Optimizations**: {{ optimize_prompt.applied_optimizations | join(', ') if optimize_prompt.applied_optimizations else 'None' }}
        
        ### Optimized Prompt
        ```
        {{ optimize_prompt.optimized_prompt }}
        ```
        
        ## Routing Decision
        - **Final Model**: {{ route_request.selected_model }}
        - **Strategy**: {{ route_request.strategy }}
        - **Routing Reason**: {{ route_request.routing_reason }}
        - **Current Load**: {{ route_request.current_load }}
        
        ## Recommendations
        {% for rec in optimize_prompt.recommendations %}
        - {{ rec }}
        {% endfor %}
        
        ## Alternative Models
        {% for score in analyze_task.all_scores if score.model != analyze_task.selected_model %}
        {% if loop.index <= 3 %}
        - **{{ score.model }}** (Score: {{ score.score }})
          - {{ score.reasons | join(', ') }}
        {% endif %}
        {% endfor %}
    dependencies:
      - route_request
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
- This pipeline produces Analysis results, Markdown documents, Reports - adjust output configuration accordingly

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
- 2. Run: python scripts/run_pipeline.py examples/llm_routing_pipeline.yaml
- 3. Monitor the output for progress and any error messages
- 4. Check the output directory for generated results

### Expected Outputs
- Generated Analysis results in the specified output directory
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

*Tutorial generated on 2025-08-27T23:40:24.396248*
