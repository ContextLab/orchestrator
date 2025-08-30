# Pipeline Tutorial: mcp_memory_workflow

## Overview

**Complexity Level**: Intermediate  
**Difficulty Score**: 55/100  
**Estimated Runtime**: 5-15 minutes  

### Purpose
This pipeline demonstrates conditional_execution, data_flow, for_loops and provides a practical example of orchestrator's capabilities for intermediate-level workflows.

### Use Cases
- General automation tasks

### Prerequisites
- Basic understanding of YAML syntax
- Familiarity with template variables and data flow
- Understanding of basic control flow concepts
- Familiarity with Model Context Protocol (MCP)

### Key Concepts
- Conditional logic and branching
- Data flow between pipeline steps
- External tool integration
- Iterative processing with loops
- Template variable substitution

## Pipeline Breakdown

### Configuration Analysis
- **input_section**: Defines the data inputs and parameters for the pipeline
- **steps_section**: Contains the sequence of operations to be executed
- **output_section**: Specifies how results are formatted and stored
- **template_usage**: Uses 1 template patterns for dynamic content
- **feature_highlights**: Demonstrates 6 key orchestrator features

### Data Flow
This pipeline processes input parameters through 6 steps to generate the specified outputs.

### Control Flow
Follows linear execution flow from first step to last step.

### Pipeline Configuration
```yaml
# MCP Memory Workflow
# Demonstrates using MCP memory for context management
id: mcp_memory_workflow
name: MCP Memory Context Management
description: Use MCP memory to maintain context across pipeline steps
version: "1.0.0"

parameters:
  user_name:
    type: string
    default: "User"
  task_description:
    type: string
    default: "Analyze sales data and create visualizations"
  output_path:
    type: string
    default: "examples/outputs/mcp_memory_workflow"

steps:
  # Initialize conversation context
  - id: init_context
    tool: mcp-memory
    action: execute
    parameters:
      action: "store"
      namespace: "conversation"
      key: "user_profile"
      value:
        name: "{{ user_name }}"
        task: "{{ task_description }}"
        started_at: "{{ execution['timestamp'] }}"
      ttl: 7200  # 2 hours
    
  # Store task breakdown
  - id: store_task_steps
    tool: mcp-memory
    action: execute
    parameters:
      action: "store"
      namespace: "conversation"
      key: "task_steps"
      value:
        - "Load and validate data"
        - "Perform statistical analysis"
        - "Create visualizations"
        - "Generate report"
      ttl: 7200
    dependencies:
      - init_context
    
  # Simulate processing first step
  - id: process_step_1
    tool: task-delegation
    action: execute
    parameters:
      task: "Identify specific data sources for: {{ task_description }}"
      requirements:
        capabilities: ["data-analysis"]
    dependencies:
      - store_task_steps
    
  # Store progress
  - id: update_progress
    tool: mcp-memory
    action: execute
    parameters:
      action: "store"
      namespace: "conversation"
      key: "progress"
      value:
        current_step: 1
        completed_steps: ["Data loading plan created"]
        next_action: "Execute data loading"
      ttl: 7200
    dependencies:
      - process_step_1
    
  # Retrieve all context
  - id: get_full_context
    tool: mcp-memory
    action: execute
    parameters:
      action: "list"
      namespace: "conversation"
    dependencies:
      - update_progress
    
  # Build context summary
  - id: build_context_summary
    tool: filesystem
    action: write
    parameters:
      path: "{{ output_path }}/{{ user_name | slugify }}_context_summary.md"
      content: |
        # Context Summary
        
        **Generated on:** {{ execution['timestamp'] }}
        
        ## Current Context State
        
        **Namespace**: conversation
        **Active Keys**: {{ get_full_context['keys'] | join(', ') }}
        
        ### Details:
        {% for key in get_full_context['keys'] %}
        - **{{ key }}**: Stored in memory
        {% endfor %}
    dependencies:
      - get_full_context
    
  # Create persistent memory export
  - id: export_memory
    tool: filesystem
    action: write
    parameters:
      path: "{{ output_path }}/{{ user_name | slugify }}_memory_export.json"
      content: |
        {
          "namespace": "conversation",
          "exported_at": "{{ execution['timestamp'] }}",
          "keys": {{ get_full_context['keys'] | tojson }},
          "metadata": {
            "user": "{{ user_name }}",
            "task": "{{ task_description }}"
          }
        }
    dependencies:
      - get_full_context
    
  # Demonstrate TTL expiration check
  - id: check_expiration
    tool: mcp-memory
    action: execute
    parameters:
      action: "retrieve"
      namespace: "conversation"
      key: "user_profile"
    dependencies:
      - export_memory
    
  # Clean up namespace (optional)
  - id: cleanup_memory
    tool: mcp-memory
    action: execute
    parameters:
      action: "clear"
      namespace: "temporary_workspace"
    dependencies:
      - export_memory
    condition: "false"  # Disabled by default

outputs:
  context_keys: "{{ get_full_context['keys'] | tojson }}"
  user_profile_found: "{{ check_expiration['found'] }}"
  memory_export_path: "{{ output_path }}/{{ user_name | slugify }}_memory_export.json"
  context_summary_path: "{{ output_path }}/{{ user_name | slugify }}_context_summary.md"
```

## Customization Guide

### Input Modifications
- Modify input parameters to match your specific data sources
- Adjust file paths and data formats as needed for your environment

### Parameter Tuning
- Adjust step parameters to customize behavior for your needs

### Step Modifications
- Add new steps by following the same pattern as existing ones
- Remove steps that aren't needed for your specific use case
- Reorder steps if your workflow requires different sequencing
- Replace tool actions with alternatives that provide similar functionality

### Output Customization
- Change output file paths and formats to match your requirements
- Modify output templates to customize the structure and content
- This pipeline produces Analysis results, JSON data, Markdown documents, Reports - adjust output configuration accordingly

## Remixing Instructions

### Compatible Patterns
- Most basic pipelines can be combined with this pattern

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

## Hands-On Exercise

### Execution Instructions
- 1. Navigate to your orchestrator project directory
- 1.5. Ensure you have access to required services: MCP tools
- 2. Run: python scripts/run_pipeline.py examples/mcp_memory_workflow.yaml
- 3. Monitor the output for progress and any error messages
- 4. Check the output directory for generated results

### Expected Outputs
- Generated Analysis results in the specified output directory
- Generated JSON data in the specified output directory
- Generated Markdown documents in the specified output directory
- Generated Reports in the specified output directory
- Execution logs showing step-by-step progress
- Completion message with runtime statistics
- No error messages or warnings (successful execution)

### Troubleshooting
- **API Authentication Errors**: Ensure all required API keys are properly configured in your environment
- **Template Resolution Errors**: Check that all input parameters are provided and template syntax is correct
- **General Execution Errors**: Check the logs for specific error messages and verify your orchestrator installation

### Verification Steps
- Check that the pipeline completed without errors
- Verify all expected output files were created
- Review the output content for quality and accuracy

---

*Tutorial generated on 2025-08-27T23:40:24.396294*
