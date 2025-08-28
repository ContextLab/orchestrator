# Pipeline Tutorial: control_flow_dynamic

## Overview

**Complexity Level**: Intermediate  
**Difficulty Score**: 60/100  
**Estimated Runtime**: 5-15 minutes  

### Purpose
This pipeline illustrates advanced control flow patterns in orchestrator. It demonstrates  for building dynamic, conditional workflows.

### Use Cases
- AI-powered content generation
- Batch processing with logic
- Conditional workflow automation
- Dynamic pipeline execution
- System administration and automation

### Prerequisites
- Basic understanding of YAML syntax
- Familiarity with template variables and data flow
- Understanding of basic control flow concepts
- Understanding of command-line interfaces and system security

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
- **template_usage**: Uses 3 template patterns for dynamic content
- **feature_highlights**: Demonstrates 6 key orchestrator features

### Data Flow
This pipeline processes input parameters through 6 steps to generate the specified outputs.

### Control Flow
Follows linear execution flow from first step to last step.

### Pipeline Configuration
```yaml
id: control-flow-dynamic
# Dynamic Flow Control Example
# Demonstrates goto and dynamic dependencies with AUTO resolution
name: Error Handling Pipeline
description: Dynamic flow control based on error conditions
version: "1.0.0"

parameters:
  operation:
    type: string
    description: Operation to perform
    default: "echo 'Hello, World!'"
  retry_limit:
    type: integer
    default: 3

steps:
  # Initial validation
  - id: validate_input
    action: generate_text
    parameters:
      prompt: |
        Validate if this is a safe operation to execute: "{{ operation }}"
        Return only "valid" if safe, or "invalid" if unsafe.
      model: <AUTO task="validate">Select a model for validation</AUTO>
      max_tokens: 10
      
  # Determine operation risk level
  - id: assess_risk
    action: generate_text
    parameters:
      prompt: |
        Assess the risk level of this operation: "{{ operation }}"
        Return ONLY one word: low, medium, or high
      model: <AUTO task="assess">Select a model for risk assessment</AUTO>
      max_tokens: 10
    dependencies:
      - validate_input
    
  # Prepare operation
  - id: prepare_operation
    action: generate_text
    parameters:
      prompt: |
        Prepare to execute operation: {{ operation }}
        Risk level: {{ assess_risk }}
        Return "ready" if prepared.
      model: <AUTO task="prepare">Select a model</AUTO>
      max_tokens: 50
    dependencies:
      - validate_input
      - assess_risk
    
  # Optional safety check for high-risk operations
  - id: safety_check
    action: generate_text
    condition: "{{ assess_risk == 'high' }}"
    parameters:
      prompt: |
        Perform additional safety check for high-risk operation: {{ operation }}
        Return "safe" or "unsafe"
      model: <AUTO task="safety">Select a model</AUTO>
      max_tokens: 10
    dependencies:
      - assess_risk
    
  # Execute operation (REAL execution)
  - id: execute_operation
    tool: terminal
    action: execute
    parameters:
      command: "{{ operation }}"
    dependencies:
      - prepare_operation
    
  # Check execution result
  - id: check_result
    action: generate_text
    parameters:
      prompt: |
        Analyze this execution output: {{ execute_operation }}
        
        Return EXACTLY one word - either "success" or "failure".
        Do not add any explanation or other text.
      model: <AUTO task="check">Select a model</AUTO>
      max_tokens: 10
    dependencies:
      - execute_operation
    
  # Generate success report
  - id: success_handler
    action: generate_text
    condition: "{{ check_result == 'success' }}"
    parameters:
      prompt: |
        Write a brief success summary. NO placeholders, NO conversational language.
        
        Operation: {{ operation }}
        Risk Level: {{ assess_risk }}
        Status: SUCCESS
        
        Format as a simple bulleted list with these exact items:
        - Operation executed successfully
        - Risk level: {{ assess_risk }}
        - Command completed without errors
      model: <AUTO task="report">Select a model</AUTO>
      max_tokens: 150
    dependencies:
      - check_result
    
  # Generate failure report
  - id: failure_handler
    action: generate_text
    condition: "{{ check_result == 'failure' }}"
    parameters:
      prompt: |
        Write a brief failure summary. NO placeholders, NO conversational language.
        
        Operation: {{ operation }}
        Risk Level: {{ assess_risk }}
        Status: FAILED
        Error: Execution did not complete successfully
        
        Format as a simple bulleted list.
      model: <AUTO task="report">Select a model</AUTO>
      max_tokens: 150
    dependencies:
      - check_result
    
  # Cleanup
  - id: cleanup
    action: generate_text
    parameters:
      prompt: |
        Return EXACTLY the word "cleaned" - nothing else.
      model: <AUTO task="cleanup">Select a model</AUTO>
      max_tokens: 10
    dependencies:
      - success_handler
      - failure_handler
      
  # Save final report
  - id: save_report
    tool: filesystem
    action: write
    parameters:
      path: "examples/outputs/control_flow_dynamic/report_{{ operation | slugify }}.md"
      content: |
        # Dynamic Flow Control Execution Report
        
        **Operation:** {{ operation }}
        **Risk Level:** {{ assess_risk }}
        
        ## Execution Summary
        
        - Validation: {{ validate_input }}
        - Preparation: {{ prepare_operation }}
        - Execution Result: {{ check_result }}
        
        ## Command Execution Details
        
        - **Command:** {{ execute_operation.command }}
        - **Return Code:** {{ execute_operation.return_code }}
        - **Success:** {{ execute_operation.success }}
        - **Execution Time:** {{ execute_operation.execution_time }}ms
        
        ### Command Output (stdout):
        ```
        {{ execute_operation.stdout }}
        ```
        
        {% if execute_operation.stderr %}
        ### Command Errors (stderr):
        ```
        {{ execute_operation.stderr }}
        ```
        {% endif %}
        
        ## Report Details
        
        {% if check_result == 'success' %}
        {{ success_handler }}
        {% else %}
        {{ failure_handler }}
        {% endif %}
        
        ## Cleanup Status
        
        {{ cleanup }}
        
        ---
        *Generated by Dynamic Flow Control Pipeline*
    dependencies:
      - cleanup

outputs:
  validation_result: "{{ validate_input }}"
  risk_level: "{{ assess_risk }}"
  execution_status: "{{ check_result }}"
  final_report: "{{ save_report.path }}"
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
- This pipeline produces Markdown documents, Reports - adjust output configuration accordingly

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
- 1.5. Ensure you have access to required services: System access
- 2. Run: python scripts/run_pipeline.py examples/control_flow_dynamic.yaml
- 3. Monitor the output for progress and any error messages
- 4. Check the output directory for generated results

### Expected Outputs
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
- Review generated content for relevance and quality

---

*Tutorial generated on 2025-08-27T23:40:24.395868*
