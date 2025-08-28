# Pipeline Tutorial: control_flow_while_loop

## Overview

**Complexity Level**: Intermediate  
**Difficulty Score**: 45/100  
**Estimated Runtime**: 10-30 minutes  

### Purpose
This pipeline illustrates advanced control flow patterns in orchestrator. It demonstrates  for building dynamic, conditional workflows.

### Use Cases
- AI-powered content generation
- Batch processing with logic
- Conditional workflow automation
- Dynamic pipeline execution

### Prerequisites
- Basic understanding of YAML syntax
- Familiarity with template variables and data flow
- Understanding of basic control flow concepts

### Key Concepts
- Data flow between pipeline steps
- Large language model integration
- Template variable substitution

## Pipeline Breakdown

### Configuration Analysis
- **input_section**: Defines the data inputs and parameters for the pipeline
- **steps_section**: Contains the sequence of operations to be executed
- **output_section**: Specifies how results are formatted and stored
- **template_usage**: Uses 6 template patterns for dynamic content
- **feature_highlights**: Demonstrates 5 key orchestrator features

### Data Flow
This pipeline processes input parameters through 5 steps to generate the specified outputs.

### Control Flow
Follows linear execution flow from first step to last step.

### Pipeline Configuration
```yaml
id: control-flow-while-loop
# While Loop Example
# Demonstrates iterative processing with a simple counter
name: Iterative Number Guessing
description: Generate numbers until we reach a target
version: "1.0.0"

parameters:
  target_number:
    type: integer
    default: 42
    description: Target number to reach
  max_attempts:
    type: integer
    default: 10
    description: Maximum attempts
    
steps:
  # Initialize the process
  - id: initialize
    action: generate_text
    parameters:
      prompt: "Starting number guessing game. Target is {{ target_number }}"
      model: <AUTO task="summarize">Select a model for initialization</AUTO>
      max_tokens: 50
      
  # Create initial state file
  - id: init_state
    tool: filesystem
    action: write
    parameters:
      path: "examples/outputs/control_flow_while_loop/state/current_guess.txt"
      content: "0"
    dependencies:
      - initialize
      
  # Iterative guessing loop
  - id: guessing_loop
    while: 'true'  # Simple always-true condition, will use max_iterations
    max_iterations: "{{ max_attempts }}"
    steps:
      # Read current guess
      - id: read_guess
        tool: filesystem
        action: read
        parameters:
          path: "examples/outputs/control_flow_while_loop/state/current_guess.txt"
          
      # Generate next guess
      - id: generate_guess
        action: generate_text
        parameters:
          prompt: |
            We're trying to guess the number {{ target_number }}.
            Current guess: {{ read_guess.content }}
            Iteration: {{ guessing_loop.iteration }}
            
            What number should we guess next? Reply with just a number between 1 and 100.
          model: <AUTO task="generate">Select a model for number generation</AUTO>
          max_tokens: 10
        dependencies:
          - read_guess
          
      # Extract number from response
      - id: extract_number
        action: analyze_text
        parameters:
          text: "Generated guess: {{ generate_guess.result | regex_search('[0-9]+') | default('25') }}"
          model: <AUTO task="analyze">Select a model for number extraction</AUTO>
          analysis_type: "extract_number"
        dependencies:
          - generate_guess
          
      # Update state
      - id: update_state
        tool: filesystem
        action: write
        parameters:
          path: "examples/outputs/control_flow_while_loop/state/current_guess.txt"
          content: "{{ generate_guess.result | regex_search('[0-9]+') | default('25') }}"
        dependencies:
          - extract_number
          
      # Log the attempt
      - id: log_attempt
        tool: filesystem
        action: write
        parameters:
          path: "examples/outputs/control_flow_while_loop/logs/attempt_{{ guessing_loop.iteration }}.txt"
          content: |
            Iteration: {{ guessing_loop.iteration }}
            Previous guess: {{ read_guess.content }}
            New guess: {{ generate_guess.result | regex_search('[0-9]+') | default('25') }}
            Target: {{ target_number }}
        dependencies:
          - update_state
          
      # Check if we found the target
      - id: check_result
        action: analyze_text  
        parameters:
          text: "{{ generate_guess.result | regex_search('[0-9]+') | default('25') }}"
          model: <AUTO task="analyze">Select a model for comparison</AUTO>
          analysis_type: "compare_to_target"
          target: "{{ target_number }}"
        dependencies:
          - log_attempt
          
      # Update loop state
      - id: update_loop_state
        action: generate_text
        parameters:
          prompt: "Attempt {{ guessing_loop.iteration }} complete. Found target: {{ check_result.matches_target | default(false) }}"
          model: <AUTO task="summarize">Select a model for status update</AUTO>
          max_tokens: 20
        dependencies:
          - check_result
    dependencies:
      - init_state
      
  # Final result
  - id: final_result
    tool: filesystem
    action: write
    parameters:
      path: "examples/outputs/control_flow_while_loop/result.txt"
      content: |
        # Number Guessing Results
        
        Target number: {{ target_number }}
        Total attempts: {{ guessing_loop.iterations | default(0) }}
        Success: {{ guessing_loop.completed | default(false) }}
    dependencies:
      - guessing_loop
      
outputs:
  target: "{{ target_number }}"
  attempts: "{{ guessing_loop.iterations | default(0) }}"
  result_file: "{{ final_result.filepath }}"
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
- This pipeline produces Analysis results - adjust output configuration accordingly

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
- 2. Run: python scripts/run_pipeline.py examples/control_flow_while_loop.yaml
- 3. Monitor the output for progress and any error messages
- 4. Check the output directory for generated results

### Expected Outputs
- Generated Analysis results in the specified output directory
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

*Tutorial generated on 2025-08-27T23:40:24.395918*
