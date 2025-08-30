# Pipeline Tutorial: until_condition_examples

## Overview

**Complexity Level**: Beginner  
**Difficulty Score**: 25/100  
**Estimated Runtime**: 2-5 minutes  

### Purpose
This pipeline demonstrates interactive_workflows, llm_integration, template_variables and provides a practical example of orchestrator's capabilities for beginner-level workflows.

### Use Cases
- AI-powered content generation

### Prerequisites
- Basic understanding of YAML syntax

### Key Concepts
- Large language model integration
- Template variable substitution

## Pipeline Breakdown

### Configuration Analysis
- **input_section**: Defines the data inputs and parameters for the pipeline
- **steps_section**: Contains the sequence of operations to be executed
- **output_section**: Specifies how results are formatted and stored
- **template_usage**: Uses 3 template patterns for dynamic content
- **feature_highlights**: Demonstrates 3 key orchestrator features

### Data Flow
This pipeline processes input parameters through 3 steps to generate the specified outputs.

### Control Flow
Follows linear execution flow from first step to last step.

### Pipeline Configuration
```yaml
name: "Until Condition Examples"
description: |
  Simple examples of until conditions that demonstrate the basic functionality
  without complex nested structures that may not be supported by the current schema.

steps:
  # Example 1: Simple counter demonstration  
  - id: initialize_counter
    name: "Initialize Counter"
    action: debug
    parameters:
      message: "Starting until condition examples - initializing counter to 0"

  # Example 2: Simple quality check 
  - id: quality_check
    name: "Quality Check Example"
    action: llm_call
    parameters:
      model: "anthropic/claude-sonnet-4-20250514"
      prompt: |
        Generate a short paragraph about artificial intelligence research.
        Focus on making it informative and well-structured.
      temperature: 0.3
    dependencies:
      - initialize_counter

  # Example 3: Evaluate the quality
  - id: evaluate_quality
    name: "Evaluate Content Quality" 
    action: llm_call
    parameters:
      model: "anthropic/claude-sonnet-4-20250514"
      prompt: |
        Rate the quality of this content on a scale of 0-1:
        {{ quality_check.result }}
        
        Consider accuracy, completeness, and clarity.
        Return only a decimal number between 0 and 1.
      temperature: 0
    dependencies:
      - quality_check

  # Example 4: Process simple items
  - id: process_first_item
    name: "Process First Item"
    action: debug
    parameters:
      message: "Processing item 1: value = {{ 1 * 2 }}"
    dependencies:
      - evaluate_quality

  - id: process_second_item
    name: "Process Second Item"  
    action: debug
    parameters:
      message: "Processing item 2: value = {{ 2 * 2 }}"
    dependencies:
      - process_first_item

  - id: process_third_item
    name: "Process Third Item"
    action: debug
    parameters:
      message: "Processing item 3: value = {{ 3 * 2 }} - threshold reached!"
    dependencies:
      - process_second_item

  # Example 5: Final report
  - id: final_report
    name: "Generate Final Report"
    action: debug
    parameters:
      message: |
        Until condition examples demonstration completed:
        - Quality score achieved: {{ evaluate_quality.result | default('N/A') }}
        - Items processing completed: 3 items processed with values 2, 4, 6
        - Threshold demonstration: value 6 >= 6 (condition met)
        
        Note: This simplified example demonstrates the concept of until conditions
        without the complex nested loop structures that require advanced schema support.
    dependencies:
      - process_third_item
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
- This pipeline produces Reports - adjust output configuration accordingly

## Remixing Instructions

### Compatible Patterns
- fact_checker.yaml - for content verification
- research workflows - for information gathering

### Extension Ideas
- Add error handling and recovery steps
- Implement conditional logic for different scenarios
- Include data validation and quality checks

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
- 1.5. Ensure you have access to required services: Anthropic API
- 2. Run: python scripts/run_pipeline.py examples/until_condition_examples.yaml
- 3. Monitor the output for progress and any error messages
- 4. Check the output directory for generated results

### Expected Outputs
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

*Tutorial generated on 2025-08-27T23:40:24.396738*
