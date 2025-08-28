# Pipeline Tutorial: validation_pipeline

## Overview

**Complexity Level**: Beginner  
**Difficulty Score**: 25/100  
**Estimated Runtime**: < 5 minutes  

### Purpose
This pipeline demonstrates json_handling, llm_integration, template_variables and provides a practical example of orchestrator's capabilities for beginner-level workflows.

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
# Validation Pipeline Example
# Uses the validation tool for data quality checks
id: validation_pipeline
name: Data Validation Pipeline
description: Validate data against schemas and extract structured information
version: "1.0.0"

steps:
  - id: read_config
    tool: filesystem
    action: read
    parameters:
      path: "examples/outputs/validation_pipeline/config/validation_schema.json"
    
  - id: read_data
    tool: filesystem
    action: read
    parameters:
      path: "examples/outputs/validation_pipeline/data/user_data.json"
    
  - id: validate_data
    tool: validation
    action: validate
    parameters:
      data: "{{ read_data.content | from_json }}"
      schema: "{{ read_config.content | from_json }}"
      mode: "strict"
    dependencies:
      - read_config
      - read_data
    
  - id: extract_info
    tool: validation
    action: extract_structured
    parameters:
      text: "John Doe, age 30, email: john@example.com, phone: +1-555-0123"
      schema:
        type: object
        properties:
          name:
            type: string
          age:
            type: integer
          email:
            type: string
            format: email
          phone:
            type: string
            pattern: "^\\+?[1-9]\\d{1,14}$"
        required: ["name", "email"]
      model: "gpt-4o-mini"
    
  - id: save_report
    tool: filesystem
    action: write
    parameters:
      path: "examples/outputs/validation_pipeline/reports/validation_report.json"
      content: |
        {
          "validation_result": {{ validate_data | to_json }},
          "extracted_data": {{ extract_info | to_json }},
          "timestamp": "{{ now() }}"
        }
    dependencies:
      - validate_data
      - extract_info
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
- This pipeline produces JSON data, Reports - adjust output configuration accordingly

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
- 1.5. Ensure you have access to required services: OpenAI API
- 2. Run: python scripts/run_pipeline.py examples/validation_pipeline.yaml
- 3. Monitor the output for progress and any error messages
- 4. Check the output directory for generated results

### Expected Outputs
- Generated JSON data in the specified output directory
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

*Tutorial generated on 2025-08-27T23:40:24.396758*
