# Pipeline Tutorial: data_processing

## Overview

**Complexity Level**: Advanced  
**Difficulty Score**: 65/100  
**Estimated Runtime**: 15+ minutes  

### Purpose
This pipeline shows how to process and analyze data using orchestrator's data processing capabilities. It demonstrates data_flow, file_inclusion, interactive_workflows for building robust data workflows.

### Use Cases
- AI-powered content generation
- Automated data processing workflows
- Business data analysis and reporting
- Data quality assessment and cleaning

### Prerequisites
- Basic understanding of YAML syntax
- Experience with intermediate pipeline patterns
- Understanding of error handling and system integration
- Familiarity with external APIs and tools

### Key Concepts
- Data flow between pipeline steps
- Large language model integration
- Template variable substitution

## Pipeline Breakdown

### Configuration Analysis
- **input_section**: Defines the data inputs and parameters for the pipeline
- **steps_section**: Contains the sequence of operations to be executed
- **output_section**: Specifies how results are formatted and stored
- **template_usage**: Uses 3 template patterns for dynamic content
- **feature_highlights**: Demonstrates 7 key orchestrator features

### Data Flow
This pipeline processes input parameters through 7 steps to generate the specified outputs.

### Control Flow
Follows linear execution flow from first step to last step.

### Pipeline Configuration
```yaml
id: data-processing
name: Data Processing Pipeline
description: Process and validate data from various sources

parameters:
  data_source:
    type: string
    required: false
    default: "examples/test_data/sample_data.json"
    description: Path to data file (CSV or JSON)
  output_format:
    type: string
    default: json
    description: Output format (json, csv, or yaml)
  output_path:
    type: string
    default: "examples/outputs/data_processing"
    description: Directory where output files will be saved

steps:
  - id: load_data
    tool: filesystem
    action: read
    parameters:
      path: "{{ data_source }}"
  
  - id: parse_data
    action: generate_text
    parameters:
      prompt: |
        Parse this data and identify its structure:
        {{ load_data }}
        
        Return ONLY one word: "json" if it's JSON, "csv" if it's CSV, or "unknown" if unclear.
      model: <AUTO task="parse">Select a model for parsing</AUTO>
      max_tokens: 10
    dependencies:
      - load_data
  
  - id: validate_data
    tool: validation
    action: validate
    parameters:
      data: "{{ load_data.content }}"
      schema:
        type: object
        properties:
          records:
            type: array
            items:
              type: object
              properties:
                id:
                  type: integer
                name:
                  type: string
                active:
                  type: boolean
              required: ["id", "name"]
      mode: lenient
    dependencies:
      - parse_data
  
  - id: transform_data
    tool: data-processing
    action: transform
    parameters:
      data: "{{ load_data.content }}"
      operation:
        transformations:
          - type: filter
            field: active
            value: true
          - type: aggregate
            operation: sum
            field: value
    dependencies:
      - validate_data
  
  - id: format_results
    action: generate_text
    parameters:
      prompt: |
        Convert this data to clean JSON format:
        {{ transform_data }}
        
        Return ONLY valid JSON without any markdown formatting, code fences, or explanations.
        Do NOT include ```json or ``` markers.
        Start directly with { and end with }
      model: <AUTO task="format">Select a model for formatting</AUTO>
      max_tokens: 500
    dependencies:
      - transform_data
  
  - id: save_results
    tool: filesystem
    action: write
    parameters:
      path: "{{ output_path }}/processed_data.{{ output_format }}"
      content: "{{ format_results }}"
    dependencies:
      - format_results
      
  - id: generate_summary
    action: generate_text
    parameters:
      prompt: |
        Generate a brief processing summary based on:
        - Original data: {{ load_data }}
        - Validation result: {{ validate_data }}
        - Transformed data: {{ transform_data }}
        
        Include:
        - Number of records processed
        - Validation status
        - Transformation applied
        
        Keep it concise (3-4 lines).
      model: <AUTO task="summary">Select a model for summary</AUTO>
      max_tokens: 150
    dependencies:
      - transform_data
      
  - id: save_report
    tool: filesystem
    action: write
    parameters:
      path: "{{ output_path }}/processing_report.md"
      content: |
        # Data Processing Report
        
        **Source File:** {{ data_source }}
        **Output Format:** {{ output_format }}
        
        ## Validation Results
        
        - Validation Status: {% if validate_data.valid %}Passed{% else %}Failed{% endif %}
        - Errors: {% if validate_data.errors %}{{ validate_data.errors | length }} errors found{% else %}None{% endif %}
        - Warnings: {% if validate_data.warnings %}{{ validate_data.warnings | length }} warnings{% else %}None{% endif %}
        
        ## Processing Summary
        
        {{ generate_summary }}
        
        ## Output Details
        
        - Transformed data saved to: {{ output_path }}/processed_data.{{ output_format }}
        - Report generated at: {{ output_path }}/processing_report.md
        
        ---
        *Generated by Data Processing Pipeline*
    dependencies:
      - save_results
      - generate_summary

outputs:
  original_data: "{{ load_data }}"
  validated: "{{ validate_data.valid }}"
  transformed: "{{ transform_data }}"
  output_file: "{{ output_path }}/processed_data.{{ output_format }}"
  summary: "{{ generate_summary }}"
```

## Customization Guide

### Input Modifications
- Modify input parameters to match your specific data sources
- Adjust file paths and data formats as needed for your environment

### Parameter Tuning
- Adjust model parameters (temperature, max_tokens) for different output styles
- Modify prompts to change the tone and focus of generated content
- Fine-tune performance parameters for your specific use case

### Step Modifications
- Add new steps by following the same pattern as existing ones
- Remove steps that aren't needed for your specific use case
- Reorder steps if your workflow requires different sequencing
- Replace tool actions with alternatives that provide similar functionality

### Output Customization
- Change output file paths and formats to match your requirements
- Modify output templates to customize the structure and content
- This pipeline produces CSV data, JSON data, Markdown documents, Reports - adjust output configuration accordingly

## Remixing Instructions

### Compatible Patterns
- fact_checker.yaml - for content verification
- research workflows - for information gathering

### Extension Ideas
- Build modular components for reusability
- Add performance monitoring and optimization
- Implement advanced security and access controls

### Combination Examples
- Combine with research workflows to gather additional data
- Use with statistical analysis for comprehensive insights
- Integrate with visualization tools for data presentation

### Advanced Variations
- Scale to handle larger datasets and more complex processing
- Add real-time processing capabilities for streaming data
- Implement distributed processing across multiple systems
- Use multiple AI models for comparison and validation

## Hands-On Exercise

### Execution Instructions
- 1. Navigate to your orchestrator project directory
- 2. Run: python scripts/run_pipeline.py examples/data_processing.yaml
- 3. Monitor the output for progress and any error messages
- 4. Check the output directory for generated results

### Expected Outputs
- Generated CSV data in the specified output directory
- Generated JSON data in the specified output directory
- Generated Markdown documents in the specified output directory
- Generated Reports in the specified output directory
- Execution logs showing step-by-step progress
- Completion message with runtime statistics
- No error messages or warnings (successful execution)

### Troubleshooting
- **Template Resolution Errors**: Check that all input parameters are provided and template syntax is correct
- **Complex Logic Errors**: Review the pipeline configuration and ensure all advanced features are properly configured
- **General Execution Errors**: Check the logs for specific error messages and verify your orchestrator installation

### Verification Steps
- Check that the pipeline completed without errors
- Verify all expected output files were created
- Review the output content for quality and accuracy
- Review generated content for relevance and quality

---

*Tutorial generated on 2025-08-27T23:40:24.395973*
