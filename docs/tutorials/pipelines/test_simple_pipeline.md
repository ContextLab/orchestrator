# Pipeline Tutorial: test_simple_pipeline

## Overview

**Complexity Level**: Intermediate  
**Difficulty Score**: 31/100  
**Estimated Runtime**: 5-15 minutes  

### Purpose
This pipeline demonstrates csv_processing, data_flow, interactive_workflows and provides a practical example of orchestrator's capabilities for intermediate-level workflows.

### Use Cases
- AI-powered content generation

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
- **template_usage**: Uses 2 template patterns for dynamic content
- **feature_highlights**: Demonstrates 5 key orchestrator features

### Data Flow
This pipeline processes input parameters through 5 steps to generate the specified outputs.

### Control Flow
Follows linear execution flow from first step to last step.

### Pipeline Configuration
```yaml
# Simple test pipeline
id: simple_test
name: Simple Test Pipeline
description: Test basic functionality

steps:
  - id: generate_data
    action: generate_text
    parameters:
      prompt: "Generate a CSV with 3 rows: id,value,category\n1,100,A\n2,200,B\n3,150,A"
      model: <AUTO>
    
  - id: save_data
    tool: filesystem
    action: write
    parameters:
      path: "{{ output_path }}/test_data.csv"
      content: "id,value,category\n1,100,A\n2,200,B\n3,150,A"
    
  - id: create_chart
    tool: visualization
    action: create_charts
    parameters:
      data: [{"id": 1, "value": 100, "category": "A"}, {"id": 2, "value": 200, "category": "B"}, {"id": 3, "value": 150, "category": "A"}]
      chart_types: ["bar", "pie"]
      output_dir: "{{ output_path }}/charts"
      title: "Test Chart"
    dependencies:
      - save_data

outputs:
  charts: "{{ create_chart.charts }}"
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
- This pipeline produces CSV data - adjust output configuration accordingly

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
- 2. Run: python scripts/run_pipeline.py examples/test_simple_pipeline.yaml
- 3. Monitor the output for progress and any error messages
- 4. Check the output directory for generated results

### Expected Outputs
- Generated CSV data in the specified output directory
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

*Tutorial generated on 2025-08-27T23:40:24.396717*
