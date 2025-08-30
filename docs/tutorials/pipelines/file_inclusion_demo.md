# Pipeline Tutorial: file_inclusion_demo

## Overview

**Complexity Level**: Intermediate  
**Difficulty Score**: 40/100  
**Estimated Runtime**: 10-30 minutes  

### Purpose
This pipeline demonstrates interactive_workflows, json_handling, llm_integration and provides a practical example of orchestrator's capabilities for intermediate-level workflows.

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
- **template_usage**: Uses 8 template patterns for dynamic content
- **feature_highlights**: Demonstrates 4 key orchestrator features

### Data Flow
This pipeline processes input parameters through 4 steps to generate the specified outputs.

### Control Flow
Follows linear execution flow from first step to last step.

### Pipeline Configuration
```yaml
id: file_inclusion_demo
name: File Inclusion Demo Pipeline
version: 1.0.0
description: |
  Demonstrates file inclusion capabilities for external content.
  
  This pipeline shows how to:
  - Include prompts from external files
  - Include configuration from JSON files
  - Include markdown instructions
  - Use both {{ file:path }} and << path >> syntax

inputs:
  research_topic:
    type: string
    description: Topic to research
    default: "artificial intelligence ethics"
  
  output_format:
    type: string
    description: Format for the final report
    default: "markdown"
    
parameters:
  max_search_results: 10

steps:
  - id: load_system_prompt
    name: Load System Prompt
    action: llm_call
    parameters:
      model: anthropic/claude-sonnet-4-20250514
      prompt: |
        {{ file:prompts/research_system_prompt.txt }}
        
        Research Topic: {{ research_topic }}
        
        Please provide an initial research plan for this topic.

  - id: conduct_web_search
    name: Conduct Web Search
    action: web_search
    depends_on: load_system_prompt
    parameters:
      query: "{{ load_system_prompt.result }}"
      max_results: "{{ max_search_results }}"

  - id: analyze_results
    name: Analyze Search Results
    action: llm_call
    depends_on: conduct_web_search
    parameters:
      model: anthropic/claude-sonnet-4-20250514
      prompt: |
        << prompts/analysis_instructions.md >>
        
        Research Topic: {{ research_topic }}
        Search Results: {{ conduct_web_search.result }}
        
        Please analyze these search results according to the instructions above.

  - id: generate_report
    name: Generate Final Report
    action: llm_call
    depends_on: analyze_results
    parameters:
      model: anthropic/claude-sonnet-4-20250514
      prompt: |
        {{ file:prompts/report_template.md }}
        
        Topic: {{ research_topic }}
        Analysis: {{ analyze_results.result }}
        Format: {{ output_format }}
        
        Generate a comprehensive report using the template above.
        
        Additional formatting guidelines:
        << templates/formatting_guidelines.txt >>

  - id: save_report
    name: Save Report to File
    action: filesystem
    depends_on: generate_report
    parameters:
      action: write
      path: "outputs/{{ research_topic | slugify }}_report.{{ output_format }}"
      content: |
        # Research Report: {{ research_topic }}
        Generated on: {{ timestamp }}
        
        {{ generate_report.result }}
        
        ---
        Report generated using file inclusion pipeline
        Configuration: {{ file:config/report_config.json }}
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
- This pipeline produces Analysis results, JSON data, Markdown documents, Reports - adjust output configuration accordingly

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
- 1.5. Ensure you have access to required services: Anthropic API
- 2. Run: python scripts/run_pipeline.py examples/file_inclusion_demo.yaml
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
- Review generated content for relevance and quality

---

*Tutorial generated on 2025-08-27T23:40:24.396147*
