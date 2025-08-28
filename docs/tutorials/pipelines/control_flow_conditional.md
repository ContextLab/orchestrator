# Pipeline Tutorial: control_flow_conditional

## Overview

**Complexity Level**: Intermediate  
**Difficulty Score**: 45/100  
**Estimated Runtime**: 5-15 minutes  

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
- Conditional logic and branching
- Data flow between pipeline steps
- Large language model integration
- Template variable substitution

## Pipeline Breakdown

### Configuration Analysis
- **input_section**: Defines the data inputs and parameters for the pipeline
- **steps_section**: Contains the sequence of operations to be executed
- **output_section**: Specifies how results are formatted and stored
- **template_usage**: Uses 4 template patterns for dynamic content
- **feature_highlights**: Demonstrates 5 key orchestrator features

### Data Flow
This pipeline processes input parameters through 5 steps to generate the specified outputs.

### Control Flow
Follows linear execution flow from first step to last step.

### Pipeline Configuration
```yaml
id: control-flow-conditional
# Conditional Execution Example
# Demonstrates simple conditional branching
name: Conditional File Processing
description: Process files differently based on their size
version: "1.0.0"

parameters:
  input_file:
    type: string
    default: "data/sample.txt"
    description: File to process
  size_threshold:
    type: integer
    default: 1000
    description: Size threshold in bytes
    
steps:
  # Read the input file
  - id: read_file
    tool: filesystem
    action: read
    parameters:
      path: "{{ input_file }}"
      
  # Check file size
  - id: check_size
    action: generate_text
    parameters:
      prompt: "Provide a technical analysis of a file that is {{ read_file.size }} bytes in size. Return only factual information about this file size without conversational language."
      model: <AUTO task="analyze">Select a model for analysis</AUTO>
      max_tokens: 50
    dependencies:
      - read_file
    
  # Process large files (compress)
  - id: compress_large
    action: generate_text
    condition: "{{ read_file.size > size_threshold }}"
    parameters:
      prompt: |
        Analyze the following text and provide exactly 3 bullet points summarizing it.
        
        IMPORTANT: 
        - Start each bullet with • 
        - Each bullet must contain actual content (never empty)
        - If the text is just repeated characters, describe it precisely (e.g., "• Contains exactly 2000 instances of the character 'A'")
        - Be factual and specific about counts and patterns
        - No introductory phrases or conclusions
        
        Text to summarize ({{ read_file.size }} bytes):
        {{ read_file.content }}
      model: <AUTO task="summarize">Select a model for text summarization</AUTO>
      max_tokens: 200
    dependencies:
      - check_size
    
  # Process small files (expand) - but skip if empty
  - id: expand_small
    action: generate_text
    condition: "{{ read_file.size <= size_threshold and read_file.size > 0 }}"
    parameters:
      prompt: |
        Expand the following text with additional relevant details and context.
        
        RULES:
        - Start directly with the expanded content
        - No conversational phrases like "Let's", "Okay", "Here's", etc.
        - IMPORTANT: If the text is ONLY repeated single characters (like "XXXX" or "AAAA"), you MUST:
          1. Describe what this repetitive pattern represents in testing contexts
          2. Explain why files with repeated characters are used in software testing
          3. Discuss the significance of the specific byte count ({{ read_file.size }} bytes)
          4. Never just output the repeated characters themselves
        - Be accurate about sizes ({{ read_file.size }} bytes, not kilobytes)
        - For any repetitive content, provide meaningful technical analysis
        - Write in professional, informative style
        - Minimum 200 words of actual expanded content
        
        Text to expand ({{ read_file.size }} bytes):
        {{ read_file.content }}
      model: <AUTO task="generate">Select a model for text generation</AUTO>
      max_tokens: 500
    dependencies:
      - check_size
  
  # Handle empty files
  - id: handle_empty
    action: generate_text
    condition: "{{ read_file.size == 0 }}"
    parameters:
      prompt: |
        Output exactly the following text with no modifications, additions, or code:
        The input file was empty. No content to process.
      model: <AUTO task="generate">Select a minimal model</AUTO>
      max_tokens: 50
    dependencies:
      - check_size
    
  # Save the result
  - id: save_result
    tool: filesystem
    action: write
    parameters:
      path: "examples/outputs/control_flow_conditional/processed_{{ input_file | basename | replace('.txt', '') }}.md"
      content: |
        # Processed File
        
        Original size: {{ read_file.size }} bytes
        Processing type: {% if read_file.size == 0 %}Empty file{% elif read_file.size > size_threshold %}Compressed{% else %}Expanded{% endif %}
        

        ## Result

        {% if handle_empty.status is not defined or handle_empty.status != 'skipped' %}{{ handle_empty }}{% elif compress_large.status is not defined or compress_large.status != 'skipped' %}{{ compress_large }}{% elif expand_small.status is not defined or expand_small.status != 'skipped' %}{{ expand_small }}{% else %}No content processed.{% endif %}
    dependencies:
      - compress_large
      - expand_small
      - handle_empty
      
outputs:
  original_size: "{{ read_file.size }}"
  processed_content: "{% if handle_empty.status is not defined or handle_empty.status != 'skipped' %}{{ handle_empty }}{% elif compress_large.status is not defined or compress_large.status != 'skipped' %}{{ compress_large }}{% elif expand_small.status is not defined or expand_small.status != 'skipped' %}{{ expand_small }}{% else %}No content processed.{% endif %}"
  output_file: "{{ save_result.path }}"
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
- This pipeline produces Analysis results, Markdown documents - adjust output configuration accordingly

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
- 2. Run: python scripts/run_pipeline.py examples/control_flow_conditional.yaml
- 3. Monitor the output for progress and any error messages
- 4. Check the output directory for generated results

### Expected Outputs
- Generated Analysis results in the specified output directory
- Generated Markdown documents in the specified output directory
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

*Tutorial generated on 2025-08-27T23:40:24.395841*
