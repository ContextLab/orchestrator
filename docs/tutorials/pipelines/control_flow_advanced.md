# Pipeline Tutorial: control_flow_advanced

## Overview

**Complexity Level**: Advanced  
**Difficulty Score**: 70/100  
**Estimated Runtime**: 15+ minutes  

### Purpose
This pipeline illustrates advanced control flow patterns in orchestrator. It demonstrates  for building dynamic, conditional workflows.

### Use Cases
- AI-powered content generation
- Batch processing with logic
- Conditional workflow automation
- Dynamic pipeline execution

### Prerequisites
- Basic understanding of YAML syntax
- Experience with intermediate pipeline patterns
- Understanding of error handling and system integration
- Familiarity with external APIs and tools

### Key Concepts
- Conditional logic and branching
- Data flow between pipeline steps
- Iterative processing with loops
- Large language model integration
- Template variable substitution

## Pipeline Breakdown

### Configuration Analysis
- **input_section**: Defines the data inputs and parameters for the pipeline
- **steps_section**: Contains the sequence of operations to be executed
- **output_section**: Specifies how results are formatted and stored
- **template_usage**: Uses 1 template patterns for dynamic content
- **feature_highlights**: Demonstrates 8 key orchestrator features

### Data Flow
This pipeline processes input parameters through 8 steps to generate the specified outputs.

### Control Flow
Follows linear execution flow from first step to last step.

### Pipeline Configuration
```yaml
id: control-flow-advanced
# Advanced Control Flow Example
# Combines conditionals and loops in a practical scenario
name: Multi-Stage Text Processing
description: Process text through multiple stages with conditional paths
version: "1.0.0"

parameters:
  input_text:
    type: string
    description: Text to process
    default: "This is a sample text for processing."
  languages:
    type: array
    default: ["es", "fr", "de"]
    description: Languages to translate to
  quality_threshold:
    type: number
    default: 0.7
    description: Quality threshold for translations
  output:
    type: string
    default: "examples/outputs/control_flow_advanced"
    description: Output directory for results
    
steps:
  # Initial analysis
  - id: analyze_text
    action: analyze_text
    parameters:
      text: "{{ input_text }}"
      model: "ollama:llama3.2:1b"
      analysis_type: "comprehensive"
      
  # Determine if text needs enhancement
  - id: check_quality
    action: generate_text
    parameters:
      prompt: |
        Based on this text analysis, determine if the text quality is below {{ quality_threshold }} (out of 1.0) and needs improvement:
        
        Analysis: {{ analyze_text }}
        
        Respond with either "improve" or "acceptable" based on the quality assessment.
      model: "ollama:llama3.2:1b"
      max_tokens: 50
    dependencies:
      - analyze_text
      
  # Conditional enhancement
  - id: enhance_text
    action: generate_text
    if: "{{ 'improve' in check_quality }}"
    parameters:
      prompt: |
        Improve the following text to make it clearer and more professional. Output ONLY the improved version, no explanations:
        
        Original text: {{ input_text }}
      model: "ollama:llama3.2:1b"
      max_tokens: 500
    dependencies:
      - check_quality
      
  # Determine which text to use (creates a concrete value we can reference)
  - id: select_text
    action: generate_text
    parameters:
      prompt: |
        {% if check_quality and 'improve' in check_quality and enhance_text %}
        Output ONLY this text without any changes or additions:
        {{ enhance_text }}
        {% else %}
        Output ONLY this text without any changes or additions:
        {{ input_text }}
        {% endif %}
      model: "ollama:llama3.2:1b"
      max_tokens: 600
    dependencies:
      - enhance_text
      - check_quality
      
  # Translate to multiple languages
  - id: translate_text
    for_each: "{{ languages }}"
    max_parallel: 2
    steps:
      # Translate
      - id: translate
        action: generate_text
        parameters:
          prompt: |
            Translate the following text to {{ $item }}. Provide ONLY the direct translation, no explanations or commentary:
            
            Text to translate: "{{ input_text }}"
          model: "ollama:llama3.2:1b"
          max_tokens: 600
          
      # Validate translation quality
      - id: validate_translation
        action: generate_text
        parameters:
          prompt: |
            Assess the quality of this translation from English to {{ $item }}:
            
            Original English: "{{ input_text }}"
            Translation: "{{ translate }}"
            
            Evaluate the translation for:
            1. Accuracy - Does it convey the same meaning as the original?
            2. Fluency - Is it natural and grammatically correct in {{ $item }}?
            3. Completeness - Are all concepts from the original included?
            
            Provide a brief quality assessment and rate it as: excellent, good, acceptable, or poor.
          model: "ollama:llama3.2:1b"
          max_tokens: 1500
        dependencies:
          - translate
          
      # Save translation
      - id: save_translation
        tool: filesystem
        action: write
        parameters:
          path: "{{ output }}/translations/{{ input_text[:50] | slugify }}_{{ $item }}.md"
          content: |
            # Translation to {{ item | upper }}
            
            ## Original Text
            {{ input_text }}
            
            ## Translated Text
            {{ translate }}
            
            ## Translation Quality Assessment
            {{ validate_translation }}
        dependencies:
          - validate_translation
    dependencies:
      - select_text
      
  # Example of multiple conditionals: Create different summary types
  - id: create_brief_summary
    action: generate_text
    if: "{{ languages | length <= 2 }}"
    parameters:
      prompt: |
        Create a brief summary of the translation process.
        Original text: {{ input_text }}
        Languages: {{ languages }}
      model: "ollama:llama3.2:1b"
      max_tokens: 100
    dependencies:
      - translate_text
      
  - id: create_detailed_summary  
    action: generate_text
    if: "{{ languages | length > 2 }}"
    parameters:
      prompt: |
        Create a detailed summary of the multi-language translation process.
        Original text: {{ input_text }}
        Languages: {{ languages }}
        Analysis: {{ analyze_text }}
      model: "ollama:llama3.2:1b"
      max_tokens: 200
    dependencies:
      - translate_text
      
  # Example of aggregation: Count successful operations
  - id: check_translation_count
    action: generate_text
    parameters:
      prompt: |
        Count how many languages were successfully translated.
        Languages attempted: {{ languages }}
        Brief summary: {{ create_brief_summary | default('N/A') }}
        Detailed summary: {{ create_detailed_summary | default('N/A') }}
        Return just a number.
      model: "ollama:llama3.2:1b"
      max_tokens: 10
    dependencies:
      - create_brief_summary
      - create_detailed_summary
  
  # Create final report
  - id: create_report
    tool: filesystem
    action: write
    parameters:
      path: "{{ output }}/{{ input_text[:50] | slugify }}_report.md"
      content: |
        # Multi-Stage Text Processing Report
        
        ## Original Text
        {{ input_text }}
        
        ## Analysis
        {{ analyze_text }}
        
        ## Quality Check Result
        {{ check_quality }}
        
        ## Enhancement Status
        {% if enhance_text %}Enhanced version was created{% else %}Original text was sufficient{% endif %}
        
        ## Final Text Used for Translation
        {% if select_text %}{{ select_text }}{% else %}{{ input_text }}{% endif %}
        
        ## Translations
        Attempted translations to: {{ languages | join(', ') }}
        
        Check the {{ output }}/translations/ directory for successful translations.
    dependencies:
      - translate_text
      
outputs:
  analysis: "{{ analyze_text }}"
  quality_check: "{{ check_quality }}"
  enhanced: "{{ enhance_text | default('N/A') }}"
  final_text: "{{ select_text }}"
  translation_count: "{{ check_translation_count }}"
  report_file: "{{ output }}/{{ input_text[:50] | slugify }}_report.md"
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
- This pipeline produces Analysis results, Markdown documents, Reports - adjust output configuration accordingly

## Remixing Instructions

### Compatible Patterns
- fact_checker.yaml - for content verification
- research workflows - for information gathering

### Extension Ideas
- Build modular components for reusability
- Add performance monitoring and optimization
- Implement advanced security and access controls

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
- 2. Run: python scripts/run_pipeline.py examples/control_flow_advanced.yaml
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
- **Complex Logic Errors**: Review the pipeline configuration and ensure all advanced features are properly configured
- **General Execution Errors**: Check the logs for specific error messages and verify your orchestrator installation

### Verification Steps
- Check that the pipeline completed without errors
- Verify all expected output files were created
- Review the output content for quality and accuracy
- Review generated content for relevance and quality

---

*Tutorial generated on 2025-08-27T23:40:24.395808*
