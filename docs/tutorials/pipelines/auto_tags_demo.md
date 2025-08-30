# Pipeline Tutorial: auto_tags_demo

## Overview

**Complexity Level**: Unknown  
**Difficulty Score**: 0/100  
**Estimated Runtime**: unknown  

### Purpose
This pipeline demonstrates  and provides a practical example of orchestrator's capabilities for unknown-level workflows.

### Use Cases
- Pipeline analysis failed: mapping values are not allowed here
  in "<unicode string>", line 40, column 138:
     ... what processing approach is best: "simple", "advanced", or "expe ... 
                                         ^

### Prerequisites
- Basic understanding of YAML syntax

### Key Concepts
- None specified

## Pipeline Breakdown

### Configuration Analysis
- **input_section**: Defines the data inputs and parameters for the pipeline
- **steps_section**: Contains the sequence of operations to be executed
- **output_section**: Specifies how results are formatted and stored
- **template_usage**: Uses 0 template patterns for dynamic content
- **feature_highlights**: Demonstrates 0 key orchestrator features

### Data Flow
This pipeline processes input parameters through 0 steps to generate the specified outputs.

### Control Flow
Follows linear execution flow from first step to last step.

### Pipeline Configuration
```yaml
# AUTO Tags Demo Pipeline
# Demonstrates dynamic decision-making using AUTO tags
id: auto_tags_demo
name: AUTO Tags Demonstration Pipeline
description: Shows how AUTO tags enable dynamic AI-driven decisions in pipelines
version: "1.0.0"

parameters:
  content:
    type: string
    default: "Artificial intelligence is revolutionizing healthcare through machine learning algorithms that can analyze medical images, predict disease outcomes, and personalize treatment plans."
  task_complexity:
    type: string
    default: "medium"

steps:
  # Dynamic content analysis
  - id: analyze_content
    action: generate_text
    parameters:
      prompt: |
        Analyze this content and determine its type:
        "{{ content }}"
        
        Return only the content type (one word): "technical", "academic", "casual", or "business"
      model: "gpt-4o-mini"
      max_tokens: 10
    
  # AUTO tag for dynamic model selection based on content analysis
  - id: select_processing_strategy
    action: generate_text
    parameters:
      prompt: |
        Content: "{{ content }}"
        Content Type: {{ analyze_content }}
        Task Complexity: {{ task_complexity }}
        
        Choose processing strategy: "simple", "advanced", or "expert"
      model: <AUTO task="strategy">Based on content type {{ analyze_content }} and complexity {{ task_complexity }}, select appropriate model for strategic decision</AUTO>
      strategy: <AUTO>Given content type "{{ analyze_content }}" and complexity "{{ task_complexity }}", what processing approach is best: "simple", "advanced", or "expert"?</AUTO>
      max_tokens: 20
    dependencies:
      - analyze_content
      
  # Dynamic formatting decision
  - id: choose_output_format
    action: generate_text
    parameters:
      prompt: |
        For the following content and strategy, choose output format:
        Content Type: {{ analyze_content }}
        Strategy: {{ select_processing_strategy.strategy }}
        
        Return one word: "markdown", "json", or "plain"
      format: <AUTO>What output format works best for {{ analyze_content }} content with {{ select_processing_strategy.strategy }} strategy?</AUTO>
      model: "gpt-4o-mini"
      max_tokens: 10
    dependencies:
      - select_processing_strategy
      
  # Content processing with dynamic parameters
  - id: process_content
    action: generate_text
    parameters:
      prompt: |
        Process this content using the {{ select_processing_strategy.strategy }} approach:
        {{ content }}
        
        Provide analysis in {{ choose_output_format.format }} format.
      model: "{{ select_processing_strategy.model }}"
      detail_level: <AUTO>For {{ select_processing_strategy.strategy }} strategy, choose detail level: "brief", "standard", or "comprehensive"</AUTO>
      max_tokens: <AUTO type="integer" min="100" max="1000">Based on strategy {{ select_processing_strategy.strategy }}, how many tokens needed for good analysis?</AUTO>
    dependencies:
      - choose_output_format
      
  # Dynamic quality assessment
  - id: assess_quality
    action: generate_text
    parameters:
      prompt: |
        Assess the quality of this analysis:
        {{ process_content }}
        
        Rate from 1-10 and provide reasoning.
      model: "gpt-4o-mini"
      criteria: <AUTO>What quality criteria should we use for {{ analyze_content }} content: "accuracy", "completeness", "clarity", or "all"?</AUTO>
      max_tokens: 200
    dependencies:
      - process_content
      
  # Conditional improvement step
  - id: improve_if_needed
    action: generate_text
    condition: <AUTO>Quality score from assessment is {{ assess_quality }}. Should we improve if score < 7? Answer 'true' or 'false'</AUTO>
    parameters:
      prompt: |
        Improve this analysis based on quality assessment:
        Original: {{ process_content }}
        Assessment: {{ assess_quality }}
        
        Provide enhanced version.
      model: "gpt-4o-mini"
      max_tokens: 500
    dependencies:
      - assess_quality
      
  # Generate final report
  - id: create_report
    action: generate_text
    parameters:
      prompt: |
        Create a summary report of the AUTO tags demo:
        
        ## AUTO Tags Demo Results
        
        **Original Content:** {{ content }}
        
        **Dynamic Decisions Made:**
        - Content Type: {{ analyze_content }}
        - Processing Strategy: {{ select_processing_strategy.strategy }}
        - Selected Model: {{ select_processing_strategy.model }}
        - Output Format: {{ choose_output_format.format }}
        - Detail Level: {{ process_content.detail_level }}
        - Token Allocation: {{ process_content.max_tokens }}
        - Quality Criteria: {{ assess_quality.criteria }}
        
        **Final Analysis:**
        {% if improve_if_needed %}
        {{ improve_if_needed }}
        {% else %}
        {{ process_content }}
        {% endif %}
        
        **Quality Assessment:**
        {{ assess_quality }}
        
        This demo shows how AUTO tags enable dynamic, context-aware pipeline behavior.
      model: "gpt-4o-mini"
      max_tokens: 1000
    dependencies:
      - improve_if_needed

outputs:
  content_type: "{{ analyze_content }}"
  processing_strategy: "{{ select_processing_strategy.strategy }}"
  selected_model: "{{ select_processing_strategy.model }}"
  output_format: "{{ choose_output_format.format }}"
  final_analysis: "{% if improve_if_needed %}{{ improve_if_needed }}{% else %}{{ process_content }}{% endif %}"
  quality_score: "{{ assess_quality }}"
  demo_report: "{{ create_report }}"
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

## Remixing Instructions

### Compatible Patterns
- Most basic pipelines can be combined with this pattern

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

## Hands-On Exercise

### Execution Instructions
- 1. Navigate to your orchestrator project directory
- 2. Run: python scripts/run_pipeline.py examples/auto_tags_demo.yaml
- 3. Monitor the output for progress and any error messages
- 4. Check the output directory for generated results

### Expected Outputs
- Text-based results printed to console
- Execution logs showing step-by-step progress
- Completion message with runtime statistics
- No error messages or warnings (successful execution)

### Troubleshooting
- **General Execution Errors**: Check the logs for specific error messages and verify your orchestrator installation

### Verification Steps
- Check that the pipeline completed without errors
- Verify all expected output files were created
- Review the output content for quality and accuracy

---

*Tutorial generated on 2025-08-27T23:40:24.395713*
