# Pipeline Tutorial: creative_image_pipeline

## Overview

**Complexity Level**: Intermediate  
**Difficulty Score**: 40/100  
**Estimated Runtime**: 5-15 minutes  

### Purpose
This pipeline showcases creative content generation capabilities. It demonstrates conditional_execution, data_flow, interactive_workflows for building AI-powered creative workflows.

### Use Cases
- Content creation and marketing materials
- Creative writing and ideation
- Visual content generation

### Prerequisites
- Basic understanding of YAML syntax
- Familiarity with template variables and data flow
- Understanding of basic control flow concepts

### Key Concepts
- Conditional logic and branching
- Data flow between pipeline steps
- Template variable substitution

## Pipeline Breakdown

### Configuration Analysis
- **input_section**: Defines the data inputs and parameters for the pipeline
- **steps_section**: Contains the sequence of operations to be executed
- **output_section**: Specifies how results are formatted and stored
- **template_usage**: Uses 6 template patterns for dynamic content
- **feature_highlights**: Demonstrates 4 key orchestrator features

### Data Flow
This pipeline processes input parameters through 4 steps to generate the specified outputs.

### Control Flow
Follows linear execution flow from first step to last step.

### Pipeline Configuration
```yaml
# Creative Image Generation Pipeline
# Demonstrates AI-powered image generation and analysis
id: creative_image_pipeline
name: Creative Image Generation Pipeline
description: Generate and analyze images based on creative prompts
version: "1.0.0"

parameters:
  base_prompt:
    type: string
    default: "A serene landscape with mountains and a lake"
  num_variations:
    type: integer
    default: 3
  art_styles:
    type: array
    default: ["photorealistic", "impressionist", "cyberpunk"]
  output_path:
    type: string
    default: "examples/outputs/creative_image_pipeline"

steps:
  - id: generate_folder_name
    action: generate
    parameters:
      prompt: |
        Convert this image prompt into a short, descriptive folder name (no spaces, use underscores):
        "{{ base_prompt }}"
        
        Return ONLY the folder name, max 30 characters, lowercase.
        Example: "sunset_over_mountains"
      topic: AUTO
    
  - id: generate_base_image
    action: execute
    tool: image-generation
    parameters:
      prompt: "{{ base_prompt }}"
      size: "1024x1024"
      style: "high quality, detailed"
      output_format: "file"
      output_path: "{{ output_path }}/{{ generate_folder_name }}"
      filename: "base.png"
    dependencies:
      - generate_folder_name
    
  - id: analyze_base
    action: execute
    tool: image-analysis
    parameters:
      image: "{{ generate_base_image.images[0].path }}"
      analysis_type: "describe"
      detail_level: "high"
      output_format: "json"
    dependencies:
      - generate_base_image
    
  - id: generate_style_variation_1
    action: execute
    tool: image-generation
    parameters:
      prompt: "{{ base_prompt }}, {{ art_styles[0] }} style"
      size: "1024x1024"
      style: "{{ art_styles[0] }}"
      output_format: "file"
      output_path: "{{ output_path }}/{{ generate_folder_name }}"
      filename: "{{ art_styles[0] }}.png"
    dependencies:
      - analyze_base
      
  - id: generate_style_variation_2
    action: execute
    tool: image-generation
    parameters:
      prompt: "{{ base_prompt }}, {{ art_styles[1] }} style"
      size: "1024x1024"
      style: "{{ art_styles[1] }}"
      output_format: "file"
      output_path: "{{ output_path }}/{{ generate_folder_name }}"
      filename: "{{ art_styles[1] }}.png"
    dependencies:
      - analyze_base
      
  - id: generate_style_variation_3
    action: execute
    tool: image-generation
    parameters:
      prompt: "{{ base_prompt }}, {{ art_styles[2] }} style"
      size: "1024x1024"
      style: "{{ art_styles[2] }}"
      output_format: "file"
      output_path: "{{ output_path }}/{{ generate_folder_name }}"
      filename: "{{ art_styles[2] }}.png"
    dependencies:
      - analyze_base
    
  - id: enhance_prompt
    action: execute
    tool: prompt-optimization
    parameters:
      prompt: "{{ base_prompt }}"
      task: "image-generation"
      optimization_goal: "artistic_quality"
      include_style_suggestions: true
    dependencies:
      - analyze_base
    
  - id: generate_enhanced
    action: execute
    tool: image-generation
    parameters:
      prompt: "{{ enhance_prompt.optimized_prompt }}"
      size: "1024x1024"
      num_images: 1
      output_format: "file"
      output_path: "{{ output_path }}/{{ generate_folder_name }}"
      filename: "enhanced.png"
    dependencies:
      - enhance_prompt
    
  - id: analyze_enhanced_1
    action: execute
    tool: image-analysis
    parameters:
      image: "{{ generate_enhanced.images[0].path }}"
      analysis_type: "classify"
      output_format: "structured"
      detail_level: "medium"
    dependencies:
      - generate_enhanced
    condition: "{{ generate_enhanced.images | length > 0 }}"
    
  - id: count_images
    action: execute
    tool: filesystem
    parameters:
      action: list
      path: "{{ output_path }}/{{ generate_folder_name }}"
      pattern: "*.png"
    dependencies:
      - generate_enhanced
      - analyze_enhanced_1
      
  - id: create_gallery_report
    action: generate
    parameters:
      prompt: |
        Generate ONLY the following markdown content without any code blocks, explanations, or additional text:
        
        # Creative Image Generation Results
        
        ## Original Prompt
        "{{ base_prompt }}"
        
        ## Output Folder
        `{{ output_path }}/{{ generate_folder_name }}/`
        
        ## Base Image Analysis
        {{ analyze_base.analysis.result }}
        
        ## Style Variations Generated
        {% for style in art_styles %}
        - {{ style }} style
        {% endfor %}
        
        ## Enhanced Prompt
        - Original: "{{ base_prompt }}"
        - Enhanced: "{{ enhance_prompt.optimized_prompt }}"
        
        ## Generated Images
        
        | Image | Description |
        |-------|-------------|
        | ![Base](base.png) | Original prompt rendering |
        {% for style in art_styles %}
        | ![{{ style|capitalize }}]({{ style }}.png) | {{ style|capitalize }} style variation |
        {% endfor %}
        | ![Enhanced](enhanced.png) | AI-enhanced prompt rendering |
        
        **Total images generated:** {{ count_images.count }}
      topic: AUTO
    dependencies:
      - count_images
      
  - id: save_gallery_report
    action: execute
    tool: filesystem
    parameters:
      action: write
      path: "{{ output_path }}/{{ generate_folder_name }}/README.md"
      content: "{{ create_gallery_report }}"
    dependencies:
      - create_gallery_report

outputs:
  output_folder: "{{ output_path }}/{{ generate_folder_name }}"
  base_image: "{{ generate_base_image.images[0] }}"
  enhanced_prompt: "{{ enhance_prompt.optimized_prompt }}"
  total_generated: "{{ count_images.count }}"
  gallery_report: "{{ output_path }}/{{ generate_folder_name }}/README.md"
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
- This pipeline produces Analysis results, Images, JSON data, Markdown documents, Reports - adjust output configuration accordingly

## Remixing Instructions

### Compatible Patterns
- Most basic pipelines can be combined with this pattern

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

## Hands-On Exercise

### Execution Instructions
- 1. Navigate to your orchestrator project directory
- 2. Run: python scripts/run_pipeline.py examples/creative_image_pipeline.yaml
- 3. Monitor the output for progress and any error messages
- 4. Check the output directory for generated results

### Expected Outputs
- Generated Analysis results in the specified output directory
- Generated Images in the specified output directory
- Generated JSON data in the specified output directory
- Generated Markdown documents in the specified output directory
- Generated Reports in the specified output directory
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

---

*Tutorial generated on 2025-08-27T23:40:24.395947*
