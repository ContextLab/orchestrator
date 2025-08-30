# Pipeline Tutorial: code_optimization

## Overview

**Complexity Level**: Intermediate  
**Difficulty Score**: 40/100  
**Estimated Runtime**: 5-15 minutes  

### Purpose
This pipeline demonstrates data_flow, interactive_workflows, llm_integration and provides a practical example of orchestrator's capabilities for intermediate-level workflows.

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
- **template_usage**: Uses 7 template patterns for dynamic content
- **feature_highlights**: Demonstrates 4 key orchestrator features

### Data Flow
This pipeline processes input parameters through 4 steps to generate the specified outputs.

### Control Flow
Follows linear execution flow from first step to last step.

### Pipeline Configuration
```yaml
id: code-optimization
name: Code Optimization Pipeline
description: Analyze and optimize code for performance and best practices

parameters:
  code_file:
    type: string
    required: true
    description: Path to the code file to optimize
  language:
    type: string
    default: python
    description: Programming language

steps:
  - id: read_code
    tool: filesystem
    action: read
    parameters:
      path: "{{code_file}}"
  
  - id: analyze_code
    action: analyze_text
    parameters:
      text: |
        Analyze this {{language}} code for optimization opportunities:
        
        ```{{language}}
        {{read_code.content}}
        ```
        
        Identify:
        1. Performance bottlenecks
        2. Code quality issues
        3. Best practice violations
      model: <AUTO task="analyze">Select model best suited for code analysis</AUTO>
      analysis_type: "code_quality"
    dependencies:
      - read_code
  
  - id: optimize_code
    action: generate_text
    parameters:
      prompt: |
        Based on this analysis:
        {{analyze_code.result}}
        
        Provide an optimized version of the {{language}} code that addresses the identified issues.
        
        IMPORTANT: 
        - Return ONLY valid {{language}} code syntax
        - Use {{language}}-specific best practices and conventions
        - Do not include any markdown formatting or explanations
        - Do not include ```{{language}}``` or any markdown blocks
        - The output must be pure, executable {{language}} code
        
        Language: {{language}}
      model: <AUTO task="generate">Select model for code generation</AUTO>
      max_tokens: 2000
    dependencies:
      - analyze_code
  
  - id: clean_optimized_code
    action: generate_text
    parameters:
      prompt: |
        Extract ONLY the {{language}} code from the following text, removing any markdown formatting or explanations:
        
        {{optimize_code.result}}
        
        Return ONLY the pure {{language}} code without any ```{{language}}``` blocks, explanations, or formatting.
        The output must be valid, executable {{language}} code with proper {{language}} syntax.
        
        Target Language: {{language}}
      model: <AUTO task="extract">Select model for code extraction</AUTO>
      max_tokens: 2000
    dependencies:
      - optimize_code
  
  - id: save_optimized_code
    tool: filesystem
    action: write
    parameters:
      path: "examples/outputs/code_optimization/optimized_{{code_file | basename}}"
      content: "{{clean_optimized_code.result}}"
    dependencies:
      - clean_optimized_code
  
  - id: save_analysis_report
    tool: filesystem
    action: write
    parameters:
      path: "examples/outputs/code_optimization/code_optimization_report_{{ execution.timestamp | slugify }}.md"
      content: |
        # Code Optimization Report
        
        **File:** {{code_file}}
        **Language:** {{language}}
        **Date:** {{ execution.timestamp | date('%Y-%m-%d %H:%M:%S') }}
        
        ## Analysis Results
        
        {{analyze_code.result}}
        
        ## Optimization Summary
        
        The optimized code has been saved to: examples/outputs/code_optimization/optimized_{{code_file | basename}}
        
        ## Original vs Optimized
        
        ### Original Code Issues Identified:
        See analysis above for detailed breakdown.
        
        ### Optimized Code Benefits:
        - Improved performance through algorithmic optimizations
        - Enhanced code quality and maintainability
        - Better error handling and validation
        - Adherence to best practices and conventions
    dependencies:
      - analyze_code
      - optimize_code

outputs:
  analysis: "{{analyze_code.result}}"
  optimized_code: "{{clean_optimized_code.result}}"
  optimized_file: "optimized_{{code_file | basename}}"
  report_file: "{{save_analysis_report.filepath}}"
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
- This pipeline produces Analysis results, Markdown documents, Reports - adjust output configuration accordingly

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
- 2. Run: python scripts/run_pipeline.py examples/code_optimization.yaml
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
- **General Execution Errors**: Check the logs for specific error messages and verify your orchestrator installation

### Verification Steps
- Check that the pipeline completed without errors
- Verify all expected output files were created
- Review the output content for quality and accuracy
- Review generated content for relevance and quality

---

*Tutorial generated on 2025-08-27T23:40:24.395771*
