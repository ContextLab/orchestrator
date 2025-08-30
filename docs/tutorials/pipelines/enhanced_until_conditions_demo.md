# Pipeline Tutorial: enhanced_until_conditions_demo

## Overview

**Complexity Level**: Intermediate  
**Difficulty Score**: 40/100  
**Estimated Runtime**: 5-15 minutes  

### Purpose
This pipeline demonstrates data_flow, template_variables, until_conditions and provides a practical example of orchestrator's capabilities for intermediate-level workflows.

### Use Cases
- General automation tasks

### Prerequisites
- Basic understanding of YAML syntax
- Familiarity with template variables and data flow
- Understanding of basic control flow concepts

### Key Concepts
- Data flow between pipeline steps
- Template variable substitution

## Pipeline Breakdown

### Configuration Analysis
- **input_section**: Defines the data inputs and parameters for the pipeline
- **steps_section**: Contains the sequence of operations to be executed
- **output_section**: Specifies how results are formatted and stored
- **template_usage**: Uses 4 template patterns for dynamic content
- **feature_highlights**: Demonstrates 4 key orchestrator features

### Data Flow
This pipeline processes input parameters through 4 steps to generate the specified outputs.

### Control Flow
Follows linear execution flow from first step to last step.

### Pipeline Configuration
```yaml
---
name: "Enhanced Until Conditions Demo"
description: "Demonstrates structured until condition evaluation with performance tracking"

pipeline:
  inputs:
    - name: target_quality
      default: 0.8
    - name: max_attempts  
      default: 5

  steps:
    - id: initialize_metrics
      action: debug
      parameters:
        message: "Starting enhanced until condition demo with target quality: {{ target_quality }}"

    - id: quality_improvement_loop
      while: "true"  # Always continue from while perspective  
      until: "<AUTO>Quality score is {{ quality_score }}. Is this >= {{ target_quality }}? Answer 'true' or 'false'.</AUTO>"
      max_iterations: "{{ max_attempts }}"
      loop_name: "quality_loop"
      steps:
        - id: simulate_quality_work
          action: debug
          parameters:
            message: "Iteration {{ $iteration }}: Current quality score: {{ quality_score | default('0.2') }}"
            
        - id: update_quality_score
          action: debug
          parameters:
            # Simulate quality improvement over iterations
            message: "Improving quality from {{ quality_score | default('0.2') }} to {{ ($iteration + 1) * 0.25 }}"
            quality_score: "{{ ($iteration + 1) * 0.25 }}"
            
        - id: check_progress
          action: debug  
          parameters:
            message: "Loop {{ quality_loop.loop_name }} iteration {{ $iteration }} complete. Quality: {{ quality_score }}"

    - id: final_report
      action: debug
      parameters:
        message: "Quality improvement loop completed. Final quality score achieved."
      dependencies:
        - quality_improvement_loop

outputs:
  - name: completion_status
    value: "Enhanced until conditions demo completed successfully"
  - name: final_iteration_count
    value: "{{ quality_loop.iteration }}"
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
- This pipeline produces Reports - adjust output configuration accordingly

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
- 2. Run: python scripts/run_pipeline.py examples/enhanced_until_conditions_demo.yaml
- 3. Monitor the output for progress and any error messages
- 4. Check the output directory for generated results

### Expected Outputs
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

*Tutorial generated on 2025-08-27T23:40:24.396075*
