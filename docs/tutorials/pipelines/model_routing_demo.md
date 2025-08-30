# Pipeline Tutorial: model_routing_demo

## Overview

**Complexity Level**: Intermediate  
**Difficulty Score**: 60/100  
**Estimated Runtime**: 5-15 minutes  

### Purpose
This pipeline demonstrates data_flow, file_inclusion, interactive_workflows and provides a practical example of orchestrator's capabilities for intermediate-level workflows.

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
- **template_usage**: Uses 10 template patterns for dynamic content
- **feature_highlights**: Demonstrates 6 key orchestrator features

### Data Flow
This pipeline processes input parameters through 6 steps to generate the specified outputs.

### Control Flow
Follows linear execution flow from first step to last step.

### Pipeline Configuration
```yaml
id: model-routing-demo
# Model Routing Demonstration Pipeline
# Shows intelligent model selection and routing strategies

name: model-routing-demonstration
description: Demonstrates intelligent model routing capabilities
version: "1.0.0"

inputs:
  task_budget: 10.00  # Dollar budget for this pipeline
  priority: "balanced"  # cost, speed, or quality
  
steps:
  # Step 1: Analyze task complexity to route appropriately
  - id: assess_requirements
    tool: multi-model-routing
    action: execute
    parameters:
      action: "route"
      tasks:
        - task: "Summarize this document in 2-3 sentences"
          context: |
            Artificial intelligence continues to revolutionize industries worldwide.
            From healthcare to finance, AI applications are becoming increasingly sophisticated.
            Machine learning models can now process vast amounts of data in real-time,
            enabling predictive analytics and automated decision-making at unprecedented scales.
        - task: "Write a Python function to calculate fibonacci numbers"
          context: "Should be efficient and handle large inputs"
        - task: "Analyze sales data trends"
          context: "Q4 2024 sales data showing 15% growth"
      routing_strategy: "{{ priority }}"  # Maps to cost_optimized, balanced, or quality_optimized
      constraints:
        total_budget: "{{ task_budget }}"
        max_latency: 30.0
      
  # Step 2: Simple task with cheap model
  - id: summarize_document
    action: generate_text
    parameters:
      prompt: |
        Provide a concise 2-3 sentence summary of this document. Focus on the key points only:
        
        Artificial intelligence continues to revolutionize industries worldwide.
        From healthcare to finance, AI applications are becoming increasingly sophisticated.
        Machine learning models can now process vast amounts of data in real-time,
        enabling predictive analytics and automated decision-making at unprecedented scales.
      # Use routing recommendation
      model: "{{ assess_requirements.recommendations[0].model }}"
    dependencies:
      - assess_requirements
      
  # Step 3: Complex task with specialized model
  - id: generate_code
    action: generate_text
    parameters:
      prompt: |
        Write a complete, production-ready Python function to calculate fibonacci numbers.
        Requirements:
        - Use efficient algorithm (memoization or iterative)
        - Include proper type hints
        - Add comprehensive docstring
        - Handle edge cases (negative numbers, zero, large inputs)
        - Return the code only, no explanations
      # Use a working model since GPT-5-nano has issues
      model: "{{ assess_requirements.recommendations[0].model }}"
    dependencies:
      - assess_requirements
      
  # Step 4: Data analysis with balanced model
  - id: analyze_data
    action: analyze_text
    parameters:
      text: |
        Q4 2024 Sales Report:
        - Total Revenue: $2.5M (15% growth YoY)
        - Units Sold: 15,000 (12% growth YoY)
        - Average Order Value: $167 (3% growth YoY)
        - Top Product Categories: Electronics (45%), Home & Garden (30%), Clothing (25%)
      analysis_type: "trends"
      prompt: "Analyze the sales data trends and provide exactly 3 key insights. For each insight, include: the finding, why it matters, and a specific action. Be concise and data-driven. No conversational text."
      # Use routing recommendation
      model: "{{ assess_requirements.recommendations[2].model }}"
    dependencies:
      - assess_requirements
      
  # Step 5: Cost optimization demonstration
  - id: batch_processing
    tool: multi-model-routing
    action: execute
    parameters:
      action: "optimize_batch"
      tasks:
        - "Translate to Spanish (provide translation only): Hello World"
        - "Translate to French (provide translation only): Good morning"
        - "Translate to German (provide translation only): Thank you"
        - "Translate to Italian (provide translation only): Goodbye"
      optimization_goal: "minimize_cost"
      constraints:
        max_budget_per_task: 0.05
    dependencies:
      - analyze_data
      
  # Step 6: Generate routing report
  - id: routing_report
    tool: filesystem
    action: write
    parameters:
      path: "{{ output_path }}/{{ priority }}_routing_analysis.md"
      content: |
        # Model Routing Results
        
        ## Configuration
        - Budget: ${{ task_budget }}
        - Priority: {{ priority }}
        
        ## Task Routing
        
        ### Document Summary
        - Assigned Model: {{ assess_requirements.recommendations[0].model }}
        - Estimated Cost: ${{ assess_requirements.recommendations[0].estimated_cost }}
        - Result: {{ summarize_document.result }}
        
        ### Code Generation
        - Assigned Model: {{ assess_requirements.recommendations[1].model }}
        - Estimated Cost: ${{ assess_requirements.recommendations[1].estimated_cost }}
        - Code Generated: 
        ```python
        {{ generate_code.result | replace('```python', '') | replace('```', '') | truncate(500) }}
        ```
        
        ### Data Analysis
        - Assigned Model: {{ assess_requirements.recommendations[2].model }}
        - Estimated Cost: ${{ assess_requirements.recommendations[2].estimated_cost }}
        - Insights: {{ analyze_data.result }}
        
        ## Batch Translation Optimization
        - Optimization Goal: {{ batch_processing.optimization_goal | default('minimize_cost') }}
        - Total Tasks: {{ batch_processing.results | length }}
        - Models Used: {{ batch_processing.models_used | join(', ') }}
        - Total Cost: ${{ batch_processing.total_cost }}
        - Average Cost per Task: ${{ (batch_processing.total_cost / (batch_processing.results | length)) | round(4) }}
        
        ### Translation Results:
        1. Spanish: {{ batch_processing.results[0] | truncate(100) }}
        2. French: {{ batch_processing.results[1] | truncate(100) }}
        3. German: {{ batch_processing.results[2] | truncate(100) }}
        4. Italian: {{ batch_processing.results[3] | truncate(100) }}
        
        ## Summary
        - Total Pipeline Cost: ${{ assess_requirements.total_estimated_cost + batch_processing.total_cost }}
        - Budget Remaining: ${{ task_budget - (assess_requirements.total_estimated_cost + batch_processing.total_cost) }}
        - Optimization Achieved: {{ priority }} routing successfully implemented
    dependencies:
      - batch_processing
      
      
outputs:
  routing_report: "{{ routing_report.path }}"
  total_cost: "{{ assess_requirements.total_estimated_cost | default(0) + batch_processing.total_cost | default(0) }}"
  models_used: "{{ (assess_requirements.models_selected | default([])) + (batch_processing.models_used | default([])) }}"
  report_path: "{{ routing_report.path }}"
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
- 2. Run: python scripts/run_pipeline.py examples/model_routing_demo.yaml
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

*Tutorial generated on 2025-08-27T23:40:24.396362*
