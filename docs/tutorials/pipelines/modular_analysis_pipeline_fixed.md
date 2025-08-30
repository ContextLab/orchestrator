# Pipeline Tutorial: modular_analysis_pipeline_fixed

## Overview

**Complexity Level**: Intermediate  
**Difficulty Score**: 45/100  
**Estimated Runtime**: 5-15 minutes  

### Purpose
This pipeline demonstrates conditional_execution, csv_processing, data_flow and provides a practical example of orchestrator's capabilities for intermediate-level workflows.

### Use Cases
- General automation tasks

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
- **template_usage**: Uses 17 template patterns for dynamic content
- **feature_highlights**: Demonstrates 5 key orchestrator features

### Data Flow
This pipeline processes input parameters through 5 steps to generate the specified outputs.

### Control Flow
Follows linear execution flow from first step to last step.

### Pipeline Configuration
```yaml
# Modular Analysis Pipeline with Sub-Pipelines (Fixed Version)
# Demonstrates using sub-pipelines for modular workflow design
id: modular_analysis
name: Modular Analysis Pipeline
description: Main pipeline that orchestrates multiple analysis sub-pipelines
version: "2.0.0"

parameters:
  dataset:
    type: string
    default: "input/dataset.csv"
  analysis_types:
    type: array
    default: ["statistical", "sentiment", "trend"]
  output_format:
    type: string
    default: "pdf"

steps:
  - id: load_data
    tool: filesystem
    action: read
    parameters:
      path: "{{ output_path }}/{{ parameters.dataset }}"
    
  - id: data_preprocessing
    tool: pipeline-executor
    parameters:
      pipeline: |
        id: data_preprocessing_sub
        name: Data Preprocessing Sub-Pipeline
        steps:
          - id: clean_data
            tool: data-processing
            action: clean
            parameters:
              data: "{{ inputs.raw_data }}"
              remove_duplicates: true
              handle_missing: "forward_fill"
          
          - id: normalize_data
            tool: data-processing
            action: transform
            parameters:
              data: "{{ clean_data.result }}"
              operation:
                type: "normalize"
                method: "min-max"
        
        outputs:
          processed_data: "{{ normalize_data.result }}"
      inputs:
        raw_data: "{{ load_data.content }}"
      inherit_context: true
      wait_for_completion: true
    dependencies:
      - load_data
    
  - id: statistical_analysis
    tool: pipeline-executor
    parameters:
      pipeline: "examples/sub_pipelines/statistical_analysis.yaml"
      inputs:
        data: "{{ data_preprocessing.outputs.processed_data }}"
        confidence_level: 0.95
      output_mapping:
        statistics: "statistical_results"
        summary: "statistical_summary"
      inherit_context: true
    dependencies:
      - data_preprocessing
    condition: "'statistical' in {{ parameters.analysis_types }}"
    
  - id: sentiment_analysis
    tool: pipeline-executor
    parameters:
      pipeline: "examples/sub_pipelines/sentiment_analysis.yaml"
      inputs:
        data: "{{ data_preprocessing.outputs.processed_data }}"
        text_column: "comments"
      output_mapping:
        sentiment_scores: "sentiment_results"
        sentiment_summary: "sentiment_summary"
      inherit_context: true
    dependencies:
      - data_preprocessing
    condition: "'sentiment' in {{ parameters.analysis_types }}"
    
  - id: trend_analysis
    tool: pipeline-executor
    parameters:
      pipeline: "examples/sub_pipelines/trend_analysis.yaml"
      inputs:
        data: "{{ data_preprocessing.outputs.processed_data }}"
        time_column: "timestamp"
        value_columns: ["sales", "revenue"]
      output_mapping:
        trends: "trend_results"
        forecasts: "trend_forecasts"
      inherit_context: true
    dependencies:
      - data_preprocessing
    condition: "'trend' in {{ parameters.analysis_types }}"
    
  - id: combine_results
    tool: data-processing
    action: merge
    parameters:
      datasets:
        - name: "statistical"
          data: "{{ statistical_analysis.outputs.statistical_results | default({}) }}"
        - name: "sentiment"
          data: "{{ sentiment_analysis.outputs.sentiment_results | default({}) }}"
        - name: "trend"
          data: "{{ trend_analysis.outputs.trend_results | default({}) }}"
      merge_strategy: "combine"
    dependencies:
      - statistical_analysis
      - sentiment_analysis
      - trend_analysis
    
  - id: generate_visualizations
    tool: visualization
    action: create_charts
    parameters:
      data: "{{ data_preprocessing.outputs.processed_data }}"
      chart_types: ["auto"]
      output_dir: "{{ output_path }}/charts"
      title: "Analysis Results"
      theme: "seaborn"
    dependencies:
      - combine_results
    
  - id: create_dashboard
    tool: visualization
    action: create_dashboard
    parameters:
      charts: "{{ generate_visualizations.charts }}"
      layout: "grid"
      title: "Analysis Dashboard"
      output_dir: "{{ output_path }}"
    dependencies:
      - generate_visualizations
    
  - id: compile_report
    tool: report-generator
    action: generate
    parameters:
      title: "Comprehensive Analysis Report"
      sections:
        - title: "Executive Summary"
          content: <AUTO>Summarize key findings from all analyses</AUTO>
          
        - title: "Data Overview"
          content: |
            Dataset: {{ parameters.dataset }}
            Preprocessing steps applied: cleaning, normalization
            Analysis types performed: {{ parameters.analysis_types | join(', ') }}
            
        - title: "Statistical Analysis"
          content: "{{ statistical_analysis.outputs.statistical_summary | default('Not performed') }}"
          condition: "'statistical' in {{ parameters.analysis_types }}"
          
        - title: "Sentiment Analysis"
          content: "{{ sentiment_analysis.outputs.sentiment_summary | default('Not performed') }}"
          condition: "'sentiment' in {{ parameters.analysis_types }}"
          
        - title: "Trend Analysis"
          content: |
            ## Identified Trends
            {{ trend_analysis.outputs.trend_results | json }}
            
            ## Forecasts
            {{ trend_analysis.outputs.trend_forecasts | json }}
          condition: "'trend' in {{ parameters.analysis_types }}"
          
        - title: "Visualizations"
          content: |
            Dashboard available at: {{ create_dashboard.url }}
            Generated charts: {{ generate_visualizations.charts | length }} files
      
      include_visualizations: true
      visualization_files: "{{ generate_visualizations.charts }}"
    dependencies:
      - create_dashboard
    
  - id: export_report
    tool: pdf-compiler
    action: compile
    parameters:
      content: "{{ compile_report.report }}"
      output_path: "{{ output_path }}/analysis_report.{{ parameters.output_format }}"
      format: "{{ parameters.output_format }}"
      include_toc: true
      include_timestamp: true
    dependencies:
      - compile_report
    
  # Save pipeline results
  - id: save_results
    tool: filesystem
    action: write
    parameters:
      path: "{{ output_path }}/results_{{ execution.timestamp | replace(':', '-') }}.md"
      content: |
        # Modular Analysis Pipeline Results
        
        **Date:** {{ execution.timestamp }}
        **Pipeline ID:** modular_analysis
        
        ## Execution Summary
        
        Pipeline completed successfully.
        
        ### Analysis Types Performed
        {{ parameters.analysis_types | join(', ') }}
        
        ### Key Results
        - Statistical Analysis: {{ 'Completed' if 'statistical' in parameters.analysis_types else 'Skipped' }}
        - Sentiment Analysis: {{ 'Completed' if 'sentiment' in parameters.analysis_types else 'Skipped' }}
        - Trend Analysis: {{ 'Completed' if 'trend' in parameters.analysis_types else 'Skipped' }}
        
        ### Output Files
        - Report: {{ output_path }}/analysis_report.{{ parameters.output_format }}
        - Dashboard: {{ create_dashboard.url | default('Not generated') }}
        - Charts: {{ generate_visualizations.charts | length }} files generated
        
        ---
        *Generated by Modular Analysis Pipeline*
    dependencies:
      - export_report

outputs:
  report_path: "{{ output_path }}/analysis_report.{{ parameters.output_format }}"
  dashboard_url: "{{ create_dashboard.url }}"
  analysis_summary: "{{ compile_report.summary }}"
  charts_generated: "{{ generate_visualizations.charts }}"
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
- This pipeline produces Analysis results, CSV data, JSON data, Markdown documents, Reports - adjust output configuration accordingly

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
- 2. Run: python scripts/run_pipeline.py examples/modular_analysis_pipeline_fixed.yaml
- 3. Monitor the output for progress and any error messages
- 4. Check the output directory for generated results

### Expected Outputs
- Generated Analysis results in the specified output directory
- Generated CSV data in the specified output directory
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

*Tutorial generated on 2025-08-27T23:40:24.396437*
