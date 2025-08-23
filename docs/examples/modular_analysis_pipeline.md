# Modular Analysis Pipeline

**Pipeline**: `examples/modular_analysis_pipeline.yaml`  
**Category**: Data Analysis  
**Complexity**: Expert  
**Key Features**: Sub-pipeline orchestration, Modular design, Data visualization, Multi-analysis integration, Conditional execution

## Overview

The Modular Analysis Pipeline demonstrates advanced pipeline orchestration by coordinating multiple specialized sub-pipelines for comprehensive data analysis. It showcases modular design principles, conditional analysis execution, data visualization generation, and comprehensive reporting capabilities.

## Key Features Demonstrated

### 1. Sub-Pipeline Orchestration
```yaml
- id: statistical_analysis
  tool: pipeline-executor
  parameters:
    pipeline: "examples/sub_pipelines/statistical_analysis.yaml"
    inputs:
      data: "{{ load_data.content }}"
      confidence_level: 0.95
```

### 2. Inline Sub-Pipeline Definition
```yaml
pipeline: |
  id: data_preprocessing_sub
  name: Data Preprocessing Sub-Pipeline
  steps:
    - id: clean_data
      tool: data-processing
    - id: normalize_data
      tool: data-processing
```

### 3. Conditional Analysis Execution
```yaml
condition: "'statistical' in {{ parameters.analysis_types }}"
condition: "'sentiment' in {{ parameters.analysis_types }}"  
condition: "'trend' in {{ parameters.analysis_types }}"
```

### 4. Comprehensive Visualization Generation
```yaml
- id: generate_visualizations
  tool: visualization
  parameters:
    chart_types: ["bar", "line", "scatter", "pie", "histogram"]
    output_dir: "{{ output_path }}/charts"
```

## Pipeline Architecture

### Input Parameters
- **dataset** (optional): Path to input data file (default: "input/dataset.csv")
- **analysis_types** (optional): Array of analyses to perform (default: ["statistical", "sentiment", "trend"])
- **output_format** (optional): Report output format (default: "pdf")

### Processing Flow

1. **Load Data** - Read input dataset from filesystem
2. **Data Preprocessing** - Execute inline preprocessing sub-pipeline
3. **Statistical Analysis** - Run statistical analysis sub-pipeline (conditional)
4. **Sentiment Analysis** - Execute sentiment analysis sub-pipeline (conditional)
5. **Trend Analysis** - Perform trend analysis sub-pipeline (conditional)
6. **Combine Results** - Merge all analysis results
7. **Generate Visualizations** - Create comprehensive charts
8. **Create Dashboard** - Build interactive dashboard
9. **Compile Report** - Generate comprehensive markdown report
10. **Save Results** - Store execution summary and results

### Sub-Pipeline Architecture

#### Data Preprocessing Sub-Pipeline (Inline)
```yaml
steps:
  - id: clean_data
    tool: data-processing
    action: clean
    parameters:
      remove_duplicates: true
      handle_missing: "forward_fill"
      
  - id: normalize_data
    tool: data-processing
    action: transform
    parameters:
      operation:
        type: "normalize"
        method: "min-max"
```

#### External Sub-Pipelines
- **Statistical Analysis**: `examples/sub_pipelines/statistical_analysis.yaml`
- **Sentiment Analysis**: `examples/sub_pipelines/sentiment_analysis.yaml`
- **Trend Analysis**: `examples/sub_pipelines/trend_analysis.yaml`

## Usage Examples

### Full Analysis Suite
```bash
python scripts/run_pipeline.py examples/modular_analysis_pipeline.yaml \
  -i dataset="examples/data/sales_data.csv" \
  -i analysis_types='["statistical", "sentiment", "trend"]'
```

### Statistical Analysis Only
```bash
python scripts/run_pipeline.py examples/modular_analysis_pipeline.yaml \
  -i dataset="examples/data/customer_data.csv" \
  -i analysis_types='["statistical"]'
```

### Sentiment and Trend Analysis
```bash
python scripts/run_pipeline.py examples/modular_analysis_pipeline.yaml \
  -i dataset="examples/data/feedback_data.csv" \
  -i analysis_types='["sentiment", "trend"]'
```

### Custom Dataset Analysis
```bash
python scripts/run_pipeline.py examples/modular_analysis_pipeline.yaml \
  -i dataset="custom_data/market_research.csv" \
  -i output_format="html"
```

## Sub-Pipeline Integration Patterns

### External Sub-Pipeline Execution
```yaml
tool: pipeline-executor
parameters:
  pipeline: "examples/sub_pipelines/statistical_analysis.yaml"
  inputs:
    data: "{{ load_data.content }}"
    confidence_level: 0.95
  output_mapping:
    statistics: "statistical_results"
    summary: "statistical_summary"
```

### Inline Sub-Pipeline Definition
```yaml
pipeline: |
  id: data_preprocessing_sub
  name: Data Preprocessing Sub-Pipeline
  steps:
    - id: clean_data
      # Sub-pipeline steps defined inline
```

### Sub-Pipeline Configuration Options
```yaml
parameters:
  inherit_context: true          # Share parent pipeline context
  wait_for_completion: true      # Block until sub-pipeline completes
  timeout: 60000                 # 60 second timeout
  error_handling: "fail"         # Fail parent on sub-pipeline error
  retry_count: 3                 # Retry failed sub-pipeline 3 times
  retry_delay: 1                 # 1 second delay between retries
```

## Conditional Analysis Execution

### Analysis Type Selection
```yaml
# Statistical analysis conditional
condition: "'statistical' in {{ parameters.analysis_types }}"

# Sentiment analysis conditional  
condition: "'sentiment' in {{ parameters.analysis_types }}"

# Trend analysis conditional
condition: "'trend' in {{ parameters.analysis_types }}"
```

### Dynamic Analysis Configuration
```yaml
analysis_types:
  statistical:
    confidence_level: 0.95
    tests: ["t_test", "chi_square", "anova"]
  sentiment:
    text_column: "comments"
    model: "vader"
  trend:
    time_column: "timestamp"
    value_columns: ["sales", "revenue"]
    forecast_periods: 12
```

## Data Visualization Pipeline

### Chart Generation
```yaml
- id: generate_visualizations
  tool: visualization
  parameters:
    chart_types: ["bar", "line", "scatter", "pie", "histogram"]
    output_dir: "{{ output_path }}/charts"
    title: "Analysis Results"
    theme: "seaborn"
```

### Dashboard Creation
```yaml
- id: create_dashboard
  tool: visualization
  parameters:
    charts: "{{ generate_visualizations.charts }}"
    layout: "grid"
    title: "Analysis Dashboard"
    output_dir: "{{ output_path }}"
```

### Visualization Types Generated
- **Bar Charts**: Category comparisons and distributions
- **Line Charts**: Time series trends and patterns
- **Scatter Plots**: Correlation and relationship analysis
- **Pie Charts**: Proportional data representation
- **Histograms**: Distribution analysis and frequency plots

## Sample Output Structure

### Generated Files
```
examples/outputs/modular_analysis/
├── input/
│   └── dataset.csv                    # Input data
├── charts/
│   ├── bar_chart_20240823_121516.png
│   ├── line_chart_20240823_121516.png
│   ├── scatter_chart_20240823_121516.png
│   ├── pie_chart_20240823_121516.png
│   └── histogram_chart_20240823_121516.png
├── dashboard_20240823_121516.html     # Interactive dashboard
├── analysis_report.md                 # Comprehensive report
└── results_2024-08-23T12-15-16.md    # Execution summary
```

### Comprehensive Analysis Report
```markdown
# Comprehensive Analysis Report

## Executive Summary
This report presents the results of comprehensive data analysis including 
statistical, sentiment, and trend analyses.

## Data Overview
- Dataset: input/dataset.csv
- Preprocessing steps applied: cleaning, normalization
- Analysis types performed: statistical, sentiment, trend

## Statistical Analysis
Statistical analysis revealed significant patterns in the data with 
95% confidence level. Key findings include...

## Sentiment Analysis
Sentiment analysis of comment data shows overall positive sentiment 
with 73% positive, 18% neutral, and 9% negative responses.

## Trend Analysis
### Identified Trends
- Sales showing upward trend with 12% month-over-month growth
- Revenue patterns correlate strongly with seasonal factors

### Forecasts
- Projected sales increase of 15% over next quarter
- Revenue forecast indicates continued growth trajectory

## Visualizations
- Dashboard available at: dashboard_20240823_121516.html
- Generated charts: 5 files
```

## Advanced Features

### Result Combination and Merging
```yaml
- id: combine_results
  tool: data-processing
  parameters:
    datasets:
      - name: "statistical"
        data: "{{ statistical_analysis.outputs.statistical_results | default({}) }}"
      - name: "sentiment"  
        data: "{{ sentiment_analysis.outputs.sentiment_results | default({}) }}"
      - name: "trend"
        data: "{{ trend_analysis.outputs.trend_results | default({}) }}"
    merge_strategy: "combine"
```

### Error Handling and Resilience
```yaml
parameters:
  error_handling: "fail"         # Strict error handling
  retry_count: 3                 # Automatic retry on failure
  retry_delay: 1                 # Delay between retries
  timeout: 60000                 # Sub-pipeline timeout
```

### Context Inheritance
```yaml
inherit_context: true            # Share parent pipeline variables
wait_for_completion: true        # Synchronous execution
```

## Technical Implementation

### Sub-Pipeline Execution
```yaml
tool: pipeline-executor
action: execute
```

### Output Mapping
```yaml
output_mapping:
  statistics: "statistical_results"      # Map sub-pipeline output
  summary: "statistical_summary"         # to parent pipeline variables
```

### Dependency Management
```yaml
dependencies:
  - statistical_analysis               # Wait for statistical analysis
  - sentiment_analysis                # Wait for sentiment analysis
  - trend_analysis                    # Wait for trend analysis
```

## Best Practices Demonstrated

1. **Modular Design**: Separate specialized analyses into sub-pipelines
2. **Conditional Execution**: Run only requested analysis types
3. **Error Resilience**: Retry failed sub-pipelines with delays
4. **Result Integration**: Combine outputs from multiple analyses
5. **Comprehensive Reporting**: Generate detailed analysis reports
6. **Visualization Integration**: Create charts and dashboards
7. **Context Management**: Share data efficiently between pipelines

## Common Use Cases

- **Business Intelligence**: Comprehensive data analysis for decision-making
- **Research Analysis**: Multi-faceted analysis of research data
- **Market Research**: Customer sentiment, trends, and statistical analysis
- **Performance Analytics**: Multi-dimensional performance evaluation
- **Quality Assurance**: Statistical validation with visualization
- **Data Science Workflows**: End-to-end analysis pipeline orchestration

## Troubleshooting

### Sub-Pipeline Failures
- Check sub-pipeline file paths and accessibility
- Verify input data format compatibility
- Review sub-pipeline logs for specific errors

### Analysis Execution Issues
- Validate analysis_types parameter format
- Ensure required data columns exist for each analysis
- Check conditional logic for analysis selection

### Visualization Problems
- Verify visualization tool availability
- Check output directory write permissions
- Ensure data format compatibility for charting

## Related Examples
- [modular_analysis_pipeline_backup.md](modular_analysis_pipeline_backup.md) - Backup version
- [modular_analysis_pipeline_fixed.md](modular_analysis_pipeline_fixed.md) - Fixed version  
- [data_processing_pipeline.md](data_processing_pipeline.md) - Data processing patterns
- [statistical_analysis.md](statistical_analysis.md) - Statistical analysis sub-pipeline

## Technical Requirements

- **Sub-Pipeline Support**: Pipeline executor tool for orchestration
- **Data Processing Tools**: Cleaning, normalization, and analysis capabilities
- **Visualization Engine**: Chart generation and dashboard creation
- **File System**: Read/write access for data and reports
- **Analysis Libraries**: Statistical, sentiment, and trend analysis tools

This pipeline demonstrates enterprise-grade modular analysis architecture suitable for complex data science workflows requiring coordinated execution of specialized analysis components.