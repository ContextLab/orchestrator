# Statistical Analysis Sub-Pipeline

**Pipeline**: `examples/statistical_analysis.yaml`  
**Category**: Data Analysis  
**Complexity**: Advanced  
**Key Features**: Statistical computation, Distribution analysis, AUTO model selection, Comprehensive reporting

## Overview

The Statistical Analysis Sub-Pipeline provides comprehensive statistical analysis capabilities designed for reuse within larger analytical workflows. It performs descriptive statistics, distribution analysis, and generates actionable insights with configurable confidence levels, making it ideal for data science and research applications.

## Key Features Demonstrated

### 1. Structured Statistical Analysis
```yaml
- id: descriptive_stats
  action: analyze_text
  parameters:
    text: |
      Analyze this dataset and provide descriptive statistics:
      Include: mean, median, standard deviation, min, max, quartiles
```

### 2. AUTO Model Selection with Task Context
```yaml
model: <AUTO task="analyze">Select model for statistical analysis</AUTO>
model: <AUTO task="generate">Select model for insight generation</AUTO>
```

### 3. Progressive Analysis Chain
```yaml
dependencies:
  - prepare_data          # Data preparation first
  - descriptive_stats     # Basic statistics next
  - distribution_analysis # Distribution analysis follows
  - generate_insights     # Insights generated last
```

### 4. Comprehensive Report Generation
```yaml
content: |
  # Statistical Analysis Report
  
  ## Descriptive Statistics
  {{ descriptive_stats.result }}
  
  ## Distribution Analysis
  {{ distribution_analysis.result }}
  
  ## Key Insights
  {{ generate_insights.result }}
```

## Pipeline Architecture

### Input Parameters
- **data** (required): Data to analyze in JSON format
- **confidence_level** (optional): Statistical confidence level (default: 0.95)

### Processing Flow

1. **Prepare Data** - Store input data for analysis
2. **Descriptive Statistics** - Calculate basic statistical measures
3. **Distribution Analysis** - Analyze data distribution characteristics
4. **Generate Insights** - Create actionable insights and recommendations
5. **Save Analysis** - Generate comprehensive statistical report

### Statistical Analysis Components

#### Descriptive Statistics
- **Central Tendency**: Mean, median, mode
- **Variability**: Standard deviation, variance, range
- **Position**: Quartiles, percentiles, min/max values
- **Shape**: Preliminary distribution characteristics

#### Distribution Analysis
- **Normality Tests**: Check for normal distribution
- **Skewness**: Measure of asymmetry
- **Kurtosis**: Measure of tail heaviness
- **Distribution Type**: Identify distribution patterns

#### Insight Generation
- **Pattern Recognition**: Identify significant patterns
- **Anomaly Detection**: Highlight unusual values
- **Recommendations**: Actionable business insights
- **Confidence Assessment**: Statistical significance evaluation

## Usage Examples

### Basic Statistical Analysis
```bash
python scripts/run_pipeline.py examples/statistical_analysis.yaml \
  -i data='[{"value": 10}, {"value": 20}, {"value": 15}, {"value": 25}]'
```

### High Confidence Analysis
```bash
python scripts/run_pipeline.py examples/statistical_analysis.yaml \
  -i data='[{"sales": 1000, "region": "north"}, {"sales": 1200, "region": "south"}]' \
  -i confidence_level=0.99
```

### Complex Dataset Analysis
```bash
python scripts/run_pipeline.py examples/statistical_analysis.yaml \
  -i data='[
    {"revenue": 50000, "expenses": 30000, "profit": 20000, "quarter": "Q1"},
    {"revenue": 55000, "expenses": 32000, "profit": 23000, "quarter": "Q2"},
    {"revenue": 48000, "expenses": 29000, "profit": 19000, "quarter": "Q3"}
  ]'
```

### Sub-Pipeline Integration
```yaml
# Used within larger pipeline
- id: run_statistical_analysis
  tool: pipeline-executor
  parameters:
    pipeline: "examples/statistical_analysis.yaml"
    inputs:
      data: "{{ processed_data }}"
      confidence_level: 0.95
```

## Statistical Methods Applied

### Descriptive Statistics Calculation
```yaml
analysis_includes:
  - mean: "Average value across dataset"
  - median: "Middle value when data is ordered"
  - standard_deviation: "Measure of data spread"
  - quartiles: "25th, 50th, 75th percentiles"
  - range: "Difference between max and min"
  - count: "Number of data points"
```

### Distribution Characteristics
```yaml
distribution_tests:
  - normality: "Shapiro-Wilk or similar test"
  - skewness: "Measure of asymmetry (-∞ to +∞)"
  - kurtosis: "Measure of tail heaviness"
  - outliers: "Values beyond 1.5 * IQR from quartiles"
```

### Confidence Interval Calculation
```yaml
confidence_intervals:
  - mean_ci: "Confidence interval for population mean"
  - proportion_ci: "Confidence interval for proportions"
  - difference_ci: "Confidence interval for differences"
```

## Sample Analysis Output

### Descriptive Statistics Report
```markdown
## Descriptive Statistics

### Summary Statistics
- **Count**: 100 observations
- **Mean**: 45.67 ± 2.34 (95% CI)
- **Median**: 44.50
- **Standard Deviation**: 12.89
- **Variance**: 166.15
- **Range**: 18.2 to 78.9

### Quartile Analysis
- **Q1 (25th percentile)**: 37.25
- **Q2 (50th percentile)**: 44.50
- **Q3 (75th percentile)**: 52.75
- **Interquartile Range**: 15.50
```

### Distribution Analysis Report
```markdown
## Distribution Analysis

### Normality Assessment
- **Shapiro-Wilk p-value**: 0.23 (not significant at α=0.05)
- **Distribution**: Approximately normal
- **Skewness**: 0.15 (slightly right-skewed)
- **Kurtosis**: -0.42 (slightly platykurtic)

### Outlier Detection
- **Method**: 1.5 × IQR rule
- **Outliers Detected**: 3 values
- **Outlier Values**: [78.9, 19.2, 76.8]
```

### Key Insights Report
```markdown
## Key Insights

### Primary Findings
1. **Data Distribution**: The dataset follows an approximately normal distribution with slight right skew, indicating most values cluster around the mean with some higher outliers.

2. **Central Tendency**: The mean (45.67) is slightly higher than the median (44.50), confirming the right skew observed in the distribution.

3. **Variability**: Standard deviation of 12.89 suggests moderate variability relative to the mean, indicating reasonably consistent data.

### Recommendations
1. **Outlier Investigation**: Examine the 3 detected outliers to determine if they represent genuine extreme values or data entry errors.

2. **Statistical Tests**: Given near-normal distribution, parametric tests are appropriate for hypothesis testing.

3. **Sample Size**: Current sample size (n=100) provides adequate power for most statistical analyses at 95% confidence level.
```

## AUTO Model Selection Strategy

### Task-Specific Model Selection
```yaml
# Analysis tasks use models optimized for data processing
model: <AUTO task="analyze">Select model for statistical analysis</AUTO>
# Likely selects: Models with strong analytical capabilities

# Generation tasks use models optimized for content creation  
model: <AUTO task="generate">Select model for insight generation</AUTO>
# Likely selects: Models with strong reasoning and explanation abilities
```

### Analysis Type Optimization
```yaml
analysis_type: "statistical"    # Hints for statistical analysis optimization
analysis_type: "distribution"   # Hints for distribution analysis optimization
```

## Integration Patterns

### As Sub-Pipeline
```yaml
# Called from main pipeline
- id: statistical_analysis
  tool: pipeline-executor
  parameters:
    pipeline: "examples/statistical_analysis.yaml"
    inputs:
      data: "{{ cleaned_data }}"
      confidence_level: 0.95
    output_mapping:
      statistics: "stat_results"
      insights: "stat_insights"
```

### Data Flow Integration
```yaml
# Input from data processing
data: "{{ data_preprocessing.normalized_data }}"

# Output to reporting
report_content: "{{ statistical_analysis.insights }}"
```

### Batch Analysis
```yaml
# Analyze multiple datasets
for_each: "{{ datasets }}"
steps:
  - tool: pipeline-executor
    parameters:
      pipeline: "examples/statistical_analysis.yaml"
      inputs:
        data: "{{ item }}"
```

## Advanced Statistical Features

### Confidence Level Impact
```yaml
confidence_level: 0.90  # 90% confidence (wider intervals, less strict)
confidence_level: 0.95  # 95% confidence (standard)
confidence_level: 0.99  # 99% confidence (narrower intervals, more strict)
```

### Multi-Variable Analysis
```yaml
# Handles complex data structures
data: [
  {"x": 10, "y": 20, "category": "A"},
  {"x": 15, "y": 25, "category": "B"}
]
# Analyzes relationships between variables
```

### Time Series Support
```yaml
# Temporal data analysis
data: [
  {"value": 100, "timestamp": "2024-01-01"},
  {"value": 105, "timestamp": "2024-01-02"}
]
# Provides trend and seasonality analysis
```

## Best Practices Demonstrated

1. **Progressive Analysis**: Build complexity through sequential steps
2. **Data Preservation**: Store input data for reproducibility
3. **Comprehensive Reporting**: Include methodology and confidence levels
4. **Model Optimization**: Use task-specific AUTO model selection
5. **Error Handling**: Graceful handling of edge cases
6. **Standardized Output**: Consistent report format and structure

## Common Use Cases

- **Quality Assurance**: Statistical validation of data quality
- **Business Intelligence**: Performance metrics analysis
- **Research Support**: Academic and scientific data analysis
- **A/B Testing**: Statistical significance testing
- **Process Control**: Manufacturing and operational statistics
- **Financial Analysis**: Risk assessment and performance metrics

## Troubleshooting

### Data Format Issues
- Ensure data is in valid JSON format
- Check for missing values and handle appropriately
- Validate data types match expected formats

### Analysis Accuracy
- Verify confidence level is appropriate for use case
- Check sample size adequacy for statistical power
- Ensure data meets assumptions for chosen tests

### Report Generation Problems
- Validate template syntax for complex data structures
- Check file system permissions for output generation
- Ensure timestamp formatting compatibility

## Related Examples
- [modular_analysis_pipeline.md](modular_analysis_pipeline.md) - Integration within modular workflows
- [data_processing_pipeline.md](data_processing_pipeline.md) - Data preparation for analysis
- [validation_pipeline.md](validation_pipeline.md) - Data validation patterns

## Technical Requirements

- **Statistical Computing**: AI-powered statistical analysis capabilities
- **Data Processing**: JSON data structure handling
- **File System**: Read/write access for data and reports
- **Template Engine**: Complex template processing for reports
- **Model Access**: Multiple AI models for different analysis tasks

This sub-pipeline provides enterprise-grade statistical analysis capabilities that can be integrated into larger analytical workflows while maintaining consistency and reliability across different data analysis scenarios.