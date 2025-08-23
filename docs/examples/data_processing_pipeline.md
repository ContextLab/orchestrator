# Comprehensive Data Processing Pipeline

**Pipeline**: `examples/data_processing_pipeline.yaml`  
**Category**: Data Processing & Analytics  
**Complexity**: Advanced  
**Key Features**: Data profiling, Schema validation, Complex transformations, Statistical analysis, Quality assessment, Professional reporting

## Overview

The Comprehensive Data Processing Pipeline provides enterprise-grade data processing capabilities including profiling, validation, cleaning, transformation, aggregation, and statistical analysis. It demonstrates a complete end-to-end data processing workflow with quality assessment and professional reporting suitable for business intelligence and data science applications.

## Key Features Demonstrated

### 1. Advanced Data Profiling
```yaml
- id: profile_data
  tool: data-processing
  action: profile
  condition: "{{ enable_profiling }}"
  parameters:
    data: "{{ read_data.content }}"
    format: "csv"
    profiling_options:
      - missing_values
      - data_types
      - statistical_summary
      - outlier_detection
      - duplicate_detection
```

### 2. Comprehensive Schema Validation
```yaml
schema:
  type: object
  properties:
    order_id:
      type: string
      pattern: "^ORD-[0-9]{6}$"
    quantity:
      type: integer
      minimum: 1
    status:
      type: string
      enum: ["pending", "processing", "shipped", "delivered", "cancelled"]
  required: ["order_id", "customer_id", "product_name", "quantity", "unit_price"]
mode: "LENIENT"  # Fix minor issues automatically
```

### 3. Multi-Stage Data Transformations
```yaml
operations:
  # Data cleaning
  - type: deduplicate
    columns: ["order_id"]
    keep: "first"
  
  # Type casting
  - type: cast
    columns:
      quantity: "integer"
      unit_price: "float"
      order_date: "datetime"
  
  # Missing value handling
  - type: fill_missing
    strategy:
      status: "pending"
      quantity: 1
  
  # Calculated fields
  - type: calculate
    expressions:
      total_amount: "quantity * unit_price"
      order_month: "DATE_FORMAT(order_date, '%Y-%m')"
```

### 4. Advanced Aggregation Operations
```yaml
- id: aggregate_monthly
  tool: data-processing
  action: aggregate
  parameters:
    group_by: ["order_month", "product_name"]
    aggregations:
      total_quantity: {column: "quantity", function: "sum"}
      total_revenue: {column: "total_amount", function: "sum"}
      average_price: {column: "unit_price", function: "mean"}
      order_count: {column: "order_id", function: "count"}
      unique_customers: {column: "customer_id", function: "count_distinct"}
```

### 5. Statistical Trend Analysis
```yaml
- id: analyze_trends
  action: analyze_text
  parameters:
    analysis_type: "statistical_trends"
    prompt: |
      Calculate growth_rate, identify top_products, detect seasonal_patterns,
      and identify anomalies from the aggregated data.
      
      Return ONLY a valid JSON object with:
      - growth_rate: month-over-month growth percentage
      - top_products: sorted by revenue
      - seasonal_patterns: time-based patterns
      - anomalies: unusual data points
```

### 6. Pivot Table Analysis
```yaml
- id: pivot_analysis
  tool: data-processing
  action: pivot
  parameters:
    index: ["product_name"]
    columns: ["status"]
    values: ["quantity"]
    aggfunc: "sum"
    fill_value: 0
```

## Pipeline Architecture

### Input Parameters
- **input_file**: CSV filename to process (default: "sales_data.csv")
- **output_path**: Output directory (default: "examples/outputs/data_processing_pipeline")
- **quality_threshold**: Minimum acceptable quality score (default: 0.95)
- **enable_profiling**: Enable data profiling analysis (default: true)

### Processing Flow

1. **Data Reading** - Loads CSV data from specified input file
2. **Data Profiling** - Comprehensive data quality and statistical profiling
3. **Schema Validation** - Validates data structure against predefined schema
4. **Data Cleaning & Transformation** - Multi-stage cleaning and enhancement
5. **Monthly Aggregation** - Groups data by time periods with calculations
6. **Statistical Analysis** - Advanced trend analysis and pattern detection
7. **Pivot Analysis** - Cross-tabulation for product status distribution
8. **Quality Assessment** - Calculates overall data quality score
9. **Data Export** - Converts processed data to CSV format
10. **Comprehensive Reporting** - Generates detailed processing report

### Advanced Data Operations

#### Data Profiling Capabilities
- **Missing Value Analysis**: Identifies and quantifies missing data
- **Data Type Detection**: Automatic type inference and validation
- **Statistical Summaries**: Min, max, mean, median, standard deviation
- **Outlier Detection**: Statistical outlier identification with percentages
- **Duplicate Detection**: Identifies and counts duplicate records

#### Transformation Operations
- **Deduplication**: Removes duplicate records based on key columns
- **Type Casting**: Converts data types for consistency
- **Missing Value Imputation**: Strategic missing value handling
- **Calculated Fields**: Dynamic column creation with expressions
- **Data Filtering**: Conditional row filtering

## Usage Examples

### Basic Sales Data Processing
```bash
python scripts/run_pipeline.py examples/data_processing_pipeline.yaml
```

### Custom Input File
```bash
python scripts/run_pipeline.py examples/data_processing_pipeline.yaml \
  --input input_file="custom_sales.csv"
```

### Disable Profiling for Speed
```bash
python scripts/run_pipeline.py examples/data_processing_pipeline.yaml \
  --input enable_profiling=false \
  --input quality_threshold=0.8
```

### Custom Output Location
```bash
python scripts/run_pipeline.py examples/data_processing_pipeline.yaml \
  --input output_path="/custom/output/path" \
  --input input_file="large_dataset.csv"
```

## Sample Output Structure

### Data Processing Report
The pipeline generates a comprehensive markdown report including:

#### Processing Summary
- Input file details and row counts
- Data profiling statistics
- Processing completion metrics

#### Data Validation Results
- Schema validation status
- Error and warning details
- Compliance assessment

#### Data Quality Assessment
```markdown
### Quality Score: 0.95/1.0

### Issues Found
- ⚠️ High outlier rate: 'quantity' column has 20.0% outliers
```

#### Column Statistics Table
| Column | Type | Missing % | Unique Values | Min | Max | Mean |
|--------|------|-----------|---------------|-----|-----|------|
| quantity | numeric | 0.0% | 5 | 1.0 | 10.0 | 4.2 |
| unit_price | numeric | 0.0% | 4 | 19.99 | 49.99 | 31.99 |

#### Monthly Aggregations
| Month | Total Quantity | Total Revenue | Avg Price | Order Count | Unique Customers |
|-------|----------------|---------------|-----------|-------------|------------------|
| 2024-01 | 15.0 | $299.85 | $19.99 | 2 | 2 |

#### Product Status Distribution (Pivot)
| Product | Pending | Processing | Shipped | Delivered | Total |
|---------|---------|------------|---------|-----------|-------|
| Widget A | 0 | 0 | 0 | 15.0 | 15.0 |

#### Statistical Analysis
- Growth rate calculations
- Top products by revenue
- Seasonal pattern detection
- Anomaly identification

### Generated Files
- **Processed Data**: `processed_data.csv` - Clean, transformed dataset
- **Processing Report**: `data_processing_report.md` - Comprehensive analysis
- **Validation Report**: `validation_report.json` - Schema validation details

Check actual outputs: [data_processing_report.md](../../examples/outputs/data_processing_pipeline/data_processing_report.md)

## Technical Implementation

### Quality Score Calculation
```yaml
prompt: |
  Score calculation:
  - Start with 1.0 (perfect score)
  - Deduct 0.1 for duplicate rows
  - Deduct 0.05 for each column with >10% missing data
  - Deduct 0.1 if validation failed
  - Deduct 0.05 for outliers if >10%
  
  Return as JSON with:
  - quality_score: calculated score between 0 and 1
  - issues_found: specific issues in data
  - recommendations: improvement suggestions
```

### Advanced Template Processing
```yaml
# Complex conditional formatting
{% if validate_schema.valid %}
✅ **Validation Passed**: All data conforms to schema requirements
{% else %}
❌ **Validation Failed**: {{ validate_schema.error }}
{% endif %}

# Dynamic table generation
{% for row in aggregate_monthly.processed_data %}
| {{ row.order_month }} | {{ row.total_quantity }} | ${{ row.total_revenue | round(2) }} |
{% endfor %}
```

### Multi-Tool Integration
```yaml
# Filesystem operations
tool: filesystem
action: read

# Data processing operations
tool: data-processing
action: transform

# Validation operations
tool: validation
action: validate
```

### Professional JSON Formatting
```yaml
content: |
  {
    "validation_success": {{ validate_schema.valid | to_json }},
    "errors": {{ validate_schema.errors | to_json }},
    "timestamp": "{{ now() }}",
    "input_file": "{{ input_file }}"
  }
```

## Advanced Features

### Conditional Processing
```yaml
condition: "{{ enable_profiling }}"  # Optional profiling step
```

### Lenient Validation Mode
```yaml
mode: "LENIENT"  # Attempts to fix minor schema issues automatically
```

### Dynamic Data Flow
```yaml
data: "{{ validate_schema.data if validate_schema.valid else read_data.content }}"
# Uses validated data if available, falls back to original
```

### Comprehensive Error Handling
- Schema validation with detailed error reporting
- Missing value strategies for data consistency
- Outlier detection with configurable thresholds
- Quality score calculation with issue tracking

### Business Intelligence Features
- Month-over-month growth calculations
- Product performance rankings
- Seasonal pattern detection
- Anomaly identification and reporting

## Common Use Cases

- **Sales Analytics**: Monthly sales reporting and trend analysis
- **Data Quality Auditing**: Comprehensive data quality assessment
- **ETL Operations**: Extract, transform, and load workflows
- **Business Intelligence**: Executive dashboards and KPI reporting
- **Data Migration**: Quality assessment during system migrations
- **Regulatory Compliance**: Data validation for compliance requirements
- **Performance Monitoring**: Data pipeline health monitoring

## Best Practices Demonstrated

1. **Comprehensive Profiling**: Detailed data understanding before processing
2. **Schema-First Validation**: Strict data quality requirements
3. **Multi-Stage Transformations**: Logical processing sequence
4. **Quality Assessment**: Quantitative quality measurement
5. **Professional Reporting**: Executive-ready documentation
6. **Error Recovery**: Graceful handling of data issues
7. **Audit Trail**: Complete processing documentation

## Troubleshooting

### Common Issues
- **Schema Validation Failures**: Check data format matches expected schema
- **Performance Issues**: Large datasets may require chunking
- **Memory Usage**: Complex aggregations may require optimization
- **File Path Errors**: Ensure input files exist and are accessible

### Performance Optimization
- **Disable Profiling**: Set `enable_profiling=false` for faster processing
- **Adjust Quality Thresholds**: Lower thresholds for faster validation
- **Chunked Processing**: Split large files for memory efficiency
- **Selective Operations**: Skip unnecessary transformation steps

### Data Quality Tips
- **Use Schema Validation**: Always validate input data structure
- **Monitor Quality Scores**: Track quality degradation over time
- **Review Profiling Reports**: Understand data characteristics
- **Handle Missing Values**: Implement appropriate imputation strategies

## Related Examples
- [data_processing.md](data_processing.md) - Basic data processing workflows
- [statistical_analysis.md](statistical_analysis.md) - Statistical analysis focus
- [validation_pipeline.md](validation_pipeline.md) - Data validation patterns
- [simple_data_processing.md](simple_data_processing.md) - Simple processing examples

## Technical Requirements

- **Tools**: filesystem, data-processing, validation tools
- **Models**: Support for statistical analysis and text generation
- **Memory**: Adequate for dataset size and aggregation operations
- **Storage**: Write access for output files and reports

This pipeline provides production-ready data processing capabilities suitable for enterprise data workflows, business intelligence applications, and data science projects requiring comprehensive data quality assessment and professional reporting.