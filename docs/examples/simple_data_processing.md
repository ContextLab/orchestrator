# Simple Data Processing Pipeline

**Pipeline**: `examples/simple_data_processing.yaml`  
**Category**: Data Processing  
**Complexity**: Beginner  
**Key Features**: CSV processing, File I/O, Data filtering, Report generation

## Overview

The Simple Data Processing Pipeline demonstrates basic data processing workflows using the filesystem and data-processing tools. It reads CSV data, applies filters, and generates both processed data files and human-readable reports.

## Key Features Demonstrated

### 1. File System Operations
```yaml
- id: read_data
  tool: filesystem
  action: read
  parameters:
    path: "data/input.csv"
```

### 2. Data Processing with Filters
```yaml
- id: process_data
  tool: data-processing
  action: filter
  parameters:
    data: "{{ read_data.content }}"
    format: "csv"
    operation:
      criteria:
        status: "active"
```

### 3. Dynamic File Naming
```yaml
path: "{{ output_path }}/output_{{ execution.timestamp | slugify }}.csv"
```

### 4. Template-Based Report Generation
```yaml
content: |
  # Simple Data Processing Report
  
  **Generated:** {{ execution.timestamp }}
  **Pipeline:** Simple Data Processing
  
  ## Processing Summary
  
  - **Input File:** data/input.csv
  - **Filter Applied:** status = "active"
  - **Output File:** {{ output_path }}/output_{{ execution.timestamp | slugify }}.csv
```

## Pipeline Structure

### Input Parameters
- **output_path** (string): Directory where output files will be saved (default: "examples/outputs/simple_data_processing")

### Processing Steps

1. **Read Data** - Loads CSV file from the data directory
2. **Process Data** - Filters records to include only those with status="active"
3. **Save Results** - Writes filtered CSV to output directory with timestamp
4. **Save Report** - Generates markdown report with processing summary

### Dependencies
- Linear dependency chain: read_data → process_data → save_results & save_report

## Usage Examples

### Basic Usage
```bash
python scripts/run_pipeline.py examples/simple_data_processing.yaml
```

### Custom Output Directory
```bash
python scripts/run_pipeline.py examples/simple_data_processing.yaml \
  -i output_path="my_analysis_results"
```

### Using Different Data File
To use a different input file, modify the pipeline YAML:
```yaml
parameters:
  path: "data/input.csv"
```

## Sample Data and Outputs

### Input Data
The pipeline processes `examples/data/input.csv` which contains:
```csv
name,status,value,created_date
Project Alpha,active,1500,2024-01-15
Project Beta,inactive,2300,2024-02-08
Project Gamma,active,3100,2024-03-05
Project Delta,inactive,1800,2024-03-20
Project Epsilon,active,2700,2024-04-12
Project Zeta,inactive,3500,2024-04-28
Project Eta,active,4200,2024-05-15
Project Theta,inactive,2100,2024-05-30
Project Iota,active,2900,2024-06-22
Project Kappa,inactive,3800,2024-07-10
```

### Filtered Output
The processed data contains only active projects:
- [View filtered_output.csv](../../examples/outputs/simple_data_processing/filtered_output.csv)

```csv
name,status,value,created_date
Project Alpha,active,1500,2024-01-15
Project Gamma,active,3100,2024-03-05
Project Epsilon,active,2700,2024-04-12
Project Eta,active,4200,2024-05-15
Project Iota,active,2900,2024-06-22
```

### Generated Report
- [View analysis_report.md](../../examples/outputs/simple_data_processing/analysis_report.md)

The report includes:
- Processing timestamp and metadata
- Input file information
- Applied filter criteria
- Output file location
- Data preview with first 500 characters

## Advanced Features

### Template Variables
The pipeline uses several template features:

#### Execution Context
```yaml
{{ execution.timestamp }}  # Current execution timestamp
```

#### Template Filters
```yaml
{{ execution.timestamp | slugify }}  # Convert to filename-safe format
{{ process_data.processed_data | truncate(500) }}  # Preview first 500 chars
```

#### Parameter Substitution
```yaml
{{ output_path }}/output_{{ execution.timestamp | slugify }}.csv
```

### Data Processing Operations
The data-processing tool supports various operations:
- **Filtering**: Include/exclude records based on criteria
- **Transformations**: Modify data structure or values  
- **Aggregations**: Group and summarize data
- **Validations**: Check data quality and format

## Best Practices Demonstrated

1. **Clear Dependencies**: Explicit step dependencies ensure proper execution order
2. **Dynamic File Names**: Timestamp-based naming prevents file overwrites
3. **Comprehensive Reporting**: Both data and readable reports generated
4. **Template Usage**: Leverages template engine for dynamic content
5. **Error Prevention**: Uses relative paths and parameter defaults

## Common Use Cases

- **Data Cleaning**: Filter and clean CSV datasets
- **ETL Pipelines**: Extract, transform, and load data workflows
- **Report Generation**: Automated data processing with documentation
- **Quality Assurance**: Data validation and filtering workflows
- **Batch Processing**: Process multiple datasets with consistent logic

## Troubleshooting

### Common Issues

#### File Not Found
```
Error: Could not read file 'data/input.csv'
```
**Solution**: Ensure input file exists in the correct location

#### Permission Errors
```
Error: Cannot write to output directory
```
**Solution**: Check write permissions for output directory

#### Data Processing Errors
```
Error: Invalid CSV format
```
**Solution**: Validate CSV structure and encoding

### Debugging Tips

1. **Check Input Data**: Verify CSV format and content structure
2. **Validate Paths**: Ensure input and output paths are correct
3. **Review Dependencies**: Confirm step execution order is logical
4. **Examine Outputs**: Check both data file and report for completeness

## Extension Examples

### Adding Data Validation
```yaml
- id: validate_data
  tool: data-processing
  action: validate
  parameters:
    data: "{{ read_data.content }}"
    schema:
      required_columns: ["name", "status", "value", "created_date"]
      data_types:
        value: "integer"
        created_date: "date"
```

### Multiple Filter Criteria
```yaml
operation:
  criteria:
    status: "active"
    value: ">= 2000"
```

### Data Aggregation
```yaml
- id: aggregate_data
  tool: data-processing
  action: aggregate
  parameters:
    data: "{{ process_data.processed_data }}"
    group_by: "status"
    operations:
      total_value: "sum(value)"
      count: "count(*)"
```

## Related Examples
- [data_processing.md](data_processing.md) - More comprehensive data workflows
- [data_processing_pipeline.md](data_processing_pipeline.md) - Advanced data processing
- [validation_pipeline.md](validation_pipeline.md) - Data validation focus
- [statistical_analysis.md](statistical_analysis.md) - Statistical processing

## Technical Requirements

- **Tools**: filesystem, data-processing
- **Input Format**: CSV with headers
- **Memory**: Minimal (loads entire CSV into memory)
- **Storage**: Write access to output directory

This pipeline serves as an excellent starting point for learning data processing workflows and understanding the basic patterns used in more complex data pipelines.