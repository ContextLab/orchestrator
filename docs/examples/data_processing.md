# Data Processing Pipeline

**Pipeline**: `examples/data_processing.yaml`  
**Category**: Data Processing & Validation  
**Complexity**: Advanced  
**Key Features**: Schema validation, Data transformation, Multi-format support, AUTO model selection, Professional tools integration

## Overview

The Data Processing Pipeline provides comprehensive data processing capabilities including loading, parsing, validation, transformation, and output generation. It demonstrates enterprise-grade data processing workflows with schema validation, flexible transformations, and multiple output format support.

## Key Features Demonstrated

### 1. Multi-Tool Integration
```yaml
# File system operations
- id: load_data
  tool: filesystem
  action: read

# Data validation with schema
- id: validate_data
  tool: validation
  action: validate

# Data transformation operations
- id: transform_data
  tool: data-processing
  action: transform
```

### 2. Schema-Based Validation
```yaml
schema:
  type: object
  properties:
    records:
      type: array
      items:
        type: object
        properties:
          id: {type: integer}
          name: {type: string}
          active: {type: boolean}
        required: ["id", "name"]
mode: lenient  # Flexible validation mode
```

### 3. Complex Data Transformations
```yaml
operation:
  transformations:
    - type: filter
      field: active
      value: true
    - type: aggregate
      operation: sum
      field: value
```

### 4. Flexible Output Formatting
```yaml
parameters:
  output_format:
    type: string
    default: json
    description: Output format (json, csv, or yaml)
    
# Dynamic file extension
path: "{{ output_path }}/processed_data.{{ output_format }}"
```

### 5. Comprehensive Reporting
```yaml
# Professional processing report
content: |
  # Data Processing Report
  
  **Source File:** {{ data_source }}
  **Output Format:** {{ output_format }}
  
  ## Validation Results
  
  - Validation Status: {% if validate_data.valid %}Passed{% else %}Failed{% endif %}
  - Errors: {% if validate_data.errors %}{{ validate_data.errors | length }} errors found{% else %}None{% endif %}
```

## Pipeline Architecture

### Input Parameters
- **data_source** (optional): Path to input data file (default: "examples/test_data/sample_data.json")
- **output_format** (optional): Output format - json, csv, or yaml (default: json)
- **output_path** (optional): Output directory path (default: "examples/outputs/data_processing")

### Processing Flow

1. **Data Loading** - Reads data from specified source file (JSON/CSV supported)
2. **Data Parsing** - Identifies data structure and format automatically
3. **Data Validation** - Validates against predefined schema with flexible rules
4. **Data Transformation** - Applies filtering and aggregation operations
5. **Result Formatting** - Formats processed data into requested output format
6. **Data Persistence** - Saves processed data to output directory
7. **Summary Generation** - Creates processing summary with key metrics
8. **Report Creation** - Generates comprehensive processing report

### Supported Data Formats

#### Input Formats
- **JSON**: Structured JSON data with records array
- **CSV**: Comma-separated values with headers
- **Auto-Detection**: Pipeline automatically identifies format

#### Output Formats
- **JSON**: Structured JSON output (default)
- **CSV**: Comma-separated values
- **YAML**: YAML format for configuration-style output

### Data Transformation Operations

#### Filtering Operations
- **Field-Based Filtering**: Filter records by field values
- **Boolean Filters**: Active/inactive record filtering
- **Complex Conditions**: Multi-criteria filtering support

#### Aggregation Operations
- **Sum Operations**: Numerical field summation
- **Count Operations**: Record counting and statistics
- **Custom Aggregations**: Flexible aggregation definitions

## Usage Examples

### Basic JSON Processing
```bash
python scripts/run_pipeline.py examples/data_processing.yaml \
  -i data_source="examples/test_data/sample_data.json"
```

### CSV Processing with Custom Output
```bash
python scripts/run_pipeline.py examples/data_processing.yaml \
  -i data_source="examples/data/customers.csv" \
  -i output_format="csv" \
  -i output_path="my_output"
```

### YAML Output Format
```bash
python scripts/run_pipeline.py examples/data_processing.yaml \
  -i data_source="examples/test_data/sample_data.json" \
  -i output_format="yaml"
```

### Custom Data Source
```bash
python scripts/run_pipeline.py examples/data_processing.yaml \
  -i data_source="/path/to/my/data.json" \
  -i output_path="/custom/output/location"
```

## Sample Processing Flow

### Input Data Structure
```json
{
  "records": [
    {
      "id": 1,
      "name": "Product A",
      "category": "Electronics",
      "price": 299.99,
      "quantity": 50,
      "active": true,
      "value": 14999.50
    },
    {
      "id": 2,
      "name": "Product B",
      "category": "Clothing", 
      "price": 49.99,
      "quantity": 200,
      "active": true,
      "value": 9998.00
    }
  ]
}
```

### Transformation Process
1. **Filter**: Keep only records where `active: true`
2. **Aggregate**: Calculate sum of `value` field across filtered records
3. **Format**: Structure results as clean JSON output

### Output Data Structure
```json
{
  "processed_data": {
    "filtered_records": [
      {"id": 1, "name": "Product A", "active": true, "value": 14999.5},
      {"id": 2, "name": "Product B", "active": true, "value": 9998.0}
    ],
    "aggregation": {
      "operation": "sum",
      "field": "value", 
      "result": 24997.5
    }
  },
  "success": true
}
```

### Generated Report
Check the actual processing report: [processing_report.md](../../examples/outputs/data_processing/processing_report.md)

## Technical Implementation

### Schema Validation System
```yaml
tool: validation
action: validate
parameters:
  data: "{{ load_data.content }}"
  schema:
    type: object
    properties:
      records:
        type: array
        items:
          type: object
          properties:
            id: {type: integer}
            name: {type: string}
            active: {type: boolean}
          required: ["id", "name"]
  mode: lenient  # Allows additional properties
```

### Data Transformation Engine
```yaml
tool: data-processing
action: transform
parameters:
  data: "{{ load_data.content }}"
  operation:
    transformations:
      - type: filter      # Remove inactive records
        field: active
        value: true
      - type: aggregate    # Sum numerical values
        operation: sum
        field: value
```

### Clean Output Generation
```yaml
prompt: |
  Convert this data to clean JSON format:
  {{ transform_data }}
  
  Return ONLY valid JSON without any markdown formatting.
  Do NOT include ```json or ``` markers.
  Start directly with { and end with }
```

### Dynamic File Management
```yaml
# Dynamic output file naming
path: "{{ output_path }}/processed_data.{{ output_format }}"

# Professional report generation
path: "{{ output_path }}/processing_report.md"
```

## Advanced Features

### Flexible Validation Modes
- **Strict Mode**: Requires exact schema compliance
- **Lenient Mode**: Allows additional properties not in schema
- **Error Reporting**: Detailed validation error messages

### Multi-Stage Processing
1. **Parse Detection**: Automatic format identification
2. **Validation**: Schema compliance checking
3. **Transformation**: Data filtering and aggregation
4. **Formatting**: Clean output generation
5. **Reporting**: Comprehensive processing documentation

### Professional Output Quality
```yaml
# Clean JSON without explanatory text
Return ONLY valid JSON without any markdown formatting, code fences, or explanations.
Do NOT include ```json or ``` markers.
Start directly with { and end with }
```

### Comprehensive Error Handling
```yaml
# Validation status tracking
Validation Status: {% if validate_data.valid %}Passed{% else %}Failed{% endif %}
Errors: {% if validate_data.errors %}{{ validate_data.errors | length }} errors found{% else %}None{% endif %}
Warnings: {% if validate_data.warnings %}{{ validate_data.warnings | length }} warnings{% else %}None{% endif %}
```

## Common Use Cases

- **ETL Operations**: Extract, transform, and load data workflows
- **Data Validation**: Schema compliance and quality assurance
- **Report Generation**: Business intelligence and analytics reporting
- **Data Migration**: Format conversion and data cleaning
- **API Response Processing**: Transform API data for consumption
- **Configuration Management**: Process and validate configuration files
- **Database Import/Export**: Prepare data for database operations

## Best Practices Demonstrated

1. **Schema-First Design**: Define validation schema before processing
2. **Multi-Stage Processing**: Break complex operations into manageable steps
3. **Error Handling**: Comprehensive validation and error reporting
4. **Format Flexibility**: Support multiple input and output formats
5. **Clean Output**: Generate production-ready data without artifacts
6. **Documentation**: Comprehensive processing reports and summaries
7. **Tool Integration**: Leverage specialized tools for specific operations

## Troubleshooting

### Common Issues
- **Schema Validation Failures**: Verify data structure matches expected schema
- **Format Detection**: Ensure input files are valid JSON or CSV
- **File Path Errors**: Check input file exists and output directory is writable
- **Transformation Errors**: Verify field names exist in input data

### Performance Considerations
- **Large Files**: May require chunking for very large datasets
- **Complex Transformations**: Multiple operations may increase processing time
- **Output Format**: JSON generally faster than CSV for complex nested data
- **Validation Mode**: Lenient mode is faster than strict validation

### Data Quality Tips
- **Use Schema Validation**: Always validate input data structure
- **Check Transformation Results**: Verify aggregations and filters work correctly
- **Monitor Processing Reports**: Review generated reports for issues
- **Test with Sample Data**: Use small datasets for initial testing

## Related Examples
- [data_processing_pipeline.md](data_processing_pipeline.md) - Advanced data processing patterns
- [simple_data_processing.md](simple_data_processing.md) - Basic data processing workflows
- [validation_pipeline.md](validation_pipeline.md) - Data validation focus
- [statistical_analysis.md](statistical_analysis.md) - Statistical data processing

## Technical Requirements

- **Tools**: filesystem, validation, data-processing tools
- **Models**: Support for text generation and parsing
- **File System**: Read/write access for input and output operations
- **Memory**: Sufficient for processing dataset size

This pipeline provides enterprise-grade data processing capabilities suitable for production data workflows, quality assurance processes, and business intelligence applications.