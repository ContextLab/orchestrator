# Data Validation Pipeline

**Pipeline**: `examples/validation_pipeline.yaml`  
**Category**: Quality & Testing  
**Complexity**: Intermediate  
**Key Features**: Data validation, Schema enforcement, Structured extraction, JSON Schema validation

## Overview

The Data Validation Pipeline demonstrates comprehensive data validation and structured information extraction capabilities. It validates data against JSON schemas, extracts structured information from unstructured text, and generates validation reports, making it essential for data quality assurance workflows.

## Key Features Demonstrated

### 1. Schema-Based Data Validation
```yaml
- id: validate_data
  tool: validation
  action: validate
  parameters:
    data: "{{ read_data.content | from_json }}"
    schema: "{{ read_config.content | from_json }}"
    mode: "strict"
```

### 2. Structured Information Extraction
```yaml
- id: extract_info
  tool: validation
  action: extract_structured
  parameters:
    text: "John Doe, age 30, email: john@example.com, phone: +1-555-0123"
    schema:
      type: object
      properties:
        name: {type: string}
        age: {type: integer}
        email: {type: string, format: email}
```

### 3. JSON Schema Definition
```yaml
schema:
  type: object
  properties:
    name: {type: string}
    age: {type: integer}
    email: {type: string, format: email}
    phone: {type: string, pattern: "^\\+?[1-9]\\d{1,14}$"}
  required: ["name", "email"]
```

### 4. Validation Report Generation
```yaml
content: |
  {
    "validation_result": {{ validate_data | to_json }},
    "extracted_data": {{ extract_info | to_json }},
    "timestamp": "{{ now() }}"
  }
```

## Pipeline Architecture

### Input Parameters
None (uses predefined data files and schemas)

### Processing Flow

1. **Read Configuration** - Load validation schema from JSON file
2. **Read Data** - Load data to be validated
3. **Validate Data** - Validate data against schema in strict mode
4. **Extract Information** - Extract structured data from unstructured text
5. **Save Report** - Generate comprehensive validation report

### Required Input Files

#### Validation Schema (`config/validation_schema.json`)
```json
{
  "type": "object",
  "properties": {
    "id": {"type": "integer", "minimum": 1},
    "name": {"type": "string", "minLength": 2},
    "email": {"type": "string", "format": "email"},
    "age": {"type": "integer", "minimum": 0, "maximum": 150},
    "active": {"type": "boolean"}
  },
  "required": ["id", "name", "email"],
  "additionalProperties": false
}
```

#### User Data (`data/user_data.json`)
```json
{
  "id": 1,
  "name": "John Doe",
  "email": "john.doe@example.com",
  "age": 30,
  "active": true
}
```

## Usage Examples

### Basic Validation Test
```bash
python scripts/run_pipeline.py examples/validation_pipeline.yaml
```

### Custom Schema Validation
```bash
# First, create custom schema and data files
mkdir -p examples/outputs/validation_pipeline/config
mkdir -p examples/outputs/validation_pipeline/data

# Then run validation
python scripts/run_pipeline.py examples/validation_pipeline.yaml
```

### Integration Testing
```bash
# Validate multiple data sets
for file in data/*.json; do
  python scripts/run_pipeline.py examples/validation_pipeline.yaml \
    --modify-input "read_data.parameters.path=$file"
done
```

## Validation Modes and Features

### Strict Mode Validation
```yaml
mode: "strict"
# Enforces all schema requirements
# Rejects data with additional properties
# Requires all mandatory fields
# Validates format constraints
```

### Lenient Mode Validation
```yaml
mode: "lenient"
# Allows additional properties
# Optional field validation
# Flexible format checking
# Warning-based validation
```

### Schema Validation Types

#### Data Type Validation
```yaml
type: "string"      # String validation
type: "integer"     # Integer validation
type: "boolean"     # Boolean validation
type: "array"       # Array validation
type: "object"      # Object validation
```

#### Format Validation
```yaml
format: "email"     # Email format validation
format: "date"      # Date format validation
format: "uri"       # URI format validation
format: "uuid"      # UUID format validation
```

#### Pattern Validation
```yaml
pattern: "^\\+?[1-9]\\d{1,14}$"  # Regex pattern matching
minLength: 2                      # Minimum string length
maxLength: 100                    # Maximum string length
```

#### Numeric Constraints
```yaml
minimum: 0          # Minimum value
maximum: 150        # Maximum value
multipleOf: 5       # Must be multiple of value
```

## Structured Information Extraction

### Text-to-Structure Conversion
```yaml
# Input: Unstructured text
text: "John Doe, age 30, email: john@example.com, phone: +1-555-0123"

# Output: Structured JSON
{
  "name": "John Doe",
  "age": 30,
  "email": "john@example.com", 
  "phone": "+1-555-0123"
}
```

### Extraction Schema Definition
```yaml
schema:
  type: object
  properties:
    name: {type: string}
    age: {type: integer}
    email: {type: string, format: email}
    phone: {type: string, pattern: "^\\+?[1-9]\\d{1,14}$"}
  required: ["name", "email"]
```

### Model-Powered Extraction
```yaml
model: "gpt-4o-mini"
# Uses AI model to intelligently extract structured data
# Handles various text formats and structures
# Validates extracted data against schema
```

## Sample Validation Results

### Successful Validation
```json
{
  "validation_result": {
    "valid": true,
    "errors": [],
    "warnings": [],
    "data": {
      "id": 1,
      "name": "John Doe",
      "email": "john.doe@example.com",
      "age": 30,
      "active": true
    }
  },
  "extracted_data": {
    "name": "John Doe",
    "age": 30,
    "email": "john@example.com",
    "phone": "+1-555-0123"
  },
  "timestamp": "2024-08-23T10:30:00Z"
}
```

### Failed Validation
```json
{
  "validation_result": {
    "valid": false,
    "errors": [
      {
        "field": "email",
        "message": "Invalid email format",
        "value": "invalid-email"
      },
      {
        "field": "age", 
        "message": "Value exceeds maximum (150)",
        "value": 200
      }
    ],
    "warnings": [
      {
        "field": "phone",
        "message": "Recommended format not followed"
      }
    ]
  }
}
```

## Advanced Validation Patterns

### Conditional Validation
```yaml
schema:
  type: object
  properties:
    user_type: {type: string, enum: ["admin", "user"]}
    permissions: {type: array}
  if:
    properties:
      user_type: {const: "admin"}
  then:
    properties:
      permissions: {minItems: 1}
```

### Nested Object Validation
```yaml
schema:
  type: object
  properties:
    address:
      type: object
      properties:
        street: {type: string}
        city: {type: string}
        zipcode: {type: string, pattern: "^\\d{5}(-\\d{4})?$"}
      required: ["street", "city"]
```

### Array Validation
```yaml
schema:
  type: object
  properties:
    tags:
      type: array
      items: {type: string}
      uniqueItems: true
      minItems: 1
      maxItems: 10
```

## Error Handling and Reporting

### Validation Error Types
- **Type Errors**: Data type mismatch
- **Format Errors**: Invalid format (email, phone, etc.)
- **Constraint Errors**: Min/max violations
- **Required Field Errors**: Missing mandatory fields
- **Pattern Errors**: Regex pattern match failures

### Error Structure
```json
{
  "field": "field_name",
  "message": "Human-readable error description", 
  "code": "ERROR_CODE",
  "value": "actual_value_that_failed",
  "expected": "expected_format_or_constraint"
}
```

### Warning Categories
- **Formatting Suggestions**: Non-critical format improvements
- **Best Practice Violations**: Recommended but not required patterns
- **Performance Hints**: Optimization suggestions
- **Security Recommendations**: Security best practices

## Integration Patterns

### Pre-Processing Validation
```yaml
# Validate data before processing
- id: validate_input
  tool: validation
  parameters:
    data: "{{ raw_input }}"
    schema: "{{ input_schema }}"
  
- id: process_data
  dependencies: [validate_input]
  condition: "{{ validate_input.valid }}"
```

### Post-Processing Validation
```yaml
# Validate results after processing  
- id: process_data
  action: transform_data
  
- id: validate_output
  tool: validation
  parameters:
    data: "{{ process_data.result }}"
    schema: "{{ output_schema }}"
```

### Batch Validation
```yaml
# Validate multiple data items
for_each: "{{ data_batch }}"
steps:
  - id: validate_item
    tool: validation
    parameters:
      data: "{{ item }}"
      schema: "{{ validation_schema }}"
```

## Best Practices Demonstrated

1. **Schema Separation**: External schema files for maintainability
2. **Strict Validation**: Comprehensive data quality enforcement
3. **Error Reporting**: Detailed validation feedback
4. **Structured Extraction**: AI-powered data extraction from text
5. **Report Generation**: Comprehensive validation documentation
6. **Template Integration**: Clean JSON output formatting

## Common Use Cases

- **API Input Validation**: Validate request data against API schemas
- **Data Migration**: Ensure data quality during system migrations
- **ETL Pipelines**: Validate data at each transformation stage
- **User Input Processing**: Validate and extract user-provided data
- **Data Quality Assurance**: Continuous data quality monitoring
- **Configuration Validation**: Validate configuration files and settings

## Troubleshooting

### Schema Issues
- Verify JSON schema syntax is correct
- Check schema file paths and accessibility
- Validate schema against JSON Schema specification

### Data Format Problems
- Ensure data is valid JSON format
- Check for encoding issues in input files
- Validate data structure matches schema expectations

### Extraction Failures
- Verify model availability and credentials
- Check text format and structure for extraction
- Ensure extraction schema is appropriate for text content

## Related Examples
- [simple_error_handling.md](simple_error_handling.md) - Error handling in validation workflows
- [data_processing_pipeline.md](data_processing_pipeline.md) - Data processing with validation
- [statistical_analysis.md](statistical_analysis.md) - Analysis with data validation

## Technical Requirements

- **Validation Engine**: JSON Schema validation capabilities
- **AI Model Access**: For structured information extraction
- **File System**: Read/write access for schemas and data
- **Template Engine**: JSON template processing
- **Schema Standards**: JSON Schema specification compliance

This pipeline provides enterprise-grade data validation and quality assurance capabilities essential for maintaining data integrity in production systems and automated workflows.