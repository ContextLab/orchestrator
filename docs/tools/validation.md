# ValidationTool Documentation

The ValidationTool provides comprehensive data validation capabilities for orchestrator pipelines, including JSON Schema validation, custom format support, and intelligent type coercion.

## Overview

The ValidationTool supports three main actions:
- `validate` - Validate data against a JSON Schema
- `infer_schema` - Automatically infer a schema from sample data
- `extract_structured` - Extract structured data from text (coming soon)

## Basic Usage

### Simple Validation

```yaml
steps:
  - id: validate_user_data
    tool: validation
    action: validate
    parameters:
      data: "{{ user_input }}"
      schema:
        type: object
        properties:
          name:
            type: string
            minLength: 1
          email:
            type: string
            format: email
          age:
            type: integer
            minimum: 0
            maximum: 150
        required: ["name", "email"]
      mode: strict
```

### Validation Modes

The tool supports three validation modes:

1. **strict** (default) - Fail on any validation error
2. **lenient** - Attempt to coerce compatible types and warn on minor issues
3. **report_only** - Never fail, only report validation issues

#### Lenient Mode Example

```yaml
steps:
  - id: validate_with_coercion
    tool: validation
    action: validate
    parameters:
      data:
        count: "42"        # String will be coerced to integer
        active: "true"     # String will be coerced to boolean
        price: "19.99"     # String will be coerced to number
      schema:
        type: object
        properties:
          count:
            type: integer
          active:
            type: boolean
          price:
            type: number
      mode: lenient
```

### Schema Inference

Automatically generate a JSON Schema from sample data:

```yaml
steps:
  - id: analyze_structure
    tool: validation
    action: infer_schema
    parameters:
      data: "{{ sample_data }}"
```

## Built-in Format Validators

The ValidationTool includes several orchestrator-specific format validators:

### model-id
Validates AI model identifiers in the format `provider/model-name`.

**Examples:**
- ✅ `openai/gpt-4`
- ✅ `anthropic/claude-3-5-sonnet`
- ✅ `google/gemini-pro`
- ❌ `gpt-4` (missing provider)
- ❌ `invalid model` (contains space)

### tool-name
Validates tool names (lowercase, alphanumeric, hyphens, underscores).

**Examples:**
- ✅ `web-search`
- ✅ `file_system`
- ✅ `validation`
- ❌ `Tool-Name` (uppercase not allowed)
- ❌ `123tool` (cannot start with number)

### file-path
Validates file system paths.

**Examples:**
- ✅ `/path/to/file.txt`
- ✅ `relative/path.py`
- ✅ `C:\\Windows\\System32\\config.sys`
- ❌ ` /starts/with/space`
- ❌ `` (empty string)

### yaml-path
Validates JSONPath expressions for data access.

**Examples:**
- ✅ `$.data.items[0]`
- ✅ `data.nested.field`
- ✅ `$[*].name`
- ✅ `.result`

### pipeline-ref
Validates pipeline identifiers.

**Examples:**
- ✅ `data-processing`
- ✅ `ml_workflow`
- ✅ `etl-pipeline-v2`

### task-ref
Validates task output references in the format `task_id.field`.

**Examples:**
- ✅ `process_data.result`
- ✅ `step-1.output`
- ✅ `validation.errors`
- ❌ `invalid_ref` (missing field)

## Custom Format Validators

You can register custom format validators in your pipeline:

### Pattern-based Validator

```python
# In your orchestrator setup
from orchestrator.tools.validation import ValidationTool

tool = ValidationTool()
tool.register_format(
    name="order-id",
    validator=r"^ORD-\d{6}$",  # Regex pattern
    description="Order ID format (ORD-XXXXXX)"
)
```

### Function-based Validator

```python
def validate_even_number(value):
    return isinstance(value, int) and value % 2 == 0

tool.register_format(
    name="even-number",
    validator=validate_even_number,
    description="Even integer validator"
)
```

## Advanced Schema Examples

### Nested Objects with Arrays

```yaml
schema:
  type: object
  properties:
    users:
      type: array
      minItems: 1
      items:
        type: object
        properties:
          id:
            type: string
            format: uuid
          profile:
            type: object
            properties:
              name:
                type: string
              email:
                type: string
                format: email
              roles:
                type: array
                items:
                  type: string
                  enum: ["admin", "user", "guest"]
        required: ["id", "profile"]
```

### Conditional Validation

```yaml
schema:
  type: object
  properties:
    type:
      type: string
      enum: ["individual", "company"]
    name:
      type: string
    company_id:
      type: string
  required: ["type", "name"]
  if:
    properties:
      type:
        const: "company"
  then:
    required: ["company_id"]
```

## Working with AUTO Tags

The ValidationTool supports AUTO tags for dynamic schema and mode selection:

```yaml
steps:
  - id: smart_validation
    tool: validation
    action: validate
    parameters:
      data: "{{ input_data }}"
      schema: <AUTO>Infer appropriate schema based on the data structure</AUTO>
      mode: <AUTO>Choose validation mode based on data quality and criticality</AUTO>
```

## Error Handling

Validation errors include detailed information:

```json
{
  "success": true,
  "valid": false,
  "errors": [
    {
      "message": "'invalid-email' is not a 'email'",
      "path": ["users", 0, "email"],
      "schema_path": ["properties", "users", "items", "properties", "email", "format"],
      "instance": "invalid-email",
      "validator": "format",
      "validator_value": "email"
    }
  ],
  "warnings": [],
  "mode": "strict"
}
```

## Type Coercion Rules (Lenient Mode)

In lenient mode, the following type coercions are supported:

| From Type | To Type | Example |
|-----------|---------|---------|
| string | integer | `"42"` → `42` |
| string | number | `"3.14"` → `3.14` |
| string | boolean | `"true"` → `true`, `"false"` → `false` |
| number | string | `42` → `"42"` |
| integer | string | `42` → `"42"` |

Boolean string values recognized:
- True: `"true"`, `"yes"`, `"1"`
- False: `"false"`, `"no"`, `"0"`

## Integration with Pipelines

### Data Quality Gate

```yaml
steps:
  - id: fetch_data
    tool: web-search
    parameters:
      query: "{{ search_term }}"
  
  - id: validate_results
    tool: validation
    action: validate
    parameters:
      data: "{{ fetch_data.results }}"
      schema:
        type: array
        items:
          type: object
          properties:
            title:
              type: string
            url:
              type: string
              format: uri
            snippet:
              type: string
          required: ["title", "url"]
      mode: strict
    
  - id: process_valid_data
    tool: data-processing
    parameters:
      data: "{{ validate_results.data }}"
    dependencies: [validate_results]
    condition: "{{ validate_results.valid == true }}"
```

### Dynamic Schema Generation

```yaml
steps:
  - id: get_sample
    tool: file-system
    action: read
    parameters:
      path: "sample_data.json"
  
  - id: generate_schema
    tool: validation
    action: infer_schema
    parameters:
      data: "{{ get_sample.content }}"
  
  - id: validate_full_dataset
    tool: validation
    action: validate
    parameters:
      data: "{{ full_dataset }}"
      schema: "{{ generate_schema.schema }}"
      mode: lenient
```

## Best Practices

1. **Use strict mode for critical data** - Financial data, user inputs, API responses
2. **Use lenient mode for data migration** - When dealing with legacy systems or varied data sources
3. **Use report_only for monitoring** - To understand data quality without breaking pipelines
4. **Leverage schema inference** - Start with inferred schemas and refine them
5. **Document custom formats** - Always provide descriptions for custom validators
6. **Test edge cases** - Validate empty arrays, null values, and boundary conditions

## Performance Considerations

- Schema compilation is cached for repeated validations
- Large datasets are validated efficiently using streaming
- Custom validators should be lightweight and fast
- Complex schemas with many conditional rules may impact performance

## Future Features

- **Structured Output Extraction**: Use LLMs to extract structured data from unstructured text
- **Schema Evolution**: Track and manage schema changes over time
- **Validation Reports**: Generate detailed validation reports with statistics
- **Auto-fix Suggestions**: Provide automated fixes for common validation errors