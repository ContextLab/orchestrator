# ValidationTool Quick Reference

## Actions

| Action | Description | Required Parameters |
|--------|-------------|-------------------|
| `validate` | Validate data against JSON Schema | `data`, `schema` |
| `infer_schema` | Generate schema from sample data | `data` |
| `extract_structured` | Extract structured data from text (coming soon) | `text`, `schema` |

## Parameters

| Parameter | Type | Description | Default | Used By |
|-----------|------|-------------|---------|---------|
| `action` | string | Action to perform | `"validate"` | All |
| `data` | any | Data to validate or analyze | - | validate, infer_schema |
| `schema` | object | JSON Schema for validation | - | validate |
| `mode` | string | Validation mode: `strict`, `lenient`, `report_only` | `"strict"` | validate |
| `text` | string | Text to extract from | - | extract_structured |
| `model` | string | Model for extraction | - | extract_structured |

## Built-in Formats

| Format | Pattern/Rule | Example |
|--------|-------------|---------|
| `model-id` | `^[\w\-]+\/[\w\-\.:]+$` | `openai/gpt-4` |
| `tool-name` | `^[a-z][a-z0-9\-_]*$` | `web-search` |
| `file-path` | Non-empty, no leading space | `/path/to/file.txt` |
| `yaml-path` | `^$?\.?[\w\[\]\.\*]+$` | `$.data.items[0]` |
| `pipeline-ref` | `^[\w\-]+$` | `data-pipeline` |
| `task-ref` | `^[\w\-]+\.[\w\-]+$` | `task1.output` |

## Common Schemas

### User Object
```yaml
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
required: ["name", "email"]
```

### API Response
```yaml
type: object
properties:
  status:
    type: string
    enum: ["success", "error"]
  data:
    type: object
  message:
    type: string
required: ["status"]
```

### File List
```yaml
type: array
items:
  type: object
  properties:
    path:
      type: string
      format: file-path
    size:
      type: integer
      minimum: 0
    modified:
      type: string
      format: date-time
```

## Type Coercion (Lenient Mode)

| From | To | Examples |
|------|-----|----------|
| string | integer | `"42"` → `42` |
| string | number | `"3.14"` → `3.14` |
| string | boolean | `"true"` → `true`, `"1"` → `true` |
| number/int | string | `42` → `"42"` |

## Examples

### Basic Validation
```yaml
- tool: validation
  parameters:
    data: "{{ input }}"
    schema:
      type: object
      properties:
        id: { type: string }
        value: { type: number }
```

### With Custom Format
```yaml
- tool: validation  
  parameters:
    data: 
      order_id: "ORD-123456"
    schema:
      type: object
      properties:
        order_id:
          type: string
          format: order-id  # Custom format
```

### Schema Inference
```yaml
- tool: validation
  action: infer_schema
  parameters:
    data: "{{ sample_data }}"
```

### AUTO Tags
```yaml
- tool: validation
  parameters:
    data: "{{ input }}"
    schema: <AUTO>Appropriate schema for user profile data</AUTO>
    mode: <AUTO>Best validation mode for this data type</AUTO>
```