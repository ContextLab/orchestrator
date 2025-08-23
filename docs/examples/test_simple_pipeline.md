# Simple Test Pipeline

**Pipeline**: `examples/test_simple_pipeline.yaml`  
**Category**: Fundamentals  
**Complexity**: Beginner  
**Key Features**: Basic pipeline structure, Text generation, File operations, Data visualization, AUTO model selection

## Overview

The Simple Test Pipeline provides a minimal example demonstrating core pipeline functionality. It generates test data, saves it to a file, and creates basic visualizations, making it perfect for testing pipeline execution, validating tool integrations, and learning fundamental pipeline concepts.

## Key Features Demonstrated

### 1. AUTO Model Selection
```yaml
- id: generate_data
  action: generate_text
  parameters:
    model: <AUTO>
```

### 2. File System Operations
```yaml
- id: save_data
  tool: filesystem
  action: write
  parameters:
    path: "{{ output_path }}/test_data.csv"
    content: "id,value,category\n1,100,A\n2,200,B\n3,150,A"
```

### 3. Data Visualization
```yaml
- id: create_chart
  tool: visualization
  action: create_charts
  parameters:
    data: [{"id": 1, "value": 100, "category": "A"}, {"id": 2, "value": 200, "category": "B"}, {"id": 3, "value": 150, "category": "A"}]
    chart_types: ["bar", "pie"]
```

### 4. Step Dependencies
```yaml
dependencies:
  - save_data  # create_chart waits for save_data to complete
```

## Pipeline Architecture

### Input Parameters
None (uses hardcoded test data)

### Processing Flow

1. **Generate Data** - Use LLM to generate sample CSV data
2. **Save Data** - Write CSV data to file system
3. **Create Charts** - Generate bar and pie charts from data

### Test Data Structure
```csv
id,value,category
1,100,A
2,200,B
3,150,A
```

- **id**: Unique identifier (1, 2, 3)
- **value**: Numeric values (100, 200, 150)  
- **category**: Categories (A, B)

## Usage Examples

### Basic Pipeline Test
```bash
python scripts/run_pipeline.py examples/test_simple_pipeline.yaml
```

### Custom Output Directory
```bash
python scripts/run_pipeline.py examples/test_simple_pipeline.yaml \
  -o examples/outputs/my_test
```

### Integration Test
```bash
# Run as part of test suite
python scripts/run_pipeline.py examples/test_simple_pipeline.yaml \
  --validate-outputs
```

## Generated Output Files

### File Structure
```
examples/outputs/test_simple_pipeline/
├── test_data.csv                    # Generated CSV data
└── charts/
    ├── bar_chart.png               # Bar chart visualization  
    └── pie_chart.png               # Pie chart visualization
```

### CSV Data Output
```csv
id,value,category
1,100,A
2,200,B
3,150,A
```

### Chart Generation
- **Bar Chart**: Shows values by ID with category colors
- **Pie Chart**: Shows proportional distribution of categories

## Step-by-Step Breakdown

### Step 1: Generate Data
```yaml
action: generate_text
parameters:
  prompt: "Generate a CSV with 3 rows: id,value,category\n1,100,A\n2,200,B\n3,150,A"
  model: <AUTO>
```

- **Purpose**: Demonstrate LLM text generation
- **Model Selection**: AUTO selects appropriate model
- **Output**: CSV-formatted text string

### Step 2: Save Data  
```yaml
tool: filesystem
action: write
parameters:
  path: "{{ output_path }}/test_data.csv"
  content: "id,value,category\n1,100,A\n2,200,B\n3,150,A"
```

- **Purpose**: Test file system write operations
- **Path Template**: Uses output_path variable
- **Content**: Hardcoded CSV data (not using generated data)

### Step 3: Create Charts
```yaml
tool: visualization
action: create_charts  
parameters:
  data: [{"id": 1, "value": 100, "category": "A"}, {"id": 2, "value": 200, "category": "B"}, {"id": 3, "value": 150, "category": "A"}]
  chart_types: ["bar", "pie"]
```

- **Purpose**: Test visualization capabilities
- **Data Format**: JSON array of objects
- **Chart Types**: Bar and pie charts
- **Dependencies**: Waits for save_data completion

## AUTO Model Selection

### Model Selection Logic
```yaml
model: <AUTO>
# System automatically selects optimal model based on:
# - Task type (text generation)
# - Prompt complexity (simple CSV generation)  
# - Cost considerations (lightweight task)
# - Availability (currently available models)
```

### Expected Model Selection
- **Task Type**: Simple text generation
- **Likely Selection**: Cost-effective model like Claude Haiku
- **Reasoning**: Minimal complexity, structured output

## Testing Capabilities

### Basic Functionality Tests
- **Text Generation**: Validates LLM integration
- **File Operations**: Tests filesystem tool functionality
- **Visualization**: Confirms chart generation capabilities
- **Dependencies**: Verifies step sequencing

### Integration Points
- **Model Selection**: Tests AUTO model routing
- **Template Variables**: Validates `{{ output_path }}` resolution
- **Tool Integration**: Confirms filesystem and visualization tools
- **Output Generation**: Tests structured output creation

### Validation Scenarios
- **Successful Execution**: All steps complete successfully
- **File Creation**: CSV file exists with correct content
- **Chart Generation**: Both bar and pie charts created
- **Output Structure**: Proper output variable population

## Common Use Cases

- **Pipeline Testing**: Validate basic pipeline functionality
- **Tool Integration**: Test individual tool capabilities
- **Development Testing**: Quick validation during development
- **CI/CD Integration**: Automated testing in build pipelines
- **Training Material**: Learning basic pipeline concepts
- **Debugging**: Minimal test case for troubleshooting

## Error Scenarios and Handling

### Potential Failure Points
1. **Model Selection**: AUTO model unavailable
2. **File Write**: Output directory permissions
3. **Visualization**: Chart generation tool errors
4. **Dependencies**: Step sequencing issues

### Troubleshooting
- **Model Issues**: Check available models and credentials
- **File Errors**: Verify output directory exists and is writable
- **Chart Problems**: Ensure visualization tool is properly configured
- **Dependency Errors**: Check step dependency logic

## Pipeline Optimization

### Performance Considerations
- **Model Selection**: AUTO chooses efficient model for simple task
- **Data Size**: Minimal data for fast processing
- **Chart Generation**: Only two simple chart types
- **Dependencies**: Minimal dependency chain

### Resource Usage
- **Compute**: Low - simple text generation task
- **Memory**: Minimal - small data set
- **Storage**: Small - CSV file and two images
- **Network**: Low - single model API call

## Extension Possibilities

### Enhanced Data Generation
```yaml
# Use generated data from step 1
content: "{{ generate_data.result }}"
```

### Additional Chart Types
```yaml
chart_types: ["bar", "pie", "line", "scatter", "histogram"]
```

### Dynamic Data
```yaml
parameters:
  row_count: 10
  categories: ["A", "B", "C", "D"]
```

### Data Validation
```yaml
- id: validate_data
  tool: data-validation
  action: validate_csv
  parameters:
    file_path: "{{ save_data.path }}"
    schema: {...}
```

## Related Examples
- [simple_data_processing.md](simple_data_processing.md) - Basic data processing patterns
- [auto_tags_demo.md](auto_tags_demo.md) - AUTO model selection examples
- [data_processing_pipeline.md](data_processing_pipeline.md) - Advanced data processing

## Technical Requirements

- **LLM Access**: Model API for text generation (AUTO selection)
- **File System**: Write permissions for output directory
- **Visualization**: Chart generation capabilities (matplotlib/plotly)
- **Template Engine**: Variable interpolation support
- **Dependency Management**: Step execution ordering

This pipeline serves as the foundation for understanding basic pipeline concepts and provides a reliable test case for validating system functionality and tool integration.