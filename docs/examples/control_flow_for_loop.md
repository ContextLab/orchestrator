# Batch File Processing with For-Each Loop

**Pipeline**: `examples/control_flow_for_loop.yaml`  
**Category**: Control Flow Examples  
**Complexity**: Intermediate  
**Key Features**: For-each loops, Parallel processing, AUTO tags, File operations

## Overview

The Batch File Processing Pipeline demonstrates for-each control flow for processing multiple files in parallel. It showcases how to iterate over lists, utilize loop variables, and coordinate parallel operations with proper dependency management.

## Key Features Demonstrated

### 1. For-Each Loop Structure
```yaml
- id: process_files
  for_each: "{{ file_list }}"  # Iterate over parameter array
  max_parallel: 2              # Process 2 files simultaneously
  steps:
    # Nested steps execute for each item
```

### 2. Loop Variable Access
```yaml
parameters:
  path: "data/{{ $item }}"     # Current list item
  content: |
    File index: {{ $index }}    # 0-based index
    Is first: {{ $is_first }}   # Boolean: first item
    Is last: {{ $is_last }}     # Boolean: last item
```

### 3. Parallel Processing with Dependencies
```yaml
steps:
  - id: read_file
    # Reads individual file
  - id: analyze_content
    dependencies: [read_file]   # Depends on file read
  - id: transform_content
    dependencies: [analyze_content]  # Sequential processing
  - id: save_file
    dependencies: [transform_content]  # Chain dependencies
```

### 4. AUTO Tags in Loops
```yaml
model: <AUTO task="analyze">Select a model for text analysis</AUTO>
# AUTO tag resolved independently for each loop iteration
```

## Pipeline Structure

### Input Parameters
- **file_list** (array): List of files to process (default: ["file1.txt", "file2.txt", "file3.txt"])
- **output_dir** (string): Output directory path (default: "examples/outputs/control_flow_for_loop")

### Processing Flow

1. **Create Output Directory** - Sets up directory structure
2. **For-Each File Processing** (parallel with max_parallel=2):
   - **Read File** - Load file content from data directory
   - **Analyze Content** - AI-powered content analysis
   - **Transform Content** - Generate concise version
   - **Save Processed File** - Write results with metadata
3. **Create Summary Report** - Generate overall processing summary

### Loop Processing Details

Each file undergoes identical processing:
1. File is read using filesystem tool
2. Content analyzed with AI model (AUTO-selected)
3. Content transformed to be more concise
4. Results saved with loop metadata (index, position flags)

## Usage Examples

### Basic Usage
```bash
python scripts/run_pipeline.py examples/control_flow_for_loop.yaml
```

### Custom File List
```bash
python scripts/run_pipeline.py examples/control_flow_for_loop.yaml \
  -i file_list='["document1.txt", "document2.txt", "report.txt"]'
```

### Custom Output Directory
```bash
python scripts/run_pipeline.py examples/control_flow_for_loop.yaml \
  -i output_dir="my_batch_results" \
  -i file_list='["data1.txt", "data2.txt"]'
```

## Sample Outputs

### Generated Files
For each input file, the pipeline creates:
- `processed_file1.txt` - Transformed version with metadata
- `processed_file2.txt` - Processed second file
- `processed_file3.txt` - Processed third file
- `summary.md` - Overall processing summary

### Sample Processed File Structure
```markdown
# Processed: file1.txt

File index: 0
Is first: true
Is last: false

## Original Size
1234 bytes

## Analysis
[AI-generated content analysis]

## Transformed Content
[Concise version of original content]
```

### Summary Report
- [View summary.md](../../examples/outputs/control_flow_for_loop/summary.md)

The summary includes:
- Total files processed count
- List of all processed files
- Output directory location

## Advanced Features

### Loop Variables Reference
Available in all loop steps:
- `{{ $item }}` - Current array element
- `{{ $index }}` - Zero-based position (0, 1, 2, ...)
- `{{ $is_first }}` - Boolean indicating first item
- `{{ $is_last }}` - Boolean indicating last item

### Parallel Execution Control
```yaml
max_parallel: 2  # Limit concurrent processing
# Balances performance vs resource usage
```

### Nested Dependencies
```yaml
dependencies:
  - read_file              # Previous step in same loop
  - transform_content      # Chain within loop
# External: - create_output_dir  # Step outside loop
```

### Template Integration with Loops
```yaml
content: |
  {% for file in file_list %}
  - {{ file }}              # Template loop in content
  {% endfor %}
```

## Performance Characteristics

### Parallel Processing Benefits
- **Throughput**: Process multiple files simultaneously
- **Resource Utilization**: Efficient CPU and I/O usage
- **Scalability**: Easily handle larger file sets

### Resource Management
- **max_parallel**: Prevents overwhelming system resources
- **Memory Usage**: Each parallel task uses separate memory
- **API Limits**: AUTO tags resolved per iteration

## Best Practices Demonstrated

1. **Resource Control**: Limited parallel execution prevents overload
2. **Dependency Management**: Clear step ordering within loops
3. **Metadata Preservation**: Loop variables capture processing context
4. **Error Isolation**: Individual file failures don't stop others
5. **Summary Generation**: Aggregate results for overview

## Common Use Cases

- **Document Processing**: Batch analyze multiple documents
- **Data Transformation**: Convert file formats across datasets
- **Content Analysis**: Analyze multiple content pieces
- **File Validation**: Check multiple files for compliance
- **Batch Operations**: Any operation needing parallelization

## Troubleshooting

### Common Issues

#### File Not Found Errors
```
Error: Could not read file 'data/missing.txt'
```
**Solution**: Ensure all files in file_list exist in data directory

#### Parallel Processing Issues
- **Memory pressure**: Reduce max_parallel value
- **API rate limits**: Lower parallel execution count
- **Resource contention**: Adjust based on system capabilities

#### Template Errors
```
Error: Undefined variable '$item'
```
**Solution**: Loop variables only available within for_each steps

### Debugging Tips

1. **Start Small**: Test with single file first
2. **Check Dependencies**: Verify step execution order
3. **Monitor Resources**: Watch CPU and memory during parallel execution
4. **Validate Outputs**: Check generated files for completeness

## Extension Examples

### Dynamic File Discovery
```yaml
parameters:
  file_pattern: "*.txt"
  
steps:
  - id: find_files
    tool: filesystem
    action: list
    parameters:
      pattern: "{{ file_pattern }}"
      
  - id: process_found_files
    for_each: "{{ find_files.files }}"
```

### Error Handling Per File
```yaml
- id: process_with_recovery
  for_each: "{{ file_list }}"
  steps:
    - id: safe_read
      tool: filesystem
      action: read
      parameters:
        path: "data/{{ $item }}"
      on_error: "continue"  # Don't stop batch on single failure
```

### Conditional Processing
```yaml
steps:
  - id: process_if_large
    condition: "{{ read_file.size > 1000 }}"
    action: analyze_text
    # Only analyze files larger than 1KB
```

## Related Examples
- [control_flow_conditional.md](control_flow_conditional.md) - Conditional processing
- [control_flow_while_loop.md](control_flow_while_loop.md) - Iterative processing
- [control_flow_advanced.md](control_flow_advanced.md) - Complex control flows
- [fact_checker.md](fact_checker.md) - Advanced parallel processing

## Technical Requirements

- **Tools**: filesystem, analyze_text, generate_text
- **Models**: AUTO-selected models for analysis and generation
- **Memory**: Scales with max_parallel setting
- **Storage**: Write access to output directory

This pipeline serves as an excellent template for any batch processing workflow requiring parallel execution with proper coordination and dependency management.