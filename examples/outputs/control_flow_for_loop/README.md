# Batch File Processing Output Examples

This directory contains real output examples from the Batch File Processing Pipeline (`examples/control_flow_for_loop.yaml`).

## Pipeline Overview

The Control Flow For-Loop Pipeline demonstrates parallel batch processing using for-each loops. It processes multiple files simultaneously, analyzes content, transforms it, and generates individual processed files with metadata.

**Pipeline Documentation**: [control_flow_for_loop.md](../../../docs/examples/control_flow_for_loop.md)

## Generated Files

### Processed Files
Each input file generates a corresponding processed output:

- **[processed_file1.txt](processed_file1.txt)** - First file processing result
- **[processed_file2.txt](processed_file2.txt)** - Second file processing result  
- **[processed_file3.txt](processed_file3.txt)** - Third file processing result

Each processed file contains:
- Original filename and processing metadata
- Loop variables (index, is_first, is_last flags)
- Original file size information
- AI-generated content analysis
- Transformed/condensed version of original content

### Summary Report
- **[summary.md](summary.md)** - Overall batch processing summary
  - Total files processed count
  - List of all input files
  - Output directory location
  - Processing completion confirmation

## Processing Details

### Loop Variables Demonstrated
Each processed file shows the for-each loop variables:
```
File index: 0              # Zero-based position in array
Is first: true             # Boolean flag for first item
Is last: false             # Boolean flag for last item
```

### Parallel Processing
- **max_parallel**: 2 files processed simultaneously  
- **Resource Management**: Balanced throughput vs system load
- **Dependencies**: Each file follows same processing chain:
  1. Read file content
  2. Analyze with AI model (AUTO-selected)
  3. Transform to more concise version
  4. Save with metadata

### Input Files
The pipeline processes these default files from `examples/data/`:
- `file1.txt` - First input file
- `file2.txt` - Second input file
- `file3.txt` - Third input file

## Usage Example

Basic batch processing:
```bash
python scripts/run_pipeline.py examples/control_flow_for_loop.yaml
```

Custom file list:
```bash
python scripts/run_pipeline.py examples/control_flow_for_loop.yaml \
  -i file_list='["document1.txt", "document2.txt", "report.txt"]'
```

Custom output directory:
```bash
python scripts/run_pipeline.py examples/control_flow_for_loop.yaml \
  -i output_dir="my_batch_results"
```

## Template Features Demonstrated

### Loop Variable Access
```yaml
path: "data/{{ $item }}"           # Current array element
content: |
  File index: {{ $index }}         # Zero-based position
  Is first: {{ $is_first }}        # First item boolean
  Is last: {{ $is_last }}          # Last item boolean
```

### Array Processing
```yaml
for_each: "{{ file_list }}"        # Iterate over parameter array
Total files processed: {{ file_list | length }}  # Array length
```

### Parallel Execution
```yaml
max_parallel: 2                    # Concurrent processing limit
# Balances performance with resource constraints
```

## Technical Architecture

- **Control Flow**: For-each loop with parallel execution
- **Dependency Chain**: read_file → analyze_content → transform_content → save_file
- **AI Integration**: AUTO tags select appropriate models per file
- **Template Engine**: Jinja2 with loop variables and filters
- **File System**: Read from data/ directory, write to outputs/

## Performance Characteristics

- **Throughput**: 2 files processed simultaneously
- **Scalability**: Easy to adjust file list and parallel limits  
- **Resource Usage**: Controlled by max_parallel setting
- **Error Isolation**: Individual file failures don't stop batch

## Related Examples
- [control_flow_conditional.md](../../../docs/examples/control_flow_conditional.md) - Conditional file processing
- [control_flow_while_loop.md](../../../docs/examples/control_flow_while_loop.md) - While loop patterns
- [fact_checker.md](../../../docs/examples/fact_checker.md) - Advanced parallel processing

This example serves as the foundation for any batch processing workflow requiring parallel execution with proper coordination and dependency management.