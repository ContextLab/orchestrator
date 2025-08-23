# Conditional File Processing Pipeline

**Pipeline**: `examples/control_flow_conditional.yaml`  
**Category**: Control Flow & Logic  
**Complexity**: Intermediate  
**Key Features**: Conditional execution, File size analysis, Dynamic processing, AUTO model selection

## Overview

The Conditional File Processing Pipeline demonstrates sophisticated conditional logic by processing files differently based on their characteristics, primarily file size. It showcases how pipelines can make dynamic decisions about processing workflows, routing content through different transformations based on runtime conditions.

## Key Features Demonstrated

### 1. Conditional Step Execution
```yaml
# Large file processing
condition: "{{ read_file.size > size_threshold }}"

# Small file processing
condition: "{{ read_file.size <= size_threshold and read_file.size > 0 }}"

# Empty file handling
condition: "{{ read_file.size == 0 }}"
```

### 2. Dynamic Model Selection
```yaml
model: <AUTO task="analyze">Select a model for analysis</AUTO>
model: <AUTO task="summarize">Select a model for text summarization</AUTO>
model: <AUTO task="generate">Select a model for text generation</AUTO>
```

### 3. Template-Based Logic
```yaml
content: |
  Processing type: {% if read_file.size == 0 %}Empty file
  {% elif read_file.size > size_threshold %}Compressed
  {% else %}Expanded{% endif %}
```

### 4. Complex Output Routing
```yaml
processed_content: "{% if handle_empty.status is not defined or handle_empty.status != 'skipped' %}{{ handle_empty }}{% elif compress_large.status is not defined or compress_large.status != 'skipped' %}{{ compress_large }}{% elif expand_small.status is not defined or expand_small.status != 'skipped' %}{{ expand_small }}{% else %}No content processed.{% endif %}"
```

## Pipeline Architecture

### Input Parameters
- **input_file** (optional): Path to file to process (default: "data/sample.txt")
- **size_threshold** (optional): Size threshold in bytes for processing decisions (default: 1000)

### Processing Flow

1. **File Reading** - Loads the input file and captures size information
2. **Size Analysis** - Analyzes file size characteristics for decision making
3. **Conditional Processing** - Routes to one of three processing paths:
   - **Large Files** (> threshold): Compression/summarization processing
   - **Small Files** (≤ threshold, > 0): Expansion/enrichment processing
   - **Empty Files** (= 0): Special empty file handling
4. **Result Aggregation** - Combines processing results with metadata
5. **Output Generation** - Saves processed content with processing details

### Processing Paths

#### Large File Processing (Compression)
- **Trigger**: File size > threshold bytes
- **Action**: Summarizes content into exactly 3 bullet points
- **Purpose**: Reduces large content to manageable summaries
- **Features**: Pattern recognition, accurate counting, factual analysis

#### Small File Processing (Expansion)
- **Trigger**: File size ≤ threshold bytes and > 0
- **Action**: Expands content with additional context and details
- **Purpose**: Enriches small content with relevant information
- **Features**: Context addition, technical analysis, professional writing

#### Empty File Processing
- **Trigger**: File size = 0 bytes
- **Action**: Provides standardized empty file message
- **Purpose**: Handles edge case gracefully
- **Features**: Consistent error messaging

## Usage Examples

### Basic File Processing
```bash
python scripts/run_pipeline.py examples/control_flow_conditional.yaml \
  -i input_file="examples/data/sample.txt"
```

### Custom Size Threshold
```bash
python scripts/run_pipeline.py examples/control_flow_conditional.yaml \
  -i input_file="examples/data/large_document.txt" \
  -i size_threshold=500
```

### Processing Different File Types
```bash
# Small file expansion
python scripts/run_pipeline.py examples/control_flow_conditional.yaml \
  -i input_file="examples/data/short_note.txt" \
  -i size_threshold=2000

# Large file compression
python scripts/run_pipeline.py examples/control_flow_conditional.yaml \
  -i input_file="examples/data/research_paper.txt" \
  -i size_threshold=100
```

### Batch Testing
```bash
# Test various file sizes
for file in examples/data/*.txt; do
  python scripts/run_pipeline.py examples/control_flow_conditional.yaml \
    -i input_file="$file" \
    -i size_threshold=1000
done
```

## Sample Output Structure

### Processing Report Format
```markdown
# Processed File

Original size: [N] bytes
Processing type: [Compressed|Expanded|Empty file]

## Result

[Processed content based on file size and type]
```

### Example Outputs

#### Small File Expansion (150 bytes)
- **Input**: Simple test content
- **Output**: Detailed technical analysis explaining the content's purpose and context
- **Processing**: Adds relevant details, explains testing significance

#### Large File Compression (2000+ bytes)
- **Input**: Lengthy documentation or code
- **Output**: Exactly 3 bullet points summarizing key aspects
- **Processing**: Identifies patterns, counts elements, provides factual summaries

#### Empty File Handling (0 bytes)
- **Input**: Empty file
- **Output**: "The input file was empty. No content to process."
- **Processing**: Standardized empty file response

### Sample Files
Check actual generated outputs in: [processed_*.md](../../examples/outputs/control_flow_conditional/)

## Technical Implementation

### Conditional Logic Design
The pipeline implements a comprehensive conditional framework:

```yaml
# Sequential condition checking
- compress_large:   # if size > threshold
- expand_small:     # if size <= threshold AND size > 0  
- handle_empty:     # if size == 0
```

### Dependency Management
```yaml
dependencies:
  - compress_large
  - expand_small  
  - handle_empty
```
All conditional steps depend on the same prerequisites but execute based on different conditions.

### Template Processing
Advanced template logic for dynamic content generation:
```yaml
# Conditional content selection
{% if handle_empty.status is not defined or handle_empty.status != 'skipped' %}
  {{ handle_empty }}
{% elif compress_large.status is not defined or compress_large.status != 'skipped' %}
  {{ compress_large }}
{% elif expand_small.status is not defined or expand_small.status != 'skipped' %}
  {{ expand_small }}
{% else %}
  No content processed.
{% endif %}
```

### File Size Awareness
The pipeline maintains size awareness throughout processing:
- Access to `{{ read_file.size }}` in all steps
- Dynamic threshold comparison
- Size-appropriate token limits and processing strategies

## Advanced Features

### Pattern Recognition and Analysis
For repetitive content (testing files with repeated characters):
```yaml
prompt: |
  IMPORTANT: If the text is ONLY repeated single characters (like "XXXX" or "AAAA"), you MUST:
  1. Describe what this repetitive pattern represents in testing contexts
  2. Explain why files with repeated characters are used in software testing
  3. Discuss the significance of the specific byte count
```

### Professional Output Standards
```yaml
RULES:
- Start directly with the expanded content
- No conversational phrases like "Let's", "Okay", "Here's", etc.
- Be accurate about sizes (N bytes, not kilobytes)
- Write in professional, informative style
```

### Robust Error Handling
- Empty file detection and handling
- Size-based processing validation
- Graceful fallbacks for edge cases

## Common Use Cases

- **Content Management Systems**: Route content based on length/complexity
- **Data Processing Pipelines**: Apply different algorithms based on data size
- **File Processing Automation**: Handle various file sizes appropriately
- **Quality Assurance**: Test boundary conditions and edge cases
- **Document Processing**: Summarize large documents, expand brief notes
- **Testing Workflows**: Validate conditional logic with various inputs

## Best Practices Demonstrated

1. **Explicit Condition Logic**: Clear, readable conditional statements
2. **Comprehensive Edge Case Handling**: Empty files, boundary conditions
3. **Dynamic Model Selection**: Task-appropriate AI model choices
4. **Professional Output Quality**: Consistent, publication-ready results
5. **Template-Based Flexibility**: Reusable conditional logic patterns
6. **Dependency Optimization**: Efficient step execution based on conditions

## Troubleshooting

### Common Issues
- **File Path Errors**: Ensure input files exist and are accessible
- **Threshold Sensitivity**: Test different thresholds for optimal results
- **Empty File Handling**: Verify empty file detection works correctly
- **Template Syntax**: Check complex template expressions for errors

### Performance Considerations
- **Large Files**: May hit token limits; adjust max_tokens accordingly
- **Model Selection**: AUTO selection optimizes for task requirements
- **Conditional Efficiency**: Only executes relevant processing paths
- **Output Size**: Compressed outputs are concise, expanded outputs are detailed

## Related Examples
- [control_flow_for_loop.md](control_flow_for_loop.md) - Loop-based control flow
- [control_flow_while_loop.md](control_flow_while_loop.md) - While loop patterns
- [control_flow_dynamic.md](control_flow_dynamic.md) - Dynamic control flow
- [error_handling_examples.md](error_handling_examples.md) - Error handling patterns

## Technical Requirements

- **Models**: Support for analysis, summarization, and text generation
- **Tools**: Filesystem access for file reading and writing
- **Templates**: Jinja2 templating for conditional logic
- **Memory**: Sufficient for file content and processing results

This pipeline demonstrates sophisticated conditional processing patterns essential for building robust, adaptive data processing workflows that can handle diverse input characteristics intelligently.