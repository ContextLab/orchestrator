# Orchestrator Examples

This directory contains working examples demonstrating the major features of the Orchestrator framework. Each example is a complete, runnable pipeline that showcases specific capabilities.

## Running Examples

To run any example:

```bash
python scripts/run_pipeline.py examples/[example_name].yaml
```

With inputs:
```bash
python scripts/run_pipeline.py examples/[example_name].yaml -i key=value -i another_key="complex value"
```

## Research Pipelines

We provide three research pipeline examples with increasing complexity:

### 1. research_minimal.yaml
- Simplest approach - just search and summarize
- Requires only web-search tool
- Quick results with minimal setup
- Best for: Quick topic overviews

### 2. research_basic.yaml  
- Uses standard LLM actions (analyze_text, generate_text)
- No specialized tools required (beyond web-search)
- Structured report with sections
- Best for: Detailed research without special tools

### 3. research_advanced_tools.yaml
- Uses specialized tools (headless-browser, report-generator, pdf-compiler)
- Can scrape full web pages and generate PDFs
- Most comprehensive output
- Best for: Professional reports with PDF output

Choose based on your available tools and output requirements.

## Available Examples

### Core Features

1. **[auto_tags_demo.yaml](auto_tags_demo.yaml)** - AUTO tags for dynamic intelligence
   - Dynamic parameter resolution
   - Conditional logic with AUTO tags
   - Error handling decisions

2. **[model_routing_demo.yaml](model_routing_demo.yaml)** - Intelligent model selection
   - Automatic model selection
   - Cost-optimized routing
   - Fallback strategies

3. **[parallel_execution.yaml](parallel_execution.yaml)** - Parallel task execution
   - Independent task parallelization
   - Resource management
   - Dependency handling

### Tool Demonstrations

4. **[web_research_pipeline.yaml](web_research_pipeline.yaml)** - Web research automation
   - Web search integration
   - Content analysis
   - Report generation

5. **[data_processing_pipeline.yaml](data_processing_pipeline.yaml)** - Data transformation
   - Multiple format support
   - Filtering and aggregation
   - Validation

6. **[code_analysis_pipeline.yaml](code_analysis_pipeline.yaml)** - Code analysis workflow
   - File system operations
   - Code parsing
   - Documentation generation

### Advanced Features

7. **[recursive_pipeline.yaml](recursive_pipeline.yaml)** - Recursive execution
   - Self-referencing pipelines
   - Dynamic iteration
   - State management

8. **[sub_pipeline_demo.yaml](sub_pipeline_demo.yaml)** - Modular pipelines
   - Pipeline composition
   - Context inheritance
   - Output aggregation

9. **[error_recovery.yaml](error_recovery.yaml)** - Error handling
   - Retry strategies
   - Fallback mechanisms
   - Graceful degradation

10. **[multimodal_pipeline.yaml](multimodal_pipeline.yaml)** - Multimodal processing
    - Image analysis
    - Audio transcription
    - Combined insights

## Quick Start Examples

### Simple AUTO Tag Example

```yaml
# auto_tag_simple.yaml
name: simple-auto-demo
description: Basic AUTO tag usage

steps:
  - id: choose_format
    tool: llm-generate
    action: generate
    parameters:
      prompt: "We need to create a report"
      format: <AUTO>Choose the best format: 'pdf' or 'markdown'</AUTO>
      
  - id: create_report
    tool: report-generator
    action: generate
    parameters:
      title: "Demo Report"
      format: "{{ choose_format.format }}"
      content: "Report in {{ choose_format.format }} format"
```

### Basic Data Processing

```yaml
# data_transform_simple.yaml
name: simple-data-transform
description: Transform CSV to JSON

steps:
  - id: read_csv
    tool: filesystem
    action: read
    parameters:
      path: "data.csv"
      
  - id: transform
    tool: data-processing
    action: transform
    parameters:
      input_data: "{{ read_csv.content }}"
      input_format: "csv"
      output_format: "json"
      
  - id: save_json
    tool: filesystem
    action: write
    parameters:
      path: "output.json"
      content: "{{ transform.result }}"
```

## Best Practices Demonstrated

1. **Error Handling** - All examples include proper error handling
2. **Resource Management** - Examples show efficient resource usage
3. **Modular Design** - Reusable components and sub-pipelines
4. **Performance** - Parallel execution where appropriate
5. **Documentation** - Clear descriptions and comments

## Contributing Examples

When adding new examples:

1. Use descriptive names that indicate the feature demonstrated
2. Include comprehensive descriptions
3. Add comments explaining non-obvious logic
4. Test the example thoroughly
5. Update this README with a description

## Troubleshooting

If an example fails:

1. Check you have the required API keys set
2. Ensure input files exist (if required)
3. Verify tool dependencies are available
4. Check the logs for detailed error messages

For more help, see the [main documentation](../docs/README.md).