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

## Complete Pipeline Index

### 🔬 Research Pipelines

1. **[research_minimal.yaml](research_minimal.yaml)** - Minimal Research Pipeline
   - Simplest approach - just search and summarize
   - Requires only web-search tool
   - Best for: Quick topic overviews

2. **[research_basic.yaml](research_basic.yaml)** - Basic Research Pipeline
   - Uses standard LLM actions (analyze_text, generate_text)
   - No specialized tools required (beyond web-search)
   - Best for: Detailed research without special tools

3. **[research_advanced_tools.yaml](research_advanced_tools.yaml)** - Research Pipeline with Advanced Tools
   - Uses specialized tools (headless-browser, report-generator, pdf-compiler)
   - Can scrape full web pages and generate PDFs
   - Best for: Professional reports with PDF output

### 🧠 Core AI Features

4. **[auto_tags_demo.yaml](auto_tags_demo.yaml)** - AUTO Tags Demonstration
   - Dynamic parameter resolution
   - Conditional logic with AUTO tags
   - Error handling decisions

5. **[model_routing_demo.yaml](model_routing_demo.yaml)** - Model Routing Demonstration
   - Automatic model selection
   - Cost-optimized routing
   - Fallback strategies

6. **[llm_routing_pipeline.yaml](llm_routing_pipeline.yaml)** - Smart LLM Routing Pipeline
   - Automatically selects the best model
   - Optimizes prompts for tasks
   - Advanced routing logic

### 📊 Data Processing

7. **[data_processing_pipeline.yaml](data_processing_pipeline.yaml)** - Comprehensive Data Processing
   - Multiple format support
   - Filtering and aggregation
   - Validation and transformation

8. **[data_processing.yaml](data_processing.yaml)** - Data Processing Pipeline
   - Process and validate data from various sources
   - Schema validation
   - Data transformation

9. **[simple_data_processing.yaml](simple_data_processing.yaml)** - Simple Data Processing
   - Read CSV files
   - Basic processing
   - Save results

10. **[recursive_data_processing.yaml](recursive_data_processing.yaml)** - Recursive Data Processing
    - Process data recursively
    - Quality threshold checking
    - Iterative improvement

11. **[validation_pipeline.yaml](validation_pipeline.yaml)** - Data Validation Pipeline
    - Validate data against schemas
    - Extract structured information
    - Data quality checks

12. **[test_validation_pipeline.yaml](test_validation_pipeline.yaml)** - Test Validation Pipeline
    - Test pipeline for validating data
    - AUTO tags integration
    - Validation testing

13. **[statistical_analysis.yaml](statistical_analysis.yaml)** - Statistical Analysis Sub-Pipeline
    - Comprehensive statistical analysis
    - Data visualization
    - Statistical reporting

### 🔄 Control Flow Examples

14. **[control_flow_conditional.yaml](control_flow_conditional.yaml)** - Conditional File Processing
    - Process files based on size
    - Conditional branching
    - Dynamic decision making

15. **[control_flow_for_loop.yaml](control_flow_for_loop.yaml)** - Batch File Processing
    - Process multiple files in parallel
    - For-each loops
    - Batch operations

16. **[control_flow_while_loop.yaml](control_flow_while_loop.yaml)** - Iterative Number Guessing
    - Generate numbers until target reached
    - While loop implementation
    - Iterative processing

17. **[control_flow_dynamic.yaml](control_flow_dynamic.yaml)** - Error Handling Pipeline
    - Dynamic flow control
    - Error condition handling
    - Adaptive execution

18. **[control_flow_advanced.yaml](control_flow_advanced.yaml)** - Multi-Stage Text Processing
    - Process text through multiple stages
    - Conditional paths
    - Complex control flow

### 🌐 Web and Search

19. **[web_research_pipeline.yaml](web_research_pipeline.yaml)** - Web Research Automation
    - Web search integration
    - Content analysis
    - Report generation

20. **[working_web_search.yaml](working_web_search.yaml)** - Web Search and Summary
    - Search the web
    - Summarize findings
    - Basic web research

### 🎨 Multimodal and Creative

21. **[multimodal_processing.yaml](multimodal_processing.yaml)** - Multimodal Content Processing
    - Process various media types
    - AI-powered analysis
    - Multi-format support

22. **[creative_image_pipeline.yaml](creative_image_pipeline.yaml)** - Creative Image Generation
    - Generate images from prompts
    - Analyze generated images
    - Creative workflows

### 💻 Code and System

23. **[code_optimization.yaml](code_optimization.yaml)** - Code Optimization Pipeline
    - Analyze code for performance
    - Best practice checking
    - Code improvement suggestions

24. **[terminal_automation.yaml](terminal_automation.yaml)** - System Information and Setup
    - Gather system information
    - Perform setup tasks
    - System automation

### 🔌 Integration and Advanced

25. **[mcp_integration_pipeline.yaml](mcp_integration_pipeline.yaml)** - MCP Integration Pipeline
    - Connect to MCP servers
    - Utilize MCP capabilities
    - Server integration

26. **[mcp_memory_workflow.yaml](mcp_memory_workflow.yaml)** - MCP Memory Context Management
    - Use MCP memory for context
    - Maintain state across steps
    - Memory management

27. **[modular_analysis_pipeline.yaml](modular_analysis_pipeline.yaml)** - Modular Analysis Pipeline
    - Orchestrate multiple sub-pipelines
    - Modular architecture
    - Component composition

28. **[interactive_pipeline.yaml](interactive_pipeline.yaml)** - Interactive Data Processing
    - User input integration
    - Approval gates
    - Feedback collection

## Pipeline Categories Summary

### By Complexity
- **Simple**: research_minimal, simple_data_processing, working_web_search
- **Intermediate**: data_processing, validation_pipeline, web_research_pipeline
- **Advanced**: modular_analysis_pipeline, recursive_data_processing, control_flow_advanced

### By Features
- **AUTO Tags**: auto_tags_demo, test_validation_pipeline, llm_routing_pipeline
- **Control Flow**: All control_flow_*.yaml pipelines
- **Data Processing**: All data_*.yaml and validation pipelines
- **Research**: All research_*.yaml pipelines
- **Integration**: mcp_*.yaml pipelines

### By Output Type
- **Reports**: research pipelines, web_research_pipeline
- **Data Files**: data processing pipelines
- **Analysis**: code_optimization, statistical_analysis
- **Interactive**: interactive_pipeline

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

## Required Tools by Pipeline

Some pipelines require specific tools to be available:

- **Web Search**: research_*.yaml, web_research_pipeline.yaml, working_web_search.yaml
- **Filesystem**: Most pipelines use filesystem for reading/writing
- **Data Processing**: data_*.yaml, validation_*.yaml pipelines
- **MCP Servers**: mcp_*.yaml pipelines
- **Specialized Tools**: 
  - creative_image_pipeline.yaml (image generation)
  - multimodal_processing.yaml (media processing)
  - code_optimization.yaml (code analysis)

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
5. Review checkpoint files in `examples/checkpoints/` for execution details

For more help, see the [main documentation](../docs/README.md).