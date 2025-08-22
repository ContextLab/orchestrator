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

## Complete Pipeline Index (41 Pipelines)

### 🧠 AI & Model Features
1. **[auto_tags_demo.yaml](auto_tags_demo.yaml)** - AUTO Tags Demonstration
   - Dynamic parameter resolution with AI-driven decisions
   - Conditional logic and model selection
   - **Docs**: [auto_tags_demo.md](../docs/examples/auto_tags_demo.md)

2. **[model_routing_demo.yaml](model_routing_demo.yaml)** - Model Routing Demonstration  
   - Intelligent model selection based on task complexity
   - Cost-optimized routing with fallback strategies
   - Quality vs efficiency tradeoffs

3. **[llm_routing_pipeline.yaml](llm_routing_pipeline.yaml)** - Smart LLM Routing Pipeline
   - Advanced routing logic with prompt optimization
   - Automatic task complexity analysis
   - Performance monitoring and adaptation

### 🔬 Research & Analysis
4. **[research_minimal.yaml](research_minimal.yaml)** - Minimal Research Pipeline
   - Simplest research workflow - search and summarize
   - JSON-structured output with source attribution
   - **Docs**: [research_minimal.md](../docs/examples/research_minimal.md)
   - **Example**: [quantum-computing-basics_summary.md](outputs/research_minimal/quantum-computing-basics_summary.md)

5. **[research_basic.yaml](research_basic.yaml)** - Basic Research Pipeline
   - Standard research with structured analysis
   - Multiple search strategies and content analysis
   - Professional report formatting

6. **[research_advanced_tools.yaml](research_advanced_tools.yaml)** - Research Pipeline with Advanced Tools
   - Uses headless browser, PDF compiler, report generator
   - Professional publication-quality output
   - Advanced scraping and citation handling

7. **[web_research_pipeline.yaml](web_research_pipeline.yaml)** - Web Research Automation
   - Comprehensive web research with validation
   - Multi-source aggregation and cross-referencing
   - Quality scoring and source credibility assessment

8. **[working_web_search.yaml](working_web_search.yaml)** - Web Search and Summary
   - Basic web search with immediate summarization
   - Quick fact-finding and verification
   - Simple output format

9. **[enhanced_research_pipeline.yaml](enhanced_research_pipeline.yaml)** - Enhanced Research Pipeline
   - Advanced research with tool integration
   - Multi-modal analysis and report generation
   - Custom formatting and visualization

### 📊 Data Processing & Analysis
10. **[simple_data_processing.yaml](simple_data_processing.yaml)** - Simple Data Processing
    - CSV filtering and basic transformations
    - **Docs**: [simple_data_processing.md](../docs/examples/simple_data_processing.md)  
    - **Outputs**: [filtered_output.csv](outputs/simple_data_processing/filtered_output.csv), [analysis_report.md](outputs/simple_data_processing/analysis_report.md)

11. **[data_processing.yaml](data_processing.yaml)** - Data Processing Pipeline
    - Multi-format data processing with validation
    - Schema checking and transformation pipelines
    - Error handling and recovery

12. **[data_processing_pipeline.yaml](data_processing_pipeline.yaml)** - Comprehensive Data Processing
    - Advanced ETL operations with multiple sources
    - Data quality assessment and reporting
    - **Outputs**: [processing report](outputs/data_processing_pipeline/data_processing_report.md)

13. **[statistical_analysis.yaml](statistical_analysis.yaml)** - Statistical Analysis Pipeline
    - Statistical computation and visualization
    - Data exploration and hypothesis testing
    - Chart generation and statistical reporting

14. **[validation_pipeline.yaml](validation_pipeline.yaml)** - Data Validation Pipeline
    - Schema validation and data quality checks  
    - Structured data extraction and verification
    - Compliance and format validation

### 🔄 Control Flow & Iteration
15. **[control_flow_conditional.yaml](control_flow_conditional.yaml)** - Conditional File Processing
    - File processing based on size, type, or content
    - Dynamic branching and decision trees
    - **Outputs**: Multiple processed files in [outputs](outputs/control_flow_conditional/)

16. **[control_flow_for_loop.yaml](control_flow_for_loop.yaml)** - Batch File Processing
    - Parallel batch processing with for-each loops
    - Loop variables and dependency management
    - **Docs**: [control_flow_for_loop.md](../docs/examples/control_flow_for_loop.md)
    - **Outputs**: [summary.md](outputs/control_flow_for_loop/summary.md)

17. **[control_flow_while_loop.yaml](control_flow_while_loop.yaml)** - Iterative Processing
    - While loop with condition-based termination
    - State management and iterative refinement
    - **Outputs**: [result.txt](outputs/control_flow_while_loop/result.txt)

18. **[control_flow_dynamic.yaml](control_flow_dynamic.yaml)** - Dynamic Flow Control
    - Runtime decision making and adaptive execution
    - Error recovery and alternative paths
    - Context-sensitive processing

19. **[control_flow_advanced.yaml](control_flow_advanced.yaml)** - Multi-Stage Complex Workflows
    - Multi-language text processing pipeline
    - Complex dependency chains and parallel processing
    - **Outputs**: [Translation results](outputs/control_flow_advanced/)

20. **[until_condition_examples.yaml](until_condition_examples.yaml)** - Until Loop Examples
    - Until condition processing patterns
    - Threshold-based termination
    - Quality-driven iteration

21. **[enhanced_until_conditions_demo.yaml](enhanced_until_conditions_demo.yaml)** - Advanced Until Conditions
    - Complex until conditions with multi-criteria evaluation
    - Dynamic threshold adjustment
    - Performance optimization patterns

### 🎨 Creative & Multimodal
22. **[creative_image_pipeline.yaml](creative_image_pipeline.yaml)** - Creative Image Generation
    - AI image generation with style variations
    - Image analysis and gallery creation
    - **Docs**: [creative_image_pipeline.md](../docs/examples/creative_image_pipeline.md)
    - **Outputs**: [Image galleries](outputs/creative_image_pipeline/)

23. **[multimodal_processing.yaml](multimodal_processing.yaml)** - Multimodal Content Processing
    - Process images, video, audio, and text
    - Cross-modal analysis and integration
    - **Outputs**: [analysis_report.md](outputs/multimodal_processing/analysis_report.md)

### ✅ Quality Assurance & Fact-Checking  
24. **[fact_checker.yaml](fact_checker.yaml)** - Intelligent Fact-Checker
    - Parallel fact-checking with source verification
    - AUTO tags for dynamic list processing
    - **Docs**: [fact_checker.md](../docs/examples/fact_checker.md)
    - **Outputs**: [fact_check_report.md](outputs/fact_checker/fact_check_report.md)

25. **[iterative_fact_checker.yaml](iterative_fact_checker.yaml)** - Iterative Fact Verification
    - Multi-pass fact checking with refinement
    - Progressive verification and evidence gathering
    - **Outputs**: [fact_checking_report.md](outputs/iterative_fact_checker/fact_checking_report.md)

26. **[iterative_fact_checker_simple.yaml](iterative_fact_checker_simple.yaml)** - Simple Iterative Fact-Checker
    - Streamlined iterative fact verification
    - Basic claim validation workflow
    - **Outputs**: [fact_checking_report.md](outputs/iterative_fact_checker_simple/fact_checking_report.md)

### 💻 Code & System Operations
27. **[code_optimization.yaml](code_optimization.yaml)** - Code Optimization Pipeline
    - Multi-language code analysis and improvement
    - Performance optimization suggestions
    - **Outputs**: [Optimization reports and improved code](outputs/code_optimization/)

28. **[terminal_automation.yaml](terminal_automation.yaml)** - System Information and Automation
    - System discovery and automated setup
    - Environment configuration and validation
    - Command execution and monitoring

### 🔌 Integration & Advanced Features
29. **[mcp_integration_pipeline.yaml](mcp_integration_pipeline.yaml)** - MCP Integration Pipeline
    - Model Context Protocol server integration
    - External service orchestration
    - **Outputs**: [Search results](outputs/mcp_integration/)

30. **[mcp_memory_workflow.yaml](mcp_memory_workflow.yaml)** - MCP Memory Context Management
    - Persistent context and state management
    - Cross-session memory and retrieval
    - **Outputs**: [User context summaries](outputs/mcp_memory_workflow/)

31. **[mcp_simple_test.yaml](mcp_simple_test.yaml)** - Basic MCP Testing
    - Simple MCP functionality validation
    - Connection testing and basic operations
    - Integration verification

32. **[modular_analysis_pipeline.yaml](modular_analysis_pipeline.yaml)** - Modular Analysis Pipeline
    - Sub-pipeline orchestration and composition
    - Modular architecture with data visualization  
    - **Outputs**: [Dashboard and charts](outputs/modular_analysis/)

33. **[interactive_pipeline.yaml](interactive_pipeline.yaml)** - Interactive Data Processing
    - User input integration and approval gates
    - Feedback collection and adaptive processing
    - **Outputs**: [Interactive results](outputs/interactive_pipeline/)

### ⚠️ Error Handling & Reliability
34. **[error_handling_examples.yaml](error_handling_examples.yaml)** - Error Handling Examples
    - Comprehensive error handling patterns
    - Recovery strategies and fallback mechanisms
    - Graceful degradation techniques

35. **[simple_error_handling.yaml](simple_error_handling.yaml)** - Basic Error Handling
    - Simple error detection and recovery
    - Basic retry logic and error reporting
    - Foundation error handling patterns

36. **[simple_timeout_test.yaml](simple_timeout_test.yaml)** - Timeout Handling Test
    - Timeout configuration and handling
    - Time-bound operation management
    - Performance monitoring

### 📁 File & Template Operations
37. **[file_inclusion_demo.yaml](file_inclusion_demo.yaml)** - Dynamic File Inclusion
    - Runtime file inclusion and template processing
    - Dynamic content aggregation
    - File-based workflow composition

### 🧪 Testing & Development
38. **[test_simple_pipeline.yaml](test_simple_pipeline.yaml)** - Simple Pipeline Testing
    - Basic pipeline testing framework
    - Validation and verification patterns
    - Development and debugging support

### 📋 Legacy & Backup Examples
39. **[modular_analysis_pipeline_backup.yaml](modular_analysis_pipeline_backup.yaml)** - Modular Analysis Backup
    - Backup version of modular analysis pipeline
    - Alternative implementation approach
    - Development history preservation

40. **[modular_analysis_pipeline_fixed.yaml](modular_analysis_pipeline_fixed.yaml)** - Fixed Modular Analysis
    - Corrected version of modular analysis pipeline
    - Bug fixes and improvements
    - Production-ready implementation

41. **[original_research_report_pipeline.yaml](original_research_report_pipeline.yaml)** - Original Research Report
    - Original research pipeline implementation
    - Baseline research functionality
    - Reference implementation

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

## Example Outputs

Some pipelines include example outputs to demonstrate their functionality:

### Simple Data Processing Pipeline

The [simple_data_processing.yaml](simple_data_processing.yaml) pipeline demonstrates basic CSV data filtering:

- **Input**: A CSV file with 10 project records containing name, status, value, and date
- **Processing**: Filters records to include only those with status="active"
- **Output**: 
  - [filtered_output.csv](outputs/simple_data_processing/filtered_output.csv) - Contains 5 filtered records (only active projects)
  - [analysis_report.md](outputs/simple_data_processing/analysis_report.md) - Markdown report with processing summary and data preview

This example shows:
- How to read files using the filesystem tool
- How to process CSV data with the data-processing tool
- How to use template variables to pass data between pipeline steps
- How to generate both data files and human-readable reports

## Troubleshooting

If an example fails:

1. Check you have the required API keys set
2. Ensure input files exist (if required)
3. Verify tool dependencies are available
4. Check the logs for detailed error messages
5. Review checkpoint files in `examples/checkpoints/` for execution details

For more help, see the [main documentation](../docs/README.md).