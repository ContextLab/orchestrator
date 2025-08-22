# Pipeline Examples Documentation

This directory contains comprehensive documentation for all 41 example pipelines in the Orchestrator framework. Each example demonstrates specific features and capabilities, with detailed explanations, usage patterns, and real output examples.

## Documentation Structure

- **Individual Pipeline Documentation**: Each `.md` file corresponds to a pipeline example
- **Output Examples**: Links to actual pipeline outputs in `examples/outputs/`
- **Usage Patterns**: Common patterns and best practices
- **Troubleshooting**: Common issues and solutions

## Complete Pipeline Index

### 🧠 AI & Model Features
- [auto_tags_demo.md](auto_tags_demo.md) - Dynamic AI-driven decisions using AUTO tags
- [model_routing_demo.md](model_routing_demo.md) - Intelligent model selection and routing  
- [llm_routing_pipeline.md](llm_routing_pipeline.md) - Advanced LLM routing with optimization

### 🔬 Research & Analysis
- [research_minimal.md](research_minimal.md) - Basic web research and summarization
- [research_basic.md](research_basic.md) - Standard research with structured analysis
- [research_advanced_tools.md](research_advanced_tools.md) - Professional research with PDF generation
- [web_research_pipeline.md](web_research_pipeline.md) - Comprehensive web research automation
- [working_web_search.md](working_web_search.md) - Simple web search and summary
- [enhanced_research_pipeline.md](enhanced_research_pipeline.md) - Enhanced research with advanced tools

### 📊 Data Processing
- [data_processing.md](data_processing.md) - Basic data processing and validation
- [data_processing_pipeline.md](data_processing_pipeline.md) - Comprehensive data workflows
- [simple_data_processing.md](simple_data_processing.md) - Simple CSV filtering and reporting
- [statistical_analysis.md](statistical_analysis.md) - Statistical analysis and visualization
- [validation_pipeline.md](validation_pipeline.md) - Data validation and quality checks

### 🔄 Control Flow Examples  
- [control_flow_conditional.md](control_flow_conditional.md) - Conditional processing based on data
- [control_flow_for_loop.md](control_flow_for_loop.md) - Parallel batch processing
- [control_flow_while_loop.md](control_flow_while_loop.md) - Iterative processing with conditions
- [control_flow_dynamic.md](control_flow_dynamic.md) - Dynamic flow control and error handling
- [control_flow_advanced.md](control_flow_advanced.md) - Multi-stage complex workflows
- [until_condition_examples.md](until_condition_examples.md) - Until loop demonstrations
- [enhanced_until_conditions_demo.md](enhanced_until_conditions_demo.md) - Advanced until conditions

### 🎨 Creative & Multimodal
- [creative_image_pipeline.md](creative_image_pipeline.md) - AI image generation and processing
- [multimodal_processing.md](multimodal_processing.md) - Multi-format media processing

### 💻 Code & System
- [code_optimization.md](code_optimization.md) - Code analysis and optimization
- [terminal_automation.md](terminal_automation.md) - System automation and information gathering

### 🔌 Integration & Advanced Features
- [mcp_integration_pipeline.md](mcp_integration_pipeline.md) - MCP server integration
- [mcp_memory_workflow.md](mcp_memory_workflow.md) - Memory context management
- [mcp_simple_test.md](mcp_simple_test.md) - Basic MCP functionality testing
- [modular_analysis_pipeline.md](modular_analysis_pipeline.md) - Modular pipeline architecture
- [interactive_pipeline.md](interactive_pipeline.md) - Interactive user workflow

### ✅ Quality & Testing
- [fact_checker.md](fact_checker.md) - Intelligent fact-checking with parallel processing
- [iterative_fact_checker.md](iterative_fact_checker.md) - Iterative fact verification
- [iterative_fact_checker_simple.md](iterative_fact_checker_simple.md) - Simplified iterative fact-checking

### ⚠️ Error Handling & Reliability
- [error_handling_examples.md](error_handling_examples.md) - Comprehensive error handling patterns
- [simple_error_handling.md](simple_error_handling.md) - Basic error handling techniques
- [simple_timeout_test.md](simple_timeout_test.md) - Timeout handling demonstration

### 📁 File & Template Operations
- [file_inclusion_demo.md](file_inclusion_demo.md) - Dynamic file inclusion techniques

### 🧪 Testing & Validation
- [test_simple_pipeline.md](test_simple_pipeline.md) - Simple pipeline testing framework

## Quick Reference Guides

### By Complexity Level
- **Beginner**: research_minimal, simple_data_processing, working_web_search, simple_error_handling
- **Intermediate**: data_processing, validation_pipeline, web_research_pipeline, fact_checker
- **Advanced**: modular_analysis_pipeline, control_flow_advanced, enhanced_research_pipeline

### By Key Features
- **AUTO Tags**: auto_tags_demo, llm_routing_pipeline, fact_checker
- **Parallel Processing**: fact_checker, control_flow_for_loop, iterative_fact_checker
- **Error Handling**: error_handling_examples, simple_error_handling, control_flow_dynamic
- **File Operations**: file_inclusion_demo, simple_data_processing, validation_pipeline
- **Web Integration**: All research_*.md, web_research_pipeline, working_web_search
- **MCP Integration**: mcp_integration_pipeline, mcp_memory_workflow, mcp_simple_test

### By Output Type
- **Reports**: Research pipelines, fact-checkers, code_optimization
- **Data Files**: Data processing pipelines, statistical_analysis  
- **Images**: creative_image_pipeline, multimodal_processing
- **Interactive**: interactive_pipeline, terminal_automation

## Getting Started

1. **Choose a pipeline** based on your use case from the index above
2. **Read the documentation** for detailed explanation and examples
3. **Run the pipeline** using: `python scripts/run_pipeline.py examples/[pipeline_name].yaml`
4. **Examine outputs** in `examples/outputs/[pipeline_name]/`

## Common Usage Patterns

### Running with Parameters
```bash
python scripts/run_pipeline.py examples/research_basic.yaml -i topic="quantum computing" -i depth="detailed"
```

### Specifying Output Directory
```bash
python scripts/run_pipeline.py examples/data_processing.yaml -o examples/outputs/my_analysis/
```

### Testing Mode
```bash
python scripts/run_pipeline.py examples/fact_checker.yaml -i document_source="examples/data/test_article.md"
```

## Best Practices

1. **Start Simple**: Begin with minimal examples and build complexity
2. **Review Outputs**: Always examine generated outputs for quality
3. **Use Real Data**: Test with actual data that matches your use case
4. **Handle Errors**: Review error handling examples for robust pipelines
5. **Monitor Resources**: Check resource usage for long-running pipelines

## Troubleshooting

### Common Issues
- **Missing API Keys**: Ensure required API keys are set in environment
- **File Not Found**: Check input file paths are correct and accessible
- **Tool Dependencies**: Verify all required tools are properly configured
- **Memory Issues**: Use streaming for large data processing

### Getting Help
1. Check the individual pipeline documentation
2. Review error messages in pipeline logs
3. Examine checkpoint files for execution details
4. Consult the main [troubleshooting guide](../advanced/troubleshooting.rst)

## Contributing

When adding new example documentation:
1. Follow the existing template structure
2. Include real working examples
3. Document all parameters and outputs
4. Add links to actual output files
5. Update this index file

---

**Next Steps**: Choose a pipeline from the index above to explore detailed documentation and examples.