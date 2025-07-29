Real-World Examples
====================

This section provides comprehensive, real-world examples that demonstrate the depth and breadth of the Orchestrator framework. Each example includes complete implementations showing how to solve complex problems using pipelines.

Current Examples
----------------

Core Framework Features
^^^^^^^^^^^^^^^^^^^^^^^

**AUTO Tags Demonstration** (``auto_tags_demo.yaml``)
  Demonstrates the power of AUTO tags for dynamic intelligence in pipelines. Shows how to use AUTO tags for model selection, parameter tuning, and decision making.

**Model Routing Demo** (``model_routing_demo.yaml``)
  Showcases intelligent model routing with cost optimization, performance considerations, and domain-specific selection.

**LLM Routing Pipeline** (``llm_routing_pipeline.yaml``)
  Advanced example of routing between different LLMs based on task requirements, capabilities, and availability.

Control Flow Examples
^^^^^^^^^^^^^^^^^^^^

**Conditional Execution** (``control_flow_conditional.yaml``)
  Basic if/else branching in pipelines based on conditions and AUTO tag evaluations.

**For Loop Processing** (``control_flow_for_loop.yaml``)
  Iterating over collections with for-each loops, including dynamic item processing.

**While Loop Control** (``control_flow_while_loop.yaml``)
  Conditional looping with while constructs and AUTO-resolved termination conditions.

**Dynamic Flow Control** (``control_flow_dynamic.yaml``)
  Advanced flow control with goto-like jumps and dynamic step execution.

**Advanced Control Flow** (``control_flow_advanced.yaml``)
  Combines multiple control flow features in a complex pipeline demonstrating real-world usage.

Data Processing
^^^^^^^^^^^^^^^

**Simple Data Processing** (``simple_data_processing.yaml``)
  Basic data loading, transformation, and analysis pipeline suitable for beginners.

**Data Processing Pipeline** (``data_processing_pipeline.yaml``)
  Comprehensive data processing with validation, transformation, and reporting.

**Recursive Data Processing** (``recursive_data_processing.yaml``)
  Demonstrates pipeline recursion for hierarchical data processing and analysis.

**Modular Analysis Pipeline** (``modular_analysis_pipeline.yaml``)
  Shows how to build modular, reusable pipeline components with sub-pipelines.

Web and Research
^^^^^^^^^^^^^^^^

**Web Research Pipeline** (``web_research_pipeline.yaml``)
  Automated web research with search, content extraction, and summarization.

**Working Web Search** (``working_web_search.yaml``)
  Simple but effective web search implementation with result processing.

**Research Pipeline** (``research_pipeline.yaml``)
  Full research automation including literature review, analysis, and report generation.

Creative and Multimodal
^^^^^^^^^^^^^^^^^^^^^^^

**Creative Image Pipeline** (``creative_image_pipeline.yaml``)
  AI-powered image generation with prompt engineering and style control.

**Multimodal Processing** (``multimodal_processing.yaml``)
  Handles multiple input types (text, images, audio) in a unified pipeline.

Interactive and Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Interactive Pipeline** (``interactive_pipeline.yaml``)
  User interaction with prompts, confirmations, and dynamic flow based on responses.

**Terminal Automation** (``terminal_automation.yaml``)
  Automates terminal commands and system operations with safety checks.

**MCP Integration Pipeline** (``mcp_integration_pipeline.yaml``)
  Demonstrates Model Context Protocol (MCP) tool integration.

**MCP Memory Workflow** (``mcp_memory_workflow.yaml``)
  Advanced MCP usage with persistent memory and context management.

Validation and Testing
^^^^^^^^^^^^^^^^^^^^^^

**Validation Pipeline** (``validation_pipeline.yaml``)
  Data validation with schema enforcement and error reporting.

**Test Validation Pipeline** (``test_validation_pipeline.yaml``)
  Automated testing framework with validation and reporting.

Sub-Pipelines
^^^^^^^^^^^^^

**Statistical Analysis** (``sub_pipelines/statistical_analysis.yaml``)
  Reusable statistical analysis component that can be included in other pipelines.

Pipeline Templates
^^^^^^^^^^^^^^^^^^

**Code Optimization** (``pipelines/code_optimization.yaml``)
  Template for code analysis, optimization, and refactoring workflows.

**Data Processing Template** (``pipelines/data_processing.yaml``)
  Reusable template for data processing workflows.

**Research Report Template** (``pipelines/research-report-template.yaml``)
  Comprehensive template for generating research reports with citations.

**Simple Research** (``pipelines/simple_research.yaml``)
  Minimal research pipeline template for quick investigations.

Running Examples
----------------

To run any example pipeline::

    # Using the orchestrator CLI
    orchestrator run examples/[pipeline-name].yaml

    # With custom inputs
    orchestrator run examples/data_processing_pipeline.yaml --input data_file=mydata.csv

    # Using Python API
    import orchestrator
    
    pipeline = orchestrator.compile("examples/web_research_pipeline.yaml")
    results = await pipeline.run(topic="quantum computing")

Example Structure
-----------------

Each example follows a consistent structure:

1. **Metadata**: Pipeline ID, name, and description
2. **Inputs**: Required and optional parameters
3. **Steps**: Task definitions with actions and parameters
4. **Control Flow**: Conditionals, loops, and dependencies
5. **Outputs**: Result formatting and file generation

Best Practices
--------------

When working with examples:

- Start with simple examples and progress to complex ones
- Modify parameters to understand their effects
- Use examples as templates for your own pipelines
- Check generated outputs in the ``outputs/`` directory
- Enable debug mode for detailed execution logs

Prerequisites
-------------

Before running examples, ensure you have:

- Orchestrator framework installed
- API keys configured for required services
- Python 3.8 or higher
- Required tools installed (e.g., playwright for web automation)

Getting Help
------------

If you encounter issues:

1. Check the example's comments for requirements
2. Verify your API keys are configured
3. Run with ``--debug`` flag for detailed logs
4. See the troubleshooting guide
5. Open an issue on GitHub

.. note::
   All examples use real AI models and tools. Ensure you have appropriate API keys configured before running examples that require external services.

.. tip::
   Examples are designed to work out-of-the-box. If an example requires specific setup, it will be noted in the pipeline's description.