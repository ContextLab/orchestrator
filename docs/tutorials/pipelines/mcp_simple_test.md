# Pipeline Tutorial: mcp_simple_test

## Overview

**Complexity Level**: Intermediate  
**Difficulty Score**: 52/100  
**Estimated Runtime**: 10-30 minutes  

### Purpose
This pipeline demonstrates data_flow, json_handling, mcp_integration and provides a practical example of orchestrator's capabilities for intermediate-level workflows.

### Use Cases
- System administration and automation

### Prerequisites
- Basic understanding of YAML syntax
- Familiarity with template variables and data flow
- Understanding of basic control flow concepts
- Understanding of command-line interfaces and system security
- Familiarity with Model Context Protocol (MCP)

### Key Concepts
- Data flow between pipeline steps
- External tool integration
- Template variable substitution

## Pipeline Breakdown

### Configuration Analysis
- **input_section**: Defines the data inputs and parameters for the pipeline
- **steps_section**: Contains the sequence of operations to be executed
- **output_section**: Specifies how results are formatted and stored
- **template_usage**: Uses 4 template patterns for dynamic content
- **feature_highlights**: Demonstrates 5 key orchestrator features

### Data Flow
This pipeline processes input parameters through 5 steps to generate the specified outputs.

### Control Flow
Follows linear execution flow from first step to last step.

### Pipeline Configuration
```yaml
# Simple MCP Test Pipeline
# Tests basic MCP server connection and DuckDuckGo search
id: mcp_simple_test
name: Simple MCP Test
description: Basic test of MCP server with DuckDuckGo search
version: "1.0.0"

parameters:
  search_query:
    type: string
    default: "Python programming"

steps:
  # Connect to DuckDuckGo MCP server
  - id: connect
    tool: mcp-server
    action: execute
    parameters:
      action: "connect"
      server_name: "duckduckgo"
      server_config:
        command: "python"
        args: ["src/orchestrator/tools/mcp_servers/duckduckgo_server.py"]
        env: {}
    
  # List available tools
  - id: list_tools
    tool: mcp-server
    action: execute
    parameters:
      action: "list_tools"
      server_name: "duckduckgo"
    dependencies:
      - connect
    
  # Execute search
  - id: search
    tool: mcp-server
    action: execute
    parameters:
      action: "execute_tool"
      server_name: "duckduckgo"
      tool_name: "search"
      tool_params:
        query: "{{ parameters.search_query }}"
        max_results: 3
    dependencies:
      - list_tools
    
  # Save results with query-based filename
  - id: save_results
    tool: filesystem
    action: write
    parameters:
      path: "examples/outputs/mcp_simple_test/{{ parameters.search_query[:20] | slugify }}_results.json"
      content: "{{ search.result | tojson(indent=2) }}"
    dependencies:
      - search
    
  # Disconnect
  - id: disconnect
    tool: mcp-server
    action: execute
    parameters:
      action: "disconnect"
      server_name: "duckduckgo"
    dependencies:
      - save_results

outputs:
  connected: "{{ connect.connected }}"
  tools_count: "{{ list_tools.tools | length if list_tools.tools else 0 }}"
  search_results: "{{ search.result }}"
```

## Customization Guide

### Input Modifications
- Modify input parameters to match your specific data sources
- Adjust file paths and data formats as needed for your environment

### Parameter Tuning
- Adjust step parameters to customize behavior for your needs

### Step Modifications
- Add new steps by following the same pattern as existing ones
- Remove steps that aren't needed for your specific use case
- Reorder steps if your workflow requires different sequencing
- Replace tool actions with alternatives that provide similar functionality

### Output Customization
- Change output file paths and formats to match your requirements
- Modify output templates to customize the structure and content
- This pipeline produces JSON data - adjust output configuration accordingly

## Remixing Instructions

### Compatible Patterns
- Most basic pipelines can be combined with this pattern

### Extension Ideas
- Add iterative processing for continuous improvement
- Implement parallel processing for better performance
- Include advanced error recovery mechanisms

### Combination Examples
- Can be combined with most other pipeline patterns

### Advanced Variations
- Scale to handle larger datasets and more complex processing
- Add real-time processing capabilities for streaming data
- Implement distributed processing across multiple systems

## Hands-On Exercise

### Execution Instructions
- 1. Navigate to your orchestrator project directory
- 1.5. Ensure you have access to required services: MCP tools
- 2. Run: python scripts/run_pipeline.py examples/mcp_simple_test.yaml
- 3. Monitor the output for progress and any error messages
- 4. Check the output directory for generated results

### Expected Outputs
- Generated JSON data in the specified output directory
- Execution logs showing step-by-step progress
- Completion message with runtime statistics
- No error messages or warnings (successful execution)

### Troubleshooting
- **API Authentication Errors**: Ensure all required API keys are properly configured in your environment
- **Template Resolution Errors**: Check that all input parameters are provided and template syntax is correct
- **General Execution Errors**: Check the logs for specific error messages and verify your orchestrator installation

### Verification Steps
- Check that the pipeline completed without errors
- Verify all expected output files were created
- Review the output content for quality and accuracy

---

*Tutorial generated on 2025-08-27T23:40:24.396317*
