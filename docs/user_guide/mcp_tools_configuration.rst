=====================
MCP Tools Configuration
=====================

The Model Context Protocol (MCP) tools configuration file (``config/mcp_tools_config.json``) defines the available tools that can be invoked through the orchestrator framework. This file specifies the tool schemas, input validation, and capabilities.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

MCP tools provide a standardized way to extend the orchestrator's capabilities by defining external tools that can be called during pipeline execution. Each tool has:

- A unique name
- A description of its functionality
- An input schema defining required and optional parameters
- Version information and capabilities

Configuration Structure
----------------------

The configuration file has the following top-level structure:

.. code-block:: json

   {
     "tools": [...],        // Array of tool definitions
     "version": "1.0.0",    // Configuration version
     "capabilities": {      // Protocol capabilities
       "tools": {
         "listChanged": true
       }
     }
   }

Tool Definition
---------------

Each tool in the ``tools`` array has the following structure:

.. code-block:: json

   {
     "name": "tool-name",
     "description": "What this tool does",
     "inputSchema": {
       "type": "object",
       "properties": {
         // Parameter definitions
       },
       "required": ["param1", "param2"]
     }
   }

Available Tools
--------------

Validation Tool
~~~~~~~~~~~~~~~

Validates data against schemas and business rules.

.. code-block:: yaml

   - action: validate_data
     tool: validation
     parameters:
       data: "{{ results.extracted_data }}"
       schema:
         type: object
         properties:
           email:
             type: string
             format: email
         required: ["email"]
       rules:
         - field: age
           condition: "value >= 18"
           message: "Must be 18 or older"

Filesystem Tool
~~~~~~~~~~~~~~~

Performs file operations like reading, writing, copying, and moving files.

.. code-block:: yaml

   - action: manage_files
     tool: filesystem
     parameters:
       action: read
       path: "/data/input.json"

   - action: save_results
     tool: filesystem
     parameters:
       action: write
       path: "/output/results.json"
       content: "{{ results | json }}"

Headless Browser Tool
~~~~~~~~~~~~~~~~~~~~

Performs web browsing, searching, and scraping tasks.

.. code-block:: yaml

   - action: search_web
     tool: headless-browser
     parameters:
       action: search
       query: "latest AI research papers"
       sources: ["arxiv", "scholar", "pubmed"]

   - action: verify_links
     tool: headless-browser
     parameters:
       action: verify
       url: "https://example.com"

Terminal Tool
~~~~~~~~~~~~~

Executes terminal commands in a sandboxed environment.

.. code-block:: yaml

   - action: run_analysis
     tool: terminal
     parameters:
       command: "python analyze.py --input data.csv"
       working_dir: "/project"
       timeout: 300
       capture_output: true

Web Search Tool
~~~~~~~~~~~~~~~

Performs web searches and returns structured results.

.. code-block:: yaml

   - action: research_topic
     tool: web-search
     parameters:
       query: "quantum computing applications"
       max_results: 20

Data Processing Tool
~~~~~~~~~~~~~~~~~~~

Processes and transforms data in various formats.

.. code-block:: yaml

   - action: transform_data
     tool: data-processing
     parameters:
       action: convert
       data: "{{ raw_data }}"
       format: json
       operation:
         to_format: csv

Custom Tool Configuration
------------------------

To add custom tools, extend the ``mcp_tools_config.json`` file:

.. code-block:: json

   {
     "name": "my-custom-tool",
     "description": "Custom tool for specific task",
     "inputSchema": {
       "type": "object",
       "properties": {
         "action": {
           "type": "string",
           "enum": ["process", "analyze", "generate"],
           "description": "Action to perform"
         },
         "input": {
           "type": "string",
           "description": "Input data or path"
         },
         "options": {
           "type": "object",
           "description": "Additional options",
           "properties": {
             "format": {
               "type": "string",
               "default": "json"
             },
             "verbose": {
               "type": "boolean",
               "default": false
             }
           }
         }
       },
       "required": ["action", "input"]
     }
   }

Complete Example Pipeline
------------------------

Here's a complete example showing how to use multiple MCP tools in a pipeline:

.. code-block:: yaml

   name: web-research-pipeline
   description: Research a topic and generate a report
   
   inputs:
     - name: topic
       type: string
       description: Research topic
   
   steps:
     # Search for information
     - id: search_topic
       action: research
       tool: web-search
       parameters:
         query: "{{ topic }} latest developments 2024"
         max_results: 10
   
     # Verify and scrape top results
     - id: scrape_articles
       for_each: "{{ results.search_topic.results[:3] }}"
       as: article
       action: scrape
       tool: headless-browser
       parameters:
         action: scrape
         url: "{{ article.url }}"
   
     # Save raw data
     - id: save_raw
       action: save
       tool: filesystem
       parameters:
         action: write
         path: "research/{{ topic }}/raw_data.json"
         content: "{{ results.scrape_articles | json }}"
   
     # Process and analyze
     - id: analyze_content
       action: analyze
       tool: data-processing
       parameters:
         action: transform
         data: "{{ results.scrape_articles }}"
         operation:
           extract: ["title", "summary", "key_points"]
           format: structured
   
     # Validate results
     - id: validate_data
       action: validate
       tool: validation
       parameters:
         data: "{{ results.analyze_content }}"
         schema:
           type: array
           items:
             type: object
             required: ["title", "summary"]
   
     # Generate report
     - id: create_report
       action: generate
       parameters:
         template: |
           # Research Report: {{ topic }}
           
           ## Summary
           {{ results.analyze_content | summarize }}
           
           ## Key Findings
           {{ results.analyze_content | format_findings }}
   
     # Save final report
     - id: save_report
       action: save
       tool: filesystem
       parameters:
         action: write
         path: "research/{{ topic }}/report.md"
         content: "{{ results.create_report }}"

Best Practices
--------------

1. **Input Validation**: Always define comprehensive input schemas with proper validation rules
2. **Error Handling**: Include error handling for tool failures
3. **Timeouts**: Set appropriate timeouts for long-running operations
4. **Security**: Be cautious with terminal commands and file system operations
5. **Resource Management**: Consider resource limits when processing large datasets

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~~

**Tool Not Found**

.. code-block:: yaml

   Error: Tool 'custom-tool' not found in MCP configuration

**Solution**: Ensure the tool is defined in ``config/mcp_tools_config.json``

**Invalid Parameters**

.. code-block:: yaml

   Error: Missing required parameter 'action' for tool 'filesystem'

**Solution**: Check the tool's input schema for required parameters

**Permission Denied**

.. code-block:: yaml

   Error: Permission denied accessing '/protected/path'

**Solution**: Ensure proper permissions or use sandboxed paths

Advanced Configuration
---------------------

Environment-Specific Tools
~~~~~~~~~~~~~~~~~~~~~~~~~

You can have different tool configurations for different environments:

.. code-block:: bash

   # Development
   cp config/mcp_tools_config.dev.json config/mcp_tools_config.json
   
   # Production
   cp config/mcp_tools_config.prod.json config/mcp_tools_config.json

Dynamic Tool Loading
~~~~~~~~~~~~~~~~~~~

Tools can be loaded dynamically based on requirements:

.. code-block:: python

   from orchestrator.mcp import MCPToolRegistry
   
   # Load custom tool configuration
   registry = MCPToolRegistry()
   registry.load_config("config/mcp_tools_config.json")
   
   # Add tool at runtime
   registry.register_tool({
       "name": "runtime-tool",
       "description": "Dynamically added tool",
       "inputSchema": {...}
   })

See Also
--------

- :doc:`/user_guide/pipelines` - Learn about pipeline configuration
- :doc:`/api/mcp` - MCP API reference
- :doc:`/tutorials/custom_tools` - Tutorial on creating custom tools