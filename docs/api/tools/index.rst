Tools API Reference
===================

The Orchestrator framework provides a comprehensive set of tools for extending AI pipeline capabilities with real-world integrations.

.. contents:: Table of Contents
   :local:
   :depth: 2

Tool Categories
---------------

.. toctree::
   :maxdepth: 2

   base
   web_tools
   system_tools
   data_tools
   report_tools
   mcp_server

Overview
--------

The tool system in Orchestrator provides:

* **Standardized Interface**: All tools implement a common base class
* **Parameter Validation**: Automatic validation of inputs and types
* **MCP Integration**: Tools are automatically exposed to AI models
* **Async Support**: Full async/await support for performance
* **Error Handling**: Robust error handling with meaningful messages

Quick Start
-----------

Using tools directly in Python:

.. code-block:: python

   from orchestrator.tools import WebSearchTool, ReportGeneratorTool
   import asyncio
   
   async def example():
       # Search the web
       search_tool = WebSearchTool()
       results = await search_tool.execute(
           query="AI trends 2024",
           max_results=5
       )
       
       # Generate report
       report_tool = ReportGeneratorTool()
       report = await report_tool.execute(
           title="AI Trends Report",
           search_results=results
       )
       
       return report
   
   # Run example
   report = asyncio.run(example())

Using tools in YAML pipelines:

.. code-block:: yaml

   name: Research Pipeline
   steps:
     - id: search
       action: search_web
       parameters:
         query: "{{ inputs.topic }}"
         max_results: 10
     
     - id: report
       action: generate_report
       parameters:
         title: "Report: {{ inputs.topic }}"
         data: "$results.search"

Tool Registry
-------------

All tools are managed through a global registry:

.. code-block:: python

   from orchestrator.tools.base import default_registry
   
   # List available tools
   tools = default_registry.list_tools()
   print("Available tools:", tools)
   
   # Get a specific tool
   search_tool = default_registry.get_tool("web-search")
   
   # Execute a tool
   result = await default_registry.execute_tool(
       "web-search",
       query="python tutorials",
       max_results=5
   )

Custom Tools
------------

Create custom tools by inheriting from the base Tool class:

.. code-block:: python

   from orchestrator.tools.base import Tool
   
   class MyCustomTool(Tool):
       def __init__(self):
           super().__init__(
               name="my-tool",
               description="My custom tool"
           )
           self.add_parameter("input", "string", "Input data")
       
       async def execute(self, **kwargs):
           input_data = kwargs["input"]
           # Process data here
           return {"result": f"Processed: {input_data}"}
   
   # Register the tool
   from orchestrator.tools.base import default_registry
   default_registry.register(MyCustomTool())

See the detailed API documentation for each tool category below.