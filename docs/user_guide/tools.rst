Tools and Extensions
===================

The Orchestrator framework includes a powerful tool system that extends the capabilities of AI pipelines with real-world integrations. Tools provide standardized interfaces for web scraping, file operations, data processing, system commands, and more.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

Tools in Orchestrator are:

* **Standardized**: All tools implement a common interface for consistency
* **Extensible**: Easy to create custom tools for specific use cases  
* **MCP Compatible**: Integrated with the Model Context Protocol for AI model access
* **Type Safe**: Parameter validation and type checking built-in
* **Async**: Full async/await support for performance

Tool Categories
---------------

Web Tools
~~~~~~~~~

For web scraping, search, and browser automation:

* **WebSearchTool**: Search the web using DuckDuckGo or other engines
* **HeadlessBrowserTool**: Browser automation for content extraction

.. code-block:: python

   from orchestrator.tools.web_tools import WebSearchTool
   
   search_tool = WebSearchTool()
   results = await search_tool.execute(
       query="machine learning best practices",
       max_results=10
   )

System Tools  
~~~~~~~~~~~~

For file operations and system commands:

* **FileSystemTool**: Read, write, and manipulate files
* **TerminalTool**: Execute shell commands safely

.. code-block:: python

   from orchestrator.tools.system_tools import FileSystemTool
   
   fs_tool = FileSystemTool()
   content = await fs_tool.execute(
       action="read",
       path="/path/to/file.txt"
   )

Data Tools
~~~~~~~~~~

For data processing and validation:

* **DataProcessingTool**: Transform and analyze data
* **ValidationTool**: Validate data against schemas and rules

.. code-block:: python

   from orchestrator.tools.data_tools import ValidationTool
   
   validator = ValidationTool()
   result = await validator.execute(
       data={"name": "John", "age": 30},
       rules=[
           {"field": "name", "type": "string", "required": True},
           {"field": "age", "type": "integer", "min": 0}
       ]
   )

Report Tools
~~~~~~~~~~~~

For document generation and compilation:

* **ReportGeneratorTool**: Generate markdown reports from data
* **PDFCompilerTool**: Compile markdown to PDF using pandoc

.. code-block:: python

   from orchestrator.tools.report_tools import ReportGeneratorTool
   
   report_tool = ReportGeneratorTool()
   report = await report_tool.execute(
       title="Research Report",
       data=search_results,
       template="research"
   )

Using Tools in Pipelines
-------------------------

Direct Tool Usage
~~~~~~~~~~~~~~~~~

Tools can be used directly in Python code:

.. code-block:: python

   import asyncio
   from orchestrator.tools import WebSearchTool, ReportGeneratorTool
   
   async def research_pipeline():
       # Search the web
       search_tool = WebSearchTool()
       results = await search_tool.execute(
           query="renewable energy 2024",
           max_results=5
       )
       
       # Generate report
       report_tool = ReportGeneratorTool()
       report = await report_tool.execute(
           title="Renewable Energy Research",
           search_results=results
       )
       
       return report
   
   # Run the pipeline
   report = asyncio.run(research_pipeline())

YAML Pipeline Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

Tools are automatically detected and integrated in YAML pipelines:

.. code-block:: yaml

   name: Research Pipeline
   description: Automated research with web search and reporting
   
   steps:
     - id: search_web
       action: search_web  # Automatically maps to WebSearchTool
       parameters:
         query: "{{ inputs.topic }}"
         max_results: 10
     
     - id: generate_report
       action: generate_report  # Automatically maps to ReportGeneratorTool
       parameters:
         title: "Research Report: {{ inputs.topic }}"
         search_results: "$results.search_web"
         template: "academic"
       dependencies:
         - search_web

MCP Server Integration
~~~~~~~~~~~~~~~~~~~~~~

Tools are automatically exposed through the MCP (Model Context Protocol) server:

.. code-block:: python

   from orchestrator.tools.mcp_server import MCPToolServer
   
   # Tools are automatically registered with MCP server
   server = MCPToolServer()
   server.register_default_tools()
   
   # AI models can now access tools through MCP
   available_tools = server.list_tools()
   print("Available tools:", available_tools)

Creating Custom Tools
----------------------

Tool Class Structure
~~~~~~~~~~~~~~~~~~~~

Create custom tools by inheriting from the base Tool class:

.. code-block:: python

   from orchestrator.tools.base import Tool
   from typing import Dict, Any
   
   class CustomTool(Tool):
       def __init__(self):
           super().__init__(
               name="custom-tool",
               description="A custom tool for specific tasks"
           )
           
           # Define parameters
           self.add_parameter(
               name="input_data",
               type="string", 
               description="Input data to process",
               required=True
           )
           
           self.add_parameter(
               name="options",
               type="object",
               description="Processing options",
               required=False,
               default={}
           )
       
       async def execute(self, **kwargs) -> Dict[str, Any]:
           # Validate parameters (automatic)
           self.validate_parameters(kwargs)
           
           # Extract parameters
           input_data = kwargs["input_data"]
           options = kwargs.get("options", {})
           
           # Implement your logic here
           result = await self.process_data(input_data, options)
           
           return {
               "success": True,
               "result": result,
               "processed_items": len(result) if isinstance(result, list) else 1
           }
       
       async def process_data(self, data: str, options: Dict[str, Any]):
           # Your custom processing logic
           return f"Processed: {data}"

Tool Registration
~~~~~~~~~~~~~~~~~

Register your custom tool with the tool registry:

.. code-block:: python

   from orchestrator.tools.base import default_registry
   
   # Create and register custom tool
   custom_tool = CustomTool()
   default_registry.register(custom_tool)
   
   # Tool is now available in pipelines and MCP server
   available_tools = default_registry.list_tools()
   print("Custom tool registered:", "custom-tool" in available_tools)

Parameter Validation
~~~~~~~~~~~~~~~~~~~~

Tools automatically validate parameters based on their definitions:

.. code-block:: python

   class ValidatedTool(Tool):
       def __init__(self):
           super().__init__("validated-tool", "Tool with validation")
           
           # String parameter with validation
           self.add_parameter(
               name="email",
               type="string",
               description="Valid email address",
               required=True
           )
           
           # Number parameter with constraints
           self.add_parameter(
               name="count",
               type="integer", 
               description="Number of items (1-100)",
               required=False,
               default=10
           )
       
       async def execute(self, **kwargs) -> Dict[str, Any]:
           # Validation happens automatically before this method
           email = kwargs["email"]
           count = kwargs.get("count", 10)
           
           # Add custom validation if needed
           if "@" not in email:
               raise ValueError("Invalid email format")
           
           if not 1 <= count <= 100:
               raise ValueError("Count must be between 1 and 100")
           
           return {"email": email, "count": count}

Error Handling
~~~~~~~~~~~~~~

Implement proper error handling in your tools:

.. code-block:: python

   class RobustTool(Tool):
       async def execute(self, **kwargs) -> Dict[str, Any]:
           try:
               # Attempt the operation
               result = await self.risky_operation(kwargs)
               
               return {
                   "success": True,
                   "result": result
               }
               
           except ConnectionError as e:
               # Network-related errors
               return {
                   "success": False,
                   "error": "connection_failed",
                   "message": str(e),
                   "retry": True
               }
               
           except ValueError as e:
               # Input validation errors
               return {
                   "success": False,
                   "error": "invalid_input",
                   "message": str(e),
                   "retry": False
               }
               
           except Exception as e:
               # Unexpected errors
               return {
                   "success": False,
                   "error": "unexpected_error",
                   "message": str(e),
                   "retry": False
               }

Best Practices
--------------

Tool Design
~~~~~~~~~~~

* **Single Responsibility**: Each tool should have a clear, focused purpose
* **Descriptive Names**: Use clear, descriptive names for tools and parameters
* **Comprehensive Documentation**: Include detailed descriptions for all parameters
* **Error Handling**: Implement robust error handling with meaningful messages

Performance
~~~~~~~~~~~

* **Async Operations**: Use async/await for I/O operations
* **Resource Management**: Clean up resources (files, connections) properly
* **Caching**: Implement caching for expensive operations when appropriate
* **Timeouts**: Set reasonable timeouts for network operations

Security
~~~~~~~~

* **Input Validation**: Validate all inputs thoroughly
* **Safe Operations**: Avoid operations that could harm the system
* **Access Control**: Implement proper access controls for sensitive operations
* **Sanitization**: Sanitize inputs to prevent injection attacks

Testing Tools
-------------

Unit Testing
~~~~~~~~~~~~

Test your tools with pytest:

.. code-block:: python

   import pytest
   from your_tool import CustomTool
   
   @pytest.mark.asyncio
   async def test_custom_tool():
       tool = CustomTool()
       
       result = await tool.execute(
           input_data="test data",
           options={"format": "json"}
       )
       
       assert result["success"] is True
       assert "result" in result

Integration Testing
~~~~~~~~~~~~~~~~~~

Test tools in pipeline context:

.. code-block:: python

   @pytest.mark.asyncio 
   async def test_tool_in_pipeline():
       from orchestrator import Orchestrator
       from orchestrator.tools.base import default_registry
       
       # Register custom tool
       tool = CustomTool()
       default_registry.register(tool)
       
       # Test in pipeline
       orchestrator = Orchestrator()
       results = await orchestrator.execute_yaml(pipeline_yaml)
       
       assert "custom_task" in results

Tool Documentation
------------------

For detailed information about specific tools, see:

* :doc:`../api/tools/web_tools` - Web scraping and search tools
* :doc:`../api/tools/system_tools` - File and system operation tools  
* :doc:`../api/tools/data_tools` - Data processing and validation tools
* :doc:`../api/tools/report_tools` - Report generation and compilation tools

Examples
--------

* :doc:`../tutorials/examples/research_assistant` - Using web and report tools
* :doc:`../tutorials/examples/data_processing_workflow` - Using data tools
* :doc:`../development/custom_tools` - Creating custom tools

For more examples and advanced usage, see the :doc:`../tutorials/index` section.