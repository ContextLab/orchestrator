MCP Server Integration
=====================

Model Context Protocol (MCP) server integration for exposing tools to AI models.

.. automodule:: orchestrator.tools.mcp_server
   :members:
   :undoc-members:
   :show-inheritance:

MCPToolServer
-------------

.. autoclass:: orchestrator.tools.mcp_server.MCPToolServer
   :members:
   :undoc-members:
   :show-inheritance:

Exposes Orchestrator tools through the Model Context Protocol, allowing AI models to discover and use tools dynamically.

**Key Features:**

* **Automatic Tool Discovery**: Automatically exposes registered tools
* **Schema Generation**: Generates MCP-compatible schemas
* **Request Routing**: Routes tool calls to appropriate tools
* **Error Handling**: Provides structured error responses
* **Async Support**: Full async/await support for performance

**Example Usage:**

.. code-block:: python

   from orchestrator.tools.mcp_server import MCPToolServer
   from orchestrator.tools.base import default_registry
   import asyncio
   
   async def setup_mcp_server():
       # Create MCP server
       server = MCPToolServer()
       
       # Register default tools
       server.register_default_tools()
       
       # List available tools
       tools = server.list_tools()
       print("Available tools:", tools)
       
       # Get tool schemas
       schemas = server.get_tool_schemas()
       print("Tool schemas:", schemas)
       
       return server
   
   # Setup server
   server = asyncio.run(setup_mcp_server())

Tool Registration
-----------------

Automatic Registration
~~~~~~~~~~~~~~~~~~~~~~

Register all tools from the default registry:

.. code-block:: python

   server = MCPToolServer()
   server.register_default_tools()

Manual Registration
~~~~~~~~~~~~~~~~~~~

Register specific tools:

.. code-block:: python

   from orchestrator.tools import WebSearchTool, ReportGeneratorTool
   
   server = MCPToolServer()
   
   # Register individual tools
   server.register_tool(WebSearchTool())
   server.register_tool(ReportGeneratorTool())

Custom Tool Registration
~~~~~~~~~~~~~~~~~~~~~~~~

Register custom tools:

.. code-block:: python

   from orchestrator.tools.base import Tool
   
   class CustomTool(Tool):
       def __init__(self):
           super().__init__("custom-tool", "My custom tool")
           self.add_parameter("input", "string", "Input data")
       
       async def execute(self, **kwargs):
           return {"result": f"Processed: {kwargs['input']}"}
   
   # Register custom tool
   server = MCPToolServer()
   server.register_tool(CustomTool())

Tool Discovery
--------------

List Available Tools
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get list of tool names
   tools = server.list_tools()
   print("Available tools:", tools)
   # Output: ['web-search', 'headless-browser', 'filesystem', 'terminal', 'data-processing', 'validation', 'report-generator', 'pdf-compiler']

Get Tool Schemas
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get MCP-compatible schemas for all tools
   schemas = server.get_tool_schemas()
   
   for schema in schemas:
       print(f"Tool: {schema['name']}")
       print(f"Description: {schema['description']}")
       print(f"Parameters: {schema['inputSchema']['properties'].keys()}")

Get Specific Tool Schema
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get schema for specific tool
   schema = server.get_tool_schema("web-search")
   print("Web Search Tool Schema:", schema)

Tool Execution
--------------

Execute Tool by Name
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Execute a tool through MCP server
   result = await server.handle_tool_call(
       tool_name="web-search",
       parameters={
           "query": "machine learning tutorials",
           "max_results": 5
       }
   )
   
   if result["success"]:
       print("Search results:", result["result"])
   else:
       print("Error:", result["error"])

Batch Tool Execution
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Execute multiple tools
   tool_calls = [
       {
           "tool": "web-search",
           "parameters": {"query": "AI trends", "max_results": 3}
       },
       {
           "tool": "filesystem", 
           "parameters": {"action": "read", "path": "./data.json"}
       }
   ]
   
   results = []
   for call in tool_calls:
       result = await server.handle_tool_call(
           tool_name=call["tool"],
           parameters=call["parameters"]
       )
       results.append(result)

Error Handling
--------------

Tool Not Found
~~~~~~~~~~~~~~

.. code-block:: python

   result = await server.handle_tool_call(
       tool_name="nonexistent-tool",
       parameters={}
   )
   
   # Returns:
   {
       "success": False,
       "error": "tool_not_found",
       "message": "Tool 'nonexistent-tool' not found",
       "available_tools": ["web-search", "filesystem", ...]
   }

Parameter Validation Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   result = await server.handle_tool_call(
       tool_name="web-search",
       parameters={}  # Missing required 'query' parameter
   )
   
   # Returns:
   {
       "success": False,
       "error": "invalid_parameters",
       "message": "Required parameter 'query' not provided for tool 'web-search'",
       "required_parameters": ["query"],
       "provided_parameters": []
   }

Tool Execution Errors
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   result = await server.handle_tool_call(
       tool_name="filesystem",
       parameters={
           "action": "read",
           "path": "/nonexistent/file.txt"
       }
   )
   
   # Returns:
   {
       "success": False,
       "error": "execution_failed",
       "message": "File not found: /nonexistent/file.txt",
       "tool_error": {
           "type": "file_not_found",
           "details": {...}
       }
   }

MCP Protocol Support
--------------------

Request/Response Format
~~~~~~~~~~~~~~~~~~~~~~~

The server follows MCP protocol specifications:

**Tool List Request:**

.. code-block:: json

   {
       "jsonrpc": "2.0",
       "id": 1,
       "method": "tools/list"
   }

**Tool List Response:**

.. code-block:: json

   {
       "jsonrpc": "2.0",
       "id": 1,
       "result": {
           "tools": [
               {
                   "name": "web-search",
                   "description": "Search the web using DuckDuckGo",
                   "inputSchema": {
                       "type": "object",
                       "properties": {
                           "query": {
                               "type": "string",
                               "description": "Search query"
                           }
                       },
                       "required": ["query"]
                   }
               }
           ]
       }
   }

**Tool Call Request:**

.. code-block:: json

   {
       "jsonrpc": "2.0",
       "id": 2,
       "method": "tools/call",
       "params": {
           "name": "web-search",
           "arguments": {
               "query": "machine learning",
               "max_results": 5
           }
       }
   }

**Tool Call Response:**

.. code-block:: json

   {
       "jsonrpc": "2.0",
       "id": 2,
       "result": {
           "content": [
               {
                   "type": "text",
                   "text": "Search completed successfully"
               }
           ],
           "isError": false
       }
   }

Server Configuration
--------------------

Basic Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   server = MCPToolServer(
       name="orchestrator-tools",
       version="1.0.0",
       description="Orchestrator Framework Tools"
   )

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   server = MCPToolServer(
       name="orchestrator-tools",
       version="1.0.0", 
       description="Orchestrator Framework Tools",
       config={
           "max_concurrent_calls": 10,
           "timeout": 30,
           "rate_limit": 100,  # calls per minute
           "logging_level": "INFO"
       }
   )

Security Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   server = MCPToolServer(
       name="orchestrator-tools",
       security_config={
           "allowed_tools": ["web-search", "filesystem"],
           "blocked_tools": ["terminal"],
           "parameter_filtering": True,
           "output_sanitization": True
       }
   )

Default Tool Detector
---------------------

.. autoclass:: orchestrator.tools.mcp_server.DefaultToolDetector
   :members:
   :undoc-members:
   :show-inheritance:

Automatically detects required tools from YAML pipeline definitions.

**Example Usage:**

.. code-block:: python

   from orchestrator.tools.mcp_server import default_tool_detector
   
   # YAML pipeline content
   pipeline_yaml = {
       "name": "research-pipeline",
       "steps": [
           {
               "id": "search",
               "action": "search_web",
               "parameters": {"query": "AI trends"}
           },
           {
               "id": "save",
               "action": "write_file", 
               "parameters": {"path": "results.txt"}
           }
       ]
   }
   
   # Detect required tools
   required_tools = default_tool_detector.detect_tools_from_yaml(pipeline_yaml)
   print("Required tools:", required_tools)
   # Output: ['web-search', 'filesystem']
   
   # Check tool availability
   availability = default_tool_detector.ensure_tools_available(required_tools)
   print("Tool availability:", availability)

Tool Detection Rules
~~~~~~~~~~~~~~~~~~~~

The detector maps pipeline actions to tools:

.. code-block:: python

   # Action to tool mapping
   action_mappings = {
       "search_web": "web-search",
       "scrape_page": "headless-browser",
       "write_file": "filesystem",
       "read_file": "filesystem",
       "run_command": "terminal",
       "validate_data": "validation",
       "process_data": "data-processing",
       "generate_report": "report-generator",
       "compile_pdf": "pdf-compiler"
   }

Custom Detection Rules
~~~~~~~~~~~~~~~~~~~~~~

Add custom detection rules:

.. code-block:: python

   # Add custom mapping
   default_tool_detector.add_action_mapping("custom_action", "custom-tool")
   
   # Detect tools with custom rules
   required_tools = default_tool_detector.detect_tools_from_yaml(pipeline_yaml)

Integration Examples
--------------------

AI Model Integration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   async def ai_model_with_tools():
       from orchestrator.integrations.openai_model import OpenAIModel
       
       # Setup MCP server
       server = MCPToolServer()
       server.register_default_tools()
       
       # Setup AI model with tool access
       model = OpenAIModel(
           model_name="gpt-4",
           api_key="your-api-key"
       )
       
       # Model can now discover and use tools
       available_tools = server.list_tools()
       
       # Execute model with tool calling capability
       response = await model.generate_with_tools(
           prompt="Search for recent AI developments and create a report",
           tools=server.get_tool_schemas()
       )
       
       return response

Pipeline Integration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   async def pipeline_with_mcp():
       from orchestrator import Orchestrator
       
       # Setup orchestrator with MCP server
       orchestrator = Orchestrator()
       
       # Setup MCP server
       server = MCPToolServer()
       server.register_default_tools()
       
       # Attach server to orchestrator
       orchestrator.mcp_server = server
       
       # Pipeline YAML with tool actions
       pipeline_yaml = """
       name: research-pipeline
       steps:
         - id: search
           action: search_web
           parameters:
             query: "{{ inputs.topic }}"
         
         - id: report
           action: generate_report
           parameters:
             data: "{{ results.search }}"
       """
       
       # Execute pipeline (tools automatically available)
       results = await orchestrator.execute_yaml(
           pipeline_yaml,
           context={"inputs": {"topic": "quantum computing"}}
       )
       
       return results

Best Practices
--------------

Tool Registration
~~~~~~~~~~~~~~~~~

* **Register Early**: Register tools before starting pipeline execution
* **Organize Tools**: Group related tools logically
* **Version Management**: Track tool versions for compatibility
* **Documentation**: Provide clear tool descriptions and examples

.. code-block:: python

   # Organized tool registration
   async def setup_tools():
       server = MCPToolServer()
       
       # Core tools
       server.register_tool(WebSearchTool())
       server.register_tool(HeadlessBrowserTool())
       
       # Data tools
       server.register_tool(DataProcessingTool())
       server.register_tool(ValidationTool())
       
       # System tools
       server.register_tool(FileSystemTool())
       server.register_tool(TerminalTool())
       
       # Report tools
       server.register_tool(ReportGeneratorTool())
       server.register_tool(PDFCompilerTool())
       
       return server

Error Handling
~~~~~~~~~~~~~~

* **Graceful Degradation**: Handle missing tools gracefully
* **Clear Errors**: Provide clear error messages
* **Retry Logic**: Implement retry for transient failures
* **Logging**: Log tool usage and errors for debugging

.. code-block:: python

   async def robust_tool_execution(server, tool_name, parameters):
       try:
           result = await server.handle_tool_call(tool_name, parameters)
           
           if result["success"]:
               return result["result"]
           else:
               # Handle tool-specific errors
               if result["error"] == "tool_not_found":
                   logger.warning(f"Tool {tool_name} not available")
                   return None
               else:
                   logger.error(f"Tool execution failed: {result['message']}")
                   raise ToolExecutionError(result["message"])
       
       except Exception as e:
           logger.error(f"Unexpected error in tool execution: {e}")
           raise

Performance
~~~~~~~~~~~

* **Connection Pooling**: Reuse connections for efficiency
* **Concurrent Execution**: Execute independent tools in parallel
* **Caching**: Cache tool results when appropriate
* **Resource Management**: Monitor and limit resource usage

.. code-block:: python

   async def optimized_tool_usage():
       server = MCPToolServer()
       
       # Execute tools in parallel when possible
       tasks = [
           server.handle_tool_call("web-search", {"query": "topic1"}),
           server.handle_tool_call("web-search", {"query": "topic2"}),
           server.handle_tool_call("filesystem", {"action": "read", "path": "data.json"})
       ]
       
       results = await asyncio.gather(*tasks, return_exceptions=True)
       
       # Process results
       successful_results = [
           result for result in results 
           if not isinstance(result, Exception) and result.get("success")
       ]
       
       return successful_results

For more examples, see :doc:`../../tutorials/examples/mcp_integration`.