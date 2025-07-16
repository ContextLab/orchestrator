Base Tool Classes
=================

The base tool system provides the foundation for all tools in the Orchestrator framework.

.. automodule:: orchestrator.tools.base
   :members:
   :undoc-members:
   :show-inheritance:

Tool Class
----------

.. autoclass:: orchestrator.tools.base.Tool
   :members:
   :undoc-members:
   :show-inheritance:

The base Tool class that all tools must inherit from. Provides:

* Parameter definition and validation
* MCP schema generation
* Standardized execution interface
* Error handling framework

**Key Methods:**

* ``execute(**kwargs)``: Main execution method (must be implemented)
* ``add_parameter(name, type, description, required, default)``: Define tool parameters
* ``get_schema()``: Get MCP-compatible schema
* ``validate_parameters(kwargs)``: Validate input parameters

**Example:**

.. code-block:: python

   from orchestrator.tools.base import Tool
   
   class ExampleTool(Tool):
       def __init__(self):
           super().__init__(
               name="example-tool",
               description="An example tool"
           )
           
           self.add_parameter(
               name="message",
               type="string",
               description="Message to process",
               required=True
           )
           
           self.add_parameter(
               name="uppercase", 
               type="boolean",
               description="Convert to uppercase",
               required=False,
               default=False
           )
       
       async def execute(self, **kwargs):
           message = kwargs["message"]
           uppercase = kwargs.get("uppercase", False)
           
           result = message.upper() if uppercase else message
           
           return {
               "success": True,
               "result": result,
               "original": message
           }

Tool Parameter
--------------

.. autoclass:: orchestrator.tools.base.ToolParameter
   :members:
   :undoc-members:
   :show-inheritance:

Defines a single parameter for a tool with validation information.

**Attributes:**

* ``name`` (str): Parameter name
* ``type`` (str): Parameter type (string, integer, boolean, object, array)
* ``description`` (str): Human-readable description
* ``required`` (bool): Whether parameter is required
* ``default`` (Any): Default value if not provided

Tool Registry
-------------

.. autoclass:: orchestrator.tools.base.ToolRegistry
   :members:
   :undoc-members:
   :show-inheritance:

Manages registration and discovery of available tools.

**Key Methods:**

* ``register(tool)``: Register a new tool
* ``get_tool(name)``: Get tool by name
* ``list_tools()``: List all registered tool names
* ``execute_tool(name, **kwargs)``: Execute a tool by name
* ``get_schemas()``: Get MCP schemas for all tools

**Global Registry:**

The framework provides a global tool registry that's automatically used by pipelines and MCP servers:

.. code-block:: python

   from orchestrator.tools.base import default_registry
   
   # Register a custom tool
   my_tool = ExampleTool()
   default_registry.register(my_tool)
   
   # List all tools
   tools = default_registry.list_tools()
   
   # Execute a tool
   result = await default_registry.execute_tool(
       "example-tool",
       message="Hello World",
       uppercase=True
   )

Parameter Types
---------------

Tools support the following parameter types:

String Parameters
~~~~~~~~~~~~~~~~~

.. code-block:: python

   self.add_parameter(
       name="text",
       type="string",
       description="Text input",
       required=True
   )

Integer Parameters
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   self.add_parameter(
       name="count",
       type="integer", 
       description="Number of items",
       required=False,
       default=10
   )

Boolean Parameters
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   self.add_parameter(
       name="enabled",
       type="boolean",
       description="Enable feature",
       required=False,
       default=False
   )

Object Parameters
~~~~~~~~~~~~~~~~~

.. code-block:: python

   self.add_parameter(
       name="config",
       type="object",
       description="Configuration object",
       required=False,
       default={}
   )

Array Parameters
~~~~~~~~~~~~~~~~

.. code-block:: python

   self.add_parameter(
       name="items",
       type="array",
       description="List of items",
       required=False,
       default=[]
   )

Error Handling
--------------

Tools should implement proper error handling:

.. code-block:: python

   async def execute(self, **kwargs):
       try:
           # Tool logic here
           result = self.process_data(kwargs)
           
           return {
               "success": True,
               "result": result
           }
           
       except ValueError as e:
           return {
               "success": False,
               "error": "invalid_input",
               "message": str(e)
           }
           
       except ConnectionError as e:
           return {
               "success": False,
               "error": "connection_failed", 
               "message": str(e),
               "retry": True
           }
           
       except Exception as e:
           return {
               "success": False,
               "error": "unexpected_error",
               "message": str(e)
           }

Best Practices
--------------

Tool Implementation
~~~~~~~~~~~~~~~~~~~

* **Inherit from Tool**: Always inherit from the base Tool class
* **Add Parameters**: Define all parameters with clear descriptions
* **Validate Inputs**: Use the built-in validation or add custom validation
* **Handle Errors**: Return structured error information
* **Document Behavior**: Include docstrings and examples

Parameter Design
~~~~~~~~~~~~~~~~

* **Descriptive Names**: Use clear, descriptive parameter names
* **Comprehensive Descriptions**: Include detailed parameter descriptions
* **Appropriate Types**: Use the correct parameter types
* **Sensible Defaults**: Provide reasonable default values
* **Required vs Optional**: Carefully consider which parameters are required

Return Values
~~~~~~~~~~~~~

Tools should return consistent response formats:

.. code-block:: python

   # Success response
   {
       "success": True,
       "result": actual_result,
       "metadata": {
           "execution_time": 1.23,
           "items_processed": 42
       }
   }
   
   # Error response
   {
       "success": False,
       "error": "error_code",
       "message": "Human readable error message",
       "retry": True  # Whether the operation can be retried
   }

Testing Tools
-------------

Test your tools thoroughly:

.. code-block:: python

   import pytest
   from your_tool import ExampleTool
   
   @pytest.fixture
   def tool():
       return ExampleTool()
   
   @pytest.mark.asyncio
   async def test_tool_execution(tool):
       result = await tool.execute(
           message="test",
           uppercase=True
       )
       
       assert result["success"] is True
       assert result["result"] == "TEST"
   
   @pytest.mark.asyncio
   async def test_tool_validation(tool):
       with pytest.raises(ValueError):
           await tool.execute()  # Missing required parameter

For more examples, see :doc:`../../tutorials/examples/custom_tools`.