Adapters API Reference
======================

This section documents the control system adapters that provide integration with different AI agent frameworks.

.. note::
   For comprehensive guides on using control systems, see the :doc:`../advanced/control_systems` documentation.

.. currentmodule:: orchestrator.adapters

Overview
--------

The Orchestrator supports multiple control systems through a pluggable adapter architecture. Each adapter provides a standardized interface for executing tasks while integrating with different underlying frameworks.

**Key Features:**
- **Unified Interface**: All adapters implement the same core methods
- **Async Support**: Full async/await support for non-blocking operations
- **Error Handling**: Built-in error recovery and retry mechanisms
- **Resource Management**: Automatic resource allocation and cleanup
- **State Management**: Checkpoint and resume capabilities

**Usage Pattern:**

.. code-block:: python

    from orchestrator.adapters.langgraph_adapter import LangGraphAdapter
    
    # Initialize adapter
    adapter = LangGraphAdapter()
    
    # Execute task through adapter
    result = await adapter.execute_task(task, context)

LangGraph Adapter
-----------------

The LangGraph adapter provides seamless integration with LangGraph workflows, enabling complex multi-agent systems to be orchestrated through the Orchestrator framework.

**Key Capabilities:**
- **Node Management**: Create and manage LangGraph nodes with custom functions
- **Edge Configuration**: Define conditional transitions between nodes
- **State Management**: Maintain state across workflow execution
- **Parallel Execution**: Execute multiple nodes concurrently when possible

**Example Usage:**

.. code-block:: python

    from orchestrator.adapters.langgraph_adapter import LangGraphAdapter
    
    # Initialize adapter
    adapter = LangGraphAdapter()
    
    # Define workflow
    workflow = {
        "nodes": [
            {"name": "analyze", "function": analyze_data},
            {"name": "process", "function": process_results}
        ],
        "edges": [
            {"source": "analyze", "target": "process"}
        ]
    }
    
    # Execute workflow
    result = await adapter.execute_workflow(workflow, input_data)

**Classes:**

.. autoclass:: orchestrator.adapters.langgraph_adapter.LangGraphAdapter
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.adapters.langgraph_adapter.LangGraphWorkflow
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.adapters.langgraph_adapter.LangGraphNode
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.adapters.langgraph_adapter.LangGraphEdge
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.adapters.langgraph_adapter.LangGraphState
   :members:
   :undoc-members:
   :show-inheritance:

MCP Adapter
-----------

The Model Context Protocol (MCP) adapter enables integration with MCP servers, providing access to tools, resources, and prompt templates.

**Key Capabilities:**
- **Tool Execution**: Call MCP tools with parameter validation
- **Resource Access**: Retrieve and manipulate MCP resources
- **Prompt Templates**: Use predefined prompt templates
- **Server Management**: Connect to and manage MCP server instances

**Example Usage:**

.. code-block:: python

    from orchestrator.adapters.mcp_adapter import MCPAdapter
    
    # Initialize adapter
    adapter = MCPAdapter()
    
    # Connect to MCP server
    await adapter.connect("stdio", {
        "command": "python",
        "args": ["/path/to/mcp_server.py"]
    })
    
    # List available tools
    tools = await adapter.list_tools()
    
    # Execute a tool
    result = await adapter.call_tool("web_search", {
        "query": "latest AI research",
        "limit": 10
    })

**Classes:**

.. autoclass:: orchestrator.adapters.mcp_adapter.MCPAdapter
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.adapters.mcp_adapter.MCPClient
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.adapters.mcp_adapter.MCPResource
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.adapters.mcp_adapter.MCPTool
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.adapters.mcp_adapter.MCPPrompt
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.adapters.mcp_adapter.MCPMessage
   :members:
   :undoc-members:
   :show-inheritance:

Creating Custom Adapters
-------------------------

You can create custom adapters to integrate with other AI frameworks:

.. code-block:: python

    from orchestrator.core.control_system import ControlSystem, ControlAction
    
    class CustomAdapter(ControlSystem):
        """Custom adapter for your AI framework."""
        
        async def initialize(self) -> None:
            """Initialize the framework connection."""
            # Implementation here
            pass
        
        async def execute_action(self, action: ControlAction) -> Any:
            """Execute an action using your framework."""
            # Implementation here
            pass
        
        async def cleanup(self) -> None:
            """Clean up resources."""
            # Implementation here
            pass

**Adapter Requirements:**
- Inherit from ``ControlSystem`` base class
- Implement ``initialize()``, ``execute_action()``, and ``cleanup()`` methods
- Handle framework-specific error conditions
- Support async operations for non-blocking execution

For detailed implementation examples, see the :doc:`../advanced/control_systems` guide.