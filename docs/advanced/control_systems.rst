Control Systems
================

The Orchestrator supports multiple control systems through its pluggable adapter architecture. This allows you to integrate with different AI agent frameworks while maintaining a consistent interface.

Available Control Systems
--------------------------

LangGraph Adapter
^^^^^^^^^^^^^^^^^^

The LangGraph adapter provides seamless integration with LangGraph workflows, enabling you to execute complex multi-agent systems through the Orchestrator.

.. code-block:: python

    from orchestrator.adapters.langgraph_adapter import LangGraphAdapter
    from orchestrator.core.pipeline import Pipeline
    
    # Create a LangGraph adapter
    adapter = LangGraphAdapter()
    
    # Define a simple workflow
    workflow = {
        "nodes": [
            {"name": "analyze", "function": analyze_data},
            {"name": "process", "function": process_results},
            {"name": "summarize", "function": create_summary}
        ],
        "edges": [
            {"source": "analyze", "target": "process"},
            {"source": "process", "target": "summarize"}
        ]
    }
    
    # Execute through the adapter
    result = await adapter.execute_workflow(workflow, input_data)

**Key Features:**

- **Node Management**: Create and manage LangGraph nodes with custom functions
- **Edge Configuration**: Define conditional transitions between nodes
- **State Management**: Maintain state across workflow execution
- **Error Handling**: Built-in error recovery and fallback mechanisms

MCP Adapter
^^^^^^^^^^^

The Model Context Protocol (MCP) adapter enables integration with MCP servers, providing access to tools, resources, and prompts.

.. code-block:: python

    from orchestrator.adapters.mcp_adapter import MCPAdapter
    
    # Initialize MCP adapter
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

**Available Operations:**

- **Tool Execution**: Call MCP tools with parameter validation
- **Resource Access**: Retrieve and manipulate MCP resources
- **Prompt Templates**: Use predefined prompt templates
- **Server Management**: Connect to and manage MCP server instances

Creating Custom Adapters
-------------------------

You can create custom adapters to integrate with other AI frameworks or systems.

.. code-block:: python

    from orchestrator.core.control_system import ControlSystem, ControlAction
    from orchestrator.core.pipeline import Pipeline
    from orchestrator.core.task import Task
    
    class CustomAdapter(ControlSystem):
        """Custom adapter for your AI framework."""
        
        def __init__(self, config: dict = None):
            super().__init__(config)
            self.framework_client = None
        
        async def initialize(self) -> None:
            """Initialize the custom framework connection."""
            self.framework_client = YourFrameworkClient()
            await self.framework_client.connect()
        
        async def execute_action(self, action: ControlAction) -> Any:
            """Execute an action using your framework."""
            if action.type == "custom_task":
                return await self.framework_client.execute_task(
                    action.parameters
                )
            else:
                raise ValueError(f"Unsupported action type: {action.type}")
        
        async def cleanup(self) -> None:
            """Clean up resources."""
            if self.framework_client:
                await self.framework_client.disconnect()

**Adapter Requirements:**

- Inherit from ``ControlSystem`` base class
- Implement ``initialize()``, ``execute_action()``, and ``cleanup()`` methods
- Handle framework-specific error conditions
- Support async operations for non-blocking execution

Configuration Management
-------------------------

Control systems can be configured through YAML or programmatically:

.. code-block:: yaml

    # config/control_systems.yaml
    adapters:
      langgraph:
        enabled: true
        max_concurrent_workflows: 5
        timeout: 300
        retry_attempts: 3
      
      mcp:
        enabled: true
        servers:
          - name: "web_tools"
            transport: "stdio"
            command: "python"
            args: ["/path/to/web_server.py"]
          - name: "file_tools"
            transport: "stdio"
            command: "python"
            args: ["/path/to/file_server.py"]

**Configuration Options:**

- **Adapter Settings**: Enable/disable specific adapters
- **Resource Limits**: Set memory, CPU, and timeout constraints
- **Server Configuration**: Configure MCP server connections
- **Retry Policies**: Define retry strategies for failed operations

Error Handling and Recovery
---------------------------

The control system architecture includes comprehensive error handling:

.. code-block:: python

    from orchestrator.core.error_handler import ErrorHandler, RetryStrategy
    
    # Configure error handling
    error_handler = ErrorHandler(
        retry_strategy=RetryStrategy(
            max_attempts=3,
            backoff_factor=2.0,
            max_backoff=60.0
        )
    )
    
    # Adapters use error handler automatically
    adapter = LangGraphAdapter(error_handler=error_handler)

**Error Recovery Features:**

- **Automatic Retry**: Exponential backoff with configurable limits
- **Circuit Breaker**: Prevent cascading failures
- **Fallback Systems**: Switch to alternative adapters on failure
- **State Recovery**: Restore execution state from checkpoints

Performance Optimization
------------------------

Control systems are optimized for high-performance execution:

.. code-block:: python

    from orchestrator.core.resource_allocator import ResourceAllocator
    
    # Configure resource allocation
    allocator = ResourceAllocator(
        max_memory="1GB",
        max_cpu_cores=4,
        max_concurrent_tasks=10
    )
    
    # Adapters respect resource limits
    adapter = LangGraphAdapter(resource_allocator=allocator)

**Performance Features:**

- **Resource Pooling**: Reuse connections and resources
- **Parallel Execution**: Execute multiple workflows concurrently
- **Memory Management**: Automatic cleanup of unused resources
- **Load Balancing**: Distribute work across available resources

Best Practices
---------------

1. **Connection Management**: Always use async context managers for adapter connections
2. **Resource Cleanup**: Implement proper cleanup in custom adapters
3. **Error Handling**: Use structured error handling with appropriate logging
4. **Configuration**: Use environment-specific configuration files
5. **Testing**: Mock external dependencies in unit tests
6. **Monitoring**: Implement health checks and monitoring for production systems

Example Integration
-------------------

Here's a complete example showing how to use multiple control systems:

.. code-block:: python

    import asyncio
    from orchestrator.core.pipeline import Pipeline
    from orchestrator.adapters.langgraph_adapter import LangGraphAdapter
    from orchestrator.adapters.mcp_adapter import MCPAdapter
    
    async def main():
        # Initialize adapters
        langgraph = LangGraphAdapter()
        mcp = MCPAdapter()
        
        # Create pipeline with multiple control systems
        pipeline = Pipeline(
            name="multi_system_pipeline",
            adapters={
                "langgraph": langgraph,
                "mcp": mcp
            }
        )
        
        # Execute pipeline
        result = await pipeline.execute({
            "input_data": "Process this information",
            "workflow_type": "analysis"
        })
        
        print(f"Pipeline result: {result}")
    
    if __name__ == "__main__":
        asyncio.run(main())
