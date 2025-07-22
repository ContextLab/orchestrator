"""Simple demonstration of MCP tools."""

import asyncio
import json

from src.orchestrator.tools.mcp_tools import (
    MCPServerTool,
    MCPMemoryTool,
    MCPResourceTool
)


async def demo_mcp_server():
    """Demonstrate MCP server connections."""
    print("\n=== MCP Server Demo ===")
    
    tool = MCPServerTool()
    
    # Connect to a server
    result = await tool.execute(
        action="connect",
        server_name="demo-server",
        server_config={
            "command": "npx",
            "args": ["-y", "example-mcp-server"],
            "env": {"DEBUG": "true"}
        }
    )
    
    if result["success"]:
        print(f"✓ Connected to server: {result['server_name']}")
        print(f"  Capabilities: {list(result['capabilities'].keys())}")
    
    # List available tools
    result = await tool.execute(
        action="list_tools",
        server_name="demo-server"
    )
    
    if result["success"]:
        print(f"\n✓ Available tools ({len(result['tools'])}):")
        for tool_info in result["tools"]:
            print(f"  - {tool_info['name']}: {tool_info['description']}")
    
    # Execute a tool
    result = await tool.execute(
        action="execute_tool",
        server_name="demo-server",
        tool_name="search",
        tool_params={"query": "orchestrator patterns"}
    )
    
    if result["success"]:
        print(f"\n✓ Tool execution result:")
        print(f"  Results found: {result['result'].get('total', 0)}")
    
    # Disconnect
    await tool.execute(
        action="disconnect",
        server_name="demo-server"
    )
    print("\n✓ Disconnected from server")


async def demo_mcp_memory():
    """Demonstrate MCP memory management."""
    print("\n=== MCP Memory Demo ===")
    
    tool = MCPMemoryTool()
    
    # Store conversation context
    context = {
        "user": "Developer",
        "task": "Build an AI orchestrator",
        "preferences": {
            "language": "Python",
            "framework": "asyncio"
        }
    }
    
    result = await tool.execute(
        action="store",
        namespace="session",
        key="context",
        value=context,
        ttl=3600  # 1 hour
    )
    
    print(f"✓ Stored context with TTL: {result['ttl']}s")
    
    # Store progress
    await tool.execute(
        action="store",
        namespace="session",
        key="progress",
        value=["Connected to MCP", "Listed tools", "Executed search"],
        ttl=3600
    )
    
    # List all keys
    result = await tool.execute(
        action="list",
        namespace="session"
    )
    
    print(f"\n✓ Active memory keys: {', '.join(result['keys'])}")
    
    # Retrieve context
    result = await tool.execute(
        action="retrieve",
        namespace="session",
        key="context"
    )
    
    if result["found"]:
        print(f"\n✓ Retrieved context:")
        print(f"  User: {result['value']['user']}")
        print(f"  Task: {result['value']['task']}")
        print(f"  Language: {result['value']['preferences']['language']}")


async def demo_mcp_resources():
    """Demonstrate MCP resource access."""
    print("\n=== MCP Resources Demo ===")
    
    tool = MCPResourceTool()
    
    # List available resources
    result = await tool.execute(
        action="list",
        server_name="demo-server"
    )
    
    print(f"✓ Available resources ({result['count']}):")
    for resource in result["resources"]:
        print(f"  - {resource['name']} ({resource['uri']})")
        print(f"    Type: {resource['mime_type']}")
    
    # Read a resource
    result = await tool.execute(
        action="read",
        server_name="demo-server",
        uri="memory://context/conversation"
    )
    
    if result["success"]:
        print(f"\n✓ Resource content:")
        print(f"  {json.dumps(result['resource']['content'], indent=2)}")
    
    # Subscribe to updates
    result = await tool.execute(
        action="subscribe",
        server_name="demo-server",
        uri="api://weather/current"
    )
    
    if result["success"]:
        print(f"\n✓ Subscribed to updates")
        print(f"  Subscription ID: {result['subscription_id']}")
        
        # Unsubscribe
        await tool.execute(
            action="unsubscribe",
            subscription_id=result["subscription_id"]
        )
        print("✓ Unsubscribed from updates")


async def main():
    """Run all demos."""
    print("MCP Tools Demonstration")
    print("=" * 50)
    
    await demo_mcp_server()
    await demo_mcp_memory()
    await demo_mcp_resources()
    
    print("\n" + "=" * 50)
    print("Demo completed!")


if __name__ == "__main__":
    asyncio.run(main())