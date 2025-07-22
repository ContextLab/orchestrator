"""Tests for MCP tools."""

import asyncio
import pytest
import time

from src.orchestrator.tools.mcp_tools import (
    MCPServerTool,
    MCPMemoryTool,
    MCPResourceTool,
    MCPConnection,
    MCPToolInfo,
    MCPResourceInfo
)


@pytest.mark.asyncio
async def test_mcp_server_connect():
    """Test MCP server connection."""
    tool = MCPServerTool()
    
    # Test connection
    result = await tool.execute(
        action="connect",
        server_name="test-server",
        server_config={
            "command": "node",
            "args": ["test-server.js"],
            "env": {"TEST": "true"}
        }
    )
    
    assert result["success"] is True
    assert result["server_name"] == "test-server"
    assert result["connected"] is True
    assert "capabilities" in result
    
    # Verify connection is stored
    assert "test-server" in tool.connections
    assert tool.connections["test-server"].connected is True


@pytest.mark.asyncio
async def test_mcp_server_list_tools():
    """Test listing MCP server tools."""
    tool = MCPServerTool()
    
    # Connect first
    await tool.execute(
        action="connect",
        server_name="test-server",
        server_config={"command": "test"}
    )
    
    # List tools
    result = await tool.execute(
        action="list_tools",
        server_name="test-server"
    )
    
    assert result["success"] is True
    assert "tools" in result
    assert len(result["tools"]) > 0
    
    # Check tool structure
    first_tool = result["tools"][0]
    assert "name" in first_tool
    assert "description" in first_tool
    assert "input_schema" in first_tool


@pytest.mark.asyncio
async def test_mcp_server_execute_tool():
    """Test executing a tool on MCP server."""
    tool = MCPServerTool()
    
    # Connect first
    await tool.execute(
        action="connect",
        server_name="test-server",
        server_config={"command": "test"}
    )
    
    # Execute search tool
    result = await tool.execute(
        action="execute_tool",
        server_name="test-server",
        tool_name="search",
        tool_params={"query": "test query"}
    )
    
    assert result["success"] is True
    assert result["tool_name"] == "search"
    assert "result" in result
    assert "results" in result["result"]


@pytest.mark.asyncio
async def test_mcp_server_disconnect():
    """Test MCP server disconnection."""
    tool = MCPServerTool()
    
    # Connect first
    await tool.execute(
        action="connect",
        server_name="test-server",
        server_config={"command": "test"}
    )
    
    # Disconnect
    result = await tool.execute(
        action="disconnect",
        server_name="test-server"
    )
    
    assert result["success"] is True
    assert result["server_name"] == "test-server"
    assert "test-server" not in tool.connections


@pytest.mark.asyncio
async def test_mcp_server_invalid_action():
    """Test invalid action handling."""
    tool = MCPServerTool()
    
    result = await tool.execute(
        action="invalid_action"
    )
    
    assert result["success"] is False
    assert "Invalid action" in result["error"]


@pytest.mark.asyncio
async def test_mcp_memory_store_retrieve():
    """Test memory store and retrieve operations."""
    tool = MCPMemoryTool()
    
    # Store a value
    result = await tool.execute(
        action="store",
        namespace="test",
        key="user_name",
        value="Alice",
        ttl=0  # Permanent
    )
    
    assert result["success"] is True
    assert result["stored"] is True
    
    # Retrieve the value
    result = await tool.execute(
        action="retrieve",
        namespace="test",
        key="user_name"
    )
    
    assert result["success"] is True
    assert result["found"] is True
    assert result["value"] == "Alice"


@pytest.mark.asyncio
async def test_mcp_memory_ttl():
    """Test memory TTL functionality."""
    tool = MCPMemoryTool()
    
    # Store with short TTL
    result = await tool.execute(
        action="store",
        namespace="test",
        key="temp_value",
        value="temporary",
        ttl=1  # 1 second
    )
    
    assert result["success"] is True
    
    # Retrieve immediately
    result = await tool.execute(
        action="retrieve",
        namespace="test",
        key="temp_value"
    )
    
    assert result["found"] is True
    assert result["value"] == "temporary"
    
    # Wait for expiration
    await asyncio.sleep(1.5)
    
    # Try to retrieve expired value
    result = await tool.execute(
        action="retrieve",
        namespace="test",
        key="temp_value"
    )
    
    assert result["success"] is True
    assert result["found"] is False
    assert result.get("expired") is True


@pytest.mark.asyncio
async def test_mcp_memory_list():
    """Test listing memory keys."""
    tool = MCPMemoryTool()
    
    # Store multiple values
    await tool.execute(
        action="store",
        namespace="test",
        key="key1",
        value="value1"
    )
    
    await tool.execute(
        action="store",
        namespace="test",
        key="key2",
        value="value2"
    )
    
    # List keys
    result = await tool.execute(
        action="list",
        namespace="test"
    )
    
    assert result["success"] is True
    assert "key1" in result["keys"]
    assert "key2" in result["keys"]
    assert result["count"] == 2


@pytest.mark.asyncio
async def test_mcp_memory_clear():
    """Test clearing memory namespace."""
    tool = MCPMemoryTool()
    
    # Store a value
    await tool.execute(
        action="store",
        namespace="test",
        key="data",
        value="test_data"
    )
    
    # Clear namespace
    result = await tool.execute(
        action="clear",
        namespace="test"
    )
    
    assert result["success"] is True
    assert result["cleared"] is True
    
    # Verify it's cleared
    result = await tool.execute(
        action="list",
        namespace="test"
    )
    
    assert result["count"] == 0


@pytest.mark.asyncio
async def test_mcp_memory_namespaces():
    """Test memory namespace isolation."""
    tool = MCPMemoryTool()
    
    # Store in different namespaces
    await tool.execute(
        action="store",
        namespace="ns1",
        key="data",
        value="namespace1"
    )
    
    await tool.execute(
        action="store",
        namespace="ns2",
        key="data",
        value="namespace2"
    )
    
    # Retrieve from each namespace
    result1 = await tool.execute(
        action="retrieve",
        namespace="ns1",
        key="data"
    )
    
    result2 = await tool.execute(
        action="retrieve",
        namespace="ns2",
        key="data"
    )
    
    assert result1["value"] == "namespace1"
    assert result2["value"] == "namespace2"


@pytest.mark.asyncio
async def test_mcp_resource_list():
    """Test listing MCP resources."""
    tool = MCPResourceTool()
    
    result = await tool.execute(
        action="list",
        server_name="test-server"
    )
    
    assert result["success"] is True
    assert "resources" in result
    assert result["count"] > 0
    
    # Check resource structure
    first_resource = result["resources"][0]
    assert "uri" in first_resource
    assert "name" in first_resource
    assert "mime_type" in first_resource


@pytest.mark.asyncio
async def test_mcp_resource_read():
    """Test reading MCP resources."""
    tool = MCPResourceTool()
    
    # Test file resource
    result = await tool.execute(
        action="read",
        server_name="test-server",
        uri="file:///data/config.json"
    )
    
    assert result["success"] is True
    assert "resource" in result
    assert "content" in result["resource"]
    assert "metadata" in result["resource"]


@pytest.mark.asyncio
async def test_mcp_resource_subscribe():
    """Test subscribing to resource updates."""
    tool = MCPResourceTool()
    
    result = await tool.execute(
        action="subscribe",
        server_name="test-server",
        uri="api://weather/current"
    )
    
    assert result["success"] is True
    assert "subscription_id" in result
    assert result["subscription_id"].startswith("sub_")
    
    # Verify subscription is stored
    assert result["subscription_id"] in tool.subscriptions


@pytest.mark.asyncio
async def test_mcp_resource_unsubscribe():
    """Test unsubscribing from resources."""
    tool = MCPResourceTool()
    
    # Subscribe first
    sub_result = await tool.execute(
        action="subscribe",
        server_name="test-server",
        uri="api://weather/current"
    )
    
    subscription_id = sub_result["subscription_id"]
    
    # Unsubscribe
    result = await tool.execute(
        action="unsubscribe",
        subscription_id=subscription_id
    )
    
    assert result["success"] is True
    assert tool.subscriptions[subscription_id]["active"] is False


@pytest.mark.asyncio
async def test_mcp_resource_uri_types():
    """Test different resource URI types."""
    tool = MCPResourceTool()
    
    # Test different URI schemes
    uris = [
        "file:///data/config.json",
        "memory://context/conversation",
        "api://weather/current"
    ]
    
    for uri in uris:
        result = await tool.execute(
            action="read",
            server_name="test-server",
            uri=uri
        )
        
        assert result["success"] is True
        assert "resource" in result
        assert "content" in result["resource"]


@pytest.mark.asyncio
async def test_mcp_connection_dataclass():
    """Test MCPConnection dataclass."""
    conn = MCPConnection(
        server_name="test",
        command="node",
        args=["server.js"],
        env={"KEY": "value"}
    )
    
    assert conn.server_name == "test"
    assert conn.command == "node"
    assert conn.args == ["server.js"]
    assert conn.env == {"KEY": "value"}
    assert conn.connected is False
    assert conn.process is None


@pytest.mark.asyncio
async def test_mcp_tool_info_dataclass():
    """Test MCPToolInfo dataclass."""
    tool_info = MCPToolInfo(
        name="search",
        description="Search the web",
        input_schema={"type": "object"},
        server="test-server"
    )
    
    assert tool_info.name == "search"
    assert tool_info.description == "Search the web"
    assert tool_info.server == "test-server"


@pytest.mark.asyncio
async def test_mcp_resource_info_dataclass():
    """Test MCPResourceInfo dataclass."""
    resource = MCPResourceInfo(
        uri="file:///test.json",
        name="Test File",
        description="A test file",
        mime_type="application/json"
    )
    
    assert resource.uri == "file:///test.json"
    assert resource.name == "Test File"
    assert resource.description == "A test file"
    assert resource.mime_type == "application/json"


if __name__ == "__main__":
    # Run some basic tests
    asyncio.run(test_mcp_server_connect())
    asyncio.run(test_mcp_memory_store_retrieve())
    asyncio.run(test_mcp_resource_list())
    print("Basic MCP tests passed!")