"""Test basic MCP memory operations with real functionality."""

import asyncio
import json
import time
from pathlib import Path
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.orchestrator.tools.mcp_tools import MCPMemoryTool

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider


class TestMCPMemoryBasic:
    """Test basic MCP memory operations without mocks."""
    
    @pytest.mark.asyncio
    async def test_memory_store_retrieve(self):
        """Test basic store and retrieve operations."""
        tool = MCPMemoryTool()
        
        # Store value
        result = await tool.execute(
            action="store",
            namespace="test",
            key="test_key",
            value={"data": "test_value", "nested": {"field": "value"}},
            ttl=60
        )
        assert result["success"] is True
        assert result["namespace"] == "test"
        assert result["key"] == "test_key"
        assert result["stored"] is True
        
        # Retrieve value
        result = await tool.execute(
            action="retrieve",
            namespace="test",
            key="test_key"
        )
        assert result["success"] is True
        assert result["found"] is True
        assert result["value"]["data"] == "test_value"
        assert result["value"]["nested"]["field"] == "value"
    
    @pytest.mark.asyncio
    async def test_memory_ttl_expiration(self):
        """Test TTL expiration with real time delays."""
        tool = MCPMemoryTool()
        
        # Store with 2 second TTL
        result = await tool.execute(
            action="store",
            namespace="test_ttl",
            key="expires_soon",
            value="temporary",
            ttl=2
        )
        assert result["success"] is True
        
        # Should exist immediately
        result = await tool.execute(
            action="retrieve",
            namespace="test_ttl",
            key="expires_soon"
        )
        assert result["found"] is True
        assert result["value"] == "temporary"
        
        # Wait for expiration (add buffer for timing)
        await asyncio.sleep(3)
        
        # Should be expired
        result = await tool.execute(
            action="retrieve",
            namespace="test_ttl",
            key="expires_soon"
        )
        assert result["found"] is False
        # Check that it's marked as expired (the tool returns 'expired': True)
        assert result.get("expired", False) is True or "expired" in str(result).lower()
    
    @pytest.mark.asyncio
    async def test_memory_list_keys(self):
        """Test listing keys in a namespace."""
        tool = MCPMemoryTool()
        
        # Clear namespace first
        await tool.execute(
            action="clear",
            namespace="test_list"
        )
        
        # Store multiple values
        test_data = {
            "key1": "value1",
            "key2": {"nested": "data"},
            "key3": [1, 2, 3]
        }
        
        for key, value in test_data.items():
            result = await tool.execute(
                action="store",
                namespace="test_list",
                key=key,
                value=value,
                ttl=0  # No expiration
            )
            assert result["success"] is True
        
        # List all keys
        result = await tool.execute(
            action="list",
            namespace="test_list"
        )
        assert result["success"] is True
        assert result["namespace"] == "test_list"
        assert set(result["keys"]) == set(["key1", "key2", "key3"])
        assert result["count"] == 3
    
    @pytest.mark.asyncio
    async def test_memory_clear_namespace(self):
        """Test clearing a namespace."""
        tool = MCPMemoryTool()
        
        # Store some values
        await tool.execute(
            action="store",
            namespace="test_clear",
            key="will_be_cleared",
            value="temporary"
        )
        
        # Verify it exists
        result = await tool.execute(
            action="retrieve",
            namespace="test_clear",
            key="will_be_cleared"
        )
        assert result["found"] is True
        
        # Clear namespace
        result = await tool.execute(
            action="clear",
            namespace="test_clear"
        )
        assert result["success"] is True
        assert result["cleared"] == 1
        
        # Verify it's gone
        result = await tool.execute(
            action="retrieve",
            namespace="test_clear",
            key="will_be_cleared"
        )
        assert result["found"] is False
    
    @pytest.mark.asyncio
    async def test_memory_namespace_isolation(self):
        """Test that namespaces are properly isolated."""
        tool = MCPMemoryTool()
        
        # Store in namespace1
        await tool.execute(
            action="store",
            namespace="namespace1",
            key="shared_key",
            value="namespace1_value"
        )
        
        # Store in namespace2
        await tool.execute(
            action="store",
            namespace="namespace2",
            key="shared_key",
            value="namespace2_value"
        )
        
        # Retrieve from namespace1
        result = await tool.execute(
            action="retrieve",
            namespace="namespace1",
            key="shared_key"
        )
        assert result["value"] == "namespace1_value"
        
        # Retrieve from namespace2
        result = await tool.execute(
            action="retrieve",
            namespace="namespace2",
            key="shared_key"
        )
        assert result["value"] == "namespace2_value"
        
        # Clear namespace1
        await tool.execute(
            action="clear",
            namespace="namespace1"
        )
        
        # namespace2 should still have its data
        result = await tool.execute(
            action="retrieve",
            namespace="namespace2",
            key="shared_key"
        )
        assert result["found"] is True
        assert result["value"] == "namespace2_value"
    
    @pytest.mark.asyncio
    async def test_memory_complex_data_types(self):
        """Test storing various complex data types."""
        tool = MCPMemoryTool()
        
        test_cases = [
            ("string", "simple string"),
            ("number", 42),
            ("float", 3.14159),
            ("boolean", True),
            ("null", None),
            ("list", [1, "two", 3.0, True, None]),
            ("dict", {
                "name": "test",
                "nested": {
                    "level": 2,
                    "data": [1, 2, 3]
                }
            }),
            ("unicode", "Hello 世界 🌍")
        ]
        
        for key, value in test_cases:
            # Store
            result = await tool.execute(
                action="store",
                namespace="test_types",
                key=key,
                value=value
            )
            assert result["success"] is True
            
            # Retrieve and verify
            result = await tool.execute(
                action="retrieve",
                namespace="test_types",
                key=key
            )
            assert result["found"] is True
            assert result["value"] == value
    
    @pytest.mark.asyncio
    async def test_memory_invalid_actions(self):
        """Test error handling for invalid actions."""
        tool = MCPMemoryTool()
        
        # Invalid action
        result = await tool.execute(
            action="invalid_action",
            namespace="test"
        )
        assert result["success"] is False
        assert "Invalid action" in result["error"]
        
        # Missing required parameters for store
        result = await tool.execute(
            action="store",
            namespace="test"
            # Missing key and value
        )
        assert result["success"] is False
        assert "key" in result["error"].lower()
        
        # Retrieve non-existent key
        result = await tool.execute(
            action="retrieve",
            namespace="nonexistent",
            key="nonexistent"
        )
        assert result["success"] is True  # Operation succeeds
        assert result["found"] is False   # But key not found