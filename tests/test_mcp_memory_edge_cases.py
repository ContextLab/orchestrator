"""Test edge cases for MCP memory operations."""

import asyncio
import json
import time
from pathlib import Path
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from orchestrator.tools.mcp_tools import MCPMemoryTool


class TestMCPMemoryEdgeCases:
    """Test edge cases and boundary conditions for MCP memory."""
    
    @pytest.mark.asyncio
    async def test_empty_namespace(self):
        """Test listing empty namespace."""
        tool = MCPMemoryTool()
        
        # Clear namespace to ensure it's empty
        await tool.execute(
            action="clear",
            namespace="empty_namespace"
        )
        
        result = await tool.execute(
            action="list",
            namespace="empty_namespace"
        )
        assert result["success"] is True
        assert result["keys"] == []
        assert result["count"] == 0
        assert result["namespace"] == "empty_namespace"
    
    @pytest.mark.asyncio
    async def test_overwrite_key(self):
        """Test overwriting existing key."""
        tool = MCPMemoryTool()
        
        # Store initial value
        result = await tool.execute(
            action="store",
            namespace="test_overwrite",
            key="mutable",
            value="initial"
        )
        assert result["success"] is True
        
        # Verify initial value
        result = await tool.execute(
            action="retrieve",
            namespace="test_overwrite",
            key="mutable"
        )
        assert result["value"] == "initial"
        
        # Overwrite with new value
        result = await tool.execute(
            action="store",
            namespace="test_overwrite",
            key="mutable",
            value="updated"
        )
        assert result["success"] is True
        
        # Verify updated value
        result = await tool.execute(
            action="retrieve",
            namespace="test_overwrite",
            key="mutable"
        )
        assert result["value"] == "updated"
    
    @pytest.mark.asyncio
    async def test_overwrite_with_different_ttl(self):
        """Test overwriting key changes TTL."""
        tool = MCPMemoryTool()
        
        # Store with long TTL
        await tool.execute(
            action="store",
            namespace="test_ttl_overwrite",
            key="ttl_key",
            value="value1",
            ttl=100
        )
        
        # Overwrite with short TTL
        await tool.execute(
            action="store",
            namespace="test_ttl_overwrite",
            key="ttl_key",
            value="value2",
            ttl=2
        )
        
        # Should exist immediately
        result = await tool.execute(
            action="retrieve",
            namespace="test_ttl_overwrite",
            key="ttl_key"
        )
        assert result["found"] is True
        assert result["value"] == "value2"
        
        # Wait for new TTL to expire
        await asyncio.sleep(3)
        
        # Should be expired despite original long TTL
        result = await tool.execute(
            action="retrieve",
            namespace="test_ttl_overwrite",
            key="ttl_key"
        )
        assert result["found"] is False
    
    @pytest.mark.asyncio
    async def test_zero_ttl_permanent_storage(self):
        """Test that zero TTL means permanent storage."""
        tool = MCPMemoryTool()
        
        # Store with zero TTL
        await tool.execute(
            action="store",
            namespace="test_permanent",
            key="permanent_key",
            value="forever",
            ttl=0
        )
        
        # Wait a bit
        await asyncio.sleep(2)
        
        # Should still exist
        result = await tool.execute(
            action="retrieve",
            namespace="test_permanent",
            key="permanent_key"
        )
        assert result["found"] is True
        assert result["value"] == "forever"
    
    @pytest.mark.asyncio
    async def test_large_value_storage(self):
        """Test storing large values."""
        tool = MCPMemoryTool()
        
        # Create a large value (1MB of data)
        large_value = {
            "data": "x" * (1024 * 1024),  # 1MB string
            "array": list(range(10000)),  # Large array
            "nested": {
                f"key_{i}": f"value_{i}" for i in range(1000)
            }
        }
        
        # Store large value
        result = await tool.execute(
            action="store",
            namespace="test_large",
            key="large_key",
            value=large_value
        )
        assert result["success"] is True
        
        # Retrieve and verify
        result = await tool.execute(
            action="retrieve",
            namespace="test_large",
            key="large_key"
        )
        assert result["found"] is True
        assert len(result["value"]["data"]) == 1024 * 1024
        assert len(result["value"]["array"]) == 10000
        assert len(result["value"]["nested"]) == 1000
    
    @pytest.mark.asyncio
    async def test_special_characters_in_keys(self):
        """Test keys with special characters."""
        tool = MCPMemoryTool()
        
        special_keys = [
            "key-with-dash",
            "key_with_underscore",
            "key.with.dots",
            "key:with:colons",
            "key/with/slashes",
            "key with spaces",
            "key@with#special$chars",
            "unicode_key_世界_🌍"
        ]
        
        for key in special_keys:
            # Store
            result = await tool.execute(
                action="store",
                namespace="test_special",
                key=key,
                value=f"value_for_{key}"
            )
            assert result["success"] is True, f"Failed to store key: {key}"
            
            # Retrieve
            result = await tool.execute(
                action="retrieve",
                namespace="test_special",
                key=key
            )
            assert result["found"] is True, f"Failed to retrieve key: {key}"
            assert result["value"] == f"value_for_{key}"
    
    @pytest.mark.asyncio
    async def test_clear_non_existent_namespace(self):
        """Test clearing a namespace that doesn't exist."""
        tool = MCPMemoryTool()
        
        result = await tool.execute(
            action="clear",
            namespace="never_existed"
        )
        assert result["success"] is True
        # When clearing non-existent namespace, cleared might be True or 0
        assert result.get("cleared") in [0, True] or result.get("cleared", 0) == 0
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent memory operations."""
        tool = MCPMemoryTool()
        
        async def store_value(key: str, value: str):
            return await tool.execute(
                action="store",
                namespace="test_concurrent",
                key=key,
                value=value
            )
        
        # Store multiple values concurrently
        tasks = [
            store_value(f"key_{i}", f"value_{i}")
            for i in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        for result in results:
            assert result["success"] is True
        
        # Verify all values stored
        list_result = await tool.execute(
            action="list",
            namespace="test_concurrent"
        )
        assert list_result["count"] == 10
    
    @pytest.mark.asyncio
    async def test_expired_keys_not_listed(self):
        """Test that expired keys are not included in list."""
        tool = MCPMemoryTool()
        
        # Clear namespace first
        await tool.execute(
            action="clear",
            namespace="test_expiry_list"
        )
        
        # Store permanent key
        await tool.execute(
            action="store",
            namespace="test_expiry_list",
            key="permanent",
            value="stays",
            ttl=0
        )
        
        # Store temporary key
        await tool.execute(
            action="store",
            namespace="test_expiry_list",
            key="temporary",
            value="expires",
            ttl=1
        )
        
        # List immediately - should have both
        result = await tool.execute(
            action="list",
            namespace="test_expiry_list"
        )
        assert result["count"] == 2
        
        # Wait for expiration
        await asyncio.sleep(2)
        
        # List again - should only have permanent
        result = await tool.execute(
            action="list",
            namespace="test_expiry_list"
        )
        assert result["count"] == 1
        assert "permanent" in result["keys"]
        assert "temporary" not in result["keys"]
    
    @pytest.mark.asyncio
    async def test_retrieve_with_metadata(self):
        """Test that retrieve returns appropriate metadata."""
        tool = MCPMemoryTool()
        
        # Store with TTL
        await tool.execute(
            action="store",
            namespace="test_metadata",
            key="meta_key",
            value={"data": "test"},
            ttl=60
        )
        
        # Retrieve
        result = await tool.execute(
            action="retrieve",
            namespace="test_metadata",
            key="meta_key"
        )
        
        assert result["success"] is True
        assert result["found"] is True
        assert result["namespace"] == "test_metadata"
        assert result["key"] == "meta_key"
        assert result["value"] == {"data": "test"}
    
    @pytest.mark.asyncio
    async def test_boundary_ttl_values(self):
        """Test boundary values for TTL."""
        tool = MCPMemoryTool()
        
        test_cases = [
            (0, True),      # Zero - permanent
            (1, False),     # Minimum TTL
            (86400, True),  # One day
            (2147483647, True)  # Max 32-bit int
        ]
        
        for ttl, should_exist_after_wait in test_cases:
            key = f"ttl_{ttl}"
            
            # Store with specific TTL
            result = await tool.execute(
                action="store",
                namespace="test_ttl_boundary",
                key=key,
                value=f"ttl_value_{ttl}",
                ttl=ttl
            )
            assert result["success"] is True
            
            # For short TTL, wait and check
            if ttl == 1:
                await asyncio.sleep(2)
                result = await tool.execute(
                    action="retrieve",
                    namespace="test_ttl_boundary",
                    key=key
                )
                assert result["found"] == should_exist_after_wait
            else:
                # Just verify it was stored
                result = await tool.execute(
                    action="retrieve",
                    namespace="test_ttl_boundary",
                    key=key
                )
                assert result["found"] is True