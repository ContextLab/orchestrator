"""Real integration tests for MCP tools with actual server processes.

NO MOCKS OR SIMULATIONS - All tests use real server processes and real data.
"""

import asyncio
import json
import os
import pytest
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider
from src.orchestrator.tools.mcp_tools import (

    MCPServerTool,
    MCPMemoryTool,
    MCPResourceTool
)
from src.orchestrator import Orchestrator, init_models


class TestRealMCPServerConnection:
    """Test real MCP server connections with actual subprocess communication."""
    
    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        cls.server_script = Path(__file__).parent.parent / "src/orchestrator/tools/mcp_servers/duckduckgo_server.py"
        assert cls.server_script.exists(), f"DuckDuckGo server script not found at {cls.server_script}"
    
    @pytest.mark.asyncio
    async def test_real_server_connection(self):
        """Test connecting to a real MCP server process."""
        tool = MCPServerTool()
        
        # Connect to real DuckDuckGo server
        result = await tool.execute(
            action="connect",
            server_name="duckduckgo-test",
            server_config={
                "command": sys.executable,
                "args": [str(self.server_script)]
            }
        )
        
        # Verify real connection
        assert result["success"] is True
        assert result["connected"] is True
        assert "capabilities" in result
        
        # Clean up
        await tool.execute(action="disconnect", server_name="duckduckgo-test")
    
    @pytest.mark.asyncio
    async def test_real_duckduckgo_search(self):
        """Test real DuckDuckGo search through MCP server."""
        tool = MCPServerTool()
        
        # Connect to server
        connect_result = await tool.execute(
            action="connect",
            server_name="duckduckgo-real",
            server_config={
                "command": sys.executable,
                "args": [str(self.server_script)]
            }
        )
        assert connect_result["connected"] is True
        
        # List available tools from real server
        list_result = await tool.execute(
            action="list_tools",
            server_name="duckduckgo-real"
        )
        assert list_result["success"] is True
        assert len(list_result["tools"]) > 0
        
        # Find search tool
        search_tool = None
        for t in list_result["tools"]:
            if t["name"] == "search":
                search_tool = t
                break
        assert search_tool is not None, "Search tool not found in server tools"
        
        # Execute real web search
        search_result = await tool.execute(
            action="execute_tool",
            server_name="duckduckgo-real",
            tool_name="search",
            tool_params={
                "query": "Python programming language",
                "max_results": 3
            }
        )
        
        # Verify real search results
        assert search_result["success"] is True
        assert "result" in search_result
        results = search_result["result"].get("results", [])
        assert len(results) > 0, "No search results returned"
        
        # Verify result structure
        first_result = results[0]
        assert "url" in first_result
        assert first_result["url"].startswith("http"), "Result should have real URL"
        assert "title" in first_result
        assert "snippet" in first_result
        
        # Clean up
        await tool.execute(action="disconnect", server_name="duckduckgo-real")
    
    @pytest.mark.asyncio
    async def test_server_error_handling(self):
        """Test error handling with real server failures."""
        tool = MCPServerTool()
        
        # Try to connect with invalid command
        result = await tool.execute(
            action="connect",
            server_name="invalid-server",
            server_config={
                "command": "nonexistent_command_12345",
                "args": []
            }
        )
        
        assert result["success"] is False
        assert result["connected"] is False
    
    @pytest.mark.asyncio
    async def test_concurrent_server_connections(self):
        """Test multiple concurrent server connections."""
        tool = MCPServerTool()
        
        # Start multiple servers concurrently
        servers = ["server1", "server2", "server3"]
        connect_tasks = []
        
        for server_name in servers:
            task = tool.execute(
                action="connect",
                server_name=server_name,
                server_config={
                    "command": sys.executable,
                    "args": [str(self.server_script)]
                }
            )
            connect_tasks.append(task)
        
        # Wait for all connections
        results = await asyncio.gather(*connect_tasks)
        
        # Verify all connected
        for i, result in enumerate(results):
            assert result["connected"] is True, f"Server {servers[i]} failed to connect"
        
        # Execute searches on all servers concurrently
        search_tasks = []
        for server_name in servers:
            task = tool.execute(
                action="execute_tool",
                server_name=server_name,
                tool_name="search",
                tool_params={
                    "query": f"test query {server_name}",
                    "max_results": 2
                }
            )
            search_tasks.append(task)
        
        search_results = await asyncio.gather(*search_tasks)
        
        # Verify all searches succeeded
        for result in search_results:
            assert result["success"] is True
            assert len(result["result"]["results"]) > 0
        
        # Disconnect all servers
        for server_name in servers:
            await tool.execute(action="disconnect", server_name=server_name)


class TestRealMCPMemory:
    """Test real memory storage and persistence."""
    
    @pytest.mark.asyncio
    async def test_memory_persistence(self):
        """Test that memory actually persists data."""
        tool = MCPMemoryTool()
        
        # Store real data with timestamp
        test_data = {
            "timestamp": time.time(),
            "content": "Real test data",
            "list": [1, 2, 3],
            "nested": {"key": "value"}
        }
        
        store_result = await tool.execute(
            action="store",
            namespace="test_namespace",
            key="test_key",
            value=test_data,
            ttl=60  # 60 seconds TTL
        )
        assert store_result["success"] is True
        
        # Retrieve and verify exact match
        retrieve_result = await tool.execute(
            action="retrieve",
            namespace="test_namespace",
            key="test_key"
        )
        assert retrieve_result["success"] is True
        assert retrieve_result["found"] is True
        assert retrieve_result["value"] == test_data
        
        # List keys in namespace
        list_result = await tool.execute(
            action="list",
            namespace="test_namespace"
        )
        assert list_result["success"] is True
        assert "test_key" in list_result["keys"]
    
    @pytest.mark.asyncio
    async def test_memory_ttl_expiration(self):
        """Test that TTL actually expires data."""
        tool = MCPMemoryTool()
        
        # Store with very short TTL
        await tool.execute(
            action="store",
            namespace="ttl_test",
            key="expire_soon",
            value="temporary data",
            ttl=1  # 1 second
        )
        
        # Verify it exists immediately
        result1 = await tool.execute(
            action="retrieve",
            namespace="ttl_test",
            key="expire_soon"
        )
        assert result1["found"] is True
        
        # Wait for expiration
        await asyncio.sleep(1.5)
        
        # Verify it's expired
        result2 = await tool.execute(
            action="retrieve",
            namespace="ttl_test",
            key="expire_soon"
        )
        assert result2["found"] is False
        assert result2.get("expired") is True


class TestRealPipelineExecution:
    """Test complete pipeline execution with real servers and data."""
    
    @pytest.mark.asyncio
    async def test_full_mcp_pipeline(self):
        """Test the complete MCP integration pipeline end-to-end."""
        # Create output directory
        output_dir = Path(tempfile.mkdtemp(prefix="mcp_test_"))
        
        try:
            # Initialize orchestrator
            model_registry = init_models()
            orchestrator = create_test_orchestrator()
            
            # Load pipeline
            pipeline_path = Path(__file__).parent.parent / "examples/mcp_integration_pipeline.yaml"
            assert pipeline_path.exists()
            
            yaml_content = pipeline_path.read_text()
            
            # Run pipeline with real search query
            results = await orchestrator.execute_yaml(
                yaml_content,
                {
                    "search_query": "artificial intelligence latest news",
                    "output_path": str(output_dir)
                }
            )
            
            # Verify pipeline executed
            assert results is not None
            
            # Check for output files
            output_files = list(output_dir.glob("*.md")) + list(output_dir.glob("*.json"))
            assert len(output_files) > 0, "No output files generated"
            
            # Verify content has real data
            for file in output_files:
                content = file.read_text()
                # Should not contain placeholder text
                assert "{{" not in content, f"File {file} contains unrendered templates"
                # Should contain some content
                assert len(content) > 100, f"File {file} is too small"
                
                # If it's a report, verify it has real URLs
                if "report" in file.name and file.suffix == ".md":
                    assert "http" in content.lower(), "Report should contain real URLs"
            
        finally:
            # Clean up
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)
    
    @pytest.mark.asyncio 
    async def test_search_with_various_queries(self):
        """Test searching with different types of queries."""
        tool = MCPServerTool()
        
        # Connect once
        await tool.execute(
            action="connect",
            server_name="search-test",
            server_config={
                "command": sys.executable,
                "args": [str(Path(__file__).parent.parent / "src/orchestrator/tools/mcp_servers/duckduckgo_server.py")]
            }
        )
        
        queries = [
            "machine learning algorithms",
            "climate change 2024",
            "Python vs JavaScript",
            "quantum computing breakthroughs"
        ]
        
        for query in queries:
            result = await tool.execute(
                action="execute_tool",
                server_name="search-test",
                tool_name="search",
                tool_params={"query": query, "max_results": 2}
            )
            
            assert result["success"] is True
            results = result["result"]["results"]
            assert len(results) > 0, f"No results for query: {query}"
            
            # Verify each result has real content
            for r in results:
                assert r["url"].startswith("http")
                assert len(r["title"]) > 0
                assert len(r["snippet"]) > 0
        
        # Disconnect
        await tool.execute(action="disconnect", server_name="search-test")


if __name__ == "__main__":
    # Run tests with real output
    pytest.main([__file__, "-v", "-s"])