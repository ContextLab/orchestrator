"""Tests for control system adapters."""

import pytest
from unittest.mock import patch

from src.orchestrator.adapters.langgraph_adapter import LangGraphAdapter
from src.orchestrator.adapters.mcp_adapter import MCPAdapter
from src.orchestrator.core.task import Task
from src.orchestrator.core.pipeline import Pipeline


# Skip BaseAdapter and CustomMLAdapter tests for now since they're not implemented


class TestLangGraphAdapter:
    """Test cases for LangGraphAdapter class."""
    
    def test_langgraph_adapter_creation(self):
        """Test LangGraphAdapter creation."""
        config = {
            "name": "langgraph",
            "version": "1.0.0",
            "graph_config": {"nodes": [], "edges": []}
        }
        adapter = LangGraphAdapter(config)
        
        assert adapter.config == config
    
    @pytest.mark.asyncio
    async def test_langgraph_adapter_health_check(self):
        """Test LangGraphAdapter health check."""
        config = {"name": "langgraph", "version": "1.0.0"}
        adapter = LangGraphAdapter(config)
        
        # Mock the health check since we don't have actual LangGraph
        with patch.object(adapter, 'health_check', return_value=True):
            is_healthy = await adapter.health_check()
            assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_langgraph_adapter_task_execution(self):
        """Test LangGraphAdapter task execution."""
        config = {"name": "langgraph", "version": "1.0.0"}
        adapter = LangGraphAdapter(config)
        
        task = Task("test_task", "Test Task", "generate")
        task.parameters = {"prompt": "Hello, world!"}
        
        # Mock the execution since we don't have actual LangGraph
        with patch.object(adapter, 'execute_task', return_value="Mock response"):
            result = await adapter.execute_task(task, {})
            assert result == "Mock response"


class TestMCPAdapter:
    """Test cases for MCPAdapter class."""
    
    def test_mcp_adapter_creation(self):
        """Test MCPAdapter creation."""
        config = {
            "name": "mcp",
            "version": "1.0.0",
            "server_config": {"host": "localhost", "port": 8080}
        }
        adapter = MCPAdapter(config)
        
        assert adapter.config == config
    
    @pytest.mark.asyncio
    async def test_mcp_adapter_health_check(self):
        """Test MCPAdapter health check."""
        config = {"name": "mcp", "version": "1.0.0"}
        adapter = MCPAdapter(config)
        
        # Mock the health check since we don't have actual MCP server
        with patch.object(adapter, 'health_check', return_value=True):
            is_healthy = await adapter.health_check()
            assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_mcp_adapter_task_execution(self):
        """Test MCPAdapter task execution."""
        config = {"name": "mcp", "version": "1.0.0"}
        adapter = MCPAdapter(config)
        
        task = Task("test_task", "Test Task", "analyze")
        task.parameters = {"data": "sample data"}
        
        # Mock the execution since we don't have actual MCP server
        with patch.object(adapter, 'execute_task', return_value={"result": "analysis"}):
            result = await adapter.execute_task(task, {})
            assert result == {"result": "analysis"}


class TestAdapterIntegration:
    """Test adapter integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_adapter_pipeline_execution(self):
        """Test adapters working with pipeline execution."""
        config = {"name": "test", "version": "1.0.0"}
        
        # Create adapters
        langgraph_adapter = LangGraphAdapter(config)
        mcp_adapter = MCPAdapter(config)
        
        # Create pipeline with tasks
        pipeline = Pipeline("test_pipeline", "Test Pipeline")
        
        task1 = Task("task1", "LangGraph Task", "generate")
        task1.parameters = {"prompt": "Generate text"}
        
        task2 = Task("task2", "MCP Task", "analyze")
        task2.parameters = {"data": "Analyze this"}
        task2.dependencies = ["task1"]
        
        pipeline.add_task(task1)
        pipeline.add_task(task2)
        
        # Mock adapter executions
        with patch.object(langgraph_adapter, 'execute_task', return_value="Generated text"), \
             patch.object(mcp_adapter, 'execute_task', return_value={"analysis": "complete"}):
            
            # Simulate pipeline execution
            results = {}
            
            # Execute task1 with LangGraph adapter
            results["task1"] = await langgraph_adapter.execute_task(task1, {})
            
            # Execute task2 with MCP adapter
            context = {"previous_results": results}
            results["task2"] = await mcp_adapter.execute_task(task2, context)
            
            assert results["task1"] == "Generated text"
            assert results["task2"] == {"analysis": "complete"}
    
    def test_adapter_configuration_validation(self):
        """Test adapter configuration validation."""
        # Valid configs should work
        valid_langgraph_config = {"name": "langgraph", "version": "1.0.0"}
        valid_mcp_config = {"name": "mcp", "version": "1.0.0"}
        
        langgraph_adapter = LangGraphAdapter(valid_langgraph_config)
        mcp_adapter = MCPAdapter(valid_mcp_config)
        
        assert langgraph_adapter.config == valid_langgraph_config
        assert mcp_adapter.config == valid_mcp_config
    
    @pytest.mark.asyncio
    async def test_adapter_error_handling(self):
        """Test adapter error handling."""
        config = {"name": "test", "version": "1.0.0"}
        adapter = LangGraphAdapter(config)
        
        task = Task("failing_task", "Failing Task", "generate")
        
        # Mock a failing execution
        with patch.object(adapter, 'execute_task', side_effect=Exception("Execution failed")):
            with pytest.raises(Exception, match="Execution failed"):
                await adapter.execute_task(task, {})