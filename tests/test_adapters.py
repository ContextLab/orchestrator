"""Tests for control system adapters."""

import pytest

from src.orchestrator.adapters.langgraph_adapter import LangGraphAdapter, LangGraphWorkflow, LangGraphNode
from src.orchestrator.adapters.mcp_adapter import MCPAdapter
from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.task import Task
from src.orchestrator.models.model_registry import ModelRegistry

# Skip BaseAdapter and CustomMLAdapter tests for now since they're not implemented


class TestLangGraphAdapter:
    """Test cases for LangGraphAdapter class."""

    def test_langgraph_adapter_creation(self):
        """Test LangGraphAdapter creation."""
        config = {
            "name": "langgraph",
            "version": "1.0.0",
            "graph_config": {"nodes": [], "edges": []},
        }
        adapter = LangGraphAdapter(config)

        assert adapter.config == config
        assert adapter.model_registry is not None
        assert adapter.execution_control is not None

    @pytest.mark.asyncio
    async def test_langgraph_adapter_health_check(self):
        """Test LangGraphAdapter health check with real model availability."""
        config = {"name": "langgraph", "version": "1.0.0"}
        adapter = LangGraphAdapter(config)

        # Real health check - depends on model availability
        is_healthy = await adapter.health_check()
        
        # Should be healthy if we have any models available
        available_models = await adapter.model_registry.get_available_models()
        expected_healthy = len(available_models) > 0
        assert is_healthy == expected_healthy

    @pytest.mark.asyncio
    async def test_langgraph_adapter_task_execution(self, populated_model_registry):
        """Test LangGraphAdapter task execution with real AI models."""
        # Skip if no AI models available
        registry = populated_model_registry
        available_models = await registry.get_available_models()
        if not available_models:
            raise AssertionError(
                "No AI models available for testing. "
                "Please configure API keys in ~/.orchestrator/.env"
            )
        
        config = {"name": "langgraph", "version": "1.0.0"}
        adapter = LangGraphAdapter(config, model_registry=registry)

        task = Task("test_task", "Test Task", "generate")
        task.parameters = {"prompt": "Say 'Hello, LangGraph!' in exactly 3 words"}

        # Real execution using AI models
        result = await adapter.execute_task(task, {})
        
        # Verify we got actual AI-generated response
        assert isinstance(result, str)
        assert len(result) > 5  # Should have actual content
        # Result should relate to the prompt
        assert any(word in result.lower() for word in ["hello", "langgraph", "three"])


class TestMCPAdapter:
    """Test cases for MCPAdapter class."""

    def test_mcp_adapter_creation(self):
        """Test MCPAdapter creation."""
        config = {
            "name": "mcp",
            "version": "1.0.0",
            "server_config": {"host": "localhost", "port": 8080},
        }
        adapter = MCPAdapter(config)

        assert adapter.config == config
        assert adapter.model_registry is not None
        assert adapter.ai_control is not None

    @pytest.mark.asyncio
    async def test_mcp_adapter_health_check(self):
        """Test MCPAdapter health check with real AI availability."""
        config = {"name": "mcp", "version": "1.0.0"}
        adapter = MCPAdapter(config)

        # Real health check - can work with either MCP or AI
        is_healthy = await adapter.health_check()
        
        # Should be healthy if AI control system is healthy
        ai_healthy = await adapter.ai_control.health_check()
        assert is_healthy == (len(adapter.clients) > 0 or ai_healthy)

    @pytest.mark.asyncio
    async def test_mcp_adapter_task_execution(self, populated_model_registry):
        """Test MCPAdapter task execution with real AI models."""
        # Skip if no AI models available
        registry = populated_model_registry
        available_models = await registry.get_available_models()
        if not available_models:
            raise AssertionError(
                "No AI models available for testing. "
                "Please configure API keys in ~/.orchestrator/.env"
            )
            
        config = {"name": "mcp", "version": "1.0.0"}
        adapter = MCPAdapter(config, model_registry=registry)

        task = Task("test_task", "Test Task", "analyze")
        task.parameters = {"prompt": "Analyze the word 'hello' and list 3 characteristics"}

        # Real execution using AI models (no MCP server needed)
        result = await adapter.execute_task(task, {})
        
        # Verify we got actual AI-generated analysis
        assert isinstance(result, str)
        assert len(result) > 20  # Should have meaningful analysis
        # Result should relate to analysis
        assert any(word in result.lower() for word in ["hello", "word", "letter", "greeting", "characteristic"])


class TestAdapterIntegration:
    """Test adapter integration scenarios."""

    @pytest.mark.asyncio
    async def test_adapter_pipeline_execution(self, populated_model_registry):
        """Test adapters working with pipeline execution using real AI models."""
        # Skip if no AI models available
        registry = populated_model_registry
        available_models = await registry.get_available_models()
        if not available_models:
            raise AssertionError(
                "No AI models available for testing. "
                "Please configure API keys in ~/.orchestrator/.env"
            )
            
        config = {"name": "test", "version": "1.0.0"}

        # Create adapters with real AI capabilities
        langgraph_adapter = LangGraphAdapter(config)
        mcp_adapter = MCPAdapter(config)

        # Create pipeline with tasks
        pipeline = Pipeline("test_pipeline", "Test Pipeline")

        task1 = Task("task1", "LangGraph Task", "generate")
        task1.parameters = {"prompt": "Generate a short sentence about AI"}

        task2 = Task("task2", "MCP Task", "analyze")
        task2.parameters = {"prompt": "Analyze the previous text and count the number of words"}
        task2.dependencies = ["task1"]

        pipeline.add_task(task1)
        pipeline.add_task(task2)

        # Real pipeline execution
        results = {}

        # Execute task1 with LangGraph adapter
        results["task1"] = await langgraph_adapter.execute_task(task1, {})

        # Execute task2 with MCP adapter using previous results
        context = {"previous_results": results}
        results["task2"] = await mcp_adapter.execute_task(task2, context)

        # Verify real AI execution
        assert isinstance(results["task1"], str)
        assert len(results["task1"]) > 10  # Should have actual generated text
        assert isinstance(results["task2"], str)
        assert len(results["task2"]) > 10  # Should have actual analysis
        # The analysis should mention words or counting
        assert any(word in results["task2"].lower() for word in ["word", "count", "number", "text"])

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
        """Test adapter error handling with real execution failures."""
        config = {"name": "test", "version": "1.0.0"}
        adapter = LangGraphAdapter(config)

        # Create a task that will fail due to missing required parameters
        task = Task("failing_task", "Failing Task", "generate")
        # No parameters provided - should cause an error
        
        # Real execution should handle errors gracefully
        try:
            result = await adapter.execute_task(task, {})
            # If no error, verify we got some default handling
            assert isinstance(result, str)
        except Exception as e:
            # Should get a meaningful error about missing parameters
            assert "Failed to execute task" in str(e) or "parameters" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_langgraph_workflow_execution(self, populated_model_registry):
        """Test LangGraph workflow functionality with real AI."""
        # Skip if no AI models available
        registry = populated_model_registry
        available_models = await registry.get_available_models()
        if not available_models:
            raise AssertionError(
                "No AI models available for testing. "
                "Please configure API keys in ~/.orchestrator/.env"
            )
            
        adapter = LangGraphAdapter()
        
        # Create a simple workflow
        workflow = LangGraphWorkflow("test_workflow")
        
        # Define a simple node function that uses the adapter
        async def analyze_node(state, **kwargs):
            task = Task("analyze", "Analyze", "analyze")
            task.parameters = {"prompt": f"Analyze this number: {kwargs.get('number', 42)}"}
            result = await adapter._execute_task(task, state.data)
            return {"analysis": result}
        
        # Add node to workflow
        node = LangGraphNode(
            name="analyzer",
            function=analyze_node,
            inputs=["number"],
            outputs=["analysis"]
        )
        workflow.add_node(node)
        
        # Register and execute workflow
        adapter.register_workflow(workflow)
        
        initial_state = {"number": 7}
        final_state = await adapter.execute_workflow("test_workflow", initial_state)
        
        # Verify execution
        assert "analysis" in final_state.data
        assert isinstance(final_state.data["analysis"], str)
        assert len(final_state.data["analysis"]) > 10
    
    @pytest.mark.asyncio
    async def test_mcp_adapter_capabilities(self, populated_model_registry):
        """Test MCP adapter capabilities reporting."""
        adapter = MCPAdapter(model_registry=populated_model_registry)
        
        capabilities = adapter.get_capabilities()
        
        # Verify capabilities structure
        assert "supports_tools" in capabilities
        assert "supports_resources" in capabilities
        assert "supports_prompts" in capabilities
        assert "supports_mcp_protocol" in capabilities
        assert capabilities["supports_mcp_protocol"] is True
        
        # Should have AI model actions
        assert "supported_actions" in capabilities
        assert len(capabilities["supported_actions"]) > 0
