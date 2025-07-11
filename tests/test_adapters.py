"""Tests for control system adapters."""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from orchestrator.adapters.langgraph_adapter import LangGraphAdapter
from orchestrator.adapters.mcp_adapter import MCPAdapter
from orchestrator.adapters.custom_adapter import CustomMLAdapter
from orchestrator.adapters.base_adapter import BaseAdapter, AdapterError
from orchestrator.core.task import Task, TaskStatus
from orchestrator.core.pipeline import Pipeline


class TestBaseAdapter:
    """Test cases for BaseAdapter class."""
    
    def test_base_adapter_creation(self):
        """Test basic adapter creation."""
        config = {"name": "test_adapter", "version": "1.0.0"}
        adapter = BaseAdapter(config)
        
        assert adapter.config == config
        assert adapter.name == "test_adapter"
        assert adapter.version == "1.0.0"
        assert adapter.capabilities == {}
    
    def test_base_adapter_validation(self):
        """Test adapter configuration validation."""
        # Valid config
        valid_config = {
            "name": "test_adapter",
            "version": "1.0.0",
            "capabilities": ["task_execution", "pipeline_management"]
        }
        
        adapter = BaseAdapter(valid_config)
        assert adapter.validate_config() is True
        
        # Invalid config
        invalid_config = {"name": ""}  # Empty name
        
        with pytest.raises(AdapterError):
            BaseAdapter(invalid_config)
    
    @pytest.mark.asyncio
    async def test_base_adapter_health_check(self):
        """Test adapter health check."""
        config = {"name": "test_adapter", "version": "1.0.0"}
        adapter = BaseAdapter(config)
        
        # Mock health check
        adapter._perform_health_check = AsyncMock(return_value=True)
        
        health = await adapter.health_check()
        
        assert health is True
        adapter._perform_health_check.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_base_adapter_initialize(self):
        """Test adapter initialization."""
        config = {"name": "test_adapter", "version": "1.0.0"}
        adapter = BaseAdapter(config)
        
        # Mock initialization
        adapter._initialize_resources = AsyncMock()
        adapter._validate_dependencies = AsyncMock()
        
        await adapter.initialize()
        
        assert adapter.initialized is True
        adapter._initialize_resources.assert_called_once()
        adapter._validate_dependencies.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_base_adapter_shutdown(self):
        """Test adapter shutdown."""
        config = {"name": "test_adapter", "version": "1.0.0"}
        adapter = BaseAdapter(config)
        
        # Mock shutdown
        adapter._cleanup_resources = AsyncMock()
        
        await adapter.shutdown()
        
        assert adapter.initialized is False
        adapter._cleanup_resources.assert_called_once()
    
    def test_base_adapter_get_capabilities(self):
        """Test getting adapter capabilities."""
        config = {
            "name": "test_adapter",
            "version": "1.0.0",
            "capabilities": {
                "supported_actions": ["generate", "analyze"],
                "parallel_execution": True,
                "streaming": False
            }
        }
        
        adapter = BaseAdapter(config)
        capabilities = adapter.get_capabilities()
        
        assert capabilities["supported_actions"] == ["generate", "analyze"]
        assert capabilities["parallel_execution"] is True
        assert capabilities["streaming"] is False
    
    def test_base_adapter_supports_capability(self):
        """Test checking if adapter supports capability."""
        config = {
            "name": "test_adapter",
            "capabilities": {
                "supported_actions": ["generate", "analyze"],
                "parallel_execution": True
            }
        }
        
        adapter = BaseAdapter(config)
        
        assert adapter.supports_capability("parallel_execution") is True
        assert adapter.supports_capability("streaming") is False
    
    def test_base_adapter_metrics(self):
        """Test adapter metrics collection."""
        config = {"name": "test_adapter", "version": "1.0.0"}
        adapter = BaseAdapter(config)
        
        # Simulate some operations
        adapter._record_operation("task_execution", success=True, duration=1.5)
        adapter._record_operation("task_execution", success=False, duration=2.0)
        
        metrics = adapter.get_metrics()
        
        assert metrics["total_operations"] == 2
        assert metrics["successful_operations"] == 1
        assert metrics["failed_operations"] == 1
        assert metrics["average_duration"] == 1.75
        assert metrics["success_rate"] == 0.5


class TestLangGraphAdapter:
    """Test cases for LangGraphAdapter class."""
    
    def test_langgraph_adapter_creation(self):
        """Test basic LangGraph adapter creation."""
        config = {
            "name": "langgraph_adapter",
            "api_key": "test_key",
            "endpoint": "https://api.langgraph.com"
        }
        
        adapter = LangGraphAdapter(config)
        
        assert adapter.name == "langgraph_adapter"
        assert adapter.api_key == "test_key"
        assert adapter.endpoint == "https://api.langgraph.com"
        assert adapter.client is not None
    
    @pytest.mark.asyncio
    async def test_langgraph_execute_task(self):
        """Test executing task through LangGraph."""
        config = {
            "name": "langgraph_adapter",
            "api_key": "test_key",
            "endpoint": "https://api.langgraph.com"
        }
        
        adapter = LangGraphAdapter(config)
        
        # Mock LangGraph client
        adapter.client = AsyncMock()
        adapter.client.execute_task = AsyncMock(return_value={"result": "task_completed"})
        
        task = Task(
            id="test_task",
            name="Test Task",
            action="generate",
            parameters={"prompt": "Hello world"}
        )
        
        context = {"pipeline_id": "test_pipeline"}
        
        result = await adapter.execute_task(task, context)
        
        assert result["result"] == "task_completed"
        adapter.client.execute_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_langgraph_execute_pipeline(self):
        """Test executing pipeline through LangGraph."""
        config = {
            "name": "langgraph_adapter",
            "api_key": "test_key"
        }
        
        adapter = LangGraphAdapter(config)
        
        # Mock LangGraph client
        adapter.client = AsyncMock()
        adapter.client.execute_pipeline = AsyncMock(return_value={
            "task1": "result1",
            "task2": "result2"
        })
        
        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="generate")
        task2 = Task(id="task2", name="Task 2", action="analyze")
        
        pipeline.add_task(task1)
        pipeline.add_task(task2)
        
        results = await adapter.execute_pipeline(pipeline)
        
        assert results["task1"] == "result1"
        assert results["task2"] == "result2"
        adapter.client.execute_pipeline.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_langgraph_streaming_execution(self):
        """Test streaming execution through LangGraph."""
        config = {
            "name": "langgraph_adapter",
            "api_key": "test_key",
            "streaming": True
        }
        
        adapter = LangGraphAdapter(config)
        
        # Mock streaming client
        adapter.client = AsyncMock()
        
        async def mock_stream():
            yield {"type": "start", "task_id": "task1"}
            yield {"type": "progress", "task_id": "task1", "progress": 0.5}
            yield {"type": "complete", "task_id": "task1", "result": "final_result"}
        
        adapter.client.stream_execution = AsyncMock(return_value=mock_stream())
        
        task = Task(id="task1", name="Task 1", action="generate")
        
        results = []
        async for event in adapter.stream_execution(task, {}):
            results.append(event)
        
        assert len(results) == 3
        assert results[0]["type"] == "start"
        assert results[1]["type"] == "progress"
        assert results[2]["type"] == "complete"
        assert results[2]["result"] == "final_result"
    
    @pytest.mark.asyncio
    async def test_langgraph_error_handling(self):
        """Test error handling in LangGraph adapter."""
        config = {
            "name": "langgraph_adapter",
            "api_key": "test_key"
        }
        
        adapter = LangGraphAdapter(config)
        
        # Mock client error
        adapter.client = AsyncMock()
        adapter.client.execute_task = AsyncMock(side_effect=Exception("LangGraph API error"))
        
        task = Task(id="test_task", name="Test Task", action="generate")
        
        with pytest.raises(AdapterError):
            await adapter.execute_task(task, {})
    
    def test_langgraph_graph_builder(self):
        """Test building LangGraph execution graph."""
        config = {"name": "langgraph_adapter", "api_key": "test_key"}
        adapter = LangGraphAdapter(config)
        
        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="generate")
        task2 = Task(id="task2", name="Task 2", action="analyze", dependencies=["task1"])
        task3 = Task(id="task3", name="Task 3", action="transform", dependencies=["task2"])
        
        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)
        
        graph = adapter._build_execution_graph(pipeline)
        
        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) == 3
        assert len(graph["edges"]) == 2
        
        # Check dependencies
        edge1 = next(e for e in graph["edges"] if e["from"] == "task1" and e["to"] == "task2")
        edge2 = next(e for e in graph["edges"] if e["from"] == "task2" and e["to"] == "task3")
        
        assert edge1 is not None
        assert edge2 is not None
    
    def test_langgraph_capabilities(self):
        """Test LangGraph adapter capabilities."""
        config = {"name": "langgraph_adapter", "api_key": "test_key"}
        adapter = LangGraphAdapter(config)
        
        capabilities = adapter.get_capabilities()
        
        assert "supported_actions" in capabilities
        assert "parallel_execution" in capabilities
        assert "streaming" in capabilities
        assert "graph_optimization" in capabilities
        assert capabilities["graph_optimization"] is True


class TestMCPAdapter:
    """Test cases for MCPAdapter class."""
    
    def test_mcp_adapter_creation(self):
        """Test basic MCP adapter creation."""
        config = {
            "name": "mcp_adapter",
            "server_url": "http://localhost:8080",
            "protocol_version": "1.0"
        }
        
        adapter = MCPAdapter(config)
        
        assert adapter.name == "mcp_adapter"
        assert adapter.server_url == "http://localhost:8080"
        assert adapter.protocol_version == "1.0"
        assert adapter.session is not None
    
    @pytest.mark.asyncio
    async def test_mcp_initialize_session(self):
        """Test MCP session initialization."""
        config = {
            "name": "mcp_adapter",
            "server_url": "http://localhost:8080"
        }
        
        adapter = MCPAdapter(config)
        
        # Mock session initialization
        adapter.session = AsyncMock()
        adapter.session.initialize = AsyncMock(return_value={"session_id": "test_session"})
        
        await adapter.initialize()
        
        assert adapter.initialized is True
        adapter.session.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mcp_execute_task(self):
        """Test executing task through MCP."""
        config = {
            "name": "mcp_adapter",
            "server_url": "http://localhost:8080"
        }
        
        adapter = MCPAdapter(config)
        
        # Mock MCP session
        adapter.session = AsyncMock()
        adapter.session.send_request = AsyncMock(return_value={
            "id": "req_123",
            "result": {"output": "task_completed"}
        })
        
        task = Task(
            id="test_task",
            name="Test Task",
            action="generate",
            parameters={"prompt": "Hello world"}
        )
        
        context = {"pipeline_id": "test_pipeline"}
        
        result = await adapter.execute_task(task, context)
        
        assert result["output"] == "task_completed"
        adapter.session.send_request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mcp_tool_invocation(self):
        """Test MCP tool invocation."""
        config = {
            "name": "mcp_adapter",
            "server_url": "http://localhost:8080"
        }
        
        adapter = MCPAdapter(config)
        
        # Mock tool invocation
        adapter.session = AsyncMock()
        adapter.session.call_tool = AsyncMock(return_value={
            "tool_name": "calculator",
            "result": {"answer": 42}
        })
        
        tool_call = {
            "tool_name": "calculator",
            "parameters": {"expression": "6 * 7"}
        }
        
        result = await adapter.invoke_tool(tool_call)
        
        assert result["answer"] == 42
        adapter.session.call_tool.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mcp_resource_access(self):
        """Test MCP resource access."""
        config = {
            "name": "mcp_adapter",
            "server_url": "http://localhost:8080"
        }
        
        adapter = MCPAdapter(config)
        
        # Mock resource access
        adapter.session = AsyncMock()
        adapter.session.read_resource = AsyncMock(return_value={
            "uri": "file:///path/to/resource.txt",
            "content": "Resource content"
        })
        
        resource_uri = "file:///path/to/resource.txt"
        
        result = await adapter.read_resource(resource_uri)
        
        assert result["content"] == "Resource content"
        adapter.session.read_resource.assert_called_once_with(resource_uri)
    
    @pytest.mark.asyncio
    async def test_mcp_sampling_support(self):
        """Test MCP sampling support."""
        config = {
            "name": "mcp_adapter",
            "server_url": "http://localhost:8080"
        }
        
        adapter = MCPAdapter(config)
        
        # Mock sampling
        adapter.session = AsyncMock()
        adapter.session.create_message = AsyncMock(return_value={
            "message_id": "msg_123",
            "content": {"text": "Generated response"}
        })
        
        sampling_request = {
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100
        }
        
        result = await adapter.sample_message(sampling_request)
        
        assert result["content"]["text"] == "Generated response"
        adapter.session.create_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mcp_prompt_management(self):
        """Test MCP prompt management."""
        config = {
            "name": "mcp_adapter",
            "server_url": "http://localhost:8080"
        }
        
        adapter = MCPAdapter(config)
        
        # Mock prompt listing
        adapter.session = AsyncMock()
        adapter.session.list_prompts = AsyncMock(return_value={
            "prompts": [
                {"name": "greeting", "description": "Greeting prompt"},
                {"name": "analysis", "description": "Analysis prompt"}
            ]
        })
        
        prompts = await adapter.list_prompts()
        
        assert len(prompts["prompts"]) == 2
        assert prompts["prompts"][0]["name"] == "greeting"
        adapter.session.list_prompts.assert_called_once()
    
    def test_mcp_protocol_compliance(self):
        """Test MCP protocol compliance."""
        config = {
            "name": "mcp_adapter",
            "server_url": "http://localhost:8080",
            "protocol_version": "1.0"
        }
        
        adapter = MCPAdapter(config)
        
        # Test protocol compliance
        assert adapter.supports_protocol_version("1.0") is True
        assert adapter.supports_protocol_version("0.9") is False
        assert adapter.supports_protocol_version("2.0") is False
    
    def test_mcp_capabilities(self):
        """Test MCP adapter capabilities."""
        config = {
            "name": "mcp_adapter",
            "server_url": "http://localhost:8080"
        }
        
        adapter = MCPAdapter(config)
        
        capabilities = adapter.get_capabilities()
        
        assert "tool_calling" in capabilities
        assert "resource_access" in capabilities
        assert "sampling" in capabilities
        assert "prompt_management" in capabilities
        assert capabilities["tool_calling"] is True
        assert capabilities["resource_access"] is True


class TestCustomMLAdapter:
    """Test cases for CustomMLAdapter class."""
    
    def test_custom_adapter_creation(self):
        """Test basic custom adapter creation."""
        config = {
            "name": "custom_ml_adapter",
            "ml_framework": "pytorch",
            "model_path": "/path/to/model",
            "device": "cuda"
        }
        
        adapter = CustomMLAdapter(config)
        
        assert adapter.name == "custom_ml_adapter"
        assert adapter.ml_framework == "pytorch"
        assert adapter.model_path == "/path/to/model"
        assert adapter.device == "cuda"
    
    @pytest.mark.asyncio
    async def test_custom_adapter_model_loading(self):
        """Test custom adapter model loading."""
        config = {
            "name": "custom_ml_adapter",
            "ml_framework": "pytorch",
            "model_path": "/path/to/model"
        }
        
        adapter = CustomMLAdapter(config)
        
        # Mock model loading
        adapter._load_model = AsyncMock(return_value="mock_model")
        
        await adapter.initialize()
        
        assert adapter.model is not None
        adapter._load_model.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_custom_adapter_inference(self):
        """Test custom adapter inference."""
        config = {
            "name": "custom_ml_adapter",
            "ml_framework": "pytorch",
            "model_path": "/path/to/model"
        }
        
        adapter = CustomMLAdapter(config)
        
        # Mock inference
        adapter.model = MagicMock()
        adapter.model.predict = AsyncMock(return_value={
            "predictions": [0.8, 0.2],
            "confidence": 0.9
        })
        
        task = Task(
            id="inference_task",
            name="Inference Task",
            action="predict",
            parameters={"input_data": [1, 2, 3, 4]}
        )
        
        result = await adapter.execute_task(task, {})
        
        assert result["predictions"] == [0.8, 0.2]
        assert result["confidence"] == 0.9
        adapter.model.predict.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_custom_adapter_training(self):
        """Test custom adapter training."""
        config = {
            "name": "custom_ml_adapter",
            "ml_framework": "pytorch",
            "model_path": "/path/to/model"
        }
        
        adapter = CustomMLAdapter(config)
        
        # Mock training
        adapter.model = MagicMock()
        adapter.model.train = AsyncMock(return_value={
            "epochs": 10,
            "final_loss": 0.05,
            "accuracy": 0.95
        })
        
        task = Task(
            id="training_task",
            name="Training Task",
            action="train",
            parameters={
                "training_data": "path/to/training_data",
                "epochs": 10,
                "learning_rate": 0.001
            }
        )
        
        result = await adapter.execute_task(task, {})
        
        assert result["epochs"] == 10
        assert result["final_loss"] == 0.05
        assert result["accuracy"] == 0.95
        adapter.model.train.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_custom_adapter_evaluation(self):
        """Test custom adapter evaluation."""
        config = {
            "name": "custom_ml_adapter",
            "ml_framework": "pytorch",
            "model_path": "/path/to/model"
        }
        
        adapter = CustomMLAdapter(config)
        
        # Mock evaluation
        adapter.model = MagicMock()
        adapter.model.evaluate = AsyncMock(return_value={
            "accuracy": 0.92,
            "precision": 0.89,
            "recall": 0.94,
            "f1_score": 0.91
        })
        
        task = Task(
            id="eval_task",
            name="Evaluation Task",
            action="evaluate",
            parameters={"test_data": "path/to/test_data"}
        )
        
        result = await adapter.execute_task(task, {})
        
        assert result["accuracy"] == 0.92
        assert result["precision"] == 0.89
        assert result["recall"] == 0.94
        assert result["f1_score"] == 0.91
        adapter.model.evaluate.assert_called_once()
    
    def test_custom_adapter_optimization(self):
        """Test custom adapter optimization."""
        config = {
            "name": "custom_ml_adapter",
            "ml_framework": "pytorch",
            "optimization": {
                "gpu_acceleration": True,
                "mixed_precision": True,
                "batch_size": 32
            }
        }
        
        adapter = CustomMLAdapter(config)
        
        # Test optimization settings
        assert adapter.optimization["gpu_acceleration"] is True
        assert adapter.optimization["mixed_precision"] is True
        assert adapter.optimization["batch_size"] == 32
    
    def test_custom_adapter_distributed_training(self):
        """Test custom adapter distributed training support."""
        config = {
            "name": "custom_ml_adapter",
            "ml_framework": "pytorch",
            "distributed": {
                "enabled": True,
                "world_size": 4,
                "rank": 0
            }
        }
        
        adapter = CustomMLAdapter(config)
        
        # Test distributed settings
        assert adapter.distributed["enabled"] is True
        assert adapter.distributed["world_size"] == 4
        assert adapter.distributed["rank"] == 0
    
    def test_custom_adapter_capabilities(self):
        """Test custom adapter capabilities."""
        config = {
            "name": "custom_ml_adapter",
            "ml_framework": "pytorch"
        }
        
        adapter = CustomMLAdapter(config)
        
        capabilities = adapter.get_capabilities()
        
        assert "supported_operations" in capabilities
        assert "train" in capabilities["supported_operations"]
        assert "predict" in capabilities["supported_operations"]
        assert "evaluate" in capabilities["supported_operations"]
        assert "optimize" in capabilities["supported_operations"]
        assert "parallel_execution" in capabilities
        assert "gpu_acceleration" in capabilities
        assert "distributed" in capabilities
    
    @pytest.mark.asyncio
    async def test_custom_adapter_resource_monitoring(self):
        """Test custom adapter resource monitoring."""
        config = {
            "name": "custom_ml_adapter",
            "ml_framework": "pytorch",
            "monitoring": True
        }
        
        adapter = CustomMLAdapter(config)
        
        # Mock resource monitoring
        adapter._get_resource_usage = AsyncMock(return_value={
            "gpu_memory_used": 2.5,
            "gpu_memory_total": 8.0,
            "cpu_usage": 0.6,
            "memory_usage": 0.4
        })
        
        resources = await adapter.get_resource_usage()
        
        assert resources["gpu_memory_used"] == 2.5
        assert resources["gpu_memory_total"] == 8.0
        assert resources["cpu_usage"] == 0.6
        assert resources["memory_usage"] == 0.4
    
    @pytest.mark.asyncio
    async def test_custom_adapter_model_versioning(self):
        """Test custom adapter model versioning."""
        config = {
            "name": "custom_ml_adapter",
            "ml_framework": "pytorch",
            "model_versioning": True
        }
        
        adapter = CustomMLAdapter(config)
        
        # Mock model versioning
        adapter._save_model_version = AsyncMock(return_value="v1.0.0")
        adapter._load_model_version = AsyncMock(return_value="loaded_model")
        
        # Save version
        version = await adapter.save_model_version("checkpoint_name")
        assert version == "v1.0.0"
        
        # Load version
        model = await adapter.load_model_version("v1.0.0")
        assert model == "loaded_model"
        
        adapter._save_model_version.assert_called_once_with("checkpoint_name")
        adapter._load_model_version.assert_called_once_with("v1.0.0")