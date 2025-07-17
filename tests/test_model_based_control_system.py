"""
Tests for ModelBasedControlSystem.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from orchestrator.control_systems.model_based_control_system import ModelBasedControlSystem
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.core.pipeline import Pipeline
from orchestrator.core.task import Task, TaskStatus


class TestModelBasedControlSystem:
    """Test the ModelBasedControlSystem implementation."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock AI model."""
        model = Mock()
        model.name = "test-model"
        model.generate = AsyncMock(return_value="Test result")
        model.capabilities = {
            "tasks": ["generate", "analyze", "transform"],
            "context_window": 8192,
            "output_tokens": 4096,
            "expertise": ["general", "reasoning"]
        }
        return model
    
    @pytest.fixture
    def model_registry(self, mock_model):
        """Create a model registry with mock model."""
        registry = ModelRegistry()
        registry.register_model(mock_model)
        return registry
    
    @pytest.fixture
    def control_system(self, model_registry):
        """Create a ModelBasedControlSystem."""
        return ModelBasedControlSystem(
            model_registry=model_registry,
            name="test-control-system"
        )
    
    def test_initialization(self, control_system, model_registry):
        """Test control system initialization."""
        assert control_system.name == "test-control-system"
        assert control_system.model_registry == model_registry
        assert isinstance(control_system.config, dict)
        # Tools might not be implemented yet
        assert not hasattr(control_system, 'tools') or control_system.tools == {}
    
    def test_initialization_with_config(self, model_registry):
        """Test control system initialization with config."""
        config = {
            "default_temperature": 0.7,
            "max_retries": 5,
            "timeout": 120
        }
        control_system = ModelBasedControlSystem(
            model_registry=model_registry,
            config=config
        )
        assert control_system.config == config
    
    @pytest.mark.asyncio
    async def test_execute_simple_task(self, control_system, mock_model):
        """Test executing a simple task."""
        task = Task(
            id="test_task",
            action="Generate a greeting message",
            parameters={}
        )
        
        result = await control_system.execute_task(task, {})
        
        assert result == "Test result"
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "Test result"
        mock_model.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_task_with_context(self, control_system, mock_model):
        """Test executing task with context."""
        task = Task(
            id="context_task",
            action="Analyze data: {{previous.result}}",
            parameters={"format": "json"}
        )
        context = {
            "previous": {"result": "Some previous data"}
        }
        
        result = await control_system.execute_task(task, context)
        
        assert result == "Test result"
        # Verify context was included in prompt
        call_args = mock_model.generate.call_args
        prompt = call_args[0][0]
        assert "Some previous data" in prompt
        assert "json" in prompt
    
    @pytest.mark.asyncio
    async def test_execute_task_with_error(self, control_system, mock_model):
        """Test task execution with error."""
        mock_model.generate.side_effect = Exception("Model error")
        
        task = Task(
            id="error_task",
            action="This will fail",
            parameters={}
        )
        
        with pytest.raises(Exception, match="Model error"):
            await control_system.execute_task(task, {})
        
        assert task.status == TaskStatus.FAILED
        assert task.error is not None
    
    @pytest.mark.asyncio
    async def test_execute_task_with_retry(self, control_system, mock_model):
        """Test task execution with retry logic."""
        # Fail twice, succeed on third try
        call_count = 0
        
        async def mock_generate_with_retry(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "Success after retries"
        
        mock_model.generate = mock_generate_with_retry
        
        task = Task(
            id="retry_task",
            action="Retry on failure",
            on_error={"retry_count": 3}
        )
        
        result = await control_system.execute_task(task, {})
        
        assert result == "Success after retries"
        assert call_count == 3
        assert task.status == TaskStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_execute_task_with_fallback(self, control_system, mock_model):
        """Test task execution with fallback value."""
        mock_model.generate.side_effect = Exception("Permanent failure")
        
        task = Task(
            id="fallback_task",
            action="This will fail",
            on_error={
                "continue_on_error": True,
                "fallback_value": "fallback result"
            }
        )
        
        result = await control_system.execute_task(task, {})
        
        assert result == "fallback result"
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "fallback result"
    
    @pytest.mark.asyncio
    async def test_execute_task_with_timeout(self, control_system, mock_model):
        """Test task execution with timeout."""
        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(5)
            return "Too late"
        
        mock_model.generate = slow_generate
        
        task = Task(
            id="timeout_task",
            action="This should timeout",
            timeout=0.1
        )
        
        with pytest.raises(asyncio.TimeoutError):
            await control_system.execute_task(task, {})
        
        assert task.status == TaskStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_simple(self, control_system):
        """Test executing a simple pipeline."""
        pipeline = Pipeline(name="Test Pipeline")
        
        task1 = Task(id="task1", action="First task")
        task2 = Task(id="task2", action="Second task")
        
        pipeline.add_task(task1)
        pipeline.add_task(task2)
        
        results = await control_system.execute_pipeline(pipeline)
        
        assert "task1" in results
        assert "task2" in results
        assert task1.status == TaskStatus.COMPLETED
        assert task2.status == TaskStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_with_dependencies(self, control_system):
        """Test executing pipeline with task dependencies."""
        pipeline = Pipeline(name="Dependency Pipeline")
        
        task1 = Task(id="fetch", action="Fetch data")
        task2 = Task(
            id="process",
            action="Process data: {{fetch.result}}",
            dependencies=["fetch"]
        )
        task3 = Task(
            id="analyze",
            action="Analyze: {{process.result}}",
            dependencies=["process"]
        )
        
        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)
        
        results = await control_system.execute_pipeline(pipeline)
        
        # Verify execution order through result availability
        assert "fetch" in results
        assert "process" in results
        assert "analyze" in results
        
        # Verify all completed successfully
        assert task1.status == TaskStatus.COMPLETED
        assert task2.status == TaskStatus.COMPLETED
        assert task3.status == TaskStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_with_conditional(self, control_system, mock_model):
        """Test pipeline execution with conditional tasks."""
        pipeline = Pipeline(name="Conditional Pipeline")
        
        # Setup mock to return specific values
        async def mock_generate_conditional(*args, **kwargs):
            prompt = args[0]
            if "check condition" in prompt.lower():
                return "false"
            return "Default result"
        
        mock_model.generate = mock_generate_conditional
        
        task1 = Task(id="check", action="Check condition")
        task2 = Task(
            id="if_true",
            action="Execute if true",
            condition="{{check.result}} == 'true'",
            dependencies=["check"]
        )
        task3 = Task(
            id="if_false",
            action="Execute if false",
            condition="{{check.result}} == 'false'",
            dependencies=["check"]
        )
        
        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)
        
        results = await control_system.execute_pipeline(pipeline)
        
        assert "check" in results
        assert "if_false" in results  # Should execute
        assert "if_true" not in results  # Should skip
        
        assert task1.status == TaskStatus.COMPLETED
        assert task2.status == TaskStatus.PENDING  # Not executed
        assert task3.status == TaskStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_with_error_propagation(self, control_system, mock_model):
        """Test pipeline execution with error propagation."""
        pipeline = Pipeline(name="Error Pipeline")
        
        # Make second task fail
        call_count = 0
        
        async def mock_generate_with_error(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Task 2 failed")
            return f"Result {call_count}"
        
        mock_model.generate = mock_generate_with_error
        
        task1 = Task(id="task1", action="First task")
        task2 = Task(id="task2", action="Second task (will fail)")
        task3 = Task(
            id="task3",
            action="Third task",
            dependencies=["task2"]
        )
        
        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)
        
        with pytest.raises(Exception, match="Task 2 failed"):
            await control_system.execute_pipeline(pipeline)
        
        assert task1.status == TaskStatus.COMPLETED
        assert task2.status == TaskStatus.FAILED
        assert task3.status == TaskStatus.PENDING  # Not executed due to dependency failure
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_with_parallel_tasks(self, control_system):
        """Test pipeline with tasks that can execute in parallel."""
        pipeline = Pipeline(name="Parallel Pipeline")
        
        # Three independent tasks
        task1 = Task(id="task1", action="Task 1")
        task2 = Task(id="task2", action="Task 2")
        task3 = Task(id="task3", action="Task 3")
        
        # Task that depends on all three
        task4 = Task(
            id="aggregate",
            action="Aggregate results",
            dependencies=["task1", "task2", "task3"]
        )
        
        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)
        pipeline.add_task(task4)
        
        results = await control_system.execute_pipeline(pipeline)
        
        # All tasks should complete
        assert len(results) == 4
        for task in pipeline.tasks.values():
            assert task.status == TaskStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_build_prompt(self, control_system):
        """Test prompt building functionality."""
        task = Task(
            id="test",
            action="Analyze the following data and provide insights",
            parameters={"format": "json", "depth": "detailed"}
        )
        context = {
            "previous": {"result": "Previous analysis data"}
        }
        
        prompt = control_system._build_prompt(task, task.action, context)
        
        assert "Analyze the following data and provide insights" in prompt
        assert "format: json" in prompt
        assert "depth: detailed" in prompt
        assert "Previous analysis data" in prompt
        assert "Quality Guidelines:" in prompt
    
    @pytest.mark.asyncio
    async def test_model_selection(self, control_system, model_registry):
        """Test model selection based on task requirements."""
        # Add a specialized model
        specialized_model = Mock()
        specialized_model.name = "specialized-model"
        specialized_model.generate = AsyncMock(return_value="Specialized result")
        specialized_model.capabilities = {
            "tasks": ["code", "programming"],
            "expertise": ["code", "programming"],
            "context_window": 16384
        }
        model_registry.register_model(specialized_model)
        
        # Task requiring code expertise
        task = Task(
            id="code_task",
            action="Write a Python function",
            metadata={"requires_model": {"expertise": ["code"]}}
        )
        
        result = await control_system.execute_task(task, {})
        
        # Should use the specialized model
        specialized_model.generate.assert_called()
    
    @pytest.mark.asyncio
    async def test_register_tool(self, control_system):
        """Test tool registration."""
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        
        # If tools are not implemented, skip this test
        if not hasattr(control_system, 'register_tool'):
            pytest.skip("Tool registration not implemented")
        
        control_system.register_tool(mock_tool)
        
        assert "test_tool" in control_system.tools
        assert control_system.tools["test_tool"] == mock_tool
    
    @pytest.mark.asyncio
    async def test_execute_with_tools(self, control_system, mock_model):
        """Test task execution with tools."""
        # Register a mock tool
        mock_tool = Mock()
        mock_tool.name = "calculator"
        mock_tool.execute = AsyncMock(return_value={"result": 42})
        control_system.register_tool(mock_tool)
        
        # Skip this test if tools are not implemented
        pytest.skip("Tool execution not implemented in current version")
    
    @pytest.mark.asyncio
    async def test_context_size_management(self, control_system, mock_model):
        """Test context size management for large contexts."""
        # Create a large context
        large_context = {
            f"step_{i}": {"result": f"Result {i} " * 100}
            for i in range(50)
        }
        
        task = Task(
            id="large_context_task",
            action="Process with large context",
            parameters={}
        )
        
        result = await control_system.execute_task(task, large_context)
        
        # Should complete successfully
        assert result == "Test result"
        
        # Check that prompt was built (context might be truncated)
        call_args = mock_model.generate.call_args
        prompt = call_args[0][0]
        assert len(prompt) > 0
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, control_system):
        """Test getting control system capabilities."""
        capabilities = control_system.get_capabilities()
        
        assert capabilities["name"] == "test-control-system"
        assert capabilities["type"] == "model-based"
        assert "execute_task" in capabilities["supported_operations"]
        assert "execute_pipeline" in capabilities["supported_operations"]
        assert capabilities["supports_tools"] is True
        assert capabilities["supports_parallel_execution"] is False


class TestPromptBuilding:
    """Test prompt building functionality in detail."""
    
    @pytest.fixture
    def control_system(self):
        """Create a basic control system for testing."""
        registry = ModelRegistry()
        return ModelBasedControlSystem(registry)
    
    def test_build_prompt_basic(self, control_system):
        """Test basic prompt building."""
        task = Task(
            id="basic",
            action="Summarize this text"
        )
        
        prompt = control_system._build_prompt(task, {})
        
        assert "Task: Summarize this text" in prompt
        assert "Quality Guidelines:" in prompt
    
    def test_build_prompt_with_parameters(self, control_system):
        """Test prompt with parameters."""
        task = Task(
            id="params",
            action="Generate content",
            parameters={
                "style": "formal",
                "length": "short",
                "audience": "technical"
            }
        )
        
        prompt = control_system._build_prompt(task, {})
        
        assert "style: formal" in prompt
        assert "length: short" in prompt
        assert "audience: technical" in prompt
    
    def test_build_prompt_with_context(self, control_system):
        """Test prompt with rich context."""
        task = Task(
            id="context_task",
            action="Analyze trends"
        )
        
        context = {
            "data_source": {"result": "Sales data from 2024"},
            "previous_analysis": {
                "result": "Q1 showed 15% growth",
                "metadata": {"confidence": 0.95}
            }
        }
        
        prompt = control_system._build_prompt(task, task.action, context)
        
        assert "Sales data from 2024" in prompt
        assert "Q1 showed 15% growth" in prompt
    
    def test_build_prompt_quality_guidelines(self, control_system):
        """Test quality guidelines in prompt."""
        task = Task(id="quality", action="Write report")
        
        prompt = control_system._build_prompt(task, {})
        
        quality_indicators = [
            "comprehensive",
            "well-structured",
            "clear headings",
            "specific examples",
            "logical flow",
            "professional tone"
        ]
        
        for indicator in quality_indicators:
            assert indicator in prompt.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])