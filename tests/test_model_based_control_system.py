"""
Tests for ModelBasedControlSystem.
"""

import pytest
import asyncio
from typing import Dict, Any

from orchestrator.control_systems.model_based_control_system import ModelBasedControlSystem
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.core.pipeline import Pipeline
from orchestrator.core.task import Task, TaskStatus
from orchestrator.core.model import Model, ModelCapabilities


class TestModelBasedControlSystem:
    """Test the ModelBasedControlSystem implementation."""
    
    @pytest.fixture
    def test_model(self):
        """Create a test AI model."""
        class TestModel(Model):
            def __init__(self):
                capabilities = ModelCapabilities(
                    supported_tasks=["generate", "analyze", "transform"],
                    context_window=8192,
                    languages=["en"]
                )
                super().__init__(
                    name="test-model",
                    provider="test",
                    capabilities=capabilities
                )
                self._generate_calls = []
                self._next_result = "Test result"
                self._should_fail = False
                self._fail_message = "Model error"
                self._fail_count = 0
                self._current_fails = 0
                
            async def generate(self, prompt, **kwargs):
                self._generate_calls.append((prompt, kwargs))
                
                if self._should_fail:
                    if self._fail_count == 0 or self._current_fails < self._fail_count:
                        self._current_fails += 1
                        raise Exception(self._fail_message)
                    else:
                        # Stop failing after fail_count reached
                        self._current_fails = 0
                        return "Success after retries"
                
                # Check for specific prompts to return different results
                if "check condition" in prompt.lower():
                    return "false"
                
                return self._next_result
            
            def set_next_result(self, result):
                self._next_result = result
                
            def set_failure_mode(self, should_fail, message="Model error", fail_count=0):
                self._should_fail = should_fail
                self._fail_message = message
                self._fail_count = fail_count
                self._current_fails = 0
                
            def get_calls(self):
                return self._generate_calls
                
            def was_called(self):
                return len(self._generate_calls) > 0
                
            def call_count(self):
                return len(self._generate_calls)
                
            async def generate_structured(self, prompt, schema, **kwargs):
                result = await self.generate(prompt, **kwargs)
                return {"value": result}
                
            async def validate_response(self, response, schema):
                return True
                
            def estimate_tokens(self, text):
                return len(text.split())
                
            def estimate_cost(self, input_tokens, output_tokens):
                return 0.0
                
            async def health_check(self):
                return True
                
        return TestModel()
    
    @pytest.fixture
    def model_registry(self, test_model):
        """Create a model registry with test model."""
        registry = ModelRegistry()
        registry.register_model(test_model)
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
    async def test_execute_simple_task(self, control_system, test_model):
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
        assert test_model.was_called()
        assert test_model.call_count() == 1
    
    @pytest.mark.asyncio
    async def test_execute_task_with_context(self, control_system, test_model):
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
        calls = test_model.get_calls()
        assert len(calls) > 0
        prompt = calls[0][0]
        assert "Some previous data" in prompt
        assert "json" in prompt
    
    @pytest.mark.asyncio
    async def test_execute_task_with_error(self, control_system, test_model):
        """Test task execution with error."""
        test_model.set_failure_mode(True, "Model error")
        
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
    async def test_execute_task_with_retry(self, control_system, test_model):
        """Test task execution with retry logic."""
        # Fail twice, succeed on third try
        test_model.set_failure_mode(True, "Temporary failure", fail_count=2)
        
        task = Task(
            id="retry_task",
            action="Retry on failure",
            on_error={"retry_count": 3}
        )
        
        result = await control_system.execute_task(task, {})
        
        assert result == "Success after retries"
        assert test_model.call_count() == 3
        assert task.status == TaskStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_execute_task_with_fallback(self, control_system, test_model):
        """Test task execution with fallback value."""
        test_model.set_failure_mode(True, "Permanent failure")
        
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
    async def test_execute_task_with_timeout(self, control_system, test_model):
        """Test task execution with timeout."""
        # Create a slow model for timeout testing
        class SlowTestModel(test_model.__class__):
            async def generate(self, prompt, **kwargs):
                await asyncio.sleep(5)
                return "Too late"
        
        # Replace the model in registry
        slow_model = SlowTestModel()
        control_system.model_registry.register_model(slow_model)
        
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
    async def test_execute_pipeline_with_conditional(self, control_system, test_model):
        """Test pipeline execution with conditional tasks."""
        pipeline = Pipeline(name="Conditional Pipeline")
        
        # The test model already returns "false" for "check condition" prompts
        
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
    async def test_execute_pipeline_with_error_propagation(self, control_system, test_model):
        """Test pipeline execution with error propagation."""
        pipeline = Pipeline(name="Error Pipeline")
        
        # Create a special model that fails on second call
        class FailOnSecondModel(test_model.__class__):
            def __init__(self):
                super().__init__()
                self._call_count = 0
                
            async def generate(self, prompt, **kwargs):
                self._call_count += 1
                if self._call_count == 2:
                    raise Exception("Task 2 failed")
                return f"Result {self._call_count}"
        
        # Replace model in registry
        fail_model = FailOnSecondModel()
        control_system.model_registry.register_model(fail_model)
        
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
        class SpecializedModel(Model):
            def __init__(self):
                capabilities = ModelCapabilities(
                    supported_tasks=["code", "programming"],
                    context_window=16384,
                    languages=["en"]
                )
                super().__init__(
                    name="specialized-model",
                    provider="test",
                    capabilities=capabilities
                )
                self.was_used = False
                
            async def generate(self, prompt, **kwargs):
                self.was_used = True
                return "Specialized result"
                
            async def generate_structured(self, prompt, schema, **kwargs):
                return {"value": await self.generate(prompt, **kwargs)}
                
            async def validate_response(self, response, schema):
                return True
                
            def estimate_tokens(self, text):
                return len(text.split())
                
            def estimate_cost(self, input_tokens, output_tokens):
                return 0.0
                
            async def health_check(self):
                return True
        
        specialized_model = SpecializedModel()
        model_registry.register_model(specialized_model)
        
        # Task requiring code expertise
        task = Task(
            id="code_task",
            action="Write a Python function",
            metadata={"requires_model": {"expertise": ["code"]}}
        )
        
        result = await control_system.execute_task(task, {})
        
        # Should use the specialized model
        assert specialized_model.was_used
    
    @pytest.mark.asyncio
    async def test_register_tool(self, control_system):
        """Test tool registration."""
        # Create a real tool instead of mock
        from orchestrator.tools.base import Tool
        
        class TestTool(Tool):
            def __init__(self):
                super().__init__(name="test_tool", description="Test tool")
                
            async def execute(self, **kwargs):
                return {"result": "tool executed"}
        
        test_tool = TestTool()
        
        # If tools are not implemented, skip this test
        if not hasattr(control_system, 'register_tool'):
            pytest.skip("Tool registration not implemented")
        
        control_system.register_tool(test_tool)
        
        assert "test_tool" in control_system.tools
        assert control_system.tools["test_tool"] == test_tool
    
    @pytest.mark.asyncio
    async def test_execute_with_tools(self, control_system, test_model):
        """Test task execution with tools."""
        # Skip this test if tools are not implemented
        pytest.skip("Tool execution not implemented in current version")
    
    @pytest.mark.asyncio
    async def test_context_size_management(self, control_system, test_model):
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
        calls = test_model.get_calls()
        assert len(calls) > 0
        prompt = calls[0][0]
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