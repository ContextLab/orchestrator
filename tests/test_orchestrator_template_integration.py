"""Tests for Orchestrator integration with TemplateManager."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.orchestrator.core.task import Task
from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.template_manager import TemplateManager
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.models.model_registry import ModelRegistry
from src.orchestrator.core.model import ModelCapabilities, ModelRequirements, ModelMetrics, ModelCost


class MockModel:
    """Mock model for testing."""
    
    def __init__(self, name="test-model"):
        self.name = name
        self.model_name = name
        self.provider = "mock"
        self.capabilities = ModelCapabilities(supported_tasks=["generate"])
        self.requirements = ModelRequirements()
        self.metrics = ModelMetrics()
        self.cost = ModelCost(is_free=True)
        self._is_available = True
        self._expertise = ["general"]
        
    async def generate(self, **kwargs):
        """Generate mock response."""
        prompt = kwargs.get("prompt", "")
        return f"Mock response for: {prompt}"
    
    def meets_requirements(self, requirements: dict) -> bool:
        """Check if model meets requirements."""
        return True  # Mock model always meets requirements


class MockControlSystem:
    """Mock control system for testing."""
    
    def __init__(self):
        self.name = "mock-control-system"
        
    async def execute_task(self, task: Task, context):
        """Execute mock task.""" 
        # Extract template manager from context
        template_manager = context.get("template_manager")
        
        # Test that template manager is available
        assert template_manager is not None, "TemplateManager not found in context"
        
        # For generate_text actions, render the prompt using template manager
        if task.action == "generate_text" and task.parameters:
            prompt = task.parameters.get("prompt", "")
            if template_manager.has_templates(prompt):
                rendered_prompt = template_manager.render(prompt)
                return f"Generated from rendered prompt: {rendered_prompt}"
            else:
                return f"Generated from prompt: {prompt}"
        
        # For filesystem actions, verify template rendering
        elif task.action == "filesystem" and task.parameters:
            content = task.parameters.get("content", "")
            if template_manager.has_templates(content):
                rendered_content = template_manager.render(content)
                return {"action": "write", "content": rendered_content, "success": True}
            else:
                return {"action": "write", "content": content, "success": True}
        
        return "Mock task execution result"
    
    def get_capabilities(self):
        """Return control system capabilities."""
        return {
            "supported_actions": ["generate_text", "filesystem"],
            "parallel_execution": True,
            "streaming": False,
            "checkpoint_support": True,
        }
    
    async def health_check(self):
        """Check if the system is healthy."""
        return True


@pytest.fixture
def mock_model_registry():
    """Create mock model registry."""
    registry = ModelRegistry()
    registry.register_model(MockModel())
    return registry


@pytest.fixture 
def orchestrator(mock_model_registry):
    """Create orchestrator with mock components."""
    control_system = MockControlSystem()
    template_manager = TemplateManager(debug_mode=True)
    
    return Orchestrator(
        model_registry=mock_model_registry,
        control_system=control_system,
        template_manager=template_manager,
        debug_templates=True
    )


class TestOrchestratorTemplateIntegration:
    """Test orchestrator integration with TemplateManager."""
    
    @pytest.mark.asyncio
    async def test_template_manager_initialization(self, orchestrator):
        """Test that TemplateManager is properly initialized."""
        assert orchestrator.template_manager is not None
        assert orchestrator.template_manager.debug_mode is True
        assert "timestamp" in orchestrator.template_manager.context
    
    @pytest.mark.asyncio
    async def test_pipeline_context_registration(self, orchestrator):
        """Test that pipeline context is registered with TemplateManager."""
        # Create test pipeline
        pipeline = Pipeline(
            id="test-pipeline",
            name="Test Pipeline",
            context={"topic": "AI Research", "count": 5}
        )
        
        # Create task
        task = Task(
            id="test-task",
            name="Test Task",
            action="generate_text",
            parameters={"prompt": "Research {{topic}} with {{count}} sources"}
        )
        pipeline.add_task(task)
        
        # Execute pipeline
        results = await orchestrator.execute_pipeline(pipeline)
        
        # Verify that template variables were resolved
        expected_result = "Generated from rendered prompt: Research AI Research with 5 sources"
        assert results["test-task"] == expected_result
    
    @pytest.mark.asyncio
    async def test_step_results_registration(self, orchestrator):
        """Test that step results are registered for subsequent steps."""
        # Create pipeline with dependent tasks
        pipeline = Pipeline(
            id="test-pipeline", 
            name="Test Pipeline",
            context={"topic": "Machine Learning"}
        )
        
        # First task
        task1 = Task(
            id="analyze",
            name="Analyze Task",
            action="generate_text", 
            parameters={"prompt": "Analyze {{topic}}"}
        )
        pipeline.add_task(task1)
        
        # Second task that depends on first
        task2 = Task(
            id="summarize",
            name="Summarize Task",
            action="generate_text",
            parameters={"prompt": "Summarize this analysis: {{analyze.result}}"},
            dependencies=["analyze"]
        )
        pipeline.add_task(task2)
        
        # Execute pipeline
        results = await orchestrator.execute_pipeline(pipeline)
        
        # Verify first task executed correctly
        assert "analyze" in results
        expected_analyze = "Generated from rendered prompt: Analyze Machine Learning"
        assert results["analyze"] == expected_analyze
        
        # Verify second task had access to first task's result
        assert "summarize" in results
        expected_summarize = f"Generated from rendered prompt: Summarize this analysis: {expected_analyze}"
        assert results["summarize"] == expected_summarize
    
    @pytest.mark.asyncio
    async def test_filesystem_template_rendering(self, orchestrator):
        """Test that filesystem operations get template rendering."""
        # Create pipeline
        pipeline = Pipeline(
            id="test-pipeline",
            name="Test Pipeline", 
            context={"report_title": "Research Report", "date": "2025-01-15"}
        )
        
        # Create filesystem task with templates
        task = Task(
            id="save-report",
            name="Save Report Task",
            action="filesystem",
            parameters={
                "action": "write",
                "path": "reports/{{report_title | slugify}}.md",
                "content": "# {{report_title}}\n\nGenerated on {{date}}"
            }
        )
        pipeline.add_task(task)
        
        # Execute pipeline
        results = await orchestrator.execute_pipeline(pipeline)
        
        # Verify templates were rendered
        result = results["save-report"]
        assert result["success"] is True
        expected_content = "# Research Report\n\nGenerated on 2025-01-15"
        assert result["content"] == expected_content
    
    @pytest.mark.asyncio
    async def test_complex_nested_templates(self, orchestrator):
        """Test complex nested template scenarios."""
        # Create pipeline with search results simulation
        pipeline = Pipeline(
            id="test-pipeline",
            name="Test Pipeline",
            context={"topic": "AI Ethics"}
        )
        
        # Mock search task
        search_task = Task(
            id="search",
            name="Search Task",
            action="generate_text",
            parameters={"prompt": "Mock search results"}
        )
        pipeline.add_task(search_task)
        
        # Task that uses complex templates
        report_task = Task(
            id="generate_report",
            name="Generate Report Task", 
            action="generate_text",
            parameters={
                "prompt": """Create a report on {{topic}}:
                
Based on search: {{search.result}}

Topic slug: {{topic | slugify}}
Current time: {{"now" | date("%Y-%m-%d")}}"""
            },
            dependencies=["search"]
        )
        pipeline.add_task(report_task)
        
        # Execute pipeline
        results = await orchestrator.execute_pipeline(pipeline)
        
        # Verify complex template rendering
        assert "generate_report" in results
        result = results["generate_report"]
        
        # Should contain resolved templates
        assert "AI Ethics" in result
        assert "ai-ethics" in result  # slugified version
        assert "2025-" in result  # current year from date filter
        assert "Mock search results" in result  # from search task
    
    @pytest.mark.asyncio
    async def test_undefined_variable_handling(self, orchestrator):
        """Test graceful handling of undefined variables."""
        # Create pipeline
        pipeline = Pipeline(
            id="test-pipeline",
            name="Test Pipeline",
            context={"known_var": "known value"}
        )
        
        # Task with undefined variable
        task = Task(
            id="test-task",
            name="Test Task",
            action="generate_text",
            parameters={
                "prompt": "Known: {{known_var}}, Unknown: {{unknown_var}}"
            }
        )
        pipeline.add_task(task)
        
        # Execute pipeline - should not fail
        results = await orchestrator.execute_pipeline(pipeline)
        
        # Verify known variable resolved and unknown becomes placeholder
        result = results["test-task"]
        assert "known value" in result
        assert "{{unknown_var}}" in result  # Should remain as placeholder
    
    @pytest.mark.asyncio
    async def test_template_manager_context_isolation(self, orchestrator):
        """Test that template manager context is isolated between pipeline runs."""
        # First pipeline
        pipeline1 = Pipeline(
            id="pipeline1",
            name="Pipeline 1", 
            context={"var": "value1"}
        )
        task1 = Task(
            id="task1",
            name="Task 1", 
            action="generate_text",
            parameters={"prompt": "Value: {{var}}"}
        )
        pipeline1.add_task(task1)
        
        # Execute first pipeline
        results1 = await orchestrator.execute_pipeline(pipeline1)
        assert "value1" in results1["task1"]
        
        # Second pipeline with different context
        pipeline2 = Pipeline(
            id="pipeline2",
            name="Pipeline 2",
            context={"var": "value2"}
        )
        task2 = Task(
            id="task2",
            name="Task 2",
            action="generate_text", 
            parameters={"prompt": "Value: {{var}}"}
        )
        pipeline2.add_task(task2)
        
        # Execute second pipeline
        results2 = await orchestrator.execute_pipeline(pipeline2)
        assert "value2" in results2["task2"]
        
        # Verify contexts didn't leak
        assert "value1" not in results2["task2"]
        assert "value2" not in results1["task1"]