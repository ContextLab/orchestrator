"""Tests for Phase 1: Core declarative engine implementation."""

import pytest
from orchestrator.engine import DeclarativePipelineEngine
from orchestrator.engine.pipeline_spec import PipelineSpec, TaskSpec


class TestDeclarativeEngine:
    """Test the core declarative pipeline engine."""
    
    @pytest.fixture
    def engine(self):
        """Create a test engine instance."""
        return DeclarativePipelineEngine()
    
    @pytest.fixture
    def simple_pipeline_yaml(self):
        """Simple test pipeline YAML."""
        return """
name: "Simple Test Pipeline"
description: "Test the declarative engine"
version: "1.0.0"

inputs:
  message:
    type: string
    description: "Message to process"

steps:
  - id: process
    action: <AUTO>process the message {{message}} and return a summary</AUTO>
    
  - id: format
    action: <AUTO>format the processed result as a nice output</AUTO>
    depends_on: [process]

outputs:
  result: "{{format.result}}"
"""
    
    def test_pipeline_parsing(self, engine, simple_pipeline_yaml):
        """Test pipeline parsing and specification creation."""
        spec = engine._parse_yaml_to_spec(simple_pipeline_yaml)
        
        assert spec.name == "Simple Test Pipeline"
        assert len(spec.steps) == 2
        assert spec.steps[0].id == "process"
        assert spec.steps[1].id == "format"
        assert spec.steps[1].depends_on == ["process"]
    
    def test_execution_order(self, engine, simple_pipeline_yaml):
        """Test topological sorting of pipeline steps."""
        spec = engine._parse_yaml_to_spec(simple_pipeline_yaml)
        execution_order = spec.get_execution_order()
        
        assert len(execution_order) == 2
        assert execution_order[0].id == "process"
        assert execution_order[1].id == "format"
    
    def test_auto_tag_detection(self, engine, simple_pipeline_yaml):
        """Test AUTO tag detection in pipeline steps."""
        spec = engine._parse_yaml_to_spec(simple_pipeline_yaml)
        auto_steps = spec.get_steps_with_auto_tags()
        
        assert len(auto_steps) == 2
        assert all(step.has_auto_tags() for step in auto_steps)
    
    @pytest.mark.asyncio
    async def test_pipeline_validation(self, engine, simple_pipeline_yaml):
        """Test pipeline validation."""
        validation = await engine.validate_pipeline(simple_pipeline_yaml)
        
        assert validation["valid"] is True
        assert validation["pipeline_name"] == "Simple Test Pipeline"
        assert validation["total_steps"] == 2
        assert validation["auto_tag_steps"] == 2
    
    def test_invalid_pipeline_yaml(self, engine):
        """Test handling of invalid YAML."""
        invalid_yaml = """
name: Invalid Pipeline
steps:
  - id: step1
    # Missing required 'action' field
"""
        with pytest.raises(ValueError):
            engine._parse_yaml_to_spec(invalid_yaml)
    
    def test_circular_dependency_detection(self, engine):
        """Test circular dependency detection."""
        circular_yaml = """
name: "Circular Pipeline"
steps:
  - id: step1
    action: "action1"
    depends_on: [step2]
  - id: step2
    action: "action2"
    depends_on: [step1]
"""
        with pytest.raises(ValueError, match="Circular dependency"):
            engine._parse_yaml_to_spec(circular_yaml)


class TestTaskSpec:
    """Test TaskSpec functionality."""
    
    def test_task_spec_creation(self):
        """Test basic TaskSpec creation."""
        task = TaskSpec(
            id="test_task",
            action="<AUTO>perform test action</AUTO>"
        )
        
        assert task.id == "test_task"
        assert task.has_auto_tags() is True
        assert task.extract_auto_content() == "perform test action"
    
    def test_template_variable_extraction(self):
        """Test template variable extraction."""
        task = TaskSpec(
            id="test_task",
            action="Process {{input}} and {{data.value}}"
        )
        
        variables = task.get_template_variables()
        assert len(variables) == 2
        assert "input" in variables
        assert "data.value" in variables
    
    def test_task_spec_validation(self):
        """Test TaskSpec validation."""
        # Missing ID
        with pytest.raises(ValueError, match="Task ID is required"):
            TaskSpec(id="", action="test")
        
        # Missing action
        with pytest.raises(ValueError, match="Task action is required"):
            TaskSpec(id="test", action="")


class TestPipelineSpec:
    """Test PipelineSpec functionality."""
    
    def test_pipeline_spec_creation(self):
        """Test basic PipelineSpec creation."""
        spec = PipelineSpec(
            name="Test Pipeline",
            steps=[
                TaskSpec(id="step1", action="action1"),
                TaskSpec(id="step2", action="action2", depends_on=["step1"])
            ]
        )
        
        assert spec.name == "Test Pipeline"
        assert len(spec.steps) == 2
        assert spec.steps[1].depends_on == ["step1"]
    
    def test_dependency_validation(self):
        """Test dependency validation."""
        # Non-existent dependency
        with pytest.raises(ValueError, match="depends on non-existent step"):
            PipelineSpec(
                name="Test",
                steps=[
                    TaskSpec(id="step1", action="action1", depends_on=["missing"])
                ]
            )
    
    def test_input_validation(self):
        """Test input validation."""
        spec = PipelineSpec(
            name="Test",
            inputs={
                "required_input": {"type": "string", "required": True},
                "optional_input": {"type": "string", "required": False}
            },
            steps=[TaskSpec(id="step1", action="test")]
        )
        
        # Missing required input
        with pytest.raises(ValueError, match="Required input"):
            spec.validate_inputs({})
        
        # Valid inputs
        assert spec.validate_inputs({"required_input": "test"}) is True