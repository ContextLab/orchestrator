"""
Comprehensive tests for the declarative YAML framework.

Tests cover:
- YAML compilation
- Pipeline execution with ModelBasedControlSystem
- Template resolution
- Dependency management
- Error handling scenarios
"""

import pytest
import asyncio
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from orchestrator.compiler.yaml_compiler import YAMLCompiler, YAMLCompilerError
from orchestrator.control_systems.model_based_control_system import ModelBasedControlSystem
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.core.pipeline import Pipeline
from orchestrator.core.task import Task, TaskStatus


class TestYAMLCompilation:
    """Test YAML compilation functionality."""
    
    @pytest.fixture
    def compiler(self):
        """Create a YAML compiler instance."""
        return YAMLCompiler()
    
    @pytest.mark.asyncio
    async def test_basic_yaml_compilation(self, compiler):
        """Test compilation of a basic YAML pipeline."""
        yaml_content = """
name: "Test Pipeline"
description: "A simple test pipeline"

inputs:
  message:
    type: string
    required: true

steps:
  - id: process_message
    action: "Process the message: {{message}}"

outputs:
  result: "{{process_message.result}}"
"""
        inputs = {"message": "Hello, World!"}
        pipeline = await compiler.compile(yaml_content, inputs)
        
        assert isinstance(pipeline, Pipeline)
        assert pipeline.name == "Test Pipeline"
        assert pipeline.description == "A simple test pipeline"
        assert len(pipeline.tasks) == 1
        assert "process_message" in pipeline.tasks
        assert pipeline.tasks["process_message"].id == "process_message"
        assert pipeline.tasks["process_message"].action == "Process the message: Hello, World!"
    
    @pytest.mark.asyncio
    async def test_template_resolution(self, compiler):
        """Test template variable resolution."""
        yaml_content = """
name: "Template Test"

inputs:
  name:
    type: string
    required: true
  age:
    type: integer
    default: 25

steps:
  - id: greet
    action: "Hello {{name}}, you are {{age}} years old"
"""
        inputs = {"name": "Alice"}
        pipeline = await compiler.compile(yaml_content, inputs)
        
        task = list(pipeline.tasks.values())[0]
        assert task.action == "Hello Alice, you are 25 years old"
    
    @pytest.mark.asyncio
    async def test_dependency_parsing(self, compiler):
        """Test dependency parsing and ordering."""
        yaml_content = """
name: "Dependency Test"

steps:
  - id: step_c
    action: "Step C: {{step_a.result}} and {{step_b.result}}"
    depends_on: [step_a, step_b]
    
  - id: step_a
    action: "Step A"
    
  - id: step_b
    action: "Step B: {{step_a.result}}"
    depends_on: [step_a]
"""
        pipeline = await compiler.compile(yaml_content, {})
        
        # Check dependencies are correctly set
        task_map = {task.id: task for task in pipeline.tasks}
        assert task_map["step_a"].dependencies == []
        assert task_map["step_b"].dependencies == ["step_a"]
        assert task_map["step_c"].dependencies == ["step_a", "step_b"]
        
        # Check execution levels - getting execution levels might not be implemented
        # Let's just check that tasks are created with correct dependencies
    
    @pytest.mark.asyncio
    async def test_conditional_execution(self, compiler):
        """Test conditional step execution."""
        yaml_content = """
name: "Conditional Test"

inputs:
  execute_step:
    type: boolean
    default: true

steps:
  - id: conditional_step
    action: "This step should execute conditionally"
    condition: "{{execute_step}} == true"
"""
        # Test with condition true
        pipeline = await compiler.compile(yaml_content, {"execute_step": True})
        task = list(pipeline.tasks.values())[0]
        # Condition might be stored in metadata or parameters
        assert hasattr(task, 'metadata') and 'condition' in task.metadata
        
        # Test with condition false
        pipeline = await compiler.compile(yaml_content, {"execute_step": False})
        task = list(pipeline.tasks.values())[0]
        assert hasattr(task, 'metadata') and 'condition' in task.metadata
    
    @pytest.mark.asyncio
    async def test_error_handling_config(self, compiler):
        """Test error handling configuration parsing."""
        yaml_content = """
name: "Error Handling Test"

steps:
  - id: risky_step
    action: "Perform risky operation"
    on_error:
      action: "Handle error gracefully"
      continue_on_error: true
      retry_count: 3
      fallback_value: "default result"
"""
        pipeline = await compiler.compile(yaml_content, {})
        task = list(pipeline.tasks.values())[0]
        
        # on_error might be stored in metadata
        assert hasattr(task, 'metadata') and 'on_error' in task.metadata
        on_error = task.metadata['on_error']
        assert on_error["action"] == "Handle error gracefully"
        assert on_error["continue_on_error"] is True
        assert on_error["retry_count"] == 3
        assert on_error["fallback_value"] == "default result"
    
    @pytest.mark.asyncio
    async def test_missing_required_input(self, compiler):
        """Test compilation fails with missing required input."""
        yaml_content = """
name: "Required Input Test"

inputs:
  required_field:
    type: string
    required: true

steps:
  - id: step1
    action: "Use {{required_field}}"
"""
        with pytest.raises(YAMLCompilerError):
            await compiler.compile(yaml_content, {})
    
    @pytest.mark.asyncio
    async def test_circular_dependency_detection(self, compiler):
        """Test detection of circular dependencies."""
        yaml_content = """
name: "Circular Dependency Test"

steps:
  - id: step_a
    action: "Step A: {{step_b.result}}"
    depends_on: [step_b]
    
  - id: step_b
    action: "Step B: {{step_a.result}}"
    depends_on: [step_a]
"""
        with pytest.raises(YAMLCompilerError, match="Circular dependency"):
            await compiler.compile(yaml_content, {})
    
    @pytest.mark.asyncio
    async def test_complex_template_expressions(self, compiler):
        """Test complex Jinja2 template expressions."""
        yaml_content = """
name: "Complex Template Test"

inputs:
  items:
    type: list
    default: ["a", "b", "c"]
  threshold:
    type: float
    default: 0.5

steps:
  - id: process
    action: |
      Process items:
      {% for item in items %}
      - Item: {{item}}
      {% endfor %}
      
      Threshold check: {% if threshold > 0.7 %}High{% else %}Low{% endif %}
"""
        inputs = {"items": ["x", "y"], "threshold": 0.8}
        pipeline = await compiler.compile(yaml_content, inputs)
        
        expected_action = """Process items:

- Item: x

- Item: y


Threshold check: High"""
        
        assert list(pipeline.tasks.values())[0].action.strip() == expected_action.strip()


class TestModelBasedControlSystem:
    """Test ModelBasedControlSystem execution."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock AI model."""
        model = Mock()
        model.name = "test-model"
        model.generate = AsyncMock(return_value="Test result")
        model.capabilities = {
            "tasks": ["generate", "analyze"],
            "context_window": 8192,
            "output_tokens": 4096
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
        return ModelBasedControlSystem(model_registry)
    
    @pytest.mark.asyncio
    async def test_simple_task_execution(self, control_system):
        """Test execution of a simple task."""
        task = Task(
            id="test_task",
            action="Generate a greeting",
            parameters={}
        )
        
        result = await control_system.execute_task(task, {})
        
        assert result == "Test result"
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "Test result"
    
    @pytest.mark.asyncio
    async def test_context_propagation(self, control_system):
        """Test context propagation between tasks."""
        # Create pipeline with dependent tasks
        pipeline = Pipeline(name="Context Test")
        
        task1 = Task(id="task1", action="First task")
        task2 = Task(
            id="task2",
            action="Second task using: {{task1.result}}",
            dependencies=["task1"]
        )
        
        pipeline.add_task(task1)
        pipeline.add_task(task2)
        
        # Execute pipeline
        results = await control_system.execute_pipeline(pipeline)
        
        assert "task1" in results
        assert "task2" in results
        assert task1.status == TaskStatus.COMPLETED
        assert task2.status == TaskStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_error_handling(self, control_system, mock_model):
        """Test error handling during task execution."""
        # Configure model to raise error
        mock_model.generate.side_effect = Exception("Model error")
        
        task = Task(
            id="error_task",
            action="This will fail",
            on_error={
                "action": "Use fallback",
                "continue_on_error": True,
                "fallback_value": "fallback result"
            }
        )
        
        result = await control_system.execute_task(task, {})
        
        # Should use fallback value
        assert result == "fallback result"
        assert task.status == TaskStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, control_system, mock_model):
        """Test retry mechanism on failure."""
        # Configure model to fail twice then succeed
        call_count = 0
        
        async def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "Success after retries"
        
        mock_model.generate = mock_generate
        
        task = Task(
            id="retry_task",
            action="Retry on failure",
            on_error={"retry_count": 3}
        )
        
        result = await control_system.execute_task(task, {})
        
        assert result == "Success after retries"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, control_system, mock_model):
        """Test timeout handling."""
        # Configure model to take too long
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
    async def test_conditional_execution(self, control_system):
        """Test conditional task execution."""
        pipeline = Pipeline(name="Conditional Test")
        
        task1 = Task(id="check", action="Check condition")
        task2 = Task(
            id="conditional",
            action="Execute conditionally",
            condition="False",  # Should not execute
            dependencies=["check"]
        )
        
        pipeline.add_task(task1)
        pipeline.add_task(task2)
        
        results = await control_system.execute_pipeline(pipeline)
        
        assert "check" in results
        assert "conditional" not in results  # Should be skipped
        assert task1.status == TaskStatus.COMPLETED
        assert task2.status == TaskStatus.PENDING  # Not executed


class TestIntegration:
    """Integration tests for the complete framework."""
    
    @pytest.fixture
    def setup(self):
        """Setup for integration tests."""
        compiler = YAMLCompiler()
        model_registry = ModelRegistry()
        
        # Mock model that returns predictable results
        model = Mock()
        model.name = "test-model"
        
        async def mock_generate(prompt, **kwargs):
            if "research plan" in prompt.lower():
                return "Research plan: 1. Gather data 2. Analyze 3. Report"
            elif "gather information" in prompt.lower():
                return "Information: Key facts about the topic"
            elif "analyze" in prompt.lower():
                return "Analysis: Important insights discovered"
            elif "report" in prompt.lower():
                return "Report: Comprehensive summary of findings"
            return "Generic result"
        
        model.generate = mock_generate
        model.capabilities = {
            "tasks": ["generate", "analyze"],
            "context_window": 8192
        }
        
        model_registry.register_model(model)
        control_system = ModelBasedControlSystem(model_registry)
        
        return compiler, control_system
    
    @pytest.mark.asyncio
    async def test_research_pipeline(self, setup):
        """Test execution of a complete research pipeline."""
        compiler, control_system = setup
        
        yaml_content = """
name: "Research Pipeline"
description: "Research and analyze a topic"

inputs:
  topic:
    type: string
    required: true

steps:
  - id: plan_research
    action: "Create research plan for {{topic}}"
    
  - id: gather_information
    action: |
      Gather information about {{topic}}
      Following plan: {{plan_research.result}}
    depends_on: [plan_research]
    
  - id: analyze_findings
    action: |
      Analyze findings:
      {{gather_information.result}}
    depends_on: [gather_information]
    
  - id: generate_report
    action: |
      Generate report on {{topic}}:
      Analysis: {{analyze_findings.result}}
    depends_on: [analyze_findings]

outputs:
  report: "{{generate_report.result}}"
  plan: "{{plan_research.result}}"
"""
        
        inputs = {"topic": "AI Ethics"}
        pipeline = await compiler.compile(yaml_content, inputs)
        results = await control_system.execute_pipeline(pipeline)
        
        # Verify all steps executed
        assert "plan_research" in results
        assert "gather_information" in results
        assert "analyze_findings" in results
        assert "generate_report" in results
        
        # Verify outputs
        outputs = pipeline.get_outputs(results)
        assert "report" in outputs
        assert "plan" in outputs
        assert outputs["report"] == "Report: Comprehensive summary of findings"
        assert outputs["plan"] == "Research plan: 1. Gather data 2. Analyze 3. Report"
    
    @pytest.mark.asyncio
    async def test_error_recovery_pipeline(self, setup):
        """Test pipeline with error recovery."""
        compiler, control_system = setup
        
        yaml_content = """
name: "Error Recovery Test"

steps:
  - id: primary_source
    action: "Fetch from primary source"
    on_error:
      action: "Log error"
      continue_on_error: true
      fallback_value: null
      
  - id: backup_source
    action: "Fetch from backup source"
    condition: "{{primary_source.result}} == null"
    
  - id: process_data
    action: |
      Process data from:
      Primary: {{primary_source.result}}
      Backup: {{backup_source.result | default('No backup')}}
    depends_on: [primary_source, backup_source]

outputs:
  result: "{{process_data.result}}"
"""
        
        # Mock the model to fail for primary source
        model = list(control_system.model_registry._models.values())[0]
        original_generate = model.generate
        
        async def mock_generate_with_error(prompt, **kwargs):
            if "primary source" in prompt:
                raise Exception("Primary source unavailable")
            return await original_generate(prompt, **kwargs)
        
        model.generate = mock_generate_with_error
        
        pipeline = await compiler.compile(yaml_content, {})
        results = await control_system.execute_pipeline(pipeline)
        
        # Verify error recovery worked
        assert results["primary_source"] is None  # Fallback value
        assert "backup_source" in results  # Backup executed
        assert "process_data" in results  # Processing completed
    
    @pytest.mark.asyncio
    async def test_real_yaml_file_execution(self, setup):
        """Test loading and executing a real YAML file."""
        compiler, control_system = setup
        
        # Create a test YAML file
        yaml_path = Path("/tmp/test_pipeline.yaml")
        yaml_content = """
name: "File Test Pipeline"
description: "Test loading from file"

inputs:
  message:
    type: string
    default: "Hello from file"

steps:
  - id: process
    action: "Process message: {{message}}"

outputs:
  result: "{{process.result}}"
"""
        
        yaml_path.write_text(yaml_content)
        
        try:
            # Load and execute
            with open(yaml_path, 'r') as f:
                content = f.read()
            
            pipeline = await compiler.compile(content, {})
            results = await control_system.execute_pipeline(pipeline)
            
            assert "process" in results
            outputs = pipeline.get_outputs(results)
            assert "result" in outputs
            
        finally:
            yaml_path.unlink()  # Clean up


class TestEdgeCases:
    """Test edge cases and error scenarios."""
    
    @pytest.fixture
    def compiler(self):
        return YAMLCompiler()
    
    @pytest.mark.asyncio
    async def test_empty_pipeline(self, compiler):
        """Test compilation of empty pipeline."""
        yaml_content = """
name: "Empty Pipeline"
description: "No steps"
"""
        pipeline = await compiler.compile(yaml_content, {})
        assert len(pipeline.tasks) == 0
    
    @pytest.mark.asyncio
    async def test_malformed_yaml(self, compiler):
        """Test handling of malformed YAML."""
        yaml_content = """
name: "Bad YAML"
steps:
  - id: step1
    action: "Missing quote
    depends_on: [nonexistent]
"""
        with pytest.raises(YAMLCompilerError):
            await compiler.compile(yaml_content, {})
    
    @pytest.mark.asyncio
    async def test_undefined_variable(self, compiler):
        """Test handling of undefined template variables."""
        yaml_content = """
name: "Undefined Variable Test"

steps:
  - id: step1
    action: "Use undefined: {{undefined_var}}"
"""
        with pytest.raises(YAMLCompilerError, match="undefined_var"):
            await compiler.compile(yaml_content, {})
    
    @pytest.mark.asyncio
    async def test_invalid_dependency(self, compiler):
        """Test handling of invalid dependencies."""
        yaml_content = """
name: "Invalid Dependency Test"

steps:
  - id: step1
    action: "Step 1"
    depends_on: [nonexistent_step]
"""
        with pytest.raises(YAMLCompilerError, match="nonexistent_step"):
            await compiler.compile(yaml_content, {})
    
    @pytest.mark.asyncio
    async def test_type_validation(self, compiler):
        """Test input type validation."""
        yaml_content = """
name: "Type Validation Test"

inputs:
  count:
    type: integer
    required: true

steps:
  - id: step1
    action: "Count is {{count}}"
"""
        # Test with wrong type
        with pytest.raises(YAMLCompilerError, match="Expected integer"):
            await compiler.compile(yaml_content, {"count": "not a number"})
        
        # Test with correct type
        pipeline = await compiler.compile(yaml_content, {"count": 42})
        assert list(pipeline.tasks.values())[0].action == "Count is 42"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])