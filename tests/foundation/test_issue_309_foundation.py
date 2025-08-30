"""
Unit tests for Issue #309: Core Architecture Foundation

Tests the foundational interfaces, specifications, and core components
that enable all subsequent development work in the refactor.
"""

import pytest
import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, AsyncMock, patch

# Import foundation components
from src.orchestrator.foundation import (
    PipelineCompilerInterface,
    ExecutionEngineInterface,
    ModelManagerInterface,
    ToolRegistryInterface,
    QualityControlInterface,
    PipelineSpecification,
    PipelineHeader,
    PipelineStep,
    PipelineResult,
    StepResult,
    FoundationConfig
)


class TestFoundationInterfaces:
    """Test all foundation interface definitions and contracts."""
    
    def test_pipeline_compiler_interface_definition(self):
        """Test that PipelineCompilerInterface is properly defined."""
        # Verify abstract methods exist
        assert hasattr(PipelineCompilerInterface, 'compile')
        assert hasattr(PipelineCompilerInterface, 'validate')
        
        # Verify it's an abstract base class
        with pytest.raises(TypeError):
            PipelineCompilerInterface()
    
    def test_execution_engine_interface_definition(self):
        """Test that ExecutionEngineInterface is properly defined."""
        # Verify abstract methods exist
        assert hasattr(ExecutionEngineInterface, 'execute')
        assert hasattr(ExecutionEngineInterface, 'execute_step')
        assert hasattr(ExecutionEngineInterface, 'get_execution_progress')
        
        # Verify it's an abstract base class
        with pytest.raises(TypeError):
            ExecutionEngineInterface()
    
    def test_model_manager_interface_definition(self):
        """Test that ModelManagerInterface is properly defined."""
        # Verify abstract methods exist
        assert hasattr(ModelManagerInterface, 'select_model')
        assert hasattr(ModelManagerInterface, 'invoke_model')
        assert hasattr(ModelManagerInterface, 'list_available_models')
        assert hasattr(ModelManagerInterface, 'get_model_capabilities')
        
        # Verify it's an abstract base class
        with pytest.raises(TypeError):
            ModelManagerInterface()
    
    def test_tool_registry_interface_definition(self):
        """Test that ToolRegistryInterface is properly defined."""
        # Verify abstract methods exist
        assert hasattr(ToolRegistryInterface, 'register_tool')
        assert hasattr(ToolRegistryInterface, 'get_tool')
        assert hasattr(ToolRegistryInterface, 'list_available_tools')
        assert hasattr(ToolRegistryInterface, 'ensure_tool_available')
        
        # Verify it's an abstract base class
        with pytest.raises(TypeError):
            ToolRegistryInterface()
    
    def test_quality_control_interface_definition(self):
        """Test that QualityControlInterface is properly defined."""
        # Verify abstract methods exist
        assert hasattr(QualityControlInterface, 'assess_output')
        assert hasattr(QualityControlInterface, 'generate_qc_report')
        assert hasattr(QualityControlInterface, 'get_improvement_recommendations')
        
        # Verify it's an abstract base class
        with pytest.raises(TypeError):
            QualityControlInterface()


class MockPipelineCompiler(PipelineCompilerInterface):
    """Mock implementation of PipelineCompilerInterface for testing."""
    
    async def compile(self, yaml_content: str, context: Optional[Dict[str, Any]] = None) -> PipelineSpecification:
        # Simple mock implementation
        return PipelineSpecification(
            header=PipelineHeader(name="test", version="1.0"),
            steps=[PipelineStep(id="step1", action="test_action")]
        )
    
    async def validate(self, spec: PipelineSpecification) -> List[str]:
        # Mock validation - return empty list (no errors)
        return []


class MockExecutionEngine(ExecutionEngineInterface):
    """Mock implementation of ExecutionEngineInterface for testing."""
    
    def __init__(self):
        self.progress = {"status": "ready", "completed_steps": [], "current_step": None}
    
    async def execute(self, spec: PipelineSpecification, inputs: Dict[str, Any]) -> PipelineResult:
        # Mock execution
        step_results = [
            StepResult(step_id="step1", status="success", output={"result": "test_output"})
        ]
        return PipelineResult(
            pipeline_name=spec.header.name,
            status="success",
            step_results=step_results,
            total_steps=len(spec.steps)
        )
    
    async def execute_step(self, step_id: str, context: Dict[str, Any]) -> StepResult:
        return StepResult(step_id=step_id, status="success", output={"result": "step_output"})
    
    def get_execution_progress(self) -> Dict[str, Any]:
        return self.progress


class MockModelManager(ModelManagerInterface):
    """Mock implementation of ModelManagerInterface for testing."""
    
    def __init__(self):
        self.available_models = ["gpt-4", "claude-3-opus", "gemini-pro"]
    
    async def select_model(self, requirements: Dict[str, Any]) -> str:
        # Simple selection logic for testing
        if requirements.get("high_performance"):
            return "gpt-4"
        return "claude-3-opus"
    
    async def invoke_model(self, model_id: str, prompt: str, **kwargs) -> str:
        return f"Response from {model_id}: {prompt[:20]}..."
    
    def list_available_models(self) -> List[str]:
        return self.available_models
    
    async def get_model_capabilities(self, model_id: str) -> Dict[str, Any]:
        return {
            "max_tokens": 4096,
            "supports_functions": True,
            "languages": ["en"]
        }


class MockToolRegistry(ToolRegistryInterface):
    """Mock implementation of ToolRegistryInterface for testing."""
    
    def __init__(self):
        self.tools = {}
    
    async def register_tool(self, tool_name: str, tool_config: Dict[str, Any]) -> None:
        self.tools[tool_name] = tool_config
    
    async def get_tool(self, tool_name: str) -> Optional[Any]:
        return self.tools.get(tool_name)
    
    def list_available_tools(self) -> List[str]:
        return list(self.tools.keys())
    
    async def ensure_tool_available(self, tool_name: str) -> bool:
        return tool_name in self.tools


class MockQualityControl(QualityControlInterface):
    """Mock implementation of QualityControlInterface for testing."""
    
    async def assess_output(self, output: Any, criteria: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "score": 0.85,
            "criteria_met": True,
            "issues": [],
            "recommendations": []
        }
    
    async def generate_qc_report(self, result: PipelineResult) -> Dict[str, Any]:
        return {
            "overall_quality": 0.85,
            "step_quality": {step.step_id: 0.8 for step in result.step_results},
            "recommendations": ["Consider adding more validation"]
        }
    
    async def get_improvement_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        return ["Improve error handling", "Add input validation"]


class TestFoundationIntegration:
    """Test integration between foundation components."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock instances of all foundation components."""
        return {
            'compiler': MockPipelineCompiler(),
            'engine': MockExecutionEngine(),
            'models': MockModelManager(),
            'tools': MockToolRegistry(),
            'qc': MockQualityControl()
        }
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_workflow(self, mock_components):
        """Test complete workflow using all foundation components."""
        compiler = mock_components['compiler']
        engine = mock_components['engine']
        models = mock_components['models']
        tools = mock_components['tools']
        qc = mock_components['qc']
        
        # 1. Compile a pipeline
        yaml_content = """
        name: test_pipeline
        version: "1.0"
        steps:
          - id: step1
            action: test_action
        """
        
        spec = await compiler.compile(yaml_content)
        assert spec.header.name == "test_pipeline"
        assert len(spec.steps) == 1
        
        # 2. Validate the pipeline
        errors = await compiler.validate(spec)
        assert len(errors) == 0
        
        # 3. Select a model
        model_id = await models.select_model({"task": "text_generation"})
        assert model_id in ["gpt-4", "claude-3-opus", "gemini-pro"]
        
        # 4. Register and ensure tools
        await tools.register_tool("test_tool", {"type": "function"})
        tool_available = await tools.ensure_tool_available("test_tool")
        assert tool_available
        
        # 5. Execute the pipeline
        inputs = {"input": "test_input"}
        result = await engine.execute(spec, inputs)
        
        assert result.status == "success"
        assert result.pipeline_name == "test_pipeline"
        assert len(result.step_results) == 1
        
        # 6. Quality control assessment
        qc_report = await qc.generate_qc_report(result)
        assert "overall_quality" in qc_report
        assert qc_report["overall_quality"] > 0.0
    
    @pytest.mark.asyncio
    async def test_model_integration(self, mock_components):
        """Test model manager integration with execution."""
        models = mock_components['models']
        
        # Test model selection
        model_id = await models.select_model({"high_performance": True})
        assert model_id == "gpt-4"
        
        model_id = await models.select_model({"cost_effective": True})
        assert model_id == "claude-3-opus"
        
        # Test model invocation
        response = await models.invoke_model("gpt-4", "Test prompt")
        assert "Response from gpt-4" in response
        
        # Test capabilities
        capabilities = await models.get_model_capabilities("gpt-4")
        assert "max_tokens" in capabilities
        assert capabilities["supports_functions"]
    
    @pytest.mark.asyncio
    async def test_tool_registry_operations(self, mock_components):
        """Test tool registry operations."""
        tools = mock_components['tools']
        
        # Initially no tools
        assert len(tools.list_available_tools()) == 0
        
        # Register tools
        await tools.register_tool("web_search", {"type": "api", "endpoint": "search"})
        await tools.register_tool("file_reader", {"type": "function"})
        
        # Verify registration
        assert len(tools.list_available_tools()) == 2
        assert "web_search" in tools.list_available_tools()
        assert "file_reader" in tools.list_available_tools()
        
        # Test tool retrieval
        web_search_tool = await tools.get_tool("web_search")
        assert web_search_tool["type"] == "api"
        
        # Test availability checking
        assert await tools.ensure_tool_available("web_search")
        assert not await tools.ensure_tool_available("nonexistent_tool")
    
    @pytest.mark.asyncio
    async def test_quality_control_workflow(self, mock_components):
        """Test quality control assessment workflow."""
        qc = mock_components['qc']
        engine = mock_components['engine']
        
        # Create a pipeline result
        step_results = [
            StepResult(step_id="step1", status="success", output={"result": "good output"}),
            StepResult(step_id="step2", status="success", output={"result": "better output"})
        ]
        
        result = PipelineResult(
            pipeline_name="test_pipeline",
            status="success",
            step_results=step_results,
            total_steps=2
        )
        
        # Test output assessment
        assessment = await qc.assess_output(
            "good output", 
            {"min_length": 5, "language": "en"}
        )
        
        assert "score" in assessment
        assert assessment["score"] > 0.0
        
        # Test QC report generation
        qc_report = await qc.generate_qc_report(result)
        
        assert "overall_quality" in qc_report
        assert "step_quality" in qc_report
        assert len(qc_report["step_quality"]) == 2
        
        # Test improvement recommendations
        recommendations = await qc.get_improvement_recommendations(assessment)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0


class TestFoundationConfig:
    """Test foundation configuration system."""
    
    def test_foundation_config_defaults(self):
        """Test default configuration values."""
        config = FoundationConfig()
        
        # Model Management defaults
        assert config.default_model is None
        assert config.model_selection_strategy == "balanced"
        
        # Execution defaults
        assert config.max_concurrent_steps == 5
        assert config.execution_timeout == 3600
        
        # Tool Registry defaults
        assert config.auto_install_tools is True
        assert config.tool_timeout == 300
        
        # Quality Control defaults
        assert config.enable_quality_checks is True
        assert config.quality_threshold == 0.7
        
        # LangGraph Integration defaults
        assert config.enable_persistence is False
        assert config.storage_backend == "memory"
        assert config.database_url is None
        
        # Progress Monitoring defaults
        assert config.show_progress_bars is True
        assert config.log_level == "INFO"
    
    def test_foundation_config_customization(self):
        """Test custom configuration values."""
        config = FoundationConfig(
            default_model="gpt-4",
            model_selection_strategy="performance",
            max_concurrent_steps=10,
            enable_persistence=True,
            storage_backend="sqlite",
            database_url="sqlite:///test.db"
        )
        
        assert config.default_model == "gpt-4"
        assert config.model_selection_strategy == "performance"
        assert config.max_concurrent_steps == 10
        assert config.enable_persistence is True
        assert config.storage_backend == "sqlite"
        assert config.database_url == "sqlite:///test.db"


class TestPipelineSpecificationStructures:
    """Test pipeline specification data structures."""
    
    def test_pipeline_header_creation(self):
        """Test PipelineHeader creation and attributes."""
        header = PipelineHeader(
            name="test_pipeline",
            version="1.0",
            description="A test pipeline"
        )
        
        assert header.name == "test_pipeline"
        assert header.version == "1.0"
        assert header.description == "A test pipeline"
    
    def test_pipeline_step_creation(self):
        """Test PipelineStep creation and attributes."""
        step = PipelineStep(
            id="step1",
            action="test_action",
            parameters={"param1": "value1"},
            dependencies=["step0"]
        )
        
        assert step.id == "step1"
        assert step.action == "test_action"
        assert step.parameters["param1"] == "value1"
        assert "step0" in step.dependencies
    
    def test_pipeline_specification_creation(self):
        """Test PipelineSpecification creation with header and steps."""
        header = PipelineHeader(name="test", version="1.0")
        steps = [
            PipelineStep(id="step1", action="action1"),
            PipelineStep(id="step2", action="action2", dependencies=["step1"])
        ]
        
        spec = PipelineSpecification(header=header, steps=steps)
        
        assert spec.header.name == "test"
        assert len(spec.steps) == 2
        assert spec.steps[0].id == "step1"
        assert spec.steps[1].dependencies == ["step1"]
    
    def test_step_result_creation(self):
        """Test StepResult creation and attributes."""
        result = StepResult(
            step_id="step1",
            status="success",
            output={"result": "test_output"},
            execution_time=1.5,
            error_message=None
        )
        
        assert result.step_id == "step1"
        assert result.status == "success"
        assert result.output["result"] == "test_output"
        assert result.execution_time == 1.5
        assert result.error_message is None
    
    def test_pipeline_result_creation(self):
        """Test PipelineResult creation with step results."""
        step_results = [
            StepResult(step_id="step1", status="success", output={"result": "output1"}),
            StepResult(step_id="step2", status="success", output={"result": "output2"})
        ]
        
        result = PipelineResult(
            pipeline_name="test_pipeline",
            status="success",
            step_results=step_results,
            total_steps=2,
            total_execution_time=3.0
        )
        
        assert result.pipeline_name == "test_pipeline"
        assert result.status == "success"
        assert len(result.step_results) == 2
        assert result.total_steps == 2
        assert result.total_execution_time == 3.0


class TestFoundationErrorScenarios:
    """Test error handling and edge cases in foundation components."""
    
    @pytest.mark.asyncio
    async def test_compilation_error_handling(self):
        """Test error handling in pipeline compilation."""
        compiler = MockPipelineCompiler()
        
        # Test with invalid YAML
        with patch.object(compiler, 'compile') as mock_compile:
            mock_compile.side_effect = Exception("Invalid YAML")
            
            with pytest.raises(Exception):
                await compiler.compile("invalid: yaml: content:")
    
    @pytest.mark.asyncio
    async def test_execution_error_handling(self):
        """Test error handling in pipeline execution."""
        engine = MockExecutionEngine()
        
        # Test with failing step execution
        with patch.object(engine, 'execute_step') as mock_execute:
            mock_execute.side_effect = Exception("Step execution failed")
            
            with pytest.raises(Exception):
                await engine.execute_step("failing_step", {})
    
    @pytest.mark.asyncio
    async def test_model_selection_edge_cases(self):
        """Test edge cases in model selection."""
        models = MockModelManager()
        
        # Test with empty requirements
        model_id = await models.select_model({})
        assert model_id is not None
        
        # Test with conflicting requirements
        model_id = await models.select_model({
            "high_performance": True,
            "cost_effective": True
        })
        assert model_id in models.available_models


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])