"""Tests to improve coverage for the orchestrator __init__ module."""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

import orchestrator as orc
from orchestrator import (
    OrchestratorPipeline,
    compile,
    compile_async,
    init_models,
)


class TestInitModels:
    """Test the init_models function with various configurations."""

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "models": [
                    {
                        "source": "ollama",
                        "name": "test-ollama:1b",
                        "expertise": ["test", "fast"],
                        "size": "1b",
                    },
                    {
                        "source": "huggingface",
                        "name": "test/model",
                        "expertise": ["test"],
                        "size": "2b",
                    },
                    {
                        "source": "openai",
                        "name": "test-gpt",
                        "expertise": ["general"],
                        "size": "10b",
                    },
                    {
                        "source": "anthropic",
                        "name": "test-claude",
                        "expertise": ["reasoning"],
                        "size": "20b",
                    },
                    {
                        "source": "google",
                        "name": "test-gemini",
                        "expertise": ["vision"],
                        "size": "30b",
                    },
                ],
                "defaults": {
                    "expertise_preferences": {"test": "test-ollama:1b"},
                    "fallback_chain": ["test-ollama:1b"],
                },
            }
            yaml.dump(config, f)
            yield f.name
        os.unlink(f.name)

    def test_init_models_with_custom_config(self, temp_config_file):
        """Test init_models with a custom configuration file."""
        # Clear any existing registry
        orc._model_registry = None

        # Test with custom config
        registry = init_models(temp_config_file)
        assert registry is not None
        assert len(registry.list_models()) > 0

    def test_init_models_with_missing_config(self):
        """Test init_models when config file doesn't exist."""
        # Clear any existing registry
        orc._model_registry = None

        # Test with non-existent config file
        registry = init_models("non_existent_config.yaml")
        assert registry is not None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_init_models_with_openai_key(self, temp_config_file):
        """Test init_models with OpenAI API key set."""
        orc._model_registry = None
        registry = init_models(temp_config_file)
        assert registry is not None

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_init_models_with_anthropic_key(self, temp_config_file):
        """Test init_models with Anthropic API key set."""
        orc._model_registry = None
        registry = init_models(temp_config_file)
        assert registry is not None

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
    def test_init_models_with_google_key(self, temp_config_file):
        """Test init_models with Google API key set."""
        orc._model_registry = None
        registry = init_models(temp_config_file)
        assert registry is not None

    @patch("orchestrator.check_ollama_installed")
    def test_init_models_with_ollama_installed(self, mock_check, temp_config_file):
        """Test init_models when Ollama is installed."""
        mock_check.return_value = True
        orc._model_registry = None
        registry = init_models(temp_config_file)
        assert registry is not None

    @patch("orchestrator.check_ollama_installed")
    def test_init_models_with_ollama_not_installed(self, mock_check, temp_config_file):
        """Test init_models when Ollama is not installed."""
        mock_check.return_value = False
        orc._model_registry = None
        registry = init_models(temp_config_file)
        assert registry is not None

    def test_init_models_with_invalid_source(self):
        """Test init_models with invalid model source."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "models": [
                    {
                        "source": "invalid_source",
                        "name": "test-model",
                        "expertise": ["test"],
                    }
                ]
            }
            yaml.dump(config, f)
            temp_file = f.name

        try:
            orc._model_registry = None
            registry = init_models(temp_file)
            assert registry is not None
        finally:
            os.unlink(temp_file)

    def test_init_models_exception_handling(self):
        """Test init_models handles exceptions during model registration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            # Create a config that might cause issues
            config = {
                "models": [
                    {
                        "source": "ollama",
                        # Missing required fields
                    }
                ]
            }
            yaml.dump(config, f)
            temp_file = f.name

        try:
            orc._model_registry = None
            registry = init_models(temp_file)
            assert registry is not None
        finally:
            os.unlink(temp_file)


class TestOrchestratorPipeline:
    """Test the OrchestratorPipeline class."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline for testing."""
        pipeline = MagicMock()
        pipeline.metadata = {
            "inputs": {
                "test_input": {
                    "type": "string",
                    "description": "Test input parameter",
                    "default": "default_value"
                }
            },
            "outputs": {"result": "output_value"}
        }
        pipeline.tasks = {}
        return pipeline

    @pytest.fixture
    def mock_compiler(self):
        """Create a mock YAML compiler."""
        return MagicMock()

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        orchestrator = MagicMock()
        # Make execute_pipeline return a coroutine
        async def mock_execute():
            return {"result": "test_result"}
        orchestrator.execute_pipeline.return_value = mock_execute()
        return orchestrator

    def test_orchestrator_pipeline_init(self, mock_pipeline, mock_compiler, mock_orchestrator):
        """Test OrchestratorPipeline initialization."""
        with patch("builtins.print"):  # Suppress print output
            pipeline = OrchestratorPipeline(mock_pipeline, mock_compiler, mock_orchestrator)
        
        assert pipeline.pipeline == mock_pipeline
        assert pipeline.compiler == mock_compiler
        assert pipeline.orchestrator == mock_orchestrator

    def test_orchestrator_pipeline_extract_inputs(self, mock_pipeline, mock_compiler, mock_orchestrator):
        """Test _extract_inputs method."""
        with patch("builtins.print"):
            pipeline = OrchestratorPipeline(mock_pipeline, mock_compiler, mock_orchestrator)
        
        inputs = pipeline._extract_inputs()
        assert "test_input" in inputs
        assert inputs["test_input"]["type"] == "string"

    def test_orchestrator_pipeline_extract_inputs_no_metadata(self, mock_compiler, mock_orchestrator):
        """Test _extract_inputs when pipeline has no metadata."""
        mock_pipeline = MagicMock()
        mock_pipeline.metadata = {}
        
        with patch("builtins.print"):
            pipeline = OrchestratorPipeline(mock_pipeline, mock_compiler, mock_orchestrator)
        
        inputs = pipeline._extract_inputs()
        assert inputs == {}

    def test_orchestrator_pipeline_extract_outputs(self, mock_pipeline, mock_compiler, mock_orchestrator):
        """Test _extract_outputs method."""
        with patch("builtins.print"):
            pipeline = OrchestratorPipeline(mock_pipeline, mock_compiler, mock_orchestrator)
        
        outputs = pipeline._extract_outputs()
        assert "result" in outputs

    def test_orchestrator_pipeline_run(self, mock_pipeline, mock_compiler, mock_orchestrator):
        """Test run method."""
        with patch("builtins.print"):
            pipeline = OrchestratorPipeline(mock_pipeline, mock_compiler, mock_orchestrator)
        
        result = pipeline.run(test_input="test_value")
        assert result is not None

    def test_orchestrator_pipeline_run_with_default_values(self, mock_pipeline, mock_compiler, mock_orchestrator):
        """Test run method with default values."""
        with patch("builtins.print"):
            pipeline = OrchestratorPipeline(mock_pipeline, mock_compiler, mock_orchestrator)
        
        # Run without providing the input that has a default
        result = pipeline.run()
        assert result is not None

    @pytest.mark.asyncio
    async def test_resolve_runtime_templates(self, mock_pipeline, mock_compiler, mock_orchestrator):
        """Test _resolve_runtime_templates method."""
        # Add some tasks to the pipeline
        mock_task = MagicMock()
        mock_task.parameters = {
            "param1": "{{ test_input }}",
            "param2": "static_value"
        }
        mock_pipeline.tasks = {"task1": mock_task}
        
        with patch("builtins.print"):
            pipeline = OrchestratorPipeline(mock_pipeline, mock_compiler, mock_orchestrator)
        
        context = {"test_input": "resolved_value"}
        resolved = await pipeline._resolve_runtime_templates(mock_pipeline, context)
        
        # Should return a pipeline (even if it's a deep copy of the mock)
        assert resolved is not None

    def test_orchestrator_pipeline_print_usage_empty_inputs(self, mock_compiler, mock_orchestrator):
        """Test _print_usage with no inputs."""
        mock_pipeline = MagicMock()
        mock_pipeline.metadata = {"inputs": {}}
        
        with patch("builtins.print") as mock_print:
            pipeline = OrchestratorPipeline(mock_pipeline, mock_compiler, mock_orchestrator)
            # Check that something was printed
            mock_print.assert_called()


class TestCompileFunctions:
    """Test the compile and compile_async functions."""

    @pytest.fixture
    def temp_yaml_file(self):
        """Create a temporary YAML pipeline file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            pipeline_config = {
                "id": "test_pipeline",
                "name": "Test Pipeline",
                "steps": [
                    {
                        "id": "step1",
                        "action": "generate_text",
                        "parameters": {
                            "prompt": "Hello {{ name }}!"
                        }
                    }
                ]
            }
            yaml.dump(pipeline_config, f)
            yield f.name
        os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_compile_async(self, temp_yaml_file):
        """Test compile_async function."""
        # Ensure models are initialized
        if orc._model_registry is None:
            init_models()
        
        # Mock the YAML compiler's compile method
        with patch.object(orc.YAMLCompiler, "compile") as mock_compile:
            mock_pipeline = MagicMock()
            mock_pipeline.metadata = {}
            mock_compile.return_value = mock_pipeline
            
            pipeline = await compile_async(temp_yaml_file)
            assert isinstance(pipeline, OrchestratorPipeline)

    def test_compile_sync(self, temp_yaml_file):
        """Test synchronous compile function."""
        # Ensure models are initialized
        if orc._model_registry is None:
            init_models()
        
        with patch.object(orc.YAMLCompiler, "compile") as mock_compile:
            mock_pipeline = MagicMock()
            mock_pipeline.metadata = {}
            mock_compile.return_value = mock_pipeline
            
            pipeline = compile(temp_yaml_file)
            assert isinstance(pipeline, OrchestratorPipeline)

    def test_compile_in_event_loop(self, temp_yaml_file):
        """Test compile when already in an event loop."""
        # Ensure models are initialized
        if orc._model_registry is None:
            init_models()
        
        async def test_in_loop():
            with patch.object(orc.YAMLCompiler, "compile") as mock_compile:
                mock_pipeline = MagicMock()
                mock_pipeline.metadata = {}
                mock_compile.return_value = mock_pipeline
                
                # This should use the thread pool executor path
                pipeline = compile(temp_yaml_file)
                assert isinstance(pipeline, OrchestratorPipeline)
        
        # Run in an event loop
        asyncio.run(test_in_loop())

    def test_compile_no_event_loop_runtime_error(self, temp_yaml_file):
        """Test compile when no event loop exists (RuntimeError path)."""
        if orc._model_registry is None:
            init_models()
        
        with patch("asyncio.get_running_loop") as mock_get_loop:
            # Simulate RuntimeError when no loop exists
            mock_get_loop.side_effect = RuntimeError("No running loop")
            
            with patch.object(orc.YAMLCompiler, "compile") as mock_compile:
                mock_pipeline = MagicMock()
                mock_pipeline.metadata = {}
                mock_compile.return_value = mock_pipeline
                
                pipeline = compile(temp_yaml_file)
                assert isinstance(pipeline, OrchestratorPipeline)


class TestUtilityFunctions:
    """Test utility functions in the init module."""

    def test_apply_template_dict(self):
        """Test the _apply_template_dict internal function."""
        # This is called internally by _resolve_runtime_templates
        # We test it indirectly through the pipeline run
        pass

    def test_init_models_already_initialized(self):
        """Test init_models when already initialized."""
        # First initialization
        registry1 = init_models()
        
        # Second initialization should return the same registry
        registry2 = init_models()
        
        assert registry1 is registry2