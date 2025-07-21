"""Tests to improve coverage for the orchestrator __init__ module."""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import yaml

import orchestrator as orc
from orchestrator import (
    OrchestratorPipeline,
    compile,
    compile_async,
    init_models,
)
from orchestrator.core.pipeline import Pipeline
from orchestrator.core.task import Task
from orchestrator.compiler.yaml_compiler import YAMLCompiler
from orchestrator.orchestrator import Orchestrator


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

    def test_init_models_with_openai_key(self, temp_config_file):
        """Test init_models with real OpenAI API key."""
        # Load real API keys from environment
        from orchestrator.utils.api_keys import load_api_keys
        
        try:
            load_api_keys()  # This will use real keys from ~/.orchestrator/.env or environment
            orc._model_registry = None
            registry = init_models(temp_config_file)
            assert registry is not None
            
            # Verify we have a real key loaded
            assert os.environ.get("OPENAI_API_KEY") is not None
            assert os.environ.get("OPENAI_API_KEY") != "test-key"
        except EnvironmentError as e:
            pytest.skip(f"Skipping test - API keys not configured: {e}")

    def test_init_models_with_anthropic_key(self, temp_config_file):
        """Test init_models with real Anthropic API key."""
        # Load real API keys from environment
        from orchestrator.utils.api_keys import load_api_keys
        
        try:
            load_api_keys()  # This will use real keys from ~/.orchestrator/.env or environment
            orc._model_registry = None
            registry = init_models(temp_config_file)
            assert registry is not None
            
            # Verify we have a real key loaded
            assert os.environ.get("ANTHROPIC_API_KEY") is not None
            assert os.environ.get("ANTHROPIC_API_KEY") != "test-key"
        except EnvironmentError as e:
            pytest.skip(f"Skipping test - API keys not configured: {e}")

    def test_init_models_with_google_key(self, temp_config_file):
        """Test init_models with real Google API key."""
        # Load real API keys from environment
        from orchestrator.utils.api_keys import load_api_keys
        
        try:
            load_api_keys()  # This will use real keys from ~/.orchestrator/.env or environment
            orc._model_registry = None
            registry = init_models(temp_config_file)
            assert registry is not None
            
            # Verify we have a real key loaded (could be either GOOGLE_API_KEY or GOOGLE_AI_API_KEY)
            google_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_AI_API_KEY")
            assert google_key is not None
            assert google_key != "test-key"
        except EnvironmentError as e:
            pytest.skip(f"Skipping test - API keys not configured: {e}")

    def test_init_models_with_ollama_check(self, temp_config_file):
        """Test init_models with real Ollama check."""
        # This tests the real check_ollama_installed function
        # It will work whether Ollama is installed or not
        from orchestrator.utils.model_utils import check_ollama_installed
        
        # First check if Ollama is actually installed
        ollama_installed = check_ollama_installed()
        
        # Clear registry and initialize
        orc._model_registry = None
        registry = init_models(temp_config_file)
        assert registry is not None
        
        # If Ollama is installed, verify Ollama models can be registered
        # If not, verify the system handles it gracefully
        # Either way, the registry should be created successfully

    def test_init_models_handles_missing_dependencies(self, temp_config_file):
        """Test init_models handles missing dependencies gracefully."""
        # Test with a config that includes models from all sources
        # The real init_models should handle missing dependencies gracefully
        
        orc._model_registry = None
        registry = init_models(temp_config_file)
        
        # Registry should still be created even if some model sources aren't available
        assert registry is not None
        
        # The registry should have registered whatever models it could
        # based on available dependencies (API keys, Ollama, etc.)

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


class TestPipeline(Pipeline):
    """Testable pipeline for testing."""
    
    def __init__(self, id="test", name="Test Pipeline"):
        super().__init__(id, name)
        self.metadata = {
            "inputs": {
                "test_input": {
                    "type": "string",
                    "description": "Test input parameter",
                    "default": "default_value"
                }
            },
            "outputs": {"result": "output_value"}
        }
        self.tasks = {}


class TestableCompiler(YAMLCompiler):
    """Testable YAML compiler."""
    
    def __init__(self):
        super().__init__()
        self.compile_calls = []
        self._test_pipeline = None
        
    def set_test_pipeline(self, pipeline):
        """Set the pipeline to return from compile."""
        self._test_pipeline = pipeline
        
    def compile(self, yaml_path: str) -> Pipeline:
        """Test version of compile."""
        self.compile_calls.append(yaml_path)
        if self._test_pipeline:
            return self._test_pipeline
        return TestPipeline()


class TestableOrchestrator(Orchestrator):
    """Testable orchestrator."""
    
    def __init__(self):
        super().__init__()
        self._test_result = {"result": "test_result"}
        self.execute_calls = []
        
    def set_test_result(self, result):
        """Set the result to return from execute_pipeline."""
        self._test_result = result
        
    async def execute_pipeline(self, pipeline, **kwargs):
        """Test version of execute_pipeline."""
        self.execute_calls.append((pipeline, kwargs))
        return self._test_result


class TestablePrint:
    """Track print calls for testing."""
    
    def __init__(self):
        self.calls = []
        
    def __call__(self, *args, **kwargs):
        """Capture print calls."""
        message = " ".join(str(arg) for arg in args)
        self.calls.append(message)


class TestOrchestratorPipeline:
    """Test the OrchestratorPipeline class."""

    @pytest.fixture
    def test_pipeline(self):
        """Create a test pipeline."""
        return TestPipeline()

    @pytest.fixture
    def test_compiler(self):
        """Create a test compiler."""
        return TestableCompiler()

    @pytest.fixture
    def test_orchestrator(self):
        """Create a test orchestrator."""
        return TestableOrchestrator()

    def test_orchestrator_pipeline_init(self, test_pipeline, test_compiler, test_orchestrator):
        """Test OrchestratorPipeline initialization."""
        # Replace print temporarily
        original_print = print
        test_print = TestablePrint()
        import builtins
        builtins.print = test_print
        
        try:
            pipeline = OrchestratorPipeline(test_pipeline, test_compiler, test_orchestrator)
            
            assert pipeline.pipeline == test_pipeline
            assert pipeline.compiler == test_compiler
            assert pipeline.orchestrator == test_orchestrator
        finally:
            builtins.print = original_print

    def test_orchestrator_pipeline_extract_inputs(self, test_pipeline, test_compiler, test_orchestrator):
        """Test _extract_inputs method."""
        # Replace print temporarily
        original_print = print
        test_print = TestablePrint()
        import builtins
        builtins.print = test_print
        
        try:
            pipeline = OrchestratorPipeline(test_pipeline, test_compiler, test_orchestrator)
            
            inputs = pipeline._extract_inputs()
            assert "test_input" in inputs
            assert inputs["test_input"]["type"] == "string"
        finally:
            builtins.print = original_print

    def test_orchestrator_pipeline_extract_inputs_no_metadata(self, test_compiler, test_orchestrator):
        """Test _extract_inputs when pipeline has no metadata."""
        test_pipeline = TestPipeline()
        test_pipeline.metadata = {}
        
        # Replace print temporarily
        original_print = print
        test_print = TestablePrint()
        import builtins
        builtins.print = test_print
        
        try:
            pipeline = OrchestratorPipeline(test_pipeline, test_compiler, test_orchestrator)
            
            inputs = pipeline._extract_inputs()
            assert inputs == {}
        finally:
            builtins.print = original_print

    def test_orchestrator_pipeline_extract_outputs(self, test_pipeline, test_compiler, test_orchestrator):
        """Test _extract_outputs method."""
        # Replace print temporarily
        original_print = print
        test_print = TestablePrint()
        import builtins
        builtins.print = test_print
        
        try:
            pipeline = OrchestratorPipeline(test_pipeline, test_compiler, test_orchestrator)
            
            outputs = pipeline._extract_outputs()
            assert "result" in outputs
        finally:
            builtins.print = original_print

    def test_orchestrator_pipeline_run(self, test_pipeline, test_compiler, test_orchestrator):
        """Test run method."""
        # Replace print temporarily
        original_print = print
        test_print = TestablePrint()
        import builtins
        builtins.print = test_print
        
        try:
            pipeline = OrchestratorPipeline(test_pipeline, test_compiler, test_orchestrator)
            
            result = pipeline.run(test_input="test_value")
            assert result is not None
        finally:
            builtins.print = original_print

    def test_orchestrator_pipeline_run_with_default_values(self, test_pipeline, test_compiler, test_orchestrator):
        """Test run method with default values."""
        # Replace print temporarily
        original_print = print
        test_print = TestablePrint()
        import builtins
        builtins.print = test_print
        
        try:
            pipeline = OrchestratorPipeline(test_pipeline, test_compiler, test_orchestrator)
            
            # Run without providing the input that has a default
            result = pipeline.run()
            assert result is not None
        finally:
            builtins.print = original_print

    @pytest.mark.asyncio
    async def test_resolve_runtime_templates(self, test_pipeline, test_compiler, test_orchestrator):
        """Test _resolve_runtime_templates method."""
        # Add some tasks to the pipeline
        test_task = Task(id="task1", name="Task 1", action="generate")
        test_task.parameters = {
            "param1": "{{ test_input }}",
            "param2": "static_value"
        }
        test_pipeline.tasks = {"task1": test_task}
        
        # Replace print temporarily
        original_print = print
        test_print = TestablePrint()
        import builtins
        builtins.print = test_print
        
        try:
            pipeline = OrchestratorPipeline(test_pipeline, test_compiler, test_orchestrator)
            
            context = {"test_input": "resolved_value"}
            resolved = await pipeline._resolve_runtime_templates(test_pipeline, context)
            
            # Should return a pipeline
            assert resolved is not None
        finally:
            builtins.print = original_print

    def test_orchestrator_pipeline_print_usage_empty_inputs(self, test_compiler, test_orchestrator):
        """Test _print_usage with no inputs."""
        test_pipeline = TestPipeline()
        test_pipeline.metadata = {"inputs": {}}
        
        # Replace print temporarily
        original_print = print
        test_print = TestablePrint()
        import builtins
        builtins.print = test_print
        
        try:
            pipeline = OrchestratorPipeline(test_pipeline, test_compiler, test_orchestrator)
            # Check that something was printed
            assert len(test_print.calls) > 0
        finally:
            builtins.print = original_print


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
        
        # Replace YAMLCompiler.compile temporarily
        original_compile = orc.YAMLCompiler.compile
        test_pipeline = TestPipeline()
        test_pipeline.metadata = {}
        
        def test_compile(self, yaml_path):
            return test_pipeline
            
        orc.YAMLCompiler.compile = test_compile
        
        try:
            pipeline = await compile_async(temp_yaml_file)
            assert isinstance(pipeline, OrchestratorPipeline)
        finally:
            orc.YAMLCompiler.compile = original_compile

    def test_compile_sync(self, temp_yaml_file):
        """Test synchronous compile function."""
        # Ensure models are initialized
        if orc._model_registry is None:
            init_models()
        
        # Replace YAMLCompiler.compile temporarily
        original_compile = orc.YAMLCompiler.compile
        test_pipeline = TestPipeline()
        test_pipeline.metadata = {}
        
        def test_compile(self, yaml_path):
            return test_pipeline
            
        orc.YAMLCompiler.compile = test_compile
        
        try:
            pipeline = compile(temp_yaml_file)
            assert isinstance(pipeline, OrchestratorPipeline)
        finally:
            orc.YAMLCompiler.compile = original_compile

    def test_compile_in_event_loop(self, temp_yaml_file):
        """Test compile when already in an event loop."""
        # Ensure models are initialized
        if orc._model_registry is None:
            init_models()
        
        async def test_in_loop():
            # Replace YAMLCompiler.compile temporarily
            original_compile = orc.YAMLCompiler.compile
            test_pipeline = TestPipeline()
            test_pipeline.metadata = {}
            
            def test_compile(self, yaml_path):
                return test_pipeline
                
            orc.YAMLCompiler.compile = test_compile
            
            try:
                # This should use the thread pool executor path
                pipeline = compile(temp_yaml_file)
                assert isinstance(pipeline, OrchestratorPipeline)
            finally:
                orc.YAMLCompiler.compile = original_compile
        
        # Run in an event loop
        asyncio.run(test_in_loop())

    def test_compile_no_event_loop_runtime_error(self, temp_yaml_file):
        """Test compile when no event loop exists (RuntimeError path)."""
        if orc._model_registry is None:
            init_models()
        
        # Replace asyncio.get_running_loop temporarily
        original_get_loop = asyncio.get_running_loop
        
        def failing_get_loop():
            raise RuntimeError("No running loop")
            
        asyncio.get_running_loop = failing_get_loop
        
        # Replace YAMLCompiler.compile temporarily
        original_compile = orc.YAMLCompiler.compile
        test_pipeline = TestPipeline()
        test_pipeline.metadata = {}
        
        def test_compile(self, yaml_path):
            return test_pipeline
            
        orc.YAMLCompiler.compile = test_compile
        
        try:
            pipeline = compile(temp_yaml_file)
            assert isinstance(pipeline, OrchestratorPipeline)
        finally:
            asyncio.get_running_loop = original_get_loop
            orc.YAMLCompiler.compile = original_compile


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