"""
Orchestrator: AI pipeline orchestration framework with intelligent ambiguity resolution.

This package provides a unified interface for executing AI pipelines defined in YAML
with automatic ambiguity resolution using LLMs.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

from .compiler.yaml_compiler import YAMLCompiler
from .core.control_system import ControlSystem
from .core.model import Model, ModelCapabilities, ModelRequirements
from .core.pipeline import Pipeline
from .core.task import Task, TaskStatus
from .integrations.huggingface_model import HuggingFaceModel
from .integrations.ollama_model import OllamaModel
from .models.model_registry import ModelRegistry
from .orchestrator import Orchestrator
from .state.state_manager import StateManager
from .tools.mcp_server import default_mcp_server, default_tool_detector

__version__ = "0.1.0"

__all__ = [
    "Orchestrator",
    "Task",
    "TaskStatus",
    "Pipeline",
    "Model",
    "ModelRegistry",
    "YAMLCompiler",
    "ControlSystem",
    "ErrorHandler",
    "ResourceAllocator",
    "StateManager",
    "HuggingFaceModel",
    "OllamaModel",
    "init_models",
    "compile",
    "compile_async",
]
__author__ = "Contextual Dynamics Lab"
__email__ = "contextualdynamics@gmail.com"

# Global instances
_model_registry = None
_orchestrator = None


def init_models(config_path: str = "models.yaml") -> ModelRegistry:
    """Initialize the pool of available models by reading models.yaml and environment."""
    global _model_registry

    from .utils.model_utils import (
        load_model_config,
        parse_model_size,
        check_ollama_installed,
    )
    from .integrations.anthropic_model import AnthropicModel
    from .integrations.google_model import GoogleModel
    from .integrations.openai_model import OpenAIModel
    import os

    print(">> Initializing model pool...")

    _model_registry = ModelRegistry()

    # Load model configuration
    config = load_model_config(config_path)
    models_config = config.get("models", [])

    # Check if Ollama is installed
    ollama_available = check_ollama_installed()
    if not ollama_available:
        print(">> ⚠️  Ollama not found - Ollama models will not be available")
        print(">>    Install from: https://ollama.ai")

    # Process each model in configuration
    for model_config in models_config:
        source = model_config.get("source")
        name = model_config.get("name")
        expertise = model_config.get("expertise", ["general"])
        size_str = model_config.get("size")

        if not source or not name:
            continue

        # Parse model size
        size_billions = parse_model_size(name, size_str)

        try:
            if source == "ollama":
                if not ollama_available:
                    continue

                # Register model for lazy loading (will be downloaded on first use)
                # Use a lazy wrapper that doesn't check availability yet
                from .integrations.lazy_ollama_model import LazyOllamaModel

                model = LazyOllamaModel(model_name=name, timeout=60)
                # Add dynamic attributes for model selection
                setattr(model, "_expertise", expertise)
                setattr(model, "_size_billions", size_billions)
                _model_registry.register_model(model)
                print(
                    f">>   📦 Registered Ollama model: {name} ({size_billions}B params) - will download on first use"
                )

            elif source == "huggingface":
                # Check if transformers is available
                try:
                    import importlib.util

                    if importlib.util.find_spec("transformers") is not None:
                        # Register for lazy loading (will be downloaded on first use)
                        from .integrations.lazy_huggingface_model import (
                            LazyHuggingFaceModel,
                        )

                    hf_model = LazyHuggingFaceModel(model_name=name)
                    # Add dynamic attributes for model selection
                    setattr(hf_model, "_expertise", expertise)
                    setattr(hf_model, "_size_billions", size_billions)
                    _model_registry.register_model(hf_model)
                    print(
                        f">>   📦 Registered HuggingFace model: {name} ({size_billions}B params) - will download on first use"
                    )
                except ImportError:
                    print(
                        f">>   ⚠️  HuggingFace model {name} configured but transformers not installed"
                    )
                    print(">>      Install with: pip install transformers torch")
                except Exception as e:
                    print(f">>   ⚠️  Failed to register HuggingFace model {name}: {e}")

            elif source == "openai" and os.environ.get("OPENAI_API_KEY"):
                # Only register if API key is available
                model = OpenAIModel(name=name, model=name)
                # Add dynamic attributes for model selection
                setattr(model, "_expertise", expertise)
                setattr(model, "_size_billions", size_billions)
                _model_registry.register_model(model)
                print(
                    f">>   ✅ Registered OpenAI model: {name} ({size_billions}B params)"
                )

            elif source == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
                # Only register if API key is available
                model = AnthropicModel(name=name, model=name)
                # Add dynamic attributes for model selection
                setattr(model, "_expertise", expertise)
                setattr(model, "_size_billions", size_billions)
                _model_registry.register_model(model)
                print(
                    f">>   ✅ Registered Anthropic model: {name} ({size_billions}B params)"
                )

            elif source == "google" and os.environ.get("GOOGLE_API_KEY"):
                # Only register if API key is available
                model = GoogleModel(name=name, model=name)
                # Add dynamic attributes for model selection
                setattr(model, "_expertise", expertise)
                setattr(model, "_size_billions", size_billions)
                _model_registry.register_model(model)
                print(
                    f">>   ✅ Registered Google model: {name} ({size_billions}B params)"
                )

        except Exception as e:
            print(f">>   ⚠️  Error registering {source} model {name}: {e}")

    print(
        f"\n>> Model initialization complete: {len(_model_registry.list_models())} models registered"
    )

    if not _model_registry.list_models():
        print(">>   ⚠️  No models available - using mock fallback")

    # Store defaults in registry for later use
    setattr(_model_registry, "_defaults", config.get("defaults", {}))

    return _model_registry


class OrchestratorPipeline:
    """Wrapper for compiled pipeline that can be called with keyword arguments."""

    def __init__(
        self, pipeline: Pipeline, compiler: YAMLCompiler, orchestrator: Orchestrator
    ):
        self.pipeline = pipeline
        self.compiler = compiler
        self.orchestrator = orchestrator
        self._print_usage()

    def _print_usage(self) -> None:
        """Print keyword arguments as shown in README."""
        print(">> keyword arguments:")

        # Extract inputs from the raw pipeline definition
        inputs = self._extract_inputs()

        if inputs:
            for name, info in inputs.items():
                if isinstance(info, dict):
                    desc = info.get("description", "No description")
                    type_str = info.get("type", "String").title()
                    required = " (required)" if info.get("required", False) else ""
                    print(f">>   {name}: {desc} (type: {type_str}){required}")
                else:
                    # Simple string description
                    print(f">>   {name}: {info} (type: String)")
        else:
            # Default inputs for research report
            print(
                ">>   topic: a word or underscore-separated phrase specifying the to-be-researched topic (type: String)"
            )
            print(
                ">>   instructions: detailed instructions to help guide the report, specify areas of particular interest (or areas to stay away from), etc. (type: String)"
            )

    def _extract_inputs(self) -> Dict[str, Any]:
        """Extract input definitions from the compiled pipeline."""
        # The inputs are stored in the pipeline's metadata during compilation
        if hasattr(self.pipeline, "metadata") and "inputs" in self.pipeline.metadata:
            return self.pipeline.metadata["inputs"]

        # If not in metadata, try to get from the original definition
        # This is a fallback - we should enhance the compilation process
        return {}

    def run(self, **kwargs: Any) -> Any:
        """Run the pipeline with given keyword arguments."""
        # Run pipeline asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._run_async(**kwargs))
        finally:
            loop.close()

    async def _run_async(self, **kwargs):
        """Async pipeline execution."""
        # Validate required inputs
        self._validate_inputs(kwargs)

        # Resolve outputs using AUTO tags
        outputs = await self._resolve_outputs(kwargs)

        # Create full context from kwargs and resolved outputs
        context = {"inputs": kwargs, "outputs": outputs}

        # Apply runtime template resolution to pipeline tasks
        resolved_pipeline = await self._resolve_runtime_templates(
            self.pipeline, context
        )

        # Execute pipeline
        # Set the resolved pipeline's context
        resolved_pipeline.context = context
        results = await self.orchestrator.execute_pipeline(resolved_pipeline)

        # Return the final output (PDF path or report content)
        if outputs and "pdf" in outputs:
            return outputs["pdf"]
        elif "final_report" in results:
            return results["final_report"]
        else:
            return results

    def _validate_inputs(self, kwargs):
        """Validate that required inputs are provided."""
        inputs_def = self._extract_inputs()

        for name, info in inputs_def.items():
            if isinstance(info, dict) and info.get("required", False):
                if name not in kwargs:
                    raise ValueError(f"Required input '{name}' not provided")

    async def _resolve_outputs(self, inputs):
        """Resolve output definitions using AUTO tags."""
        outputs = {}
        outputs_def = self._extract_outputs()

        if outputs_def:
            from jinja2 import Template

            for name, value in outputs_def.items():
                if isinstance(value, str):
                    if value.startswith("<AUTO>") and value.endswith("</AUTO>"):
                        # Resolve AUTO tag
                        auto_content = value[6:-7]  # Remove <AUTO> tags
                        if hasattr(
                            self.orchestrator.yaml_compiler, "ambiguity_resolver"
                        ):
                            resolved = await self.orchestrator.yaml_compiler.ambiguity_resolver.resolve(
                                auto_content, f"outputs.{name}"
                            )
                            outputs[name] = resolved
                        else:
                            outputs[name] = (
                                f"report_{inputs.get('topic', 'research')}.pdf"
                            )
                    else:
                        # Regular template - render with current context
                        try:
                            template = Template(value)
                            outputs[name] = template.render(
                                inputs=inputs, outputs=outputs
                            )
                        except Exception:
                            outputs[name] = value
                else:
                    outputs[name] = value

        return outputs

    def _extract_outputs(self):
        """Extract output definitions from the compiled pipeline."""
        if hasattr(self.pipeline, "metadata") and "outputs" in self.pipeline.metadata:
            return self.pipeline.metadata["outputs"]
        return {}

    async def _resolve_runtime_templates(
        self, pipeline: Pipeline, context: Dict[str, Any]
    ) -> Pipeline:
        """Resolve templates in pipeline tasks at runtime."""
        import copy

        # Create a deep copy to avoid modifying the original
        resolved_pipeline = copy.deepcopy(pipeline)

        # Resolve templates in each task
        for task_id, task in resolved_pipeline.tasks.items():
            if hasattr(task, "parameters"):
                task.parameters = await self._resolve_task_templates(
                    task.parameters, context
                )

        return resolved_pipeline

    async def _resolve_task_templates(self, obj, context):
        """Recursively resolve templates in task parameters."""
        from jinja2 import Template

        if isinstance(obj, str):
            if "{{" in obj and "}}" in obj:
                try:
                    template = Template(obj)
                    return template.render(**context)
                except Exception:
                    # If template resolution fails, return original
                    return obj
            return obj
        elif isinstance(obj, dict):
            return {
                k: await self._resolve_task_templates(v, context)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [await self._resolve_task_templates(item, context) for item in obj]
        else:
            return obj


async def compile_async(yaml_path: str) -> "OrchestratorPipeline":
    """Compile a YAML pipeline file into an executable OrchestratorPipeline (async version)."""
    global _orchestrator, _model_registry

    # Ensure models are initialized
    if _model_registry is None:
        init_models()

    # Create orchestrator with mock control system (will be replaced)
    from .core.control_system import MockControlSystem

    control_system = MockControlSystem()
    _orchestrator = Orchestrator(control_system=control_system)

    # Set up model for ambiguity resolution
    model_keys = _model_registry.list_models() if _model_registry else []
    if model_keys:
        best_model_key = model_keys[0]  # Assume first is best (gemma2:27b)
        # Get model object directly from the models dict
        best_model = _model_registry.models[best_model_key]
        _orchestrator.yaml_compiler.ambiguity_resolver.model = best_model
        print(f">> Using model for AUTO resolution: {best_model_key}")

    # Load and compile YAML
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Pipeline file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        yaml_content = f.read()

    # Parse YAML to detect required tools
    import yaml as yaml_lib

    pipeline_def = yaml_lib.safe_load(yaml_content)

    # Auto-detect and register tools
    required_tools = default_tool_detector.detect_tools_from_yaml(pipeline_def)
    if required_tools:
        print(f">> Detected required tools: {', '.join(required_tools)}")
        availability = default_tool_detector.ensure_tools_available(required_tools)
        for tool, available in availability.items():
            status = "✅" if available else "❌"
            print(f">>   {status} {tool}")

    # Start MCP server if tools are required
    if required_tools and any(availability.values()):
        print(">> Starting MCP tool server...")
        await default_mcp_server.start_server()

    # Compile pipeline
    pipeline = await _orchestrator.yaml_compiler.compile(yaml_content, {})

    # Return callable pipeline
    return OrchestratorPipeline(pipeline, _orchestrator.yaml_compiler, _orchestrator)


def compile(yaml_path: str) -> "OrchestratorPipeline":
    """Compile a YAML pipeline file into an executable OrchestratorPipeline."""
    # Check if we're already in an event loop
    try:
        asyncio.get_running_loop()
        # We're in an async context, need to run in a new thread or return a coroutine
        import concurrent.futures

        def run_in_new_loop():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(compile_async(yaml_path))
            finally:
                new_loop.close()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_new_loop)
            return future.result()

    except RuntimeError:
        # No event loop running, we can create one
        return asyncio.run(compile_async(yaml_path))


__all__ = [
    "Task",
    "TaskStatus",
    "Pipeline",
    "Model",
    "ModelCapabilities",
    "ModelRequirements",
    "ControlSystem",
    "YAMLCompiler",
    "ModelRegistry",
    "StateManager",
    "Orchestrator",
    "init_models",
    "compile",
]
