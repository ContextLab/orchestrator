"""
Orchestrator: AI pipeline orchestration framework with intelligent ambiguity resolution.

This package provides a unified interface for executing AI pipelines defined in YAML
with automatic ambiguity resolution using LLMs.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict

from .compiler.yaml_compiler import YAMLCompiler
from .compiler.control_flow_compiler import ControlFlowCompiler
from .core.control_system import ControlSystem
from .core.model import Model, ModelCapabilities, ModelRequirements
from .core.pipeline import Pipeline
from .core.pipeline_status_tracker import PipelineStatusTracker
from .core.pipeline_resume_manager import PipelineResumeManager, ResumeStrategy
from .core.task import Task, TaskStatus
from .integrations.huggingface_model import HuggingFaceModel
from .integrations.ollama_model import OllamaModel
from .models.model_registry import ModelRegistry
from .models.registry_singleton import get_model_registry
from .orchestrator import Orchestrator
from .state.state_manager import StateManager
from .tools.mcp_server import default_mcp_server, default_tool_detector
from .control_flow import (
    ConditionalHandler,
    ForLoopHandler,
    WhileLoopHandler,
    DynamicFlowHandler,
    ControlFlowAutoResolver,
)
from .engine.control_flow_engine import ControlFlowEngine

__version__ = "0.1.0"

__all__ = [
    "Orchestrator",
    "Task",
    "TaskStatus",
    "Pipeline",
    "Model",
    "ModelRegistry",
    "YAMLCompiler",
    "ControlFlowCompiler",
    "ControlSystem",
    "ErrorHandler",
    "ResourceAllocator",
    "StateManager",
    "HuggingFaceModel",
    "OllamaModel",
    "init_models",
    "compile",
    "compile_async",
    "PipelineStatusTracker",
    "PipelineResumeManager",
    "ResumeStrategy",
    "ConditionalHandler",
    "ForLoopHandler",
    "WhileLoopHandler",
    "DynamicFlowHandler",
    "ControlFlowAutoResolver",
    "ControlFlowEngine",
]
__author__ = "Contextual Dynamics Lab"
__email__ = "contextualdynamics@gmail.com"

# Global instances
_model_registry = None
_orchestrator = None


def init_models(config_path: str = None) -> ModelRegistry:
    """Initialize the pool of available models by reading models.yaml and environment."""
    global _model_registry

    import os

    from .integrations.anthropic_model import AnthropicModel
    from .integrations.google_model import GoogleModel
    from .integrations.openai_model import OpenAIModel
    from .utils.model_utils import check_ollama_installed
    from .utils.model_config_loader import get_model_config_loader
    from .utils.api_keys_flexible import load_api_keys_optional

    print(">> Initializing model pool...")
    print(f">> Current environment: CI={os.environ.get('CI', 'false')}, GITHUB_ACTIONS={os.environ.get('GITHUB_ACTIONS', 'false')}")

    # Load available API keys (doesn't require all keys to be present)
    available_keys = load_api_keys_optional()
    print(f">> Available keys returned: {list(available_keys.keys()) if available_keys else 'None'}")
    if available_keys:
        print(f">> Found API keys for: {', '.join(available_keys.keys())}")
        # Also set them in environment for backward compatibility
        provider_env_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_AI_API_KEY",
            "huggingface": "HF_TOKEN",
            "openai": "OPENAI_API_KEY",
        }
        for provider, api_key in available_keys.items():
            env_var = provider_env_map.get(provider)
            if env_var and not os.environ.get(env_var):
                os.environ[env_var] = api_key
    else:
        print(">> No API keys found - only local models will be available")

    _model_registry = get_model_registry()

    # Load model configuration using the new loader
    loader = get_model_config_loader()
    config = loader.load_config()
    models_config = config.get("models", {})

    # Check if Ollama is installed
    ollama_available = check_ollama_installed()
    if not ollama_available:
        print(">> ⚠️  Ollama not found - Ollama models will not be available")
        print(">>    Install from: https://ollama.ai")

    # Process each model in configuration (list format)
    if not isinstance(models_config, list):
        print(">> ⚠️  Invalid models configuration format - expected list")
        models_config = []

    # Process each model
    for model_config in models_config:
        provider = model_config.get("source")
        name = model_config.get("name")

        # Parse size
        size_str = str(model_config.get("size", "1b"))
        if size_str.endswith("b"):
            size_billions = float(size_str[:-1])
        else:
            size_billions = float(size_str)

        # Get expertise
        expertise = model_config.get("expertise", ["general"])

        if not provider or not name:
            continue

        try:
            if provider == "ollama":
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

            elif provider == "huggingface":
                # Skip HuggingFace models if disabled via environment variable
                if (
                    os.environ.get("ORCHESTRATOR_SKIP_HUGGINGFACE", "").lower()
                    == "true"
                ):
                    continue

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

            elif provider == "openai" and "openai" in available_keys:
                # Only register if API key is available
                model = OpenAIModel(model_name=name, api_key=available_keys["openai"])
                # Add dynamic attributes for model selection
                setattr(model, "_expertise", expertise)
                setattr(model, "_size_billions", size_billions)
                _model_registry.register_model(model)
                print(
                    f">>   ✅ Registered OpenAI model: {name} ({size_billions}B params)"
                )

            elif provider == "anthropic" and "anthropic" in available_keys:
                # Only register if API key is available
                model = AnthropicModel(
                    model_name=name, api_key=available_keys["anthropic"]
                )
                # Add dynamic attributes for model selection
                setattr(model, "_expertise", expertise)
                setattr(model, "_size_billions", size_billions)
                _model_registry.register_model(model)
                print(
                    f">>   ✅ Registered Anthropic model: {name} ({size_billions}B params)"
                )

            elif provider == "google" and "google" in available_keys:
                # Only register if API key is available
                model = GoogleModel(model_name=name, api_key=available_keys["google"])
                # Add dynamic attributes for model selection
                setattr(model, "_expertise", expertise)
                setattr(model, "_size_billions", size_billions)
                _model_registry.register_model(model)
                print(
                    f">>   ✅ Registered Google model: {name} ({size_billions}B params)"
                )

        except Exception as e:
            print(f">>   ⚠️  Error registering {provider} model {name}: {e}")

    print(
        f"\n>> Model initialization complete: {len(_model_registry.list_models())} models registered"
    )

    if not _model_registry.list_models():
        print(">>   ⚠️  No models available - ensure models.yaml is properly configured")

    # Store defaults in registry for later use
    setattr(_model_registry, "_defaults", config.get("defaults", {}))

    # Enable auto-registration for new models
    _model_registry.enable_auto_registration()
    print(">> Auto-registration enabled for new models")

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
        try:
            # Check if there's an event loop running
            asyncio.get_running_loop()
            # We're in an async context but called from sync code
            # This is not ideal but we need to handle it
            asyncio.ensure_future(self._run_async(**kwargs))
            # Can't await here since this is a sync method
            raise RuntimeError(
                "Cannot call synchronous run() method from within an async context. "
                "Use 'await run_async()' instead."
            )
        except RuntimeError as e:
            if "no running event loop" in str(e):
                # No event loop running, use asyncio.run which handles cleanup properly
                return asyncio.run(self._run_async(**kwargs))
            else:
                raise

    async def run_async(self, **kwargs: Any) -> Any:
        """Run the pipeline asynchronously with given keyword arguments."""
        return await self._run_async(**kwargs)

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
    global _orchestrator

    # Ensure models are initialized
    if _model_registry is None:
        init_models()

    # Create orchestrator with hybrid control system that handles both models and tools
    from .control_systems.hybrid_control_system import HybridControlSystem

    # We need models to create a control system
    if not _model_registry or not _model_registry.models:
        raise RuntimeError(
            "No models available. Run init_models() first or ensure API keys are set."
        )

    control_system = HybridControlSystem(_model_registry)
    _orchestrator = Orchestrator(
        model_registry=_model_registry, control_system=control_system
    )

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
        # We're in an async context but called from sync code
        raise RuntimeError(
            "Cannot call synchronous compile_yaml() from within an async context. "
            "Use 'await compile_yaml_async()' instead."
        )

    except RuntimeError as e:
        if "no running event loop" in str(e):
            # No event loop running, use asyncio.run
            return asyncio.run(compile_async(yaml_path))
        else:
            raise


__all__ = [
    "Task",
    "TaskStatus",
    "Pipeline",
    "Model",
    "ModelCapabilities",
    "ModelRequirements",
    "ControlSystem",
    "YAMLCompiler",
    "ControlFlowCompiler",
    "ModelRegistry",
    "StateManager",
    "Orchestrator",
    "init_models",
    "compile",
    "ConditionalHandler",
    "ForLoopHandler",
    "WhileLoopHandler",
    "DynamicFlowHandler",
    "ControlFlowAutoResolver",
    "ControlFlowEngine",
]
