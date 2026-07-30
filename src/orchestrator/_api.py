"""Runtime helpers for the :mod:`orchestrator` public facade.

This module holds the eager, dependency-heavy runtime helpers that used to live
directly in ``orchestrator/__init__.py``. Keeping them here lets the package
facade stay lazy (PEP 562) so that ``import orchestrator`` costs almost nothing
and does not require model-provider SDKs to be installed.

Nothing here is part of the public import path; use ``orchestrator.init_models``,
``orchestrator.compile`` and ``orchestrator.compile_async`` instead.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict

from .compiler.yaml_compiler import YAMLCompiler
from .core.pipeline import Pipeline
from .models.model_registry import ModelRegistry
from .models.registry_singleton import get_model_registry
from .orchestrator import Orchestrator
from .tools.mcp_server import default_mcp_server, default_tool_detector

logger = logging.getLogger(__name__)

# Global instances
_model_registry = None
_orchestrator = None


def init_models(config_path: str = None) -> ModelRegistry:
    """Initialize the pool of available models by reading models.yaml and environment.

    This is the eager entry point: it reads credentials and probes providers
    immediately. Callers that may never need a model should instead hand
    :func:`populate_model_registry` to a
    :class:`~orchestrator.models.lazy_registry.LazyModelRegistry`, which defers
    all of that until a model is actually requested.
    """
    global _model_registry

    _model_registry = get_model_registry()
    populate_model_registry(_model_registry)
    return _model_registry


def populate_model_registry(registry: ModelRegistry) -> ModelRegistry:
    """Register every model the current environment can actually serve.

    Reads ``~/.orchestrator/.env`` for provider credentials, loads
    ``models.yaml`` and probes for a local Ollama install. This is the step
    that touches the user's credentials, so it must only run when a model is
    genuinely required.
    """
    import os

    from .integrations.anthropic_model import AnthropicModel
    from .integrations.google_model import GoogleModel
    from .integrations.openai_model import OpenAIModel
    from .utils.model_utils import check_ollama_installed
    from .utils.model_config_loader import get_model_config_loader
    from .utils.api_keys_flexible import load_api_keys_optional

    logger.info("Initializing model pool")

    # Load available API keys (doesn't require all keys to be present)
    available_keys = load_api_keys_optional()
    if available_keys:
        logger.info("Found API keys for: %s", ", ".join(sorted(available_keys)))
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
        logger.info("No API keys found - only local models will be available")

    # Load model configuration using the new loader
    loader = get_model_config_loader()
    config = loader.load_config()
    models_config = config.get("models", {})

    # Check if Ollama is installed
    ollama_available = check_ollama_installed()
    if not ollama_available:
        logger.info(
            "Ollama not found - Ollama models unavailable (install from https://ollama.ai)"
        )

    # Process each model in configuration (list format)
    if not isinstance(models_config, list):
        logger.warning(
            "Invalid models configuration format: expected a list, got %s",
            type(models_config).__name__,
        )
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
                registry.register_model(model)
                logger.info(
                    "Registered Ollama model %s (%sB) - downloads on first use",
                    name,
                    size_billions,
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
                        registry.register_model(hf_model)
                        logger.info(
                            "Registered HuggingFace model %s (%sB) - downloads on first use",
                            name,
                            size_billions,
                        )
                except ImportError:
                    logger.info(
                        "HuggingFace model %s configured but transformers is not "
                        "installed (pip install 'py-orc[multimedia]')",
                        name,
                    )
                except Exception as e:
                    logger.warning(
                        "Could not register HuggingFace model %s: %s", name, e
                    )

            elif provider == "openai" and "openai" in available_keys:
                # Only register if API key is available
                model = OpenAIModel(model_name=name, api_key=available_keys["openai"])
                # Add dynamic attributes for model selection
                setattr(model, "_expertise", expertise)
                setattr(model, "_size_billions", size_billions)
                registry.register_model(model)
                logger.info("Registered OpenAI model %s (%sB)", name, size_billions)

            elif provider == "anthropic" and "anthropic" in available_keys:
                # Only register if API key is available
                model = AnthropicModel(
                    model_name=name, api_key=available_keys["anthropic"]
                )
                # Add dynamic attributes for model selection
                setattr(model, "_expertise", expertise)
                setattr(model, "_size_billions", size_billions)
                registry.register_model(model)
                logger.info("Registered Anthropic model %s (%sB)", name, size_billions)

            elif provider == "google" and "google" in available_keys:
                # Only register if API key is available
                model = GoogleModel(model_name=name, api_key=available_keys["google"])
                # Add dynamic attributes for model selection
                setattr(model, "_expertise", expertise)
                setattr(model, "_size_billions", size_billions)
                registry.register_model(model)
                logger.info("Registered Google model %s (%sB)", name, size_billions)

        except Exception as e:
            # One line per provider, naming the real cause. Registering a
            # model is best-effort: a missing provider SDK must not stop the
            # models that *are* usable from being registered.
            logger.warning("Could not register %s model %s: %s", provider, name, e)

    registered = registry.list_models()
    if registered:
        logger.info("Model pool ready: %d model(s) registered", len(registered))
    else:
        logger.info(
            "Model pool ready: no models registered "
            "(no provider credentials and no local models)"
        )

    # Store defaults in registry for later use
    setattr(registry, "_defaults", config.get("defaults", {}))

    # Enable auto-registration for new models
    registry.enable_auto_registration()

    return registry


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


