"""Orchestrator: AI pipeline orchestration framework with intelligent ambiguity resolution.

This package provides a unified interface for executing AI pipelines defined in
YAML with automatic ambiguity resolution using LLMs.

Import policy
-------------
Public names are resolved lazily (PEP 562). ``import orchestrator`` performs no
heavy work: it does not import model-provider SDKs, browsers, multimedia
libraries or the execution engine. Each name is loaded from its defining module
the first time it is accessed, so an optional dependency that is not installed
breaks only the feature that needs it -- never the package import.

    import orchestrator                      # cheap; core deps only
    from orchestrator import Orchestrator    # loads the execution stack
    from orchestrator import WebSearchTool   # would need the [web] extra

This replaces the previous eager facade, which imported ~40 third-party
packages at ``import orchestrator`` time and made the package unimportable
whenever any optional dependency was missing.
"""

from ._lazy import lazy_exports

__version__ = "0.1.0"
__author__ = "Contextual Dynamics Lab"
__email__ = "contextualdynamics@gmail.com"

# Exported name -> module that defines it, relative to this package.
_EXPORTS: dict[str, str] = {
    # --- Core domain types ---
    "Task": ".core.task",
    "TaskStatus": ".core.task",
    "Pipeline": ".core.pipeline",
    "Model": ".core.model",
    "ModelCapabilities": ".core.model",
    "ModelRequirements": ".core.model",
    "ControlSystem": ".core.control_system",
    "PipelineStatusTracker": ".core.pipeline_status_tracker",
    "PipelineResumeManager": ".core.pipeline_resume_manager",
    "ResumeStrategy": ".core.pipeline_resume_manager",
    "ErrorHandler": ".core.error_handling",
    "ResourceAllocator": ".core.resource_allocator",
    # --- Compilation ---
    "YAMLCompiler": ".compiler.yaml_compiler",
    "ControlFlowCompiler": ".compiler.control_flow_compiler",
    # --- Execution ---
    "Orchestrator": ".orchestrator",
    "ControlFlowEngine": ".engine.control_flow_engine",
    # --- Control flow ---
    "ConditionalHandler": ".control_flow",
    "ForLoopHandler": ".control_flow",
    "WhileLoopHandler": ".control_flow",
    "DynamicFlowHandler": ".control_flow",
    "ControlFlowAutoResolver": ".control_flow",
    # --- Models ---
    "ModelRegistry": ".models.model_registry",
    "get_model_registry": ".models.registry_singleton",
    # --- Model integrations (each needs its provider extra) ---
    "HuggingFaceModel": ".integrations.huggingface_model",
    "OllamaModel": ".integrations.ollama_model",
    # --- State ---
    "StateManager": ".state.state_manager",
    # --- Tools / MCP ---
    "default_mcp_server": ".tools.mcp_server",
    "default_tool_detector": ".tools.mcp_server",
    # --- High-level helpers (see orchestrator._api) ---
    "init_models": "._api",
    "compile": "._api",
    "compile_async": "._api",
    "OrchestratorPipeline": "._api",
    # --- Optional API layer ---
    "PipelineAPI": ".api",
    "AdvancedPipelineCompiler": ".api",
    "PipelineExecutor": ".api",
    "create_pipeline_api": ".api",
    "create_advanced_pipeline_compiler": ".api",
    "create_pipeline_executor": ".api",
    # --- Error hierarchy ---
    "OrchestratorError": ".core.exceptions",
    "PipelineError": ".core.exceptions",
    "TaskError": ".core.exceptions",
    "ModelError": ".core.exceptions",
    "ValidationError": ".core.exceptions",
    "ResourceError": ".core.exceptions",
    "StateError": ".core.exceptions",
    "ToolError": ".core.exceptions",
    "ControlSystemError": ".core.exceptions",
    "CompilationError": ".core.exceptions",
    "AdapterError": ".core.exceptions",
    "ConfigurationError": ".core.exceptions",
    "NetworkError": ".core.exceptions",
    "TimeoutError": ".core.exceptions",
    "ModelNotFoundError": ".core.exceptions",
    "NoEligibleModelsError": ".core.exceptions",
    "TaskExecutionError": ".core.exceptions",
    "PipelineExecutionError": ".core.exceptions",
    "CircularDependencyError": ".core.exceptions",
    "InvalidDependencyError": ".core.exceptions",
}

__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS, globals())
