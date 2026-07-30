"""State management and checkpointing.

This module provides both legacy and modern state management systems:

RECOMMENDED (New Projects):
- Use orchestrator.Orchestrator(use_langgraph_state=True)
- Enhanced features with LangGraph integration
- Better performance and scalability

LEGACY (Backward Compatibility):
- StateManager - Legacy state management (still supported)
- Deprecated modules: simple_state_manager, adaptive_checkpoint

MIGRATION:
- LegacyStateManagerAdapter - Provides seamless compatibility
- See docs/migration/langgraph-state-management.md for migration guide

Names are resolved lazily: the LangGraph-backed managers require the
``langgraph`` extra, and importing this package must not force that dependency
on users who only need the legacy ``StateManager``.
"""

from .._lazy import lazy_exports

_EXPORTS = {
    # Legacy (backward compatibility)
    "StateManager": ".state_manager",
    # Modern LangGraph-based system (requires the [langgraph] extra)
    "LangGraphGlobalContextManager": ".langgraph_state_manager",
    "LegacyStateManagerAdapter": ".legacy_compatibility",
    # Global context schema and utilities
    "PipelineGlobalState": ".global_context",
    "ExecutionMetadata": ".global_context",
    "ErrorContext": ".global_context",
    "ToolExecutionResults": ".global_context",
    "ModelInteractions": ".global_context",
    "DebugContext": ".global_context",
    "PerformanceMetrics": ".global_context",
    "SecurityContext": ".global_context",
    "PipelineStatus": ".global_context",
    "create_initial_pipeline_state": ".global_context",
    "validate_pipeline_state": ".global_context",
    "merge_pipeline_states": ".global_context",
}

__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS, globals())
