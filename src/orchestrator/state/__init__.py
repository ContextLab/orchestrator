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
"""

# Legacy imports (maintained for backward compatibility)
from .state_manager import StateManager

# Modern LangGraph-based imports
from .langgraph_state_manager import LangGraphGlobalContextManager
from .legacy_compatibility import LegacyStateManagerAdapter
from .global_context import (
    PipelineGlobalState,
    ExecutionMetadata,
    ErrorContext,
    ToolExecutionResults,
    ModelInteractions,
    DebugContext,
    PerformanceMetrics,
    SecurityContext,
    PipelineStatus,
    create_initial_pipeline_state,
    validate_pipeline_state,
    merge_pipeline_states,
)

__all__ = [
    # Legacy (backward compatibility)
    "StateManager",
    
    # Modern LangGraph-based system  
    "LangGraphGlobalContextManager",
    "LegacyStateManagerAdapter",
    
    # Global context schema and utilities
    "PipelineGlobalState",
    "ExecutionMetadata",
    "ErrorContext", 
    "ToolExecutionResults",
    "ModelInteractions",
    "DebugContext",
    "PerformanceMetrics",
    "SecurityContext",
    "PipelineStatus",
    "create_initial_pipeline_state",
    "validate_pipeline_state", 
    "merge_pipeline_states",
]
