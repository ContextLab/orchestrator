"""
Runtime dependency resolution and loop expansion system.

This package implements the runtime dependency resolution architecture
outlined in Issue #211, providing progressive template resolution and
dynamic loop expansion.
"""

from .execution_state import (
    PipelineExecutionState,
    UnresolvedItem,
    LoopContext,
    ItemStatus
)

from .dependency_resolver import (
    DependencyResolver,
    ResolutionResult
)

from .loop_expander import (
    LoopExpander,
    LoopTask,
    ExpandedTask
)

from .orchestrator_integration import RuntimeResolutionIntegration

__all__ = [
    'PipelineExecutionState',
    'UnresolvedItem',
    'LoopContext',
    'ItemStatus',
    'DependencyResolver',
    'ResolutionResult',
    'LoopExpander',
    'LoopTask',
    'ExpandedTask',
    'RuntimeResolutionIntegration',
]