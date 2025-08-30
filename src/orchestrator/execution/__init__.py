"""
Execution package for orchestrator framework.
Provides advanced execution engines, variable management, and state persistence.
"""

from .error_handler_executor import ErrorHandlerExecutor
from .variables import (
    VariableManager,
    VariableContext,
    VariableScope,
    VariableType,
    Variable,
    VariableMetadata
)
from .state import (
    ExecutionContext,
    FileStateManager,
    ExecutionStatus,
    ExecutionMetrics,
    Checkpoint,
    PersistenceFormat,
    create_execution_context,
    load_execution_context
)
from .integration import (
    ExecutionStateBridge,
    VariableManagerAdapter
)

__all__ = [
    "ErrorHandlerExecutor",
    # Variable Management
    "VariableManager",
    "VariableContext",
    "VariableScope",
    "VariableType",
    "Variable",
    "VariableMetadata",
    # State Management
    "ExecutionContext",
    "FileStateManager",
    "ExecutionStatus",
    "ExecutionMetrics",
    "Checkpoint",
    "PersistenceFormat",
    "create_execution_context",
    "load_execution_context",
    # Integration
    "ExecutionStateBridge",
    "VariableManagerAdapter"
]