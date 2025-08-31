"""
Execution package for orchestrator framework.
Provides advanced execution engines, variable management, state persistence,
progress tracking, and recovery mechanisms.
"""

from .error_handler_executor import ErrorHandlerExecutor
from .engine import StateGraphEngine, ExecutionState, ExecutionError, StepExecutionError
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
from .progress import (
    ProgressTracker,
    ProgressTrackerProtocol,
    ProgressEvent,
    ProgressEventType,
    StepProgress,
    ExecutionProgress,
    StepStatus,
    create_progress_tracker
)
from .recovery import (
    RecoveryManager,
    RecoveryStrategy,
    ErrorSeverity,
    ErrorCategory,
    ErrorInfo,
    RetryConfig,
    RecoveryPlan,
    create_recovery_manager,
    network_error_handler,
    timeout_error_handler,
    critical_error_handler
)
from .integration import (
    ExecutionStateBridge,
    VariableManagerAdapter,
    ComprehensiveExecutionManager,
    create_comprehensive_execution_manager
)

__all__ = [
    "ErrorHandlerExecutor",
    "StateGraphEngine",
    "ExecutionState", 
    "ExecutionError",
    "StepExecutionError",
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
    # Progress Tracking
    "ProgressTracker",
    "ProgressTrackerProtocol",
    "ProgressEvent",
    "ProgressEventType",
    "StepProgress",
    "ExecutionProgress",
    "StepStatus",
    "create_progress_tracker",
    # Recovery Management
    "RecoveryManager",
    "RecoveryStrategy",
    "ErrorSeverity",
    "ErrorCategory",
    "ErrorInfo",
    "RetryConfig",
    "RecoveryPlan",
    "create_recovery_manager",
    "network_error_handler",
    "timeout_error_handler",
    "critical_error_handler",
    # Integration
    "ExecutionStateBridge",
    "VariableManagerAdapter",
    "ComprehensiveExecutionManager",
    "create_comprehensive_execution_manager"
]