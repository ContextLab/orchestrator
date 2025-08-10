"""LangGraph Built-in Checkpointing and Persistence - Issue #205

This module provides automatic step-level checkpointing, durable execution,
human-in-the-loop capabilities, and seamless recovery using LangGraph's 
native persistence infrastructure.

Phase 1 Implementation: Core Automatic Checkpointing
- AutomaticCheckpointingGraph: LangGraph workflow with step-level checkpointing
- DurableExecutionManager: Fault-tolerant pipeline execution with recovery
- ExecutionRecoveryStrategy: Recovery strategy enumeration
- CheckpointedExecutionError: Exception with checkpoint context

Future Phases:
- HumanInteractionManager: Human-in-the-loop capabilities (Phase 2)
- CheckpointMigrationManager: Legacy checkpoint migration (Phase 3)
"""

from .automatic_graph import (
    AutomaticCheckpointingGraph,
    CheckpointedExecutionError,
    CheckpointMetadata
)
from .durable_executor import (
    DurableExecutionManager,
    ExecutionRecoveryStrategy,
    ExecutionResult,
    RecoveryContext
)

__all__ = [
    "AutomaticCheckpointingGraph",
    "CheckpointedExecutionError", 
    "CheckpointMetadata",
    "DurableExecutionManager",
    "ExecutionRecoveryStrategy",
    "ExecutionResult",
    "RecoveryContext",
]