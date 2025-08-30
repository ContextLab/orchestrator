"""
Foundation package for the refactored orchestrator architecture.

This package contains the core interfaces and abstract classes that define
the foundational architecture for the new orchestrator system.
"""

from .interfaces import (
    PipelineCompilerInterface,
    ExecutionEngineInterface,
    ModelManagerInterface,
    ToolRegistryInterface,
    QualityControlInterface,
)
from .pipeline_spec import PipelineSpecification, PipelineHeader, PipelineStep
from .result import PipelineResult, StepResult

# Optional imports for state graph components
try:
    from .state_graph import StateGraphCompiler, StateGraphExecutor
    _has_state_graph = True
except ImportError:
    StateGraphCompiler = None
    StateGraphExecutor = None
    _has_state_graph = False

# Configuration data class
try:
    from .config import FoundationConfig
except ImportError:
    # Create a simple FoundationConfig if not found elsewhere
    from dataclasses import dataclass
    from typing import Optional
    
    @dataclass
    class FoundationConfig:
        """Foundation configuration with default values."""
        default_model: Optional[str] = None
        model_selection_strategy: str = "balanced"
        max_concurrent_steps: int = 5
        enable_quality_checks: bool = True
        enable_persistence: bool = False

__all__ = [
    "PipelineCompilerInterface",
    "ExecutionEngineInterface", 
    "ModelManagerInterface",
    "ToolRegistryInterface",
    "QualityControlInterface",
    "PipelineSpecification",
    "PipelineHeader",
    "PipelineStep",
    "PipelineResult",
    "StepResult",
    "FoundationConfig",
]

if _has_state_graph:
    __all__.extend(["StateGraphCompiler", "StateGraphExecutor"])