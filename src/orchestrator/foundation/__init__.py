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
from .state_graph import StateGraphCompiler, StateGraphExecutor
from .pipeline_spec import PipelineSpecification, PipelineHeader, PipelineStep
from .result import PipelineResult, StepResult

__all__ = [
    "PipelineCompilerInterface",
    "ExecutionEngineInterface", 
    "ModelManagerInterface",
    "ToolRegistryInterface",
    "QualityControlInterface",
    "StateGraphCompiler",
    "StateGraphExecutor",
    "PipelineSpecification",
    "PipelineHeader",
    "PipelineStep",
    "PipelineResult",
    "StepResult",
]