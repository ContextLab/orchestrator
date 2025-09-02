"""
Compatibility module for foundation components.

This module provides backward compatibility during the migration to the new architecture.
All foundation components have been migrated to the execution and api modules.
"""

# Import compatibility layer for backward compatibility
from ._compatibility import (
    ExecutionEngineInterface,
    FoundationConfig,
    PipelineHeader,
    PipelineSpecification,
    PipelineStep,
    PipelineResult,
    StepResult,
)

# Legacy interfaces for backward compatibility
try:
    from .interfaces import (
        PipelineCompilerInterface,
        ModelManagerInterface,
        ToolRegistryInterface,
        QualityControlInterface,
    )
except ImportError:
    # Provide stubs if files don't exist
    class PipelineCompilerInterface:
        pass
    class ModelManagerInterface:
        pass
    class ToolRegistryInterface:
        pass
    class QualityControlInterface:
        pass

# Re-export for backward compatibility
__all__ = [
    "ExecutionEngineInterface",
    "FoundationConfig",
    "PipelineHeader", 
    "PipelineSpecification",
    "PipelineStep",
    "PipelineResult",
    "StepResult",
    "PipelineCompilerInterface",
    "ModelManagerInterface",
    "ToolRegistryInterface", 
    "QualityControlInterface",
]