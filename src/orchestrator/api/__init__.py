"""
API Interface Package for the Orchestrator Framework.

This package provides clean, user-friendly interfaces for pipeline operations,
hiding the complexity of the underlying architecture while providing full access
to all framework capabilities.

Main Components:
- PipelineAPI: Primary interface for pipeline compilation and execution
- AdvancedPipelineCompiler: Specialized compilation with enhanced YAML integration
- PipelineExecutor: Advanced execution with monitoring and control
- Exception classes: Structured error handling for API operations
- Convenience functions: Quick access helpers for common operations

Example usage:
    >>> from orchestrator.api import PipelineAPI, AdvancedPipelineCompiler, PipelineExecutor
    >>> 
    >>> # Basic API usage
    >>> api = create_pipeline_api(development_mode=True)
    >>> pipeline = await api.compile_pipeline(yaml_content, context)
    >>> execution = await api.execute_pipeline(pipeline)
    >>> 
    >>> # Advanced compilation with validation
    >>> compiler = AdvancedPipelineCompiler(enable_preprocessing=True)
    >>> pipeline, report = await compiler.compile_with_validation(yaml_content)
    >>> 
    >>> # Advanced execution with monitoring
    >>> executor = PipelineExecutor(max_concurrent_executions=5)
    >>> execution = await executor.execute_with_monitoring(pipeline, context)
    >>> async for status in executor.monitor_execution(execution.execution_id):
    ...     print(f"Progress: {status['progress_percentage']:.1f}%")
"""

from .core import (
    PipelineAPI,
    PipelineAPIError,
    CompilationError,
    ExecutionError,
    create_pipeline_api,
)
from .pipeline import (
    AdvancedPipelineCompiler,
    PipelineCompilerError,
    PipelineValidationError,
    create_advanced_pipeline_compiler,
)
from .execution import (
    PipelineExecutor,
    PipelineExecutionError,
    ExecutionControlError,
    create_pipeline_executor,
)

# Version information
__version__ = "2.0.0"
__author__ = "Orchestrator Team"

# Main exports
__all__ = [
    # Main API class
    "PipelineAPI",
    
    # Exception classes
    "PipelineAPIError",
    "CompilationError", 
    "ExecutionError",
    
    # Convenience functions
    "create_pipeline_api",
    
    # Specialized Pipeline Operations
    "AdvancedPipelineCompiler",
    "PipelineCompilerError",
    "PipelineValidationError",
    "create_advanced_pipeline_compiler",
    
    # Specialized Execution Operations
    "PipelineExecutor",
    "PipelineExecutionError",
    "ExecutionControlError",
    "create_pipeline_executor",
    
    # Version info
    "__version__",
]

# Convenience aliases for backwards compatibility
OrchestratorAPI = PipelineAPI
create_api = create_pipeline_api