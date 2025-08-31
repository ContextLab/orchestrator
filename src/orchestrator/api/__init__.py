"""
API Interface Package for the Orchestrator Framework.

This package provides clean, user-friendly interfaces for pipeline operations,
hiding the complexity of the underlying architecture while providing full access
to all framework capabilities.

Main Components:
- PipelineAPI: Primary interface for pipeline compilation and execution
- Exception classes: Structured error handling for API operations
- Convenience functions: Quick access helpers for common operations

Example usage:
    >>> from orchestrator.api import PipelineAPI, create_pipeline_api
    >>> 
    >>> # Create API instance
    >>> api = create_pipeline_api(development_mode=True)
    >>> 
    >>> # Compile and execute pipeline
    >>> pipeline = await api.compile_pipeline(yaml_content, context)
    >>> execution = await api.execute_pipeline(pipeline)
    >>> status = api.get_execution_status(execution.execution_id)
"""

from .core import (
    PipelineAPI,
    PipelineAPIError,
    CompilationError,
    ExecutionError,
    create_pipeline_api,
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
    
    # Version info
    "__version__",
]

# Convenience aliases for backwards compatibility
OrchestratorAPI = PipelineAPI
create_api = create_pipeline_api