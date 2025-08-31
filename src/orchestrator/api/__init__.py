"""
API Interface Package for the Orchestrator Framework.

This package provides clean, user-friendly interfaces for pipeline operations,
hiding the complexity of the underlying architecture while providing full access
to all framework capabilities.

Main Components:
- PipelineAPI: Primary interface for pipeline compilation and execution
- AdvancedPipelineCompiler: Specialized compilation with enhanced YAML integration
- PipelineExecutor: Advanced execution with monitoring and control
- Comprehensive error handling: Structured error handling with recovery mechanisms
- Type definitions: Complete type system for API operations
- Convenience functions: Quick access helpers for common operations

Example usage:
    >>> from orchestrator.api import (
    ...     PipelineAPI, AdvancedPipelineCompiler, PipelineExecutor,
    ...     CompilationRequest, ExecutionRequest, APIErrorHandler
    ... )
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
    ...
    >>> # Error handling with recovery
    >>> error_handler = create_api_error_handler()
    >>> try:
    ...     pipeline = await api.compile_pipeline(yaml_content)
    ... except Exception as e:
    ...     api_error = error_handler.handle_error(e, operation="compilation")
    ...     print(f"Error: {api_error.message}")
    ...     print(f"Recovery: {api_error.recovery_guidance.user_actions}")
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
from .errors import (
    OrchestratorAPIError,
    PipelineCompilationError,
    YAMLValidationError,
    TemplateProcessingError,
    PipelineExecutionError as APIExecutionError,
    ExecutionTimeoutError,
    StepExecutionError,
    APIConfigurationError,
    ModelRegistryError,
    ResourceError,
    NetworkError,
    UserInputError,
    APIErrorHandler,
    APIErrorCategory,
    RecoveryGuidance,
    APIErrorContext,
    create_api_error_handler,
    handle_api_exception,
)
from .types import (
    APIOperation,
    ValidationLevel,
    CompilationMode,
    ExecutionMode,
    CompilationRequest,
    ExecutionRequest,
    APIResponse,
    CompilationResult,
    ExecutionResult,
    ExecutionStatusInfo,
    ProgressUpdate,
    APIConfiguration,
    PipelineCompilerProtocol,
    ExecutionManagerProtocol,
    ProgressMonitorProtocol,
    ProgressCallback,
    StatusCallback,
    ErrorCallback,
    CompletionCallback,
    PipelineCompilationDict,
    PipelineExecutionDict,
    ExecutionStatusDict,
    ValidationResult,
    ResourceUsage,
    StepSummary,
    APIEndpoint,
    API_DOCUMENTATION,
    PipelineCompilationResponse,
    PipelineExecutionResponse,
    ExecutionStatusResponse,
    ProgressUpdateResponse,
)

# Version information
__version__ = "2.0.0"
__author__ = "Orchestrator Team"

# Main exports
__all__ = [
    # Main API class
    "PipelineAPI",
    
    # Core Exception classes
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
    
    # Comprehensive Error Handling
    "OrchestratorAPIError",
    "PipelineCompilationError",
    "YAMLValidationError",
    "TemplateProcessingError",
    "APIExecutionError",
    "ExecutionTimeoutError",
    "StepExecutionError",
    "APIConfigurationError",
    "ModelRegistryError",
    "ResourceError",
    "NetworkError",
    "UserInputError",
    "APIErrorHandler",
    "APIErrorCategory",
    "RecoveryGuidance",
    "APIErrorContext",
    "create_api_error_handler",
    "handle_api_exception",
    
    # Type Definitions
    "APIOperation",
    "ValidationLevel",
    "CompilationMode",
    "ExecutionMode",
    "CompilationRequest",
    "ExecutionRequest",
    "APIResponse",
    "CompilationResult",
    "ExecutionResult",
    "ExecutionStatusInfo",
    "ProgressUpdate",
    "APIConfiguration",
    
    # Protocol Types
    "PipelineCompilerProtocol",
    "ExecutionManagerProtocol",
    "ProgressMonitorProtocol",
    
    # Callback Types
    "ProgressCallback",
    "StatusCallback",
    "ErrorCallback",
    "CompletionCallback",
    
    # Typed Dictionary Types
    "PipelineCompilationDict",
    "PipelineExecutionDict",
    "ExecutionStatusDict",
    "ValidationResult",
    "ResourceUsage",
    "StepSummary",
    
    # Documentation Types
    "APIEndpoint",
    "API_DOCUMENTATION",
    
    # Response Types
    "PipelineCompilationResponse",
    "PipelineExecutionResponse",
    "ExecutionStatusResponse",
    "ProgressUpdateResponse",
    
    # Version info
    "__version__",
]

# Convenience aliases for backwards compatibility
OrchestratorAPI = PipelineAPI
create_api = create_pipeline_api