"""Comprehensive error hierarchy for the Orchestrator framework.

This module defines a structured exception hierarchy that provides:
1. Clear categorization of different error types
2. Consistent error handling across the framework
3. Rich error context for debugging
4. Proper inheritance structure
"""

from typing import Any, Dict, Optional


class OrchestratorError(Exception):
    """Base exception for all Orchestrator-related errors.
    
    This is the root of the exception hierarchy. All custom exceptions
    in the Orchestrator framework should inherit from this class.
    
    Attributes:
        message: Human-readable error message
        details: Optional dictionary containing additional error context
        error_code: Optional error code for programmatic handling
    """
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.error_code = error_code
        
    def __str__(self):
        """Return string representation of the error."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "error_code": self.error_code
        }


# Pipeline-related errors
class PipelineError(OrchestratorError):
    """Base class for pipeline-related errors."""
    pass


class PipelineCompilationError(PipelineError):
    """Raised when pipeline compilation fails."""
    pass


class PipelineExecutionError(PipelineError):
    """Raised when pipeline execution fails."""
    pass


class CircularDependencyError(PipelineError):
    """Raised when a circular dependency is detected in the pipeline."""
    
    def __init__(self, cycle: list, **kwargs):
        message = f"Circular dependency detected: {' -> '.join(cycle)}"
        super().__init__(message, details={"cycle": cycle}, **kwargs)


class InvalidDependencyError(PipelineError):
    """Raised when a dependency refers to a non-existent task."""
    
    def __init__(self, task_id: str, missing_dependency: str, **kwargs):
        message = f"Task '{task_id}' depends on non-existent task '{missing_dependency}'"
        super().__init__(
            message, 
            details={"task_id": task_id, "missing_dependency": missing_dependency},
            **kwargs
        )


class GraphGenerationError(PipelineError):
    """Raised when automatic graph generation fails."""
    
    def __init__(self, message: str, pipeline_id: Optional[str] = None, **kwargs):
        if pipeline_id:
            full_message = f"Graph generation failed for pipeline '{pipeline_id}': {message}"
            details = {"pipeline_id": pipeline_id}
        else:
            full_message = f"Graph generation failed: {message}"
            details = {}
            
        super().__init__(
            full_message,
            details=details,
            **kwargs
        )


# Task-related errors
class TaskError(OrchestratorError):
    """Base class for task-related errors."""
    pass


class TaskExecutionError(TaskError):
    """Raised when task execution fails."""
    
    def __init__(self, task_id: str, reason: str, **kwargs):
        message = f"Task '{task_id}' failed: {reason}"
        super().__init__(
            message,
            details={"task_id": task_id, "reason": reason},
            **kwargs
        )


class TaskValidationError(TaskError):
    """Raised when task validation fails."""
    pass


class TaskTimeoutError(TaskError):
    """Raised when task execution times out."""
    
    def __init__(self, task_id: str, timeout: float, **kwargs):
        message = f"Task '{task_id}' timed out after {timeout} seconds"
        super().__init__(
            message,
            details={"task_id": task_id, "timeout": timeout},
            **kwargs
        )


# Model-related errors
class ModelError(OrchestratorError):
    """Base class for model-related errors."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a requested model is not found."""
    
    def __init__(self, model_id: str, **kwargs):
        message = f"Model '{model_id}' not found in registry"
        super().__init__(message, details={"model_id": model_id}, **kwargs)


class NoEligibleModelsError(ModelError):
    """Raised when no models meet the requirements."""
    
    def __init__(self, requirements: Dict[str, Any], **kwargs):
        message = "No models meet the specified requirements"
        super().__init__(message, details={"requirements": requirements}, **kwargs)


class ModelExecutionError(ModelError):
    """Raised when model execution fails."""
    pass


class ModelConfigurationError(ModelError):
    """Raised when model configuration is invalid."""
    pass


# Validation errors
class ValidationError(OrchestratorError):
    """Base class for validation errors."""
    pass


class SchemaValidationError(ValidationError):
    """Raised when schema validation fails."""
    
    def __init__(self, validation_errors: list, **kwargs):
        message = f"Schema validation failed: {len(validation_errors)} errors"
        super().__init__(
            message,
            details={"validation_errors": validation_errors},
            **kwargs
        )


class YAMLValidationError(ValidationError):
    """Raised when YAML validation fails."""
    pass


class ParameterValidationError(ValidationError):
    """Raised when parameter validation fails."""
    
    def __init__(self, parameter: str, reason: str, **kwargs):
        message = f"Invalid parameter '{parameter}': {reason}"
        super().__init__(
            message,
            details={"parameter": parameter, "reason": reason},
            **kwargs
        )


# Resource errors
class ResourceError(OrchestratorError):
    """Base class for resource-related errors."""
    pass


class ResourceAllocationError(ResourceError):
    """Raised when resource allocation fails."""
    
    def __init__(self, resource_type: str, requested: Any, available: Any, **kwargs):
        message = f"Cannot allocate {requested} {resource_type} (only {available} available)"
        super().__init__(
            message,
            details={
                "resource_type": resource_type,
                "requested": requested,
                "available": available
            },
            **kwargs
        )


class ResourceLimitError(ResourceError):
    """Raised when resource limits are exceeded."""
    pass


# State management errors
class StateError(OrchestratorError):
    """Base class for state-related errors."""
    pass


class StateManagerError(StateError):
    """Raised when state management operations fail."""
    pass


class StateCorruptionError(StateError):
    """Raised when state corruption is detected."""
    
    def __init__(self, reason: str, **kwargs):
        message = f"State corruption detected: {reason}"
        super().__init__(message, details={"reason": reason}, **kwargs)


# Tool errors
class ToolError(OrchestratorError):
    """Base class for tool-related errors."""
    pass


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not found."""
    
    def __init__(self, tool_name: str, **kwargs):
        message = f"Tool '{tool_name}' not found"
        super().__init__(message, details={"tool_name": tool_name}, **kwargs)


class ToolExecutionError(ToolError):
    """Raised when tool execution fails."""
    
    def __init__(self, tool_name: str, reason: str, **kwargs):
        message = f"Tool '{tool_name}' execution failed: {reason}"
        super().__init__(
            message,
            details={"tool_name": tool_name, "reason": reason},
            **kwargs
        )


# Control system errors
class ControlSystemError(OrchestratorError):
    """Base class for control system errors."""
    pass


class CircuitBreakerOpenError(ControlSystemError):
    """Raised when circuit breaker is open."""
    
    def __init__(self, system_name: str, **kwargs):
        message = f"Circuit breaker is open for system '{system_name}'"
        super().__init__(message, details={"system_name": system_name}, **kwargs)


class SystemUnavailableError(ControlSystemError):
    """Raised when system is unavailable."""
    
    def __init__(self, system_name: str, reason: str = "Unknown", **kwargs):
        message = f"System '{system_name}' is unavailable: {reason}"
        super().__init__(
            message,
            details={"system_name": system_name, "reason": reason},
            **kwargs
        )


# Compilation errors
class CompilationError(OrchestratorError):
    """Base class for compilation-related errors."""
    pass


class YAMLCompilerError(CompilationError):
    """Base exception for YAML compiler errors."""
    pass


class AmbiguityResolutionError(CompilationError):
    """Raised when ambiguity resolution fails."""
    
    def __init__(self, ambiguity_type: str, context: str, **kwargs):
        message = f"Failed to resolve {ambiguity_type} ambiguity in context: {context}"
        super().__init__(
            message,
            details={"ambiguity_type": ambiguity_type, "context": context},
            **kwargs
        )


# Adapter errors
class AdapterError(OrchestratorError):
    """Base class for adapter-related errors."""
    pass


class AdapterConfigurationError(AdapterError):
    """Raised when adapter configuration is invalid."""
    pass


class AdapterConnectionError(AdapterError):
    """Raised when adapter connection fails."""
    
    def __init__(self, adapter_name: str, reason: str, **kwargs):
        message = f"Failed to connect adapter '{adapter_name}': {reason}"
        super().__init__(
            message,
            details={"adapter_name": adapter_name, "reason": reason},
            **kwargs
        )


# Configuration errors
class ConfigurationError(OrchestratorError):
    """Base class for configuration-related errors."""
    pass


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration is missing."""
    
    def __init__(self, config_key: str, **kwargs):
        message = f"Missing required configuration: {config_key}"
        super().__init__(message, details={"config_key": config_key}, **kwargs)


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration is invalid."""
    
    def __init__(self, config_key: str, reason: str, **kwargs):
        message = f"Invalid configuration for '{config_key}': {reason}"
        super().__init__(
            message,
            details={"config_key": config_key, "reason": reason},
            **kwargs
        )


# Network and API errors
class NetworkError(OrchestratorError):
    """Base class for network-related errors."""
    pass


class APIError(NetworkError):
    """Base class for API-related errors."""
    
    def __init__(self, service: str, status_code: Optional[int] = None, **kwargs):
        message = f"API error for service '{service}'"
        if status_code:
            message += f" (status code: {status_code})"
        super().__init__(
            message,
            details={"service": service, "status_code": status_code},
            **kwargs
        )


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""
    
    def __init__(self, service: str, retry_after: Optional[float] = None, **kwargs):
        super().__init__(service, **kwargs)
        self.message = f"Rate limit exceeded for service '{service}'"
        if retry_after:
            self.message += f" (retry after {retry_after} seconds)"
            self.details["retry_after"] = retry_after


class AuthenticationError(APIError):
    """Raised when authentication fails."""
    
    def __init__(self, service: str, **kwargs):
        super().__init__(service, status_code=401, **kwargs)
        self.message = f"Authentication failed for service '{service}'"


# Timeout errors
class TimeoutError(OrchestratorError):
    """Base class for timeout-related errors."""
    
    def __init__(self, operation: str, timeout: float, **kwargs):
        message = f"Operation '{operation}' timed out after {timeout} seconds"
        super().__init__(
            message,
            details={"operation": operation, "timeout": timeout},
            **kwargs
        )


# Helper function to get error hierarchy
def get_error_hierarchy() -> Dict[str, list]:
    """Get the complete error hierarchy as a dictionary."""
    hierarchy = {}
    
    def add_subclasses(base_class, result_dict):
        subclasses = base_class.__subclasses__()
        if subclasses:
            result_dict[base_class.__name__] = [cls.__name__ for cls in subclasses]
            for subclass in subclasses:
                add_subclasses(subclass, result_dict)
    
    add_subclasses(OrchestratorError, hierarchy)
    return hierarchy