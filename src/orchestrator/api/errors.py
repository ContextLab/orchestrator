"""
Comprehensive Error Handling for the Orchestrator API Framework.

This module provides structured error handling with comprehensive recovery mechanisms,
detailed error reporting, context-aware error classification, and integration with
the underlying execution and compilation systems.

The error system follows a hierarchical structure:
- OrchestratorAPIError: Base class for all API errors
- Compilation errors: Issues during pipeline compilation and validation
- Execution errors: Issues during pipeline execution and monitoring
- Configuration errors: Issues with API setup and configuration
- Resource errors: Issues with external dependencies and resources

Each error includes detailed context, recovery suggestions, and integration
with the underlying recovery management system.
"""

from __future__ import annotations

import logging
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Type, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Import foundation error handling components
from ..execution import (
    ErrorSeverity,
    ErrorCategory,
    ErrorInfo,
    RecoveryStrategy,
    RecoveryManager,
    RecoveryPlan
)

logger = logging.getLogger(__name__)


class APIErrorCategory(Enum):
    """Extended error categories specific to API operations."""
    # Core API categories
    COMPILATION = "compilation"           # Pipeline compilation errors
    EXECUTION = "execution"               # Pipeline execution errors
    VALIDATION = "validation"             # Pipeline validation errors
    CONFIGURATION = "configuration"       # API configuration errors
    
    # Resource and dependency categories
    MODEL_REGISTRY = "model_registry"     # Model registry errors
    YAML_PROCESSING = "yaml_processing"   # YAML parsing/processing errors
    TEMPLATE_PROCESSING = "template"      # Template processing errors
    DEPENDENCY_RESOLUTION = "dependency"  # Dependency resolution errors
    
    # Runtime categories  
    STATE_MANAGEMENT = "state"           # Execution state errors
    PROGRESS_TRACKING = "progress"       # Progress tracking errors
    MONITORING = "monitoring"            # Execution monitoring errors
    RESOURCE_MANAGEMENT = "resource"     # Resource allocation errors
    
    # External integration categories
    MODEL_INTEGRATION = "model"          # Model integration errors
    TOOL_INTEGRATION = "tool"            # Tool integration errors
    AUTHENTICATION = "authentication"    # Authentication errors
    NETWORK = "network"                  # Network connectivity errors
    
    # User interaction categories
    INPUT_VALIDATION = "input"           # Input validation errors
    OUTPUT_PROCESSING = "output"         # Output processing errors
    USER_CONFIGURATION = "user_config"   # User configuration errors
    PERMISSION = "permission"            # Permission/access errors


@dataclass
class APIErrorContext:
    """Comprehensive context information for API errors."""
    error_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    
    # API context
    operation: Optional[str] = None
    api_method: Optional[str] = None
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Pipeline context
    pipeline_id: Optional[str] = None
    execution_id: Optional[str] = None
    step_id: Optional[str] = None
    step_name: Optional[str] = None
    
    # System context
    thread_id: Optional[int] = None
    process_id: Optional[int] = None
    memory_usage_mb: Optional[float] = None
    
    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)
    related_errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary format."""
        result = {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
        }
        
        # Add non-None fields
        for key, value in [
            ("operation", self.operation),
            ("api_method", self.api_method),
            ("request_id", self.request_id),
            ("user_id", self.user_id),
            ("pipeline_id", self.pipeline_id),
            ("execution_id", self.execution_id),
            ("step_id", self.step_id),
            ("step_name", self.step_name),
            ("thread_id", self.thread_id),
            ("process_id", self.process_id),
            ("memory_usage_mb", self.memory_usage_mb),
        ]:
            if value is not None:
                result[key] = value
        
        if self.metadata:
            result["metadata"] = self.metadata
        if self.related_errors:
            result["related_errors"] = self.related_errors
            
        return result


@dataclass
class RecoveryGuidance:
    """Comprehensive recovery guidance for API errors."""
    # Recovery strategy
    strategy: RecoveryStrategy
    automatic_recovery: bool = False
    
    # User actions
    user_actions: List[str] = field(default_factory=list)
    system_actions: List[str] = field(default_factory=list)
    
    # Recovery details
    recovery_steps: List[str] = field(default_factory=list)
    estimated_recovery_time: Optional[int] = None  # seconds
    confidence_level: float = 0.5  # 0.0 to 1.0
    
    # Additional guidance
    documentation_links: List[str] = field(default_factory=list)
    troubleshooting_tips: List[str] = field(default_factory=list)
    related_issues: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert recovery guidance to dictionary format."""
        return {
            "strategy": self.strategy.value,
            "automatic_recovery": self.automatic_recovery,
            "user_actions": self.user_actions,
            "system_actions": self.system_actions,
            "recovery_steps": self.recovery_steps,
            "estimated_recovery_time": self.estimated_recovery_time,
            "confidence_level": self.confidence_level,
            "documentation_links": self.documentation_links,
            "troubleshooting_tips": self.troubleshooting_tips,
            "related_issues": self.related_issues
        }


class OrchestratorAPIError(Exception):
    """
    Base exception class for all Orchestrator API errors.
    
    Provides comprehensive error information including context, severity,
    recovery guidance, and integration with the underlying error handling system.
    """
    
    def __init__(
        self,
        message: str,
        category: APIErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[APIErrorContext] = None,
        recovery_guidance: Optional[RecoveryGuidance] = None,
        original_exception: Optional[Exception] = None,
        traceback_info: Optional[str] = None
    ):
        """
        Initialize API error with comprehensive context.
        
        Args:
            message: Human-readable error message
            category: Error category for classification
            severity: Error severity level
            context: Additional error context
            recovery_guidance: Recovery guidance and suggestions
            original_exception: Original exception that caused this error
            traceback_info: Optional traceback information
        """
        super().__init__(message)
        
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or APIErrorContext()
        self.recovery_guidance = recovery_guidance
        self.original_exception = original_exception
        
        # Capture traceback if not provided
        if traceback_info is None and original_exception:
            self.traceback_info = traceback.format_exception(
                type(original_exception), original_exception, original_exception.__traceback__
            )
        else:
            self.traceback_info = traceback_info
        
        # Log error for monitoring
        self._log_error()
    
    def _log_error(self):
        """Log error for monitoring and debugging."""
        logger.error(
            f"[{self.context.error_id}] {self.__class__.__name__}: {self.message}",
            extra={
                "error_id": self.context.error_id,
                "category": self.category.value,
                "severity": self.severity.value,
                "pipeline_id": self.context.pipeline_id,
                "execution_id": self.context.execution_id
            }
        )
    
    def to_error_info(self) -> ErrorInfo:
        """Convert to foundation ErrorInfo object."""
        # Map API category to foundation category
        foundation_category_map = {
            APIErrorCategory.COMPILATION: ErrorCategory.VALIDATION,
            APIErrorCategory.EXECUTION: ErrorCategory.EXECUTION,
            APIErrorCategory.VALIDATION: ErrorCategory.VALIDATION,
            APIErrorCategory.CONFIGURATION: ErrorCategory.CONFIGURATION,
            APIErrorCategory.MODEL_REGISTRY: ErrorCategory.DEPENDENCY,
            APIErrorCategory.YAML_PROCESSING: ErrorCategory.VALIDATION,
            APIErrorCategory.TEMPLATE_PROCESSING: ErrorCategory.VALIDATION,
            APIErrorCategory.DEPENDENCY_RESOLUTION: ErrorCategory.DEPENDENCY,
            APIErrorCategory.STATE_MANAGEMENT: ErrorCategory.SYSTEM,
            APIErrorCategory.PROGRESS_TRACKING: ErrorCategory.SYSTEM,
            APIErrorCategory.MONITORING: ErrorCategory.SYSTEM,
            APIErrorCategory.RESOURCE_MANAGEMENT: ErrorCategory.RESOURCE,
            APIErrorCategory.MODEL_INTEGRATION: ErrorCategory.DEPENDENCY,
            APIErrorCategory.TOOL_INTEGRATION: ErrorCategory.DEPENDENCY,
            APIErrorCategory.AUTHENTICATION: ErrorCategory.AUTHENTICATION,
            APIErrorCategory.NETWORK: ErrorCategory.NETWORK,
            APIErrorCategory.INPUT_VALIDATION: ErrorCategory.USER,
            APIErrorCategory.OUTPUT_PROCESSING: ErrorCategory.SYSTEM,
            APIErrorCategory.USER_CONFIGURATION: ErrorCategory.USER,
            APIErrorCategory.PERMISSION: ErrorCategory.AUTHENTICATION
        }
        
        return ErrorInfo(
            error_id=self.context.error_id,
            category=foundation_category_map.get(self.category, ErrorCategory.UNKNOWN),
            severity=self.severity,
            message=self.message,
            exception=self.original_exception,
            traceback=self.traceback_info,
            step_id=self.context.step_id,
            step_name=self.context.step_name,
            execution_id=self.context.execution_id,
            timestamp=self.context.timestamp,
            context=self.context.to_dict()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format for serialization."""
        result = {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "context": self.context.to_dict()
        }
        
        if self.recovery_guidance:
            result["recovery_guidance"] = self.recovery_guidance.to_dict()
        
        if self.original_exception:
            result["original_exception"] = {
                "type": type(self.original_exception).__name__,
                "message": str(self.original_exception)
            }
            
        return result


# Compilation Error Classes

class PipelineCompilationError(OrchestratorAPIError):
    """Error during pipeline compilation process."""
    
    def __init__(
        self,
        message: str,
        yaml_content: Optional[str] = None,
        context_variables: Optional[Dict[str, Any]] = None,
        validation_errors: Optional[List[str]] = None,
        **kwargs
    ):
        # Create specific context
        context = kwargs.get('context', APIErrorContext())
        context.operation = "pipeline_compilation"
        context.metadata.update({
            "yaml_length": len(yaml_content) if yaml_content else 0,
            "context_variables": list(context_variables.keys()) if context_variables else [],
            "validation_errors": validation_errors or []
        })
        
        # Create recovery guidance
        recovery_guidance = RecoveryGuidance(
            strategy=RecoveryStrategy.MANUAL_INTERVENTION,
            user_actions=[
                "Review YAML syntax and structure",
                "Check template variable definitions",
                "Validate model and tool references",
                "Review compilation context variables"
            ],
            troubleshooting_tips=[
                "Use YAML validator to check syntax",
                "Ensure all template variables are defined",
                "Verify model registry contains referenced models",
                "Check for circular dependencies in pipeline steps"
            ],
            documentation_links=[
                "/docs/pipeline-specification",
                "/docs/yaml-reference",
                "/docs/template-variables"
            ]
        )
        
        super().__init__(
            message=message,
            category=APIErrorCategory.COMPILATION,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_guidance=recovery_guidance,
            **kwargs
        )


class YAMLValidationError(PipelineCompilationError):
    """Error during YAML validation and parsing."""
    
    def __init__(
        self,
        message: str,
        yaml_line: Optional[int] = None,
        yaml_column: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.get('context', APIErrorContext())
        context.operation = "yaml_validation"
        context.metadata.update({
            "yaml_line": yaml_line,
            "yaml_column": yaml_column
        })
        
        recovery_guidance = RecoveryGuidance(
            strategy=RecoveryStrategy.MANUAL_INTERVENTION,
            user_actions=[
                "Fix YAML syntax errors",
                "Check indentation and structure",
                "Validate YAML against schema"
            ],
            recovery_steps=[
                "1. Use a YAML validator to identify syntax issues",
                "2. Fix indentation and structural problems",
                "3. Retry compilation"
            ]
        )
        
        super().__init__(
            message=message,
            context=context,
            recovery_guidance=recovery_guidance,
            **kwargs
        )


class TemplateProcessingError(PipelineCompilationError):
    """Error during template variable processing."""
    
    def __init__(
        self,
        message: str,
        template_variables: Optional[List[str]] = None,
        missing_variables: Optional[List[str]] = None,
        **kwargs
    ):
        context = kwargs.get('context', APIErrorContext())
        context.operation = "template_processing"
        context.metadata.update({
            "template_variables": template_variables or [],
            "missing_variables": missing_variables or []
        })
        
        recovery_guidance = RecoveryGuidance(
            strategy=RecoveryStrategy.MANUAL_INTERVENTION,
            user_actions=[
                "Provide missing template variables",
                "Check variable naming and syntax",
                "Verify context variable definitions"
            ],
            recovery_steps=[
                "1. Identify missing variables: " + str(missing_variables) if missing_variables else "",
                "2. Add variables to compilation context",
                "3. Retry compilation with complete context"
            ]
        )
        
        super().__init__(
            message=message,
            context=context,
            recovery_guidance=recovery_guidance,
            **kwargs
        )


# Execution Error Classes

class PipelineExecutionError(OrchestratorAPIError):
    """Error during pipeline execution."""
    
    def __init__(
        self,
        message: str,
        pipeline_id: Optional[str] = None,
        execution_id: Optional[str] = None,
        failed_step: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', APIErrorContext())
        context.operation = "pipeline_execution"
        context.pipeline_id = pipeline_id
        context.execution_id = execution_id
        context.step_name = failed_step
        
        recovery_guidance = RecoveryGuidance(
            strategy=RecoveryStrategy.RETRY,
            automatic_recovery=True,
            user_actions=[
                "Check execution logs for detailed error information",
                "Verify pipeline inputs and context",
                "Review step-specific error details"
            ],
            system_actions=[
                "Attempt automatic retry with backoff",
                "Save execution state for recovery",
                "Generate detailed error report"
            ]
        )
        
        super().__init__(
            message=message,
            category=APIErrorCategory.EXECUTION,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_guidance=recovery_guidance,
            **kwargs
        )


class ExecutionTimeoutError(PipelineExecutionError):
    """Error when pipeline execution times out."""
    
    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[int] = None,
        elapsed_seconds: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.get('context', APIErrorContext())
        context.operation = "execution_timeout"
        context.metadata.update({
            "timeout_seconds": timeout_seconds,
            "elapsed_seconds": elapsed_seconds
        })
        
        recovery_guidance = RecoveryGuidance(
            strategy=RecoveryStrategy.RETRY_WITH_BACKOFF,
            automatic_recovery=False,
            user_actions=[
                "Increase execution timeout if appropriate",
                "Check for infinite loops or hanging operations",
                "Review resource availability and performance"
            ],
            recovery_steps=[
                "1. Analyze step execution times",
                "2. Adjust timeout settings",
                "3. Optimize slow operations",
                "4. Retry with extended timeout"
            ]
        )
        
        super().__init__(
            message=message,
            context=context,
            recovery_guidance=recovery_guidance,
            **kwargs
        )


class StepExecutionError(PipelineExecutionError):
    """Error during individual step execution."""
    
    def __init__(
        self,
        message: str,
        step_id: Optional[str] = None,
        step_type: Optional[str] = None,
        step_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        context = kwargs.get('context', APIErrorContext())
        context.operation = "step_execution"
        context.step_id = step_id
        context.metadata.update({
            "step_type": step_type,
            "step_config": step_config or {}
        })
        
        recovery_guidance = RecoveryGuidance(
            strategy=RecoveryStrategy.RETRY,
            automatic_recovery=True,
            user_actions=[
                "Review step configuration and inputs",
                "Check step-specific requirements",
                "Verify model/tool availability"
            ],
            troubleshooting_tips=[
                f"Step type: {step_type}" if step_type else "Check step type configuration",
                "Verify input data format and availability",
                "Check model/tool connectivity and authentication"
            ]
        )
        
        super().__init__(
            message=message,
            context=context,
            recovery_guidance=recovery_guidance,
            **kwargs
        )


# Configuration Error Classes

class APIConfigurationError(OrchestratorAPIError):
    """Error in API configuration or setup."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        **kwargs
    ):
        context = kwargs.get('context', APIErrorContext())
        context.operation = "api_configuration"
        context.metadata.update({
            "config_key": config_key,
            "config_value": str(config_value) if config_value is not None else None
        })
        
        recovery_guidance = RecoveryGuidance(
            strategy=RecoveryStrategy.MANUAL_INTERVENTION,
            user_actions=[
                "Review API configuration settings",
                "Check configuration file syntax and values",
                "Verify environment variables and dependencies"
            ],
            recovery_steps=[
                "1. Identify configuration issue",
                "2. Update configuration with correct values", 
                "3. Restart API with new configuration"
            ]
        )
        
        super().__init__(
            message=message,
            category=APIErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_guidance=recovery_guidance,
            **kwargs
        )


class ModelRegistryError(APIConfigurationError):
    """Error with model registry configuration or access."""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        registry_type: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', APIErrorContext())
        context.operation = "model_registry_access"
        context.metadata.update({
            "model_name": model_name,
            "registry_type": registry_type
        })
        
        recovery_guidance = RecoveryGuidance(
            strategy=RecoveryStrategy.MANUAL_INTERVENTION,
            user_actions=[
                "Verify model registry configuration",
                "Check model availability and permissions",
                "Update model registry settings if needed"
            ],
            troubleshooting_tips=[
                f"Model: {model_name}" if model_name else "Check model name and availability",
                "Verify model registry connectivity",
                "Check authentication credentials"
            ]
        )
        
        super().__init__(
            message=message,
            context=context,
            recovery_guidance=recovery_guidance,
            **kwargs
        )


# Resource Error Classes

class ResourceError(OrchestratorAPIError):
    """Error related to resource availability or allocation."""
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', APIErrorContext())
        context.operation = "resource_management"
        context.metadata.update({
            "resource_type": resource_type,
            "resource_id": resource_id
        })
        
        recovery_guidance = RecoveryGuidance(
            strategy=RecoveryStrategy.RETRY_WITH_BACKOFF,
            automatic_recovery=True,
            user_actions=[
                "Check resource availability and limits",
                "Verify resource configuration",
                "Monitor resource usage patterns"
            ],
            system_actions=[
                "Retry resource allocation with backoff",
                "Monitor resource pool status",
                "Implement resource usage optimization"
            ]
        )
        
        super().__init__(
            message=message,
            category=APIErrorCategory.RESOURCE_MANAGEMENT,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recovery_guidance=recovery_guidance,
            **kwargs
        )


class NetworkError(OrchestratorAPIError):
    """Error related to network connectivity or external services."""
    
    def __init__(
        self,
        message: str,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.get('context', APIErrorContext())
        context.operation = "network_request"
        context.metadata.update({
            "endpoint": endpoint,
            "status_code": status_code
        })
        
        recovery_guidance = RecoveryGuidance(
            strategy=RecoveryStrategy.RETRY_WITH_BACKOFF,
            automatic_recovery=True,
            user_actions=[
                "Check network connectivity",
                "Verify service endpoint availability",
                "Review authentication credentials"
            ],
            system_actions=[
                "Implement exponential backoff retry",
                "Switch to backup endpoints if available",
                "Monitor service health status"
            ],
            estimated_recovery_time=30
        )
        
        super().__init__(
            message=message,
            category=APIErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recovery_guidance=recovery_guidance,
            **kwargs
        )


# User Input Error Classes

class UserInputError(OrchestratorAPIError):
    """Error related to user input validation or processing."""
    
    def __init__(
        self,
        message: str,
        input_field: Optional[str] = None,
        expected_type: Optional[str] = None,
        provided_value: Optional[Any] = None,
        **kwargs
    ):
        context = kwargs.get('context', APIErrorContext())
        context.operation = "input_validation"
        context.metadata.update({
            "input_field": input_field,
            "expected_type": expected_type,
            "provided_value": str(provided_value) if provided_value is not None else None
        })
        
        recovery_guidance = RecoveryGuidance(
            strategy=RecoveryStrategy.MANUAL_INTERVENTION,
            user_actions=[
                "Review input requirements and format",
                "Correct invalid input values",
                "Check data types and constraints"
            ],
            recovery_steps=[
                f"1. Fix {input_field} field" if input_field else "1. Review input fields",
                f"2. Ensure type is {expected_type}" if expected_type else "2. Check data types",
                "3. Retry operation with corrected input"
            ]
        )
        
        super().__init__(
            message=message,
            category=APIErrorCategory.INPUT_VALIDATION,
            severity=ErrorSeverity.LOW,
            context=context,
            recovery_guidance=recovery_guidance,
            **kwargs
        )


# Error Handler Integration

class APIErrorHandler:
    """
    Centralized error handler for API operations with integration to recovery management.
    
    Provides consistent error handling, logging, recovery guidance, and integration
    with the underlying execution recovery system.
    """
    
    def __init__(self, recovery_manager: Optional[RecoveryManager] = None):
        """
        Initialize error handler.
        
        Args:
            recovery_manager: Optional recovery manager for automatic recovery
        """
        self.recovery_manager = recovery_manager
        self._error_handlers: Dict[Type[Exception], Callable] = {}
        self._error_history: List[OrchestratorAPIError] = []
        
        # Register default error handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default error handlers for common exception types."""
        from ..compiler import YAMLCompilerError
        
        self._error_handlers.update({
            YAMLCompilerError: self._handle_yaml_compiler_error,
            FileNotFoundError: self._handle_file_not_found_error,
            PermissionError: self._handle_permission_error,
            TimeoutError: self._handle_timeout_error,
            ConnectionError: self._handle_network_error,
            ValueError: self._handle_value_error,
            TypeError: self._handle_type_error,
        })
    
    def handle_error(
        self,
        exception: Exception,
        context: Optional[APIErrorContext] = None,
        operation: Optional[str] = None
    ) -> OrchestratorAPIError:
        """
        Handle an exception and convert it to structured API error.
        
        Args:
            exception: The original exception
            context: Optional error context
            operation: Optional operation name
            
        Returns:
            Structured API error with recovery guidance
        """
        # Create context if not provided
        if context is None:
            context = APIErrorContext()
        if operation:
            context.operation = operation
        
        # Check for specific error handlers
        exception_type = type(exception)
        if exception_type in self._error_handlers:
            api_error = self._error_handlers[exception_type](exception, context)
        else:
            # Generic error handling
            api_error = self._handle_generic_error(exception, context)
        
        # Add to error history
        self._error_history.append(api_error)
        
        # Attempt automatic recovery if configured
        if self.recovery_manager and api_error.recovery_guidance.automatic_recovery:
            self._attempt_automatic_recovery(api_error)
        
        return api_error
    
    def _handle_yaml_compiler_error(
        self, 
        exception: Exception, 
        context: APIErrorContext
    ) -> YAMLValidationError:
        """Handle YAML compiler errors."""
        return YAMLValidationError(
            message=f"YAML compilation failed: {str(exception)}",
            context=context,
            original_exception=exception
        )
    
    def _handle_file_not_found_error(
        self, 
        exception: FileNotFoundError, 
        context: APIErrorContext
    ) -> APIConfigurationError:
        """Handle file not found errors."""
        return APIConfigurationError(
            message=f"Required file not found: {str(exception)}",
            context=context,
            original_exception=exception
        )
    
    def _handle_permission_error(
        self, 
        exception: PermissionError, 
        context: APIErrorContext
    ) -> OrchestratorAPIError:
        """Handle permission errors."""
        return OrchestratorAPIError(
            message=f"Permission denied: {str(exception)}",
            category=APIErrorCategory.PERMISSION,
            severity=ErrorSeverity.HIGH,
            context=context,
            original_exception=exception
        )
    
    def _handle_timeout_error(
        self, 
        exception: TimeoutError, 
        context: APIErrorContext
    ) -> ExecutionTimeoutError:
        """Handle timeout errors."""
        return ExecutionTimeoutError(
            message=f"Operation timed out: {str(exception)}",
            context=context,
            original_exception=exception
        )
    
    def _handle_network_error(
        self, 
        exception: ConnectionError, 
        context: APIErrorContext
    ) -> NetworkError:
        """Handle network/connection errors."""
        return NetworkError(
            message=f"Network error: {str(exception)}",
            context=context,
            original_exception=exception
        )
    
    def _handle_value_error(
        self, 
        exception: ValueError, 
        context: APIErrorContext
    ) -> UserInputError:
        """Handle value errors."""
        return UserInputError(
            message=f"Invalid input value: {str(exception)}",
            context=context,
            original_exception=exception
        )
    
    def _handle_type_error(
        self, 
        exception: TypeError, 
        context: APIErrorContext
    ) -> UserInputError:
        """Handle type errors."""
        return UserInputError(
            message=f"Invalid input type: {str(exception)}",
            context=context,
            original_exception=exception
        )
    
    def _handle_generic_error(
        self, 
        exception: Exception, 
        context: APIErrorContext
    ) -> OrchestratorAPIError:
        """Handle generic/unknown errors."""
        return OrchestratorAPIError(
            message=f"Unexpected error: {str(exception)}",
            category=APIErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            original_exception=exception,
            recovery_guidance=RecoveryGuidance(
                strategy=RecoveryStrategy.MANUAL_INTERVENTION,
                user_actions=[
                    "Review error details and context",
                    "Check system logs for additional information",
                    "Contact support if issue persists"
                ]
            )
        )
    
    def _attempt_automatic_recovery(self, api_error: OrchestratorAPIError):
        """Attempt automatic recovery if configured."""
        if not self.recovery_manager:
            return
        
        try:
            # Convert to foundation ErrorInfo
            error_info = api_error.to_error_info()
            
            # Attempt recovery
            recovery_plan = self.recovery_manager.create_recovery_plan(error_info)
            if recovery_plan.strategy == RecoveryStrategy.RETRY:
                logger.info(f"Attempting automatic recovery for error {api_error.context.error_id}")
                # Recovery implementation would go here
                
        except Exception as recovery_error:
            logger.error(f"Automatic recovery failed: {recovery_error}")
    
    def get_error_history(self) -> List[OrchestratorAPIError]:
        """Get history of handled errors."""
        return self._error_history.copy()
    
    def clear_error_history(self):
        """Clear error history."""
        self._error_history.clear()


# Convenience functions for error handling

def create_api_error_handler(recovery_manager: Optional[RecoveryManager] = None) -> APIErrorHandler:
    """Create an API error handler with optional recovery manager."""
    return APIErrorHandler(recovery_manager=recovery_manager)


def handle_api_exception(
    exception: Exception,
    operation: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> OrchestratorAPIError:
    """
    Convenience function to handle exceptions and convert to API errors.
    
    Args:
        exception: The original exception
        operation: Optional operation name
        context: Optional additional context
        
    Returns:
        Structured API error
    """
    error_context = APIErrorContext()
    if operation:
        error_context.operation = operation
    if context:
        error_context.metadata.update(context)
    
    handler = APIErrorHandler()
    return handler.handle_error(exception, error_context, operation)