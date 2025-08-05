"""
Enhanced error handling infrastructure for task failure recovery.
Provides comprehensive error handler configuration and error context management.
"""

from __future__ import annotations

import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


@dataclass
class ErrorHandler:
    """Advanced error handler configuration for task failure recovery."""
    
    # Handler task configuration
    handler_task_id: Optional[str] = None          # Task to execute on error
    handler_action: Optional[str] = None           # Direct action (alternative to task_id)
    
    # Error filtering - determines which errors this handler should catch
    error_types: List[str] = field(default_factory=list)  # Specific error types to catch
    error_patterns: List[str] = field(default_factory=list)  # Error message patterns
    error_codes: List[Union[int, str]] = field(default_factory=list)  # HTTP codes, exit codes, etc.
    
    # Recovery behavior
    retry_with_handler: bool = True                # Retry original after handler success
    max_handler_retries: int = 0                   # Retries for the handler itself
    propagate_error: bool = False                  # Propagate if handler fails
    continue_on_handler_failure: bool = False      # Continue pipeline if handler fails
    
    # Fallback configuration
    fallback_value: Optional[Any] = None           # Value to return on handler failure
    fallback_result: Optional[Dict[str, Any]] = None  # Complete result structure
    
    # Context and logging
    capture_error_context: bool = True             # Include full error context
    log_level: str = "error"                       # Logging level for handler execution
    timeout: Optional[float] = None                # Handler execution timeout (seconds)
    
    # Priority and ordering
    priority: int = 100                            # Handler priority (lower = higher priority)
    enabled: bool = True                           # Whether this handler is enabled
    
    def __post_init__(self):
        """Validate error handler configuration."""
        if not self.handler_task_id and not self.handler_action and not self.fallback_value:
            raise ValueError("ErrorHandler must specify at least one of: handler_task_id, handler_action, or fallback_value")
        
        if self.handler_task_id and self.handler_action:
            raise ValueError("ErrorHandler cannot specify both handler_task_id and handler_action")
        
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("Handler timeout must be positive")
        
        if self.max_handler_retries < 0:
            raise ValueError("max_handler_retries must be non-negative")
    
    def matches_error(self, error: Exception, task_id: str = None) -> bool:
        """Check if this handler should handle the given error."""
        if not self.enabled:
            return False
        
        # Check error types
        if self.error_types:
            error_type_name = type(error).__name__
            # Support wildcard matching
            if "*" in self.error_types:
                return True
            if error_type_name in self.error_types:
                return True
            # Check inheritance
            for error_type_str in self.error_types:
                try:
                    # Try to match by inheritance (e.g., "Exception" matches all exceptions)
                    if error_type_str == "Exception" or issubclass(type(error), Exception):
                        if error_type_str in str(type(error).__mro__):
                            return True
                except:
                    pass
        
        # Check error message patterns
        if self.error_patterns:
            error_message = str(error)
            for pattern in self.error_patterns:
                if pattern in error_message:
                    return True
                # Support regex patterns
                try:
                    import re
                    if re.search(pattern, error_message, re.IGNORECASE):
                        return True
                except:
                    pass
        
        # Check error codes (for HTTP errors, etc.)
        if self.error_codes:
            error_code = getattr(error, 'code', None) or getattr(error, 'status_code', None)
            if error_code in self.error_codes:
                return True
        
        # If no filters specified, handle all errors
        if not self.error_types and not self.error_patterns and not self.error_codes:
            return True
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'handler_task_id': self.handler_task_id,
            'handler_action': self.handler_action,
            'error_types': self.error_types,
            'error_patterns': self.error_patterns,
            'error_codes': self.error_codes,
            'retry_with_handler': self.retry_with_handler,
            'max_handler_retries': self.max_handler_retries,
            'propagate_error': self.propagate_error,
            'continue_on_handler_failure': self.continue_on_handler_failure,
            'fallback_value': self.fallback_value,
            'fallback_result': self.fallback_result,
            'capture_error_context': self.capture_error_context,
            'log_level': self.log_level,
            'timeout': self.timeout,
            'priority': self.priority,
            'enabled': self.enabled
        }


@dataclass
class ErrorContext:
    """Comprehensive error context for handlers with full execution details."""
    
    # Task information
    failed_task_id: str
    failed_task_name: str = ""
    task_parameters: Dict[str, Any] = field(default_factory=dict)
    task_dependencies: List[str] = field(default_factory=list)
    task_result: Optional[Any] = None
    task_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Error details
    error_type: str = ""
    error_message: str = ""
    error_traceback: Optional[str] = None
    error_timestamp: datetime = field(default_factory=datetime.now)
    error_code: Optional[Union[int, str]] = None
    
    # Execution context
    pipeline_context: Dict[str, Any] = field(default_factory=dict)
    execution_attempt: int = 1
    previous_attempts: List[Dict[str, Any]] = field(default_factory=list)
    pipeline_id: Optional[str] = None
    pipeline_step: Optional[int] = None
    
    # Recovery information
    recovery_suggestions: List[str] = field(default_factory=list)
    similar_failures: List[str] = field(default_factory=list)
    error_frequency: Optional[Dict[str, int]] = None
    
    # System context
    system_info: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize error context with additional information."""
        if not self.failed_task_name:
            self.failed_task_name = self.failed_task_id
        
        # Capture system information
        if not self.system_info:
            import platform
            import psutil
            try:
                self.system_info = {
                    'platform': platform.platform(),
                    'python_version': platform.python_version(),
                    'memory_usage': psutil.virtual_memory().percent,
                    'cpu_usage': psutil.cpu_percent(),
                    'disk_usage': psutil.disk_usage('/').percent if platform.system() != 'Windows' else None
                }
            except ImportError:
                # psutil not available
                self.system_info = {
                    'platform': platform.platform(),
                    'python_version': platform.python_version()
                }
    
    def add_recovery_suggestion(self, suggestion: str) -> None:
        """Add a recovery suggestion."""
        if suggestion not in self.recovery_suggestions:
            self.recovery_suggestions.append(suggestion)
    
    def add_similar_failure(self, failure_info: str) -> None:
        """Add information about a similar failure."""
        if failure_info not in self.similar_failures:
            self.similar_failures.append(failure_info)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for serialization."""
        return {
            'failed_task_id': self.failed_task_id,
            'failed_task_name': self.failed_task_name,
            'task_parameters': self.task_parameters,
            'task_dependencies': self.task_dependencies,
            'task_result': self.task_result,
            'task_metadata': self.task_metadata,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'error_traceback': self.error_traceback,
            'error_timestamp': self.error_timestamp.isoformat(),
            'error_code': self.error_code,
            'pipeline_context': self.pipeline_context,
            'execution_attempt': self.execution_attempt,
            'previous_attempts': self.previous_attempts,
            'pipeline_id': self.pipeline_id,
            'pipeline_step': self.pipeline_step,
            'recovery_suggestions': self.recovery_suggestions,
            'similar_failures': self.similar_failures,
            'error_frequency': self.error_frequency,
            'system_info': self.system_info,
            'resource_usage': self.resource_usage
        }
    
    @classmethod
    def from_exception(
        cls,
        failed_task_id: str,
        error: Exception,
        task_parameters: Dict[str, Any] = None,
        pipeline_context: Dict[str, Any] = None,
        **kwargs
    ) -> ErrorContext:
        """Create ErrorContext from an exception."""
        return cls(
            failed_task_id=failed_task_id,
            task_parameters=task_parameters or {},
            pipeline_context=pipeline_context or {},
            error_type=type(error).__name__,
            error_message=str(error),
            error_traceback=traceback.format_exc(),
            error_code=getattr(error, 'code', None) or getattr(error, 'status_code', None),
            **kwargs
        )


@dataclass
class ErrorHandlerResult:
    """Result of error handler execution."""
    
    # Execution results
    success: bool
    handler_id: Optional[str] = None
    handler_output: Optional[Any] = None
    execution_time: Optional[float] = None
    
    # Error information (if handler failed)
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    handler_traceback: Optional[str] = None
    
    # Recovery information
    should_retry_original: bool = False
    should_continue_pipeline: bool = False
    fallback_value: Optional[Any] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    handler_attempts: int = 1
    context_modifications: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'success': self.success,
            'handler_id': self.handler_id,
            'handler_output': self.handler_output,
            'execution_time': self.execution_time,
            'error_message': self.error_message,
            'error_type': self.error_type,
            'handler_traceback': self.handler_traceback,
            'should_retry_original': self.should_retry_original,
            'should_continue_pipeline': self.should_continue_pipeline,
            'fallback_value': self.fallback_value,
            'timestamp': self.timestamp.isoformat(),
            'handler_attempts': self.handler_attempts,
            'context_modifications': self.context_modifications
        }


def create_error_handler(
    handler_task_id: Optional[str] = None,
    handler_action: Optional[str] = None,
    error_types: List[str] = None,
    error_patterns: List[str] = None,
    retry_with_handler: bool = True,
    **kwargs
) -> ErrorHandler:
    """Convenience function to create ErrorHandler with validation."""
    return ErrorHandler(
        handler_task_id=handler_task_id,
        handler_action=handler_action,
        error_types=error_types or [],
        error_patterns=error_patterns or [],
        retry_with_handler=retry_with_handler,
        **kwargs
    )


def create_simple_error_handler(handler_action: str, error_types: List[str] = None) -> ErrorHandler:
    """Create a simple error handler for basic use cases."""
    return ErrorHandler(
        handler_action=handler_action,
        error_types=error_types or ["*"],
        retry_with_handler=True,
        capture_error_context=True
    )