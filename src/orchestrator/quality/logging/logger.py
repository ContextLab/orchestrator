"""
Comprehensive structured logging framework for orchestrator quality control.

This module provides a sophisticated logging system that captures detailed execution
insights, integrates with external monitoring systems, and supports multiple 
verbosity levels for pipeline debugging and quality assurance.

Key Features:
- Structured logging with JSON formatting for external systems
- Context-aware logging with execution and validation metadata
- Performance-focused logging with minimal overhead
- Integration with quality validation events
- Support for multiple log levels and filtering
- External monitoring system compatibility (Prometheus, Grafana, etc.)
"""

from __future__ import annotations

import json
import logging
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Iterator
from pathlib import Path
import traceback

from ...execution.state import ExecutionContext, ExecutionStatus
from ...execution.progress import ProgressTracker, ProgressEvent, ProgressEventType


class LogLevel(Enum):
    """Enhanced log levels for quality control system."""
    TRACE = 5       # Very detailed debugging
    DEBUG = 10      # Detailed debugging information
    INFO = 20       # General information
    WARNING = 30    # Warning conditions
    ERROR = 40      # Error conditions
    CRITICAL = 50   # Critical errors
    AUDIT = 60      # Audit trail events


class LogCategory(Enum):
    """Log event categories for structured logging."""
    EXECUTION = "execution"
    VALIDATION = "validation"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    MONITORING = "monitoring"
    SECURITY = "security"
    AUDIT = "audit"
    USER_ACTION = "user_action"
    SYSTEM = "system"
    INTEGRATION = "integration"


@dataclass
class LogContext:
    """Structured logging context for comprehensive event tracking."""
    timestamp: str
    level: str
    category: LogCategory
    component: str
    operation: str
    execution_id: Optional[str] = None
    pipeline_id: Optional[str] = None
    session_id: Optional[str] = None
    step_id: Optional[str] = None
    user_id: Optional[str] = None
    trace_id: Optional[str] = None
    
    # Performance metrics
    duration_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    
    # Quality metrics
    validation_score: Optional[float] = None
    rule_violations: Optional[int] = None
    quality_level: Optional[str] = None
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = None
    error_details: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['category'] = self.category.value
        # Remove None values for cleaner logs
        return {k: v for k, v in result.items() if v is not None}


@dataclass
class QualityEvent:
    """Quality-specific logging event with validation integration."""
    event_type: str
    severity: str
    source_component: str
    validation_result: Optional[Dict[str, Any]] = None
    rule_violations: Optional[List[Dict[str, Any]]] = None
    quality_score: Optional[float] = None
    recommendations: Optional[List[str]] = None
    remediation_actions: Optional[List[str]] = None


class StructuredLogger:
    """
    Advanced structured logger with quality control integration.
    
    Provides comprehensive logging capabilities for pipeline execution,
    validation events, performance metrics, and quality assurance.
    """
    
    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        enable_structured: bool = True,
        enable_performance_logging: bool = True,
        enable_quality_logging: bool = True,
        buffer_size: int = 1000,
        flush_interval: float = 5.0
    ):
        self.name = name
        self.level = level
        self.enable_structured = enable_structured
        self.enable_performance_logging = enable_performance_logging
        self.enable_quality_logging = enable_quality_logging
        
        # Initialize standard logger
        self._logger = logging.getLogger(f"orchestrator.quality.{name}")
        self._logger.setLevel(level.value)
        
        # Context tracking
        self._context_stack: List[Dict[str, Any]] = []
        self._local = threading.local()
        
        # Performance tracking
        self._operation_times: Dict[str, float] = {}
        self._performance_buffer: List[Dict[str, Any]] = []
        
        # Quality event tracking
        self._quality_events: List[QualityEvent] = []
        self._validation_context: Optional[Dict[str, Any]] = None
        
        # Buffering for high-performance logging
        self._log_buffer: List[Dict[str, Any]] = []
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval
        self._last_flush = time.time()
        self._buffer_lock = threading.Lock()

    def _get_current_context(self) -> Dict[str, Any]:
        """Get current logging context from thread-local storage."""
        if not hasattr(self._local, 'context'):
            self._local.context = {}
        return self._local.context

    def _should_log(self, level: LogLevel) -> bool:
        """Check if message should be logged at given level."""
        return level.value >= self.level.value

    def _create_log_context(
        self,
        level: LogLevel,
        category: LogCategory,
        operation: str,
        **kwargs
    ) -> LogContext:
        """Create structured log context with current metadata."""
        context = self._get_current_context()
        
        return LogContext(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=level.name,
            category=category,
            component=self.name,
            operation=operation,
            execution_id=context.get('execution_id'),
            pipeline_id=context.get('pipeline_id'),
            session_id=context.get('session_id'),
            step_id=context.get('step_id'),
            user_id=context.get('user_id'),
            trace_id=context.get('trace_id'),
            **kwargs
        )

    def _format_message(self, context: LogContext, message: str, **kwargs) -> str:
        """Format log message with structured context."""
        if not self.enable_structured:
            return message
            
        log_data = context.to_dict()
        log_data['message'] = message
        log_data.update(kwargs)
        
        return json.dumps(log_data, default=str)

    def _emit_log(self, level: LogLevel, context: LogContext, message: str, **kwargs):
        """Emit log message through appropriate handlers."""
        if not self._should_log(level):
            return
            
        formatted_message = self._format_message(context, message, **kwargs)
        
        # Use buffering for high-volume logging
        if self._buffer_size > 0:
            with self._buffer_lock:
                self._log_buffer.append({
                    'level': level,
                    'message': formatted_message,
                    'timestamp': time.time()
                })
                
                if (len(self._log_buffer) >= self._buffer_size or 
                    time.time() - self._last_flush > self._flush_interval):
                    self._flush_buffer()
        else:
            # Direct logging
            self._logger.log(level.value, formatted_message)

    def _flush_buffer(self):
        """Flush buffered log messages."""
        if not self._log_buffer:
            return
            
        try:
            for entry in self._log_buffer:
                self._logger.log(entry['level'].value, entry['message'])
            self._log_buffer.clear()
            self._last_flush = time.time()
        except Exception as e:
            # Fallback logging to prevent log loss
            self._logger.error(f"Failed to flush log buffer: {e}")

    @contextmanager
    def context(self, **context_vars) -> Iterator[None]:
        """Context manager for adding logging context."""
        current_context = self._get_current_context()
        old_values = {}
        
        # Save old values and set new ones
        for key, value in context_vars.items():
            old_values[key] = current_context.get(key)
            current_context[key] = value
            
        try:
            yield
        finally:
            # Restore old values
            for key, old_value in old_values.items():
                if old_value is None:
                    current_context.pop(key, None)
                else:
                    current_context[key] = old_value

    @contextmanager
    def operation_timer(self, operation: str, category: LogCategory = LogCategory.PERFORMANCE) -> Iterator[None]:
        """Context manager for timing operations."""
        if not self.enable_performance_logging:
            yield
            return
            
        start_time = time.perf_counter()
        operation_key = f"{self.name}.{operation}"
        
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            context = self._create_log_context(
                LogLevel.DEBUG,
                category,
                operation,
                duration_ms=duration_ms
            )
            
            self.debug(
                f"Operation '{operation}' completed",
                context=context,
                performance_metrics={'duration_ms': duration_ms}
            )

    def set_validation_context(self, validation_context: Dict[str, Any]):
        """Set validation context for quality logging."""
        self._validation_context = validation_context

    def log_quality_event(
        self,
        event: QualityEvent,
        level: LogLevel = LogLevel.INFO,
        **kwargs
    ):
        """Log quality-specific events with validation integration."""
        if not self.enable_quality_logging:
            return
            
        context = self._create_log_context(
            level,
            LogCategory.QUALITY,
            f"quality_event_{event.event_type}",
            quality_level=event.severity,
            validation_score=event.quality_score,
            rule_violations=len(event.rule_violations) if event.rule_violations else None,
            **kwargs
        )
        
        self._quality_events.append(event)
        
        self._emit_log(
            level,
            context,
            f"Quality event: {event.event_type}",
            event_data=asdict(event)
        )

    def log_validation_result(
        self,
        validation_result: Dict[str, Any],
        level: LogLevel = LogLevel.INFO,
        **kwargs
    ):
        """Log validation results with quality metrics."""
        if not self.enable_quality_logging:
            return
            
        severity = validation_result.get('severity', 'unknown')
        score = validation_result.get('quality_score', 0.0)
        violations = validation_result.get('violations', [])
        
        context = self._create_log_context(
            level,
            LogCategory.VALIDATION,
            'validation_result',
            quality_level=severity,
            validation_score=score,
            rule_violations=len(violations),
            **kwargs
        )
        
        self._emit_log(
            level,
            context,
            f"Validation completed with {len(violations)} violations",
            validation_result=validation_result
        )

    # Standard logging methods with structured context
    def trace(self, message: str, category: LogCategory = LogCategory.EXECUTION, **kwargs):
        """Log trace-level message."""
        context = self._create_log_context(LogLevel.TRACE, category, 'trace', **kwargs)
        self._emit_log(LogLevel.TRACE, context, message)

    def debug(self, message: str, category: LogCategory = LogCategory.EXECUTION, **kwargs):
        """Log debug-level message."""
        context = self._create_log_context(LogLevel.DEBUG, category, 'debug', **kwargs)
        self._emit_log(LogLevel.DEBUG, context, message)

    def info(self, message: str, category: LogCategory = LogCategory.EXECUTION, **kwargs):
        """Log info-level message."""
        context = self._create_log_context(LogLevel.INFO, category, 'info', **kwargs)
        self._emit_log(LogLevel.INFO, context, message)

    def warning(self, message: str, category: LogCategory = LogCategory.EXECUTION, **kwargs):
        """Log warning-level message."""
        context = self._create_log_context(LogLevel.WARNING, category, 'warning', **kwargs)
        self._emit_log(LogLevel.WARNING, context, message)

    def error(self, message: str, category: LogCategory = LogCategory.EXECUTION, exception: Optional[Exception] = None, **kwargs):
        """Log error-level message with optional exception details."""
        error_details = None
        if exception:
            error_details = {
                'exception_type': type(exception).__name__,
                'exception_message': str(exception),
                'traceback': traceback.format_exc()
            }
            
        context = self._create_log_context(
            LogLevel.ERROR, 
            category, 
            'error', 
            error_details=error_details,
            **kwargs
        )
        self._emit_log(LogLevel.ERROR, context, message)

    def critical(self, message: str, category: LogCategory = LogCategory.EXECUTION, exception: Optional[Exception] = None, **kwargs):
        """Log critical-level message with optional exception details."""
        error_details = None
        if exception:
            error_details = {
                'exception_type': type(exception).__name__,
                'exception_message': str(exception),
                'traceback': traceback.format_exc()
            }
            
        context = self._create_log_context(
            LogLevel.CRITICAL, 
            category, 
            'critical', 
            error_details=error_details,
            **kwargs
        )
        self._emit_log(LogLevel.CRITICAL, context, message)

    def audit(self, message: str, user_action: str, **kwargs):
        """Log audit trail events."""
        context = self._create_log_context(
            LogLevel.AUDIT, 
            LogCategory.AUDIT, 
            user_action,
            **kwargs
        )
        self._emit_log(LogLevel.AUDIT, context, message, user_action=user_action)

    def flush(self):
        """Force flush of buffered logs."""
        with self._buffer_lock:
            self._flush_buffer()

    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get accumulated quality metrics for reporting."""
        return {
            'total_quality_events': len(self._quality_events),
            'quality_events_by_type': {
                event.event_type: len([e for e in self._quality_events if e.event_type == event.event_type])
                for event in self._quality_events
            },
            'average_quality_score': (
                sum(e.quality_score for e in self._quality_events if e.quality_score is not None) / 
                len([e for e in self._quality_events if e.quality_score is not None])
                if any(e.quality_score is not None for e in self._quality_events) else 0.0
            )
        }


# Global logger registry for centralized management
_logger_registry: Dict[str, StructuredLogger] = {}
_registry_lock = threading.Lock()


def get_logger(
    name: str,
    level: LogLevel = LogLevel.INFO,
    **kwargs
) -> StructuredLogger:
    """
    Get or create a structured logger instance.
    
    Provides centralized logger management with consistent configuration
    across the quality control system.
    """
    with _registry_lock:
        if name not in _logger_registry:
            _logger_registry[name] = StructuredLogger(name, level, **kwargs)
        return _logger_registry[name]


def flush_all_loggers():
    """Flush all registered loggers."""
    with _registry_lock:
        for logger in _logger_registry.values():
            logger.flush()


def get_all_quality_metrics() -> Dict[str, Any]:
    """Get quality metrics from all registered loggers."""
    metrics = {}
    with _registry_lock:
        for name, logger in _logger_registry.items():
            metrics[name] = logger.get_quality_metrics()
    return metrics