"""
Error Handling and Recovery System for Pipeline Execution.

This module provides comprehensive error handling, recovery mechanisms,
retry logic, and failure management for pipeline execution with integration
to checkpoint and resume functionality.
"""

from __future__ import annotations

import logging
import time
import traceback
import threading
from typing import Any, Dict, List, Optional, Set, Callable, Union, Type
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import json
import asyncio
from contextlib import contextmanager, asynccontextmanager

from .state import ExecutionContext, ExecutionStatus, Checkpoint
from .progress import ProgressTracker, ProgressEvent, ProgressEventType

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Recovery strategies for handling failures."""
    FAIL_FAST = "fail_fast"                    # Immediately fail execution
    RETRY = "retry"                            # Retry failed step
    SKIP = "skip"                              # Skip failed step and continue
    ROLLBACK = "rollback"                      # Rollback to previous checkpoint
    RETRY_WITH_BACKOFF = "retry_with_backoff"  # Retry with exponential backoff
    MANUAL_INTERVENTION = "manual_intervention" # Require manual intervention
    ALTERNATIVE_PATH = "alternative_path"       # Try alternative execution path


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"           # Warning level, execution can continue
    MEDIUM = "medium"     # Error level, step fails but execution might continue
    HIGH = "high"         # Critical error, execution should stop
    CRITICAL = "critical" # System-level error, immediate shutdown required


class ErrorCategory(Enum):
    """Categories of errors that can occur."""
    VALIDATION = "validation"           # Input/output validation errors
    EXECUTION = "execution"             # Step execution errors
    TIMEOUT = "timeout"                 # Timeout errors
    RESOURCE = "resource"               # Resource availability errors
    NETWORK = "network"                 # Network connectivity errors
    AUTHENTICATION = "authentication"   # Authentication/authorization errors
    SYSTEM = "system"                   # System-level errors
    DEPENDENCY = "dependency"           # Missing dependency errors
    CONFIGURATION = "configuration"     # Configuration errors
    USER = "user"                       # User-caused errors
    UNKNOWN = "unknown"                 # Unclassified errors


@dataclass
class ErrorInfo:
    """Detailed error information."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    exception: Optional[Exception] = None
    traceback: Optional[str] = None
    step_id: Optional[str] = None
    step_name: Optional[str] = None
    execution_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error info to dictionary."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['category'] = self.category.value
        result['severity'] = self.severity.value
        result['exception'] = str(self.exception) if self.exception else None
        return result


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 60.0     # seconds
    backoff_factor: float = 2.0
    exponential_backoff: bool = True
    retry_on_categories: Set[ErrorCategory] = field(
        default_factory=lambda: {ErrorCategory.NETWORK, ErrorCategory.TIMEOUT, ErrorCategory.RESOURCE}
    )
    retry_condition: Optional[Callable[[ErrorInfo], bool]] = None
    
    def should_retry(self, error_info: ErrorInfo, attempt: int) -> bool:
        """Check if error should be retried."""
        if attempt >= self.max_attempts:
            return False
        
        if self.retry_condition:
            return self.retry_condition(error_info)
        
        return error_info.category in self.retry_on_categories
    
    def get_delay(self, attempt: int) -> float:
        """Get delay for retry attempt."""
        if not self.exponential_backoff:
            return self.initial_delay
        
        delay = self.initial_delay * (self.backoff_factor ** attempt)
        return min(delay, self.max_delay)


@dataclass
class RecoveryPlan:
    """Plan for recovering from failure."""
    strategy: RecoveryStrategy
    target_checkpoint: Optional[str] = None
    alternative_steps: List[str] = field(default_factory=list)
    retry_config: Optional[RetryConfig] = None
    manual_instructions: Optional[str] = None
    timeout: Optional[timedelta] = None
    
    def is_automated(self) -> bool:
        """Check if recovery can be automated."""
        return self.strategy not in (RecoveryStrategy.MANUAL_INTERVENTION,)


class RecoveryManager:
    """
    Comprehensive recovery management system.
    
    Handles error detection, classification, retry logic, and recovery
    strategies for pipeline execution failures.
    """
    
    def __init__(
        self,
        execution_context: Optional[ExecutionContext] = None,
        progress_tracker: Optional[ProgressTracker] = None
    ):
        """
        Initialize recovery manager.
        
        Args:
            execution_context: Execution context for state management
            progress_tracker: Progress tracker for event integration
        """
        self.execution_context = execution_context
        self.progress_tracker = progress_tracker
        
        # Recovery state
        self._error_history: List[ErrorInfo] = []
        self._recovery_plans: Dict[str, RecoveryPlan] = {}
        self._retry_counts: Dict[str, int] = {}  # step_id -> count
        self._active_recoveries: Dict[str, RecoveryPlan] = {}
        
        # Error handlers
        self._error_handlers: Dict[ErrorCategory, List[Callable[[ErrorInfo], Optional[RecoveryPlan]]]] = {}
        self._global_error_handlers: List[Callable[[ErrorInfo], Optional[RecoveryPlan]]] = []
        
        # Threading and safety
        self._lock = threading.RLock()
        self._shutdown = False
        
        # Default recovery configurations
        self._default_retry_config = RetryConfig()
        self._default_strategy_map = {
            ErrorSeverity.LOW: RecoveryStrategy.RETRY,
            ErrorSeverity.MEDIUM: RecoveryStrategy.RETRY_WITH_BACKOFF,
            ErrorSeverity.HIGH: RecoveryStrategy.ROLLBACK,
            ErrorSeverity.CRITICAL: RecoveryStrategy.FAIL_FAST
        }
        
        logger.info("Initialized RecoveryManager")
    
    def register_error_handler(
        self,
        category: ErrorCategory,
        handler: Callable[[ErrorInfo], Optional[RecoveryPlan]]
    ) -> None:
        """Register error handler for specific category."""
        with self._lock:
            if category not in self._error_handlers:
                self._error_handlers[category] = []
            self._error_handlers[category].append(handler)
            
            logger.debug(f"Registered error handler for category {category.value}")
    
    def register_global_error_handler(
        self,
        handler: Callable[[ErrorInfo], Optional[RecoveryPlan]]
    ) -> None:
        """Register global error handler for all categories."""
        with self._lock:
            self._global_error_handlers.append(handler)
            logger.debug("Registered global error handler")
    
    def set_recovery_plan(self, step_id: str, plan: RecoveryPlan) -> None:
        """Set recovery plan for specific step."""
        with self._lock:
            self._recovery_plans[step_id] = plan
            logger.debug(f"Set recovery plan for step {step_id}: {plan.strategy.value}")
    
    def classify_error(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorInfo:
        """
        Classify error and create error info.
        
        Args:
            exception: The exception that occurred
            context: Additional context information
        
        Returns:
            Classified error information
        """
        error_message = str(exception)
        error_type = type(exception).__name__
        
        # Classify error category
        category = self._classify_error_category(exception, error_message)
        
        # Determine severity
        severity = self._determine_error_severity(exception, category)
        
        # Create error info
        error_info = ErrorInfo(
            error_id=f"err_{int(time.time() * 1000)}",
            category=category,
            severity=severity,
            message=error_message,
            exception=exception,
            traceback=traceback.format_exc(),
            context=context or {},
            execution_id=self.execution_context.execution_id if self.execution_context else None
        )
        
        logger.debug(f"Classified error: {category.value}/{severity.value} - {error_message}")
        return error_info
    
    def _classify_error_category(self, exception: Exception, message: str) -> ErrorCategory:
        """Classify error into category based on exception type and message."""
        exception_type = type(exception).__name__
        message_lower = message.lower()
        
        # Network-related errors
        if any(term in message_lower for term in ['connection', 'network', 'socket', 'dns', 'host']):
            return ErrorCategory.NETWORK
        
        # Timeout errors
        if any(term in message_lower for term in ['timeout', 'expired', 'deadline']):
            return ErrorCategory.TIMEOUT
        
        # Authentication errors
        if any(term in message_lower for term in ['authentication', 'authorization', 'forbidden', 'unauthorized']):
            return ErrorCategory.AUTHENTICATION
        
        # Resource errors
        if any(term in message_lower for term in ['resource', 'memory', 'disk', 'space', 'quota']):
            return ErrorCategory.RESOURCE
        
        # Validation errors
        if any(term in message_lower for term in ['validation', 'invalid', 'format', 'schema']):
            return ErrorCategory.VALIDATION
        
        # Configuration errors
        if any(term in message_lower for term in ['configuration', 'config', 'setting', 'parameter']):
            return ErrorCategory.CONFIGURATION
        
        # Dependency errors
        if any(term in message_lower for term in ['dependency', 'import', 'module', 'package']):
            return ErrorCategory.DEPENDENCY
        
        # System errors
        if any(term in exception_type.lower() for term in ['system', 'os', 'io', 'file']):
            return ErrorCategory.SYSTEM
        
        # Default to execution error
        return ErrorCategory.EXECUTION
    
    def _determine_error_severity(self, exception: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity based on exception and category."""
        exception_type = type(exception).__name__
        
        # Critical system errors
        if category == ErrorCategory.SYSTEM and 'Critical' in exception_type:
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if category in (ErrorCategory.AUTHENTICATION, ErrorCategory.CONFIGURATION):
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if category in (ErrorCategory.NETWORK, ErrorCategory.RESOURCE, ErrorCategory.DEPENDENCY):
            return ErrorSeverity.MEDIUM
        
        # Low severity errors (mostly validation and timeout)
        if category in (ErrorCategory.VALIDATION, ErrorCategory.TIMEOUT):
            return ErrorSeverity.LOW
        
        # Default to medium
        return ErrorSeverity.MEDIUM
    
    def handle_error(
        self,
        exception: Exception,
        step_id: Optional[str] = None,
        step_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> RecoveryPlan:
        """
        Handle error and create recovery plan.
        
        Args:
            exception: The exception that occurred
            step_id: ID of the step where error occurred
            step_name: Name of the step where error occurred
            context: Additional context information
        
        Returns:
            Recovery plan for the error
        """
        with self._lock:
            # Classify error
            error_info = self.classify_error(exception, context)
            error_info.step_id = step_id
            error_info.step_name = step_name
            
            # Add to error history
            self._error_history.append(error_info)
            
            # Try to get recovery plan from handlers
            recovery_plan = self._get_recovery_plan(error_info)
            
            # If no specific plan, create default based on severity
            if not recovery_plan:
                recovery_plan = self._create_default_recovery_plan(error_info)
            
            # Track recovery attempt
            if step_id:
                self._active_recoveries[step_id] = recovery_plan
            
            # Emit progress event
            if self.progress_tracker:
                self.progress_tracker._emit_event(ProgressEvent(
                    event_type=ProgressEventType.STEP_FAILED,
                    execution_id=error_info.execution_id,
                    step_id=step_id,
                    step_name=step_name,
                    message=error_info.message,
                    data={
                        "error_id": error_info.error_id,
                        "category": error_info.category.value,
                        "severity": error_info.severity.value,
                        "recovery_strategy": recovery_plan.strategy.value
                    }
                ))
            
            logger.warning(f"Error handled for step {step_id}: {error_info.message}")
            logger.info(f"Recovery plan: {recovery_plan.strategy.value}")
            
            return recovery_plan
    
    def _get_recovery_plan(self, error_info: ErrorInfo) -> Optional[RecoveryPlan]:
        """Get recovery plan from registered handlers."""
        # Try category-specific handlers first
        if error_info.category in self._error_handlers:
            for handler in self._error_handlers[error_info.category]:
                try:
                    plan = handler(error_info)
                    if plan:
                        return plan
                except Exception as e:
                    logger.error(f"Error in recovery handler: {e}")
        
        # Try global handlers
        for handler in self._global_error_handlers:
            try:
                plan = handler(error_info)
                if plan:
                    return plan
            except Exception as e:
                logger.error(f"Error in global recovery handler: {e}")
        
        # Try step-specific recovery plan
        if error_info.step_id and error_info.step_id in self._recovery_plans:
            return self._recovery_plans[error_info.step_id]
        
        return None
    
    def _create_default_recovery_plan(self, error_info: ErrorInfo) -> RecoveryPlan:
        """Create default recovery plan based on error severity."""
        strategy = self._default_strategy_map.get(error_info.severity, RecoveryStrategy.RETRY)
        
        recovery_plan = RecoveryPlan(strategy=strategy)
        
        # Configure retry for appropriate strategies
        if strategy in (RecoveryStrategy.RETRY, RecoveryStrategy.RETRY_WITH_BACKOFF):
            recovery_plan.retry_config = self._default_retry_config
        
        # Set timeout based on severity
        if error_info.severity == ErrorSeverity.HIGH:
            recovery_plan.timeout = timedelta(minutes=5)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            recovery_plan.timeout = timedelta(minutes=2)
        
        return recovery_plan
    
    async def execute_recovery(
        self,
        step_id: str,
        recovery_plan: RecoveryPlan,
        step_executor: Callable[[], Any]
    ) -> bool:
        """
        Execute recovery plan.
        
        Args:
            step_id: ID of the step to recover
            recovery_plan: Recovery plan to execute
            step_executor: Function to execute the step
        
        Returns:
            True if recovery succeeded, False otherwise
        """
        logger.info(f"Executing recovery for step {step_id}: {recovery_plan.strategy.value}")
        
        try:
            if recovery_plan.strategy == RecoveryStrategy.FAIL_FAST:
                return False
            
            elif recovery_plan.strategy == RecoveryStrategy.SKIP:
                logger.info(f"Skipping step {step_id} as part of recovery")
                return True
            
            elif recovery_plan.strategy in (RecoveryStrategy.RETRY, RecoveryStrategy.RETRY_WITH_BACKOFF):
                return await self._execute_retry_recovery(step_id, recovery_plan, step_executor)
            
            elif recovery_plan.strategy == RecoveryStrategy.ROLLBACK:
                return await self._execute_rollback_recovery(step_id, recovery_plan)
            
            elif recovery_plan.strategy == RecoveryStrategy.ALTERNATIVE_PATH:
                return await self._execute_alternative_path_recovery(step_id, recovery_plan)
            
            elif recovery_plan.strategy == RecoveryStrategy.MANUAL_INTERVENTION:
                return await self._execute_manual_intervention_recovery(step_id, recovery_plan)
            
            else:
                logger.error(f"Unknown recovery strategy: {recovery_plan.strategy}")
                return False
                
        except Exception as e:
            logger.error(f"Error during recovery execution: {e}")
            return False
    
    async def _execute_retry_recovery(
        self,
        step_id: str,
        recovery_plan: RecoveryPlan,
        step_executor: Callable[[], Any]
    ) -> bool:
        """Execute retry recovery strategy."""
        retry_config = recovery_plan.retry_config or self._default_retry_config
        current_attempt = self._retry_counts.get(step_id, 0)
        
        while current_attempt < retry_config.max_attempts:
            current_attempt += 1
            self._retry_counts[step_id] = current_attempt
            
            # Wait for retry delay
            if current_attempt > 1:  # No delay for first attempt
                delay = retry_config.get_delay(current_attempt - 1)
                logger.info(f"Waiting {delay:.2f} seconds before retry attempt {current_attempt}")
                await asyncio.sleep(delay)
            
            try:
                logger.info(f"Retry attempt {current_attempt} for step {step_id}")
                await step_executor()
                
                # Success - clear retry count and return
                self._retry_counts.pop(step_id, None)
                logger.info(f"Step {step_id} succeeded on retry attempt {current_attempt}")
                return True
                
            except Exception as e:
                error_info = self.classify_error(e)
                
                # Check if we should continue retrying
                if not retry_config.should_retry(error_info, current_attempt):
                    logger.warning(f"Not retrying step {step_id} due to error category: {error_info.category.value}")
                    break
                
                logger.warning(f"Retry attempt {current_attempt} failed for step {step_id}: {e}")
        
        # All retries exhausted
        logger.error(f"All {retry_config.max_attempts} retry attempts failed for step {step_id}")
        return False
    
    async def _execute_rollback_recovery(
        self,
        step_id: str,
        recovery_plan: RecoveryPlan
    ) -> bool:
        """Execute rollback recovery strategy."""
        if not self.execution_context:
            logger.error("Cannot rollback: no execution context available")
            return False
        
        target_checkpoint = recovery_plan.target_checkpoint
        if not target_checkpoint:
            # Find most recent checkpoint
            checkpoints = self.execution_context.get_checkpoints()
            if not checkpoints:
                logger.error("Cannot rollback: no checkpoints available")
                return False
            target_checkpoint = checkpoints[-1].checkpoint_id
        
        try:
            logger.info(f"Rolling back to checkpoint {target_checkpoint}")
            success = self.execution_context.restore_checkpoint(target_checkpoint)
            
            if success and self.progress_tracker:
                self.progress_tracker.create_checkpoint_event(
                    execution_id=self.execution_context.execution_id,
                    checkpoint_id=target_checkpoint,
                    step_id=step_id
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    async def _execute_alternative_path_recovery(
        self,
        step_id: str,
        recovery_plan: RecoveryPlan
    ) -> bool:
        """Execute alternative path recovery strategy."""
        if not recovery_plan.alternative_steps:
            logger.error(f"No alternative steps defined for step {step_id}")
            return False
        
        logger.info(f"Executing alternative path for step {step_id}")
        # This would need to be implemented based on the specific execution engine
        # For now, we'll mark it as successful if alternative steps are defined
        return True
    
    async def _execute_manual_intervention_recovery(
        self,
        step_id: str,
        recovery_plan: RecoveryPlan
    ) -> bool:
        """Execute manual intervention recovery strategy."""
        logger.warning(f"Manual intervention required for step {step_id}")
        
        if recovery_plan.manual_instructions:
            logger.info(f"Manual instructions: {recovery_plan.manual_instructions}")
        
        # This would typically involve notifying operators and waiting for manual resolution
        # For now, we'll return False to indicate manual intervention is needed
        return False
    
    def get_error_history(
        self,
        step_id: Optional[str] = None,
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None,
        limit: Optional[int] = None
    ) -> List[ErrorInfo]:
        """Get error history with optional filtering."""
        with self._lock:
            errors = self._error_history
            
            if step_id:
                errors = [e for e in errors if e.step_id == step_id]
            
            if category:
                errors = [e for e in errors if e.category == category]
            
            if severity:
                errors = [e for e in errors if e.severity == severity]
            
            if limit:
                errors = errors[-limit:]
            
            return errors.copy()
    
    def get_retry_count(self, step_id: str) -> int:
        """Get current retry count for step."""
        with self._lock:
            return self._retry_counts.get(step_id, 0)
    
    def reset_retry_count(self, step_id: str) -> None:
        """Reset retry count for step."""
        with self._lock:
            self._retry_counts.pop(step_id, None)
            logger.debug(f"Reset retry count for step {step_id}")
    
    def is_step_recovering(self, step_id: str) -> bool:
        """Check if step is currently being recovered."""
        with self._lock:
            return step_id in self._active_recoveries
    
    def get_recovery_status(self) -> Dict[str, Any]:
        """Get overall recovery status."""
        with self._lock:
            total_errors = len(self._error_history)
            error_counts_by_category = {}
            error_counts_by_severity = {}
            
            for error in self._error_history:
                error_counts_by_category[error.category.value] = error_counts_by_category.get(error.category.value, 0) + 1
                error_counts_by_severity[error.severity.value] = error_counts_by_severity.get(error.severity.value, 0) + 1
            
            return {
                "total_errors": total_errors,
                "active_recoveries": len(self._active_recoveries),
                "error_counts_by_category": error_counts_by_category,
                "error_counts_by_severity": error_counts_by_severity,
                "steps_with_retries": len(self._retry_counts),
                "registered_handlers": {
                    "category_specific": sum(len(handlers) for handlers in self._error_handlers.values()),
                    "global": len(self._global_error_handlers)
                }
            }
    
    @contextmanager
    def handle_step_errors(self, step_id: str, step_name: str):
        """Context manager for handling step execution errors."""
        try:
            yield
        except Exception as e:
            recovery_plan = self.handle_error(e, step_id, step_name)
            # Re-raise the exception - the caller should handle recovery execution
            raise
    
    def cleanup(self, execution_id: Optional[str] = None) -> None:
        """Clean up recovery data."""
        with self._lock:
            if execution_id:
                # Clean up data for specific execution
                self._error_history = [e for e in self._error_history if e.execution_id != execution_id]
                logger.debug(f"Cleaned up recovery data for execution {execution_id}")
            else:
                # Clean up all data
                self._error_history.clear()
                self._retry_counts.clear()
                self._active_recoveries.clear()
                logger.debug("Cleaned up all recovery data")
    
    def shutdown(self) -> None:
        """Shutdown recovery manager."""
        with self._lock:
            self._shutdown = True
            self.cleanup()
            self._error_handlers.clear()
            self._global_error_handlers.clear()
            logger.info("Recovery manager shut down")


def create_recovery_manager(
    execution_context: Optional[ExecutionContext] = None,
    progress_tracker: Optional[ProgressTracker] = None
) -> RecoveryManager:
    """
    Create and configure a recovery manager instance.
    
    Args:
        execution_context: Execution context for state integration
        progress_tracker: Progress tracker for event integration
    
    Returns:
        Configured RecoveryManager instance
    """
    return RecoveryManager(
        execution_context=execution_context,
        progress_tracker=progress_tracker
    )


# Common error handler examples
def network_error_handler(error_info: ErrorInfo) -> Optional[RecoveryPlan]:
    """Example error handler for network errors."""
    if error_info.category == ErrorCategory.NETWORK:
        return RecoveryPlan(
            strategy=RecoveryStrategy.RETRY_WITH_BACKOFF,
            retry_config=RetryConfig(
                max_attempts=5,
                initial_delay=2.0,
                max_delay=30.0,
                backoff_factor=1.5
            )
        )
    return None


def timeout_error_handler(error_info: ErrorInfo) -> Optional[RecoveryPlan]:
    """Example error handler for timeout errors."""
    if error_info.category == ErrorCategory.TIMEOUT:
        return RecoveryPlan(
            strategy=RecoveryStrategy.RETRY,
            retry_config=RetryConfig(
                max_attempts=3,
                initial_delay=5.0
            )
        )
    return None


def critical_error_handler(error_info: ErrorInfo) -> Optional[RecoveryPlan]:
    """Example error handler for critical errors."""
    if error_info.severity == ErrorSeverity.CRITICAL:
        return RecoveryPlan(
            strategy=RecoveryStrategy.FAIL_FAST,
            manual_instructions="Critical system error - check logs and system health"
        )
    return None