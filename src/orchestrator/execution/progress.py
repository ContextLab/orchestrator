"""
Real-time Progress Tracking System for Pipeline Execution.

This module provides comprehensive progress tracking capabilities including
real-time status updates, step-level progress monitoring, and integration
with the StateGraphEngine execution system.
"""

from __future__ import annotations

import logging
import time
import threading
from typing import Any, Dict, List, Optional, Set, Callable, Protocol, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import json
from contextlib import contextmanager

from .state import ExecutionContext, ExecutionStatus, ExecutionMetrics
from .variables import VariableManager

logger = logging.getLogger(__name__)


class ProgressEventType(Enum):
    """Types of progress events."""
    EXECUTION_STARTED = "execution_started"
    EXECUTION_COMPLETED = "execution_completed"
    EXECUTION_FAILED = "execution_failed"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    STEP_SKIPPED = "step_skipped"
    CHECKPOINT_CREATED = "checkpoint_created"
    PROGRESS_UPDATE = "progress_update"
    CUSTOM_EVENT = "custom_event"


class StepStatus(Enum):
    """Status of individual pipeline steps."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class StepProgress:
    """Progress information for individual pipeline step."""
    step_id: str
    step_name: str
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[timedelta] = None
    progress_percentage: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_completed(self) -> bool:
        """Check if step is completed."""
        return self.status in (StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED)
    
    @property
    def is_successful(self) -> bool:
        """Check if step completed successfully."""
        return self.status == StepStatus.COMPLETED
    
    def start(self) -> None:
        """Mark step as started."""
        self.status = StepStatus.RUNNING
        self.start_time = datetime.now()
    
    def complete(self, progress_percentage: float = 100.0) -> None:
        """Mark step as completed successfully."""
        self.status = StepStatus.COMPLETED
        self.progress_percentage = progress_percentage
        self.end_time = datetime.now()
        if self.start_time:
            self.duration = self.end_time - self.start_time
    
    def fail(self, error_message: str) -> None:
        """Mark step as failed."""
        self.status = StepStatus.FAILED
        self.error_message = error_message
        self.end_time = datetime.now()
        if self.start_time:
            self.duration = self.end_time - self.start_time
    
    def skip(self, reason: str = "Condition not met") -> None:
        """Mark step as skipped."""
        self.status = StepStatus.SKIPPED
        self.metadata["skip_reason"] = reason
        self.end_time = datetime.now()
        if self.start_time:
            self.duration = self.end_time - self.start_time


@dataclass
class ProgressEvent:
    """Event representing a progress update."""
    event_type: ProgressEventType
    timestamp: datetime = field(default_factory=datetime.now)
    execution_id: Optional[str] = None
    step_id: Optional[str] = None
    step_name: Optional[str] = None
    message: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['event_type'] = self.event_type.value
        return result


@dataclass
class ExecutionProgress:
    """Overall execution progress information."""
    execution_id: str
    total_steps: int
    completed_steps: int = 0
    failed_steps: int = 0
    skipped_steps: int = 0
    running_steps: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    
    @property
    def progress_percentage(self) -> float:
        """Calculate overall progress percentage."""
        if self.total_steps == 0:
            return 0.0
        return (self.completed_steps + self.failed_steps + self.skipped_steps) / self.total_steps * 100
    
    @property
    def is_completed(self) -> bool:
        """Check if execution is completed."""
        return (self.completed_steps + self.failed_steps + self.skipped_steps) >= self.total_steps
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get execution duration if available."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return datetime.now() - self.start_time
        return None


class ProgressTrackerProtocol(Protocol):
    """Protocol for progress tracking integration with StateGraphEngine."""
    
    def on_execution_started(self, execution_id: str, total_steps: int) -> None:
        """Called when execution starts."""
        ...
    
    def on_execution_completed(self, execution_id: str, success: bool) -> None:
        """Called when execution completes."""
        ...
    
    def on_step_started(self, execution_id: str, step_id: str, step_name: str) -> None:
        """Called when step starts."""
        ...
    
    def on_step_completed(self, execution_id: str, step_id: str, step_name: str, success: bool) -> None:
        """Called when step completes."""
        ...
    
    def update_step_progress(self, execution_id: str, step_id: str, progress: float) -> None:
        """Update progress for specific step."""
        ...


class ProgressTracker:
    """
    Comprehensive progress tracking system.
    
    Provides real-time progress monitoring, event handling, and integration
    with the execution context and variable management systems.
    """
    
    def __init__(
        self,
        execution_context: Optional[ExecutionContext] = None,
        variable_manager: Optional[VariableManager] = None
    ):
        """
        Initialize progress tracker.
        
        Args:
            execution_context: Execution context for state management
            variable_manager: Variable manager for event integration
        """
        self.execution_context = execution_context
        self.variable_manager = variable_manager
        
        # Progress tracking state
        self._executions: Dict[str, ExecutionProgress] = {}
        self._steps: Dict[str, Dict[str, StepProgress]] = {}  # execution_id -> step_id -> progress
        
        # Event handling
        self._event_handlers: List[Callable[[ProgressEvent], None]] = []
        self._real_time_handlers: List[Callable[[str, Dict[str, Any]], None]] = []
        
        # Threading and safety
        self._lock = threading.RLock()
        self._shutdown = False
        
        # Performance monitoring
        self._event_count = 0
        self._handler_performance: Dict[str, List[float]] = {}
        
        # Integration with variable manager
        if self.variable_manager:
            self._setup_variable_integration()
        
        logger.info("Initialized ProgressTracker")
    
    def _setup_variable_integration(self) -> None:
        """Set up integration with variable manager for progress tracking."""
        def handle_variable_change(name: str, old_value: Any, new_value: Any) -> None:
            """Handle variable changes for progress tracking."""
            if name.startswith("progress."):
                # Extract step information from variable name
                parts = name.split(".")
                if len(parts) >= 3:
                    step_id = parts[1]
                    metric = parts[2]
                    
                    # Update step progress based on variable change
                    if self.execution_context:
                        self._handle_progress_variable_change(
                            self.execution_context.execution_id,
                            step_id,
                            metric,
                            new_value
                        )
        
        self.variable_manager.on_variable_changed(handle_variable_change)
    
    def _handle_progress_variable_change(
        self,
        execution_id: str,
        step_id: str,
        metric: str,
        value: Any
    ) -> None:
        """Handle progress-related variable changes."""
        with self._lock:
            if execution_id in self._steps and step_id in self._steps[execution_id]:
                step_progress = self._steps[execution_id][step_id]
                
                if metric == "percentage" and isinstance(value, (int, float)):
                    step_progress.progress_percentage = min(100.0, max(0.0, float(value)))
                    self._emit_event(ProgressEvent(
                        event_type=ProgressEventType.PROGRESS_UPDATE,
                        execution_id=execution_id,
                        step_id=step_id,
                        step_name=step_progress.step_name,
                        data={"progress_percentage": step_progress.progress_percentage}
                    ))
                elif metric == "status" and isinstance(value, str):
                    try:
                        new_status = StepStatus(value)
                        old_status = step_progress.status
                        step_progress.status = new_status
                        
                        if old_status != new_status:
                            self._handle_step_status_change(execution_id, step_id, new_status)
                    except ValueError:
                        logger.warning(f"Invalid step status value: {value}")
    
    def _handle_step_status_change(
        self,
        execution_id: str,
        step_id: str,
        new_status: StepStatus
    ) -> None:
        """Handle step status changes."""
        if new_status == StepStatus.RUNNING:
            self._emit_event(ProgressEvent(
                event_type=ProgressEventType.STEP_STARTED,
                execution_id=execution_id,
                step_id=step_id
            ))
        elif new_status == StepStatus.COMPLETED:
            self._emit_event(ProgressEvent(
                event_type=ProgressEventType.STEP_COMPLETED,
                execution_id=execution_id,
                step_id=step_id
            ))
        elif new_status == StepStatus.FAILED:
            self._emit_event(ProgressEvent(
                event_type=ProgressEventType.STEP_FAILED,
                execution_id=execution_id,
                step_id=step_id
            ))
    
    def start_execution(self, execution_id: str, total_steps: int) -> None:
        """Start tracking execution progress."""
        with self._lock:
            execution_progress = ExecutionProgress(
                execution_id=execution_id,
                total_steps=total_steps,
                start_time=datetime.now()
            )
            
            self._executions[execution_id] = execution_progress
            self._steps[execution_id] = {}
            
            self._emit_event(ProgressEvent(
                event_type=ProgressEventType.EXECUTION_STARTED,
                execution_id=execution_id,
                data={"total_steps": total_steps}
            ))
            
            logger.info(f"Started tracking execution {execution_id} with {total_steps} steps")
    
    def complete_execution(self, execution_id: str, success: bool = True) -> None:
        """Complete execution tracking."""
        with self._lock:
            if execution_id not in self._executions:
                logger.warning(f"Execution {execution_id} not found for completion")
                return
            
            execution_progress = self._executions[execution_id]
            execution_progress.end_time = datetime.now()
            
            event_type = (ProgressEventType.EXECUTION_COMPLETED 
                         if success else ProgressEventType.EXECUTION_FAILED)
            
            self._emit_event(ProgressEvent(
                event_type=event_type,
                execution_id=execution_id,
                data={
                    "success": success,
                    "duration": execution_progress.duration.total_seconds() if execution_progress.duration else None,
                    "progress_percentage": execution_progress.progress_percentage
                }
            ))
            
            logger.info(f"Completed tracking execution {execution_id} (success: {success})")
    
    def start_step(self, execution_id: str, step_id: str, step_name: str) -> None:
        """Start tracking step progress."""
        with self._lock:
            if execution_id not in self._executions:
                logger.warning(f"Execution {execution_id} not found for step start")
                return
            
            if execution_id not in self._steps:
                self._steps[execution_id] = {}
            
            step_progress = StepProgress(
                step_id=step_id,
                step_name=step_name
            )
            step_progress.start()
            
            self._steps[execution_id][step_id] = step_progress
            
            # Update execution counters
            execution_progress = self._executions[execution_id]
            execution_progress.running_steps += 1
            
            self._emit_event(ProgressEvent(
                event_type=ProgressEventType.STEP_STARTED,
                execution_id=execution_id,
                step_id=step_id,
                step_name=step_name
            ))
            
            logger.debug(f"Started tracking step {step_id} ({step_name}) in execution {execution_id}")
    
    def complete_step(
        self,
        execution_id: str,
        step_id: str,
        success: bool = True,
        error_message: Optional[str] = None,
        progress_percentage: float = 100.0
    ) -> None:
        """Complete step tracking."""
        with self._lock:
            if (execution_id not in self._steps or 
                step_id not in self._steps[execution_id]):
                logger.warning(f"Step {step_id} in execution {execution_id} not found for completion")
                return
            
            step_progress = self._steps[execution_id][step_id]
            execution_progress = self._executions[execution_id]
            
            # Update step status
            if success:
                step_progress.complete(progress_percentage)
                execution_progress.completed_steps += 1
                event_type = ProgressEventType.STEP_COMPLETED
            else:
                step_progress.fail(error_message or "Unknown error")
                execution_progress.failed_steps += 1
                event_type = ProgressEventType.STEP_FAILED
            
            # Update execution counters
            execution_progress.running_steps = max(0, execution_progress.running_steps - 1)
            
            self._emit_event(ProgressEvent(
                event_type=event_type,
                execution_id=execution_id,
                step_id=step_id,
                step_name=step_progress.step_name,
                message=error_message,
                data={
                    "success": success,
                    "progress_percentage": step_progress.progress_percentage,
                    "duration": step_progress.duration.total_seconds() if step_progress.duration else None
                }
            ))
            
            logger.debug(f"Completed step {step_id} in execution {execution_id} (success: {success})")
    
    def skip_step(self, execution_id: str, step_id: str, reason: str = "Condition not met") -> None:
        """Mark step as skipped."""
        with self._lock:
            if (execution_id not in self._steps or 
                step_id not in self._steps[execution_id]):
                logger.warning(f"Step {step_id} in execution {execution_id} not found for skipping")
                return
            
            step_progress = self._steps[execution_id][step_id]
            execution_progress = self._executions[execution_id]
            
            step_progress.skip(reason)
            execution_progress.skipped_steps += 1
            execution_progress.running_steps = max(0, execution_progress.running_steps - 1)
            
            self._emit_event(ProgressEvent(
                event_type=ProgressEventType.STEP_SKIPPED,
                execution_id=execution_id,
                step_id=step_id,
                step_name=step_progress.step_name,
                message=reason
            ))
            
            logger.debug(f"Skipped step {step_id} in execution {execution_id}: {reason}")
    
    def update_step_progress(
        self,
        execution_id: str,
        step_id: str,
        progress_percentage: float,
        message: Optional[str] = None
    ) -> None:
        """Update progress for specific step."""
        with self._lock:
            if (execution_id not in self._steps or 
                step_id not in self._steps[execution_id]):
                logger.warning(f"Step {step_id} in execution {execution_id} not found for progress update")
                return
            
            step_progress = self._steps[execution_id][step_id]
            step_progress.progress_percentage = min(100.0, max(0.0, progress_percentage))
            
            if message:
                step_progress.metadata["last_message"] = message
            
            self._emit_event(ProgressEvent(
                event_type=ProgressEventType.PROGRESS_UPDATE,
                execution_id=execution_id,
                step_id=step_id,
                step_name=step_progress.step_name,
                message=message,
                data={"progress_percentage": progress_percentage}
            ))
    
    def get_execution_progress(self, execution_id: str) -> Optional[ExecutionProgress]:
        """Get overall execution progress."""
        with self._lock:
            return self._executions.get(execution_id)
    
    def get_step_progress(self, execution_id: str, step_id: str) -> Optional[StepProgress]:
        """Get specific step progress."""
        with self._lock:
            if execution_id in self._steps:
                return self._steps[execution_id].get(step_id)
            return None
    
    def get_all_step_progress(self, execution_id: str) -> Dict[str, StepProgress]:
        """Get all step progress for execution."""
        with self._lock:
            return self._steps.get(execution_id, {}).copy()
    
    def add_event_handler(self, handler: Callable[[ProgressEvent], None]) -> None:
        """Add event handler for progress events."""
        with self._lock:
            self._event_handlers.append(handler)
            logger.debug(f"Added progress event handler: {handler.__name__}")
    
    def remove_event_handler(self, handler: Callable[[ProgressEvent], None]) -> None:
        """Remove event handler."""
        with self._lock:
            if handler in self._event_handlers:
                self._event_handlers.remove(handler)
                logger.debug(f"Removed progress event handler: {handler.__name__}")
    
    def add_real_time_handler(self, handler: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add real-time progress handler."""
        with self._lock:
            self._real_time_handlers.append(handler)
            logger.debug(f"Added real-time progress handler: {handler.__name__}")
    
    def _emit_event(self, event: ProgressEvent) -> None:
        """Emit progress event to all handlers."""
        self._event_count += 1
        
        # Call event handlers
        for handler in self._event_handlers:
            try:
                start_time = time.time()
                handler(event)
                duration = time.time() - start_time
                
                handler_name = getattr(handler, '__name__', str(handler))
                if handler_name not in self._handler_performance:
                    self._handler_performance[handler_name] = []
                self._handler_performance[handler_name].append(duration)
                
            except Exception as e:
                logger.error(f"Error in progress event handler: {e}")
        
        # Call real-time handlers with simplified data
        real_time_data = {
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "execution_id": event.execution_id,
            "step_id": event.step_id,
            "step_name": event.step_name,
            "message": event.message,
            "data": event.data
        }
        
        for handler in self._real_time_handlers:
            try:
                handler(event.event_type.value, real_time_data)
            except Exception as e:
                logger.error(f"Error in real-time progress handler: {e}")
    
    @contextmanager
    def track_step(self, execution_id: str, step_id: str, step_name: str):
        """Context manager for tracking step execution."""
        self.start_step(execution_id, step_id, step_name)
        try:
            yield
            self.complete_step(execution_id, step_id, success=True)
        except Exception as e:
            self.complete_step(execution_id, step_id, success=False, error_message=str(e))
            raise
    
    def create_checkpoint_event(
        self,
        execution_id: str,
        checkpoint_id: str,
        step_id: Optional[str] = None
    ) -> None:
        """Create checkpoint event."""
        self._emit_event(ProgressEvent(
            event_type=ProgressEventType.CHECKPOINT_CREATED,
            execution_id=execution_id,
            step_id=step_id,
            data={"checkpoint_id": checkpoint_id}
        ))
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the progress tracker."""
        with self._lock:
            avg_handler_performance = {}
            for handler_name, durations in self._handler_performance.items():
                avg_handler_performance[handler_name] = {
                    "avg_duration": sum(durations) / len(durations),
                    "max_duration": max(durations),
                    "min_duration": min(durations),
                    "call_count": len(durations)
                }
            
            return {
                "total_events": self._event_count,
                "active_executions": len(self._executions),
                "handler_performance": avg_handler_performance,
                "total_handlers": len(self._event_handlers) + len(self._real_time_handlers)
            }
    
    def cleanup(self, execution_id: Optional[str] = None) -> None:
        """Clean up progress tracking data."""
        with self._lock:
            if execution_id:
                # Clean up specific execution
                self._executions.pop(execution_id, None)
                self._steps.pop(execution_id, None)
                logger.debug(f"Cleaned up progress data for execution {execution_id}")
            else:
                # Clean up all data
                self._executions.clear()
                self._steps.clear()
                logger.debug("Cleaned up all progress tracking data")
    
    def shutdown(self) -> None:
        """Shutdown progress tracker."""
        with self._lock:
            self._shutdown = True
            self.cleanup()
            self._event_handlers.clear()
            self._real_time_handlers.clear()
            logger.info("Progress tracker shut down")


def create_progress_tracker(
    execution_context: Optional[ExecutionContext] = None,
    variable_manager: Optional[VariableManager] = None
) -> ProgressTracker:
    """
    Create and configure a progress tracker instance.
    
    Args:
        execution_context: Execution context for state integration
        variable_manager: Variable manager for event integration
    
    Returns:
        Configured ProgressTracker instance
    """
    return ProgressTracker(
        execution_context=execution_context,
        variable_manager=variable_manager
    )