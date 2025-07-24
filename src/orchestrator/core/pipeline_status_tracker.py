"""Pipeline status tracking and monitoring."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import asyncio
import logging

from .pipeline import Pipeline
from .task import TaskStatus


class PipelineStatus(Enum):
    """Pipeline execution status."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionMetrics:
    """Metrics for a pipeline execution."""

    start_time: float = 0.0
    end_time: Optional[float] = None
    task_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Get execution duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def average_task_duration(self) -> float:
        """Get average task execution duration."""
        durations = [
            m.get("duration", 0) for m in self.task_metrics.values() if "duration" in m
        ]
        return sum(durations) / len(durations) if durations else 0.0


@dataclass
class PipelineExecution:
    """Represents a single pipeline execution."""

    execution_id: str
    pipeline: Pipeline
    status: PipelineStatus = PipelineStatus.PENDING
    metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)
    context: Dict[str, Any] = field(default_factory=dict)
    completed_tasks: Set[str] = field(default_factory=set)
    failed_tasks: Set[str] = field(default_factory=set)
    running_tasks: Set[str] = field(default_factory=set)

    def update_task_status(
        self, task_id: str, status: TaskStatus, metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update the status of a task."""
        # Remove from previous status sets
        self.running_tasks.discard(task_id)
        self.completed_tasks.discard(task_id)
        self.failed_tasks.discard(task_id)

        # Add to appropriate set
        if status == TaskStatus.RUNNING:
            self.running_tasks.add(task_id)
        elif status == TaskStatus.COMPLETED:
            self.completed_tasks.add(task_id)
        elif status == TaskStatus.FAILED:
            self.failed_tasks.add(task_id)

        # Update metrics if provided
        if metrics:
            self.metrics.task_metrics[task_id] = metrics

    @property
    def progress(self) -> float:
        """Get execution progress as percentage."""
        total_tasks = len(self.pipeline.tasks)
        if total_tasks == 0:
            return 100.0
        completed = len(self.completed_tasks) + len(self.failed_tasks)
        return (completed / total_tasks) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "execution_id": self.execution_id,
            "pipeline_id": self.pipeline.id,
            "pipeline_name": self.pipeline.name,
            "status": self.status.value,
            "progress": self.progress,
            "duration": self.metrics.duration,
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "running_tasks": len(self.running_tasks),
            "total_tasks": len(self.pipeline.tasks),
            "errors": len(self.metrics.errors),
            "warnings": len(self.metrics.warnings),
        }


class PipelineStatusTracker:
    """Centralized pipeline status tracking and monitoring."""

    def __init__(self, max_history: int = 1000):
        """Initialize the status tracker.

        Args:
            max_history: Maximum number of completed executions to keep in history
        """
        self.executions: Dict[str, PipelineExecution] = {}
        self.history: List[PipelineExecution] = []
        self.max_history = max_history
        self.logger = logging.getLogger(__name__)
        self._lock = asyncio.Lock()

        # Statistics
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0

        # Event handlers
        self.status_change_handlers: List[Any] = []
        self.task_status_handlers: List[Any] = []

    async def start_execution(
        self,
        execution_id: str,
        pipeline: Pipeline,
        context: Optional[Dict[str, Any]] = None,
    ) -> PipelineExecution:
        """Start tracking a new pipeline execution."""
        async with self._lock:
            if execution_id in self.executions:
                raise ValueError(f"Execution {execution_id} already exists")

            execution = PipelineExecution(
                execution_id=execution_id,
                pipeline=pipeline,
                status=PipelineStatus.RUNNING,
                context=context or {},
            )
            execution.metrics.start_time = time.time()

            self.executions[execution_id] = execution
            self.total_executions += 1

            await self._notify_status_change(execution_id, PipelineStatus.RUNNING)

            self.logger.info(
                f"Started tracking execution {execution_id} for pipeline {pipeline.id}"
            )
            return execution

    async def update_status(self, execution_id: str, status: PipelineStatus) -> None:
        """Update the status of a pipeline execution."""
        async with self._lock:
            if execution_id not in self.executions:
                raise ValueError(f"Execution {execution_id} not found")

            execution = self.executions[execution_id]
            old_status = execution.status
            execution.status = status

            # Update metrics
            if status in (
                PipelineStatus.COMPLETED,
                PipelineStatus.FAILED,
                PipelineStatus.CANCELLED,
            ):
                execution.metrics.end_time = time.time()

                # Update statistics
                if status == PipelineStatus.COMPLETED:
                    self.successful_executions += 1
                elif status == PipelineStatus.FAILED:
                    self.failed_executions += 1

                # Move to history
                self._move_to_history(execution_id)

            await self._notify_status_change(execution_id, status, old_status)

            self.logger.info(
                f"Updated execution {execution_id} status: {old_status.value} -> {status.value}"
            )

    async def update_task_status(
        self,
        execution_id: str,
        task_id: str,
        status: TaskStatus,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update the status of a task within an execution."""
        async with self._lock:
            if execution_id not in self.executions:
                raise ValueError(f"Execution {execution_id} not found")

            execution = self.executions[execution_id]
            execution.update_task_status(task_id, status, metrics)

            # Check if pipeline should transition to completed/failed
            if execution.status == PipelineStatus.RUNNING:
                if execution.pipeline.is_complete():
                    await self.update_status(execution_id, PipelineStatus.COMPLETED)
                elif execution.pipeline.is_failed():
                    await self.update_status(execution_id, PipelineStatus.FAILED)

            await self._notify_task_status(execution_id, task_id, status)

    def get_execution(self, execution_id: str) -> Optional[PipelineExecution]:
        """Get execution details by ID."""
        return self.executions.get(execution_id)

    def get_running_executions(self) -> List[PipelineExecution]:
        """Get all currently running executions."""
        return [
            exec
            for exec in self.executions.values()
            if exec.status == PipelineStatus.RUNNING
        ]

    def get_execution_summary(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of an execution."""
        execution = self.get_execution(execution_id)
        if execution:
            return execution.to_dict()

        # Check history
        for hist_exec in self.history:
            if hist_exec.execution_id == execution_id:
                return hist_exec.to_dict()

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics."""
        running = len(self.get_running_executions())

        return {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "running_executions": running,
            "success_rate": (
                self.successful_executions / self.total_executions * 100
                if self.total_executions > 0
                else 0
            ),
            "history_size": len(self.history),
        }

    def add_error(self, execution_id: str, error: Dict[str, Any]) -> None:
        """Add an error to an execution."""
        if execution_id in self.executions:
            self.executions[execution_id].metrics.errors.append(
                {**error, "timestamp": time.time()}
            )

    def add_warning(self, execution_id: str, warning: Dict[str, Any]) -> None:
        """Add a warning to an execution."""
        if execution_id in self.executions:
            self.executions[execution_id].metrics.warnings.append(
                {**warning, "timestamp": time.time()}
            )

    def _move_to_history(self, execution_id: str) -> None:
        """Move completed execution to history."""
        if execution_id in self.executions:
            execution = self.executions.pop(execution_id)
            self.history.append(execution)

            # Trim history if needed
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history :]

    async def _notify_status_change(
        self,
        execution_id: str,
        new_status: PipelineStatus,
        old_status: Optional[PipelineStatus] = None,
    ) -> None:
        """Notify status change handlers."""
        for handler in self.status_change_handlers:
            try:
                await handler(execution_id, new_status, old_status)
            except Exception as e:
                self.logger.error(f"Error in status change handler: {e}")

    async def _notify_task_status(
        self, execution_id: str, task_id: str, status: TaskStatus
    ) -> None:
        """Notify task status handlers."""
        for handler in self.task_status_handlers:
            try:
                await handler(execution_id, task_id, status)
            except Exception as e:
                self.logger.error(f"Error in task status handler: {e}")

    def register_status_handler(self, handler) -> None:
        """Register a pipeline status change handler."""
        self.status_change_handlers.append(handler)

    def register_task_handler(self, handler) -> None:
        """Register a task status change handler."""
        self.task_status_handlers.append(handler)

    def clear_history(self) -> None:
        """Clear execution history."""
        self.history.clear()

    def reset_statistics(self) -> None:
        """Reset execution statistics."""
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
