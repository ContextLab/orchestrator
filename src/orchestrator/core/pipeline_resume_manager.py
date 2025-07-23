"""Pipeline resume and restart management."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .pipeline import Pipeline
from .task import TaskStatus
from ..state.state_manager import StateManager


@dataclass
class ResumeStrategy:
    """Configuration for pipeline resume behavior."""

    # Resume options
    retry_failed_tasks: bool = True
    reset_running_tasks: bool = True
    preserve_completed_tasks: bool = True
    max_retry_attempts: int = 3

    # Checkpointing options
    checkpoint_on_task_completion: bool = True
    checkpoint_on_error: bool = True
    checkpoint_interval_seconds: float = 60.0

    # Recovery options
    recover_from_latest: bool = True
    validate_checkpoint_integrity: bool = True


@dataclass
class ResumeState:
    """State information for pipeline resume."""

    execution_id: str
    pipeline_id: str
    checkpoint_id: str
    completed_tasks: Set[str] = field(default_factory=set)
    failed_tasks: Dict[str, int] = field(default_factory=dict)  # task_id -> retry_count
    task_results: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "execution_id": self.execution_id,
            "pipeline_id": self.pipeline_id,
            "checkpoint_id": self.checkpoint_id,
            "completed_tasks": list(self.completed_tasks),
            "failed_tasks": self.failed_tasks,
            "task_results": self.task_results,
            "context": self.context,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ResumeState:
        """Create from dictionary."""
        data = data.copy()
        data["completed_tasks"] = set(data.get("completed_tasks", []))
        return cls(**data)


class PipelineResumeManager:
    """Manages pipeline resume and restart operations."""

    def __init__(
        self, state_manager: StateManager, default_strategy: Optional[ResumeStrategy] = None
    ):
        """Initialize resume manager.

        Args:
            state_manager: State manager for checkpointing
            default_strategy: Default resume strategy
        """
        self.state_manager = state_manager
        self.default_strategy = default_strategy or ResumeStrategy()
        self.logger = logging.getLogger(__name__)
        self._checkpoint_tasks: Dict[str, asyncio.Task] = {}

    async def create_resume_checkpoint(
        self,
        execution_id: str,
        pipeline: Pipeline,
        completed_tasks: Set[str],
        task_results: Dict[str, Any],
        context: Dict[str, Any],
        failed_tasks: Optional[Dict[str, int]] = None,
    ) -> str:
        """Create a checkpoint for resume.

        Args:
            execution_id: Execution ID
            pipeline: Pipeline being executed
            completed_tasks: Set of completed task IDs
            task_results: Results from completed tasks
            context: Execution context
            failed_tasks: Failed tasks with retry counts

        Returns:
            Checkpoint ID
        """
        resume_state = ResumeState(
            execution_id=execution_id,
            pipeline_id=pipeline.id,
            checkpoint_id="",  # Will be set after saving
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks or {},
            task_results=task_results,
            context=context,
        )

        # Save checkpoint
        checkpoint_id = await self.state_manager.save_checkpoint(
            execution_id,
            {
                "pipeline": pipeline.to_dict(),
                "resume_state": resume_state.to_dict(),
            },
            metadata={
                "type": "resume_checkpoint",
                "completed_count": len(completed_tasks),
                "total_count": len(pipeline.tasks),
            },
        )

        resume_state.checkpoint_id = checkpoint_id
        self.logger.info(f"Created resume checkpoint {checkpoint_id} for execution {execution_id}")

        return checkpoint_id

    async def can_resume(self, execution_id: str) -> bool:
        """Check if a pipeline execution can be resumed.

        Args:
            execution_id: Execution ID to check

        Returns:
            True if resumable, False otherwise
        """
        checkpoints = await self.state_manager.list_checkpoints(execution_id)

        # Look for resume checkpoints
        for checkpoint in checkpoints:
            metadata = checkpoint.get("metadata", {})
            if metadata.get("type") == "resume_checkpoint":
                return True

        return False

    async def get_resume_state(self, execution_id: str) -> Optional[Tuple[Pipeline, ResumeState]]:
        """Get resume state for an execution.

        Args:
            execution_id: Execution ID

        Returns:
            Tuple of (Pipeline, ResumeState) or None if not found
        """
        # Get latest resume checkpoint
        checkpoints = await self.state_manager.list_checkpoints(execution_id)

        resume_checkpoint = None
        for checkpoint in checkpoints:
            metadata = checkpoint.get("metadata", {})
            if metadata.get("type") == "resume_checkpoint":
                resume_checkpoint = checkpoint
                break

        if not resume_checkpoint:
            return None

        # Restore checkpoint
        restored = await self.state_manager.restore_checkpoint(
            execution_id, resume_checkpoint["checkpoint_id"]
        )

        if not restored:
            return None

        state = restored["state"]

        # Reconstruct pipeline and resume state
        pipeline = Pipeline.from_dict(state["pipeline"])
        resume_state = ResumeState.from_dict(state["resume_state"])

        return pipeline, resume_state

    async def prepare_pipeline_for_resume(
        self,
        pipeline: Pipeline,
        resume_state: ResumeState,
        strategy: Optional[ResumeStrategy] = None,
    ) -> Pipeline:
        """Prepare a pipeline for resume by updating task states.

        Args:
            pipeline: Pipeline to prepare
            resume_state: Resume state information
            strategy: Resume strategy (uses default if None)

        Returns:
            Prepared pipeline
        """
        strategy = strategy or self.default_strategy

        # Update task states based on resume state
        for task_id, task in pipeline.tasks.items():
            if task_id in resume_state.completed_tasks:
                if strategy.preserve_completed_tasks:
                    task.status = TaskStatus.COMPLETED
                    task.result = resume_state.task_results.get(task_id)
                else:
                    task.reset()

            elif task_id in resume_state.failed_tasks:
                retry_count = resume_state.failed_tasks[task_id]

                if strategy.retry_failed_tasks and retry_count < strategy.max_retry_attempts:
                    task.reset()
                    task.metadata["retry_count"] = retry_count
                else:
                    task.status = TaskStatus.FAILED

            elif task.status == TaskStatus.RUNNING:
                if strategy.reset_running_tasks:
                    task.reset()
                else:
                    task.status = TaskStatus.FAILED

        self.logger.info(
            f"Prepared pipeline {pipeline.id} for resume: "
            f"{len(resume_state.completed_tasks)} completed, "
            f"{len(resume_state.failed_tasks)} failed"
        )

        return pipeline

    async def resume_pipeline(
        self, execution_id: str, strategy: Optional[ResumeStrategy] = None
    ) -> Optional[Tuple[Pipeline, Dict[str, Any]]]:
        """Resume a pipeline execution.

        Args:
            execution_id: Execution ID to resume
            strategy: Resume strategy (uses default if None)

        Returns:
            Tuple of (prepared pipeline, context) or None if not resumable
        """
        # Get resume state
        result = await self.get_resume_state(execution_id)
        if not result:
            self.logger.warning(f"No resume state found for execution {execution_id}")
            return None

        pipeline, resume_state = result

        # Prepare pipeline
        prepared_pipeline = await self.prepare_pipeline_for_resume(pipeline, resume_state, strategy)

        # Prepare context with resume information
        context = resume_state.context.copy()
        context["is_resume"] = True
        context["resume_checkpoint_id"] = resume_state.checkpoint_id
        context["completed_tasks"] = resume_state.completed_tasks
        context["task_results"] = resume_state.task_results

        return prepared_pipeline, context

    async def start_periodic_checkpointing(
        self,
        execution_id: str,
        pipeline: Pipeline,
        get_state_func,
        interval: Optional[float] = None,
    ):
        """Start periodic checkpointing for a pipeline.

        Args:
            execution_id: Execution ID
            pipeline: Pipeline being executed
            get_state_func: Function to get current execution state
            interval: Checkpoint interval in seconds (uses strategy default if None)
        """
        interval = interval or self.default_strategy.checkpoint_interval_seconds

        async def checkpoint_loop():
            while True:
                await asyncio.sleep(interval)

                try:
                    # Get current state
                    state = await get_state_func()

                    # Create checkpoint
                    await self.create_resume_checkpoint(
                        execution_id=execution_id,
                        pipeline=pipeline,
                        completed_tasks=state["completed_tasks"],
                        task_results=state["task_results"],
                        context=state["context"],
                        failed_tasks=state.get("failed_tasks"),
                    )

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error during periodic checkpoint: {e}")

        # Start checkpoint task
        task = asyncio.create_task(checkpoint_loop())
        self._checkpoint_tasks[execution_id] = task

        self.logger.info(f"Started periodic checkpointing for {execution_id} every {interval}s")

    async def stop_periodic_checkpointing(self, execution_id: str):
        """Stop periodic checkpointing for an execution.

        Args:
            execution_id: Execution ID
        """
        if execution_id in self._checkpoint_tasks:
            task = self._checkpoint_tasks.pop(execution_id)
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            self.logger.info(f"Stopped periodic checkpointing for {execution_id}")

    async def get_resume_history(self, pipeline_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get resume history for a pipeline.

        Args:
            pipeline_id: Pipeline ID
            limit: Maximum number of entries

        Returns:
            List of resume checkpoint summaries
        """
        # List all checkpoints
        all_checkpoints = await self.state_manager.list_checkpoints(limit=limit * 2)

        # Filter for resume checkpoints of this pipeline
        resume_history = []

        for checkpoint in all_checkpoints:
            metadata = checkpoint.get("metadata", {})

            if metadata.get("type") == "resume_checkpoint":
                # Load checkpoint to check pipeline ID
                restored = await self.state_manager.restore_checkpoint(checkpoint["checkpoint_id"])

                if restored:
                    state = restored["state"]
                    resume_state = state.get("resume_state", {})

                    if resume_state.get("pipeline_id") == pipeline_id:
                        resume_history.append(
                            {
                                "checkpoint_id": checkpoint["checkpoint_id"],
                                "execution_id": resume_state.get("execution_id"),
                                "timestamp": checkpoint["timestamp"],
                                "completed_count": metadata.get("completed_count", 0),
                                "total_count": metadata.get("total_count", 0),
                            }
                        )

                if len(resume_history) >= limit:
                    break

        return resume_history

    def get_resume_statistics(self) -> Dict[str, Any]:
        """Get resume manager statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "active_checkpointing_tasks": len(self._checkpoint_tasks),
            "default_strategy": {
                "retry_failed_tasks": self.default_strategy.retry_failed_tasks,
                "max_retry_attempts": self.default_strategy.max_retry_attempts,
                "checkpoint_interval": self.default_strategy.checkpoint_interval_seconds,
            },
        }
