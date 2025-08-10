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
from ..state.langgraph_state_manager import LangGraphGlobalContextManager


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
        self,
        state_manager: StateManager,
        langgraph_manager: Optional[LangGraphGlobalContextManager] = None,
        default_strategy: Optional[ResumeStrategy] = None,
    ):
        """Initialize resume manager.

        Args:
            state_manager: State manager for checkpointing
            langgraph_manager: Optional LangGraph state manager for enhanced features
            default_strategy: Default resume strategy
        """
        self.state_manager = state_manager
        self.langgraph_manager = langgraph_manager
        self.default_strategy = default_strategy or ResumeStrategy()
        self.logger = logging.getLogger(__name__)
        self._checkpoint_tasks: Dict[str, asyncio.Task] = {}
        self._use_langgraph = langgraph_manager is not None

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

        # Prepare checkpoint data
        checkpoint_data = {
            "pipeline": pipeline.to_dict(),
            "resume_state": resume_state.to_dict(),
        }

        checkpoint_context = {
            "type": "resume_checkpoint",
            "completed_count": len(completed_tasks),
            "total_count": len(pipeline.tasks),
            "execution_id": execution_id,
            "pipeline_id": pipeline.id,
            "timestamp": time.time()
        }

        # Save checkpoint using appropriate manager
        if self._use_langgraph and self.langgraph_manager:
            # Use LangGraph state manager for enhanced checkpointing
            try:
                checkpoint_id = await self.langgraph_manager.save_checkpoint(
                    execution_id, checkpoint_data, checkpoint_context
                )
                
                # Also update global state if available
                global_state_updates = {
                    "intermediate_results": {
                        "resume_checkpoint_id": checkpoint_id,
                        "completed_tasks": list(completed_tasks),
                        "failed_tasks": failed_tasks or {},
                    },
                    "execution_metadata": {
                        "current_step": "resume_checkpoint",
                        "completed_steps": list(completed_tasks),
                        "failed_steps": list(failed_tasks.keys()) if failed_tasks else [],
                    }
                }
                
                await self.langgraph_manager.update_global_state(
                    execution_id, global_state_updates
                )
                
            except Exception as e:
                self.logger.warning(f"LangGraph checkpoint failed, falling back to legacy: {e}")
                checkpoint_id = await self.state_manager.save_checkpoint(
                    execution_id, checkpoint_data, checkpoint_context
                )
        else:
            # Use legacy state manager
            checkpoint_id = await self.state_manager.save_checkpoint(
                execution_id, checkpoint_data, checkpoint_context
            )

        resume_state.checkpoint_id = checkpoint_id
        self.logger.info(
            f"Created resume checkpoint {checkpoint_id} for execution {execution_id} "
            f"({'LangGraph' if self._use_langgraph else 'legacy'} mode)"
        )

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

    async def get_resume_state(
        self, execution_id: str
    ) -> Optional[Tuple[Pipeline, ResumeState]]:
        """Get resume state for an execution.

        Args:
            execution_id: Execution ID

        Returns:
            Tuple of (Pipeline, ResumeState) or None if not found
        """
        if self._use_langgraph and self.langgraph_manager:
            # Try LangGraph first for enhanced resume state
            try:
                checkpoints = await self.langgraph_manager.list_checkpoints(execution_id)
                
                # Look for resume checkpoint
                resume_checkpoint = None
                for checkpoint in checkpoints:
                    if isinstance(checkpoint, dict) and checkpoint.get("metadata", {}).get("type") == "resume_checkpoint":
                        resume_checkpoint = checkpoint
                        break
                
                if resume_checkpoint:
                    restored = await self.langgraph_manager.restore_checkpoint(
                        execution_id, resume_checkpoint["checkpoint_id"]
                    )
                    
                    if restored and "state" in restored:
                        state = restored["state"]
                        pipeline = Pipeline.from_dict(state["pipeline"])
                        resume_state = ResumeState.from_dict(state["resume_state"])
                        
                        # Enhance resume state with LangGraph global state
                        global_state = await self.langgraph_manager.get_global_state(execution_id)
                        if global_state:
                            # Add enhanced information from global state
                            if "intermediate_results" in global_state:
                                resume_state.context["langgraph_state"] = global_state["intermediate_results"]
                            if "execution_metadata" in global_state:
                                resume_state.context["execution_metadata"] = global_state["execution_metadata"]
                        
                        return pipeline, resume_state
                        
            except Exception as e:
                self.logger.warning(f"LangGraph resume state retrieval failed, falling back to legacy: {e}")

        # Fallback to legacy state manager
        try:
            checkpoints = await self.state_manager.list_checkpoints(execution_id)
        except Exception:
            # If list_checkpoints doesn't exist in legacy manager, try alternative approach
            return None

        resume_checkpoint = None
        for checkpoint in checkpoints:
            metadata = checkpoint.get("metadata", {})
            if metadata.get("type") == "resume_checkpoint":
                resume_checkpoint = checkpoint
                break

        if not resume_checkpoint:
            return None

        # Restore checkpoint
        try:
            restored = await self.state_manager.restore_checkpoint(
                execution_id, resume_checkpoint["checkpoint_id"]
            )
        except Exception:
            # Handle legacy checkpoint format differences
            try:
                restored = await self.state_manager.restore_checkpoint(
                    resume_checkpoint["checkpoint_id"]
                )
            except Exception as e:
                self.logger.error(f"Failed to restore checkpoint: {e}")
                return None

        if not restored:
            return None

        state = restored.get("state", restored)  # Handle different response formats

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

                if (
                    strategy.retry_failed_tasks
                    and retry_count < strategy.max_retry_attempts
                ):
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
        prepared_pipeline = await self.prepare_pipeline_for_resume(
            pipeline, resume_state, strategy
        )

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

        self.logger.info(
            f"Started periodic checkpointing for {execution_id} every {interval}s"
        )

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

    async def get_resume_history(
        self, pipeline_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
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
                restored = await self.state_manager.restore_checkpoint(
                    checkpoint["checkpoint_id"]
                )

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
        stats = {
            "active_checkpointing_tasks": len(self._checkpoint_tasks),
            "default_strategy": {
                "retry_failed_tasks": self.default_strategy.retry_failed_tasks,
                "max_retry_attempts": self.default_strategy.max_retry_attempts,
                "checkpoint_interval": self.default_strategy.checkpoint_interval_seconds,
            },
            "manager_type": "langgraph" if self._use_langgraph else "legacy",
        }
        
        # Add LangGraph-specific statistics
        if self._use_langgraph and self.langgraph_manager:
            try:
                langgraph_stats = self.langgraph_manager.get_statistics()
                stats["langgraph_stats"] = langgraph_stats
            except Exception:
                pass
                
        return stats

    async def create_named_resume_checkpoint(
        self,
        execution_id: str,
        checkpoint_name: str,
        description: str,
        pipeline: Pipeline,
        completed_tasks: Set[str],
        task_results: Dict[str, Any],
        context: Dict[str, Any],
        failed_tasks: Optional[Dict[str, int]] = None,
    ) -> Optional[str]:
        """Create a named checkpoint for resume (LangGraph only).

        Args:
            execution_id: Execution ID
            checkpoint_name: Human-readable name for the checkpoint
            description: Description of the checkpoint
            pipeline: Pipeline being executed
            completed_tasks: Set of completed task IDs
            task_results: Results from completed tasks
            context: Execution context
            failed_tasks: Failed tasks with retry counts

        Returns:
            Checkpoint ID if successful, None if not using LangGraph
        """
        if not self._use_langgraph or not self.langgraph_manager:
            self.logger.warning("Named checkpoints only available with LangGraph state manager")
            return None

        # Create standard resume checkpoint first
        checkpoint_id = await self.create_resume_checkpoint(
            execution_id, pipeline, completed_tasks, task_results, context, failed_tasks
        )

        # Create named checkpoint reference
        try:
            named_checkpoint_id = await self.langgraph_manager.create_checkpoint(
                execution_id,
                checkpoint_name,
                {
                    "resume_checkpoint_id": checkpoint_id,
                    "description": description,
                    "checkpoint_type": "named_resume_checkpoint",
                    "completed_tasks": list(completed_tasks),
                    "failed_tasks": failed_tasks or {},
                    "timestamp": time.time()
                }
            )
            
            self.logger.info(
                f"Created named resume checkpoint '{checkpoint_name}' ({named_checkpoint_id}) "
                f"for execution {execution_id}"
            )
            
            return named_checkpoint_id
            
        except Exception as e:
            self.logger.error(f"Failed to create named resume checkpoint: {e}")
            return checkpoint_id  # Return standard checkpoint ID as fallback

    async def get_enhanced_resume_metrics(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get enhanced resume metrics (LangGraph only).

        Args:
            execution_id: Execution ID

        Returns:
            Enhanced metrics if using LangGraph, None otherwise
        """
        if not self._use_langgraph or not self.langgraph_manager:
            return None

        try:
            global_state = await self.langgraph_manager.get_global_state(execution_id)
            if not global_state:
                return None

            # Extract resume-related metrics
            metrics = {
                "execution_id": execution_id,
                "current_status": global_state.get("execution_metadata", {}).get("status"),
                "completed_tasks": global_state.get("execution_metadata", {}).get("completed_steps", []),
                "failed_tasks": global_state.get("execution_metadata", {}).get("failed_steps", []),
                "pending_tasks": global_state.get("execution_metadata", {}).get("pending_steps", []),
                "retry_count": global_state.get("execution_metadata", {}).get("retry_count", 0),
                "performance_metrics": global_state.get("performance_metrics", {}),
                "error_context": global_state.get("error_context", {}),
                "checkpoint_history": global_state.get("checkpoint_history", []),
                "total_execution_time": None
            }

            # Calculate execution time if available
            start_time = global_state.get("execution_metadata", {}).get("start_time")
            if start_time:
                metrics["total_execution_time"] = time.time() - start_time

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to get enhanced resume metrics: {e}")
            return None

    async def optimize_checkpoint_storage(self, execution_id: str, keep_last_n: int = 5) -> bool:
        """Optimize checkpoint storage by keeping only recent checkpoints (LangGraph only).

        Args:
            execution_id: Execution ID
            keep_last_n: Number of recent checkpoints to keep

        Returns:
            True if optimization was performed, False otherwise
        """
        if not self._use_langgraph or not self.langgraph_manager:
            return False

        try:
            return await self.langgraph_manager.cleanup_checkpoints(execution_id, keep_last_n)
        except Exception as e:
            self.logger.error(f"Failed to optimize checkpoint storage: {e}")
            return False
