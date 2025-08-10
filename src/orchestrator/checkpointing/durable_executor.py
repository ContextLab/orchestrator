"""Durable Execution Manager - Issue #205

Manages durable pipeline execution with automatic recovery.
Provides failure detection, recovery mechanisms, and execution progress tracking.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

# Internal imports
from ..core.pipeline import Pipeline
from ..core.task import Task, TaskStatus
from ..core.exceptions import PipelineExecutionError
from ..state.global_context import (
    PipelineGlobalState, 
    create_initial_pipeline_state,
    PipelineStatus
)
from ..state.langgraph_state_manager import LangGraphGlobalContextManager
from .automatic_graph import AutomaticCheckpointingGraph

logger = logging.getLogger(__name__)


class ExecutionRecoveryStrategy(Enum):
    """Strategy for execution recovery."""
    RESUME_FROM_LAST_CHECKPOINT = "resume_from_last"
    RESTART_FROM_BEGINNING = "restart_from_beginning"
    RESUME_FROM_SPECIFIC_CHECKPOINT = "resume_from_specific"
    SKIP_FAILED_STEP = "skip_failed_step"


@dataclass
class ExecutionResult:
    """Result of durable pipeline execution."""
    execution_id: str
    pipeline_id: str
    status: PipelineStatus
    final_state: Optional[PipelineGlobalState]
    execution_time: float
    checkpoint_count: int
    recovery_count: int
    error_info: Optional[Dict[str, Any]] = None


@dataclass 
class RecoveryContext:
    """Context information for execution recovery."""
    execution_id: str
    failure_step: str
    failure_checkpoint_id: str
    failure_timestamp: float
    recovery_strategy: ExecutionRecoveryStrategy
    recovery_metadata: Dict[str, Any]


class DurableExecutionManager:
    """
    Manages durable pipeline execution with automatic recovery.
    
    This manager provides fault-tolerant pipeline execution by leveraging
    automatic checkpointing and implementing sophisticated recovery mechanisms.
    """
    
    def __init__(
        self,
        langgraph_manager: LangGraphGlobalContextManager,
        default_recovery_strategy: ExecutionRecoveryStrategy = ExecutionRecoveryStrategy.RESUME_FROM_LAST_CHECKPOINT,
        max_recovery_attempts: int = 3,
        recovery_delay_seconds: float = 1.0,
        enable_failure_analysis: bool = True,
        checkpoint_retention_hours: int = 24,
    ):
        """
        Initialize durable execution manager.
        
        Args:
            langgraph_manager: LangGraph state manager for persistence
            default_recovery_strategy: Default strategy for execution recovery
            max_recovery_attempts: Maximum number of recovery attempts
            recovery_delay_seconds: Delay between recovery attempts
            enable_failure_analysis: Enable detailed failure analysis
            checkpoint_retention_hours: Hours to retain checkpoints
        """
        self.langgraph_manager = langgraph_manager
        self.default_recovery_strategy = default_recovery_strategy
        self.max_recovery_attempts = max_recovery_attempts
        self.recovery_delay_seconds = recovery_delay_seconds
        self.enable_failure_analysis = enable_failure_analysis
        self.checkpoint_retention_hours = checkpoint_retention_hours
        
        # Execution tracking
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        self.recovery_history: Dict[str, List[RecoveryContext]] = {}
        
        # Performance metrics
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "recovery_attempts": 0,
            "successful_recoveries": 0,
            "average_execution_time": 0.0,
            "average_recovery_time": 0.0
        }
        
        logger.info("DurableExecutionManager initialized")
    
    async def execute_pipeline_durably(
        self,
        pipeline: Pipeline,
        config: Dict[str, Any],
        recovery_strategy: Optional[ExecutionRecoveryStrategy] = None
    ) -> ExecutionResult:
        """
        Execute pipeline with full durability guarantees.
        
        Args:
            pipeline: Pipeline to execute
            config: Execution configuration
            recovery_strategy: Optional recovery strategy override
            
        Returns:
            Execution result with status and metrics
        """
        execution_id = config.get("execution_id", f"durable_exec_{uuid.uuid4().hex[:8]}")
        thread_id = config.get("configurable", {}).get("thread_id", execution_id)
        start_time = time.time()
        
        # Track execution
        self.active_executions[execution_id] = {
            "pipeline_id": pipeline.id,
            "thread_id": thread_id,
            "start_time": start_time,
            "status": PipelineStatus.RUNNING,
            "recovery_attempts": 0
        }
        
        self.metrics["total_executions"] += 1
        
        logger.info(f"Starting durable execution: {execution_id} for pipeline: {pipeline.id}")
        
        try:
            # Check for existing execution to resume
            existing_state = await self._check_for_resumable_execution(thread_id)
            if existing_state:
                logger.info(f"Resuming existing execution: {execution_id}")
                return await self._resume_execution(
                    execution_id, pipeline, existing_state, config
                )
            
            # Start new durable execution
            result = await self._execute_new_pipeline(
                execution_id, pipeline, config, recovery_strategy
            )
            
            # Update metrics
            execution_time = time.time() - start_time
            self._update_execution_metrics(result.status, execution_time)
            
            logger.info(
                f"Durable execution completed: {execution_id} in {execution_time:.2f}s "
                f"with {result.checkpoint_count} checkpoints"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Durable execution failed: {execution_id} - {e}")
            
            # Attempt recovery if enabled
            if recovery_strategy or self.default_recovery_strategy:
                return await self._attempt_recovery(
                    execution_id, pipeline, config, e, recovery_strategy
                )
            
            # Create failure result
            self.metrics["failed_executions"] += 1
            return ExecutionResult(
                execution_id=execution_id,
                pipeline_id=pipeline.id,
                status=PipelineStatus.FAILED,
                final_state=None,
                execution_time=time.time() - start_time,
                checkpoint_count=0,
                recovery_count=0,
                error_info={"error": str(e), "type": type(e).__name__}
            )
            
        finally:
            # Clean up active execution tracking
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
    
    async def _execute_new_pipeline(
        self,
        execution_id: str,
        pipeline: Pipeline,
        config: Dict[str, Any],
        recovery_strategy: Optional[ExecutionRecoveryStrategy] = None
    ) -> ExecutionResult:
        """Execute a new pipeline with automatic checkpointing."""
        thread_id = config.get("configurable", {}).get("thread_id", execution_id)
        
        # Create automatic checkpointing graph
        checkpointing_graph = AutomaticCheckpointingGraph(
            pipeline=pipeline,
            langgraph_manager=self.langgraph_manager,
            checkpoint_frequency=config.get("checkpoint_frequency", "every_step"),
            max_checkpoint_overhead_ms=config.get("max_checkpoint_overhead_ms", 100.0)
        )
        
        # Initialize pipeline state
        initial_state = create_initial_pipeline_state(
            pipeline_id=pipeline.id,
            thread_id=thread_id,
            execution_id=execution_id,
            inputs=config.get("inputs", {}),
            user_id=config.get("user_id"),
            session_id=config.get("session_id")
        )
        
        # Add execution metadata
        initial_state["execution_metadata"]["durable_execution"] = True
        initial_state["execution_metadata"]["recovery_strategy"] = (
            recovery_strategy.value if recovery_strategy else self.default_recovery_strategy.value
        )
        initial_state["execution_metadata"]["max_recovery_attempts"] = self.max_recovery_attempts
        
        try:
            # Execute with checkpoints
            final_state = await checkpointing_graph.execute_with_checkpoints(
                initial_state=initial_state,
                thread_id=thread_id
            )
            
            # Get checkpoint statistics
            checkpoint_stats = checkpointing_graph.get_checkpoint_statistics()
            
            return ExecutionResult(
                execution_id=execution_id,
                pipeline_id=pipeline.id,
                status=PipelineStatus.COMPLETED,
                final_state=final_state,
                execution_time=final_state["execution_metadata"]["total_execution_time"],
                checkpoint_count=checkpoint_stats["total_checkpoints"],
                recovery_count=0
            )
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {execution_id} - {e}")
            raise
    
    async def _check_for_resumable_execution(self, thread_id: str) -> Optional[PipelineGlobalState]:
        """Check if there's an existing execution that can be resumed."""
        try:
            # Get current state from LangGraph manager
            current_state = await self.langgraph_manager.get_global_state(thread_id)
            
            if not current_state:
                return None
            
            # Check if execution was interrupted (not completed or failed)
            execution_status = current_state.get("execution_metadata", {}).get("status")
            if execution_status in [PipelineStatus.RUNNING, PipelineStatus.PAUSED]:
                logger.info(f"Found resumable execution for thread: {thread_id}")
                return current_state
            
            return None
            
        except Exception as e:
            logger.warning(f"Error checking for resumable execution: {e}")
            return None
    
    async def _resume_execution(
        self,
        execution_id: str,
        pipeline: Pipeline,
        existing_state: PipelineGlobalState,
        config: Dict[str, Any]
    ) -> ExecutionResult:
        """Resume execution from existing state."""
        thread_id = existing_state["thread_id"]
        resume_start_time = time.time()
        
        logger.info(f"Resuming execution {execution_id} from existing state")
        
        # Update state for resume
        existing_state["execution_metadata"]["resume_timestamp"] = resume_start_time
        existing_state["execution_metadata"]["status"] = PipelineStatus.RUNNING
        
        # Determine resume point
        completed_steps = existing_state.get("execution_metadata", {}).get("completed_steps", [])
        failed_steps = existing_state.get("execution_metadata", {}).get("failed_steps", [])
        
        # Create modified pipeline for resumption
        resume_pipeline = self._create_resume_pipeline(pipeline, completed_steps, failed_steps)
        
        # Execute remaining steps
        try:
            checkpointing_graph = AutomaticCheckpointingGraph(
                pipeline=resume_pipeline,
                langgraph_manager=self.langgraph_manager
            )
            
            final_state = await checkpointing_graph.execute_with_checkpoints(
                initial_state=existing_state,
                thread_id=thread_id
            )
            
            # Calculate total execution time including original execution
            original_start = existing_state.get("execution_metadata", {}).get("execution_start_time", resume_start_time)
            total_execution_time = time.time() - original_start
            
            checkpoint_stats = checkpointing_graph.get_checkpoint_statistics()
            
            return ExecutionResult(
                execution_id=execution_id,
                pipeline_id=pipeline.id,
                status=PipelineStatus.COMPLETED,
                final_state=final_state,
                execution_time=total_execution_time,
                checkpoint_count=checkpoint_stats["total_checkpoints"],
                recovery_count=1
            )
            
        except Exception as e:
            logger.error(f"Failed to resume execution {execution_id}: {e}")
            raise
    
    def _create_resume_pipeline(
        self, 
        original_pipeline: Pipeline, 
        completed_steps: List[str], 
        failed_steps: List[str]
    ) -> Pipeline:
        """Create a pipeline for resumption by filtering out completed steps."""
        # This is a simplified implementation
        # In a full implementation, we would create a new pipeline with only remaining tasks
        
        # For now, return the original pipeline and rely on the execution logic
        # to skip completed steps based on the state
        return original_pipeline
    
    async def _attempt_recovery(
        self,
        execution_id: str,
        pipeline: Pipeline,
        config: Dict[str, Any],
        original_error: Exception,
        recovery_strategy: Optional[ExecutionRecoveryStrategy] = None
    ) -> ExecutionResult:
        """Attempt to recover from execution failure."""
        strategy = recovery_strategy or self.default_recovery_strategy
        recovery_start_time = time.time()
        
        # Track recovery attempt
        if execution_id not in self.recovery_history:
            self.recovery_history[execution_id] = []
        
        current_attempt = len(self.recovery_history[execution_id]) + 1
        
        if current_attempt > self.max_recovery_attempts:
            logger.error(f"Max recovery attempts exceeded for execution: {execution_id}")
            self.metrics["failed_executions"] += 1
            return ExecutionResult(
                execution_id=execution_id,
                pipeline_id=pipeline.id,
                status=PipelineStatus.FAILED,
                final_state=None,
                execution_time=time.time() - recovery_start_time,
                checkpoint_count=0,
                recovery_count=current_attempt - 1,
                error_info={
                    "error": "Max recovery attempts exceeded",
                    "original_error": str(original_error)
                }
            )
        
        logger.info(f"Attempting recovery {current_attempt}/{self.max_recovery_attempts} for execution: {execution_id}")
        self.metrics["recovery_attempts"] += 1
        
        # Create recovery context
        recovery_context = RecoveryContext(
            execution_id=execution_id,
            failure_step="unknown",
            failure_checkpoint_id="unknown", 
            failure_timestamp=time.time(),
            recovery_strategy=strategy,
            recovery_metadata={"attempt": current_attempt, "original_error": str(original_error)}
        )
        
        self.recovery_history[execution_id].append(recovery_context)
        
        # Wait before recovery attempt
        if self.recovery_delay_seconds > 0:
            await asyncio.sleep(self.recovery_delay_seconds)
        
        try:
            # Execute recovery based on strategy
            if strategy == ExecutionRecoveryStrategy.RESUME_FROM_LAST_CHECKPOINT:
                result = await self._recover_from_last_checkpoint(execution_id, pipeline, config)
            elif strategy == ExecutionRecoveryStrategy.RESTART_FROM_BEGINNING:
                result = await self._recover_with_restart(execution_id, pipeline, config)
            else:
                # Fallback to restart
                result = await self._recover_with_restart(execution_id, pipeline, config)
            
            if result.status == PipelineStatus.COMPLETED:
                self.metrics["successful_recoveries"] += 1
                logger.info(f"Recovery successful for execution: {execution_id}")
            
            return result
            
        except Exception as recovery_error:
            logger.error(f"Recovery attempt failed for execution {execution_id}: {recovery_error}")
            
            # Try next recovery attempt recursively
            return await self._attempt_recovery(
                execution_id, pipeline, config, recovery_error, recovery_strategy
            )
    
    async def _recover_from_last_checkpoint(
        self,
        execution_id: str,
        pipeline: Pipeline,
        config: Dict[str, Any]
    ) -> ExecutionResult:
        """Recover execution from the last available checkpoint."""
        thread_id = config.get("configurable", {}).get("thread_id", execution_id)
        
        # Get the last state from LangGraph
        last_state = await self.langgraph_manager.get_global_state(thread_id)
        
        if last_state:
            logger.info(f"Recovering from last checkpoint for execution: {execution_id}")
            return await self._resume_execution(execution_id, pipeline, last_state, config)
        else:
            logger.warning(f"No checkpoint found for recovery, restarting: {execution_id}")
            return await self._recover_with_restart(execution_id, pipeline, config)
    
    async def _recover_with_restart(
        self,
        execution_id: str,
        pipeline: Pipeline,
        config: Dict[str, Any]
    ) -> ExecutionResult:
        """Recover by restarting the execution from the beginning."""
        logger.info(f"Restarting execution from beginning: {execution_id}")
        
        # Create new execution ID to avoid conflicts
        new_execution_id = f"{execution_id}_restart_{int(time.time())}"
        config = config.copy()
        config["execution_id"] = new_execution_id
        
        return await self._execute_new_pipeline(
            new_execution_id, pipeline, config, None
        )
    
    def _update_execution_metrics(self, status: PipelineStatus, execution_time: float):
        """Update execution performance metrics."""
        if status == PipelineStatus.COMPLETED:
            self.metrics["successful_executions"] += 1
        else:
            self.metrics["failed_executions"] += 1
        
        # Update average execution time
        total_successful = self.metrics["successful_executions"]
        if total_successful > 0:
            current_avg = self.metrics["average_execution_time"]
            self.metrics["average_execution_time"] = (
                (current_avg * (total_successful - 1) + execution_time) / total_successful
            )
    
    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of an execution."""
        if execution_id in self.active_executions:
            execution_info = self.active_executions[execution_id]
            thread_id = execution_info["thread_id"]
            
            # Get current state
            current_state = await self.langgraph_manager.get_global_state(thread_id)
            
            return {
                "execution_id": execution_id,
                "pipeline_id": execution_info["pipeline_id"],
                "status": execution_info["status"],
                "start_time": execution_info["start_time"],
                "current_step": current_state.get("execution_metadata", {}).get("current_step") if current_state else None,
                "completed_steps": current_state.get("execution_metadata", {}).get("completed_steps", []) if current_state else [],
                "recovery_attempts": execution_info["recovery_attempts"],
                "checkpoint_count": len(current_state.get("execution_metadata", {}).get("checkpoints", [])) if current_state else 0
            }
        
        return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get execution performance metrics."""
        return {
            **self.metrics,
            "active_executions": len(self.active_executions),
            "recovery_success_rate": (
                self.metrics["successful_recoveries"] / max(self.metrics["recovery_attempts"], 1)
            ) * 100,
            "execution_success_rate": (
                self.metrics["successful_executions"] / max(self.metrics["total_executions"], 1)
            ) * 100
        }
    
    async def cleanup_old_executions(self, max_age_hours: Optional[int] = None) -> int:
        """Clean up old execution data and checkpoints."""
        max_age = max_age_hours or self.checkpoint_retention_hours
        cleanup_count = 0
        
        # This would implement cleanup logic for old checkpoints and execution data
        # For now, it's a placeholder
        
        logger.info(f"Cleaned up {cleanup_count} old executions older than {max_age} hours")
        return cleanup_count
    
    async def pause_execution(self, execution_id: str) -> bool:
        """Pause an active execution."""
        if execution_id in self.active_executions:
            self.active_executions[execution_id]["status"] = PipelineStatus.PAUSED
            
            # Update state in LangGraph
            thread_id = self.active_executions[execution_id]["thread_id"]
            await self.langgraph_manager.update_global_state(
                thread_id,
                {"execution_metadata": {"status": PipelineStatus.PAUSED, "pause_timestamp": time.time()}}
            )
            
            logger.info(f"Paused execution: {execution_id}")
            return True
        
        return False
    
    async def resume_paused_execution(self, execution_id: str) -> bool:
        """Resume a paused execution."""
        if execution_id in self.active_executions:
            if self.active_executions[execution_id]["status"] == PipelineStatus.PAUSED:
                self.active_executions[execution_id]["status"] = PipelineStatus.RUNNING
                
                # Update state in LangGraph
                thread_id = self.active_executions[execution_id]["thread_id"]
                await self.langgraph_manager.update_global_state(
                    thread_id,
                    {"execution_metadata": {"status": PipelineStatus.RUNNING, "resume_timestamp": time.time()}}
                )
                
                logger.info(f"Resumed execution: {execution_id}")
                return True
        
        return False