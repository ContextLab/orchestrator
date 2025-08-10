"""Automatic Checkpointing Graph - Issue #205

LangGraph workflow with automatic step-level checkpointing.
Provides automatic checkpoint creation after each pipeline step with performance optimization.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass

# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.base import BaseCheckpointSaver

# Internal imports  
from ..core.pipeline import Pipeline
from ..core.task import Task, TaskStatus
from ..state.global_context import (
    PipelineGlobalState,
    create_initial_pipeline_state,
    validate_pipeline_state,
    PipelineStatus
)
from ..state.langgraph_state_manager import LangGraphGlobalContextManager
from ..core.exceptions import PipelineExecutionError

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for automatic checkpoints."""
    step_id: str
    checkpoint_type: str  # "pre_step", "post_step", "error"
    timestamp: float
    performance_metrics: Dict[str, Any]
    error_info: Optional[Dict[str, Any]] = None


class CheckpointedExecutionError(PipelineExecutionError):
    """Exception with checkpoint information for recovery."""
    
    def __init__(self, original_error: Exception, checkpoint_id: str):
        super().__init__(f"Execution failed at checkpoint {checkpoint_id}: {original_error}")
        self.original_error = original_error
        self.checkpoint_id = checkpoint_id


class AutomaticCheckpointingGraph:
    """
    LangGraph workflow with automatic step-level checkpointing.
    
    This class creates a StateGraph that automatically checkpoints state
    before and after each pipeline step, providing durable execution
    with minimal performance overhead.
    """
    
    def __init__(
        self,
        pipeline: Pipeline,
        langgraph_manager: LangGraphGlobalContextManager,
        checkpoint_frequency: str = "every_step",
        max_checkpoint_overhead_ms: float = 100.0,
        enable_pre_step_checkpoints: bool = True,
        enable_post_step_checkpoints: bool = True,
        enable_error_checkpoints: bool = True,
    ):
        """
        Initialize automatic checkpointing graph.
        
        Args:
            pipeline: Pipeline to execute with checkpointing
            langgraph_manager: LangGraph state manager for persistence
            checkpoint_frequency: Checkpointing frequency ("every_step", "every_n_steps", "time_based")
            max_checkpoint_overhead_ms: Maximum allowed checkpoint overhead
            enable_pre_step_checkpoints: Create checkpoints before each step
            enable_post_step_checkpoints: Create checkpoints after each step
            enable_error_checkpoints: Create checkpoints on errors
        """
        self.pipeline = pipeline
        self.langgraph_manager = langgraph_manager
        self.checkpoint_frequency = checkpoint_frequency
        self.max_checkpoint_overhead_ms = max_checkpoint_overhead_ms
        self.enable_pre_step_checkpoints = enable_pre_step_checkpoints
        self.enable_post_step_checkpoints = enable_post_step_checkpoints
        self.enable_error_checkpoints = enable_error_checkpoints
        
        # Performance tracking
        self.checkpoint_metrics = {
            "total_checkpoints": 0,
            "total_checkpoint_time_ms": 0,
            "average_checkpoint_time_ms": 0,
            "checkpoint_errors": 0,
            "performance_optimizations": 0
        }
        
        # Build the checkpointed graph
        self.graph = self._build_checkpointed_graph()
        
        logger.info(f"AutomaticCheckpointingGraph initialized for pipeline: {pipeline.id}")
        
    def _build_checkpointed_graph(self) -> StateGraph:
        """
        Build LangGraph StateGraph with automatic checkpointing.
        
        Returns:
            Compiled StateGraph with checkpointing capabilities
        """
        # Create StateGraph using PipelineGlobalState
        graph = StateGraph(PipelineGlobalState)
        
        # Add entry point
        graph.add_node("initialize_execution", self._initialize_execution_node)
        graph.set_entry_point("initialize_execution")
        
        # Add nodes for each pipeline step with checkpointing
        previous_node = "initialize_execution"
        
        for task_id in self.pipeline.get_execution_order():
            task = self.pipeline.get_task(task_id)
            
            # Create checkpointed step node
            step_node_name = f"step_{task_id}"
            graph.add_node(step_node_name, self._create_checkpointed_step(task))
            
            # Connect to previous node
            graph.add_edge(previous_node, step_node_name)
            previous_node = step_node_name
        
        # Add finalization node
        graph.add_node("finalize_execution", self._finalize_execution_node)
        graph.add_edge(previous_node, "finalize_execution")
        graph.add_edge("finalize_execution", END)
        
        # Compile with checkpointer from LangGraph manager
        return graph.compile(checkpointer=self.langgraph_manager.checkpointer)
    
    def _create_checkpointed_step(self, task: Task) -> Callable:
        """
        Create a step function with automatic checkpointing.
        
        Args:
            task: Task to wrap with checkpointing
            
        Returns:
            Async function that executes task with checkpointing
        """
        async def checkpointed_step(state: PipelineGlobalState) -> PipelineGlobalState:
            step_id = task.id
            start_time = time.time()
            
            try:
                # Pre-step checkpoint
                if self.enable_pre_step_checkpoints:
                    pre_checkpoint_start = time.time()
                    pre_checkpoint_id = await self._create_step_checkpoint(
                        state, step_id, "pre_step"
                    )
                    pre_checkpoint_time = (time.time() - pre_checkpoint_start) * 1000
                    
                    # Update state with checkpoint info
                    if "execution_metadata" not in state:
                        state["execution_metadata"] = {}
                    if "checkpoints" not in state["execution_metadata"]:
                        state["execution_metadata"]["checkpoints"] = []
                    
                    state["execution_metadata"]["checkpoints"].append({
                        "step_id": step_id,
                        "checkpoint_type": "pre_step", 
                        "checkpoint_id": pre_checkpoint_id,
                        "timestamp": time.time(),
                        "creation_time_ms": pre_checkpoint_time
                    })
                
                # Update current step
                state["execution_metadata"]["current_step"] = step_id
                state["execution_metadata"]["step_start_time"] = start_time
                
                # Execute the actual task
                logger.debug(f"Executing step: {step_id}")
                execution_result = await self._execute_task_with_context(task, state)
                
                # Update state with task result
                if "intermediate_results" not in state:
                    state["intermediate_results"] = {}
                state["intermediate_results"][step_id] = execution_result
                
                # Update execution metadata
                state["execution_metadata"]["completed_steps"].append(step_id)
                state["execution_metadata"]["step_end_time"] = time.time()
                
                # Post-step checkpoint
                if self.enable_post_step_checkpoints:
                    post_checkpoint_start = time.time()
                    post_checkpoint_id = await self._create_step_checkpoint(
                        state, step_id, "post_step", execution_result
                    )
                    post_checkpoint_time = (time.time() - post_checkpoint_start) * 1000
                    
                    state["execution_metadata"]["checkpoints"].append({
                        "step_id": step_id,
                        "checkpoint_type": "post_step",
                        "checkpoint_id": post_checkpoint_id,
                        "timestamp": time.time(),
                        "creation_time_ms": post_checkpoint_time,
                        "result_size": len(str(execution_result)) if execution_result else 0
                    })
                
                # Performance optimization check
                total_step_time = (time.time() - start_time) * 1000
                checkpoint_overhead = sum(
                    cp.get("creation_time_ms", 0) 
                    for cp in state["execution_metadata"]["checkpoints"]
                    if cp["step_id"] == step_id
                )
                
                if checkpoint_overhead > self.max_checkpoint_overhead_ms:
                    logger.warning(
                        f"Step {step_id} checkpoint overhead ({checkpoint_overhead:.1f}ms) "
                        f"exceeds limit ({self.max_checkpoint_overhead_ms}ms)"
                    )
                    self.checkpoint_metrics["performance_optimizations"] += 1
                
                logger.debug(f"Completed step: {step_id} in {total_step_time:.1f}ms")
                return state
                
            except Exception as e:
                # Error checkpoint
                if self.enable_error_checkpoints:
                    try:
                        error_checkpoint_id = await self._create_error_checkpoint(
                            state, step_id, e
                        )
                        
                        # Update state with error info
                        if "error_context" not in state:
                            state["error_context"] = {}
                        
                        state["error_context"] = {
                            "failed_step": step_id,
                            "error_message": str(e),
                            "error_type": type(e).__name__,
                            "error_checkpoint_id": error_checkpoint_id,
                            "timestamp": time.time()
                        }
                        
                        state["execution_metadata"]["failed_steps"].append(step_id)
                        
                    except Exception as checkpoint_error:
                        logger.error(f"Failed to create error checkpoint: {checkpoint_error}")
                        self.checkpoint_metrics["checkpoint_errors"] += 1
                
                # Wrap exception with checkpoint information
                raise CheckpointedExecutionError(e, error_checkpoint_id if 'error_checkpoint_id' in locals() else "unknown")
        
        return checkpointed_step
    
    async def _initialize_execution_node(self, state: PipelineGlobalState) -> PipelineGlobalState:
        """Initialize execution with proper state structure."""
        logger.info(f"Initializing checkpointed execution for pipeline: {self.pipeline.id}")
        
        # Ensure all required state fields exist
        if "execution_metadata" not in state:
            state["execution_metadata"] = {}
        
        state["execution_metadata"].update({
            "pipeline_id": self.pipeline.id,
            "execution_start_time": time.time(),
            "current_step": "initializing",
            "completed_steps": [],
            "failed_steps": [],
            "checkpoints": [],
            "checkpoint_frequency": self.checkpoint_frequency
        })
        
        if "intermediate_results" not in state:
            state["intermediate_results"] = {}
        
        if "performance_metrics" not in state:
            state["performance_metrics"] = {}
        
        # Create initial execution checkpoint
        initial_checkpoint_id = await self._create_step_checkpoint(
            state, "initialization", "execution_start"
        )
        
        state["execution_metadata"]["initial_checkpoint_id"] = initial_checkpoint_id
        
        return state
    
    async def _finalize_execution_node(self, state: PipelineGlobalState) -> PipelineGlobalState:
        """Finalize execution with completion checkpoint."""
        logger.info(f"Finalizing checkpointed execution for pipeline: {self.pipeline.id}")
        
        # Update execution metadata
        state["execution_metadata"].update({
            "status": PipelineStatus.COMPLETED,
            "execution_end_time": time.time(),
            "current_step": "completed",
            "total_execution_time": time.time() - state["execution_metadata"]["execution_start_time"]
        })
        
        # Add performance metrics
        state["performance_metrics"]["checkpoint_metrics"] = self.checkpoint_metrics.copy()
        
        # Create final completion checkpoint
        completion_checkpoint_id = await self._create_step_checkpoint(
            state, "finalization", "execution_complete"
        )
        
        state["execution_metadata"]["completion_checkpoint_id"] = completion_checkpoint_id
        
        logger.info(
            f"Pipeline {self.pipeline.id} completed with {len(state['execution_metadata']['completed_steps'])} steps "
            f"and {len(state['execution_metadata']['checkpoints'])} checkpoints"
        )
        
        return state
    
    async def _execute_task_with_context(self, task: Task, state: PipelineGlobalState) -> Any:
        """
        Execute a task with proper context from the pipeline state.
        
        Args:
            task: Task to execute
            state: Current pipeline state
            
        Returns:
            Task execution result
        """
        # Build execution context from state
        context = {
            "pipeline_id": state.get("execution_metadata", {}).get("pipeline_id"),
            "execution_id": state.get("thread_id"),
            "current_step": task.id,
            "previous_results": state.get("intermediate_results", {}),
            "global_variables": state.get("global_variables", {}),
            "step_start_time": time.time()
        }
        
        # Add any task-specific context
        if hasattr(task, 'context'):
            context.update(task.context)
        
        # Execute the task - this would integrate with the existing task execution system
        # For now, we'll simulate execution but this needs to integrate with the actual orchestrator
        try:
            # This is a placeholder - needs integration with actual task execution
            logger.debug(f"Simulating execution of task: {task.id}")
            await asyncio.sleep(0.1)  # Simulate work
            
            result = {
                "task_id": task.id,
                "status": "completed",
                "output": f"Simulated output for {task.id}",
                "execution_time": time.time() - context["step_start_time"]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Task {task.id} execution failed: {e}")
            raise
    
    async def _create_step_checkpoint(
        self, 
        state: PipelineGlobalState, 
        step_id: str, 
        checkpoint_type: str,
        execution_result: Optional[Any] = None
    ) -> str:
        """
        Create a checkpoint for the current step.
        
        Args:
            state: Current pipeline state
            step_id: ID of the step
            checkpoint_type: Type of checkpoint (pre_step, post_step, etc.)
            execution_result: Optional execution result to include
            
        Returns:
            Checkpoint ID
        """
        checkpoint_start_time = time.time()
        
        try:
            # Create checkpoint description
            description = f"{checkpoint_type} checkpoint for step: {step_id}"
            
            # Prepare checkpoint metadata
            checkpoint_metadata = {
                "step_id": step_id,
                "checkpoint_type": checkpoint_type,
                "pipeline_id": self.pipeline.id,
                "timestamp": checkpoint_start_time,
                "execution_result": execution_result
            }
            
            # Use LangGraph manager to create checkpoint
            thread_id = state.get("thread_id", f"auto_checkpoint_{uuid.uuid4().hex[:8]}")
            
            checkpoint_id = await self.langgraph_manager.create_checkpoint(
                thread_id=thread_id,
                description=description,
                metadata=checkpoint_metadata
            )
            
            # Update performance metrics
            checkpoint_time_ms = (time.time() - checkpoint_start_time) * 1000
            self.checkpoint_metrics["total_checkpoints"] += 1
            self.checkpoint_metrics["total_checkpoint_time_ms"] += checkpoint_time_ms
            self.checkpoint_metrics["average_checkpoint_time_ms"] = (
                self.checkpoint_metrics["total_checkpoint_time_ms"] / 
                self.checkpoint_metrics["total_checkpoints"]
            )
            
            logger.debug(f"Created {checkpoint_type} checkpoint {checkpoint_id} in {checkpoint_time_ms:.1f}ms")
            return checkpoint_id
            
        except Exception as e:
            self.checkpoint_metrics["checkpoint_errors"] += 1
            logger.error(f"Failed to create checkpoint for step {step_id}: {e}")
            raise
    
    async def _create_error_checkpoint(
        self, 
        state: PipelineGlobalState, 
        step_id: str, 
        error: Exception
    ) -> str:
        """
        Create a checkpoint capturing error state for debugging.
        
        Args:
            state: Current pipeline state
            step_id: ID of the failed step
            error: Exception that occurred
            
        Returns:
            Error checkpoint ID
        """
        try:
            # Enhance state with error information
            error_state = state.copy()
            error_state["error_context"] = {
                "failed_step": step_id,
                "error_message": str(error),
                "error_type": type(error).__name__,
                "stack_trace": str(error.__traceback__) if hasattr(error, '__traceback__') else None,
                "timestamp": time.time()
            }
            
            return await self._create_step_checkpoint(
                error_state, step_id, "error", {"error": str(error)}
            )
            
        except Exception as checkpoint_error:
            logger.error(f"Failed to create error checkpoint: {checkpoint_error}")
            # Return a placeholder ID to avoid breaking the flow
            return f"error_checkpoint_failed_{uuid.uuid4().hex[:8]}"
    
    async def execute_with_checkpoints(
        self, 
        initial_state: Optional[PipelineGlobalState] = None,
        thread_id: Optional[str] = None
    ) -> PipelineGlobalState:
        """
        Execute the pipeline with automatic checkpointing.
        
        Args:
            initial_state: Optional initial state (created if not provided)
            thread_id: Optional thread ID for execution
            
        Returns:
            Final execution state
        """
        # Create or use provided thread ID
        if not thread_id:
            thread_id = f"auto_exec_{uuid.uuid4().hex[:8]}"
        
        # Create initial state if not provided
        if not initial_state:
            initial_state = create_initial_pipeline_state(
                pipeline_id=self.pipeline.id,
                thread_id=thread_id,
                execution_id=f"exec_{uuid.uuid4().hex}",
                inputs={}
            )
        
        # Configure execution
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            logger.info(f"Starting checkpointed execution for pipeline: {self.pipeline.id}")
            
            # Execute the graph
            result = await self.graph.ainvoke(initial_state, config)
            
            logger.info(f"Checkpointed execution completed for pipeline: {self.pipeline.id}")
            return result
            
        except Exception as e:
            logger.error(f"Checkpointed execution failed for pipeline {self.pipeline.id}: {e}")
            raise
    
    def get_checkpoint_statistics(self) -> Dict[str, Any]:
        """Get checkpoint performance statistics."""
        return {
            **self.checkpoint_metrics,
            "pipeline_id": self.pipeline.id,
            "checkpoint_frequency": self.checkpoint_frequency,
            "max_checkpoint_overhead_ms": self.max_checkpoint_overhead_ms,
            "features": {
                "pre_step_checkpoints": self.enable_pre_step_checkpoints,
                "post_step_checkpoints": self.enable_post_step_checkpoints,
                "error_checkpoints": self.enable_error_checkpoints
            }
        }