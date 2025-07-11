"""Main orchestration engine."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Set

from .compiler.yaml_compiler import YAMLCompiler
from .core.control_system import ControlSystem, MockControlSystem
from .core.pipeline import Pipeline
from .core.task import Task, TaskStatus
from .models.model_registry import ModelRegistry
from .state.state_manager import StateManager


class ExecutionError(Exception):
    """Raised when pipeline execution fails."""
    pass


class Orchestrator:
    """
    Main orchestration engine.
    
    Coordinates the execution of pipelines by managing tasks,
    dependencies, and control systems.
    """
    
    def __init__(
        self,
        model_registry: Optional[ModelRegistry] = None,
        control_system: Optional[ControlSystem] = None,
        state_manager: Optional[StateManager] = None,
        yaml_compiler: Optional[YAMLCompiler] = None,
        max_concurrent_tasks: int = 10,
    ) -> None:
        """
        Initialize orchestrator.
        
        Args:
            model_registry: Model registry for model selection
            control_system: Control system for task execution
            state_manager: State manager for checkpointing
            yaml_compiler: YAML compiler for pipeline parsing
            max_concurrent_tasks: Maximum concurrent tasks
        """
        self.model_registry = model_registry or ModelRegistry()
        self.control_system = control_system or MockControlSystem()
        self.state_manager = state_manager or StateManager()
        self.yaml_compiler = yaml_compiler or YAMLCompiler()
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Execution state
        self.running_pipelines: Dict[str, Pipeline] = {}
        self.execution_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.execution_history: List[Dict[str, Any]] = []
    
    async def execute_pipeline(
        self,
        pipeline: Pipeline,
        checkpoint_enabled: bool = True,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Execute a pipeline.
        
        Args:
            pipeline: Pipeline to execute
            checkpoint_enabled: Whether to enable checkpointing
            max_retries: Maximum number of retries for failed tasks
            
        Returns:
            Execution results
            
        Raises:
            ExecutionError: If execution fails
        """
        execution_id = f"{pipeline.id}_{int(time.time())}"
        
        try:
            # Register pipeline as running
            self.running_pipelines[execution_id] = pipeline
            
            # Create execution context
            context = {
                "pipeline_id": pipeline.id,
                "execution_id": execution_id,
                "checkpoint_enabled": checkpoint_enabled,
                "max_retries": max_retries,
                "start_time": time.time(),
            }
            
            # Save initial checkpoint if enabled
            if checkpoint_enabled:
                await self.state_manager.save_checkpoint(
                    execution_id,
                    self._get_pipeline_state(pipeline),
                    context
                )
            
            # Execute pipeline
            results = await self._execute_pipeline_internal(pipeline, context)
            
            # Record successful execution
            execution_record = {
                "execution_id": execution_id,
                "pipeline_id": pipeline.id,
                "status": "completed",
                "results": results,
                "execution_time": time.time() - context["start_time"],
                "completed_at": time.time(),
            }
            self.execution_history.append(execution_record)
            
            return results
            
        except Exception as e:
            # Record failed execution
            execution_record = {
                "execution_id": execution_id,
                "pipeline_id": pipeline.id,
                "status": "failed",
                "error": str(e),
                "execution_time": time.time() - context.get("start_time", time.time()),
                "failed_at": time.time(),
            }
            self.execution_history.append(execution_record)
            
            raise ExecutionError(f"Pipeline execution failed: {e}") from e
            
        finally:
            # Clean up
            if execution_id in self.running_pipelines:
                del self.running_pipelines[execution_id]
    
    async def _execute_pipeline_internal(
        self,
        pipeline: Pipeline,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Internal pipeline execution logic."""
        results = {}
        
        # Get execution order
        execution_order = pipeline.get_execution_order()
        
        # Execute tasks level by level
        for level_index, level in enumerate(execution_order):
            context["current_level"] = level_index
            
            # Execute tasks in parallel within the level
            level_results = await self._execute_level(pipeline, level, context, results)
            
            # Check for failures
            failed_tasks = [
                task_id for task_id in level
                if pipeline.get_task(task_id).status == TaskStatus.FAILED
            ]
            
            if failed_tasks:
                # Handle failures based on policy
                await self._handle_task_failures(pipeline, failed_tasks, context)
            
            # Update results
            results.update(level_results)
            
            # Save checkpoint after each level
            if context.get("checkpoint_enabled", False):
                await self.state_manager.save_checkpoint(
                    context["execution_id"],
                    self._get_pipeline_state(pipeline),
                    context
                )
        
        return results
    
    async def _execute_level(
        self,
        pipeline: Pipeline,
        level_tasks: List[str],
        context: Dict[str, Any],
        previous_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute tasks in a single level."""
        # Create tasks for parallel execution
        tasks = []
        for task_id in level_tasks:
            task = pipeline.get_task(task_id)
            task_context = {
                **context,
                "task_id": task_id,
                "previous_results": previous_results,
            }
            tasks.append(self._execute_task(task, task_context))
        
        # Execute tasks with concurrency control
        results = {}
        if tasks:
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for task_id, result in zip(level_tasks, task_results):
                if isinstance(result, Exception):
                    # Task failed
                    task = pipeline.get_task(task_id)
                    task.fail(result)
                    results[task_id] = {"error": str(result)}
                else:
                    # Task succeeded
                    results[task_id] = result
        
        return results
    
    async def _execute_task(
        self,
        task: Task,
        context: Dict[str, Any],
    ) -> Any:
        """Execute a single task."""
        async with self.execution_semaphore:
            # Mark task as running
            task.start()
            
            try:
                # Execute task using control system
                result = await self.control_system.execute_task(task, context)
                
                # Mark task as completed
                task.complete(result)
                
                return result
                
            except Exception as e:
                # Mark task as failed
                task.fail(e)
                
                # Check if task can be retried
                if task.can_retry():
                    # Reset task and retry
                    task.reset()
                    return await self._execute_task(task, context)
                else:
                    raise e
    
    async def _handle_task_failures(
        self,
        pipeline: Pipeline,
        failed_tasks: List[str],
        context: Dict[str, Any],
    ) -> None:
        """Handle task failures based on policy."""
        for task_id in failed_tasks:
            task = pipeline.get_task(task_id)
            failure_policy = task.metadata.get("on_failure", "fail")
            
            if failure_policy == "continue":
                # Continue with other tasks
                continue
            elif failure_policy == "skip":
                # Skip dependent tasks
                self._skip_dependent_tasks(pipeline, task_id)
            elif failure_policy == "fail":
                # Fail entire pipeline
                raise ExecutionError(f"Task '{task_id}' failed and policy is 'fail'")
            elif failure_policy == "retry":
                # Retry is handled in _execute_task
                continue
    
    def _skip_dependent_tasks(self, pipeline: Pipeline, failed_task_id: str) -> None:
        """Skip tasks that depend on a failed task."""
        # Find all tasks that depend on the failed task
        to_skip = set()
        
        def find_dependents(task_id: str) -> None:
            for tid, task in pipeline.tasks.items():
                if task_id in task.dependencies and tid not in to_skip:
                    to_skip.add(tid)
                    find_dependents(tid)  # Recursively find dependents
        
        find_dependents(failed_task_id)
        
        # Skip all dependent tasks
        for task_id in to_skip:
            task = pipeline.get_task(task_id)
            task.skip(f"Dependency '{failed_task_id}' failed")
    
    def _get_pipeline_state(self, pipeline: Pipeline) -> Dict[str, Any]:
        """Get current pipeline state for checkpointing."""
        return {
            "pipeline_id": pipeline.id,
            "tasks": {task_id: task.to_dict() for task_id, task in pipeline.tasks.items()},
            "context": pipeline.context,
            "metadata": pipeline.metadata,
        }
    
    async def execute_yaml(
        self,
        yaml_content: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute pipeline from YAML content.
        
        Args:
            yaml_content: YAML pipeline definition
            context: Template context variables
            **kwargs: Additional execution parameters
            
        Returns:
            Execution results
        """
        # Compile YAML to pipeline
        pipeline = await self.yaml_compiler.compile(yaml_content, context)
        
        # Execute pipeline
        return await self.execute_pipeline(pipeline, **kwargs)
    
    async def execute_yaml_file(
        self,
        yaml_file: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute pipeline from YAML file.
        
        Args:
            yaml_file: Path to YAML file
            context: Template context variables
            **kwargs: Additional execution parameters
            
        Returns:
            Execution results
        """
        with open(yaml_file, 'r') as f:
            yaml_content = f.read()
        
        return await self.execute_yaml(yaml_content, context, **kwargs)
    
    async def recover_pipeline(
        self,
        execution_id: str,
        from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Recover a failed pipeline from checkpoint.
        
        Args:
            execution_id: Execution ID to recover
            from_checkpoint: Specific checkpoint to recover from
            
        Returns:
            Recovery results
        """
        # Load checkpoint
        checkpoint = await self.state_manager.restore_checkpoint(
            execution_id,
            from_checkpoint
        )
        
        if not checkpoint:
            raise ExecutionError(f"No checkpoint found for execution '{execution_id}'")
        
        # Reconstruct pipeline
        pipeline = Pipeline.from_dict(checkpoint["state"])
        
        # Reset failed tasks to pending
        for task in pipeline.tasks.values():
            if task.status == TaskStatus.FAILED:
                task.reset()
        
        # Re-execute pipeline
        return await self.execute_pipeline(pipeline)
    
    def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """
        Get execution status.
        
        Args:
            execution_id: Execution ID
            
        Returns:
            Status information
        """
        if execution_id in self.running_pipelines:
            pipeline = self.running_pipelines[execution_id]
            return {
                "execution_id": execution_id,
                "status": "running",
                "progress": pipeline.get_progress(),
                "pipeline": pipeline.to_dict(),
            }
        
        # Check execution history
        for record in self.execution_history:
            if record["execution_id"] == execution_id:
                return record
        
        return {"execution_id": execution_id, "status": "not_found"}
    
    def list_running_pipelines(self) -> List[str]:
        """List all running pipeline execution IDs."""
        return list(self.running_pipelines.keys())
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get execution history.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of execution records
        """
        return self.execution_history[-limit:]
    
    def clear_execution_history(self) -> None:
        """Clear execution history."""
        self.execution_history.clear()
    
    async def shutdown(self) -> None:
        """Shutdown orchestrator and clean up resources."""
        # Wait for running pipelines to complete
        if self.running_pipelines:
            await asyncio.sleep(1)  # Give some time for cleanup
        
        # Clear state
        self.running_pipelines.clear()
        self.execution_history.clear()
    
    def __repr__(self) -> str:
        """String representation of orchestrator."""
        return f"Orchestrator(running_pipelines={len(self.running_pipelines)})"