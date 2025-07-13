"""Main orchestration engine."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Set

from .compiler.yaml_compiler import YAMLCompiler
from .core.control_system import ControlSystem, MockControlSystem
from .core.pipeline import Pipeline
from .core.task import Task, TaskStatus
from .core.error_handler import ErrorHandler
from .core.resource_allocator import ResourceAllocator
from .models.model_registry import ModelRegistry
from .state.state_manager import StateManager
from .executor.parallel_executor import ParallelExecutor


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
        error_handler: Optional[ErrorHandler] = None,
        resource_allocator: Optional[ResourceAllocator] = None,
        parallel_executor: Optional[ParallelExecutor] = None,
        max_concurrent_tasks: int = 10,
    ) -> None:
        """
        Initialize orchestrator.
        
        Args:
            model_registry: Model registry for model selection
            control_system: Control system for task execution
            state_manager: State manager for checkpointing
            yaml_compiler: YAML compiler for pipeline parsing
            error_handler: Error handler for fault tolerance
            resource_allocator: Resource allocator for task scheduling
            parallel_executor: Parallel executor for concurrent execution
            max_concurrent_tasks: Maximum concurrent tasks
        """
        self.model_registry = model_registry or ModelRegistry()
        self.control_system = control_system or MockControlSystem()
        self.state_manager = state_manager or StateManager()
        
        # Register default models if registry is empty
        if not self.model_registry.models:
            self._register_default_models()
        self.yaml_compiler = yaml_compiler or YAMLCompiler()
        self.error_handler = error_handler or ErrorHandler()
        self.resource_allocator = resource_allocator or ResourceAllocator()
        self.parallel_executor = parallel_executor or ParallelExecutor()
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
        
        # Get execution levels (groups of tasks that can run in parallel)
        execution_levels = pipeline.get_execution_levels()
        
        # Execute tasks level by level
        for level_index, level in enumerate(execution_levels):
            context["current_level"] = level_index
            
            # Execute tasks in parallel within the level
            level_results = await self._execute_level(pipeline, level, context, results)
            
            # Check for failures
            failed_tasks = [
                task_id for task_id in level
                if pipeline.get_task(task_id) and pipeline.get_task(task_id).status == TaskStatus.FAILED
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
        # Allocate resources for tasks
        resource_allocations = {}
        for task_id in level_tasks:
            task = pipeline.get_task(task_id)
            if task is None:
                raise ValueError(f"Task '{task_id}' not found in pipeline")
            
            # Determine resource requirements
            resource_requirements = self._get_task_resource_requirements(task)
            
            # Create resource request
            from .core.resource_allocator import ResourceRequest
            request = ResourceRequest(
                task_id=task_id,
                resources=resource_requirements,
                min_resources={k: v * 0.5 for k, v in resource_requirements.items()},
                priority=task.metadata.get("priority", 1.0)
            )
            
            # Request resources
            allocation_success = await self.resource_allocator.request_resources(request)
            resource_allocations[task_id] = allocation_success
        
        try:
            # Execute tasks using parallel executor
            execution_tasks = []
            scheduled_task_ids = []
            results = {}
            
            for task_id in level_tasks:
                task = pipeline.get_task(task_id)
                
                # Skip tasks that are already marked as skipped
                if task.status == TaskStatus.SKIPPED:
                    results[task_id] = {"status": "skipped"}
                    continue
                    
                task_context = {
                    **context,
                    "task_id": task_id,
                    "previous_results": previous_results,
                    "resource_allocation": resource_allocations[task_id],
                }
                execution_tasks.append(self._execute_task_with_resources(task, task_context))
                scheduled_task_ids.append(task_id)
            
            # Execute tasks with proper error handling
            if execution_tasks:
                # Execute tasks concurrently with semaphore control
                task_results = await asyncio.gather(*execution_tasks, return_exceptions=True)
                
                for task_id, result in zip(scheduled_task_ids, task_results):
                    if isinstance(result, Exception):
                        # Task failed - use error handler
                        task = pipeline.get_task(task_id)
                        try:
                            handled_error = await self.error_handler.handle_error(result, context)
                        except Exception:
                            # Fallback to original error
                            handled_error = result
                        task.fail(handled_error)
                        results[task_id] = {"error": str(handled_error)}
                    else:
                        # Task succeeded
                        results[task_id] = result
            
            # Note: Skipped tasks are already handled above at line 236
            # This loop was redundant and has been removed
        
        finally:
            # Release resources
            for task_id, allocation_success in resource_allocations.items():
                if allocation_success:
                    await self.resource_allocator.release_resources(task_id)
        
        return results
    
    async def _execute_task_with_resources(
        self,
        task: Task,
        context: Dict[str, Any],
    ) -> Any:
        """Execute a single task with resource management."""
        # Mark task as running
        task.start()
        
        try:
            # Select appropriate model for the task
            model = await self._select_model_for_task(task, context)
            if model:
                context["model"] = model
            
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
                return await self._execute_task_with_resources(task, context)
            else:
                raise e
    
    async def _execute_task(
        self,
        task: Task,
        context: Dict[str, Any],
    ) -> Any:
        """Execute a single task (legacy method)."""
        async with self.execution_semaphore:
            return await self._execute_task_with_resources(task, context)
    
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
            "id": pipeline.id,
            "name": pipeline.name,
            "tasks": {task_id: task.to_dict() for task_id, task in pipeline.tasks.items()},
            "context": pipeline.context,
            "metadata": pipeline.metadata,
            "created_at": pipeline.created_at,
            "version": pipeline.version,
            "description": pipeline.description,
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
    
    def _get_task_resource_requirements(self, task: Task) -> Dict[str, Any]:
        """Get resource requirements for a task."""
        # Extract resource requirements from task metadata
        requirements = {
            "cpu": task.metadata.get("cpu_cores", 1),
            "memory": task.metadata.get("memory_mb", 512),
            "gpu": task.metadata.get("gpu_required", False),
            "gpu_memory": task.metadata.get("gpu_memory_mb", 0),
            "timeout": task.timeout or 300,  # Default 5 minutes
        }
        
        # Add model-specific requirements
        if "requires_model" in task.metadata:
            model_name = task.metadata["requires_model"]
            model = self.model_registry.get_model(model_name)
            if model:
                requirements.update({
                    "model_memory": model.requirements.memory_gb * 1024,
                    "model_gpu": model.requirements.requires_gpu,
                    "model_gpu_memory": (model.requirements.gpu_memory_gb or 0) * 1024,
                })
        
        return requirements
    
    async def _select_model_for_task(self, task: Task, context: Dict[str, Any]) -> Optional[Any]:
        """Select appropriate model for task execution."""
        # Check if task specifies a model
        if "requires_model" in task.metadata:
            model_name = task.metadata["requires_model"]
            return self.model_registry.get_model(model_name)
        
        # Check if task requires AI capabilities
        if task.action in ["generate", "analyze", "transform", "chat"]:
            # Select best model for the task
            requirements = {
                "tasks": [task.action],
                "context_window": len(str(task.parameters).encode()) // 4,  # Rough token estimate
            }
            
            return await self.model_registry.select_model(requirements)
        
        return None
    
    def _register_default_models(self) -> None:
        """Register default models for testing and basic functionality."""
        from .core.model import MockModel
        
        # Register a default mock model that can handle basic tasks
        default_model = MockModel(
            name="default-mock",
            provider="mock",
        )
        
        # Set up the model to handle common tasks
        default_model.set_response("generate", "Generated content")
        default_model.set_response("analyze", {"analysis": "Analysis result"})
        default_model.set_response("transform", {"transformed": "Transformed data"})
        default_model.set_response("chat", "Chat response")
        
        self.model_registry.register_model(default_model)
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the orchestrator."""
        return {
            "total_executions": len(self.execution_history),
            "successful_executions": len([
                record for record in self.execution_history
                if record["status"] == "completed"
            ]),
            "failed_executions": len([
                record for record in self.execution_history
                if record["status"] == "failed"
            ]),
            "running_pipelines": len(self.running_pipelines),
            "average_execution_time": sum(
                record.get("execution_time", 0) for record in self.execution_history
            ) / len(self.execution_history) if self.execution_history else 0,
            "resource_utilization": await self.resource_allocator.get_utilization(),
            "error_rate": len([
                record for record in self.execution_history
                if record["status"] == "failed"
            ]) / len(self.execution_history) if self.execution_history else 0,
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components."""
        health_status = {
            "orchestrator": "healthy",
            "model_registry": "healthy",
            "control_system": "healthy",
            "state_manager": "healthy",
            "error_handler": "healthy",
            "resource_allocator": "healthy",
            "parallel_executor": "healthy",
        }
        
        # Check each component
        try:
            # Check model registry
            available_models = await self.model_registry.get_available_models()
            if not available_models:
                health_status["model_registry"] = "warning"
        except Exception:
            health_status["model_registry"] = "unhealthy"
        
        # Check control system
        try:
            capabilities = self.control_system.get_capabilities()
            if not capabilities:
                health_status["control_system"] = "warning"
        except Exception:
            health_status["control_system"] = "unhealthy"
        
        # Check state manager
        try:
            if not await self.state_manager.is_healthy():
                health_status["state_manager"] = "unhealthy"
        except Exception:
            health_status["state_manager"] = "unhealthy"
        
        # Check resource allocator
        try:
            utilization = await self.resource_allocator.get_utilization()
            if utilization.get("cpu_usage", 0) > 0.9:
                health_status["resource_allocator"] = "warning"
        except Exception:
            health_status["resource_allocator"] = "unhealthy"
        
        # Overall health
        unhealthy_components = [
            comp for comp, status in health_status.items()
            if status == "unhealthy"
        ]
        
        if unhealthy_components:
            health_status["overall"] = "unhealthy"
        elif any(status == "warning" for status in health_status.values()):
            health_status["overall"] = "warning"
        else:
            health_status["overall"] = "healthy"
        
        return health_status
    
    async def shutdown(self) -> None:
        """Shutdown orchestrator and clean up resources."""
        # Wait for running pipelines to complete
        if self.running_pipelines:
            await asyncio.sleep(1)  # Give some time for cleanup
        
        # Shutdown components (only if they have shutdown methods)
        if hasattr(self.resource_allocator, 'shutdown'):
            await self.resource_allocator.shutdown()
        elif hasattr(self.resource_allocator, 'cleanup'):
            await self.resource_allocator.cleanup()
        
        if hasattr(self.parallel_executor, 'shutdown'):
            if asyncio.iscoroutinefunction(self.parallel_executor.shutdown):
                await self.parallel_executor.shutdown()
            else:
                self.parallel_executor.shutdown()
        
        if hasattr(self.state_manager, 'shutdown'):
            await self.state_manager.shutdown()
        
        # Clear state
        self.running_pipelines.clear()
        self.execution_history.clear()
    
    def __repr__(self) -> str:
        """String representation of orchestrator."""
        return f"Orchestrator(running_pipelines={len(self.running_pipelines)})"