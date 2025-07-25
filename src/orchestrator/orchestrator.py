"""Main orchestration engine."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from .compiler.yaml_compiler import YAMLCompiler
from .core.control_system import ControlSystem
from .core.error_handler import ErrorHandler
from .core.pipeline import Pipeline
from .core.pipeline_status_tracker import PipelineStatusTracker
from .core.pipeline_resume_manager import PipelineResumeManager, ResumeStrategy
from .core.resource_allocator import ResourceAllocator
from .core.task import Task, TaskStatus
from .executor.parallel_executor import ParallelExecutor
from .models.model_registry import ModelRegistry
from .models.registry_singleton import get_model_registry
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
        self.model_registry = model_registry or get_model_registry()

        # Use a proper control system if none provided
        if control_system is None:
            # We need models to create a real control system
            if not self.model_registry.models:
                raise RuntimeError(
                    "No control system provided and no models available. "
                    "Initialize models first with init_models() or provide a control system."
                )

            from .control_systems.hybrid_control_system import HybridControlSystem

            control_system = HybridControlSystem(self.model_registry)

        self.control_system = control_system
        self.state_manager = state_manager or StateManager()

        # No default models - must be explicitly initialized
        self.yaml_compiler = yaml_compiler or YAMLCompiler(
            model_registry=self.model_registry
        )
        self.error_handler = error_handler or ErrorHandler()
        self.resource_allocator = resource_allocator or ResourceAllocator()
        self.parallel_executor = parallel_executor or ParallelExecutor()
        self.max_concurrent_tasks = max_concurrent_tasks

        # Execution state
        self.running_pipelines: Dict[str, Pipeline] = (
            {}
        )  # Keep for backward compatibility
        self.execution_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.execution_history: List[Dict[str, Any]] = (
            []
        )  # Keep for backward compatibility

        # New status tracker and resume manager
        self.status_tracker = PipelineStatusTracker()
        self.resume_manager = PipelineResumeManager(self.state_manager)

        # Logger
        self.logger = logging.getLogger(__name__)

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

        # Create execution context early to avoid UnboundLocalError
        context = {
            "pipeline_id": pipeline.id,
            "pipeline_metadata": pipeline.metadata,  # Include pipeline metadata for model selection
            "execution_id": execution_id,
            "checkpoint_enabled": checkpoint_enabled,
            "max_retries": max_retries,
            "start_time": time.time(),
        }

        try:
            # Register pipeline as running
            self.running_pipelines[execution_id] = pipeline

            # Register with new status tracker
            await self.status_tracker.start_execution(execution_id, pipeline, context)

            # Save initial checkpoint if enabled
            if checkpoint_enabled:
                await self.state_manager.save_checkpoint(
                    execution_id, self._get_pipeline_state(pipeline), context
                )

            # Execute pipeline
            results = await self._execute_pipeline_internal(pipeline, context)

            # Extract outputs if defined in pipeline metadata
            final_result = results
            if pipeline.metadata.get("outputs"):
                outputs = self._extract_outputs(pipeline, results)
                final_result = {"steps": results, "outputs": outputs}

            # Record successful execution
            execution_record = {
                "execution_id": execution_id,
                "pipeline_id": pipeline.id,
                "status": "completed",
                "results": final_result,
                "execution_time": time.time() - context["start_time"],
                "completed_at": time.time(),
            }
            self.execution_history.append(execution_record)

            return final_result

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

    async def _execute_step(self, task, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single step/task.

        This method is used by tests to override behavior for minimal responses.

        Args:
            task: Task object to execute
            context: Execution context

        Returns:
            Step execution result
        """
        # Execute task using control system
        result = await self.control_system.execute_task(task, context)
        return result

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
                task_id
                for task_id in level
                if pipeline.get_task(task_id)
                and pipeline.get_task(task_id).status == TaskStatus.FAILED
            ]

            if failed_tasks:
                # Handle failures based on policy
                await self._handle_task_failures(pipeline, failed_tasks, context)

            # Update results
            results.update(level_results)

            # Save checkpoint after each level
            if context.get("checkpoint_enabled", False):
                await self.state_manager.save_checkpoint(
                    context["execution_id"], self._get_pipeline_state(pipeline), context
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
                priority=task.metadata.get("priority", 1.0),
            )

            # Request resources
            allocation_success = await self.resource_allocator.request_resources(
                request
            )
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
                execution_tasks.append(
                    self._execute_task_with_resources(task, task_context)
                )
                scheduled_task_ids.append(task_id)

            # Execute tasks with proper error handling
            if execution_tasks:
                # Execute tasks concurrently with semaphore control
                task_results = await asyncio.gather(
                    *execution_tasks, return_exceptions=True
                )

                for task_id, result in zip(scheduled_task_ids, task_results):
                    if isinstance(result, Exception):
                        # Task failed - use error handler
                        task = pipeline.get_task(task_id)
                        try:
                            handled_error = await self.error_handler.handle_error(
                                result, context
                            )
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

            # Execute task using _execute_step (allows test override)
            result = await self._execute_step(task, context)

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
            "tasks": {
                task_id: task.to_dict() for task_id, task in pipeline.tasks.items()
            },
            "context": pipeline.context,
            "metadata": pipeline.metadata,
            "created_at": pipeline.created_at,
            "version": pipeline.version,
            "description": pipeline.description,
        }

    def _extract_outputs(
        self, pipeline: Pipeline, results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract outputs from pipeline results based on output definitions.

        Args:
            pipeline: Pipeline with output definitions in metadata
            results: Step execution results

        Returns:
            Extracted outputs
        """
        outputs = {}
        output_defs = pipeline.metadata.get("outputs", {})

        # Use Jinja2 for template rendering to support filters
        from jinja2 import Template, TemplateError

        for output_name, output_expr in output_defs.items():
            try:
                if isinstance(output_expr, str) and "{{" in output_expr:
                    # Render template with results context
                    template = Template(output_expr)
                    # Create a context that includes all step results
                    # Also create objects with .result attribute for backward compatibility
                    context = {}
                    for step_id, step_result in results.items():
                        context[step_id] = step_result
                        # Create an object-like dict with result attribute
                        if isinstance(step_result, str):
                            context[step_id] = type(
                                "Result", (), {"result": step_result}
                            )()
                        elif isinstance(step_result, dict) and "result" in step_result:
                            context[step_id] = type("Result", (), step_result)()
                        elif isinstance(step_result, dict):
                            # If dict doesn't have 'result' key, wrap the whole dict
                            context[step_id] = type(
                                "Result", (), {"result": step_result}
                            )()

                    # Render the template
                    value = template.render(**context)

                    # Try to convert to appropriate type
                    if isinstance(value, str):
                        if value.isdigit():
                            value = int(value)
                        elif value.replace(".", "", 1).isdigit():
                            value = float(value)
                        elif value.lower() in ("true", "false"):
                            value = value.lower() == "true"
                        elif value.startswith("{") and value.endswith("}"):
                            # Try to parse as dict
                            try:
                                import ast

                                value = ast.literal_eval(value)
                            except Exception:
                                pass  # Keep as string if parsing fails
                        elif value.startswith("[") and value.endswith("]"):
                            # Try to parse as list
                            try:
                                import ast

                                value = ast.literal_eval(value)
                            except Exception:
                                pass  # Keep as string if parsing fails

                    outputs[output_name] = value
                else:
                    # Direct value assignment
                    outputs[output_name] = output_expr
            except TemplateError as e:
                # If template rendering fails, set to None
                self.logger.warning(f"Failed to extract output '{output_name}': {e}")
                outputs[output_name] = None
            except Exception as e:
                # Catch any other errors
                self.logger.warning(f"Error extracting output '{output_name}': {e}")
                outputs[output_name] = None

        return outputs

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
        with open(yaml_file, "r") as f:
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
            execution_id, from_checkpoint
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

    async def resume_pipeline(
        self, execution_id: str, resume_strategy: Optional[ResumeStrategy] = None
    ) -> Dict[str, Any]:
        """
        Resume a failed or interrupted pipeline execution.

        Args:
            execution_id: ID of the execution to resume
            resume_strategy: Optional custom resume strategy

        Returns:
            Execution result

        Raises:
            ExecutionError: If resume fails
        """
        # Check if resumable
        if not await self.resume_manager.can_resume(execution_id):
            raise ExecutionError(
                f"No resume checkpoint found for execution {execution_id}"
            )

        # Get resume state
        result = await self.resume_manager.resume_pipeline(
            execution_id, resume_strategy
        )
        if not result:
            raise ExecutionError(
                f"Failed to load resume state for execution {execution_id}"
            )

        pipeline, context = result

        self.logger.info(
            f"Resuming execution {execution_id} for pipeline {pipeline.id}"
        )

        # Execute with resume context
        return await self.execute_pipeline(pipeline, context=context)

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
                requirements.update(
                    {
                        "model_memory": model.requirements.memory_gb * 1024,
                        "model_gpu": model.requirements.requires_gpu,
                        "model_gpu_memory": (model.requirements.gpu_memory_gb or 0)
                        * 1024,
                    }
                )

        return requirements

    async def _select_model_for_task(
        self, task: Task, context: Dict[str, Any]
    ) -> Optional[Any]:
        """Select appropriate model for task execution."""
        # Check if pipeline specifies a default model
        pipeline_metadata = context.get("pipeline_metadata", {})
        if pipeline_metadata.get("model"):
            # Use pipeline-level model specification
            model_spec = pipeline_metadata["model"]
            if isinstance(model_spec, str):
                # Handle format: "gpt-4o-mini" or "openai:gpt-4o-mini"
                if ":" in model_spec:
                    provider, model_name = model_spec.split(":", 1)
                    return self.model_registry.get_model(model_name, provider)
                else:
                    # Try to get model by name, letting registry figure out provider
                    return self.model_registry.get_model(model_spec)
        
        # Check if task specifies model requirements
        if "requires_model" in task.metadata:
            model_req = task.metadata["requires_model"]

            # Handle string format (specific model name)
            if isinstance(model_req, str):
                return self.model_registry.get_model(model_req)

            # Handle dict format (requirements)
            if isinstance(model_req, dict):
                # Map task action to supported task types
                task_type = task.action
                if task.action == "generate_text":
                    task_type = "generate"

                requirements = {
                    "tasks": [task_type],
                    "context_window": len(str(task.parameters).encode())
                    // 4,  # Rough token estimate
                }
                # Merge task-specific requirements
                requirements.update(model_req)

                return await self.model_registry.select_model(requirements)

        # Check if task requires AI capabilities
        if task.action in ["generate", "analyze", "transform", "chat", "generate_text"]:
            # Map task action to supported task types
            task_type = task.action
            if task.action == "generate_text":
                task_type = "generate"

            # Infer requirements based on task action
            requirements = {
                "tasks": [task_type],
                "context_window": len(str(task.parameters).encode())
                // 4,  # Rough token estimate
            }

            # Add default expertise based on action
            if task.action in ["generate_text", "generate"]:
                requirements["expertise"] = ["general"]
            elif task.action == "analyze":
                requirements["expertise"] = ["reasoning", "analysis"]
            elif task.action == "transform":
                requirements["expertise"] = ["general"]

            return await self.model_registry.select_model(requirements)

        return None

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the orchestrator."""
        return {
            "total_executions": len(self.execution_history),
            "successful_executions": len(
                [
                    record
                    for record in self.execution_history
                    if record["status"] == "completed"
                ]
            ),
            "failed_executions": len(
                [
                    record
                    for record in self.execution_history
                    if record["status"] == "failed"
                ]
            ),
            "running_pipelines": len(self.running_pipelines),
            "average_execution_time": (
                sum(
                    record.get("execution_time", 0) for record in self.execution_history
                )
                / len(self.execution_history)
                if self.execution_history
                else 0
            ),
            "resource_utilization": await self.resource_allocator.get_utilization(),
            "error_rate": (
                len(
                    [
                        record
                        for record in self.execution_history
                        if record["status"] == "failed"
                    ]
                )
                / len(self.execution_history)
                if self.execution_history
                else 0
            ),
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
            comp for comp, status in health_status.items() if status == "unhealthy"
        ]

        if unhealthy_components:
            health_status["overall"] = "unhealthy"
        elif any(status == "warning" for status in health_status.values()):
            health_status["overall"] = "warning"
        else:
            health_status["overall"] = "healthy"

        return health_status

    async def execute_pipeline_from_dict(
        self,
        pipeline_dict: Dict[str, Any],
        inputs: Optional[Dict[str, Any]] = None,
        context: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute pipeline from dictionary definition.

        This method is used by PipelineExecutorTool for recursive execution.

        Args:
            pipeline_dict: Pipeline definition as dictionary
            inputs: Input parameters for the pipeline
            context: Recursion context or other context data
            **kwargs: Additional execution parameters

        Returns:
            Execution results
        """
        # Import here to avoid circular import
        import yaml

        # Convert dict to YAML for compilation
        yaml_content = yaml.dump(pipeline_dict)

        # Create compilation context with inputs
        compile_context = inputs or {}

        # Store recursion context if provided
        recursion_context = None
        if context and hasattr(context, "shared_state"):
            # This is a RecursionContext object
            recursion_context = context
            # Remove it from kwargs since execute_pipeline doesn't accept it
            kwargs.pop("recursion_context", None)

        # Compile and execute
        pipeline = await self.yaml_compiler.compile(yaml_content, compile_context)

        # Execute with inputs merged into pipeline context
        if inputs:
            pipeline.context.update(inputs)

        # Store recursion context in pipeline metadata if provided
        if recursion_context:
            pipeline.metadata = pipeline.metadata or {}
            pipeline.metadata["recursion_context"] = recursion_context

        # Execute pipeline
        result = await self.execute_pipeline(pipeline, **kwargs)

        # Return with outputs key for consistency
        if isinstance(result, dict) and "outputs" in result:
            return result
        else:
            return {"outputs": result, "steps_executed": len(pipeline.tasks)}

    async def shutdown(self) -> None:
        """Shutdown orchestrator and clean up resources."""
        # Wait for running pipelines to complete
        if self.running_pipelines:
            await asyncio.sleep(1)  # Give some time for cleanup

        # Shutdown components (only if they have shutdown methods)
        if hasattr(self.resource_allocator, "shutdown"):
            await self.resource_allocator.shutdown()
        elif hasattr(self.resource_allocator, "cleanup"):
            await self.resource_allocator.cleanup()

        if hasattr(self.parallel_executor, "shutdown"):
            if asyncio.iscoroutinefunction(self.parallel_executor.shutdown):
                await self.parallel_executor.shutdown()
            else:
                self.parallel_executor.shutdown()

        if hasattr(self.state_manager, "shutdown"):
            await self.state_manager.shutdown()

        # Clear state
        self.running_pipelines.clear()
        self.execution_history.clear()

    def __repr__(self) -> str:
        """String representation of orchestrator."""
        return f"Orchestrator(running_pipelines={len(self.running_pipelines)})"
