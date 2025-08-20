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
from .core.template_manager import TemplateManager
from .executor.parallel_executor import ParallelExecutor
from .models.model_registry import ModelRegistry
from .models.registry_singleton import get_model_registry
from .state.state_manager import StateManager
from .state.langgraph_state_manager import LangGraphGlobalContextManager
from .state.legacy_compatibility import LegacyStateManagerAdapter
from .core.exceptions import PipelineExecutionError
from .runtime import RuntimeResolutionIntegration

# Import checkpointing components for Issue #205
try:
    from .checkpointing.durable_executor import DurableExecutionManager, ExecutionRecoveryStrategy
    CHECKPOINTING_AVAILABLE = True
except ImportError:
    CHECKPOINTING_AVAILABLE = False

# Use PipelineExecutionError instead of custom ExecutionError
ExecutionError = PipelineExecutionError


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
        template_manager: Optional[TemplateManager] = None,
        max_concurrent_tasks: int = 10,
        debug_templates: bool = False,
        use_langgraph_state: bool = False,
        langgraph_storage_type: str = "memory",
        langgraph_database_url: Optional[str] = None,
        # Issue #205: Automatic Checkpointing Parameters
        use_automatic_checkpointing: bool = False,
        checkpoint_frequency: str = "every_step",
        max_checkpoint_overhead_ms: float = 100.0,
        recovery_strategy: Optional[ExecutionRecoveryStrategy] = None,
        max_recovery_attempts: int = 3,
    ) -> None:
        """
        Initialize orchestrator.

        Args:
            model_registry: Model registry for model selection
            control_system: Control system for task execution
            state_manager: State manager for checkpointing (legacy)
            yaml_compiler: YAML compiler for pipeline parsing
            error_handler: Error handler for fault tolerance
            resource_allocator: Resource allocator for task scheduling
            parallel_executor: Parallel executor for concurrent execution
            template_manager: Template manager for deferred rendering
            max_concurrent_tasks: Maximum concurrent tasks
            debug_templates: Enable debug mode for template rendering
            use_langgraph_state: Use new LangGraph-based state management
            langgraph_storage_type: Storage backend for LangGraph ("memory", "sqlite", "postgres")
            langgraph_database_url: Database URL for persistent storage backends
            use_automatic_checkpointing: Enable automatic step-level checkpointing (Issue #205)
            checkpoint_frequency: Frequency of checkpointing ("every_step", "every_n_steps")
            max_checkpoint_overhead_ms: Maximum allowed checkpoint creation time
            recovery_strategy: Strategy for execution recovery on failures
            max_recovery_attempts: Maximum number of recovery attempts
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
        
        # Initialize state management - either LangGraph or legacy
        self.use_langgraph_state = use_langgraph_state
        if use_langgraph_state:
            if state_manager is not None:
                raise ValueError("Cannot specify both use_langgraph_state=True and state_manager")
            
            # Create LangGraph state manager
            self.langgraph_state_manager = LangGraphGlobalContextManager(
                storage_type=langgraph_storage_type,
                database_url=langgraph_database_url
            )
            
            # Create legacy compatibility adapter
            self.state_manager = LegacyStateManagerAdapter(self.langgraph_state_manager)
            
            logging.getLogger(__name__).info(f"Initialized LangGraph state management with {langgraph_storage_type} backend")
        else:
            self.state_manager = state_manager or StateManager()
            self.langgraph_state_manager = None

        # Initialize template manager
        self.template_manager = template_manager or TemplateManager(debug_mode=debug_templates)
        
        # No default models - must be explicitly initialized
        # Use ControlFlowCompiler to handle for_each, while, and conditionals
        if yaml_compiler is None:
            from .compiler.control_flow_compiler import ControlFlowCompiler
            self.yaml_compiler = ControlFlowCompiler(
                model_registry=self.model_registry
            )
        else:
            self.yaml_compiler = yaml_compiler
        self.error_handler = error_handler or ErrorHandler()
        self.resource_allocator = resource_allocator or ResourceAllocator()
        self.parallel_executor = parallel_executor or ParallelExecutor()
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Initialize control flow handlers for runtime expansion
        from .control_flow import WhileLoopHandler, DynamicFlowHandler
        from .control_flow.auto_resolver import ControlFlowAutoResolver
        self.control_flow_resolver = ControlFlowAutoResolver(self.model_registry)
        self.while_loop_handler = WhileLoopHandler(self.control_flow_resolver)
        self.dynamic_flow_handler = DynamicFlowHandler(self.control_flow_resolver)

        # Execution state
        self.running_pipelines: Dict[str, Pipeline] = (
            {}
        )  # Keep for backward compatibility
        self.execution_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.execution_history: List[Dict[str, Any]] = (
            []
        )  # Keep for backward compatibility
        
        # Runtime resolution system (Issue #211)
        self.runtime_resolution = None  # Will be initialized per pipeline

        # New status tracker and resume manager
        self.status_tracker = PipelineStatusTracker()
        # Pass LangGraph manager to resume manager if available
        langgraph_manager = getattr(self.state_manager, 'langgraph_manager', None)
        if hasattr(self.state_manager, 'backend_manager'):
            langgraph_manager = self.state_manager.backend_manager
        
        self.resume_manager = PipelineResumeManager(
            self.state_manager, 
            langgraph_manager=langgraph_manager
        )

        # Issue #205: Initialize automatic checkpointing
        self.use_automatic_checkpointing = use_automatic_checkpointing and CHECKPOINTING_AVAILABLE
        self.checkpoint_frequency = checkpoint_frequency
        self.max_checkpoint_overhead_ms = max_checkpoint_overhead_ms
        self.recovery_strategy = recovery_strategy
        self.max_recovery_attempts = max_recovery_attempts
        
        # Initialize durable executor if checkpointing is enabled
        if self.use_automatic_checkpointing:
            if not self.use_langgraph_state:
                self.logger.warning("Automatic checkpointing requires LangGraph state management. Disabling automatic checkpointing.")
                self.use_automatic_checkpointing = False
            elif not CHECKPOINTING_AVAILABLE:
                self.logger.warning("Checkpointing components not available. Disabling automatic checkpointing.")
                self.use_automatic_checkpointing = False
            else:
                self.durable_executor = DurableExecutionManager(
                    langgraph_manager=self.langgraph_state_manager,
                    default_recovery_strategy=recovery_strategy or ExecutionRecoveryStrategy.RESUME_FROM_LAST_CHECKPOINT,
                    max_recovery_attempts=max_recovery_attempts,
                    recovery_delay_seconds=1.0
                )
                self.logger.info(f"Automatic checkpointing enabled with {checkpoint_frequency} frequency")
        else:
            self.durable_executor = None

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

        # Issue #205: Check if automatic checkpointing should be used
        if self.use_automatic_checkpointing and checkpoint_enabled:
            return await self._execute_pipeline_with_checkpointing(
                pipeline, execution_id, max_retries
            )

        # Create execution context early to avoid UnboundLocalError
        context = {
            "pipeline_id": pipeline.id,
            "pipeline_metadata": pipeline.metadata,  # Include pipeline metadata for model selection
            "pipeline_context": pipeline.context,  # Include pipeline context with inputs
            "execution_id": execution_id,
            "checkpoint_enabled": checkpoint_enabled,
            "max_retries": max_retries,
            "start_time": time.time(),
        }
        
        # Also merge pipeline context directly into execution context for backward compatibility
        context.update(pipeline.context)
        
        # Initialize runtime resolution system (Issue #211)
        self.runtime_resolution = RuntimeResolutionIntegration(pipeline.id)
        self.runtime_resolution.register_pipeline_context(pipeline.context)
        self.logger.info(f"Initialized runtime resolution for pipeline {pipeline.id}")
        
        # Initialize template manager context with pipeline parameters and execution metadata
        self.template_manager.clear_context()
        self.template_manager.register_context("pipeline_id", pipeline.id)
        self.template_manager.register_context("execution_id", execution_id)
        
        # Add execution metadata
        from datetime import datetime
        execution_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.template_manager.register_context("execution", {
            "timestamp": execution_timestamp,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S")
        })
        
        # Register all pipeline context (including inputs)
        for key, value in pipeline.context.items():
            self.template_manager.register_context(key, value)
            
        # Also register 'inputs' as a separate object for backward compatibility
        if pipeline.context:
            self.template_manager.register_context("inputs", pipeline.context)

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

    async def _execute_pipeline_with_checkpointing(
        self, 
        pipeline: Pipeline, 
        execution_id: str, 
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Execute pipeline with automatic checkpointing (Issue #205).
        
        This method uses the DurableExecutionManager to provide automatic
        step-level checkpointing, recovery, and durable execution.
        
        Args:
            pipeline: Pipeline to execute
            execution_id: Execution identifier
            max_retries: Maximum number of retries for failed tasks
            
        Returns:
            Execution results with checkpoint information
            
        Raises:
            ExecutionError: If execution fails
        """
        if not self.durable_executor:
            raise ExecutionError("Automatic checkpointing not available")
            
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting checkpointed execution: {execution_id} for pipeline: {pipeline.id}")
            
            # Track running pipeline
            self.running_pipelines[execution_id] = pipeline
            
            # Prepare execution configuration
            config = {
                "execution_id": execution_id,
                "configurable": {
                    "thread_id": f"thread_{execution_id}",
                },
                "inputs": pipeline.context.copy(),
                "checkpoint_frequency": self.checkpoint_frequency,
                "max_checkpoint_overhead_ms": self.max_checkpoint_overhead_ms,
                "user_id": pipeline.context.get("user_id"),
                "session_id": pipeline.context.get("session_id")
            }
            
            # Execute with automatic checkpointing
            execution_result = await self.durable_executor.execute_pipeline_durably(
                pipeline=pipeline,
                config=config,
                recovery_strategy=self.recovery_strategy
            )
            
            # Convert execution result to legacy format
            execution_time = time.time() - start_time
            
            if execution_result.status.value == "completed":
                success_results = {
                    "status": "success",
                    "execution_id": execution_id,
                    "pipeline_id": pipeline.id,
                    "execution_time": execution_time,
                    "checkpoint_count": execution_result.checkpoint_count,
                    "recovery_count": execution_result.recovery_count,
                    "results": {},
                    "metadata": {
                        "durable_execution": True,
                        "automatic_checkpointing": True,
                        "checkpoint_frequency": self.checkpoint_frequency
                    }
                }
                
                # Extract results from final state if available
                if execution_result.final_state:
                    intermediate_results = execution_result.final_state.get("intermediate_results", {})
                    success_results["results"] = intermediate_results
                    
                    # Add execution metadata
                    execution_metadata = execution_result.final_state.get("execution_metadata", {})
                    success_results["metadata"]["completed_steps"] = execution_metadata.get("completed_steps", [])
                    success_results["metadata"]["checkpoints"] = execution_metadata.get("checkpoints", [])
                
                # Record successful execution
                execution_record = {
                    "execution_id": execution_id,
                    "pipeline_id": pipeline.id,
                    "status": "success",
                    "start_time": start_time,
                    "end_time": time.time(),
                    "execution_time": execution_time,
                    "checkpoint_count": execution_result.checkpoint_count,
                    "recovery_count": execution_result.recovery_count
                }
                self.execution_history.append(execution_record)
                
                self.logger.info(
                    f"Checkpointed execution completed: {execution_id} in {execution_time:.2f}s "
                    f"with {execution_result.checkpoint_count} checkpoints and {execution_result.recovery_count} recoveries"
                )
                
                return success_results
                
            else:
                # Execution failed
                error_msg = f"Checkpointed execution failed: {execution_id}"
                if execution_result.error_info:
                    error_msg += f" - {execution_result.error_info.get('error', 'Unknown error')}"
                
                execution_record = {
                    "execution_id": execution_id,
                    "pipeline_id": pipeline.id,
                    "status": "failed",
                    "start_time": start_time,
                    "end_time": time.time(),
                    "execution_time": time.time() - start_time,
                    "error": error_msg,
                    "checkpoint_count": execution_result.checkpoint_count,
                    "recovery_count": execution_result.recovery_count
                }
                self.execution_history.append(execution_record)
                
                raise ExecutionError(error_msg)
                
        except Exception as e:
            self.logger.error(f"Checkpointed execution failed: {execution_id} - {e}")
            
            # Record failed execution
            execution_record = {
                "execution_id": execution_id,
                "pipeline_id": pipeline.id,
                "status": "failed",
                "start_time": start_time,
                "end_time": time.time(),
                "execution_time": time.time() - start_time,
                "error": str(e),
                "checkpoint_count": 0,
                "recovery_count": 0
            }
            self.execution_history.append(execution_record)
            
            raise ExecutionError(f"Checkpointed pipeline execution failed: {e}") from e
            
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
        import time
        start_time = time.time()
        
        # Pre-render templates in task parameters before execution
        if task.parameters and self.template_manager:
            rendered_params = {}
            # Build comprehensive context for template rendering
            render_context = {
                **context.get("pipeline_params", {}),  # Pipeline parameters
                **context.get("previous_results", {}),  # Previous step results
                "$item": context.get("$item"),         # Loop item if in loop
                "$index": context.get("$index"),       # Loop index if in loop
            }
            
            for key, value in task.parameters.items():
                if isinstance(value, str) and ("{{" in value or "{%" in value):
                    try:
                        # Render the template
                        rendered_value = self.template_manager.render(
                            value,
                            additional_context=render_context
                        )
                        rendered_params[key] = rendered_value
                        
                        # Log prompt rendering for debugging
                        if key == "prompt":
                            self.logger.debug(f"Pre-rendered prompt: '{value[:100]}...' -> '{rendered_value[:100]}...'")
                    except Exception as e:
                        self.logger.warning(f"Failed to pre-render template for {key}: {e}")
                        rendered_params[key] = value
                else:
                    rendered_params[key] = value
            
            # Update task parameters with rendered values
            task.parameters = rendered_params
        
        # Execute task using control system
        try:
            result = await self.control_system.execute_task(task, context)
            execution_time = time.time() - start_time
            
            # Enhanced result tracking for LangGraph
            if self.use_langgraph_state:
                enhanced_result = {
                    "result": result,
                    "task_metadata": {
                        "task_id": task.id,
                        "task_type": task.action,
                        "execution_time": execution_time,
                        "status": "completed",
                        "model_used": task.parameters.get('model', None),
                        "dependencies": task.dependencies,
                        "timestamp": time.time()
                    }
                }
                
                # Try to capture more detailed execution information
                if hasattr(self.control_system, 'last_model_response'):
                    enhanced_result["model_response"] = self.control_system.last_model_response
                
                return enhanced_result
            else:
                return result
                
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Enhanced error tracking for LangGraph
            if self.use_langgraph_state:
                error_result = {
                    "error": str(e),
                    "task_metadata": {
                        "task_id": task.id,
                        "task_type": task.action,
                        "execution_time": execution_time,
                        "status": "failed",
                        "error_type": type(e).__name__,
                        "timestamp": time.time()
                    }
                }
                raise ExecutionError(f"Task {task.id} failed", error_result)
            else:
                raise

    async def _execute_pipeline_internal(
        self,
        pipeline: Pipeline,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Internal pipeline execution logic."""
        results = {}

        # Get execution levels (groups of tasks that can run in parallel)
        execution_levels = pipeline.get_execution_levels()

        # Track completed tasks for while loop checking
        completed_tasks = set()
        
        # Debug: Print all tasks and their metadata
        self.logger.info("=== Pipeline tasks at start ===")
        for task_id, task in pipeline.tasks.items():
            self.logger.info(f"Task {task_id}: action={task.action}, is_while_loop={task.metadata.get('is_while_loop', False)}, has_steps={'steps' in task.metadata}")
        
        # Execute tasks level by level
        while True:
            # Get execution levels (groups of tasks that can run in parallel)
            execution_levels = pipeline.get_execution_levels()
            self.logger.debug(f"Raw execution levels: {execution_levels}")
            
            # Filter out completed tasks and tasks that depend on pending while loops
            remaining_levels = []
            pending_while_loops = set()
            
            # First, identify all pending while loops
            for task_id, task in pipeline.tasks.items():
                if (task.status == TaskStatus.PENDING 
                    and task.metadata.get("is_while_loop", False)):
                    pending_while_loops.add(task_id)
            
            for level in execution_levels:
                remaining_tasks = []
                for task_id in level:
                    task = pipeline.get_task(task_id)
                    if not task or task_id in completed_tasks or task.status != TaskStatus.PENDING:
                        continue
                        
                    # Check if this task depends on any pending while loops
                    depends_on_while_loop = any(
                        dep in pending_while_loops for dep in task.dependencies
                    )
                    
                    if not depends_on_while_loop:
                        remaining_tasks.append(task_id)
                    else:
                        self.logger.debug(f"Task {task_id} depends on pending while loop, skipping")
                        
                if remaining_tasks:
                    remaining_levels.append(remaining_tasks)
                    
            self.logger.debug(f"Remaining levels after filtering: {remaining_levels}")
            self.logger.debug(f"Completed tasks: {completed_tasks}")
            
            if not remaining_levels:
                # Check for while loops that need expansion
                self.logger.info("No remaining levels, checking for while loops to expand...")
                self.logger.info(f"Current results: {list(results.keys())}")
                self.logger.info(f"Current completed tasks: {list(completed_tasks)}")
                if await self._expand_while_loops(pipeline, context, results):
                    self.logger.info("While loops were expanded, recalculating levels")
                    continue  # New tasks added, recalculate levels
                else:
                    self.logger.info("No while loops to expand, finishing execution")
                    break  # No more tasks to execute
            
            # Find the first level with executable tasks
            executable_tasks = []
            level_index = 0
            
            for idx, level in enumerate(remaining_levels):
                level_index = idx
                self.logger.info(f"Level {level_index} contains tasks: {level}")
                
                # Skip while loop placeholder tasks
                executable_tasks = []
                for task_id in level:
                    task = pipeline.get_task(task_id)
                    if task and not (hasattr(task, "metadata") and task.metadata.get("is_while_loop")):
                        executable_tasks.append(task_id)
                    elif task and hasattr(task, "metadata") and task.metadata.get("is_while_loop"):
                        self.logger.info(f"Skipping while loop placeholder: {task_id}")
                
                # If we found executable tasks, use this level
                if executable_tasks:
                    break
                    
            context["current_level"] = level_index
            
            if executable_tasks:
                # Execute tasks in parallel within the level
                self.logger.warning(f"ORCHESTRATOR: Executing level {level_index} with tasks: {executable_tasks}")
                self.logger.warning(f"ORCHESTRATOR: Accumulated results so far: {list(results.keys())}")
                level_results = await self._execute_level(pipeline, executable_tasks, context, results)

                # Check for failures
                failed_tasks = [
                    task_id
                    for task_id in executable_tasks
                    if pipeline.get_task(task_id)
                    and pipeline.get_task(task_id).status == TaskStatus.FAILED
                ]

                if failed_tasks:
                    # Handle failures based on policy
                    await self._handle_task_failures(pipeline, failed_tasks, context)

                # Update results with level results
                results.update(level_results)
                
                # Mark tasks as completed
                completed_tasks.update(executable_tasks)
                
                # Process goto jumps from completed tasks
                goto_target = await self._process_goto_jumps(
                    pipeline, context, results, set(executable_tasks)
                )
                
                if goto_target:
                    # A goto was triggered, recalculate execution levels
                    self.logger.info(f"Goto triggered, jumping to {goto_target}")
                    # Mark the goto source task as having jumped
                    # This prevents infinite loops if the same task is reached again
                    for task_id in executable_tasks:
                        task = pipeline.get_task(task_id)
                        if task and task.metadata.get("goto"):
                            task.metadata["goto_executed"] = True
                
                # IMPORTANT: Update context with step results for template rendering
                # This ensures that subsequent steps can access previous step results
                # via template variables like {{ step_id.result }}
                context["previous_results"] = results.copy()
                
                # Register all results with template manager for deferred rendering
                self.template_manager.register_all_results(results)
                
                # Resolve dynamic dependencies for pending tasks
                await self._resolve_dynamic_dependencies(pipeline, context, results)

                # Save checkpoint after each level
                if context.get("checkpoint_enabled", False):
                    await self.state_manager.save_checkpoint(
                        context["execution_id"], self._get_pipeline_state(pipeline), context
                    )
            else:
                # No executable tasks found in any level
                # Check if we need to expand while loops
                self.logger.info("No executable tasks found in any level")
                
                # Try to expand while loops
                if await self._expand_while_loops(pipeline, context, results):
                    # Tasks were added, continue to recalculate levels
                    continue
                    
                # If no while loops expanded and we have no executable tasks,
                # we might be done or stuck
                self.logger.info("No tasks to execute and no while loops to expand")
                
                # Check if there are any pending tasks at all
                pending_tasks = [
                    task_id for task_id, task in pipeline.tasks.items()
                    if task.status == TaskStatus.PENDING
                ]
                
                if pending_tasks:
                    # We have pending tasks but can't execute them
                    # This might be due to while loops that need to complete
                    self.logger.warning(f"Have {len(pending_tasks)} pending tasks but none are executable: {pending_tasks}")
                    
                    # Check if we have any while loops that might need to be marked complete
                    pending_while_loop_tasks = [
                        t for t in pending_tasks 
                        if pipeline.get_task(t).metadata.get("is_while_loop", False)
                    ]
                    
                    if pending_while_loop_tasks:
                        # Try to check while loops for completion
                        made_progress = False
                        for task_id in pending_while_loop_tasks:
                            task = pipeline.get_task(task_id)
                            if task:
                                loop_id = task.id
                                # Check if this while loop should be marked as complete
                                current_iteration = 0
                                for t_id in results:
                                    if t_id.startswith(f"{loop_id}_") and "_" in t_id[len(loop_id)+1:]:
                                        parts = t_id[len(loop_id)+1:].split("_", 1)
                                        if parts[0].isdigit():
                                            current_iteration = max(current_iteration, int(parts[0]) + 1)
                                
                                max_iterations = task.metadata.get("max_iterations", 10)
                                if isinstance(max_iterations, str):
                                    max_iterations = int(max_iterations)
                                    
                                # Check condition one more time
                                condition = task.metadata.get("while_condition", "false")
                                should_continue = await self.while_loop_handler.should_continue(
                                    loop_id, condition, context, results, current_iteration, max_iterations
                                )
                                
                                self.logger.info(f"While loop {loop_id}: should_continue={should_continue}, current_iteration={current_iteration}, max_iterations={max_iterations}")
                                
                                if not should_continue and task.status == TaskStatus.PENDING:
                                    task.complete({"iterations": current_iteration, "status": "completed"})
                                    self.logger.info(f"Marking while loop {loop_id} as completed")
                                    made_progress = True
                                    # Continue to recalculate levels
                                    break
                        
                        if made_progress:
                            # We marked a while loop as complete, continue execution
                            continue
                    else:
                        # Have non-while-loop pending tasks that can't execute
                        # This is likely a dependency issue
                        self.logger.error("Deadlock detected - pending tasks with unmet dependencies")
                        break
                else:
                    # No pending tasks, we're done
                    self.logger.info("No pending tasks remaining")
                    break

        return results
    
    async def _process_goto_jumps(
        self, 
        pipeline: Pipeline, 
        context: Dict[str, Any], 
        results: Dict[str, Any],
        completed_tasks: Set[str]
    ) -> Optional[str]:
        """Process goto directives from completed tasks.
        
        Args:
            pipeline: Pipeline being executed
            context: Execution context
            results: Results from completed steps
            completed_tasks: Set of completed task IDs
            
        Returns:
            Task ID to jump to, or None
        """
        # Check recently completed tasks for goto directives
        for task_id in completed_tasks:
            task = pipeline.get_task(task_id)
            if not task or task.status != TaskStatus.COMPLETED:
                continue
                
            # Check for goto in task metadata
            goto_target = task.metadata.get("goto")
            if goto_target:
                # Process goto with dynamic flow handler
                resolved_target = await self.dynamic_flow_handler.process_goto(
                    task, context, results, pipeline.tasks
                )
                
                if resolved_target and resolved_target in pipeline.tasks:
                    self.logger.info(f"Task {task_id} jumping to {resolved_target}")
                    # Mark intermediate tasks as skipped
                    self._skip_tasks_between(pipeline, task_id, resolved_target)
                    return resolved_target
                    
        return None
    
    def _skip_tasks_between(self, pipeline: Pipeline, from_task: str, to_task: str) -> None:
        """Skip tasks between a goto source and target.
        
        Args:
            pipeline: Pipeline containing tasks
            from_task: Source task ID
            to_task: Target task ID
        """
        # Get execution order
        levels = pipeline.get_execution_levels()
        task_order = []
        for level in levels:
            task_order.extend(level)
            
        # Find positions
        try:
            from_idx = task_order.index(from_task)
            to_idx = task_order.index(to_task)
        except ValueError:
            # Tasks not found in order
            return
            
        # Skip tasks between source and target
        if to_idx > from_idx:
            for i in range(from_idx + 1, to_idx):
                task_id = task_order[i]
                task = pipeline.get_task(task_id)
                if task and task.status == TaskStatus.PENDING:
                    task.skip(f"Skipped due to goto from {from_task} to {to_task}")
                    self.logger.info(f"Skipped task {task_id} due to goto")
    
    async def _resolve_dynamic_dependencies(
        self, 
        pipeline: Pipeline, 
        context: Dict[str, Any], 
        results: Dict[str, Any]
    ) -> None:
        """Resolve dynamic dependencies for pending tasks.
        
        Args:
            pipeline: Pipeline being executed
            context: Execution context
            results: Results from completed steps
        """
        for task in pipeline.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
                
            # Check for dynamic dependencies
            if task.metadata.get("dynamic_dependencies"):
                resolved_deps = await self.dynamic_flow_handler.resolve_dynamic_dependencies(
                    task, context, results, pipeline.tasks
                )
                
                # Update task dependencies
                task.dependencies = resolved_deps
                self.logger.info(f"Resolved dynamic dependencies for {task.id}: {resolved_deps}")
    
    async def _expand_while_loops(
        self, pipeline: Pipeline, context: Dict[str, Any], step_results: Dict[str, Any]
    ) -> bool:
        """Expand while loops that need another iteration.
        
        Args:
            pipeline: Pipeline being executed
            context: Execution context
            step_results: Results from completed steps
            
        Returns:
            True if any loops were expanded
        """
        expanded = False
        self.logger.info(f"_expand_while_loops called. Total tasks: {len(pipeline.tasks)}")
        
        for task in list(pipeline.tasks.values()):
            # Check if task is a while loop placeholder
            if (
                hasattr(task, "metadata")
                and task.metadata.get("is_while_loop")
                and task.status == TaskStatus.PENDING
            ):
                loop_id = task.id
                max_iterations = task.metadata.get("max_iterations", 10)
                self.logger.info(f"While loop {loop_id}: max_iterations from metadata = {max_iterations} (type: {type(max_iterations)})")
                if isinstance(max_iterations, str):
                    max_iterations = int(max_iterations)
                    self.logger.info(f"While loop {loop_id}: converted max_iterations to int: {max_iterations}")
                condition = task.metadata.get("while_condition", "false")
                
                # Determine current iteration
                current_iteration = 0
                for t_id in step_results:
                    if t_id.startswith(f"{loop_id}_") and "_" in t_id[len(loop_id)+1:]:
                        # Extract iteration number
                        parts = t_id[len(loop_id)+1:].split("_", 1)
                        if parts[0].isdigit():
                            current_iteration = max(current_iteration, int(parts[0]) + 1)
                
                self.logger.info(f"While loop {loop_id}: current iteration = {current_iteration}, max = {max_iterations}")
                self.logger.info(f"Condition to check: {condition}")
                
                # Check if we've already created tasks for this iteration
                next_iter_task_id = f"{loop_id}_{current_iteration}_result"
                if next_iter_task_id in pipeline.tasks:
                    self.logger.info(f"Tasks for iteration {current_iteration} already exist, skipping")
                    continue
                
                # Create a clean context for while loop handler
                loop_context = context.copy()
                # Ensure template manager is available for condition rendering
                if "template_manager" not in loop_context and "_template_manager" not in loop_context:
                    loop_context["_template_manager"] = self.template_manager
                
                # Check if should continue
                should_continue = await self.while_loop_handler.should_continue(
                    loop_id,
                    condition,
                    loop_context,
                    step_results,
                    current_iteration,
                    max_iterations,
                )
                
                # Clean context for task creation (remove non-serializable objects)
                clean_context = {k: v for k, v in context.items() 
                               if k not in ["_template_manager", "template_manager"]}
                
                if should_continue:
                    # Create tasks for next iteration
                    # Get the full task definition including while loop body
                    loop_def = task.to_dict()
                    # Add the while loop body steps from metadata
                    if "steps" in task.metadata:
                        loop_def["steps"] = task.metadata["steps"]
                    
                    self.logger.info(f"Creating iteration {current_iteration} for loop {loop_id}")
                    self.logger.info(f"Loop def keys: {list(loop_def.keys())}")
                    if "steps" in loop_def:
                        self.logger.info(f"Loop has {len(loop_def['steps'])} steps")
                    else:
                        self.logger.warning(f"Loop {loop_id} has no steps!")
                    
                    iteration_tasks = await self.while_loop_handler.create_iteration_tasks(
                        loop_def,
                        current_iteration,
                        clean_context,
                        step_results,
                    )
                    
                    self.logger.info(f"Created {len(iteration_tasks)} tasks for iteration {current_iteration}")
                    
                    # Add tasks to pipeline
                    added_count = 0
                    for iter_task in iteration_tasks:
                        # Check if task already exists to avoid duplicates
                        if iter_task.id not in pipeline.tasks:
                            pipeline.add_task(iter_task)
                            added_count += 1
                            self.logger.info(f"Added task: {iter_task.id} (action={iter_task.action})")
                        else:
                            self.logger.debug(f"Task {iter_task.id} already exists, skipping")
                    
                    self.logger.info(f"Added {added_count} new tasks to pipeline")
                    
                    expanded = True
                    self.logger.info(
                        f"Expanded while loop {loop_id} iteration {current_iteration}"
                    )
                else:
                    # Mark loop as completed
                    if task.status == TaskStatus.PENDING:
                        task.complete({"iterations": current_iteration, "status": "completed"})
                        self.logger.info(
                            f"While loop {loop_id} completed after {current_iteration} iterations"
                        )
                        expanded = True  # Mark as expanded so we recalculate levels
        
        self.logger.info(f"_expand_while_loops returning: {expanded}")
        return expanded
    
    async def _prepare_task_for_execution(self, task: Task, context: Dict[str, Any], 
                                          completed_steps: Set[str]) -> Dict[str, Any]:
        """
        Prepare task for execution.
        
        For now, we'll let the control system handle template rendering
        to avoid premature dependency checks.
        
        Args:
            task: Task to prepare
            context: Execution context
            completed_steps: Set of completed step IDs
            
        Returns:
            Task parameters (unrendered)
        """
        # Return original parameters - control system will render templates as needed
        return task.parameters

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

            # Track completed steps from previous results
            completed_steps = set(previous_results.keys())
            
            # Track if any ForEachTasks were expanded (requires level recalculation)
            for_each_expanded = False
            
            for task_id in level_tasks:
                task = pipeline.get_task(task_id)

                # Handle runtime for_each expansion
                from .core.for_each_task import ForEachTask
                if isinstance(task, ForEachTask):
                    # Expand ForEachTask at runtime
                    expanded_task_ids = await self._expand_for_each_task(
                        task, pipeline, context, previous_results
                    )
                    
                    # DON'T add expanded tasks to the current level!
                    # They have dependencies that may not be satisfied yet.
                    # They'll be picked up in the next level calculation.
                    # for expanded_id in expanded_task_ids:
                    #     if expanded_id not in level_tasks:
                    #         level_tasks.append(expanded_id)
                    
                    # Mark the ForEachTask as completed
                    results[task_id] = task.result
                    
                    # Signal that we need to recalculate levels
                    for_each_expanded = True
                    continue
                
                # Skip tasks that are already marked as skipped
                if task.status == TaskStatus.SKIPPED:
                    results[task_id] = {"status": "skipped"}
                    # Register skipped task with None value in template manager
                    # This allows Jinja conditionals like {% if enhance_text.result %} to work
                    self.template_manager.register_context(task_id, None)
                    continue
                
                # For conditional tasks, we'll check the condition after dependencies are satisfied
                # This happens inside _execute_task_with_resources
                
                # Prepare task by rendering templates JIT
                try:
                    rendered_params = await self._prepare_task_for_execution(
                        task, context, completed_steps
                    )
                except Exception as e:
                    # Handle template rendering errors
                    self.logger.error(f"Failed to prepare task {task_id}: {e}")
                    task.fail(e)
                    results[task_id] = {"error": str(e)}
                    continue

                # Build comprehensive task context with all available data
                task_context = {
                    **context,
                    "task_id": task_id,
                    "previous_results": previous_results,
                    "pipeline_params": pipeline.context,  # Add pipeline context (contains parameters) for template rendering
                    "resource_allocation": resource_allocations.get(task_id, {"cpu": 1, "memory": 512}),  # Default for dynamic tasks
                    "template_manager": self.template_manager,
                    "_template_manager": self.template_manager,  # Also add with underscore for compatibility
                }
                
                # Flatten previous_results for direct template access
                # This allows templates to use {{ step_id }} instead of {{ previous_results.step_id }}
                for step_id, result in previous_results.items():
                    if step_id not in task_context:
                        task_context[step_id] = result
                
                # Add execution metadata for templates
                from datetime import datetime
                task_context["execution"] = {
                    "timestamp": datetime.now().isoformat(),
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "time": datetime.now().strftime("%H:%M:%S"),
                }
                
                # Ensure pipeline parameters are directly accessible
                if isinstance(pipeline.context, dict):
                    for key, value in pipeline.context.items():
                        if key not in task_context and key not in ["execution", "task_id", "previous_results"]:
                            task_context[key] = value
                
                # Add loop metadata to context if this is a loop iteration task
                if task.metadata.get("is_while_loop_child"):
                    loop_id = task.metadata.get("loop_id")
                    iteration = task.metadata.get("loop_iteration", 0)
                    
                    # Register loop metadata with template manager
                    if loop_id:
                        self.template_manager.register_context(loop_id, {
                            "iteration": iteration,
                            "id": loop_id
                        })
                        # Also register under the actual loop name for templates
                        self.template_manager.register_context("guessing_loop", {
                            "iteration": iteration,
                            "id": loop_id
                        })
                
                # Debug: Log task metadata to see what's available
                self.logger.debug(f"Task {task_id} metadata: {task.metadata}")
                
                # If this is a ForEachTask child with loop_context, add loop variables to task_context
                if task.metadata.get("is_for_each_child") and "loop_context" in task.metadata:
                    loop_ctx = task.metadata["loop_context"]
                    # Add ALL loop context variables to the execution context
                    # This ensures step results from outside the loop are available
                    for key, value in loop_ctx.items():
                        if key not in task_context and not key.startswith("_"):
                            task_context[key] = value
                            if key in ["item", "index", "$item", "$index"]:
                                self.logger.info(f"Added ForEachTask loop variable '{key}' = {value} to execution context")
                
                # Add loop context mapping for for_each loops
                # Check for both old and new metadata formats
                is_loop_task = (task.metadata.get("is_for_each_child") or 
                               task.metadata.get("loop_context") or
                               task.metadata.get("loop_id"))
                
                if is_loop_task:
                    # Try to get loop info from different metadata formats
                    if task.metadata.get("is_for_each_child"):
                        # Old format
                        parent_id = task.metadata.get("parent_for_each")
                        iteration_idx = task.metadata.get("iteration_index", 0)
                        loop_step_id = task.metadata.get("loop_step_id")
                    elif task.metadata.get("loop_context"):
                        # New format with loop_context
                        loop_ctx = task.metadata["loop_context"]
                        parent_id = loop_ctx.get("loop_id") or task.metadata.get("loop_id")
                        iteration_idx = loop_ctx.get("index", 0)
                        # Extract step name from task ID
                        # Format: parent_id_index_stepname (e.g., process_items_0_save -> save)
                        # But stepname might have underscores (e.g., process_items_0_generate_unique -> generate_unique)
                        if parent_id and '_' in task_id:
                            # Remove parent_id and index to get step name
                            prefix = f"{parent_id}_{iteration_idx}_"
                            if task_id.startswith(prefix):
                                loop_step_id = task_id[len(prefix):]
                            else:
                                # Fallback: take everything after second underscore
                                parts = task_id.split('_')
                                if len(parts) > 2:
                                    loop_step_id = '_'.join(parts[2:])
                                else:
                                    loop_step_id = parts[-1] if parts else task_id
                        else:
                            loop_step_id = task_id
                    else:
                        # Fallback to direct metadata
                        parent_id = task.metadata.get("loop_id")
                        iteration_idx = task.metadata.get("loop_index", 0)
                        # Same logic for extracting step name
                        if parent_id and '_' in task_id:
                            prefix = f"{parent_id}_{iteration_idx}_"
                            if task_id.startswith(prefix):
                                loop_step_id = task_id[len(prefix):]
                            else:
                                parts = task_id.split('_')
                                if len(parts) > 2:
                                    loop_step_id = '_'.join(parts[2:])
                                else:
                                    loop_step_id = parts[-1] if parts else task_id
                        else:
                            loop_step_id = task_id
                    
                    self.logger.info(f"Creating loop context mapping for task {task_id}: parent={parent_id}, iteration={iteration_idx}, step={loop_step_id}")
                    self.logger.info(f"  Task metadata: is_for_each_child={task.metadata.get('is_for_each_child')}, parent_for_each={task.metadata.get('parent_for_each')}")
                    
                    # Create a mapping from short names to full task IDs for this iteration
                    loop_context_mapping = {}
                    
                    # Get all tasks from the same loop iteration
                    for other_task_id, other_task in pipeline.tasks.items():
                        # Check if task is from same loop and iteration
                        is_same_loop = False
                        other_iteration = -1
                        other_step_name = None
                        
                        if other_task.metadata.get("parent_for_each") == parent_id:
                            # Old format
                            is_same_loop = True
                            other_iteration = other_task.metadata.get("iteration_index", -1)
                            other_step_name = other_task.metadata.get("loop_step_id")
                            self.logger.debug(f"    Found same-loop task (old format): {other_task_id}, iteration={other_iteration}, step={other_step_name}")
                        elif other_task.metadata.get("loop_context"):
                            # New format with loop_context
                            other_loop_ctx = other_task.metadata["loop_context"]
                            other_loop_id = other_loop_ctx.get("loop_id") or other_task.metadata.get("loop_id")
                            if other_loop_id == parent_id:
                                is_same_loop = True
                                other_iteration = other_loop_ctx.get("index", -1)
                                # Extract step name from task ID (handle underscores in step names)
                                if '_' in other_task_id:
                                    prefix = f"{other_loop_id}_{other_iteration}_"
                                    if other_task_id.startswith(prefix):
                                        other_step_name = other_task_id[len(prefix):]
                                        self.logger.debug(f"    Extracted step name '{other_step_name}' from task ID '{other_task_id}' using prefix '{prefix}'")
                                    else:
                                        parts = other_task_id.split('_')
                                        if len(parts) > 2:
                                            other_step_name = '_'.join(parts[2:])
                                        else:
                                            other_step_name = parts[-1] if parts else other_task_id
                                        self.logger.debug(f"    Extracted step name '{other_step_name}' from task ID '{other_task_id}' using fallback")
                                else:
                                    other_step_name = other_task_id
                        elif other_task.metadata.get("loop_id") == parent_id:
                            # Direct metadata format
                            is_same_loop = True
                            other_iteration = other_task.metadata.get("loop_index", -1)
                            # Extract step name from task ID (handle underscores in step names)
                            if '_' in other_task_id:
                                prefix = f"{parent_id}_{other_iteration}_"
                                if other_task_id.startswith(prefix):
                                    other_step_name = other_task_id[len(prefix):]
                                    self.logger.debug(f"    [Direct] Extracted step name '{other_step_name}' from task ID '{other_task_id}' using prefix '{prefix}'")
                                else:
                                    parts = other_task_id.split('_')
                                    if len(parts) > 2:
                                        other_step_name = '_'.join(parts[2:])
                                    else:
                                        other_step_name = parts[-1] if parts else other_task_id
                                    self.logger.debug(f"    [Direct] Extracted step name '{other_step_name}' from task ID '{other_task_id}' using fallback")
                            else:
                                other_step_name = other_task_id
                        
                        if is_same_loop and other_iteration == iteration_idx and other_step_name:
                            # Map the short step ID to the full task ID
                            loop_context_mapping[other_step_name] = other_task_id
                            self.logger.info(f"  Mapped '{other_step_name}' -> '{other_task_id}'")
                    
                    # Add the mapping to the task context
                    task_context["_loop_context_mapping"] = loop_context_mapping
                    self.logger.info(f"Added loop context mapping with {len(loop_context_mapping)} entries to task context")
                    
                    # Also add loop-specific variables to context
                    if "loop_context" in task.metadata:
                        loop_ctx = task.metadata["loop_context"]
                        self.logger.info(f"Adding {len(loop_ctx)} loop context variables to task context for {task_id}")
                        for key, value in loop_ctx.items():
                            if key not in task_context:
                                task_context[key] = value
                                self.logger.debug(f"  Added loop variable '{key}' = {value}")
                
                # Pre-render templates in task parameters before execution
                if task.parameters and self.template_manager:
                    # First, register ALL previous results with the template manager
                    # This ensures all step results are available for template rendering
                    if "previous_results" in task_context:
                        self.logger.info(f"DEBUG: Registering {len(task_context['previous_results'])} results for task {task_id}")
                        for step_id, result in task_context["previous_results"].items():
                            # Register the result for template access
                            self.template_manager.register_context(step_id, result)
                            self.logger.info(f"DEBUG: Registered {step_id} = {str(result)[:100] if isinstance(result, str) else type(result).__name__}")
                            # If it's a string, make it directly accessible
                            if isinstance(result, str):
                                self.template_manager.register_context(f"{step_id}_str", result)
                            # If it's a dict with 'result' key, also register that
                            elif isinstance(result, dict) and 'result' in result:
                                self.template_manager.register_context(f"{step_id}_result", result['result'])
                    
                    # If we have loop context mapping, also register the loop-specific results with short names
                    if "_loop_context_mapping" in task_context and "previous_results" in task_context:
                        loop_context_mapping = task_context["_loop_context_mapping"]
                        self.logger.debug(f"Registering {len(loop_context_mapping)} loop results with short names for task {task_id}")
                        
                        # Register results from this iteration with their short names
                        for short_name, full_task_id in loop_context_mapping.items():
                            if full_task_id in task_context["previous_results"]:
                                result = task_context["previous_results"][full_task_id]
                                self.template_manager.register_context(short_name, result)
                                self.logger.debug(f"  Registered '{short_name}' = {str(result)[:50] if isinstance(result, str) else 'complex'}")
                    
                    # Also register pipeline parameters
                    if isinstance(pipeline.context, dict):
                        for key, value in pipeline.context.items():
                            if key not in ["previous_results", "_template_manager"]:
                                self.template_manager.register_context(key, value)
                    
                    # Register loop variables if present
                    for loop_var in ["$item", "$index", "item", "index"]:
                        if loop_var in task_context:
                            self.template_manager.register_context(loop_var, task_context[loop_var])
                    
                    rendered_params = {}
                    for key, value in task.parameters.items():
                        if isinstance(value, str) and ("{{" in value or "{%" in value):
                            # This is a template string - render it with full context
                            try:
                                rendered_value = self.template_manager.render(
                                    value,
                                    additional_context=task_context
                                )
                                rendered_params[key] = rendered_value
                                self.logger.debug(f"Pre-rendered {key} for task {task_id}: '{value[:50]}...' -> '{rendered_value[:50]}...'")
                            except Exception as e:
                                self.logger.warning(f"Failed to pre-render template for {key} in task {task_id}: {e}")
                                rendered_params[key] = value
                        else:
                            rendered_params[key] = value
                    task.parameters = rendered_params
                
                execution_tasks.append(
                    self._execute_task_with_resources(task, task_context)
                )
                scheduled_task_ids.append(task_id)
            
            # If we expanded any ForEachTasks, return early to trigger level recalculation
            if for_each_expanded:
                self.logger.info("ForEachTasks were expanded, returning early to recalculate levels")
                return results

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
                        
                        # Register with runtime resolution system (Issue #211)
                        if self.runtime_resolution:
                            self.runtime_resolution.register_task_result(task_id, result)
                        
                        # Register result immediately with template manager so subsequent tasks can use it
                        self.template_manager.register_context(task_id, result)
                        
                        # For loop iteration tasks, also register under the original unprefixed name
                        task = pipeline.get_task(task_id)
                        if task and task.metadata.get("is_while_loop_child"):
                            # Extract original task name from prefixed ID
                            # Format: loop_id_iteration_original_name
                            parts = task_id.split("_", 3)  # Split at most 3 times
                            if len(parts) >= 4:  # loop_id, iteration, original_name...
                                original_name = parts[3]  # Everything after the iteration number
                                self.template_manager.register_context(original_name, result)
                                self.logger.debug(f"Registered while loop task result under original name: {original_name}")
                        
                        # Also handle for_each loop tasks
                        if task and task.metadata.get("is_for_each_child"):
                            # Extract the step name from the task ID
                            # Format: parent_loop_id_iteration_index_step_name
                            parent_id = task.metadata.get("parent_for_each")
                            iteration_idx = task.metadata.get("iteration_index")
                            
                            if parent_id is not None and iteration_idx is not None:
                                # Remove the prefix to get the step name
                                prefix = f"{parent_id}_{iteration_idx}_"
                                if task_id.startswith(prefix):
                                    step_name = task_id[len(prefix):]
                                    # Register under the short step name for template access within the loop
                                    self.template_manager.register_context(step_name, result)
                                    self.logger.info(f"Registered for_each loop task result: {step_name} = {str(result)[:100]}")
                                    
                                    # Also register with common aliases for better template compatibility
                                    if isinstance(result, dict) and 'result' in result:
                                        self.template_manager.register_context(f"{step_name}_result", result['result'])
                        
                        # Additional registration for loop_step_id if present
                        if task and task.metadata.get("loop_step_id"):
                            loop_step_id = task.metadata["loop_step_id"]
                            if loop_step_id and loop_step_id != task_id:
                                self.template_manager.register_context(loop_step_id, result)
                                self.logger.debug(f"Registered loop task under loop_step_id: {loop_step_id}")

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
        # Check if this is a conditional task that needs condition evaluation
        from .control_flow.conditional import ConditionalTask
        if isinstance(task, ConditionalTask) and hasattr(task, 'should_execute'):
            # Check if all dependencies have been satisfied first
            previous_results = context.get("previous_results", {})
            missing_deps = []
            for dep in task.dependencies:
                if dep not in previous_results:
                    missing_deps.append(dep)
            
            if missing_deps:
                # Dependencies not satisfied, can't evaluate condition yet
                self.logger.error(
                    f"Task '{task.id}' has unsatisfied dependencies: {missing_deps}. "
                    f"This should not happen - tasks should only be scheduled after dependencies complete."
                )
                raise ExecutionError(
                    f"Task '{task.id}' scheduled before dependencies {missing_deps} completed"
                )
            
            # Import here to avoid circular dependency
            from .control_flow.auto_resolver import ControlFlowAutoResolver
            resolver = ControlFlowAutoResolver(self.model_registry)
            
            # Register previous results with template manager before condition evaluation
            # This ensures step results are available for condition template rendering
            template_manager = context.get("template_manager") or context.get("_template_manager")
            if template_manager and previous_results:
                self.logger.warning(f"ORCHESTRATOR: Registering {len(previous_results)} previous results with template manager before condition evaluation for task '{task.id}'")
                self.logger.warning(f"ORCHESTRATOR: Previous results keys: {list(previous_results.keys())}")
                template_manager.register_all_results(previous_results)
            else:
                self.logger.warning(f"ORCHESTRATOR: No template manager or previous results for task '{task.id}'. TM: {template_manager is not None}, PR: {len(previous_results) if previous_results else 0}")
            
            # Now we can safely evaluate the condition
            try:
                should_execute = await task.should_execute(
                    context, 
                    previous_results,
                    resolver
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to evaluate condition for task '{task.id}': {e}"
                )
                # If condition evaluation fails, skip the task with error details
                task.skip(f"Condition evaluation failed: {e}")
                return {"status": "skipped", "reason": "condition_error", "error": str(e)}
            
            if not should_execute:
                # Skip this task
                task.skip("Condition evaluated to false")
                return {"status": "skipped", "reason": "condition_false"}
        
        # Mark task as running
        task.start()

        try:
            # Select appropriate model for the task
            model = await self._select_model_for_task(task, context)
            if model:
                context["model"] = model

            # Determine timeout - check task.timeout, then metadata, then pipeline config
            timeout_seconds = None
            if task.timeout:
                timeout_seconds = task.timeout
            elif task.metadata.get("timeout"):
                timeout_seconds = task.metadata.get("timeout")
            elif context.get("pipeline_metadata", {}).get("default_timeout"):
                timeout_seconds = context["pipeline_metadata"]["default_timeout"]

            # Execute task with timeout if specified
            if timeout_seconds:
                try:
                    result = await asyncio.wait_for(
                        self._execute_step(task, context),
                        timeout=timeout_seconds
                    )
                except asyncio.TimeoutError:
                    raise TimeoutError(
                        f"Task '{task.id}' exceeded timeout of {timeout_seconds} seconds"
                    )
            else:
                # Execute without timeout
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

    async def _expand_for_each_task(
        self,
        for_each_task,  # ForEachTask type
        pipeline: Pipeline,
        context: Dict[str, Any],
        step_results: Dict[str, Any]
    ) -> List[str]:
        """Expand ForEachTask at runtime with actual step results.
        
        Args:
            for_each_task: The ForEachTask to expand
            pipeline: The pipeline to modify
            context: Execution context
            step_results: Results from previous steps
            
        Returns:
            List of expanded task IDs
        """
        from .core.for_each_task import ForEachTask
        from .core.task import Task
        
        self.logger.info(f"Expanding ForEachTask '{for_each_task.id}' at runtime")
        
        # Resolve the for_each expression using actual step results
        resolved_items = await self.control_flow_resolver.resolve_iterator(
            for_each_task.for_each_expr, context, step_results
        )
        
        self.logger.info(f"Resolved {len(resolved_items)} items for ForEachTask '{for_each_task.id}'")
        
        # Create expanded tasks
        expanded_tasks = []
        expanded_task_ids = []
        
        # Import the ForLoopHandler to reuse its loop context management
        from .control_flow.loops import ForLoopHandler
        loop_handler = ForLoopHandler(self.control_flow_resolver)
        
        for idx, item in enumerate(resolved_items):
            # Create comprehensive context for this iteration
            # Include ALL available data for template rendering
            loop_context = {
                **pipeline.context,  # Include all pipeline context (contains parameters)
                **context.get("pipeline_params", {}),  # Alternative pipeline params location
                **step_results,  # Include all previous step results directly
                "$item": item,
                "$index": idx,
                "$is_first": idx == 0,
                "$is_last": idx == len(resolved_items) - 1,
                # Also register without $ prefix for template compatibility
                "item": item,
                "index": idx,
                "is_first": idx == 0,
                "is_last": idx == len(resolved_items) - 1,
                f"${for_each_task.loop_var}": item if for_each_task.loop_var != "$item" else item
            }
            
            # Also flatten step results for direct access in templates
            for step_id, result in step_results.items():
                if step_id not in loop_context:
                    loop_context[step_id] = result
            
            # Add execution metadata
            from datetime import datetime
            loop_context["execution"] = {
                "timestamp": datetime.now().isoformat(),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M:%S"),
            }
            
            # Process each step in the loop body
            for step_def in for_each_task.loop_steps:
                # Create unique task ID
                task_id = f"{for_each_task.id}_{idx}_{step_def['id']}"
                
                # Process parameters - but DON'T render templates yet!
                # Templates will be rendered at execution time when all dependencies are available
                params = step_def.get("parameters", {})
                
                # For now, just do simple string replacement for loop variables
                # This is a temporary fix until proper partial rendering is implemented
                rendered_params = {}
                for key, value in params.items():
                    if isinstance(value, str):
                        # Replace only loop-specific variables, leave other templates intact
                        rendered_value = value
                        rendered_value = rendered_value.replace("{{ item }}", str(item))
                        rendered_value = rendered_value.replace("{{ $item }}", str(item))
                        rendered_value = rendered_value.replace("{{item}}", str(item))
                        rendered_value = rendered_value.replace("{{$item}}", str(item))
                        rendered_value = rendered_value.replace("{{ index }}", str(idx))
                        rendered_value = rendered_value.replace("{{ $index }}", str(idx))
                        rendered_value = rendered_value.replace("{{index}}", str(idx))
                        rendered_value = rendered_value.replace("{{$index}}", str(idx))
                        rendered_params[key] = rendered_value
                    else:
                        rendered_params[key] = value
                
                # Handle dependencies
                task_deps = []
                
                # Add dependencies from the ForEachTask itself
                if idx == 0:
                    # First iteration depends on ForEachTask dependencies
                    task_deps.extend(for_each_task.dependencies)
                    self.logger.info(f"DEBUG: First iteration of {for_each_task.id}, adding deps: {for_each_task.dependencies}")
                elif for_each_task.max_parallel == 1:
                    # Sequential execution - depend on previous iteration
                    prev_task_id = f"{for_each_task.id}_{idx-1}_{step_def['id']}"
                    task_deps.append(prev_task_id)
                else:
                    # Parallel execution - depend on ForEachTask dependencies
                    task_deps.extend(for_each_task.dependencies)
                
                # Add internal dependencies within the loop body
                for dep in step_def.get("dependencies", []):
                    if dep in [s["id"] for s in for_each_task.loop_steps]:
                        # Internal dependency
                        task_deps.append(f"{for_each_task.id}_{idx}_{dep}")
                    else:
                        # External dependency
                        task_deps.append(dep)
                
                # Create the task
                task = Task(
                    id=task_id,
                    name=step_def.get("name", f"{step_def['id']} (item {idx})"),
                    action=step_def.get("action"),
                    parameters=rendered_params,
                    dependencies=task_deps,
                    metadata={
                        **step_def.get("metadata", {}),
                        "is_for_each_child": True,
                        "parent_for_each": for_each_task.id,
                        "iteration_index": idx,
                        "loop_context": {**loop_context, "loop_id": for_each_task.id, "index": idx},  # Ensure loop_id and index are in loop_context
                        "loop_step_id": step_def['id'],  # Store the original step ID for mapping
                        "loop_task_id": task_id  # Store the full task ID
                    }
                )
                self.logger.info(f"DEBUG: Created task {task_id} with deps: {task_deps}, params: {list(rendered_params.keys())}")
                
                # Handle tool field
                if "tool" in step_def:
                    task.metadata["tool"] = step_def["tool"]
                
                expanded_tasks.append(task)
                expanded_task_ids.append(task_id)
        
        # Add completion task if requested
        if for_each_task.add_completion_task and expanded_tasks:
            # Get all last tasks from each iteration
            last_task_ids = []
            for idx in range(len(resolved_items)):
                if for_each_task.loop_steps:
                    last_step_id = for_each_task.loop_steps[-1]["id"]
                    last_task_ids.append(f"{for_each_task.id}_{idx}_{last_step_id}")
            
            completion_task = Task(
                id=for_each_task.id + "_complete",
                name=f"Complete {for_each_task.id} loop",
                action="loop_complete",
                parameters={"loop_id": for_each_task.id, "iterations": len(resolved_items)},
                dependencies=last_task_ids,
                metadata={
                    "is_loop_completion": True,
                    "loop_id": for_each_task.id,
                    "control_flow_type": "for_each",
                }
            )
            expanded_tasks.append(completion_task)
            expanded_task_ids.append(completion_task.id)
        
        # Add expanded tasks to pipeline
        for task in expanded_tasks:
            pipeline.add_task(task)
        
        # Mark the ForEachTask as expanded
        for_each_task.mark_expanded(expanded_task_ids, resolved_items)
        
        self.logger.info(f"Expanded ForEachTask '{for_each_task.id}' into {len(expanded_tasks)} tasks")
        
        return expanded_task_ids

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
        basic_state = {
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
        
        # Enhanced state when using LangGraph
        if self.use_langgraph_state:
            # Add enhanced execution metadata
            basic_state["execution_metadata"] = {
                "orchestrator_version": "1.0.0",
                "model_registry_size": len(self.model_registry.models) if self.model_registry else 0,
                "control_system_type": type(self.control_system).__name__ if self.control_system else None,
                "template_manager_enabled": self.template_manager is not None,
                "max_concurrent_tasks": self.max_concurrent_tasks,
                "langgraph_storage_type": self.langgraph_state_manager.storage_type if self.langgraph_state_manager else None
            }
            
            # Add performance metrics
            basic_state["performance_metrics"] = {
                "running_pipelines_count": len(self.running_pipelines),
                "total_execution_history": len(self.execution_history) if hasattr(self, 'execution_history') else 0
            }
        
        return basic_state

    async def _update_global_execution_state(self, execution_id: str, task_id: str, task_result: Dict[str, Any]) -> None:
        """Update global state with task execution results (LangGraph only)."""
        if not self.use_langgraph_state:
            return
            
        # Get thread_id from execution_id mapping
        if hasattr(self.state_manager, "_execution_to_thread_mapping"):
            thread_id = self.state_manager._execution_to_thread_mapping.get(execution_id)
            if thread_id:
                try:
                    # Extract enhanced task metadata
                    task_metadata = task_result.get("task_metadata", {})
                    actual_result = task_result.get("result", task_result)
                    
                    # Prepare state updates
                    updates = {
                        "tool_results": {
                            "tool_calls": {task_id: {"result": actual_result}},
                            "tool_outputs": {task_id: actual_result},
                            "execution_times": {task_id: task_metadata.get("execution_time", 0)},
                            "tool_metadata": {task_id: task_metadata}
                        },
                        "intermediate_results": {task_id: actual_result}
                    }
                    
                    # Update model interactions if available
                    if "model_response" in task_result:
                        updates["model_interactions"] = {
                            "model_calls": [{
                                "task_id": task_id,
                                "model": task_metadata.get("model_used"),
                                "response": task_result["model_response"],
                                "timestamp": task_metadata.get("timestamp")
                            }]
                        }
                    
                    # Update global state
                    await self.langgraph_state_manager.update_global_state(
                        thread_id, updates, step_name=task_id
                    )
                    
                except Exception as e:
                    logging.getLogger(__name__).warning(f"Failed to update global state for task {task_id}: {e}")

    def _process_enhanced_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process enhanced results from LangGraph execution, extracting actual results."""
        if not self.use_langgraph_state:
            return results
            
        processed_results = {}
        for task_id, task_result in results.items():
            if isinstance(task_result, dict) and "result" in task_result:
                # Extract the actual result from enhanced structure
                processed_results[task_id] = task_result["result"]
            else:
                processed_results[task_id] = task_result
                
        return processed_results

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
                        # Create an object-like dict for template access
                        if isinstance(step_result, str):
                            context[step_id] = type(
                                "Result", (), {"result": step_result, "value": step_result}
                            )()
                        elif isinstance(step_result, dict):
                            # Create Result object with all dict keys as attributes
                            # This allows both step.field and step.result access
                            attrs = dict(step_result)
                            # Add result attribute for backward compatibility if not present
                            if "result" not in attrs:
                                attrs["result"] = step_result
                            context[step_id] = type("Result", (), attrs)()
                        else:
                            # For other types, wrap as result
                            context[step_id] = type(
                                "Result", (), {"result": step_result, "value": step_result}
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
            model_req = task.metadata["requires_model"]
            
            # Only look up model if it's a string (specific model name)
            if isinstance(model_req, str):
                model = self.model_registry.get_model(model_req)
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

    # Enhanced LangGraph state management methods
    
    async def get_pipeline_global_state(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive pipeline global state (only available with LangGraph).
        
        Args:
            execution_id: Pipeline execution ID
            
        Returns:
            Global state dictionary or None if not found
        """
        if not self.use_langgraph_state:
            raise ValueError("Global state only available when use_langgraph_state=True")
            
        # Get thread_id from execution_id mapping in legacy adapter
        if hasattr(self.state_manager, "_execution_to_thread_mapping"):
            thread_id = self.state_manager._execution_to_thread_mapping.get(execution_id)
            if thread_id:
                return await self.langgraph_state_manager.get_global_state(thread_id)
        return None
    
    async def create_named_checkpoint(self, execution_id: str, name: str, description: str = "") -> Optional[str]:
        """
        Create a named checkpoint with description (only available with LangGraph).
        
        Args:
            execution_id: Pipeline execution ID
            name: Checkpoint name
            description: Optional description
            
        Returns:
            Checkpoint ID or None if failed
        """
        if not self.use_langgraph_state:
            raise ValueError("Named checkpoints only available when use_langgraph_state=True")
            
        if hasattr(self.state_manager, "_execution_to_thread_mapping"):
            thread_id = self.state_manager._execution_to_thread_mapping.get(execution_id)
            if thread_id:
                return await self.langgraph_state_manager.create_checkpoint(
                    thread_id, description, {"name": name}
                )
        return None
    
    async def get_pipeline_metrics(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive pipeline execution metrics (only available with LangGraph).
        
        Args:
            execution_id: Pipeline execution ID
            
        Returns:
            Metrics dictionary or None if not found
        """
        if not self.use_langgraph_state:
            raise ValueError("Pipeline metrics only available when use_langgraph_state=True")
            
        state = await self.get_pipeline_global_state(execution_id)
        if state:
            return {
                "execution_metadata": state.get("execution_metadata", {}),
                "performance_metrics": state.get("performance_metrics", {}),
                "model_interactions": state.get("model_interactions", {}),
                "tool_results": state.get("tool_results", {}),
                "error_context": state.get("error_context", {})
            }
        return None
    
    def get_state_manager_type(self) -> str:
        """Get the type of state manager being used."""
        return "langgraph" if self.use_langgraph_state else "legacy"
    
    def get_langgraph_manager(self) -> Optional[LangGraphGlobalContextManager]:
        """Get direct access to LangGraph manager (if enabled)."""
        return self.langgraph_state_manager

    # Issue #205: Automatic Checkpointing Utility Methods
    
    def is_automatic_checkpointing_enabled(self) -> bool:
        """Check if automatic checkpointing is enabled."""
        return self.use_automatic_checkpointing
    
    async def get_execution_status_with_checkpoints(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get execution status including checkpoint information."""
        if self.durable_executor:
            return await self.durable_executor.get_execution_status(execution_id)
        return None
    
    def get_checkpoint_performance_metrics(self) -> Dict[str, Any]:
        """Get checkpoint performance metrics."""
        if self.durable_executor:
            return self.durable_executor.get_performance_metrics()
        return {
            "automatic_checkpointing_enabled": False,
            "message": "Automatic checkpointing not enabled"
        }
    
    async def pause_checkpointed_execution(self, execution_id: str) -> bool:
        """Pause a checkpointed execution."""
        if self.durable_executor:
            return await self.durable_executor.pause_execution(execution_id)
        return False
    
    async def resume_checkpointed_execution(self, execution_id: str) -> bool:
        """Resume a paused checkpointed execution."""
        if self.durable_executor:
            return await self.durable_executor.resume_paused_execution(execution_id)
        return False
    
    async def get_pipeline_global_state(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get pipeline global state for LangGraph execution."""
        if self.langgraph_state_manager:
            return await self.langgraph_state_manager.get_global_state(thread_id)
        return None
    
    async def create_named_checkpoint(self, thread_id: str, name: str, description: str) -> Optional[str]:
        """Create a named checkpoint for LangGraph execution."""
        if self.langgraph_state_manager:
            return await self.langgraph_state_manager.create_checkpoint(
                thread_id=thread_id,
                description=f"{name}: {description}",
                metadata={"checkpoint_name": name, "user_created": True}
            )
        return None
    
    async def list_execution_checkpoints(self, thread_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for an execution."""
        if self.langgraph_state_manager:
            return await self.langgraph_state_manager.list_checkpoints(thread_id)
        return []

    def get_pipeline_metrics(self, pipeline_id: str) -> Dict[str, Any]:
        """Get comprehensive metrics for a pipeline execution."""
        metrics = {
            "pipeline_id": pipeline_id,
            "automatic_checkpointing_enabled": self.use_automatic_checkpointing,
            "state_management_type": self.get_state_manager_type()
        }
        
        if self.use_automatic_checkpointing and self.durable_executor:
            metrics.update(self.durable_executor.get_performance_metrics())
            
        return metrics

    def __repr__(self) -> str:
        """String representation of orchestrator."""
        return f"Orchestrator(running_pipelines={len(self.running_pipelines)})"
