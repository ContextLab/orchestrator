"""Execution engine with control flow support."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set
from datetime import datetime

from ..core.pipeline import Pipeline
from ..core.task import Task, TaskStatus
from ..compiler.control_flow_compiler import ControlFlowCompiler
from ..control_flow import (
    ConditionalHandler,
    ForLoopHandler,
    WhileLoopHandler,
    DynamicFlowHandler,
    ControlFlowAutoResolver
)
from ..tools.base import default_registry
from ..models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class ControlFlowEngine:
    """Pipeline execution engine with advanced control flow support."""
    
    def __init__(
        self,
        model_registry: Optional[ModelRegistry] = None,
        tool_registry=None
    ):
        """Initialize control flow engine.
        
        Args:
            model_registry: Model registry for LLM access
            tool_registry: Tool registry for task execution
        """
        self.model_registry = model_registry
        self.tool_registry = tool_registry or default_registry
        self.compiler = ControlFlowCompiler(model_registry=model_registry)
        
        # Initialize control flow handlers
        self.control_flow_resolver = ControlFlowAutoResolver(model_registry)
        self.conditional_handler = ConditionalHandler(self.control_flow_resolver)
        self.for_loop_handler = ForLoopHandler(self.control_flow_resolver)
        self.while_loop_handler = WhileLoopHandler(self.control_flow_resolver)
        self.dynamic_flow_handler = DynamicFlowHandler(self.control_flow_resolver)
        
        # Execution state
        self.completed_tasks: Set[str] = set()
        self.step_results: Dict[str, Any] = {}
        self.skipped_tasks: Set[str] = set()
        
    async def execute_yaml(
        self,
        yaml_content: str,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a pipeline from YAML content.
        
        Args:
            yaml_content: YAML pipeline definition
            inputs: Input parameters
            
        Returns:
            Execution results
        """
        try:
            # Compile pipeline
            logger.info("Compiling pipeline with control flow support")
            pipeline = await self.compiler.compile(yaml_content, inputs)
            
            # Execute pipeline
            return await self.execute_pipeline(pipeline, inputs)
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise
    
    async def execute_pipeline(
        self,
        pipeline: Pipeline,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a compiled pipeline with control flow.
        
        Args:
            pipeline: Compiled pipeline
            context: Execution context
            
        Returns:
            Execution results
        """
        # Initialize execution state
        self.completed_tasks.clear()
        self.step_results.clear()
        self.skipped_tasks.clear()
        
        # Add pipeline info to context
        context['pipeline'] = {
            'id': pipeline.id,
            'name': pipeline.name,
            'version': pipeline.version
        }
        
        # Execute tasks
        start_time = datetime.now()
        
        try:
            while not self._is_pipeline_complete(pipeline):
                # Get ready tasks considering control flow
                ready_tasks = await self._get_ready_tasks(pipeline, context)
                
                if not ready_tasks:
                    # Check for while loops that need expansion
                    if await self._expand_while_loops(pipeline, context):
                        continue
                    
                    # No tasks ready and no loops to expand
                    if self._has_pending_tasks(pipeline):
                        raise RuntimeError("Pipeline deadlock: tasks pending but none ready")
                    break
                
                # Execute ready tasks
                await self._execute_tasks(ready_tasks, pipeline, context)
                
            # Build final results
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': not self._has_failed_tasks(pipeline),
                'pipeline': pipeline.name,
                'execution_time': execution_time,
                'completed_tasks': list(self.completed_tasks),
                'skipped_tasks': list(self.skipped_tasks),
                'results': self.step_results,
                'context': context
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution error: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': False,
                'pipeline': pipeline.name,
                'execution_time': execution_time,
                'error': str(e),
                'completed_tasks': list(self.completed_tasks),
                'results': self.step_results
            }
    
    async def _get_ready_tasks(
        self,
        pipeline: Pipeline,
        context: Dict[str, Any]
    ) -> List[Task]:
        """Get tasks ready for execution considering control flow.
        
        Args:
            pipeline: Pipeline being executed
            context: Execution context
            
        Returns:
            List of ready tasks
        """
        ready_tasks = []
        
        for task in pipeline.tasks.values():
            # Skip completed or running tasks
            if task.status != TaskStatus.PENDING:
                continue
                
            # Check conditional execution
            should_execute = await self.conditional_handler.evaluate_condition(
                task, context, self.step_results
            )
            
            if not should_execute:
                # Skip this task
                task.skip("Condition evaluated to false")
                self.skipped_tasks.add(task.id)
                continue
            
            # Resolve dynamic dependencies
            dependencies = await self.dynamic_flow_handler.resolve_dynamic_dependencies(
                task, context, self.step_results, pipeline.tasks
            )
            
            # Check if all dependencies are satisfied
            if all(dep in self.completed_tasks or dep in self.skipped_tasks 
                   for dep in dependencies):
                ready_tasks.append(task)
                
        return ready_tasks
    
    async def _execute_tasks(
        self,
        tasks: List[Task],
        pipeline: Pipeline,
        context: Dict[str, Any]
    ):
        """Execute a batch of tasks.
        
        Args:
            tasks: Tasks to execute
            pipeline: Pipeline being executed
            context: Execution context
        """
        # Execute tasks concurrently
        async def execute_task(task: Task):
            try:
                # Mark as running
                task.start()
                
                # Execute task action
                tool = self.tool_registry.get_tool(task.action)
                if not tool:
                    raise ValueError(f"Unknown action: {task.action}")
                
                # Execute with timeout
                if task.timeout:
                    result = await asyncio.wait_for(
                        tool.execute(**task.parameters),
                        timeout=task.timeout
                    )
                else:
                    result = await tool.execute(**task.parameters)
                
                # Mark as completed
                task.complete(result)
                self.completed_tasks.add(task.id)
                self.step_results[task.id] = result
                
                logger.info(f"Task {task.id} completed successfully")
                
                # Check for flow control
                jump_target = await self.dynamic_flow_handler.process_goto(
                    task, context, self.step_results, pipeline.tasks
                )
                
                if jump_target:
                    logger.info(f"Task {task.id} jumping to {jump_target}")
                    # Mark intermediate tasks as skipped
                    self._skip_tasks_until(pipeline, task.id, jump_target)
                
            except Exception as e:
                task.fail(e)
                logger.error(f"Task {task.id} failed: {e}")
                
                # Handle retry logic
                if task.can_retry():
                    task.reset()
                    logger.info(f"Task {task.id} will retry (attempt {task.retry_count + 1})")
                else:
                    raise
        
        # Execute all ready tasks
        await asyncio.gather(*[execute_task(task) for task in tasks])
    
    async def _expand_while_loops(
        self,
        pipeline: Pipeline,
        context: Dict[str, Any]
    ) -> bool:
        """Expand while loops that need another iteration.
        
        Args:
            pipeline: Pipeline being executed
            context: Execution context
            
        Returns:
            True if any loops were expanded
        """
        expanded = False
        
        for task in list(pipeline.tasks.values()):
            # Check if task is a while loop placeholder
            if (hasattr(task, 'metadata') and 
                task.metadata.get('is_while_loop') and
                task.status == TaskStatus.PENDING):
                
                loop_id = task.id
                max_iterations = task.metadata.get('max_iterations', 10)
                condition = task.metadata.get('while_condition', 'false')
                
                # Determine current iteration
                current_iteration = 0
                for t_id in self.completed_tasks:
                    if t_id.startswith(f"{loop_id}_") and "_result" in t_id:
                        current_iteration += 1
                
                # Check if should continue
                should_continue = await self.while_loop_handler.should_continue(
                    loop_id, condition, context, self.step_results,
                    current_iteration, max_iterations
                )
                
                if should_continue:
                    # Create tasks for next iteration
                    iteration_tasks = await self.while_loop_handler.create_iteration_tasks(
                        task.to_dict(), current_iteration, context, self.step_results
                    )
                    
                    # Add tasks to pipeline
                    for iter_task in iteration_tasks:
                        pipeline.add_task(iter_task)
                    
                    expanded = True
                    logger.info(f"Expanded while loop {loop_id} iteration {current_iteration}")
                else:
                    # Mark loop as completed
                    task.complete({'iterations': current_iteration})
                    self.completed_tasks.add(task.id)
                    
        return expanded
    
    def _skip_tasks_until(self, pipeline: Pipeline, from_task: str, to_task: str):
        """Skip tasks between from_task and to_task.
        
        Args:
            pipeline: Pipeline being executed
            from_task: Starting task ID
            to_task: Target task ID
        """
        # Get execution order
        try:
            execution_order = pipeline.get_execution_order()
        except Exception:
            # If we can't get order due to dynamic changes, skip
            return
            
        # Find positions
        try:
            from_idx = execution_order.index(from_task)
            to_idx = execution_order.index(to_task)
        except ValueError:
            return
            
        # Skip intermediate tasks
        for idx in range(from_idx + 1, to_idx):
            task_id = execution_order[idx]
            if task_id in pipeline.tasks:
                task = pipeline.tasks[task_id]
                if task.status == TaskStatus.PENDING:
                    task.skip("Skipped due to flow jump")
                    self.skipped_tasks.add(task_id)
    
    def _is_pipeline_complete(self, pipeline: Pipeline) -> bool:
        """Check if pipeline execution is complete.
        
        Args:
            pipeline: Pipeline being executed
            
        Returns:
            True if complete
        """
        for task in pipeline.tasks.values():
            if task.status in {TaskStatus.PENDING, TaskStatus.RUNNING}:
                # Check if it's a while loop that might expand
                if (hasattr(task, 'metadata') and 
                    task.metadata.get('is_while_loop') and
                    task.status == TaskStatus.PENDING):
                    continue
                return False
        return True
    
    def _has_pending_tasks(self, pipeline: Pipeline) -> bool:
        """Check if there are pending tasks.
        
        Args:
            pipeline: Pipeline being executed
            
        Returns:
            True if there are pending tasks
        """
        return any(task.status == TaskStatus.PENDING for task in pipeline.tasks.values())
    
    def _has_failed_tasks(self, pipeline: Pipeline) -> bool:
        """Check if there are failed tasks.
        
        Args:
            pipeline: Pipeline being executed
            
        Returns:
            True if there are failed tasks
        """
        return any(task.status == TaskStatus.FAILED for task in pipeline.tasks.values())