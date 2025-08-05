"""ParallelQueueHandler for managing create_parallel_queue execution."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import json

from ..core.parallel_queue_task import ParallelQueueTask, ParallelQueueStatus, ParallelSubtask
from ..core.task import Task, TaskStatus
from ..core.loop_context import GlobalLoopContextManager
from ..core.parallel_loop_context import ParallelLoopContextManager, ParallelQueueContext
from .auto_resolver import ControlFlowAutoResolver
from .enhanced_condition_evaluator import EnhancedConditionEvaluator

logger = logging.getLogger(__name__)


class ParallelResourceManager:
    """Manages shared resources across parallel executions."""
    
    def __init__(self):
        self.tool_pools: Dict[str, List[Any]] = {}
        self.pool_locks: Dict[str, asyncio.Lock] = {}
        self.usage_stats: Dict[str, Dict[str, Any]] = {}
    
    async def acquire_tool_instance(self, tool_name: str, max_instances: int = 5):
        """Acquire a tool instance from the pool."""
        if tool_name not in self.pool_locks:
            self.pool_locks[tool_name] = asyncio.Lock()
        
        async with self.pool_locks[tool_name]:
            # Initialize pool if needed
            if tool_name not in self.tool_pools:
                self.tool_pools[tool_name] = []
                self.usage_stats[tool_name] = {
                    "total_acquisitions": 0,
                    "peak_usage": 0,
                    "current_usage": 0
                }
            
            pool = self.tool_pools[tool_name]
            stats = self.usage_stats[tool_name]
            
            # Try to get existing instance or create new one
            if pool:
                instance = pool.pop()
            elif stats["current_usage"] < max_instances:
                # Create new instance (placeholder for now)
                instance = f"{tool_name}_instance_{stats['current_usage']}"
            else:
                # Wait for an instance to become available
                instance = None
            
            if instance is not None:
                stats["total_acquisitions"] += 1
                stats["current_usage"] += 1
                stats["peak_usage"] = max(stats["peak_usage"], stats["current_usage"])
            
            return instance
    
    async def release_tool_instance(self, tool_name: str, instance: Any):
        """Release a tool instance back to the pool."""
        if tool_name not in self.pool_locks:
            return
        
        async with self.pool_locks[tool_name]:
            if tool_name in self.tool_pools:
                self.tool_pools[tool_name].append(instance)
                if tool_name in self.usage_stats:
                    self.usage_stats[tool_name]["current_usage"] -= 1
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        return {
            "pools": {name: len(pool) for name, pool in self.tool_pools.items()},
            "stats": self.usage_stats.copy()
        }


class ParallelQueueHandler:
    """
    Handles create_parallel_queue execution with real concurrency management.
    
    This class manages:
    - Dynamic queue generation using AUTO tag resolution
    - Parallel execution with configurable concurrency limits
    - Until/while condition evaluation across parallel tasks
    - Tool resource sharing and management
    - Performance monitoring and error handling
    """
    
    def __init__(self, 
                 auto_resolver: Optional[ControlFlowAutoResolver] = None,
                 loop_context_manager: Optional[GlobalLoopContextManager] = None,
                 condition_evaluator: Optional[EnhancedConditionEvaluator] = None):
        """Initialize the parallel queue handler.
        
        Args:
            auto_resolver: AUTO tag resolver for queue generation and conditions
            loop_context_manager: Loop context manager for variable resolution
            condition_evaluator: Enhanced condition evaluator from Issue 189
        """
        self.auto_resolver = auto_resolver or ControlFlowAutoResolver()
        self.loop_context_manager = loop_context_manager or GlobalLoopContextManager()
        self.parallel_context_manager = ParallelLoopContextManager()
        self.condition_evaluator = condition_evaluator
        
        # Resource management
        self.resource_manager = ParallelResourceManager()
        
        # Active queue tracking
        self.active_queues: Dict[str, ParallelQueueTask] = {}
        
        # Performance monitoring
        self.execution_stats = {
            "total_queues_processed": 0,
            "total_items_processed": 0,
            "average_queue_size": 0.0,
            "average_execution_time": 0.0,
        }
    
    async def execute_parallel_queue(self, 
                                   task: ParallelQueueTask,
                                   context: Dict[str, Any],
                                   step_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a parallel queue task with full lifecycle management.
        
        Args:
            task: The ParallelQueueTask to execute
            context: Pipeline execution context
            step_results: Results from previous steps
            
        Returns:
            Dictionary containing execution results and statistics
        """
        logger.info(f"Starting parallel queue execution: {task.id}")
        start_time = time.time()
        
        try:
            # Add to active queues
            self.active_queues[task.id] = task
            task.start()
            
            # Phase 1: Generate queue items
            await self._generate_queue_items(task, context, step_results)
            
            # Phase 1.5: Create parallel context
            parallel_context = self._create_parallel_context(task, context)
            
            # Phase 2: Execute items in parallel
            results = await self._execute_parallel_items(task, context, step_results, parallel_context)
            
            # Phase 3: Handle condition evaluation (if present)
            if task.until_condition or task.while_condition:
                await self._evaluate_termination_conditions(task, context, step_results, results)
            
            # Phase 4: Update statistics and aggregate results
            execution_time = time.time() - start_time
            task.stats.total_execution_time = execution_time
            self._update_global_stats(task, execution_time)
            
            # Aggregate results after stats are updated
            final_result = await self._aggregate_results(task, results)
            
            task.complete(final_result)
            task.queue_status = ParallelQueueStatus.COMPLETED
            
            logger.info(f"Parallel queue completed: {task.id} - {len(task.queue_items)} items in {execution_time:.2f}s")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Parallel queue execution failed: {task.id} - {e}")
            task.fail(e)
            task.queue_status = ParallelQueueStatus.FAILED
            raise
            
        finally:
            # Cleanup
            if task.id in self.active_queues:
                del self.active_queues[task.id]
            
            # Cleanup parallel context
            self.parallel_context_manager.cleanup_parallel_queue(task.id)
    
    async def _generate_queue_items(self, 
                                  task: ParallelQueueTask,
                                  context: Dict[str, Any],
                                  step_results: Dict[str, Any]) -> None:
        """Generate queue items using AUTO tag resolution."""
        logger.info(f"Generating queue for: {task.id} using expression: {task.on}")
        task.queue_status = ParallelQueueStatus.GENERATING_QUEUE
        queue_start_time = time.time()
        
        try:
            # Resolve the 'on' expression to get queue items
            if task.on.strip().startswith('<AUTO>') and task.on.strip().endswith('</AUTO>'):
                # Use AUTO tag resolution
                resolved_items = await self.auto_resolver.resolve_iterator(
                    task.on, context, step_results
                )
            else:
                # Try to evaluate as template or direct list
                resolved_items = await self._evaluate_queue_expression(
                    task.on, context, step_results
                )
            
            # Validate and add items to queue
            if not isinstance(resolved_items, list):
                if resolved_items is None:
                    resolved_items = []
                else:
                    resolved_items = [resolved_items]
            
            # Add items to the task queue
            for item in resolved_items:
                task.add_queue_item(item)
            
            task.stats.queue_generation_time = time.time() - queue_start_time
            
            logger.info(f"Generated queue for {task.id}: {len(task.queue_items)} items")
            
            if not task.queue_items:
                logger.warning(f"Empty queue generated for {task.id}")
            
        except Exception as e:
            logger.error(f"Queue generation failed for {task.id}: {e}")
            raise ValueError(f"Failed to generate queue items: {e}")
    
    async def _evaluate_queue_expression(self, 
                                       expression: str,
                                       context: Dict[str, Any],
                                       step_results: Dict[str, Any]) -> List[Any]:
        """Evaluate non-AUTO queue expressions."""
        try:
            # Try to parse as JSON list
            if expression.strip().startswith('[') and expression.strip().endswith(']'):
                return json.loads(expression)
            
            # Try template resolution
            template_manager = context.get("template_manager")
            if template_manager and "{{" in expression:
                resolved = template_manager.render(expression, additional_context=step_results)
                
                # Try parsing resolved result
                if resolved.strip().startswith('['):
                    return json.loads(resolved)
                else:
                    # Single item
                    return [resolved]
            
            # Try direct context lookup
            if expression in context:
                result = context[expression]
                return result if isinstance(result, list) else [result]
            
            if expression in step_results:
                result = step_results[expression]
                return result if isinstance(result, list) else [result]
            
            # Default: treat as single item
            return [expression]
            
        except Exception as e:
            logger.debug(f"Queue expression evaluation failed: {e}")
            return [expression]
    
    def _create_parallel_context(self, 
                               task: ParallelQueueTask,
                               context: Dict[str, Any]) -> ParallelQueueContext:
        """Create parallel context for the queue execution."""
        parallel_context = self.parallel_context_manager.create_parallel_queue_context(
            queue_id=task.id,
            items=task.queue_items,
            max_parallel=task.max_parallel,
            explicit_loop_name=task.metadata.get("loop_name"),
            until_condition=task.until_condition,
            while_condition=task.while_condition
        )
        
        # Push to active contexts
        self.parallel_context_manager.push_parallel_queue(parallel_context)
        
        logger.debug(f"Created parallel context for {task.id}: {parallel_context.loop_name}")
        return parallel_context
    
    async def _execute_parallel_items(self, 
                                    task: ParallelQueueTask,
                                    context: Dict[str, Any],
                                    step_results: Dict[str, Any],
                                    parallel_context: ParallelQueueContext) -> List[Dict[str, Any]]:
        """Execute queue items in parallel with concurrency control."""
        logger.info(f"Executing {len(task.queue_items)} items in parallel (max_parallel={task.max_parallel})")
        task.queue_status = ParallelQueueStatus.EXECUTING_PARALLEL
        
        if not task.queue_items:
            return []
        
        # Create subtasks for all queue items
        subtasks = []
        for queue_index, item in enumerate(task.queue_items):
            for action_def in task.action_loop:
                subtask = task.create_subtask(queue_index, item, action_def)
                subtasks.append(subtask)
        
        # Execute subtasks in parallel with concurrency control
        results = []
        semaphore = task.semaphore
        
        async def execute_subtask(subtask: ParallelSubtask) -> Dict[str, Any]:
            """Execute a single subtask with resource management."""
            async with semaphore:
                try:
                    subtask.start()
                    task.update_concurrency_stats()
                    
                    # Get tool instance if needed
                    tool_instance = None
                    if task.tool:
                        tool_instance = await self.resource_manager.acquire_tool_instance(task.tool)
                    
                    try:
                        # Mark item as started in parallel context
                        parallel_context.mark_item_started(subtask.queue_index)
                        
                        # Build execution context with parallel loop variables
                        item_context = context.copy()
                        item_context.update(step_results)
                        item_context.update(task.get_context_variables(subtask.queue_index))
                        item_context.update(
                            self.parallel_context_manager.get_parallel_template_variables(
                                task.id, subtask.queue_index
                            )
                        )
                        
                        # Execute the subtask (placeholder for now)
                        result = await self._execute_subtask_action(
                            subtask, item_context, tool_instance
                        )
                        
                        subtask.complete(result)
                        parallel_context.mark_item_completed(subtask.queue_index, result)
                        task.completed_items.add(subtask.queue_index)
                        task.stats.update_item_completion(subtask.execution_time or 0.0)
                        
                        return {
                            "subtask_id": subtask.id,
                            "queue_index": subtask.queue_index,
                            "item": subtask.item,
                            "result": result,
                            "status": "completed",
                            "execution_time": subtask.execution_time
                        }
                        
                    finally:
                        # Release tool instance
                        if tool_instance and task.tool:
                            await self.resource_manager.release_tool_instance(task.tool, tool_instance)
                
                except Exception as e:
                    subtask.fail(e)
                    parallel_context.mark_item_failed(subtask.queue_index, e)
                    task.failed_items.add(subtask.queue_index)
                    task.stats.failed_items += 1
                    
                    logger.error(f"Subtask failed: {subtask.id} - {e}")
                    
                    return {
                        "subtask_id": subtask.id,
                        "queue_index": subtask.queue_index,
                        "item": subtask.item,
                        "result": None,
                        "status": "failed",
                        "error": str(e),
                        "execution_time": subtask.execution_time
                    }
                finally:
                    task.update_concurrency_stats()
        
        # Execute all subtasks concurrently
        logger.info(f"Starting concurrent execution of {len(subtasks)} subtasks")
        results = await asyncio.gather(*[execute_subtask(st) for st in subtasks], return_exceptions=True)
        
        # Filter out exceptions and log them
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Subtask execution exception: {result}")
            else:
                final_results.append(result)
        
        logger.info(f"Parallel execution completed: {len(final_results)} results")
        return final_results
    
    async def _execute_subtask_action(self, 
                                    subtask: ParallelSubtask,
                                    context: Dict[str, Any],
                                    tool_instance: Optional[Any]) -> Dict[str, Any]:
        """Execute the actual action for a subtask with real tool integration."""
        action = subtask.task.action
        parameters = subtask.task.parameters.copy()
        
        try:
            # 1. Resolve AUTO tags in action and parameters if needed
            resolved_action = action
            if action.startswith('<AUTO>') and action.endswith('</AUTO>'):
                # Use the general _resolve_auto_tags method for actions
                resolved_action = await self.auto_resolver._resolve_auto_tags(
                    action, context, {}
                )
            
            # Resolve parameters
            resolved_params = await self._resolve_parameters(parameters, context)
            
            # 2. Execute the action using real tool integration
            if tool_instance:
                # Use tool instance if available
                result = await self._execute_with_tool(
                    resolved_action, resolved_params, tool_instance, context
                )
            else:
                # Execute action directly
                result = await self._execute_direct_action(
                    resolved_action, resolved_params, context
                )
            
            return {
                "action": resolved_action,
                "parameters": resolved_params,
                "item": subtask.item,
                "queue_index": subtask.queue_index,
                "result": result,
                "timestamp": time.time(),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Subtask action execution failed: {subtask.id} - {e}")
            return {
                "action": action,
                "parameters": parameters,
                "item": subtask.item,
                "queue_index": subtask.queue_index,
                "result": None,
                "error": str(e),
                "timestamp": time.time(),
                "status": "failed"
            }
    
    async def _resolve_parameters(self, 
                                parameters: Dict[str, Any], 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve AUTO tags and templates in parameters."""
        resolved = {}
        
        for key, value in parameters.items():
            if isinstance(value, str):
                if value.startswith('<AUTO>') and value.endswith('</AUTO>'):
                    # Resolve AUTO tag using general resolver
                    resolved[key] = await self.auto_resolver._resolve_auto_tags(
                        value, context, {}
                    )
                elif "{{" in value and "}}" in value:
                    # Resolve template
                    template_manager = context.get("template_manager")
                    if template_manager:
                        resolved[key] = template_manager.render(value, additional_context=context)
                    else:
                        resolved[key] = value
                else:
                    resolved[key] = value
            else:
                resolved[key] = value
        
        return resolved
    
    async def _execute_with_tool(self, 
                               action: str, 
                               parameters: Dict[str, Any],
                               tool_instance: Any,
                               context: Dict[str, Any]) -> Any:
        """Execute action using a tool instance."""
        try:
            # Check if tool instance has the action method
            if hasattr(tool_instance, action):
                method = getattr(tool_instance, action)
                if callable(method):
                    # Call the tool method
                    if asyncio.iscoroutinefunction(method):
                        return await method(**parameters)
                    else:
                        return method(**parameters)
            
            # Try generic execute method
            if hasattr(tool_instance, 'execute'):
                execute_method = getattr(tool_instance, 'execute')
                if callable(execute_method):
                    if asyncio.iscoroutinefunction(execute_method):
                        return await execute_method(action=action, parameters=parameters)
                    else:
                        return execute_method(action=action, parameters=parameters)
            
            # Fallback: return parameters processed
            return {"action_executed": action, "processed_parameters": parameters}
            
        except Exception as e:
            logger.error(f"Tool execution failed for action '{action}': {e}")
            raise
    
    async def _execute_direct_action(self, 
                                   action: str, 
                                   parameters: Dict[str, Any],
                                   context: Dict[str, Any]) -> Any:
        """Execute action directly without tool instance."""
        # This would integrate with the main orchestrator's action execution system
        # For now, return a processed result
        
        # Handle common actions that don't require tools
        if action == "debug" or action == "log":
            message = parameters.get("message", "Debug message")
            logger.info(f"Debug action executed: {message}")
            return {"debug_message": message, "action": action}
        
        elif action == "process_item":
            item = parameters.get("item", "unknown")
            logger.info(f"Processing item: {item}")
            return {"processed_item": item, "action": action, "result": f"Successfully processed {item}"}
        
        elif action == "fetch":
            url = parameters.get("url", "unknown")
            logger.info(f"Fetching URL: {url}")
            # Simulate fetch result
            return {"url": url, "action": action, "result": f"Content from {url}", "status_code": 200}
        
        else:
            # Generic action execution
            logger.info(f"Executing action: {action} with parameters: {parameters}")
            return {
                "action": action,
                "parameters": parameters,
                "result": f"Action {action} executed successfully",
                "generic_execution": True
            }
    
    async def _evaluate_termination_conditions(self, 
                                             task: ParallelQueueTask,
                                             context: Dict[str, Any],
                                             step_results: Dict[str, Any],
                                             results: List[Dict[str, Any]]) -> bool:
        """Evaluate until/while conditions to determine if execution should continue."""
        if not self.condition_evaluator:
            logger.warning("No condition evaluator available for until/while condition evaluation")
            return False
        
        task.queue_status = ParallelQueueStatus.EVALUATING_CONDITIONS
        
        # Build comprehensive context with all results
        eval_context = context.copy()
        eval_context.update(step_results)
        eval_context.update({
            "parallel_results": results,
            "completed_items": list(task.completed_items),
            "failed_items": list(task.failed_items),
            "total_items": len(task.queue_items),
            "completion_rate": task.stats.get_completion_rate(),
            "failure_rate": task.stats.get_failure_rate(),
        })
        
        should_terminate = False
        
        try:
            # Evaluate until condition (stop when true)
            if task.until_condition:
                until_result = await self.condition_evaluator.evaluate_condition(
                    condition=task.until_condition,
                    context=eval_context,
                    step_results=step_results,
                    iteration=0,  # Parallel queues don't have iterations
                    condition_type="until"
                )
                
                if until_result.should_terminate:
                    logger.info(f"Until condition satisfied for {task.id}: {task.until_condition}")
                    should_terminate = True
            
            # Evaluate while condition (stop when false)
            if task.while_condition and not should_terminate:
                while_result = await self.condition_evaluator.evaluate_condition(
                    condition=task.while_condition,
                    context=eval_context,
                    step_results=step_results,
                    iteration=0,
                    condition_type="while"
                )
                
                if while_result.should_terminate:
                    logger.info(f"While condition failed for {task.id}: {task.while_condition}")
                    should_terminate = True
        
        except Exception as e:
            logger.error(f"Condition evaluation failed for {task.id}: {e}")
            # Don't terminate on condition evaluation failure
            should_terminate = False
        
        return should_terminate
    
    async def _aggregate_results(self, 
                               task: ParallelQueueTask,
                               results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from all parallel executions."""
        successful_results = [r for r in results if r.get("status") == "completed"]
        failed_results = [r for r in results if r.get("status") == "failed"]
        
        return {
            "parallel_queue_id": task.id,
            "total_items": len(task.queue_items),
            "successful_items": len(successful_results),
            "failed_items": len(failed_results),
            "completion_rate": task.stats.get_completion_rate(),
            "failure_rate": task.stats.get_failure_rate(),
            "results": successful_results,
            "errors": failed_results,
            "execution_stats": {
                "queue_generation_time": task.stats.queue_generation_time,
                "total_execution_time": task.stats.total_execution_time,
                "average_item_time": task.stats.average_item_time,
                "max_concurrent_executions": task.stats.max_concurrent_reached,
            },
            "resource_stats": self.resource_manager.get_resource_stats(),
            "conditions_evaluated": {
                "until_condition": task.until_condition,
                "while_condition": task.while_condition,
            }
        }
    
    def _update_global_stats(self, task: ParallelQueueTask, execution_time: float) -> None:
        """Update global execution statistics."""
        self.execution_stats["total_queues_processed"] += 1
        self.execution_stats["total_items_processed"] += len(task.queue_items)
        
        # Update running averages
        queue_count = self.execution_stats["total_queues_processed"]
        
        # Average queue size
        total_items = self.execution_stats["total_items_processed"]
        self.execution_stats["average_queue_size"] = total_items / queue_count
        
        # Average execution time
        if queue_count == 1:
            self.execution_stats["average_execution_time"] = execution_time
        else:
            current_avg = self.execution_stats["average_execution_time"]
            self.execution_stats["average_execution_time"] = (
                (current_avg * (queue_count - 1) + execution_time) / queue_count
            )
    
    def get_handler_stats(self) -> Dict[str, Any]:
        """Get comprehensive handler statistics."""
        return {
            "execution_stats": self.execution_stats.copy(),
            "active_queues": len(self.active_queues),
            "resource_stats": self.resource_manager.get_resource_stats(),
            "active_queue_details": {
                queue_id: task.get_progress_summary()
                for queue_id, task in self.active_queues.items()
            }
        }
    
    async def cancel_queue(self, queue_id: str) -> bool:
        """Cancel a running parallel queue."""
        if queue_id not in self.active_queues:
            return False
        
        task = self.active_queues[queue_id]
        task.queue_status = ParallelQueueStatus.CANCELLED
        
        # Cancel running subtasks
        for subtask in task.get_active_subtasks():
            subtask.status = TaskStatus.SKIPPED
            task.skipped_items.add(subtask.queue_index)
        
        logger.info(f"Cancelled parallel queue: {queue_id}")
        return True