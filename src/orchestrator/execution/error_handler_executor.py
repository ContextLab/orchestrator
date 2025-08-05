"""
Advanced error handler executor for comprehensive task failure recovery.
Orchestrates error handler execution with retry logic, context passing, and analytics.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union

from ..core.error_handling import ErrorContext, ErrorHandler, ErrorHandlerResult
from ..core.error_handler_registry import ErrorHandlerRegistry
from ..core.task import Task, TaskStatus
from ..engine.pipeline_spec import TaskSpec

logger = logging.getLogger(__name__)


class ErrorHandlerExecutor:
    """Executes error handlers with comprehensive recovery logic and analytics."""
    
    def __init__(self, task_executor, handler_registry: Optional[ErrorHandlerRegistry] = None):
        """Initialize with task executor and handler registry."""
        self.task_executor = task_executor
        self.handler_registry = handler_registry or ErrorHandlerRegistry()
        
        # Execution tracking
        self.execution_history: Dict[str, List[Dict[str, Any]]] = {}
        self.active_handlers: Dict[str, Dict[str, Any]] = {}  # Currently executing handlers
        
        # Circuit breaker for infinite loops
        self.max_handler_chain_depth = 5
        self.max_total_retries_per_task = 10
        
        # Performance tracking
        self.execution_metrics = {
            'total_errors_handled': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'average_recovery_time': 0.0,
            'handler_chain_executions': 0
        }
    
    async def handle_task_error(
        self,
        failed_task: Union[Task, TaskSpec],
        error: Exception,
        context: Dict[str, Any],
        pipeline: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Main error handling orchestration with comprehensive recovery logic.
        
        Args:
            failed_task: The task that failed
            error: The exception that occurred
            context: Execution context
            pipeline: Pipeline instance (for handler task lookup)
            
        Returns:
            Recovery result with handler execution details
        """
        start_time = time.time()
        task_id = failed_task.id if hasattr(failed_task, 'id') else str(failed_task)
        
        logger.info(f"Handling error for task '{task_id}': {type(error).__name__}: {error}")
        
        # Check for infinite loop prevention
        if self._should_prevent_handler_execution(task_id, error):
            logger.warning(f"Preventing handler execution for task '{task_id}' due to loop detection")
            return self._build_final_error_result(failed_task, error, None, "Handler loop prevented")
        
        try:
            # 1. Build comprehensive error context
            error_context = self._build_error_context(failed_task, error, context, pipeline)
            
            # 2. Find and prioritize applicable handlers
            handlers = self._find_applicable_handlers(failed_task, error)
            
            if not handlers:
                logger.info(f"No error handlers found for task '{task_id}'")
                self.handler_registry.record_error_occurrence(task_id, type(error).__name__, handled=False)
                return self._build_final_error_result(failed_task, error, error_context, "No handlers found")
            
            logger.info(f"Found {len(handlers)} error handlers for task '{task_id}'")
            
            # 3. Execute handlers in priority order
            handler_results = []
            for i, (handler_id, handler) in enumerate(handlers):
                logger.info(f"Executing error handler {i+1}/{len(handlers)}: '{handler_id}'")
                
                handler_result = await self._execute_handler(
                    handler_id, handler, error_context, failed_task, context, pipeline
                )
                
                handler_results.append(handler_result)
                
                # Check if handler succeeded
                if handler_result.success:
                    logger.info(f"Handler '{handler_id}' succeeded")
                    
                    # Record successful handling
                    self.handler_registry.record_error_occurrence(task_id, type(error).__name__, handled=True)
                    
                    # Decide on retry behavior
                    if handler.retry_with_handler:
                        logger.info(f"Retrying original task '{task_id}' after successful handler")
                        retry_result = await self._retry_original_task(
                            failed_task, context, handler_result, error_context
                        )
                        
                        # Update metrics
                        execution_time = time.time() - start_time
                        self._update_execution_metrics(True, execution_time)
                        
                        return retry_result
                    else:
                        # Handler succeeded but no retry requested
                        result = self._build_handler_success_result(
                            failed_task, handler_result, error_context
                        )
                        
                        execution_time = time.time() - start_time
                        self._update_execution_metrics(True, execution_time)
                        
                        return result
                else:
                    logger.warning(f"Handler '{handler_id}' failed: {handler_result.error_message}")
                    
                    # Check if we should continue to next handler or stop
                    if not handler.continue_on_handler_failure:
                        logger.info(f"Stopping handler chain after failure of '{handler_id}'")
                        break
            
            # 4. All handlers failed or completed without success
            self.handler_registry.record_error_occurrence(task_id, type(error).__name__, handled=False)
            
            # Check for fallback values from any handler
            fallback_result = self._check_for_fallback_values(handler_results, failed_task, error_context)
            if fallback_result:
                logger.info(f"Using fallback value from failed handlers for task '{task_id}'")
                execution_time = time.time() - start_time
                self._update_execution_metrics(False, execution_time)
                return fallback_result
            
            # No successful handlers or fallbacks
            execution_time = time.time() - start_time
            self._update_execution_metrics(False, execution_time)
            
            return self._build_final_error_result(
                failed_task, error, error_context, "All handlers failed", handler_results
            )
            
        except Exception as handler_execution_error:
            logger.error(f"Error handler execution failed: {handler_execution_error}")
            execution_time = time.time() - start_time
            self._update_execution_metrics(False, execution_time)
            
            return self._build_final_error_result(
                failed_task, error, None, f"Handler execution error: {handler_execution_error}"
            )
    
    async def _execute_handler(
        self,
        handler_id: str,
        handler: ErrorHandler,
        error_context: ErrorContext,
        failed_task: Union[Task, TaskSpec],
        context: Dict[str, Any],
        pipeline: Optional[Any] = None
    ) -> ErrorHandlerResult:
        """Execute a specific error handler with retry logic and timeout."""
        start_time = time.time()
        task_id = failed_task.id if hasattr(failed_task, 'id') else str(failed_task)
        
        # Track active handler execution
        self.active_handlers[handler_id] = {
            'task_id': task_id,
            'start_time': start_time,
            'attempts': 0
        }
        
        try:
            # Build handler execution context
            handler_context = self._build_handler_context(error_context, context)
            
            # Execute handler with retries
            for attempt in range(handler.max_handler_retries + 1):
                self.active_handlers[handler_id]['attempts'] = attempt + 1
                
                try:
                    if attempt > 0:
                        logger.info(f"Retrying handler '{handler_id}', attempt {attempt + 1}/{handler.max_handler_retries + 1}")
                        await asyncio.sleep(min(2 ** (attempt - 1), 10))  # Exponential backoff, max 10s
                    
                    # Execute the handler
                    if handler.handler_task_id:
                        # Execute handler task from pipeline
                        handler_result = await self._execute_handler_task(
                            handler.handler_task_id, handler_context, handler, pipeline
                        )
                    elif handler.handler_action:
                        # Execute handler action directly
                        handler_result = await self._execute_handler_action(
                            handler.handler_action, handler_context, handler, failed_task
                        )
                    else:
                        # Use fallback value
                        handler_result = self._build_fallback_result(handler, error_context)
                    
                    # Handler succeeded
                    execution_time = time.time() - start_time
                    self.handler_registry.record_handler_execution(
                        handler_id, True, execution_time, error_context.error_type, task_id
                    )
                    
                    return ErrorHandlerResult(
                        success=True,
                        handler_id=handler_id,
                        handler_output=handler_result,
                        execution_time=execution_time,
                        should_retry_original=handler.retry_with_handler,
                        should_continue_pipeline=handler.continue_on_handler_failure,
                        handler_attempts=attempt + 1
                    )
                    
                except Exception as handler_error:
                    logger.warning(f"Handler '{handler_id}' attempt {attempt + 1} failed: {handler_error}")
                    
                    if attempt == handler.max_handler_retries:
                        # Final attempt failed
                        execution_time = time.time() - start_time
                        self.handler_registry.record_handler_execution(
                            handler_id, False, execution_time, error_context.error_type, task_id
                        )
                        
                        return ErrorHandlerResult(
                            success=False,
                            handler_id=handler_id,
                            error_message=str(handler_error),
                            error_type=type(handler_error).__name__,
                            handler_traceback=None,  # Could add traceback if needed
                            fallback_value=handler.fallback_value,
                            should_continue_pipeline=handler.continue_on_handler_failure,
                            handler_attempts=attempt + 1
                        )
        
        finally:
            # Clean up active handler tracking
            self.active_handlers.pop(handler_id, None)
        
        # Should not reach here
        return ErrorHandlerResult(
            success=False,
            handler_id=handler_id,
            error_message="Handler execution completed unexpectedly"
        )
    
    async def _execute_handler_task(
        self,
        handler_task_id: str,
        handler_context: Dict[str, Any],
        handler: ErrorHandler,
        pipeline: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Execute a handler task from the pipeline."""
        if not pipeline:
            raise ValueError(f"Pipeline required to execute handler task '{handler_task_id}'")
        
        # Get handler task from pipeline
        handler_task = None
        if hasattr(pipeline, 'get_task'):
            handler_task = pipeline.get_task(handler_task_id)
        elif hasattr(pipeline, 'tasks') and isinstance(pipeline.tasks, dict):
            handler_task = pipeline.tasks.get(handler_task_id)
        
        if not handler_task:
            raise ValueError(f"Handler task '{handler_task_id}' not found in pipeline")
        
        logger.debug(f"Executing handler task '{handler_task_id}'")
        
        # Execute with timeout if specified
        if handler.timeout:
            try:
                return await asyncio.wait_for(
                    self.task_executor.execute_task(handler_task, handler_context),
                    timeout=handler.timeout
                )
            except asyncio.TimeoutError:
                raise Exception(f"Handler task '{handler_task_id}' exceeded timeout of {handler.timeout}s")
        else:
            return await self.task_executor.execute_task(handler_task, handler_context)
    
    async def _execute_handler_action(
        self,
        handler_action: str,
        handler_context: Dict[str, Any],
        handler: ErrorHandler,
        failed_task: Union[Task, TaskSpec]
    ) -> Dict[str, Any]:
        """Execute a handler action directly."""
        task_id = failed_task.id if hasattr(failed_task, 'id') else str(failed_task)
        
        # Create temporary task spec for handler action
        handler_spec = TaskSpec(
            id=f"{task_id}_error_handler",
            action=handler_action,
            timeout=handler.timeout
        )
        
        logger.debug(f"Executing handler action for task '{task_id}': {handler_action[:100]}...")
        
        # Execute with timeout if specified  
        if handler.timeout:
            try:
                return await asyncio.wait_for(
                    self.task_executor.execute_task(handler_spec, handler_context),
                    timeout=handler.timeout
                )
            except asyncio.TimeoutError:
                raise Exception(f"Handler action exceeded timeout of {handler.timeout}s")
        else:
            return await self.task_executor.execute_task(handler_spec, handler_context)
    
    async def _retry_original_task(
        self,
        failed_task: Union[Task, TaskSpec],
        context: Dict[str, Any],
        handler_result: ErrorHandlerResult,
        error_context: ErrorContext
    ) -> Dict[str, Any]:
        """Retry the original task after successful handler execution."""
        task_id = failed_task.id if hasattr(failed_task, 'id') else str(failed_task)
        
        logger.info(f"Retrying original task '{task_id}' after successful error handler")
        
        # Update context with any modifications from handler
        updated_context = context.copy()
        if handler_result.context_modifications:
            updated_context.update(handler_result.context_modifications)
        
        # Add error context for the retry
        updated_context['_error_recovery'] = {
            'previous_error': error_context.to_dict(),
            'handler_result': handler_result.to_dict(),
            'retry_attempt': error_context.execution_attempt + 1
        }
        
        try:
            # Execute the original task again
            retry_result = await self.task_executor.execute_task(failed_task, updated_context)
            
            logger.info(f"Original task '{task_id}' succeeded after error handler recovery")
            
            # Add recovery information to result
            retry_result['recovered_from_error'] = True
            retry_result['recovery_handler'] = handler_result.handler_id
            retry_result['original_error'] = {
                'type': error_context.error_type,
                'message': error_context.error_message
            }
            
            return retry_result
            
        except Exception as retry_error:
            logger.warning(f"Original task '{task_id}' failed again after handler recovery: {retry_error}")
            
            # Return comprehensive failure result
            return {
                'task_id': task_id,
                'success': False,
                'error': str(retry_error),
                'error_type': type(retry_error).__name__,
                'original_error': {
                    'type': error_context.error_type,
                    'message': error_context.error_message
                },
                'recovery_attempted': True,
                'recovery_handler': handler_result.handler_id,
                'recovery_handler_succeeded': handler_result.success,
                'retry_failed': True,
                'timestamp': time.time()
            }
    
    def _build_error_context(
        self,
        failed_task: Union[Task, TaskSpec],
        error: Exception,
        context: Dict[str, Any],
        pipeline: Optional[Any] = None
    ) -> ErrorContext:
        """Build comprehensive error context for handlers."""
        task_id = failed_task.id if hasattr(failed_task, 'id') else str(failed_task)
        task_name = getattr(failed_task, 'name', task_id)
        
        # Extract task information
        task_parameters = {}
        task_dependencies = []
        task_result = None
        task_metadata = {}
        
        if hasattr(failed_task, 'parameters'):
            task_parameters = failed_task.parameters or {}
        elif hasattr(failed_task, 'inputs'):
            task_parameters = failed_task.inputs or {}
        
        if hasattr(failed_task, 'dependencies'):
            task_dependencies = failed_task.dependencies or []
        elif hasattr(failed_task, 'depends_on'):
            task_dependencies = failed_task.depends_on or []
        
        if hasattr(failed_task, 'result'):
            task_result = failed_task.result
        
        if hasattr(failed_task, 'metadata'):
            task_metadata = failed_task.metadata or {}
        
        # Get execution attempt from context
        execution_attempt = 1
        if '_error_recovery' in context:
            execution_attempt = context['_error_recovery'].get('retry_attempt', 1)
        
        # Create error context
        error_context = ErrorContext.from_exception(
            failed_task_id=task_id,
            error=error,
            task_parameters=task_parameters,
            pipeline_context=context,
            failed_task_name=task_name,
            task_dependencies=task_dependencies,
            task_result=task_result,
            task_metadata=task_metadata,
            execution_attempt=execution_attempt,
            pipeline_id=getattr(pipeline, 'id', None) if pipeline else None
        )
        
        # Add recovery suggestions
        suggestions = self.handler_registry.get_recovery_suggestions(
            error_context.error_type, task_id
        )
        for suggestion in suggestions:
            error_context.add_recovery_suggestion(suggestion)
        
        return error_context
    
    def _build_handler_context(self, error_context: ErrorContext, original_context: Dict[str, Any]) -> Dict[str, Any]:
        """Build execution context for error handlers."""
        handler_context = original_context.copy()
        
        # Add error information
        handler_context['error'] = error_context.to_dict()
        
        # Add convenience fields for easier access in handlers
        handler_context['error_type'] = error_context.error_type
        handler_context['error_message'] = error_context.error_message
        handler_context['failed_task_id'] = error_context.failed_task_id
        handler_context['failed_task_name'] = error_context.failed_task_name
        handler_context['task_parameters'] = error_context.task_parameters
        handler_context['execution_attempt'] = error_context.execution_attempt
        
        return handler_context
    
    def _find_applicable_handlers(
        self, failed_task: Union[Task, TaskSpec], error: Exception
    ) -> List[tuple[str, ErrorHandler]]:
        """Find and prioritize handlers for the given error."""
        task_id = failed_task.id if hasattr(failed_task, 'id') else str(failed_task)
        
        # Get handlers from registry
        registry_handlers = self.handler_registry.find_matching_handlers(error, task_id)
        
        # Get handlers from task specification
        task_handlers = []
        if hasattr(failed_task, 'get_error_handlers'):
            task_spec_handlers = failed_task.get_error_handlers()
            for i, handler in enumerate(task_spec_handlers):
                if handler.matches_error(error, task_id):
                    handler_id = f"{task_id}_handler_{i}"
                    task_handlers.append((handler_id, handler))
        
        # Combine and sort by priority
        all_handlers = registry_handlers + task_handlers
        all_handlers.sort(key=lambda x: x[1].priority)
        
        return all_handlers
    
    def _should_prevent_handler_execution(self, task_id: str, error: Exception) -> bool:
        """Check if handler execution should be prevented to avoid infinite loops."""
        # Check total retry count for this task
        error_stats = self.handler_registry.get_error_statistics(task_id)
        total_errors = error_stats.get('total_errors', 0)
        
        if total_errors >= self.max_total_retries_per_task:
            logger.warning(f"Task '{task_id}' has exceeded maximum retry limit ({self.max_total_retries_per_task})")
            return True
        
        # Check for rapid repeated failures (potential infinite loop)
        if task_id in self.execution_history:
            recent_failures = self.execution_history[task_id][-5:]  # Last 5 executions
            if len(recent_failures) >= 5:
                # Check if all recent executions were failures of the same error type
                error_type = type(error).__name__
                if all(exec_info.get('error_type') == error_type for exec_info in recent_failures):
                    time_span = recent_failures[-1]['timestamp'] - recent_failures[0]['timestamp']
                    if time_span < 60:  # 5 failures of same type in less than 1 minute
                        logger.warning(f"Detected rapid repeated failures for task '{task_id}', preventing handler execution")
                        return True
        
        return False
    
    def _build_final_error_result(
        self,
        failed_task: Union[Task, TaskSpec],
        error: Exception,
        error_context: Optional[ErrorContext],
        reason: str,
        handler_results: List[ErrorHandlerResult] = None
    ) -> Dict[str, Any]:
        """Build final error result when all handlers fail or no handlers exist."""
        task_id = failed_task.id if hasattr(failed_task, 'id') else str(failed_task)
        
        result = {
            'task_id': task_id,
            'success': False,
            'error': str(error),
            'error_type': type(error).__name__,
            'error_handling_reason': reason,
            'timestamp': time.time(),
            'handlers_attempted': len(handler_results) if handler_results else 0
        }
        
        if error_context:
            result['error_context'] = error_context.to_dict()
        
        if handler_results:
            result['handler_results'] = [hr.to_dict() for hr in handler_results]
        
        return result
    
    def _build_handler_success_result(
        self,
        failed_task: Union[Task, TaskSpec],
        handler_result: ErrorHandlerResult,
        error_context: ErrorContext
    ) -> Dict[str, Any]:
        """Build result for successful handler execution without retry."""
        task_id = failed_task.id if hasattr(failed_task, 'id') else str(failed_task)
        
        return {
            'task_id': task_id,
            'success': True,
            'result': handler_result.handler_output,
            'recovered_from_error': True,
            'recovery_handler': handler_result.handler_id,
            'original_error': {
                'type': error_context.error_type,
                'message': error_context.error_message
            },
            'handler_execution_time': handler_result.execution_time,
            'timestamp': time.time()
        }
    
    def _build_fallback_result(self, handler: ErrorHandler, error_context: ErrorContext) -> Dict[str, Any]:
        """Build result using handler's fallback configuration."""
        return {
            'success': True,
            'result': handler.fallback_value,
            'fallback_used': True,
            'original_error': {
                'type': error_context.error_type,
                'message': error_context.error_message
            }
        }
    
    def _check_for_fallback_values(
        self,
        handler_results: List[ErrorHandlerResult],
        failed_task: Union[Task, TaskSpec],
        error_context: ErrorContext
    ) -> Optional[Dict[str, Any]]:
        """Check if any failed handlers provided fallback values."""
        for handler_result in handler_results:
            if not handler_result.success and handler_result.fallback_value is not None:
                task_id = failed_task.id if hasattr(failed_task, 'id') else str(failed_task)
                
                return {
                    'task_id': task_id,
                    'success': True,
                    'result': handler_result.fallback_value,
                    'fallback_used': True,
                    'fallback_from_handler': handler_result.handler_id,
                    'original_error': {
                        'type': error_context.error_type,
                        'message': error_context.error_message
                    },
                    'timestamp': time.time()
                }
        
        return None
    
    def _update_execution_metrics(self, success: bool, execution_time: float) -> None:
        """Update execution metrics for analytics."""
        self.execution_metrics['total_errors_handled'] += 1
        
        if success:
            self.execution_metrics['successful_recoveries'] += 1
        else:
            self.execution_metrics['failed_recoveries'] += 1
        
        # Update average execution time
        current_avg = self.execution_metrics['average_recovery_time']
        total_handled = self.execution_metrics['total_errors_handled']
        
        self.execution_metrics['average_recovery_time'] = (
            (current_avg * (total_handled - 1)) + execution_time
        ) / total_handled
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get current execution metrics."""
        return self.execution_metrics.copy()
    
    def clear_execution_history(self) -> None:
        """Clear execution history (useful for testing)."""
        self.execution_history.clear()
        self.active_handlers.clear()
        
    def get_active_handlers(self) -> Dict[str, Dict[str, Any]]:
        """Get currently executing handlers."""
        return self.active_handlers.copy()