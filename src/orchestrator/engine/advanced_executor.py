"""Advanced task executor with conditional execution, loops, and error handling."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .enhanced_executor import EnhancedTaskExecutor
from .pipeline_spec import ErrorHandling, TaskSpec

logger = logging.getLogger(__name__)


class ConditionalExecutor:
    """Handles conditional execution logic for tasks."""

    def __init__(self):
        self.expression_cache = {}

    def evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a conditional expression."""
        if not condition:
            return True

        try:
            # Cache resolved conditions for performance
            cache_key = f"{condition}_{hash(str(sorted(context.items())))}"
            if cache_key in self.expression_cache:
                return self.expression_cache[cache_key]

            resolved_condition = self._resolve_condition_variables(condition, context)
            result = self._evaluate_expression(resolved_condition)

            self.expression_cache[cache_key] = result
            logger.debug(
                f"Condition '{condition}' -> '{resolved_condition}' -> {result}"
            )

            return result

        except Exception as e:
            logger.warning(f"Failed to evaluate condition '{condition}': {e}")
            return False

    def _resolve_condition_variables(
        self, condition: str, context: Dict[str, Any]
    ) -> str:
        """Resolve variables in condition expression."""
        import re

        def replace_var(match):
            var_path = match.group(1).strip()
            value = self._get_nested_value(context, var_path)

            # Convert to string representation for evaluation
            if isinstance(value, bool):
                return str(value).lower()
            elif isinstance(value, (int, float)):
                return str(value)
            elif isinstance(value, str):
                return f'"{value}"'
            elif value is None:
                return "None"
            else:
                return f'"{str(value)}"'

        return re.sub(r"\{\{([^}]+)\}\}", replace_var, condition)

    def _get_nested_value(self, context: Dict[str, Any], path: str) -> Any:
        """Get nested value from context using dot notation."""
        parts = path.split(".")
        value = context

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                logger.warning(f"Variable path '{path}' not found in context")
                return None

        return value

    def _evaluate_expression(self, expression: str) -> bool:
        """Evaluate a boolean expression safely."""
        # Simple expression evaluator - can be enhanced with ast module for safety
        expression = expression.strip()

        # Handle simple comparisons
        if " == " in expression:
            left, right = expression.split(" == ", 1)
            left_val = left.strip().strip('"')
            right_val = right.strip().strip('"')
            return left_val == right_val
        elif " != " in expression:
            left, right = expression.split(" != ", 1)
            return left.strip() != right.strip()
        elif " > " in expression:
            left, right = expression.split(" > ", 1)
            return float(left.strip()) > float(right.strip())
        elif " < " in expression:
            left, right = expression.split(" < ", 1)
            return float(left.strip()) < float(right.strip())
        elif " >= " in expression:
            left, right = expression.split(" >= ", 1)
            return float(left.strip()) >= float(right.strip())
        elif " <= " in expression:
            left, right = expression.split(" <= ", 1)
            return float(left.strip()) <= float(right.strip())
        elif expression.lower() in ["true", "false"]:
            return expression.lower() == "true"
        elif " and " in expression:
            parts = expression.split(" and ")
            return all(self._evaluate_expression(part) for part in parts)
        elif " or " in expression:
            parts = expression.split(" or ")
            return any(self._evaluate_expression(part) for part in parts)
        else:
            # Try to evaluate as a simple boolean
            try:
                return bool(eval(expression))
            except Exception:
                logger.warning(f"Could not evaluate expression: {expression}")
                return False


class LoopExecutor:
    """Handles loop execution for iterative tasks."""

    def __init__(self, task_executor):
        self.task_executor = task_executor
        self.conditional_executor = ConditionalExecutor()

    async def execute_loop(
        self, task_spec: TaskSpec, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a task with loop configuration."""
        if not task_spec.loop:
            raise ValueError("Task has no loop configuration")

        loop_spec = task_spec.loop
        loop_items = self._resolve_loop_items(loop_spec.foreach, context)

        if not loop_items:
            logger.warning(f"No items found for loop in task {task_spec.id}")
            return {"loop_results": [], "iteration_count": 0}

        # Limit iterations if specified
        if loop_spec.max_iterations:
            loop_items = loop_items[: loop_spec.max_iterations]

        logger.info(
            f"Executing loop for task {task_spec.id} with {len(loop_items)} iterations"
        )

        if loop_spec.parallel:
            return await self._execute_parallel_loop(task_spec, loop_items, context)
        else:
            return await self._execute_sequential_loop(task_spec, loop_items, context)

    def _resolve_loop_items(
        self, foreach_expression: str, context: Dict[str, Any]
    ) -> List[Any]:
        """Resolve the items to iterate over."""
        import re

        # Extract variable path from {{variable}} syntax
        match = re.search(r"\{\{([^}]+)\}\}", foreach_expression)
        if match:
            var_path = match.group(1).strip()
            value = self._get_nested_value(context, var_path)

            if isinstance(value, list):
                return value
            elif isinstance(value, dict):
                return list(value.items())
            elif isinstance(value, str):
                return list(value)
            else:
                return [value] if value is not None else []
        else:
            # Handle literal arrays
            try:
                import ast

                return ast.literal_eval(foreach_expression)
            except Exception:
                logger.warning(f"Could not resolve loop items: {foreach_expression}")
                return []

    def _get_nested_value(self, context: Dict[str, Any], path: str) -> Any:
        """Get nested value from context using dot notation."""
        parts = path.split(".")
        value = context

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None

        return value

    async def _execute_sequential_loop(
        self, task_spec: TaskSpec, loop_items: List[Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute loop iterations sequentially."""
        results = []

        for i, item in enumerate(loop_items):
            iteration_context = context.copy()
            iteration_context["loop_item"] = item
            iteration_context["loop_index"] = i
            iteration_context["loop_total"] = len(loop_items)

            # Check break condition
            if task_spec.loop.break_condition:
                if self.conditional_executor.evaluate_condition(
                    task_spec.loop.break_condition, iteration_context
                ):
                    logger.info(
                        f"Breaking loop for task {task_spec.id} at iteration {i}"
                    )
                    break

            try:
                # Create iteration-specific task
                iteration_task = self._create_iteration_task(task_spec, i)
                result = await self.task_executor.execute_task(
                    iteration_task, iteration_context
                )

                if task_spec.loop.collect_results:
                    results.append(result)

                # Update context with iteration result
                iteration_context[f"{task_spec.id}_iteration_{i}"] = result

            except Exception as e:
                logger.warning(
                    f"Loop iteration {i} failed for task {task_spec.id}: {e}"
                )
                if task_spec.loop.collect_results:
                    results.append({"error": str(e), "iteration": i})

        return {
            "loop_results": results,
            "iteration_count": len(results),
            "loop_completed": True,
        }

    async def _execute_parallel_loop(
        self, task_spec: TaskSpec, loop_items: List[Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute loop iterations in parallel."""
        tasks = []

        for i, item in enumerate(loop_items):
            iteration_context = context.copy()
            iteration_context["loop_item"] = item
            iteration_context["loop_index"] = i
            iteration_context["loop_total"] = len(loop_items)

            iteration_task = self._create_iteration_task(task_spec, i)
            task = asyncio.create_task(
                self.task_executor.execute_task(iteration_task, iteration_context)
            )
            tasks.append((i, task))

        # Wait for all iterations to complete
        results = []
        for i, task in tasks:
            try:
                result = await task
                if task_spec.loop.collect_results:
                    results.append(result)
            except Exception as e:
                logger.warning(
                    f"Parallel loop iteration {i} failed for task {task_spec.id}: {e}"
                )
                if task_spec.loop.collect_results:
                    results.append({"error": str(e), "iteration": i})

        return {
            "loop_results": results,
            "iteration_count": len(results),
            "loop_completed": True,
            "execution_mode": "parallel",
        }

    def _create_iteration_task(self, task_spec: TaskSpec, iteration: int) -> TaskSpec:
        """Create a task specification for a loop iteration."""
        return TaskSpec(
            id=f"{task_spec.id}_iter_{iteration}",
            action=task_spec.action,
            inputs=task_spec.inputs,
            tools=task_spec.tools,
            depends_on=[],  # Dependencies handled at loop level
            condition=None,  # Conditions handled at loop level
            on_error=task_spec.on_error,
            loop=None,  # Remove loop to prevent recursion
            model_requirements=task_spec.model_requirements,
            timeout=task_spec.timeout,
            cache_results=task_spec.cache_results,
            tags=task_spec.tags + [f"iteration_{iteration}"],
        )


class ErrorRecoveryExecutor:
    """Handles advanced error recovery and retry logic."""

    def __init__(self, task_executor):
        self.task_executor = task_executor

    async def execute_with_error_handling(
        self, task_spec: TaskSpec, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute task with sophisticated error handling."""
        if not task_spec.has_error_handling():
            return await self.task_executor.execute_task(task_spec, context)

        error_config = task_spec.on_error
        max_retries = (
            error_config.retry_count if isinstance(error_config, ErrorHandling) else 0
        )
        retry_delay = (
            error_config.retry_delay if isinstance(error_config, ErrorHandling) else 1.0
        )

        last_error = None

        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(
                        f"Retrying task {task_spec.id}, attempt {attempt + 1}/{max_retries + 1}"
                    )
                    await asyncio.sleep(
                        retry_delay * (2 ** (attempt - 1))
                    )  # Exponential backoff

                result = await self.task_executor.execute_task(task_spec, context)

                if attempt > 0:
                    logger.info(
                        f"Task {task_spec.id} succeeded after {attempt} retries"
                    )

                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Task {task_spec.id} failed on attempt {attempt + 1}: {e}"
                )

                if attempt == max_retries:
                    # Final attempt failed, handle error
                    return await self._handle_final_error(
                        task_spec, last_error, context
                    )

        # Should not reach here
        return await self._handle_final_error(task_spec, last_error, context)

    async def _handle_final_error(
        self, task_spec: TaskSpec, error: Exception, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle error after all retries have been exhausted."""
        error_config = task_spec.on_error

        if isinstance(error_config, ErrorHandling):
            # Execute error handling action
            if error_config.action:
                try:
                    error_context = context.copy()
                    error_context["error"] = str(error)
                    error_context["error_type"] = type(error).__name__
                    error_context["failed_task"] = task_spec.id

                    error_task = TaskSpec(
                        id=f"{task_spec.id}_error_handler", action=error_config.action
                    )

                    error_result = await self.task_executor.execute_task(
                        error_task, error_context
                    )

                    return {
                        "task_id": task_spec.id,
                        "success": False,
                        "error": str(error),
                        "error_handled": True,
                        "error_handler_result": error_result,
                        "fallback_value": error_config.fallback_value,
                        "continue_pipeline": error_config.continue_on_error,
                    }

                except Exception as handler_error:
                    logger.error(
                        f"Error handler for task {task_spec.id} also failed: {handler_error}"
                    )

            # Return fallback result
            return {
                "task_id": task_spec.id,
                "success": False,
                "error": str(error),
                "error_handled": True,
                "fallback_value": error_config.fallback_value,
                "continue_pipeline": error_config.continue_on_error,
            }

        else:
            # Simple error handling
            raise error


class AdvancedTaskExecutor(EnhancedTaskExecutor):
    """Advanced task executor with conditional execution, loops, and error handling."""

    def __init__(self, model_registry=None, tool_registry=None):
        super().__init__(model_registry, tool_registry)
        self.conditional_executor = ConditionalExecutor()
        self.loop_executor = LoopExecutor(super())
        self.error_recovery_executor = ErrorRecoveryExecutor(super())
        self.execution_cache = {}

    async def execute_task(
        self, task_spec: TaskSpec, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute task with advanced features (conditions, loops, error handling)."""
        logger.info(f"Advanced execution for task: {task_spec.id}")

        try:
            # 1. Check cache if enabled
            if task_spec.cache_results:
                cached_result = self._get_cached_result(task_spec, context)
                if cached_result:
                    logger.debug(f"Using cached result for task {task_spec.id}")
                    return cached_result

            # 2. Check condition if present
            if task_spec.has_condition():
                if not self.conditional_executor.evaluate_condition(
                    task_spec.condition, context
                ):
                    logger.info(f"Skipping task {task_spec.id} due to condition")
                    return {
                        "task_id": task_spec.id,
                        "success": True,
                        "skipped": True,
                        "reason": "condition_not_met",
                        "condition": task_spec.condition,
                    }

            # 3. Handle timeout if specified
            if task_spec.timeout:
                result = await asyncio.wait_for(
                    self._execute_task_core(task_spec, context),
                    timeout=task_spec.timeout,
                )
            else:
                result = await self._execute_task_core(task_spec, context)

            # 4. Cache result if enabled
            if task_spec.cache_results and result.get("success", True):
                self._cache_result(task_spec, context, result)

            return result

        except asyncio.TimeoutError:
            logger.error(
                f"Task {task_spec.id} timed out after {task_spec.timeout} seconds"
            )
            return {
                "task_id": task_spec.id,
                "success": False,
                "error": f"Task timed out after {task_spec.timeout} seconds",
                "timeout": True,
            }
        except Exception as e:
            logger.error(f"Advanced task {task_spec.id} failed: {str(e)}")
            return await self.error_recovery_executor.execute_with_error_handling(
                task_spec, context
            )

    async def _execute_task_core(
        self, task_spec: TaskSpec, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the core task logic."""
        # Handle loops
        if task_spec.has_loop():
            result = await self.loop_executor.execute_loop(task_spec, context)
        else:
            # Execute with error handling
            result = await self.error_recovery_executor.execute_with_error_handling(
                task_spec, context
            )

        # Add execution metadata
        result["execution_metadata"] = task_spec.get_execution_metadata()
        result["timestamp"] = datetime.now().isoformat()

        return result

    def _get_cached_result(
        self, task_spec: TaskSpec, context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get cached result for task if available."""
        cache_key = self._generate_cache_key(task_spec, context)
        return self.execution_cache.get(cache_key)

    def _cache_result(
        self, task_spec: TaskSpec, context: Dict[str, Any], result: Dict[str, Any]
    ):
        """Cache task result."""
        cache_key = self._generate_cache_key(task_spec, context)
        self.execution_cache[cache_key] = result

    def _generate_cache_key(self, task_spec: TaskSpec, context: Dict[str, Any]) -> str:
        """Generate cache key for task and context."""
        # Simple cache key - can be enhanced
        import hashlib

        key_data = {
            "task_id": task_spec.id,
            "action": task_spec.action,
            "inputs": task_spec.inputs,
            "context_keys": sorted(context.keys()),
        }

        key_str = str(sorted(key_data.items()))
        return hashlib.md5(key_str.encode()).hexdigest()

    def clear_cache(self):
        """Clear the execution cache."""
        self.execution_cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.execution_cache),
            "cache_keys": list(self.execution_cache.keys()),
        }
