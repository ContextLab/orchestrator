"""Conditional execution handler for pipelines."""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass

from ..core.task import Task, TaskStatus
from .auto_resolver import ControlFlowAutoResolver


@dataclass
class ConditionalTask(Task):
    """Task with conditional execution support."""

    condition: Optional[str] = None
    condition_cache_key: Optional[str] = None
    else_task_id: Optional[str] = None

    def should_execute(
        self,
        context: Dict[str, Any],
        step_results: Dict[str, Any],
        resolver: ControlFlowAutoResolver,
    ) -> bool:
        """Check if task should execute based on condition.

        Args:
            context: Pipeline execution context
            step_results: Results from previous steps
            resolver: AUTO tag resolver

        Returns:
            True if task should execute
        """
        if not self.condition:
            return True

        # Synchronous wrapper for async resolution
        import asyncio

        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're already in an async context
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    resolver.resolve_condition(
                        self.condition, context, step_results, self.condition_cache_key
                    ),
                )
                return future.result()
        else:
            # We can create a new event loop
            return asyncio.run(
                resolver.resolve_condition(
                    self.condition, context, step_results, self.condition_cache_key
                )
            )


class ConditionalHandler:
    """Handles conditional execution of pipeline steps."""

    def __init__(self, auto_resolver: Optional[ControlFlowAutoResolver] = None):
        """Initialize conditional handler.

        Args:
            auto_resolver: AUTO tag resolver for conditions
        """
        self.auto_resolver = auto_resolver or ControlFlowAutoResolver()

    async def evaluate_condition(
        self, task: Task, context: Dict[str, Any], step_results: Dict[str, Any]
    ) -> bool:
        """Evaluate whether a task should execute.

        Args:
            task: Task to evaluate
            context: Pipeline execution context
            step_results: Results from previous steps

        Returns:
            True if task should execute
        """
        # Check if task has condition attribute
        condition = getattr(task, "condition", None)
        if hasattr(task, "metadata") and "condition" in task.metadata:
            condition = task.metadata["condition"]

        if not condition:
            return True

        # Get cache key if available
        cache_key = None
        if hasattr(task, "condition_cache_key"):
            cache_key = task.condition_cache_key
        elif hasattr(task, "metadata") and "condition_cache_key" in task.metadata:
            cache_key = task.metadata["condition_cache_key"]

        # Resolve and evaluate condition
        return await self.auto_resolver.resolve_condition(
            condition, context, step_results, cache_key
        )

    def get_next_task(
        self, task: Task, condition_result: bool, pipeline_tasks: Dict[str, Task]
    ) -> Optional[str]:
        """Get next task ID based on condition result.

        Args:
            task: Current task
            condition_result: Result of condition evaluation
            pipeline_tasks: All tasks in pipeline

        Returns:
            Next task ID to execute, or None
        """
        if condition_result:
            # Condition is true, execute normally
            return None

        # Check for else branch
        else_task_id = None
        if hasattr(task, "else_task_id"):
            else_task_id = task.else_task_id
        elif hasattr(task, "metadata") and "else_task_id" in task.metadata:
            else_task_id = task.metadata["else_task_id"]

        if else_task_id and else_task_id in pipeline_tasks:
            return else_task_id

        # No else branch, skip task
        return None

    def create_conditional_task(self, task_def: Dict[str, Any]) -> ConditionalTask:
        """Create a conditional task from definition.

        Args:
            task_def: Task definition dictionary

        Returns:
            ConditionalTask instance
        """
        # Extract conditional properties
        condition = task_def.get("if") or task_def.get("condition")
        else_task_id = task_def.get("else")
        condition_cache_key = task_def.get("condition_cache_key")

        # Create base task parameters
        task_params = {
            "id": task_def["id"],
            "name": task_def.get("name", task_def["id"]),
            "action": task_def.get("action", "conditional"),
            "parameters": task_def.get("parameters", {}),
            "dependencies": task_def.get("depends_on", []),
            "metadata": task_def.get("metadata", {}),
            "timeout": task_def.get("timeout"),
            "max_retries": task_def.get("max_retries", 3),
        }

        # Add conditional properties to metadata
        if condition:
            task_params["metadata"]["condition"] = condition
        if else_task_id:
            task_params["metadata"]["else_task_id"] = else_task_id
        if condition_cache_key:
            task_params["metadata"]["condition_cache_key"] = condition_cache_key

        # Create conditional task
        return ConditionalTask(
            **task_params,
            condition=condition,
            condition_cache_key=condition_cache_key,
            else_task_id=else_task_id,
        )

    def get_conditional_dependencies(
        self, tasks: Dict[str, Task], completed_tasks: Set[str]
    ) -> Dict[str, List[str]]:
        """Get adjusted dependencies considering conditional execution.

        Args:
            tasks: All tasks in pipeline
            completed_tasks: Set of completed task IDs

        Returns:
            Adjusted dependencies map
        """
        adjusted_deps = {}

        for task_id, task in tasks.items():
            deps = list(task.dependencies)

            # Check if any dependencies were skipped due to conditions
            for dep_id in task.dependencies:
                if dep_id not in completed_tasks and dep_id in tasks:
                    dep_task = tasks[dep_id]
                    if dep_task.status == TaskStatus.SKIPPED:
                        # Remove skipped dependency
                        deps.remove(dep_id)

                        # Add the skipped task's dependencies instead
                        for trans_dep in dep_task.dependencies:
                            if trans_dep not in deps:
                                deps.append(trans_dep)

            adjusted_deps[task_id] = deps

        return adjusted_deps

    async def process_conditional_step(
        self, step_def: Dict[str, Any], context: Dict[str, Any], step_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a step definition with conditional execution.

        Args:
            step_def: Step definition from YAML
            context: Pipeline execution context
            step_results: Results from previous steps

        Returns:
            Processed step definition with resolved conditions
        """
        # Check for condition
        if "if" in step_def or "condition" in step_def:
            condition = step_def.get("if") or step_def.get("condition")

            # Resolve AUTO tags in condition
            if isinstance(condition, str) and "<AUTO>" in condition:
                resolved_condition = await self.auto_resolver._resolve_auto_tags(
                    condition, context, step_results
                )
                step_def["condition"] = resolved_condition

        # Check for else reference
        if "else" in step_def:
            else_ref = step_def["else"]

            # Resolve AUTO tags in else reference
            if isinstance(else_ref, str) and "<AUTO>" in else_ref:
                valid_targets = list(context.get("all_step_ids", []))
                resolved_else = await self.auto_resolver.resolve_target(
                    else_ref, context, step_results, valid_targets
                )
                step_def["else"] = resolved_else

        return step_def
