"""Dynamic flow control for goto and dynamic dependencies."""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass

from ..core.task import Task
from .auto_resolver import ControlFlowAutoResolver


@dataclass
class FlowJump:
    """Represents a flow control jump."""

    from_task: str
    to_task: str
    condition: Optional[str] = None
    is_resolved: bool = False


class DynamicFlowHandler:
    """Handles dynamic flow control including goto and dynamic dependencies."""

    def __init__(self, auto_resolver: Optional[ControlFlowAutoResolver] = None):
        """Initialize dynamic flow handler.

        Args:
            auto_resolver: AUTO tag resolver
        """
        self.auto_resolver = auto_resolver or ControlFlowAutoResolver()
        self.flow_jumps: Dict[str, FlowJump] = {}
        self.dynamic_dependencies: Dict[str, List[str]] = {}

    async def process_goto(
        self,
        task: Task,
        context: Dict[str, Any],
        step_results: Dict[str, Any],
        all_tasks: Dict[str, Task],
    ) -> Optional[str]:
        """Process goto directive in a task.

        Args:
            task: Task potentially containing goto
            context: Pipeline execution context
            step_results: Results from previous steps
            all_tasks: All tasks in the pipeline

        Returns:
            Target task ID to jump to, or None
        """
        # Check for goto in task metadata
        goto_target = None
        if hasattr(task, "metadata") and "goto" in task.metadata:
            goto_target = task.metadata["goto"]
        elif hasattr(task, "goto"):
            goto_target = task.goto

        if not goto_target:
            return None

        # Resolve AUTO tags in goto target
        valid_targets = list(all_tasks.keys())
        resolved_target = await self.auto_resolver.resolve_target(
            goto_target, context, step_results, valid_targets
        )

        # Record jump
        jump = FlowJump(from_task=task.id, to_task=resolved_target, is_resolved=True)
        self.flow_jumps[task.id] = jump

        return resolved_target

    async def resolve_dynamic_dependencies(
        self,
        task: Task,
        context: Dict[str, Any],
        step_results: Dict[str, Any],
        all_tasks: Dict[str, Task],
    ) -> List[str]:
        """Resolve dynamic dependencies for a task.

        Args:
            task: Task with potential dynamic dependencies
            context: Pipeline execution context
            step_results: Results from previous steps
            all_tasks: All tasks in the pipeline

        Returns:
            List of resolved dependency task IDs
        """
        # Check for dynamic dependencies
        dynamic_deps = None
        if hasattr(task, "metadata") and "dynamic_dependencies" in task.metadata:
            dynamic_deps = task.metadata["dynamic_dependencies"]
        elif hasattr(task, "dynamic_dependencies"):
            dynamic_deps = task.dynamic_dependencies

        if not dynamic_deps:
            # Return static dependencies
            return list(task.dependencies)

        # Resolve AUTO tags in dependencies
        valid_steps = list(all_tasks.keys())
        resolved_deps = await self.auto_resolver.resolve_dependencies(
            dynamic_deps, context, step_results, valid_steps
        )

        # Combine with static dependencies
        all_deps = list(task.dependencies)
        for dep in resolved_deps:
            if dep not in all_deps:
                all_deps.append(dep)

        # Cache resolved dependencies
        self.dynamic_dependencies[task.id] = resolved_deps

        return all_deps

    def apply_flow_jumps(
        self, execution_order: List[str], completed_tasks: Set[str]
    ) -> List[str]:
        """Apply flow jumps to modify execution order.

        Args:
            execution_order: Original execution order
            completed_tasks: Set of completed task IDs

        Returns:
            Modified execution order considering jumps
        """
        modified_order = []
        skip_until = None

        for task_id in execution_order:
            # Check if we should skip
            if skip_until and task_id != skip_until:
                continue
            elif skip_until and task_id == skip_until:
                skip_until = None

            # Check if this task has a jump
            if task_id in self.flow_jumps and task_id in completed_tasks:
                jump = self.flow_jumps[task_id]
                if jump.is_resolved:
                    # Skip to target
                    skip_until = jump.to_task
                    continue

            modified_order.append(task_id)

        return modified_order

    def get_dynamic_execution_graph(
        self, tasks: Dict[str, Task], completed_tasks: Set[str]
    ) -> Dict[str, List[str]]:
        """Build execution graph considering dynamic dependencies.

        Args:
            tasks: All tasks in pipeline
            completed_tasks: Set of completed task IDs

        Returns:
            Dependency graph with dynamic dependencies resolved
        """
        graph = {}

        for task_id, task in tasks.items():
            # Skip already completed tasks
            if task_id in completed_tasks:
                graph[task_id] = []
                continue

            # Get dependencies (static + dynamic)
            deps = list(task.dependencies)

            # Add resolved dynamic dependencies
            if task_id in self.dynamic_dependencies:
                for dep in self.dynamic_dependencies[task_id]:
                    if dep not in deps:
                        deps.append(dep)

            # Filter out completed dependencies
            active_deps = [d for d in deps if d not in completed_tasks]
            graph[task_id] = active_deps

        return graph

    async def process_step_flow_control(
        self,
        step_def: Dict[str, Any],
        context: Dict[str, Any],
        step_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process flow control directives in step definition.

        Args:
            step_def: Step definition from YAML
            context: Pipeline execution context
            step_results: Results from previous steps

        Returns:
            Processed step definition
        """
        # Process goto
        if "goto" in step_def:
            goto_expr = step_def["goto"]

            # Store goto expression in metadata for runtime processing
            if "metadata" not in step_def:
                step_def["metadata"] = {}
            step_def["metadata"]["goto"] = goto_expr
            
            # For non-AUTO goto, keep it at top level for backward compatibility
            # For AUTO tags, remove from top level to prevent early resolution
            if isinstance(goto_expr, str) and "<AUTO>" in goto_expr:
                del step_def["goto"]

        # Process dynamic dependencies
        if "depends_on" in step_def:
            deps = step_def["depends_on"]

            # Check if dependencies contain AUTO tags
            has_auto = False
            if isinstance(deps, str) and "<AUTO>" in deps:
                has_auto = True
            elif isinstance(deps, list):
                for dep in deps:
                    if isinstance(dep, str) and "<AUTO>" in dep:
                        has_auto = True
                        break

            if has_auto:
                # Mark as dynamic and store original expression
                if "metadata" not in step_def:
                    step_def["metadata"] = {}
                step_def["metadata"]["dynamic_dependencies"] = deps
                # Keep minimal static dependencies for initial graph
                step_def["depends_on"] = []

        return step_def

    def validate_jumps(self, all_tasks: Dict[str, Task]) -> List[str]:
        """Validate all flow jumps.

        Args:
            all_tasks: All tasks in pipeline

        Returns:
            List of validation errors
        """
        errors = []

        for task_id, jump in self.flow_jumps.items():
            # Check source exists
            if jump.from_task not in all_tasks:
                errors.append(f"Jump source '{jump.from_task}' does not exist")

            # Check target exists
            if jump.to_task not in all_tasks:
                errors.append(f"Jump target '{jump.to_task}' does not exist")

            # Check for circular jumps
            if jump.to_task == jump.from_task:
                errors.append(f"Circular jump detected in task '{jump.from_task}'")

        return errors

    def create_dynamic_task(self, task_def: Dict[str, Any]) -> Task:
        """Create a task with dynamic flow control support.

        Args:
            task_def: Task definition

        Returns:
            Task instance with flow control metadata
        """
        # Extract flow control properties
        goto = task_def.get("goto")
        dynamic_deps = task_def.get("metadata", {}).get("dynamic_dependencies")

        # Create task
        task = Task(
            id=task_def["id"],
            name=task_def.get("name", task_def["id"]),
            action=task_def.get("action", "execute"),
            parameters=task_def.get("parameters", {}),
            dependencies=task_def.get("depends_on", []),
            metadata=task_def.get("metadata", {}),
            timeout=task_def.get("timeout"),
            max_retries=task_def.get("max_retries", 3),
        )

        # Add flow control metadata
        if goto:
            task.metadata["goto"] = goto
        if dynamic_deps:
            task.metadata["dynamic_dependencies"] = dynamic_deps

        return task
