"""Loop handlers for for-each and while loops in pipelines."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import copy

from ..core.task import Task
from .auto_resolver import ControlFlowAutoResolver


@dataclass
class LoopContext:
    """Context for loop execution."""

    iteration: int = 0
    max_iterations: Optional[int] = None
    items: List[Any] = field(default_factory=list)
    current_item: Any = None
    current_index: int = 0
    loop_id: str = ""
    parent_context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert loop context to dictionary for template access."""
        return {
            "$iteration": self.iteration,
            "$index": self.current_index,
            "$item": self.current_item,
            "$items": self.items,
            "$loop_id": self.loop_id,
            "$is_first": self.current_index == 0,
            "$is_last": (
                self.current_index == len(self.items) - 1 if self.items else False
            ),
        }


class ForLoopHandler:
    """Handles for-each loop execution in pipelines."""

    def __init__(self, auto_resolver: Optional[ControlFlowAutoResolver] = None):
        """Initialize for loop handler.

        Args:
            auto_resolver: AUTO tag resolver
        """
        self.auto_resolver = auto_resolver or ControlFlowAutoResolver()

    async def expand_for_loop(
        self,
        loop_def: Dict[str, Any],
        context: Dict[str, Any],
        step_results: Dict[str, Any],
    ) -> List[Task]:
        """Expand a for-each loop into individual tasks.

        Args:
            loop_def: Loop definition from YAML
            context: Pipeline execution context
            step_results: Results from previous steps

        Returns:
            List of expanded tasks
        """
        # Extract loop configuration
        loop_id = loop_def.get("id", "loop")
        for_each_expr = loop_def.get("for_each", [])
        loop_var = loop_def.get("loop_var", "$item")
        max_parallel = loop_def.get("max_parallel", 1)

        # Resolve iterator
        items = await self.auto_resolver.resolve_iterator(
            for_each_expr, context, step_results
        )

        # Get loop body steps
        body_steps = loop_def.get("steps", [])
        if not body_steps and "action" in loop_def:
            # Single action shorthand
            body_steps = [
                {
                    "id": f"{loop_id}_item",
                    "action": loop_def["action"],
                    "parameters": loop_def.get("parameters", {}),
                }
            ]

        # Expand loop into tasks
        expanded_tasks = []

        for idx, item in enumerate(items):
            # Create loop context
            loop_context = LoopContext(
                iteration=idx,
                items=items,
                current_item=item,
                current_index=idx,
                loop_id=loop_id,
                parent_context=context,
            )

            # Process each step in loop body
            for step_def in body_steps:
                # Create unique task ID
                task_id = f"{loop_id}_{idx}_{step_def['id']}"

                # Deep copy step definition
                task_def = copy.deepcopy(step_def)
                task_def["id"] = task_id
                task_def["metadata"] = task_def.get("metadata", {})

                # Add loop context to metadata
                task_def["metadata"]["loop_context"] = loop_context.to_dict()
                task_def["metadata"]["loop_id"] = loop_id
                task_def["metadata"]["loop_index"] = idx
                
                # Preserve pipeline inputs in metadata
                if "inputs" in context:
                    task_def["metadata"]["pipeline_inputs"] = context["inputs"]
                elif isinstance(context, dict):
                    # Extract input parameters from context
                    pipeline_inputs = {}
                    for key, value in context.items():
                        if key not in ["all_step_ids", "_", "step_results"]:
                            pipeline_inputs[key] = value
                    if pipeline_inputs:
                        task_def["metadata"]["pipeline_inputs"] = pipeline_inputs

                # Process parameters with loop variables
                if "parameters" in task_def:
                    processed_params = await self._process_loop_variables(
                        task_def["parameters"],
                        loop_context,
                        context,
                        step_results,
                        loop_var,
                    )
                    task_def["parameters"] = processed_params

                # Adjust dependencies
                original_deps = task_def.get("dependencies", [])
                task_def["dependencies"] = []

                # Add dependencies from previous iteration if sequential
                if max_parallel == 1 and idx > 0:
                    prev_task_id = f"{loop_id}_{idx-1}_{step_def['id']}"
                    task_def["dependencies"].append(prev_task_id)

                # Add dependencies within same iteration
                for dep in original_deps:
                    if dep in [s["id"] for s in body_steps]:
                        # Internal dependency
                        task_def["dependencies"].append(f"{loop_id}_{idx}_{dep}")
                    else:
                        # External dependency
                        task_def["dependencies"].append(dep)

                # Create task
                expanded_tasks.append(self._create_task_from_def(task_def))

        # Add a completion task to represent the whole for_each loop
        if expanded_tasks:
            # Get all the last tasks from each iteration
            last_task_ids = []
            for idx in range(len(items)):
                if body_steps:
                    last_step_id = body_steps[-1]["id"]
                    last_task_ids.append(f"{loop_id}_{idx}_{last_step_id}")
            
            # Create completion task that depends on all iterations
            completion_task = Task(
                id=loop_id,  # Use the original loop ID
                name=f"Complete {loop_id} loop",
                action="loop_complete",
                parameters={"loop_id": loop_id, "iterations": len(items)},
                dependencies=last_task_ids,
                metadata={
                    "is_loop_completion": True,
                    "loop_id": loop_id,
                    "control_flow_type": "for_each",
                },
            )
            expanded_tasks.append(completion_task)

        return expanded_tasks

    async def _process_loop_variables(
        self,
        obj: Any,
        loop_context: LoopContext,
        context: Dict[str, Any],
        step_results: Dict[str, Any],
        loop_var: str = "$item",
    ) -> Any:
        """Process loop variables in parameters.

        Args:
            obj: Object to process (dict, list, or string)
            loop_context: Current loop context
            context: Pipeline context
            step_results: Step results
            loop_var: Custom loop variable name

        Returns:
            Processed object with loop variables replaced
        """
        if isinstance(obj, str):
            # Replace loop variables
            result = obj

            # First, replace custom loop variable
            if loop_var != "$item":
                # Handle both {{var}} and {{ var }} formats
                result = result.replace(
                    f"{{{{{loop_var}}}}}", str(loop_context.current_item)
                )
                result = result.replace(
                    f"{{{{ {loop_var} }}}}", str(loop_context.current_item)
                )

            # Then replace standard loop variables
            for var_name, var_value in loop_context.to_dict().items():
                # Handle both {{var}} and {{ var }} formats
                result = result.replace(f"{{{{{var_name}}}}}", str(var_value))
                result = result.replace(f"{{{{ {var_name} }}}}", str(var_value))

            # Resolve AUTO tags with loop context
            if "<AUTO>" in result:
                enhanced_context = context.copy()
                enhanced_context.update(loop_context.to_dict())
                # Also add custom loop var to context
                if loop_var != "$item":
                    enhanced_context[loop_var] = loop_context.current_item
                result = await self.auto_resolver._resolve_auto_tags(
                    result, enhanced_context, step_results
                )

            return result

        elif isinstance(obj, dict):
            return {
                k: await self._process_loop_variables(
                    v, loop_context, context, step_results, loop_var
                )
                for k, v in obj.items()
            }

        elif isinstance(obj, list):
            return [
                await self._process_loop_variables(
                    item, loop_context, context, step_results, loop_var
                )
                for item in obj
            ]

        return obj

    def _create_task_from_def(self, task_def: Dict[str, Any]) -> Task:
        """Create a Task instance from definition.

        Args:
            task_def: Task definition

        Returns:
            Task instance
        """
        # Ensure metadata includes the tool field if present
        metadata = task_def.get("metadata", {})
        if "tool" in task_def:
            metadata["tool"] = task_def["tool"]
            
        return Task(
            id=task_def["id"],
            name=task_def.get("name", task_def["id"]),
            action=task_def["action"],
            parameters=task_def.get("parameters", {}),
            dependencies=task_def.get("dependencies", []),
            metadata=metadata,
            timeout=task_def.get("timeout"),
            max_retries=task_def.get("max_retries", 3),
        )


class WhileLoopHandler:
    """Handles while loop execution in pipelines."""

    def __init__(self, auto_resolver: Optional[ControlFlowAutoResolver] = None):
        """Initialize while loop handler.

        Args:
            auto_resolver: AUTO tag resolver
        """
        self.auto_resolver = auto_resolver or ControlFlowAutoResolver()
        self.loop_states = {}  # Track loop state across iterations

    async def should_continue(
        self,
        loop_id: str,
        condition: str,
        context: Dict[str, Any],
        step_results: Dict[str, Any],
        iteration: int,
        max_iterations: int,
    ) -> bool:
        """Check if while loop should continue.

        Args:
            loop_id: Unique loop identifier
            condition: Loop condition potentially with AUTO tags
            context: Pipeline execution context
            step_results: Results from previous steps
            iteration: Current iteration number
            max_iterations: Maximum allowed iterations

        Returns:
            True if loop should continue
        """
        # Check max iterations
        if iteration >= max_iterations:
            return False

        # Get loop state
        loop_state = self.loop_states.get(loop_id, {})

        # Build enhanced context with loop info
        enhanced_context = context.copy()
        enhanced_context.update(
            {
                "$iteration": iteration,
                "$loop_state": loop_state,
                "current_result": (
                    step_results.get(f"{loop_id}_{iteration-1}_result")
                    if iteration > 0
                    else None
                ),
            }
        )

        # Evaluate condition
        cache_key = f"{loop_id}_iter_{iteration}"
        result = await self.auto_resolver.resolve_condition(
            condition, enhanced_context, step_results, cache_key
        )

        return result

    async def create_iteration_tasks(
        self,
        loop_def: Dict[str, Any],
        iteration: int,
        context: Dict[str, Any],
        step_results: Dict[str, Any],
    ) -> List[Task]:
        """Create tasks for a single while loop iteration.

        Args:
            loop_def: While loop definition
            iteration: Current iteration number
            context: Pipeline execution context
            step_results: Results from previous steps

        Returns:
            List of tasks for this iteration
        """
        loop_id = loop_def.get("id", "while_loop")
        body_steps = loop_def.get("steps", [])

        # Get or create loop state
        if loop_id not in self.loop_states:
            self.loop_states[loop_id] = {}

        loop_state = self.loop_states[loop_id]

        # Create tasks for this iteration
        iteration_tasks = []

        for step_def in body_steps:
            # Create unique task ID
            task_id = f"{loop_id}_{iteration}_{step_def['id']}"

            # Deep copy step definition
            task_def = copy.deepcopy(step_def)
            task_def["id"] = task_id
            task_def["metadata"] = task_def.get("metadata", {})

            # Add loop metadata
            task_def["metadata"]["loop_id"] = loop_id
            task_def["metadata"]["loop_iteration"] = iteration
            task_def["metadata"]["is_while_loop"] = True

            # Process parameters with loop variables
            if "parameters" in task_def:
                enhanced_context = context.copy()
                enhanced_context.update(
                    {"$iteration": iteration, "$loop_state": loop_state}
                )

                task_def["parameters"] = await self._process_loop_params(
                    task_def["parameters"], enhanced_context, step_results
                )

            # Adjust dependencies
            original_deps = task_def.get("dependencies", [])
            task_def["dependencies"] = []

            # Add dependencies from previous iteration
            if iteration > 0:
                for dep in original_deps:
                    if dep in [s["id"] for s in body_steps]:
                        # Add dependency from previous iteration
                        prev_task_id = f"{loop_id}_{iteration-1}_{dep}"
                        task_def["dependencies"].append(prev_task_id)

            # Add dependencies within same iteration
            for dep in original_deps:
                if dep in [s["id"] for s in body_steps]:
                    task_def["dependencies"].append(f"{loop_id}_{iteration}_{dep}")
                else:
                    # External dependency (only for first iteration)
                    if iteration == 0:
                        task_def["dependencies"].append(dep)

            # Create task
            iteration_tasks.append(self._create_task_from_def(task_def))

        # Add a sentinel task to capture iteration result
        result_task = Task(
            id=f"{loop_id}_{iteration}_result",
            name=f"Capture {loop_id} iteration {iteration} result",
            action="capture_result",
            parameters={"loop_id": loop_id, "iteration": iteration},
            dependencies=[t.id for t in iteration_tasks],
            metadata={
                "is_loop_result": True,
                "loop_id": loop_id,
                "iteration": iteration,
            },
        )
        iteration_tasks.append(result_task)

        return iteration_tasks

    async def _process_loop_params(
        self, obj: Any, context: Dict[str, Any], step_results: Dict[str, Any]
    ) -> Any:
        """Process parameters with loop context.

        Args:
            obj: Parameter object
            context: Enhanced context with loop info
            step_results: Step results

        Returns:
            Processed parameters
        """
        if isinstance(obj, str):
            # Replace loop variables
            result = obj
            result = result.replace("{{$iteration}}", str(context.get("$iteration", 0)))

            # Handle loop state references
            if "{{$loop_state" in result:
                import re

                state_pattern = r"\{\{\$loop_state\.([^}]+)\}\}"

                def replace_state(match):
                    key = match.group(1)
                    loop_state = context.get("$loop_state", {})
                    return str(loop_state.get(key, ""))

                result = re.sub(state_pattern, replace_state, result)

            # Resolve AUTO tags
            if "<AUTO>" in result:
                result = await self.auto_resolver._resolve_auto_tags(
                    result, context, step_results
                )

            return result

        elif isinstance(obj, dict):
            return {
                k: await self._process_loop_params(v, context, step_results)
                for k, v in obj.items()
            }

        elif isinstance(obj, list):
            return [
                await self._process_loop_params(item, context, step_results)
                for item in obj
            ]

        return obj

    def update_loop_state(self, loop_id: str, key: str, value: Any):
        """Update while loop state.

        Args:
            loop_id: Loop identifier
            key: State key
            value: State value
        """
        if loop_id not in self.loop_states:
            self.loop_states[loop_id] = {}
        self.loop_states[loop_id][key] = value

    def get_loop_state(self, loop_id: str) -> Dict[str, Any]:
        """Get while loop state.

        Args:
            loop_id: Loop identifier

        Returns:
            Loop state dictionary
        """
        return self.loop_states.get(loop_id, {})

    def _create_task_from_def(self, task_def: Dict[str, Any]) -> Task:
        """Create a Task instance from definition.

        Args:
            task_def: Task definition

        Returns:
            Task instance
        """
        # Ensure metadata includes the tool field if present
        metadata = task_def.get("metadata", {})
        if "tool" in task_def:
            metadata["tool"] = task_def["tool"]
            
        return Task(
            id=task_def["id"],
            name=task_def.get("name", task_def["id"]),
            action=task_def["action"],
            parameters=task_def.get("parameters", {}),
            dependencies=task_def.get("dependencies", []),
            metadata=metadata,
            timeout=task_def.get("timeout"),
            max_retries=task_def.get("max_retries", 3),
        )
