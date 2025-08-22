"""Loop handlers for for-each and while loops in pipelines."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import copy

from ..core.task import Task
from ..core.loop_context import LoopContextVariables, GlobalLoopContextManager
from .auto_resolver import ControlFlowAutoResolver
from .enhanced_condition_evaluator import EnhancedConditionEvaluator


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

    def __init__(self, auto_resolver: Optional[ControlFlowAutoResolver] = None, loop_context_manager: Optional[GlobalLoopContextManager] = None):
        """Initialize for loop handler.

        Args:
            auto_resolver: AUTO tag resolver
            loop_context_manager: Global loop context manager for named loops
        """
        self.auto_resolver = auto_resolver or ControlFlowAutoResolver()
        self.loop_context_manager = loop_context_manager or GlobalLoopContextManager()

    async def expand_for_loop(
        self,
        loop_def: Dict[str, Any],
        context: Dict[str, Any],
        step_results: Dict[str, Any],
    ) -> List[Task]:
        """Expand a for-each loop into individual tasks with named loop support.

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
        explicit_loop_name = loop_def.get("loop_name")  # Optional explicit name
        loop_var = loop_def.get("loop_var", "$item")  # Custom loop variable name
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
            # Create named loop context with auto-generation if needed
            loop_context = self.loop_context_manager.create_loop_context(
                step_id=loop_id,
                item=item,
                index=idx,
                items=items,
                explicit_loop_name=explicit_loop_name
            )
            
            # Push loop context (available to all nested operations)
            self.loop_context_manager.push_loop(loop_context)
            
            try:
                # Process loop body with named loop context
                for step_def in body_steps:
                    task_id = f"{loop_id}_{idx}_{step_def['id']}"
                    
                    # Create enhanced context with all loop variables
                    enhanced_context = context.copy()
                    enhanced_context.update(step_results)
                    enhanced_context.update(self.loop_context_manager.get_accessible_loop_variables())
                    
                    # Add custom loop variable
                    if loop_var != "$item":
                        # Remove the $ prefix if present for template usage
                        var_name = loop_var.lstrip('$') if loop_var.startswith('$') else loop_var
                        enhanced_context[var_name] = item
                    
                    # Deep copy step definition
                    task_def = copy.deepcopy(step_def)
                    task_def["id"] = task_id
                    task_def["metadata"] = task_def.get("metadata", {})
                    
                    # Add loop metadata
                    task_def["metadata"]["loop_context"] = loop_context.get_debug_info()
                    task_def["metadata"]["loop_id"] = loop_id
                    task_def["metadata"]["loop_name"] = loop_context.loop_name
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
                    
                    # Process parameters with all accessible loop variables
                    if "parameters" in task_def:
                        task_def["parameters"] = await self._process_templates_with_named_loops(
                            task_def["parameters"], enhanced_context
                        )
                    
                    # Process action field if it contains templates
                    if "action" in task_def:
                        task_def["action"] = await self._process_templates_with_named_loops(
                            task_def["action"], enhanced_context
                        )
                    
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
                    
                    # Handle nested loops
                    if "for_each" in task_def or "while" in task_def:
                        # Recursive loop expansion with current context
                        nested_tasks = await self.expand_for_loop(task_def, enhanced_context, step_results)
                        expanded_tasks.extend(nested_tasks)
                    else:
                        # Regular task
                        expanded_tasks.append(self._create_task_from_def(task_def))
                        
            finally:
                # Keep loop in history but remove from active
                # (unless we need it for cross-step access)
                if loop_context.is_auto_generated or not self._has_active_nested_loops(loop_context.loop_name):
                    self.loop_context_manager.pop_loop(loop_context.loop_name)

        # Add a completion task to represent the whole for_each loop
        # Only add completion task if explicitly requested or if there are multiple steps per iteration
        add_completion_task = loop_def.get("add_completion_task", len(body_steps) > 1)
        
        if expanded_tasks and add_completion_task:
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
    
    async def _process_templates_with_named_loops(self, obj: Any, context: Dict[str, Any]) -> Any:
        """Process templates with named loop variable support.
        
        Args:
            obj: Object to process (dict, list, or string)
            context: Enhanced context with all loop variables
            
        Returns:
            Processed object with all templates rendered
        """
        if isinstance(obj, str):
            # Handle named loop variable references and AUTO tags
            result = obj
            
            # Process named loop variables ($loop_name.variable)
            for loop_name, loop_context in self.loop_context_manager.active_loops.items():
                loop_vars = loop_context.to_template_dict()
                for var_name, var_value in loop_vars.items():
                    if var_name.startswith(f'${loop_name}.'):
                        # Replace ${loop_name}.variable with actual value
                        result = result.replace(f"{{{{{var_name}}}}}", str(var_value))
                        result = result.replace(f"{{{{ {var_name} }}}}", str(var_value))
            
            # Process default loop variables ($item, $index, etc.)
            current_loop = self.loop_context_manager.get_current_loop()
            if current_loop:
                default_vars = current_loop.to_template_dict(is_current_loop=True)
                for var_name, var_value in default_vars.items():
                    if not '.' in var_name:  # Only process default $ variables
                        result = result.replace(f"{{{{{var_name}}}}}", str(var_value))
                        result = result.replace(f"{{{{ {var_name} }}}}", str(var_value))
            
            # Process variables directly in context (like custom loop variables)
            for var_name, var_value in context.items():
                if isinstance(var_value, (str, int, float, bool)) and not var_name.startswith('_'):
                    # Replace both {{ var }} and {{var}} patterns
                    result = result.replace(f"{{{{ {var_name} }}}}", str(var_value))
                    result = result.replace(f"{{{{{var_name}}}}}", str(var_value))
            
            # Handle AUTO tags with enhanced context
            if "<AUTO>" in result:
                result = await self.auto_resolver._resolve_auto_tags(result, context, {})
            
            return result
            
        elif isinstance(obj, dict):
            return {k: await self._process_templates_with_named_loops(v, context) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [await self._process_templates_with_named_loops(item, context) for item in obj]
        
        return obj
    
    def _has_active_nested_loops(self, loop_name: str) -> bool:
        """Check if any nested loops are still active that depend on this loop."""
        # For now, simple implementation - can be enhanced later
        # Auto-generated loops don't need to persist for cross-step access
        return False

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

    def __init__(self, auto_resolver: Optional[ControlFlowAutoResolver] = None, loop_context_manager: Optional[GlobalLoopContextManager] = None):
        """Initialize while loop handler.

        Args:
            auto_resolver: AUTO tag resolver
            loop_context_manager: Global loop context manager for named loops
        """
        self.auto_resolver = auto_resolver or ControlFlowAutoResolver()
        self.loop_context_manager = loop_context_manager or GlobalLoopContextManager()
        self.loop_states = {}  # Track loop state across iterations
        
        # Enhanced condition evaluator with structured evaluation
        self.enhanced_evaluator = EnhancedConditionEvaluator(self.auto_resolver)

    async def should_continue(
        self,
        loop_id: str,
        condition: str,
        context: Dict[str, Any],
        step_results: Dict[str, Any],
        iteration: int,
        max_iterations: int,
        until_condition: Optional[str] = None,
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
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"WhileLoopHandler.should_continue: loop_id={loop_id}, iteration={iteration}, max_iterations={max_iterations}")
        if iteration >= max_iterations:
            logger.info(f"Stopping loop {loop_id}: iteration {iteration} >= max_iterations {max_iterations}")
            return False

        # Get loop state
        loop_state = self.loop_states.get(loop_id, {})

        # Build enhanced context with loop info
        enhanced_context = context.copy()
        enhanced_context.update(
            {
                "$iteration": iteration,
                "iteration": iteration,  # Also available without $ for Jinja2 templates
                "$loop_state": loop_state,
                "loop_state": loop_state,  # Also available without $ for Jinja2 templates
                "current_result": (
                    step_results.get(f"{loop_id}_{iteration-1}_result")
                    if iteration > 0
                    else None
                ),
            }
        )

        # First, render the condition template if it contains templates
        rendered_condition = condition
        if "{{" in condition and "}}" in condition:
            # Use template manager to render the condition
            template_manager = context.get("template_manager") or context.get("_template_manager")
            if template_manager:
                try:
                    # Pass step_results and enhanced context for rendering
                    all_context = {**step_results, **enhanced_context}
                    rendered_condition = template_manager.render(condition, additional_context=all_context)
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"While loop {loop_id}: Rendered condition from '{condition}' to '{rendered_condition}'")
                except Exception as e:
                    # Log the error but try to continue with original condition
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to render while loop condition template: {e}")
            else:
                # Simple template replacement when no template manager
                rendered_condition = self._simple_template_replace(condition, enhanced_context)
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"While loop {loop_id}: Simple template rendering: '{condition}' -> '{rendered_condition}'")
        
        # Evaluate while condition (continue when true)
        cache_key = f"{loop_id}_iter_{iteration}"
        while_result = await self.auto_resolver.resolve_condition(
            rendered_condition, enhanced_context, step_results, cache_key
        )
        
        logger.info(f"WhileLoopHandler.should_continue: while condition '{rendered_condition}' evaluated to {while_result}")
        
        # If while condition is false, stop loop
        if not while_result:
            logger.info(f"Stopping loop {loop_id}: while condition is false")
            return False
        
        # If we have an until condition, evaluate it (stop when true)
        if until_condition:
            # Render until condition template
            rendered_until = until_condition
            if "{{" in until_condition and "}}" in until_condition:
                template_manager = context.get("template_manager") or context.get("_template_manager")
                if template_manager:
                    try:
                        all_context = {**step_results, **enhanced_context}
                        rendered_until = template_manager.render(until_condition, additional_context=all_context)
                        logger.info(f"Until loop {loop_id}: Rendered until condition from '{until_condition}' to '{rendered_until}'")
                    except Exception as e:
                        logger.warning(f"Failed to render until condition template: {e}")
                else:
                    # Simple template replacement when no template manager
                    rendered_until = self._simple_template_replace(until_condition, enhanced_context)
                    logger.info(f"Until loop {loop_id}: Simple template rendering: '{until_condition}' -> '{rendered_until}'")
            
            # Evaluate until condition
            until_cache_key = f"{loop_id}_until_iter_{iteration}"
            until_result = await self.auto_resolver.resolve_condition(
                rendered_until, enhanced_context, step_results, until_cache_key
            )
            
            logger.info(f"WhileLoopHandler.should_continue: until condition '{rendered_until}' evaluated to {until_result}")
            
            # If until condition is true, stop loop
            if until_result:
                logger.info(f"Stopping loop {loop_id}: until condition is satisfied")
                return False
        
        # Continue loop (while=true, until=false or not present)
        return True


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
        explicit_loop_name = loop_def.get("loop_name")  # Optional explicit name
        body_steps = loop_def.get("steps", [])

        # Get or create loop state
        if loop_id not in self.loop_states:
            self.loop_states[loop_id] = {}

        loop_state = self.loop_states[loop_id]
        
        # Create while loop context (no items, just iteration tracking)
        while_context = self.loop_context_manager.create_loop_context(
            step_id=loop_id,
            item=None,  # While loops don't have items
            index=iteration,
            items=[],  # Empty items list for while loops
            explicit_loop_name=explicit_loop_name
        )
        
        # Push while loop context
        self.loop_context_manager.push_loop(while_context)

        # Create tasks for this iteration
        iteration_tasks = []

        try:
            for step_def in body_steps:
                # Create unique task ID
                task_id = f"{loop_id}_{iteration}_{step_def['id']}"

                # Create enhanced context with all loop variables
                enhanced_context = context.copy()
                enhanced_context.update(step_results)
                enhanced_context.update(self.loop_context_manager.get_accessible_loop_variables())
                enhanced_context.update({
                    "$loop_state": loop_state,
                    "loop_state": loop_state,  # Also available without $ for Jinja2
                    "$iteration": iteration,
                    "iteration": iteration,  # Also available without $ for Jinja2
                    "loop_id": loop_id
                })

                # Deep copy step definition
                task_def = copy.deepcopy(step_def)
                task_def["id"] = task_id
                task_def["metadata"] = task_def.get("metadata", {})
                
                # Remove any while loop specific metadata that shouldn't be inherited
                task_def["metadata"].pop("is_while_loop", None)
                task_def["metadata"].pop("while_condition", None)
                task_def["metadata"].pop("while", None)
                task_def["metadata"].pop("max_iterations", None)

                # Add loop metadata
                task_def["metadata"]["loop_id"] = loop_id
                task_def["metadata"]["loop_name"] = while_context.loop_name
                task_def["metadata"]["loop_iteration"] = iteration
                task_def["metadata"]["is_while_loop_child"] = True
                task_def["metadata"]["loop_context"] = while_context.get_debug_info()
                # Add loop variables for template rendering
                task_def["metadata"]["loop_variables"] = {
                    "$iteration": iteration,
                    "iteration": iteration,
                    "$loop_state": loop_state,
                    "loop_state": loop_state,
                    "loop_id": loop_id
                }
                
                # Process action field if it contains templates
                if "action" in task_def:
                    task_def["action"] = await self._process_templates_with_named_loops(
                        task_def["action"], enhanced_context
                    )
                
                # Process parameters with all accessible loop variables
                if "parameters" in task_def:
                    task_def["parameters"] = await self._process_templates_with_named_loops(
                        task_def["parameters"], enhanced_context
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
                    "loop_name": while_context.loop_name,
                    "iteration": iteration,
                },
            )
            iteration_tasks.append(result_task)
            
        finally:
            # Remove from active loops after processing
            if while_context.is_auto_generated or not self._has_active_nested_loops(while_context.loop_name):
                self.loop_context_manager.pop_loop(while_context.loop_name)

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
            # Replace loop variables - handle both $iteration and iteration syntax
            result = result.replace("{{$iteration}}", str(context.get("$iteration", 0)))
            result = result.replace("{{ $iteration }}", str(context.get("$iteration", 0)))
            result = result.replace("{{iteration}}", str(context.get("$iteration", 0)))
            result = result.replace("{{ iteration }}", str(context.get("$iteration", 0)))
            
            # Handle loop_id.iteration pattern (e.g., {{ guessing_loop.iteration }})
            loop_id = context.get("loop_id")
            if loop_id:
                result = result.replace(f"{{{{{loop_id}.iteration}}}}", str(context.get("$iteration", 0)))
                result = result.replace(f"{{{{ {loop_id}.iteration }}}}", str(context.get("$iteration", 0)))

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
    
    async def _process_templates_with_named_loops(self, obj: Any, context: Dict[str, Any]) -> Any:
        """Process templates with named loop variable support for while loops.
        
        Args:
            obj: Object to process (dict, list, or string)
            context: Enhanced context with all loop variables
            
        Returns:
            Processed object with all templates rendered
        """
        if isinstance(obj, str):
            # Handle named loop variable references and AUTO tags
            result = obj
            
            # Process named loop variables ($loop_name.variable)
            for loop_name, loop_context in self.loop_context_manager.active_loops.items():
                loop_vars = loop_context.to_template_dict()
                for var_name, var_value in loop_vars.items():
                    if var_name.startswith(f'${loop_name}.'):
                        # Replace ${loop_name}.variable with actual value
                        result = result.replace(f"{{{{{var_name}}}}}", str(var_value))
                        result = result.replace(f"{{{{ {var_name} }}}}", str(var_value))
            
            # Process default loop variables ($index, etc.)
            current_loop = self.loop_context_manager.get_current_loop()
            if current_loop:
                default_vars = current_loop.to_template_dict(is_current_loop=True)
                for var_name, var_value in default_vars.items():
                    if not '.' in var_name:  # Only process default $ variables
                        result = result.replace(f"{{{{{var_name}}}}}", str(var_value))
                        result = result.replace(f"{{{{ {var_name} }}}}", str(var_value))
            
            # Handle AUTO tags with enhanced context
            if "<AUTO>" in result:
                result = await self.auto_resolver._resolve_auto_tags(result, context, {})
            
            return result
            
        elif isinstance(obj, dict):
            return {k: await self._process_templates_with_named_loops(v, context) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [await self._process_templates_with_named_loops(item, context) for item in obj]
        
        return obj
    
    def _has_active_nested_loops(self, loop_name: str) -> bool:
        """Check if any nested loops are still active that depend on this loop."""
        # For now, simple implementation - can be enhanced later
        return False

    def _simple_template_replace(self, template: str, context: Dict[str, Any]) -> str:
        """Simple template variable replacement for conditions."""
        import re
        
        def replace_var(match):
            var_name = match.group(1).strip()
            
            # Handle dot notation
            if '.' in var_name:
                parts = var_name.split('.')
                value = context
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        return match.group(0)  # Return original if not found
                return str(value)
            
            # Simple variable
            if var_name in context:
                return str(context[var_name])
            
            return match.group(0)  # Return original if not found
        
        # Replace {{ variable }} patterns
        template_pattern = r'\{\{\s*([^}]+)\s*\}\}'
        return re.sub(template_pattern, replace_var, template)

    async def should_continue_enhanced(
        self,
        loop_id: str,
        condition: Optional[str] = None,
        until_condition: Optional[str] = None,
        context: Dict[str, Any] = None,
        step_results: Dict[str, Any] = None,
        iteration: int = 0,
        max_iterations: int = 100,
    ) -> Dict[str, Any]:
        """Enhanced version using structured condition evaluation.
        
        Returns detailed evaluation results including performance metrics.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Check max iterations
        if iteration >= max_iterations:
            return {
                "should_continue": False,
                "termination_reason": "max_iterations_reached",
                "evaluation_results": [],
                "performance": self.enhanced_evaluator.get_performance_stats()
            }
        
        context = context or {}
        step_results = step_results or {}
        evaluation_results = []
        
        # Build enhanced context with loop info
        enhanced_context = context.copy()
        loop_state = self.loop_states.get(loop_id, {})
        enhanced_context.update({
            "$iteration": iteration,
            "iteration": iteration,  # Also available without $ for Jinja2 templates
            "$loop_state": loop_state,
            "loop_state": loop_state,  # Also available without $ for Jinja2 templates
            "current_result": (
                step_results.get(f"{loop_id}_{iteration-1}_result")
                if iteration > 0
                else None
            ),
        })
        
        # Evaluate while condition first (if present)
        if condition:
            while_result = await self.enhanced_evaluator.evaluate_condition(
                condition=condition,
                context=enhanced_context,
                step_results=step_results,
                iteration=iteration,
                condition_type="while"
            )
            evaluation_results.append(while_result)
            
            logger.info(f"While condition evaluation: {while_result.to_dict()}")
            
            # If while condition says stop, return immediately
            if while_result.should_terminate:
                return {
                    "should_continue": False,
                    "termination_reason": "while_condition_false",
                    "evaluation_results": evaluation_results,
                    "performance": self.enhanced_evaluator.get_performance_stats()
                }
        
        # Evaluate until condition (if present)
        if until_condition:
            until_result = await self.enhanced_evaluator.evaluate_condition(
                condition=until_condition,
                context=enhanced_context,
                step_results=step_results,
                iteration=iteration,
                condition_type="until"
            )
            evaluation_results.append(until_result)
            
            logger.info(f"Until condition evaluation: {until_result.to_dict()}")
            
            # If until condition says stop, return immediately
            if until_result.should_terminate:
                return {
                    "should_continue": False,
                    "termination_reason": "until_condition_met",
                    "evaluation_results": evaluation_results,
                    "performance": self.enhanced_evaluator.get_performance_stats()
                }
        
        # Continue loop
        return {
            "should_continue": True,
            "termination_reason": None,
            "evaluation_results": evaluation_results,
            "performance": self.enhanced_evaluator.get_performance_stats()
        }

    def get_condition_debug_info(self, loop_id: str) -> Dict[str, Any]:
        """Get debugging information for loop conditions."""
        return {
            "loop_id": loop_id,
            "loop_state": self.loop_states.get(loop_id, {}),
            "evaluator_performance": self.enhanced_evaluator.get_performance_stats(),
        }

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
