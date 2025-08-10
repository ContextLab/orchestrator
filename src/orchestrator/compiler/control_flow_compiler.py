"""Enhanced YAML compiler with control flow support."""

from typing import Any, Dict, List, Optional

from .yaml_compiler import YAMLCompiler, YAMLCompilerError
from ..core.pipeline import Pipeline
from ..core.task import Task
from ..core.action_loop_task import ActionLoopTask
from ..core.for_each_task import ForEachTask
from ..control_flow import (
    ConditionalHandler,
    ForLoopHandler,
    WhileLoopHandler,
    DynamicFlowHandler,
    ControlFlowAutoResolver,
)
from ..control_flow.action_loop_handler import ActionLoopHandler


class ControlFlowCompiler(YAMLCompiler):
    """YAML compiler with advanced control flow support."""

    def __init__(self, *args, **kwargs):
        """Initialize control flow compiler."""
        # Extract model_registry if provided
        model_registry = kwargs.get("model_registry", None)

        super().__init__(*args, **kwargs)

        # Initialize control flow handlers
        self.control_flow_resolver = ControlFlowAutoResolver(model_registry)
        self.conditional_handler = ConditionalHandler(self.control_flow_resolver)
        self.for_loop_handler = ForLoopHandler(self.control_flow_resolver)
        self.while_loop_handler = WhileLoopHandler(self.control_flow_resolver)
        self.dynamic_flow_handler = DynamicFlowHandler(self.control_flow_resolver)
        self.action_loop_handler = ActionLoopHandler(auto_resolver=self.control_flow_resolver)

    async def compile(
        self,
        yaml_content: str,
        context: Optional[Dict[str, Any]] = None,
        resolve_ambiguities: bool = True,
    ) -> Pipeline:
        """Compile YAML with control flow support.

        Args:
            yaml_content: YAML content as string
            context: Template context variables
            resolve_ambiguities: Whether to resolve AUTO tags

        Returns:
            Compiled Pipeline object with control flow
        """
        try:
            # Step 1: Parse YAML safely
            raw_pipeline = self._parse_yaml(yaml_content)

            # Step 2: Validate against schema
            self.schema_validator.validate(raw_pipeline)

            # Step 3: Merge default values with context
            merged_context = self._merge_defaults_with_context(
                raw_pipeline, context or {}
            )

            # Step 4: Process templates
            pipeline_def = self._process_templates(raw_pipeline, merged_context)

            # Step 5: Resolve ambiguities (including control flow AUTO tags)
            if resolve_ambiguities:
                pipeline_def = await self._resolve_ambiguities(pipeline_def)

            # Step 6: Process control flow constructs
            pipeline_def = await self._process_control_flow(
                pipeline_def, merged_context
            )

            # Step 7: Build pipeline with expanded tasks
            pipeline = self._build_pipeline(pipeline_def, merged_context)

            return pipeline

        except Exception as e:
            raise YAMLCompilerError(f"Failed to compile pipeline: {e}") from e

    async def _process_control_flow(
        self, pipeline_def: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process control flow constructs in pipeline definition.

        Args:
            pipeline_def: Pipeline definition
            context: Execution context

        Returns:
            Pipeline definition with expanded control flow
        """
        steps = pipeline_def.get("steps", [])
        processed_steps = []
        step_results = {}  # Simulated results for compile-time resolution

        # Collect all step IDs for validation
        all_step_ids = self._collect_step_ids(steps)
        context["all_step_ids"] = all_step_ids

        for step_def in steps:
            # Process different control flow types
            if "for_each" in step_def:
                # Check if for_each expression contains AUTO tags or runtime dependencies
                for_each_expr = str(step_def.get("for_each", ""))
                
                # Check for runtime dependencies: AUTO tags or step result references
                has_runtime_deps = (
                    "<AUTO>" in for_each_expr or
                    any(f"{step_id}." in for_each_expr for step_id in all_step_ids)
                )
                
                if has_runtime_deps:
                    # Create ForEachTask for runtime expansion
                    for_each_task = self._create_for_each_task(step_def)
                    processed_steps.append(self._task_to_step_def(for_each_task))
                else:
                    # Static loop - expand at compile time
                    expanded_tasks = await self.for_loop_handler.expand_for_loop(
                        step_def, context, step_results
                    )
                    # Convert tasks back to step definitions
                    for task in expanded_tasks:
                        processed_steps.append(self._task_to_step_def(task))

            elif "while" in step_def:
                # Handle while loop (compile-time setup only)
                processed_step = await self._process_while_loop(
                    step_def, context, step_results
                )
                processed_steps.append(processed_step)
            
            elif "action_loop" in step_def:
                # Handle action_loop (compile-time setup only)
                processed_step = await self._process_action_loop(
                    step_def, context, step_results
                )
                processed_steps.append(processed_step)

            else:
                # Process regular step with potential control flow
                processed_step = await self._process_regular_step(
                    step_def, context, step_results
                )
                processed_steps.append(processed_step)

        pipeline_def["steps"] = processed_steps
        return pipeline_def

    async def _process_regular_step(
        self,
        step_def: Dict[str, Any],
        context: Dict[str, Any],
        step_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process a regular step with control flow features.

        Args:
            step_def: Step definition
            context: Execution context
            step_results: Simulated step results

        Returns:
            Processed step definition
        """
        # Process conditional execution
        step_def = await self.conditional_handler.process_conditional_step(
            step_def, context, step_results
        )

        # Process dynamic flow control
        step_def = await self.dynamic_flow_handler.process_step_flow_control(
            step_def, context, step_results
        )

        return step_def

    async def _process_while_loop(
        self,
        loop_def: Dict[str, Any],
        context: Dict[str, Any],
        step_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process while loop definition.

        Args:
            loop_def: While loop definition
            context: Execution context
            step_results: Simulated step results

        Returns:
            Processed loop definition
        """
        # Mark as while loop for runtime expansion
        loop_def["metadata"] = loop_def.get("metadata", {})
        loop_def["metadata"]["is_while_loop"] = True
        loop_def["metadata"]["max_iterations"] = loop_def.get("max_iterations", 10)

        # Store the while condition (whether it has AUTO tags or not)
        if "while" in loop_def:
            loop_def["metadata"]["while_condition"] = loop_def["while"]

        return loop_def

    async def _process_action_loop(
        self,
        loop_def: Dict[str, Any],
        context: Dict[str, Any],
        step_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process action_loop definition.

        Args:
            loop_def: Action loop definition
            context: Execution context
            step_results: Simulated step results

        Returns:
            Processed action loop definition
        """
        # Mark as action loop for runtime execution
        loop_def["metadata"] = loop_def.get("metadata", {})
        loop_def["metadata"]["is_action_loop"] = True
        loop_def["metadata"]["max_iterations"] = loop_def.get("max_iterations", 100)
        
        # Store loop configuration
        if "action_loop" in loop_def:
            loop_def["metadata"]["action_loop"] = loop_def["action_loop"]
        
        # Store termination conditions
        if "until" in loop_def:
            loop_def["metadata"]["until"] = loop_def["until"]
        if "while_condition" in loop_def:
            loop_def["metadata"]["while_condition"] = loop_def["while_condition"]
        
        # Store additional loop configuration
        if "break_on_error" in loop_def:
            loop_def["metadata"]["break_on_error"] = loop_def["break_on_error"]
        if "iteration_timeout" in loop_def:
            loop_def["metadata"]["iteration_timeout"] = loop_def["iteration_timeout"]
        
        # Set action to action_loop for proper routing
        loop_def["action"] = "action_loop"
        
        # Validate action_loop structure
        action_loop = loop_def.get("action_loop", [])
        if not isinstance(action_loop, list):
            raise YAMLCompilerError("action_loop must be a list of actions")
        
        if not action_loop:
            raise YAMLCompilerError("action_loop cannot be empty")
        
        # Validate that each action has required fields
        for i, action_def in enumerate(action_loop):
            if not isinstance(action_def, dict):
                raise YAMLCompilerError(f"Action {i} in action_loop must be a dictionary")
            if "action" not in action_def:
                raise YAMLCompilerError(f"Action {i} in action_loop must have 'action' field")
        
        # Validate termination conditions
        if not loop_def.get("until") and not loop_def.get("while_condition"):
            raise YAMLCompilerError("action_loop must have either 'until' or 'while_condition'")
        
        if loop_def.get("until") and loop_def.get("while_condition"):
            raise YAMLCompilerError("action_loop cannot have both 'until' and 'while_condition'")

        return loop_def

    def _collect_step_ids(self, steps: List[Dict[str, Any]]) -> List[str]:
        """Collect all step IDs including those in loops.

        Args:
            steps: List of step definitions

        Returns:
            List of all step IDs
        """
        step_ids = []

        for step in steps:
            step_ids.append(step["id"])

            # Check for nested steps in loops
            if "steps" in step:
                nested_ids = self._collect_step_ids(step["steps"])
                step_ids.extend(nested_ids)

        return step_ids

    def _create_for_each_task(self, loop_def: Dict[str, Any]) -> ForEachTask:
        """Create a ForEachTask for runtime expansion.
        
        Args:
            loop_def: Loop definition from YAML
            
        Returns:
            ForEachTask instance configured for runtime expansion
        """
        # Extract loop configuration
        task_id = loop_def.get("id", "for_each_loop")
        for_each_expr = loop_def.get("for_each", "")
        loop_steps = loop_def.get("steps", [])
        max_parallel = loop_def.get("max_parallel", 1)
        loop_var = loop_def.get("loop_var", "$item")
        loop_name = loop_def.get("loop_name")
        add_completion_task = loop_def.get("add_completion_task", len(loop_steps) > 1)
        
        # Handle single action shorthand
        if not loop_steps and "action" in loop_def:
            loop_steps = [
                {
                    "id": f"{task_id}_item",
                    "action": loop_def["action"],
                    "parameters": loop_def.get("parameters", {}),
                }
            ]
        
        # Create ForEachTask
        for_each_task = ForEachTask(
            id=task_id,
            name=loop_def.get("name", f"For each: {task_id}"),
            action="for_each_runtime",  # Special action type for runtime expansion
            parameters={},  # No parameters needed at this stage
            dependencies=loop_def.get("dependencies", []),
            for_each_expr=for_each_expr,
            loop_steps=loop_steps,
            max_parallel=max_parallel,
            loop_var=loop_var,
            loop_name=loop_name,
            add_completion_task=add_completion_task,
            metadata=loop_def.get("metadata", {})
        )
        
        # Add any additional metadata
        if "timeout" in loop_def:
            for_each_task.timeout = loop_def["timeout"]
        if "max_retries" in loop_def:
            for_each_task.max_retries = loop_def["max_retries"]
            
        return for_each_task

    def _task_to_step_def(self, task: Task) -> Dict[str, Any]:
        """Convert Task object back to step definition.

        Args:
            task: Task object

        Returns:
            Step definition dictionary
        """
        step_def = {
            "id": task.id,
            "name": task.name,
            "action": task.action,
            "parameters": task.parameters,
        }

        if task.dependencies:
            step_def["depends_on"] = task.dependencies

        if task.metadata:
            step_def["metadata"] = task.metadata
            # Extract tool field from metadata if present
            if "tool" in task.metadata:
                step_def["tool"] = task.metadata["tool"]

        if task.timeout:
            step_def["timeout"] = task.timeout

        if task.max_retries != 3:  # Only include if not default
            step_def["max_retries"] = task.max_retries

        # Extract control flow properties from metadata
        if "condition" in task.metadata:
            step_def["if"] = task.metadata["condition"]
        if "else_task_id" in task.metadata:
            step_def["else"] = task.metadata["else_task_id"]
        if "goto" in task.metadata:
            step_def["goto"] = task.metadata["goto"]

        return step_def

    def _build_task(self, task_def: Dict[str, Any], available_steps: List[str]) -> Task:
        """Build Task object with control flow support.

        Args:
            task_def: Task definition
            available_steps: List of all step IDs in the pipeline

        Returns:
            Task object
        """
        # First, ensure metadata is properly built
        # This is important for conditional tasks that also have tools
        if "metadata" not in task_def:
            task_def["metadata"] = {}
        
        # Copy important fields to metadata if not already there
        # This ensures fields like 'tool' are preserved for conditional tasks
        if "tool" in task_def and "tool" not in task_def["metadata"]:
            task_def["metadata"]["tool"] = task_def["tool"]
        if "continue_on_error" in task_def and "continue_on_error" not in task_def["metadata"]:
            task_def["metadata"]["continue_on_error"] = task_def["continue_on_error"]
            
        # Extract condition before building base task to prevent early rendering
        condition = task_def.pop("condition", None) or task_def.pop("if", None)
        
        # First, let the parent build the base task with template analysis
        base_task = super()._build_task(task_def, available_steps)
        
        # Restore condition to task_def
        if condition:
            task_def["condition"] = condition
        
        # Check for condition in task definition
        if "if" in task_def or "condition" in task_def:
            conditional_task = self.conditional_handler.create_conditional_task(task_def)
            # Copy template metadata from base task
            conditional_task.template_metadata = base_task.template_metadata
            return conditional_task

        # Check for goto at top level
        if "goto" in task_def:
            dynamic_task = self.dynamic_flow_handler.create_dynamic_task(task_def)
            # Copy template metadata from base task
            dynamic_task.template_metadata = base_task.template_metadata
            return dynamic_task

        # Check for control flow metadata
        if "metadata" in task_def:
            metadata = task_def["metadata"]

            # Create action loop task if needed
            if "is_action_loop" in metadata:
                action_loop_task = self._create_action_loop_task(task_def, base_task)
                return action_loop_task

            # Create conditional task if needed
            if "condition" in metadata:
                conditional_task = self.conditional_handler.create_conditional_task(task_def)
                # Copy template metadata from base task
                conditional_task.template_metadata = base_task.template_metadata
                return conditional_task

            # Create dynamic flow task if needed
            if "goto" in metadata or "dynamic_dependencies" in metadata:
                dynamic_task = self.dynamic_flow_handler.create_dynamic_task(task_def)
                # Copy template metadata from base task
                dynamic_task.template_metadata = base_task.template_metadata
                return dynamic_task

        # Return the base task if no control flow
        return base_task

    def _create_action_loop_task(self, task_def: Dict[str, Any], base_task: Task) -> ActionLoopTask:
        """Create ActionLoopTask from task definition.
        
        Args:
            task_def: Task definition with action loop metadata
            base_task: Base task object
            
        Returns:
            ActionLoopTask instance
        """
        metadata = task_def.get("metadata", {})
        
        # Extract action loop configuration from metadata
        action_loop = metadata.get("action_loop", [])
        until = metadata.get("until")
        while_condition = metadata.get("while_condition") 
        max_iterations = metadata.get("max_iterations", 100)
        break_on_error = metadata.get("break_on_error", False)
        iteration_timeout = metadata.get("iteration_timeout")
        
        # Create ActionLoopTask with all base task properties
        action_loop_task = ActionLoopTask(
            # Base task properties
            id=base_task.id,
            name=base_task.name,
            action=base_task.action,
            parameters=base_task.parameters,
            dependencies=base_task.dependencies,
            status=base_task.status,
            result=base_task.result,
            error=base_task.error,
            metadata=base_task.metadata,
            timeout=base_task.timeout,
            max_retries=base_task.max_retries,
            retry_count=base_task.retry_count,
            created_at=base_task.created_at,
            started_at=base_task.started_at,
            completed_at=base_task.completed_at,
            template_metadata=base_task.template_metadata,
            rendered_parameters=base_task.rendered_parameters,
            dependencies_satisfied=base_task.dependencies_satisfied,
            in_loop_context=base_task.in_loop_context,
            loop_context=base_task.loop_context,
            output_metadata=base_task.output_metadata,
            output_info=base_task.output_info,
            # ActionLoopTask specific properties
            action_loop=action_loop,
            until=until,
            while_condition=while_condition,
            max_iterations=max_iterations,
            break_on_error=break_on_error,
            iteration_timeout=iteration_timeout
        )
        
        return action_loop_task
