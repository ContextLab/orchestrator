"""Enhanced YAML compiler with control flow support."""

from typing import Any, Dict, List, Optional

from .yaml_compiler import YAMLCompiler, YAMLCompilerError
from ..core.pipeline import Pipeline
from ..core.task import Task
from ..control_flow import (
    ConditionalHandler,
    ForLoopHandler,
    WhileLoopHandler,
    DynamicFlowHandler,
    ControlFlowAutoResolver,
)


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
            pipeline = self._build_pipeline(pipeline_def)

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
                # Handle for-each loop
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

        # Process condition for AUTO tags
        if "while" in loop_def and "<AUTO>" in loop_def["while"]:
            # Store original condition for runtime resolution
            loop_def["metadata"]["while_condition"] = loop_def["while"]

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

    def _build_task(self, task_def: Dict[str, Any]) -> Task:
        """Build Task object with control flow support.

        Args:
            task_def: Task definition

        Returns:
            Task object
        """
        # Check for condition in task definition
        if "if" in task_def or "condition" in task_def:
            return self.conditional_handler.create_conditional_task(task_def)

        # Check for goto at top level
        if "goto" in task_def:
            return self.dynamic_flow_handler.create_dynamic_task(task_def)

        # Check for control flow metadata
        if "metadata" in task_def:
            metadata = task_def["metadata"]

            # Create conditional task if needed
            if "condition" in metadata:
                return self.conditional_handler.create_conditional_task(task_def)

            # Create dynamic flow task if needed
            if "goto" in metadata or "dynamic_dependencies" in metadata:
                return self.dynamic_flow_handler.create_dynamic_task(task_def)

        # Default to parent implementation
        return super()._build_task(task_def)
