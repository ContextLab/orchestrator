"""Model-based control system that uses real AI models for task execution."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.control_system import ControlSystem
from ..core.pipeline import Pipeline
from ..core.task import Task
from ..models.model_registry import ModelRegistry


class ModelBasedControlSystem(ControlSystem):
    """Control system that executes tasks using real AI models."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        name: str = "model-based-control-system",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize model-based control system.

        Args:
            model_registry: Registry of available models
            name: Control system name
            config: Configuration dictionary
        """
        if config is None:
            config = {
                "capabilities": {
                    "supported_actions": [
                        "generate",
                        "generate_text",
                        "analyze",
                        "transform",
                        "execute",
                        "search",
                        "extract",
                        "filter",
                        "synthesize",
                        "create",
                        "validate",
                        "optimize",
                        "review",
                        "write",
                        "compile",
                        "process",
                    ],
                    "parallel_execution": True,
                    "streaming": True,
                    "checkpoint_support": True,
                },
                "base_priority": 20,
            }

        super().__init__(name, config)
        self.model_registry = model_registry
        self._execution_history = []

    async def execute_task(self, task: Task, context: Dict[str, Any]) -> Any:
        """
        Execute a task using appropriate AI model.

        Args:
            task: Task to execute
            context: Execution context

        Returns:
            Task execution result
        """
        # Validate required parameters for text generation actions
        if task.action in ["generate_text", "generate"] and (
            not task.parameters or "prompt" not in task.parameters
        ):
            raise ValueError(
                f"Task '{task.id}' with action '{task.action}' requires a 'prompt' parameter"
            )

        # Record execution
        self._execution_history.append(
            {
                "task_id": task.id,
                "action": task.action,
                "parameters": task.parameters,
                "context": context,
            }
        )

        # Get the model from context if available
        model = context.get("model")

        # If no model in context, select one based on task
        if not model:
            requirements = self._get_task_requirements(task)
            model = await self.model_registry.select_model(requirements)

        # Extract the actual action/prompt from the task
        if task.action in ["generate", "generate_text"] and task.parameters.get(
            "prompt"
        ):
            # For generate actions, use the prompt parameter
            prompt_text = task.parameters["prompt"]

            # Substitute template variables from previous results
            if (
                isinstance(prompt_text, str)
                and "{" in prompt_text
                and "}" in prompt_text
            ):
                import re

                template_vars = re.findall(r"\{(\w+)\}", prompt_text)
                for var in template_vars:
                    if var in context.get("previous_results", {}):
                        result_value = context["previous_results"][var]
                        # Handle different result types
                        if isinstance(result_value, dict) and "result" in result_value:
                            result_value = result_value["result"]
                        elif (
                            isinstance(result_value, dict) and "output" in result_value
                        ):
                            result_value = result_value["output"]
                        prompt_text = prompt_text.replace(
                            f"{{{var}}}", str(result_value)
                        )

            prompt = prompt_text
        else:
            # For other actions, use the action as the prompt
            action_text = str(task.action)  # Convert to string in case it's not

            # Handle AUTO tags by extracting the content
            auto_tag_pattern = r"<AUTO>(.*?)</AUTO>"
            auto_match = re.search(auto_tag_pattern, action_text, re.DOTALL)
            if auto_match:
                action_text = auto_match.group(1).strip()

            # Build the full prompt with context
            prompt = self._build_prompt(task, action_text, context)

        try:
            # Use the model to generate result
            # Get generation parameters from task
            temperature = (
                task.parameters.get("temperature", 0.7) if task.parameters else 0.7
            )
            max_tokens = (
                task.parameters.get("max_tokens", 1000) if task.parameters else 1000
            )

            result = await model.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Parse the result based on expected format
            return self._parse_result(result, task)

        except Exception as e:
            # Log the error and re-raise it
            print(f">> Model execution error: {str(e)}")
            raise RuntimeError(f"Failed to execute task {task.id}: {str(e)}") from e

    def _get_task_requirements(self, task: Task) -> Dict[str, Any]:
        """Get model requirements based on task."""
        # Determine task type
        task_types = []
        action_lower = str(task.action).lower()  # Convert to string first
        print(
            f">> DEBUG: Processing action: '{action_lower}' (type: {type(task.action)})"
        )

        # Map action to supported task types
        if "generate_text" in action_lower or action_lower == "generate_text":
            # Special case for generate_text action - map to "generate"
            task_types.append("generate")
            print(">> DEBUG: Mapped generate_text to generate")
        elif any(word in action_lower for word in ["generate", "create", "write"]):
            task_types.append("generate")
        if any(word in action_lower for word in ["analyze", "extract", "identify"]):
            task_types.append("analyze")
        if any(word in action_lower for word in ["transform", "convert", "modify"]):
            task_types.append("transform")
        if any(word in action_lower for word in ["code", "program", "implement"]):
            task_types.append("code")

        # Default to general task
        if not task_types:
            task_types = ["generate"]

        # Debug print
        context_estimate = len(str(task.parameters)) // 4
        requirements = {
            "tasks": task_types,
            "context_window": context_estimate,  # Rough estimate
            "expertise": self._determine_expertise(task),
        }
        print(f">> DEBUG: Task requirements for {task.action}: {requirements}")
        return requirements

    def _determine_expertise(self, task: Task) -> list[str]:
        """Determine required expertise based on task."""
        expertise = ["general"]
        action_lower = str(task.action).lower()  # Convert to string first

        if any(word in action_lower for word in ["research", "analyze", "synthesize"]):
            expertise.append("reasoning")
        if any(word in action_lower for word in ["code", "implement", "debug"]):
            expertise.append("code")
        if any(word in action_lower for word in ["creative", "story", "write"]):
            expertise.append("creative")

        return expertise

    def _build_prompt(
        self, task: Task, action_text: str, context: Dict[str, Any]
    ) -> str:
        """Build full prompt for the model."""
        # Start with the action text
        prompt_parts = [action_text]

        # Add task parameters if any
        if task.parameters:
            # Check if we have template variables to resolve
            params_to_add = []
            for key, value in task.parameters.items():
                # Skip adding action again if it's in parameters
                if key != "action":
                    # Check if value contains template variables
                    if isinstance(value, str) and "{" in value and "}" in value:
                        # Try to resolve template variables from context
                        import re

                        template_vars = re.findall(r"\{(\w+)\}", value)
                        for var in template_vars:
                            if var in context.get("previous_results", {}):
                                value = value.replace(
                                    f"{{{var}}}", str(context["previous_results"][var])
                                )
                    params_to_add.append(f"- {key}: {value}")

            if params_to_add:
                prompt_parts.append("\nTask Parameters:")
                prompt_parts.extend(params_to_add)

        # Add relevant context from previous results
        if "previous_results" in context and context["previous_results"]:
            prev_results = context["previous_results"]

            # First, add dependency results if they exist
            if task.dependencies:
                prompt_parts.append("\nPrevious Step Results:")
                for dep in task.dependencies:
                    if dep in prev_results:
                        result_str = str(prev_results[dep])
                        # Limit result length but keep it meaningful
                        if len(result_str) > 1200:
                            # For very long results, include beginning and end
                            result_str = (
                                result_str[:800]
                                + "\n... [content truncated] ...\n"
                                + result_str[-400:]
                            )
                        prompt_parts.append(f"\n{dep} result:\n{result_str}")

            # Also add any additional context that might be relevant
            if len(prev_results) > 0 and not task.dependencies:
                # For tasks without explicit dependencies, show the latest results
                prompt_parts.append("\nAvailable Context:")
                # Show the 3 most recent results
                recent_results = list(prev_results.items())[-3:]
                for step_id, result in recent_results:
                    result_str = str(result)
                    if len(result_str) > 400:
                        result_str = result_str[:400] + "... [truncated]"
                    prompt_parts.append(f"\n{step_id}: {result_str}")

        # Add instructions for using the context
        if "previous_results" in context and context["previous_results"]:
            prompt_parts.append(
                "\nInstructions: Use the provided context and previous results to inform your response. Be specific and reference concrete information from the previous steps when relevant."
            )

        # Add quality enhancement instructions
        prompt_parts.extend(
            [
                "\nQuality Guidelines:",
                "- Provide comprehensive and well-structured responses",
                "- Use clear headings and formatting where appropriate",
                "- Include specific examples and concrete details",
                "- Ensure logical flow and coherence",
                "- Be thorough but concise",
                "- Maintain professional tone unless otherwise specified",
            ]
        )

        return "\n".join(prompt_parts)

    def _parse_result(self, result: str, task: Task) -> Any:
        """Parse model result based on expected format."""
        # For now, return the raw result
        # In the future, this could parse JSON, extract specific fields, etc.
        return result

    async def execute_pipeline(self, pipeline: Pipeline) -> Dict[str, Any]:
        """Execute an entire pipeline."""
        results = {}

        # Get execution levels
        execution_levels = pipeline.get_execution_levels()

        # Execute each level
        for level in execution_levels:
            for task_id in level:
                task = pipeline.get_task(task_id)
                if task:
                    # Build context with previous results
                    context = {
                        "pipeline_id": pipeline.id,
                        "previous_results": results,
                    }

                    # Execute task
                    result = await self.execute_task(task, context)
                    results[task_id] = result

        return results

    def get_capabilities(self) -> Dict[str, Any]:
        """Return system capabilities."""
        return self._capabilities

    async def health_check(self) -> bool:
        """Check if the system is healthy."""
        # Check if we have any models available
        try:
            available_models = await self.model_registry.get_available_models()
            return len(available_models) > 0
        except Exception:
            return False

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history for debugging."""
        return self._execution_history
