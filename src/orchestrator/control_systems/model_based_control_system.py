"""Model-based control system that uses real AI models for task execution."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from ..core.control_system import ControlSystem
from ..core.pipeline import Pipeline
from ..core.task import Task
from ..core.unified_template_resolver import UnifiedTemplateResolver, TemplateResolutionContext
from ..models.model_registry import ModelRegistry
from ..utils.output_sanitizer import sanitize_output

logger = logging.getLogger(__name__)


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
        
        # Initialize unified template resolver
        self.unified_template_resolver = UnifiedTemplateResolver(debug_mode=True)

    async def _execute_task_impl(self, task: Task, context: Dict[str, Any]) -> Any:
        """
        Execute a task using appropriate AI model.

        Args:
            task: Task to execute
            context: Execution context

        Returns:
            Task execution result
        """
        # Use UnifiedTemplateResolver to render templates in task parameters
        if task.parameters:
            # Build enhanced step_results that includes loop iteration context
            enhanced_step_results = dict(context.get("previous_results", {}))
            
            # For loop tasks, add results from sibling tasks in the same iteration using loop context mapping
            if "_loop_context_mapping" in context and task.metadata.get("is_for_each_child"):
                loop_context_mapping = context["_loop_context_mapping"]
                available_results = list(context.get("previous_results", {}).keys())
                logger.info(f"STREAM_C_MODEL_DEBUG: Enhancing step_results for AI model task {task.id}")
                logger.info(f"STREAM_C_MODEL_DEBUG:   Loop context mapping has {len(loop_context_mapping)} entries: {list(loop_context_mapping.keys())}")
                logger.info(f"STREAM_C_MODEL_DEBUG:   Available previous_results: {available_results}")
                
                # Add results using both full task IDs and short names for cross-step references
                for short_name, full_task_id in loop_context_mapping.items():
                    if full_task_id in context.get("previous_results", {}):
                        result = context["previous_results"][full_task_id]
                        # Register with short name for cross-step references like {{ read_file.content }}
                        enhanced_step_results[short_name] = result
                        logger.info(f"STREAM_C_MODEL_DEBUG:     Mapped loop result '{short_name}' -> {type(result).__name__} (available)")
                        
                        # Also ensure it's available with the full task ID
                        enhanced_step_results[full_task_id] = result
                    else:
                        logger.info(f"STREAM_C_MODEL_DEBUG:     Skip '{short_name}' -> '{full_task_id}' (not available yet)")
            else:
                if "_loop_context_mapping" not in context:
                    logger.info(f"STREAM_C_MODEL_DEBUG: No loop context mapping for AI model task {task.id}")
                if not task.metadata.get("is_for_each_child"):
                    logger.info(f"STREAM_C_MODEL_DEBUG: AI model task {task.id} is not a for_each_child")
            
            # Collect comprehensive context for template resolution
            template_context = self.unified_template_resolver.collect_context(
                pipeline_id=context.get("pipeline_id"),
                task_id=task.id,
                pipeline_inputs=context.get("pipeline_inputs", {}),
                pipeline_parameters=context.get("pipeline_params", {}),
                step_results=enhanced_step_results,
                additional_context={
                    # Add loop variables
                    "$item": context.get("$item"),
                    "$index": context.get("$index"),
                    "$is_first": context.get("$is_first"),
                    "$is_last": context.get("$is_last"),
                    "$iteration": context.get("$iteration"),
                    # Add any other context variables
                    **{k: v for k, v in context.items() 
                       if k not in ["pipeline_id", "pipeline_inputs", "pipeline_params", "previous_results", "_template_manager"]}
                }
            )
            
            # Resolve templates in task parameters
            rendered_params = self.unified_template_resolver.resolve_templates(
                task.parameters, template_context
            )
            
            # Update task parameters with rendered values
            task.parameters = rendered_params
            
            # Log template resolution for debugging
            logger.info(f"Task {task.id}: Template resolution completed using UnifiedTemplateResolver")
            if "prompt" in rendered_params:
                logger.debug(f"  Resolved prompt length: {len(str(rendered_params['prompt']))} chars")
        
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
        
        # Check if model is specified in task parameters
        if not model and task.parameters and "model" in task.parameters:
            model_spec = task.parameters["model"]
            # Check if it's an AUTO tag (should have been resolved by now)
            if isinstance(model_spec, str) and not model_spec.startswith("<AUTO"):
                # It's a model name/path, get it from registry
                model = self.model_registry.get_model(model_spec)
                if not model:
                    # Model not found, raise error
                    raise ValueError(f"Model '{model_spec}' not found in registry")

        # If still no model, select one based on task requirements
        if not model:
            requirements = self._get_task_requirements(task)
            model = await self.model_registry.select_model(requirements)

        # Extract the actual action/prompt from the task
        if task.action in ["generate", "generate_text"] and task.parameters.get(
            "prompt"
        ):
            # For generate actions, use the prompt parameter
            # Templates have already been rendered above
            prompt = task.parameters["prompt"]
        elif task.action == "analyze_text" and task.parameters:
            # For analyze_text action, build analysis prompt
            text = task.parameters.get("text", "")
            analysis_type = task.parameters.get("analysis_type", "comprehensive")
            custom_prompt = task.parameters.get("prompt", "")
            
            # If a custom prompt is provided, use it with the text
            if custom_prompt:
                prompt = f"{custom_prompt}\n\nData:\n{text}"
                # Debug logging
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Task {task.id}: Using custom prompt for analyze_text")
                logger.info(f"  Prompt: {custom_prompt[:100]}...")
                logger.info(f"  Text: {text[:100]}...")
                logger.info(f"  Full prompt length: {len(prompt)} chars")
            else:
                # Build default analysis prompt
                prompt = f"Perform a {analysis_type} analysis of the following text:\n\n{text}"
                
                if analysis_type == "comprehensive":
                    prompt += "\n\nProvide a detailed analysis covering structure, content, style, and key themes."
                elif analysis_type == "quality":
                    prompt += "\n\nAssess the quality of this text on a scale of 0-1, considering clarity, coherence, and completeness."
                elif analysis_type == "trends":
                    prompt += "\n\nIdentify and analyze key trends, patterns, and insights from this data."
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
            # Check if this is a structured generation task
            if task.action == "generate-structured":
                # Use structured generation
                if not task.parameters or "schema" not in task.parameters:
                    raise ValueError(
                        f"Task '{task.id}' with action 'generate-structured' requires a 'schema' parameter"
                    )
                
                temperature = (
                    task.parameters.get("temperature", 0.7) if task.parameters else 0.7
                )
                
                # Build structured generation kwargs
                structured_kwargs = {
                    "prompt": prompt,
                    "schema": task.parameters["schema"],
                    "temperature": temperature,
                }
                
                # Add any additional kwargs, excluding system parameters
                if task.parameters:
                    for key, value in task.parameters.items():
                        if key not in ["prompt", "schema", "temperature", "model", "max_tokens"]:
                            structured_kwargs[key] = value
                
                result = await model.generate_structured(**structured_kwargs)
                
                # Parse the result based on expected format
                parsed_result = self._parse_result(result, task)
                
                # Apply output sanitization to clean conversational markers
                sanitized_result = sanitize_output(parsed_result)
                
                return sanitized_result
            
            else:
                # Use regular generation
                # Get generation parameters from task
                temperature = (
                    task.parameters.get("temperature", 0.7) if task.parameters else 0.7
                )
                max_tokens = (
                    task.parameters.get("max_tokens", 1000) if task.parameters else 1000
                )
                
                # Build generation kwargs
                # Handle model-specific parameter names
                gen_kwargs = {
                    "prompt": prompt,
                    "temperature": temperature,
                }
                
                # Check if model requires max_completion_tokens instead of max_tokens
                # GPT-5 models from OpenAI require max_completion_tokens
                if model and hasattr(model, 'provider') and model.provider == 'openai':
                    # Check if it's a GPT-5 model
                    if hasattr(model, 'name') and 'gpt-5' in model.name.lower():
                        gen_kwargs["max_completion_tokens"] = max_tokens
                    else:
                        gen_kwargs["max_tokens"] = max_tokens
                else:
                    gen_kwargs["max_tokens"] = max_tokens
                
                # Add response_format if specified
                if task.parameters and "response_format" in task.parameters:
                    gen_kwargs["response_format"] = task.parameters["response_format"]

                result = await model.generate(**gen_kwargs)

                # Parse the result based on expected format
                parsed_result = self._parse_result(result, task)
                
                # Apply output sanitization to clean conversational markers
                sanitized_result = sanitize_output(parsed_result)
                
                return sanitized_result

        except Exception as e:
            # Log the error and re-raise it
            logger.error(f"Model execution error: {str(e)}")
            raise RuntimeError(f"Failed to execute task {task.id}: {str(e)}") from e

    def _get_task_requirements(self, task: Task) -> Dict[str, Any]:
        """Get model requirements based on task."""
        # Determine task type
        task_types = []
        action_lower = str(task.action).lower()  # Convert to string first
        logger.debug(f"Processing action: '{action_lower}' (type: {type(task.action)})")

        # Map action to supported task types
        if "generate_text" in action_lower or action_lower == "generate_text":
            # Special case for generate_text action - map to "generate"
            task_types.append("generate")
            logger.debug("Mapped generate_text to generate")
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

        # Calculate requirements
        context_estimate = len(str(task.parameters)) // 4
        requirements = {
            "tasks": task_types,
            "context_window": context_estimate,  # Rough estimate
            "expertise": self._determine_expertise(task),
        }
        logger.debug(f"Task requirements for {task.action}: {requirements}")
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

    def _parse_result(self, result: Any, task: Task) -> Any:
        """Parse model result based on expected format."""
        # For generate-structured actions, the result should already be a structured object
        if task.action == "generate-structured":
            # If it's still a string, try to parse it as JSON
            if isinstance(result, str):
                try:
                    import json
                    return json.loads(result)
                except json.JSONDecodeError:
                    # If parsing fails, return the string as-is
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Task {task.id}: Could not parse generate-structured result as JSON")
                    return result
            else:
                # Already structured, return as-is
                return result
        
        # For other actions, return the raw result
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
