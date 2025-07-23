"""Runtime AUTO tag resolver with real model support."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from ..models.model_registry import ModelRegistry, ModelNotFoundError


class RuntimeAutoResolver:
    """
    Resolves AUTO tags at runtime with access to execution context.

    Unlike the compile-time AmbiguityResolver, this resolver:
    - Has access to previous step results
    - Can use runtime context and template values
    - Makes real model calls for intelligent resolution
    - Uses real models from the registry
    """

    def __init__(self, model_registry: ModelRegistry) -> None:
        """
        Initialize runtime AUTO resolver.

        Args:
            model_registry: Registry of real models to use

        Raises:
            ValueError: If no models are available
        """
        self.model_registry = model_registry

        # Verify we have real models available
        available_models = model_registry.list_models()
        if not available_models:
            raise ValueError(
                "No models available for AUTO tag resolution. "
                "Please configure at least one model provider."
            )

        # Cache for resolved AUTO tags (within same execution)
        self._resolution_cache: Dict[str, str] = {}

    async def resolve(
        self,
        auto_content: str,
        context: Dict[str, Any],
        task_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Resolve AUTO tag content to executable action.

        Args:
            auto_content: Content inside <AUTO> tags
            context: Full execution context including previous results
            task_config: Task configuration (model preferences, etc.)

        Returns:
            Resolved action string

        Raises:
            RuntimeError: If resolution fails
        """
        # Create cache key from content and available context
        cache_key = self._create_cache_key(auto_content, context)
        if cache_key in self._resolution_cache:
            return self._resolution_cache[cache_key]

        try:
            # Select appropriate model for resolution
            model = await self._select_model(task_config)

            # Build resolution prompt
            prompt = self._build_resolution_prompt(auto_content, context)

            # Call real model for resolution
            resolved = await model.generate(
                prompt=prompt,
                temperature=0.3,  # Lower temperature for consistency
                max_tokens=500,
            )

            # Clean and validate resolution
            resolved = self._clean_resolution(resolved)

            # Cache the result
            self._resolution_cache[cache_key] = resolved

            return resolved

        except Exception as e:
            raise RuntimeError(f"Failed to resolve AUTO tag '{auto_content}': {e}") from e

    async def _select_model(
        self,
        task_config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Select best available model for AUTO tag resolution.

        Args:
            task_config: Task configuration with model preferences

        Returns:
            Selected model instance

        Raises:
            ModelNotFoundError: If no suitable model found
        """
        # Check for specific model request in config
        if task_config and "model" in task_config:
            model_name = task_config["model"]
            try:
                return self.model_registry.get_model(model_name)
            except ModelNotFoundError:
                # Continue to auto-selection
                pass

        # Auto-select based on requirements
        requirements = {
            "task_type": "reasoning",  # AUTO resolution needs reasoning
            "min_context_window": 4096,
            "preferred_size": "small",  # Prefer smaller models for speed
        }

        return await self.model_registry.select_model(requirements)

    def _build_resolution_prompt(
        self,
        auto_content: str,
        context: Dict[str, Any],
    ) -> str:
        """
        Build prompt for model to resolve AUTO tag.

        Args:
            auto_content: Content to resolve
            context: Execution context

        Returns:
            Formatted prompt
        """
        # Extract relevant context
        previous_results = context.get("previous_results", {})
        inputs = context.get("inputs", {})

        # Build context summary
        context_summary = []

        if inputs:
            context_summary.append("Input values:")
            for key, value in inputs.items():
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                context_summary.append(f"  - {key}: {value}")

        if previous_results:
            context_summary.append("\nPrevious step results:")
            for step_id, result in previous_results.items():
                if isinstance(result, dict):
                    result_str = str(result.get("result", result))[:100] + "..."
                else:
                    result_str = str(result)[:100] + "..."
                context_summary.append(f"  - {step_id}: {result_str}")

        context_str = "\n".join(context_summary) if context_summary else "No context available"

        # Build the prompt
        prompt = f"""You are an AI assistant helping to resolve ambiguous task descriptions into concrete, executable actions.

Task Description: {auto_content}

Available Context:
{context_str}

Please convert the task description into a clear, specific, executable instruction. The instruction should:
1. Be concrete and actionable
2. Include all necessary details from the context
3. Be suitable for an AI model to execute
4. Preserve any template variables ({{variable}}) that appear in the original

Respond with ONLY the resolved instruction, no explanations or metadata."""

        return prompt

    def _clean_resolution(self, resolved: str) -> str:
        """
        Clean and validate resolved content.

        Args:
            resolved: Raw resolution from model

        Returns:
            Cleaned resolution
        """
        # Remove any markdown formatting
        resolved = re.sub(r"```[^`]*```", "", resolved)
        resolved = re.sub(r"`([^`]+)`", r"\1", resolved)

        # Remove common prefixes models might add
        prefixes_to_remove = [
            "Here is the resolved instruction:",
            "The resolved instruction is:",
            "Resolved:",
            "Instruction:",
            "Action:",
        ]

        for prefix in prefixes_to_remove:
            if resolved.lower().startswith(prefix.lower()):
                resolved = resolved[len(prefix) :].strip()

        # Ensure it's not empty
        resolved = resolved.strip()
        if not resolved:
            raise ValueError("Model returned empty resolution")

        return resolved

    def _create_cache_key(
        self,
        auto_content: str,
        context: Dict[str, Any],
    ) -> str:
        """
        Create cache key for resolution.

        Args:
            auto_content: AUTO tag content
            context: Execution context

        Returns:
            Cache key
        """
        # Include relevant context in cache key
        context_parts = []

        # Include input values
        inputs = context.get("inputs", {})
        for key in sorted(inputs.keys()):
            context_parts.append(f"input_{key}={inputs[key]}")

        # Include previous result keys (not values as they're too large)
        prev_results = context.get("previous_results", {})
        if prev_results:
            context_parts.append(f"steps={','.join(sorted(prev_results.keys()))}")

        context_str = "|".join(context_parts)
        return f"{auto_content}||{context_str}"
