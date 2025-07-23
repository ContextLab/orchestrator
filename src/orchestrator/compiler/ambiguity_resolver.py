"""Ambiguity resolver that uses AI models for intelligent resolution."""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Dict, Optional

from ..core.model import Model
from ..models.model_registry import ModelRegistry


class AmbiguityType(Enum):
    """Types of ambiguities that can be resolved."""

    ACTION = "action"
    PARAMETER = "parameter"
    MODEL = "model"
    RESOURCE = "resource"


class AmbiguityResolutionError(Exception):
    """Raised when ambiguity resolution fails."""

    pass


class AmbiguityResolver:
    """
    Resolves ambiguous specifications using AI models.

    This resolver uses real AI models to intelligently resolve ambiguities
    in YAML configurations, making actual API calls to determine the best
    choices based on context and requirements.
    """

    def __init__(
        self, model: Optional[Model] = None, model_registry: Optional[ModelRegistry] = None
    ) -> None:
        """
        Initialize ambiguity resolver.

        Args:
            model: Specific model to use for resolution
            model_registry: Model registry to select models from

        Raises:
            ValueError: If no model is provided and registry has no models
        """
        self.model = model
        self.model_registry = model_registry
        self.resolution_cache: Dict[str, Any] = {}

        # If no model provided, try to get one from registry
        if not model and model_registry:
            available_models = model_registry.list_models()
            if available_models:
                # Try to get a good model for text generation
                try:
                    self.model = model_registry.select_model({"tasks": ["generate"]})
                except Exception:
                    # If selection fails, just get the first available
                    self.model = model_registry.get_model(available_models[0])

        # Verify we have a model for AI resolution
        if not self.model:
            raise ValueError(
                "No AI model available for ambiguity resolution. " "A real model must be provided."
            )

    async def resolve(self, content: str, context_path: str) -> Any:
        """
        Resolve ambiguous content using AI.

        Args:
            content: Ambiguous content to resolve
            context_path: Context path (e.g., "config.output.format")

        Returns:
            Resolved value

        Raises:
            AmbiguityResolutionError: If resolution fails
        """
        # Check cache first
        cache_key = f"{content}:{context_path}"
        if cache_key in self.resolution_cache:
            return self.resolution_cache[cache_key]

        try:
            # Try AI resolution with the model
            if self.model:
                result = await self._resolve_with_ai(content, context_path)
            else:
                raise AmbiguityResolutionError("No AI model available for ambiguity resolution")

            # Cache result
            self.resolution_cache[cache_key] = result
            return result

        except Exception as e:
            raise AmbiguityResolutionError(
                f"Failed to resolve ambiguity '{content}' at {context_path}: {e}"
            ) from e

    async def _resolve_with_ai(self, content: str, context_path: str) -> Any:
        """Use AI model to intelligently resolve ambiguity."""
        # Determine the expected type from context
        expected_type = self._infer_type_from_context(content, context_path)

        # Build a prompt that guides the AI to return the right type
        prompt = self._build_resolution_prompt(content, context_path, expected_type)

        # Get AI response
        response = await self.model.generate(prompt, temperature=0.1, max_tokens=100)

        # Parse the response based on expected type
        return self._parse_ai_response(response, expected_type, content)

    def _infer_type_from_context(self, content: str, context_path: str) -> str:
        """Infer the expected return type from context."""
        content_lower = content.lower()

        # Check for explicit type indicators
        if any(
            word in content_lower for word in ["true", "false", "enable", "disable", "yes", "no"]
        ):
            return "boolean"
        elif any(
            word in content_lower
            for word in ["number", "count", "size", "batch", "timeout", "limit"]
        ):
            return "number"
        elif "list" in content_lower or "array" in content_lower:
            return "list"
        elif re.search(r"\b(json|yaml|xml|csv|text)\b", content_lower):
            return "format"
        elif "algorithm" in content_lower or "method" in content_lower:
            return "choice"

        # Check context path
        if any(word in context_path.lower() for word in ["enable", "disable", "support"]):
            return "boolean"
        elif any(word in context_path.lower() for word in ["size", "count", "limit", "timeout"]):
            return "number"
        elif "format" in context_path.lower():
            return "format"

        return "string"

    def _build_resolution_prompt(self, content: str, context_path: str, expected_type: str) -> str:
        """Build a prompt for the AI model."""
        type_instructions = {
            "boolean": "Respond with only 'true' or 'false'.",
            "number": "Respond with only a number (integer or decimal).",
            "list": "Respond with a JSON array of strings.",
            "format": "Respond with only one of: json, yaml, xml, csv, text.",
            "choice": "Respond with only the chosen option.",
            "string": "Respond with a short string value.",
        }

        instruction = type_instructions.get(expected_type, type_instructions["string"])

        prompt = f"""You are resolving an ambiguous configuration value.

Context: {context_path}
Ambiguous content: {content}

{instruction}

Think about what would be the most appropriate value given the context.
Respond with only the value, no explanation."""

        return prompt

    def _parse_ai_response(self, response: str, expected_type: str, original_content: str) -> Any:
        """Parse AI response into the expected type."""
        response = response.strip()

        if expected_type == "boolean":
            return response.lower() in ["true", "yes", "1", "enable", "on"]

        elif expected_type == "number":
            # Try to extract a number from the response
            numbers = re.findall(r"-?\d+(?:\.\d+)?", response)
            if numbers:
                return float(numbers[0]) if "." in numbers[0] else int(numbers[0])
            # Default numbers based on context
            if "batch" in original_content.lower():
                return 32
            elif "timeout" in original_content.lower():
                return 30
            elif "retry" in original_content.lower():
                return 3
            elif "limit" in original_content.lower():
                return 1000
            return 10

        elif expected_type == "list":
            # Try to parse as JSON array
            try:
                result = json.loads(response)
                if isinstance(result, list):
                    return result
            except Exception:
                pass
            # Try to split by common delimiters
            if "," in response:
                return [item.strip() for item in response.split(",")]
            return [response]

        elif expected_type == "format":
            # Extract format keyword
            formats = ["json", "yaml", "xml", "csv", "text"]
            response_lower = response.lower()
            for fmt in formats:
                if fmt in response_lower:
                    return fmt
            return "json"  # Default

        elif expected_type == "choice":
            # Return the response as-is for choices
            return response

        else:
            # String - clean up the response
            return response.strip("\"'")

    def clear_cache(self) -> None:
        """Clear the resolution cache."""
        self.resolution_cache.clear()

    def get_cache_size(self) -> int:
        """Get the size of the resolution cache."""
        return len(self.resolution_cache)

    def set_resolution_strategy(self, ambiguity_type: str, strategy_func) -> None:
        """Set custom resolution strategy for ambiguity type."""
        self.resolution_strategies[ambiguity_type] = strategy_func
