"""Enhanced ambiguity resolver using structured outputs for reliable formatting."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ..core.model import Model
from ..models.model_registry import ModelRegistry
from .ambiguity_resolver import AmbiguityResolutionError


class BooleanResponse(BaseModel):
    """Schema for boolean responses."""
    value: bool = Field(description="The boolean value (true or false)")


class NumberResponse(BaseModel):
    """Schema for numeric responses."""
    value: Union[int, float] = Field(description="The numeric value")


class ListResponse(BaseModel):
    """Schema for list responses."""
    items: List[str] = Field(description="List of string items")


class ChoiceResponse(BaseModel):
    """Schema for choice responses."""
    choice: str = Field(description="The selected choice")


class StringResponse(BaseModel):
    """Schema for string responses."""
    value: str = Field(description="The string value")


class StructuredAmbiguityResolver:
    """
    Enhanced ambiguity resolver using structured outputs.
    
    This resolver leverages structured output capabilities of models
    to ensure responses are always in the expected format.
    """

    def __init__(
        self,
        model: Optional[Model] = None,
        model_registry: Optional[ModelRegistry] = None,
    ) -> None:
        """
        Initialize structured ambiguity resolver.

        Args:
            model: Specific model to use for resolution
            model_registry: Model registry to select models from

        Raises:
            ValueError: If no model is provided and registry has no models
        """
        self.model = model
        self.model_registry = model_registry
        self.resolution_cache: Dict[str, Any] = {}

        # If no model provided and we have a registry, defer model selection
        if not model and model_registry:
            available_models = model_registry.list_models()
            if not available_models:
                raise ValueError(
                    "No AI model available for ambiguity resolution. "
                    "A real model must be provided."
                )
        elif not model and not model_registry:
            raise ValueError(
                "No AI model available for ambiguity resolution. "
                "A real model must be provided."
            )

    async def resolve(self, content: str, context_path: str) -> Any:
        """
        Resolve ambiguous content using structured outputs.

        Args:
            content: Ambiguous content to resolve
            context_path: Context path (e.g., "config.output.format")

        Returns:
            Resolved value in the appropriate type

        Raises:
            AmbiguityResolutionError: If resolution fails
        """
        # Check cache first
        cache_key = f"{content}:{context_path}"
        if cache_key in self.resolution_cache:
            return self.resolution_cache[cache_key]

        try:
            # Ensure we have a model
            if not self.model and self.model_registry:
                # Try to get a model that supports structured output
                requirements = {
                    "tasks": ["generate"],
                    "capabilities": ["structured_output"]
                }
                try:
                    self.model = await self.model_registry.select_model(requirements)
                except Exception:
                    # Fall back to any model that can generate
                    self.model = await self.model_registry.select_model({"tasks": ["generate"]})

            if not self.model:
                raise AmbiguityResolutionError("No AI model available for resolution")

            # Determine expected type from content and context
            expected_type = self._infer_type_from_context(content, context_path)
            
            # Resolve using structured output if supported
            if hasattr(self.model.capabilities, 'supports_structured_output') and \
               self.model.capabilities.supports_structured_output:
                result = await self._resolve_with_structured_output(content, expected_type)
            else:
                # Fall back to traditional parsing
                result = await self._resolve_with_parsing(content, context_path, expected_type)

            # Cache result
            self.resolution_cache[cache_key] = result
            return result

        except Exception as e:
            raise AmbiguityResolutionError(
                f"Failed to resolve ambiguity '{content}' at {context_path}: {e}"
            ) from e

    async def _resolve_with_structured_output(self, content: str, expected_type: str) -> Any:
        """Use structured output for reliable formatting."""
        
        # Build prompt with clear instructions
        prompt = self._build_structured_prompt(content, expected_type)
        
        # Get appropriate schema based on expected type
        schema = self._get_schema_for_type(expected_type)
        
        try:
            # Use generate_structured method
            response = await self.model.generate_structured(
                prompt=prompt,
                schema=schema,
                temperature=0.1  # Low temperature for consistency
            )
            
            # Extract value from structured response
            return self._extract_value_from_structured(response, expected_type)
            
        except Exception as e:
            # If structured output fails, fall back to parsing
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Structured output failed, falling back to parsing: {e}")
            return await self._resolve_with_parsing(content, "", expected_type)

    async def _resolve_with_parsing(self, content: str, context_path: str, expected_type: str) -> Any:
        """Fall back to traditional response parsing."""
        prompt = self._build_fallback_prompt(content, expected_type)
        response = await self.model.generate(prompt, temperature=0.1, max_tokens=100)
        return self._parse_response(response, expected_type, content)

    def _infer_type_from_context(self, content: str, context_path: str) -> str:
        """Infer the expected return type from content and context."""
        content_lower = content.lower()

        # Check for explicit type indicators
        if "list" in content_lower or "array" in content_lower or \
           ("separated" in content_lower and "commas" in content_lower):
            return "list"
        elif any(word in content_lower for word in 
                ["number", "count", "size", "how many", "quantity"]):
            return "number"
        elif any(word in content_lower for word in 
                ["true", "false", "yes", "no", "enable", "disable"]) or \
             "answer with 'true' or 'false'" in content_lower:
            return "boolean"
        elif "choose" in content_lower or "select" in content_lower or \
             "which" in content_lower:
            return "choice"
        else:
            return "string"

    def _get_schema_for_type(self, expected_type: str) -> Dict[str, Any]:
        """Get JSON schema for the expected type."""
        schemas = {
            "boolean": BooleanResponse.model_json_schema(),
            "number": NumberResponse.model_json_schema(),
            "list": ListResponse.model_json_schema(),
            "choice": ChoiceResponse.model_json_schema(),
            "string": StringResponse.model_json_schema(),
        }
        return schemas.get(expected_type, schemas["string"])

    def _build_structured_prompt(self, content: str, expected_type: str) -> str:
        """Build prompt for structured output."""
        type_descriptions = {
            "boolean": "Determine if the statement is true or false.",
            "number": "Extract or calculate the numeric value.",
            "list": "Create a list of items based on the request.",
            "choice": "Select the most appropriate option.",
            "string": "Provide a concise text response.",
        }
        
        description = type_descriptions.get(expected_type, type_descriptions["string"])
        
        return f"""Task: {description}

Question: {content}

Provide your answer in the required format."""

    def _build_fallback_prompt(self, content: str, expected_type: str) -> str:
        """Build prompt for fallback parsing."""
        type_instructions = {
            "boolean": "Answer with exactly one word: either 'true' or 'false'.",
            "number": "Answer with only a number (integer or decimal).",
            "list": "Answer with items separated by commas, no other formatting.",
            "choice": "Answer with only the chosen option, no explanation.",
            "string": "Answer with a short string value.",
        }
        
        instruction = type_instructions.get(expected_type, type_instructions["string"])
        
        return f"""{content}

{instruction}
Do not explain. Answer only."""

    def _extract_value_from_structured(self, response: Dict[str, Any], expected_type: str) -> Any:
        """Extract value from structured response."""
        if expected_type == "boolean":
            return response.get("value", False)
        elif expected_type == "number":
            return response.get("value", 0)
        elif expected_type == "list":
            return response.get("items", [])
        elif expected_type == "choice":
            return response.get("choice", "")
        else:  # string
            return response.get("value", "")

    def _parse_response(self, response: str, expected_type: str, original_content: str) -> Any:
        """Parse unstructured response based on expected type."""
        response = response.strip()
        
        if expected_type == "boolean":
            response_lower = response.lower()
            if "true" in response_lower or "yes" in response_lower:
                return True
            elif "false" in response_lower or "no" in response_lower:
                return False
            # Default based on content
            return "yes" in original_content.lower()
            
        elif expected_type == "number":
            # Try to extract number
            import re
            numbers = re.findall(r'-?\d+\.?\d*', response)
            if numbers:
                try:
                    # Try float first
                    return float(numbers[0])
                except ValueError:
                    return int(numbers[0])
            return 0
            
        elif expected_type == "list":
            # Handle markdown code blocks
            if "```" in response:
                # Extract content between code blocks
                import re
                match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
                if match:
                    response = match.group(1).strip()
            
            # Try JSON parsing
            try:
                result = json.loads(response)
                if isinstance(result, list):
                    return [str(item) for item in result]
            except:
                pass
            
            # Try comma separation
            if "," in response:
                return [item.strip() for item in response.split(",") if item.strip()]
            
            # Single item
            return [response] if response else []
            
        elif expected_type == "choice":
            # Return first non-empty line
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    return line
            return response
            
        else:  # string
            return response