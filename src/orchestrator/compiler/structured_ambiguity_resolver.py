"""Enhanced ambiguity resolver using structured outputs for reliable formatting."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ValidationError

from ..core.model import Model
from ..models.model_registry import ModelRegistry
from .ambiguity_resolver import AmbiguityResolutionError
from .utils import async_retry

logger = logging.getLogger(__name__)


class BooleanResponse(BaseModel):
    """Schema for boolean responses."""

    value: bool = Field(description="The boolean value (true or false)")
    reasoning: Optional[str] = Field(
        None, description="Brief explanation for the choice"
    )


class NumberResponse(BaseModel):
    """Schema for numeric responses."""

    value: Union[int, float] = Field(description="The numeric value")
    unit: Optional[str] = Field(None, description="Unit of measurement if applicable")
    reasoning: Optional[str] = Field(
        None, description="Brief explanation for the choice"
    )


class ListResponse(BaseModel):
    """Schema for list responses."""

    items: List[str] = Field(description="List of string items")
    reasoning: Optional[str] = Field(
        None, description="Brief explanation for the choices"
    )


class ChoiceResponse(BaseModel):
    """Schema for choice responses."""

    choice: str = Field(description="The selected choice")
    reasoning: Optional[str] = Field(
        None, description="Brief explanation for the choice"
    )


class StringResponse(BaseModel):
    """Schema for string responses."""

    value: str = Field(description="The string value")
    reasoning: Optional[str] = Field(None, description="Brief explanation")


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
                    "supports_structured_output": True,
                }
                try:
                    self.model = await self.model_registry.select_model(requirements)
                    logger.info(
                        f"Selected model with structured output support: {self.model.name}"
                    )
                    logger.debug(f"Model capabilities: {self.model.capabilities}")
                except Exception as e:
                    logger.warning(f"No model with structured output found: {e}")
                    # Fall back to any model that can generate
                    self.model = await self.model_registry.select_model(
                        {"tasks": ["generate"]}
                    )
                    logger.info(f"Selected fallback model: {self.model.name}")
                    logger.debug(f"Model capabilities: {self.model.capabilities}")

            if not self.model:
                raise AmbiguityResolutionError("No AI model available for resolution")

            logger.info(f"Ambiguity resolver initialized with model: {self.model.name}")

            # Determine expected type from content and context
            expected_type = self._infer_type_from_context(content, context_path)
            logger.debug(
                f"Inferred type '{expected_type}' for content: {content[:50]}..."
            )

            # Resolve using structured output if supported
            # Check if model has generate_structured method
            if hasattr(self.model, "generate_structured") and callable(
                getattr(self.model, "generate_structured")
            ):
                logger.debug(
                    f"Using structured output resolution with model {self.model.name}"
                )
                result = await self._resolve_with_structured_output(
                    content, expected_type
                )
            else:
                # Fall back to traditional parsing
                logger.debug(
                    f"Using traditional parsing (model {self.model.name} doesn't have generate_structured method)"
                )
                result = await self._resolve_with_parsing(
                    content, context_path, expected_type
                )

            # Log result
            logger.info(
                f"Resolved '{content[:30]}...' to: {result} (type: {type(result).__name__})"
            )

            # Cache result
            self.resolution_cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"Failed to resolve ambiguity: {e}")
            raise AmbiguityResolutionError(
                f"Failed to resolve ambiguity '{content}' at {context_path}: {e}"
            ) from e

    async def _resolve_with_structured_output(
        self, content: str, expected_type: str
    ) -> Any:
        """Use structured output for reliable formatting."""

        # Build prompt with clear instructions
        prompt = self._build_structured_prompt(content, expected_type)

        # Get appropriate schema based on expected type
        schema = self._get_schema_for_type(expected_type)

        try:
            # Use generate_structured method with retry
            logger.debug(f"Calling generate_structured with schema: {schema}")

            @async_retry(exceptions=(Exception,), max_attempts=3, delay=1.0)
            async def _call_structured():
                return await self.model.generate_structured(
                    prompt=prompt,
                    schema=schema,
                    temperature=0.1,  # Low temperature for consistency
                )

            response = await _call_structured()
            logger.debug(f"Structured output response: {response}")

            # Validate and extract value from structured response
            result = self._validate_and_extract(response, expected_type)
            logger.debug(f"Extracted value: {result} (type: {type(result).__name__})")
            return result

        except Exception as e:
            # If structured output fails, fall back to parsing
            logger.warning(
                f"Structured output failed for '{content[:30]}...', falling back to parsing: {e}"
            )
            import traceback

            logger.debug(f"Traceback: {traceback.format_exc()}")
            return await self._resolve_with_parsing(content, "", expected_type)

    async def _resolve_with_parsing(
        self, content: str, context_path: str, expected_type: str
    ) -> Any:
        """Fall back to traditional response parsing."""
        try:
            prompt = self._build_fallback_prompt(content, expected_type)

            @async_retry(exceptions=(Exception,), max_attempts=2, delay=0.5)
            async def _call_generate():
                return await self.model.generate(
                    prompt, temperature=0.1, max_tokens=100
                )

            response = await _call_generate()
            return self._parse_response(response, expected_type, content)
        except Exception as e:
            logger.error(f"Fallback parsing failed for '{content[:30]}...': {e}")
            raise AmbiguityResolutionError(
                f"Both structured output and fallback parsing failed for '{content[:50]}...': {e}"
            ) from e

    def _infer_type_from_context(self, content: str, context_path: str) -> str:
        """Infer the expected return type from content and context."""
        content_lower = content.lower()

        # Check for explicit type indicators in content
        # Check number indicators first (more specific)
        if any(
            word in content_lower
            for word in [
                "number",
                "count",
                "size",
                "how many",
                "quantity",
                "amount",
                "batch size",
                "workers",
                "threads",
                "timeout",
                "duration",
                "retries",
                "attempts",
                "iterations",
            ]
        ):
            return "number"
        elif (
            "list" in content_lower
            or "array" in content_lower
            or (
                "separated" in content_lower
                and ("comma" in content_lower or "items" in content_lower)
            )
            or (
                "items" in content_lower
                and any(
                    word in content_lower
                    for word in ["what", "which", "provide", "list"]
                )
            )
            or (
                "top" in content_lower
                and any(char in content for char in ["3", "5", "10"])
            )
            or (
                "example" in content_lower
                and any(str(i) in content for i in range(2, 21))
            )
            or any(
                phrase in content_lower
                for phrase in [
                    "enumerate",
                    "name all",
                    "what are the",
                    "which are the",
                    "give me",
                ]
            )
        ):
            return "list"
        elif any(
            phrase in content_lower
            for phrase in [
                "true or false",
                "yes or no",
                "enable or disable",
                "answer true",
                "answer false",
                "answer yes",
                "answer no",
            ]
        ) or (
            any(
                phrase in content_lower
                for phrase in [
                    "should we",
                    "is it",
                    "are we",
                    "do we",
                    "can we",
                    "is ",
                    "should i",
                    "would it",
                ]
            )
            and "?" in content
            and not any(
                word in content_lower
                for word in [
                    "name",
                    "call",
                    "title",
                    "what",
                    "which",
                    "who",
                    "where",
                    "when",
                    "how",
                    "many",
                    "much",
                ]
            )
        ):
            return "boolean"
        elif any(
            word in content_lower
            for word in ["choose", "select", "which", "what type", "what kind"]
        ) and any(sep in content_lower for sep in [" or ", ", or ", ", "]):
            return "choice"

        # Check context path for additional hints
        path_lower = context_path.lower()
        if any(
            word in path_lower
            for word in ["count", "size", "num", "max", "min", "limit"]
        ):
            return "number"
        elif any(
            word in path_lower
            for word in ["enable", "disable", "is_", "has_", "should_"]
        ):
            return "boolean"
        elif "strategy" in path_lower or "method" in path_lower or "mode" in path_lower:
            return "choice"

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
            "boolean": "Determine if the statement is true or false. Return a JSON object with a 'value' field containing true or false.",
            "number": "Extract or calculate the numeric value. Return a JSON object with a 'value' field containing the number.",
            "list": "Create a list of items based on the request. Return a JSON object with an 'items' field containing an array of strings.",
            "choice": "Select the most appropriate option. Return a JSON object with a 'choice' field containing the selected option.",
            "string": "Provide a concise text response. Return a JSON object with a 'value' field containing the text.",
        }

        description = type_descriptions.get(expected_type, type_descriptions["string"])

        # Add schema examples for clarity
        examples = {
            "boolean": '{"value": true, "reasoning": "optional explanation"}',
            "number": '{"value": 42, "unit": "optional unit", "reasoning": "optional explanation"}',
            "list": '{"items": ["item1", "item2", "item3"], "reasoning": "optional explanation"}',
            "choice": '{"choice": "selected_option", "reasoning": "optional explanation"}',
            "string": '{"value": "your response", "reasoning": "optional explanation"}',
        }

        example = examples.get(expected_type, examples["string"])

        return f"""Task: {description}

Question: {content}

Return a JSON object following this exact structure:
{example}

The reasoning field is optional but the primary field is required."""

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

    def _validate_and_extract(
        self, response: Dict[str, Any], expected_type: str
    ) -> Any:
        """Validate response with Pydantic model and extract value."""
        try:
            if expected_type == "boolean":
                validated = BooleanResponse(**response)
                return validated.value
            elif expected_type == "number":
                validated = NumberResponse(**response)
                return validated.value
            elif expected_type == "list":
                validated = ListResponse(**response)
                return validated.items
            elif expected_type == "choice":
                validated = ChoiceResponse(**response)
                return validated.choice
            else:  # string
                validated = StringResponse(**response)
                return validated.value
        except ValidationError as e:
            logger.warning(f"Validation error for {expected_type}: {e}")
            # Try to extract value directly if validation fails
            return self._extract_value_fallback(response, expected_type)

    def _extract_value_fallback(
        self, response: Dict[str, Any], expected_type: str
    ) -> Any:
        """Fallback extraction when Pydantic validation fails."""
        if expected_type == "boolean":
            # Try to find any boolean-like value
            for key in ["value", "result", "answer", "response"]:
                if key in response:
                    val = response[key]
                    if isinstance(val, bool):
                        return val
                    elif isinstance(val, str):
                        return val.lower() in ["true", "yes", "1"]
            return False
        elif expected_type == "number":
            # Try to find any numeric value
            for key in ["value", "result", "number", "answer"]:
                if key in response and isinstance(response[key], (int, float)):
                    return response[key]
            return 0
        elif expected_type == "list":
            # Try to find any list-like value
            for key in ["items", "list", "values", "results"]:
                if key in response and isinstance(response[key], list):
                    return response[key]
            return []
        elif expected_type == "choice":
            # Try to find any string value
            for key in ["choice", "selected", "option", "value"]:
                if key in response and isinstance(response[key], str):
                    return response[key]
            return ""
        else:  # string
            # Try to find any string value
            for key in ["value", "text", "result", "answer"]:
                if key in response and isinstance(response[key], str):
                    return response[key]
            return ""

    def _parse_response(
        self, response: str, expected_type: str, original_content: str
    ) -> Any:
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

            numbers = re.findall(r"-?\d+\.?\d*", response)
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

                match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
                if match:
                    response = match.group(1).strip()

            # Try JSON parsing
            try:
                result = json.loads(response)
                if isinstance(result, list):
                    return [str(item) for item in result]
            except Exception:
                pass

            # Try comma separation
            if "," in response:
                return [item.strip() for item in response.split(",") if item.strip()]

            # Single item
            return [response] if response else []

        elif expected_type == "choice":
            # Return first non-empty line
            lines = response.split("\n")
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    return line
            return response

        else:  # string
            return response
