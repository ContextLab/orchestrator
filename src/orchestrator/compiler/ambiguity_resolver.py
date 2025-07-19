"""Ambiguity resolver that uses AI models for intelligent resolution."""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Union

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
        self, 
        model: Optional[Model] = None,
        model_registry: Optional[ModelRegistry] = None,
        fallback_to_heuristics: bool = False
    ) -> None:
        """
        Initialize ambiguity resolver.

        Args:
            model: Specific model to use for resolution
            model_registry: Model registry to select models from
            fallback_to_heuristics: Whether to fall back to heuristics if AI fails
            
        Raises:
            ValueError: If no model is provided and registry has no models
        """
        self.model = model
        self.model_registry = model_registry
        self.fallback_to_heuristics = fallback_to_heuristics
        self.resolution_cache: Dict[str, Any] = {}
        
        # Heuristic strategies for fallback
        self.resolution_strategies = {
            "parameter": self._resolve_parameter_heuristic,
            "value": self._resolve_value_heuristic,
            "list": self._resolve_list_heuristic,
            "boolean": self._resolve_boolean_heuristic,
            "number": self._resolve_number_heuristic,
            "string": self._resolve_string_heuristic,
        }
        
        # If no model provided, try to get one from registry
        if not model and model_registry:
            available_models = model_registry.list_models()
            if available_models:
                # Try to get a good model for text generation
                try:
                    self.model = model_registry.select_model({"tasks": ["generate"]})
                except:
                    # If selection fails, just get the first available
                    self.model = model_registry.get_model(available_models[0])
        
        # Verify we have a model for AI resolution
        if not self.model and not self.fallback_to_heuristics:
            raise ValueError(
                "No AI model available for ambiguity resolution. "
                "Either provide a model or enable fallback_to_heuristics."
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
            # Try AI resolution first if we have a model
            if self.model:
                result = await self._resolve_with_ai(content, context_path)
            elif self.fallback_to_heuristics:
                # Fall back to heuristic resolution
                result = await self._resolve_with_heuristics(content, context_path)
            else:
                raise AmbiguityResolutionError(
                    "No AI model available and heuristic fallback is disabled"
                )

            # Cache result
            self.resolution_cache[cache_key] = result
            return result

        except Exception as e:
            if self.fallback_to_heuristics and self.model:
                # If AI failed, try heuristics
                try:
                    result = await self._resolve_with_heuristics(content, context_path)
                    self.resolution_cache[cache_key] = result
                    return result
                except:
                    pass
            
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
        if any(word in content_lower for word in ["true", "false", "enable", "disable", "yes", "no"]):
            return "boolean"
        elif any(word in content_lower for word in ["number", "count", "size", "batch", "timeout", "limit"]):
            return "number"
        elif "list" in content_lower or "array" in content_lower:
            return "list"
        elif re.search(r'\b(json|yaml|xml|csv|text)\b', content_lower):
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
            "string": "Respond with a short string value."
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
            numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
            if numbers:
                return float(numbers[0]) if '.' in numbers[0] else int(numbers[0])
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
            except:
                pass
            # Try to split by common delimiters
            if ',' in response:
                return [item.strip() for item in response.split(',')]
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
            return response.strip('"\'')

    async def _resolve_with_heuristics(self, content: str, context_path: str) -> Any:
        """Fall back to heuristic resolution."""
        # Classify ambiguity type
        resolution_type = self._classify_ambiguity(content, context_path)
        
        # Apply resolution strategy
        if resolution_type in self.resolution_strategies:
            strategy = self.resolution_strategies[resolution_type]
            return await strategy(content, context_path)
        else:
            # Default to string
            return content

    def _classify_ambiguity(self, content: str, context_path: str) -> str:
        """Classify the type of ambiguity for heuristic resolution."""
        content_lower = content.lower()

        # Check for explicit choice patterns
        if ("choose" in content_lower or "select" in content_lower):
            if any(word in content_lower for word in ["true", "false"]):
                return "boolean"
            elif any(word in content_lower for word in ["number", "size", "count", "amount"]):
                return "number"
            elif "list" in content_lower or "array" in content_lower:
                return "list"
            else:
                return "value"

        # Check for parameter context
        if "parameters" in context_path:
            return "parameter"

        # Check for specific patterns
        if re.search(r'"[^"]*"', content):
            return "string"
        
        # Context-based classification
        path_parts = context_path.lower().split('.')
        if any(part in ["format", "type", "style", "method"] for part in path_parts):
            return "string"
        elif any(part in ["enable", "disable", "support"] for part in path_parts):
            return "boolean"
        elif any(part in ["count", "size", "limit", "timeout", "batch"] for part in path_parts):
            return "number"
        elif any(part in ["languages", "formats", "items", "sources", "tags", "options", "list"] for part in path_parts):
            return "list"
        
        return "string"

    # Heuristic resolution methods
    async def _resolve_parameter_heuristic(self, content: str, context_path: str) -> Any:
        """Heuristic parameter resolution."""
        # Check context for common patterns
        if "format" in context_path:
            return "json"
        elif "method" in context_path:
            return "default"
        elif "type" in context_path:
            return "auto"
        return "default"

    async def _resolve_value_heuristic(self, content: str, context_path: str) -> Any:
        """Heuristic value resolution."""
        # Extract choices if present
        choices = self._extract_choices(content)
        if choices:
            return choices[0]
        
        # Check for specific patterns
        if "format" in context_path:
            return "json"
        elif "method" in context_path:
            return "auto"
        elif "style" in context_path:
            return "default"
        
        return "default"

    async def _resolve_list_heuristic(self, content: str, context_path: str) -> List[str]:
        """Heuristic list resolution."""
        # Context-based defaults
        if "source" in context_path:
            return ["web", "documents", "database"]
        elif "format" in context_path:
            return ["json", "csv", "xml"]
        elif "language" in context_path:
            return ["en", "es", "fr"]
        return ["item1", "item2", "item3"]

    async def _resolve_boolean_heuristic(self, content: str, context_path: str) -> bool:
        """Heuristic boolean resolution."""
        content_lower = content.lower()
        
        # Check for explicit indicators
        positive = ["enable", "true", "yes", "allow", "support"]
        negative = ["disable", "false", "no", "deny", "block"]
        
        for word in positive:
            if word in content_lower:
                return True
        
        for word in negative:
            if word in content_lower:
                return False
        
        # Context-based defaults
        if "enable" in context_path or "support" in context_path:
            return True
        
        return False

    async def _resolve_number_heuristic(self, content: str, context_path: str) -> Union[int, float]:
        """Heuristic number resolution."""
        # Context-based defaults
        if "batch" in context_path:
            return 32
        elif "timeout" in context_path:
            return 30
        elif "retry" in context_path:
            return 3
        elif "size" in context_path:
            return 100
        elif "limit" in context_path:
            return 1000
        return 10

    async def _resolve_string_heuristic(self, content: str, context_path: str) -> str:
        """Heuristic string resolution."""
        # Extract quoted content
        quotes = self._extract_quotes(content)
        if quotes:
            return quotes[0]
        
        # Context-based defaults
        if "query" in context_path:
            return "default search query"
        elif "format" in context_path:
            return "json"
        elif "method" in context_path:
            return "auto"
        elif "style" in context_path:
            return "default"
        elif "type" in context_path:
            return "standard"
        
        return "default"

    def _extract_choices(self, content: str) -> List[str]:
        """Extract choices from content."""
        # Pattern for "Choose: A, B, C" or "Select from: A, B, C"
        pattern = r'(?:choose|select)(?:\s+from)?:\s*([^.]+)'
        match = re.search(pattern, content, re.IGNORECASE)
        
        if match:
            choices_text = match.group(1)
            # Split by comma and clean
            choices = [c.strip() for c in choices_text.split(',')]
            # Remove 'or' from choices
            choices = [c.replace(' or ', '').replace('or ', '').strip() for c in choices]
            return [c for c in choices if c]
        
        return []

    def _extract_quotes(self, content: str) -> List[str]:
        """Extract quoted strings from content."""
        pattern = r'"([^"]*)"'
        return re.findall(pattern, content)

    def _fallback_resolution(self, content: str, context_path: str) -> Any:
        """Ultimate fallback resolution."""
        # This is used by tests that expect specific behavior
        if "query" in context_path:
            return "default search query"
        elif "format" in context_path:
            return "json"
        elif "method" in context_path:
            return "auto"
        elif "style" in context_path:
            return "default"
        elif "type" in context_path:
            return "standard"
        return "default"

    def clear_cache(self) -> None:
        """Clear the resolution cache."""
        self.resolution_cache.clear()

    def get_cache_size(self) -> int:
        """Get the size of the resolution cache."""
        return len(self.resolution_cache)

    def set_resolution_strategy(self, ambiguity_type: str, strategy_func) -> None:
        """Set custom resolution strategy for ambiguity type."""
        self.resolution_strategies[ambiguity_type] = strategy_func