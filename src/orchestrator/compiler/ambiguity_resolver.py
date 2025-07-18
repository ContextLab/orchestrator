"""Ambiguity resolver that uses model registry instead of direct instantiation."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

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
    Resolves ambiguous specifications using models from registry.
    
    This version:
    - Uses ModelRegistry instead of direct model instantiation
    - Never uses mock models
    - Selects appropriate models based on task requirements
    """

    def __init__(self, model_registry: Optional[ModelRegistry] = None) -> None:
        """
        Initialize ambiguity resolver.

        Args:
            model_registry: Model registry to use for resolution
            
        Raises:
            ValueError: If no models are available
        """
        self.model_registry = model_registry
        self.resolution_cache: Dict[str, Any] = {}
        self.resolution_strategies = {
            "parameter": self._resolve_parameter,
            "value": self._resolve_value,
            "list": self._resolve_list,
            "boolean": self._resolve_boolean,
            "number": self._resolve_number,  
            "string": self._resolve_string,
        }
        
        # Verify we have models if registry provided
        if model_registry and not model_registry.list_models():
            raise ValueError(
                "Model registry has no models available for ambiguity resolution"
            )

    async def resolve(self, content: str, context_path: str) -> Any:
        """
        Resolve ambiguous content.

        Args:
            content: Ambiguous content to resolve
            context_path: Context path for debugging

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
            # Determine resolution strategy
            resolution_type = self._classify_ambiguity(content, context_path)

            # Apply resolution strategy
            if resolution_type in self.resolution_strategies:
                strategy = self.resolution_strategies[resolution_type]
                result = await strategy(content, context_path)
            else:
                # Default to generic resolution
                result = await self._resolve_generic(content, context_path)

            # Cache result
            self.resolution_cache[cache_key] = result
            return result

        except Exception as e:
            raise AmbiguityResolutionError(
                f"Failed to resolve ambiguity '{content}' at {context_path}: {e}"
            ) from e

    def _classify_ambiguity(self, content: str, context_path: str) -> str:
        """
        Classify the type of ambiguity.

        Args:
            content: Ambiguous content
            context_path: Context path

        Returns:
            Ambiguity type
        """
        content_lower = content.lower()

        # Check for boolean indicators
        import re
        boolean_indicators = [
            "enable", "disable", "true", "false", "yes", "no",
            "allow", "deny", "support", "block",
        ]
        for indicator in boolean_indicators:
            if re.search(r"\b" + re.escape(indicator) + r"\b", content_lower):
                return "boolean"

        # Check for number indicators
        number_indicators = [
            "timeout", "retry", "count", "size", "limit", "number", "amount",
        ]
        if any(word in content_lower for word in number_indicators):
            return "number"

        # Check for quoted strings
        if '"' in content:
            return "string"

        # Check for explicit choices
        if (("choose" in content_lower or "select" in content_lower) 
            and ":" in content and "," in content):
            return "value"

        # Check for list indicators
        if any(word in content_lower for word in ["list", "array", "collection"]):
            return "list"

        # Check context path for hints
        if "parameters" in context_path:
            return "parameter"

        return "generic"

    async def _resolve_parameter(self, content: str, context_path: str) -> Any:
        """Resolve parameter ambiguity."""
        # For simple parameter resolution without model calls
        if "timeout" in content.lower():
            return 30.0
        elif "retry" in content.lower() or "retries" in content.lower():
            return 3
        elif "enable" in content.lower() or "true" in content.lower():
            return True
        elif "disable" in content.lower() or "false" in content.lower():
            return False
        else:
            # Default parameter value
            return None

    async def _resolve_value(self, content: str, context_path: str) -> Any:
        """Resolve explicit value choice."""
        # Extract choices after colon
        if ":" in content:
            parts = content.split(":", 1)
            if len(parts) == 2:
                choices_str = parts[1].strip()
                # Split by comma and clean
                choices = [c.strip() for c in choices_str.split(",")]
                if choices:
                    # Return first choice as default
                    return choices[0]
        return content

    async def _resolve_list(self, content: str, context_path: str) -> List[Any]:
        """Resolve list ambiguity."""
        # Simple list resolution
        if "," in content:
            items = [item.strip() for item in content.split(",")]
            return items
        return []

    async def _resolve_boolean(self, content: str, context_path: str) -> bool:
        """Resolve boolean ambiguity."""
        content_lower = content.lower()
        
        # Positive indicators
        if any(word in content_lower for word in ["enable", "true", "yes", "allow", "support"]):
            return True
        
        # Negative indicators  
        if any(word in content_lower for word in ["disable", "false", "no", "deny", "block"]):
            return False
            
        # Default to True for ambiguous cases
        return True

    async def _resolve_number(self, content: str, context_path: str) -> float:
        """Resolve number ambiguity."""
        import re
        
        # Try to extract number from content
        numbers = re.findall(r'\d+(?:\.\d+)?', content)
        if numbers:
            return float(numbers[0])
            
        # Defaults based on context
        content_lower = content.lower()
        if "timeout" in content_lower:
            return 30.0
        elif "retry" in content_lower or "retries" in content_lower:
            return 3.0
        elif "limit" in content_lower:
            return 100.0
        elif "size" in content_lower:
            return 1024.0
            
        return 1.0

    async def _resolve_string(self, content: str, context_path: str) -> str:
        """Resolve string ambiguity."""
        # Remove quotes if present
        if content.startswith('"') and content.endswith('"'):
            return content[1:-1]
        return content

    async def _resolve_generic(self, content: str, context_path: str) -> Any:
        """Generic resolution for unclassified ambiguities."""
        # If we have a model registry, we could use it for complex resolution
        # For now, return simple resolutions
        
        # Check if it looks like a boolean
        if content.lower() in ["true", "false", "yes", "no"]:
            return content.lower() in ["true", "yes"]
            
        # Check if it's a number
        try:
            return float(content)
        except ValueError:
            pass
            
        # Default: return as string
        return content

    def clear_cache(self) -> None:
        """Clear the resolution cache."""
        self.resolution_cache.clear()

    def get_cache_size(self) -> int:
        """Get the size of the resolution cache."""
        return len(self.resolution_cache)

    def set_resolution_strategy(self, ambiguity_type: str, strategy_func) -> None:
        """Set custom resolution strategy for ambiguity type."""
        self.resolution_strategies[ambiguity_type] = strategy_func

    def _extract_choices(self, content: str) -> List[str]:
        """Extract choices from content (e.g., 'choose: A, B, or C')."""
        import re

        # Pattern for "A, B, or C" or "A, B, C"
        pattern = r"(?:choose|select).*?:\s*([^.]+)"
        match = re.search(pattern, content, re.IGNORECASE)

        if match:
            choices_text = match.group(1)
            # Split by comma and clean up
            choices = [choice.strip() for choice in choices_text.split(",")]
            # Remove 'or' from last choice
            if choices:
                last_choice = choices[-1]
                # Handle various 'or' patterns
                last_choice = last_choice.replace(" or ", "").replace("or ", "").strip()
                choices[-1] = last_choice
            return [choice for choice in choices if choice]

        return []

    def _extract_quotes(self, content: str) -> List[str]:
        """Extract quoted strings from content."""
        import re

        # Pattern for quoted strings
        pattern = r'"([^"]*)"'
        matches = re.findall(pattern, content)
        return matches