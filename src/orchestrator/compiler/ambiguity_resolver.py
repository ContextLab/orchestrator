"""Ambiguity resolver for AUTO tag resolution."""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from ..core.model import Model


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
    Resolves ambiguous specifications using LLMs.
    
    The resolver analyzes AUTO tag content and generates appropriate
    concrete values based on context and task requirements.
    """
    
    def __init__(self, model: Optional[Model] = None, fallback_to_mock: bool = True) -> None:
        """
        Initialize ambiguity resolver.
        
        Args:
            model: Model to use for resolution (uses mock if None)
            fallback_to_mock: Whether to fallback to mock model if real model unavailable
        """
        if model is None:
            # Try to use a real model first
            try:
                from ..integrations.openai_model import OpenAIModel
                model = OpenAIModel()
            except Exception:
                try:
                    from ..integrations.anthropic_model import AnthropicModel
                    model = AnthropicModel()
                except Exception:
                    if fallback_to_mock:
                        # Fallback to mock model
                        from ..core.model import MockModel
                        model = MockModel()
                    else:
                        raise RuntimeError("No AI model available for ambiguity resolution")
        
        self.model = model
        self.resolution_cache: Dict[str, Any] = {}
        self.resolution_strategies = {
            "parameter": self._resolve_parameter,
            "value": self._resolve_value,
            "list": self._resolve_list,
            "boolean": self._resolve_boolean,
            "number": self._resolve_number,
            "string": self._resolve_string,
        }
    
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
        
        # Check for boolean indicators first (use word boundaries for more precision)
        import re
        boolean_indicators = ["enable", "disable", "true", "false", "yes", "no", "allow", "deny", "support", "block"]
        for indicator in boolean_indicators:
            # Use word boundaries to avoid matching partial words like "supported" matching "support"
            if re.search(r'\b' + re.escape(indicator) + r'\b', content_lower):
                return "boolean"
        
        # Check for number indicators (but not if it's quoted content)
        number_indicators = ["timeout", "retry", "count", "size", "limit", "number", "amount"]
        if any(word in content_lower for word in number_indicators):
            return "number"
        
        # Check for quoted strings (should be handled as string)
        if '"' in content:
            return "string"
        
        # Check for explicit choices first - this is the most specific pattern
        if ("choose" in content_lower or "select" in content_lower) and ":" in content and "," in content:
            # Looks like explicit choices (e.g., "choose: a, b, c", "select from: a, b, c")
            return "value"
        
        # Check for specific patterns in content first (most specific)
        if "choose" in content_lower or "select" in content_lower:
            # Check content patterns first (more specific than generic choose)
            if "true" in content_lower or "false" in content_lower:
                return "boolean"
            elif any(word in content_lower for word in ["number", "size", "count", "amount"]):
                return "number"
            elif "list" in content_lower or "array" in content_lower or "items" in content_lower or "languages" in content_lower:
                return "list"
            # Check for strong boolean context hints even with choose/select
            elif (any(pattern in context_path for pattern in ["enable_", "disable_", "support_", "allow_", "deny_"]) or
                  # Also consider specific option/config paths that might be boolean
                  ("other_option" in context_path and "config." in context_path)):
                return "boolean"
            # Check for parameter context (should take precedence over generic "value")
            elif "parameters" in context_path:
                return "parameter"
            # Default to value for choose/select patterns
            else:
                return "value"
        
        # Check context path for strong type hints (only if no content patterns matched)
        if "enable" in context_path or "support" in context_path or "disable" in context_path:
            return "boolean"
        elif "count" in context_path or "size" in context_path or "limit" in context_path or "timeout" in context_path or "retry" in context_path or "number" in context_path:
            return "number"
        elif "languages" in context_path or "items" in context_path or "list" in context_path or "tags" in context_path or "options" in context_path:
            return "list"
        elif "parameters" in context_path:
            return "parameter"
        elif "format" in context_path or "type" in context_path:
            return "string"
        
        return "string"
    
    async def _resolve_parameter(self, content: str, context_path: str) -> Any:
        """Resolve parameter ambiguity."""
        # Common parameter patterns
        if "format" in content.lower():
            return "json"
        elif "method" in content.lower():
            return "default"
        elif "type" in content.lower():
            return "auto"
        
        return await self._resolve_generic(content, context_path)
    
    async def _resolve_value(self, content: str, context_path: str) -> str:
        """Resolve value ambiguity."""
        # Extract choices if present
        choices = self._extract_choices(content)
        if choices:
            return choices[0]  # Return first choice
        
        return await self._resolve_generic(content, context_path)
    
    async def _resolve_list(self, content: str, context_path: str) -> List[Any]:
        """Resolve list ambiguity."""
        # Return a default list based on context
        content_lower = content.lower()
        if "source" in content_lower:
            return ["web", "documents", "database"]
        elif "format" in content_lower:
            return ["json", "csv", "xml"]
        elif "language" in content_lower or "languages" in content_lower:
            return ["en", "es", "fr"]
        
        return ["item1", "item2", "item3"]
    
    async def _resolve_boolean(self, content: str, context_path: str) -> bool:
        """Resolve boolean ambiguity."""
        # Look for positive/negative indicators
        positive_words = ["enable", "true", "yes", "allow", "support"]
        negative_words = ["disable", "false", "no", "deny", "block"]
        
        content_lower = content.lower()
        
        for word in positive_words:
            if word in content_lower:
                return True
        
        for word in negative_words:
            if word in content_lower:
                return False
        
        # Default based on context
        if "enable" in context_path or "support" in context_path:
            return True
        
        return False
    
    async def _resolve_number(self, content: str, context_path: str) -> Union[int, float]:
        """Resolve number ambiguity."""
        # Look for number hints in content
        if "batch" in content.lower():
            return 32
        elif "timeout" in content.lower():
            return 30
        elif "retry" in content.lower():
            return 3
        elif "size" in content.lower():
            return 100
        elif "limit" in content.lower():
            return 1000
        
        return 10  # Default number
    
    async def _resolve_string(self, content: str, context_path: str) -> str:
        """Resolve string ambiguity."""
        # Extract quoted strings if present
        quotes = self._extract_quotes(content)
        if quotes:
            return quotes[0]
        
        return await self._resolve_generic(content, context_path)
    
    async def _resolve_generic(self, content: str, context_path: str) -> str:
        """Generic resolution using model."""
        # Create a detailed resolution prompt
        prompt = f"""You are an AI pipeline orchestration expert. Resolve this ambiguous specification to a specific, concrete value.

CONTEXT:
- Path: {context_path}
- Ambiguous content: {content}

RULES:
1. Provide a specific, actionable value
2. Consider the context path to understand the parameter's purpose
3. Return only the resolved value without explanation
4. If multiple options exist, choose the most common/standard option
5. For parameters, prefer widely-supported values
6. For actions, prefer standard operations
7. For formats, prefer JSON unless context suggests otherwise

EXAMPLES:
- "Choose the best format" in data context → "json"
- "Select appropriate method" in analysis context → "statistical"
- "Determine timeout value" → "30"
- "Pick suitable model" → "gpt-3.5-turbo"

Resolved value:"""
        
        try:
            result = await self.model.generate(prompt, temperature=0.1, max_tokens=50)
            return result.strip().strip('"').strip("'")
        except Exception as e:
            # Fallback to simple heuristics
            return self._fallback_resolution(content, context_path)
    
    def _fallback_resolution(self, content: str, context_path: str) -> str:
        """Fallback resolution using simple heuristics."""
        # Common fallbacks based on content
        if "query" in content.lower():
            return "default search query"
        elif "format" in content.lower():
            return "json"
        elif "method" in content.lower():
            return "auto"
        elif "style" in content.lower():
            return "default"
        elif "type" in content.lower():
            return "standard"
        
        return "default"
    
    def _extract_choices(self, content: str) -> List[str]:
        """Extract choices from content (e.g., 'choose: A, B, or C')."""
        import re
        
        # Pattern for "A, B, or C" or "A, B, C"
        pattern = r'(?:choose|select).*?:\s*([^.]+)'
        match = re.search(pattern, content, re.IGNORECASE)
        
        if match:
            choices_text = match.group(1)
            # Split by comma and clean up
            choices = [choice.strip() for choice in choices_text.split(',')]
            # Remove 'or' from last choice
            if choices:
                last_choice = choices[-1]
                # Handle various 'or' patterns
                last_choice = last_choice.replace(' or ', '').replace('or ', '').strip()
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
    
    def clear_cache(self) -> None:
        """Clear the resolution cache."""
        self.resolution_cache.clear()
    
    def get_cache_size(self) -> int:
        """Get the size of the resolution cache."""
        return len(self.resolution_cache)
    
    def set_resolution_strategy(self, ambiguity_type: str, strategy_func) -> None:
        """Set custom resolution strategy for ambiguity type."""
        self.resolution_strategies[ambiguity_type] = strategy_func