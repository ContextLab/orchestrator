"""Mock ambiguity resolver for testing without models."""

from typing import Any, Dict, Optional
from .ambiguity_resolver import AmbiguityResolver


class MockAmbiguityResolver:
    """Mock ambiguity resolver that provides default resolutions."""
    
    def __init__(self, model=None, model_registry=None):
        """Initialize mock resolver."""
        self.model = model
        self.model_registry = model_registry
        self.resolution_cache = {}
        
    async def resolve(self, content: str, context_path: str) -> Any:
        """Resolve ambiguous content with default values.
        
        Args:
            content: Ambiguous content to resolve
            context_path: Context path
            
        Returns:
            Default resolved value
        """
        content_lower = content.lower()
        
        # Boolean resolutions
        if any(word in content_lower for word in ['should', 'do we', 'is', 'are', 'true', 'false']):
            if 'not' in content_lower or 'false' in content_lower:
                return False
            return True
            
        # Numeric resolutions
        if 'how many' in content_lower:
            return 3
        if 'number' in content_lower:
            return 1
            
        # List resolutions  
        if 'list' in content_lower or 'array' in content_lower:
            return ["item1", "item2", "item3"]
            
        # Method/strategy selection
        if 'method' in content_lower or 'strategy' in content_lower:
            if 'advanced' in content_lower:
                return 'advanced'
            elif 'simple' in content_lower:
                return 'simple'
            else:
                return 'default'
                
        # Format selection
        if 'format' in content_lower:
            return 'json'
            
        # Default string
        return "default_resolution"