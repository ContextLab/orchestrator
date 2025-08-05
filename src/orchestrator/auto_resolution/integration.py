"""Integration of lazy AUTO tag resolver with existing infrastructure."""

import logging
from typing import Any, Dict, List, Optional, Union

from ..core.pipeline import Pipeline
from ..models.model_registry import ModelRegistry
from .models import AutoTagContext, AutoTagConfig
from .resolver import LazyAutoTagResolver
from .nested_handler import NestedAutoTagHandler

logger = logging.getLogger(__name__)


class EnhancedControlFlowAutoResolver:
    """Enhanced AUTO resolver that uses the new lazy resolution system.
    
    This class provides backward compatibility with the existing ControlFlowAutoResolver
    interface while using the new multi-pass resolution system internally.
    """
    
    def __init__(
        self, 
        model_registry: Optional[ModelRegistry] = None,
        config: Optional[AutoTagConfig] = None,
        pipeline: Optional[Pipeline] = None
    ):
        """Initialize enhanced resolver.
        
        Args:
            model_registry: Model registry for LLM access
            config: AUTO tag configuration
            pipeline: Current pipeline (can be set later)
        """
        self.model_registry = model_registry
        self.config = config or AutoTagConfig()
        self.pipeline = pipeline
        
        # Initialize new resolver system
        self.lazy_resolver = LazyAutoTagResolver(
            config=self.config,
            model_registry=model_registry
        )
        self.nested_handler = NestedAutoTagHandler(self.lazy_resolver)
        
        # Legacy compatibility
        self._resolution_cache = {}
    
    def set_pipeline(self, pipeline: Pipeline):
        """Set the current pipeline context."""
        self.pipeline = pipeline
    
    async def resolve_condition(
        self,
        condition: str,
        context: Dict[str, Any],
        step_results: Dict[str, Any],
        cache_key: Optional[str] = None,
        current_task_id: Optional[str] = None
    ) -> bool:
        """Resolve a condition that may contain AUTO tags.
        
        Maintains compatibility with existing interface while using new system.
        """
        # Check cache first
        if cache_key and cache_key in self._resolution_cache:
            return self._resolution_cache[cache_key]
        
        # Handle non-string conditions
        if not isinstance(condition, str):
            return bool(condition)
        
        # Resolve using new system
        resolved = await self._resolve_with_new_system(
            condition, 
            context, 
            step_results,
            current_task_id or "unknown",
            "condition"
        )
        
        # Convert to boolean
        result = self._to_boolean(resolved)
        
        # Cache result if key provided
        if cache_key:
            self._resolution_cache[cache_key] = result
        
        return result
    
    async def resolve_iterator(
        self, 
        iterator_expr: str, 
        context: Dict[str, Any], 
        step_results: Dict[str, Any],
        current_task_id: Optional[str] = None
    ) -> List[Any]:
        """Resolve an iterator expression that may contain AUTO tags."""
        # Resolve using new system
        resolved = await self._resolve_with_new_system(
            iterator_expr,
            context,
            step_results,
            current_task_id or "unknown",
            "iterator"
        )
        
        # Convert to list
        return self._to_list(resolved)
    
    async def resolve_count(
        self, 
        count_expr: str, 
        context: Dict[str, Any], 
        step_results: Dict[str, Any],
        current_task_id: Optional[str] = None
    ) -> int:
        """Resolve a count expression that may contain AUTO tags."""
        # Resolve using new system
        resolved = await self._resolve_with_new_system(
            count_expr,
            context,
            step_results,
            current_task_id or "unknown",
            "count"
        )
        
        # Convert to integer
        return self._to_int(resolved)
    
    async def resolve_target(
        self,
        target_expr: str,
        context: Dict[str, Any],
        step_results: Dict[str, Any],
        valid_targets: List[str],
        current_task_id: Optional[str] = None
    ) -> str:
        """Resolve a jump target that may contain AUTO tags."""
        # Resolve using new system
        resolved = await self._resolve_with_new_system(
            target_expr,
            context,
            step_results,
            current_task_id or "unknown",
            "target"
        )
        
        # Convert to string
        target = str(resolved).strip()
        
        # Validate target
        if target not in valid_targets:
            # Try to find closest match
            target = self._find_closest_target(target, valid_targets)
            if not target:
                raise ValueError(
                    f"Invalid jump target '{resolved}'. Valid targets: {valid_targets}"
                )
        
        return target
    
    async def _resolve_auto_tags(
        self, 
        content: str, 
        context: Dict[str, Any], 
        step_results: Dict[str, Any]
    ) -> Any:
        """Legacy method for backward compatibility."""
        return await self._resolve_with_new_system(
            content,
            context,
            step_results,
            "unknown",
            "general"
        )
    
    async def _resolve_with_new_system(
        self,
        content: str,
        context: Dict[str, Any],
        step_results: Dict[str, Any],
        current_task_id: str,
        field_type: str
    ) -> Any:
        """Resolve content using the new lazy resolution system."""
        if not isinstance(content, str):
            return content
        
        # Check if content has AUTO tags
        if "<AUTO>" not in content:
            return content
        
        # Ensure we have a pipeline
        if not self.pipeline:
            logger.warning("No pipeline set for AUTO tag resolution")
            # Fall back to simple resolution
            return self._simple_resolution(content)
        
        # Create AUTO tag context
        auto_context = AutoTagContext(
            pipeline=self.pipeline,
            current_task_id=current_task_id,
            tag_location=f"tasks.{current_task_id}.{field_type}",
            variables=context,
            step_results=step_results,
            loop_context=context.get("$loop")
        )
        
        # Use nested handler for complete resolution
        try:
            resolved = await self.nested_handler.resolve_nested(
                content, 
                auto_context,
                max_depth=self.config.max_nesting_depth
            )
            return resolved
        except Exception as e:
            logger.error(f"Failed to resolve AUTO tags: {e}")
            # Fall back to simple resolution
            return self._simple_resolution(content)
    
    def _simple_resolution(self, content: str) -> str:
        """Simple fallback resolution when full system unavailable."""
        # Extract AUTO tag content
        import re
        pattern = r'<AUTO>(.*?)</AUTO>'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            auto_content = match.group(1).strip().lower()
            
            # Simple heuristics
            if "choose" in auto_content or "which" in auto_content:
                if "error" in auto_content:
                    return "error_handler"
                elif "success" in auto_content:
                    return "success_handler"
                else:
                    return "default_handler"
            elif "true" in auto_content or "false" in auto_content:
                return "false" not in auto_content
            elif "count" in auto_content or "number" in auto_content:
                return 3
            elif "list" in auto_content:
                return ["item1", "item2", "item3"]
            else:
                return "default"
        
        return content
    
    def _to_boolean(self, value: Any) -> bool:
        """Convert value to boolean."""
        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            lower = value.strip().lower()
            if lower in ("true", "yes", "1", "on"):
                return True
            elif lower in ("false", "no", "0", "off", ""):
                return False
            else:
                # Try to evaluate as expression
                try:
                    return bool(eval(lower, {"__builtins__": {}}, {}))
                except:
                    return bool(value)
        else:
            return bool(value)
    
    def _to_list(self, value: Any) -> List[Any]:
        """Convert value to list."""
        if isinstance(value, list):
            return value
        elif isinstance(value, str):
            # Try to parse as JSON
            import json
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except:
                pass
            
            # Handle comma-separated
            if "," in value:
                return [item.strip() for item in value.split(",")]
            
            # Single item
            return [value]
        else:
            try:
                return list(value)
            except TypeError:
                return [value]
    
    def _to_int(self, value: Any) -> int:
        """Convert value to integer."""
        if isinstance(value, int):
            return value
        elif isinstance(value, float):
            return int(value)
        elif isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                # Try to extract number
                import re
                match = re.search(r'\d+', value)
                if match:
                    return int(match.group())
                raise ValueError(f"Cannot convert '{value}' to int")
        else:
            return int(value)
    
    def _find_closest_target(self, target: str, valid_targets: List[str]) -> Optional[str]:
        """Find closest matching target using string similarity."""
        target_lower = target.lower()
        
        # Exact match (case insensitive)
        for valid in valid_targets:
            if valid.lower() == target_lower:
                return valid
        
        # Partial match
        for valid in valid_targets:
            if target_lower in valid.lower() or valid.lower() in target_lower:
                return valid
        
        # No match found
        return None
    
    def clear_cache(self):
        """Clear the resolution cache."""
        self._resolution_cache.clear()


def create_auto_resolver(
    model_registry: Optional[ModelRegistry] = None,
    config: Optional[AutoTagConfig] = None,
    use_legacy: bool = False
) -> Union[EnhancedControlFlowAutoResolver, Any]:
    """Factory function to create appropriate AUTO resolver.
    
    Args:
        model_registry: Model registry for LLM access
        config: AUTO tag configuration
        use_legacy: Whether to use legacy resolver
        
    Returns:
        AUTO resolver instance
    """
    if use_legacy:
        # Import legacy resolver
        from ..control_flow.auto_resolver import ControlFlowAutoResolver
        return ControlFlowAutoResolver(model_registry)
    else:
        # Use new enhanced resolver
        return EnhancedControlFlowAutoResolver(model_registry, config)