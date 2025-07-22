"""AUTO tag resolution engine for control flow constructs."""

import re
from typing import Any, Dict, List, Optional, Union
import json

from ..compiler.ambiguity_resolver import AmbiguityResolver
from ..models.model_registry import ModelRegistry


class ControlFlowAutoResolver:
    """Handles AUTO tag resolution for control flow constructs with runtime context."""
    
    def __init__(self, model_registry: Optional[ModelRegistry] = None):
        """Initialize the control flow AUTO resolver.
        
        Args:
            model_registry: Model registry for LLM access
        """
        self.model_registry = model_registry
        
        # Try to create ambiguity resolver, but make it optional
        try:
            self.ambiguity_resolver = AmbiguityResolver(model_registry)
        except ValueError:
            # No model available, create a placeholder
            self.ambiguity_resolver = None
            
        self.auto_tag_pattern = re.compile(r"<AUTO>(.*?)</AUTO>", re.DOTALL)
        self._resolution_cache = {}
        
    async def resolve_condition(
        self, 
        condition: str, 
        context: Dict[str, Any],
        step_results: Dict[str, Any],
        cache_key: Optional[str] = None
    ) -> bool:
        """Resolve a condition that may contain AUTO tags.
        
        Args:
            condition: Condition string potentially containing AUTO tags
            context: Pipeline execution context
            step_results: Results from previous steps
            cache_key: Optional cache key for deterministic resolution
            
        Returns:
            Boolean result of condition evaluation
        """
        # Check cache first
        if cache_key and cache_key in self._resolution_cache:
            return self._resolution_cache[cache_key]
            
        # Resolve AUTO tags in condition
        resolved_condition = await self._resolve_auto_tags(
            condition, context, step_results
        )
        
        # Evaluate the resolved condition
        try:
            # Build evaluation context
            eval_context = self._build_eval_context(context, step_results)
            
            # For simple boolean strings
            if isinstance(resolved_condition, str):
                lower_cond = resolved_condition.strip().lower()
                if lower_cond in ("true", "yes", "1"):
                    result = True
                elif lower_cond in ("false", "no", "0"):
                    result = False
                else:
                    # Try to evaluate as Python expression
                    result = self._safe_eval(resolved_condition, eval_context)
            else:
                result = bool(resolved_condition)
                
            # Cache result if key provided
            if cache_key:
                self._resolution_cache[cache_key] = result
                
            return result
            
        except Exception as e:
            raise ValueError(f"Failed to evaluate condition: {e}")
    
    async def resolve_iterator(
        self,
        iterator_expr: str,
        context: Dict[str, Any],
        step_results: Dict[str, Any]
    ) -> List[Any]:
        """Resolve an iterator expression that may contain AUTO tags.
        
        Args:
            iterator_expr: Iterator expression potentially containing AUTO tags
            context: Pipeline execution context
            step_results: Results from previous steps
            
        Returns:
            List of items to iterate over
        """
        # Resolve AUTO tags
        resolved_expr = await self._resolve_auto_tags(
            iterator_expr, context, step_results
        )
        
        # Convert to list if needed
        if isinstance(resolved_expr, str):
            # Try to parse as JSON array
            try:
                result = json.loads(resolved_expr)
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass
                
            # Split comma-separated values
            if "," in resolved_expr:
                return [item.strip() for item in resolved_expr.split(",")]
                
            # Single item
            return [resolved_expr]
            
        elif isinstance(resolved_expr, list):
            return resolved_expr
        else:
            # Convert other iterables to list
            try:
                return list(resolved_expr)
            except TypeError:
                return [resolved_expr]
    
    async def resolve_count(
        self,
        count_expr: str,
        context: Dict[str, Any],
        step_results: Dict[str, Any]
    ) -> int:
        """Resolve a count expression that may contain AUTO tags.
        
        Args:
            count_expr: Count expression potentially containing AUTO tags
            context: Pipeline execution context
            step_results: Results from previous steps
            
        Returns:
            Integer count value
        """
        resolved = await self._resolve_auto_tags(count_expr, context, step_results)
        
        try:
            if isinstance(resolved, (int, float)):
                return int(resolved)
            elif isinstance(resolved, str):
                return int(resolved.strip())
            else:
                raise ValueError(f"Cannot convert {type(resolved)} to count")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to resolve count: {e}")
    
    async def resolve_target(
        self,
        target_expr: str,
        context: Dict[str, Any],
        step_results: Dict[str, Any],
        valid_targets: List[str]
    ) -> str:
        """Resolve a jump target that may contain AUTO tags.
        
        Args:
            target_expr: Target expression potentially containing AUTO tags
            context: Pipeline execution context
            step_results: Results from previous steps
            valid_targets: List of valid target step IDs
            
        Returns:
            Resolved target step ID
        """
        resolved = await self._resolve_auto_tags(target_expr, context, step_results)
        
        if not isinstance(resolved, str):
            resolved = str(resolved)
            
        resolved = resolved.strip()
        
        # Validate target
        if resolved not in valid_targets:
            raise ValueError(
                f"Invalid jump target '{resolved}'. Valid targets: {valid_targets}"
            )
            
        return resolved
    
    async def resolve_dependencies(
        self,
        deps_expr: Union[str, List[str]],
        context: Dict[str, Any],
        step_results: Dict[str, Any],
        valid_steps: List[str]
    ) -> List[str]:
        """Resolve dynamic dependencies that may contain AUTO tags.
        
        Args:
            deps_expr: Dependencies expression potentially containing AUTO tags
            context: Pipeline execution context
            step_results: Results from previous steps
            valid_steps: List of valid step IDs
            
        Returns:
            List of resolved dependency step IDs
        """
        # Handle list of dependencies
        if isinstance(deps_expr, list):
            resolved_deps = []
            for dep in deps_expr:
                if isinstance(dep, str) and self.auto_tag_pattern.search(dep):
                    resolved = await self._resolve_auto_tags(dep, context, step_results)
                    if isinstance(resolved, list):
                        resolved_deps.extend(resolved)
                    else:
                        resolved_deps.append(str(resolved))
                else:
                    resolved_deps.append(str(dep))
            return resolved_deps
            
        # Handle single dependency expression
        resolved = await self._resolve_auto_tags(deps_expr, context, step_results)
        
        if isinstance(resolved, str):
            # Parse comma-separated or JSON array
            try:
                deps = json.loads(resolved)
                if isinstance(deps, list):
                    return [str(d) for d in deps]
            except json.JSONDecodeError:
                if "," in resolved:
                    return [d.strip() for d in resolved.split(",")]
                return [resolved.strip()]
                
        elif isinstance(resolved, list):
            return [str(d) for d in resolved]
        else:
            return [str(resolved)]
    
    async def _resolve_auto_tags(
        self,
        content: str,
        context: Dict[str, Any],
        step_results: Dict[str, Any]
    ) -> Any:
        """Resolve AUTO tags in content with runtime context.
        
        Args:
            content: Content potentially containing AUTO tags
            context: Pipeline execution context
            step_results: Results from previous steps
            
        Returns:
            Content with AUTO tags resolved
        """
        if not isinstance(content, str):
            return content
            
        # Check if string contains AUTO tags
        matches = self.auto_tag_pattern.findall(content)
        if not matches:
            return content
            
        # Build enhanced context for AUTO resolution
        enhanced_context = self._build_eval_context(context, step_results)
        
        # If entire string is a single AUTO tag, resolve directly
        if len(matches) == 1 and content.strip() == f"<AUTO>{matches[0]}</AUTO>":
            prompt = matches[0].strip()
            
            # Check if we have a resolver
            if not self.ambiguity_resolver:
                # Return a reasonable default
                return self._get_default_resolution(prompt)
                
            # Add context to prompt
            context_prompt = self._build_context_prompt(prompt, enhanced_context)
            return await self.ambiguity_resolver.resolve(context_prompt, "control_flow")
            
        # Otherwise resolve each AUTO tag and substitute
        resolved_content = content
        for match in matches:
            prompt = match.strip()
            
            if not self.ambiguity_resolver:
                resolved_value = self._get_default_resolution(prompt)
            else:
                context_prompt = self._build_context_prompt(prompt, enhanced_context)
                resolved_value = await self.ambiguity_resolver.resolve(
                    context_prompt, "control_flow"
                )
            
            # Convert to string for substitution
            resolved_str = (
                str(resolved_value) 
                if not isinstance(resolved_value, str) 
                else resolved_value
            )
            resolved_content = resolved_content.replace(
                f"<AUTO>{match}</AUTO>", resolved_str
            )
            
        return resolved_content
    
    def _build_eval_context(
        self, 
        context: Dict[str, Any], 
        step_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build evaluation context combining pipeline context and step results.
        
        Args:
            context: Pipeline execution context
            step_results: Results from previous steps
            
        Returns:
            Combined evaluation context
        """
        eval_context = context.copy()
        eval_context["results"] = step_results
        eval_context["steps"] = step_results  # Alias for convenience
        
        # Add helper functions
        eval_context["len"] = len
        eval_context["str"] = str
        eval_context["int"] = int
        eval_context["float"] = float
        eval_context["bool"] = bool
        eval_context["list"] = list
        eval_context["dict"] = dict
        
        return eval_context
    
    def _build_context_prompt(
        self, 
        prompt: str, 
        context: Dict[str, Any]
    ) -> str:
        """Build enhanced prompt with context information.
        
        Args:
            prompt: Original AUTO tag prompt
            context: Evaluation context
            
        Returns:
            Enhanced prompt with context
        """
        # Extract relevant context info
        context_parts = []
        
        # Add recent step results
        if "results" in context and context["results"]:
            recent_results = list(context["results"].items())[-3:]  # Last 3 results
            if recent_results:
                context_parts.append("Recent step results:")
                for step_id, result in recent_results:
                    result_summary = str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
                    context_parts.append(f"  - {step_id}: {result_summary}")
        
        # Add pipeline info
        if "pipeline" in context:
            pipeline_info = context["pipeline"]
            if isinstance(pipeline_info, dict) and "name" in pipeline_info:
                context_parts.append(f"Pipeline: {pipeline_info['name']}")
        
        # Build final prompt
        if context_parts:
            context_str = "\n".join(context_parts)
            return f"{context_str}\n\n{prompt}"
        else:
            return prompt
    
    def _safe_eval(self, expression: str, context: Dict[str, Any]) -> Any:
        """Safely evaluate a Python expression.
        
        Args:
            expression: Python expression to evaluate
            context: Variables available during evaluation
            
        Returns:
            Result of expression evaluation
        """
        # Simple expression evaluation using compile
        try:
            # Replace template variables {{ var }} with context lookups
            import re
            
            def replace_var(match):
                var_path = match.group(1).strip()
                parts = var_path.split('.')
                
                # Navigate through context
                value = context
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        # Try direct lookup in context
                        if part in context:
                            value = context[part]
                        else:
                            return "None"
                
                # Return Python representation
                if isinstance(value, str):
                    return f'"{value}"'
                else:
                    return str(value)
            
            # Replace template variables
            resolved_expr = re.sub(r'\{\{([^}]+)\}\}', replace_var, expression)
            
            # If no replacement happened, try direct evaluation
            if resolved_expr == expression:
                # Check if it's a simple boolean string
                lower_expr = expression.strip().lower()
                if lower_expr == "true":
                    return True
                elif lower_expr == "false":
                    return False
            
            # Compile and evaluate
            code = compile(resolved_expr, '<string>', 'eval')
            return eval(code, {"__builtins__": {}}, context)
            
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expression}': {e}")
    
    def clear_cache(self):
        """Clear the resolution cache."""
        self._resolution_cache.clear()
    
    def _get_default_resolution(self, prompt: str) -> Any:
        """Get default resolution when no model is available.
        
        Args:
            prompt: The AUTO prompt
            
        Returns:
            Default value based on prompt content
        """
        prompt_lower = prompt.lower()
        
        # Boolean defaults
        if any(word in prompt_lower for word in ['should', 'do we', 'is', 'are']):
            if 'not' in prompt_lower or 'false' in prompt_lower:
                return False
            return True
            
        # Numeric defaults
        if 'how many' in prompt_lower:
            return 3
        if 'number' in prompt_lower:
            return 1
            
        # List defaults
        if 'list' in prompt_lower or 'array' in prompt_lower:
            return ["item1", "item2", "item3"]
            
        # Target selection
        if 'which' in prompt_lower and 'handler' in prompt_lower:
            if 'error' in prompt_lower:
                return 'error_handler'
            elif 'success' in prompt_lower:
                return 'success_handler'
            else:
                return 'default_handler'
                
        # Method selection
        if 'method' in prompt_lower:
            return 'default'
            
        # Default string
        return "default"