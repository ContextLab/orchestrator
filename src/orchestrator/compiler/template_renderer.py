"""Template rendering utilities for YAML pipelines."""

import re
from datetime import datetime
from typing import Any, Dict


class TemplateRenderer:
    """Handles template variable rendering in pipelines."""
    
    @staticmethod
    def render(text: str, context: Dict[str, Any]) -> str:
        """Render template variables in text."""
        if not text or not context:
            return text
        
        # Handle Jinja2-style templates
        text = TemplateRenderer._render_jinja2(text, context)
        
        # Handle simple {{variable}} templates
        text = TemplateRenderer._render_simple(text, context)
        
        return text
    
    @staticmethod
    def _render_simple(text: str, context: Dict[str, Any]) -> str:
        """Render simple {{variable}} style templates."""
        def replace_var(match) -> str:
            var_expr = match.group(1).strip()
            
            # Special variables
            if var_expr == 'execution.timestamp':
                return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Handle simple variables
            if var_expr in context:
                return str(context[var_expr])
            
            # Handle dot notation
            if '.' in var_expr:
                value = TemplateRenderer._get_nested_value(var_expr, context)
                if value is not None:
                    return str(value)
            
            # Handle filters
            if '|' in var_expr:
                value = TemplateRenderer._apply_filter(var_expr, context)
                if value is not None:
                    return str(value)
            
            return match.group(0)  # Return original if not resolved
        
        return re.sub(r'\{\{([^}]+)\}\}', replace_var, text)
    
    @staticmethod
    def _render_jinja2(text: str, context: Dict[str, Any]) -> str:
        """Render Jinja2-style templates."""
        # Handle {% for %} loops
        for_pattern = r'\{%\s*for\s+(\w+)\s+in\s+([^%]+)\s*%\}(.*?)\{%\s*endfor\s*%\}'
        
        def replace_for(match) -> str:
            var_name = match.group(1)
            collection_expr = match.group(2).strip()
            loop_body = match.group(3)
            
            # Get collection
            collection = TemplateRenderer._evaluate_expression(collection_expr, context)
            if not isinstance(collection, (list, tuple)):
                return match.group(0)
            
            # Render each iteration
            results = []
            for item in collection:
                loop_context = context.copy()
                loop_context[var_name] = item
                rendered = TemplateRenderer.render(loop_body, loop_context)
                results.append(rendered)
            
            return ''.join(results)
        
        text = re.sub(for_pattern, replace_for, text, flags=re.DOTALL)
        
        # Handle {% if %} conditions
        if_pattern = r'\{%\s*if\s+([^%]+)\s*%\}(.*?)(?:\{%\s*else\s*%\}(.*?))?\{%\s*endif\s*%\}'
        
        def replace_if(match) -> str:
            condition_expr = match.group(1).strip()
            if_body = match.group(2)
            else_body = match.group(3) or ''
            
            # Evaluate condition
            condition = TemplateRenderer._evaluate_expression(condition_expr, context)
            
            if condition:
                return TemplateRenderer.render(if_body, context)
            else:
                return TemplateRenderer.render(else_body, context)
        
        text = re.sub(if_pattern, replace_if, text, flags=re.DOTALL)
        
        return text
    
    @staticmethod
    def _get_nested_value(expr: str, context: Dict[str, Any]) -> Any:
        """Get value from nested dictionary using dot notation."""
        parts = expr.split('.')
        value = context
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        
        return value
    
    @staticmethod
    def _apply_filter(expr: str, context: Dict[str, Any]) -> Any:
        """Apply filter to a variable."""
        parts = expr.split('|', 1)
        if len(parts) != 2:
            return None
        
        var_name = parts[0].strip()
        filter_expr = parts[1].strip()
        
        # Get base value
        value = TemplateRenderer._get_nested_value(var_name, context) if '.' in var_name else context.get(var_name)
        if value is None:
            return None
        
        # Apply filters
        if filter_expr == 'lower':
            return str(value).lower()
        elif filter_expr == 'upper':
            return str(value).upper()
        elif filter_expr == "replace(' ', '_')":
            return str(value).replace(' ', '_')
        elif filter_expr.startswith('truncate('):
            match = re.match(r'truncate\((\d+)\)', filter_expr)
            if match:
                length = int(match.group(1))
                return str(value)[:length]
        elif filter_expr.startswith('default('):
            match = re.match(r'default\(["\']([^"\']*)["\']', filter_expr)
            if match and not value:
                return match.group(1)
        elif filter_expr == 'length':
            return len(value) if hasattr(value, '__len__') else 0
        elif filter_expr.startswith('join('):
            match = re.match(r'join\(["\']([^"\']*)["\']', filter_expr)
            if match and isinstance(value, (list, tuple)):
                separator = match.group(1)
                return separator.join(str(v) for v in value)
        
        return value
    
    @staticmethod
    def _evaluate_expression(expr: str, context: Dict[str, Any]) -> Any:
        """Evaluate a simple expression."""
        expr = expr.strip()
        
        # Handle string literals
        if (expr.startswith('"') and expr.endswith('"')) or (expr.startswith("'") and expr.endswith("'")):
            return expr[1:-1]
        
        # Handle numbers
        try:
            return int(expr)
        except ValueError:
            try:
                return float(expr)
            except ValueError:
                pass
        
        # Handle boolean
        if expr.lower() == 'true':
            return True
        elif expr.lower() == 'false':
            return False
        
        # Handle variables
        if '.' in expr:
            return TemplateRenderer._get_nested_value(expr, context)
        else:
            return context.get(expr)