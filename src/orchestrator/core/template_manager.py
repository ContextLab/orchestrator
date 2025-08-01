"""Template management system for deferred template rendering."""

import logging
from typing import Any, Dict, List, Union, Optional
from jinja2 import Environment, StrictUndefined, Template, TemplateSyntaxError, UndefinedError
from jinja2.filters import FILTERS
from datetime import datetime
import json
import re

logger = logging.getLogger(__name__)


class ChainableUndefined(StrictUndefined):
    """Custom undefined class that allows chaining and provides helpful error messages."""
    
    def __getattr__(self, name):
        """Allow chaining on undefined variables to prevent immediate errors."""
        logger.warning(f"Accessing undefined attribute '{name}' on undefined variable '{self._undefined_name}'")
        return ChainableUndefined(name=f"{self._undefined_name}.{name}")
    
    def __getitem__(self, key):
        """Allow indexing on undefined variables."""
        logger.warning(f"Accessing undefined index '{key}' on undefined variable '{self._undefined_name}'")
        return ChainableUndefined(name=f"{self._undefined_name}[{key}]")
    
    def __str__(self):
        """Return a placeholder string instead of raising error."""
        placeholder = f"{{{{{self._undefined_name}}}}}"
        logger.warning(f"Undefined variable '{self._undefined_name}' rendered as placeholder: {placeholder}")
        return placeholder
    
    def __len__(self):
        """Return 0 for undefined collections."""
        logger.warning(f"Length requested for undefined variable '{self._undefined_name}', returning 0")
        return 0
    
    def __iter__(self):
        """Return empty iterator for undefined collections."""
        logger.warning(f"Iteration requested for undefined variable '{self._undefined_name}', returning empty iterator")
        return iter([])


class DeferredTemplate:
    """A template that will be rendered later when context is available."""
    
    def __init__(self, template_string: str, template_manager: 'TemplateManager'):
        self.template_string = template_string
        self.template_manager = template_manager
        self._compiled_template = None
    
    def compile(self) -> Template:
        """Compile the template if not already compiled."""
        if self._compiled_template is None:
            try:
                self._compiled_template = self.template_manager.env.from_string(self.template_string)
            except TemplateSyntaxError as e:
                logger.error(f"Template syntax error in '{self.template_string}': {e}")
                # Return template that renders to original string
                self._compiled_template = self.template_manager.env.from_string("{{ original }}")
        return self._compiled_template
    
    def render(self, additional_context: Optional[Dict[str, Any]] = None) -> str:
        """Render the template with current context."""
        try:
            template = self.compile()
            context = {**self.template_manager.context, **(additional_context or {})}
            
            # Add original template string as fallback
            if 'original' not in context:
                context['original'] = self.template_string
                
            result = template.render(context)
            logger.debug(f"Successfully rendered template: '{self.template_string}' -> '{result}'")
            return result
            
        except UndefinedError as e:
            logger.warning(f"Undefined variable in template '{self.template_string}': {e}")
            return self.template_string
        except Exception as e:
            logger.error(f"Error rendering template '{self.template_string}': {e}")
            return self.template_string
    
    def __str__(self):
        """Render when converted to string."""
        return self.render()


class TemplateManager:
    """Manages template rendering with deferred evaluation and context management."""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.context: Dict[str, Any] = {}
        
        # Set up Jinja2 environment with custom filters and undefined handling
        self.env = Environment(
            undefined=ChainableUndefined,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters
        self._setup_custom_filters()
        
        # Initialize with basic context
        self._setup_base_context()
    
    def _setup_custom_filters(self):
        """Set up custom Jinja2 filters for common pipeline operations."""
        
        def slugify(text: str) -> str:
            """Convert text to URL-safe slug."""
            if not isinstance(text, str):
                text = str(text)
            text = re.sub(r'[^\w\s-]', '', text.lower())
            return re.sub(r'[-\s]+', '-', text).strip('-')
        
        def truncate_words(text: str, count: int = 50) -> str:
            """Truncate text to specified number of words."""
            if not isinstance(text, str):
                text = str(text)
            words = text.split()
            if len(words) <= count:
                return text
            return ' '.join(words[:count]) + '...'
        
        def to_json(obj: Any) -> str:
            """Convert object to JSON string."""
            try:
                return json.dumps(obj, indent=2, default=str)
            except Exception:
                return str(obj)
        
        def from_json(text: str) -> Any:
            """Parse JSON string to object."""
            if not isinstance(text, str):
                return text
            try:
                return json.loads(text)
            except Exception:
                return text
        
        def date_format(date_obj: Any, format_str: str = '%Y-%m-%d %H:%M:%S') -> str:
            """Format date object or string."""
            # Debug logging
            logger.debug(f"date_format called with date_obj={date_obj}, format_str={format_str}")
            
            if isinstance(date_obj, str):
                if date_obj == 'now':
                    date_obj = datetime.now()
                else:
                    try:
                        # Handle various ISO format variations
                        date_str = date_obj.replace('Z', '+00:00')
                        # Try parsing as ISO format
                        if 'T' in date_str:
                            date_obj = datetime.fromisoformat(date_str)
                        else:
                            # Try other common formats
                            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                                try:
                                    date_obj = datetime.strptime(date_str, fmt)
                                    break
                                except ValueError:
                                    continue
                            else:
                                return date_obj  # Return original if can't parse
                    except (ValueError, AttributeError):
                        return str(date_obj)
            elif not isinstance(date_obj, datetime):
                return str(date_obj)
            
            # Now we have a datetime object
            try:
                # Ensure format_str is a string
                if not isinstance(format_str, str):
                    logger.warning(f"Format string is not a string: {format_str} (type: {type(format_str)})")
                    format_str = '%Y-%m-%d %H:%M:%S'
                
                result = date_obj.strftime(format_str)
                logger.debug(f"date_format returning: {result}")
                return result
            except Exception as e:
                logger.error(f"Date formatting error with format '{format_str}': {e}")
                logger.error(f"Date object type: {type(date_obj)}, value: {date_obj}")
                # Try to return something sensible
                try:
                    return date_obj.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    return str(date_obj)
        
        def markdown_format(text: str) -> str:
            """Basic markdown formatting improvements."""
            if not isinstance(text, str):
                text = str(text)
            
            # Ensure proper line breaks for headers
            text = re.sub(r'\n(#{1,6}\s)', r'\n\n\1', text)
            # Ensure proper spacing around lists
            text = re.sub(r'\n(\s*[-*+]\s)', r'\n\n\1', text)
            text = re.sub(r'\n(\s*\d+\.\s)', r'\n\n\1', text)
            
            return text.strip()
        
        # Register custom filters
        self.env.filters.update({
            'slugify': slugify,
            'truncate_words': truncate_words,
            'to_json': to_json,
            'from_json': from_json,
            'date': date_format,
            'markdown_format': markdown_format,
        })
    
    def _setup_base_context(self):
        """Initialize base context with common variables."""
        self.context.update({
            'timestamp': datetime.now().isoformat(),
            'debug_mode': self.debug_mode,
        })
    
    def register_context(self, key: str, value: Any):
        """Register a value for template resolution."""
        if self.debug_mode:
            logger.info(f"Registering template context: {key} = {type(value).__name__}")
        
        # For dictionaries that already have structured data (like web search results),
        # register them directly without wrapping
        if isinstance(value, dict) and any(k in value for k in ['results', 'total_results', 'query', 'backend']):
            # This looks like a tool result with direct properties - don't wrap it
            self.context[key] = value
        # Handle special case where value is a dict with 'result' attribute
        # This maintains backward compatibility with existing templates
        elif isinstance(value, dict) and 'result' in value:
            # Create object with result attribute for template compatibility
            class ResultWrapper:
                def __init__(self, data):
                    self.__dict__.update(data)
                    
                def __getattr__(self, name):
                    return self.__dict__.get(name, ChainableUndefined(name=f"{key}.{name}"))
            
            wrapper = ResultWrapper(value)
            self.context[key] = wrapper
        else:
            # For string results, wrap them to have .result attribute
            if isinstance(value, str):
                class StringResultWrapper(str):
                    """String wrapper that also provides .result attribute."""
                    def __new__(cls, result_value):
                        # Create a new string instance
                        instance = str.__new__(cls, result_value)
                        instance.result = result_value
                        instance._value = result_value
                        return instance
                        
                    def __getattr__(self, name):
                        if name == 'result':
                            return self._value if hasattr(self, '_value') else str(self)
                        # For other attributes, return None instead of undefined
                        # This allows Jinja2 conditionals to work properly
                        return None
                
                self.context[key] = StringResultWrapper(value)
            else:
                self.context[key] = value
    
    def register_all_results(self, results: Dict[str, Any]):
        """Register all step results at once."""
        if self.debug_mode:
            logger.info(f"Registering all results: {list(results.keys())}")
        
        # Register individual results
        for key, value in results.items():
            self.register_context(key, value)
        
        # Also register as 'previous_results' for backward compatibility
        self.context['previous_results'] = results
    
    def has_templates(self, text: str) -> bool:
        """Check if text contains Jinja2 template syntax."""
        if not isinstance(text, str):
            return False
        return bool(re.search(r'\{\{.*?\}\}|\{\%.*?\%\}', text))
    
    def render(self, template_string: str, additional_context: Optional[Dict[str, Any]] = None) -> str:
        """Render a template string with current context."""
        if not isinstance(template_string, str):
            return str(template_string)
        
        if not self.has_templates(template_string):
            return template_string
        
        # Build context before try block so it's available in except blocks
        context = {**self.context, **(additional_context or {})}
        
        # Debug: Log when rendering conditions
        if "read_file.size" in template_string and "{{ read_file.size" in template_string:
            logger.warning(f"TEMPLATE MANAGER: Attempting to render condition template: {template_string[:100]}...")
            logger.warning(f"Available context keys: {list(context.keys())}")
        
        try:
            template = self.env.from_string(template_string)
            
            # Debug: Check specific variables
            if 'analyze_findings' in context and '{{ analyze_findings.result }}' in template_string:
                logger.debug(f"analyze_findings type: {type(context['analyze_findings'])}")
                logger.debug(f"analyze_findings value: {str(context['analyze_findings'])[:100]}...")
                if hasattr(context['analyze_findings'], 'result'):
                    logger.debug(f"analyze_findings.result exists: {str(context['analyze_findings'].result)[:100]}...")
                else:
                    logger.debug("analyze_findings.result DOES NOT EXIST")
                    
            result = template.render(context)
            
            if self.debug_mode:
                logger.info(f"Template rendered: '{template_string}' -> '{result}'")
            
            return result
            
        except UndefinedError as e:
            error_msg = str(e)
            # Don't show the whole template if it's large
            template_preview = template_string[:200] + "..." if len(template_string) > 200 else template_string
            logger.warning(f"Undefined variable in template '{template_preview}': {error_msg}")
            logger.warning(f"Available variables in context: {list(context.keys())}")
            
            # Try to extract the specific undefined variable from the error message
            import re
            # Look for patterns like "'variable_name' is undefined"
            match = re.search(r"'([^']+)' is undefined", error_msg)
            if match:
                undefined_var = match.group(1)
                logger.warning(f"Specifically, '{undefined_var}' is undefined")
                # Check if it's a nested access
                if '.' in undefined_var:
                    base_var = undefined_var.split('.')[0]
                    if base_var in context:
                        logger.warning(f"Base variable '{base_var}' exists with type: {type(context[base_var])}")
                        logger.warning(f"Trying to access attribute: {undefined_var}")
                else:
                    logger.warning(f"Variable '{undefined_var}' not found in context")
            
            return template_string
        except Exception as e:
            logger.error(f"Error rendering template: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            # Don't show long templates in error messages
            if len(template_string) > 200:
                logger.error(f"Error rendering template (first 200 chars): {template_string[:200]}...")
            else:
                logger.error(f"Error rendering template: '{template_string}'")
            logger.error(f"Available variables in context: {list(context.keys())}")
            
            # Check if it's an AttributeError on a specific variable
            if isinstance(e, Exception) and "has no attribute" in str(e):
                # Try to extract which variable/attribute caused the issue
                import re
                match = re.search(r"'(\w+)' object.* has no attribute '(\w+)'", str(e))
                if match:
                    obj_type, attr = match.groups()
                    logger.error(f"AttributeError: '{obj_type}' object missing attribute '{attr}'")
                    # Check if the base variable exists
                    for var_name, var_value in context.items():
                        if type(var_value).__name__ == obj_type or str(type(var_value)) == f"<class '{obj_type}'>":
                            logger.error(f"Variable '{var_name}' is a {obj_type} with keys/attrs: {list(vars(var_value).keys()) if hasattr(var_value, '__dict__') else list(var_value.keys()) if isinstance(var_value, dict) else 'N/A'}")
            
            # Import traceback to get more details
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return template_string
    
    def render_dict(self, data: Dict[str, Any], additional_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Recursively render all string values in a dictionary."""
        if not isinstance(data, dict):
            return data
        
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.render(value, additional_context)
            elif isinstance(value, dict):
                result[key] = self.render_dict(value, additional_context)
            elif isinstance(value, list):
                result[key] = self.render_list(value, additional_context)
            else:
                result[key] = value
        
        return result
    
    def render_list(self, data: List[Any], additional_context: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Recursively render all string values in a list."""
        if not isinstance(data, list):
            return data
        
        result = []
        for item in data:
            if isinstance(item, str):
                result.append(self.render(item, additional_context))
            elif isinstance(item, dict):
                result.append(self.render_dict(item, additional_context))
            elif isinstance(item, list):
                result.append(self.render_list(item, additional_context))
            else:
                result.append(item)
        
        return result
    
    def defer_render(self, template_string: str) -> DeferredTemplate:
        """Create a deferred template for lazy evaluation."""
        return DeferredTemplate(template_string, self)
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about current context."""
        return {
            'context_keys': list(self.context.keys()),
            'context_types': {k: type(v).__name__ for k, v in self.context.items()},
            'debug_mode': self.debug_mode,
        }
    
    def clear_context(self):
        """Clear all context except base context."""
        self.context.clear()
        self._setup_base_context()
    
    def deep_render(self, data: Any, additional_context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Recursively render all template strings in any data structure.
        
        This is the universal template rendering function that handles:
        - Strings with template syntax
        - Dictionaries (renders all values)
        - Lists (renders all items)
        - Nested structures of any depth
        - Preserves non-string types
        
        Args:
            data: Any data structure that may contain template strings
            additional_context: Additional context variables for rendering
            
        Returns:
            The same data structure with all templates rendered
        """
        if isinstance(data, str):
            # Only render if it contains template syntax
            if self.has_templates(data):
                try:
                    result = self.render(data, additional_context)
                    return result
                except Exception as e:
                    # More detailed error logging
                    import traceback
                    logger.error(f"Error rendering template: {e}")
                    logger.error(f"Template preview: {data[:200]}...")
                    logger.debug(f"Full traceback: {traceback.format_exc()}")
                    # Check if it's an undefined variable error
                    if "is undefined" in str(e):
                        logger.warning(f"Template contains undefined variables, returning original")
                    # Return original string if rendering fails
                    return data
            return data
        elif isinstance(data, dict):
            # Recursively render all dictionary values
            return {key: self.deep_render(value, additional_context) 
                    for key, value in data.items()}
        elif isinstance(data, list):
            # Recursively render all list items
            return [self.deep_render(item, additional_context) 
                    for item in data]
        elif isinstance(data, tuple):
            # Handle tuples by converting to list, rendering, then back to tuple
            return tuple(self.deep_render(list(data), additional_context))
        else:
            # Return all other types as-is (int, float, bool, None, objects, etc.)
            return data