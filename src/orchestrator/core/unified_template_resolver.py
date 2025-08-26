"""Unified template resolution system for the orchestrator.

This module provides a centralized template resolution layer that:
1. Collects context from all sources (pipeline, loop, task, tool)
2. Resolves templates BEFORE passing to tools
3. Properly exposes structured data to template engine
4. Works consistently across all components

Issue #223: Template resolution system needs comprehensive fixes
"""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from contextlib import contextmanager

from .template_manager import TemplateManager
from .context_manager import ContextManager
from .loop_context import LoopContextVariables, GlobalLoopContextManager

logger = logging.getLogger(__name__)


@dataclass
class TemplateResolutionContext:
    """Comprehensive context for template resolution."""
    
    # Core execution context
    pipeline_id: Optional[str] = None
    task_id: Optional[str] = None
    tool_name: Optional[str] = None
    
    # Pipeline-level context
    pipeline_inputs: Dict[str, Any] = None
    pipeline_parameters: Dict[str, Any] = None
    
    # Execution results
    step_results: Dict[str, Any] = None
    previous_results: Dict[str, Any] = None
    
    # Loop context
    loop_variables: Dict[str, Any] = None
    active_loops: Dict[str, LoopContextVariables] = None
    
    # Tool context
    tool_parameters: Dict[str, Any] = None
    
    # Additional context
    additional_context: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize empty dicts for None values."""
        if self.pipeline_inputs is None:
            self.pipeline_inputs = {}
        if self.pipeline_parameters is None:
            self.pipeline_parameters = {}
        if self.step_results is None:
            self.step_results = {}
        if self.previous_results is None:
            self.previous_results = {}
        if self.loop_variables is None:
            self.loop_variables = {}
        if self.active_loops is None:
            self.active_loops = {}
        if self.tool_parameters is None:
            self.tool_parameters = {}
        if self.additional_context is None:
            self.additional_context = {}
    
    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for template rendering."""
        context = {}
        
        # Add pipeline inputs directly
        context.update(self.pipeline_inputs)
        
        # Add pipeline parameters
        if self.pipeline_parameters:
            context.update(self.pipeline_parameters)
            context["parameters"] = self.pipeline_parameters
        
        # Add step results
        if self.step_results:
            context.update(self.step_results)
            context["step_results"] = self.step_results
        
        # Add previous results for backward compatibility
        if self.previous_results:
            context["previous_results"] = self.previous_results
        
        # Add loop variables
        if self.loop_variables:
            context.update(self.loop_variables)
            # Also add loop variables without $ prefix for Jinja2 compatibility
            for key, value in self.loop_variables.items():
                if key.startswith('$') and '.' not in key:  # Only for top-level $ variables, not nested ones
                    clean_key = key[1:]  # Remove $ prefix
                    context[clean_key] = value
        
        # Add tool parameters
        if self.tool_parameters:
            context.update(self.tool_parameters)
        
        # Add execution metadata
        if self.pipeline_id:
            context["pipeline_id"] = self.pipeline_id
        if self.task_id:
            context["task_id"] = self.task_id
        if self.tool_name:
            context["tool_name"] = self.tool_name
        
        # Add additional context last (highest priority)
        if self.additional_context:
            context.update(self.additional_context)
        
        return context


class UnifiedTemplateResolver:
    """Unified template resolution system.
    
    This class coordinates template resolution across all orchestrator components,
    ensuring consistent behavior and proper context collection.
    """
    
    def __init__(
        self,
        template_manager: Optional[TemplateManager] = None,
        context_manager: Optional[ContextManager] = None,
        loop_context_manager: Optional[GlobalLoopContextManager] = None,
        debug_mode: bool = False
    ):
        """Initialize the unified template resolver.
        
        Args:
            template_manager: Existing template manager or None to create new
            context_manager: Existing context manager or None to create new
            loop_context_manager: Existing loop context manager or None to create new
            debug_mode: Enable debug logging
        """
        self.debug_mode = debug_mode
        
        # Initialize or use existing managers
        self.template_manager = template_manager or TemplateManager(debug_mode=debug_mode)
        self.context_manager = context_manager or ContextManager()
        self.loop_context_manager = loop_context_manager or GlobalLoopContextManager()
        
        # Ensure template manager is initialized in context manager
        self.context_manager.initialize_template_manager(self.template_manager)
        
        # Ensure template manager uses the same loop context manager
        self.template_manager.loop_context_manager = self.loop_context_manager
        
        # Current resolution context
        self._current_context: Optional[TemplateResolutionContext] = None
        
        logger.info("UnifiedTemplateResolver initialized")
    
    def collect_context(
        self,
        pipeline_id: Optional[str] = None,
        task_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        pipeline_inputs: Optional[Dict[str, Any]] = None,
        pipeline_parameters: Optional[Dict[str, Any]] = None,
        step_results: Optional[Dict[str, Any]] = None,
        tool_parameters: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> TemplateResolutionContext:
        """Collect comprehensive context for template resolution.
        
        Args:
            pipeline_id: Current pipeline ID
            task_id: Current task ID
            tool_name: Current tool name
            pipeline_inputs: Pipeline input parameters
            pipeline_parameters: Pipeline parameters
            step_results: Results from executed steps
            tool_parameters: Tool-specific parameters
            additional_context: Any additional context variables
            
        Returns:
            Comprehensive template resolution context
        """
        # Collect loop variables from loop context manager
        loop_variables = self.loop_context_manager.get_accessible_loop_variables()
        active_loops = dict(self.loop_context_manager.active_loops)
        
        # Build comprehensive context
        context = TemplateResolutionContext(
            pipeline_id=pipeline_id,
            task_id=task_id,
            tool_name=tool_name,
            pipeline_inputs=pipeline_inputs or {},
            pipeline_parameters=pipeline_parameters or {},
            step_results=step_results or {},
            previous_results=step_results or {},  # For backward compatibility
            loop_variables=loop_variables,
            active_loops=active_loops,
            tool_parameters=tool_parameters or {},
            additional_context=additional_context or {}
        )
        
        if self.debug_mode:
            logger.debug(f"Collected context for pipeline={pipeline_id}, task={task_id}, tool={tool_name}")
            logger.debug(f"Context keys: {list(context.to_flat_dict().keys())}")
        
        return context
    
    def register_context(self, context: TemplateResolutionContext) -> None:
        """Register context with the template manager.
        
        Args:
            context: Template resolution context to register
        """
        self._current_context = context
        
        # Clear existing template manager context
        self.template_manager.clear_context()
        
        # Register all context variables
        flat_context = context.to_flat_dict()
        for key, value in flat_context.items():
            if not key.startswith("_"):  # Skip internal variables
                self.template_manager.register_context(key, value)
        
        # Register loop contexts with template manager
        for loop_context in context.active_loops.values():
            self.template_manager.register_loop_context(loop_context)
        
        if self.debug_mode:
            logger.debug(f"Registered {len(flat_context)} context variables with template manager")
    
    def resolve_templates(
        self,
        data: Any,
        context: Optional[TemplateResolutionContext] = None
    ) -> Any:
        """Resolve all templates in data using comprehensive context.
        
        Args:
            data: Data that may contain templates (str, dict, list, etc.)
            context: Template resolution context (uses current if None)
            
        Returns:
            Data with all templates resolved
        """
        # Use provided context or current context
        if context:
            self.register_context(context)
        elif self._current_context is None:
            logger.warning("No template resolution context available")
            return data
        
        # Pre-process templates to convert $variable syntax to valid Jinja2
        preprocessed_data = self._preprocess_dollar_variables(data)
        
        # Use template manager's deep render for comprehensive resolution
        try:
            resolved = self.template_manager.deep_render(preprocessed_data)
            
            if self.debug_mode:
                if isinstance(data, str) and data != resolved:
                    logger.debug(f"Template resolved: '{data[:100]}...' -> '{resolved[:100]}...'")
                elif isinstance(data, (dict, list)):
                    logger.debug(f"Templates resolved in {type(data).__name__} structure")
            
            return resolved
        
        except Exception as e:
            logger.error(f"Template resolution failed: {e}")
            logger.error(f"Data type: {type(data)}, Context: {self._current_context is not None}")
            # Return original data if resolution fails
            return data
    
    def resolve_before_tool_execution(
        self,
        tool_name: str,
        tool_parameters: Dict[str, Any],
        context: TemplateResolutionContext
    ) -> Dict[str, Any]:
        """Resolve templates in tool parameters before tool execution.
        
        This is the key integration point for ensuring templates are resolved
        BEFORE being passed to tools.
        
        Args:
            tool_name: Name of the tool being executed
            tool_parameters: Tool parameters that may contain templates
            context: Template resolution context
            
        Returns:
            Tool parameters with all templates resolved
        """
        # Update context with tool information
        context.tool_name = tool_name
        context.tool_parameters = tool_parameters
        
        # Register context and resolve templates
        self.register_context(context)
        resolved_parameters = self.resolve_templates(tool_parameters)
        
        if self.debug_mode:
            logger.info(f"Resolved templates for tool '{tool_name}'")
            for key, (original, resolved) in self._compare_dicts(tool_parameters, resolved_parameters).items():
                if original != resolved:
                    logger.debug(f"  {key}: '{original}' -> '{resolved}'")
        
        return resolved_parameters
    
    def validate_templates(self, template: str) -> List[str]:
        """Validate template syntax and check for undefined variables.
        
        Args:
            template: Template string to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        if self.context_manager:
            return self.context_manager.validate_template(template)
        else:
            # Fallback validation using template manager
            try:
                self.template_manager.render(template)
                return []
            except Exception as e:
                return [str(e)]
    
    @contextmanager
    def resolution_context(
        self,
        pipeline_id: Optional[str] = None,
        task_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        pipeline_inputs: Optional[Dict[str, Any]] = None,
        pipeline_parameters: Optional[Dict[str, Any]] = None,
        step_results: Optional[Dict[str, Any]] = None,
        tool_parameters: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ):
        """Context manager for template resolution scope.
        
        Args:
            pipeline_id: Current pipeline ID
            task_id: Current task ID  
            tool_name: Current tool name
            pipeline_inputs: Pipeline input parameters
            pipeline_parameters: Pipeline parameters
            step_results: Results from executed steps
            tool_parameters: Tool-specific parameters
            additional_context: Any additional context variables
        """
        # Collect and register context
        context = self.collect_context(
            pipeline_id=pipeline_id,
            task_id=task_id,
            tool_name=tool_name,
            pipeline_inputs=pipeline_inputs,
            pipeline_parameters=pipeline_parameters,
            step_results=step_results,
            tool_parameters=tool_parameters,
            additional_context=additional_context
        )
        
        # Store previous context
        previous_context = self._current_context
        
        try:
            # Set current context
            self.register_context(context)
            yield context
        finally:
            # Restore previous context
            self._current_context = previous_context
            if previous_context:
                self.register_context(previous_context)
            else:
                self.template_manager.clear_context()
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the current template resolution state.
        
        Returns:
            Debug information dictionary
        """
        info = {
            "has_current_context": self._current_context is not None,
            "template_manager_debug": self.template_manager.get_debug_info(),
            "loop_context_manager": {
                "active_loops": list(self.loop_context_manager.active_loops.keys()),
                "historical_loops": list(self.loop_context_manager.loop_history.keys())
            }
        }
        
        if self._current_context:
            flat_context = self._current_context.to_flat_dict()
            info["current_context"] = {
                "pipeline_id": self._current_context.pipeline_id,
                "task_id": self._current_context.task_id,
                "tool_name": self._current_context.tool_name,
                "context_keys": list(flat_context.keys()),
                "context_size": len(flat_context)
            }
        
        return info
    
    def _preprocess_dollar_variables(self, data: Any) -> Any:
        """Preprocess templates to convert $variable syntax to valid Jinja2.
        
        Converts {{ $variable }} to {{ variable }} since Jinja2 doesn't support
        variables starting with $.
        
        Args:
            data: Data that may contain $variable templates
            
        Returns:
            Data with $variable syntax converted to valid Jinja2
        """
        import re
        
        if isinstance(data, str):
            # Replace {{ $variable }} with {{ variable }}
            # This regex captures $variable patterns inside {{ }} blocks
            pattern = r'\{\{\s*\$([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}'
            result = re.sub(pattern, r'{{ \1 }}', data)
            
            # Also handle $variable in {% %} blocks (for loops, conditions, etc.)
            pattern_control = r'\{%\s*([^%]*)\$([a-zA-Z_][a-zA-Z0-9_]*)([^%]*)\s*%\}'
            result = re.sub(pattern_control, r'{% \1\2\3 %}', result)
            
            if self.debug_mode and result != data:
                logger.debug(f"Preprocessed $variables: '{data[:100]}...' -> '{result[:100]}...'")
            
            return result
        elif isinstance(data, dict):
            return {key: self._preprocess_dollar_variables(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._preprocess_dollar_variables(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._preprocess_dollar_variables(item) for item in data)
        else:
            return data
    
    def _compare_dicts(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, tuple]:
        """Compare two dictionaries and return differences.
        
        Args:
            dict1: Original dictionary
            dict2: Modified dictionary
            
        Returns:
            Dictionary mapping keys to (original_value, new_value) tuples
        """
        differences = {}
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        for key in all_keys:
            val1 = dict1.get(key, "<missing>")
            val2 = dict2.get(key, "<missing>")
            if val1 != val2:
                differences[key] = (val1, val2)
        
        return differences