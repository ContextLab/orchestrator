"""Unified context management for template rendering and variable resolution."""

import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

from .task import Task
from .template_manager import TemplateManager
from .loop_context import LoopContextVariables, GlobalLoopContextManager

logger = logging.getLogger(__name__)


class ContextManager:
    """Unified context management for all pipeline executions.
    
    Provides a hierarchical context system that ensures variables and templates
    are properly resolved across all execution layers (global, pipeline, task, 
    loop, tool).
    """
    
    def __init__(self):
        """Initialize the context manager."""
        self.context_stack: List[Dict[str, Any]] = []
        self.global_context: Dict[str, Any] = {}
        self.template_manager: Optional[TemplateManager] = None
        
    def initialize_template_manager(self, template_manager: Optional[TemplateManager] = None) -> None:
        """Initialize or set the template manager.
        
        Args:
            template_manager: Existing template manager to use, or None to create new
        """
        if template_manager is not None:
            self.template_manager = template_manager
        else:
            self.template_manager = TemplateManager()
            
    def push_context(self, context: Dict[str, Any], context_type: str = "task") -> None:
        """Push a new context onto the stack.
        
        Args:
            context: The context dictionary to push
            context_type: Type of context (global, pipeline, task, loop, tool)
        """
        # Add metadata about context type
        context["_context_type"] = context_type
        context["_context_id"] = f"{context_type}_{len(self.context_stack)}"
        
        logger.debug(f"Pushing {context_type} context: {list(context.keys())}")
        self.context_stack.append(context)
        
        # Update template manager with new context
        if self.template_manager:
            merged = self.get_merged_context()
            for key, value in merged.items():
                if not key.startswith("_"):
                    self.template_manager.register_context(key, value)
                    
    def pop_context(self) -> Dict[str, Any]:
        """Pop the current context from the stack.
        
        Returns:
            The popped context dictionary
        """
        if not self.context_stack:
            raise RuntimeError("Cannot pop from empty context stack")
            
        context = self.context_stack.pop()
        logger.debug(f"Popping {context.get('_context_type', 'unknown')} context")
        
        # Update template manager with remaining context
        if self.template_manager and self.context_stack:
            merged = self.get_merged_context()
            # Clear and re-register all variables
            self.template_manager.context.clear()
            for key, value in merged.items():
                if not key.startswith("_"):
                    self.template_manager.register_context(key, value)
                    
        return context
        
    def get_merged_context(self) -> Dict[str, Any]:
        """Get the merged context from all levels.
        
        Returns a dictionary with all contexts merged, with more specific
        contexts overriding more general ones.
        """
        merged = self.global_context.copy()
        
        # Merge contexts in order (earlier contexts are overridden by later ones)
        for context in self.context_stack:
            for key, value in context.items():
                if not key.startswith("_"):  # Skip metadata
                    merged[key] = value
                    
        return merged
        
    def register_variable(self, key: str, value: Any, level: str = "current") -> None:
        """Register a variable at a specific context level.
        
        Args:
            key: Variable name
            value: Variable value
            level: Context level (current, task, pipeline, global)
        """
        if level == "global":
            self.global_context[key] = value
        elif level == "current" and self.context_stack:
            self.context_stack[-1][key] = value
        elif level == "pipeline":
            # Find the pipeline context (usually the first non-global context)
            for context in self.context_stack:
                if context.get("_context_type") == "pipeline":
                    context[key] = value
                    break
        elif level == "task":
            # Find the most recent task context
            for context in reversed(self.context_stack):
                if context.get("_context_type") == "task":
                    context[key] = value
                    break
                    
        # Update template manager
        if self.template_manager:
            self.template_manager.register_context(key, value)
            
        logger.debug(f"Registered variable '{key}' at level '{level}'")
    
    def register_loop_context(self, loop_context: LoopContextVariables):
        """Register a loop context for template access.
        
        Args:
            loop_context: The loop context to register
        """
        if self.template_manager:
            self.template_manager.register_loop_context(loop_context)
        logger.debug(f"Registered loop context: {loop_context.loop_name}")
    
    def unregister_loop_context(self, loop_name: str):
        """Unregister a loop context.
        
        Args:
            loop_name: Name of the loop context to remove
        """
        if self.template_manager:
            self.template_manager.unregister_loop_context(loop_name)
        logger.debug(f"Unregistered loop context: {loop_name}")
        
    def render_template(self, template: Union[str, Dict, List]) -> Union[str, Dict, List]:
        """Render a template string using the current context.
        
        Args:
            template: Template string, dict, or list with Jinja2 syntax
            
        Returns:
            Rendered content with all placeholders resolved
        """
        if not self.template_manager:
            logger.warning("No template manager available for rendering")
            return template
            
        # Ensure all current context is registered
        merged = self.get_merged_context()
        for key, value in merged.items():
            if not key.startswith("_"):
                self.template_manager.register_context(key, value)
                
        # Use deep_render to handle nested structures
        return self.template_manager.deep_render(template)
        
    def validate_template(self, template: str) -> List[str]:
        """Validate template syntax and check for undefined variables.
        
        Args:
            template: Template string to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not self.template_manager:
            errors.append("No template manager available for validation")
            return errors
            
        try:
            # Try to render with current context
            rendered = self.template_manager.render(template)
            
            # Check for remaining placeholders
            import re
            remaining = re.findall(r'{{\s*(\w+)', rendered)
            if remaining:
                errors.append(f"Undefined variables: {', '.join(set(remaining))}")
                
        except Exception as e:
            errors.append(f"Template syntax error: {str(e)}")
            
        return errors
        
    @contextmanager
    def task_context(self, task: Task, pipeline_context: Dict[str, Any]):
        """Context manager for task execution.
        
        Args:
            task: The task being executed
            pipeline_context: Pipeline-level context to inherit
        """
        # Create task context
        task_context = {
            "task_id": task.id,
            "task_name": task.name,
            "task": task,
            **pipeline_context  # Inherit pipeline context
        }
        
        # Add task metadata if available
        if task.metadata:
            task_context.update(task.metadata)
            
        # Add pipeline inputs if preserved in metadata
        if "pipeline_inputs" in task.metadata:
            task_context["inputs"] = task.metadata["pipeline_inputs"]
            
        self.push_context(task_context, "task")
        try:
            yield task_context
        finally:
            self.pop_context()
            
    @contextmanager
    def loop_context(self, loop_id: str, item: Any, index: int):
        """Context manager for loop iterations.
        
        Args:
            loop_id: Identifier for the loop
            item: Current item in the loop
            index: Current index in the loop
        """
        loop_context = {
            "loop_id": loop_id,
            "$item": item,
            "$index": index,
            "item": item,  # Alternative syntax
            "index": index
        }
        
        self.push_context(loop_context, "loop")
        try:
            yield loop_context
        finally:
            self.pop_context()
            
    @contextmanager
    def tool_context(self, tool_name: str, parameters: Dict[str, Any]):
        """Context manager for tool execution.
        
        Args:
            tool_name: Name of the tool being executed
            parameters: Tool parameters
        """
        tool_context = {
            "tool_name": tool_name,
            "tool_parameters": parameters,
            **parameters  # Make parameters directly accessible
        }
        
        self.push_context(tool_context, "tool")
        try:
            yield tool_context
        finally:
            self.pop_context()
            
    @contextmanager
    def pipeline_context(self, pipeline_id: str, inputs: Dict[str, Any], 
                        parameters: Optional[Dict[str, Any]] = None):
        """Context manager for pipeline execution.
        
        Args:
            pipeline_id: Pipeline identifier
            inputs: Pipeline inputs
            parameters: Pipeline parameters
        """
        pipeline_context = {
            "pipeline_id": pipeline_id,
            "inputs": inputs,
            "parameters": parameters or {},
            **inputs  # Make inputs directly accessible
        }
        
        # Merge parameters into context
        if parameters:
            pipeline_context.update(parameters)
            
        self.push_context(pipeline_context, "pipeline")
        try:
            yield pipeline_context
        finally:
            self.pop_context()
            
    def debug_context(self) -> str:
        """Get a debug representation of the current context stack.
        
        Returns:
            String representation of context hierarchy
        """
        lines = ["Context Stack:"]
        lines.append(f"  Global: {list(self.global_context.keys())}")
        
        for i, context in enumerate(self.context_stack):
            ctx_type = context.get("_context_type", "unknown")
            ctx_keys = [k for k in context.keys() if not k.startswith("_")]
            lines.append(f"  [{i}] {ctx_type}: {ctx_keys}")
            
        lines.append(f"  Merged keys: {list(self.get_merged_context().keys())}")
        return "\n".join(lines)