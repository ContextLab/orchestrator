"""
Variable Management System for Pipeline Execution.

This module provides comprehensive variable management capabilities for pipeline
execution, including data flow between steps, context isolation, and variable
resolution with dependency tracking.
"""

from __future__ import annotations

import logging
import copy
from typing import Any, Dict, List, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading
import re
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class VariableScope(Enum):
    """Defines the scope of a variable."""
    GLOBAL = "global"           # Available across entire pipeline
    STEP = "step"              # Available only within specific step
    LOOP = "loop"              # Available only within loop iteration
    TEMPORARY = "temporary"     # Available only for specific operation


class VariableType(Enum):
    """Defines the type classification of a variable."""
    INPUT = "input"            # Input parameter to pipeline/step
    OUTPUT = "output"          # Output result from step
    INTERMEDIATE = "intermediate"  # Intermediate computation result
    CONFIGURATION = "configuration"  # Configuration parameter
    SYSTEM = "system"          # System-generated variable


@dataclass
class VariableMetadata:
    """Metadata for a variable."""
    name: str
    scope: VariableScope
    var_type: VariableType
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    source_step: Optional[str] = None
    description: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    version: int = 1
    
    def update_timestamp(self):
        """Update the last modified timestamp."""
        self.updated_at = datetime.now()
        self.version += 1


@dataclass
class Variable:
    """Represents a variable with metadata and value."""
    name: str
    value: Any
    metadata: VariableMetadata
    
    def __post_init__(self):
        """Ensure metadata name matches variable name."""
        self.metadata.name = self.name
    
    def copy(self) -> Variable:
        """Create a deep copy of the variable."""
        return Variable(
            name=self.name,
            value=copy.deepcopy(self.value),
            metadata=copy.deepcopy(self.metadata)
        )
    
    def update_value(self, new_value: Any, source_step: Optional[str] = None):
        """Update variable value and metadata."""
        self.value = new_value
        self.metadata.update_timestamp()
        if source_step:
            self.metadata.source_step = source_step


class VariableManager:
    """
    Comprehensive variable management system for pipeline execution.
    
    Provides variable storage, retrieval, scoping, and dependency tracking
    with thread-safe operations and context isolation.
    """
    
    def __init__(self, pipeline_id: str = "default"):
        """
        Initialize the variable manager.
        
        Args:
            pipeline_id: Unique identifier for the pipeline execution
        """
        self.pipeline_id = pipeline_id
        self._variables: Dict[str, Variable] = {}
        self._scope_stack: List[Dict[str, str]] = []  # Stack of scope contexts
        self._lock = threading.RLock()  # Thread safety
        
        # Variable resolution and dependency tracking
        self._dependency_graph: Dict[str, Set[str]] = {}
        self._variable_templates: Dict[str, str] = {}
        self._resolution_cache: Dict[str, Any] = {}
        
        # Event handlers
        self._change_handlers: List[Callable[[str, Any, Any], None]] = []
        self._creation_handlers: List[Callable[[str, Any], None]] = []
        
        # Context isolation
        self._context_id = 0
        self._context_variables: Dict[str, Dict[str, Variable]] = {}
        
        logger.info(f"Initialized VariableManager for pipeline: {pipeline_id}")
    
    def set_variable(
        self,
        name: str,
        value: Any,
        scope: VariableScope = VariableScope.GLOBAL,
        var_type: VariableType = VariableType.INTERMEDIATE,
        source_step: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        context_id: Optional[str] = None
    ) -> None:
        """
        Set a variable with full metadata.
        
        Args:
            name: Variable name
            value: Variable value
            scope: Variable scope
            var_type: Variable type classification
            source_step: Step that created this variable
            description: Human-readable description
            tags: Set of tags for categorization
            context_id: Context identifier for isolation
        """
        with self._lock:
            # Get old value for change notification
            old_value = None
            if name in self._variables:
                old_value = self._variables[name].value
            
            # Create metadata
            metadata = VariableMetadata(
                name=name,
                scope=scope,
                var_type=var_type,
                source_step=source_step,
                description=description,
                tags=tags or set()
            )
            
            # Create variable
            variable = Variable(name=name, value=value, metadata=metadata)
            
            # Store in appropriate context
            if context_id:
                if context_id not in self._context_variables:
                    self._context_variables[context_id] = {}
                self._context_variables[context_id][name] = variable
            else:
                self._variables[name] = variable
            
            logger.debug(f"Set variable '{name}' = {type(value).__name__} in scope {scope.value}")
            
            # Notify handlers
            if old_value is not None:
                self._notify_change_handlers(name, old_value, value)
            else:
                self._notify_creation_handlers(name, value)
            
            # Clear resolution cache entries that depend on this variable
            self._invalidate_resolution_cache(name)
    
    def get_variable(
        self,
        name: str,
        default: Any = None,
        context_id: Optional[str] = None,
        resolve_templates: bool = True
    ) -> Any:
        """
        Get a variable value with optional template resolution.
        
        Args:
            name: Variable name
            default: Default value if variable not found
            context_id: Context to search in addition to global
            resolve_templates: Whether to resolve template expressions
            
        Returns:
            Variable value or default
        """
        with self._lock:
            # Check if this is a template variable
            if name in self._variable_templates:
                template = self._variable_templates[name]
                return self.resolve_template(template)
            
            # Search in context first if specified
            if context_id and context_id in self._context_variables:
                if name in self._context_variables[context_id]:
                    value = self._context_variables[context_id][name].value
                    return self._resolve_value(value) if resolve_templates else value
            
            # Search in global variables
            if name in self._variables:
                value = self._variables[name].value
                return self._resolve_value(value) if resolve_templates else value
            
            # Search in scope stack (for nested contexts)
            for scope_context in reversed(self._scope_stack):
                if name in scope_context:
                    context_var_name = scope_context[name]
                    if context_var_name in self._variables:
                        value = self._variables[context_var_name].value
                        return self._resolve_value(value) if resolve_templates else value
            
            return default
    
    def get_variable_metadata(
        self,
        name: str,
        context_id: Optional[str] = None
    ) -> Optional[VariableMetadata]:
        """
        Get variable metadata.
        
        Args:
            name: Variable name
            context_id: Context to search in
            
        Returns:
            Variable metadata or None if not found
        """
        with self._lock:
            # Search in context first
            if context_id and context_id in self._context_variables:
                if name in self._context_variables[context_id]:
                    return self._context_variables[context_id][name].metadata
            
            # Search in global variables
            if name in self._variables:
                return self._variables[name].metadata
            
            return None
    
    def has_variable(
        self,
        name: str,
        context_id: Optional[str] = None
    ) -> bool:
        """
        Check if a variable exists.
        
        Args:
            name: Variable name
            context_id: Context to search in
            
        Returns:
            True if variable exists
        """
        with self._lock:
            # Check context first
            if context_id and context_id in self._context_variables:
                if name in self._context_variables[context_id]:
                    return True
            
            # Check global variables
            return name in self._variables
    
    def delete_variable(
        self,
        name: str,
        context_id: Optional[str] = None
    ) -> bool:
        """
        Delete a variable.
        
        Args:
            name: Variable name
            context_id: Context to delete from
            
        Returns:
            True if variable was deleted
        """
        with self._lock:
            deleted = False
            
            # Delete from context
            if context_id and context_id in self._context_variables:
                if name in self._context_variables[context_id]:
                    del self._context_variables[context_id][name]
                    deleted = True
            
            # Delete from global variables
            if name in self._variables:
                del self._variables[name]
                deleted = True
            
            if deleted:
                logger.debug(f"Deleted variable '{name}'")
                self._invalidate_resolution_cache(name)
            
            return deleted
    
    def list_variables(
        self,
        scope: Optional[VariableScope] = None,
        var_type: Optional[VariableType] = None,
        context_id: Optional[str] = None,
        include_metadata: bool = False
    ) -> Dict[str, Any]:
        """
        List variables with optional filtering.
        
        Args:
            scope: Filter by scope
            var_type: Filter by variable type
            context_id: Context to list from
            include_metadata: Whether to include metadata
            
        Returns:
            Dictionary of variable names to values/metadata
        """
        with self._lock:
            result = {}
            
            # Get variables from specified context or global
            variables_to_check = {}
            if context_id and context_id in self._context_variables:
                variables_to_check.update(self._context_variables[context_id])
            variables_to_check.update(self._variables)
            
            for name, variable in variables_to_check.items():
                # Apply filters
                if scope and variable.metadata.scope != scope:
                    continue
                if var_type and variable.metadata.var_type != var_type:
                    continue
                
                # Include in result
                if include_metadata:
                    result[name] = {
                        'value': variable.value,
                        'metadata': variable.metadata
                    }
                else:
                    result[name] = variable.value
            
            return result
    
    def create_context(self) -> str:
        """
        Create a new isolated context for variable management.
        
        Returns:
            Context identifier
        """
        with self._lock:
            self._context_id += 1
            context_id = f"ctx_{self._context_id}_{datetime.now().timestamp()}"
            self._context_variables[context_id] = {}
            logger.debug(f"Created context: {context_id}")
            return context_id
    
    def destroy_context(self, context_id: str) -> None:
        """
        Destroy a context and all its variables.
        
        Args:
            context_id: Context to destroy
        """
        with self._lock:
            if context_id in self._context_variables:
                var_count = len(self._context_variables[context_id])
                del self._context_variables[context_id]
                logger.debug(f"Destroyed context '{context_id}' with {var_count} variables")
    
    def push_scope(self, scope_mapping: Dict[str, str]) -> None:
        """
        Push a new scope context onto the stack.
        
        Args:
            scope_mapping: Mapping of local names to global variable names
        """
        with self._lock:
            self._scope_stack.append(scope_mapping)
            logger.debug(f"Pushed scope with {len(scope_mapping)} mappings")
    
    def pop_scope(self) -> Optional[Dict[str, str]]:
        """
        Pop the current scope context from the stack.
        
        Returns:
            The popped scope mapping or None if stack is empty
        """
        with self._lock:
            if self._scope_stack:
                scope = self._scope_stack.pop()
                logger.debug(f"Popped scope with {len(scope)} mappings")
                return scope
            return None
    
    def set_template(self, name: str, template: str) -> None:
        """
        Set a variable template for dynamic resolution.
        
        Args:
            name: Variable name
            template: Template string with ${var} placeholders
        """
        with self._lock:
            self._variable_templates[name] = template
            self._invalidate_resolution_cache(name)
    
    def resolve_template(self, template: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Resolve a template string using current variables.
        
        Args:
            template: Template string with ${var} placeholders
            context: Additional context variables
            
        Returns:
            Resolved value
        """
        # Don't use cache when context is provided
        if context is None and template in self._resolution_cache:
            return self._resolution_cache[template]
        
        resolved = self._resolve_template_impl(template, context)
        
        # Only cache when no context is provided
        if context is None:
            self._resolution_cache[template] = resolved
        
        return resolved
    
    def add_dependency(self, variable_name: str, depends_on: Set[str]) -> None:
        """
        Add dependency relationships for variables.
        
        Args:
            variable_name: Name of the dependent variable
            depends_on: Set of variable names this depends on
        """
        with self._lock:
            if variable_name not in self._dependency_graph:
                self._dependency_graph[variable_name] = set()
            self._dependency_graph[variable_name].update(depends_on)
    
    def get_dependencies(self, variable_name: str) -> Set[str]:
        """
        Get dependencies for a variable.
        
        Args:
            variable_name: Variable name
            
        Returns:
            Set of variable names this depends on
        """
        return self._dependency_graph.get(variable_name, set())
    
    def get_dependents(self, variable_name: str) -> Set[str]:
        """
        Get variables that depend on this variable.
        
        Args:
            variable_name: Variable name
            
        Returns:
            Set of variable names that depend on this variable
        """
        dependents = set()
        for var, deps in self._dependency_graph.items():
            if variable_name in deps:
                dependents.add(var)
        return dependents
    
    def on_variable_changed(self, handler: Callable[[str, Any, Any], None]) -> None:
        """
        Register a handler for variable change events.
        
        Args:
            handler: Function that takes (name, old_value, new_value)
        """
        self._change_handlers.append(handler)
    
    def on_variable_created(self, handler: Callable[[str, Any], None]) -> None:
        """
        Register a handler for variable creation events.
        
        Args:
            handler: Function that takes (name, value)
        """
        self._creation_handlers.append(handler)
    
    def export_state(self, context_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Export variable state for persistence.
        
        Args:
            context_id: Specific context to export, or None for all
            
        Returns:
            Serializable state dictionary
        """
        with self._lock:
            state = {
                'pipeline_id': self.pipeline_id,
                'timestamp': datetime.now().isoformat(),
                'global_variables': {},
                'contexts': {},
                'dependency_graph': {k: list(v) for k, v in self._dependency_graph.items()},
                'variable_templates': dict(self._variable_templates),
            }
            
            # Export global variables
            for name, variable in self._variables.items():
                state['global_variables'][name] = {
                    'value': self._serialize_value(variable.value),
                    'metadata': {
                        'scope': variable.metadata.scope.value,
                        'var_type': variable.metadata.var_type.value,
                        'created_at': variable.metadata.created_at.isoformat(),
                        'updated_at': variable.metadata.updated_at.isoformat(),
                        'source_step': variable.metadata.source_step,
                        'description': variable.metadata.description,
                        'tags': list(variable.metadata.tags),
                        'version': variable.metadata.version,
                    }
                }
            
            # Export context variables
            if context_id:
                if context_id in self._context_variables:
                    state['contexts'][context_id] = self._export_context(context_id)
            else:
                for ctx_id in self._context_variables:
                    state['contexts'][ctx_id] = self._export_context(ctx_id)
            
            return state
    
    def import_state(self, state: Dict[str, Any]) -> None:
        """
        Import variable state from a dictionary.
        
        Args:
            state: State dictionary to import
        """
        with self._lock:
            self.pipeline_id = state.get('pipeline_id', self.pipeline_id)
            
            # Clear existing state
            self._variables.clear()
            self._context_variables.clear()
            self._dependency_graph.clear()
            self._variable_templates.clear()
            self._resolution_cache.clear()
            
            # Import global variables
            for name, var_data in state.get('global_variables', {}).items():
                metadata = VariableMetadata(
                    name=name,
                    scope=VariableScope(var_data['metadata']['scope']),
                    var_type=VariableType(var_data['metadata']['var_type']),
                    created_at=datetime.fromisoformat(var_data['metadata']['created_at']),
                    updated_at=datetime.fromisoformat(var_data['metadata']['updated_at']),
                    source_step=var_data['metadata']['source_step'],
                    description=var_data['metadata']['description'],
                    tags=set(var_data['metadata']['tags']),
                    version=var_data['metadata']['version'],
                )
                
                variable = Variable(
                    name=name,
                    value=self._deserialize_value(var_data['value']),
                    metadata=metadata
                )
                self._variables[name] = variable
            
            # Import contexts
            for ctx_id, ctx_data in state.get('contexts', {}).items():
                self._context_variables[ctx_id] = {}
                for name, var_data in ctx_data.items():
                    # Similar import logic for context variables
                    pass
            
            # Import dependency graph
            for var, deps in state.get('dependency_graph', {}).items():
                self._dependency_graph[var] = set(deps)
            
            # Import templates
            self._variable_templates.update(state.get('variable_templates', {}))
            
            logger.info(f"Imported variable state with {len(self._variables)} global variables")
    
    def _resolve_value(self, value: Any) -> Any:
        """Resolve template expressions in a value."""
        if isinstance(value, str) and '${' in value:
            return self.resolve_template(value)
        return value
    
    def _resolve_template_impl(self, template: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Implementation of template resolution."""
        # Simple template resolution using regex
        def replace_var(match):
            var_name = match.group(1)
            
            # Check context first
            if context and var_name in context:
                return str(context[var_name])
            
            # Check variables
            value = self.get_variable(var_name, resolve_templates=False)
            if value is not None:
                return str(value)
            
            # Return placeholder if not found
            return f"${{{var_name}}}"
        
        # Replace ${variable} patterns
        resolved = re.sub(r'\$\{([^}]+)\}', replace_var, template)
        
        # Only try JSON parsing if the result looks like JSON
        if resolved.startswith(('{', '[', '"')) or resolved in ('true', 'false', 'null'):
            try:
                return json.loads(resolved)
            except (json.JSONDecodeError, ValueError):
                pass
        
        return resolved
    
    def _notify_change_handlers(self, name: str, old_value: Any, new_value: Any) -> None:
        """Notify registered change handlers."""
        for handler in self._change_handlers:
            try:
                handler(name, old_value, new_value)
            except Exception as e:
                logger.error(f"Error in change handler for variable '{name}': {e}")
    
    def _notify_creation_handlers(self, name: str, value: Any) -> None:
        """Notify registered creation handlers."""
        for handler in self._creation_handlers:
            try:
                handler(name, value)
            except Exception as e:
                logger.error(f"Error in creation handler for variable '{name}': {e}")
    
    def _invalidate_resolution_cache(self, variable_name: str) -> None:
        """Invalidate resolution cache entries that depend on a variable."""
        keys_to_remove = []
        for key in self._resolution_cache:
            if f"${{{variable_name}}}" in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._resolution_cache[key]
        
        # Also invalidate any template variables that depend on this variable
        # and recursively invalidate templates that depend on those templates
        affected_templates = []
        for template_name, template_content in self._variable_templates.items():
            if f"${{{variable_name}}}" in template_content:
                affected_templates.append(template_name)
                if template_content in self._resolution_cache:
                    del self._resolution_cache[template_content]
        
        # Recursively invalidate templates that depend on the affected templates
        for template_name in affected_templates:
            self._invalidate_dependent_templates(template_name, set())
    
    def _invalidate_dependent_templates(self, template_name: str, visited: Set[str]) -> None:
        """Recursively invalidate templates that depend on a given template."""
        if template_name in visited:
            return  # Avoid cycles
        visited.add(template_name)
        
        # Find templates that depend on this template
        for other_template_name, template_content in self._variable_templates.items():
            if f"${{{template_name}}}" in template_content:
                if template_content in self._resolution_cache:
                    del self._resolution_cache[template_content]
                # Recursively check dependencies
                self._invalidate_dependent_templates(other_template_name, visited)
    
    def _export_context(self, context_id: str) -> Dict[str, Any]:
        """Export variables from a specific context."""
        context_data = {}
        for name, variable in self._context_variables[context_id].items():
            context_data[name] = {
                'value': self._serialize_value(variable.value),
                'metadata': {
                    'scope': variable.metadata.scope.value,
                    'var_type': variable.metadata.var_type.value,
                    'created_at': variable.metadata.created_at.isoformat(),
                    'updated_at': variable.metadata.updated_at.isoformat(),
                    'source_step': variable.metadata.source_step,
                    'description': variable.metadata.description,
                    'tags': list(variable.metadata.tags),
                    'version': variable.metadata.version,
                }
            }
        return context_data
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for state export."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, (list, dict)):
            return value  # JSON-serializable
        else:
            # For complex objects, convert to string representation
            return str(value)
    
    def _deserialize_value(self, value: Any) -> Any:
        """Deserialize a value from state import."""
        return value  # Simple implementation, could be enhanced


class VariableContext:
    """
    Context manager for variable scoping and isolation.
    
    Provides a clean way to manage variable scopes and ensure
    proper cleanup of temporary variables.
    """
    
    def __init__(self, variable_manager: VariableManager):
        """
        Initialize variable context.
        
        Args:
            variable_manager: The variable manager to use
        """
        self.variable_manager = variable_manager
        self.context_id = None
        self.scope_mapping = None
    
    def __enter__(self) -> VariableContext:
        """Enter the context and create isolation."""
        self.context_id = self.variable_manager.create_context()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context and clean up."""
        if self.context_id:
            self.variable_manager.destroy_context(self.context_id)
        if self.scope_mapping:
            self.variable_manager.pop_scope()
    
    def set_variable(self, name: str, value: Any, **kwargs) -> None:
        """Set a variable in this context."""
        self.variable_manager.set_variable(
            name, value, context_id=self.context_id, **kwargs
        )
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a variable from this context."""
        return self.variable_manager.get_variable(
            name, default, context_id=self.context_id
        )
    
    def with_scope(self, scope_mapping: Dict[str, str]) -> VariableContext:
        """Add scope mapping to this context."""
        self.scope_mapping = scope_mapping
        self.variable_manager.push_scope(scope_mapping)
        return self