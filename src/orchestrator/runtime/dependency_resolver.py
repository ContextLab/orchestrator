"""
Runtime Dependency Resolution Engine.

This module implements the progressive dependency resolution system for runtime
template rendering and loop expansion as outlined in Issue #211.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from jinja2 import Environment, StrictUndefined, TemplateSyntaxError, UndefinedError, meta
import ast

from .execution_state import PipelineExecutionState, UnresolvedItem, ItemStatus

logger = logging.getLogger(__name__)


@dataclass
class ResolutionResult:
    """Result of a resolution attempt."""
    success: bool
    resolved_items: List[str] = None
    failed_items: List[str] = None
    unresolved_items: List[str] = None
    error_message: Optional[str] = None
    iterations: int = 0
    
    def __post_init__(self):
        if self.resolved_items is None:
            self.resolved_items = []
        if self.failed_items is None:
            self.failed_items = []
        if self.unresolved_items is None:
            self.unresolved_items = []


class DependencyResolver:
    """
    Progressive dependency resolution engine.
    
    Iteratively resolves templates, AUTO tags, and expressions as their
    dependencies become available during pipeline execution.
    """
    
    def __init__(self, execution_state: PipelineExecutionState, max_iterations: int = 100):
        """
        Initialize the dependency resolver.
        
        Args:
            execution_state: Pipeline execution state to work with
            max_iterations: Maximum resolution iterations to prevent infinite loops
        """
        self.state = execution_state
        self.max_iterations = max_iterations
        
        # Initialize Jinja2 environment for template parsing
        self.jinja_env = Environment(undefined=StrictUndefined)
        
        # Regex patterns for different types of references
        self.patterns = {
            'jinja_var': re.compile(r'{{\s*([^}|]+?)(?:\|[^}]+)?\s*}}'),  # {{ variable }}
            'jinja_expr': re.compile(r'{%\s*(.+?)\s*%}'),  # {% expression %}
            'auto_tag': re.compile(r'<AUTO(?:\s+[^>]*)?>(.+?)</AUTO>', re.DOTALL),  # <AUTO>...</AUTO>
            'step_result': re.compile(r'\b(\w+)\.result\b'),  # step_id.result
            'step_value': re.compile(r'\b(\w+)\.value\b'),  # step_id.value
            'array_access': re.compile(r'\b(\w+)\[(\d+)\]'),  # array[index]
            'dict_access': re.compile(r'\b(\w+)\[[\'"]([\w-]+)[\'"]\]'),  # dict['key']
        }
        
        logger.info(f"Initialized DependencyResolver with max_iterations={max_iterations}")
    
    def extract_dependencies(self, content: str) -> Set[str]:
        """
        Extract all dependencies from a template string.
        
        Args:
            content: Template string to analyze
            
        Returns:
            Set of dependency names
        """
        dependencies = set()
        
        # Extract Jinja2 variable references
        for match in self.patterns['jinja_var'].finditer(content):
            var_expr = match.group(1).strip()
            # Handle dotted access (e.g., task.result)
            if '.' in var_expr:
                base_var = var_expr.split('.')[0]
                dependencies.add(base_var)
            # Handle array/dict access
            elif '[' in var_expr:
                base_var = var_expr.split('[')[0]
                dependencies.add(base_var)
            else:
                dependencies.add(var_expr)
        
        # Extract variables from Jinja2 expressions
        for match in self.patterns['jinja_expr'].finditer(content):
            expr = match.group(1).strip()
            # Parse expression to find variable references
            deps = self._extract_from_jinja_expression(expr)
            dependencies.update(deps)
        
        # Extract step result references
        for match in self.patterns['step_result'].finditer(content):
            step_id = match.group(1)
            dependencies.add(step_id)
        
        for match in self.patterns['step_value'].finditer(content):
            step_id = match.group(1)
            dependencies.add(step_id)
        
        # Extract array access
        for match in self.patterns['array_access'].finditer(content):
            array_name = match.group(1)
            dependencies.add(array_name)
        
        # Extract dict access  
        for match in self.patterns['dict_access'].finditer(content):
            dict_name = match.group(1)
            dependencies.add(dict_name)
        
        # Extract AUTO tag dependencies (the content might reference variables)
        for match in self.patterns['auto_tag'].finditer(content):
            auto_content = match.group(1)
            # Recursively extract dependencies from AUTO tag content
            auto_deps = self.extract_dependencies(auto_content)
            dependencies.update(auto_deps)
        
        # Filter out common Python/Jinja2 keywords and functions
        keywords = {
            'if', 'else', 'elif', 'for', 'in', 'and', 'or', 'not', 'is',
            'true', 'false', 'none', 'True', 'False', 'None',
            'range', 'len', 'str', 'int', 'float', 'list', 'dict',
            'loop', 'item', 'index'  # Common loop variables
        }
        dependencies = dependencies - keywords
        
        return dependencies
    
    def _extract_from_jinja_expression(self, expression: str) -> Set[str]:
        """
        Extract variable references from a Jinja2 expression.
        
        Args:
            expression: Jinja2 expression (e.g., "if x > 5", "for item in items")
            
        Returns:
            Set of variable names
        """
        dependencies = set()
        
        try:
            # Use Jinja2's parser to extract variables
            ast = self.jinja_env.parse(f"{{% {expression} %}}")
            variables = meta.find_undeclared_variables(ast)
            dependencies.update(variables)
        except TemplateSyntaxError:
            # Fallback to simple parsing if Jinja2 parsing fails
            # Extract potential variable names (alphanumeric + underscore)
            tokens = re.findall(r'\b([a-zA-Z_]\w*)\b', expression)
            dependencies.update(tokens)
        
        return dependencies
    
    def can_resolve(self, item: UnresolvedItem) -> bool:
        """
        Check if an item can be resolved with current context.
        
        Args:
            item: Item to check
            
        Returns:
            True if all dependencies are satisfied
        """
        if item.status == ItemStatus.FAILED:
            return False
        
        available_context = set(self.state.get_available_context().keys())
        
        # Check dependencies
        if not item.dependencies.issubset(available_context):
            missing = item.dependencies - available_context
            logger.debug(f"Item '{item.id}' missing dependencies: {missing}")
            return False
        
        # Check context requirements
        if not item.context_requirements.issubset(available_context):
            missing = item.context_requirements - available_context
            logger.debug(f"Item '{item.id}' missing context: {missing}")
            return False
        
        return True
    
    def resolve_template(self, template_str: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Resolve a template string with available context.
        
        Args:
            template_str: Template string to resolve
            context: Optional additional context
            
        Returns:
            Resolved string
            
        Raises:
            TemplateSyntaxError: If template syntax is invalid
            UndefinedError: If required variables are undefined
        """
        if context is None:
            context = self.state.get_available_context()
        else:
            # Merge with global context
            context = {**self.state.get_available_context(), **context}
        
        try:
            template = self.jinja_env.from_string(template_str)
            return template.render(context)
        except UndefinedError as e:
            logger.warning(f"Template rendering failed - undefined variable: {e}")
            raise
        except TemplateSyntaxError as e:
            logger.error(f"Template syntax error: {e}")
            raise
    
    def resolve_auto_tag(self, auto_content: str, tag_metadata: Optional[Dict[str, Any]] = None) -> Any:
        """
        Resolve an AUTO tag content.
        
        For now, this is a placeholder that returns the content as-is.
        In the full implementation, this would call the AUTO tag resolver.
        
        Args:
            auto_content: Content inside the AUTO tag
            tag_metadata: Optional metadata about the tag
            
        Returns:
            Resolved value (could be any type)
        """
        # First resolve any templates within the AUTO tag content
        try:
            resolved_content = self.resolve_template(auto_content)
        except (UndefinedError, TemplateSyntaxError):
            resolved_content = auto_content
        
        # In a real implementation, this would:
        # 1. Parse the AUTO tag directive
        # 2. Call the appropriate model/service
        # 3. Return the generated value
        
        # For testing, we'll return a mock resolved value based on content
        if "list" in resolved_content.lower() or "items" in resolved_content.lower():
            return ["item1", "item2", "item3"]
        elif "model" in resolved_content.lower():
            return "gpt-4"
        elif "number" in resolved_content.lower():
            return 42
        else:
            return resolved_content
    
    def resolve_expression(self, expression: str) -> Any:
        """
        Resolve a Python expression.
        
        Args:
            expression: Python expression to evaluate
            
        Returns:
            Evaluated result
        """
        context = self.state.get_available_context()
        
        try:
            # Safely evaluate the expression
            # In production, this should use a sandboxed evaluator
            result = eval(expression, {"__builtins__": {}}, context)
            return result
        except Exception as e:
            logger.error(f"Failed to evaluate expression '{expression}': {e}")
            raise
    
    def resolve_item(self, item: UnresolvedItem) -> Tuple[bool, Any]:
        """
        Attempt to resolve a single item.
        
        Args:
            item: Item to resolve
            
        Returns:
            Tuple of (success, resolved_value)
        """
        if not self.can_resolve(item):
            return False, None
        
        item.increment_attempts()
        
        try:
            if item.item_type == "template":
                resolved = self.resolve_template(item.content)
                item.mark_resolved(resolved)
                return True, resolved
            
            elif item.item_type == "auto_tag":
                # Extract AUTO tag content
                match = self.patterns['auto_tag'].search(item.content)
                if match:
                    auto_content = match.group(1)
                    resolved = self.resolve_auto_tag(auto_content, item.metadata)
                    item.mark_resolved(resolved)
                    return True, resolved
                else:
                    raise ValueError(f"Invalid AUTO tag format: {item.content}")
            
            elif item.item_type == "expression":
                resolved = self.resolve_expression(item.content)
                item.mark_resolved(resolved)
                return True, resolved
            
            elif item.item_type == "loop":
                # For loop iterators, resolve the expression
                resolved = self.resolve_template(item.content)
                # Try to evaluate as Python expression to get actual list
                try:
                    resolved = self.resolve_expression(resolved)
                except:
                    # If not evaluable, treat as string list
                    if isinstance(resolved, str) and ',' in resolved:
                        resolved = [s.strip() for s in resolved.split(',')]
                item.mark_resolved(resolved)
                return True, resolved
            
            elif item.item_type == "condition":
                # Resolve condition template then evaluate as boolean
                resolved_str = self.resolve_template(item.content)
                # Evaluate as boolean expression
                resolved = self.resolve_expression(resolved_str)
                item.mark_resolved(bool(resolved))
                return True, bool(resolved)
            
            else:
                logger.warning(f"Unknown item type: {item.item_type}")
                return False, None
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to resolve item '{item.id}': {error_msg}")
            item.mark_failed(error_msg)
            return False, None
    
    def resolve_all_pending(self) -> ResolutionResult:
        """
        Iteratively resolve all pending items until convergence.
        
        Returns:
            Resolution result with statistics
        """
        result = ResolutionResult(success=True)
        iterations = 0
        total_resolved = []
        total_failed = []
        
        while iterations < self.max_iterations:
            iterations += 1
            progress_made = False
            
            # Get current unresolved items
            unresolved = list(self.state.unresolved_items)
            
            for item in unresolved:
                if item.status == ItemStatus.FAILED:
                    continue
                
                success, resolved_value = self.resolve_item(item)
                
                if success:
                    # Update state
                    self.state.mark_item_resolved(item.id, resolved_value)
                    
                    # Register resolved value based on type
                    if item.item_type == "template":
                        self.state.register_template(item.id, resolved_value)
                    elif item.item_type == "auto_tag":
                        self.state.register_auto_tag(item.id, resolved_value)
                    
                    total_resolved.append(item.id)
                    progress_made = True
                    logger.info(f"Resolved item '{item.id}' in iteration {iterations}")
                
                elif item.status == ItemStatus.FAILED:
                    total_failed.append(item.id)
            
            # Check if we're done
            remaining_unresolved = [
                item for item in self.state.unresolved_items
                if item.status not in [ItemStatus.RESOLVED, ItemStatus.FAILED]
            ]
            
            if not remaining_unresolved:
                # All items either resolved or failed
                break
            
            if not progress_made:
                # No progress made - check for circular dependencies
                has_circular, cycle = self.state.has_circular_dependencies()
                if has_circular:
                    result.success = False
                    result.error_message = f"Circular dependency detected: {' -> '.join(cycle)}"
                    break
                
                # Check if items are truly blocked
                resolvable = self.state.get_resolvable_items()
                if not resolvable:
                    # Items are blocked by missing dependencies
                    missing_deps = set()
                    for item in remaining_unresolved:
                        available = set(self.state.get_available_context().keys())
                        missing = item.dependencies - available
                        missing_deps.update(missing)
                    
                    result.success = False
                    result.error_message = f"Cannot resolve - missing dependencies: {missing_deps}"
                    break
        
        # Finalize result
        result.iterations = iterations
        result.resolved_items = total_resolved
        result.failed_items = total_failed
        result.unresolved_items = [
            item.id for item in self.state.unresolved_items
            if item.status == ItemStatus.UNRESOLVED
        ]
        
        if iterations >= self.max_iterations:
            result.success = False
            result.error_message = f"Maximum iterations ({self.max_iterations}) reached"
        
        # Update state metadata
        self.state.metadata['resolution_iterations'] = iterations
        
        logger.info(f"Resolution complete: {len(total_resolved)} resolved, "
                   f"{len(total_failed)} failed, {len(result.unresolved_items)} unresolved "
                   f"in {iterations} iterations")
        
        return result
    
    def resolve_single_template(self, template_str: str, 
                               item_id: Optional[str] = None) -> Tuple[bool, str]:
        """
        Convenience method to resolve a single template string.
        
        Args:
            template_str: Template to resolve
            item_id: Optional ID for the item
            
        Returns:
            Tuple of (success, resolved_string)
        """
        if item_id is None:
            item_id = f"template_{id(template_str)}"
        
        # Extract dependencies
        deps = self.extract_dependencies(template_str)
        
        # Create unresolved item
        item = UnresolvedItem(
            id=item_id,
            content=template_str,
            item_type="template",
            dependencies=deps
        )
        
        # Add to state
        self.state.add_unresolved_item(item)
        
        # Try to resolve
        success, resolved = self.resolve_item(item)
        
        if success:
            self.state.mark_item_resolved(item_id, resolved)
        
        return success, resolved if success else template_str
    
    def get_resolution_order(self) -> List[str]:
        """
        Get the optimal order for resolving items based on dependencies.
        
        Returns:
            List of item IDs in resolution order
        """
        # Build dependency graph
        graph = {}
        in_degree = {}
        
        for item in self.state.unresolved_items:
            if item.status == ItemStatus.FAILED:
                continue
            
            graph[item.id] = list(item.dependencies)
            in_degree[item.id] = len(item.dependencies)
        
        # Topological sort
        queue = [item_id for item_id, degree in in_degree.items() if degree == 0]
        order = []
        
        while queue:
            current = queue.pop(0)
            order.append(current)
            
            # Update in-degrees
            for item_id in graph:
                if current in graph[item_id]:
                    graph[item_id].remove(current)
                    in_degree[item_id] -= 1
                    if in_degree[item_id] == 0:
                        queue.append(item_id)
        
        return order