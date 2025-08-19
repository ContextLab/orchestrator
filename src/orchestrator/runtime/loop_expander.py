"""
Runtime Loop Expansion System.

This module implements dynamic loop expansion for for_each and while loops
as part of the runtime dependency resolution system (Issue #211).
"""

from __future__ import annotations

import logging
import copy
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import uuid

from .execution_state import PipelineExecutionState, LoopContext, UnresolvedItem
from .dependency_resolver import DependencyResolver

logger = logging.getLogger(__name__)


@dataclass
class LoopTask:
    """Represents a loop that needs runtime expansion."""
    id: str
    loop_type: str  # 'for_each' or 'while'
    condition_expr: Optional[str] = None  # For while loops
    iterator_expr: Optional[str] = None  # For for_each loops
    loop_steps: List[Dict[str, Any]] = field(default_factory=list)
    max_parallel: int = 1
    max_iterations: int = 1000
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime state
    current_iteration: int = 0
    expanded_task_ids: List[str] = field(default_factory=list)
    completed: bool = False


@dataclass
class ExpandedTask:
    """Represents a task expanded from a loop."""
    id: str
    original_step_id: str
    loop_id: str
    iteration: int
    action: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    metadata: Dict[str, Any]
    loop_context: LoopContext
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for execution."""
        return {
            'id': self.id,
            'action': self.action,
            'parameters': self.parameters,
            'dependencies': self.dependencies,
            'metadata': {
                **self.metadata,
                'is_loop_child': True,  # Always true for ExpandedTask
                'loop_id': self.loop_id,
                'iteration': self.iteration,
                'original_step_id': self.original_step_id,
                'loop_context': {
                    'item': self.loop_context.item,
                    'index': self.loop_context.index,
                    'iteration': self.loop_context.iteration,
                    'is_first': self.loop_context.is_first,
                    'is_last': self.loop_context.is_last,
                }
            }
        }


class LoopExpander:
    """
    Handles runtime expansion of loops.
    
    Expands for_each and while loops when their dependencies are satisfied,
    creating concrete tasks with proper context.
    """
    
    def __init__(self, execution_state: PipelineExecutionState, 
                 dependency_resolver: DependencyResolver):
        """
        Initialize the loop expander.
        
        Args:
            execution_state: Pipeline execution state
            dependency_resolver: Dependency resolver for evaluating expressions
        """
        self.state = execution_state
        self.resolver = dependency_resolver
        self.active_loops: Dict[str, LoopTask] = {}
        
        logger.info("Initialized LoopExpander")
    
    def can_expand(self, loop_task: LoopTask) -> bool:
        """
        Check if a loop can be expanded.
        
        Args:
            loop_task: Loop task to check
            
        Returns:
            True if loop can be expanded
        """
        if loop_task.loop_type == "for_each":
            return self._can_expand_for_each(loop_task)
        elif loop_task.loop_type == "while":
            return self._can_expand_while(loop_task)
        else:
            logger.error(f"Unknown loop type: {loop_task.loop_type}")
            return False
    
    def _can_expand_for_each(self, loop_task: LoopTask) -> bool:
        """Check if a for_each loop can be expanded."""
        if not loop_task.iterator_expr:
            logger.error(f"For_each loop {loop_task.id} has no iterator expression")
            return False
        
        # Check if iterator can be resolved
        try:
            # Extract dependencies from iterator expression
            deps = self.resolver.extract_dependencies(loop_task.iterator_expr)
            available_context = set(self.state.get_available_context().keys())
            
            # Check if all dependencies are available
            if not deps.issubset(available_context):
                missing = deps - available_context
                logger.debug(f"Loop {loop_task.id} missing dependencies: {missing}")
                return False
            
            # Try to resolve the iterator
            resolved = self.resolver.resolve_template(loop_task.iterator_expr)
            
            # Try to evaluate as expression to get actual items
            if resolved.startswith('[') or resolved.startswith('('):
                # Looks like a Python list/tuple
                try:
                    items = self.resolver.resolve_expression(resolved)
                    return isinstance(items, (list, tuple))
                except:
                    pass
            
            # Check if it's already a list in context
            if resolved in available_context:
                items = self.state.get_available_context()[resolved]
                return isinstance(items, (list, tuple))
            
            return True  # Assume it can be resolved
            
        except Exception as e:
            logger.debug(f"Cannot expand for_each loop {loop_task.id}: {e}")
            return False
    
    def _can_expand_while(self, loop_task: LoopTask) -> bool:
        """Check if a while loop can be expanded (next iteration)."""
        if not loop_task.condition_expr:
            logger.error(f"While loop {loop_task.id} has no condition expression")
            return False
        
        # Check if condition can be evaluated
        try:
            deps = self.resolver.extract_dependencies(loop_task.condition_expr)
            available_context = set(self.state.get_available_context().keys())
            
            if not deps.issubset(available_context):
                missing = deps - available_context
                logger.debug(f"Loop {loop_task.id} condition missing dependencies: {missing}")
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Cannot evaluate while loop {loop_task.id} condition: {e}")
            return False
    
    def expand_for_each(self, loop_task: LoopTask) -> List[ExpandedTask]:
        """
        Expand a for_each loop into concrete tasks.
        
        Args:
            loop_task: For_each loop to expand
            
        Returns:
            List of expanded tasks
        """
        if not self._can_expand_for_each(loop_task):
            return []
        
        # Resolve iterator to get items
        try:
            resolved_expr = self.resolver.resolve_template(loop_task.iterator_expr)
            
            # Try to evaluate as expression
            if resolved_expr.startswith('[') or resolved_expr.startswith('('):
                items = self.resolver.resolve_expression(resolved_expr)
            elif resolved_expr in self.state.get_available_context():
                items = self.state.get_available_context()[resolved_expr]
            else:
                # Try direct evaluation
                items = self.resolver.resolve_expression(resolved_expr)
            
            if not isinstance(items, (list, tuple)):
                logger.error(f"Iterator for loop {loop_task.id} did not resolve to a list: {type(items)}")
                return []
            
        except Exception as e:
            logger.error(f"Failed to resolve iterator for loop {loop_task.id}: {e}")
            return []
        
        logger.info(f"Expanding for_each loop {loop_task.id} with {len(items)} items")
        
        expanded_tasks = []
        
        # Get parent loop context if nested
        parent_context = self.state.get_current_loop_context()
        
        for idx, item in enumerate(items):
            # Create loop context for this iteration
            loop_context = LoopContext(
                loop_id=loop_task.id,
                iteration=idx,
                item=item,
                index=idx,
                is_first=(idx == 0),
                is_last=(idx == len(items) - 1),
                total_items=len(items),
                parent_context=parent_context
            )
            
            # Push context for template resolution
            self.state.push_loop_context(f"{loop_task.id}_{idx}", loop_context)
            
            try:
                # Expand each step in the loop body
                for step_def in loop_task.loop_steps:
                    expanded_task = self._expand_loop_step(
                        loop_task, step_def, idx, loop_context
                    )
                    if expanded_task:
                        expanded_tasks.append(expanded_task)
                        loop_task.expanded_task_ids.append(expanded_task.id)
            
            finally:
                # Pop context after expanding this iteration
                self.state.pop_loop_context()
        
        # Mark loop as completed
        loop_task.completed = True
        logger.info(f"Expanded for_each loop {loop_task.id} into {len(expanded_tasks)} tasks")
        
        return expanded_tasks
    
    def expand_while_iteration(self, loop_task: LoopTask) -> List[ExpandedTask]:
        """
        Expand next iteration of a while loop.
        
        Args:
            loop_task: While loop to expand
            
        Returns:
            List of expanded tasks for this iteration, or empty if condition is false
        """
        if not self._can_expand_while(loop_task):
            return []
        
        # Check iteration limit
        if loop_task.current_iteration >= loop_task.max_iterations:
            logger.warning(f"While loop {loop_task.id} reached max iterations ({loop_task.max_iterations})")
            loop_task.completed = True
            return []
        
        # Evaluate condition
        try:
            condition_result = self._evaluate_condition(loop_task.condition_expr)
            if not condition_result:
                logger.info(f"While loop {loop_task.id} condition is false, stopping")
                loop_task.completed = True
                return []
        except Exception as e:
            logger.error(f"Failed to evaluate while loop {loop_task.id} condition: {e}")
            loop_task.completed = True
            return []
        
        logger.info(f"Expanding while loop {loop_task.id} iteration {loop_task.current_iteration}")
        
        expanded_tasks = []
        
        # Get parent loop context if nested
        parent_context = self.state.get_current_loop_context()
        
        # Create loop context for this iteration
        loop_context = LoopContext(
            loop_id=loop_task.id,
            iteration=loop_task.current_iteration,
            item=None,  # While loops don't have items
            index=loop_task.current_iteration,
            is_first=(loop_task.current_iteration == 0),
            is_last=False,  # Unknown for while loops
            parent_context=parent_context
        )
        
        # Push context for template resolution
        self.state.push_loop_context(
            f"{loop_task.id}_{loop_task.current_iteration}", 
            loop_context
        )
        
        try:
            # Expand each step in the loop body
            for step_def in loop_task.loop_steps:
                expanded_task = self._expand_loop_step(
                    loop_task, step_def, loop_task.current_iteration, loop_context
                )
                if expanded_task:
                    expanded_tasks.append(expanded_task)
                    loop_task.expanded_task_ids.append(expanded_task.id)
        
        finally:
            # Pop context after expanding this iteration
            self.state.pop_loop_context()
        
        # Increment iteration counter
        loop_task.current_iteration += 1
        
        logger.info(f"Expanded while loop {loop_task.id} iteration into {len(expanded_tasks)} tasks")
        
        return expanded_tasks
    
    def _expand_loop_step(self, loop_task: LoopTask, step_def: Dict[str, Any],
                         iteration: int, loop_context: LoopContext) -> Optional[ExpandedTask]:
        """
        Expand a single step within a loop iteration.
        
        Args:
            loop_task: Parent loop task
            step_def: Step definition to expand
            iteration: Current iteration number
            loop_context: Loop context for this iteration
            
        Returns:
            Expanded task or None if expansion fails
        """
        try:
            # Generate unique task ID
            original_step_id = step_def.get('id', f"step_{uuid.uuid4().hex[:8]}")
            task_id = f"{loop_task.id}_{iteration}_{original_step_id}"
            
            # Resolve parameters with loop context
            parameters = step_def.get('parameters', {})
            resolved_params = self._resolve_parameters(parameters, loop_context)
            
            # Handle dependencies
            dependencies = []
            
            # Add loop task dependencies (for first iteration or all if parallel)
            if iteration == 0 or loop_task.max_parallel > 1:
                dependencies.extend(loop_task.dependencies)
            
            # Add previous iteration dependency for sequential execution
            if loop_task.max_parallel == 1 and iteration > 0:
                prev_task_id = f"{loop_task.id}_{iteration-1}_{original_step_id}"
                dependencies.append(prev_task_id)
            
            # Add step-level dependencies
            step_deps = step_def.get('dependencies', [])
            for dep in step_deps:
                # Check if dependency is within the loop
                if any(s.get('id') == dep for s in loop_task.loop_steps):
                    # Internal dependency - reference task from same iteration
                    dependencies.append(f"{loop_task.id}_{iteration}_{dep}")
                else:
                    # External dependency
                    dependencies.append(dep)
            
            # Create expanded task
            expanded_task = ExpandedTask(
                id=task_id,
                original_step_id=original_step_id,
                loop_id=loop_task.id,
                iteration=iteration,
                action=step_def.get('action', 'execute'),
                parameters=resolved_params,
                dependencies=dependencies,
                metadata={
                    **step_def.get('metadata', {}),
                    'is_loop_child': True,
                    'loop_type': loop_task.loop_type,
                    'loop_iteration': iteration,
                },
                loop_context=loop_context
            )
            
            logger.debug(f"Expanded step {original_step_id} -> {task_id} with {len(dependencies)} dependencies")
            
            return expanded_task
            
        except Exception as e:
            logger.error(f"Failed to expand loop step: {e}")
            return None
    
    def _resolve_parameters(self, parameters: Dict[str, Any], 
                           loop_context: LoopContext) -> Dict[str, Any]:
        """
        Resolve parameters with loop context.
        
        Args:
            parameters: Original parameters
            loop_context: Loop context for resolution
            
        Returns:
            Resolved parameters
        """
        # Deep copy to avoid modifying original
        resolved = copy.deepcopy(parameters)
        
        # Add loop context to available context for resolution
        additional_context = loop_context.to_dict()
        
        # Resolve each parameter
        for key, value in resolved.items():
            if isinstance(value, str):
                # Check if it contains templates
                if '{{' in value or '{%' in value:
                    try:
                        resolved[key] = self.resolver.resolve_template(value, additional_context)
                    except Exception as e:
                        logger.warning(f"Failed to resolve parameter {key}: {e}")
            elif isinstance(value, dict):
                # Recursively resolve nested dicts
                resolved[key] = self._resolve_parameters(value, loop_context)
            elif isinstance(value, list):
                # Resolve list items
                resolved[key] = [
                    self.resolver.resolve_template(item, additional_context)
                    if isinstance(item, str) and ('{{' in item or '{%' in item)
                    else item
                    for item in value
                ]
        
        return resolved
    
    def _evaluate_condition(self, condition_expr: str) -> bool:
        """
        Evaluate a condition expression.
        
        Args:
            condition_expr: Condition to evaluate
            
        Returns:
            Boolean result
        """
        try:
            # First resolve any templates
            resolved = self.resolver.resolve_template(condition_expr)
            
            # If it's already a boolean string, convert it
            if resolved.lower() in ['true', 'false']:
                return resolved.lower() == 'true'
            
            # Evaluate the resolved expression
            result = self.resolver.resolve_expression(resolved)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to evaluate condition '{condition_expr}': {e}")
            raise
    
    def register_loop(self, loop_task: LoopTask) -> None:
        """
        Register a loop for tracking.
        
        Args:
            loop_task: Loop task to register
        """
        self.active_loops[loop_task.id] = loop_task
        logger.info(f"Registered {loop_task.loop_type} loop: {loop_task.id}")
    
    def get_loop(self, loop_id: str) -> Optional[LoopTask]:
        """
        Get a registered loop by ID.
        
        Args:
            loop_id: Loop ID
            
        Returns:
            Loop task or None if not found
        """
        return self.active_loops.get(loop_id)
    
    def is_loop_complete(self, loop_id: str) -> bool:
        """
        Check if a loop is complete.
        
        Args:
            loop_id: Loop ID
            
        Returns:
            True if loop is complete
        """
        loop = self.active_loops.get(loop_id)
        return loop.completed if loop else True
    
    def get_expandable_loops(self) -> List[LoopTask]:
        """
        Get all loops that can be expanded now.
        
        Returns:
            List of expandable loops
        """
        expandable = []
        
        for loop_task in self.active_loops.values():
            if not loop_task.completed and self.can_expand(loop_task):
                expandable.append(loop_task)
        
        return expandable
    
    def expand_all_ready(self) -> List[ExpandedTask]:
        """
        Expand all loops that are ready.
        
        Returns:
            List of all expanded tasks
        """
        all_expanded = []
        
        for loop_task in self.get_expandable_loops():
            if loop_task.loop_type == "for_each":
                expanded = self.expand_for_each(loop_task)
            elif loop_task.loop_type == "while":
                expanded = self.expand_while_iteration(loop_task)
            else:
                continue
            
            all_expanded.extend(expanded)
        
        return all_expanded