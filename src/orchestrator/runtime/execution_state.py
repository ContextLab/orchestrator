"""
Pipeline Execution State Management for Runtime Dependency Resolution.

This module implements the centralized state management system for runtime
dependency resolution as outlined in Issue #211. It maintains all variables,
results, and contexts during pipeline execution.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import copy

logger = logging.getLogger(__name__)


class ItemStatus(Enum):
    """Status of an item in the resolution system."""
    UNRESOLVED = "unresolved"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class UnresolvedItem:
    """Represents an item waiting for dependency resolution."""
    id: str
    content: str  # Template string or expression to resolve
    item_type: str  # 'template', 'loop', 'condition', 'auto_tag'
    dependencies: Set[str] = field(default_factory=set)
    context_requirements: Set[str] = field(default_factory=set)
    status: ItemStatus = ItemStatus.UNRESOLVED
    resolution_attempts: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, UnresolvedItem):
            return False
        return self.id == other.id
    
    def mark_resolved(self, resolved_value: Any):
        """Mark this item as resolved with the given value."""
        self.status = ItemStatus.RESOLVED
        self.metadata['resolved_value'] = resolved_value
        self.metadata['resolved_at'] = datetime.now().isoformat()
    
    def mark_failed(self, error: str):
        """Mark this item as failed with an error message."""
        self.status = ItemStatus.FAILED
        self.error_message = error
        self.metadata['failed_at'] = datetime.now().isoformat()
        
    def increment_attempts(self):
        """Increment resolution attempt counter."""
        self.resolution_attempts += 1


@dataclass
class LoopContext:
    """Context for a loop iteration."""
    loop_id: str
    iteration: int
    item: Any = None
    index: Optional[int] = None
    is_first: bool = False
    is_last: bool = False
    total_items: Optional[int] = None
    parent_context: Optional[LoopContext] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert loop context to dictionary for template rendering."""
        return {
            'loop_id': self.loop_id,
            'iteration': self.iteration,
            'item': self.item,
            'index': self.index,
            'is_first': self.is_first,
            'is_last': self.is_last,
            'total_items': self.total_items,
            **self.variables,
            # Also provide common loop variable names
            '$item': self.item,
            '$index': self.index,
            '$is_first': self.is_first,
            '$is_last': self.is_last,
        }


class PipelineExecutionState:
    """
    Centralized state management for pipeline execution.
    
    Maintains all variables, results, and contexts during pipeline execution,
    enabling runtime dependency resolution and dynamic loop expansion.
    """
    
    def __init__(self, pipeline_id: str = "default"):
        """
        Initialize execution state.
        
        Args:
            pipeline_id: Unique identifier for this pipeline execution
        """
        self.pipeline_id = pipeline_id
        self.start_time = datetime.now()
        
        # Core state storage
        self.global_context = {
            'variables': {},      # User-defined and system variables
            'results': {},        # Task execution results
            'templates': {},      # Resolved template values
            'auto_tags': {},      # Resolved AUTO tag values
        }
        
        # Loop management
        self.loop_contexts: Dict[str, LoopContext] = {}
        self.active_loops: List[str] = []  # Stack of active loop IDs
        
        # Dependency tracking
        self.unresolved_items: Set[UnresolvedItem] = set()
        self.resolved_items: Dict[str, Any] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}  # item_id -> dependencies
        
        # Execution tracking
        self.executed_tasks: Set[str] = set()
        self.pending_tasks: Set[str] = set()
        self.failed_tasks: Dict[str, str] = {}  # task_id -> error message
        
        # Metadata
        self.metadata = {
            'pipeline_id': pipeline_id,
            'start_time': self.start_time.isoformat(),
            'resolution_iterations': 0,
            'total_tasks_executed': 0,
        }
        
        logger.info(f"Initialized PipelineExecutionState for pipeline: {pipeline_id}")
    
    def register_variable(self, name: str, value: Any) -> None:
        """
        Register a variable in the global context.
        
        Args:
            name: Variable name
            value: Variable value
        """
        self.global_context['variables'][name] = value
        logger.debug(f"Registered variable '{name}': {type(value).__name__}")
        
        # Trigger resolution check for items depending on this variable
        self._trigger_resolution_check(name)
    
    def register_result(self, task_id: str, result: Any) -> None:
        """
        Register a task execution result.
        
        Args:
            task_id: Task identifier
            result: Task execution result
        """
        self.global_context['results'][task_id] = result
        self.executed_tasks.add(task_id)
        self.pending_tasks.discard(task_id)
        self.metadata['total_tasks_executed'] += 1
        
        logger.info(f"Registered result for task '{task_id}': {type(result).__name__}")
        
        # Also register common access patterns
        if isinstance(result, dict):
            # Register dict.result if it exists
            if 'result' in result:
                self.global_context['results'][f"{task_id}_result"] = result['result']
            # Register dict.value if it exists
            if 'value' in result:
                self.global_context['results'][f"{task_id}_value"] = result['value']
        elif isinstance(result, str):
            # Register string results with _str suffix for easy access
            self.global_context['results'][f"{task_id}_str"] = result
        
        # Trigger resolution check
        self._trigger_resolution_check(task_id)
    
    def register_template(self, template_id: str, resolved_value: str) -> None:
        """
        Register a resolved template value.
        
        Args:
            template_id: Template identifier
            resolved_value: Resolved template string
        """
        self.global_context['templates'][template_id] = resolved_value
        logger.debug(f"Registered resolved template '{template_id}'")
    
    def register_auto_tag(self, tag_id: str, resolved_value: Any) -> None:
        """
        Register a resolved AUTO tag value.
        
        Args:
            tag_id: AUTO tag identifier
            resolved_value: Resolved AUTO tag value
        """
        self.global_context['auto_tags'][tag_id] = resolved_value
        logger.debug(f"Registered resolved AUTO tag '{tag_id}': {type(resolved_value).__name__}")
        
        # Trigger resolution check
        self._trigger_resolution_check(f"auto_{tag_id}")
    
    def push_loop_context(self, loop_id: str, context: LoopContext) -> None:
        """
        Push a new loop context onto the stack.
        
        Args:
            loop_id: Loop identifier
            context: Loop context to push
        """
        self.loop_contexts[loop_id] = context
        self.active_loops.append(loop_id)
        logger.debug(f"Pushed loop context for '{loop_id}', iteration {context.iteration}")
    
    def pop_loop_context(self) -> Optional[LoopContext]:
        """
        Pop the current loop context from the stack.
        
        Returns:
            The popped loop context, or None if stack is empty
        """
        if not self.active_loops:
            return None
        
        loop_id = self.active_loops.pop()
        context = self.loop_contexts.pop(loop_id, None)
        if context:
            logger.debug(f"Popped loop context for '{loop_id}'")
        return context
    
    def get_current_loop_context(self) -> Optional[LoopContext]:
        """
        Get the current (top) loop context without removing it.
        
        Returns:
            Current loop context or None if not in a loop
        """
        if not self.active_loops:
            return None
        
        current_loop_id = self.active_loops[-1]
        return self.loop_contexts.get(current_loop_id)
    
    def get_available_context(self) -> Dict[str, Any]:
        """
        Get all available context for template rendering.
        
        Returns:
            Combined context from all sources
        """
        context = {}
        
        # Add global variables
        context.update(self.global_context['variables'])
        
        # Add task results
        context.update(self.global_context['results'])
        
        # Add resolved templates
        context.update(self.global_context['templates'])
        
        # Add resolved AUTO tags
        for tag_id, value in self.global_context['auto_tags'].items():
            context[f"auto_{tag_id}"] = value
        
        # Add current loop context if in a loop
        current_loop = self.get_current_loop_context()
        if current_loop:
            context.update(current_loop.to_dict())
            
            # Add nested loop contexts
            parent = current_loop.parent_context
            depth = 1
            while parent:
                parent_dict = parent.to_dict()
                # Prefix parent loop variables to avoid conflicts
                for key, value in parent_dict.items():
                    context[f"parent_{depth}_{key}"] = value
                parent = parent.parent_context
                depth += 1
        
        # Add system variables
        context['pipeline_id'] = self.pipeline_id
        context['execution_time'] = (datetime.now() - self.start_time).total_seconds()
        context['timestamp'] = datetime.now().isoformat()
        
        return context
    
    def add_unresolved_item(self, item: UnresolvedItem) -> None:
        """
        Add an item that needs resolution.
        
        Args:
            item: Unresolved item to track
        """
        self.unresolved_items.add(item)
        
        # Track dependencies
        if item.dependencies:
            self.dependency_graph[item.id] = item.dependencies
        
        logger.debug(f"Added unresolved item '{item.id}' ({item.item_type}) with {len(item.dependencies)} dependencies")
    
    def mark_item_resolved(self, item_id: str, resolved_value: Any) -> None:
        """
        Mark an unresolved item as resolved.
        
        Args:
            item_id: Item identifier
            resolved_value: Resolved value
        """
        # Find and update the item
        for item in self.unresolved_items:
            if item.id == item_id:
                item.mark_resolved(resolved_value)
                self.resolved_items[item_id] = resolved_value
                self.unresolved_items.discard(item)
                
                # Clean up dependency graph
                self.dependency_graph.pop(item_id, None)
                
                logger.info(f"Marked item '{item_id}' as resolved")
                
                # Trigger resolution check for dependent items
                self._trigger_resolution_check(item_id)
                break
    
    def mark_item_failed(self, item_id: str, error: str) -> None:
        """
        Mark an unresolved item as failed.
        
        Args:
            item_id: Item identifier
            error: Error message
        """
        for item in self.unresolved_items:
            if item.id == item_id:
                item.mark_failed(error)
                logger.error(f"Item '{item_id}' failed: {error}")
                break
    
    def mark_task_failed(self, task_id: str, error: str) -> None:
        """
        Mark a task as failed.
        
        Args:
            task_id: Task identifier
            error: Error message
        """
        self.failed_tasks[task_id] = error
        self.pending_tasks.discard(task_id)
        logger.error(f"Task '{task_id}' failed: {error}")
    
    def get_unresolved_by_type(self, item_type: str) -> List[UnresolvedItem]:
        """
        Get all unresolved items of a specific type.
        
        Args:
            item_type: Type to filter by
            
        Returns:
            List of unresolved items of the specified type
        """
        return [item for item in self.unresolved_items if item.item_type == item_type]
    
    def get_resolvable_items(self) -> List[UnresolvedItem]:
        """
        Get items that can potentially be resolved now.
        
        Returns:
            List of items whose dependencies are satisfied
        """
        available_context = set(self.get_available_context().keys())
        resolvable = []
        
        for item in self.unresolved_items:
            if item.status == ItemStatus.FAILED:
                continue
                
            # Check if all dependencies are available
            if item.dependencies.issubset(available_context):
                # Check if all context requirements are met
                if item.context_requirements.issubset(available_context):
                    resolvable.append(item)
        
        return resolvable
    
    def has_circular_dependencies(self) -> Tuple[bool, Optional[List[str]]]:
        """
        Check for circular dependencies in unresolved items.
        
        Returns:
            Tuple of (has_circular, cycle_path) where cycle_path is the circular dependency chain
        """
        # Build adjacency list
        graph = {}
        for item in self.unresolved_items:
            graph[item.id] = list(item.dependencies)
        
        # DFS to detect cycles
        visited = set()
        rec_stack = set()
        path = []
        cycle_found = []
        
        def has_cycle(node: str) -> bool:
            nonlocal cycle_found
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle - extract it
                    cycle_start = path.index(neighbor)
                    cycle_found = path[cycle_start:]
                    return True
            
            path.pop()
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                path = []
                if has_cycle(node):
                    return True, cycle_found
        
        return False, None
    
    def _trigger_resolution_check(self, resolved_item: str) -> None:
        """
        Trigger a check for items that might now be resolvable.
        
        Args:
            resolved_item: The item that was just resolved
        """
        # Find items that depend on the resolved item
        dependent_items = [
            item for item in self.unresolved_items
            if resolved_item in item.dependencies or resolved_item in item.context_requirements
        ]
        
        if dependent_items:
            logger.debug(f"Resolution of '{resolved_item}' may unblock {len(dependent_items)} items")
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current execution state.
        
        Returns:
            Dictionary with execution statistics and status
        """
        return {
            'pipeline_id': self.pipeline_id,
            'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'tasks_executed': len(self.executed_tasks),
            'tasks_pending': len(self.pending_tasks),
            'tasks_failed': len(self.failed_tasks),
            'items_unresolved': len(self.unresolved_items),
            'items_resolved': len(self.resolved_items),
            'active_loops': len(self.active_loops),
            'resolution_iterations': self.metadata.get('resolution_iterations', 0),
            'has_circular_deps': self.has_circular_dependencies()[0],
        }
    
    def export_state(self) -> Dict[str, Any]:
        """
        Export the complete execution state for persistence or debugging.
        
        Returns:
            Complete state as a dictionary
        """
        return {
            'pipeline_id': self.pipeline_id,
            'start_time': self.start_time.isoformat(),
            'global_context': copy.deepcopy(self.global_context),
            'loop_contexts': {
                loop_id: {
                    'loop_id': ctx.loop_id,
                    'iteration': ctx.iteration,
                    'item': ctx.item,
                    'index': ctx.index,
                    'variables': ctx.variables,
                }
                for loop_id, ctx in self.loop_contexts.items()
            },
            'active_loops': self.active_loops.copy(),
            'unresolved_items': [
                {
                    'id': item.id,
                    'content': item.content,
                    'type': item.item_type,
                    'status': item.status.value,
                    'dependencies': list(item.dependencies),
                    'attempts': item.resolution_attempts,
                }
                for item in self.unresolved_items
            ],
            'resolved_items': dict(self.resolved_items),
            'executed_tasks': list(self.executed_tasks),
            'pending_tasks': list(self.pending_tasks),
            'failed_tasks': dict(self.failed_tasks),
            'metadata': dict(self.metadata),
        }
    
    def import_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Import execution state from a dictionary.
        
        Args:
            state_dict: State dictionary to import
        """
        self.pipeline_id = state_dict.get('pipeline_id', 'imported')
        self.start_time = datetime.fromisoformat(state_dict['start_time'])
        self.global_context = copy.deepcopy(state_dict.get('global_context', {}))
        
        # Reconstruct loop contexts
        self.loop_contexts = {}
        for loop_id, ctx_data in state_dict.get('loop_contexts', {}).items():
            context = LoopContext(
                loop_id=ctx_data['loop_id'],
                iteration=ctx_data['iteration'],
                item=ctx_data.get('item'),
                index=ctx_data.get('index'),
                variables=ctx_data.get('variables', {})
            )
            self.loop_contexts[loop_id] = context
        
        self.active_loops = state_dict.get('active_loops', []).copy()
        self.resolved_items = dict(state_dict.get('resolved_items', {}))
        self.executed_tasks = set(state_dict.get('executed_tasks', []))
        self.pending_tasks = set(state_dict.get('pending_tasks', []))
        self.failed_tasks = dict(state_dict.get('failed_tasks', {}))
        self.metadata = dict(state_dict.get('metadata', {}))
        
        # Reconstruct unresolved items
        self.unresolved_items = set()
        for item_data in state_dict.get('unresolved_items', []):
            item = UnresolvedItem(
                id=item_data['id'],
                content=item_data['content'],
                item_type=item_data['type'],
                dependencies=set(item_data.get('dependencies', [])),
                status=ItemStatus(item_data.get('status', 'unresolved')),
                resolution_attempts=item_data.get('attempts', 0)
            )
            self.unresolved_items.add(item)
        
        logger.info(f"Imported execution state for pipeline: {self.pipeline_id}")