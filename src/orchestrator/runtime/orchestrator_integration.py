"""
Integration module for runtime dependency resolution with existing orchestrator.

This module bridges the new runtime resolution system (Issue #211) with the
existing orchestrator implementation, providing backward compatibility while
enabling progressive template resolution.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass

from .execution_state import PipelineExecutionState, UnresolvedItem, LoopContext
from .dependency_resolver import DependencyResolver
from .loop_expander import LoopExpander, LoopTask, ExpandedTask

logger = logging.getLogger(__name__)


class RuntimeResolutionIntegration:
    """
    Integrates runtime resolution system with existing orchestrator.
    
    This class provides methods to:
    - Convert existing for_each and while tasks to LoopTask format
    - Register pipeline context and results with execution state
    - Resolve templates progressively during execution
    - Expand loops at runtime when dependencies are satisfied
    """
    
    def __init__(self, pipeline_id: str = "default"):
        """
        Initialize runtime resolution integration.
        
        Args:
            pipeline_id: Pipeline identifier
        """
        self.state = PipelineExecutionState(pipeline_id)
        self.resolver = DependencyResolver(self.state)
        self.expander = LoopExpander(self.state, self.resolver)
        
        logger.info(f"Initialized RuntimeResolutionIntegration for pipeline {pipeline_id}")
    
    def register_pipeline_context(self, context: Dict[str, Any]) -> None:
        """
        Register pipeline context (parameters, variables, etc.).
        
        Args:
            context: Pipeline context dictionary
        """
        # Register pipeline parameters
        if "parameters" in context:
            for key, value in context["parameters"].items():
                self.state.register_variable(key, value)
        
        # Register other context variables
        for key, value in context.items():
            if key not in ["parameters", "tasks", "metadata"]:
                self.state.register_variable(key, value)
        
        logger.info(f"Registered {len(context)} context items")
    
    def register_task_result(self, task_id: str, result: Any) -> None:
        """
        Register a task execution result.
        
        Args:
            task_id: Task identifier
            result: Task execution result
        """
        self.state.register_result(task_id, result)
        
        # Also register specific result fields for easier access
        if isinstance(result, dict):
            # Register 'result' field if present
            if 'result' in result:
                self.state.register_result(f"{task_id}_result", result['result'])
            
            # Register 'value' field if present
            if 'value' in result:
                self.state.register_result(f"{task_id}_value", result['value'])
        
        logger.debug(f"Registered result for task {task_id}")
    
    def convert_for_each_task(self, for_each_task) -> LoopTask:
        """
        Convert existing ForEachTask to LoopTask format.
        
        Args:
            for_each_task: ForEachTask instance
            
        Returns:
            Converted LoopTask
        """
        loop_task = LoopTask(
            id=for_each_task.id,
            loop_type="for_each",
            iterator_expr=for_each_task.for_each_expr,
            loop_steps=for_each_task.loop_steps,
            max_parallel=getattr(for_each_task, 'max_parallel', 1),
            dependencies=list(for_each_task.dependencies),
            metadata=for_each_task.metadata
        )
        
        self.expander.register_loop(loop_task)
        return loop_task
    
    def convert_while_task(self, while_task) -> LoopTask:
        """
        Convert existing while loop task to LoopTask format.
        
        Args:
            while_task: While loop task
            
        Returns:
            Converted LoopTask
        """
        loop_task = LoopTask(
            id=while_task.id,
            loop_type="while",
            condition_expr=while_task.metadata.get('condition', 'true'),
            loop_steps=while_task.metadata.get('steps', []),
            max_iterations=while_task.metadata.get('max_iterations', 100),
            dependencies=list(while_task.dependencies),
            metadata=while_task.metadata
        )
        
        self.expander.register_loop(loop_task)
        return loop_task
    
    def add_unresolved_template(self, template_id: str, template_str: str) -> None:
        """
        Add a template that needs resolution.
        
        Args:
            template_id: Template identifier
            template_str: Template string to resolve
        """
        deps = self.resolver.extract_dependencies(template_str)
        
        item = UnresolvedItem(
            id=template_id,
            content=template_str,
            item_type="template",
            dependencies=deps
        )
        
        self.state.add_unresolved_item(item)
        logger.debug(f"Added unresolved template {template_id} with {len(deps)} dependencies")
    
    def resolve_pending_items(self) -> Dict[str, Any]:
        """
        Attempt to resolve all pending items.
        
        Returns:
            Dictionary with resolution results
        """
        result = self.resolver.resolve_all_pending()
        
        return {
            "success": result.success,
            "resolved": result.resolved_items,
            "failed": result.failed_items,
            "unresolved": result.unresolved_items,
            "iterations": result.iterations,
            "error": result.error_message
        }
    
    def expand_ready_loops(self) -> List[Dict[str, Any]]:
        """
        Expand all loops that have satisfied dependencies.
        
        Returns:
            List of expanded task dictionaries
        """
        expanded_tasks = self.expander.expand_all_ready()
        
        # Convert ExpandedTask objects to dictionaries
        task_dicts = [task.to_dict() for task in expanded_tasks]
        
        logger.info(f"Expanded {len(task_dicts)} tasks from ready loops")
        return task_dicts
    
    def can_expand_loop(self, loop_id: str) -> bool:
        """
        Check if a specific loop can be expanded.
        
        Args:
            loop_id: Loop identifier
            
        Returns:
            True if loop can be expanded
        """
        loop = self.expander.get_loop(loop_id)
        if not loop:
            return False
        
        return self.expander.can_expand(loop)
    
    def expand_specific_loop(self, loop_id: str) -> List[Dict[str, Any]]:
        """
        Expand a specific loop.
        
        Args:
            loop_id: Loop identifier
            
        Returns:
            List of expanded task dictionaries
        """
        loop = self.expander.get_loop(loop_id)
        if not loop:
            logger.warning(f"Loop {loop_id} not found")
            return []
        
        if loop.loop_type == "for_each":
            expanded = self.expander.expand_for_each(loop)
        elif loop.loop_type == "while":
            expanded = self.expander.expand_while_iteration(loop)
        else:
            logger.error(f"Unknown loop type: {loop.loop_type}")
            return []
        
        # Convert to dictionaries
        return [task.to_dict() for task in expanded]
    
    def resolve_template_with_context(self, template_str: str, 
                                     additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Resolve a template string with current context.
        
        Args:
            template_str: Template to resolve
            additional_context: Additional context to use
            
        Returns:
            Resolved string
        """
        try:
            if additional_context:
                # Temporarily add additional context
                for key, value in additional_context.items():
                    self.state.register_variable(f"_temp_{key}", value)
                
                # Create merged context
                context = {**self.state.get_available_context(), **additional_context}
                result = self.resolver.resolve_template(template_str, context)
                
                # Clean up temporary variables
                for key in additional_context:
                    self.state.global_context['variables'].pop(f"_temp_{key}", None)
                
                return result
            else:
                return self.resolver.resolve_template(template_str)
                
        except Exception as e:
            logger.warning(f"Failed to resolve template: {e}")
            return template_str
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get summary of execution state.
        
        Returns:
            Execution summary dictionary
        """
        summary = self.state.get_execution_summary()
        
        # Add loop information
        summary['loops'] = {
            'registered': len(self.expander.active_loops),
            'completed': sum(1 for l in self.expander.active_loops.values() if l.completed),
            'expandable': len(self.expander.get_expandable_loops())
        }
        
        return summary
    
    def handle_loop_context_for_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get loop context for a task if it's part of a loop.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Loop context dictionary or None
        """
        # Check if task_id matches loop task pattern (loop_id_iteration_step)
        parts = task_id.split('_')
        if len(parts) < 3:
            return None
        
        # Try to extract loop_id and iteration
        for loop_id in self.expander.active_loops:
            if task_id.startswith(f"{loop_id}_"):
                # This task is part of a loop
                loop_ctx = self.state.get_current_loop_context()
                if loop_ctx:
                    return loop_ctx.to_dict()
        
        return None
    
    def register_auto_tag(self, tag_id: str, tag_content: str, resolved_value: Any) -> None:
        """
        Register a resolved AUTO tag.
        
        Args:
            tag_id: Tag identifier
            tag_content: Original tag content
            resolved_value: Resolved value
        """
        self.state.register_auto_tag(tag_id, resolved_value)
        logger.debug(f"Registered AUTO tag {tag_id}")