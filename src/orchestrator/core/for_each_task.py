"""ForEachTask type for runtime loop expansion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .task import Task, TaskStatus


@dataclass
class ForEachTask(Task):
    """
    Task representing an unexpanded for_each loop.
    
    This task type is created at compile time when a for_each loop
    contains AUTO tags or other runtime-dependent expressions.
    The actual expansion happens at runtime when step results are available.
    """
    
    # Loop configuration
    for_each_expr: str = ""  # The expression to evaluate (may contain AUTO tags)
    loop_steps: List[Dict[str, Any]] = field(default_factory=list)  # Steps to execute per item
    max_parallel: int = 1  # Maximum parallel iterations
    loop_var: str = "$item"  # Variable name for current item
    loop_name: Optional[str] = None  # Optional explicit loop name
    add_completion_task: bool = False  # Whether to add a completion task
    
    # Runtime state
    resolved_items: Optional[List[Any]] = None  # Items after resolving for_each_expr
    expanded_task_ids: List[str] = field(default_factory=list)  # IDs of expanded tasks
    
    def __post_init__(self):
        """Initialize ForEachTask with loop metadata."""
        super().__post_init__()
        
        # Mark this as a for_each task in metadata
        self.metadata["is_for_each_task"] = True
        self.metadata["requires_runtime_expansion"] = True
        
        # Store loop configuration in metadata for debugging
        self.metadata["loop_config"] = {
            "for_each_expr": self.for_each_expr,
            "max_parallel": self.max_parallel,
            "loop_var": self.loop_var,
            "num_steps": len(self.loop_steps),
            "add_completion": self.add_completion_task
        }
    
    def is_ready_for_expansion(self, available_results: Dict[str, Any]) -> bool:
        """
        Check if this task is ready for expansion.
        
        Args:
            available_results: Results from completed tasks
            
        Returns:
            True if all dependencies are satisfied and expansion can proceed
        """
        # Check if all dependencies have results
        for dep in self.dependencies:
            if dep not in available_results:
                return False
        return True
    
    def mark_expanded(self, task_ids: List[str], items: List[Any]) -> None:
        """
        Mark this task as expanded with the generated task IDs.
        
        Args:
            task_ids: IDs of the expanded tasks
            items: The resolved items being iterated over
        """
        self.expanded_task_ids = task_ids
        self.resolved_items = items
        self.status = TaskStatus.COMPLETED  # Mark as completed since expansion is done
        self.result = {
            "type": "for_each_expansion",
            "num_items": len(items),
            "num_tasks": len(task_ids),
            "task_ids": task_ids
        }
    
    def get_expansion_info(self) -> Dict[str, Any]:
        """
        Get information about the expansion for logging/debugging.
        
        Returns:
            Dictionary with expansion details
        """
        return {
            "task_id": self.id,
            "for_each_expr": self.for_each_expr,
            "resolved_items": self.resolved_items,
            "expanded_task_ids": self.expanded_task_ids,
            "max_parallel": self.max_parallel,
            "status": self.status.value
        }