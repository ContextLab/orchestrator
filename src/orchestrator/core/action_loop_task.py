"""Action loop task for sequential task iteration with termination conditions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .task import Task, TaskStatus


@dataclass
class ActionLoopTask(Task):
    """Task that executes actions in a loop until condition is met.
    
    This extends the base Task class to support iterative execution of action
    sequences with termination conditions and tool integration.
    """
    
    # Loop definition
    action_loop: List[Dict[str, Any]] = field(default_factory=list)
    until: Optional[str] = None  # Termination condition (may contain AUTO tags)
    while_condition: Optional[str] = None  # Alternative: continue condition
    
    # Loop control
    max_iterations: int = 100  # Safety limit
    break_on_error: bool = False  # Stop loop on action failure
    iteration_timeout: Optional[int] = None  # Timeout per iteration
    
    # Runtime state
    current_iteration: int = 0
    loop_results: List[Dict[str, Any]] = field(default_factory=list)
    terminated_by: Optional[str] = None  # 'condition', 'max_iterations', 'error', 'timeout'
    iteration_start_time: Optional[float] = None
    
    # Tool execution tracking
    tool_executions: Dict[str, int] = field(default_factory=dict)
    tool_errors: Dict[str, List[str]] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate action loop task after initialization."""
        super().__post_init__()
        
        # Validate action_loop
        if not self.action_loop:
            raise ValueError("action_loop cannot be empty")
        
        if not isinstance(self.action_loop, list):
            raise ValueError("action_loop must be a list of actions")
        
        # Validate termination conditions
        if not self.until and not self.while_condition:
            raise ValueError("Either 'until' or 'while_condition' must be specified")
        
        if self.until and self.while_condition:
            raise ValueError("Cannot specify both 'until' and 'while_condition'")
        
        # Validate max_iterations
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        
        # Validate action definitions
        for i, action_def in enumerate(self.action_loop):
            if not isinstance(action_def, dict):
                raise ValueError(f"Action {i} must be a dictionary")
            
            if "action" not in action_def:
                raise ValueError(f"Action {i} must have 'action' field")
    
    def start_iteration(self) -> None:
        """Mark start of current iteration."""
        self.iteration_start_time = time.time()
        
    def complete_iteration(self, iteration_results: Dict[str, Any]) -> None:
        """Mark completion of current iteration and store results."""
        self.loop_results.append(iteration_results)
        self.current_iteration += 1
        self.iteration_start_time = None
    
    def get_iteration_duration(self) -> Optional[float]:
        """Get duration of current iteration in seconds."""
        if self.iteration_start_time is None:
            return None
        return time.time() - self.iteration_start_time
    
    def record_tool_execution(self, tool_name: str, success: bool = True, error: str = None) -> None:
        """Record tool execution statistics."""
        # Update execution count
        if tool_name not in self.tool_executions:
            self.tool_executions[tool_name] = 0
        self.tool_executions[tool_name] += 1
        
        # Record errors
        if not success and error:
            if tool_name not in self.tool_errors:
                self.tool_errors[tool_name] = []
            self.tool_errors[tool_name].append(error)
    
    def get_loop_statistics(self) -> Dict[str, Any]:
        """Get comprehensive loop execution statistics."""
        total_actions = len(self.action_loop) * self.current_iteration
        total_tool_executions = sum(self.tool_executions.values())
        total_errors = sum(len(errors) for errors in self.tool_errors.values())
        
        return {
            "iterations_completed": self.current_iteration,
            "max_iterations": self.max_iterations,
            "terminated_by": self.terminated_by,
            "total_actions_executed": total_actions,
            "total_tool_executions": total_tool_executions,
            "total_errors": total_errors,
            "tool_usage": dict(self.tool_executions),
            "tool_error_summary": {
                tool: len(errors) for tool, errors in self.tool_errors.items()
            },
            "success_rate": (
                (total_tool_executions - total_errors) / total_tool_executions
                if total_tool_executions > 0 else 1.0
            )
        }
    
    def has_termination_condition(self) -> bool:
        """Check if task has a termination condition."""
        return bool(self.until or self.while_condition)
    
    def get_termination_condition(self) -> str:
        """Get the termination condition string."""
        if self.until:
            return self.until
        elif self.while_condition:
            return self.while_condition
        return ""
    
    def is_until_condition(self) -> bool:
        """Check if using 'until' termination condition."""
        return bool(self.until)
    
    def is_while_condition(self) -> bool:
        """Check if using 'while' continuation condition."""
        return bool(self.while_condition)
    
    def should_check_timeout(self) -> bool:
        """Check if current iteration should be checked for timeout."""
        return (
            self.iteration_timeout is not None 
            and self.iteration_start_time is not None
            and self.get_iteration_duration() > self.iteration_timeout
        )
    
    def can_continue_iteration(self) -> bool:
        """Check if loop can continue to next iteration."""
        if self.current_iteration >= self.max_iterations:
            return False
        
        if self.terminated_by is not None:
            return False
        
        if self.should_check_timeout():
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert action loop task to dictionary representation."""
        base_dict = super().to_dict()
        
        # Add action loop specific fields
        base_dict.update({
            "action_loop": self.action_loop,
            "until": self.until,
            "while_condition": self.while_condition,
            "max_iterations": self.max_iterations,
            "break_on_error": self.break_on_error,
            "iteration_timeout": self.iteration_timeout,
            "current_iteration": self.current_iteration,
            "loop_results": self.loop_results,
            "terminated_by": self.terminated_by,
            "tool_executions": self.tool_executions,
            "tool_errors": self.tool_errors,
            "loop_statistics": self.get_loop_statistics()
        })
        
        return base_dict
    
    @classmethod
    def from_task_definition(cls, task_def: Dict[str, Any]) -> ActionLoopTask:
        """Create ActionLoopTask from task definition dictionary."""
        # Make a copy to avoid modifying the original
        task_def = task_def.copy()
        
        # Extract action loop specific fields
        action_loop = task_def.pop("action_loop", [])
        until = task_def.pop("until", None)
        while_condition = task_def.pop("while_condition", None)
        max_iterations = task_def.pop("max_iterations", 100)
        break_on_error = task_def.pop("break_on_error", False)
        iteration_timeout = task_def.pop("iteration_timeout", None)
        
        # Handle loop_results and other runtime state if present
        current_iteration = task_def.pop("current_iteration", 0)
        loop_results = task_def.pop("loop_results", [])
        terminated_by = task_def.pop("terminated_by", None)
        tool_executions = task_def.pop("tool_executions", {})
        tool_errors = task_def.pop("tool_errors", {})
        
        # Remove computed fields that shouldn't be in constructor
        task_def.pop("loop_statistics", None)
        
        # Move non-Task fields to metadata before creating Task
        metadata = task_def.get("metadata", {})
        special_fields = ["requires_model", "produces", "location", "tool", "on_error"]
        for field in special_fields:
            if field in task_def:
                metadata[field] = task_def.pop(field)
        task_def["metadata"] = metadata
        
        # Ensure required fields are present
        if "action" not in task_def:
            task_def["action"] = "action_loop"
        if "name" not in task_def:
            task_def["name"] = task_def.get("id", "action_loop_task")
        
        # Create base task from remaining fields
        base_task = Task.from_dict(task_def)
        
        # Create ActionLoopTask with all fields
        action_loop_task = cls(
            id=base_task.id,
            name=base_task.name,
            action=base_task.action,
            parameters=base_task.parameters,
            dependencies=base_task.dependencies,
            status=base_task.status,
            result=base_task.result,
            error=base_task.error,
            metadata=base_task.metadata,
            timeout=base_task.timeout,
            max_retries=base_task.max_retries,
            retry_count=base_task.retry_count,
            created_at=base_task.created_at,
            started_at=base_task.started_at,
            completed_at=base_task.completed_at,
            template_metadata=base_task.template_metadata,
            rendered_parameters=base_task.rendered_parameters,
            dependencies_satisfied=base_task.dependencies_satisfied,
            in_loop_context=base_task.in_loop_context,
            loop_context=base_task.loop_context,
            output_metadata=base_task.output_metadata,
            output_info=base_task.output_info,
            # ActionLoopTask specific fields
            action_loop=action_loop,
            until=until,
            while_condition=while_condition,
            max_iterations=max_iterations,
            break_on_error=break_on_error,
            iteration_timeout=iteration_timeout,
            current_iteration=current_iteration,
            loop_results=loop_results,
            terminated_by=terminated_by,
            tool_executions=tool_executions,
            tool_errors=tool_errors
        )
        
        return action_loop_task
    
    def __repr__(self) -> str:
        """String representation of action loop task."""
        condition = self.until or self.while_condition or "no condition"
        return (
            f"ActionLoopTask(id='{self.id}', "
            f"actions={len(self.action_loop)}, "
            f"condition='{condition[:50]}...', "
            f"iteration={self.current_iteration}/{self.max_iterations}, "
            f"status={self.status.value})"
        )