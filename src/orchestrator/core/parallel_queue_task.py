"""ParallelQueueTask model for dynamic parallel task execution."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union
from enum import Enum

from .task import Task, TaskStatus


class ParallelQueueStatus(Enum):
    """Status of parallel queue execution."""
    
    INITIALIZING = "initializing"
    GENERATING_QUEUE = "generating_queue"
    EXECUTING_PARALLEL = "executing_parallel"
    EVALUATING_CONDITIONS = "evaluating_conditions"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ParallelQueueStats:
    """Statistics for parallel queue execution."""
    
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    concurrent_executions: int = 0
    max_concurrent_reached: int = 0
    
    # Timing statistics
    queue_generation_time: float = 0.0
    total_execution_time: float = 0.0
    average_item_time: float = 0.0
    
    # Resource usage
    memory_peak_mb: float = 0.0
    active_tool_instances: Dict[str, int] = field(default_factory=dict)
    
    def update_item_completion(self, execution_time: float) -> None:
        """Update statistics when an item completes."""
        self.completed_items += 1
        
        # Update average execution time
        if self.completed_items == 1:
            self.average_item_time = execution_time
        else:
            # Running average
            total_time = self.average_item_time * (self.completed_items - 1) + execution_time
            self.average_item_time = total_time / self.completed_items
    
    def get_completion_rate(self) -> float:
        """Get completion rate as percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.completed_items / self.total_items) * 100
    
    def get_failure_rate(self) -> float:
        """Get failure rate as percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.failed_items / self.total_items) * 100


@dataclass
class ParallelSubtask:
    """Individual subtask within a parallel queue."""
    
    id: str
    queue_index: int
    item: Any
    task: Task
    status: TaskStatus = TaskStatus.PENDING
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[Exception] = None
    
    @property
    def execution_time(self) -> Optional[float]:
        """Get subtask execution time."""
        if self.started_at is None:
            return None
        end_time = self.completed_at or time.time()
        return end_time - self.started_at
    
    def start(self) -> None:
        """Mark subtask as started."""
        self.status = TaskStatus.RUNNING
        self.started_at = time.time()
    
    def complete(self, result: Any = None) -> None:
        """Mark subtask as completed."""
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = time.time()
    
    def fail(self, error: Exception) -> None:
        """Mark subtask as failed."""
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = time.time()


@dataclass
class ParallelQueueTask(Task):
    """
    Task that generates and executes subtasks in parallel based on a dynamic queue.
    
    This extends the base Task class to support parallel execution with:
    - Dynamic queue generation using AUTO tag resolution
    - Configurable concurrency limits
    - Until/while condition evaluation across parallel tasks
    - Tool resource sharing and management
    - Comprehensive progress tracking and statistics
    """
    
    # Queue generation
    on: str = ""  # Expression to generate queue items (supports AUTO tags)
    
    # Parallel execution configuration
    max_parallel: int = 10
    action_loop: List[Dict[str, Any]] = field(default_factory=list)
    tool: Optional[str] = None
    
    # Condition integration (Issue 189)
    until_condition: Optional[str] = None
    while_condition: Optional[str] = None
    condition_check_interval: float = 1.0  # Seconds between condition checks
    
    # Runtime state
    queue_status: ParallelQueueStatus = ParallelQueueStatus.INITIALIZING
    queue_items: List[Any] = field(default_factory=list)
    subtasks: Dict[str, ParallelSubtask] = field(default_factory=dict)
    active_subtasks: Dict[str, str] = field(default_factory=dict)  # item_id -> subtask_id
    
    # Completion tracking
    completed_items: Set[int] = field(default_factory=set)
    failed_items: Set[int] = field(default_factory=set)
    skipped_items: Set[int] = field(default_factory=set)
    
    # Performance and resource tracking
    stats: ParallelQueueStats = field(default_factory=ParallelQueueStats)
    semaphore: Optional[asyncio.Semaphore] = field(default=None, init=False, repr=False)
    
    # Loop context integration
    queue_loop_context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Additional validation for parallel queue tasks."""
        super().__post_init__()
        
        if not self.on:
            raise ValueError("ParallelQueueTask must have 'on' expression for queue generation")
        
        if self.max_parallel <= 0:
            raise ValueError("max_parallel must be positive")
        
        if not self.action_loop:
            raise ValueError("ParallelQueueTask must have action_loop defined")
        
        # Initialize semaphore for concurrency control
        self.semaphore = asyncio.Semaphore(self.max_parallel)
    
    def generate_subtask_id(self, queue_index: int, action_index: int = 0) -> str:
        """Generate unique ID for subtask."""
        return f"{self.id}_parallel_{queue_index}_{action_index}"
    
    def add_queue_item(self, item: Any) -> int:
        """Add item to queue and return its index."""
        queue_index = len(self.queue_items)
        self.queue_items.append(item)
        self.stats.total_items += 1
        return queue_index
    
    def create_subtask(self, queue_index: int, item: Any, action_def: Dict[str, Any]) -> ParallelSubtask:
        """Create a subtask for a queue item and action."""
        subtask_id = self.generate_subtask_id(queue_index)
        
        # Create the underlying task
        task_def = action_def.copy()
        task_def["id"] = subtask_id
        task_def["name"] = f"{self.name} - Item {queue_index}"
        
        # Add metadata
        task_def.setdefault("metadata", {})
        task_def["metadata"].update({
            "parent_parallel_queue": self.id,
            "queue_index": queue_index,
            "queue_item": item,
            "is_parallel_subtask": True,
            "tool": self.tool,  # Inherit tool from parent
        })
        
        # Create task instance
        task = Task(
            id=subtask_id,
            name=task_def["name"],
            action=task_def["action"],
            parameters=task_def.get("parameters", {}),
            dependencies=task_def.get("dependencies", []),
            metadata=task_def["metadata"],
            timeout=task_def.get("timeout", self.timeout),
            max_retries=task_def.get("max_retries", self.max_retries),
        )
        
        # Create parallel subtask wrapper
        subtask = ParallelSubtask(
            id=subtask_id,
            queue_index=queue_index,
            item=item,
            task=task
        )
        
        self.subtasks[subtask_id] = subtask
        return subtask
    
    def get_active_subtasks(self) -> List[ParallelSubtask]:
        """Get list of currently running subtasks."""
        return [
            subtask for subtask in self.subtasks.values()
            if subtask.status == TaskStatus.RUNNING
        ]
    
    def get_completed_subtasks(self) -> List[ParallelSubtask]:
        """Get list of completed subtasks."""
        return [
            subtask for subtask in self.subtasks.values()
            if subtask.status == TaskStatus.COMPLETED
        ]
    
    def get_failed_subtasks(self) -> List[ParallelSubtask]:
        """Get list of failed subtasks."""
        return [
            subtask for subtask in self.subtasks.values()
            if subtask.status == TaskStatus.FAILED
        ]
    
    def update_concurrency_stats(self) -> None:
        """Update concurrency statistics."""
        active_count = len(self.get_active_subtasks())
        self.stats.concurrent_executions = active_count
        if active_count > self.stats.max_concurrent_reached:
            self.stats.max_concurrent_reached = active_count
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get comprehensive progress summary."""
        return {
            "status": self.queue_status.value,
            "total_items": len(self.queue_items),
            "completed": len(self.completed_items),
            "failed": len(self.failed_items),
            "skipped": len(self.skipped_items),
            "active": len(self.get_active_subtasks()),
            "completion_rate": self.stats.get_completion_rate(),
            "failure_rate": self.stats.get_failure_rate(),
            "concurrent_executions": self.stats.concurrent_executions,
            "max_concurrent_reached": self.stats.max_concurrent_reached,
            "average_item_time": self.stats.average_item_time,
            "has_until_condition": self.until_condition is not None,
            "has_while_condition": self.while_condition is not None,
        }
    
    def should_continue_execution(self) -> bool:
        """Check if parallel execution should continue based on current state."""
        # Check if we have more items to process
        pending_items = len(self.queue_items) - len(self.completed_items) - len(self.failed_items) - len(self.skipped_items)
        active_items = len(self.get_active_subtasks())
        
        return (pending_items > 0 or active_items > 0) and self.queue_status not in {
            ParallelQueueStatus.COMPLETED,
            ParallelQueueStatus.FAILED,
            ParallelQueueStatus.CANCELLED
        }
    
    def get_context_variables(self, queue_index: int) -> Dict[str, Any]:
        """Get loop context variables for a specific queue item."""
        if queue_index >= len(self.queue_items):
            return {}
        
        # Provide both $ prefixed and non-prefixed versions for template compatibility
        variables = {
            # $ prefixed versions (original)
            "$item": self.queue_items[queue_index],
            "$index": queue_index,
            "$queue": self.queue_items,
            "$queue_size": len(self.queue_items),
            "$is_first": queue_index == 0,
            "$is_last": queue_index == len(self.queue_items) - 1,
            "$parallel_queue_id": self.id,
            "$parent_task": self.id,
            
            # Non-prefixed versions for template engine compatibility
            "item": self.queue_items[queue_index],
            "index": queue_index,
            "queue": self.queue_items,
            "queue_size": len(self.queue_items),
            "is_first": queue_index == 0,
            "is_last": queue_index == len(self.queue_items) - 1,
            "parallel_queue_id": self.id,
            "parent_task": self.id,
        }
        
        return variables
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with parallel queue specific fields."""
        base_dict = super().to_dict()
        
        # Add parallel queue specific fields
        base_dict.update({
            "queue_type": "parallel_queue",
            "on": self.on,
            "max_parallel": self.max_parallel,
            "action_loop": self.action_loop,
            "tool": self.tool,
            "until_condition": self.until_condition,
            "while_condition": self.while_condition,
            "queue_status": self.queue_status.value,
            "queue_items": self.queue_items,
            "progress": self.get_progress_summary(),
            "stats": {
                "total_items": self.stats.total_items,
                "completed_items": self.stats.completed_items,
                "failed_items": self.stats.failed_items,
                "completion_rate": self.stats.get_completion_rate(),
                "failure_rate": self.stats.get_failure_rate(),
                "queue_generation_time": self.stats.queue_generation_time,
                "total_execution_time": self.stats.total_execution_time,
                "average_item_time": self.stats.average_item_time,
            }
        })
        
        return base_dict
    
    @classmethod
    def from_task_definition(cls, task_def: Dict[str, Any]) -> ParallelQueueTask:
        """Create ParallelQueueTask from YAML task definition."""
        # Extract parallel queue specific fields
        # Handle multiple syntaxes:
        # 1. Direct: create_parallel_queue: { on: ..., action_loop: ... }
        # 2. Action: action: create_parallel_queue, parameters: { on: ..., action_loop: ... }
        # 3. Nested task: create_parallel_queue: { on: ..., task: { action_loop: ... } }
        parallel_config = task_def.get("create_parallel_queue", {})
        
        # If not found at top level, check parameters for action syntax
        if not parallel_config and task_def.get("action") == "create_parallel_queue":
            parallel_config = task_def.get("parameters", {})
        
        if not parallel_config:
            raise ValueError("Task definition must contain 'create_parallel_queue' section or use 'action: create_parallel_queue' with parameters")
        
        # Extract action_loop - check both direct and nested under 'task'
        action_loop = parallel_config.get("action_loop", [])
        if not action_loop and "task" in parallel_config:
            # Handle nested task structure: task: { action_loop: [...] }
            task_config = parallel_config["task"]
            if isinstance(task_config, dict):
                action_loop = task_config.get("action_loop", [])
        
        # Create the task
        task = cls(
            id=task_def["id"],
            name=task_def.get("name", task_def["id"]),
            action="create_parallel_queue",
            parameters=task_def.get("parameters", {}),
            dependencies=task_def.get("dependencies", []),
            metadata=task_def.get("metadata", {}),
            timeout=task_def.get("timeout"),
            max_retries=task_def.get("max_retries", 3),
            
            # Parallel queue specific
            # Handle YAML parsing issue where "on" becomes boolean True
            on=parallel_config.get("on") or parallel_config.get(True, ""),
            max_parallel=parallel_config.get("max_parallel", 10),
            action_loop=action_loop,
            tool=parallel_config.get("tool"),
            until_condition=parallel_config.get("until"),
            while_condition=parallel_config.get("while"),
        )
        
        return task