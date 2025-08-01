"""Core task abstraction for the orchestrator framework."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .template_metadata import TemplateMetadata


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """
    Core task abstraction for the orchestrator.

    A task represents a single unit of work in a pipeline with dependencies,
    parameters, and execution metadata.
    """

    id: str
    name: str
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[int] = None
    max_retries: int = 3
    retry_count: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Template tracking
    template_metadata: Dict[str, TemplateMetadata] = field(default_factory=dict)
    rendered_parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies_satisfied: bool = False
    
    # Loop context tracking
    in_loop_context: bool = False
    loop_context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate task after initialization."""
        if not self.id:
            raise ValueError("Task ID cannot be empty")
        if not self.name:
            raise ValueError("Task name cannot be empty")
        if not self.action:
            raise ValueError("Task action cannot be empty")
        if self.max_retries < 0:
            raise ValueError("Max retries must be non-negative")

        # Validate dependencies
        if self.id in self.dependencies:
            raise ValueError(f"Task {self.id} cannot depend on itself")

    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """
        Check if all dependencies are satisfied.

        Args:
            completed_tasks: Set of completed task IDs

        Returns:
            True if all dependencies are satisfied, False otherwise
        """
        return all(dep in completed_tasks for dep in self.dependencies)

    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries and self.status == TaskStatus.FAILED

    def start(self) -> None:
        """Mark task as started."""
        self.status = TaskStatus.RUNNING
        self.started_at = time.time()

    def complete(self, result: Any = None) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = time.time()
        self.error = None

    def fail(self, error: Exception) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = time.time()
        self.retry_count += 1

    def skip(self, reason: str = "") -> None:
        """Mark task as skipped."""
        self.status = TaskStatus.SKIPPED
        self.completed_at = time.time()
        if reason:
            self.metadata["skip_reason"] = reason

    def reset(self) -> None:
        """Reset task to pending state."""
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.started_at = None
        self.completed_at = None
    
    def check_dependencies_satisfied(self, completed_steps: Set[str], 
                                     available_contexts: Optional[Set[str]] = None) -> bool:
        """
        Check if all dependencies for templates are satisfied.
        
        Args:
            completed_steps: Set of step IDs that have completed
            available_contexts: Set of available special contexts (e.g., "$item")
            
        Returns:
            True if all template dependencies are satisfied
        """
        if available_contexts is None:
            available_contexts = set()
        
        # Add current loop context if in a loop
        if self.in_loop_context:
            available_contexts.update(self.loop_context.keys())
        
        # Check each template's dependencies
        for param_name, metadata in self.template_metadata.items():
            if not metadata.can_render_with_context(completed_steps, available_contexts):
                return False
        
        self.dependencies_satisfied = True
        return True
    
    def get_missing_dependencies(self, completed_steps: Set[str]) -> Dict[str, Set[str]]:
        """
        Get missing dependencies for each template parameter.
        
        Returns:
            Dict mapping parameter names to their missing dependencies
        """
        missing = {}
        for param_name, metadata in self.template_metadata.items():
            missing_deps = metadata.get_missing_dependencies(completed_steps)
            if missing_deps:
                missing[param_name] = missing_deps
        return missing

    @property
    def execution_time(self) -> Optional[float]:
        """Get task execution time in seconds."""
        if self.started_at is None:
            return None
        end_time = self.completed_at or time.time()
        return end_time - self.started_at

    @property
    def is_terminal(self) -> bool:
        """Check if task is in a terminal state."""
        return self.status in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.SKIPPED,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "action": self.action,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "result": self.result,
            "error": str(self.error) if self.error else None,
            "metadata": self.metadata,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "execution_time": self.execution_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Task:
        """Create task from dictionary representation."""
        # Convert status back to enum
        if "status" in data:
            data["status"] = TaskStatus(data["status"])

        # Handle error
        if "error" in data and data["error"]:
            data["error"] = Exception(data["error"])

        # Remove computed properties
        data.pop("execution_time", None)

        return cls(**data)

    def __repr__(self) -> str:
        """String representation of task."""
        return f"Task(id='{self.id}', name='{self.name}', status={self.status.value})"

    def __eq__(self, other: object) -> bool:
        """Check equality based on task ID."""
        if not isinstance(other, Task):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash based on task ID."""
        return hash(self.id)
