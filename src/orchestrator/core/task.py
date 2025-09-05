"""Core task abstraction for the orchestrator framework."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .template_metadata import TemplateMetadata
from .output_metadata import OutputMetadata, OutputInfo


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
    
    # Output tracking
    output_metadata: Optional[OutputMetadata] = None
    output_info: Optional[OutputInfo] = None

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

    # Output-related properties and methods
    @property
    def produces(self) -> Optional[str]:
        """Get the output type this task produces."""
        return self.output_metadata.produces if self.output_metadata else None
    
    @property
    def location(self) -> Optional[str]:
        """Get the output location (resolved if output_info exists, template if not)."""
        if self.output_info and self.output_info.location:
            return self.output_info.location
        return self.output_metadata.location if self.output_metadata else None
    
    @property
    def output_format(self) -> Optional[str]:
        """Get the output format/MIME type."""
        if self.output_info and self.output_info.format:
            return self.output_info.format
        return self.output_metadata.format if self.output_metadata else None
    
    def has_output_metadata(self) -> bool:
        """Check if task has output metadata defined."""
        return self.output_metadata is not None
    
    def has_output_info(self) -> bool:
        """Check if task has actual output information."""
        return self.output_info is not None
    
    def set_output_metadata(self, produces: Optional[str] = None, 
                           location: Optional[str] = None,
                           format: Optional[str] = None,
                           **kwargs) -> None:
        """Set output metadata for this task."""
        from .output_metadata import create_output_metadata
        self.output_metadata = create_output_metadata(
            produces=produces,
            location=location,
            format=format,
            **kwargs
        )
    
    def register_output(self, result: Any, location: Optional[str] = None,
                       format: Optional[str] = None) -> OutputInfo:
        """Register the actual output after task execution."""
        from .output_metadata import OutputFormatDetector
        
        # Detect format if not provided
        if not format:
            format = OutputFormatDetector.detect_from_content(result, location)
        
        # Create output info
        self.output_info = OutputInfo(
            task_id=self.id,
            output_type=self.produces,
            location=location,
            format=format,
            result=result
        )
        
        # Validate against metadata if available
        if self.output_metadata:
            self.output_info.validate_against_metadata(self.output_metadata)
        
        return self.output_info
    
    def get_output_reference(self, field: Optional[str] = None) -> str:
        """Get template reference string for this task's output."""
        if field:
            return f"{{{{{self.id}.{field}}}}}"
        else:
            return f"{{{{{self.id}.result}}}}"
    
    def validate_output_consistency(self) -> List[str]:
        """Validate consistency of output metadata."""
        if not self.output_metadata:
            return []
        
        issues = self.output_metadata.validate_consistency()
        
        # Additional task-specific validations
        if self.output_info:
            if not self.output_info.validate_against_metadata(self.output_metadata):
                issues.extend(self.output_info.validation_errors)
        
        return issues

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation."""
        result_dict = {
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
        
        # Add output metadata if present
        if self.output_metadata:
            result_dict["output_metadata"] = {
                "produces": self.output_metadata.produces,
                "location": self.output_metadata.location,
                "format": self.output_metadata.format,
                "schema": self.output_metadata.schema,
                "size_limit": self.output_metadata.size_limit,
                "validation_rules": self.output_metadata.validation_rules,
                "description": self.output_metadata.description,
                "tags": self.output_metadata.tags
            }
        
        # Add output info if present
        if self.output_info:
            result_dict["output_info"] = self.output_info.to_dict()
        
        return result_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Task:
        """Create task from dictionary representation."""
        # Extract output metadata and info before creating task
        output_metadata_data = data.pop("output_metadata", None)
        output_info_data = data.pop("output_info", None)
        
        # Convert status back to enum
        if "status" in data:
            data["status"] = TaskStatus(data["status"])

        # Handle error
        if "error" in data and data["error"]:
            data["error"] = Exception(data["error"])

        # Remove computed properties
        data.pop("execution_time", None)
        
        # Handle depends_on -> dependencies mapping
        if "depends_on" in data:
            data["dependencies"] = data.pop("depends_on")

        # Create task
        task = cls(**data)
        
        # Reconstruct output metadata if present
        if output_metadata_data:
            task.output_metadata = OutputMetadata(**output_metadata_data)
        
        # Reconstruct output info if present
        if output_info_data:
            # Convert datetime strings back to datetime objects
            from datetime import datetime
            for date_field in ['created_at', 'modified_at', 'accessed_at']:
                if output_info_data.get(date_field):
                    output_info_data[date_field] = datetime.fromisoformat(output_info_data[date_field])
            
            task.output_info = OutputInfo(**output_info_data)

        return task

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
