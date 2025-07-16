"""Pipeline abstraction for the orchestrator framework."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .task import Task, TaskStatus


class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected in the pipeline."""

    pass


class InvalidDependencyError(Exception):
    """Raised when a dependency refers to a non-existent task."""

    pass


@dataclass
class Pipeline:
    """
    Pipeline represents a collection of tasks with dependencies.

    A pipeline manages the execution order of tasks based on their dependencies
    and provides methods for validation and execution planning.
    """

    id: str
    name: str
    tasks: Dict[str, Task] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    version: str = "1.0.0"
    description: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate pipeline after initialization."""
        if not self.id:
            raise ValueError("Pipeline ID cannot be empty")
        if not self.name:
            raise ValueError("Pipeline name cannot be empty")

        # Validate tasks and dependencies
        self._validate_dependencies()

    def add_task(self, task: Task) -> None:
        """
        Add a task to the pipeline.

        Args:
            task: Task to add

        Raises:
            ValueError: If task with same ID already exists
        """
        if task.id in self.tasks:
            raise ValueError(f"Task with ID '{task.id}' already exists")

        self.tasks[task.id] = task
        self._validate_dependencies()

    def remove_task(self, task_id: str) -> Optional[Task]:
        """
        Remove a task from the pipeline and return it.

        Args:
            task_id: ID of task to remove

        Returns:
            Removed task, or None if task doesn't exist

        Raises:
            ValueError: If other tasks depend on it
        """
        if task_id not in self.tasks:
            return None

        # Check if any tasks depend on this task
        dependents = self._get_dependents(task_id)
        if dependents:
            raise ValueError(
                f"Cannot remove task '{task_id}' because it has dependents: {dependents}"
            )

        task = self.tasks[task_id]
        del self.tasks[task_id]
        return task

    def remove_task_strict(self, task_id: str) -> Task:
        """
        Remove a task from the pipeline (legacy interface that raises exceptions).

        Args:
            task_id: ID of task to remove

        Returns:
            Removed task

        Raises:
            ValueError: If task doesn't exist or other tasks depend on it
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task '{task_id}' does not exist")

        # Check if any tasks depend on this task
        dependents = self._get_dependents(task_id)
        if dependents:
            raise ValueError(
                f"Cannot remove task '{task_id}' - tasks {dependents} depend on it"
            )

        task = self.tasks[task_id]
        del self.tasks[task_id]
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a task by ID.

        Args:
            task_id: Task ID

        Returns:
            Task object, or None if not found
        """
        return self.tasks.get(task_id)

    def get_task_safe(self, task_id: str) -> Optional[Task]:
        """
        Get a task by ID without raising exceptions.

        Args:
            task_id: Task ID

        Returns:
            Task object or None if not found
        """
        return self.tasks.get(task_id)

    def get_task_strict(self, task_id: str) -> Task:
        """
        Get a task by ID (legacy interface that raises exceptions).

        Args:
            task_id: Task ID

        Returns:
            Task object

        Raises:
            ValueError: If task doesn't exist
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task '{task_id}' does not exist")
        return self.tasks[task_id]

    def _validate_dependencies(self) -> None:
        """Validate task dependencies."""
        # Check for invalid dependencies
        for task_id, task in self.tasks.items():
            for dep in task.dependencies:
                if dep not in self.tasks:
                    raise InvalidDependencyError(
                        f"Task '{task_id}' depends on non-existent task '{dep}'"
                    )

        # Check for circular dependencies
        cycles = self._detect_cycles()
        if cycles:
            raise CircularDependencyError(f"Circular dependencies detected: {cycles}")

    def _detect_cycles(self) -> List[List[str]]:
        """
        Detect circular dependencies using DFS.

        Returns:
            List of cycles found
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {task_id: WHITE for task_id in self.tasks}
        cycles = []

        def dfs(task_id: str, path: List[str]) -> None:
            color[task_id] = GRAY
            path.append(task_id)

            for dep in self.tasks[task_id].dependencies:
                if color[dep] == GRAY:
                    # Found cycle
                    cycle_start = path.index(dep)
                    cycles.append(path[cycle_start:] + [dep])
                elif color[dep] == WHITE:
                    dfs(dep, path[:])

            color[task_id] = BLACK

        for task_id in self.tasks:
            if color[task_id] == WHITE:
                dfs(task_id, [])

        return cycles

    def _get_dependents(self, task_id: str) -> List[str]:
        """Get tasks that depend on the given task."""
        dependents = []
        for tid, task in self.tasks.items():
            if task_id in task.dependencies:
                dependents.append(tid)
        return dependents

    def get_execution_order(self) -> List[str]:
        """
        Get flat execution order of tasks.

        Returns:
            List of task IDs in execution order
        """
        levels = self.get_execution_levels()
        return [task_id for level in levels for task_id in level]

    def get_ready_tasks(self, completed_tasks: Set[str]) -> List[Task]:
        """
        Get tasks that are ready to execute.

        Args:
            completed_tasks: Set of completed task IDs

        Returns:
            List of Task objects ready for execution
        """
        ready_tasks = []
        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.PENDING and task.is_ready(completed_tasks):
                ready_tasks.append(task)
        return ready_tasks

    def get_ready_task_ids(self, completed_tasks: Set[str]) -> List[str]:
        """
        Get task IDs that are ready to execute (legacy interface).

        Args:
            completed_tasks: Set of completed task IDs

        Returns:
            List of task IDs ready for execution
        """
        ready_tasks = []
        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.PENDING and task.is_ready(completed_tasks):
                ready_tasks.append(task_id)
        return ready_tasks

    def get_failed_tasks(self) -> List[str]:
        """Get list of failed task IDs."""
        return [
            task_id
            for task_id, task in self.tasks.items()
            if task.status == TaskStatus.FAILED
        ]

    def get_completed_tasks(self) -> List[str]:
        """Get list of completed task IDs."""
        return [
            task_id
            for task_id, task in self.tasks.items()
            if task.status == TaskStatus.COMPLETED
        ]

    def get_running_tasks(self) -> List[str]:
        """Get list of running task IDs."""
        return [
            task_id
            for task_id, task in self.tasks.items()
            if task.status == TaskStatus.RUNNING
        ]

    def reset(self) -> None:
        """Reset all tasks to pending state."""
        for task in self.tasks.values():
            task.reset()

    def is_complete(self) -> bool:
        """Check if all tasks are completed."""
        return all(task.status == TaskStatus.COMPLETED for task in self.tasks.values())

    def is_failed(self) -> bool:
        """Check if any critical task has failed."""
        return any(task.status == TaskStatus.FAILED for task in self.tasks.values())

    def get_progress(self) -> Dict[str, int]:
        """Get pipeline execution progress."""
        status_counts = defaultdict(int)
        for task in self.tasks.values():
            status_counts[task.status.value] += 1

        return {
            "total": len(self.tasks),
            "pending": status_counts[TaskStatus.PENDING.value],
            "running": status_counts[TaskStatus.RUNNING.value],
            "completed": status_counts[TaskStatus.COMPLETED.value],
            "failed": status_counts[TaskStatus.FAILED.value],
            "skipped": status_counts[TaskStatus.SKIPPED.value],
        }

    def get_critical_path(self) -> List[str]:
        """
        Get the critical path (longest path) through the pipeline.

        Returns:
            List of task IDs in the critical path
        """
        # Build reverse graph
        reverse_graph = defaultdict(list)
        for task_id, task in self.tasks.items():
            for dep in task.dependencies:
                reverse_graph[task_id].append(dep)

        # Find longest path using DFS
        memo = {}

        def longest_path(task_id: str) -> Tuple[int, List[str]]:
            if task_id in memo:
                return memo[task_id]

            if not reverse_graph[task_id]:  # No dependencies
                memo[task_id] = (1, [task_id])
                return memo[task_id]

            max_length = 0
            max_path = []

            for dep in reverse_graph[task_id]:
                length, path = longest_path(dep)
                if length > max_length:
                    max_length = length
                    max_path = path

            result = (max_length + 1, max_path + [task_id])
            memo[task_id] = result
            return result

        # Find the longest path among all tasks
        max_length = 0
        critical_path = []

        for task_id in self.tasks:
            length, path = longest_path(task_id)
            if length > max_length:
                max_length = length
                critical_path = path

        return critical_path

    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            "context": self.context,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "version": self.version,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Pipeline:
        """Create pipeline from dictionary representation."""
        # Convert tasks back to Task objects
        tasks = {}
        if "tasks" in data:
            for task_id, task_data in data["tasks"].items():
                tasks[task_id] = Task.from_dict(task_data)

        data["tasks"] = tasks
        return cls(**data)

    def __repr__(self) -> str:
        """String representation of pipeline."""
        return f"Pipeline(id='{self.id}', name='{self.name}', tasks={len(self.tasks)})"

    def __len__(self) -> int:
        """Number of tasks in pipeline."""
        return len(self.tasks)

    def __contains__(self, task_id: str) -> bool:
        """Check if task exists in pipeline."""
        return task_id in self.tasks

    def __iter__(self):
        """Iterate over task IDs."""
        return iter(self.tasks)

    def has_task(self, task_id: str) -> bool:
        """
        Check if pipeline has a task with given ID.

        Args:
            task_id: Task ID to check

        Returns:
            True if task exists, False otherwise
        """
        return task_id in self.tasks

    def get_execution_order_flat(self) -> List[str]:
        """
        Get flat execution order of tasks.

        Returns:
            List of task IDs in execution order
        """
        levels = self.get_execution_levels()
        return [task_id for level in levels for task_id in level]

    def get_execution_levels(self) -> List[List[str]]:
        """
        Get tasks grouped by execution level (parallel groups).

        Returns:
            List of lists, where each inner list contains task IDs that can
            be executed in parallel at that level
        """
        # Build dependency graph
        in_degree = {
            task_id: len(task.dependencies) for task_id, task in self.tasks.items()
        }
        graph = defaultdict(list)

        for task_id, task in self.tasks.items():
            for dep in task.dependencies:
                graph[dep].append(task_id)

        # Topological sort with level grouping
        levels = []
        queue = deque([task_id for task_id, degree in in_degree.items() if degree == 0])

        while queue:
            current_level = []
            level_size = len(queue)

            for _ in range(level_size):
                task_id = queue.popleft()
                current_level.append(task_id)

                for neighbor in graph[task_id]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            if current_level:
                levels.append(current_level)

        # Verify all tasks are included
        total_tasks = sum(len(level) for level in levels)
        if total_tasks != len(self.tasks):
            raise CircularDependencyError(
                "Cannot determine execution order - circular dependencies detected"
            )

        return levels

    def get_dependencies(self, task_id: str) -> List[str]:
        """
        Get dependencies for a task.

        Args:
            task_id: Task ID

        Returns:
            List of dependency task IDs
        """
        if task_id not in self.tasks:
            return []
        return list(self.tasks[task_id].dependencies)

    def get_dependents(self, task_id: str) -> List[str]:
        """
        Get tasks that depend on the given task.

        Args:
            task_id: Task ID

        Returns:
            List of dependent task IDs
        """
        return self._get_dependents(task_id)

    def is_valid(self) -> bool:
        """
        Check if pipeline is valid.

        Returns:
            True if valid, False otherwise
        """
        try:
            self._validate_dependencies()
            return True
        except (InvalidDependencyError, CircularDependencyError):
            return False

    def get_status(self) -> Dict[str, int]:
        """
        Get pipeline status summary.

        Returns:
            Dictionary with status counts
        """
        status_counts = defaultdict(int)
        for task in self.tasks.values():
            status_counts[task.status.value] += 1

        return {
            "total_tasks": len(self.tasks),
            "pending_tasks": status_counts[TaskStatus.PENDING.value],
            "running_tasks": status_counts[TaskStatus.RUNNING.value],
            "completed_tasks": status_counts[TaskStatus.COMPLETED.value],
            "failed_tasks": status_counts[TaskStatus.FAILED.value],
            "skipped_tasks": status_counts[TaskStatus.SKIPPED.value],
        }

    def clear_tasks(self) -> None:
        """Clear all tasks from the pipeline."""
        self.tasks.clear()

    @property
    def task_count(self) -> int:
        """Get number of tasks in pipeline."""
        return len(self.tasks)

    def __eq__(self, other: object) -> bool:
        """Check equality based on pipeline ID."""
        if not isinstance(other, Pipeline):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash based on pipeline ID."""
        return hash(self.id)
