"""Control system abstraction for the orchestrator framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from .pipeline import Pipeline
from .task import Task


class ControlAction(Enum):
    """Control actions for task execution."""

    EXECUTE = "execute"
    SKIP = "skip"
    WAIT = "wait"
    RETRY = "retry"
    FAIL = "fail"


class ControlSystem(ABC):
    """Abstract base class for control system adapters."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize control system.

        Args:
            name: Control system name
            config: Configuration dictionary
        """
        if not name:
            raise ValueError("Control system name cannot be empty")

        self.name = name
        self.config = config or {}
        self._capabilities = self._load_capabilities()

    @abstractmethod
    async def execute_task(self, task: Task, context: Dict[str, Any]) -> Any:
        """
        Execute a single task.

        Args:
            task: Task to execute
            context: Execution context

        Returns:
            Task execution result
        """
        pass

    @abstractmethod
    async def execute_pipeline(self, pipeline: Pipeline) -> Dict[str, Any]:
        """
        Execute an entire pipeline.

        Args:
            pipeline: Pipeline to execute

        Returns:
            Pipeline execution results
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Return system capabilities.

        Returns:
            Dictionary of capabilities
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the system is healthy.

        Returns:
            True if healthy, False otherwise
        """
        pass

    def _load_capabilities(self) -> Dict[str, Any]:
        """Load system capabilities from config."""
        return self.config.get("capabilities", {})

    def supports_capability(self, capability: str) -> bool:
        """Check if system supports a specific capability."""
        return capability in self._capabilities

    def can_execute_task(self, task: Task) -> bool:
        """
        Check if system can execute a specific task.

        Args:
            task: Task to check

        Returns:
            True if task can be executed
        """
        # Check if task action is supported
        supported_actions = self._capabilities.get("supported_actions", [])
        if supported_actions and task.action not in supported_actions:
            return False

        # Check if task requires specific capabilities
        required_capabilities = task.metadata.get("required_capabilities", [])
        return all(self.supports_capability(cap) for cap in required_capabilities)

    def get_priority(self, task: Task) -> int:
        """
        Get execution priority for a task.

        Args:
            task: Task to get priority for

        Returns:
            Priority score (higher = more priority)
        """
        base_priority = self.config.get("base_priority", 0)

        # Adjust priority based on task metadata
        if "priority" in task.metadata:
            return task.metadata["priority"]

        return base_priority

    def __repr__(self) -> str:
        """String representation of control system."""
        return f"ControlSystem(name='{self.name}')"

    def __eq__(self, other: object) -> bool:
        """Check equality based on name."""
        if not isinstance(other, ControlSystem):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        """Hash based on name."""
        return hash(self.name)


class MockControlSystem(ControlSystem):
    """Mock control system implementation for testing."""

    def __init__(
        self,
        name: str = "mock-control-system",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize mock control system."""
        if config is None:
            config = {
                "capabilities": {
                    "supported_actions": [
                        "generate",
                        "analyze",
                        "transform",
                        "execute",
                    ],
                    "parallel_execution": True,
                    "streaming": False,
                    "checkpoint_support": True,
                },
                "base_priority": 10,
            }

        super().__init__(name, config)
        self._task_results = {}
        self._execution_history = []

    def set_task_result(self, task_id: str, result: Any) -> None:
        """Set canned result for a task."""
        self._task_results[task_id] = result

    async def execute_task(self, task: Task, context: Dict[str, Any]) -> Any:
        """Execute a mock task."""
        # Record execution
        self._execution_history.append(
            {
                "task_id": task.id,
                "action": task.action,
                "parameters": task.parameters,
                "context": context,
            }
        )

        # Return canned result if available
        if task.id in self._task_results:
            return self._task_results[task.id]

        # Generate mock result based on action
        if task.action == "generate":
            return f"Generated content for task {task.id}"
        elif task.action == "analyze":
            return {"analysis": f"Analysis result for task {task.id}"}
        elif task.action == "transform":
            return {"transformed_data": f"Transformed data for task {task.id}"}
        elif task.action == "execute":
            return {"execution_result": f"Execution result for task {task.id}"}
        else:
            return f"Mock result for action '{task.action}' in task {task.id}"

    async def execute_pipeline(self, pipeline: Pipeline) -> Dict[str, Any]:
        """Execute a mock pipeline."""
        results: Dict[str, Any] = {}

        # Get execution levels (groups of tasks that can run in parallel)
        execution_levels = pipeline.get_execution_levels()

        # Execute tasks level by level
        for level in execution_levels:
            level_results: Dict[str, Any] = {}

            for task_id in level:
                task = pipeline.get_task(task_id)

                # Build context with results from previous tasks
                context = {"pipeline_id": pipeline.id, "results": results}

                # Execute task
                result = await self.execute_task(task, context)
                level_results[task_id] = result

                # Mark task as completed
                task.complete(result)

            # Add level results to overall results
            results.update(level_results)

        return results

    def get_capabilities(self) -> Dict[str, Any]:
        """Get mock capabilities."""
        return self._capabilities

    async def health_check(self) -> bool:
        """Mock health check."""
        return True

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self._execution_history.copy()

    def clear_execution_history(self) -> None:
        """Clear execution history."""
        self._execution_history.clear()

    def clear_task_results(self) -> None:
        """Clear task results."""
        self._task_results.clear()
