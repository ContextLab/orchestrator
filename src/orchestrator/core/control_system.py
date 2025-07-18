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
