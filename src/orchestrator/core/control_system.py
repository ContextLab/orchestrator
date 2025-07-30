"""Control system abstraction for the orchestrator framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional

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

    async def execute_task(self, task: Task, context: Dict[str, Any]) -> Any:
        """
        Execute a single task with automatic template rendering.

        Args:
            task: Task to execute
            context: Execution context

        Returns:
            Task execution result
        """
        # Pre-process task parameters with template rendering
        rendered_task = self._render_task_templates(task, context)
        
        # Call the implementation-specific execution
        return await self._execute_task_impl(rendered_task, context)
    
    @abstractmethod
    async def _execute_task_impl(self, task: Task, context: Dict[str, Any]) -> Any:
        """
        Execute a single task (to be implemented by subclasses).

        Args:
            task: Task to execute (with templates already rendered)
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
    
    def _render_task_templates(self, task: Task, context: Dict[str, Any]) -> Task:
        """
        Render all template strings in task parameters using deep template rendering.
        
        Args:
            task: Task with potential template strings
            context: Execution context with previous results
            
        Returns:
            Task with all templates rendered
        """
        # Import here to avoid circular dependency
        from .template_manager import TemplateManager
        
        # Create a copy of the task to avoid modifying the original
        import copy
        rendered_task = copy.deepcopy(task)
        
        # Skip if no parameters to render
        if not rendered_task.parameters:
            return rendered_task
        
        # Create template manager and register context
        template_manager = TemplateManager()
        
        # Register previous results if available
        if "previous_results" in context:
            template_manager.register_all_results(context["previous_results"])
        
        # Register pipeline parameters if available
        if "pipeline_metadata" in context and isinstance(context["pipeline_metadata"], dict):
            params = context["pipeline_metadata"].get("parameters", {})
            for param_name, param_value in params.items():
                template_manager.register_context(param_name, param_value)
            
            # Also check for inputs
            inputs = context["pipeline_metadata"].get("inputs", {})
            for input_name, input_value in inputs.items():
                template_manager.register_context(input_name, input_value)
        
        # Register pipeline context values (which should contain inputs)
        if "pipeline_context" in context and isinstance(context["pipeline_context"], dict):
            for key, value in context["pipeline_context"].items():
                template_manager.register_context(key, value)
        
        # Add execution metadata
        from datetime import datetime
        template_manager.register_context("execution", {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S")
        })
        
        # Register other context values
        for key, value in context.items():
            if key not in ["previous_results", "_template_manager", "pipeline_metadata", "pipeline_context"] and not key.startswith("_"):
                # Skip if already registered as a result
                if "previous_results" in context and key in context["previous_results"]:
                    continue
                template_manager.register_context(key, value)
        
        # Deep render all parameters
        rendered_task.parameters = template_manager.deep_render(rendered_task.parameters)
        
        # Also render the action if it's a string with templates
        if isinstance(rendered_task.action, str):
            rendered_task.action = template_manager.deep_render(rendered_task.action)
        
        # Store template_manager in context for tools to use
        context["_template_manager"] = template_manager
        
        return rendered_task
