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
        # No longer checking for pre-rendered parameters
        # We'll render templates on-demand here
        
        # Fallback to legacy rendering for backward compatibility
        # Import here to avoid circular dependency
        from .template_manager import TemplateManager
        
        # Create a copy of the task to avoid modifying the original
        import copy
        rendered_task = copy.deepcopy(task)
        
        # Skip if no parameters to render
        if not rendered_task.parameters:
            return rendered_task
        
        # Use template manager from context if available, otherwise create new one
        template_manager = context.get("template_manager")
        if template_manager is None:
            # Log warning that we're creating a new template manager
            import logging
            logging.warning("Creating new TemplateManager in ControlSystem - pipeline inputs may be lost!")
            # Create new template manager only if not provided
            template_manager = TemplateManager()
            
            # Register all context values since this is a new template manager
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
            
            # Register other context values (including direct pipeline inputs like 'topic')
            # Skip only internal keys and already registered results
            skip_keys = [
                "previous_results", "_template_manager", "pipeline_metadata", 
                "pipeline_context", "model", "pipeline", "pipeline_id",
                "execution_id", "checkpoint_enabled", "max_retries", 
                "start_time", "current_level", "resource_allocation",
                "task_id", "template_manager"
            ]
            
            for key, value in context.items():
                if key not in skip_keys and not key.startswith("_"):
                    # Skip if already registered as a result
                    if "previous_results" in context and key in context["previous_results"]:
                        continue
                    # Register all other values (this includes pipeline inputs like 'topic')
                    template_manager.register_context(key, value)
        else:
            # Template manager already exists, ensure all necessary context is registered
            
            # Ensure pipeline inputs like 'topic' are registered
            # Skip only internal keys and already registered results
            skip_keys = [
                "previous_results", "_template_manager", "pipeline_metadata", 
                "pipeline_context", "model", "pipeline", "pipeline_id",
                "execution_id", "checkpoint_enabled", "max_retries", 
                "start_time", "current_level", "resource_allocation",
                "task_id", "template_manager"
            ]
            
            # Register any missing context values (especially pipeline inputs)
            for key, value in context.items():
                if key not in skip_keys and not key.startswith("_"):
                    # Only register if not already in template manager
                    if key not in template_manager.context:
                        template_manager.register_context(key, value)
            
            # Register loop context if this is a for_each task
            if "loop_context" in rendered_task.metadata:
                loop_ctx = rendered_task.metadata["loop_context"]
                for var_name, var_value in loop_ctx.items():
                    template_manager.register_context(var_name, var_value)
            
            # Register pipeline inputs if stored in task metadata
            if "pipeline_inputs" in rendered_task.metadata:
                pipeline_inputs = rendered_task.metadata["pipeline_inputs"]
                for input_name, input_value in pipeline_inputs.items():
                    template_manager.register_context(input_name, input_value)
            
            # Special handling for loop task results
            # If this is a task within a loop, register previous results from the same iteration
            if "loop_id" in rendered_task.metadata and "loop_index" in rendered_task.metadata:
                loop_id = rendered_task.metadata["loop_id"]
                loop_index = rendered_task.metadata["loop_index"]
                
                # Look for results from the same loop iteration
                if "previous_results" in context:
                    for result_id, result_value in context["previous_results"].items():
                        # Check if this result is from the same loop iteration
                        if result_id.startswith(f"{loop_id}_{loop_index}_"):
                            # Extract the simple task name (e.g., "translate" from "translate_text_0_translate")
                            simple_name = result_id.replace(f"{loop_id}_{loop_index}_", "")
                            # Register with simple name for use within loop templates
                            template_manager.register_context(simple_name, result_value)
        
        # Deep render all parameters, but skip content for filesystem write operations
        # Debug logging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"_render_task_templates for task {rendered_task.id}: tool={rendered_task.metadata.get('tool')}, action={rendered_task.action}")
        if rendered_task.parameters and "action" in rendered_task.parameters:
            logger.info(f"Parameters contain action={rendered_task.parameters.get('action')}")
        if rendered_task.parameters and "content" in rendered_task.parameters:
            logger.info(f"Content parameter present, first 100 chars: {str(rendered_task.parameters.get('content'))[:100]}")
        
        # Check both cases:
        # 1. tool="filesystem" with action="write" (new style)
        # 2. action="filesystem" with parameters.action="write" (YAML tool style)
        is_filesystem_write = False
        
        # Case 1: Direct action="write" with tool="filesystem"
        if (rendered_task.metadata.get("tool") == "filesystem" and 
            rendered_task.action == "write" and 
            rendered_task.parameters and 
            "content" in rendered_task.parameters):
            is_filesystem_write = True
            
        # Case 2: action="filesystem" with write in parameters
        elif (rendered_task.action == "filesystem" and 
              rendered_task.parameters and 
              rendered_task.parameters.get("action") == "write" and
              "content" in rendered_task.parameters):
            is_filesystem_write = True
            
        if is_filesystem_write:
            logger.info("FileSystemTool write detected - preserving content parameter for runtime rendering")
            # Render all parameters except content
            rendered_params = {}
            for key, value in rendered_task.parameters.items():
                if key == "content":
                    # Keep content as-is for runtime rendering
                    rendered_params[key] = value
                else:
                    rendered_params[key] = template_manager.deep_render(value)
            rendered_task.parameters = rendered_params
        else:
            # Normal deep render for all other cases
            rendered_task.parameters = template_manager.deep_render(rendered_task.parameters)
        
        # Also render the action if it's a string with templates
        if isinstance(rendered_task.action, str):
            rendered_task.action = template_manager.deep_render(rendered_task.action)
        
        # Store template_manager in context for tools to use
        # This ensures tools get the same template_manager with all registered context
        context["_template_manager"] = template_manager
        
        return rendered_task
