"""Hybrid control system that handles both model-based tasks and tool operations."""

from typing import Any, Dict, Optional
import re
from pathlib import Path
from datetime import datetime

from .model_based_control_system import ModelBasedControlSystem
from ..core.task import Task
from ..models.model_registry import ModelRegistry
from ..tools.system_tools import FileSystemTool
from ..tools.data_tools import DataProcessingTool
from ..tools.validation import ValidationTool
from ..compiler.template_renderer import TemplateRenderer


class HybridControlSystem(ModelBasedControlSystem):
    """Control system that handles both AI models and tool operations."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        name: str = "hybrid-control-system",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize hybrid control system."""
        # Initialize parent with extended capabilities
        if config is None:
            config = {
                "capabilities": {
                    "supported_actions": [
                        # Model-based actions
                        "generate",
                        "analyze",
                        "transform",
                        "execute",
                        "search",
                        "extract",
                        "filter",
                        "synthesize",
                        "create",
                        "validate",
                        "optimize",
                        "review",
                        # File operations
                        "save_output",
                        "save_to_file",
                        "write_file",
                        "read_file",
                        "save",
                        "write",
                    ],
                    "parallel_execution": True,
                    "streaming": True,
                    "checkpoint_support": True,
                },
                "base_priority": 25,  # Higher priority than model-based alone
            }

        super().__init__(model_registry, name, config)

        # Initialize tools
        self.filesystem_tool = FileSystemTool()
        self.data_processing_tool = DataProcessingTool()
        self.validation_tool = ValidationTool()

    async def execute_task(self, task: Task, context: Dict[str, Any]) -> Any:
        """Execute task with support for both models and tools."""
        action_str = str(task.action).lower()

        # Check if this is a simple echo/print operation
        if self._is_echo_operation(action_str):
            return await self._handle_echo_operation(task, context)

        # Check if this is a file operation
        if self._is_file_operation(action_str):
            return await self._handle_file_operation(task, context)
            
        # Check if this is a data processing operation
        if action_str == "process":
            return await self._handle_data_processing(task, context)
            
        # Check if this is a validation operation
        if action_str == "validate":
            return await self._handle_validation(task, context)

        # Otherwise use model-based execution
        return await super().execute_task(task, context)

    def _is_file_operation(self, action: str) -> bool:
        """Check if action is a file operation."""
        # Simple check for 'file' action
        if action.strip() == "file":
            return True
            
        # More specific patterns that indicate file operations
        file_patterns = [
            r"write.*to\s+(?:a\s+)?(?:file|path)",  # Write ... to file/path
            r"save.*to\s+(?:a\s+)?(?:file|path)",  # Save ... to file/path
            r"write.*following.*content.*to",  # Write the following content to
            r"save.*following.*content.*to",  # Save the following content to
            r"create.*file\s+at",  # Create file at
            r"export.*to\s+file",  # Export to file
            r"store.*in\s+file",  # Store in file
            r"write.*to\s+[^\s]+\.(txt|md|json|yaml|yml|csv|html)",  # Write to filename.ext
            r"save.*to\s+[^\s]+\.(txt|md|json|yaml|yml|csv|html)",  # Save to filename.ext
        ]
        return any(
            re.search(pattern, action, re.IGNORECASE | re.DOTALL)
            for pattern in file_patterns
        )

    def _is_echo_operation(self, action: str) -> bool:
        """Check if action is a simple echo/print operation."""
        echo_patterns = [
            r"^echo\s+",
            r"^print\s+",
            r"^display\s+",
            r"^show\s+",
            r"^output\s+",
        ]
        return any(
            re.match(pattern, action, re.IGNORECASE) for pattern in echo_patterns
        )

    async def _handle_echo_operation(
        self, task: Task, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle simple echo/print operations."""
        action_text = str(task.action)

        # Extract the message to echo (everything after the command)
        match = re.match(
            r"^(?:echo|print|display|show|output)\s+(.+)",
            action_text,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            message = match.group(1).strip()
        else:
            message = action_text

        # Build template context
        template_context = self._build_template_context(context)

        # Resolve templates in message
        if "{{" in message:
            message = self._resolve_templates(message, template_context)

        # Return the echoed message
        return {
            "result": message,
            "status": "success",
            "action": "echo",
        }

    async def _handle_file_operation(
        self, task: Task, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle file operations."""
        action_text = str(task.action)

        # If the action is simply "file", use the FileSystemTool
        if action_text.strip() == "file" and task.parameters:
            # Build template context with all available data
            template_context = self._build_template_context(context)
            
            # Resolve templates in parameters
            resolved_params = {}
            for key, value in task.parameters.items():
                if isinstance(value, str) and "{{" in value:
                    resolved_params[key] = self._resolve_templates(value, template_context)
                else:
                    resolved_params[key] = value
            
            # Use FileSystemTool directly
            return await self.filesystem_tool.execute(**resolved_params)
        
        # Otherwise handle as a write operation
        # Extract file path from action
        file_path = self._extract_file_path(action_text, task.parameters)
        if not file_path:
            file_path = f"output/{task.id}_output.txt"

        # Extract content
        content = self._extract_content(action_text, task.parameters)

        # Build template context with all available data
        template_context = self._build_template_context(context)

        # Resolve templates in file path
        if "{{" in file_path:
            file_path = self._resolve_templates(file_path, template_context)

        # Resolve templates in content
        if "{{" in content:
            content = self._resolve_templates(content, template_context)

        # Ensure parent directory exists
        file_path_obj = Path(file_path)
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Write the file
        try:
            file_path_obj.write_text(content)
            return {
                "success": True,
                "filepath": str(file_path_obj),
                "size": len(content),
                "message": f"Successfully wrote {len(content)} bytes to {file_path_obj}",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to write file: {e}",
            }

    def _extract_file_path(
        self, action_text: str, parameters: Dict[str, Any]
    ) -> Optional[str]:
        """Extract file path from action or parameters."""
        # Check parameters first
        if "filepath" in parameters:
            return parameters["filepath"]
        if "filename" in parameters:
            return parameters["filename"]
        if "path" in parameters:
            return parameters["path"]

        # Try to extract from action text
        patterns = [
            r"to (?:a )?(?:markdown )?file at ([^\n]+?)(?=:|\s*$)",  # Match until colon or end of line
            r"to ([^\s]+\.(?:md|txt|json|yaml|yml|csv|html))(?=:|\s|$)",  # Match file with extension
            r"Save.*to ([^\n]+?)(?=:|\s*$)",  # Save to ... (until colon or end)
            r"Write.*to ([^\n]+?)(?=:|\s*$)",  # Write to ... (until colon or end)
            r"(?:file|path):\s*([^\n]+?)(?=:|\s*$)",  # file: or path: prefix
        ]

        for pattern in patterns:
            match = re.search(pattern, action_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _extract_content(self, action_text: str, parameters: Dict[str, Any]) -> str:
        """Extract content to write from action or parameters."""
        # Check parameters first
        if "content" in parameters:
            return str(parameters["content"])
        if "data" in parameters:
            return str(parameters["data"])
        if "text" in parameters:
            return str(parameters["text"])

        # Try to extract from action text after a colon
        content_match = re.search(r":\s*\n(.*)", action_text, re.DOTALL)
        if content_match:
            return content_match.group(1).strip()

        # If action contains "following content", everything after that is content
        following_match = re.search(
            r"following content[:\s]*\n?(.*)", action_text, re.DOTALL | re.IGNORECASE
        )
        if following_match:
            return following_match.group(1).strip()

        return action_text

    def _build_template_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build complete context for template resolution."""
        template_context = context.copy()

        # Add execution metadata
        template_context.update(
            {
                "execution": {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "time": datetime.now().strftime("%H:%M:%S"),
                }
            }
        )

        # Flatten previous_results for easier access
        if "previous_results" in context:
            for step_id, result in context["previous_results"].items():
                if isinstance(result, dict):
                    template_context[step_id] = result
                else:
                    template_context[step_id] = {"result": result}

        # Also check for results in the main context (from pipeline execution)
        # Look for any keys that look like task results
        for key, value in context.items():
            if key not in template_context and key not in [
                "model",
                "pipeline",
                "execution",
                "task_id",
            ]:
                # Check if this might be a task result
                if isinstance(value, (str, int, float, bool, list, dict)):
                    template_context[key] = value

        return template_context

    def _resolve_templates(self, text: str, context: Dict[str, Any]) -> str:
        """Resolve template variables in text."""
        # Use the TemplateRenderer for consistent behavior
        return TemplateRenderer.render(text, context)
    
    async def _handle_data_processing(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle data processing operations."""
        if task.parameters:
            # Build template context
            template_context = self._build_template_context(context)
            
            # Resolve templates in parameters
            resolved_params = {}
            for key, value in task.parameters.items():
                if isinstance(value, str) and "{{" in value:
                    resolved_params[key] = self._resolve_templates(value, template_context)
                else:
                    resolved_params[key] = value
            
            # Use DataProcessingTool
            result = await self.data_processing_tool.execute(**resolved_params)
            
            # If the result contains processed_data, return it in the expected format
            if isinstance(result, dict) and "processed_data" in result:
                return result
            else:
                # Wrap the result to match expected format
                return {"processed_data": result, "success": True}
        
        return {"error": "No parameters provided for data processing", "success": False}
    
    async def _handle_validation(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle validation operations."""
        if task.parameters:
            # Build template context
            template_context = self._build_template_context(context)
            
            # Resolve templates in parameters
            resolved_params = {}
            for key, value in task.parameters.items():
                if isinstance(value, str) and "{{" in value:
                    resolved_params[key] = self._resolve_templates(value, template_context)
                else:
                    resolved_params[key] = value
            
            # Use ValidationTool
            return await self.validation_tool.execute(**resolved_params)
        
        return {"error": "No parameters provided for validation", "success": False}
