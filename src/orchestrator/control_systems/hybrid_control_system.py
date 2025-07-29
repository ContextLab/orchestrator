"""Hybrid control system that handles both model-based tasks and tool operations."""

from typing import Any, Dict, Optional
import re
from pathlib import Path
from datetime import datetime

from .model_based_control_system import ModelBasedControlSystem
from ..core.task import Task
from ..models.model_registry import ModelRegistry
from ..tools.system_tools import FileSystemTool, TerminalTool
from ..tools.data_tools import DataProcessingTool
from ..tools.validation import ValidationTool
from ..tools.web_tools import WebSearchTool, HeadlessBrowserTool
from ..tools.report_tools import ReportGeneratorTool, PDFCompilerTool
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
        self.terminal_tool = TerminalTool()
        self.data_processing_tool = DataProcessingTool()
        self.validation_tool = ValidationTool()
        self.web_search_tool = WebSearchTool()
        self.headless_browser_tool = HeadlessBrowserTool()
        self.report_generator_tool = ReportGeneratorTool()
        self.pdf_compiler_tool = PDFCompilerTool()

    async def execute_task(self, task: Task, context: Dict[str, Any]) -> Any:
        """Execute task with support for both models and tools."""
        action_str = str(task.action).lower()

        # Check if a specific tool is requested
        tool_name = task.metadata.get("tool")
        if tool_name:
            return await self._handle_tool_execution(task, tool_name, context)

        # Check if this is a control flow operation
        if action_str == "control_flow":
            return await self._handle_control_flow(task, context)

        # Check if this is a simple echo/print operation
        if self._is_echo_operation(action_str):
            return await self._handle_echo_operation(task, context)

        # Check if this is a file operation
        if self._is_file_operation(action_str) or action_str == "filesystem":
            return await self._handle_file_operation(task, context)

        # Check if this is a data processing operation
        if action_str == "process":
            return await self._handle_data_processing(task, context)

        # Check if this is a validation operation
        if action_str == "validate":
            return await self._handle_validation(task, context)

        # Otherwise use model-based execution
        return await super().execute_task(task, context)

    async def _handle_tool_execution(self, task: Task, tool_name: str, context: Dict[str, Any]) -> Any:
        """Handle execution with a specific tool."""
        # Map tool names to handlers
        tool_handlers = {
            "filesystem": self._handle_file_operation,
            "terminal": self._handle_terminal_operation,
            "data-processing": self._handle_data_processing,
            "validation": self._handle_validation,
            "web-search": self._handle_web_search,
            "headless-browser": self._handle_headless_browser,
            "report-generator": self._handle_report_generator,
            "pdf-compiler": self._handle_pdf_compiler,
            "pipeline-executor": self._handle_pipeline_executor,
        }
        
        if tool_name in tool_handlers:
            return await tool_handlers[tool_name](task, context)
        else:
            # Tool not found - this is an error, not a fallback situation
            available_tools = ", ".join(tool_handlers.keys())
            raise ValueError(f"Tool '{tool_name}' not found in hybrid control system. Available tools: {available_tools}")

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
        action_text = str(task.action).strip()
        

        # If task has parameters and tool is filesystem, use FileSystemTool for all operations
        if task.metadata.get("tool") == "filesystem" and task.parameters:
            # Build template context with all available data
            template_context = self._build_template_context(context)

            # Resolve templates in parameters
            resolved_params = {}
            for key, value in task.parameters.items():
                if isinstance(value, str):
                    # Always resolve templates for string values
                    resolved_value = self._resolve_templates(
                        value, template_context
                    )
                    resolved_params[key] = resolved_value
                else:
                    resolved_params[key] = value

            # Add the action parameter from the task
            resolved_params["action"] = action_text
            
            # Use FileSystemTool directly
            return await self.filesystem_tool.execute(**resolved_params)
        
        # If the action is a known filesystem operation, use FileSystemTool
        filesystem_actions = ["read", "write", "copy", "move", "delete", "list", "file"]
        if action_text in filesystem_actions and task.parameters:
            # Build template context with all available data
            template_context = self._build_template_context(context)

            # Resolve templates in parameters
            resolved_params = {}
            for key, value in task.parameters.items():
                if isinstance(value, str):
                    # Always resolve templates for string values
                    resolved_value = self._resolve_templates(
                        value, template_context
                    )
                    # Debug: Check if content parameter still has unrendered templates
                    if key == "content" and ("{{" in resolved_value or "{%" in resolved_value):
                        print(f">> WARNING: Content still has unrendered templates after resolution")
                        print(f">> First unrendered template check:")
                        print(f">> - summarize_results in context: {'summarize_results' in template_context}")
                        if 'summarize_results' in template_context:
                            print(f">> - summarize_results type: {type(template_context['summarize_results'])}")
                            if isinstance(template_context['summarize_results'], dict):
                                print(f">> - summarize_results keys: {list(template_context['summarize_results'].keys())}")
                    resolved_params[key] = resolved_value
                else:
                    resolved_params[key] = value

            # Add the action parameter from the task
            resolved_params["action"] = action_text
            
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
        
        # DEBUG: Print what's available in the context for template rendering
        task_id = context.get("task_id", context.get("current_task_id"))
        if task_id == "save_report":
            print(f">> DEBUG: Building template context for save_report")
            print(f">> Available step IDs in previous_results: {list(context.get('previous_results', {}).keys())}")
            print(f">> Available step IDs in template_context: {list(template_context.keys())}")
            if "create_report" in template_context:
                print(f">> create_report keys: {list(template_context['create_report'].keys()) if isinstance(template_context['create_report'], dict) else 'Not a dict'}")
            if "distribution_plan" in template_context:
                print(f">> distribution_plan keys: {list(template_context['distribution_plan'].keys()) if isinstance(template_context['distribution_plan'], dict) else 'Not a dict'}")

        # Add pipeline parameters if available
        if "pipeline_metadata" in context and isinstance(context["pipeline_metadata"], dict):
            # Check for both 'parameters' and 'inputs' keys
            params = context["pipeline_metadata"].get("parameters", {})
            for param_name, param_value in params.items():
                if param_name not in template_context:
                    template_context[param_name] = param_value
            
            # Also add inputs
            inputs = context["pipeline_metadata"].get("inputs", {})
            for input_name, input_value in inputs.items():
                if input_name not in template_context:
                    template_context[input_name] = input_value
        
        # Add pipeline context values (which should contain inputs)
        if "pipeline_context" in context and isinstance(context["pipeline_context"], dict):
            for key, value in context["pipeline_context"].items():
                if key not in template_context:
                    template_context[key] = value
        
        # Also check for results in the main context (from pipeline execution)
        # Look for any keys that look like task results
        for key, value in context.items():
            if key not in template_context and key not in [
                "model",
                "pipeline",
                "execution",
                "task_id",
                "pipeline_id",
                "pipeline_metadata",
                "execution_id",
                "checkpoint_enabled",
                "max_retries",
                "start_time",
                "current_level",
                "resource_allocation"
            ]:
                # Check if this might be a task result or parameter
                if isinstance(value, (str, int, float, bool, list, dict)):
                    template_context[key] = value

        return template_context

    def _resolve_templates(self, text: str, context: Dict[str, Any]) -> str:
        """Resolve template variables in text using Jinja2."""
        try:
            from jinja2 import Template, Environment, StrictUndefined
            
            # Create Jinja2 environment with custom filters
            env = Environment(undefined=StrictUndefined)
            
            # Add custom filters
            from datetime import datetime
            import json
            
            # Date filter
            def date_filter(value, format="%Y-%m-%d %H:%M:%S"):
                if isinstance(value, str):
                    try:
                        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    except:
                        value = datetime.now()
                elif not isinstance(value, datetime):
                    value = datetime.now()
                return value.strftime(format)
            
            # JSON filters
            def from_json_filter(value):
                if isinstance(value, str):
                    try:
                        return json.loads(value)
                    except:
                        return value
                return value
            
            # Register filters
            env.filters['date'] = date_filter
            env.filters['from_json'] = from_json_filter
            env.filters['to_json'] = lambda v: json.dumps(v, default=str)
            env.filters['slugify'] = lambda v: str(v).lower().replace(' ', '-').replace('_', '-')
            env.filters['default'] = lambda v, d='': v if v is not None else d
            
            # Render template
            template = env.from_string(text)
            return template.render(**context)
        except Exception as e:
            # If Jinja2 fails, it might be due to missing data
            # Try to return the original text with basic variable substitution
            result = text
            
            # Do basic variable substitution for common patterns
            for key, value in context.items():
                if isinstance(value, (str, int, float)):
                    result = result.replace(f"{{{{{key}}}}}", str(value))
            
            # If we still have unrendered templates, return original
            if "{{" in result or "{%" in result:
                return text
            return result

    async def _handle_data_processing(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle data processing operations."""
        if task.parameters:
            # Build template context
            template_context = self._build_template_context(context)

            # Resolve templates in parameters
            resolved_params = {}
            for key, value in task.parameters.items():
                if isinstance(value, str):
                    # Always resolve templates for string values
                    resolved_params[key] = self._resolve_templates(
                        value, template_context
                    )
                else:
                    resolved_params[key] = value
            
            # Add the action parameter from the task
            resolved_params["action"] = str(task.action).strip()

            # Special handling for transform_spec
            if "transform_spec" in resolved_params:
                import json

                transform_spec = resolved_params["transform_spec"]
                data_str = resolved_params.get("data", "")

                # Parse the JSON data if it's a string
                parsed_data = None
                try:
                    if isinstance(data_str, str):
                        parsed_data = json.loads(data_str)
                    else:
                        parsed_data = data_str
                except Exception:
                    parsed_data = data_str

                # Apply transformations
                processed_data = {}
                for field, expr in transform_spec.items():
                    try:
                        # Create safe evaluation context with necessary builtins
                        safe_builtins = {
                            "sum": sum,
                            "len": len,
                            "min": min,
                            "max": max,
                            "abs": abs,
                            "round": round,
                            "int": int,
                            "float": float,
                            "str": str,
                            "bool": bool,
                            "list": list,
                            "dict": dict,
                            "set": set,
                            "tuple": tuple,
                        }
                        # Provide both raw string data and parsed data in context
                        eval_context = {
                            "data": data_str,  # Raw string data for json.loads(data)
                            "parsed_data": parsed_data,  # Pre-parsed data
                            "json": json,
                        }
                        # Evaluate the expression
                        processed_data[field] = eval(
                            expr, {"__builtins__": safe_builtins}, eval_context
                        )
                    except Exception as e:
                        processed_data[field] = f"Error: {str(e)}"

                return {"processed_data": processed_data, "success": True}

            # Use DataProcessingTool for standard operations
            result = await self.data_processing_tool.execute(**resolved_params)

            # If the result contains processed_data, return it in the expected format
            if isinstance(result, dict) and "processed_data" in result:
                return result
            elif isinstance(result, dict) and "result" in result:
                # Convert result to processed_data format
                return {"processed_data": result["result"], "success": True}
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
                    # Special handling for 'data' parameter - preserve structure
                    if key == "data":
                        # Extract the referenced value directly from context
                        import re

                        match = re.search(r"\{\{(.+?)\}\}", value)
                        if match:
                            ref_path = match.group(1).strip()
                            parts = ref_path.split(".")
                            result = template_context
                            for part in parts:
                                if isinstance(result, dict) and part in result:
                                    result = result[part]
                                else:
                                    # Fallback to string resolution
                                    result = self._resolve_templates(
                                        value, template_context
                                    )
                                    break
                            resolved_params[key] = result
                        else:
                            resolved_params[key] = value
                    else:
                        resolved_params[key] = self._resolve_templates(
                            value, template_context
                        )
                else:
                    resolved_params[key] = value

            # Use ValidationTool
            return await self.validation_tool.execute(**resolved_params)

        return {"error": "No parameters provided for validation", "success": False}
    
    async def _handle_terminal_operation(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle terminal/command execution operations."""
        # Execute using terminal tool
        params = task.parameters.copy()
        params["action"] = task.action
        return await self.terminal_tool.execute(**params)
    
    async def _handle_web_search(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle web search operations."""
        # Build template context
        template_context = self._build_template_context(context)
        
        # Resolve templates in parameters
        resolved_params = {}
        for key, value in task.parameters.items():
            if isinstance(value, str) and ("{{" in value or "{%" in value):
                resolved_params[key] = self._resolve_templates(value, template_context)
            else:
                resolved_params[key] = value
        
        # Add the action parameter
        resolved_params["action"] = task.action
        
        # Execute using web search tool
        return await self.web_search_tool.execute(**resolved_params)
    
    async def _handle_headless_browser(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle headless browser operations."""
        # Build template context
        template_context = self._build_template_context(context)
        
        # Resolve templates in parameters
        resolved_params = {}
        for key, value in task.parameters.items():
            if isinstance(value, str) and ("{{" in value or "{%" in value):
                resolved_params[key] = self._resolve_templates(value, template_context)
            else:
                resolved_params[key] = value
        
        # Add the action parameter
        resolved_params["action"] = task.action
        
        # Execute using headless browser tool
        return await self.headless_browser_tool.execute(**resolved_params)
    
    async def _handle_control_flow(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle control flow operations."""
        # For control flow tasks, we need to return a success response
        # The actual control flow execution is handled by the orchestrator
        # This is just a placeholder to allow the task to complete
        
        # Get control flow type from metadata
        cf_type = None
        if "for_each" in task.metadata:
            cf_type = "for_each"
        elif "while" in task.metadata:
            cf_type = "while"
        elif "if" in task.metadata:
            cf_type = "if"
        else:
            cf_type = "unknown"
        
        # Build a descriptive result
        result = f"Control flow ({cf_type}) executed"
        
        if cf_type == "for_each":
            items = task.metadata.get("for_each", [])
            if isinstance(items, str):
                # Parse the string representation of the list
                try:
                    import ast
                    items = ast.literal_eval(items)
                except:
                    pass
            result = f"Processed {len(items) if isinstance(items, list) else 0} items"
        
        # Return success
        return {
            "result": result,
            "status": "success",
            "control_flow_type": cf_type,
            "metadata": task.metadata
        }

    async def _handle_report_generator(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle report generation operations."""
        # Build template context
        template_context = self._build_template_context(context)
        
        # Resolve templates in parameters
        resolved_params = {}
        for key, value in task.parameters.items():
            if isinstance(value, str) and ("{{" in value or "{%" in value):
                resolved_params[key] = self._resolve_templates(value, template_context)
            elif isinstance(value, list):
                # Handle lists with templates
                resolved_list = []
                for item in value:
                    if isinstance(item, str) and ("{{" in item or "{%" in item):
                        resolved_list.append(self._resolve_templates(item, template_context))
                    else:
                        resolved_list.append(item)
                resolved_params[key] = resolved_list
            else:
                resolved_params[key] = value
        
        # Add the action parameter
        resolved_params["action"] = task.action
        
        # Execute using report generator tool
        return await self.report_generator_tool.execute(**resolved_params)
    
    async def _handle_pdf_compiler(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle PDF compilation operations."""
        # Execute using PDF compiler tool
        params = task.parameters.copy()
        params["action"] = task.action
        return await self.pdf_compiler_tool.execute(**params)
    
    async def _handle_pipeline_executor(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle pipeline execution operations."""
        # Execute using pipeline executor tool
        from orchestrator.tools.pipeline_recursion_tools import PipelineExecutorTool
        
        if not hasattr(self, 'pipeline_executor_tool'):
            self.pipeline_executor_tool = PipelineExecutorTool()
        
        params = task.parameters.copy()
        return await self.pipeline_executor_tool.execute(**params)
