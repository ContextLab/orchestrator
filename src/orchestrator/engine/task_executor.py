"""Universal task executor for automatic execution of declarative tasks."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core.model import Model
from ..core.output_tracker import OutputTracker
from ..core.template_resolver import TemplateResolver
from ..tools.base import Tool, default_registry
from ..execution.error_handler_executor import ErrorHandlerExecutor
from .auto_resolver import EnhancedAutoResolver
from .pipeline_spec import TaskSpec

logger = logging.getLogger(__name__)


class UniversalTaskExecutor:
    """Executes any task defined declaratively with automatic tool and model selection."""

    def __init__(self, model_registry=None, tool_registry=None, output_tracker=None, error_handler_executor=None):
        self.model_registry = model_registry
        self.tool_registry = tool_registry or default_registry
        self.auto_resolver = EnhancedAutoResolver()
        self.execution_context = {}
        
        # Output tracking integration
        self.output_tracker = output_tracker or OutputTracker()
        self.template_resolver = TemplateResolver(self.output_tracker)
        
        # Advanced error handling integration
        self.error_handler_executor = error_handler_executor or ErrorHandlerExecutor(self)

    async def execute_task(
        self, task_spec: TaskSpec, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single task from its specification."""
        logger.info(f"Executing task: {task_spec.id}")

        try:
            # 0. Register output metadata if task has output specification
            if task_spec.has_output_metadata():
                output_metadata = task_spec.get_output_metadata()
                self.output_tracker.register_task_metadata(task_spec.id, output_metadata)
                logger.debug(f"Registered output metadata for task {task_spec.id}")

            # 1. Resolve AUTO tags if present
            if task_spec.has_auto_tags():
                resolved_spec = await self._resolve_auto_tags(task_spec, context)
            else:
                resolved_spec = {
                    "prompt": task_spec.action,
                    "tools": task_spec.tools or [],
                    "output_format": "structured",
                }

            # 2. Resolve template variables in prompt (including output references)
            prompt = self._resolve_template_variables_with_outputs(resolved_spec["prompt"], context)

            # 3. Determine required tools
            required_tools = self._get_required_tools(task_spec, resolved_spec)

            # 4. Execute the task
            result = await self._execute_with_tools_and_model(
                prompt=prompt,
                tools=required_tools,
                task_spec=task_spec,
                context=context,
                output_format=resolved_spec.get("output_format", "structured"),
            )

            # 5. Structure and validate result
            structured_result = self._structure_result(result, task_spec, resolved_spec)

            # 6. Register actual output
            output_info = await self._register_task_output(task_spec, structured_result, context)
            if output_info:
                structured_result["output_info"] = {
                    "location": output_info.location,
                    "format": output_info.format,
                    "output_type": output_info.output_type
                }

            logger.info(f"Task {task_spec.id} completed successfully")
            return structured_result

        except Exception as e:
            logger.error(f"Task {task_spec.id} failed: {str(e)}")

            # Use advanced error handling if available
            if task_spec.has_advanced_error_handling():
                return await self.error_handler_executor.handle_task_error(
                    failed_task=task_spec,
                    error=e,
                    context=context,
                    pipeline=None  # Pipeline reference not available at this level
                )
            # Fall back to legacy error handling
            elif task_spec.on_error:
                return await self._handle_task_error(task_spec, e, context)
            else:
                raise

    async def _resolve_auto_tags(
        self, task_spec: TaskSpec, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve AUTO tags in task specification."""
        auto_content = task_spec.extract_auto_content()

        # Set up auto resolver with available model
        if self.model_registry and hasattr(self.model_registry, "get_best_model"):
            model = self.model_registry.get_best_model(["reasoning"])
            self.auto_resolver.model = model

        return await self.auto_resolver.resolve_auto_tag(auto_content, context)

    def _resolve_template_variables(self, prompt: str, context: Dict[str, Any]) -> str:
        """Resolve template variables like {{variable}} in the prompt."""
        import re

        def replace_var(match):
            var_path = match.group(1).strip()

            # Handle nested access like {{results.search.data}}
            if "." in var_path:
                parts = var_path.split(".")
                value = context

                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        logger.warning(
                            f"Template variable path '{var_path}' not found in context"
                        )
                        return match.group(0)  # Return original if not found

                return str(value)
            else:
                # Simple variable access
                if var_path in context:
                    return str(context[var_path])
                else:
                    logger.warning(
                        f"Template variable '{var_path}' not found in context"
                    )
                    return match.group(0)  # Return original if not found

        return re.sub(r"\{\{([^}]+)\}\}", replace_var, prompt)

    def _resolve_template_variables_with_outputs(self, prompt: str, context: Dict[str, Any]) -> str:
        """Resolve template variables including output references from other tasks."""
        import re

        def replace_var(match):
            var_path = match.group(1).strip()

            # Handle nested access like {{task_id.result}} or {{task_id.location}}
            if "." in var_path:
                parts = var_path.split(".", 1)
                task_id = parts[0]
                field = parts[1]
                
                # Try to get from output tracker first
                if self.output_tracker.has_output(task_id):
                    try:
                        value = self.output_tracker.get_output(task_id, field)
                        return str(value)
                    except (KeyError, AttributeError):
                        pass
                
                # Fall back to regular context resolution
                value = context
                for part in var_path.split("."):
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        logger.warning(
                            f"Template variable path '{var_path}' not found in context or outputs"
                        )
                        return match.group(0)  # Return original if not found
                
                return str(value)
            else:
                # Simple variable access - check context first
                if var_path in context:
                    return str(context[var_path])
                
                # Check if it's a task result reference
                if self.output_tracker.has_output(var_path):
                    try:
                        value = self.output_tracker.get_output(var_path)
                        return str(value)
                    except (KeyError, AttributeError):
                        pass
                
                logger.warning(
                    f"Template variable '{var_path}' not found in context or outputs"
                )
                return match.group(0)  # Return original if not found

        return re.sub(r"\{\{([^}]+)\}\}", replace_var, prompt)

    def _get_required_tools(
        self, task_spec: TaskSpec, resolved_spec: Dict[str, Any]
    ) -> List[Tool]:
        """Get tools required for task execution."""
        tool_names = set()

        # Add explicitly specified tools
        if task_spec.tools:
            tool_names.update(task_spec.tools)

        # Add tools suggested by AUTO resolver
        if "tools" in resolved_spec:
            tool_names.update(resolved_spec["tools"])

        # Get tool instances from registry
        tools = []
        for tool_name in tool_names:
            tool = self.tool_registry.get_tool(tool_name)
            if tool:
                tools.append(tool)
            else:
                logger.warning(f"Tool '{tool_name}' not found in registry")

        return tools

    async def _execute_with_tools_and_model(
        self,
        prompt: str,
        tools: List[Tool],
        task_spec: TaskSpec,
        context: Dict[str, Any],
        output_format: str,
    ) -> Any:
        """Execute task using appropriate tools and model."""

        # If tools are specified, execute with tools
        if tools:
            return await self._execute_with_tools(prompt, tools, task_spec, context)

        # If no tools, execute with model only
        elif self.model_registry:
            return await self._execute_with_model_only(prompt, task_spec, output_format)

        else:
            raise ValueError(
                f"No tools or models available to execute task: {task_spec.id}"
            )

    async def _execute_with_tools(
        self,
        prompt: str,
        tools: List[Tool],
        task_spec: TaskSpec,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute task using specified tools."""
        results = {}

        # For now, execute tools in sequence
        # TODO: Implement intelligent tool orchestration
        for tool in tools:
            try:
                logger.debug(f"Executing tool: {tool.name}")

                # Extract parameters for tool from context and task inputs
                tool_params = self._extract_tool_parameters(
                    tool, task_spec, context, prompt
                )

                # Execute tool
                tool_result = await tool.execute(**tool_params)
                results[tool.name] = tool_result

                # If tool execution was successful, break (for simple case)
                if isinstance(tool_result, dict) and tool_result.get("success"):
                    results["primary_result"] = tool_result
                    break

            except Exception as e:
                logger.warning(f"Tool {tool.name} execution failed: {str(e)}")
                results[f"{tool.name}_error"] = str(e)

        return results

    async def _execute_with_model_only(
        self, prompt: str, task_spec: TaskSpec, output_format: str
    ) -> str:
        """Execute task using model only (no tools)."""
        if not self.model_registry:
            raise ValueError("No model registry available for model-only execution")

        # Get appropriate model for task
        model = self._select_model_for_task(task_spec)

        # Execute with model
        if output_format == "json" or output_format == "structured":
            # Try to get structured output
            try:
                schema = {
                    "type": "object",
                    "properties": {"result": {"type": "string"}},
                }
                return await model.generate_structured(prompt, schema)
            except Exception:
                # Fallback to regular generation
                return await model.generate(prompt)
        else:
            return await model.generate(prompt)

    def _extract_tool_parameters(
        self, tool: Tool, task_spec: TaskSpec, context: Dict[str, Any], prompt: str
    ) -> Dict[str, Any]:
        """Extract appropriate parameters for tool execution."""
        params = {}

        # Start with task inputs
        params.update(task_spec.inputs)

        # Add context data
        for key, value in context.items():
            if not key.startswith("_"):  # Skip internal context variables
                params[key] = value

        # Tool-specific parameter mapping
        if tool.name == "web-search":
            params["query"] = params.get("query", prompt)
            params["max_results"] = params.get("max_results", 10)

        elif tool.name == "report-generator":
            params["title"] = params.get("title", "Generated Report")
            params["content"] = params.get("content", prompt)

        elif tool.name == "data-processing":
            params["action"] = params.get("action", "analyze")
            params["data"] = params.get("data", context.get("data"))

        elif tool.name == "headless-browser":
            params["action"] = params.get("action", "scrape")
            params["url"] = params.get("url", context.get("url"))

        elif tool.name == "filesystem":
            params["action"] = params.get("action", "read")
            params["path"] = params.get("path", context.get("path"))

        # Remove None values
        return {k: v for k, v in params.items() if v is not None}

    def _select_model_for_task(self, task_spec: TaskSpec) -> Model:
        """Select appropriate model for task execution."""
        if not self.model_registry:
            raise ValueError("No model registry available")

        # Use model requirements from task spec if available
        if task_spec.model_requirements:
            required_capabilities = task_spec.model_requirements.get("capabilities", [])
            min_size = task_spec.model_requirements.get("min_size")

            if hasattr(self.model_registry, "get_best_model"):
                return self.model_registry.get_best_model(
                    required_capabilities, min_size
                )

        # Default to general purpose model
        if hasattr(self.model_registry, "get_default_model"):
            return self.model_registry.get_default_model()

        # Fallback to first available model
        models = getattr(self.model_registry, "models", [])
        if models:
            return models[0]

        raise ValueError("No models available in registry")

    def _structure_result(
        self, result: Any, task_spec: TaskSpec, resolved_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Structure the raw result into a standardized format."""
        structured = {
            "task_id": task_spec.id,
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "result": result,
        }

        # Extract specific fields based on result type
        if isinstance(result, dict):
            # If result is already structured, preserve important fields
            if "success" in result:
                structured["success"] = result["success"]

            # Extract common result patterns
            if "content" in result:
                structured["content"] = result["content"]
            if "data" in result:
                structured["data"] = result["data"]
            if "results" in result:
                structured["results"] = result["results"]
            if "insights" in result:
                structured["insights"] = result["insights"]
            if "summary" in result:
                structured["summary"] = result["summary"]

        return structured

    async def _handle_task_error(
        self, task_spec: TaskSpec, error: Exception, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle task error based on legacy on_error specification."""
        logger.info(
            f"Handling error for task {task_spec.id} with legacy strategy: {task_spec.on_error}"
        )

        # Handle string-based error configuration
        if isinstance(task_spec.on_error, str):
            if task_spec.on_error.startswith("<AUTO>"):
                # Execute error handling as another AUTO task
                error_action = task_spec.on_error[6:-7]  # Remove <AUTO> tags
                error_context = context.copy()
                error_context["error"] = str(error)
                error_context["failed_task"] = task_spec.id

                error_spec = TaskSpec(
                    id=f"{task_spec.id}_error_handler", action=error_action
                )

                return await self.execute_task(error_spec, error_context)
            else:
                # Simple error response
                return {
                    "task_id": task_spec.id,
                    "success": False,
                    "error": str(error),
                    "error_handler": task_spec.on_error,
                    "timestamp": datetime.now().isoformat(),
                }
        
        # Handle ErrorHandling object (from pipeline_spec.py)
        elif hasattr(task_spec.on_error, 'action'):
            error_context = context.copy()
            error_context["error"] = str(error)
            error_context["error_type"] = type(error).__name__
            error_context["failed_task"] = task_spec.id

            error_spec = TaskSpec(
                id=f"{task_spec.id}_error_handler", 
                action=task_spec.on_error.action
            )

            try:
                error_result = await self.execute_task(error_spec, error_context)
                return {
                    "task_id": task_spec.id,
                    "success": False,
                    "error": str(error),
                    "error_handled": True,
                    "error_handler_result": error_result,
                    "fallback_value": task_spec.on_error.fallback_value,
                    "continue_pipeline": task_spec.on_error.continue_on_error,
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as handler_error:
                logger.error(f"Error handler for task {task_spec.id} also failed: {handler_error}")
                return {
                    "task_id": task_spec.id,
                    "success": False,
                    "error": str(error),
                    "error_handled": False,
                    "handler_error": str(handler_error),
                    "fallback_value": task_spec.on_error.fallback_value,
                    "continue_pipeline": task_spec.on_error.continue_on_error,
                    "timestamp": datetime.now().isoformat(),
                }
        
        else:
            # Unknown error configuration format
            return {
                "task_id": task_spec.id,
                "success": False,
                "error": str(error),
                "error_handler": "unknown_format",
                "timestamp": datetime.now().isoformat(),
            }

    async def _register_task_output(self, task_spec: TaskSpec, result: Dict[str, Any], 
                                   context: Dict[str, Any]) -> Optional[Any]:
        """Register task output with the output tracker."""
        if not task_spec.has_output_metadata():
            return None
        
        # Determine output location
        location = None
        if task_spec.location:
            try:
                # Resolve location template with current context and outputs
                location = self.template_resolver.resolve_template(
                    task_spec.location, 
                    default_values=context
                )
                
                # Ensure directory exists if it's a file path
                if location and ('/' in location or '\\' in location):
                    location = self.template_resolver.resolve_file_path(location, ensure_dir=True)
            except Exception as e:
                logger.warning(f"Failed to resolve output location template: {e}")
                location = task_spec.location  # Use unresolved template
        
        # Determine format
        format_type = task_spec.format
        if not format_type and location:
            # Try to detect format from location
            from ..core.output_metadata import OutputFormatDetector
            format_type = OutputFormatDetector.detect_from_location(location)
        
        # Extract actual result content
        actual_result = result.get("result", result)
        
        # Handle file output - save result to file if needed
        if location and task_spec.is_file_output():
            try:
                await self._save_result_to_file(actual_result, location, format_type)
            except Exception as e:
                logger.error(f"Failed to save result to file {location}: {e}")
        
        # Register with output tracker
        try:
            output_info = self.output_tracker.register_output(
                task_id=task_spec.id,
                result=actual_result,
                location=location,
                format=format_type
            )
            logger.debug(f"Registered output for task {task_spec.id} at {location}")
            return output_info
        except Exception as e:
            logger.error(f"Failed to register output for task {task_spec.id}: {e}")
            return None

    async def _save_result_to_file(self, result: Any, location: str, format_type: Optional[str]) -> None:
        """Save task result to file."""
        import os
        import json
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(location), exist_ok=True)
        
        try:
            if format_type == 'application/json' or location.endswith('.json'):
                # JSON output
                with open(location, 'w', encoding='utf-8') as f:
                    if isinstance(result, (dict, list)):
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    else:
                        json.dump({"result": result}, f, indent=2, ensure_ascii=False)
            
            elif format_type in ['text/markdown', 'text/plain'] or location.endswith(('.md', '.txt')):
                # Text/Markdown output
                with open(location, 'w', encoding='utf-8') as f:
                    if isinstance(result, dict):
                        # Extract content from structured result
                        content = result.get('content', result.get('result', str(result)))
                    else:
                        content = str(result)
                    f.write(content)
            
            elif format_type == 'text/html' or location.endswith('.html'):
                # HTML output
                with open(location, 'w', encoding='utf-8') as f:
                    if isinstance(result, dict):
                        content = result.get('content', result.get('result', str(result)))
                    else:
                        content = str(result)
                    
                    # Basic HTML wrapper if content doesn't look like HTML
                    if not content.strip().startswith('<'):
                        content = f"<html><body><pre>{content}</pre></body></html>"
                    f.write(content)
            
            else:
                # Default: save as text
                with open(location, 'w', encoding='utf-8') as f:
                    f.write(str(result))
            
            logger.debug(f"Saved result to {location}")
            
        except Exception as e:
            logger.error(f"Failed to save result to {location}: {e}")
            raise
    
    def get_error_handler_executor(self) -> ErrorHandlerExecutor:
        """Get the error handler executor instance."""
        return self.error_handler_executor
    
    def set_error_handler_executor(self, executor: ErrorHandlerExecutor) -> None:
        """Set a custom error handler executor."""
        self.error_handler_executor = executor
