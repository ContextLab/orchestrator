"""Universal task executor for automatic execution of declarative tasks."""

import logging
from datetime import datetime
from typing import Any, Dict, List

from ..core.model import Model
from ..tools.base import Tool, default_registry
from .auto_resolver import EnhancedAutoResolver
from .pipeline_spec import TaskSpec

logger = logging.getLogger(__name__)


class UniversalTaskExecutor:
    """Executes any task defined declaratively with automatic tool and model selection."""

    def __init__(self, model_registry=None, tool_registry=None):
        self.model_registry = model_registry
        self.tool_registry = tool_registry or default_registry
        self.auto_resolver = EnhancedAutoResolver()
        self.execution_context = {}

    async def execute_task(self, task_spec: TaskSpec, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single task from its specification."""
        logger.info(f"Executing task: {task_spec.id}")

        try:
            # 1. Resolve AUTO tags if present
            if task_spec.has_auto_tags():
                resolved_spec = await self._resolve_auto_tags(task_spec, context)
            else:
                resolved_spec = {
                    "prompt": task_spec.action,
                    "tools": task_spec.tools or [],
                    "output_format": "structured",
                }

            # 2. Resolve template variables in prompt
            prompt = self._resolve_template_variables(resolved_spec["prompt"], context)

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

            logger.info(f"Task {task_spec.id} completed successfully")
            return structured_result

        except Exception as e:
            logger.error(f"Task {task_spec.id} failed: {str(e)}")

            # Handle error based on task configuration
            if task_spec.on_error:
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
                        logger.warning(f"Template variable path '{var_path}' not found in context")
                        return match.group(0)  # Return original if not found

                return str(value)
            else:
                # Simple variable access
                if var_path in context:
                    return str(context[var_path])
                else:
                    logger.warning(f"Template variable '{var_path}' not found in context")
                    return match.group(0)  # Return original if not found

        return re.sub(r"\{\{([^}]+)\}\}", replace_var, prompt)

    def _get_required_tools(self, task_spec: TaskSpec, resolved_spec: Dict[str, Any]) -> List[Tool]:
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
            raise ValueError(f"No tools or models available to execute task: {task_spec.id}")

    async def _execute_with_tools(
        self, prompt: str, tools: List[Tool], task_spec: TaskSpec, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute task using specified tools."""
        results = {}

        # For now, execute tools in sequence
        # TODO: Implement intelligent tool orchestration
        for tool in tools:
            try:
                logger.debug(f"Executing tool: {tool.name}")

                # Extract parameters for tool from context and task inputs
                tool_params = self._extract_tool_parameters(tool, task_spec, context, prompt)

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
                schema = {"type": "object", "properties": {"result": {"type": "string"}}}
                return await model.generate_structured(prompt, schema)
            except:
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
                return self.model_registry.get_best_model(required_capabilities, min_size)

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
        """Handle task error based on on_error specification."""
        logger.info(f"Handling error for task {task_spec.id} with strategy: {task_spec.on_error}")

        if task_spec.on_error.startswith("<AUTO>"):
            # Execute error handling as another AUTO task
            task_spec.on_error[6:-7]  # Remove <AUTO> tags
            error_context = context.copy()
            error_context["error"] = str(error)
            error_context["failed_task"] = task_spec.id

            error_spec = TaskSpec(id=f"{task_spec.id}_error_handler", action=task_spec.on_error)

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
