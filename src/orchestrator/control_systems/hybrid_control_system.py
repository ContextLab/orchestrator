"""Hybrid control system that handles both model-based tasks and tool operations."""

from typing import Any, Dict, Optional
import logging
import re
from pathlib import Path
from datetime import datetime

from .model_based_control_system import ModelBasedControlSystem
from ..core.task import Task
from ..core.action_loop_task import ActionLoopTask
from ..core.unified_template_resolver import UnifiedTemplateResolver, TemplateResolutionContext
from ..models.model_registry import ModelRegistry
from ..tools.system_tools import FileSystemTool, TerminalTool
from ..tools.data_tools import DataProcessingTool
from ..tools.validation import ValidationTool
from ..tools.web_tools import WebSearchTool, HeadlessBrowserTool
from ..tools.report_tools import ReportGeneratorTool, PDFCompilerTool
from ..tools.checkpoint_tool import CheckpointTool
from ..tools.code_execution import PythonExecutorTool
from ..tools.multimodal_tools import (
    ImageGenerationTool, 
    ImageAnalysisTool,
    AudioProcessingTool,
    VideoProcessingTool
)
from ..tools.user_interaction_tools import (
    UserPromptTool, 
    ApprovalGateTool, 
    FeedbackCollectionTool
)
from ..tools.llm_tools import (
    TaskDelegationTool,
    MultiModelRoutingTool,
    PromptOptimizationTool
)
from ..tools.mcp_tools import (
    MCPServerTool,
    MCPMemoryTool,
    MCPResourceTool
)
from ..tools.pipeline_recursion_tools import (
    PipelineExecutorTool,
    RecursionControlTool
)
from ..compiler.template_renderer import TemplateRenderer
from ..runtime import RuntimeResolutionIntegration

logger = logging.getLogger(__name__)


class HybridControlSystem(ModelBasedControlSystem):
    """Control system that handles both AI models and tool operations."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        name: str = "hybrid-control-system",
        config: Optional[Dict[str, Any]] = None,
        template_resolver: Optional[UnifiedTemplateResolver] = None,
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

        # Use provided template resolver or create new one
        # Issue #287: Support RecursiveTemplateResolver for advanced loop patterns  
        self.hybrid_template_resolver = template_resolver or UnifiedTemplateResolver(debug_mode=True)

        # Initialize tools
        self.filesystem_tool = FileSystemTool()
        self.terminal_tool = TerminalTool()
        self.data_processing_tool = DataProcessingTool()
        self.validation_tool = ValidationTool()
        self.web_search_tool = WebSearchTool()
        self.headless_browser_tool = HeadlessBrowserTool()
        self.report_generator_tool = ReportGeneratorTool()
        self.pdf_compiler_tool = PDFCompilerTool()
        self.checkpoint_tool = CheckpointTool()
        self.image_generation_tool = ImageGenerationTool()
        self.image_analysis_tool = ImageAnalysisTool()
        self.audio_processing_tool = AudioProcessingTool()
        self.video_processing_tool = VideoProcessingTool()
        
        # Import and initialize VisualizationTool
        from ..tools.visualization_tools import VisualizationTool
        self.visualization_tool = VisualizationTool()
        
        # Initialize Python executor tool
        self.python_executor_tool = PythonExecutorTool()
        
        # Initialize user interaction tools (Issue #165)
        self.user_prompt_tool = UserPromptTool()
        self.approval_gate_tool = ApprovalGateTool()
        self.feedback_collection_tool = FeedbackCollectionTool()
        
        # Initialize LLM routing tools (Issue #166)
        self.task_delegation_tool = TaskDelegationTool()
        self.multi_model_routing_tool = MultiModelRoutingTool()
        self.prompt_optimization_tool = PromptOptimizationTool()
        
        # Initialize MCP tools (Issue #167)
        self.mcp_server_tool = MCPServerTool()
        self.mcp_memory_tool = MCPMemoryTool()
        self.mcp_resource_tool = MCPResourceTool()
        
        # Initialize recursion control tool (Issue #172)
        self.recursion_control_tool = RecursionControlTool()
        
        # Initialize runtime resolution system (Issue #211)
        self.runtime_resolution = None  # Will be initialized per pipeline

    async def _execute_task_impl(self, task: Task, context: Dict[str, Any]) -> Any:
        """Execute task with support for both models and tools."""
        action_str = str(task.action).lower()

        # Check if a specific tool is requested
        tool_name = task.metadata.get("tool")
        if tool_name:
            print(f"Routing to tool handler: {tool_name}")
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

        # Check if this is a loop completion marker
        if action_str == "loop_complete":
            return await self._handle_loop_complete(task, context)
            
        # Check if this is a capture result marker
        if action_str == "capture_result":
            return await self._handle_capture_result(task, context)
            
        # Check if this is a condition evaluation
        if action_str == "evaluate_condition":
            return await self._handle_evaluate_condition(task, context)
            
        # Check if this is a parallel queue execution
        if action_str == "create_parallel_queue":
            return await self._handle_create_parallel_queue(task, context)
        
        # Check if this is an action loop
        if action_str == "action_loop":
            return await self._handle_action_loop(task, context)
        
        # Check if this is text analysis
        if action_str == "analyze_text" or action_str == "analyze":
            return await self._handle_analyze_text(task, context)
        
        # Check if this is text generation
        if action_str == "generate_text" or action_str == "generate":
            return await self._handle_generate_text(task, context)

        # Otherwise use model-based execution
        return await super()._execute_task_impl(task, context)

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
            "checkpoint": self._handle_checkpoint,
            "image-generation": self._handle_image_generation,
            "image-analysis": self._handle_image_analysis,
            "audio-processing": self._handle_audio_processing,
            "video-processing": self._handle_video_processing,
            "task-delegation": self._handle_task_delegation,
            "multi-model-routing": self._handle_multi_model_routing,
            "prompt-optimization": self._handle_prompt_optimization_real,
            "user-prompt": self._handle_user_prompt,
            "approval-gate": self._handle_approval_gate,
            "feedback-collection": self._handle_feedback_collection,
            "mcp-server": self._handle_mcp_server,
            "mcp-memory": self._handle_mcp_memory,
            "mcp-resource": self._handle_mcp_resource,
            "visualization": self._handle_visualization,
            "python-executor": self._handle_python_executor,
            "recursion-control": self._handle_recursion_control,
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

        # Templates have already been rendered by ControlSystem._render_task_templates
        # The action text already contains rendered values

        # Return the echoed message
        return {
            "result": message,
            "status": "success",
            "action": "echo",
        }

    def _prepare_template_context(self, context: Dict[str, Any]) -> TemplateResolutionContext:
        """Prepare comprehensive template context using UnifiedTemplateResolver.
        
        This replaces the old template manager registration methods with the unified system.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Initialize runtime resolution if not already done
        if self.runtime_resolution is None:
            pipeline_id = context.get("pipeline_id", "default")
            self.runtime_resolution = RuntimeResolutionIntegration(pipeline_id)
            logger.info(f"Initialized runtime resolution for pipeline {pipeline_id}")
        
        # Collect comprehensive template context
        # Extract pipeline inputs from context (they're merged directly in orchestrator)
        pipeline_inputs = {k: v for k, v in context.items() 
                         if k not in ["pipeline_id", "pipeline_metadata", "pipeline_context", 
                                    "execution_id", "checkpoint_enabled", "max_retries", "start_time",
                                    "previous_results", "_loop_context_mapping", "$item", "$index", 
                                    "item", "index", "iteration", "$iteration", "_template_manager",
                                    "pipeline_params", "template_manager", "unified_template_resolver",
                                    "template_resolution_context", "task_id", "current_level", 
                                    "resource_allocation"]}
        
        
        template_context = self.hybrid_template_resolver.collect_context(
            pipeline_id=context.get("pipeline_id"),
            task_id=context.get("task_id"),
            pipeline_inputs=pipeline_inputs,
            pipeline_parameters=context.get("pipeline_params", {}),
            step_results=context.get("previous_results", {}),
            additional_context={
                # Add results from runtime resolution
                **self._get_runtime_resolution_context(context),
                # Add loop context mapping for short names
                **context.get("_loop_context_mapping", {}),
                # Add loop variables
                "$item": context.get("$item"),
                "$index": context.get("$index"),
                "$is_first": context.get("$is_first"),
                "$is_last": context.get("$is_last"),
                "$iteration": context.get("$iteration"),
                "$loop_state": context.get("$loop_state"),
                # Add other context variables
                **{k: v for k, v in context.items() 
                   if k not in ["pipeline_id", "pipeline_inputs", "pipeline_params", "previous_results", "_template_manager"]
                   and not k.startswith("_")}
            }
        )
        
        logger.info(f"Prepared template context with {len(template_context.to_flat_dict())} variables")
        return template_context
    
    def _get_runtime_resolution_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get context from runtime resolution system if available."""
        if self.runtime_resolution is None:
            return {}
        
        try:
            # Get state variables from runtime resolution
            return self.runtime_resolution.state.get_all_variables()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Could not get runtime resolution context: {e}")
            return {}
    
    async def _handle_file_operation(
        self, task: Task, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle file operations."""
        action_text = str(task.action).strip()
        

        # If task has parameters and tool is filesystem, use FileSystemTool for all operations
        if task.metadata.get("tool") == "filesystem" and task.parameters:
            # Use UnifiedTemplateResolver for template resolution
            resolved_params = task.parameters.copy()
            if "action" not in resolved_params:
                resolved_params["action"] = action_text
            
            # Prepare template context using the unified system
            template_context = self._prepare_template_context(context)
            
            # Add task-specific metadata to context
            if "loop_variables" in task.metadata:
                loop_vars = task.metadata["loop_variables"]
                template_context.additional_context.update(loop_vars)
            
            # Resolve templates using UnifiedTemplateResolver
            resolved_params = self.hybrid_template_resolver.resolve_before_tool_execution(
                "filesystem", resolved_params, template_context
            )
            
            # Pass the unified resolver components to the tool for runtime rendering
            resolved_params["unified_template_resolver"] = self.hybrid_template_resolver
            resolved_params["template_resolution_context"] = template_context
            
            # Also pass loop context mapping if available
            if "_loop_context_mapping" in context:
                resolved_params["_loop_context_mapping"] = context["_loop_context_mapping"]
            
            print(f"   📋 Using UnifiedTemplateResolver for filesystem tool with {len(context.get('previous_results', {}))} results")
            
            return await self.filesystem_tool.execute(**resolved_params)
        
        # If the action is a known filesystem operation, use FileSystemTool
        filesystem_actions = ["read", "write", "copy", "move", "delete", "list", "file"]
        if action_text in filesystem_actions and task.parameters:
            # Use UnifiedTemplateResolver for template resolution
            resolved_params = task.parameters.copy()
            resolved_params["action"] = action_text
            
            # Prepare template context using the unified system
            template_context = self._prepare_template_context(context)
            
            # Add task-specific metadata to context
            if "loop_variables" in task.metadata:
                loop_vars = task.metadata["loop_variables"]
                template_context.additional_context.update(loop_vars)
            
            # Resolve templates using UnifiedTemplateResolver
            resolved_params = self.hybrid_template_resolver.resolve_before_tool_execution(
                "filesystem", resolved_params, template_context
            )
            
            # Pass the unified resolver components to the tool for runtime rendering
            resolved_params["unified_template_resolver"] = self.hybrid_template_resolver
            resolved_params["template_resolution_context"] = template_context
            
            # Also pass loop context mapping if available
            if "_loop_context_mapping" in context:
                resolved_params["_loop_context_mapping"] = context["_loop_context_mapping"]
            
            print(f"   📋 Using UnifiedTemplateResolver for filesystem tool with {len(context.get('previous_results', {}))} results")
            
            return await self.filesystem_tool.execute(**resolved_params)

        # Otherwise handle as a write operation
        # Extract file path from action
        file_path = self._extract_file_path(action_text, task.parameters)
        if not file_path:
            file_path = f"output/{task.id}_output.txt"

        # Extract content
        content = self._extract_content(action_text, task.parameters)

        # Templates have already been rendered by ControlSystem._render_task_templates
        # No need to render again

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
        """Build complete context for template resolution using UnifiedTemplateResolver.
        
        This method is kept for backward compatibility but now uses the unified system.
        """
        # Use the new unified template context preparation
        template_context_obj = self._prepare_template_context(context)
        
        # Add execution metadata
        execution_metadata = {
            "execution": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M:%S"),
            }
        }
        
        # Convert to flat dict and add execution metadata
        flat_context = template_context_obj.to_flat_dict()
        flat_context.update(execution_metadata)
        
        return flat_context

    def _resolve_templates(self, text: str, context: Dict[str, Any]) -> str:
        """Resolve template variables in text using UnifiedTemplateResolver.
        
        This method is kept for backward compatibility but now uses the unified system.
        """
        try:
            # Prepare template context using the unified system
            template_context_obj = self._prepare_template_context(context)
            
            # Use the unified template resolver
            resolved = self.hybrid_template_resolver.resolve_templates(text, template_context_obj)
            
            return resolved
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Template resolution failed: {e}")
            # Return original text as fallback
            return text

    async def _handle_data_processing(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle data processing operations."""
        if task.parameters:
            # Templates have already been rendered by ControlSystem._render_task_templates
            resolved_params = task.parameters.copy()
            
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
            # Templates have already been rendered by ControlSystem._render_task_templates
            # Use ValidationTool
            return await self.validation_tool.execute(**task.parameters)

        return {"error": "No parameters provided for validation", "success": False}
    
    async def _handle_terminal_operation(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle terminal/command execution operations."""
        # Execute using terminal tool
        params = task.parameters.copy()
        params["action"] = task.action
        return await self.terminal_tool.execute(**params)
    
    async def _handle_web_search(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle web search operations."""
        # Templates have already been rendered by ControlSystem._render_task_templates
        params = task.parameters.copy()
        params["action"] = task.action
        
        # Execute using web search tool
        return await self.web_search_tool.execute(**params)
    
    async def _handle_headless_browser(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle headless browser operations."""
        # Templates have already been rendered by ControlSystem._render_task_templates
        params = task.parameters.copy()
        params["action"] = task.action
        
        # Execute using headless browser tool
        return await self.headless_browser_tool.execute(**params)
    
    async def _handle_control_flow(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle control flow operations."""
        # For control flow tasks, we need to return a success response
        # The actual control flow execution is handled by the orchestrator
        # This is just a placeholder to allow the task to complete
        
        # Get control flow type from metadata
        cf_type = None
        if "for_each" in task.metadata:
            cf_type = "for_each"
        elif "while" in task.metadata or task.metadata.get("is_while_loop"):
            cf_type = "while"
            # While loops should NOT be executed - they need to be expanded
            raise ValueError("While loop tasks should not be executed directly. They must be expanded by the orchestrator.")
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

    async def _handle_capture_result(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle capture result markers for while loops."""
        loop_id = task.parameters.get("loop_id", "unknown")
        iteration = task.parameters.get("iteration", 0)
        
        # Get the results from the previous tasks in this iteration
        iteration_results = {}
        for task_id, result in context.get("previous_results", {}).items():
            # Check if this task is from the current iteration
            if task_id.startswith(f"{loop_id}_{iteration}_") and not task_id.endswith("_result"):
                # Extract the step name from the task_id
                step_name = task_id.replace(f"{loop_id}_{iteration}_", "")
                iteration_results[step_name] = result
        
        # Return the aggregated results for this iteration
        return {
            "result": f"Iteration {iteration} completed",
            "status": "success",
            "loop_id": loop_id,
            "iteration": iteration,
            "iteration_results": iteration_results,
        }
    
    async def _handle_evaluate_condition(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle condition evaluation using the condition evaluator."""
        from ..actions.condition_evaluator import get_condition_evaluator
        
        # Get condition from parameters
        condition = task.parameters.get("condition", "")
        if not condition:
            return {
                "result": False,
                "status": "error",
                "error": "No condition provided"
            }
        
        # Prepare evaluation context - start with the full context
        eval_context = context.copy()
        
        # Add task parameters (but don't overwrite existing context)
        for key, value in task.parameters.items():
            if key not in eval_context:
                eval_context[key] = value
        
        # Flatten previous results if present
        if "previous_results" in context:
            for step_id, result in context["previous_results"].items():
                if step_id not in eval_context:
                    eval_context[step_id] = result
        
        # Get appropriate evaluator
        evaluator = get_condition_evaluator(condition, eval_context)
        
        # Execute evaluation
        result = await evaluator.execute(
            condition=condition,
            context=eval_context
        )
        
        # Add some useful metadata
        result["evaluator_type"] = type(evaluator).__name__
        
        return result
    
    async def _handle_loop_complete(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle loop completion markers."""
        loop_id = task.parameters.get("loop_id", "unknown")
        iterations = task.parameters.get("iterations", 0)
        cf_type = task.metadata.get("control_flow_type", "loop")
        
        return {
            "result": f"Completed {iterations} iterations",
            "status": "success",
            "control_flow_type": cf_type,
            "loop_id": loop_id,
            "iterations": iterations,
        }

    async def _handle_report_generator(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle report generation operations."""
        # Templates have already been rendered by ControlSystem._render_task_templates
        params = task.parameters.copy()
        params["action"] = task.action
        
        # Execute using report generator tool
        return await self.report_generator_tool.execute(**params)
    
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
    
    async def _handle_checkpoint(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle checkpoint inspection operations."""
        # Execute using checkpoint tool
        params = task.parameters.copy()
        params["action"] = task.action
        return await self.checkpoint_tool.execute(**params)
    
    async def _handle_image_generation(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle image generation operations."""
        # Execute using image generation tool
        params = task.parameters.copy()
        return await self.image_generation_tool.execute(**params)
    
    async def _handle_image_analysis(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle image analysis operations."""
        # Execute using image analysis tool
        params = task.parameters.copy()
        return await self.image_analysis_tool.execute(**params)
    
    async def _handle_audio_processing(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle audio processing operations."""
        # Execute using audio processing tool
        params = task.parameters.copy()
        return await self.audio_processing_tool.execute(**params)
    
    async def _handle_video_processing(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle video processing operations."""
        # Execute using video processing tool
        params = task.parameters.copy()
        return await self.video_processing_tool.execute(**params)
    
    async def _handle_prompt_optimization_placeholder(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle prompt optimization operations (DEPRECATED - kept for reference)."""
        # This is the old placeholder implementation - replaced by _handle_prompt_optimization_real
        prompt = task.parameters.get("prompt", "")
        task_type = task.parameters.get("task", "image-generation")
        goal = task.parameters.get("optimization_goal", "quality")
        
        # Use the model to enhance the prompt
        model = await self.model_registry.select_model({"tasks": ["generate"]})
        if not model:
            return {
                "success": False,
                "error": "No model available for prompt optimization"
            }
        
        optimization_prompt = f"""Optimize this prompt for {task_type}:
Original prompt: "{prompt}"
Goal: {goal}

Provide an enhanced, more detailed version that will produce better results.
Just return the optimized prompt, nothing else."""
        
        try:
            optimized = await model.generate(optimization_prompt, temperature=0.7)
            return {
                "success": True,
                "optimized_prompt": optimized.strip(),
                "original_prompt": prompt,
                "optimization_goal": goal
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "optimized_prompt": prompt  # Fallback to original
            }
    
    async def _handle_create_parallel_queue(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle parallel queue execution with real functionality."""
        from ..core.parallel_queue_task import ParallelQueueTask
        from ..control_flow.parallel_queue_handler import ParallelQueueHandler
        from ..control_flow.auto_resolver import ControlFlowAutoResolver
        from ..core.loop_context import GlobalLoopContextManager
        
        # Ensure we have a ParallelQueueTask
        if not isinstance(task, ParallelQueueTask):
            # Convert regular task to ParallelQueueTask if needed
            task_def = task.to_dict()
            task = ParallelQueueTask.from_task_definition(task_def)
        
        # Initialize handler with real components
        auto_resolver = ControlFlowAutoResolver(model_registry=self.model_registry)
        loop_context_manager = GlobalLoopContextManager()
        
        # Initialize handler
        handler = ParallelQueueHandler(
            auto_resolver=auto_resolver,
            loop_context_manager=loop_context_manager,
            model_registry=self.model_registry
        )
        
        # Get step results from context
        step_results = context.get("previous_results", {})
        
        # Execute the parallel queue
        result = await handler.execute_parallel_queue(task, context, step_results)
        
        return result

    async def _handle_action_loop(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle action loop execution with real functionality."""
        from ..control_flow.action_loop_handler import ActionLoopHandler
        from ..core.template_manager import TemplateManager
        from ..control_flow.auto_resolver import ControlFlowAutoResolver
        
        # Ensure we have an ActionLoopTask
        if not isinstance(task, ActionLoopTask):
            # Convert regular task to ActionLoopTask if needed
            task_dict = task.to_dict()
            
            # Extract action loop configuration from task metadata
            metadata = task.metadata or {}
            if "action_loop" in metadata:
                task_dict["action_loop"] = metadata["action_loop"]
            if "until" in metadata:
                task_dict["until"] = metadata["until"]
            if "while_condition" in metadata:
                task_dict["while_condition"] = metadata["while_condition"]
            if "max_iterations" in metadata:
                task_dict["max_iterations"] = metadata["max_iterations"]
            if "break_on_error" in metadata:
                task_dict["break_on_error"] = metadata["break_on_error"]
            if "iteration_timeout" in metadata:
                task_dict["iteration_timeout"] = metadata["iteration_timeout"]
            
            action_loop_task = ActionLoopTask.from_task_definition(task_dict)
        else:
            action_loop_task = task
        
        # Initialize action loop handler with real components
        auto_resolver = ControlFlowAutoResolver(model_registry=self.model_registry)
        template_manager = TemplateManager()
        
        # Update template manager with context
        for key, value in context.items():
            template_manager.register_context(key, value)
        
        # Pass template manager in context for tool integration
        enhanced_context = context.copy()
        enhanced_context["_template_manager"] = template_manager
        
        # Initialize handler
        handler = ActionLoopHandler(
            auto_resolver=auto_resolver,
            template_manager=template_manager
        )
        
        # Execute the action loop
        result = await handler.execute_action_loop(action_loop_task, enhanced_context)
        
        return result
    
    async def _handle_user_prompt(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle user prompt operations with real stdin/stdout."""
        params = task.parameters.copy()
        
        # Pass template manager for runtime rendering if available
        if hasattr(self, '_template_manager') and self._template_manager:
            params['template_manager'] = self._template_manager
        
        return await self.user_prompt_tool.execute(**params)
    
    async def _handle_approval_gate(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle approval gate operations with real user interaction."""
        params = task.parameters.copy()
        
        # Pass template manager for runtime rendering if available
        if hasattr(self, '_template_manager') and self._template_manager:
            params['template_manager'] = self._template_manager
        
        return await self.approval_gate_tool.execute(**params)
    
    async def _handle_feedback_collection(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle feedback collection with real user input."""
        params = task.parameters.copy()
        
        # Pass template manager for runtime rendering if available
        if hasattr(self, '_template_manager') and self._template_manager:
            params['template_manager'] = self._template_manager
        
        return await self.feedback_collection_tool.execute(**params)
    
    async def _handle_task_delegation(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle task delegation using TaskDelegationTool."""
        params = task.parameters.copy()
        
        # Handle AUTO tags if present in requirements
        if "requirements" in params and isinstance(params["requirements"], dict):
            for key, value in params["requirements"].items():
                if isinstance(value, str) and value.startswith("<AUTO>"):
                    # Use runtime resolution for AUTO tags if available
                    if self.runtime_resolution:
                        resolved = await self.runtime_resolution.resolve_auto_tag(
                            value, context
                        )
                        params["requirements"][key] = resolved
                    else:
                        # Fallback: extract the prompt from AUTO tag
                        import re
                        match = re.match(r'<AUTO>(.*?)</AUTO>', value)
                        if match:
                            # Use a model to resolve the AUTO tag
                            auto_prompt = match.group(1)
                            model = await self.model_registry.select_model({"tasks": ["generate"]})
                            if model:
                                response = await model.generate(
                                    f"{auto_prompt}\nBased on the task: '{params.get('task', '')}'\nRespond with only: simple, moderate, or complex"
                                )
                                # Extract complexity from response
                                response_lower = response.lower()
                                if "complex" in response_lower:
                                    params["requirements"][key] = "complex"
                                elif "moderate" in response_lower:
                                    params["requirements"][key] = "moderate"
                                else:
                                    params["requirements"][key] = "simple"
        
        # Ensure model registry is accessible
        if not hasattr(self.task_delegation_tool, 'model_registry'):
            self.task_delegation_tool.model_registry = self.model_registry
        
        return await self.task_delegation_tool._execute_impl(**params)
    
    async def _handle_multi_model_routing(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle multi-model routing using MultiModelRoutingTool."""
        params = task.parameters.copy()
        
        # Ensure model registry is accessible
        if not hasattr(self.multi_model_routing_tool, 'model_registry'):
            self.multi_model_routing_tool.model_registry = self.model_registry
        
        return await self.multi_model_routing_tool._execute_impl(**params)
    
    async def _handle_prompt_optimization_real(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle prompt optimization using real PromptOptimizationTool."""
        params = task.parameters.copy()
        
        # Ensure model registry is accessible
        if not hasattr(self.prompt_optimization_tool, 'model_registry'):
            self.prompt_optimization_tool.model_registry = self.model_registry
        
        return await self.prompt_optimization_tool._execute_impl(**params)
    
    async def _handle_analyze_text(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle text analysis using AI models."""
        from ..core.model import Model
        
        # Get parameters
        params = task.parameters.copy()
        text = params.get("text", "")
        analysis_type = params.get("analysis_type", "general")
        custom_prompt = params.get("prompt", "")
        
        # Build the full prompt - if custom prompt exists, combine it with the text
        if custom_prompt:
            prompt = f"{custom_prompt}\n\nData:\n{text}"
        else:
            prompt = f"Analyze the following text for {analysis_type}:\n\n{text}"
        
        # Add system instruction to avoid conversational text
        if analysis_type == "trends":
            prompt += "\n\nProvide only the analysis results. Do not include conversational text like 'If you want...' or 'I can help...' or questions to the user."
        
        model_spec = params.get("model", "<AUTO>")
        
        # Select model
        if self.model_registry:
            if model_spec == "<AUTO>" or model_spec.startswith("<AUTO>"):
                # Auto-select model
                requirements = {
                    "tasks": ["analyze", "generate"],
                    "context_window": len(prompt.encode()) // 4  # Rough token estimate
                }
                model = await self.model_registry.select_model(requirements)
            else:
                # Get specific model
                model = self.model_registry.get_model(model_spec)
        else:
            # Fallback to creating a model directly
            from ..models.openai_model import OpenAIModel
            model = OpenAIModel(name="gpt-4")
        
        if not model:
            return {
                "success": False,
                "error": "No suitable model found for text analysis"
            }
        
        # Call model
        try:
            # Log the model call for debugging
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Calling model {model.name if hasattr(model, 'name') else str(model)} for analyze_text")
            logger.debug(f"Prompt length: {len(prompt)} chars, First 200 chars: {prompt[:200]}")
            
            # Incorporate quality instructions into the prompt itself
            if analysis_type == "trends" or analysis_type == "text_generation":
                prompt = f"Instructions: Provide clear, concise, and accurate responses. Avoid conversational fillers or questions back to the user.\n\n{prompt}"
            
            response = await model.generate(
                prompt=prompt,
                max_tokens=params.get("max_tokens", 2000),  # Increased for GPT-5
                temperature=params.get("temperature", 0.7)
            )
            
            logger.info(f"Model response length: {len(response) if response else 0} chars")
            if not response:
                logger.warning("Model returned empty response!")
                logger.warning(f"Prompt was: {prompt[:500]}")
                # Return a default structure for empty responses
                return {
                    "action": "analyze_text",
                    "analysis_type": analysis_type,
                    "result": {"error": "Model returned empty response"},
                    "model_used": model.name if hasattr(model, 'name') else str(model),
                    "success": False
                }
            
            # Try to parse as JSON if expected
            result = response
            if "json" in analysis_type.lower() or "JSON" in prompt or "JSON" in str(prompt):
                try:
                    import json
                    result = json.loads(response)
                except json.JSONDecodeError:
                    # Clean up response and try again
                    cleaned = response.strip()
                    if cleaned.startswith("```json"):
                        cleaned = cleaned[7:]
                    if cleaned.startswith("```"):
                        cleaned = cleaned[3:]
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3]
                    try:
                        result = json.loads(cleaned.strip())
                    except json.JSONDecodeError:
                        # Log the failure for debugging
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(f"Failed to parse JSON from analyze_text response. First 200 chars: {cleaned[:200]}")
                        # Keep as string if can't parse
                        pass
            
            return {
                "action": "analyze_text",
                "analysis_type": analysis_type,
                "result": result,
                "model_used": model.name if hasattr(model, 'name') else str(model),
                "success": True
            }
            
        except Exception as e:
            return {
                "action": "analyze_text",
                "success": False,
                "error": str(e)
            }
    
    async def _handle_generate_text(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle text generation using AI models."""
        # Use the model-based control system for direct text generation
        # This bypasses the analyze_text handler which adds "Data:" formatting
        params = task.parameters.copy()
        
        # Get the prompt directly from parameters
        if "prompt" not in params:
            raise ValueError(f"Task '{task.id}' with action 'generate_text' requires a 'prompt' parameter")
        
        prompt = params["prompt"]
        
        # Get model from parameters or select appropriate model
        model_spec = params.get("model")
        model = None
        
        if model_spec:
            if isinstance(model_spec, str) and not model_spec.startswith("<AUTO"):
                # Direct model specification
                model = self.model_registry.get_model(model_spec)
            elif isinstance(model_spec, str) and model_spec.startswith("<AUTO"):
                # AUTO tag - let model registry select
                requirements = {"tasks": ["generate"], "context_window": len(prompt) // 4}
                model = await self.model_registry.select_model(requirements)
        else:
            # No model specified, select appropriate model
            requirements = {"tasks": ["generate"], "context_window": len(prompt) // 4}
            model = await self.model_registry.select_model(requirements)
        
        if not model:
            return {
                "success": False,
                "error": "No suitable model found for text generation"
            }
        
        try:
            # Call model directly for generation
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Calling model {model.name if hasattr(model, 'name') else str(model)} for generate_text")
            
            response = await model.generate(
                prompt=prompt,
                max_tokens=params.get("max_tokens", 1000),
                temperature=params.get("temperature", 0.7)
            )
            
            logger.info(f"Model response length: {len(response) if response else 0} chars")
            if not response:
                logger.warning("Model returned empty response!")
                logger.warning(f"Prompt was: {prompt[:500]}")
                return {
                    "action": "generate_text",
                    "result": {"error": "Model returned empty response"},
                    "model_used": model.name if hasattr(model, 'name') else str(model),
                    "success": False
                }
            
            return {
                "action": "generate_text",
                "result": response,
                "model_used": model.name if hasattr(model, 'name') else str(model),
                "success": True
            }
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error in text generation: {str(e)}")
            return {
                "action": "generate_text",
                "success": False,
                "error": str(e)
            }
    
    async def _handle_mcp_server(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle MCP server operations."""
        return await self.mcp_server_tool.execute(**task.parameters, context=context)
    
    async def _handle_mcp_memory(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle MCP memory operations."""
        return await self.mcp_memory_tool.execute(**task.parameters, context=context)
    
    async def _handle_mcp_resource(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle MCP resource operations."""
        return await self.mcp_resource_tool.execute(**task.parameters, context=context)
    
    async def _handle_visualization(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle visualization operations."""
        params = task.parameters.copy()
        if "action" not in params and task.action:
            params["action"] = task.action
        return await self.visualization_tool.execute(**params)
    
    async def _handle_python_executor(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle Python code execution."""
        params = task.parameters.copy()
        
        # Pass context to the code if needed
        code = params.get("code", "")
        if "context" in code:
            # Inject context as a variable
            context_setup = f"""
import json
context = {repr(context)}
"""
            params["code"] = context_setup + code
        
        result = await self.python_executor_tool.execute(**params)
        
        # Try to parse stdout as JSON for structured results
        if result.get("success") and result.get("stdout"):
            try:
                import json
                result["result"] = json.loads(result["stdout"].strip())
            except:
                # If not JSON, just use the stdout as result
                result["result"] = result["stdout"].strip()
        
        return result

    async def _handle_recursion_control(self, task: Task, context: Dict[str, Any]) -> Any:
        """Handle recursion control operations."""
        params = task.parameters.copy()
        
        # Pass the action from the task
        params['action'] = task.action
        
        # Pass the execution_id from context if available
        if 'execution_id' in context and 'context_id' not in params:
            params['context_id'] = context['execution_id']
        
        # Execute the recursion control tool
        return await self.recursion_control_tool.execute(**params)

