"""Enhanced task executor with smart tool discovery and automatic execution."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List

from ..core.model import Model
from ..tools.base import Tool
from ..tools.discovery import ToolDiscoveryEngine, ToolMatch
from .pipeline_spec import TaskSpec
from .task_executor import UniversalTaskExecutor

logger = logging.getLogger(__name__)


class EnhancedTaskExecutor(UniversalTaskExecutor):
    """Enhanced task executor with smart tool discovery and orchestration."""

    def __init__(self, model_registry=None, tool_registry=None):
        super().__init__(model_registry, tool_registry)
        self.discovery_engine = ToolDiscoveryEngine(tool_registry)
        self.execution_strategies = self._build_execution_strategies()

    def _build_execution_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Build execution strategies for different tool combinations."""
        return {
            "sequential": {
                "description": "Execute tools one after another",
                "max_tools": 5,
                "timeout": 300,
            },
            "parallel": {
                "description": "Execute independent tools in parallel",
                "max_tools": 3,
                "timeout": 180,
            },
            "pipeline": {
                "description": "Execute tools in data pipeline fashion",
                "max_tools": 4,
                "timeout": 240,
            },
            "adaptive": {
                "description": "Dynamically choose strategy based on tools",
                "max_tools": 6,
                "timeout": 360,
            },
        }

    async def execute_task(self, task_spec: TaskSpec, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with enhanced tool discovery and orchestration."""
        logger.info(f"Enhanced execution for task: {task_spec.id}")

        try:
            # 1. Resolve AUTO tags and get enhanced task specification
            enhanced_spec = await self._get_enhanced_task_spec(task_spec, context)

            # 2. Discover optimal tools for the task
            tool_matches = await self._discover_optimal_tools(enhanced_spec, context)

            # 3. Select execution strategy
            strategy = self._select_execution_strategy(tool_matches, enhanced_spec)

            # 4. Execute with selected strategy
            result = await self._execute_with_strategy(
                enhanced_spec, tool_matches, strategy, context
            )

            # 5. Post-process and validate results
            final_result = await self._post_process_results(result, task_spec, enhanced_spec)

            logger.info(f"Enhanced task {task_spec.id} completed successfully")
            return final_result

        except Exception as e:
            logger.error(f"Enhanced task {task_spec.id} failed: {str(e)}")
            return await self._handle_enhanced_error(task_spec, e, context)

    async def _get_enhanced_task_spec(
        self, task_spec: TaskSpec, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get enhanced task specification with resolved AUTO tags and enriched metadata."""
        if task_spec.has_auto_tags():
            # Resolve AUTO tags
            auto_content = task_spec.extract_auto_content()
            resolved_spec = await self.auto_resolver.resolve_auto_tag(auto_content, context)
        else:
            resolved_spec = {
                "prompt": task_spec.action,
                "tools": task_spec.tools or [],
                "output_format": "structured",
            }

        # Enrich with task metadata
        resolved_spec.update(
            {
                "task_id": task_spec.id,
                "original_action": task_spec.action,
                "context_variables": list(context.keys()),
                "dependencies": task_spec.depends_on,
                "model_requirements": task_spec.model_requirements,
            }
        )

        return resolved_spec

    async def _discover_optimal_tools(
        self, enhanced_spec: Dict[str, Any], context: Dict[str, Any]
    ) -> List[ToolMatch]:
        """Discover optimal tools for enhanced task execution."""
        action_description = enhanced_spec.get("prompt", enhanced_spec.get("original_action", ""))

        # Use discovery engine to find tools
        discovered_tools = self.discovery_engine.discover_tools_for_action(
            action_description, context
        )

        # Add explicitly specified tools
        explicit_tools = enhanced_spec.get("tools", [])
        for tool_name in explicit_tools:
            if tool_name not in [match.tool_name for match in discovered_tools]:
                discovered_tools.append(
                    ToolMatch(
                        tool_name=tool_name,
                        confidence=1.0,
                        reasoning="Explicitly specified in task",
                        parameters={},
                    )
                )

        # Validate tool availability and capabilities
        validated_tools = await self._validate_discovered_tools(discovered_tools, enhanced_spec)

        logger.debug(f"Discovered {len(validated_tools)} validated tools for task")
        return validated_tools

    async def _validate_discovered_tools(
        self, tool_matches: List[ToolMatch], enhanced_spec: Dict[str, Any]
    ) -> List[ToolMatch]:
        """Validate that discovered tools are available and capable."""
        validated = []
        available_tools = self.tool_registry.list_tools()

        for match in tool_matches:
            if match.tool_name in available_tools:
                tool = self.tool_registry.get_tool(match.tool_name)
                if tool and await self._validate_tool_capability(tool, enhanced_spec):
                    validated.append(match)
                else:
                    logger.warning(f"Tool {match.tool_name} failed capability validation")
            else:
                logger.warning(f"Tool {match.tool_name} not available in registry")
                # Try to suggest alternatives
                suggestions = self.discovery_engine.suggest_missing_tools([match.tool_name])
                if suggestions.get(match.tool_name):
                    alt_tool = suggestions[match.tool_name][0]
                    logger.info(f"Using alternative tool {alt_tool} for {match.tool_name}")
                    match.tool_name = alt_tool
                    match.reasoning += f" (alternative for {match.tool_name})"
                    validated.append(match)

        return validated

    async def _validate_tool_capability(self, tool: Tool, enhanced_spec: Dict[str, Any]) -> bool:
        """Validate that a tool can handle the required task."""
        # Basic validation - can be enhanced with more sophisticated checks
        try:
            # Check if tool has required methods
            if not hasattr(tool, "execute"):
                return False

            # Check tool capabilities if available
            if hasattr(tool, "capabilities"):
                required_caps = enhanced_spec.get("required_capabilities", [])
                if required_caps and not any(cap in tool.capabilities for cap in required_caps):
                    return False

            return True

        except Exception as e:
            logger.warning(f"Tool validation failed for {tool.name}: {e}")
            return False

    def _select_execution_strategy(
        self, tool_matches: List[ToolMatch], enhanced_spec: Dict[str, Any]
    ) -> str:
        """Select optimal execution strategy based on tools and task."""
        num_tools = len(tool_matches)

        if num_tools == 0:
            return "model_only"
        elif num_tools == 1:
            return "sequential"
        elif num_tools <= 2:
            # Check if tools can run in parallel
            if self._can_run_parallel(tool_matches):
                return "parallel"
            else:
                return "sequential"
        elif num_tools <= 4:
            # Check if this looks like a data pipeline
            if self._is_pipeline_pattern(tool_matches, enhanced_spec):
                return "pipeline"
            else:
                return "sequential"
        else:
            return "adaptive"

    def _can_run_parallel(self, tool_matches: List[ToolMatch]) -> bool:
        """Check if tools can be executed in parallel."""
        # Simple heuristic: tools with different primary functions can run in parallel
        tool_types = set()

        for match in tool_matches:
            if "search" in match.tool_name:
                tool_types.add("search")
            elif "data" in match.tool_name:
                tool_types.add("data")
            elif "report" in match.tool_name:
                tool_types.add("report")
            elif "browser" in match.tool_name:
                tool_types.add("web")
            else:
                tool_types.add("other")

        # If all tools are different types, they can likely run in parallel
        return len(tool_types) == len(tool_matches)

    def _is_pipeline_pattern(
        self, tool_matches: List[ToolMatch], enhanced_spec: Dict[str, Any]
    ) -> bool:
        """Check if tools form a data pipeline pattern."""
        action = enhanced_spec.get("prompt", "").lower()

        # Common pipeline patterns
        pipeline_indicators = [
            "search" in action and "analyze" in action,
            "extract" in action and "process" in action,
            "collect" in action and "summarize" in action,
            "gather" in action and "report" in action,
        ]

        return any(pipeline_indicators)

    async def _execute_with_strategy(
        self,
        enhanced_spec: Dict[str, Any],
        tool_matches: List[ToolMatch],
        strategy: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute tools using the selected strategy."""
        logger.info(f"Executing with strategy: {strategy}")

        if strategy == "model_only":
            return await self._execute_model_only_enhanced(enhanced_spec, context)
        elif strategy == "sequential":
            return await self._execute_sequential(enhanced_spec, tool_matches, context)
        elif strategy == "parallel":
            return await self._execute_parallel(enhanced_spec, tool_matches, context)
        elif strategy == "pipeline":
            return await self._execute_pipeline(enhanced_spec, tool_matches, context)
        elif strategy == "adaptive":
            return await self._execute_adaptive(enhanced_spec, tool_matches, context)
        else:
            raise ValueError(f"Unknown execution strategy: {strategy}")

    async def _execute_model_only_enhanced(
        self, enhanced_spec: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute task using model only with enhanced prompt generation."""
        prompt = self._resolve_template_variables(enhanced_spec["prompt"], context)

        # Enhance prompt with context information
        enhanced_prompt = self._enhance_prompt_with_context(prompt, context)

        if self.model_registry:
            model = self._select_model_for_enhanced_task(enhanced_spec)

            output_format = enhanced_spec.get("output_format", "structured")
            if output_format in ["json", "structured"]:
                try:
                    schema = self._generate_output_schema(enhanced_spec)
                    result = await model.generate_structured(enhanced_prompt, schema)
                except:
                    result = await model.generate(enhanced_prompt)
            else:
                result = await model.generate(enhanced_prompt)

            return {"result": result, "method": "model_only", "prompt_used": enhanced_prompt}
        else:
            raise ValueError("No model registry available for model-only execution")

    async def _execute_sequential(
        self, enhanced_spec: Dict[str, Any], tool_matches: List[ToolMatch], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute tools sequentially with result chaining."""
        results = {"method": "sequential", "tool_results": {}}
        current_context = context.copy()

        for i, match in enumerate(tool_matches):
            logger.debug(f"Executing tool {i+1}/{len(tool_matches)}: {match.tool_name}")

            try:
                tool = self.tool_registry.get_tool(match.tool_name)
                if not tool:
                    continue

                # Prepare parameters for this tool
                tool_params = self._prepare_enhanced_tool_params(
                    tool, match, enhanced_spec, current_context
                )

                # Execute tool
                tool_result = await tool.execute(**tool_params)
                results["tool_results"][match.tool_name] = tool_result

                # Update context with result for next tool
                current_context[f"{match.tool_name}_result"] = tool_result
                if isinstance(tool_result, dict):
                    current_context.update(
                        {
                            f"{match.tool_name}_{k}": v
                            for k, v in tool_result.items()
                            if k not in ["success", "timestamp"]
                        }
                    )

            except Exception as e:
                logger.warning(f"Tool {match.tool_name} failed: {e}")
                results["tool_results"][f"{match.tool_name}_error"] = str(e)

        # Use last successful result as primary result
        successful_results = [
            result
            for result in results["tool_results"].values()
            if isinstance(result, dict) and result.get("success", True)
        ]

        if successful_results:
            results["result"] = successful_results[-1]

        return results

    async def _execute_parallel(
        self, enhanced_spec: Dict[str, Any], tool_matches: List[ToolMatch], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute tools in parallel and combine results."""
        logger.debug(f"Executing {len(tool_matches)} tools in parallel")

        # Create tasks for parallel execution
        tasks = []
        for match in tool_matches:
            tool = self.tool_registry.get_tool(match.tool_name)
            if tool:
                params = self._prepare_enhanced_tool_params(tool, match, enhanced_spec, context)
                task = asyncio.create_task(self._execute_single_tool(tool, params, match.tool_name))
                tasks.append((match.tool_name, task))

        # Wait for all tasks to complete
        results = {"method": "parallel", "tool_results": {}}
        for tool_name, task in tasks:
            try:
                result = await task
                results["tool_results"][tool_name] = result
            except Exception as e:
                logger.warning(f"Parallel tool {tool_name} failed: {e}")
                results["tool_results"][f"{tool_name}_error"] = str(e)

        # Combine parallel results intelligently
        combined_result = self._combine_parallel_results(results["tool_results"])
        results["result"] = combined_result

        return results

    async def _execute_pipeline(
        self, enhanced_spec: Dict[str, Any], tool_matches: List[ToolMatch], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute tools as a data pipeline."""
        logger.debug("Executing tools as data pipeline")

        # Order tools for pipeline execution
        ordered_tools = self._order_tools_for_pipeline(tool_matches, enhanced_spec)

        # Execute as pipeline with data flow
        results = {"method": "pipeline", "tool_results": {}, "data_flow": []}
        pipeline_data = None

        for i, match in enumerate(ordered_tools):
            tool = self.tool_registry.get_tool(match.tool_name)
            if not tool:
                continue

            # Prepare parameters with pipeline data
            params = self._prepare_enhanced_tool_params(tool, match, enhanced_spec, context)
            if pipeline_data is not None:
                params["data"] = pipeline_data
                params["input_data"] = pipeline_data

            try:
                result = await tool.execute(**params)
                results["tool_results"][match.tool_name] = result
                results["data_flow"].append(
                    {
                        "step": i + 1,
                        "tool": match.tool_name,
                        "input_size": len(str(pipeline_data)) if pipeline_data else 0,
                        "output_size": len(str(result)) if result else 0,
                    }
                )

                # Extract data for next stage
                pipeline_data = self._extract_pipeline_data(result)

            except Exception as e:
                logger.warning(f"Pipeline tool {match.tool_name} failed: {e}")
                results["tool_results"][f"{match.tool_name}_error"] = str(e)
                break

        results["result"] = pipeline_data
        return results

    async def _execute_adaptive(
        self, enhanced_spec: Dict[str, Any], tool_matches: List[ToolMatch], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute tools using adaptive strategy."""
        logger.debug("Executing with adaptive strategy")

        # Group tools by type and execute in intelligent order
        tool_groups = self._group_tools_by_type(tool_matches)
        results = {"method": "adaptive", "tool_results": {}, "execution_plan": []}

        for group_name, group_tools in tool_groups.items():
            logger.debug(f"Executing tool group: {group_name}")

            if len(group_tools) == 1:
                # Single tool - execute directly
                match = group_tools[0]
                result = await self._execute_single_tool_enhanced(match, enhanced_spec, context)
                results["tool_results"][match.tool_name] = result
            else:
                # Multiple tools - choose best sub-strategy
                if group_name in ["search", "data"]:
                    # These can often run in parallel
                    group_result = await self._execute_parallel(enhanced_spec, group_tools, context)
                else:
                    # Default to sequential
                    group_result = await self._execute_sequential(
                        enhanced_spec, group_tools, context
                    )

                results["tool_results"].update(group_result["tool_results"])

            results["execution_plan"].append(
                {
                    "group": group_name,
                    "tools": [t.tool_name for t in group_tools],
                    "strategy": (
                        "parallel"
                        if len(group_tools) > 1 and group_name in ["search", "data"]
                        else "sequential"
                    ),
                }
            )

        # Combine all results
        all_results = [
            result
            for result in results["tool_results"].values()
            if isinstance(result, dict) and result.get("success", True)
        ]

        if all_results:
            results["result"] = self._intelligently_combine_results(all_results, enhanced_spec)

        return results

    def _group_tools_by_type(self, tool_matches: List[ToolMatch]) -> Dict[str, List[ToolMatch]]:
        """Group tools by their primary function type."""
        groups = {}

        for match in tool_matches:
            tool_name = match.tool_name

            if "search" in tool_name or "web" in tool_name:
                group = "search"
            elif "data" in tool_name or "process" in tool_name:
                group = "data"
            elif "report" in tool_name or "generate" in tool_name:
                group = "generate"
            elif "browser" in tool_name or "scrape" in tool_name:
                group = "web"
            elif "file" in tool_name or "filesystem" in tool_name:
                group = "file"
            else:
                group = "other"

            if group not in groups:
                groups[group] = []
            groups[group].append(match)

        return groups

    async def _execute_single_tool(self, tool: Tool, params: Dict[str, Any], tool_name: str) -> Any:
        """Execute a single tool with error handling."""
        try:
            return await tool.execute(**params)
        except Exception as e:
            logger.warning(f"Single tool execution failed for {tool_name}: {e}")
            raise

    async def _execute_single_tool_enhanced(
        self, match: ToolMatch, enhanced_spec: Dict[str, Any], context: Dict[str, Any]
    ) -> Any:
        """Execute a single tool with enhanced parameter preparation."""
        tool = self.tool_registry.get_tool(match.tool_name)
        if not tool:
            raise ValueError(f"Tool {match.tool_name} not found")

        params = self._prepare_enhanced_tool_params(tool, match, enhanced_spec, context)
        return await self._execute_single_tool(tool, params, match.tool_name)

    def _prepare_enhanced_tool_params(
        self, tool: Tool, match: ToolMatch, enhanced_spec: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare enhanced parameters for tool execution."""
        params = match.parameters.copy()

        # Add basic context
        prompt = enhanced_spec.get("prompt", "")
        resolved_prompt = self._resolve_template_variables(prompt, context)

        # Tool-specific parameter enhancement
        if "search" in match.tool_name:
            params.update(
                {
                    "query": params.get("query", resolved_prompt),
                    "max_results": params.get("max_results", 10),
                    "include_snippets": True,
                }
            )

        elif "data" in match.tool_name:
            params.update(
                {
                    "action": params.get("action", "analyze"),
                    "data": context.get("data", resolved_prompt),
                }
            )

        elif "report" in match.tool_name:
            params.update(
                {
                    "title": params.get("title", "Generated Report"),
                    "content": params.get("content", resolved_prompt),
                    "format": params.get("format", "markdown"),
                }
            )

        elif "browser" in match.tool_name:
            params.update(
                {
                    "action": params.get("action", "scrape"),
                    "url": context.get("url", params.get("url")),
                }
            )

        elif "file" in match.tool_name:
            params.update(
                {
                    "action": params.get("action", "read"),
                    "path": context.get("path", params.get("path")),
                }
            )

        # Add general context variables
        for key, value in context.items():
            if key not in params and not key.startswith("_"):
                params[key] = value

        # Remove None values
        return {k: v for k, v in params.items() if v is not None}

    def _enhance_prompt_with_context(self, prompt: str, context: Dict[str, Any]) -> str:
        """Enhance prompt with relevant context information."""
        enhanced_prompt = prompt

        # Add context summary if there's relevant data
        context_info = []

        if "data" in context:
            data = context["data"]
            if isinstance(data, (list, dict)):
                context_info.append(f"Available data: {type(data).__name__} with {len(data)} items")

        if "url" in context:
            context_info.append(f"URL context: {context['url']}")

        if "file" in context or "path" in context:
            file_info = context.get("file", context.get("path", ""))
            context_info.append(f"File context: {file_info}")

        if context_info:
            enhanced_prompt += f"\n\nContext: {'; '.join(context_info)}"

        return enhanced_prompt

    def _select_model_for_enhanced_task(self, enhanced_spec: Dict[str, Any]) -> Model:
        """Select optimal model for enhanced task execution."""
        if not self.model_registry:
            raise ValueError("No model registry available")

        # Use model requirements from enhanced spec
        model_reqs = enhanced_spec.get("model_requirements", {})
        required_capabilities = model_reqs.get("capabilities", [])

        # Infer capabilities from task type
        prompt = enhanced_spec.get("prompt", "").lower()
        if "analyze" in prompt or "examine" in prompt:
            required_capabilities.append("reasoning")
        if "generate" in prompt or "create" in prompt:
            required_capabilities.append("generation")
        if "code" in prompt or "program" in prompt:
            required_capabilities.append("coding")

        # Select best model
        if hasattr(self.model_registry, "get_best_model"):
            return self.model_registry.get_best_model(required_capabilities)
        else:
            return self._select_model_for_task(enhanced_spec)

    def _generate_output_schema(self, enhanced_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Generate appropriate output schema for structured results."""
        output_format = enhanced_spec.get("output_format", "structured")

        if output_format == "json":
            return {
                "type": "object",
                "properties": {
                    "result": {"type": "string"},
                    "data": {"type": "array"},
                    "insights": {"type": "array"},
                },
            }
        else:
            return {
                "type": "object",
                "properties": {"result": {"type": "string"}, "summary": {"type": "string"}},
            }

    def _combine_parallel_results(self, tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligently combine results from parallel tool execution."""
        combined = {
            "combined_results": True,
            "individual_results": tool_results,
            "summary": "Combined results from parallel execution",
        }

        # Extract common result patterns
        all_data = []
        all_insights = []
        all_content = []

        for tool_name, result in tool_results.items():
            if isinstance(result, dict):
                if "data" in result:
                    all_data.extend(
                        result["data"] if isinstance(result["data"], list) else [result["data"]]
                    )
                if "insights" in result:
                    all_insights.extend(
                        result["insights"]
                        if isinstance(result["insights"], list)
                        else [result["insights"]]
                    )
                if "content" in result:
                    all_content.append(result["content"])
                if "result" in result:
                    all_content.append(str(result["result"]))

        if all_data:
            combined["data"] = all_data
        if all_insights:
            combined["insights"] = all_insights
        if all_content:
            combined["content"] = "\n\n".join(all_content)
            combined["result"] = combined["content"]

        return combined

    def _order_tools_for_pipeline(
        self, tool_matches: List[ToolMatch], enhanced_spec: Dict[str, Any]
    ) -> List[ToolMatch]:
        """Order tools for optimal pipeline execution."""
        # Define tool execution order priorities
        order_priority = {
            "web-search": 1,
            "headless-browser": 1,
            "filesystem": 1,
            "data-processing": 2,
            "validation": 2,
            "report-generator": 3,
            "terminal": 4,
        }

        # Sort by priority, then by confidence
        return sorted(
            tool_matches, key=lambda x: (order_priority.get(x.tool_name, 99), -x.confidence)
        )

    def _extract_pipeline_data(self, result: Any) -> Any:
        """Extract data from tool result for pipeline continuation."""
        if isinstance(result, dict):
            # Look for common data fields
            for field in ["data", "results", "content", "output", "result"]:
                if field in result:
                    return result[field]

            # Return the whole result if no specific field found
            return result
        else:
            return result

    def _intelligently_combine_results(
        self, results: List[Dict[str, Any]], enhanced_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Intelligently combine multiple results based on task context."""
        if not results:
            return {}

        if len(results) == 1:
            return results[0]

        combined = {
            "combined_results": True,
            "result_count": len(results),
            "combination_method": "intelligent",
        }

        # Analyze task to determine best combination strategy
        prompt = enhanced_spec.get("prompt", "").lower()

        if "search" in prompt and "analyze" in prompt:
            # Search + analysis task
            search_results = [r for r in results if "search" in str(r) or "data" in r]
            analysis_results = [r for r in results if "insights" in r or "analysis" in r]

            if search_results:
                combined["search_data"] = search_results[0]
            if analysis_results:
                combined["analysis"] = analysis_results[0]
                combined["result"] = analysis_results[0]
            else:
                combined["result"] = search_results[0] if search_results else results[0]

        elif "generate" in prompt or "create" in prompt:
            # Generation task - use last result
            combined["result"] = results[-1]

        else:
            # Default combination - merge data and use best result
            all_data = []
            for result in results:
                if isinstance(result, dict) and "data" in result:
                    data = result["data"]
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)

            if all_data:
                combined["data"] = all_data

            # Use result with highest confidence or most complete data
            best_result = max(results, key=lambda x: len(str(x)) if isinstance(x, dict) else 0)
            combined["result"] = best_result

        return combined

    async def _post_process_results(
        self, result: Dict[str, Any], task_spec: TaskSpec, enhanced_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Post-process and validate execution results."""
        structured_result = {
            "task_id": task_spec.id,
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "execution_method": result.get("method", "unknown"),
            "result": result.get("result", result),
        }

        # Add execution metadata
        if "tool_results" in result:
            structured_result["tool_results"] = result["tool_results"]
            structured_result["tools_used"] = list(result["tool_results"].keys())

        if "execution_plan" in result:
            structured_result["execution_plan"] = result["execution_plan"]

        # Validate result completeness
        if not structured_result["result"]:
            structured_result["success"] = False
            structured_result["error"] = "No valid result produced"

        return structured_result

    async def _handle_enhanced_error(
        self, task_spec: TaskSpec, error: Exception, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle errors in enhanced execution with better recovery."""
        logger.error(f"Enhanced execution error for {task_spec.id}: {str(error)}")

        # Try fallback execution
        if task_spec.on_error:
            return await super()._handle_task_error(task_spec, error, context)

        # Enhanced error response
        return {
            "task_id": task_spec.id,
            "success": False,
            "error": str(error),
            "error_type": type(error).__name__,
            "timestamp": datetime.now().isoformat(),
            "execution_method": "enhanced",
            "recovery_attempted": bool(task_spec.on_error),
        }
