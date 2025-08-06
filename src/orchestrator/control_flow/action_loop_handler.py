"""Action loop handler for sequential task iteration with termination conditions."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from ..core.action_loop_task import ActionLoopTask
from ..core.action_loop_context import ActionResult, EnhancedLoopContext, IterationContext
from ..tools.base import ToolRegistry, default_registry
from ..control_flow.enhanced_condition_evaluator import EnhancedConditionEvaluator
from ..control_flow.auto_resolver import ControlFlowAutoResolver
from ..core.template_manager import TemplateManager
from ..models import get_model_registry


class ActionLoopHandler:
    """Handler for executing action loops with tool integration and termination conditions."""
    
    def __init__(
        self, 
        tool_registry: Optional[ToolRegistry] = None,
        auto_resolver: Optional[ControlFlowAutoResolver] = None,
        template_manager: Optional[TemplateManager] = None
    ):
        """Initialize action loop handler.
        
        Args:
            tool_registry: Registry of available tools
            auto_resolver: AUTO tag resolver for conditions and actions
            template_manager: Template manager for rendering templates
        """
        self.tool_registry = tool_registry or default_registry
        self.auto_resolver = auto_resolver or ControlFlowAutoResolver(get_model_registry())
        self.template_manager = template_manager or TemplateManager()
        
        # Enhanced condition evaluator for termination conditions
        self.condition_evaluator = EnhancedConditionEvaluator(self.auto_resolver)
        
        self.logger = logging.getLogger(__name__)
    
    async def execute_action_loop(
        self, 
        task: ActionLoopTask, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the complete action loop with termination conditions.
        
        Args:
            task: The action loop task to execute
            context: Pipeline execution context
            
        Returns:
            Dictionary containing loop execution results
        """
        self.logger.info(f"Starting action loop execution: {task.id}")
        
        # Initialize loop context
        loop_context = EnhancedLoopContext(
            loop_id=task.id,
            loop_metadata={"task_metadata": task.metadata}
        )
        
        # Update template manager with initial context
        self._update_template_context(context, loop_context)
        
        try:
            # Execute loop iterations
            while task.can_continue_iteration():
                # Check iteration timeout before starting
                if task.should_check_timeout():
                    task.terminated_by = "timeout"
                    self.logger.warning(f"Loop {task.id} iteration {task.current_iteration} timed out")
                    break
                
                # Start iteration
                task.start_iteration()
                iteration_ctx = loop_context.start_iteration()
                
                self.logger.info(
                    f"Starting iteration {task.current_iteration + 1}/{task.max_iterations} "
                    f"for loop {task.id}"
                )
                
                # Execute all actions in the iteration
                iteration_success = await self._execute_iteration(
                    task, iteration_ctx, loop_context, context
                )
                
                # Complete iteration
                task.complete_iteration(iteration_ctx.get_all_results())
                loop_context.complete_iteration()
                
                # Check for break on error
                if not iteration_success and task.break_on_error:
                    task.terminated_by = "error"
                    self.logger.warning(f"Loop {task.id} terminated due to error in iteration {task.current_iteration}")
                    break
                
                # Check termination condition
                should_terminate = await self._check_termination_condition(
                    task, loop_context, context
                )
                
                if should_terminate:
                    task.terminated_by = "condition"
                    self.logger.info(f"Loop {task.id} terminated by condition after {task.current_iteration} iterations")
                    break
            
            # Check if terminated by max iterations
            if task.current_iteration >= task.max_iterations and task.terminated_by is None:
                task.terminated_by = "max_iterations"
                self.logger.warning(f"Loop {task.id} reached maximum iterations ({task.max_iterations})")
            
            # Build final results
            final_results = self._build_final_results(task, loop_context)
            
            self.logger.info(
                f"Action loop {task.id} completed: {task.current_iteration} iterations, "
                f"terminated by {task.terminated_by}"
            )
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error executing action loop {task.id}: {e}", exc_info=True)
            task.terminated_by = "error"
            raise
    
    async def _execute_iteration(
        self,
        task: ActionLoopTask,
        iteration_ctx: IterationContext,
        loop_context: EnhancedLoopContext,
        base_context: Dict[str, Any]
    ) -> bool:
        """Execute all actions in a single iteration.
        
        Returns:
            bool: True if iteration succeeded, False if any action failed
        """
        iteration_success = True
        
        # Build iteration context for template rendering
        iteration_template_context = self._build_iteration_context(
            base_context, loop_context, iteration_ctx
        )
        
        # Execute each action in sequence
        for action_idx, action_def in enumerate(task.action_loop):
            action_name = action_def.get("name", f"action_{action_idx}")
            
            try:
                self.logger.debug(
                    f"Executing action '{action_name}' in iteration {task.current_iteration} "
                    f"of loop {task.id}"
                )
                
                # Execute the action
                action_result = await self._execute_action_in_loop(
                    action_def, action_name, iteration_template_context, task
                )
                
                # Record result in iteration context
                iteration_ctx.add_action_result(action_result)
                
                # Update template context with new result for subsequent actions
                if action_result.success and action_name:
                    iteration_template_context[action_name] = action_result.result
                
                # Track tool execution
                if action_result.tool_used:
                    task.record_tool_execution(
                        action_result.tool_used, 
                        action_result.success,
                        action_result.error
                    )
                    
                    # Record in loop context
                    loop_context.record_tool_result(
                        action_result.tool_used,
                        action_result.result,
                        action_result.execution_time
                    )
                    
                    if not action_result.success:
                        loop_context.record_tool_error(action_result.tool_used)
                
                # Check if action failed
                if not action_result.success:
                    iteration_success = False
                    self.logger.warning(
                        f"Action '{action_name}' failed in iteration {task.current_iteration}: "
                        f"{action_result.error}"
                    )
                
            except Exception as e:
                # Record action failure
                action_result = ActionResult(
                    action_name=action_name,
                    action_def=action_def,
                    status="failed",
                    error=str(e),
                    execution_time=0.0
                )
                iteration_ctx.add_action_result(action_result)
                iteration_success = False
                
                self.logger.error(
                    f"Error executing action '{action_name}' in iteration {task.current_iteration}: {e}",
                    exc_info=True
                )
        
        return iteration_success
    
    async def _execute_action_in_loop(
        self,
        action_def: Dict[str, Any],
        action_name: str,
        context: Dict[str, Any],
        task: ActionLoopTask
    ) -> ActionResult:
        """Execute a single action within a loop iteration."""
        start_time = time.time()
        
        try:
            # Check if action specifies a tool
            tool_name = action_def.get("tool")
            if tool_name:
                result = await self._execute_tool_action(
                    tool_name, action_def, context
                )
            else:
                # Route action based on content
                result = await self._route_and_execute_action(
                    action_def, context
                )
            
            execution_time = time.time() - start_time
            
            return ActionResult(
                action_name=action_name,
                action_def=action_def,
                status="completed" if result.get("success", True) else "failed",
                result=result,
                tool_used=result.get("tool_used"),
                execution_time=execution_time,
                error=result.get("error") if not result.get("success", True) else None
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ActionResult(
                action_name=action_name,
                action_def=action_def,
                status="failed",
                error=str(e),
                execution_time=execution_time
            )
    
    async def _execute_tool_action(
        self,
        tool_name: str,
        action_def: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute action using specific tool."""
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found in registry")
        
        # Prepare parameters
        params = action_def.get("parameters", {}).copy()
        
        # Add action if not in parameters
        if "action" in action_def and "action" not in params:
            params["action"] = action_def["action"]
        
        # Render templates in parameters
        if params:
            params = self.template_manager.deep_render(params, additional_context=context)
        
        # Add template manager to context for runtime template resolution
        params["template_manager"] = self.template_manager
        
        # Execute tool
        result = await tool.execute(**params)
        
        # Add tool metadata to result
        if isinstance(result, dict):
            result["tool_used"] = tool_name
        else:
            result = {"result": result, "tool_used": tool_name, "success": True}
        
        return result
    
    async def _route_and_execute_action(
        self,
        action_def: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Route and execute action based on action content."""
        action_str = str(action_def.get("action", "")).strip()
        
        # Check for AUTO tags - resolve them first
        if "<AUTO>" in action_str:
            resolved_action = await self.auto_resolver._resolve_auto_tags(
                action_str, context, {}
            )
            action_def = action_def.copy()
            action_def["action"] = resolved_action
            action_str = resolved_action
        
        # Route to appropriate tool based on action patterns
        tool_name = self._determine_tool_from_action(action_str.lower())
        
        if tool_name:
            return await self._execute_tool_action(tool_name, action_def, context)
        
        # Fallback: try to execute as model-based action
        return await self._execute_model_action(action_def, context)
    
    def _determine_tool_from_action(self, action_str: str) -> Optional[str]:
        """Determine appropriate tool based on action string."""
        # Filesystem operations
        if any(pattern in action_str for pattern in [
            "write", "read", "save", "file", "create file", "list files",
            "copy", "move", "delete file"
        ]):
            return "filesystem"
        
        # Web operations  
        if any(pattern in action_str for pattern in [
            "search", "web search", "google", "find online"
        ]):
            return "web-search"
        
        if any(pattern in action_str for pattern in [
            "scrape", "browse", "visit", "web page", "download page"
        ]):
            return "headless-browser"
        
        # Terminal operations
        if any(pattern in action_str for pattern in [
            "run command", "execute", "shell", "terminal", "bash"
        ]):
            return "terminal"
        
        # Data processing
        if any(pattern in action_str for pattern in [
            "process data", "transform", "parse", "extract", "filter"
        ]):
            return "data-processing"
        
        # Validation
        if any(pattern in action_str for pattern in [
            "validate", "verify", "check", "test"
        ]):
            return "validation"
        
        return None
    
    async def _execute_model_action(
        self,
        action_def: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute action using model-based approach."""
        action_str = action_def.get("action", "")
        
        # Use AUTO resolver to generate response
        try:
            result = await self.auto_resolver._resolve_auto_tags(
                f"<AUTO>{action_str}</AUTO>", context, {}
            )
            
            return {
                "result": result,
                "success": True,
                "tool_used": "auto_resolver"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "success": False,
                "tool_used": "auto_resolver"
            }
    
    async def _check_termination_condition(
        self,
        task: ActionLoopTask,
        loop_context: EnhancedLoopContext,
        base_context: Dict[str, Any]
    ) -> bool:
        """Check if loop should terminate based on condition."""
        if not task.has_termination_condition():
            return False
        
        condition = task.get_termination_condition()
        
        # Build evaluation context
        eval_context = base_context.copy()
        eval_context.update(loop_context.to_template_dict())
        
        # Add current iteration results
        if loop_context.current_iteration_context:
            eval_context.update(loop_context.current_iteration_context.get_all_results())
        
        try:
            # Evaluate condition using enhanced condition evaluator
            result = await self.condition_evaluator.evaluate_condition(
                condition=condition,
                context=eval_context,
                step_results={},
                iteration=task.current_iteration,
                condition_type="until" if task.is_until_condition() else "while"
            )
            
            # Record condition evaluation
            loop_context.record_condition_evaluation(
                condition=condition,
                result=result.result,
                metadata={
                    "condition_type": "until" if task.is_until_condition() else "while",
                    "resolved_expression": result.resolved_expression,
                    "evaluation_time": result.evaluation_time
                }
            )
            
            # For 'until' conditions: terminate when condition is true
            # For 'while' conditions: terminate when condition is false
            if task.is_until_condition():
                return result.result  # Terminate when true
            else:
                return not result.result  # Terminate when false (while condition failed)
        
        except Exception as e:
            self.logger.error(f"Error evaluating termination condition: {e}")
            # On evaluation error, terminate loop for safety
            loop_context.record_condition_evaluation(
                condition=condition,
                result=True,
                metadata={"error": str(e), "terminated_on_error": True}
            )
            return True
    
    def _build_iteration_context(
        self,
        base_context: Dict[str, Any],
        loop_context: EnhancedLoopContext,
        iteration_ctx: IterationContext
    ) -> Dict[str, Any]:
        """Build context for template rendering within iteration."""
        context = base_context.copy()
        
        # Add loop context
        context.update(loop_context.to_template_dict())
        
        # Add current iteration results
        context.update(iteration_ctx.get_all_results())
        
        # Add previous iteration results for easy access
        if loop_context.has_previous_results:
            previous_results = loop_context.get_previous_results(1)
            if previous_results:
                for key, value in previous_results[0].items():
                    context[f"prev_{key}"] = value
        
        return context
    
    def _update_template_context(
        self,
        base_context: Dict[str, Any],
        loop_context: EnhancedLoopContext
    ) -> None:
        """Update template manager with current context."""
        template_context = base_context.copy()
        template_context.update(loop_context.to_template_dict())
        
        # Update template manager context
        for key, value in template_context.items():
            self.template_manager.register_context(key, value)
    
    def _build_final_results(
        self,
        task: ActionLoopTask,
        loop_context: EnhancedLoopContext
    ) -> Dict[str, Any]:
        """Build final results dictionary."""
        # Set termination reason in loop context
        loop_context.termination_reason = task.terminated_by
        
        # Get final summary
        final_summary = loop_context.get_final_summary()
        
        # Collect all iteration results
        all_results = []
        for iteration_ctx in loop_context.iterations:
            if iteration_ctx.is_complete:
                all_results.append({
                    "iteration": iteration_ctx.iteration,
                    "duration": iteration_ctx.duration,
                    "results": iteration_ctx.get_all_results(),
                    "summary": iteration_ctx.get_iteration_summary()
                })
        
        return {
            "success": True,
            "loop_id": task.id,
            "iterations_completed": task.current_iteration,
            "terminated_by": task.terminated_by,
            "total_duration": loop_context.total_duration,
            "all_results": all_results,
            "final_results": (
                all_results[-1]["results"] if all_results else {}
            ),
            "loop_statistics": task.get_loop_statistics(),
            "tool_statistics": loop_context.get_tool_statistics(),
            "condition_evaluations": loop_context.condition_evaluations,
            "summary": final_summary,
            "debug_info": loop_context.get_debug_info()
        }