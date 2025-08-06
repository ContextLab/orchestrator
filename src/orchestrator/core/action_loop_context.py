"""Enhanced loop context for action loops with tool integration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ActionResult:
    """Result from executing an action within a loop iteration."""
    
    action_name: Optional[str] = None  # Named action identifier
    action_def: Dict[str, Any] = field(default_factory=dict)  # Original action definition
    status: str = "completed"  # completed, failed, skipped
    result: Any = None  # Action result data
    tool_used: Optional[str] = None  # Tool that was used (if any)
    execution_time: float = 0.0  # Time taken to execute
    error: Optional[str] = None  # Error message if failed
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert action result to dictionary."""
        return {
            "action_name": self.action_name,
            "action_def": self.action_def,
            "status": self.status,
            "result": self.result,
            "tool_used": self.tool_used,
            "execution_time": self.execution_time,
            "error": self.error,
            "metadata": self.metadata
        }
    
    @property
    def success(self) -> bool:
        """Check if action execution was successful."""
        return self.status == "completed"
    
    @property
    def failed(self) -> bool:
        """Check if action execution failed."""
        return self.status == "failed"


@dataclass 
class IterationContext:
    """Context for a single loop iteration."""
    
    iteration: int = 0  # Current iteration number (0-based)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    action_results: Dict[str, ActionResult] = field(default_factory=dict)  # Named action results
    unnamed_results: List[ActionResult] = field(default_factory=list)  # Unnamed action results
    iteration_metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
        """Get iteration duration in seconds."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time
    
    @property
    def is_complete(self) -> bool:
        """Check if iteration is complete."""
        return self.end_time is not None
    
    def complete_iteration(self) -> None:
        """Mark iteration as complete."""
        self.end_time = time.time()
    
    def add_action_result(self, result: ActionResult) -> None:
        """Add action result to iteration context."""
        if result.action_name:
            self.action_results[result.action_name] = result
        else:
            self.unnamed_results.append(result)
    
    def get_action_result(self, name: str) -> Optional[ActionResult]:
        """Get named action result."""
        return self.action_results.get(name)
    
    def get_all_results(self) -> Dict[str, Any]:
        """Get all results as a flat dictionary for template access."""
        results = {}
        
        # Add named results
        for name, result in self.action_results.items():
            results[name] = result.result
            
        # Add numbered results for unnamed actions
        for i, result in enumerate(self.unnamed_results):
            results[f"action_{i}"] = result.result
        
        return results
    
    def get_iteration_summary(self) -> Dict[str, Any]:
        """Get summary of iteration execution."""
        total_actions = len(self.action_results) + len(self.unnamed_results)
        successful_actions = sum(
            1 for result in self.action_results.values() if result.success
        ) + sum(
            1 for result in self.unnamed_results if result.success
        )
        
        tools_used = set()
        for result in list(self.action_results.values()) + self.unnamed_results:
            if result.tool_used:
                tools_used.add(result.tool_used)
        
        return {
            "iteration": self.iteration,
            "duration": self.duration,
            "total_actions": total_actions,
            "successful_actions": successful_actions,
            "success_rate": successful_actions / total_actions if total_actions > 0 else 1.0,
            "tools_used": list(tools_used),
            "is_complete": self.is_complete
        }


@dataclass
class EnhancedLoopContext:
    """Enhanced loop context with tool integration and comprehensive state tracking."""
    
    loop_id: str = ""
    start_time: float = field(default_factory=time.time)
    iterations: List[IterationContext] = field(default_factory=list)
    current_iteration: int = 0
    termination_reason: Optional[str] = None
    
    # Tool execution tracking
    tool_results: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    tool_execution_times: Dict[str, List[float]] = field(default_factory=dict)
    tool_error_count: Dict[str, int] = field(default_factory=dict)
    
    # Loop state and metadata
    loop_metadata: Dict[str, Any] = field(default_factory=dict)
    condition_evaluations: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def total_duration(self) -> float:
        """Get total loop duration in seconds."""
        return time.time() - self.start_time
    
    @property
    def current_iteration_context(self) -> Optional[IterationContext]:
        """Get current iteration context."""
        if self.current_iteration < len(self.iterations):
            return self.iterations[self.current_iteration]
        return None
    
    @property
    def previous_iteration_context(self) -> Optional[IterationContext]:
        """Get previous iteration context."""
        if self.current_iteration > 0 and len(self.iterations) > self.current_iteration - 1:
            return self.iterations[self.current_iteration - 1]
        return None
    
    @property
    def is_first_iteration(self) -> bool:
        """Check if this is the first iteration."""
        return self.current_iteration == 0
    
    @property
    def has_previous_results(self) -> bool:
        """Check if there are results from previous iterations."""
        return self.current_iteration > 0 and len(self.iterations) > 0
    
    def start_iteration(self) -> IterationContext:
        """Start a new iteration and return its context."""
        iteration_ctx = IterationContext(iteration=self.current_iteration)
        
        # Extend iterations list if needed
        while len(self.iterations) <= self.current_iteration:
            self.iterations.append(IterationContext(iteration=len(self.iterations)))
        
        # Update current iteration context
        self.iterations[self.current_iteration] = iteration_ctx
        return iteration_ctx
    
    def complete_iteration(self) -> None:
        """Complete current iteration and advance to next."""
        if self.current_iteration < len(self.iterations):
            self.iterations[self.current_iteration].complete_iteration()
        self.current_iteration += 1
    
    def record_tool_result(self, tool_name: str, result: Dict[str, Any], execution_time: float) -> None:
        """Record tool execution result."""
        if tool_name not in self.tool_results:
            self.tool_results[tool_name] = []
            self.tool_execution_times[tool_name] = []
        
        self.tool_results[tool_name].append(result)
        self.tool_execution_times[tool_name].append(execution_time)
    
    def record_tool_error(self, tool_name: str) -> None:
        """Record tool execution error."""
        if tool_name not in self.tool_error_count:
            self.tool_error_count[tool_name] = 0
        self.tool_error_count[tool_name] += 1
    
    def get_tool_result(self, tool_name: str, iteration: int = -1) -> Optional[Any]:
        """Get result from specific tool execution."""
        if tool_name not in self.tool_results:
            return None
        
        tool_results = self.tool_results[tool_name]
        if iteration == -1 and tool_results:
            return tool_results[-1]
        elif 0 <= iteration < len(tool_results):
            return tool_results[iteration]
        return None
    
    def get_all_tool_results(self, tool_name: str) -> List[Dict[str, Any]]:
        """Get all results from a specific tool."""
        return self.tool_results.get(tool_name, [])
    
    def get_tool_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tool execution statistics."""
        stats = {}
        
        for tool_name in self.tool_results:
            executions = len(self.tool_results[tool_name])
            errors = self.tool_error_count.get(tool_name, 0)
            avg_time = (
                sum(self.tool_execution_times.get(tool_name, [])) / executions
                if executions > 0 else 0.0
            )
            
            stats[tool_name] = {
                "executions": executions,
                "errors": errors,
                "success_rate": (executions - errors) / executions if executions > 0 else 1.0,
                "average_execution_time": avg_time,
                "total_time": sum(self.tool_execution_times.get(tool_name, []))
            }
        
        return stats
    
    def get_previous_results(self, count: int = 1) -> List[Dict[str, Any]]:
        """Get results from previous N iterations."""
        results = []
        start_idx = max(0, self.current_iteration - count)
        
        for i in range(start_idx, self.current_iteration):
            if i < len(self.iterations):
                results.append(self.iterations[i].get_all_results())
        
        return results
    
    def get_named_result(self, name: str, iteration: int = -1) -> Any:
        """Get result by name from specific iteration."""
        if iteration == -1:
            # Get from most recent iteration that has this result
            for i in range(len(self.iterations) - 1, -1, -1):
                if i < len(self.iterations):
                    result = self.iterations[i].get_action_result(name)
                    if result:
                        return result.result
        elif 0 <= iteration < len(self.iterations):
            result = self.iterations[iteration].get_action_result(name)
            if result:
                return result.result
        
        return None
    
    def record_condition_evaluation(self, condition: str, result: bool, metadata: Dict[str, Any] = None) -> None:
        """Record condition evaluation for debugging."""
        evaluation = {
            "iteration": self.current_iteration,
            "condition": condition,
            "result": result,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        self.condition_evaluations.append(evaluation)
    
    def to_template_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for template access."""
        template_dict = {
            # With $ prefix for condition evaluation
            "$loop_id": self.loop_id,
            "$iteration": self.current_iteration,
            "$is_first": self.is_first_iteration,
            "$has_previous": self.has_previous_results,
            "$total_duration": self.total_duration,
            "$termination_reason": self.termination_reason,
            # Without $ prefix for Jinja2 template rendering
            "loop_id": self.loop_id,
            "iteration": self.current_iteration,
            "is_first": self.is_first_iteration,
            "has_previous": self.has_previous_results,
            "total_duration": self.total_duration,
            "termination_reason": self.termination_reason
        }
        
        # Add current iteration results
        if self.current_iteration_context:
            current_results = self.current_iteration_context.get_all_results()
            template_dict.update(current_results)
        
        # Add previous iteration results
        if self.has_previous_results:
            previous_results = self.get_previous_results(1)
            if previous_results:
                template_dict["$previous"] = previous_results[0]
                template_dict["$last_result"] = previous_results[0]
        
        # Add tool results for easy access
        for tool_name, results in self.tool_results.items():
            if results:
                template_dict[f"${tool_name}_result"] = results[-1]
                template_dict[f"${tool_name}_results"] = results
        
        # Add loop metadata
        template_dict.update(self.loop_metadata)
        
        return template_dict
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get comprehensive debug information."""
        return {
            "loop_id": self.loop_id,
            "current_iteration": self.current_iteration,
            "total_duration": self.total_duration,
            "iterations_summary": [
                ctx.get_iteration_summary() for ctx in self.iterations
            ],
            "tool_statistics": self.get_tool_statistics(),
            "condition_evaluations": self.condition_evaluations,
            "termination_reason": self.termination_reason,
            "loop_metadata": self.loop_metadata
        }
    
    def get_final_summary(self) -> Dict[str, Any]:
        """Get final loop execution summary."""
        completed_iterations = len([ctx for ctx in self.iterations if ctx.is_complete])
        total_actions = sum(
            len(ctx.action_results) + len(ctx.unnamed_results) 
            for ctx in self.iterations
        )
        successful_actions = sum(
            sum(1 for result in ctx.action_results.values() if result.success) +
            sum(1 for result in ctx.unnamed_results if result.success)
            for ctx in self.iterations
        )
        
        return {
            "loop_id": self.loop_id,
            "completed_iterations": completed_iterations,
            "total_duration": self.total_duration,
            "total_actions": total_actions,
            "successful_actions": successful_actions,
            "overall_success_rate": (
                successful_actions / total_actions if total_actions > 0 else 1.0
            ),
            "termination_reason": self.termination_reason,
            "tool_statistics": self.get_tool_statistics(),
            "condition_evaluations_count": len(self.condition_evaluations)
        }