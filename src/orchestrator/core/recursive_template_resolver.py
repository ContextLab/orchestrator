"""Recursive Template Resolution System for Advanced Loop Context.

This module provides enhanced template resolution for complex iterative patterns:
- Recursive template resolution for nested loop contexts
- Loop iteration history access: loop.iterations[-1].step.result
- Multi-level variable access with complex nesting
- Advanced template validation with type safety

Issue #287: Advanced Infrastructure Pipeline Development - Stream A
"""

import logging
import re
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from .unified_template_resolver import UnifiedTemplateResolver, TemplateResolutionContext
from .loop_context import GlobalLoopContextManager
from .template_manager import TemplateManager

logger = logging.getLogger(__name__)


@dataclass
class LoopIterationData:
    """Data structure for storing loop iteration history."""
    
    iteration: int
    step_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_step_result(self, step_name: str, property_path: Optional[str] = None) -> Any:
        """Get result from a specific step, optionally accessing nested properties."""
        if step_name not in self.step_results:
            return None
        
        result = self.step_results[step_name]
        
        # If no property path, return the entire result
        if not property_path:
            return result
        
        # Navigate property path (e.g., "result" or "action")
        current = result
        for prop in property_path.split('.'):
            if isinstance(current, dict) and prop in current:
                current = current[prop]
            else:
                return None
        
        return current


@dataclass 
class LoopIterationHistory:
    """Container for loop iteration history with advanced access patterns."""
    
    loop_id: str
    iterations: List[LoopIterationData] = field(default_factory=list)
    iteration_count: int = 0
    
    def add_iteration(self, iteration_data: LoopIterationData):
        """Add new iteration to history."""
        self.iterations.append(iteration_data)
        self.iteration_count = len(self.iterations)
        logger.debug(f"Added iteration {iteration_data.iteration} to loop {self.loop_id}")
    
    def get_iteration(self, index: int) -> Optional[LoopIterationData]:
        """Get iteration by index (supports negative indexing)."""
        if not self.iterations:
            return None
        
        try:
            return self.iterations[index]
        except IndexError:
            return None
    
    def get_last_iteration(self) -> Optional[LoopIterationData]:
        """Get the most recent iteration."""
        return self.get_iteration(-1)
    
    def get_step_from_iteration(self, iteration_index: int, step_name: str, property_path: Optional[str] = None) -> Any:
        """Get specific step result from a specific iteration."""
        iteration_data = self.get_iteration(iteration_index)
        if not iteration_data:
            return None
        
        return iteration_data.get_step_result(step_name, property_path)
    
    def __getitem__(self, index: int) -> Optional[LoopIterationData]:
        """Support array-like access: iterations[0], iterations[-1]"""
        return self.get_iteration(index)
    
    def __len__(self) -> int:
        """Support len() calls"""
        return len(self.iterations)
    
    def __iter__(self):
        """Support iteration"""
        return iter(self.iterations)


class RecursiveTemplateResolver(UnifiedTemplateResolver):
    """Enhanced template resolver with recursive loop iteration support.
    
    This class extends UnifiedTemplateResolver to handle complex patterns like:
    - {{ loop_name.iterations[-1].step_name.result }}
    - {{ loop_name.iteration_count }}
    - Complex multi-level variable access
    """
    
    def __init__(
        self,
        template_manager: Optional[TemplateManager] = None,
        context_manager: Optional = None,
        loop_context_manager: Optional[GlobalLoopContextManager] = None,
        debug_mode: bool = False
    ):
        """Initialize the recursive template resolver.
        
        Args:
            template_manager: Existing template manager or None to create new
            context_manager: Existing context manager or None to create new  
            loop_context_manager: Existing loop context manager or None to create new
            debug_mode: Enable debug logging
        """
        super().__init__(template_manager, context_manager, loop_context_manager, debug_mode)
        
        # Enhanced iteration tracking
        self.loop_iteration_history: Dict[str, LoopIterationHistory] = {}
        self.loop_step_results: Dict[str, Dict[str, Any]] = {}  # loop_id -> iteration -> step -> results
        
        logger.info("RecursiveTemplateResolver initialized with advanced loop iteration tracking")
    
    def register_loop_iteration(
        self,
        loop_id: str,
        iteration: int,
        step_results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Register results from a loop iteration for template access.
        
        Args:
            loop_id: Loop identifier
            iteration: Iteration number
            step_results: Results from all steps in this iteration
            metadata: Optional metadata for this iteration
        """
        # Initialize history for this loop if not exists
        if loop_id not in self.loop_iteration_history:
            self.loop_iteration_history[loop_id] = LoopIterationHistory(loop_id=loop_id)
        
        # Create iteration data
        iteration_data = LoopIterationData(
            iteration=iteration,
            step_results=step_results.copy(),
            metadata=metadata or {}
        )
        
        # Add to history
        self.loop_iteration_history[loop_id].add_iteration(iteration_data)
        
        # Update loop step results tracking
        if loop_id not in self.loop_step_results:
            self.loop_step_results[loop_id] = {}
        self.loop_step_results[loop_id][iteration] = step_results.copy()
        
        if self.debug_mode:
            logger.debug(f"Registered iteration {iteration} for loop {loop_id} with {len(step_results)} step results")
            logger.debug(f"Step result keys: {list(step_results.keys())}")
    
    def get_loop_iteration_variables(self) -> Dict[str, Any]:
        """Get template variables for all loop iterations.
        
        Returns:
            Dictionary of loop iteration variables for templates
        """
        variables = {}
        
        for loop_id, history in self.loop_iteration_history.items():
            # Add the entire history object for complex access
            variables[f"{loop_id}.iterations"] = history.iterations
            variables[f"{loop_id}.iteration_count"] = history.iteration_count
            
            # Add convenience access to last iteration
            last_iteration = history.get_last_iteration()
            if last_iteration:
                variables[f"{loop_id}.last_iteration"] = last_iteration
                
                # Add direct access to last iteration's step results
                for step_name, step_result in last_iteration.step_results.items():
                    variables[f"{loop_id}.last.{step_name}"] = step_result
        
        if self.debug_mode:
            logger.debug(f"Generated {len(variables)} loop iteration variables")
        
        return variables
    
    def resolve_recursive_patterns(self, template: str, context: Dict[str, Any]) -> str:
        """Resolve complex recursive template patterns.
        
        Args:
            template: Template string with potential recursive patterns
            context: Template resolution context
            
        Returns:
            Template with recursive patterns resolved
        """
        if not isinstance(template, str):
            return template
        
        # Pattern: {{ loop_name.iterations[index].step_name.property }}
        iteration_pattern = r'\{\{\s*(\w+)\.iterations\[(-?\d+)\]\.(\w+)(?:\.(\w+))?\s*\}\}'
        
        def replace_iteration_access(match):
            loop_name = match.group(1)
            index = int(match.group(2))
            step_name = match.group(3)
            property_name = match.group(4)  # Optional property access
            
            if loop_name not in self.loop_iteration_history:
                logger.warning(f"Loop '{loop_name}' not found in iteration history")
                return match.group(0)  # Return original if not found
            
            history = self.loop_iteration_history[loop_name]
            iteration_data = history.get_iteration(index)
            
            if not iteration_data:
                logger.warning(f"Iteration {index} not found for loop '{loop_name}'")
                return match.group(0)  # Return original if not found
            
            result = iteration_data.get_step_result(step_name, property_name)
            if result is None:
                logger.warning(f"Step '{step_name}' result not found in iteration {index} of loop '{loop_name}'")
                return match.group(0)  # Return original if not found
            
            # Convert result to string for template replacement
            if isinstance(result, (dict, list)):
                # For complex objects, we might need JSON serialization or special handling
                import json
                try:
                    return json.dumps(result)
                except (TypeError, ValueError):
                    return str(result)
            
            return str(result)
        
        # Apply pattern replacement
        resolved = re.sub(iteration_pattern, replace_iteration_access, template)
        
        # Pattern: {{ loop_name.iteration_count }}
        count_pattern = r'\{\{\s*(\w+)\.iteration_count\s*\}\}'
        
        def replace_iteration_count(match):
            loop_name = match.group(1)
            
            if loop_name not in self.loop_iteration_history:
                return "0"  # Default to 0 if loop not found
            
            return str(self.loop_iteration_history[loop_name].iteration_count)
        
        resolved = re.sub(count_pattern, replace_iteration_count, resolved)
        
        if self.debug_mode and resolved != template:
            logger.debug(f"Resolved recursive patterns: '{template[:100]}...' -> '{resolved[:100]}...'")
        
        return resolved
    
    def collect_enhanced_context(
        self,
        pipeline_id: Optional[str] = None,
        task_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        pipeline_inputs: Optional[Dict[str, Any]] = None,
        pipeline_parameters: Optional[Dict[str, Any]] = None,
        step_results: Optional[Dict[str, Any]] = None,
        tool_parameters: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> TemplateResolutionContext:
        """Collect enhanced context including loop iteration history.
        
        Args:
            pipeline_id: Current pipeline ID
            task_id: Current task ID
            tool_name: Current tool name
            pipeline_inputs: Pipeline input parameters
            pipeline_parameters: Pipeline parameters  
            step_results: Results from executed steps
            tool_parameters: Tool-specific parameters
            additional_context: Any additional context variables
            
        Returns:
            Enhanced template resolution context
        """
        # Get base context from parent
        base_context = self.collect_context(
            pipeline_id=pipeline_id,
            task_id=task_id,
            tool_name=tool_name,
            pipeline_inputs=pipeline_inputs,
            pipeline_parameters=pipeline_parameters,
            step_results=step_results,
            tool_parameters=tool_parameters,
            additional_context=additional_context
        )
        
        # Add loop iteration variables
        iteration_vars = self.get_loop_iteration_variables()
        
        # Merge with additional context
        enhanced_additional = base_context.additional_context.copy()
        enhanced_additional.update(iteration_vars)
        
        # Create enhanced context
        enhanced_context = TemplateResolutionContext(
            pipeline_id=base_context.pipeline_id,
            task_id=base_context.task_id,
            tool_name=base_context.tool_name,
            pipeline_inputs=base_context.pipeline_inputs,
            pipeline_parameters=base_context.pipeline_parameters,
            step_results=base_context.step_results,
            previous_results=base_context.previous_results,
            loop_variables=base_context.loop_variables,
            active_loops=base_context.active_loops,
            tool_parameters=base_context.tool_parameters,
            additional_context=enhanced_additional
        )
        
        if self.debug_mode:
            logger.debug(f"Enhanced context with {len(iteration_vars)} iteration variables")
        
        return enhanced_context
    
    def resolve_templates(
        self,
        data: Any,
        context: Optional[TemplateResolutionContext] = None
    ) -> Any:
        """Resolve templates with recursive pattern support.
        
        Args:
            data: Data that may contain templates
            context: Template resolution context (uses current if None)
            
        Returns:
            Data with all templates resolved, including recursive patterns
        """
        # Pre-process recursive patterns before standard template resolution
        if isinstance(data, str):
            # Get context for recursive pattern resolution
            if context:
                flat_context = context.to_flat_dict()
            elif self._current_context:
                flat_context = self._current_context.to_flat_dict()
            else:
                flat_context = {}
            
            # Add iteration variables to context
            iteration_vars = self.get_loop_iteration_variables()
            flat_context.update(iteration_vars)
            
            # Resolve recursive patterns first
            preprocessed_data = self.resolve_recursive_patterns(data, flat_context)
        else:
            preprocessed_data = data
        
        # Then apply standard template resolution
        return super().resolve_templates(preprocessed_data, context)
    
    def clear_loop_history(self, loop_id: Optional[str] = None):
        """Clear loop iteration history.
        
        Args:
            loop_id: Specific loop to clear, or None to clear all
        """
        if loop_id:
            if loop_id in self.loop_iteration_history:
                del self.loop_iteration_history[loop_id]
                logger.debug(f"Cleared history for loop {loop_id}")
        else:
            self.loop_iteration_history.clear()
            self.loop_step_results.clear()
            logger.debug("Cleared all loop iteration history")
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information including loop iteration history.
        
        Returns:
            Enhanced debug information dictionary
        """
        base_info = super().get_debug_info()
        
        # Add recursive template resolver info
        recursive_info = {
            "tracked_loops": list(self.loop_iteration_history.keys()),
            "total_iterations": sum(
                history.iteration_count 
                for history in self.loop_iteration_history.values()
            ),
            "iteration_variables": len(self.get_loop_iteration_variables())
        }
        
        # Add detailed loop information
        loop_details = {}
        for loop_id, history in self.loop_iteration_history.items():
            loop_details[loop_id] = {
                "iteration_count": history.iteration_count,
                "last_iteration_steps": list(history.get_last_iteration().step_results.keys()) if history.get_last_iteration() else []
            }
        
        base_info["recursive_resolver"] = recursive_info
        base_info["loop_details"] = loop_details
        
        return base_info