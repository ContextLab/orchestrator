"""
StateGraphConstructor - Generates optimized LangGraph StateGraph structures.

This module converts analysis results into optimized LangGraph StateGraph instances,
implementing the final step of automatic graph generation from Issue #200.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Any, Callable, Optional, TYPE_CHECKING

from .types import (
    DependencyGraph, ParallelGroup, ControlFlowMap, DataFlowSchema, 
    ParsedPipeline, ParsedStep
)

if TYPE_CHECKING:
    try:
        from langgraph.graph import StateGraph
    except ImportError:
        StateGraph = Any

logger = logging.getLogger(__name__)


class StateGraphConstructor:
    """
    Converts analysis results into optimized LangGraph StateGraph.
    This is where the automatic graph generation magic happens.
    """
    
    def __init__(self, model_registry=None, tool_registry=None):
        self.model_registry = model_registry
        self.tool_registry = tool_registry
        logger.info("StateGraphConstructor initialized")
        
    async def construct_graph(self,
                            dependency_graph: DependencyGraph,
                            parallel_groups: List[ParallelGroup],
                            control_flow: ControlFlowMap,
                            data_schema: DataFlowSchema,
                            original_pipeline: ParsedPipeline) -> Any:  # StateGraph when LangGraph is available
        """Generate optimized StateGraph from all analysis components."""
        logger.debug(f"Constructing StateGraph for pipeline: {original_pipeline.id}")
        
        try:
            # Try to create actual LangGraph StateGraph
            graph = await self._create_langgraph_state_graph(
                dependency_graph, parallel_groups, control_flow, data_schema, original_pipeline
            )
            logger.info(f"Successfully created LangGraph StateGraph for {original_pipeline.id}")
            return graph
            
        except ImportError:
            logger.warning("LangGraph not available, returning enhanced placeholder")
            return await self._create_enhanced_placeholder(
                dependency_graph, parallel_groups, control_flow, data_schema, original_pipeline
            )
        except Exception as e:
            logger.error(f"Failed to create StateGraph: {e}")
            return await self._create_enhanced_placeholder(
                dependency_graph, parallel_groups, control_flow, data_schema, original_pipeline
            )
            
    async def _create_langgraph_state_graph(self,
                                          dependency_graph: DependencyGraph,
                                          parallel_groups: List[ParallelGroup],
                                          control_flow: ControlFlowMap,
                                          data_schema: DataFlowSchema,
                                          original_pipeline: ParsedPipeline) -> Any:
        """Create actual LangGraph StateGraph."""
        try:
            from langgraph.graph import StateGraph, START, END
            from typing_extensions import TypedDict
        except ImportError:
            raise ImportError("LangGraph not available")
        
        # Define state structure from data schema
        state_class = self._create_state_class(data_schema, original_pipeline, dependency_graph)
        
        # Create the StateGraph
        graph = StateGraph(state_class)
        
        # Add nodes for each step
        await self._add_step_nodes(graph, dependency_graph, original_pipeline)
        
        # Add edges based on dependencies
        await self._add_dependency_edges(graph, dependency_graph)
        
        # Add conditional edges from control flow
        await self._add_conditional_edges(graph, control_flow)
        
        # Add parallel execution edges
        await self._add_parallel_edges(graph, parallel_groups, dependency_graph)
        
        # Set entry and exit points
        await self._configure_entry_exit_points(graph, dependency_graph)
        
        # Compile the graph
        compiled_graph = graph.compile()
        
        logger.info(f"LangGraph StateGraph compiled successfully with {len(dependency_graph.nodes)} nodes")
        return compiled_graph
        
    def _create_state_class(self, data_schema: DataFlowSchema, pipeline: ParsedPipeline, dependency_graph: DependencyGraph = None) -> type:
        """Create TypedDict state class from data flow schema."""
        try:
            from typing_extensions import TypedDict
        except ImportError:
            from typing import TypedDict
        
        # Build state fields from pipeline inputs and step outputs
        state_fields = {}
        
        # Add pipeline inputs
        for input_name, input_spec in pipeline.inputs.items():
            state_fields[input_name] = type(input_spec.get('default', str))
        
        # Add step outputs  
        if dependency_graph:
            for step_id, step in dependency_graph.nodes.items():
                # Each step can produce a result
                state_fields[f"{step_id}_result"] = dict
                state_fields[f"{step_id}_status"] = str
        
        # Add metadata fields
        state_fields['pipeline_id'] = str
        state_fields['current_step'] = str
        state_fields['execution_history'] = list
        state_fields['error_state'] = dict
        
        # Create the TypedDict class dynamically using types.new_class
        import types
        
        def create_state_class():
            return state_fields
            
        PipelineState = types.new_class(
            'PipelineState', 
            (TypedDict,), 
            {},
            lambda ns: ns.update(state_fields)
        )
        return PipelineState
        
    async def _add_step_nodes(self, graph: Any, dependency_graph: DependencyGraph, pipeline: ParsedPipeline) -> None:
        """Add nodes for each pipeline step."""
        for step_id, step in dependency_graph.nodes.items():
            node_function = await self._create_node_function(step, step_id)
            graph.add_node(step_id, node_function)
            logger.debug(f"Added node: {step_id}")
            
    async def _create_node_function(self, step: ParsedStep, step_id: str) -> Callable:
        """Create the actual function that will execute for this step."""
        async def step_executor(state: Dict[str, Any]) -> Dict[str, Any]:
            """Execute the step and return updated state."""
            logger.debug(f"Executing step: {step_id}")
            
            try:
                # Update current step in state
                state['current_step'] = step_id
                
                # Add to execution history
                if 'execution_history' not in state:
                    state['execution_history'] = []
                state['execution_history'].append(step_id)
                
                # Execute the step based on its type
                if step.tool:
                    result = await self._execute_tool_step(step, state)
                elif step.action:
                    result = await self._execute_action_step(step, state)
                else:
                    result = await self._execute_custom_step(step, state)
                
                # Store result in state
                state[f"{step_id}_result"] = result
                state[f"{step_id}_status"] = "completed"
                
                logger.info(f"Step {step_id} completed successfully")
                return state
                
            except Exception as e:
                logger.error(f"Step {step_id} failed: {e}")
                state[f"{step_id}_status"] = "failed"
                state['error_state'] = {
                    'failed_step': step_id,
                    'error_message': str(e),
                    'error_type': type(e).__name__
                }
                return state
                
        return step_executor
        
    async def _execute_tool_step(self, step: ParsedStep, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a step that uses a tool."""
        if self.tool_registry and step.tool in self.tool_registry:
            tool = self.tool_registry[step.tool]
            # Resolve input templates
            resolved_inputs = await self._resolve_input_templates(step.inputs, state)
            return await tool.execute(resolved_inputs)
        else:
            # Placeholder execution for testing
            return {
                "tool": step.tool,
                "inputs": step.inputs,
                "status": "simulated",
                "message": f"Tool {step.tool} executed (placeholder)"
            }
            
    async def _execute_action_step(self, step: ParsedStep, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a step that uses an action."""
        # Placeholder for action execution
        resolved_inputs = await self._resolve_input_templates(step.inputs, state)
        return {
            "action": step.action,
            "inputs": resolved_inputs,
            "status": "simulated",
            "message": f"Action {step.action} executed (placeholder)"
        }
        
    async def _execute_custom_step(self, step: ParsedStep, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a custom step."""
        return {
            "step_id": step.id,
            "inputs": step.inputs,
            "status": "simulated",
            "message": f"Custom step {step.id} executed (placeholder)"
        }
        
    async def _resolve_input_templates(self, inputs: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve Jinja2 template variables in step inputs."""
        from jinja2 import Template
        
        resolved = {}
        for key, value in inputs.items():
            if isinstance(value, str) and '{{' in value and '}}' in value:
                try:
                    template = Template(value)
                    resolved[key] = template.render(**state)
                except Exception as e:
                    logger.warning(f"Failed to resolve template '{value}': {e}")
                    resolved[key] = value
            else:
                resolved[key] = value
        return resolved
        
    async def _add_dependency_edges(self, graph: Any, dependency_graph: DependencyGraph) -> None:
        """Add edges based on step dependencies."""
        for edge in dependency_graph.edges:
            graph.add_edge(edge.source, edge.target)
            logger.debug(f"Added dependency edge: {edge.source} -> {edge.target}")
            
    async def _add_conditional_edges(self, graph: Any, control_flow: ControlFlowMap) -> None:
        """Add conditional edges from control flow analysis."""
        for step_id, conditional_logic in control_flow.conditionals.items():
            # Create condition function for LangGraph
            def create_condition_router(logic):
                async def condition_router(state: Dict[str, Any]) -> str:
                    """Route based on condition evaluation."""
                    try:
                        is_true = await logic.evaluator_function(state)
                        return logic.true_path if is_true else (logic.false_path or 'END')
                    except Exception as e:
                        logger.error(f"Condition evaluation failed: {e}")
                        return logic.false_path or 'END'
                return condition_router
            
            router = create_condition_router(conditional_logic)
            paths = [conditional_logic.true_path]
            if conditional_logic.false_path:
                paths.append(conditional_logic.false_path)
            
            graph.add_conditional_edges(step_id, router, paths)
            logger.debug(f"Added conditional edge for step: {step_id}")
            
    async def _add_parallel_edges(self, graph: Any, parallel_groups: List[ParallelGroup], dependency_graph: DependencyGraph) -> None:
        """Add edges to enable parallel execution."""
        for group in parallel_groups:
            if len(group.steps) > 1:
                # For LangGraph, parallel execution is achieved through proper edge structure
                # Steps with the same dependencies can execute in parallel automatically
                logger.debug(f"Configured parallel execution for group: {group.steps}")
                
    async def _configure_entry_exit_points(self, graph: Any, dependency_graph: DependencyGraph) -> None:
        """Configure graph entry and exit points."""
        try:
            from langgraph.graph import START, END
            
            # Find entry points (nodes with no dependencies)
            entry_points = []
            for node_id in dependency_graph.nodes.keys():
                has_incoming = any(edge.target == node_id for edge in dependency_graph.edges)
                if not has_incoming:
                    entry_points.append(node_id)
                    
            # Connect START to entry points
            for entry_point in entry_points:
                graph.add_edge(START, entry_point)
                
            # Find exit points (nodes with no dependents)
            exit_points = []
            for node_id in dependency_graph.nodes.keys():
                has_outgoing = any(edge.source == node_id for edge in dependency_graph.edges)
                if not has_outgoing:
                    exit_points.append(node_id)
                    
            # Connect exit points to END
            for exit_point in exit_points:
                graph.add_edge(exit_point, END)
                
            logger.info(f"Configured {len(entry_points)} entry points and {len(exit_points)} exit points")
            
        except ImportError:
            logger.warning("LangGraph START/END not available")
            
    async def _create_enhanced_placeholder(self,
                                         dependency_graph: DependencyGraph,
                                         parallel_groups: List[ParallelGroup],
                                         control_flow: ControlFlowMap,
                                         data_schema: DataFlowSchema,
                                         original_pipeline: ParsedPipeline) -> Dict[str, Any]:
        """Create an enhanced placeholder with detailed execution plan."""
        
        execution_plan = await self._generate_execution_plan(dependency_graph, parallel_groups, control_flow)
        
        return {
            "pipeline_id": original_pipeline.id,
            "type": "enhanced_placeholder",
            "nodes": list(dependency_graph.nodes.keys()),
            "edges": [{"source": e.source, "target": e.target, "type": e.dependency_type.value} 
                     for e in dependency_graph.edges],
            "parallel_groups": [{"steps": g.steps, "speedup": g.estimated_speedup} 
                              for g in parallel_groups],
            "conditionals": list(control_flow.conditionals.keys()),
            "loops": list(control_flow.loops.keys()),
            "goto_statements": control_flow.goto_statements,
            "execution_plan": execution_plan,
            "estimated_total_speedup": self._calculate_total_speedup(parallel_groups),
            "status": "ready_for_execution"
        }
        
    async def _generate_execution_plan(self, dependency_graph: DependencyGraph, 
                                     parallel_groups: List[ParallelGroup], 
                                     control_flow: ControlFlowMap) -> List[Dict[str, Any]]:
        """Generate detailed execution plan."""
        execution_levels = dependency_graph.get_execution_levels()
        plan = []
        
        for level, steps in execution_levels.items():
            level_info = {
                "level": level,
                "steps": steps,
                "execution_type": "sequential"
            }
            
            # Check if any steps in this level can run in parallel
            for group in parallel_groups:
                if any(step in steps for step in group.steps):
                    level_info["execution_type"] = "parallel"
                    level_info["parallel_groups"] = [group.steps]
                    level_info["estimated_speedup"] = group.estimated_speedup
                    break
                    
            plan.append(level_info)
            
        return plan
        
    def _calculate_total_speedup(self, parallel_groups: List[ParallelGroup]) -> float:
        """Calculate estimated total speedup from parallelization."""
        if not parallel_groups:
            return 1.0
            
        # Simple speedup calculation - in reality this would be more complex
        total_speedup = 1.0
        for group in parallel_groups:
            if group.estimated_speedup > 1.0:
                total_speedup *= group.estimated_speedup
                
        return min(total_speedup, 10.0)  # Cap at 10x speedup