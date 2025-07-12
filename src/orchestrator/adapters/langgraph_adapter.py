"""LangGraph adapter for integrating with LangGraph workflows."""

import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..core.control_system import ControlSystem, ControlAction
from ..core.task import Task, TaskStatus
from ..core.pipeline import Pipeline


@dataclass 
class LangGraphNode:
    """Represents a node in a LangGraph workflow."""
    name: str
    function: Callable
    inputs: List[str]
    outputs: List[str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LangGraphEdge:
    """Represents an edge in a LangGraph workflow."""
    source: str
    target: str
    condition: Optional[Callable] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LangGraphState:
    """State management for LangGraph workflows."""
    
    def __init__(self):
        self.data = {}
        self.history = []
        self.current_node = None
    
    def get(self, key: str, default=None):
        """Get value from state."""
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set value in state."""
        self.data[key] = value
        self.history.append({"action": "set", "key": key, "value": value})
    
    def update(self, updates: Dict[str, Any]):
        """Update multiple values in state."""
        self.data.update(updates)
        self.history.append({"action": "update", "updates": updates})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "data": self.data.copy(),
            "current_node": self.current_node,
            "history_length": len(self.history)
        }


class LangGraphWorkflow:
    """LangGraph workflow representation."""
    
    def __init__(self, name: str):
        self.name = name
        self.nodes: Dict[str, LangGraphNode] = {}
        self.edges: List[LangGraphEdge] = []
        self.entry_point = None
        self.checkpoints = []
        self.metadata = {}
    
    def add_node(self, node: LangGraphNode):
        """Add a node to the workflow."""
        self.nodes[node.name] = node
        
        # Set as entry point if it's the first node
        if self.entry_point is None:
            self.entry_point = node.name
    
    def add_edge(self, edge: LangGraphEdge):
        """Add an edge to the workflow."""
        if edge.source not in self.nodes:
            raise ValueError(f"Source node '{edge.source}' not found")
        if edge.target not in self.nodes:
            raise ValueError(f"Target node '{edge.target}' not found")
        
        self.edges.append(edge)
    
    def get_next_nodes(self, current_node: str, state: LangGraphState) -> List[str]:
        """Get next nodes to execute based on current state."""
        next_nodes = []
        
        for edge in self.edges:
            if edge.source == current_node:
                # Check condition if specified
                if edge.condition is None or edge.condition(state):
                    next_nodes.append(edge.target)
        
        return next_nodes
    
    async def execute(self, initial_state: Dict[str, Any] = None) -> LangGraphState:
        """Execute the workflow."""
        state = LangGraphState()
        if initial_state:
            state.update(initial_state)
        
        # Start from entry point
        current_nodes = [self.entry_point] if self.entry_point else []
        
        while current_nodes:
            next_nodes = []
            
            # Execute all current nodes in parallel
            for node_name in current_nodes:
                state.current_node = node_name
                node = self.nodes[node_name]
                
                try:
                    # Prepare inputs for the node
                    inputs = {key: state.get(key) for key in node.inputs}
                    
                    # Execute node function
                    if asyncio.iscoroutinefunction(node.function):
                        result = await node.function(state, **inputs)
                    else:
                        result = node.function(state, **inputs)
                    
                    # Update state with outputs
                    if isinstance(result, dict):
                        for output_key in node.outputs:
                            if output_key in result:
                                state.set(output_key, result[output_key])
                    
                    # Get next nodes
                    next_nodes.extend(self.get_next_nodes(node_name, state))
                    
                except Exception as e:
                    state.set(f"error_{node_name}", str(e))
                    # Continue execution but mark error
            
            current_nodes = list(set(next_nodes))  # Remove duplicates
        
        return state


class LangGraphAdapter(ControlSystem):
    """Adapter for integrating Orchestrator with LangGraph workflows."""
    
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {"name": "langgraph_adapter"}
        
        super().__init__(config.get("name", "langgraph_adapter"))
        self.config = config
        self.workflows: Dict[str, LangGraphWorkflow] = {}
        self.active_executions: Dict[str, LangGraphState] = {}
    
    def register_workflow(self, workflow: LangGraphWorkflow):
        """Register a LangGraph workflow."""
        self.workflows[workflow.name] = workflow
    
    def create_workflow_from_pipeline(self, pipeline: Pipeline) -> LangGraphWorkflow:
        """Create a LangGraph workflow from an Orchestrator pipeline."""
        workflow = LangGraphWorkflow(pipeline.id)
        
        # Convert tasks to nodes
        for task_id in pipeline:
            task = pipeline.get_task(task_id)
            
            # Create node function
            async def node_function(state: LangGraphState, task=task, **kwargs):
                # Execute the task (simplified)
                result = await self._execute_task(task, state.data)
                return {"output": result}
            
            node = LangGraphNode(
                name=task_id,
                function=node_function,
                inputs=list(task.parameters.keys()),
                outputs=["output"],
                metadata={"original_task": task}
            )
            
            workflow.add_node(node)
        
        # Convert dependencies to edges
        for task_id in pipeline:
            task = pipeline.get_task(task_id)
            for dep_id in task.dependencies:
                edge = LangGraphEdge(
                    source=dep_id,
                    target=task_id
                )
                workflow.add_edge(edge)
        
        return workflow
    
    async def execute_task(self, task: Task, context: Dict[str, Any] = None) -> Any:
        """Execute a single task."""
        return await self._execute_task(task, context or {})
    
    async def execute_pipeline(self, pipeline: Pipeline) -> Dict[str, Any]:
        """Execute an entire pipeline."""
        workflow = self.create_workflow_from_pipeline(pipeline)
        state = await workflow.execute()
        return state.data
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return system capabilities."""
        return {
            "supports_workflows": True,
            "supports_conditional_execution": True,
            "supports_parallel_execution": True,
            "supported_actions": ["generate", "analyze", "transform"]
        }
    
    async def health_check(self) -> bool:
        """Check if the system is healthy."""
        return True  # Simplified for testing
    
    async def _execute_task(self, task: Task, state_data: Dict[str, Any]) -> Any:
        """Execute a task within LangGraph context."""
        # This would integrate with the actual Orchestrator execution
        # For now, return a mock result
        return f"Executed {task.id} with state: {state_data}"
    
    async def execute_workflow(self, workflow_name: str, initial_state: Dict[str, Any] = None) -> LangGraphState:
        """Execute a registered workflow."""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow '{workflow_name}' not found")
        
        workflow = self.workflows[workflow_name]
        state = await workflow.execute(initial_state)
        
        # Store execution state
        execution_id = f"{workflow_name}_{len(self.active_executions)}"
        self.active_executions[execution_id] = state
        
        return state
    
    async def decide_action(self, task: Task, context: Dict[str, Any]) -> ControlAction:
        """Decide control action based on LangGraph workflow logic."""
        # Check if task is part of a workflow
        workflow_name = context.get("workflow_name")
        if workflow_name and workflow_name in self.workflows:
            workflow = self.workflows[workflow_name]
            
            # Get current state
            execution_id = context.get("execution_id")
            if execution_id in self.active_executions:
                state = self.active_executions[execution_id]
                
                # Determine action based on workflow state
                if task.id in workflow.nodes:
                    node = workflow.nodes[task.id]
                    
                    # Check if all inputs are available
                    inputs_ready = all(
                        state.get(input_key) is not None 
                        for input_key in node.inputs
                    )
                    
                    if inputs_ready:
                        return ControlAction.EXECUTE
                    else:
                        return ControlAction.WAIT
        
        # Default action
        return ControlAction.EXECUTE
    
    def get_workflow_status(self, workflow_name: str) -> Dict[str, Any]:
        """Get status of a workflow."""
        if workflow_name not in self.workflows:
            return {"error": "Workflow not found"}
        
        workflow = self.workflows[workflow_name]
        
        # Count active executions for this workflow
        active_count = sum(
            1 for exec_id in self.active_executions 
            if exec_id.startswith(f"{workflow_name}_")
        )
        
        return {
            "name": workflow.name,
            "nodes": len(workflow.nodes),
            "edges": len(workflow.edges),
            "entry_point": workflow.entry_point,
            "active_executions": active_count,
            "metadata": workflow.metadata
        }
    
    def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """Get status of a specific execution."""
        if execution_id not in self.active_executions:
            return {"error": "Execution not found"}
        
        state = self.active_executions[execution_id]
        return {
            "execution_id": execution_id,
            "current_node": state.current_node,
            "state_data": state.data,
            "history_length": len(state.history),
            "status": "active" if state.current_node else "completed"
        }
    
    def cleanup_execution(self, execution_id: str) -> bool:
        """Clean up a completed execution."""
        if execution_id in self.active_executions:
            del self.active_executions[execution_id]
            return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return {
            "workflows_registered": len(self.workflows),
            "active_executions": len(self.active_executions),
            "total_nodes": sum(len(wf.nodes) for wf in self.workflows.values()),
            "total_edges": sum(len(wf.edges) for wf in self.workflows.values())
        }