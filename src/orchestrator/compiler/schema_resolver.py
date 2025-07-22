"""Graph-based schema resolution for pipeline validation."""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..core.pipeline import Pipeline
from ..core.task import Task
from ..tools.validation import SchemaState


@dataclass
class GraphNode:
    """Node in the pipeline graph representing a task."""
    task_id: str
    task: Task
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    schema_state: SchemaState = SchemaState.AMBIGUOUS
    predecessors: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)
    
    def has_fixed_schema(self) -> bool:
        """Check if both input and output schemas are fixed."""
        return self.schema_state == SchemaState.FIXED
    
    def has_partial_schema(self) -> bool:
        """Check if schemas are partially known."""
        return self.schema_state == SchemaState.PARTIAL


@dataclass
class GraphEdge:
    """Edge in the pipeline graph representing data flow."""
    source: str
    target: str
    data_mapping: Dict[str, str] = field(default_factory=dict)  # source_field -> target_field
    transform: Optional[str] = None  # Optional transformation expression


class PipelineGraph:
    """Graph representation of a pipeline for schema resolution."""
    
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, List[GraphEdge]] = defaultdict(list)
        self.reverse_edges: Dict[str, List[GraphEdge]] = defaultdict(list)
        self._build_graph()
    
    def _build_graph(self):
        """Build graph from pipeline tasks."""
        # Create nodes
        for task_id, task in self.pipeline.tasks.items():
            self.nodes[task_id] = GraphNode(
                task_id=task_id,
                task=task,
                predecessors=list(task.dependencies)
            )
        
        # Create edges and update successors
        for task_id, task in self.pipeline.tasks.items():
            for dep_id in task.dependencies:
                if dep_id in self.nodes:
                    # Create edge
                    edge = GraphEdge(source=dep_id, target=task_id)
                    self.edges[dep_id].append(edge)
                    self.reverse_edges[task_id].append(edge)
                    
                    # Update successor list
                    self.nodes[dep_id].successors.append(task_id)
    
    def get_roots(self) -> List[str]:
        """Get root nodes (no dependencies)."""
        return [node_id for node_id, node in self.nodes.items() 
                if not node.predecessors]
    
    def get_leaves(self) -> List[str]:
        """Get leaf nodes (no successors)."""
        return [node_id for node_id, node in self.nodes.items() 
                if not node.successors]
    
    def topological_sort(self) -> List[str]:
        """Get topological ordering of nodes."""
        in_degree = {node_id: len(node.predecessors) 
                    for node_id, node in self.nodes.items()}
        
        queue = deque([node_id for node_id, degree in in_degree.items() 
                      if degree == 0])
        result = []
        
        while queue:
            node_id = queue.popleft()
            result.append(node_id)
            
            for successor in self.nodes[node_id].successors:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)
        
        return result


class SchemaResolver:
    """Resolves schemas for pipeline tasks using graph traversal."""
    
    def __init__(self, tool_registry=None, model_registry=None):
        self.tool_registry = tool_registry
        self.model_registry = model_registry
        self.resolution_cache: Dict[str, Any] = {}
    
    async def resolve_schemas(self, pipeline_graph: PipelineGraph) -> Dict[str, Tuple[Optional[Dict], Optional[Dict]]]:
        """
        Resolve schemas for all tasks in the pipeline.
        
        Returns:
            Dictionary mapping task_id to (input_schema, output_schema)
        """
        # Phase 1: Static Analysis
        await self._static_schema_extraction(pipeline_graph)
        
        # Phase 2: Forward Propagation
        await self._forward_propagation(pipeline_graph)
        
        # Phase 3: Backward Propagation
        await self._backward_propagation(pipeline_graph)
        
        # Phase 4: Constraint Satisfaction
        await self._apply_constraints(pipeline_graph)
        
        # Collect results
        results = {}
        for node_id, node in pipeline_graph.nodes.items():
            results[node_id] = (node.input_schema, node.output_schema)
        
        return results
    
    async def _static_schema_extraction(self, graph: PipelineGraph):
        """Extract schemas from tool definitions and explicit schemas."""
        for node in graph.nodes.values():
            task = node.task
            
            # Check if task has explicit schemas
            if "validation" in task.metadata:
                validation_config = task.metadata["validation"]
                if "input_schema" in validation_config:
                    node.input_schema = validation_config["input_schema"]
                if "output_schema" in validation_config:
                    node.output_schema = validation_config["output_schema"]
            
            # Try to get schemas from tool registry
            if self.tool_registry and not (node.input_schema and node.output_schema):
                tool_name = task.action
                if not tool_name.startswith("<AUTO>"):  # Not an AUTO tag
                    tool = self.tool_registry.get(tool_name)
                    if tool:
                        # Extract parameter schemas
                        if not node.input_schema:
                            node.input_schema = self._extract_tool_input_schema(tool)
                        if not node.output_schema:
                            node.output_schema = self._extract_tool_output_schema(tool)
            
            # Update schema state
            if node.input_schema and node.output_schema:
                node.schema_state = SchemaState.FIXED
            elif node.input_schema or node.output_schema:
                node.schema_state = SchemaState.PARTIAL
    
    async def _forward_propagation(self, graph: PipelineGraph):
        """Propagate schemas forward through the graph."""
        # Process in topological order
        for node_id in graph.topological_sort():
            node = graph.nodes[node_id]
            
            if node.schema_state == SchemaState.FIXED:
                continue
            
            # Collect schemas from predecessors
            predecessor_outputs = []
            for pred_id in node.predecessors:
                pred_node = graph.nodes.get(pred_id)
                if pred_node and pred_node.output_schema:
                    predecessor_outputs.append({
                        "task_id": pred_id,
                        "schema": pred_node.output_schema
                    })
            
            # Try to infer input schema from predecessors
            if not node.input_schema and predecessor_outputs:
                node.input_schema = self._infer_input_from_predecessors(
                    node, predecessor_outputs, graph
                )
                if node.input_schema:
                    node.schema_state = SchemaState.PARTIAL
    
    async def _backward_propagation(self, graph: PipelineGraph):
        """Propagate schemas backward through the graph."""
        # Process in reverse topological order
        for node_id in reversed(graph.topological_sort()):
            node = graph.nodes[node_id]
            
            if node.schema_state == SchemaState.FIXED:
                continue
            
            # Collect expected inputs from successors
            successor_inputs = []
            for succ_id in node.successors:
                succ_node = graph.nodes.get(succ_id)
                if succ_node and succ_node.input_schema:
                    successor_inputs.append({
                        "task_id": succ_id,
                        "schema": succ_node.input_schema
                    })
            
            # Try to infer output schema from successors
            if not node.output_schema and successor_inputs:
                node.output_schema = self._infer_output_from_successors(
                    node, successor_inputs, graph
                )
                if node.output_schema:
                    node.schema_state = SchemaState.PARTIAL if not node.input_schema else SchemaState.FIXED
    
    async def _apply_constraints(self, graph: PipelineGraph):
        """Apply cross-cutting constraints and resolve remaining ambiguities."""
        # Apply pipeline-wide constraints
        if "validation" in graph.pipeline.metadata:
            global_validation = graph.pipeline.metadata["validation"]
            
            # Apply global constraints to ambiguous nodes
            for node in graph.nodes.values():
                if node.schema_state == SchemaState.AMBIGUOUS:
                    # Apply any global patterns or constraints
                    if "default_schemas" in global_validation:
                        defaults = global_validation["default_schemas"]
                        if not node.input_schema and "default_input" in defaults:
                            node.input_schema = defaults["default_input"]
                        if not node.output_schema and "default_output" in defaults:
                            node.output_schema = defaults["default_output"]
    
    def _extract_tool_input_schema(self, tool) -> Optional[Dict[str, Any]]:
        """Extract input schema from tool definition."""
        if not hasattr(tool, "parameters"):
            return None
        
        properties = {}
        required = []
        
        for param in tool.parameters:
            param_schema = {
                "type": self._map_param_type(param.get("type", "string"))
            }
            
            if "description" in param:
                param_schema["description"] = param["description"]
            
            properties[param["name"]] = param_schema
            
            if param.get("required", False):
                required.append(param["name"])
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    def _extract_tool_output_schema(self, tool) -> Optional[Dict[str, Any]]:
        """Extract output schema from tool definition."""
        # Tools typically return a standard format
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "result": {"type": "any"},
                "error": {"type": "string"}
            },
            "required": ["success"]
        }
    
    def _map_param_type(self, param_type: str) -> str:
        """Map parameter type to JSON Schema type."""
        type_mapping = {
            "string": "string",
            "integer": "integer",
            "number": "number",
            "boolean": "boolean",
            "array": "array",
            "object": "object",
            "any": "any"
        }
        return type_mapping.get(param_type, "string")
    
    def _infer_input_from_predecessors(
        self, 
        node: GraphNode, 
        predecessor_outputs: List[Dict],
        graph: PipelineGraph
    ) -> Optional[Dict[str, Any]]:
        """Infer input schema from predecessor outputs."""
        # Simple case: single predecessor
        if len(predecessor_outputs) == 1:
            pred = predecessor_outputs[0]
            # Check if there's a direct mapping
            edges = graph.reverse_edges.get(node.task_id, [])
            for edge in edges:
                if edge.source == pred["task_id"]:
                    # Use predecessor's output schema as base
                    return pred["schema"]
        
        # Multiple predecessors: combine schemas
        properties = {}
        for pred in predecessor_outputs:
            pred_id = pred["task_id"]
            if pred["schema"].get("type") == "object":
                # Nest under task ID
                properties[pred_id] = pred["schema"]
            else:
                # Wrap non-object schemas
                properties[pred_id] = {
                    "type": "object",
                    "properties": {
                        "result": pred["schema"]
                    }
                }
        
        return {
            "type": "object",
            "properties": properties
        }
    
    def _infer_output_from_successors(
        self,
        node: GraphNode,
        successor_inputs: List[Dict],
        graph: PipelineGraph
    ) -> Optional[Dict[str, Any]]:
        """Infer output schema from successor inputs."""
        # Look for common patterns in what successors expect
        if not successor_inputs:
            return None
        
        # For now, return a generic output schema
        # This will be enhanced with AUTO tag analysis
        return {
            "type": "object",
            "properties": {
                "result": {"type": "any"}
            }
        }