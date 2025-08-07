"""
Type definitions and data structures for the automatic graph generation system.

This module defines the core data structures used throughout the graph generation
pipeline, including parsed pipeline representations, dependency graphs, control flow
maps, and other foundational types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Union
from abc import ABC, abstractmethod


class StepType(Enum):
    """Types of pipeline steps supported by the graph generator."""
    STANDARD = "standard"
    PARALLEL_MAP = "parallel_map"
    LOOP = "loop" 
    WHILE = "while"
    FOR = "for"
    CONDITIONAL = "conditional"


class DependencyType(Enum):
    """Types of dependencies between pipeline steps."""
    EXPLICIT = "explicit"      # From depends_on arrays
    IMPLICIT = "implicit"      # From template variable references
    CONDITIONAL = "conditional"  # From conditional execution paths
    CONTROL_FLOW = "control_flow"  # From loops and control structures


class ParallelizationType(Enum):
    """Types of parallelization strategies."""
    INDEPENDENT = "independent"  # Steps with no shared dependencies
    MAP_REDUCE = "map_reduce"    # Parallel processing over collections
    FAN_OUT_FAN_IN = "fan_out_fan_in"  # Broadcast input to multiple processors


@dataclass
class InputSchema:
    """Schema definition for pipeline inputs."""
    name: str
    type: str
    required: bool = True
    default: Any = None
    description: Optional[str] = None
    enum: Optional[List[Any]] = None
    range: Optional[List[Union[int, float]]] = None
    example: Any = None
    
    def validate_value(self, value: Any) -> bool:
        """Validate that a value matches this input schema."""
        # Basic type validation - can be enhanced
        if self.enum and value not in self.enum:
            return False
        if self.range and isinstance(value, (int, float)):
            return self.range[0] <= value <= self.range[1]
        return True


@dataclass
class OutputSchema:
    """Schema definition for step outputs."""
    name: str
    type: str
    description: Optional[str] = None
    schema: Optional[Dict[str, Any]] = None  # For complex types
    computed_as: Optional[str] = None  # Template expression for computed outputs
    format: Optional[str] = None  # e.g., "markdown", "json"
    
    
@dataclass
class ParsedStep:
    """Parsed representation of a pipeline step."""
    id: str
    type: StepType = StepType.STANDARD
    tool: Optional[str] = None
    action: Optional[str] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, OutputSchema] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[str] = None
    model_requirements: Optional[Dict[str, Any]] = None
    
    # Control flow specific fields
    items: Optional[str] = None  # For parallel_map iteration
    substeps: Optional[List[ParsedStep]] = None  # For nested steps
    max_iterations: Optional[int] = None  # For loops
    goto: Optional[str] = None  # For goto statements
    else_step: Optional[str] = None  # For conditional branching
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    original_definition: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedPipeline:
    """Parsed representation of a complete pipeline."""
    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    version: str = "1.0.0"
    inputs: Dict[str, InputSchema] = field(default_factory=dict)
    outputs: Dict[str, OutputSchema] = field(default_factory=dict)
    steps: List[ParsedStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    
    def get_step_by_id(self, step_id: str) -> Optional[ParsedStep]:
        """Get a step by its ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None
        
    def get_step_ids(self) -> List[str]:
        """Get all step IDs in order."""
        return [step.id for step in self.steps]


@dataclass 
class DependencyEdge:
    """Represents a dependency edge between two steps."""
    source: str
    target: str
    dependency_type: DependencyType
    condition: Optional[str] = None  # For conditional dependencies
    weight: float = 1.0  # For optimization algorithms
    metadata: Dict[str, Any] = field(default_factory=dict)


class DependencyGraph:
    """Graph representation of step dependencies."""
    
    def __init__(self):
        self.nodes: Dict[str, ParsedStep] = {}
        self.edges: List[DependencyEdge] = []
        self._adjacency_list: Dict[str, List[str]] = {}
        self._reverse_adjacency_list: Dict[str, List[str]] = {}
        self._execution_order: Optional[List[str]] = None
        
    def add_node(self, step_id: str, step: ParsedStep) -> None:
        """Add a step node to the graph."""
        self.nodes[step_id] = step
        if step_id not in self._adjacency_list:
            self._adjacency_list[step_id] = []
        if step_id not in self._reverse_adjacency_list:
            self._reverse_adjacency_list[step_id] = []
            
    def add_edge(self, source: str, target: str, 
                 dependency_type: DependencyType = DependencyType.EXPLICIT,
                 condition: Optional[str] = None,
                 weight: float = 1.0) -> None:
        """Add a dependency edge between two steps."""
        if source not in self.nodes or target not in self.nodes:
            raise ValueError(f"Both source ({source}) and target ({target}) must be added as nodes first")
            
        edge = DependencyEdge(
            source=source,
            target=target, 
            dependency_type=dependency_type,
            condition=condition,
            weight=weight
        )
        self.edges.append(edge)
        
        # Update adjacency lists
        if target not in self._adjacency_list[source]:
            self._adjacency_list[source].append(target)
        if source not in self._reverse_adjacency_list[target]:
            self._reverse_adjacency_list[target].append(source)
            
    def has_node(self, step_id: str) -> bool:
        """Check if a node exists in the graph."""
        return step_id in self.nodes
        
    def has_cycles(self) -> bool:
        """Check if the graph has circular dependencies."""
        visited = set()
        recursion_stack = set()
        
        def has_cycle_util(node: str) -> bool:
            visited.add(node)
            recursion_stack.add(node)
            
            for neighbor in self._adjacency_list.get(node, []):
                if neighbor not in visited:
                    if has_cycle_util(neighbor):
                        return True
                elif neighbor in recursion_stack:
                    return True
                    
            recursion_stack.remove(node)
            return False
            
        for node in self.nodes:
            if node not in visited:
                if has_cycle_util(node):
                    return True
        return False
        
    def find_cycles(self) -> List[List[str]]:
        """Find all cycles in the graph."""
        cycles = []
        visited = set()
        recursion_stack = []
        
        def find_cycles_util(node: str, path: List[str]) -> None:
            if node in path:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return
                
            if node in visited:
                return
                
            visited.add(node)
            path.append(node)
            
            for neighbor in self._adjacency_list.get(node, []):
                find_cycles_util(neighbor, path.copy())
                
        for node in self.nodes:
            if node not in visited:
                find_cycles_util(node, [])
                
        return cycles
        
    def get_entry_points(self) -> List[str]:
        """Get nodes with no incoming dependencies."""
        return [node for node in self.nodes if not self._reverse_adjacency_list[node]]
        
    def get_terminal_nodes(self) -> List[str]:
        """Get nodes with no outgoing dependencies.""" 
        return [node for node in self.nodes if not self._adjacency_list[node]]
        
    def topological_sort(self) -> List[str]:
        """Get topological ordering of nodes."""
        if self.has_cycles():
            raise ValueError("Cannot perform topological sort on graph with cycles")
            
        in_degree = {node: 0 for node in self.nodes}
        for edge in self.edges:
            in_degree[edge.target] += 1
            
        queue = [node for node in self.nodes if in_degree[node] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for neighbor in self._adjacency_list[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    
        return result
        
    def get_execution_levels(self) -> Dict[int, List[str]]:
        """Group nodes by execution level for parallel processing."""
        levels = {}
        node_levels = {}  # Cache computed levels
        
        def get_level(node: str, visiting: set) -> int:
            if node in node_levels:
                return node_levels[node]
                
            if node in visiting:
                # Cycle detected, break it
                return 0
                
            visiting.add(node)
            dependencies = self._reverse_adjacency_list[node]
            
            if not dependencies:
                level = 0  # Entry point
            else:
                max_dep_level = max(get_level(dep, visiting) for dep in dependencies)
                level = max_dep_level + 1
                
            visiting.remove(node)
            node_levels[node] = level
            return level
                
        for node in self.nodes:
            level = get_level(node, set())
            if level not in levels:
                levels[level] = []
            levels[level].append(node)
            
        return levels
        
    def set_execution_order(self, order: List[str]) -> None:
        """Set optimized execution order."""
        self._execution_order = order
        
    def get_execution_order(self) -> List[str]:
        """Get execution order (topological if not set)."""
        if self._execution_order:
            return self._execution_order
        else:
            return self.topological_sort()


@dataclass
class ParallelGroup:
    """Group of steps that can execute in parallel."""
    steps: List[str]
    parallelization_type: ParallelizationType
    execution_level: int
    estimated_speedup: float = 1.0
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    max_concurrency: Optional[int] = None


@dataclass
class ConditionalLogic:
    """Represents conditional execution logic."""
    condition_template: str
    evaluator_function: Callable[[Dict[str, Any]], bool]
    required_data: List[str]
    true_path: str
    false_path: Optional[str] = None


@dataclass 
class LoopLogic:
    """Represents loop execution logic."""
    loop_type: StepType  # LOOP, WHILE, FOR
    condition_template: Optional[str] = None
    items_expression: Optional[str] = None  # For FOR loops
    max_iterations: int = 100
    substeps: List[ParsedStep] = field(default_factory=list)


@dataclass
class ParallelMapLogic:
    """Represents parallel map execution logic."""
    items_expression: str  # Template expression that evaluates to list
    item_variable_name: str = "item"
    substeps: List[ParsedStep] = field(default_factory=list)
    max_concurrency: Optional[int] = None


class ControlFlowMap:
    """Map of all control flow logic in a pipeline."""
    
    def __init__(self):
        self.conditionals: Dict[str, ConditionalLogic] = {}
        self.loops: Dict[str, LoopLogic] = {}
        self.parallel_maps: Dict[str, ParallelMapLogic] = {}
        self.goto_statements: Dict[str, str] = {}
        self.dynamic_routing: Dict[str, Any] = {}
        
    def add_conditional(self, step_id: str, logic: ConditionalLogic) -> None:
        """Add conditional logic for a step."""
        self.conditionals[step_id] = logic
        
    def add_loop(self, step_id: str, logic: LoopLogic) -> None:
        """Add loop logic for a step."""
        self.loops[step_id] = logic
        
    def add_parallel_map(self, step_id: str, logic: ParallelMapLogic) -> None:
        """Add parallel map logic for a step."""
        self.parallel_maps[step_id] = logic
        
    def add_goto(self, step_id: str, target: str) -> None:
        """Add goto statement for a step."""
        self.goto_statements[step_id] = target
        
    def add_dynamic_routing(self, step_id: str, routing_logic: Any) -> None:
        """Add dynamic routing logic for a step."""
        self.dynamic_routing[step_id] = routing_logic


@dataclass
class ValidationError:
    """Represents a data flow validation error."""
    step_id: str
    input_name: str
    variable: str
    error: str
    suggestion: Optional[str] = None
    line_number: Optional[int] = None


class DataFlowSchema:
    """Schema representing complete data flow through a pipeline."""
    
    def __init__(self):
        self.inputs: Dict[str, InputSchema] = {}
        self.outputs: Dict[str, OutputSchema] = {}
        self.step_schemas: Dict[str, Dict[str, OutputSchema]] = {}
        self.validation_errors: List[ValidationError] = []
        
    def add_step_schema(self, step_id: str, outputs: Dict[str, OutputSchema]) -> None:
        """Add output schema for a step."""
        self.step_schemas[step_id] = outputs
        
    def get_available_variables(self) -> Dict[str, str]:
        """Get all available variables and their sources."""
        variables = {}
        
        # Add input variables
        for input_name in self.inputs:
            variables[f"inputs.{input_name}"] = "pipeline_input"
            
        # Add step output variables
        for step_id, outputs in self.step_schemas.items():
            for output_name in outputs:
                variables[f"{step_id}.{output_name}"] = f"step_output:{step_id}"
                
        return variables
        
    def validate_variable_reference(self, variable_path: str) -> bool:
        """Check if a variable reference is valid."""
        available = self.get_available_variables()
        return variable_path in available