"""Pipeline and task specifications for declarative execution."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
import re


@dataclass
class TaskSpec:
    """Specification for a single pipeline task."""
    
    id: str
    action: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    tools: Optional[List[str]] = None
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[str] = None
    on_error: Optional[str] = None
    foreach: Optional[str] = None
    model_requirements: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and process task specification."""
        if not self.id:
            raise ValueError("Task ID is required")
        if not self.action:
            raise ValueError("Task action is required")
            
        # Ensure depends_on is a list
        if isinstance(self.depends_on, str):
            self.depends_on = [self.depends_on]
    
    def has_auto_tags(self) -> bool:
        """Check if action contains AUTO tags."""
        return "<AUTO>" in self.action and "</AUTO>" in self.action
    
    def extract_auto_content(self) -> str:
        """Extract content from AUTO tags."""
        if not self.has_auto_tags():
            return self.action
            
        match = re.search(r'<AUTO>(.*?)</AUTO>', self.action, re.DOTALL)
        return match.group(1).strip() if match else self.action
    
    def get_template_variables(self) -> List[str]:
        """Extract template variables from action (e.g., {{topic}})."""
        return re.findall(r'\{\{([^}]+)\}\}', self.action)


@dataclass  
class PipelineSpec:
    """Specification for a complete pipeline."""
    
    name: str
    description: str = ""
    version: str = "1.0.0"
    inputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    steps: List[TaskSpec] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and process pipeline specification."""
        if not self.name:
            raise ValueError("Pipeline name is required")
        if not self.steps:
            raise ValueError("Pipeline must have at least one step")
            
        # Convert step dicts to TaskSpec objects if needed
        processed_steps = []
        for step in self.steps:
            if isinstance(step, dict):
                processed_steps.append(TaskSpec(**step))
            elif isinstance(step, TaskSpec):
                processed_steps.append(step)
            else:
                raise ValueError(f"Invalid step type: {type(step)}")
        self.steps = processed_steps
        
        # Validate dependencies
        self._validate_dependencies()
    
    def _validate_dependencies(self):
        """Validate that all dependencies exist and there are no cycles."""
        step_ids = {step.id for step in self.steps}
        
        # Check that all dependencies exist
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in step_ids:
                    raise ValueError(f"Step '{step.id}' depends on non-existent step '{dep}'")
        
        # Check for circular dependencies using topological sort
        self._check_circular_dependencies()
    
    def _check_circular_dependencies(self):
        """Check for circular dependencies using DFS."""
        visited = set()
        rec_stack = set()
        
        def has_cycle(step_id: str) -> bool:
            visited.add(step_id)
            rec_stack.add(step_id)
            
            # Find step by ID
            step = next((s for s in self.steps if s.id == step_id), None)
            if not step:
                return False
                
            # Visit all dependencies
            for dep in step.depends_on:
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True
            
            rec_stack.remove(step_id)
            return False
        
        # Check all steps
        for step in self.steps:
            if step.id not in visited:
                if has_cycle(step.id):
                    raise ValueError(f"Circular dependency detected involving step '{step.id}'")
    
    def get_execution_order(self) -> List[TaskSpec]:
        """Get steps in valid execution order (topological sort)."""
        # Build dependency graph
        in_degree = {step.id: 0 for step in self.steps}
        graph = {step.id: [] for step in self.steps}
        step_map = {step.id: step for step in self.steps}
        
        for step in self.steps:
            for dep in step.depends_on:
                graph[dep].append(step.id)
                in_degree[step.id] += 1
        
        # Topological sort using Kahn's algorithm
        queue = [step_id for step_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(step_map[current])
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) != len(self.steps):
            raise ValueError("Cannot determine execution order - circular dependencies exist")
        
        return result
    
    def get_steps_with_auto_tags(self) -> List[TaskSpec]:
        """Get all steps that contain AUTO tags."""
        return [step for step in self.steps if step.has_auto_tags()]
    
    def get_required_tools(self) -> List[str]:
        """Get all tools required by pipeline steps."""
        tools = set()
        for step in self.steps:
            if step.tools:
                tools.update(step.tools)
        return list(tools)
    
    def validate_inputs(self, provided_inputs: Dict[str, Any]) -> bool:
        """Validate that required inputs are provided."""
        for input_name, input_spec in self.inputs.items():
            required = input_spec.get('required', True)
            if required and input_name not in provided_inputs:
                raise ValueError(f"Required input '{input_name}' not provided")
        return True