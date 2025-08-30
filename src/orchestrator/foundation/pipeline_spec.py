"""
Pipeline specification data structures for the refactored architecture.

This module defines the data structures that represent compiled YAML pipelines
in their executable form, ready for LangGraph StateGraph execution.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pathlib import Path


class ModelSelectionStrategy(Enum):
    """Strategy for model selection."""
    COST = "cost"
    PERFORMANCE = "performance" 
    BALANCED = "balanced"


@dataclass
class PipelineHeader:
    """
    Pipeline header containing metadata and configuration.
    
    Based on the specification from the PRD:
    ```yaml
    id: research_report_pipeline
    name: Research Report Generator
    description: Generate comprehensive research reports
    orchestrator: ollama:llama3.2-70b
    default_model: openai:gpt-5
    experts:
      - web_search: gemini:gemini-2.0-flash-thinking-exp
    selection_schema: balanced
    inputs:
      - output_dir: /examples/outputs/research_report_pipeline/
      - topic: None
    ```
    """
    
    id: str
    name: str
    description: Optional[str] = None
    orchestrator: Optional[str] = None
    default_model: Optional[str] = None
    experts: Dict[str, str] = field(default_factory=dict)
    selection_strategy: ModelSelectionStrategy = ModelSelectionStrategy.BALANCED
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    
    def __post_init__(self):
        """Validate header after initialization."""
        if not self.id:
            raise ValueError("Pipeline ID cannot be empty")
        if not self.name:
            raise ValueError("Pipeline name cannot be empty")
        
        # Convert string selection strategy to enum
        if isinstance(self.selection_strategy, str):
            self.selection_strategy = ModelSelectionStrategy(self.selection_strategy)


@dataclass 
class PipelineStep:
    """
    Individual pipeline step specification.
    
    Based on the specification from the PRD:
    ```yaml
    - id: research_phase
      name: Research Phase
      description: Conduct comprehensive research
      dependencies: []
      tools: [web_search, document_analyzer]
      model: gemini:gemini-2.0-flash-thinking-exp
      personality: "research_specialist"
      condition: "topic is not None"
      vars:
        - research_findings: "Structured research data"
      products:
        - research_findings: "research_summary.md"
    ```
    """
    
    id: str
    name: str
    description: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    model: Optional[str] = None
    personality: Optional[str] = None
    condition: Optional[str] = None
    variables: Dict[str, str] = field(default_factory=dict)  # var_name -> description
    products: Dict[str, str] = field(default_factory=dict)   # var_name -> filename
    parameters: Dict[str, Any] = field(default_factory=dict)  # Additional step parameters
    timeout: Optional[int] = None  # Step timeout in seconds
    retry_count: int = 0  # Number of retries on failure
    
    def __post_init__(self):
        """Validate step after initialization."""
        if not self.id:
            raise ValueError("Step ID cannot be empty")
        if not self.name:
            raise ValueError("Step name cannot be empty")
        
        # Validate dependencies don't include self
        if self.id in self.dependencies:
            raise ValueError(f"Step '{self.id}' cannot depend on itself")


@dataclass
class PipelineSpecification:
    """
    Complete compiled pipeline specification ready for execution.
    
    This represents the result of compiling a YAML pipeline definition
    into an executable format that can be run by the execution engine.
    """
    
    header: PipelineHeader
    steps: List[PipelineStep]
    compiled_at: float = field(default_factory=lambda: __import__('time').time())
    compiler_version: str = "2.0.0"
    
    # LLM-generated metadata (filled during compilation)
    intention: Optional[str] = None      # LLM-generated intention summary
    architecture: Optional[str] = None  # LLM-generated architecture description
    
    def __post_init__(self):
        """Validate specification after initialization."""
        # Ensure all step IDs are unique
        step_ids = [step.id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            duplicates = [id for id in step_ids if step_ids.count(id) > 1]
            raise ValueError(f"Duplicate step IDs found: {duplicates}")
        
        # Validate all dependencies exist
        for step in self.steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    raise ValueError(f"Step '{step.id}' has invalid dependency '{dep}'")
        
        # Check for circular dependencies
        self._validate_no_cycles()
    
    def _validate_no_cycles(self):
        """Validate no circular dependencies exist."""
        from collections import defaultdict, deque
        
        # Build adjacency list
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        for step in self.steps:
            in_degree[step.id] = len(step.dependencies)
            for dep in step.dependencies:
                graph[dep].append(step.id)
        
        # Topological sort to detect cycles
        queue = deque([step.id for step in self.steps if in_degree[step.id] == 0])
        processed = 0
        
        while queue:
            current = queue.popleft()
            processed += 1
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if processed != len(self.steps):
            raise ValueError("Circular dependencies detected in pipeline steps")
    
    def get_step(self, step_id: str) -> Optional[PipelineStep]:
        """Get step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None
    
    def get_execution_order(self) -> List[List[str]]:
        """Get execution order grouped by parallel execution levels."""
        from collections import defaultdict, deque
        
        # Build dependency graph
        in_degree = {step.id: len(step.dependencies) for step in self.steps}
        graph = defaultdict(list)
        
        for step in self.steps:
            for dep in step.dependencies:
                graph[dep].append(step.id)
        
        # Topological sort with level grouping
        levels = []
        queue = deque([step_id for step_id, degree in in_degree.items() if degree == 0])
        
        while queue:
            current_level = []
            level_size = len(queue)
            
            for _ in range(level_size):
                step_id = queue.popleft()
                current_level.append(step_id)
                
                for neighbor in graph[step_id]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
            
            if current_level:
                levels.append(current_level)
        
        return levels
    
    def get_dependencies(self, step_id: str) -> List[str]:
        """Get dependencies for a step."""
        step = self.get_step(step_id)
        return step.dependencies if step else []
    
    def get_dependents(self, step_id: str) -> List[str]:
        """Get steps that depend on the given step."""
        dependents = []
        for step in self.steps:
            if step_id in step.dependencies:
                dependents.append(step.id)
        return dependents
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "header": {
                "id": self.header.id,
                "name": self.header.name,
                "description": self.header.description,
                "orchestrator": self.header.orchestrator,
                "default_model": self.header.default_model,
                "experts": self.header.experts,
                "selection_strategy": self.header.selection_strategy.value,
                "inputs": self.header.inputs,
                "outputs": self.header.outputs,
                "version": self.header.version,
            },
            "steps": [
                {
                    "id": step.id,
                    "name": step.name,
                    "description": step.description,
                    "dependencies": step.dependencies,
                    "tools": step.tools,
                    "model": step.model,
                    "personality": step.personality,
                    "condition": step.condition,
                    "variables": step.variables,
                    "products": step.products,
                    "parameters": step.parameters,
                    "timeout": step.timeout,
                    "retry_count": step.retry_count,
                }
                for step in self.steps
            ],
            "compiled_at": self.compiled_at,
            "compiler_version": self.compiler_version,
            "intention": self.intention,
            "architecture": self.architecture,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PipelineSpecification:
        """Create from dictionary representation."""
        header_data = data["header"]
        header = PipelineHeader(
            id=header_data["id"],
            name=header_data["name"],
            description=header_data.get("description"),
            orchestrator=header_data.get("orchestrator"),
            default_model=header_data.get("default_model"),
            experts=header_data.get("experts", {}),
            selection_strategy=ModelSelectionStrategy(header_data.get("selection_strategy", "balanced")),
            inputs=header_data.get("inputs", {}),
            outputs=header_data.get("outputs", {}),
            version=header_data.get("version", "1.0.0"),
        )
        
        steps = []
        for step_data in data["steps"]:
            step = PipelineStep(
                id=step_data["id"],
                name=step_data["name"],
                description=step_data.get("description"),
                dependencies=step_data.get("dependencies", []),
                tools=step_data.get("tools", []),
                model=step_data.get("model"),
                personality=step_data.get("personality"),
                condition=step_data.get("condition"),
                variables=step_data.get("variables", {}),
                products=step_data.get("products", {}),
                parameters=step_data.get("parameters", {}),
                timeout=step_data.get("timeout"),
                retry_count=step_data.get("retry_count", 0),
            )
            steps.append(step)
        
        return cls(
            header=header,
            steps=steps,
            compiled_at=data.get("compiled_at", __import__('time').time()),
            compiler_version=data.get("compiler_version", "2.0.0"),
            intention=data.get("intention"),
            architecture=data.get("architecture"),
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"PipelineSpecification(id='{self.header.id}', steps={len(self.steps)})"