"""Pipeline and task specifications for declarative execution."""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..core.output_metadata import OutputMetadata, create_output_metadata, validate_output_specification


@dataclass
class ErrorHandling:
    """Error handling specification for a task."""

    action: str
    continue_on_error: bool = False
    retry_count: int = 0
    retry_delay: float = 1.0
    fallback_value: Optional[Any] = None


@dataclass
class LoopSpec:
    """Loop specification for iterative execution."""

    foreach: str
    max_iterations: Optional[int] = None
    parallel: bool = False
    collect_results: bool = True
    break_condition: Optional[str] = None


@dataclass
class TaskSpec:
    """Specification for a single pipeline task."""

    id: str
    action: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    tools: Optional[List[str]] = None
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[str] = None
    on_error: Optional[Union[str, ErrorHandling]] = None
    loop: Optional[LoopSpec] = None
    foreach: Optional[str] = None  # Simple foreach - converted to LoopSpec
    model_requirements: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = None
    cache_results: bool = True
    tags: List[str] = field(default_factory=list)
    
    # Output tracking fields
    produces: Optional[str] = None              # Output type descriptor
    location: Optional[str] = None              # Output location (supports templates)
    format: Optional[str] = None                # Output format/MIME type
    output_schema: Optional[Dict[str, Any]] = None  # JSON schema for output validation
    size_limit: Optional[int] = None            # Maximum output size in bytes
    output_description: Optional[str] = None    # Description of the output

    def __post_init__(self) -> None:
        """Validate and process task specification."""
        if not self.id:
            raise ValueError("Task ID is required")
        if not self.action:
            raise ValueError("Task action is required")

        # Ensure depends_on is a list
        if isinstance(self.depends_on, str):
            self.depends_on = [self.depends_on]

        # Convert simple foreach to LoopSpec
        if self.foreach and not self.loop:
            self.loop = LoopSpec(foreach=self.foreach)
            self.foreach = None

        # Convert simple error handling to ErrorHandling object
        if isinstance(self.on_error, str):
            self.on_error = ErrorHandling(action=self.on_error)

        # Ensure tags is a list
        if isinstance(self.tags, str):
            self.tags = [self.tags]

        # Validate output specification consistency
        if self.produces or self.location or self.format:
            validation_issues = validate_output_specification(
                produces=self.produces,
                location=self.location,
                format=self.format
            )
            if validation_issues:
                raise ValueError(f"Output specification validation failed: {'; '.join(validation_issues)}")

    def has_auto_tags(self) -> bool:
        """Check if action contains AUTO tags."""
        return "<AUTO>" in self.action and "</AUTO>" in self.action

    def extract_auto_content(self) -> str:
        """Extract content from AUTO tags."""
        if not self.has_auto_tags():
            return self.action

        match = re.search(r"<AUTO>(.*?)</AUTO>", self.action, re.DOTALL)
        return match.group(1).strip() if match else self.action

    def get_template_variables(self) -> List[str]:
        """Extract template variables from action (e.g., {{topic}})."""
        return re.findall(r"\{\{([^}]+)\}\}", self.action)

    def has_condition(self) -> bool:
        """Check if task has a condition."""
        return bool(self.condition)

    def has_loop(self) -> bool:
        """Check if task has loop configuration."""
        return bool(self.loop)

    def has_error_handling(self) -> bool:
        """Check if task has error handling configuration."""
        return bool(self.on_error)

    def is_iterative(self) -> bool:
        """Check if task is iterative (has loop or foreach)."""
        return self.has_loop()

    def get_loop_variable(self) -> Optional[str]:
        """Get the loop variable for iterative tasks."""
        return self.loop.foreach if self.loop else None

    def should_retry_on_error(self) -> bool:
        """Check if task should retry on error."""
        return (
            isinstance(self.on_error, ErrorHandling) and self.on_error.retry_count > 0
        )

    def should_continue_on_error(self) -> bool:
        """Check if pipeline should continue on task error."""
        return (
            isinstance(self.on_error, ErrorHandling) and self.on_error.continue_on_error
        )

    def get_condition_variables(self) -> List[str]:
        """Extract variables from condition expression."""
        if not self.condition:
            return []
        return re.findall(r"\{\{([^}]+)\}\}", self.condition)

    def is_conditional(self) -> bool:
        """Check if task execution is conditional."""
        return bool(self.condition)

    def get_execution_metadata(self) -> Dict[str, Any]:
        """Get metadata about task execution requirements."""
        return {
            "has_condition": self.has_condition(),
            "has_loop": self.has_loop(),
            "has_error_handling": self.has_error_handling(),
            "is_iterative": self.is_iterative(),
            "is_conditional": self.is_conditional(),
            "timeout": self.timeout,
            "cache_results": self.cache_results,
            "tags": self.tags,
            "condition_variables": self.get_condition_variables(),
            "template_variables": self.get_template_variables(),
            "has_output_metadata": self.has_output_metadata(),
            "output_metadata": self.get_output_metadata(),
        }

    def has_output_metadata(self) -> bool:
        """Check if task has output metadata defined."""
        return bool(self.produces or self.location or self.format)

    def get_output_metadata(self) -> Optional[OutputMetadata]:
        """Get output metadata for this task spec."""
        if not self.has_output_metadata():
            return None
        
        return create_output_metadata(
            produces=self.produces,
            location=self.location,
            format=self.format,
            schema=self.output_schema,
            size_limit=self.size_limit,
            description=self.output_description
        )

    def validate_output_consistency(self) -> List[str]:
        """Validate consistency of output metadata fields."""
        if not self.has_output_metadata():
            return []
        
        issues = validate_output_specification(
            produces=self.produces,
            location=self.location,
            format=self.format
        )
        
        # Additional validations
        if self.size_limit is not None and self.size_limit <= 0:
            issues.append("size_limit must be positive")
        
        return issues

    def get_output_reference(self, field: Optional[str] = None) -> str:
        """Get template reference string for this task's output."""
        if field:
            return f"{{{{{self.id}.{field}}}}}"
        else:
            return f"{{{{{self.id}.result}}}}"

    def is_file_output(self) -> bool:
        """Check if this task produces file output."""
        if not self.location:
            return False
        
        # Check if location looks like a file path
        return bool(
            self.location.startswith('./') or 
            self.location.startswith('/') or
            '/' in self.location or
            '\\' in self.location
        )


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

    def __post_init__(self) -> None:
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

    def _validate_dependencies(self) -> None:
        """Validate that all dependencies exist and there are no cycles."""
        step_ids = {step.id for step in self.steps}

        # Check that all dependencies exist
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in step_ids:
                    raise ValueError(
                        f"Step '{step.id}' depends on non-existent step '{dep}'"
                    )

        # Check for circular dependencies using topological sort
        self._check_circular_dependencies()

    def _check_circular_dependencies(self) -> None:
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
                    raise ValueError(
                        f"Circular dependency detected involving step '{step.id}'"
                    )

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
            raise ValueError(
                "Cannot determine execution order - circular dependencies exist"
            )

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
            required = input_spec.get("required", True)
            if required and input_name not in provided_inputs:
                raise ValueError(f"Required input '{input_name}' not provided")
        return True

    def get_steps_with_output_metadata(self) -> List[TaskSpec]:
        """Get all steps that have output metadata defined."""
        return [step for step in self.steps if step.has_output_metadata()]

    def get_output_producing_steps(self) -> Dict[str, TaskSpec]:
        """Get mapping of output types to steps that produce them."""
        output_map = {}
        for step in self.steps:
            if step.produces:
                output_map[step.produces] = step
        return output_map

    def validate_output_references(self) -> List[str]:
        """Validate that all output references in the pipeline are valid."""
        issues = []
        step_ids = {step.id for step in self.steps}
        
        for step in self.steps:
            # Check if step references other steps' outputs
            step_content = str(step.inputs) + str(step.action)
            
            # Look for references like {{ step_id.field }}
            import re
            references = re.findall(r'{{\s*([^}]+)\s*}}', step_content)
            
            for ref in references:
                if '.' in ref:
                    parts = ref.split('.')
                    referenced_step = parts[0].strip()
                    if referenced_step in step_ids and referenced_step != step.id:
                        # Check if the referenced step comes before this step in dependencies
                        if referenced_step not in step.depends_on:
                            issues.append(
                                f"Step '{step.id}' references output from '{referenced_step}' "
                                f"but doesn't declare it as a dependency"
                            )
        
        return issues

    def get_pipeline_output_summary(self) -> Dict[str, Any]:
        """Get summary of all pipeline outputs."""
        summary = {
            "total_steps": len(self.steps),
            "steps_with_outputs": len(self.get_steps_with_output_metadata()),
            "output_types": {},
            "file_outputs": [],
            "structured_outputs": []
        }
        
        for step in self.steps:
            if step.has_output_metadata():
                if step.produces:
                    if step.produces not in summary["output_types"]:
                        summary["output_types"][step.produces] = []
                    summary["output_types"][step.produces].append(step.id)
                
                if step.is_file_output():
                    summary["file_outputs"].append({
                        "step_id": step.id,
                        "location": step.location,
                        "format": step.format
                    })
                
                if step.output_schema:
                    summary["structured_outputs"].append({
                        "step_id": step.id,
                        "schema": step.output_schema
                    })
        
        return summary
