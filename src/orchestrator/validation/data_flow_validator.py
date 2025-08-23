"""Data flow validation for pipeline compilation.

This module provides comprehensive data flow validation to ensure:
- Template variable references point to valid task outputs
- Data flow between pipeline steps is valid
- Output/input compatibility between connected tasks
- Data transformations are valid
- Referenced task outputs exist

Issue #241 Stream 4: Data Flow Validation
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from jinja2 import Environment, TemplateSyntaxError, meta

logger = logging.getLogger(__name__)


@dataclass
class DataFlowError:
    """Represents a data flow validation error."""
    
    task_id: str
    parameter_name: Optional[str]
    error_type: str
    message: str
    variable_reference: Optional[str] = None
    source_task: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    severity: str = "error"  # error, warning
    
    def __str__(self) -> str:
        if self.parameter_name:
            context = f"Task '{self.task_id}' parameter '{self.parameter_name}'"
        else:
            context = f"Task '{self.task_id}'"
        
        result = f"{context} {self.error_type}: {self.message}"
        if self.suggestions:
            result += f"\nSuggestions: {', '.join(self.suggestions)}"
        return result


@dataclass 
class DataFlowResult:
    """Result of data flow validation."""
    
    valid: bool
    errors: List[DataFlowError] = field(default_factory=list)
    warnings: List[DataFlowError] = field(default_factory=list)
    data_flow_graph: Dict[str, Set[str]] = field(default_factory=dict)  # task_id -> dependencies
    available_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # task_id -> outputs
    validated_tasks: int = 0
    
    @property
    def has_errors(self) -> bool:
        """Check if there are validation errors."""
        return bool(self.errors)
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are validation warnings."""
        return bool(self.warnings)
    
    def summary(self) -> str:
        """Get a summary of validation results."""
        return (f"Data flow validation: {self.validated_tasks} tasks, "
                f"{len(self.errors)} errors, {len(self.warnings)} warnings")


@dataclass
class TaskOutputSchema:
    """Schema information for a task's outputs."""
    
    task_id: str
    outputs: Dict[str, Any] = field(default_factory=dict)  # output_name -> schema/type info
    produces: Optional[str] = None  # what the task produces (file, data, etc.)
    format: Optional[str] = None    # output format
    
    def get_available_variables(self) -> Set[str]:
        """Get all variable names this task makes available."""
        variables = set()
        
        # Standard outputs
        variables.add(f"{self.task_id}.result")
        variables.add(f"{self.task_id}.output")
        
        # Tool-specific outputs based on known patterns
        # These are common output patterns from various tools
        if self.task_id:
            variables.update({
                f"{self.task_id}.content",
                f"{self.task_id}.data", 
                f"{self.task_id}.text",
                f"{self.task_id}.response",
                f"{self.task_id}.value",
                f"{self.task_id}.results",
                f"{self.task_id}.outputs",
                f"{self.task_id}.status",
                f"{self.task_id}.success",
                f"{self.task_id}.error"
            })
        
        # Add explicit outputs if defined
        for output_name in self.outputs.keys():
            variables.add(f"{self.task_id}.{output_name}")
            
        return variables


class DataFlowValidator:
    """Validates data flow between pipeline steps."""
    
    def __init__(self, 
                 development_mode: bool = False,
                 tool_validator: Optional[Any] = None):
        """
        Initialize data flow validator.
        
        Args:
            development_mode: If True, allows some validation bypasses for development
            tool_validator: ToolValidator instance to get tool schemas
        """
        self.development_mode = development_mode
        self.tool_validator = tool_validator
        
        # Jinja2 environment for template analysis
        self.jinja_env = Environment()
        
        # Pattern for extracting template variables
        self.template_var_pattern = re.compile(r'\{\{\s*([^}]+)\s*\}\}')
        
        logger.debug(f"DataFlowValidator initialized (development_mode={development_mode})")
    
    def validate_pipeline_data_flow(self, pipeline_def: Dict[str, Any]) -> DataFlowResult:
        """
        Validate data flow for an entire pipeline.
        
        Args:
            pipeline_def: Complete pipeline definition
            
        Returns:
            DataFlowResult with validation details
        """
        errors: List[DataFlowError] = []
        warnings: List[DataFlowError] = []
        data_flow_graph: Dict[str, Set[str]] = {}
        available_outputs: Dict[str, Dict[str, Any]] = {}
        validated_tasks = 0
        
        steps = pipeline_def.get("steps", [])
        
        # First pass: collect all task outputs and build output schemas
        task_schemas: Dict[str, TaskOutputSchema] = {}
        
        for step in steps:
            if not isinstance(step, dict):
                continue
                
            task_id = step.get("id", "unknown")
            validated_tasks += 1
            
            # Create output schema for this task
            schema = self._create_task_output_schema(step)
            task_schemas[task_id] = schema
            available_outputs[task_id] = schema.outputs
            
            # Initialize data flow graph entry
            data_flow_graph[task_id] = set()
        
        # Second pass: validate data flow and template references
        for step in steps:
            if not isinstance(step, dict):
                continue
                
            task_id = step.get("id", "unknown")
            
            # Validate template references in this task
            task_errors, task_warnings, task_dependencies = self._validate_task_data_flow(
                step, task_schemas, pipeline_def.get("inputs", {})
            )
            
            errors.extend(task_errors)
            warnings.extend(task_warnings)
            data_flow_graph[task_id].update(task_dependencies)
        
        # Third pass: validate data flow graph for cycles and missing dependencies
        graph_errors, graph_warnings = self._validate_data_flow_graph(data_flow_graph, steps)
        errors.extend(graph_errors)
        warnings.extend(graph_warnings)
        
        # Determine overall validity
        valid = len(errors) == 0 or (self.development_mode and self._are_errors_bypassable(errors))
        
        return DataFlowResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            data_flow_graph=data_flow_graph,
            available_outputs=available_outputs,
            validated_tasks=validated_tasks
        )
    
    def _create_task_output_schema(self, step: Dict[str, Any]) -> TaskOutputSchema:
        """Create output schema for a task based on its definition."""
        task_id = step.get("id", "unknown")
        
        # Get explicit outputs if defined
        outputs = {}
        if "outputs" in step:
            outputs = step["outputs"]
        elif "produces" in step:
            # Infer outputs from produces field
            produces = step["produces"]
            if produces == "file":
                outputs["file_path"] = {"type": "string", "description": "Path to generated file"}
            elif produces == "data":
                outputs["data"] = {"type": "object", "description": "Generated data"}
            elif produces == "text":
                outputs["text"] = {"type": "string", "description": "Generated text"}
        
        # Get tool/action name for schema inference
        tool_name = step.get("action") or step.get("tool")
        
        # Use tool validator to get tool schema if available
        if self.tool_validator and tool_name:
            tool_schema = self.tool_validator.get_tool_schema(tool_name)
            if tool_schema:
                # Try to infer outputs from tool schema
                output_schema = tool_schema.get("outputSchema", {})
                if output_schema:
                    outputs.update(output_schema.get("properties", {}))
        
        return TaskOutputSchema(
            task_id=task_id,
            outputs=outputs,
            produces=step.get("produces"),
            format=step.get("format")
        )
    
    def _validate_task_data_flow(self, 
                               step: Dict[str, Any], 
                               task_schemas: Dict[str, TaskOutputSchema],
                               pipeline_inputs: Dict[str, Any]) -> Tuple[List[DataFlowError], List[DataFlowError], Set[str]]:
        """
        Validate data flow for a single task.
        
        Returns:
            Tuple of (errors, warnings, dependencies)
        """
        errors: List[DataFlowError] = []
        warnings: List[DataFlowError] = []
        dependencies: Set[str] = set()
        
        task_id = step.get("id", "unknown")
        parameters = step.get("parameters", {})
        
        # Check explicit dependencies
        explicit_deps = step.get("dependencies", step.get("depends_on", []))
        if isinstance(explicit_deps, str):
            explicit_deps = [explicit_deps.strip()] if explicit_deps.strip() else []
        dependencies.update(explicit_deps)
        
        # Analyze template variables in parameters
        param_errors, param_warnings, param_deps = self._analyze_parameters_data_flow(
            parameters, task_id, task_schemas, pipeline_inputs
        )
        errors.extend(param_errors)
        warnings.extend(param_warnings)
        dependencies.update(param_deps)
        
        # Analyze template variables in other fields
        for field_name in ["condition", "while", "if", "for_each"]:
            if field_name in step:
                field_value = step[field_name]
                if isinstance(field_value, str):
                    field_errors, field_warnings, field_deps = self._analyze_template_string(
                        field_value, task_id, field_name, task_schemas, pipeline_inputs
                    )
                    errors.extend(field_errors)
                    warnings.extend(field_warnings)
                    dependencies.update(field_deps)
        
        return errors, warnings, dependencies
    
    def _analyze_parameters_data_flow(self, 
                                    parameters: Dict[str, Any],
                                    task_id: str,
                                    task_schemas: Dict[str, TaskOutputSchema],
                                    pipeline_inputs: Dict[str, Any],
                                    param_path: str = "") -> Tuple[List[DataFlowError], List[DataFlowError], Set[str]]:
        """Recursively analyze parameters for data flow validation."""
        errors: List[DataFlowError] = []
        warnings: List[DataFlowError] = []
        dependencies: Set[str] = set()
        
        for param_name, param_value in parameters.items():
            current_path = f"{param_path}.{param_name}" if param_path else param_name
            
            if isinstance(param_value, str):
                # Analyze template string
                param_errors, param_warnings, param_deps = self._analyze_template_string(
                    param_value, task_id, current_path, task_schemas, pipeline_inputs
                )
                errors.extend(param_errors)
                warnings.extend(param_warnings)
                dependencies.update(param_deps)
                
            elif isinstance(param_value, dict):
                # Recursively analyze nested parameters
                nested_errors, nested_warnings, nested_deps = self._analyze_parameters_data_flow(
                    param_value, task_id, task_schemas, pipeline_inputs, current_path
                )
                errors.extend(nested_errors)
                warnings.extend(nested_warnings)
                dependencies.update(nested_deps)
                
            elif isinstance(param_value, list):
                # Analyze list items
                for i, item in enumerate(param_value):
                    if isinstance(item, (str, dict)):
                        item_path = f"{current_path}[{i}]"
                        if isinstance(item, str):
                            item_errors, item_warnings, item_deps = self._analyze_template_string(
                                item, task_id, item_path, task_schemas, pipeline_inputs
                            )
                        else:
                            item_errors, item_warnings, item_deps = self._analyze_parameters_data_flow(
                                item, task_id, task_schemas, pipeline_inputs, item_path
                            )
                        errors.extend(item_errors)
                        warnings.extend(item_warnings)
                        dependencies.update(item_deps)
        
        return errors, warnings, dependencies
    
    def _analyze_template_string(self, 
                               template_str: str,
                               task_id: str,
                               parameter_name: str,
                               task_schemas: Dict[str, TaskOutputSchema],
                               pipeline_inputs: Dict[str, Any]) -> Tuple[List[DataFlowError], List[DataFlowError], Set[str]]:
        """Analyze a template string for data flow validation."""
        errors: List[DataFlowError] = []
        warnings: List[DataFlowError] = []
        dependencies: Set[str] = set()
        
        # Extract template variables
        template_vars = self._extract_template_variables(template_str)
        
        for var_ref in template_vars:
            # Parse variable reference (e.g., "task_name.output", "inputs.param")
            validation_result = self._validate_variable_reference(
                var_ref, task_id, parameter_name, task_schemas, pipeline_inputs
            )
            
            if validation_result["valid"]:
                if validation_result.get("dependency"):
                    dependencies.add(validation_result["dependency"])
            else:
                error = DataFlowError(
                    task_id=task_id,
                    parameter_name=parameter_name,
                    error_type=validation_result["error_type"],
                    message=validation_result["message"],
                    variable_reference=var_ref,
                    source_task=validation_result.get("source_task"),
                    suggestions=validation_result.get("suggestions", []),
                    severity=validation_result.get("severity", "error")
                )
                
                if error.severity == "warning":
                    warnings.append(error)
                else:
                    errors.append(error)
        
        return errors, warnings, dependencies
    
    def _extract_template_variables(self, template_str: str) -> List[str]:
        """Extract all template variable references from a string."""
        variables = []
        
        # Find all {{ variable }} patterns
        matches = self.template_var_pattern.findall(template_str)
        
        for match in matches:
            # Clean up the variable reference
            var = match.strip()
            
            # Handle filters and complex expressions
            # Take the base variable before any filters or operations
            var = var.split('|')[0].strip()  # Remove filters
            var = var.split(' ')[0].strip()   # Remove operations
            
            # Skip literals and complex expressions
            if (not var.startswith('"') and 
                not var.startswith("'") and 
                not var.isdigit() and
                not var.startswith('[') and
                not var.startswith('{')):
                variables.append(var)
        
        # Also check for Jinja2 control structures ({% %})
        control_pattern = re.compile(r'\{%\s*(?:for|if)\s+([^%]+)\s*%\}')
        control_matches = control_pattern.findall(template_str)
        
        for match in control_matches:
            # Extract variable references from control structures
            # This is more complex parsing, simplified for now
            parts = match.split()
            for part in parts:
                if '.' in part and not part.startswith('"') and not part.startswith("'"):
                    variables.append(part.strip())
        
        return variables
    
    def _validate_variable_reference(self, 
                                   var_ref: str,
                                   task_id: str,
                                   parameter_name: str,
                                   task_schemas: Dict[str, TaskOutputSchema],
                                   pipeline_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single variable reference.
        
        Returns dict with validation result and metadata.
        """
        # Handle special variables
        if var_ref in ['item', 'index', 'loop', 'iteration', 'is_first', 'is_last']:
            return {"valid": True, "type": "loop_variable"}
        
        if var_ref.startswith('$'):
            # Loop variables like $item, $index
            return {"valid": True, "type": "loop_variable"}
        
        # Parse dotted variable reference
        parts = var_ref.split('.')
        if len(parts) < 1:
            return {
                "valid": False,
                "error_type": "invalid_reference",
                "message": f"Invalid variable reference: '{var_ref}'"
            }
        
        base_var = parts[0]
        
        # Check for pipeline inputs
        if base_var == "inputs":
            if len(parts) > 1:
                input_name = parts[1]
                if input_name in pipeline_inputs:
                    return {"valid": True, "type": "pipeline_input"}
                else:
                    available_inputs = list(pipeline_inputs.keys())
                    return {
                        "valid": False,
                        "error_type": "undefined_input",
                        "message": f"Undefined pipeline input: '{input_name}'",
                        "suggestions": self._suggest_similar_names(input_name, available_inputs)
                    }
            else:
                return {"valid": True, "type": "pipeline_inputs"}
        
        # Check for task output references
        if base_var in task_schemas:
            source_task = base_var
            schema = task_schemas[source_task]
            
            # Check if this creates a dependency
            if source_task != task_id:  # Can't depend on self
                # Validate the specific output reference
                if len(parts) > 1:
                    output_field = parts[1]
                    available_vars = schema.get_available_variables()
                    full_ref = f"{source_task}.{output_field}"
                    
                    if full_ref in available_vars:
                        return {
                            "valid": True,
                            "type": "task_output",
                            "dependency": source_task,
                            "source_task": source_task
                        }
                    else:
                        # Extract just the output names for suggestions
                        available_outputs = [var.split('.', 1)[1] for var in available_vars if '.' in var]
                        return {
                            "valid": False,
                            "error_type": "undefined_output",
                            "message": f"Task '{source_task}' does not produce output '{output_field}'",
                            "source_task": source_task,
                            "suggestions": self._suggest_similar_names(output_field, available_outputs)
                        }
                else:
                    return {
                        "valid": True,
                        "type": "task_reference",
                        "dependency": source_task,
                        "source_task": source_task
                    }
            else:
                return {
                    "valid": False,
                    "error_type": "self_reference",
                    "message": f"Task cannot reference its own outputs"
                }
        
        # Check if it might be a missing task
        available_tasks = list(task_schemas.keys())
        if base_var not in available_tasks:
            suggestions = self._suggest_similar_names(base_var, available_tasks)
            
            # In development mode, this might be a warning instead of an error
            severity = "warning" if self.development_mode else "error"
            
            return {
                "valid": False,
                "error_type": "undefined_task",
                "message": f"Undefined task reference: '{base_var}'",
                "suggestions": suggestions,
                "severity": severity
            }
        
        return {"valid": True, "type": "unknown"}
    
    def _suggest_similar_names(self, target: str, available: List[str]) -> List[str]:
        """Suggest similar names for typos."""
        suggestions = []
        target_lower = target.lower()
        
        # Exact match (case insensitive)
        for name in available:
            if name.lower() == target_lower:
                suggestions.append(f"Did you mean '{name}' (case mismatch)?")
        
        # Partial matches
        for name in available:
            if target_lower in name.lower() or name.lower() in target_lower:
                suggestions.append(f"Did you mean '{name}'?")
        
        # Similar length and characters
        for name in available:
            if abs(len(name) - len(target)) <= 2:
                # Simple character overlap check
                overlap = sum(1 for c in target_lower if c in name.lower())
                if overlap >= min(len(target), len(name)) * 0.6:
                    suggestions.append(f"Did you mean '{name}'?")
        
        return suggestions[:3]  # Limit to top 3 suggestions
    
    def _validate_data_flow_graph(self, 
                                data_flow_graph: Dict[str, Set[str]], 
                                steps: List[Dict[str, Any]]) -> Tuple[List[DataFlowError], List[DataFlowError]]:
        """Validate the overall data flow graph for issues."""
        errors: List[DataFlowError] = []
        warnings: List[DataFlowError] = []
        
        # Check for circular dependencies
        def has_cycle(graph: Dict[str, Set[str]]) -> Optional[List[str]]:
            """Detect cycles in dependency graph using DFS."""
            WHITE, GRAY, BLACK = 0, 1, 2
            colors = {node: WHITE for node in graph}
            
            def dfs(node: str, path: List[str]) -> Optional[List[str]]:
                if colors[node] == GRAY:
                    # Found a cycle
                    cycle_start = path.index(node)
                    return path[cycle_start:] + [node]
                
                if colors[node] == BLACK:
                    return None
                
                colors[node] = GRAY
                path.append(node)
                
                for neighbor in graph.get(node, set()):
                    if neighbor in graph:  # Only check if neighbor exists
                        cycle = dfs(neighbor, path.copy())
                        if cycle:
                            return cycle
                
                colors[node] = BLACK
                return None
            
            for node in graph:
                if colors[node] == WHITE:
                    cycle = dfs(node, [])
                    if cycle:
                        return cycle
            
            return None
        
        cycle = has_cycle(data_flow_graph)
        if cycle:
            cycle_str = " -> ".join(cycle)
            errors.append(DataFlowError(
                task_id=cycle[0],
                parameter_name=None,
                error_type="circular_dependency",
                message=f"Circular dependency detected: {cycle_str}",
                suggestions=["Remove or restructure dependencies to break the cycle"]
            ))
        
        # Check for missing dependencies
        all_task_ids = {step.get("id") for step in steps if step.get("id")}
        
        for task_id, deps in data_flow_graph.items():
            for dep in deps:
                if dep not in all_task_ids:
                    errors.append(DataFlowError(
                        task_id=task_id,
                        parameter_name=None,
                        error_type="missing_dependency",
                        message=f"Task '{task_id}' depends on non-existent task '{dep}'",
                        suggestions=[f"Remove dependency on '{dep}' or add the missing task to the pipeline"]
                    ))
        
        return errors, warnings
    
    def _are_errors_bypassable(self, errors: List[DataFlowError]) -> bool:
        """Check if errors can be bypassed in development mode."""
        if not self.development_mode:
            return False
        
        # In development mode, allow bypassing certain types of errors
        bypassable_types = {
            "undefined_task",
            "undefined_output", 
            "undefined_input"
        }
        
        for error in errors:
            if error.error_type not in bypassable_types:
                return False
        
        return True
    
    def validate_single_reference(self, 
                                variable_ref: str,
                                available_outputs: Dict[str, Set[str]],
                                pipeline_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single variable reference against available outputs.
        
        Args:
            variable_ref: Variable reference to validate (e.g., "task1.result")
            available_outputs: Dict mapping task_id to set of available output names
            pipeline_inputs: Available pipeline inputs
            
        Returns:
            Validation result dictionary
        """
        parts = variable_ref.split('.')
        if len(parts) < 1:
            return {
                "valid": False,
                "error": "Invalid variable reference format"
            }
        
        base_var = parts[0]
        
        # Check pipeline inputs
        if base_var == "inputs":
            if len(parts) > 1:
                input_name = parts[1]
                return {
                    "valid": input_name in pipeline_inputs,
                    "type": "pipeline_input",
                    "error": f"Undefined input: {input_name}" if input_name not in pipeline_inputs else None
                }
            return {"valid": True, "type": "pipeline_inputs"}
        
        # Check task outputs
        if base_var in available_outputs:
            if len(parts) > 1:
                output_name = parts[1]
                return {
                    "valid": output_name in available_outputs[base_var],
                    "type": "task_output", 
                    "dependency": base_var,
                    "error": f"Task '{base_var}' does not have output '{output_name}'" if output_name not in available_outputs[base_var] else None
                }
            return {"valid": True, "type": "task_reference", "dependency": base_var}
        
        return {
            "valid": False,
            "type": "unknown",
            "error": f"Undefined task or variable: {base_var}"
        }