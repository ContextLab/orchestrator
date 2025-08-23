"""Tool configuration validation for pipeline compilation."""

import logging
from typing import Any, Dict, List, Optional, Set, Type, Union
from dataclasses import dataclass

from ..tools.base import Tool, ToolRegistry, default_registry
from .template_validator import TemplateValidationError


logger = logging.getLogger(__name__)


@dataclass
class ToolValidationError:
    """Represents a tool validation error."""
    
    task_id: str
    tool_name: str
    parameter_name: Optional[str]
    error_type: str
    message: str
    severity: str = "error"  # error, warning
    
    def __str__(self) -> str:
        if self.parameter_name:
            return f"Task '{self.task_id}': Tool '{self.tool_name}' parameter '{self.parameter_name}' {self.error_type}: {self.message}"
        else:
            return f"Task '{self.task_id}': Tool '{self.tool_name}' {self.error_type}: {self.message}"


@dataclass
class ToolValidationResult:
    """Result of tool validation."""
    
    valid: bool
    errors: List[ToolValidationError]
    warnings: List[ToolValidationError]
    validated_tasks: int
    tool_availability: Dict[str, bool]
    
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
        return (f"Validated {self.validated_tasks} tasks: "
                f"{len(self.errors)} errors, {len(self.warnings)} warnings")


class ToolValidator:
    """Validates tool configurations in pipeline definitions."""
    
    def __init__(self, 
                 tool_registry: Optional[ToolRegistry] = None,
                 development_mode: bool = False,
                 allow_unknown_tools: bool = False):
        """
        Initialize tool validator.
        
        Args:
            tool_registry: Tool registry to use for validation (defaults to global registry)
            development_mode: If True, allows some validation bypasses for development
            allow_unknown_tools: If True, allows unknown tools (useful for external tools)
        """
        self.tool_registry = tool_registry or default_registry
        self.development_mode = development_mode
        # In development mode, automatically allow unknown tools
        self.allow_unknown_tools = allow_unknown_tools or development_mode
        
        # Load all available tools to ensure registry is populated
        self._ensure_tools_loaded()
        
        # Cache tool schemas for performance
        self._tool_schemas: Dict[str, Dict[str, Any]] = {}
        self._refresh_tool_schemas()
    
    def _ensure_tools_loaded(self) -> None:
        """Ensure all default tools are loaded in the registry."""
        try:
            from ..tools.base import register_default_tools
            registered_count = register_default_tools()
            logger.debug(f"Ensured {registered_count} tools are registered")
        except Exception as e:
            logger.warning(f"Failed to ensure tools are loaded: {e}")
    
    def _refresh_tool_schemas(self) -> None:
        """Refresh cached tool schemas."""
        self._tool_schemas.clear()
        
        for tool_name in self.tool_registry.list_tools():
            tool = self.tool_registry.get_tool(tool_name)
            if tool:
                self._tool_schemas[tool_name] = tool.get_schema()
                logger.debug(f"Cached schema for tool '{tool_name}'")
    
    def validate_pipeline_tools(self, pipeline_def: Dict[str, Any]) -> ToolValidationResult:
        """
        Validate all tools used in a pipeline definition.
        
        Args:
            pipeline_def: Complete pipeline definition
            
        Returns:
            ToolValidationResult with validation details
        """
        errors: List[ToolValidationError] = []
        warnings: List[ToolValidationError] = []
        validated_tasks = 0
        tool_availability: Dict[str, bool] = {}
        
        steps = pipeline_def.get("steps", [])
        
        for step in steps:
            if not isinstance(step, dict):
                continue
                
            task_id = step.get("id", "unknown")
            validated_tasks += 1
            
            # Get tool/action name
            tool_name = self._extract_tool_name(step)
            if not tool_name:
                errors.append(ToolValidationError(
                    task_id=task_id,
                    tool_name="unknown",
                    parameter_name=None,
                    error_type="missing_tool",
                    message="No tool/action specified"
                ))
                continue
            
            # Check tool availability
            tool_available = self._check_tool_availability(tool_name)
            tool_availability[tool_name] = tool_available
            
            if not tool_available:
                if self.allow_unknown_tools:
                    warnings.append(ToolValidationError(
                        task_id=task_id,
                        tool_name=tool_name,
                        parameter_name=None,
                        error_type="unknown_tool",
                        message=f"Tool '{tool_name}' not found in registry (allowed in current mode)",
                        severity="warning"
                    ))
                else:
                    errors.append(ToolValidationError(
                        task_id=task_id,
                        tool_name=tool_name,
                        parameter_name=None,
                        error_type="unknown_tool",
                        message=f"Tool '{tool_name}' not found in registry"
                    ))
                continue
            
            # Validate tool parameters
            parameters = step.get("parameters", {})
            tool_errors, tool_warnings = self._validate_tool_parameters(
                task_id, tool_name, parameters
            )
            errors.extend(tool_errors)
            warnings.extend(tool_warnings)
        
        # Determine overall validity
        valid = len(errors) == 0 or (self.development_mode and self._are_errors_bypassable(errors))
        
        return ToolValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            validated_tasks=validated_tasks,
            tool_availability=tool_availability
        )
    
    def _extract_tool_name(self, step: Dict[str, Any]) -> Optional[str]:
        """Extract tool name from step definition."""
        # Check 'action' field first (preferred)
        if "action" in step:
            return step["action"]
        
        # Check 'tool' field (legacy)
        if "tool" in step:
            return step["tool"]
        
        # Check for special control flow steps
        if "create_parallel_queue" in step:
            return "create_parallel_queue"
        
        if "action_loop" in step:
            return "action_loop"
        
        # Check for control flow steps
        if any(key in step for key in ["for_each", "while", "if", "condition"]):
            return "control_flow"
        
        return None
    
    def _check_tool_availability(self, tool_name: str) -> bool:
        """Check if a tool is available in the registry."""
        # Handle special control flow "tools"
        if tool_name in ["control_flow", "create_parallel_queue", "action_loop"]:
            return True
            
        return tool_name in self._tool_schemas
    
    def _validate_tool_parameters(self, task_id: str, tool_name: str, parameters: Dict[str, Any]) -> tuple[List[ToolValidationError], List[ToolValidationError]]:
        """Validate parameters for a specific tool."""
        errors: List[ToolValidationError] = []
        warnings: List[ToolValidationError] = []
        
        # Skip validation for special control flow tools
        if tool_name in ["control_flow", "create_parallel_queue", "action_loop"]:
            return errors, warnings
        
        tool_schema = self._tool_schemas.get(tool_name)
        if not tool_schema:
            # This should have been caught earlier
            return errors, warnings
        
        input_schema = tool_schema.get("inputSchema", {})
        properties = input_schema.get("properties", {})
        required_params = set(input_schema.get("required", []))
        
        # Check for missing required parameters
        provided_params = set(parameters.keys())
        missing_required = required_params - provided_params
        
        for param_name in missing_required:
            # Skip if this might be a template that will be resolved at runtime
            if self._might_be_runtime_resolved(task_id, tool_name, param_name, parameters):
                warnings.append(ToolValidationError(
                    task_id=task_id,
                    tool_name=tool_name,
                    parameter_name=param_name,
                    error_type="missing_required_param",
                    message=f"Required parameter '{param_name}' not provided (may be resolved at runtime)",
                    severity="warning"
                ))
            else:
                errors.append(ToolValidationError(
                    task_id=task_id,
                    tool_name=tool_name,
                    parameter_name=param_name,
                    error_type="missing_required_param",
                    message=f"Required parameter '{param_name}' not provided"
                ))
        
        # Check parameter types and formats
        for param_name, param_value in parameters.items():
            if param_name in properties:
                param_errors, param_warnings = self._validate_parameter_value(
                    task_id, tool_name, param_name, param_value, properties[param_name]
                )
                errors.extend(param_errors)
                warnings.extend(param_warnings)
            else:
                # Unknown parameter
                warnings.append(ToolValidationError(
                    task_id=task_id,
                    tool_name=tool_name,
                    parameter_name=param_name,
                    error_type="unknown_parameter",
                    message=f"Parameter '{param_name}' not recognized by tool schema",
                    severity="warning"
                ))
        
        return errors, warnings
    
    def _validate_parameter_value(self, task_id: str, tool_name: str, param_name: str, 
                                param_value: Any, param_schema: Dict[str, Any]) -> tuple[List[ToolValidationError], List[ToolValidationError]]:
        """Validate a specific parameter value against its schema."""
        errors: List[ToolValidationError] = []
        warnings: List[ToolValidationError] = []
        
        expected_type = param_schema.get("type", "any")
        
        # Skip validation for template strings (they'll be resolved at runtime)
        if self._is_template_string(param_value):
            return errors, warnings
        
        # Type validation
        if not self._check_parameter_type(param_value, expected_type):
            # In development mode, this might be a warning instead of error
            error_msg = f"Parameter value type mismatch. Expected {expected_type}, got {type(param_value).__name__}"
            
            if self.development_mode and self._can_coerce_type(param_value, expected_type):
                warnings.append(ToolValidationError(
                    task_id=task_id,
                    tool_name=tool_name,
                    parameter_name=param_name,
                    error_type="type_mismatch",
                    message=f"{error_msg} (can be coerced)",
                    severity="warning"
                ))
            else:
                errors.append(ToolValidationError(
                    task_id=task_id,
                    tool_name=tool_name,
                    parameter_name=param_name,
                    error_type="type_mismatch",
                    message=error_msg
                ))
        
        # Format validation if specified
        format_spec = param_schema.get("format")
        if format_spec and isinstance(param_value, str):
            if not self._validate_parameter_format(param_value, format_spec):
                warnings.append(ToolValidationError(
                    task_id=task_id,
                    tool_name=tool_name,
                    parameter_name=param_name,
                    error_type="format_mismatch",
                    message=f"Parameter value does not match expected format '{format_spec}'",
                    severity="warning"
                ))
        
        return errors, warnings
    
    def _might_be_runtime_resolved(self, task_id: str, tool_name: str, param_name: str, 
                                 parameters: Dict[str, Any]) -> bool:
        """Check if a parameter might be resolved at runtime through templates or dependencies."""
        # Check if other parameters contain references that might affect this one
        for param_value in parameters.values():
            if self._is_template_string(param_value):
                # If any parameter is templated, the missing ones might be resolved dynamically
                return True
        
        # Tool-specific logic
        if tool_name == "filesystem" and param_name in ["content", "path"]:
            # Filesystem operations often have runtime-resolved paths and content
            return True
        
        if tool_name in ["llm_tools", "task-delegation"] and param_name in ["prompt", "task"]:
            # LLM tools often have runtime-resolved prompts
            return True
        
        return False
    
    def _is_template_string(self, value: Any) -> bool:
        """Check if a value contains template syntax."""
        if not isinstance(value, str):
            return False
        
        # Check for Jinja2 template syntax
        return "{{" in value or "{%" in value or value.startswith("$")
    
    def _check_parameter_type(self, value: Any, expected_type: str) -> bool:
        """Check if parameter value matches expected type."""
        if expected_type == "any":
            return True
        
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type is None:
            # Unknown type, assume valid
            return True
        
        return isinstance(value, expected_python_type)
    
    def _can_coerce_type(self, value: Any, expected_type: str) -> bool:
        """Check if value can be coerced to expected type."""
        try:
            if expected_type == "string":
                str(value)
                return True
            elif expected_type == "integer" and isinstance(value, (str, float)):
                int(value)
                return True
            elif expected_type == "number" and isinstance(value, str):
                float(value)
                return True
            elif expected_type == "boolean" and isinstance(value, str):
                return value.lower() in ["true", "false", "yes", "no", "1", "0"]
        except (ValueError, TypeError):
            return False
        
        return False
    
    def _validate_parameter_format(self, value: str, format_spec: str) -> bool:
        """Validate parameter format."""
        import re
        
        format_patterns = {
            "email": r"^[^@]+@[^@]+\.[^@]+$",
            "uri": r"^https?://",
            "url": r"^https?://",
            "file-path": r"^[^\0]+$",  # Just check for non-null characters
            "model-id": r"^[\w\-]+\/[\w\-\.:]+$",
            "tool-name": r"^[a-z][a-z0-9\-_]*$",
        }
        
        pattern = format_patterns.get(format_spec)
        if pattern:
            return bool(re.match(pattern, value))
        
        # Unknown format, assume valid
        return True
    
    def _are_errors_bypassable(self, errors: List[ToolValidationError]) -> bool:
        """Check if errors can be bypassed in development mode."""
        if not self.development_mode:
            return False
        
        # In development mode, allow bypassing certain types of errors
        bypassable_types = {
            "unknown_tool",
            "unknown_parameter", 
            "type_mismatch",
            "format_mismatch"
        }
        
        for error in errors:
            if error.error_type not in bypassable_types:
                return False
        
        return True
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return self.tool_registry.list_tools()
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get schema for a specific tool."""
        return self._tool_schemas.get(tool_name)
    
    def validate_single_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ToolValidationResult:
        """
        Validate a single tool configuration.
        
        Args:
            tool_name: Name of the tool to validate
            parameters: Parameters to validate
            
        Returns:
            ToolValidationResult for the single tool
        """
        task_id = "single_validation"
        errors, warnings = self._validate_tool_parameters(task_id, tool_name, parameters)
        
        tool_availability = {tool_name: self._check_tool_availability(tool_name)}
        
        return ToolValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            validated_tasks=1,
            tool_availability=tool_availability
        )