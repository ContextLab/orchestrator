"""
YAML schema definitions and validation for advanced error handling.
Provides validation for the new ErrorHandler YAML syntax and backward compatibility.
"""

from typing import Any, Dict, List, Optional, Union

from ..core.exceptions import YAMLCompilerError


class ErrorHandlerSchemaValidator:
    """Validates error handler configurations in YAML pipelines."""
    
    def __init__(self):
        self.supported_error_types = [
            "*",  # Wildcard for all errors
            "Exception",
            "ValueError", 
            "TypeError",
            "FileNotFoundError",
            "PermissionError",
            "TimeoutError",
            "ConnectionError",
            "HTTPError",
            "APIError",
            "ModelError",
            "ToolError",
            "AuthenticationError",
            "AuthorizationError",
        ]
    
    def validate_error_handler_config(self, handler_config: Any, context_path: str = "") -> List[str]:
        """
        Validate a single error handler configuration.
        
        Args:
            handler_config: Error handler configuration (dict, string, or ErrorHandler)
            context_path: Path context for error reporting
            
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        if isinstance(handler_config, str):
            # Simple string format - just validate it's not empty
            if not handler_config.strip():
                issues.append(f"{context_path}: Empty error handler action")
            return issues
        
        if not isinstance(handler_config, dict):
            issues.append(f"{context_path}: Error handler must be string or dict, got {type(handler_config)}")
            return issues
            
        # Validate required fields
        if not handler_config.get("handler_task_id") and not handler_config.get("handler_action") and not handler_config.get("fallback_value"):
            issues.append(f"{context_path}: Error handler must specify at least one of: handler_task_id, handler_action, or fallback_value")
        
        # Validate mutual exclusivity
        if handler_config.get("handler_task_id") and handler_config.get("handler_action"):
            issues.append(f"{context_path}: Error handler cannot specify both handler_task_id and handler_action")
        
        # Validate error types
        error_types = handler_config.get("error_types", [])
        if error_types:
            if not isinstance(error_types, list):
                issues.append(f"{context_path}: error_types must be a list")
            else:
                for error_type in error_types:
                    if not isinstance(error_type, str):
                        issues.append(f"{context_path}: error_types must contain strings, got {type(error_type)}")
        
        # Validate error patterns
        error_patterns = handler_config.get("error_patterns", [])
        if error_patterns:
            if not isinstance(error_patterns, list):
                issues.append(f"{context_path}: error_patterns must be a list")
            else:
                for pattern in error_patterns:
                    if not isinstance(pattern, str):
                        issues.append(f"{context_path}: error_patterns must contain strings, got {type(pattern)}")
                    else:
                        # Try to validate regex patterns
                        try:
                            import re
                            re.compile(pattern)
                        except re.error as e:
                            issues.append(f"{context_path}: Invalid regex pattern '{pattern}': {e}")
        
        # Validate error codes
        error_codes = handler_config.get("error_codes", [])
        if error_codes:
            if not isinstance(error_codes, list):
                issues.append(f"{context_path}: error_codes must be a list")
            else:
                for code in error_codes:
                    if not isinstance(code, (int, str)):
                        issues.append(f"{context_path}: error_codes must contain integers or strings, got {type(code)}")
        
        # Validate numeric fields
        numeric_fields = {
            "max_handler_retries": (int, 0, 100),
            "priority": (int, 1, 10000),
            "timeout": (float, 0.1, 3600.0)
        }
        
        for field, (expected_type, min_val, max_val) in numeric_fields.items():
            value = handler_config.get(field)
            if value is not None:
                if not isinstance(value, (int, float)):
                    issues.append(f"{context_path}: {field} must be a number, got {type(value)}")
                elif value < min_val or value > max_val:
                    issues.append(f"{context_path}: {field} must be between {min_val} and {max_val}, got {value}")
        
        # Validate boolean fields
        boolean_fields = [
            "retry_with_handler", "propagate_error", "continue_on_handler_failure",
            "capture_error_context", "enabled"
        ]
        
        for field in boolean_fields:
            value = handler_config.get(field)
            if value is not None and not isinstance(value, bool):
                issues.append(f"{context_path}: {field} must be boolean, got {type(value)}")
        
        # Validate log level
        log_level = handler_config.get("log_level")
        if log_level is not None:
            valid_levels = ["debug", "info", "warning", "error", "critical"]
            if log_level.lower() not in valid_levels:
                issues.append(f"{context_path}: log_level must be one of {valid_levels}, got {log_level}")
        
        return issues
    
    def validate_task_error_handling(self, task_config: Dict[str, Any], task_id: str) -> List[str]:
        """
        Validate error handling configuration for a task.
        
        Args:
            task_config: Task configuration dictionary
            task_id: Task ID for error reporting
            
        Returns:
            List of validation issues
        """
        issues = []
        context_path = f"task '{task_id}'"
        
        # Check legacy on_error field
        on_error = task_config.get("on_error")
        if on_error is not None:
            if isinstance(on_error, list):
                # New list format
                for i, handler_config in enumerate(on_error):
                    handler_issues = self.validate_error_handler_config(
                        handler_config, f"{context_path}.on_error[{i}]"
                    )
                    issues.extend(handler_issues)
            else:
                # Legacy format or single handler
                handler_issues = self.validate_error_handler_config(
                    on_error, f"{context_path}.on_error"
                )
                issues.extend(handler_issues)
        
        # Check separate error_handlers field if present
        error_handlers = task_config.get("error_handlers")
        if error_handlers is not None:
            if not isinstance(error_handlers, list):
                issues.append(f"{context_path}: error_handlers must be a list")
            else:
                for i, handler_config in enumerate(error_handlers):
                    handler_issues = self.validate_error_handler_config(
                        handler_config, f"{context_path}.error_handlers[{i}]"
                    )
                    issues.extend(handler_issues)
        
        # Validate that we don't have conflicting error handling formats
        if on_error is not None and error_handlers is not None:
            if isinstance(on_error, list):
                issues.append(f"{context_path}: Cannot specify both on_error (list format) and error_handlers")
            # Legacy on_error with separate error_handlers is allowed for migration
        
        return issues
    
    def validate_pipeline_error_handling(self, pipeline_config: Dict[str, Any]) -> List[str]:
        """
        Validate error handling configuration for an entire pipeline.
        
        Args:
            pipeline_config: Pipeline configuration dictionary
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Validate each step's error handling
        steps = pipeline_config.get("steps", [])
        for step in steps:
            if isinstance(step, dict):
                step_id = step.get("id", "unnamed_step")
                step_issues = self.validate_task_error_handling(step, step_id)
                issues.extend(step_issues)
        
        # Check for global error handling configuration if supported
        global_error_handlers = pipeline_config.get("global_error_handlers")
        if global_error_handlers is not None:
            if not isinstance(global_error_handlers, list):
                issues.append("global_error_handlers must be a list")
            else:
                for i, handler_config in enumerate(global_error_handlers):
                    handler_issues = self.validate_error_handler_config(
                        handler_config, f"global_error_handlers[{i}]"
                    )
                    issues.extend(handler_issues)
        
        return issues
    
    def get_schema_examples(self) -> Dict[str, Any]:
        """
        Get example YAML configurations for error handling.
        
        Returns:
            Dictionary with example configurations
        """
        return {
            "simple_string": "Log error and continue",
            
            "simple_dict": {
                "handler_action": "Log the error: {{error_message}}",
                "error_types": ["ValueError", "TypeError"],
                "continue_on_handler_failure": True
            },
            
            "advanced_handler": {
                "handler_task_id": "error_recovery_task",
                "error_types": ["ConnectionError", "TimeoutError"],
                "error_patterns": ["connection.*refused", "timeout.*occurred"],
                "error_codes": [404, 500, 503],
                "retry_with_handler": True,
                "max_handler_retries": 3,
                "timeout": 30.0,
                "priority": 10,
                "continue_on_handler_failure": False,
                "fallback_value": "Default response when all else fails",
                "capture_error_context": True,
                "log_level": "warning"
            },
            
            "multiple_handlers": [
                {
                    "handler_action": "Retry with exponential backoff",
                    "error_types": ["ConnectionError"],
                    "priority": 1,
                    "retry_with_handler": True,
                    "max_handler_retries": 3
                },
                {
                    "handler_task_id": "send_alert",
                    "error_types": ["*"],
                    "priority": 100,
                    "continue_on_handler_failure": True,
                    "fallback_value": "Error handled by alert system"
                }
            ],
            
            "task_with_error_handling": {
                "id": "process_data",
                "action": "Process the input data",
                "parameters": {
                    "data": "{{input_data}}"
                },
                "on_error": [
                    {
                        "handler_action": "Log processing error and use backup data",
                        "error_types": ["ValueError", "KeyError"],
                        "fallback_value": "backup_data.json",
                        "priority": 1
                    },
                    {
                        "handler_task_id": "notify_admin",
                        "error_types": ["*"],
                        "priority": 99
                    }
                ]
            }
        }
    
    def convert_legacy_to_new_format(self, legacy_config: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert legacy error handling format to new ErrorHandler format.
        
        Args:
            legacy_config: Legacy on_error configuration
            
        Returns:
            New ErrorHandler format configuration
        """
        if isinstance(legacy_config, str):
            # Simple string action
            return {
                "handler_action": legacy_config,
                "error_types": ["*"],
                "retry_with_handler": False,
                "continue_on_handler_failure": True
            }
        
        if isinstance(legacy_config, dict):
            # Legacy ErrorHandling object format
            new_config = {
                "handler_action": legacy_config.get("action", "Handle error"),
                "error_types": ["*"],
                "continue_on_handler_failure": legacy_config.get("continue_on_error", False),
                "retry_with_handler": legacy_config.get("retry_count", 0) > 0,
                "max_handler_retries": legacy_config.get("retry_count", 0),
                "fallback_value": legacy_config.get("fallback_value")
            }
            
            # Remove None values
            return {k: v for k, v in new_config.items() if v is not None}
        
        raise ValueError(f"Cannot convert legacy format: {type(legacy_config)}")


def validate_error_handler_yaml(yaml_content: str) -> List[str]:
    """
    Validate error handling in YAML content.
    
    Args:
        yaml_content: YAML pipeline content
        
    Returns:
        List of validation issues
    """
    try:
        import yaml
        pipeline_config = yaml.safe_load(yaml_content)
        
        validator = ErrorHandlerSchemaValidator()
        return validator.validate_pipeline_error_handling(pipeline_config)
        
    except yaml.YAMLError as e:
        return [f"Invalid YAML syntax: {e}"]
    except Exception as e:
        return [f"Validation error: {e}"]


def get_error_handler_schema() -> Dict[str, Any]:
    """
    Get JSON schema for error handler validation.
    
    Returns:
        JSON schema dictionary
    """
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Error Handler Configuration",
        "type": "object",
        "properties": {
            "handler_task_id": {
                "type": "string",
                "description": "ID of task to execute on error"
            },
            "handler_action": {
                "type": "string", 
                "description": "Action to execute directly on error"
            },
            "error_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of error types to catch (e.g., ['ValueError', 'ConnectionError'])"
            },
            "error_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of regex patterns to match error messages"
            },
            "error_codes": {
                "type": "array",
                "items": {"type": ["integer", "string"]},
                "description": "List of error codes to match (HTTP codes, exit codes, etc.)"
            },
            "retry_with_handler": {
                "type": "boolean",
                "default": True,
                "description": "Whether to retry original task after successful handler"
            },
            "max_handler_retries": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "default": 0,
                "description": "Maximum retries for the handler itself"
            },
            "propagate_error": {
                "type": "boolean",
                "default": False,
                "description": "Whether to propagate error if handler fails"
            },
            "continue_on_handler_failure": {
                "type": "boolean",
                "default": False,
                "description": "Whether to continue pipeline if handler fails"
            },
            "fallback_value": {
                "description": "Value to return if handler fails"
            },
            "fallback_result": {
                "type": "object",
                "description": "Complete result structure to return if handler fails"
            },
            "capture_error_context": {
                "type": "boolean",
                "default": True,
                "description": "Whether to capture full error context"
            },
            "log_level": {
                "type": "string",
                "enum": ["debug", "info", "warning", "error", "critical"],
                "default": "error",
                "description": "Logging level for handler execution"
            },
            "timeout": {
                "type": "number",
                "minimum": 0.1,
                "maximum": 3600,
                "description": "Handler execution timeout in seconds"
            },
            "priority": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10000,
                "default": 100,
                "description": "Handler priority (lower number = higher priority)"
            },
            "enabled": {
                "type": "boolean",
                "default": True,
                "description": "Whether this handler is enabled"
            }
        },
        "anyOf": [
            {"required": ["handler_task_id"]},
            {"required": ["handler_action"]},
            {"required": ["fallback_value"]}
        ]
    }