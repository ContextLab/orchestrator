"""Comprehensive validation tool with JSON Schema and structured output support."""

import copy
import re
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union

import jsonschema
from jsonschema import Draft7Validator, ValidationError
from pydantic import BaseModel, Field, create_model

from ..core.model import Model
from .base import Tool


class SchemaState(Enum):
    """State of schema resolution."""
    FIXED = "fixed"  # Fully determined at compile time
    PARTIAL = "partial"  # Some parts known, others ambiguous
    AMBIGUOUS = "ambiguous"  # Cannot be determined until runtime


class ValidationMode(Enum):
    """Validation strictness modes."""
    STRICT = "strict"  # Fail on any validation error
    LENIENT = "lenient"  # Coerce types where safe, warn on minor issues
    REPORT_ONLY = "report_only"  # Never fail, just report issues


class ValidationResult(BaseModel):
    """Result of validation operation."""
    valid: bool = Field(description="Whether validation passed")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Validation errors")
    warnings: List[Dict[str, Any]] = Field(default_factory=list, description="Validation warnings")
    data: Optional[Any] = Field(default=None, description="Validated/coerced data")
    schema_used: Optional[Dict[str, Any]] = Field(default=None, description="Schema used for validation")


class FormatValidator:
    """Registry for custom format validators."""
    
    def __init__(self):
        self._validators: Dict[str, Callable[[str], bool]] = {}
        self._patterns: Dict[str, str] = {}
        self._register_builtin_formats()
    
    def _register_builtin_formats(self):
        """Register orchestrator-specific format validators."""
        # Model ID format
        self.register_pattern(
            "model-id",
            r"^[\w\-]+\/[\w\-\.:]+$",
            "Model identifier (provider/model-name)"
        )
        
        # Tool name format
        self.register_pattern(
            "tool-name",
            r"^[a-z][a-z0-9\-_]*$",
            "Tool name (lowercase, alphanumeric, hyphens, underscores)"
        )
        
        # File path format
        self.register_function(
            "file-path",
            lambda x: len(x) > 0 and not x.startswith(" "),
            "Valid file system path"
        )
        
        # YAML/JSONPath format
        self.register_pattern(
            "yaml-path",
            r"^\$?\.?[\w\[\]\.\*]+$",
            "JSONPath expression"
        )
        
        # Pipeline reference format
        self.register_pattern(
            "pipeline-ref",
            r"^[\w\-]+$",
            "Pipeline identifier"
        )
        
        # Task reference format
        self.register_pattern(
            "task-ref",
            r"^[\w\-]+\.[\w\-]+$",
            "Task output reference (task_id.field)"
        )
    
    def register_pattern(self, name: str, pattern: str, description: str = ""):
        """Register a regex-based format validator."""
        self._patterns[name] = pattern
        # Compile pattern for efficiency and to ensure full match
        compiled_pattern = re.compile(pattern)
        self._validators[name] = lambda x: bool(compiled_pattern.fullmatch(str(x)))
    
    def register_function(self, name: str, validator: Callable[[Any], bool], description: str = ""):
        """Register a function-based format validator."""
        self._validators[name] = validator
    
    def validate_format(self, format_name: str, value: Any) -> bool:
        """Validate a value against a format."""
        if format_name not in self._validators:
            return True  # Unknown format, pass by default
        
        try:
            return self._validators[format_name](value)
        except Exception:
            return False
    
    def get_format_pattern(self, format_name: str) -> Optional[str]:
        """Get regex pattern for a format if available."""
        return self._patterns.get(format_name)


class SchemaValidator:
    """Core schema validation engine."""
    
    def __init__(self):
        self.format_validator = FormatValidator()
        self._setup_custom_validator()
    
    def _setup_custom_validator(self):
        """Setup JSON Schema validator with custom format support."""
        # Create custom validator class with our format checker
        self.format_checker = jsonschema.FormatChecker()
        
        # Add custom format validators
        for format_name in ["model-id", "tool-name", "file-path", "yaml-path", "pipeline-ref", "task-ref"]:
            self.format_checker.checks(format_name)(
                lambda instance, format_name=format_name: self.format_validator.validate_format(format_name, instance)
            )
        
        self.validator_class = jsonschema.validators.create(
            meta_schema=Draft7Validator.META_SCHEMA,
            validators=Draft7Validator.VALIDATORS
        )
    
    def validate(self, data: Any, schema: Dict[str, Any], mode: ValidationMode = ValidationMode.STRICT) -> ValidationResult:
        """Validate data against a JSON Schema."""
        result = ValidationResult(valid=True, schema_used=schema)
        
        try:
            # Create validator instance
            validator = Draft7Validator(
                schema,
                format_checker=self.format_checker
            )
            
            # Collect all validation errors
            errors = list(validator.iter_errors(data))
            
            if errors:
                if mode == ValidationMode.LENIENT:
                    # In lenient mode, try to coerce and re-validate
                    data_copy = self._deep_copy_data(data)
                    any_coerced = False
                    
                    for error in errors:
                        coerced = self._try_coerce(error, data_copy)
                        if coerced is not None:
                            self._update_data_at_path(data_copy, error.path, coerced)
                            any_coerced = True
                            result.warnings.append({
                                "message": error.message,
                                "path": list(error.path),
                                "schema_path": list(error.schema_path),
                                "instance": error.instance,
                                "validator": error.validator,
                                "validator_value": error.validator_value,
                                "coerced_to": coerced,
                                "severity": "warning"
                            })
                    
                    # Re-validate coerced data
                    if any_coerced:
                        re_errors = list(validator.iter_errors(data_copy))
                        if not re_errors:
                            # Coercion successful, update original data
                            result.valid = True
                            result.data = data_copy
                        else:
                            # Still has errors after coercion
                            result.valid = False
                            for error in re_errors:
                                result.errors.append({
                                    "message": error.message,
                                    "path": list(error.path),
                                    "schema_path": list(error.schema_path),
                                    "instance": error.instance,
                                    "validator": error.validator,
                                    "validator_value": error.validator_value
                                })
                    else:
                        # No coercion possible
                        result.valid = False
                        for error in errors:
                            result.errors.append({
                                "message": error.message,
                                "path": list(error.path),
                                "schema_path": list(error.schema_path),
                                "instance": error.instance,
                                "validator": error.validator,
                                "validator_value": error.validator_value
                            })
                
                elif mode == ValidationMode.REPORT_ONLY:
                    # Report errors as warnings
                    result.valid = True  # Don't fail in report-only mode
                    for error in errors:
                        result.warnings.append({
                            "message": error.message,
                            "path": list(error.path),
                            "schema_path": list(error.schema_path),
                            "instance": error.instance,
                            "validator": error.validator,
                            "validator_value": error.validator_value,
                            "severity": "info"
                        })
                
                else:  # STRICT mode
                    result.valid = False
                    for error in errors:
                        result.errors.append({
                            "message": error.message,
                            "path": list(error.path),
                            "schema_path": list(error.schema_path),
                            "instance": error.instance,
                            "validator": error.validator,
                            "validator_value": error.validator_value
                        })
            
            if not result.data:
                result.data = data
            
        except jsonschema.SchemaError as e:
            result.valid = False
            result.errors.append({
                "message": f"Invalid schema: {str(e)}",
                "path": [],
                "severity": "error"
            })
        
        return result
    
    def _try_coerce(self, error: ValidationError, data: Any) -> Optional[Any]:
        """Try to coerce data to match schema requirement."""
        if error.validator == "type":
            expected_type = error.validator_value
            instance = error.instance
            
            # String to number coercion
            if expected_type == "number" and isinstance(instance, str):
                try:
                    return float(instance)
                except ValueError:
                    return None
            
            # String to integer coercion
            if expected_type == "integer" and isinstance(instance, str):
                try:
                    return int(instance)
                except ValueError:
                    return None
            
            # Number to string coercion
            if expected_type == "string" and isinstance(instance, (int, float)):
                return str(instance)
            
            # Boolean coercion
            if expected_type == "boolean":
                if isinstance(instance, str):
                    if instance.lower() in ["true", "yes", "1"]:
                        return True
                    elif instance.lower() in ["false", "no", "0"]:
                        return False
        
        return None
    
    def _deep_copy_data(self, data: Any) -> Any:
        """Create a deep copy of the data."""
        return copy.deepcopy(data)
    
    def _update_data_at_path(self, data: Any, path: Any, value: Any):
        """Update data at the given path with the new value."""
        # Convert path to list if it's not already
        path_list = list(path) if hasattr(path, '__iter__') else [path]
        
        if not path_list:
            return
        
        current = data
        for i, key in enumerate(path_list[:-1]):
            if isinstance(current, dict):
                current = current[key]
            elif isinstance(current, list):
                current = current[int(key)]
        
        # Set the final value
        final_key = path_list[-1]
        if isinstance(current, dict):
            current[final_key] = value
        elif isinstance(current, list):
            current[int(final_key)] = value


class ValidationTool(Tool):
    """Comprehensive validation tool with structured output support."""
    
    def __init__(self, model: Optional[Model] = None):
        super().__init__(
            name="validation",
            description="Validate data against schemas with structured output support"
        )
        
        # Parameters
        self.add_parameter("action", "string", "Action: 'validate', 'extract_structured', 'infer_schema'", required=False, default="validate")
        self.add_parameter("data", "any", "Data to validate or text to extract from", required=False)
        self.add_parameter("schema", "object", "JSON Schema for validation", required=False)
        self.add_parameter("mode", "string", "Validation mode: 'strict', 'lenient', 'report_only'", required=False, default="strict")
        self.add_parameter("model", "string", "Model to use for structured extraction", required=False)
        self.add_parameter("text", "string", "Text to extract structured data from", required=False)
        self.add_parameter("pydantic_model", "string", "Pydantic model class name for validation", required=False)
        
        # Core components
        self.schema_validator = SchemaValidator()
        self.format_validator = self.schema_validator.format_validator
        self.model = model
        self.model_name = None
    
    def register_format(self, name: str, validator: Union[str, Callable], description: str = ""):
        """Register a custom format validator."""
        if isinstance(validator, str):
            self.format_validator.register_pattern(name, validator, description)
        else:
            self.format_validator.register_function(name, validator, description)
        
        # Also register with the schema validator's format checker
        self.schema_validator.format_checker.checks(name)(
            lambda instance: self.format_validator.validate_format(name, instance)
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute validation action."""
        action = kwargs.get("action", "validate")
        
        try:
            if action == "validate":
                return await self._validate_data(kwargs)
            elif action == "extract_structured":
                return await self._extract_structured(kwargs)
            elif action == "infer_schema":
                return await self._infer_schema(kwargs)
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action": action
            }
    
    async def _validate_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against a schema."""
        data = params.get("data")
        schema = params.get("schema")
        mode_str = params.get("mode", "strict")
        
        if data is None:
            return {
                "success": False,
                "error": "No data provided for validation"
            }
        
        if not schema:
            return {
                "success": False,
                "error": "No schema provided for validation"
            }
        
        # Parse validation mode
        try:
            mode = ValidationMode(mode_str)
        except ValueError:
            mode = ValidationMode.STRICT
        
        # Perform validation
        result = self.schema_validator.validate(data, schema, mode)
        
        return {
            "success": True,
            "valid": result.valid,
            "errors": result.errors,
            "warnings": result.warnings,
            "data": result.data,
            "mode": mode.value
        }
    
    async def _extract_structured(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured data from text using LLM."""
        text = params.get("text")
        schema = params.get("schema")
        model_name = params.get("model", self.model_name)
        
        if not text:
            return {
                "success": False,
                "error": "No text provided for extraction"
            }
        
        if not schema:
            return {
                "success": False,
                "error": "No schema provided for extraction"
            }
        
        # Check if we have a model available
        if not self.model and not model_name:
            return {
                "success": False,
                "error": "No model available for structured extraction. Please provide 'model' parameter or initialize with a model."
            }
        
        try:
            # Convert JSON Schema to Pydantic model dynamically
            self._json_schema_to_pydantic(schema)
            
            # Build extraction prompt
            prompt = f"""Extract structured data from the following text according to the provided schema.
            
Text:
{text}

Return the extracted data as a JSON object that matches the schema."""
            
            # If we have a direct model, use it
            if self.model:
                # For now, use basic extraction since we don't have LangChain integration
                # This is a placeholder for proper structured output
                response = await self.model.generate(prompt)
                
                # Try to parse JSON from response
                import json
                try:
                    # Look for JSON in the response
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        extracted_data = json.loads(json_str)
                    else:
                        raise ValueError("No JSON found in response")
                except (json.JSONDecodeError, ValueError) as e:
                    return {
                        "success": False,
                        "error": f"Failed to parse model response as JSON: {str(e)}",
                        "raw_response": response
                    }
                
                # Validate extracted data against schema
                validation_result = self.schema_validator.validate(extracted_data, schema)
                
                return {
                    "success": True,
                    "valid": validation_result.valid,
                    "data": validation_result.data or extracted_data,
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings,
                    "model_used": self.model.__class__.__name__
                }
            else:
                # TODO: Implement LangChain structured output when model_name is provided
                return {
                    "success": False,
                    "error": "LangChain structured output integration not yet implemented",
                    "note": "Currently only direct model extraction is supported"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Extraction failed: {str(e)}"
            }
    
    async def _infer_schema(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Infer a JSON Schema from sample data."""
        data = params.get("data")
        
        if data is None:
            return {
                "success": False,
                "error": "No data provided for schema inference"
            }
        
        # Basic schema inference
        schema = self._infer_schema_from_data(data)
        
        return {
            "success": True,
            "schema": schema,
            "data_type": type(data).__name__
        }
    
    def _infer_schema_from_data(self, data: Any) -> Dict[str, Any]:
        """Infer a basic JSON Schema from data."""
        if isinstance(data, dict):
            properties = {}
            required = []
            
            for key, value in data.items():
                properties[key] = self._infer_schema_from_data(value)
                if value is not None:
                    required.append(key)
            
            return {
                "type": "object",
                "properties": properties,
                "required": required
            }
        
        elif isinstance(data, list):
            if not data:
                return {"type": "array", "items": {}}
            
            # Infer from first item (simple approach)
            return {
                "type": "array",
                "items": self._infer_schema_from_data(data[0])
            }
        
        elif isinstance(data, str):
            schema = {"type": "string"}
            
            # Try to detect format
            if re.match(r"^\S+@\S+\.\S+$", data):
                schema["format"] = "email"
            elif re.match(r"^\d{4}-\d{2}-\d{2}$", data):
                schema["format"] = "date"
            elif re.match(r"^https?://", data):
                schema["format"] = "uri"
            
            return schema
        
        elif isinstance(data, int):
            return {"type": "integer"}
        
        elif isinstance(data, float):
            return {"type": "number"}
        
        elif isinstance(data, bool):
            return {"type": "boolean"}
        
        elif data is None:
            return {"type": "null"}
        
        else:
            return {}
    
    def _json_schema_to_pydantic(self, schema: Dict[str, Any]) -> Type[BaseModel]:
        """Convert JSON Schema to Pydantic model dynamically."""
        # This is a simplified conversion - full implementation would handle all JSON Schema features
        
        if schema.get("type") != "object":
            raise ValueError("Only object schemas are supported for extraction")
        
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        
        # Build field definitions
        fields = {}
        for field_name, field_schema in properties.items():
            field_type = self._get_python_type(field_schema)
            
            if field_name in required:
                fields[field_name] = (field_type, Field(..., description=field_schema.get("description", "")))
            else:
                fields[field_name] = (Optional[field_type], Field(None, description=field_schema.get("description", "")))
        
        # Create dynamic Pydantic model
        return create_model("ExtractedData", **fields)
    
    def _get_python_type(self, schema: Dict[str, Any]) -> Type:
        """Convert JSON Schema type to Python type."""
        json_type = schema.get("type", "string")
        
        if json_type == "string":
            return str
        elif json_type == "integer":
            return int
        elif json_type == "number":
            return float
        elif json_type == "boolean":
            return bool
        elif json_type == "array":
            item_type = self._get_python_type(schema.get("items", {}))
            return List[item_type]
        elif json_type == "object":
            # For nested objects, use Dict for simplicity
            return Dict[str, Any]
        else:
            return Any