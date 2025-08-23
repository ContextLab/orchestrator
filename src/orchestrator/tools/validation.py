"""Comprehensive validation tool with JSON Schema and structured output support."""

import copy
import re
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union

import jsonschema
from jsonschema import Draft7Validator, ValidationError

# Try to import pydantic, install if needed
try:
    from pydantic import BaseModel, Field, create_model
except ImportError:
    import subprocess
    import sys
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Pydantic not found. Installing...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pydantic"])
        from pydantic import BaseModel, Field, create_model

        logger.info("Pydantic installed successfully")
    except Exception as e:
        raise ImportError(
            f"Failed to install pydantic: {e}. Please install manually with: pip install pydantic"
        )

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
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Validation errors"
    )
    warnings: List[Dict[str, Any]] = Field(
        default_factory=list, description="Validation warnings"
    )
    data: Optional[Any] = Field(default=None, description="Validated/coerced data")
    schema_used: Optional[Dict[str, Any]] = Field(
        default=None, description="Schema used for validation"
    )


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
            "Model identifier (provider/model-name)",
        )

        # Tool name format
        self.register_pattern(
            "tool-name",
            r"^[a-z][a-z0-9\-_]*$",
            "Tool name (lowercase, alphanumeric, hyphens, underscores)",
        )

        # File path format
        self.register_function(
            "file-path",
            lambda x: len(x) > 0 and not x.startswith(" "),
            "Valid file system path",
        )

        # YAML/JSONPath format
        self.register_pattern(
            "yaml-path", r"^\$?\.?[\w\[\]\.\*]+$", "JSONPath expression"
        )

        # Pipeline reference format
        self.register_pattern("pipeline-ref", r"^[\w\-]+$", "Pipeline identifier")

        # Task reference format
        self.register_pattern(
            "task-ref", r"^[\w\-]+\.[\w\-]+$", "Task output reference (task_id.field)"
        )

    def register_pattern(self, name: str, pattern: str, description: str = ""):
        """Register a regex-based format validator."""
        self._patterns[name] = pattern
        # Compile pattern for efficiency and to ensure full match
        compiled_pattern = re.compile(pattern)
        self._validators[name] = lambda x: bool(compiled_pattern.fullmatch(str(x)))

    def register_function(
        self, name: str, validator: Callable[[Any], bool], description: str = ""
    ):
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
        for format_name in [
            "model-id",
            "tool-name",
            "file-path",
            "yaml-path",
            "pipeline-ref",
            "task-ref",
        ]:
            self.format_checker.checks(format_name)(
                lambda instance, format_name=format_name: self.format_validator.validate_format(
                    format_name, instance
                )
            )

        self.validator_class = jsonschema.validators.create(
            meta_schema=Draft7Validator.META_SCHEMA,
            validators=Draft7Validator.VALIDATORS,
        )

    def validate(
        self,
        data: Any,
        schema: Dict[str, Any],
        mode: ValidationMode = ValidationMode.STRICT,
    ) -> ValidationResult:
        """Validate data against a JSON Schema."""
        result = ValidationResult(valid=True, schema_used=schema)

        try:
            # Create validator instance
            validator = Draft7Validator(schema, format_checker=self.format_checker)

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
                            result.warnings.append(
                                {
                                    "message": error.message,
                                    "path": list(error.path),
                                    "schema_path": list(error.schema_path),
                                    "instance": error.instance,
                                    "validator": error.validator,
                                    "validator_value": error.validator_value,
                                    "coerced_to": coerced,
                                    "severity": "warning",
                                }
                            )

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
                                result.errors.append(
                                    {
                                        "message": error.message,
                                        "path": list(error.path),
                                        "schema_path": list(error.schema_path),
                                        "instance": error.instance,
                                        "validator": error.validator,
                                        "validator_value": error.validator_value,
                                    }
                                )
                    else:
                        # No coercion possible
                        result.valid = False
                        for error in errors:
                            result.errors.append(
                                {
                                    "message": error.message,
                                    "path": list(error.path),
                                    "schema_path": list(error.schema_path),
                                    "instance": error.instance,
                                    "validator": error.validator,
                                    "validator_value": error.validator_value,
                                }
                            )

                elif mode == ValidationMode.REPORT_ONLY:
                    # Report errors as warnings
                    result.valid = True  # Don't fail in report-only mode
                    for error in errors:
                        result.warnings.append(
                            {
                                "message": error.message,
                                "path": list(error.path),
                                "schema_path": list(error.schema_path),
                                "instance": error.instance,
                                "validator": error.validator,
                                "validator_value": error.validator_value,
                                "severity": "info",
                            }
                        )

                else:  # STRICT mode
                    result.valid = False
                    for error in errors:
                        result.errors.append(
                            {
                                "message": error.message,
                                "path": list(error.path),
                                "schema_path": list(error.schema_path),
                                "instance": error.instance,
                                "validator": error.validator,
                                "validator_value": error.validator_value,
                            }
                        )

            if not result.data:
                result.data = data

        except jsonschema.SchemaError as e:
            result.valid = False
            result.errors.append(
                {
                    "message": f"Invalid schema: {str(e)}",
                    "path": [],
                    "severity": "error",
                }
            )

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
        path_list = list(path) if hasattr(path, "__iter__") else [path]

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
            description="Validate data against schemas with structured output support",
        )

        # Parameters
        self.add_parameter(
            "action",
            "string",
            "Action: 'validate', 'extract_structured', 'infer_schema', 'quality_check'",
            required=False,
            default="validate",
        )
        self.add_parameter(
            "data", "any", "Data to validate or text to extract from", required=False
        )
        self.add_parameter(
            "schema", "object", "JSON Schema for validation", required=False
        )
        self.add_parameter(
            "mode",
            "string",
            "Validation mode: 'strict', 'lenient', 'report_only'",
            required=False,
            default="strict",
        )
        self.add_parameter(
            "model", "string", "Model to use for structured extraction", required=False
        )
        self.add_parameter(
            "text", "string", "Text to extract structured data from", required=False
        )
        self.add_parameter(
            "pydantic_model",
            "string",
            "Pydantic model class name for validation",
            required=False,
        )
        self.add_parameter(
            "threshold",
            "number",
            "Quality threshold for validation (0.0-1.0)",
            required=False,
            default=0.8,
        )

        # Core components
        self.schema_validator = SchemaValidator()
        self.format_validator = self.schema_validator.format_validator
        self.model = model
        self.model_name = None

    def register_format(
        self, name: str, validator: Union[str, Callable], description: str = ""
    ):
        """Register a custom format validator."""
        if isinstance(validator, str):
            self.format_validator.register_pattern(name, validator, description)
        else:
            self.format_validator.register_function(name, validator, description)

        # Also register with the schema validator's format checker
        self.schema_validator.format_checker.checks(name)(
            lambda instance: self.format_validator.validate_format(name, instance)
        )

    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Execute validation action."""
        action = kwargs.get("action", "validate")

        try:
            if action == "validate":
                return await self._validate_data(kwargs)
            elif action == "extract_structured":
                return await self._extract_structured(kwargs)
            elif action == "infer_schema":
                return await self._infer_schema(kwargs)
            elif action == "quality_check":
                return await self._validate_quality_check(kwargs)
            else:
                return {"result": None, "success": False, "error": f"Unknown action: {action}"}
        except Exception as e:
            return {"result": None, "success": False, "error": str(e)}

    async def _validate_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against a schema."""
        data = params.get("data")
        schema = params.get("schema")
        mode_str = params.get("mode", "strict")

        if data is None:
            return {"result": None, "success": False, "error": "No data provided for validation"}

        if not schema:
            return {"result": None, "success": False, "error": "No schema provided for validation"}
        
        # Parse data string if needed
        if isinstance(data, str):
            import json
            import csv
            import io
            
            # First try JSON
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                # Try CSV if JSON fails
                try:
                    # Check if it looks like CSV (has commas and newlines)
                    if ',' in data and '\n' in data:
                        reader = csv.DictReader(io.StringIO(data))
                        data = list(reader)
                        
                        # Convert numeric strings to appropriate types based on schema
                        if isinstance(data, list) and data and schema.get("type") == "object":
                            # For CSV validation, each row should match the schema
                            for row in data:
                                for field, value in row.items():
                                    if field in schema.get("properties", {}):
                                        field_type = schema["properties"][field].get("type")
                                        if field_type == "integer" and value:
                                            try:
                                                row[field] = int(value)
                                            except ValueError:
                                                pass
                                        elif field_type == "number" and value:
                                            try:
                                                row[field] = float(value)
                                            except ValueError:
                                                pass
                    else:
                        # Not CSV, return error
                        return {"result": None, "success": False, "error": f"Invalid data format: not JSON or CSV"}
                except Exception as e:
                    return {"result": None, "success": False, "error": f"Failed to parse CSV data: {str(e)}"}

        # Parse validation mode
        try:
            mode = ValidationMode(mode_str)
        except ValueError:
            mode = ValidationMode.STRICT

        # Handle CSV validation (array of objects)
        if isinstance(data, list) and all(isinstance(row, dict) for row in data):
            # Validate each row against the schema
            all_valid = True
            all_errors = []
            all_warnings = []
            validated_data = []
            
            for i, row in enumerate(data):
                result = self.schema_validator.validate(row, schema, mode)
                if not result.valid:
                    all_valid = False
                    # Add row number to error messages
                    for error in result.errors:
                        error["row"] = i + 1
                        all_errors.append(error)
                for warning in result.warnings:
                    warning["row"] = i + 1
                    all_warnings.append(warning)
                validated_data.append(result.data or row)
            
            return {
                "result": {
                    "valid": all_valid,
                    "errors": all_errors,
                    "warnings": all_warnings,
                    "data": validated_data,
                    "mode": mode.value,
                    "rows_validated": len(data),
                },
                "success": True,
                "error": None,
            }
        else:
            # Perform single object validation
            result = self.schema_validator.validate(data, schema, mode)

            return {
                "result": {
                    "valid": result.valid,
                    "errors": result.errors,
                    "warnings": result.warnings,
                    "data": result.data,
                    "mode": mode.value,
                },
                "success": True,
                "error": None,
            }

    async def _extract_structured(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured data from text using LLM."""
        text = params.get("text")
        schema = params.get("schema")
        model_name = params.get("model", self.model_name)

        if not text:
            return {"result": None, "success": False, "error": "No text provided for extraction"}

        if not schema:
            return {"result": None, "success": False, "error": "No schema provided for extraction"}

        # Check if we have a model available
        if not self.model and not model_name:
            return {
                "result": None,
                "success": False,
                "error": "No model available for structured extraction. Please provide 'model' parameter or initialize with a model.",
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
                    json_start = response.find("{")
                    json_end = response.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        extracted_data = json.loads(json_str)
                    else:
                        raise ValueError("No JSON found in response")
                except (json.JSONDecodeError, ValueError) as e:
                    return {
                        "result": None,
                        "success": False,
                        "error": f"Failed to parse model response as JSON: {str(e)}",
                    }

                # Validate extracted data against schema
                validation_result = self.schema_validator.validate(
                    extracted_data, schema
                )

                return {
                    "result": {
                        "valid": validation_result.valid,
                        "data": validation_result.data or extracted_data,
                        "errors": validation_result.errors,
                        "warnings": validation_result.warnings,
                        "model_used": self.model.__class__.__name__,
                    },
                    "success": True,
                    "error": None,
                }
            else:
                # TODO: Implement LangChain structured output when model_name is provided
                return {
                    "result": None,
                    "success": False,
                    "error": "LangChain structured output integration not yet implemented",
                }

        except Exception as e:
            return {"result": None, "success": False, "error": f"Extraction failed: {str(e)}"}

    async def _infer_schema(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Infer a JSON Schema from sample data."""
        data = params.get("data")

        if data is None:
            return {"result": None, "success": False, "error": "No data provided for schema inference"}

        # Basic schema inference
        schema = self._infer_schema_from_data(data)

        return {"result": {"schema": schema, "data_type": type(data).__name__}, "success": True, "error": None}

    def _infer_schema_from_data(self, data: Any) -> Dict[str, Any]:
        """Infer a basic JSON Schema from data."""
        if isinstance(data, dict):
            properties = {}
            required = []

            for key, value in data.items():
                properties[key] = self._infer_schema_from_data(value)
                if value is not None:
                    required.append(key)

            return {"type": "object", "properties": properties, "required": required}

        elif isinstance(data, list):
            if not data:
                return {"type": "array", "items": {}}

            # Infer from first item (simple approach)
            return {"type": "array", "items": self._infer_schema_from_data(data[0])}

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
                fields[field_name] = (
                    field_type,
                    Field(..., description=field_schema.get("description", "")),
                )
            else:
                fields[field_name] = (
                    Optional[field_type],
                    Field(None, description=field_schema.get("description", "")),
                )

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

    async def _validate_quality_check(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive quality check on data."""
        data = params.get("data")
        threshold = params.get("threshold", 0.8)
        
        if data is None:
            return {"result": None, "success": False, "error": "No data provided for quality check"}
        
        # Parse data if it's a string
        if isinstance(data, str):
            import json
            import csv
            import io
            
            # Try JSON first
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                # Try CSV if JSON fails
                try:
                    if ',' in data and '\n' in data:
                        reader = csv.DictReader(io.StringIO(data))
                        data = list(reader)
                        if not data:
                            data = []
                    else:
                        return {"result": None, "success": False, "error": "Invalid data format: not JSON or CSV"}
                except Exception as e:
                    return {"result": None, "success": False, "error": f"Failed to parse data: {str(e)}"}
        
        # Initialize quality metrics
        completeness_score = 0.0
        accuracy_score = 0.0
        consistency_score = 0.0
        
        try:
            # Analyze data based on type
            if isinstance(data, list) and data:
                # Analyze array of records
                completeness_score = self._analyze_completeness(data)
                accuracy_score = self._analyze_accuracy(data)
                consistency_score = self._analyze_consistency(data)
            elif isinstance(data, dict):
                # Analyze single record
                completeness_score = self._analyze_completeness([data])
                accuracy_score = self._analyze_accuracy([data])
                consistency_score = self._analyze_consistency([data])
            elif isinstance(data, list) and not data:
                # Empty data
                completeness_score = 0.0
                accuracy_score = 1.0  # No data to be inaccurate
                consistency_score = 1.0  # Empty data is consistent
            else:
                # Other data types - basic checks
                completeness_score = 1.0 if data is not None else 0.0
                accuracy_score = 1.0  # Assume accurate if not null
                consistency_score = 1.0  # Single value is consistent
            
            # Calculate overall score
            overall_score = (completeness_score + accuracy_score + consistency_score) / 3.0
            
            # Determine validity based on threshold
            is_valid = overall_score >= threshold
            
            result = {
                'completeness': round(completeness_score, 3),
                'accuracy': round(accuracy_score, 3),
                'consistency': round(consistency_score, 3),
                'overall_score': round(overall_score, 3),
                'valid': is_valid
            }
            
            return {
                "result": result,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            return {"result": None, "success": False, "error": f"Quality check failed: {str(e)}"}
    
    def _analyze_completeness(self, data: List[Dict]) -> float:
        """Analyze data completeness."""
        if not data:
            return 0.0
        
        total_fields = 0
        filled_fields = 0
        
        # Get all possible field names
        all_fields = set()
        for record in data:
            if isinstance(record, dict):
                all_fields.update(record.keys())
        
        if not all_fields:
            return 1.0  # No fields to check
        
        # Check completeness for each record
        for record in data:
            if isinstance(record, dict):
                for field in all_fields:
                    total_fields += 1
                    value = record.get(field)
                    # Consider non-null, non-empty values as complete
                    if value is not None and value != "" and str(value).strip():
                        filled_fields += 1
        
        return filled_fields / total_fields if total_fields > 0 else 1.0
    
    def _analyze_accuracy(self, data: List[Dict]) -> float:
        """Analyze data accuracy based on format validation."""
        if not data:
            return 1.0  # No data to be inaccurate
        
        total_values = 0
        accurate_values = 0
        
        for record in data:
            if isinstance(record, dict):
                for field, value in record.items():
                    if value is not None and value != "":
                        total_values += 1
                        # Basic accuracy checks
                        if self._is_value_accurate(field, value):
                            accurate_values += 1
        
        return accurate_values / total_values if total_values > 0 else 1.0
    
    def _analyze_consistency(self, data: List[Dict]) -> float:
        """Analyze data consistency."""
        if not data or len(data) < 2:
            return 1.0  # Single records are consistent
        
        # Track data types and formats for each field
        field_types = {}
        field_formats = {}
        
        for record in data:
            if isinstance(record, dict):
                for field, value in record.items():
                    if value is not None and value != "":
                        value_type = type(value).__name__
                        
                        # Track types for this field
                        if field not in field_types:
                            field_types[field] = {}
                        field_types[field][value_type] = field_types[field].get(value_type, 0) + 1
                        
                        # Track formats for string fields
                        if isinstance(value, str):
                            format_pattern = self._detect_format_pattern(value)
                            if format_pattern:
                                if field not in field_formats:
                                    field_formats[field] = {}
                                field_formats[field][format_pattern] = field_formats[field].get(format_pattern, 0) + 1
        
        # Calculate consistency scores
        total_fields = 0
        consistent_fields = 0
        
        for field in field_types:
            total_fields += 1
            # Field is consistent if one type represents >80% of values
            type_counts = field_types[field]
            total_count = sum(type_counts.values())
            max_type_count = max(type_counts.values())
            
            type_consistency = max_type_count / total_count
            
            # Check format consistency for strings
            format_consistency = 1.0
            if field in field_formats:
                format_counts = field_formats[field]
                total_format_count = sum(format_counts.values())
                max_format_count = max(format_counts.values())
                format_consistency = max_format_count / total_format_count
            
            # Combined consistency score for this field
            field_consistency = (type_consistency + format_consistency) / 2
            if field_consistency >= 0.8:
                consistent_fields += 1
        
        return consistent_fields / total_fields if total_fields > 0 else 1.0
    
    def _is_value_accurate(self, field: str, value: Any) -> bool:
        """Check if a value appears to be accurate for its field."""
        # Basic accuracy checks
        
        # Email validation
        if 'email' in field.lower():
            if isinstance(value, str):
                import re
                return bool(re.match(r'^[^@]+@[^@]+\.[^@]+$', value))
            return False
        
        # Date validation
        if 'date' in field.lower():
            if isinstance(value, str):
                import re
                # Check for common date formats
                date_patterns = [
                    r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                    r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                    r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
                ]
                return any(re.match(pattern, value) for pattern in date_patterns)
            return False
        
        # Numeric fields
        if any(keyword in field.lower() for keyword in ['id', 'count', 'number', 'quantity', 'amount']):
            if isinstance(value, str):
                try:
                    float(value)
                    return True
                except ValueError:
                    return False
            return isinstance(value, (int, float))
        
        # URL validation
        if 'url' in field.lower() or 'link' in field.lower():
            if isinstance(value, str):
                import re
                return bool(re.match(r'https?://', value))
            return False
        
        # Default: assume accurate if not obviously wrong
        return True
    
    def _detect_format_pattern(self, value: str) -> Optional[str]:
        """Detect common format patterns in string values."""
        import re
        
        # Common patterns
        if re.match(r'^\d{4}-\d{2}-\d{2}$', value):
            return 'date_iso'
        elif re.match(r'^\d{2}/\d{2}/\d{4}$', value):
            return 'date_us'
        elif re.match(r'^[^@]+@[^@]+\.[^@]+$', value):
            return 'email'
        elif re.match(r'^https?://', value):
            return 'url'
        elif re.match(r'^\d+$', value):
            return 'numeric'
        elif re.match(r'^\d*\.\d+$', value):
            return 'decimal'
        elif re.match(r'^[A-Z]{2,3}$', value):
            return 'code_upper'
        elif re.match(r'^[a-z]{2,3}$', value):
            return 'code_lower'
        else:
            return None
