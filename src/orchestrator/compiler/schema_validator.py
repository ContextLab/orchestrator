"""Schema validator for pipeline definitions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import jsonschema


class SchemaValidationError(Exception):
    """Raised when schema validation fails."""
    pass


class SchemaValidator:
    """
    Validates pipeline definitions against JSON schema.
    
    Ensures that YAML pipeline definitions conform to the expected
    structure and contain all required fields.
    """
    
    def __init__(self, custom_schema: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize schema validator.
        
        Args:
            custom_schema: Custom schema to use instead of default
        """
        self.schema = custom_schema or self._get_default_schema()
        self.validator = jsonschema.Draft7Validator(self.schema)
    
    def validate(self, pipeline_def: Dict[str, Any]) -> None:
        """
        Validate pipeline definition against schema.
        
        Args:
            pipeline_def: Pipeline definition to validate
            
        Raises:
            SchemaValidationError: If validation fails
        """
        try:
            self.validator.validate(pipeline_def)
        except jsonschema.ValidationError as e:
            raise SchemaValidationError(f"Schema validation failed: {e.message}") from e
    
    def is_valid(self, pipeline_def: Dict[str, Any]) -> bool:
        """
        Check if pipeline definition is valid.
        
        Args:
            pipeline_def: Pipeline definition to check
            
        Returns:
            True if valid, False otherwise
        """
        try:
            self.validate(pipeline_def)
            return True
        except SchemaValidationError:
            return False
    
    def get_validation_errors(self, pipeline_def: Dict[str, Any]) -> List[str]:
        """
        Get list of validation errors.
        
        Args:
            pipeline_def: Pipeline definition to validate
            
        Returns:
            List of error messages
        """
        errors = []
        for error in self.validator.iter_errors(pipeline_def):
            error_path = " -> ".join(str(p) for p in error.absolute_path)
            if error_path:
                errors.append(f"At {error_path}: {error.message}")
            else:
                errors.append(error.message)
        return errors
    
    def _get_default_schema(self) -> Dict[str, Any]:
        """Get default pipeline schema."""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["name", "steps"],
            "properties": {
                "id": {
                    "type": "string",
                    "pattern": "^[a-zA-Z][a-zA-Z0-9_-]*$"
                },
                "name": {
                    "type": "string",
                    "minLength": 1
                },
                "version": {
                    "type": "string",
                    "pattern": "^\\d+\\.\\d+\\.\\d+$"
                },
                "description": {
                    "type": "string"
                },
                "context": {
                    "type": "object",
                    "properties": {
                        "timeout": {
                            "type": "integer",
                            "minimum": 1
                        },
                        "max_retries": {
                            "type": "integer",
                            "minimum": 0
                        },
                        "checkpoint_strategy": {
                            "type": "string",
                            "enum": ["adaptive", "fixed", "none"]
                        }
                    }
                },
                "metadata": {
                    "type": "object"
                },
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["id", "action"],
                        "properties": {
                            "id": {
                                "type": "string",
                                "pattern": "^[a-zA-Z][a-zA-Z0-9_-]*$"
                            },
                            "name": {
                                "type": "string"
                            },
                            "action": {
                                "type": "string",
                                "minLength": 1
                            },
                            "parameters": {
                                "type": "object"
                            },
                            "dependencies": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "pattern": "^[a-zA-Z][a-zA-Z0-9_-]*$"
                                }
                            },
                            "timeout": {
                                "type": "integer",
                                "minimum": 1
                            },
                            "max_retries": {
                                "type": "integer",
                                "minimum": 0
                            },
                            "on_failure": {
                                "type": "string",
                                "enum": ["continue", "fail", "retry", "skip"]
                            },
                            "requires_model": {
                                "type": "object",
                                "properties": {
                                    "min_size": {
                                        "type": "string"
                                    },
                                    "expertise": {
                                        "type": "string",
                                        "enum": ["low", "medium", "high", "very-high"]
                                    },
                                    "capabilities": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        }
                                    }
                                }
                            },
                            "metadata": {
                                "type": "object"
                            }
                        }
                    }
                },
                "inputs": {
                    "type": "object"
                },
                "outputs": {
                    "type": "object"
                }
            }
        }
    
    def add_custom_validator(self, property_name: str, validator_func) -> None:
        """
        Add custom validator for a property.
        
        Args:
            property_name: Name of the property to validate
            validator_func: Validation function
        """
        # Create a new validator class with custom validation
        all_validators = dict(jsonschema.Draft7Validator.TYPE_CHECKER.checkers)
        all_validators[property_name] = validator_func
        
        CustomValidator = jsonschema.validators.create(
            meta_schema=jsonschema.Draft7Validator.META_SCHEMA,
            validators=jsonschema.Draft7Validator.VALIDATORS,
            type_checker=jsonschema.Draft7Validator.TYPE_CHECKER.redefine_many(all_validators)
        )
        
        self.validator = CustomValidator(self.schema)
    
    def validate_task_dependencies(self, pipeline_def: Dict[str, Any]) -> List[str]:
        """
        Validate task dependencies.
        
        Args:
            pipeline_def: Pipeline definition
            
        Returns:
            List of dependency validation errors
        """
        errors = []
        
        # Get all task IDs
        steps = pipeline_def.get("steps", [])
        task_ids = {step["id"] for step in steps}
        
        # Check dependencies
        for step in steps:
            dependencies = step.get("dependencies", [])
            for dep in dependencies:
                if dep not in task_ids:
                    errors.append(f"Task '{step['id']}' depends on non-existent task '{dep}'")
                if dep == step["id"]:
                    errors.append(f"Task '{step['id']}' cannot depend on itself")
        
        return errors
    
    def validate_unique_task_ids(self, pipeline_def: Dict[str, Any]) -> List[str]:
        """
        Validate that all task IDs are unique.
        
        Args:
            pipeline_def: Pipeline definition
            
        Returns:
            List of uniqueness validation errors
        """
        errors = []
        
        steps = pipeline_def.get("steps", [])
        task_ids = []
        
        for step in steps:
            task_id = step.get("id")
            if task_id in task_ids:
                errors.append(f"Duplicate task ID: '{task_id}'")
            else:
                task_ids.append(task_id)
        
        return errors
    
    def validate_complete(self, pipeline_def: Dict[str, Any]) -> List[str]:
        """
        Perform complete validation including custom checks.
        
        Args:
            pipeline_def: Pipeline definition
            
        Returns:
            List of all validation errors
        """
        errors = []
        
        # Schema validation
        errors.extend(self.get_validation_errors(pipeline_def))
        
        # Custom validations
        errors.extend(self.validate_task_dependencies(pipeline_def))
        errors.extend(self.validate_unique_task_ids(pipeline_def))
        
        return errors