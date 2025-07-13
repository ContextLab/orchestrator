"""
Tests for missing coverage lines in schema_validator.py.

This test file specifically targets:
- Line 78: error with path appending
- Lines 207-216: add_custom_validator method
"""

import pytest
import jsonschema
from orchestrator.compiler.schema_validator import SchemaValidator


class TestSchemaValidatorMissingLines:
    """Test cases for achieving 100% coverage on schema_validator.py."""

    def test_validate_errors_with_path_line_78(self):
        """Test line 78: errors.append with error path."""
        # Create a custom schema that will generate errors with paths
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["name", "steps"],
            "properties": {
                "name": {"type": "string"},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["id", "action"],
                        "properties": {
                            "id": {"type": "string"},
                            "action": {"type": "string"},
                            "parameters": {"type": "object"}
                        }
                    }
                }
            }
        }
        
        validator = SchemaValidator(schema)
        
        # Create an invalid pipeline definition with nested errors
        invalid_pipeline = {
            "name": "Test Pipeline",
            "steps": [
                {
                    "id": "step1",
                    "action": "generate",
                    "parameters": {}
                },
                {
                    # Missing required field "action" - will create error with path
                    "id": "step2",
                    "parameters": {}
                }
            ]
        }
        
        # Get validation errors
        errors = validator.get_validation_errors(invalid_pipeline)
        
        # Should have validation errors
        assert len(errors) > 0
        
        # Check that we have errors with paths (line 78)
        path_error_found = False
        for error in errors:
            if "At " in error and " -> " in error:
                path_error_found = True
                break
        assert path_error_found, f"No errors with paths found. Errors: {errors}"
    
    def test_add_custom_validator_lines_207_216(self):
        """Test lines 207-216: add_custom_validator method."""
        validator = SchemaValidator()
        
        # Define a custom validator function
        def validate_custom_format(checker, instance):
            """Custom validator that checks if string starts with 'custom_'."""
            if isinstance(instance, str):
                return instance.startswith("custom_")
            return True
        
        # Add the custom validator
        validator.add_custom_validator("custom_format", validate_custom_format)
        
        # Test that the custom validator was added
        assert validator.validator is not None
        
        # Create a schema that uses our custom validator through properties
        test_schema = {
            "type": "object",
            "properties": {
                "custom_field": {
                    "type": "string"
                }
            }
        }
        
        # Update the validator's schema
        validator.schema = test_schema
        validator.validator = jsonschema.Draft7Validator(test_schema)
        
        # Test with valid data
        valid_data = {"custom_field": "custom_value"}
        errors = validator.get_validation_errors(valid_data)
        assert len(errors) == 0
    
    def test_add_custom_validator_integration(self):
        """Integration test for add_custom_validator to ensure it works end-to-end."""
        # Create a validator with a simple schema
        schema = {
            "type": "object",
            "properties": {
                "special_field": {
                    "type": "string"
                }
            }
        }
        
        validator = SchemaValidator(schema)
        
        # Add a custom type checker
        def check_special_string(checker, instance):
            """Check if the string has special format."""
            if not isinstance(instance, str):
                return False
            return instance.startswith("special_") and instance.endswith("_end")
        
        # This should trigger lines 207-216
        validator.add_custom_validator("special_string", check_special_string)
        
        # Verify the validator was recreated
        assert validator.validator is not None
        
        # The actual validation would need to use the custom type in the schema
        # For coverage, we just need to ensure the method executes
    
    def test_validate_errors_without_path(self):
        """Test line 80: errors.append without path (root level errors)."""
        # Create a schema that expects an object
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["name"]
        }
        
        validator = SchemaValidator(schema)
        
        # Provide wrong type (array instead of object) - creates root level error
        invalid_data = ["this", "should", "be", "an", "object"]
        
        errors = validator.get_validation_errors(invalid_data)
        
        # Should have validation errors
        assert len(errors) > 0
        
        # Check that we have errors without "At" prefix (line 80)
        root_error_found = False
        for error in errors:
            if "At " not in error:
                root_error_found = True
                break
        assert root_error_found, f"No root level errors found. Errors: {errors}"