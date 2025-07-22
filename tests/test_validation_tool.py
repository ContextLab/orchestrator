"""Tests for the comprehensive ValidationTool."""

import pytest

from orchestrator.tools.validation import (
    ValidationTool, 
    ValidationMode, 
    SchemaValidator,
    FormatValidator
)


class TestFormatValidator:
    """Test custom format validators."""
    
    def test_builtin_formats(self):
        """Test built-in orchestrator formats."""
        validator = FormatValidator()
        
        # Test model-id format
        assert validator.validate_format("model-id", "openai/gpt-4")
        assert validator.validate_format("model-id", "anthropic/claude-3-5-sonnet")
        assert not validator.validate_format("model-id", "invalid model")
        
        # Test tool-name format  
        assert validator.validate_format("tool-name", "web-search")
        assert validator.validate_format("tool-name", "file_system")
        assert not validator.validate_format("tool-name", "Tool-Name")  # No uppercase
        assert not validator.validate_format("tool-name", "123tool")  # Can't start with number
        
        # Test file-path format
        assert validator.validate_format("file-path", "/path/to/file.txt")
        assert validator.validate_format("file-path", "relative/path.py")
        assert not validator.validate_format("file-path", "")
        assert not validator.validate_format("file-path", " /starts/with/space")
        
        # Test yaml-path format
        assert validator.validate_format("yaml-path", "$.data.items[0]")
        assert validator.validate_format("yaml-path", "data.nested.field")
        assert validator.validate_format("yaml-path", "$[*].name")
        
        # Test task-ref format
        assert validator.validate_format("task-ref", "process_data.result")
        assert validator.validate_format("task-ref", "step-1.output")
        assert not validator.validate_format("task-ref", "invalid_ref")
    
    def test_custom_format_registration(self):
        """Test registering custom formats."""
        validator = FormatValidator()
        
        # Register pattern-based format
        validator.register_pattern(
            "phone",
            r"^\+?[1-9]\d{1,14}$",
            "E.164 phone number"
        )
        
        assert validator.validate_format("phone", "+1234567890")
        assert validator.validate_format("phone", "12345")  # Valid: 5 digits total
        assert not validator.validate_format("phone", "023")  # Invalid: starts with 0
        assert not validator.validate_format("phone", "")  # Invalid: empty
        assert not validator.validate_format("phone", "abc")  # Invalid: not numeric
        
        # Register function-based format
        validator.register_function(
            "even_number",
            lambda x: isinstance(x, int) and x % 2 == 0,
            "Even number"
        )
        
        assert validator.validate_format("even_number", 4)
        assert not validator.validate_format("even_number", 3)


class TestSchemaValidator:
    """Test JSON Schema validation."""
    
    def test_basic_validation(self):
        """Test basic schema validation."""
        validator = SchemaValidator()
        
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
                "email": {"type": "string", "format": "email"}
            },
            "required": ["name", "email"]
        }
        
        # Valid data
        data = {
            "name": "John Doe",
            "age": 30,
            "email": "john@example.com"
        }
        
        result = validator.validate(data, schema)
        assert result.valid
        assert len(result.errors) == 0
        
        # Invalid data - missing required field
        data = {
            "name": "John Doe",
            "age": 30
        }
        
        result = validator.validate(data, schema)
        assert not result.valid
        assert len(result.errors) == 1
        assert "email" in result.errors[0]["message"]
    
    def test_validation_modes(self):
        """Test different validation modes."""
        validator = SchemaValidator()
        
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"}
            }
        }
        
        # String that can be coerced to integer
        data = {"count": "42"}
        
        # Strict mode - should fail
        result = validator.validate(data, schema, ValidationMode.STRICT)
        assert not result.valid
        assert len(result.errors) == 1
        
        # Lenient mode - should coerce and warn
        result = validator.validate(data, schema, ValidationMode.LENIENT)
        assert result.valid
        assert len(result.warnings) == 1
        assert result.warnings[0]["coerced_to"] == 42
        
        # Report-only mode - should pass with warnings
        result = validator.validate(data, schema, ValidationMode.REPORT_ONLY)
        assert result.valid
        assert len(result.warnings) == 1
    
    def test_nested_schemas(self):
        """Test validation of nested schemas."""
        validator = SchemaValidator()
        
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "contacts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "enum": ["email", "phone"]},
                                    "value": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            }
        }
        
        # Valid nested data
        data = {
            "user": {
                "name": "Alice",
                "contacts": [
                    {"type": "email", "value": "alice@example.com"},
                    {"type": "phone", "value": "+1234567890"}
                ]
            }
        }
        
        result = validator.validate(data, schema)
        assert result.valid


class TestValidationTool:
    """Test the ValidationTool."""
    
    @pytest.fixture
    def validation_tool(self):
        """Create ValidationTool instance."""
        return ValidationTool()
    
    @pytest.mark.asyncio
    async def test_validate_action(self, validation_tool):
        """Test basic validation action."""
        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string", "format": "uuid"},
                "name": {"type": "string"},
                "active": {"type": "boolean"}
            },
            "required": ["id", "name"]
        }
        
        # Valid data
        result = await validation_tool.execute(
            action="validate",
            data={
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "name": "Test Item",
                "active": True
            },
            schema=schema
        )
        
        assert result["success"]
        assert result["valid"]
        assert len(result["errors"]) == 0
        
        # Invalid data
        result = await validation_tool.execute(
            action="validate",
            data={
                "id": "not-a-uuid",
                "active": "yes"  # Wrong type
            },
            schema=schema
        )
        
        assert result["success"]  # Tool executed successfully
        assert not result["valid"]  # But validation failed
        assert len(result["errors"]) >= 2  # Missing name, invalid format
    
    @pytest.mark.asyncio
    async def test_infer_schema_action(self, validation_tool):
        """Test schema inference from data."""
        sample_data = {
            "user": {
                "name": "John Doe",
                "email": "john@example.com",
                "age": 30,
                "active": True
            },
            "scores": [95, 87, 92],
            "created": "2024-01-15"
        }
        
        result = await validation_tool.execute(
            action="infer_schema",
            data=sample_data
        )
        
        assert result["success"]
        assert "schema" in result
        
        schema = result["schema"]
        assert schema["type"] == "object"
        assert "user" in schema["properties"]
        assert schema["properties"]["user"]["type"] == "object"
        assert schema["properties"]["scores"]["type"] == "array"
        
        # Check format detection
        user_props = schema["properties"]["user"]["properties"]
        assert user_props["email"].get("format") == "email"
        assert schema["properties"]["created"].get("format") == "date"
    
    @pytest.mark.asyncio
    async def test_custom_format_validation(self, validation_tool):
        """Test validation with custom formats."""
        # Register custom format
        validation_tool.register_format(
            "order-id",
            r"^ORD-\d{6}$",
            "Order ID format"
        )
        
        schema = {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "format": "order-id"},
                "model_id": {"type": "string", "format": "model-id"}
            }
        }
        
        # Valid data
        result = await validation_tool.execute(
            action="validate",
            data={
                "order_id": "ORD-123456",
                "model_id": "openai/gpt-4"
            },
            schema=schema
        )
        
        assert result["valid"]
        
        # Invalid format
        result = await validation_tool.execute(
            action="validate",
            data={
                "order_id": "ORDER-123",  # Wrong format
                "model_id": "gpt-4"  # Missing provider
            },
            schema=schema
        )
        
        assert not result["valid"]
    
    @pytest.mark.asyncio
    async def test_lenient_validation(self, validation_tool):
        """Test lenient validation mode."""
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "rate": {"type": "number"},
                "active": {"type": "boolean"}
            }
        }
        
        # Data with coercible types
        result = await validation_tool.execute(
            action="validate",
            data={
                "count": "10",  # String to int
                "rate": "3.14",  # String to float
                "active": "true"  # String to bool
            },
            schema=schema,
            mode="lenient"
        )
        
        assert result["valid"]
        assert len(result["warnings"]) == 3  # All values were coerced
        
        # Check coercion in warnings
        for warning in result["warnings"]:
            assert "coerced_to" in warning


if __name__ == "__main__":
    pytest.main([__file__, "-v"])