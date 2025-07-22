"""Test structured extraction in ValidationTool."""

import pytest
from orchestrator.tools.validation import ValidationTool


class TestStructuredExtraction:
    """Test the extract_structured action."""
    
    @pytest.fixture
    def validation_tool(self):
        """Create ValidationTool instance."""
        return ValidationTool()
    
    @pytest.mark.asyncio
    async def test_extract_structured_no_model(self, validation_tool):
        """Test extraction without a model."""
        result = await validation_tool.execute(
            action="extract_structured",
            text="Some text",
            schema={"type": "object", "properties": {"name": {"type": "string"}}}
        )
        
        assert not result["success"]
        assert "No model available" in result["error"]
    
    @pytest.mark.asyncio
    async def test_extract_structured_no_text(self, validation_tool):
        """Test extraction without text."""
        result = await validation_tool.execute(
            action="extract_structured",
            schema={"type": "object"}
        )
        
        assert not result["success"]
        assert "No text provided" in result["error"]
    
    @pytest.mark.asyncio
    async def test_extract_structured_no_schema(self, validation_tool):
        """Test extraction without schema."""
        result = await validation_tool.execute(
            action="extract_structured",
            text="Some text"
        )
        
        assert not result["success"]
        assert "No schema provided" in result["error"]
    
    @pytest.mark.asyncio
    async def test_pydantic_conversion(self, validation_tool):
        """Test JSON Schema to Pydantic conversion."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person's name"},
                "age": {"type": "integer", "description": "Person's age"},
                "email": {"type": "string", "format": "email"},
                "active": {"type": "boolean"},
                "scores": {"type": "array", "items": {"type": "number"}},
                "metadata": {"type": "object"}
            },
            "required": ["name", "email"]
        }
        
        # Convert to Pydantic
        model = validation_tool._json_schema_to_pydantic(schema)
        
        # Check model name
        assert model.__name__ == "ExtractedData"
        
        # Check fields
        fields = model.model_fields
        assert "name" in fields
        assert "age" in fields
        assert "email" in fields
        
        # Check required fields
        assert fields["name"].is_required()
        assert fields["email"].is_required()
        assert not fields["age"].is_required()
        
        # Test creating instance
        instance = model(
            name="John Doe",
            email="john@example.com",
            age=30,
            active=True,
            scores=[95.5, 87.0],
            metadata={"role": "admin"}
        )
        
        assert instance.name == "John Doe"
        assert instance.email == "john@example.com"
        assert instance.age == 30
        assert instance.active is True
        assert instance.scores == [95.5, 87.0]
        assert instance.metadata == {"role": "admin"}
    
    @pytest.mark.asyncio
    async def test_pydantic_conversion_non_object(self, validation_tool):
        """Test conversion with non-object schema."""
        schema = {"type": "array", "items": {"type": "string"}}
        
        with pytest.raises(ValueError, match="Only object schemas are supported"):
            validation_tool._json_schema_to_pydantic(schema)
    
    @pytest.mark.asyncio
    async def test_python_type_conversion(self, validation_tool):
        """Test JSON Schema type to Python type conversion."""
        # String
        assert validation_tool._get_python_type({"type": "string"}) == str
        
        # Integer
        assert validation_tool._get_python_type({"type": "integer"}) == int
        
        # Number
        assert validation_tool._get_python_type({"type": "number"}) == float
        
        # Boolean
        assert validation_tool._get_python_type({"type": "boolean"}) == bool
        
        # Array
        list_type = validation_tool._get_python_type({"type": "array", "items": {"type": "string"}})
        assert list_type.__origin__ == list
        assert list_type.__args__[0] == str
        
        # Object
        dict_type = validation_tool._get_python_type({"type": "object"})
        assert dict_type.__origin__ == dict
        
        # Unknown type
        from typing import Any
        assert validation_tool._get_python_type({"type": "unknown"}) == Any