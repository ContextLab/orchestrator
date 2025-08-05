"""Structured output parsing for AUTO tag resolution."""

import json
from typing import Any, Dict, Optional, Type
from jsonschema import validate, ValidationError as JsonSchemaError

from .models import ParseError


class StructuredOutputParser:
    """Handles structured output parsing with schema validation."""
    
    def __init__(self):
        """Initialize parser."""
        pass
    
    def parse_json_response(self, response: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Parse JSON response and validate against schema.
        
        Args:
            response: Raw response string
            schema: JSON schema to validate against
            
        Returns:
            Parsed and validated data
            
        Raises:
            ParseError: If parsing or validation fails
        """
        # Try to extract JSON from response
        json_str = self._extract_json(response)
        
        # Parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ParseError(f"Failed to parse JSON: {e}")
        
        # Validate against schema
        try:
            validate(instance=data, schema=schema)
        except JsonSchemaError as e:
            raise ParseError(f"Schema validation failed: {e}")
        
        return data
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that may contain other content.
        
        Handles cases where JSON is embedded in markdown code blocks or
        surrounded by other text.
        """
        # First, try the whole text
        text = text.strip()
        if text.startswith('{') or text.startswith('['):
            return text
        
        # Look for JSON in code blocks
        import re
        
        # Try ```json blocks
        json_block_pattern = r'```json\s*([\s\S]*?)\s*```'
        match = re.search(json_block_pattern, text)
        if match:
            return match.group(1)
        
        # Try generic ``` blocks
        code_block_pattern = r'```\s*([\s\S]*?)\s*```'
        match = re.search(code_block_pattern, text)
        if match:
            potential_json = match.group(1)
            if potential_json.strip().startswith(('{', '[')):
                return potential_json
        
        # Look for JSON-like content
        json_pattern = r'(\{[\s\S]*\}|\[[\s\S]*\])'
        matches = re.findall(json_pattern, text)
        
        # Try each match to see if it's valid JSON
        for match in matches:
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue
        
        # If no JSON found, raise error
        raise ParseError("No valid JSON found in response")
    
    def create_schema_prompt(self, schema: Dict[str, Any]) -> str:
        """Create a prompt snippet explaining the expected schema.
        
        Args:
            schema: JSON schema
            
        Returns:
            Human-readable schema explanation
        """
        prompt_parts = ["Respond with a JSON object matching this schema:"]
        
        # Add schema
        prompt_parts.append("```json")
        prompt_parts.append(json.dumps(schema, indent=2))
        prompt_parts.append("```")
        
        # Add examples if possible
        example = self._generate_example(schema)
        if example:
            prompt_parts.append("\nExample response:")
            prompt_parts.append("```json")
            prompt_parts.append(json.dumps(example, indent=2))
            prompt_parts.append("```")
        
        return "\n".join(prompt_parts)
    
    def _generate_example(self, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate an example that matches the schema."""
        if schema.get("type") != "object":
            return None
        
        example = {}
        properties = schema.get("properties", {})
        
        for prop_name, prop_schema in properties.items():
            prop_type = prop_schema.get("type")
            
            if prop_type == "string":
                if "enum" in prop_schema:
                    example[prop_name] = prop_schema["enum"][0]
                else:
                    example[prop_name] = f"example_{prop_name}"
            elif prop_type == "number":
                example[prop_name] = 0.0
            elif prop_type == "integer":
                example[prop_name] = 0
            elif prop_type == "boolean":
                example[prop_name] = True
            elif prop_type == "array":
                example[prop_name] = ["example_item"]
            elif prop_type == "object":
                example[prop_name] = {"key": "value"}
            elif prop_type == "null":
                example[prop_name] = None
        
        return example
    
    def validate_response(self, data: Any, expected_type: str) -> bool:
        """Validate that response matches expected type.
        
        Args:
            data: Response data
            expected_type: Expected type (string, number, boolean, object, array, null)
            
        Returns:
            True if valid, False otherwise
        """
        type_map = {
            "string": str,
            "number": (int, float),
            "boolean": bool,
            "object": dict,
            "array": list,
            "null": type(None)
        }
        
        expected_python_type = type_map.get(expected_type)
        if expected_python_type is None:
            return True  # Unknown type, assume valid
        
        return isinstance(data, expected_python_type)