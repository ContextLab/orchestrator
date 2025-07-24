"""AUTO tag schema inference for intelligent validation."""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..core.model import Model
from ..tools.validation import SchemaState


@dataclass
class InferenceContext:
    """Context for schema inference from AUTO tags."""

    auto_tag_content: str
    task_id: str
    predecessors: List[Dict[str, Any]]  # List of {task_id, output_schema}
    successors: List[Dict[str, Any]]  # List of {task_id, input_schema}
    pipeline_context: Dict[str, Any]
    tool_registry: Optional[Any] = None
    model: Optional[Model] = None


class AutoTagSchemaInferer:
    """Infers JSON schemas from AUTO tags using context analysis."""

    def __init__(self, model: Optional[Model] = None):
        self.model = model
        self.inference_cache: Dict[str, Dict[str, Any]] = {}

        # Keywords that suggest specific schema structures
        self.structure_keywords = {
            "extract": ["object", "properties"],
            "list": ["array", "items"],
            "summarize": ["string"],
            "analyze": ["object", "properties"],
            "generate": ["string", "object"],
            "process": ["any"],
            "transform": ["any"],
            "validate": ["boolean", "object"],
            "classify": ["string", "enum"],
            "categorize": ["string", "enum"],
            "count": ["integer"],
            "calculate": ["number"],
            "measure": ["number"],
        }

        # Field indicators
        self.field_indicators = {
            "email": {"type": "string", "format": "email"},
            "url": {"type": "string", "format": "uri"},
            "link": {"type": "string", "format": "uri"},
            "date": {"type": "string", "format": "date"},
            "time": {"type": "string", "format": "time"},
            "datetime": {"type": "string", "format": "date-time"},
            "phone": {"type": "string", "format": "phone"},
            "number": {"type": "number"},
            "count": {"type": "integer"},
            "amount": {"type": "number"},
            "price": {"type": "number"},
            "percentage": {"type": "number", "minimum": 0, "maximum": 100},
            "age": {"type": "integer", "minimum": 0},
            "name": {"type": "string"},
            "title": {"type": "string"},
            "description": {"type": "string"},
            "summary": {"type": "string"},
            "id": {"type": "string"},
            "identifier": {"type": "string"},
            "code": {"type": "string"},
            "status": {"type": "string"},
            "flag": {"type": "boolean"},
            "enabled": {"type": "boolean"},
            "active": {"type": "boolean"},
        }

    async def infer_schema(
        self, context: InferenceContext
    ) -> Tuple[Optional[Dict], Optional[Dict], SchemaState]:
        """
        Infer input and output schemas from AUTO tag.

        Returns:
            Tuple of (input_schema, output_schema, schema_state)
        """
        # Check cache
        cache_key = f"{context.task_id}:{context.auto_tag_content}"
        if cache_key in self.inference_cache:
            cached = self.inference_cache[cache_key]
            return cached["input"], cached["output"], cached["state"]

        # Parse AUTO tag content
        content = context.auto_tag_content.strip()

        # Try pattern-based inference first
        input_schema, output_schema, state = self._pattern_based_inference(
            content, context
        )

        # If still ambiguous and model available, use AI inference
        if state == SchemaState.AMBIGUOUS and self.model:
            input_schema, output_schema, state = await self._ai_based_inference(
                content, context, input_schema, output_schema
            )

        # Cache result
        self.inference_cache[cache_key] = {
            "input": input_schema,
            "output": output_schema,
            "state": state,
        }

        return input_schema, output_schema, state

    def _pattern_based_inference(
        self, content: str, context: InferenceContext
    ) -> Tuple[Optional[Dict], Optional[Dict], SchemaState]:
        """Infer schema using pattern matching and heuristics."""
        content_lower = content.lower()

        # Determine primary action
        primary_action = None
        for action, indicators in self.structure_keywords.items():
            if action in content_lower:
                primary_action = action
                break

        # Extract mentioned fields
        mentioned_fields = self._extract_mentioned_fields(content)

        # Build output schema based on action and fields
        output_schema = self._build_output_schema(
            primary_action, mentioned_fields, content
        )

        # Infer input schema from predecessors and content
        input_schema = self._build_input_schema(context, content)

        # Determine state
        state = SchemaState.FIXED
        if not output_schema or not input_schema:
            state = SchemaState.AMBIGUOUS
        elif self._has_dynamic_references(content):
            state = SchemaState.PARTIAL

        return input_schema, output_schema, state

    def _extract_mentioned_fields(self, content: str) -> List[Tuple[str, Dict]]:
        """Extract field names and their likely schemas from content."""
        fields = []

        # Look for explicit field mentions
        # Pattern: "extract/include/with {field1}, {field2}, and {field3}"
        field_pattern = r"(?:extract|include|with|including)\s+([^.!?]+)"
        matches = re.findall(field_pattern, content.lower())

        for match in matches:
            # Split by commas and "and"
            parts = re.split(r"[,\s]+and\s+|,\s*", match)
            for part in parts:
                part = part.strip()
                # Check against known field indicators
                for field_name, field_schema in self.field_indicators.items():
                    if field_name in part:
                        fields.append((field_name, field_schema.copy()))
                        break
                else:
                    # Generic string field
                    if part and len(part) < 50:  # Reasonable field name length
                        clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", part)
                        fields.append((clean_name, {"type": "string"}))

        return fields

    def _build_output_schema(
        self, action: Optional[str], fields: List[Tuple[str, Dict]], content: str
    ) -> Optional[Dict[str, Any]]:
        """Build output schema based on action and fields."""
        if not action:
            action = "process"  # Default action

        # Get base structure from action
        if action in ["extract", "analyze", "generate"]:
            if fields:
                # Object with properties
                properties = {name: schema for name, schema in fields}
                return {
                    "type": "object",
                    "properties": properties,
                    "required": [
                        name for name, _ in fields[:3]
                    ],  # First 3 fields required
                }
            else:
                # Generic object
                return {"type": "object"}

        elif action in ["list"]:
            # Array structure
            if fields:
                # Array of objects
                properties = {name: schema for name, schema in fields}
                return {
                    "type": "array",
                    "items": {"type": "object", "properties": properties},
                }
            else:
                return {"type": "array", "items": {"type": "string"}}

        elif action in ["summarize"]:
            return {"type": "string", "minLength": 10}

        elif action in ["count", "calculate", "measure"]:
            return {"type": "number"}

        elif action in ["classify", "categorize"]:
            # Try to extract categories
            categories = self._extract_categories(content)
            if categories:
                return {"type": "string", "enum": categories}
            else:
                return {"type": "string"}

        elif action in ["validate"]:
            return {
                "type": "object",
                "properties": {
                    "valid": {"type": "boolean"},
                    "errors": {"type": "array", "items": {"type": "string"}},
                    "warnings": {"type": "array", "items": {"type": "string"}},
                },
            }

        # Default
        return None

    def _build_input_schema(
        self, context: InferenceContext, content: str
    ) -> Optional[Dict[str, Any]]:
        """Build input schema from context and content analysis."""
        # If single predecessor, likely uses its output
        if len(context.predecessors) == 1:
            pred = context.predecessors[0]
            if pred.get("output_schema"):
                return pred["output_schema"]

        # Check for template variables in content
        template_vars = re.findall(r"\{\{([^}]+)\}\}", content)
        if template_vars:
            properties = {}
            for var in template_vars:
                var_name = var.strip().split(".")[0]  # Get root variable
                if var_name in context.pipeline_context:
                    # Infer type from context value
                    value = context.pipeline_context[var_name]
                    properties[var_name] = self._infer_type_from_value(value)
                else:
                    # Default to string
                    properties[var_name] = {"type": "string"}

            if properties:
                return {"type": "object", "properties": properties}

        # Multiple predecessors
        if len(context.predecessors) > 1:
            properties = {}
            for pred in context.predecessors:
                if pred.get("output_schema"):
                    properties[pred["task_id"]] = pred["output_schema"]

            if properties:
                return {"type": "object", "properties": properties}

        return None

    def _extract_categories(self, content: str) -> Optional[List[str]]:
        """Extract category values from content."""
        # Look for patterns like "classify as X, Y, or Z"
        pattern = r"(?:classify|categorize)\s+(?:as|into)\s+([^.!?]+)"
        match = re.search(pattern, content.lower())

        if match:
            categories_text = match.group(1)
            # Split by commas, "or", "and"
            categories = re.split(r"[,\s]+(?:or|and)\s+|,\s*", categories_text)
            return [cat.strip().strip("\"'") for cat in categories if cat.strip()]

        return None

    def _has_dynamic_references(self, content: str) -> bool:
        """Check if content has dynamic references that can't be resolved at compile time."""
        # Check for runtime template variables
        if re.search(r"\{\{[^}]+\}\}", content):
            return True

        # Check for conditional language
        conditional_keywords = [
            "if",
            "when",
            "based on",
            "depending on",
            "according to",
        ]
        content_lower = content.lower()
        for keyword in conditional_keywords:
            if keyword in content_lower:
                return True

        return False

    def _infer_type_from_value(self, value: Any) -> Dict[str, str]:
        """Infer JSON Schema type from a Python value."""
        if isinstance(value, bool):
            return {"type": "boolean"}
        elif isinstance(value, int):
            return {"type": "integer"}
        elif isinstance(value, float):
            return {"type": "number"}
        elif isinstance(value, str):
            return {"type": "string"}
        elif isinstance(value, list):
            return {"type": "array"}
        elif isinstance(value, dict):
            return {"type": "object"}
        else:
            return {"type": "string"}  # Default

    async def _ai_based_inference(
        self,
        content: str,
        context: InferenceContext,
        partial_input_schema: Optional[Dict],
        partial_output_schema: Optional[Dict],
    ) -> Tuple[Optional[Dict], Optional[Dict], SchemaState]:
        """Use AI model to infer schemas from AUTO tag."""
        if not self.model:
            return partial_input_schema, partial_output_schema, SchemaState.AMBIGUOUS

        # Build prompt for schema inference
        prompt = self._build_inference_prompt(
            content, context, partial_input_schema, partial_output_schema
        )

        try:
            # Call model for inference
            response = await self.model.generate(prompt)

            # Parse response to extract schemas
            schemas = self._parse_ai_response(response)

            if schemas:
                return (
                    schemas.get("input_schema", partial_input_schema),
                    schemas.get("output_schema", partial_output_schema),
                    SchemaState.PARTIAL,  # AI inference is not guaranteed to be complete
                )
        except Exception as e:
            # Log error and return partial schemas
            print(f"AI schema inference failed: {e}")

        return partial_input_schema, partial_output_schema, SchemaState.AMBIGUOUS

    def _build_inference_prompt(
        self,
        content: str,
        context: InferenceContext,
        partial_input: Optional[Dict],
        partial_output: Optional[Dict],
    ) -> str:
        """Build prompt for AI-based schema inference."""
        prompt = f"""Given an AUTO tag in a data pipeline, infer the JSON schemas for input and output.

AUTO tag content: {content}

Context:
- Task ID: {context.task_id}
- Number of predecessors: {len(context.predecessors)}
- Number of successors: {len(context.successors)}

Partial schemas already inferred:
- Input schema: {partial_input}
- Output schema: {partial_output}

Please provide refined JSON schemas for both input and output that would be appropriate for this operation.
Return the result as a JSON object with "input_schema" and "output_schema" keys.
"""
        return prompt

    def _parse_ai_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse AI response to extract schemas."""
        try:
            # Try to extract JSON from response
            import json

            # Look for JSON block
            json_match = re.search(r"\{[\s\S]+\}", response)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass

        return None
