"""Determines what to do with resolved AUTO tag value."""

import json
from typing import Any, Dict, Optional

from .models import (
    AutoTagContext,
    RequirementsAnalysis,
    ActionPlan,
    ParseError
)
from .structured_output import StructuredOutputParser


class ActionDeterminer:
    """Determines what to do with resolved AUTO tag value."""
    
    ACTION_SCHEMA = {
        "type": "object",
        "properties": {
            "action_type": {
                "type": "string",
                "enum": ["return_value", "call_tool", "save_file", "update_context", "chain_resolution"],
                "description": "What to do with the resolved value"
            },
            "tool_name": {
                "type": "string",
                "description": "Name of tool to call (if action_type is call_tool)"
            },
            "tool_parameters": {
                "type": "object",
                "description": "Parameters for tool call"
            },
            "file_path": {
                "type": "string",
                "description": "Path to save file (if action_type is save_file)"
            },
            "context_updates": {
                "type": "object",
                "description": "Context variables to update (if action_type is update_context)"
            },
            "next_auto_tag": {
                "type": "string",
                "description": "Next AUTO tag to resolve (if action_type is chain_resolution)"
            }
        },
        "required": ["action_type"]
    }
    
    def __init__(self, structured_parser: Optional[StructuredOutputParser] = None):
        self.parser = structured_parser or StructuredOutputParser()
    
    async def determine(
        self,
        resolved_value: Any,
        requirements: RequirementsAnalysis,
        context: AutoTagContext,
        model: str
    ) -> ActionPlan:
        """Determine action plan for resolved value.
        
        Args:
            resolved_value: The resolved AUTO tag value
            requirements: Requirements from analysis
            context: Full resolution context
            model: Model to use for determination
            
        Returns:
            Action plan for handling the value
            
        Raises:
            ParseError: If action determination fails
        """
        # Build determination prompt
        determination_prompt = self._build_determination_prompt(
            resolved_value,
            requirements,
            context
        )
        
        try:
            # Call model to determine action
            response = await self._call_model_with_schema(
                model,
                determination_prompt,
                self.ACTION_SCHEMA
            )
            
            # Create ActionPlan object
            action_plan = ActionPlan(
                action_type=response["action_type"],
                tool_name=response.get("tool_name"),
                tool_parameters=response.get("tool_parameters"),
                file_path=response.get("file_path"),
                context_updates=response.get("context_updates"),
                next_auto_tag=response.get("next_auto_tag"),
                model_used=model
            )
            
            return action_plan
            
        except Exception as e:
            raise ParseError(f"Failed to determine action: {e}")
    
    def _build_determination_prompt(
        self,
        resolved_value: Any,
        requirements: RequirementsAnalysis,
        context: AutoTagContext
    ) -> str:
        """Build prompt for action determination."""
        # Get value preview
        value_preview = self._get_value_preview(resolved_value)
        
        # Get tag location info
        location_parts = context.tag_location.split('.')
        field_name = location_parts[-1] if location_parts else "unknown"
        
        prompt = f"""Determine what to do with this resolved AUTO tag value.

Tag Location: {context.tag_location}
Field Name: {field_name}
Current Task: {context.current_task_id}

Resolved Value Type: {type(resolved_value).__name__}
Resolved Value Preview: {value_preview}

Requirements:
- Tools Needed: {requirements.tools_needed}
- Output Format: {requirements.output_format}
- Expected Type: {requirements.expected_output_type}

Context:
- Is this in a task parameter? {'.parameters.' in context.tag_location}
- Is this a task action? {'action' in context.tag_location}
- Is this a tool specification? {'tool' in context.tag_location}
- Is this a file location? {'location' in context.tag_location or 'produces' in context.tag_location}

Determine the appropriate action:

1. **return_value**: Simply return the value to be used in place of the AUTO tag
   - Use when the value should directly replace the AUTO tag
   - Most common for parameters, actions, and simple replacements

2. **call_tool**: Pass the value to a tool for processing
   - Use when tools_needed specified a tool
   - Use when the value needs further processing
   - Specify tool_name and tool_parameters

3. **save_file**: Save the value to a file
   - Use when the location suggests file output
   - Use when produces indicates a file type
   - Specify file_path

4. **update_context**: Update pipeline context with the value
   - Use when the value should be available to future steps
   - Use for intermediate results that need persistence
   - Specify context_updates as key-value pairs

5. **chain_resolution**: Trigger resolution of another AUTO tag
   - Use when the resolved value contains another AUTO tag
   - Use for multi-step resolution processes
   - Specify next_auto_tag

Choose the most appropriate action based on:
- The tag's location in the pipeline
- The requirements analysis
- The nature of the resolved value
- Standard pipeline patterns"""
        
        return prompt
    
    def _get_value_preview(self, value: Any) -> str:
        """Get a safe preview of the resolved value."""
        try:
            if isinstance(value, str):
                # Show first 200 chars
                if len(value) > 200:
                    return value[:200] + "..."
                return value
            elif isinstance(value, (dict, list)):
                # Show structure with limited depth
                preview = json.dumps(value, indent=2)
                if len(preview) > 500:
                    # Show just the structure
                    if isinstance(value, dict):
                        keys = list(value.keys())[:5]
                        return f"Dict with {len(value)} keys: {keys}..."
                    else:
                        return f"List with {len(value)} items"
                return preview
            elif isinstance(value, (int, float, bool)):
                return str(value)
            elif value is None:
                return "null"
            else:
                return f"<{type(value).__name__} object>"
        except Exception:
            return "<unprintable value>"
    
    async def _call_model_with_schema(
        self,
        model: str,
        prompt: str,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call model with structured output schema using real API."""
        from .model_caller import ModelCaller
        
        model_caller = ModelCaller()
        
        # Add schema to prompt
        schema_prompt = self.parser.create_schema_prompt(schema)
        full_prompt = f"{prompt}\n\n{schema_prompt}"
        
        # Call real model
        response = await model_caller.call_model(
            model=model,
            prompt=full_prompt,
            temperature=0.3,
            json_mode=True if model.startswith("gpt") else False
        )
        
        # Parse response
        return self.parser.parse_json_response(response, schema)