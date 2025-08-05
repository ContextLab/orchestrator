"""Executes the constructed prompt to resolve AUTO tag."""

import json
from typing import Any, Dict, Optional

from .models import (
    AutoTagContext,
    RequirementsAnalysis,
    PromptConstruction,
    ParseError,
    ValidationError
)
from .structured_output import StructuredOutputParser


class ResolutionExecutor:
    """Executes the constructed prompt to resolve AUTO tag."""
    
    def __init__(
        self, 
        model_registry: Optional[Any] = None,
        structured_parser: Optional[StructuredOutputParser] = None
    ):
        self.model_registry = model_registry
        self.parser = structured_parser or StructuredOutputParser()
    
    async def execute(
        self,
        prompt_data: PromptConstruction,
        context: AutoTagContext,
        requirements: RequirementsAnalysis
    ) -> Any:
        """Execute prompt and return resolved value.
        
        Args:
            prompt_data: Constructed prompt with metadata
            context: Full resolution context
            requirements: Requirements from analysis
            
        Returns:
            Resolved value of appropriate type
            
        Raises:
            ParseError: If response parsing fails
            ValidationError: If response doesn't match expected type
        """
        # Determine which model to use
        target_model = prompt_data.target_model
        if not target_model:
            # Use model from pipeline metadata or default
            target_model = context.pipeline.metadata.get("model", "gpt-4o-mini")
        
        # Execute based on whether we need structured output
        if prompt_data.output_schema:
            response = await self._execute_with_schema(
                target_model,
                prompt_data.prompt,
                prompt_data.output_schema,
                prompt_data.system_prompt,
                prompt_data.temperature,
                prompt_data.max_tokens
            )
        else:
            response = await self._execute_regular(
                target_model,
                prompt_data.prompt,
                prompt_data.system_prompt,
                prompt_data.temperature,
                prompt_data.max_tokens
            )
        
        # Parse response based on expected type
        parsed_value = self._parse_response(
            response,
            requirements.expected_output_type,
            requirements.output_format
        )
        
        # Validate response
        if not self._validate_response(parsed_value, requirements):
            raise ValidationError(
                f"Response does not match requirements. "
                f"Expected type: {requirements.expected_output_type}, "
                f"Got: {type(parsed_value).__name__}"
            )
        
        return parsed_value
    
    async def _execute_with_schema(
        self,
        model: str,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int]
    ) -> Any:
        """Execute prompt with structured output schema."""
        # Add schema instructions to prompt
        schema_prompt = self.parser.create_schema_prompt(schema)
        full_prompt = f"{prompt}\n\n{schema_prompt}"
        
        # Call model
        response = await self._call_model(
            model,
            full_prompt,
            system_prompt,
            temperature,
            max_tokens
        )
        
        # Parse structured response
        return self.parser.parse_json_response(response, schema)
    
    async def _execute_regular(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int]
    ) -> str:
        """Execute regular prompt without structured output."""
        return await self._call_model(
            model,
            prompt,
            system_prompt,
            temperature,
            max_tokens
        )
    
    async def _call_model(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int]
    ) -> str:
        """Call the model with given parameters using real API."""
        from .model_caller import ModelCaller
        
        model_caller = ModelCaller()
        
        # Use JSON mode for structured outputs when using GPT models
        json_mode = "json" in prompt.lower() or "schema" in prompt.lower()
        
        return await model_caller.call_model(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode if model.startswith("gpt") else False
        )
    
    def _parse_response(
        self,
        response: Any,
        expected_type: str,
        output_format: Optional[str]
    ) -> Any:
        """Parse response based on expected type and format."""
        # If already parsed (from structured output), validate and return
        if isinstance(response, dict) and expected_type == "object":
            return response
        elif isinstance(response, list) and expected_type == "array":
            return response
        
        # Parse string responses based on format
        if isinstance(response, str):
            if output_format == "json" or expected_type in ["object", "array"]:
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    # Try to extract JSON from response
                    extracted = self.parser._extract_json(response)
                    return json.loads(extracted)
            
            elif expected_type == "number":
                # Try to parse as number
                try:
                    # Try float first
                    value = float(response.strip())
                    # If it's a whole number, return as int
                    if value.is_integer():
                        return int(value)
                    return value
                except ValueError:
                    raise ParseError(f"Expected number, got: {response}")
            
            elif expected_type == "boolean":
                # Parse boolean
                cleaned = response.strip().lower()
                if cleaned in ["true", "yes", "1", "on"]:
                    return True
                elif cleaned in ["false", "no", "0", "off"]:
                    return False
                else:
                    raise ParseError(f"Expected boolean, got: {response}")
            
            elif expected_type == "null":
                # Check for null
                cleaned = response.strip().lower()
                if cleaned in ["null", "none", "nil", ""]:
                    return None
                else:
                    raise ParseError(f"Expected null, got: {response}")
        
        # Default: return as-is
        return response
    
    def _validate_response(
        self,
        value: Any,
        requirements: RequirementsAnalysis
    ) -> bool:
        """Validate response against requirements."""
        # Check type
        if not self.parser.validate_response(value, requirements.expected_output_type):
            return False
        
        # Check constraints
        if requirements.constraints:
            # Length constraints
            if "max_length" in requirements.constraints and isinstance(value, str):
                if len(value) > requirements.constraints["max_length"]:
                    return False
            
            if "min_length" in requirements.constraints and isinstance(value, str):
                if len(value) < requirements.constraints["min_length"]:
                    return False
            
            # Array constraints
            if "max_items" in requirements.constraints and isinstance(value, list):
                if len(value) > requirements.constraints["max_items"]:
                    return False
            
            if "min_items" in requirements.constraints and isinstance(value, list):
                if len(value) < requirements.constraints["min_items"]:
                    return False
        
        return True