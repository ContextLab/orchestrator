"""Constructs prompts based on requirements and context."""

import json
from typing import Any, Dict, List, Optional

from .models import (
    AutoTagContext, 
    RequirementsAnalysis, 
    PromptConstruction,
    ParseError
)
from .structured_output import StructuredOutputParser


class PromptConstructor:
    """Constructs prompts based on requirements and context."""
    
    PROMPT_SCHEMA = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The constructed prompt to fulfill the AUTO tag"
            },
            "target_model": {
                "type": "string",
                "description": "Specific model to execute this prompt (optional)"
            },
            "temperature": {
                "type": "number",
                "minimum": 0,
                "maximum": 2,
                "description": "Temperature for generation (0-2)"
            },
            "max_tokens": {
                "type": "integer",
                "minimum": 1,
                "description": "Maximum tokens to generate"
            },
            "output_schema": {
                "type": "object",
                "description": "JSON schema for structured output (if needed)"
            },
            "system_prompt": {
                "type": "string",
                "description": "System prompt to set context (optional)"
            }
        },
        "required": ["prompt"]
    }
    
    def __init__(self, structured_parser: Optional[StructuredOutputParser] = None):
        self.parser = structured_parser or StructuredOutputParser()
    
    async def construct(
        self,
        auto_tag: str,
        context: AutoTagContext,
        requirements: RequirementsAnalysis,
        model: str
    ) -> PromptConstruction:
        """Construct prompt based on requirements.
        
        Args:
            auto_tag: Original AUTO tag content
            context: Full resolution context
            requirements: Requirements from analysis phase
            model: Model to use for construction
            
        Returns:
            Constructed prompt with metadata
            
        Raises:
            ParseError: If prompt construction fails
        """
        # Resolve context references
        resolved_context = self._resolve_context_references(
            auto_tag, 
            context, 
            requirements.data_dependencies
        )
        
        # Build construction prompt
        construction_prompt = self._build_construction_prompt(
            auto_tag, 
            context, 
            requirements, 
            resolved_context
        )
        
        try:
            # Call model to construct prompt
            response = await self._call_model_with_schema(
                model,
                construction_prompt,
                self.PROMPT_SCHEMA
            )
            
            # Validate response has required fields
            if "prompt" not in response:
                raise ParseError(f"Model response missing 'prompt' field. Got: {response}")
            
            # Create PromptConstruction object
            prompt_construction = PromptConstruction(
                prompt=response["prompt"],
                target_model=response.get("target_model"),
                temperature=response.get("temperature", 0.7),
                max_tokens=response.get("max_tokens"),
                output_schema=response.get("output_schema"),
                system_prompt=response.get("system_prompt"),
                resolved_context=resolved_context,
                model_used=model
            )
            
            return prompt_construction
            
        except Exception as e:
            raise ParseError(f"Failed to construct prompt: {e}")
    
    def _resolve_context_references(
        self,
        auto_tag: str,
        context: AutoTagContext,
        dependencies: List[str]
    ) -> Dict[str, Any]:
        """Resolve context variables referenced in AUTO tag."""
        resolved = {}
        
        # Get full context
        full_context = context.get_full_context()
        
        # Resolve each dependency
        for dep in dependencies:
            value = self._resolve_reference(dep, full_context)
            if value is not None:
                resolved[dep] = value
        
        # Also include any template variables in the AUTO tag
        import re
        template_pattern = r'\{\{\s*([^}]+)\s*\}\}'
        matches = re.findall(template_pattern, auto_tag)
        
        for match in matches:
            # Parse the reference
            parts = match.strip().split('.')
            base = parts[0]
            
            # Resolve the full reference
            value = self._resolve_reference(match.strip(), full_context)
            if value is not None:
                resolved[match.strip()] = value
        
        return resolved
    
    def _resolve_reference(self, reference: str, context: Dict[str, Any]) -> Any:
        """Resolve a single reference from context."""
        # Split reference into parts
        parts = reference.split('.')
        
        # Start with the base
        current = None
        
        # Check in different context sections
        if parts[0] in context.get("variables", {}):
            current = context["variables"][parts[0]]
        elif parts[0] in context.get("step_results", {}):
            current = context["step_results"][parts[0]]
        elif parts[0] in context.get("loop", {}):
            current = context["loop"][parts[0]]
        else:
            return None
        
        # Navigate through nested attributes
        for part in parts[1:]:
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif hasattr(current, part):
                current = getattr(current, part)
            elif part.isdigit() and isinstance(current, (list, tuple)):
                idx = int(part)
                if 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return None
            else:
                return None
        
        return current
    
    def _build_construction_prompt(
        self,
        auto_tag: str,
        context: AutoTagContext,
        requirements: RequirementsAnalysis,
        resolved_context: Dict[str, Any]
    ) -> str:
        """Build prompt for prompt construction."""
        
        # Format requirements
        req_summary = {
            "tools_needed": requirements.tools_needed,
            "output_format": requirements.output_format,
            "expected_type": requirements.expected_output_type,
            "constraints": requirements.constraints,
            "model_requirements": requirements.model_requirements
        }
        
        prompt = f"""Construct a prompt to fulfill this AUTO tag based on the requirements and available context.

Original AUTO tag: {auto_tag}

Requirements Analysis:
{json.dumps(req_summary, indent=2)}

Available Context Data:
{json.dumps(resolved_context, indent=2)}

Task Location: {context.tag_location}
Current Task: {context.current_task_id}

Instructions for Prompt Construction:

1. **Incorporate Context**: 
   - Replace all {{{{ variable }}}} references with actual values from context
   - Ensure all necessary data is included in the prompt
   - Make the prompt self-contained

2. **Specify Output Format**:
   - If output_format is specified, include clear instructions
   - For structured outputs (json, yaml), provide examples
   - Be explicit about format requirements

3. **Apply Constraints**:
   - Include any constraints from requirements
   - Add length limits, content restrictions, etc.
   - Make constraints clear and enforceable

4. **Optimize for Model**:
   - Choose appropriate target_model if specific capabilities needed
   - Set temperature based on task (lower for factual, higher for creative)
   - Estimate max_tokens needed
   - Create output_schema if structured output required

5. **System Prompt** (if needed):
   - Add system prompt for specific behaviors
   - Use for role-setting or consistent formatting
   - Keep concise and focused

Create a prompt that is:
- Clear and unambiguous
- Self-contained with all necessary context
- Specific about expected output
- Appropriately detailed for the task

Remember: The prompt will be sent to an LLM to generate the final result."""
        
        return prompt
    
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