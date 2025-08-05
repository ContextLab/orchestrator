"""Analyzes AUTO tags to determine requirements using real LLM calls."""

import json
import logging
from typing import Any, Dict, List, Optional

from .models import AutoTagContext, RequirementsAnalysis, ParseError
from .structured_output import StructuredOutputParser
from .model_caller import ModelCaller

logger = logging.getLogger(__name__)


class RequirementsAnalyzer:
    """Analyzes AUTO tag to determine requirements."""
    
    REQUIREMENTS_SCHEMA = {
        "type": "object",
        "properties": {
            "tools_needed": {
                "type": "array", 
                "items": {"type": "string"},
                "description": "List of tools needed to accomplish the task"
            },
            "output_format": {
                "type": "string",
                "description": "Expected output format (e.g., 'json', 'markdown', 'text', 'yaml')"
            },
            "constraints": {
                "type": "object",
                "description": "Any constraints or requirements for the task"
            },
            "data_dependencies": {
                "type": "array", 
                "items": {"type": "string"},
                "description": "List of context variables or step results needed"
            },
            "model_requirements": {
                "type": "object",
                "properties": {
                    "capabilities": {"type": "array", "items": {"type": "string"}},
                    "min_context_length": {"type": "integer"},
                    "requires_vision": {"type": "boolean"},
                    "requires_function_calling": {"type": "boolean"}
                },
                "description": "Specific model requirements"
            },
            "expected_output_type": {
                "type": "string",
                "enum": ["string", "number", "boolean", "object", "array", "null"],
                "description": "Expected type of the resolved value"
            }
        },
        "required": ["expected_output_type"]
    }
    
    def __init__(
        self, 
        structured_parser: Optional[StructuredOutputParser] = None,
        model_caller: Optional[ModelCaller] = None
    ):
        self.parser = structured_parser or StructuredOutputParser()
        self.model_caller = model_caller or ModelCaller()
    
    async def analyze(
        self,
        auto_tag: str,
        context: AutoTagContext,
        model: str
    ) -> RequirementsAnalysis:
        """Analyze requirements from AUTO tag using real LLM.
        
        Args:
            auto_tag: The AUTO tag content to analyze
            context: Full resolution context
            model: Model to use for analysis
            
        Returns:
            Requirements analysis result
            
        Raises:
            ParseError: If analysis fails to parse
        """
        # Build analysis prompt
        analysis_prompt = self._build_analysis_prompt(auto_tag, context)
        
        # Add schema instructions
        schema_prompt = self.parser.create_schema_prompt(self.REQUIREMENTS_SCHEMA)
        full_prompt = f"{analysis_prompt}\n\n{schema_prompt}"
        
        # Call real model
        try:
            logger.debug(f"Analyzing requirements with {model}")
            response = await self.model_caller.call_model(
                model=model,
                prompt=full_prompt,
                temperature=0.3,  # Lower temperature for structured output
                json_mode=True if model.startswith("gpt") else False
            )
            
            # Parse response
            parsed_response = self.parser.parse_json_response(response, self.REQUIREMENTS_SCHEMA)
            
            # Create RequirementsAnalysis object
            requirements = RequirementsAnalysis(
                tools_needed=parsed_response.get("tools_needed", []),
                output_format=parsed_response.get("output_format"),
                constraints=parsed_response.get("constraints", {}),
                data_dependencies=parsed_response.get("data_dependencies", []),
                model_requirements=parsed_response.get("model_requirements", {}),
                expected_output_type=parsed_response.get("expected_output_type", "string"),
                model_used=model
            )
            
            logger.info(f"Requirements analysis complete: {requirements.expected_output_type}")
            return requirements
            
        except Exception as e:
            logger.error(f"Failed to analyze requirements: {e}")
            raise ParseError(f"Failed to analyze requirements: {e}")
    
    def _build_analysis_prompt(self, auto_tag: str, context: AutoTagContext) -> str:
        """Build prompt for requirements analysis."""
        # Serialize pipeline structure
        pipeline_structure = self._serialize_pipeline(context.pipeline)
        
        # Get available context
        available_vars = list(context.variables.keys())
        available_results = list(context.step_results.keys())
        
        # Build comprehensive prompt
        prompt = f"""Analyze this AUTO tag to determine its requirements for resolution.

AUTO tag: {auto_tag}

Current Context:
- Current task ID: {context.current_task_id}
- Tag location in pipeline: {context.tag_location}
- Resolution depth: {context.resolution_depth}

Available Variables:
{json.dumps(available_vars, indent=2)}

Available Step Results:
{json.dumps(available_results, indent=2)}

Pipeline Structure:
{pipeline_structure}

Analyze the AUTO tag and determine:

1. **Tools Needed**: What tools (if any) are required to accomplish this task? Consider:
   - Web search tools
   - File system tools
   - API calling tools
   - Data processing tools
   - Code execution tools

2. **Output Format**: What format should the output be in?
   - text: Plain text response
   - json: Structured JSON data
   - markdown: Formatted markdown
   - yaml: YAML configuration
   - code: Programming code
   - Other specific formats

3. **Constraints**: What constraints or requirements exist?
   - Length limits
   - Format requirements
   - Content restrictions
   - Quality requirements

4. **Data Dependencies**: Which variables or step results are referenced?
   - Look for {{{{ variable }}}} references
   - Identify required context data
   - Note any conditional dependencies

5. **Model Requirements**: What model capabilities are needed?
   - Text generation
   - Code understanding
   - Vision capabilities
   - Function calling
   - Large context window

6. **Expected Output Type**: What is the fundamental type of the output?
   - string: Text output
   - number: Numeric value
   - boolean: True/false
   - object: Structured data
   - array: List of items
   - null: No output expected

Be thorough but realistic. Only specify tools and requirements that are actually needed for this specific AUTO tag."""
        
        return prompt
    
    def _serialize_pipeline(self, pipeline) -> str:
        """Serialize pipeline structure for context."""
        # Create a simplified view of the pipeline
        structure = {
            "id": pipeline.id,
            "name": pipeline.name,
            "steps": []
        }
        
        for task_id, task in pipeline.tasks.items():
            step_info = {
                "id": task_id,
                "action": task.action,
                "dependencies": list(task.dependencies),
                "has_parameters": bool(task.parameters)
            }
            
            # Add relevant metadata
            if hasattr(task, 'metadata') and task.metadata:
                if "tool" in task.metadata:
                    step_info["tool"] = task.metadata["tool"]
                if "produces" in task.metadata:
                    step_info["produces"] = task.metadata["produces"]
                    
            structure["steps"].append(step_info)
        
        return json.dumps(structure, indent=2)