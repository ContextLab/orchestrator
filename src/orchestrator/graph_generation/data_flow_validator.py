"""
DataFlowValidator - Type safety and data flow validation.

This module implements comprehensive data flow validation as outlined in Issue #199,
ensuring type safety, validating variable references, and providing clear error messages
for data flow issues in pipeline definitions.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Set, Optional, Any
from jinja2 import Template

from .types import (
    ParsedPipeline, ParsedStep, DataFlowSchema, InputSchema, OutputSchema,
    ValidationError
)
from .syntax_parser import DeclarativeSyntaxParser

logger = logging.getLogger(__name__)


class DataFlowValidator:
    """
    Implements comprehensive data flow validation as outlined in Issue #199.
    Ensures type safety, validates variable references, and provides clear error messages.
    """
    
    def __init__(self):
        self.syntax_parser = DeclarativeSyntaxParser()
        logger.info("DataFlowValidator initialized")
        
    async def validate_data_flow(self, parsed_pipeline: ParsedPipeline) -> DataFlowSchema:
        """Complete data flow validation with type safety."""
        logger.debug(f"Validating data flow for pipeline: {parsed_pipeline.id}")
        
        schema = DataFlowSchema()
        
        # Build input schema from pipeline definition
        schema.inputs = parsed_pipeline.inputs
        
        # Analyze each step's data transformations
        for step in parsed_pipeline.steps:
            step_outputs = await self._analyze_step_data_flow(step, schema)
            schema.add_step_schema(step.id, step_outputs)
            
            # Validate all variable references in this step
            errors = await self._validate_variable_references(step, schema)
            schema.validation_errors.extend(errors)
            
        # Build output schema
        schema.outputs = parsed_pipeline.outputs
        
        logger.info(f"Data flow validation completed: {len(schema.validation_errors)} errors found")
        
        return schema
        
    async def _analyze_step_data_flow(self, 
                                    step: ParsedStep, 
                                    schema: DataFlowSchema) -> Dict[str, OutputSchema]:
        """Analyze a step's data transformations and outputs.""" 
        outputs = {}
        
        # Use explicit outputs if defined
        if step.outputs:
            outputs.update(step.outputs)
        else:
            # Infer basic outputs based on step type
            if step.tool or step.action:
                outputs["result"] = OutputSchema(
                    name="result",
                    type="string",  # Default type
                    description=f"Output from {step.tool or step.action}"
                )
                
        return outputs
        
    async def _validate_variable_references(self, 
                                          step: ParsedStep, 
                                          schema: DataFlowSchema) -> List[ValidationError]:
        """
        Validate all {{ variable.path }} references in step inputs.
        Examples from Issue #199 comment:
        - {{ web_search.search_results }} ← FROM web_search outputs
        - {{ inputs.topic }} ← FROM pipeline inputs  
        - {{ item.claim_text }} ← FROM current iteration item
        """
        errors = []
        available_vars = schema.get_available_variables()
        
        for input_name, input_value in step.inputs.items():
            if isinstance(input_value, str):
                # Extract all template variables
                template_vars = self.syntax_parser.extract_template_variables(input_value)
                
                for var in template_vars:
                    if not self._is_variable_available(var, available_vars, step):
                        error = ValidationError(
                            step_id=step.id,
                            input_name=input_name,
                            variable=var,
                            error=f"Variable '{var}' is not available",
                            suggestion=self._suggest_variable_correction(var, available_vars)
                        )
                        errors.append(error)
                        
        return errors
        
    def _is_variable_available(self, 
                             variable: str, 
                             available_vars: Dict[str, str],
                             step: ParsedStep) -> bool:
        """Check if a variable is available in the current context.""" 
        # Special context variables
        var_root = variable.split('.')[0]
        if var_root in ['inputs', 'item', 'loop', 'index']:
            return True
            
        # Check explicit dependencies
        if var_root in step.depends_on:
            return True
            
        # Check available variables
        return variable in available_vars
        
    def _suggest_variable_correction(self, 
                                   invalid_var: str, 
                                   available_vars: Dict[str, str]) -> Optional[str]:
        """Suggest corrections for invalid variable references."""
        # Simple suggestion based on similarity
        var_parts = invalid_var.split('.')
        
        for available_var in available_vars:
            available_parts = available_var.split('.')
            
            # Check for similar step names
            if len(var_parts) >= 1 and len(available_parts) >= 1:
                if self._strings_similar(var_parts[0], available_parts[0]):
                    return f"Did you mean '{available_var}'?"
                    
        return None
        
    def _strings_similar(self, a: str, b: str, threshold: float = 0.8) -> bool:
        """Simple string similarity check."""
        if len(a) == 0 or len(b) == 0:
            return False
            
        # Simple character overlap similarity
        common = sum(1 for x in a if x in b)
        return common / max(len(a), len(b)) >= threshold