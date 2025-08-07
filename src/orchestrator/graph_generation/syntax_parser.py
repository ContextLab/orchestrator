"""
DeclarativeSyntaxParser - Parses enhanced YAML syntax from Issue #199.

This module handles parsing the enhanced declarative pipeline syntax that allows users
to specify steps + dependencies without needing to understand graph concepts. It supports
both legacy and new syntax formats for backwards compatibility.

Key Features:
- Type-safe input/output definitions
- Declarative step specifications with depends_on arrays  
- Advanced control flow (conditions, loops, parallel_map)
- Template variable extraction and validation
- Backwards compatibility with existing pipeline formats
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.exceptions import YAMLCompilerError
from .types import (
    ParsedPipeline, ParsedStep, InputSchema, OutputSchema, 
    StepType, ValidationError
)

logger = logging.getLogger(__name__)


class SyntaxParsingError(YAMLCompilerError):
    """Raised when syntax parsing fails."""
    pass


class DeclarativeSyntaxParser:
    """
    Parses the enhanced declarative syntax from Issue #199 comment.
    Supports all new features while maintaining backwards compatibility with existing pipelines.
    """
    
    def __init__(self):
        # Template variable pattern for extracting {{ variable.path }} references
        self.template_var_pattern = re.compile(r'\{\{\s*([^}]+)\s*\}\}')
        
        # Supported step types
        self.step_types = {
            'standard': StepType.STANDARD,
            'parallel_map': StepType.PARALLEL_MAP,
            'loop': StepType.LOOP,
            'while': StepType.WHILE, 
            'for': StepType.FOR,
            'conditional': StepType.CONDITIONAL
        }
        
        logger.info("DeclarativeSyntaxParser initialized")
        
    async def parse_pipeline_definition(self, pipeline_def: Dict[str, Any]) -> ParsedPipeline:
        """
        Parse and validate declarative pipeline syntax with full validation.
        
        Supports both legacy and new declarative formats from Issue #199.
        
        Args:
            pipeline_def: Raw pipeline definition from YAML
            
        Returns:
            ParsedPipeline object with validated structure
            
        Raises:
            SyntaxParsingError: If parsing or validation fails
        """
        try:
            logger.debug(f"Parsing pipeline definition: {pipeline_def.get('id', 'unknown')}")
            
            # Determine format and parse accordingly
            if self._is_legacy_format(pipeline_def):
                logger.debug("Detected legacy pipeline format")
                parsed = await self._convert_legacy_format(pipeline_def)
            else:
                logger.debug("Detected new declarative format")
                parsed = await self._parse_new_declarative_format(pipeline_def)
                
            # Comprehensive validation
            await self._validate_parsed_pipeline(parsed)
            
            logger.info(f"Successfully parsed pipeline '{parsed.id}' with {len(parsed.steps)} steps")
            return parsed
            
        except Exception as e:
            raise SyntaxParsingError(f"Failed to parse pipeline definition: {e}") from e
            
    def _is_legacy_format(self, pipeline_def: Dict[str, Any]) -> bool:
        """
        Determine if this is a legacy pipeline format.
        
        Legacy format characteristics:
        - Has 'steps' array with action/tool fields
        - Uses 'dependencies' instead of 'depends_on'
        - No typed inputs/outputs sections
        """
        # Check for new format indicators
        has_typed_inputs = (
            'inputs' in pipeline_def and 
            isinstance(pipeline_def['inputs'], dict) and
            any(isinstance(v, dict) and 'type' in v for v in pipeline_def['inputs'].values())
        )
        
        has_typed_outputs = (
            'outputs' in pipeline_def and
            isinstance(pipeline_def['outputs'], dict) and
            any(isinstance(v, dict) and 'type' in v for v in pipeline_def['outputs'].values())
        )
        
        # Check for new step format indicators
        has_new_step_format = False
        if 'steps' in pipeline_def:
            for step in pipeline_def['steps']:
                if isinstance(step, dict):
                    if 'depends_on' in step or 'type' in step and step['type'] in self.step_types:
                        has_new_step_format = True
                        break
        
        # If it has new format indicators, it's not legacy
        return not (has_typed_inputs or has_typed_outputs or has_new_step_format)
        
    async def _convert_legacy_format(self, pipeline_def: Dict[str, Any]) -> ParsedPipeline:
        """
        Convert legacy pipeline format to new ParsedPipeline structure.
        
        This maintains backwards compatibility with existing pipeline files.
        """
        # Extract basic pipeline info
        pipeline_id = pipeline_def.get('id', 'unknown')
        name = pipeline_def.get('name', pipeline_id)
        description = pipeline_def.get('description')
        version = pipeline_def.get('version', '1.0.0')
        
        # Convert legacy inputs (parameters)
        inputs = {}
        if 'parameters' in pipeline_def:
            for param_name, param_def in pipeline_def['parameters'].items():
                if isinstance(param_def, dict):
                    inputs[param_name] = InputSchema(
                        name=param_name,
                        type=param_def.get('type', 'string'),
                        required=param_def.get('required', True),
                        default=param_def.get('default'),
                        description=param_def.get('description'),
                        enum=param_def.get('enum'),
                        range=param_def.get('range')
                    )
                else:
                    # Simple default value
                    inputs[param_name] = InputSchema(
                        name=param_name,
                        type='string',
                        required=False,
                        default=param_def
                    )
                    
        # Convert legacy outputs
        outputs = {}
        if 'outputs' in pipeline_def:
            for output_name, output_value in pipeline_def['outputs'].items():
                outputs[output_name] = OutputSchema(
                    name=output_name,
                    type='string',  # Default type for legacy
                    computed_as=output_value if isinstance(output_value, str) else None
                )
                
        # Convert legacy steps
        steps = []
        if 'steps' in pipeline_def:
            for i, step_def in enumerate(pipeline_def['steps']):
                parsed_step = await self._convert_legacy_step(step_def, i)
                steps.append(parsed_step)
                
        return ParsedPipeline(
            id=pipeline_id,
            name=name,
            description=description,
            version=version,
            inputs=inputs,
            outputs=outputs,
            steps=steps,
            metadata=pipeline_def.get('metadata', {}),
            config=pipeline_def.get('config', {})
        )
        
    async def _convert_legacy_step(self, step_def: Dict[str, Any], index: int) -> ParsedStep:
        """Convert a legacy step definition to new format."""
        # Extract basic step info
        step_id = step_def.get('id', f"step_{index}")
        tool = step_def.get('tool')
        action = step_def.get('action')
        
        # Convert legacy dependencies field
        depends_on = []
        if 'dependencies' in step_def:
            depends_on = step_def['dependencies']
        elif 'depends_on' in step_def:
            depends_on = step_def['depends_on']
            
        # Extract inputs (parameters)
        inputs = step_def.get('parameters', {}).copy()
        
        # Extract condition
        condition = step_def.get('condition')
        
        # Extract model requirements (requires_model)
        model_requirements = step_def.get('requires_model')
        
        return ParsedStep(
            id=step_id,
            type=StepType.STANDARD,
            tool=tool,
            action=action,
            inputs=inputs,
            depends_on=depends_on,
            condition=condition,
            model_requirements=model_requirements,
            original_definition=step_def
        )
        
    async def _parse_new_declarative_format(self, pipeline_def: Dict[str, Any]) -> ParsedPipeline:
        """
        Parse the new declarative format from Issue #199:
        - Enhanced input/output definitions with type safety
        - Declarative step specifications  
        - Automatic dependency resolution via depends_on arrays
        - Control flow with conditions, loops, parallel_map
        """
        # Extract basic pipeline info
        pipeline_id = pipeline_def['id']  # Required in new format
        name = pipeline_def.get('name', pipeline_id)
        description = pipeline_def.get('description')
        version = pipeline_def.get('version', '1.0.0')
        
        # Parse type-safe inputs
        inputs = await self._parse_typed_inputs(pipeline_def.get('inputs', {}))
        
        # Parse type-safe outputs  
        outputs = await self._parse_typed_outputs(pipeline_def.get('outputs', {}))
        
        # Parse declarative steps
        steps = []
        for step_def in pipeline_def.get('steps', []):
            parsed_step = await self._parse_declarative_step(step_def, inputs)
            steps.append(parsed_step)
            
        # Handle advanced control flow steps
        if 'advanced_steps' in pipeline_def:
            for advanced_step in pipeline_def['advanced_steps']:
                parsed_advanced = await self._parse_advanced_control_flow(advanced_step)
                steps.append(parsed_advanced)
                
        return ParsedPipeline(
            id=pipeline_id,
            name=name,
            description=description,
            version=version,
            inputs=inputs,
            outputs=outputs,
            steps=steps,
            metadata=pipeline_def.get('metadata', {}),
            config=pipeline_def.get('config', {})
        )
        
    async def _parse_typed_inputs(self, inputs_def: Dict[str, Any]) -> Dict[str, InputSchema]:
        """Parse type-safe input definitions from new format."""
        inputs = {}
        
        for input_name, input_spec in inputs_def.items():
            if isinstance(input_spec, dict):
                # Full input schema
                inputs[input_name] = InputSchema(
                    name=input_name,
                    type=input_spec.get('type', 'string'),
                    required=input_spec.get('required', True),
                    default=input_spec.get('default'),
                    description=input_spec.get('description'),
                    enum=input_spec.get('enum'),
                    range=input_spec.get('range'),
                    example=input_spec.get('example')
                )
            else:
                # Simple default value
                inputs[input_name] = InputSchema(
                    name=input_name,
                    type='string',
                    required=False,
                    default=input_spec
                )
                
        return inputs
        
    async def _parse_typed_outputs(self, outputs_def: Dict[str, Any]) -> Dict[str, OutputSchema]:
        """Parse type-safe output definitions from new format."""
        outputs = {}
        
        for output_name, output_spec in outputs_def.items():
            if isinstance(output_spec, dict):
                # Full output schema
                outputs[output_name] = OutputSchema(
                    name=output_name,
                    type=output_spec.get('type', 'string'),
                    description=output_spec.get('description'),
                    schema=output_spec.get('schema'),
                    computed_as=output_spec.get('source'),  # 'source' field in new format
                    format=output_spec.get('format')
                )
            else:
                # Simple template expression
                outputs[output_name] = OutputSchema(
                    name=output_name,
                    type='string',
                    computed_as=output_spec if isinstance(output_spec, str) else None
                )
                
        return outputs
        
    async def _parse_declarative_step(self, 
                                    step_def: Dict[str, Any], 
                                    inputs_schema: Dict[str, InputSchema]) -> ParsedStep:
        """Parse a single declarative step from the new format."""
        # Required fields
        step_id = step_def['id']
        
        # Determine step type
        step_type_str = step_def.get('type', 'standard')
        step_type = self.step_types.get(step_type_str, StepType.STANDARD)
        
        # Basic step info
        tool = step_def.get('tool')
        action = step_def.get('action')
        inputs = step_def.get('inputs', {})
        depends_on = step_def.get('depends_on', [])
        condition = step_def.get('condition')
        model_requirements = step_def.get('model', step_def.get('requires_model'))
        
        # Parse output schema
        outputs = {}
        if 'outputs' in step_def:
            for output_name, output_spec in step_def['outputs'].items():
                if isinstance(output_spec, dict):
                    outputs[output_name] = OutputSchema(
                        name=output_name,
                        type=output_spec.get('type', 'string'),
                        description=output_spec.get('description'),
                        schema=output_spec.get('schema'),
                        computed_as=output_spec.get('computed_as'),
                        format=output_spec.get('format')
                    )
                else:
                    outputs[output_name] = OutputSchema(
                        name=output_name,
                        type='string',
                        description=output_spec if isinstance(output_spec, str) else None
                    )
                    
        # Handle special step types
        items = None
        substeps = None
        max_iterations = None
        goto = None
        else_step = None
        
        if step_type == StepType.PARALLEL_MAP:
            items = step_def.get('items')
            if 'substeps' in step_def:
                substeps = []
                for substep_def in step_def['substeps']:
                    substep = await self._parse_declarative_step(substep_def, inputs_schema)
                    substeps.append(substep)
                    
        elif step_type in [StepType.LOOP, StepType.WHILE, StepType.FOR]:
            max_iterations = step_def.get('max_iterations', 100)
            if 'steps' in step_def:
                substeps = []
                for substep_def in step_def['steps']:
                    substep = await self._parse_declarative_step(substep_def, inputs_schema)
                    substeps.append(substep)
                    
        # Handle control flow
        goto = step_def.get('goto')
        else_step = step_def.get('else_step', step_def.get('else'))
        
        return ParsedStep(
            id=step_id,
            type=step_type,
            tool=tool,
            action=action,
            inputs=inputs,
            outputs=outputs,
            depends_on=depends_on,
            condition=condition,
            model_requirements=model_requirements,
            items=items,
            substeps=substeps,
            max_iterations=max_iterations,
            goto=goto,
            else_step=else_step,
            metadata=step_def.get('metadata', {}),
            original_definition=step_def
        )
        
    async def _parse_advanced_control_flow(self, step_def: Dict[str, Any]) -> ParsedStep:
        """Parse advanced control flow steps from the advanced_steps section."""
        # This is similar to regular step parsing but focuses on control flow
        return await self._parse_declarative_step(step_def, {})
        
    async def _validate_parsed_pipeline(self, parsed: ParsedPipeline) -> None:
        """Comprehensive validation of the parsed pipeline."""
        validation_errors = []
        
        # Validate basic structure
        if not parsed.id:
            validation_errors.append("Pipeline must have an id")
            
        if not parsed.steps:
            validation_errors.append("Pipeline must have at least one step")
            
        # Validate step IDs are unique
        step_ids = [step.id for step in parsed.steps]
        if len(step_ids) != len(set(step_ids)):
            duplicates = [sid for sid in set(step_ids) if step_ids.count(sid) > 1]
            validation_errors.append(f"Duplicate step IDs found: {duplicates}")
            
        # Validate dependencies exist
        for step in parsed.steps:
            for dep_id in step.depends_on:
                if dep_id not in step_ids:
                    validation_errors.append(
                        f"Step '{step.id}' depends on undefined step '{dep_id}'"
                    )
                    
        # Validate template variables in inputs
        available_vars = set()
        available_vars.update(f"inputs.{name}" for name in parsed.inputs.keys())
        # Also add legacy parameter names without the "inputs." prefix for backward compatibility
        available_vars.update(parsed.inputs.keys())
        
        # Build a map of all possible step outputs for conditional reference validation
        all_possible_outputs = set()
        for step in parsed.steps:
            # Add default output if step has tool/action
            if step.tool or step.action:
                all_possible_outputs.add(f"{step.id}.result")
            # Add explicit outputs
            for output_name in step.outputs.keys():
                all_possible_outputs.add(f"{step.id}.{output_name}")
        
        for step in parsed.steps:
            step_vars = set()
            
            # Extract template variables from inputs
            for input_name, input_value in step.inputs.items():
                if isinstance(input_value, str):
                    template_vars = self.extract_template_variables(input_value)
                    step_vars.update(template_vars)
                    
            # Extract from condition
            if step.condition:
                condition_vars = self.extract_template_variables(step.condition)
                step_vars.update(condition_vars)
                
            # Validate all variables are available
            for var in step_vars:
                # Find the context where this variable is used
                context_text = ""
                for input_name, input_value in step.inputs.items():
                    if isinstance(input_value, str) and var in self.extract_template_variables(input_value):
                        context_text = input_value
                        break
                if not context_text and step.condition and var in self.extract_template_variables(step.condition):
                    context_text = step.condition
                    
                if not self._is_variable_available_enhanced(var, available_vars, step.depends_on, all_possible_outputs, context_text):
                    validation_errors.append(
                        f"Step '{step.id}' references undefined variable '{var}'"
                    )
                    
            # Add this step's outputs to available variables for subsequent steps
            # Add default output if step has tool/action
            if step.tool or step.action:
                available_vars.add(f"{step.id}.result")
            for output_name in step.outputs.keys():
                available_vars.add(f"{step.id}.{output_name}")
                
        if validation_errors:
            raise SyntaxParsingError(f"Pipeline validation failed: {'; '.join(validation_errors)}")
            
        logger.debug(f"Pipeline validation passed for '{parsed.id}'")
        
    def extract_template_variables(self, text: str) -> List[str]:
        """
        Extract template variables from text.
        
        Examples:
        - "{{ web_search.results }}" → ["web_search.results"]
        - "{{ inputs.topic }}" → ["inputs.topic"]  
        - "{{ item.claim_text }}" → ["item.claim_text"]
        """
        if not isinstance(text, str):
            return []
            
        matches = self.template_var_pattern.findall(text)
        variables = []
        
        for match in matches:
            # Clean up the variable (remove filters, etc.)
            var = match.strip()
            
            # Handle Jinja2 filters (e.g., "variable | filter")
            if '|' in var:
                var = var.split('|')[0].strip()
                
            # Handle array indexing (e.g., "variable[0]")
            if '[' in var:
                var = var.split('[')[0].strip()
            
            # Only add non-empty variables
            if var and len(var) > 0:
                variables.append(var)
            
        return variables
        
    def _is_variable_available(self, 
                              variable: str, 
                              available_vars: Set[str], 
                              step_dependencies: List[str]) -> bool:
        """Check if a variable is available given current context."""
        # Special variables that are always available
        special_vars = ['item', 'loop', 'index', 'iteration_count']
        var_root = variable.split('.')[0]
        
        if var_root in special_vars:
            return True
            
        # Check if variable is in available set
        if variable in available_vars:
            return True
            
        # Check if variable comes from a dependency
        var_step = var_root
        if var_step in step_dependencies:
            return True
            
        # Handle complex expressions with operators
        if any(op in variable for op in ['*', '+', '-', '/', '>', '<', '==', '!=', '>=', '<=']):
            # For expressions like "inputs.depth * 5", extract the variable parts
            import re
            # Find variable patterns in the expression
            var_patterns = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*', variable)
            for var_pattern in var_patterns:
                # Check if this looks like a template variable (has dots or is known root)
                if '.' in var_pattern or var_pattern in ['inputs'] + list(available_vars) + step_dependencies:
                    var_parts = var_pattern.split('.')
                    var_root_check = var_parts[0]
                    if var_root_check in ['inputs'] or var_root_check in step_dependencies or var_pattern in available_vars:
                        continue  # This variable is valid
                    else:
                        return False  # Found an invalid variable
            return True  # All variables in expression are valid
            
        # Handle Jinja2 filters and functions
        if '|' in variable:
            # Extract the main variable before filters
            main_var = variable.split('|')[0].strip()
            # Recursively check the main variable
            return self._is_variable_available(main_var, available_vars, step_dependencies)
            
        # Handle array/object access
        if '[' in variable and ']' in variable:
            # Extract variable before array access
            base_var = variable.split('[')[0].strip()
            return self._is_variable_available(base_var, available_vars, step_dependencies)
            
        # Check for pipeline inputs (inputs.*)
        if variable.startswith('inputs.'):
            return True
            
        # Check if the root variable is available (for chained access)
        available_roots = set()
        for av in available_vars:
            if '.' in av:
                available_roots.add(av.split('.')[0])
        available_roots.update(step_dependencies)
        available_roots.add('inputs')
        
        if var_root in available_roots:
            return True
                
        return False
        
    def _is_variable_available_enhanced(self, 
                                      variable: str, 
                                      available_vars: Set[str], 
                                      step_dependencies: List[str],
                                      all_possible_outputs: Set[str],
                                      context_text: str = "") -> bool:
        """Enhanced variable availability check that handles conditional references."""
        # First try the standard check
        if self._is_variable_available(variable, available_vars, step_dependencies):
            return True
            
        # Check if this is a conditional reference (has default filter)
        if '| default(' in context_text:
            # This is a conditional reference, check if the variable could potentially exist
            var_root = variable.split('.')[0]
            var_full = variable
            
            # Check if this variable exists in all possible outputs (from conditional steps)
            if var_full in all_possible_outputs:
                return True
                
            # Check if the root step exists (even if it's conditional)
            step_ids = {output.split('.')[0] for output in all_possible_outputs}
            if var_root in step_ids:
                return True
                
        return False
        
    def get_step_dependencies_from_template_vars(self, step: ParsedStep) -> Set[str]:
        """Extract implicit dependencies from template variables."""
        dependencies = set()
        
        # Check all input values
        for input_value in step.inputs.values():
            if isinstance(input_value, str):
                template_vars = self.extract_template_variables(input_value)
                for var in template_vars:
                    var_step = var.split('.')[0]
                    if var_step not in ['inputs', 'item', 'loop', 'index']:
                        dependencies.add(var_step)
                        
        # Check condition
        if step.condition:
            condition_vars = self.extract_template_variables(step.condition)
            for var in condition_vars:
                var_step = var.split('.')[0]
                if var_step not in ['inputs', 'item', 'loop', 'index']:
                    dependencies.add(var_step)
                    
        return dependencies