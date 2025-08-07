"""
Enhanced YAML Processor - Issue #199 Declarative Improvements

This module implements the enhanced declarative pipeline syntax from Issue #199, providing
a more intuitive and user-friendly way to define pipelines where users never need to
understand graph concepts.

Key Features:
- Type-safe input/output definitions with validation
- Declarative step specifications with automatic dependency resolution
- Enhanced control flow (parallel_map, loops, conditions)  
- Intelligent defaults and auto-discovery
- Complex data flow with schema validation
- Integration with AutomaticGraphGenerator for seamless processing

The enhanced syntax allows users to focus on WHAT they want to accomplish rather than
HOW the graph should be structured, fully implementing the Issue #199 vision.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Type
from enum import Enum

from ..core.exceptions import YAMLCompilerError, ValidationError

logger = logging.getLogger(__name__)


class StepType(Enum):
    """Enhanced step types from Issue #199."""
    STANDARD = "standard"
    PARALLEL_MAP = "parallel_map"
    LOOP = "loop"
    WHILE = "while"
    FOR = "for"
    CONDITIONAL = "conditional"
    WORKFLOW = "workflow"  # Sub-workflow support


class DataType(Enum):
    """Supported data types for type safety."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    FILE = "file"
    JSON = "json"
    ANY = "any"


@dataclass
class TypeSafeInput:
    """Type-safe input definition from Issue #199 enhanced syntax."""
    name: str
    type: DataType
    description: str = ""
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None
    range: Optional[List[Union[int, float]]] = None
    example: Any = None
    validation_rules: Optional[Dict[str, Any]] = None


@dataclass
class TypeSafeOutput:
    """Type-safe output definition with schema validation."""
    name: str
    type: DataType
    description: str = ""
    source: str = ""  # Template for extracting value
    schema: Optional[Dict[str, Any]] = None
    format: Optional[str] = None
    location: Optional[str] = None  # For file outputs


@dataclass
class EnhancedStep:
    """Enhanced step definition supporting all Issue #199 features."""
    id: str
    type: StepType = StepType.STANDARD
    tool: Optional[str] = None
    action: Optional[str] = None
    model: Optional[Dict[str, Any]] = None
    
    # Enhanced dependency management
    depends_on: List[str] = None
    condition: Optional[str] = None
    
    # Input/Output with type safety
    inputs: Dict[str, Any] = None
    outputs: Dict[str, TypeSafeOutput] = None
    
    # Control flow features
    items: Optional[str] = None  # For parallel_map and loops
    max_parallel: Optional[int] = None
    max_iterations: Optional[int] = None
    loop_condition: Optional[str] = None
    goto: Optional[str] = None
    else_step: Optional[str] = None
    
    # Sub-workflow support
    steps: Optional[List['EnhancedStep']] = None
    
    # Advanced features
    timeout: Optional[int] = None
    retry_config: Optional[Dict[str, Any]] = None
    continue_on_error: bool = False
    dynamic_routing: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = []
        if self.inputs is None:
            self.inputs = {}
        if self.outputs is None:
            self.outputs = {}


@dataclass
class EnhancedPipeline:
    """Complete enhanced pipeline definition implementing Issue #199 vision."""
    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    version: str = "1.0.0"
    type: str = "workflow"
    
    # Type-safe input/output definitions
    inputs: Dict[str, TypeSafeInput] = None
    outputs: Dict[str, TypeSafeOutput] = None
    
    # Enhanced step definitions
    steps: List[EnhancedStep] = None
    advanced_steps: List[EnhancedStep] = None
    
    # Configuration and metadata
    config: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    # Advanced features
    error_handling: Optional[Dict[str, Any]] = None
    monitoring: Optional[Dict[str, Any]] = None
    optimization: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.inputs is None:
            self.inputs = {}
        if self.outputs is None:
            self.outputs = {}
        if self.steps is None:
            self.steps = []
        if self.advanced_steps is None:
            self.advanced_steps = []
        if self.config is None:
            self.config = {}
        if self.metadata is None:
            self.metadata = {}


class EnhancedYAMLProcessor:
    """
    Processes enhanced YAML syntax from Issue #199 and converts to internal representation.
    
    This processor handles the new declarative syntax while maintaining backwards compatibility
    with existing pipeline formats. It provides comprehensive validation and intelligent
    defaults as outlined in the Issue #199 vision.
    """
    
    def __init__(self):
        self.supported_data_types = {
            'string': DataType.STRING,
            'str': DataType.STRING, 
            'integer': DataType.INTEGER,
            'int': DataType.INTEGER,
            'float': DataType.FLOAT,
            'number': DataType.FLOAT,
            'boolean': DataType.BOOLEAN,
            'bool': DataType.BOOLEAN,
            'array': DataType.ARRAY,
            'list': DataType.ARRAY,
            'object': DataType.OBJECT,
            'dict': DataType.OBJECT,
            'file': DataType.FILE,
            'json': DataType.JSON,
            'any': DataType.ANY
        }
        
        logger.info("EnhancedYAMLProcessor initialized with Issue #199 syntax support")
        
    async def process_enhanced_yaml(self, yaml_content: Dict[str, Any]) -> EnhancedPipeline:
        """
        Main entry point for processing enhanced YAML syntax.
        
        Args:
            yaml_content: Parsed YAML dictionary
            
        Returns:
            EnhancedPipeline object ready for graph generation
            
        Raises:
            YAMLCompilerError: If YAML processing fails
            ValidationError: If validation fails
        """
        try:
            # Determine format and process accordingly
            if self._is_enhanced_format(yaml_content):
                logger.info("Processing enhanced Issue #199 format")
                pipeline = await self._process_enhanced_format(yaml_content)
            else:
                logger.info("Processing legacy format with compatibility layer")
                pipeline = await self._process_legacy_format_enhanced(yaml_content)
                
            # Validate the complete pipeline
            await self._validate_enhanced_pipeline(pipeline)
            
            # Apply intelligent defaults
            pipeline = await self._apply_intelligent_defaults(pipeline)
            
            logger.info(f"Successfully processed enhanced pipeline: {pipeline.id}")
            return pipeline
            
        except Exception as e:
            logger.error(f"Enhanced YAML processing failed: {e}")
            raise YAMLCompilerError(f"Failed to process enhanced YAML: {e}") from e
            
    def _is_enhanced_format(self, yaml_content: Dict[str, Any]) -> bool:
        """Detect if YAML uses the enhanced Issue #199 syntax."""
        enhanced_indicators = [
            # Pipeline type specification (not just any 'type' key)
            yaml_content.get('type') == 'workflow',
            # Type-safe inputs with object definitions
            'inputs' in yaml_content and isinstance(yaml_content['inputs'], dict) and 
            any(isinstance(v, dict) and 'type' in v for v in yaml_content['inputs'].values()),
            # Type-safe outputs with object definitions  
            'outputs' in yaml_content and isinstance(yaml_content['outputs'], dict) and
            any(isinstance(v, dict) and ('type' in v or 'source' in v) for v in yaml_content['outputs'].values()),
            # Advanced steps section
            'advanced_steps' in yaml_content,
            # Enhanced step types
            any(isinstance(step, dict) and step.get('type') in ['parallel_map', 'loop', 'while', 'for'] 
                for step in yaml_content.get('steps', []) + yaml_content.get('advanced_steps', []))
        ]
        
        # Need at least 2 indicators for enhanced format to avoid false positives
        return sum(bool(indicator) for indicator in enhanced_indicators) >= 2
        
    async def _process_enhanced_format(self, yaml_content: Dict[str, Any]) -> EnhancedPipeline:
        """Process the new enhanced format from Issue #199."""
        
        # Extract basic pipeline information
        if 'id' not in yaml_content:
            raise ValidationError("Pipeline must have an 'id' field")
        pipeline_id = yaml_content['id']
        pipeline_name = yaml_content.get('name', pipeline_id)
        pipeline_desc = yaml_content.get('description', '')
        pipeline_version = yaml_content.get('version', '1.0.0')
        pipeline_type = yaml_content.get('type', 'workflow')
        
        # Process type-safe inputs
        inputs = {}
        if 'inputs' in yaml_content:
            inputs = await self._process_type_safe_inputs(yaml_content['inputs'])
            
        # Process type-safe outputs  
        outputs = {}
        if 'outputs' in yaml_content:
            outputs = await self._process_type_safe_outputs(yaml_content['outputs'])
            
        # Process enhanced steps
        steps = []
        if 'steps' in yaml_content:
            steps = await self._process_enhanced_steps(yaml_content['steps'])
            
        # Process advanced steps
        advanced_steps = []
        if 'advanced_steps' in yaml_content:
            advanced_steps = await self._process_enhanced_steps(yaml_content['advanced_steps'])
            
        # Extract configuration
        config = yaml_content.get('config', {})
        metadata = yaml_content.get('metadata', {})
        error_handling = yaml_content.get('error_handling')
        monitoring = yaml_content.get('monitoring')
        optimization = yaml_content.get('optimization')
        
        return EnhancedPipeline(
            id=pipeline_id,
            name=pipeline_name,
            description=pipeline_desc,
            version=pipeline_version,
            type=pipeline_type,
            inputs=inputs,
            outputs=outputs,
            steps=steps,
            advanced_steps=advanced_steps,
            config=config,
            metadata=metadata,
            error_handling=error_handling,
            monitoring=monitoring,
            optimization=optimization
        )
        
    async def _process_type_safe_inputs(self, inputs_def: Dict[str, Any]) -> Dict[str, TypeSafeInput]:
        """Process type-safe input definitions."""
        processed_inputs = {}
        
        for input_name, input_spec in inputs_def.items():
            if isinstance(input_spec, dict):
                # Enhanced format with type safety
                input_type = self._parse_data_type(input_spec.get('type', 'string'))
                
                processed_input = TypeSafeInput(
                    name=input_name,
                    type=input_type,
                    description=input_spec.get('description', ''),
                    required=input_spec.get('required', True),
                    default=input_spec.get('default'),
                    enum=input_spec.get('enum'),
                    range=input_spec.get('range'),
                    example=input_spec.get('example'),
                    validation_rules=input_spec.get('validation')
                )
                
                processed_inputs[input_name] = processed_input
            else:
                # Simple format - convert to type-safe
                processed_inputs[input_name] = TypeSafeInput(
                    name=input_name,
                    type=DataType.ANY,
                    default=input_spec,
                    required=False
                )
                
        return processed_inputs
        
    async def _process_type_safe_outputs(self, outputs_def: Dict[str, Any]) -> Dict[str, TypeSafeOutput]:
        """Process type-safe output definitions."""
        processed_outputs = {}
        
        for output_name, output_spec in outputs_def.items():
            if isinstance(output_spec, dict):
                # Enhanced format
                output_type = self._parse_data_type(output_spec.get('type', 'any'))
                
                processed_output = TypeSafeOutput(
                    name=output_name,
                    type=output_type,
                    description=output_spec.get('description', ''),
                    source=output_spec.get('source', output_spec.get('value', '')),
                    schema=output_spec.get('schema'),
                    format=output_spec.get('format'),
                    location=output_spec.get('location')
                )
                
                processed_outputs[output_name] = processed_output
            else:
                # Simple string format
                processed_outputs[output_name] = TypeSafeOutput(
                    name=output_name,
                    type=DataType.STRING,
                    source=str(output_spec)
                )
                
        return processed_outputs
        
    async def _process_enhanced_steps(self, steps_def: List[Dict[str, Any]]) -> List[EnhancedStep]:
        """Process enhanced step definitions with full Issue #199 support."""
        processed_steps = []
        
        for step_def in steps_def:
            # Determine step type
            step_type = self._determine_step_type(step_def)
            
            # Process step outputs with type safety
            outputs = {}
            if 'outputs' in step_def:
                for output_name, output_spec in step_def['outputs'].items():
                    if isinstance(output_spec, dict):
                        output_type = self._parse_data_type(output_spec.get('type', 'any'))
                        outputs[output_name] = TypeSafeOutput(
                            name=output_name,
                            type=output_type,
                            description=output_spec.get('description', ''),
                            schema=output_spec.get('schema'),
                            format=output_spec.get('format')
                        )
                    else:
                        outputs[output_name] = TypeSafeOutput(
                            name=output_name,
                            type=DataType.ANY,
                            description=str(output_spec)
                        )
            
            # Handle nested steps for complex control flow
            nested_steps = None
            if 'steps' in step_def and isinstance(step_def['steps'], list):
                nested_steps = await self._process_enhanced_steps(step_def['steps'])
            
            enhanced_step = EnhancedStep(
                id=step_def['id'],
                type=step_type,
                tool=step_def.get('tool'),
                action=step_def.get('action'),
                model=step_def.get('model'),
                depends_on=step_def.get('depends_on', step_def.get('dependencies', [])),
                condition=step_def.get('condition', step_def.get('if')),
                inputs=step_def.get('inputs', step_def.get('parameters', {})),
                outputs=outputs,
                items=step_def.get('items'),
                max_parallel=step_def.get('max_parallel'),
                max_iterations=step_def.get('max_iterations'),
                loop_condition=step_def.get('loop_condition'),
                goto=step_def.get('goto'),
                else_step=step_def.get('else_step'),
                steps=nested_steps,
                timeout=step_def.get('timeout'),
                retry_config=step_def.get('retry_config'),
                continue_on_error=step_def.get('continue_on_error', False),
                dynamic_routing=step_def.get('dynamic_routing')
            )
            
            processed_steps.append(enhanced_step)
            
        return processed_steps
        
    def _determine_step_type(self, step_def: Dict[str, Any]) -> StepType:
        """Determine the enhanced step type based on step definition."""
        if 'type' in step_def:
            type_str = step_def['type'].lower()
            if type_str == 'parallel_map':
                return StepType.PARALLEL_MAP
            elif type_str in ['loop', 'while']:
                return StepType.LOOP
            elif type_str == 'for':
                return StepType.FOR
            elif type_str == 'conditional':
                return StepType.CONDITIONAL
            elif type_str == 'workflow':
                return StepType.WORKFLOW
                
        # Auto-detect based on fields
        if 'items' in step_def or 'for_each' in step_def:
            return StepType.PARALLEL_MAP
        elif 'loop_condition' in step_def or 'max_iterations' in step_def:
            return StepType.LOOP
        elif 'condition' in step_def and 'else_step' in step_def:
            return StepType.CONDITIONAL
        elif 'steps' in step_def and isinstance(step_def['steps'], list):
            return StepType.WORKFLOW
        else:
            return StepType.STANDARD
            
    def _parse_data_type(self, type_spec: Union[str, Dict[str, Any]]) -> DataType:
        """Parse data type specification into DataType enum."""
        if isinstance(type_spec, dict):
            # Complex type specification - use base type
            base_type = type_spec.get('base', type_spec.get('type', 'any'))
            type_str = str(base_type).lower()
        else:
            type_str = str(type_spec).lower()
            
        return self.supported_data_types.get(type_str, DataType.ANY)
        
    async def _process_legacy_format_enhanced(self, yaml_content: Dict[str, Any]) -> EnhancedPipeline:
        """Process legacy format with enhanced features for backwards compatibility."""
        
        # Convert legacy format to enhanced format
        pipeline_id = yaml_content.get('id', yaml_content.get('name', 'unnamed_pipeline'))
        
        # Basic pipeline info
        enhanced_pipeline = EnhancedPipeline(
            id=pipeline_id,
            name=yaml_content.get('name', pipeline_id),
            description=yaml_content.get('description', ''),
            version=yaml_content.get('version', '1.0.0'),
            type='workflow'
        )
        
        # Convert parameters to type-safe inputs
        if 'parameters' in yaml_content:
            inputs = {}
            for param_name, param_value in yaml_content['parameters'].items():
                if isinstance(param_value, dict):
                    # Structured parameter definition
                    inputs[param_name] = TypeSafeInput(
                        name=param_name,
                        type=self._infer_data_type(param_value),
                        default=param_value.get('default'),
                        required=param_value.get('required', False),
                        description=param_value.get('description', '')
                    )
                else:
                    # Simple parameter - treat as default value
                    inputs[param_name] = TypeSafeInput(
                        name=param_name,
                        type=self._infer_data_type(param_value),
                        default=param_value,
                        required=False
                    )
            enhanced_pipeline.inputs = inputs
            
        # Convert outputs if present
        if 'outputs' in yaml_content:
            outputs = {}
            for output_name, output_spec in yaml_content['outputs'].items():
                outputs[output_name] = TypeSafeOutput(
                    name=output_name,
                    type=DataType.ANY,
                    source=str(output_spec)
                )
            enhanced_pipeline.outputs = outputs
            
        # Convert steps
        if 'steps' in yaml_content:
            enhanced_steps = await self._convert_legacy_steps(yaml_content['steps'])
            enhanced_pipeline.steps = enhanced_steps
            
        return enhanced_pipeline
        
    async def _convert_legacy_steps(self, legacy_steps: List[Dict[str, Any]]) -> List[EnhancedStep]:
        """Convert legacy step format to enhanced format."""
        enhanced_steps = []
        
        for step_def in legacy_steps:
            # Determine enhanced step type
            step_type = self._determine_step_type(step_def)
            
            enhanced_step = EnhancedStep(
                id=step_def['id'],
                type=step_type,
                tool=step_def.get('tool'),
                action=step_def.get('action'),
                depends_on=step_def.get('dependencies', step_def.get('depends_on', [])),
                condition=step_def.get('condition'),
                inputs=step_def.get('parameters', step_def.get('inputs', {})),
                continue_on_error=step_def.get('continue_on_error', False)
            )
            
            enhanced_steps.append(enhanced_step)
            
        return enhanced_steps
        
    def _infer_data_type(self, value: Any) -> DataType:
        """Infer data type from a value."""
        if isinstance(value, dict) and 'type' in value:
            return self._parse_data_type(value['type'])
        elif isinstance(value, str):
            return DataType.STRING
        elif isinstance(value, int):
            return DataType.INTEGER
        elif isinstance(value, float):
            return DataType.FLOAT
        elif isinstance(value, bool):
            return DataType.BOOLEAN
        elif isinstance(value, list):
            return DataType.ARRAY
        elif isinstance(value, dict):
            return DataType.OBJECT
        else:
            return DataType.ANY
            
    async def _validate_enhanced_pipeline(self, pipeline: EnhancedPipeline) -> None:
        """Comprehensive validation of enhanced pipeline."""
        
        # Validate basic pipeline structure
        if not pipeline.id:
            raise ValidationError("Pipeline must have an ID")
            
        # Validate inputs
        await self._validate_type_safe_inputs(pipeline.inputs)
        
        # Validate outputs
        await self._validate_type_safe_outputs(pipeline.outputs)
        
        # Validate steps
        await self._validate_enhanced_steps(pipeline.steps + pipeline.advanced_steps)
        
        # Validate step dependencies
        await self._validate_step_dependencies(pipeline.steps + pipeline.advanced_steps)
        
        logger.debug(f"Enhanced pipeline validation passed for: {pipeline.id}")
        
    async def _validate_type_safe_inputs(self, inputs: Dict[str, TypeSafeInput]) -> None:
        """Validate type-safe input definitions."""
        for input_name, input_def in inputs.items():
            # Validate required inputs have no default
            if input_def.required and input_def.default is not None:
                logger.warning(f"Required input '{input_name}' has default value - will be treated as optional")
                
            # Validate enum constraints
            if input_def.enum and input_def.default and input_def.default not in input_def.enum:
                raise ValidationError(f"Input '{input_name}' default value not in enum: {input_def.enum}")
                
            # Validate range constraints
            if input_def.range and input_def.default is not None:
                if input_def.type in [DataType.INTEGER, DataType.FLOAT]:
                    if len(input_def.range) == 2:
                        min_val, max_val = input_def.range
                        if not (min_val <= input_def.default <= max_val):
                            raise ValidationError(f"Input '{input_name}' default value outside range: {input_def.range}")
                            
    async def _validate_type_safe_outputs(self, outputs: Dict[str, TypeSafeOutput]) -> None:
        """Validate type-safe output definitions."""
        for output_name, output_def in outputs.items():
            # Validate source template
            if not output_def.source:
                logger.warning(f"Output '{output_name}' has no source defined")
                
            # Validate file outputs have location  
            if output_def.type == DataType.FILE and not output_def.location:
                logger.warning(f"File output '{output_name}' should specify location")
                
    async def _validate_enhanced_steps(self, steps: List[EnhancedStep]) -> None:
        """Validate enhanced step definitions."""
        step_ids = set()
        
        for step in steps:
            # Check for duplicate IDs
            if step.id in step_ids:
                raise ValidationError(f"Duplicate step ID: {step.id}")
            step_ids.add(step.id)
            
            # Validate step has either tool or action (except for control flow steps)
            if step.type in [StepType.PARALLEL_MAP, StepType.LOOP, StepType.WHILE, StepType.FOR, StepType.CONDITIONAL]:
                # Control flow steps may not need tool/action if they have nested steps
                if not step.tool and not step.action and not step.steps:
                    raise ValidationError(f"Control flow step '{step.id}' must specify either tool/action or nested steps")
            else:
                # Regular steps must have tool or action
                if not step.tool and not step.action:
                    raise ValidationError(f"Step '{step.id}' must specify either tool or action")
                
            # Validate parallel_map steps
            if step.type == StepType.PARALLEL_MAP:
                if not step.items:
                    raise ValidationError(f"Parallel map step '{step.id}' must specify items")
                    
            # Validate loop steps
            if step.type in [StepType.LOOP, StepType.WHILE]:
                if not step.loop_condition and not step.max_iterations:
                    raise ValidationError(f"Loop step '{step.id}' must specify loop_condition or max_iterations")
                    
            # Validate nested steps
            if step.steps:
                await self._validate_enhanced_steps(step.steps)
                
    async def _validate_step_dependencies(self, steps: List[EnhancedStep]) -> None:
        """Validate step dependencies are valid."""
        step_ids = {step.id for step in steps}
        
        for step in steps:
            for dep_id in step.depends_on:
                if dep_id not in step_ids:
                    raise ValidationError(f"Step '{step.id}' depends on undefined step: {dep_id}")
                    
    async def _apply_intelligent_defaults(self, pipeline: EnhancedPipeline) -> EnhancedPipeline:
        """Apply intelligent defaults as outlined in Issue #199."""
        
        # Auto-detect missing output definitions
        for step in pipeline.steps + pipeline.advanced_steps:
            if not step.outputs and (step.tool or step.action):
                # Create intelligent default outputs
                step.outputs = {
                    'result': TypeSafeOutput(
                        name='result', 
                        type=DataType.ANY,
                        description=f"Output from {step.tool or step.action}"
                    )
                }
                
        # Auto-infer pipeline outputs if not specified
        if not pipeline.outputs and pipeline.steps:
            # Use the last step's outputs as pipeline outputs
            last_step = pipeline.steps[-1]
            if last_step.outputs:
                pipeline.outputs = {
                    f"final_{output_name}": TypeSafeOutput(
                        name=f"final_{output_name}",
                        type=output.type,
                        source=f"{{{{ {last_step.id}.{output_name} }}}}"
                    )
                    for output_name, output in last_step.outputs.items()
                }
                
        logger.debug(f"Applied intelligent defaults to pipeline: {pipeline.id}")
        return pipeline


# Export key classes for external use
__all__ = [
    'EnhancedYAMLProcessor',
    'EnhancedPipeline', 
    'EnhancedStep',
    'TypeSafeInput',
    'TypeSafeOutput',
    'StepType',
    'DataType'
]