"""Enhanced YAML compiler with automatic graph generation support.

This compiler extends the existing YAMLCompiler to support the automatic graph generation
system from Issue #200. It detects enhanced YAML syntax from Issue #199 and uses the
AutomaticGraphGenerator to create optimized execution graphs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .yaml_compiler import YAMLCompiler
from ..core.pipeline import Pipeline
from ..core.exceptions import YAMLCompilerError
from ..graph_generation.automatic_generator import AutomaticGraphGenerator
from ..graph_generation.enhanced_yaml_processor import EnhancedYAMLProcessor

logger = logging.getLogger(__name__)


class EnhancedYAMLCompiler(YAMLCompiler):
    """
    Enhanced YAML compiler with automatic graph generation support.
    
    This compiler maintains full backwards compatibility with existing YAML pipelines
    while adding support for:
    - Issue #199 enhanced declarative syntax
    - Automatic graph generation via AutomaticGraphGenerator
    - Type-safe input/output validation
    - Advanced control flow constructs
    """
    
    def __init__(self, 
                 automatic_graph_generator: Optional[AutomaticGraphGenerator] = None,
                 enable_auto_generation: bool = True,
                 **kwargs):
        """
        Initialize enhanced YAML compiler.
        
        Args:
            automatic_graph_generator: AutomaticGraphGenerator instance
            enable_auto_generation: Whether to enable automatic graph generation
            **kwargs: Arguments passed to parent YAMLCompiler
        """
        super().__init__(**kwargs)
        
        # Initialize graph generation components
        self.enable_auto_generation = enable_auto_generation
        self.automatic_graph_generator = automatic_graph_generator or AutomaticGraphGenerator(
            model_registry=kwargs.get('model_registry')
        )
        self.enhanced_yaml_processor = EnhancedYAMLProcessor()
        
        logger.info(f"EnhancedYAMLCompiler initialized with auto-generation: {enable_auto_generation}")
        
    async def compile(self,
                     yaml_content: str,
                     context: Optional[Dict[str, Any]] = None,
                     resolve_ambiguities: bool = True,
                     use_auto_generation: Optional[bool] = None) -> Pipeline:
        """
        Compile YAML content to Pipeline object with optional automatic graph generation.
        
        Args:
            yaml_content: YAML content as string
            context: Template context variables
            resolve_ambiguities: Whether to resolve AUTO tags
            use_auto_generation: Override auto-generation setting for this compilation
            
        Returns:
            Compiled Pipeline object with optional StateGraph integration
            
        Raises:
            YAMLCompilerError: If compilation fails
        """
        try:
            # Determine if we should use automatic graph generation
            should_use_auto_gen = (
                use_auto_generation if use_auto_generation is not None 
                else self.enable_auto_generation
            )
            
            # Step 1: Process file inclusions and parse YAML
            processed_yaml_content = await self._process_file_inclusions(yaml_content)
            raw_pipeline = self._parse_yaml(processed_yaml_content)
            
            # Step 2: Check if this is enhanced YAML format
            is_enhanced_format = (
                should_use_auto_gen and 
                self._is_enhanced_yaml_format(raw_pipeline)
            )
            
            if is_enhanced_format:
                logger.info(f"Detected enhanced YAML format for pipeline: {raw_pipeline.get('id', 'unknown')}")
                return await self._compile_with_auto_generation(
                    raw_pipeline, context, resolve_ambiguities
                )
            else:
                logger.info(f"Using legacy compilation for pipeline: {raw_pipeline.get('id', 'unknown')}")
                return await self._compile_legacy_format(
                    raw_pipeline, context, resolve_ambiguities
                )
                
        except Exception as e:
            logger.error(f"Enhanced YAML compilation failed: {e}")
            raise YAMLCompilerError(f"Failed to compile enhanced YAML: {e}") from e
            
    async def _compile_with_auto_generation(self,
                                          raw_pipeline: Dict[str, Any],
                                          context: Optional[Dict[str, Any]],
                                          resolve_ambiguities: bool) -> Pipeline:
        """
        Compile enhanced YAML format using automatic graph generation.
        """
        logger.debug("Compiling with automatic graph generation")
        
        # Step 1: Validate enhanced format
        self.schema_validator.validate(raw_pipeline)
        
        # Step 2: Process enhanced YAML through EnhancedYAMLProcessor
        enhanced_pipeline = await self.enhanced_yaml_processor.process_enhanced_yaml(raw_pipeline)
        
        # Step 3: Generate optimized StateGraph
        state_graph = await self.automatic_graph_generator.generate_graph(
            raw_pipeline, context=context
        )
        
        # Step 4: Create Pipeline object with StateGraph integration
        pipeline = await self._create_pipeline_from_enhanced(
            enhanced_pipeline, raw_pipeline, context, resolve_ambiguities
        )
        
        # Step 5: Attach StateGraph to pipeline for execution
        pipeline.metadata['state_graph'] = state_graph
        pipeline.metadata['graph_generation_stats'] = self.automatic_graph_generator.get_generation_stats()
        pipeline.metadata['compilation_method'] = 'automatic_graph_generation'
        
        logger.info(f"Successfully compiled enhanced pipeline with automatic graph generation: {pipeline.id}")
        return pipeline
        
    async def _compile_legacy_format(self,
                                   raw_pipeline: Dict[str, Any],
                                   context: Optional[Dict[str, Any]],
                                   resolve_ambiguities: bool) -> Pipeline:
        """
        Compile legacy format using parent YAMLCompiler logic.
        """
        logger.debug("Compiling legacy format")
        
        # Use parent class logic for legacy compilation
        # Step 3: Validate against schema
        self.schema_validator.validate(raw_pipeline)
        
        # Step 4: Validate error handling configurations
        error_issues = self.error_handler_validator.validate_pipeline_error_handling(raw_pipeline)
        if error_issues:
            raise YAMLCompilerError(f"Error handler validation failed: {'; '.join(error_issues)}")

        # Step 5: Merge default values with context
        merged_context = self._merge_defaults_with_context(
            raw_pipeline, context or {}
        )

        # Step 6: Process templates
        processed = self._process_templates(raw_pipeline, merged_context)

        # Step 7: Detect and resolve ambiguities
        if resolve_ambiguities:
            resolved = await self._resolve_ambiguities(processed)
        else:
            resolved = processed

        # Step 8: Create pipeline object from processed YAML
        pipeline = await self._create_legacy_pipeline_from_yaml(resolved, merged_context)
        
        # Mark as legacy compilation
        pipeline.metadata['compilation_method'] = 'legacy'
        
        logger.info(f"Successfully compiled legacy pipeline: {pipeline.id}")
        return pipeline
        
    async def _create_legacy_pipeline_from_yaml(self,
                                              yaml_data: Dict[str, Any],
                                              context: Dict[str, Any]) -> Pipeline:
        """
        Create Pipeline object from legacy YAML format.
        """
        from ..core.task import Task
        
        # Extract basic pipeline info
        pipeline_id = yaml_data.get('id', yaml_data.get('name', 'unknown_pipeline'))
        pipeline_name = yaml_data.get('name', pipeline_id)
        pipeline_desc = yaml_data.get('description', '')
        pipeline_version = yaml_data.get('version', '1.0.0')
        
        # Create pipeline
        pipeline = Pipeline(
            id=pipeline_id,
            name=pipeline_name,
            description=pipeline_desc,
            version=pipeline_version,
            context=context
        )
        
        # Add tasks from steps
        steps = yaml_data.get('steps', [])
        for step_data in steps:
            task = Task(
                id=step_data['id'],
                name=step_data.get('name', step_data['id']),
                action=step_data.get('action', 'unknown_action'),
                parameters=step_data.get('parameters', {}),
                dependencies=step_data.get('dependencies', [])
            )
            
            # Store original step data in metadata
            task.metadata['original_step_data'] = step_data
            
            pipeline.add_task(task)
            
        return pipeline
        
    async def _create_pipeline_from_enhanced(self,
                                           enhanced_pipeline,
                                           raw_pipeline: Dict[str, Any],
                                           context: Optional[Dict[str, Any]],
                                           resolve_ambiguities: bool) -> Pipeline:
        """
        Create Pipeline object from enhanced pipeline format.
        """
        from ..core.task import Task
        
        # Create pipeline object
        pipeline = Pipeline(
            id=enhanced_pipeline.id,
            name=enhanced_pipeline.name or enhanced_pipeline.id,
            context=context or {},
            metadata=enhanced_pipeline.metadata,
            version=enhanced_pipeline.version,
            description=enhanced_pipeline.description
        )
        
        # Convert enhanced steps to Task objects
        all_steps = enhanced_pipeline.steps + enhanced_pipeline.advanced_steps
        
        for step in all_steps:
            # Convert enhanced step to task
            task = await self._convert_enhanced_step_to_task(step, context, resolve_ambiguities)
            pipeline.add_task(task)
            
        # Process advanced configurations
        if enhanced_pipeline.config:
            pipeline.metadata.update(enhanced_pipeline.config)
            
        # Store enhanced format metadata
        pipeline.metadata['enhanced_syntax'] = True
        pipeline.metadata['type_safe_inputs'] = len(enhanced_pipeline.inputs)
        pipeline.metadata['type_safe_outputs'] = len(enhanced_pipeline.outputs)
        pipeline.metadata['advanced_steps'] = len(enhanced_pipeline.advanced_steps)
        
        return pipeline
        
    async def _convert_enhanced_step_to_task(self, 
                                           enhanced_step, 
                                           context: Optional[Dict[str, Any]],
                                           resolve_ambiguities: bool) -> Task:
        """
        Convert enhanced step format to Task object.
        """
        from ..core.task import Task
        
        # Build task parameters - Task constructor expects specific fields
        task_params = {
            'id': enhanced_step.id,
            'name': enhanced_step.id,  # Use id as name if no specific name
            'action': enhanced_step.action or enhanced_step.tool or 'unknown_action',
            'parameters': enhanced_step.inputs,
            'dependencies': enhanced_step.depends_on,
        }
        
        # Create task first, then add metadata
        task = Task(**task_params)
        
        # Store enhanced step information in metadata
        task.metadata['enhanced_step'] = True
        task.metadata['step_type'] = enhanced_step.type.value
        task.metadata['original_tool'] = enhanced_step.tool
        task.metadata['condition'] = enhanced_step.condition
        task.metadata['continue_on_error'] = enhanced_step.continue_on_error
        
        # Handle model requirements
        if enhanced_step.model:
            if resolve_ambiguities and self.ambiguity_resolver:
                # Resolve model using AUTO tag logic
                auto_tag_content = f"<AUTO task=\"{enhanced_step.action or 'general'}\">{enhanced_step.model}</AUTO>"
                resolved_model = await self.ambiguity_resolver.resolve_auto_tag_async(auto_tag_content)
                task.metadata['resolved_model'] = resolved_model
            else:
                task.metadata['model'] = enhanced_step.model
                
        # Handle control flow parameters in metadata
        if enhanced_step.items:
            task.metadata['items'] = enhanced_step.items
        if enhanced_step.max_iterations:
            task.metadata['max_iterations'] = enhanced_step.max_iterations
        if enhanced_step.goto:
            task.metadata['goto'] = enhanced_step.goto
            
        # Store output schema information
        if enhanced_step.outputs:
            task.metadata['output_schema'] = {
                name: {
                    'type': output.type.value,
                    'description': output.description
                }
                for name, output in enhanced_step.outputs.items()
            }
            
        return task
        
    def _is_enhanced_yaml_format(self, yaml_data: Dict[str, Any]) -> bool:
        """
        Detect if YAML uses enhanced Issue #199 syntax.
        """
        return self.enhanced_yaml_processor._is_enhanced_format(yaml_data)
        
    def get_generation_stats(self) -> Dict[str, Any]:
        """
        Get automatic graph generation statistics.
        """
        return self.automatic_graph_generator.get_generation_stats()
        
    def clear_generation_cache(self) -> None:
        """
        Clear the automatic graph generation cache.
        """
        self.automatic_graph_generator.clear_cache()
        logger.info("Automatic graph generation cache cleared")


# Alias for backwards compatibility
AutoGraphYAMLCompiler = EnhancedYAMLCompiler