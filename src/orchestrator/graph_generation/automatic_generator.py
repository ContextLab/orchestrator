"""
AutomaticGraphGenerator - Core engine for converting declarative pipelines to LangGraph.

This module implements the main graph generation system from Issue #200, which enables
users to specify steps + dependencies in declarative syntax and automatically generates
optimized LangGraph StateGraph structures.

Key Features:
- Zero graph knowledge required from users
- Automatic dependency resolution (explicit and implicit)
- Intelligent parallel execution detection
- Advanced control flow support (conditions, loops, parallel_map)
- Type-safe data flow validation
- Integration with AutoDebugger for self-healing
- NO MOCK implementations - all real execution
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..core.exceptions import GraphGenerationError, CircularDependencyError
from ..models.model_registry import ModelRegistry
from .syntax_parser import DeclarativeSyntaxParser, ParsedPipeline
from .enhanced_yaml_processor import (
    EnhancedYAMLProcessor, EnhancedPipeline, EnhancedStep,
    TypeSafeInput, TypeSafeOutput, StepType, DataType
)
from .dependency_resolver import EnhancedDependencyResolver, DependencyGraph
from .parallel_detector import ParallelExecutionDetector, ParallelGroup
from .control_flow_analyzer import ControlFlowAnalyzer, ControlFlowMap
from .data_flow_validator import DataFlowValidator, DataFlowSchema
from .state_graph_constructor import StateGraphConstructor
from .auto_debugger import AutoDebugger, AutoDebugResult

if TYPE_CHECKING:
    from langgraph.graph import StateGraph

logger = logging.getLogger(__name__)


class AutomaticGraphGenerator:
    """
    Converts declarative pipeline definitions into optimized LangGraph StateGraph structures.
    Implements the vision from Issue #199 where users specify steps + dependencies 
    and the system auto-generates optimal execution graphs.
    
    This class serves as the main entry point for the automatic graph generation system,
    coordinating all analysis components to produce optimized execution graphs.
    """
    
    def __init__(self, 
                 model_registry: Optional[ModelRegistry] = None,
                 tool_registry: Optional[Any] = None,  # UniversalToolRegistry from Issue #203
                 auto_debugger: Optional[Any] = None):  # AutoDebugger from Issue #201
        """
        Initialize automatic graph generator with ecosystem integration.
        
        Args:
            model_registry: Model registry for intelligent model selection
            tool_registry: Universal tool registry with MCP integration
            auto_debugger: AutoDebugger for self-healing capabilities
        """
        self.model_registry = model_registry
        self.tool_registry = tool_registry
        self.auto_debugger = auto_debugger or AutoDebugger(
            model_registry=model_registry,
            tool_registry=tool_registry
        )
        
        # Initialize core analysis components
        self.syntax_parser = DeclarativeSyntaxParser()
        self.enhanced_yaml_processor = EnhancedYAMLProcessor()  # Issue #199 support
        self.dependency_resolver = EnhancedDependencyResolver()
        self.parallel_detector = ParallelExecutionDetector()
        self.control_flow_analyzer = ControlFlowAnalyzer()
        self.data_flow_validator = DataFlowValidator()
        self.state_graph_constructor = StateGraphConstructor(
            model_registry=model_registry,
            tool_registry=tool_registry
        )
        
        # Performance tracking
        self._generation_stats = {
            "total_generations": 0,
            "successful_generations": 0,
            "average_generation_time": 0.0,
            "cache_hits": 0
        }
        
        # Simple caching for repeated pipeline definitions
        self._pipeline_cache: Dict[str, StateGraph] = {}
        
        logger.info("AutomaticGraphGenerator initialized with ecosystem integrations")
        
    async def generate_graph(self, 
                            pipeline_def: Dict[str, Any], 
                            context: Optional[Dict[str, Any]] = None,
                            use_cache: bool = True) -> StateGraph:
        """
        Main entry point - converts declarative YAML to optimized LangGraph.
        
        This method implements the complete pipeline-to-graph transformation:
        1. Parse declarative syntax (supports Issue #199 enhanced format)
        2. Resolve dependencies (explicit and implicit)  
        3. Detect parallel execution opportunities
        4. Analyze control flow (conditions, loops, etc.)
        5. Validate data flow and type safety
        6. Generate optimized StateGraph
        7. Integrate self-healing capabilities
        
        Args:
            pipeline_def: Pipeline definition dictionary from YAML
            context: Optional context for pipeline execution
            use_cache: Whether to use cached results for identical pipelines
            
        Returns:
            Optimized LangGraph StateGraph ready for execution
            
        Raises:
            GraphGenerationError: If graph generation fails
            CircularDependencyError: If circular dependencies are detected
        """
        start_time = time.time()
        self._generation_stats["total_generations"] += 1
        
        try:
            # Generate cache key for identical pipeline definitions
            cache_key = self._generate_cache_key(pipeline_def) if use_cache else None
            
            if cache_key and cache_key in self._pipeline_cache:
                self._generation_stats["cache_hits"] += 1
                logger.info(f"Using cached graph for pipeline {pipeline_def.get('id', 'unknown')}")
                return self._pipeline_cache[cache_key]
            
            logger.info(f"Starting graph generation for pipeline: {pipeline_def.get('id', 'unknown')}")
            
            # PHASE 1: Parse and validate declarative syntax (with Issue #199 support)
            logger.debug("Phase 1: Parsing declarative syntax")
            if self._is_enhanced_yaml_format(pipeline_def):
                logger.debug("Detected Issue #199 enhanced YAML format")
                enhanced_pipeline = await self.enhanced_yaml_processor.process_enhanced_yaml(pipeline_def)
                parsed_pipeline = await self._convert_enhanced_to_parsed(enhanced_pipeline)
            else:
                logger.debug("Using legacy format processing")
                parsed_pipeline = await self._parse_declarative_syntax(pipeline_def)
            
            # PHASE 2: Advanced dependency analysis
            logger.debug("Phase 2: Analyzing dependencies")
            dependency_graph = await self._analyze_dependencies(parsed_pipeline)
            
            # PHASE 3: Detect parallel execution opportunities  
            logger.debug("Phase 3: Detecting parallel execution opportunities")
            parallel_groups = await self._detect_parallelization(dependency_graph)
            
            # PHASE 4: Control flow analysis (loops, conditions, goto)
            logger.debug("Phase 4: Analyzing control flow")
            control_flow_map = await self._analyze_control_flow(parsed_pipeline)
            
            # PHASE 5: Data flow validation and type checking
            logger.debug("Phase 5: Validating data flow")
            data_flow_schema = await self._validate_data_flow(parsed_pipeline)
            
            # PHASE 6: Generate optimized StateGraph
            logger.debug("Phase 6: Constructing optimized StateGraph")
            optimized_graph = await self._construct_state_graph(
                dependency_graph=dependency_graph,
                parallel_groups=parallel_groups,
                control_flow=control_flow_map,
                data_schema=data_flow_schema,
                original_pipeline=parsed_pipeline
            )
            
            # PHASE 7: Integration with AutoDebugger for self-healing
            if self.auto_debugger:
                logger.debug("Phase 7: Integrating self-healing capabilities")
                optimized_graph = await self._integrate_auto_debugging(optimized_graph)
            
            # Cache successful generation
            if cache_key:
                self._pipeline_cache[cache_key] = optimized_graph
                
            # Update performance stats
            generation_time = time.time() - start_time
            self._update_generation_stats(generation_time, success=True)
            
            logger.info(f"Successfully generated graph in {generation_time:.3f}s for pipeline: {parsed_pipeline.id}")
            
            return optimized_graph
            
        except Exception as e:
            # Update failure stats
            generation_time = time.time() - start_time
            self._update_generation_stats(generation_time, success=False)
            
            # If generation fails, use AutoDebugger to self-correct
            if self.auto_debugger:
                logger.warning(f"Graph generation failed, attempting auto-correction: {e}")
                try:
                    return await self._auto_fix_generation_failure(pipeline_def, e, context)
                except Exception as auto_fix_error:
                    logger.error(f"AutoDebugger failed to fix generation error: {auto_fix_error}")
                    raise GraphGenerationError(
                        f"Graph generation failed and auto-correction unsuccessful: {e}"
                    ) from e
            else:
                logger.error(f"Graph generation failed: {e}")
                raise GraphGenerationError(f"Failed to generate graph: {e}") from e
                
    async def _parse_declarative_syntax(self, pipeline_def: Dict[str, Any]) -> ParsedPipeline:
        """Parse and validate declarative pipeline syntax."""
        try:
            return await self.syntax_parser.parse_pipeline_definition(pipeline_def)
        except Exception as e:
            raise GraphGenerationError(f"Failed to parse pipeline definition: {e}") from e
            
    async def _analyze_dependencies(self, parsed_pipeline: ParsedPipeline) -> DependencyGraph:
        """Build comprehensive dependency graph with explicit and implicit dependencies."""
        try:
            return await self.dependency_resolver.resolve_dependencies(parsed_pipeline.steps)
        except CircularDependencyError:
            # Re-raise circular dependency errors without wrapping
            raise
        except Exception as e:
            raise GraphGenerationError(f"Failed to analyze dependencies: {e}") from e
            
    async def _detect_parallelization(self, dependency_graph: DependencyGraph) -> List[ParallelGroup]:
        """Identify steps that can execute in parallel for automatic optimization."""
        try:
            return await self.parallel_detector.detect_parallel_groups(dependency_graph)
        except Exception as e:
            raise GraphGenerationError(f"Failed to detect parallel execution opportunities: {e}") from e
            
    async def _analyze_control_flow(self, parsed_pipeline: ParsedPipeline) -> ControlFlowMap:
        """Analyze advanced control flow patterns (conditions, loops, etc.)."""
        try:
            return await self.control_flow_analyzer.analyze_control_flow(parsed_pipeline)
        except Exception as e:
            raise GraphGenerationError(f"Failed to analyze control flow: {e}") from e
            
    async def _validate_data_flow(self, parsed_pipeline: ParsedPipeline) -> DataFlowSchema:
        """Validate data flow and ensure type safety."""
        try:
            return await self.data_flow_validator.validate_data_flow(parsed_pipeline)
        except Exception as e:
            raise GraphGenerationError(f"Failed to validate data flow: {e}") from e
            
    async def _construct_state_graph(self,
                                   dependency_graph: DependencyGraph,
                                   parallel_groups: List[ParallelGroup],
                                   control_flow: ControlFlowMap,
                                   data_schema: DataFlowSchema,
                                   original_pipeline: ParsedPipeline) -> StateGraph:
        """Generate optimized LangGraph StateGraph from analysis results."""
        try:
            return await self.state_graph_constructor.construct_graph(
                dependency_graph=dependency_graph,
                parallel_groups=parallel_groups,
                control_flow=control_flow,
                data_schema=data_schema,
                original_pipeline=original_pipeline
            )
        except Exception as e:
            raise GraphGenerationError(f"Failed to construct StateGraph: {e}") from e
            
    async def _integrate_auto_debugging(self, graph: StateGraph) -> StateGraph:
        """Integrate AutoDebugger self-healing capabilities into the graph."""
        try:
            # This will be implemented when AutoDebugger (Issue #201) is available
            logger.debug("AutoDebugger integration placeholder - will be implemented with Issue #201")
            return graph
        except Exception as e:
            logger.warning(f"Failed to integrate auto-debugging capabilities: {e}")
            # Don't fail graph generation if auto-debugging integration fails
            return graph
            
    async def _auto_fix_generation_failure(self,
                                         pipeline_def: Dict[str, Any],
                                         error: Exception,
                                         context: Optional[Dict[str, Any]]) -> StateGraph:
        """Use AutoDebugger to automatically fix generation failures."""
        
        # Build pipeline context for AutoDebugger
        pipeline_context = {
            'pipeline_def': pipeline_def,
            'error': str(error),
            'error_type': type(error).__name__,
            'context': context or {}
        }
        
        # Create instructions for AutoDebugger
        instructions = f"""
        Fix the pipeline graph generation failure for this pipeline definition:
        
        Pipeline ID: {pipeline_def.get('id', 'unknown')}
        Error: {str(error)}
        
        The pipeline definition that failed to generate:
        {pipeline_def}
        
        Please analyze the error and provide a corrected pipeline definition
        that will successfully generate a valid graph.
        """
        
        try:
            # Use AutoDebugger to fix the issue
            debug_result: AutoDebugResult = await self.auto_debugger.auto_debug(
                initial_instructions=instructions,
                pipeline_context=pipeline_context,
                error_context=str(error),
                available_tools=self._get_available_tool_names()
            )
            
            if debug_result.success and debug_result.final_result:
                logger.info(f"AutoDebugger successfully fixed generation failure: {debug_result.debug_summary}")
                
                # Try to parse the fixed pipeline definition
                fixed_pipeline = debug_result.final_result
                if isinstance(fixed_pipeline, dict):
                    # Recursively try to generate graph with fixed definition
                    return await self.generate_graph(fixed_pipeline, context)
                else:
                    # AutoDebugger provided a different type of fix
                    logger.warning(f"AutoDebugger provided non-dict result: {type(fixed_pipeline)}")
                    raise error
            else:
                logger.warning(f"AutoDebugger failed to fix generation failure: {debug_result.error_message}")
                raise error
                
        except Exception as debug_error:
            logger.error(f"AutoDebugger encountered error during fix attempt: {debug_error}")
            raise GraphGenerationError(f"Auto-fix failed: {debug_error}") from error
    
    def _get_available_tool_names(self) -> List[str]:
        """Get list of available tool names for AutoDebugger."""
        if self.tool_registry and hasattr(self.tool_registry, 'get_available_tools'):
            try:
                return list(self.tool_registry.get_available_tools())
            except Exception as e:
                logger.warning(f"Failed to get available tools: {e}")
                return []
        else:
            # Return common tool names as fallback
            return [
                'web-search', 'filesystem', 'headless-browser', 
                'pdf-compiler', 'terminal', 'analyze_text', 
                'generate_text', 'fact_checker'
            ]
            
    def _generate_cache_key(self, pipeline_def: Dict[str, Any]) -> str:
        """Generate cache key for pipeline definition."""
        import hashlib
        import json
        
        # Create deterministic string representation
        pipeline_str = json.dumps(pipeline_def, sort_keys=True, separators=(',', ':'))
        return hashlib.md5(pipeline_str.encode()).hexdigest()
        
    def _update_generation_stats(self, generation_time: float, success: bool) -> None:
        """Update performance statistics."""
        self._generation_stats["total_generations"] += 1
        
        if success:
            self._generation_stats["successful_generations"] += 1
            
        # Update running average of generation time
        total = self._generation_stats["total_generations"]
        current_avg = self._generation_stats["average_generation_time"]
        if total > 1:
            self._generation_stats["average_generation_time"] = (
                (current_avg * (total - 1) + generation_time) / total
            )
        else:
            self._generation_stats["average_generation_time"] = generation_time
        
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring."""
        success_rate = 0.0
        if self._generation_stats["total_generations"] > 0:
            success_rate = (
                self._generation_stats["successful_generations"] / 
                self._generation_stats["total_generations"]
            )
            
        return {
            **self._generation_stats,
            "success_rate": success_rate,
            "cache_hit_rate": (
                self._generation_stats["cache_hits"] / 
                max(1, self._generation_stats["total_generations"])
            )
        }
        
    def clear_cache(self) -> None:
        """Clear the pipeline cache."""
        self._pipeline_cache.clear()
        logger.info("Pipeline cache cleared")
        
    def _is_enhanced_yaml_format(self, pipeline_def: Dict[str, Any]) -> bool:
        """Detect if pipeline uses Issue #199 enhanced YAML format."""
        return self.enhanced_yaml_processor._is_enhanced_format(pipeline_def)
        
    async def _convert_enhanced_to_parsed(self, enhanced_pipeline: EnhancedPipeline) -> ParsedPipeline:
        """Convert enhanced pipeline format to parsed pipeline format for processing."""
        from .types import ParsedStep, InputSchema, OutputSchema
        
        # Convert enhanced steps to parsed steps
        parsed_steps = []
        
        # Process main steps
        for enhanced_step in enhanced_pipeline.steps:
            parsed_step = await self._convert_enhanced_step_to_parsed(enhanced_step)
            parsed_steps.append(parsed_step)
            
        # Process advanced steps
        for enhanced_step in enhanced_pipeline.advanced_steps:
            parsed_step = await self._convert_enhanced_step_to_parsed(enhanced_step)
            parsed_steps.append(parsed_step)
            
        # Convert inputs to dictionary of InputSchema instances
        inputs_dict = {}
        for input_name, type_safe_input in enhanced_pipeline.inputs.items():
            inputs_dict[input_name] = InputSchema(
                name=input_name,
                type=type_safe_input.type.value,
                required=type_safe_input.required,
                default=type_safe_input.default,
                description=type_safe_input.description,
                enum=type_safe_input.enum,
                range=type_safe_input.range,
                example=type_safe_input.example
            )
            
        # Convert outputs to dictionary of OutputSchema instances
        outputs_dict = {}
        for output_name, type_safe_output in enhanced_pipeline.outputs.items():
            outputs_dict[output_name] = OutputSchema(
                name=output_name,
                type=type_safe_output.type.value,
                description=type_safe_output.description,
                schema=type_safe_output.schema,
                computed_as=type_safe_output.source,
                format=type_safe_output.format
            )
            
        return ParsedPipeline(
            id=enhanced_pipeline.id,
            name=enhanced_pipeline.name or enhanced_pipeline.id,
            description=enhanced_pipeline.description,
            version=enhanced_pipeline.version,
            steps=parsed_steps,
            inputs=inputs_dict,
            outputs=outputs_dict,
            config=enhanced_pipeline.config,
            metadata=enhanced_pipeline.metadata
        )
        
    async def _convert_enhanced_step_to_parsed(self, enhanced_step: EnhancedStep) -> ParsedStep:
        """Convert enhanced step to parsed step format."""
        from .types import ParsedStep, StepType as ParsedStepType
        
        # Convert step type
        if enhanced_step.type == StepType.PARALLEL_MAP:
            step_type = ParsedStepType.PARALLEL_MAP
        elif enhanced_step.type in [StepType.LOOP, StepType.WHILE, StepType.FOR]:
            step_type = ParsedStepType.LOOP
        elif enhanced_step.type == StepType.CONDITIONAL:
            step_type = ParsedStepType.CONDITIONAL
        else:
            step_type = ParsedStepType.STANDARD
            
        # Convert outputs to simple dictionary
        outputs = {}
        for output_name, type_safe_output in enhanced_step.outputs.items():
            outputs[output_name] = type_safe_output.description or f"Output from {enhanced_step.id}"
            
        # Convert nested steps if present
        substeps = None
        if enhanced_step.steps:
            substeps = []
            for nested_step in enhanced_step.steps:
                parsed_nested_step = await self._convert_enhanced_step_to_parsed(nested_step)
                substeps.append(parsed_nested_step)

        return ParsedStep(
            id=enhanced_step.id,
            type=step_type,
            tool=enhanced_step.tool,
            action=enhanced_step.action,
            model_requirements=enhanced_step.model,
            inputs=enhanced_step.inputs,
            outputs=outputs,
            depends_on=enhanced_step.depends_on,
            condition=enhanced_step.condition,
            items=enhanced_step.items,
            substeps=substeps,
            max_iterations=enhanced_step.max_iterations,
            goto=enhanced_step.goto,
            else_step=enhanced_step.else_step
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"AutomaticGraphGenerator("
            f"model_registry={self.model_registry is not None}, "
            f"tool_registry={self.tool_registry is not None}, "
            f"auto_debugger={self.auto_debugger is not None}, "
            f"enhanced_yaml_support=True, "
            f"total_generations={self._generation_stats['total_generations']}"
            f")"
        )