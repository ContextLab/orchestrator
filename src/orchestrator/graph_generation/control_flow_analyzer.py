"""
ControlFlowAnalyzer - Handles advanced control flow patterns.

This module analyzes and processes control flow patterns from Issue #199 including:
- Conditional execution with automatic edge creation
- Loop structures with termination conditions  
- Goto statements for complex control flow
- Dynamic routing based on runtime conditions
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Callable, Any
from jinja2 import Template

from .types import (
    ParsedPipeline, ParsedStep, ControlFlowMap, ConditionalLogic, 
    LoopLogic, ParallelMapLogic, StepType
)
from .syntax_parser import DeclarativeSyntaxParser

logger = logging.getLogger(__name__)


class ControlFlowAnalyzer:
    """
    Handles all control flow patterns mentioned in Issue #199:
    - Conditional routing with automatic edge creation
    - Loop structures with termination conditions  
    - Goto statements for complex control flow
    - Dynamic routing based on runtime conditions
    """
    
    def __init__(self):
        self.syntax_parser = DeclarativeSyntaxParser()
        logger.info("ControlFlowAnalyzer initialized")
        
    async def analyze_control_flow(self, parsed_pipeline: ParsedPipeline) -> ControlFlowMap:
        """Comprehensive control flow analysis and optimization."""
        logger.debug(f"Analyzing control flow for pipeline: {parsed_pipeline.id}")
        
        control_map = ControlFlowMap()
        
        for step in parsed_pipeline.steps:
            # CONDITIONAL EXECUTION
            if step.condition:
                conditional_logic = await self._parse_conditional_logic(step)
                control_map.add_conditional(step.id, conditional_logic)
                
            # LOOP STRUCTURES  
            if step.type in [StepType.LOOP, StepType.WHILE, StepType.FOR]:
                loop_logic = await self._parse_loop_structure(step)
                control_map.add_loop(step.id, loop_logic)
                
            # PARALLEL MAP (from new syntax)
            if step.type == StepType.PARALLEL_MAP:
                parallel_logic = await self._parse_parallel_map(step)
                control_map.add_parallel_map(step.id, parallel_logic)
                
            # GOTO STATEMENTS
            if step.goto:
                control_map.add_goto(step.id, step.goto)
                
        logger.info(f"Analyzed control flow: {len(control_map.conditionals)} conditionals, "
                   f"{len(control_map.loops)} loops, {len(control_map.parallel_maps)} parallel maps")
        
        return control_map
        
    async def _parse_conditional_logic(self, step: ParsedStep) -> ConditionalLogic:
        """
        Parse conditional execution logic into LangGraph conditional edges.
        Example: condition: "{{ analyze_results.needs_verification }}"
        """
        condition_template = step.condition
        
        # Analyze template variables in condition
        template_vars = self.syntax_parser.extract_template_variables(condition_template)
        required_data = [var.split('.')[0] for var in template_vars]
        
        # Create conditional edge function
        async def condition_evaluator(state: Dict[str, Any]) -> bool:
            """Real-time condition evaluation using Jinja2"""
            try:
                template = Template(condition_template)
                result = template.render(**state)
                
                # Convert string result to boolean
                return self._evaluate_condition_result(result)
                
            except Exception as e:
                logger.error(f"Failed to evaluate condition '{condition_template}': {e}")
                return False
                
        return ConditionalLogic(
            condition_template=condition_template,
            evaluator_function=condition_evaluator,
            required_data=required_data,
            true_path=step.id,
            false_path=step.else_step
        )
        
    async def _parse_loop_structure(self, step: ParsedStep) -> LoopLogic:
        """Parse loop execution logic."""
        return LoopLogic(
            loop_type=step.type,
            condition_template=step.condition,
            max_iterations=step.max_iterations or 100,
            substeps=step.substeps or []
        )
        
    async def _parse_parallel_map(self, step: ParsedStep) -> ParallelMapLogic:
        """Parse parallel map execution logic.""" 
        return ParallelMapLogic(
            items_expression=step.items or "[]",
            item_variable_name="item",
            substeps=step.substeps or [],
            max_concurrency=getattr(step, 'max_concurrency', None)
        )
        
    def _evaluate_condition_result(self, result: str) -> bool:
        """Convert condition evaluation result to boolean."""
        if isinstance(result, bool):
            return result
        elif isinstance(result, str):
            result_lower = result.lower().strip()
            return result_lower in ['true', 'yes', '1', 'on', 'enabled']
        else:
            return bool(result)
            
    async def analyze_complex_control_patterns(self, parsed_pipeline: ParsedPipeline) -> Dict[str, Any]:
        """
        Analyze complex control flow patterns for advanced optimization.
        Identifies patterns like:
        - Nested conditionals that can be optimized
        - Loop structures with early exit conditions
        - Fan-out/fan-in patterns
        - State machine patterns
        """
        patterns = {
            'nested_conditionals': [],
            'early_exit_loops': [],
            'fan_out_patterns': [],
            'fan_in_patterns': [],
            'state_machines': [],
            'retry_patterns': []
        }
        
        # Analyze each pattern type
        patterns['nested_conditionals'] = await self._detect_nested_conditionals(parsed_pipeline)
        patterns['early_exit_loops'] = await self._detect_early_exit_loops(parsed_pipeline)
        patterns['fan_out_patterns'] = await self._detect_fan_out_patterns(parsed_pipeline)
        patterns['fan_in_patterns'] = await self._detect_fan_in_patterns(parsed_pipeline)
        patterns['state_machines'] = await self._detect_state_machines(parsed_pipeline)
        patterns['retry_patterns'] = await self._detect_retry_patterns(parsed_pipeline)
        
        logger.info(f"Complex pattern analysis complete: {sum(len(p) for p in patterns.values())} patterns detected")
        return patterns
        
    async def _detect_nested_conditionals(self, pipeline: ParsedPipeline) -> List[Dict[str, Any]]:
        """Detect nested conditional patterns that can be optimized."""
        nested_patterns = []
        
        for step in pipeline.steps:
            if step.condition and step.else_step:
                # Look for else_step that also has conditions
                else_step = next((s for s in pipeline.steps if s.id == step.else_step), None)
                if else_step and else_step.condition:
                    pattern = {
                        'type': 'nested_conditional',
                        'root_step': step.id,
                        'conditions': [step.condition, else_step.condition],
                        'optimization_potential': 'high'
                    }
                    nested_patterns.append(pattern)
                    
        return nested_patterns
        
    async def _detect_early_exit_loops(self, pipeline: ParsedPipeline) -> List[Dict[str, Any]]:
        """Detect loops with early exit conditions."""
        early_exit_patterns = []
        
        for step in pipeline.steps:
            if step.type in [StepType.LOOP, StepType.WHILE] and step.substeps:
                # Look for break/continue conditions in substeps
                for substep in step.substeps:
                    if substep.condition and ('break' in substep.condition.lower() or 
                                            'exit' in substep.condition.lower() or
                                            'return' in substep.condition.lower()):
                        pattern = {
                            'type': 'early_exit_loop',
                            'loop_step': step.id,
                            'exit_condition': substep.condition,
                            'optimization_potential': 'medium'
                        }
                        early_exit_patterns.append(pattern)
                        break
                        
        return early_exit_patterns
        
    async def _detect_fan_out_patterns(self, pipeline: ParsedPipeline) -> List[Dict[str, Any]]:
        """Detect fan-out patterns where one step feeds multiple parallel steps."""
        fan_out_patterns = []
        
        # Build dependency map
        dependents = {}
        for step in pipeline.steps:
            for dep in step.depends_on:
                if dep not in dependents:
                    dependents[dep] = []
                dependents[dep].append(step.id)
                
        # Find steps that feed 3+ other steps
        for source_step, dependent_steps in dependents.items():
            if len(dependent_steps) >= 3:
                # Check if dependents can run in parallel
                pattern = {
                    'type': 'fan_out',
                    'source_step': source_step,
                    'target_steps': dependent_steps,
                    'parallelization_opportunity': True,
                    'optimization_potential': 'high'
                }
                fan_out_patterns.append(pattern)
                
        return fan_out_patterns
        
    async def _detect_fan_in_patterns(self, pipeline: ParsedPipeline) -> List[Dict[str, Any]]:
        """Detect fan-in patterns where multiple steps feed into one step."""
        fan_in_patterns = []
        
        for step in pipeline.steps:
            if len(step.depends_on) >= 3:
                pattern = {
                    'type': 'fan_in',
                    'target_step': step.id,
                    'source_steps': step.depends_on,
                    'synchronization_point': True,
                    'optimization_potential': 'medium'
                }
                fan_in_patterns.append(pattern)
                
        return fan_in_patterns
        
    async def _detect_state_machines(self, pipeline: ParsedPipeline) -> List[Dict[str, Any]]:
        """Detect state machine patterns with goto statements."""
        state_machine_patterns = []
        
        goto_steps = [step for step in pipeline.steps if step.goto]
        if len(goto_steps) >= 2:
            # Analyze goto relationships
            state_transitions = {}
            for step in goto_steps:
                if step.condition:
                    state_transitions[step.id] = {
                        'condition': step.condition,
                        'next_state': step.goto,
                        'current_state': step.id
                    }
                    
            if state_transitions:
                pattern = {
                    'type': 'state_machine',
                    'states': list(state_transitions.keys()),
                    'transitions': state_transitions,
                    'optimization_potential': 'low'  # State machines are usually already optimized
                }
                state_machine_patterns.append(pattern)
                
        return state_machine_patterns
        
    async def _detect_retry_patterns(self, pipeline: ParsedPipeline) -> List[Dict[str, Any]]:
        """Detect retry/error handling patterns."""
        retry_patterns = []
        
        for step in pipeline.steps:
            # Look for retry indicators in step names or conditions
            if any(keyword in step.id.lower() for keyword in ['retry', 'attempt', 'fallback']):
                # Check if it has error handling
                if step.condition and any(keyword in step.condition.lower() for keyword in ['error', 'failed', 'exception']):
                    pattern = {
                        'type': 'retry_pattern',
                        'retry_step': step.id,
                        'error_condition': step.condition,
                        'max_attempts': getattr(step, 'max_iterations', 1),
                        'optimization_potential': 'medium'
                    }
                    retry_patterns.append(pattern)
                    
        return retry_patterns
        
    async def optimize_control_flow(self, control_map: ControlFlowMap, patterns: Dict[str, Any]) -> ControlFlowMap:
        """Apply optimizations based on detected patterns."""
        optimized_map = control_map.copy()
        
        # Optimize nested conditionals
        for pattern in patterns['nested_conditionals']:
            await self._optimize_nested_conditional(optimized_map, pattern)
            
        # Optimize fan-out patterns
        for pattern in patterns['fan_out_patterns']:
            await self._optimize_fan_out(optimized_map, pattern)
            
        # Add early termination for loops
        for pattern in patterns['early_exit_loops']:
            await self._optimize_early_exit_loop(optimized_map, pattern)
            
        logger.info("Control flow optimizations applied")
        return optimized_map
        
    async def _optimize_nested_conditional(self, control_map: ControlFlowMap, pattern: Dict[str, Any]) -> None:
        """Optimize nested conditionals by flattening logic."""
        root_step = pattern['root_step']
        conditions = pattern['conditions']
        
        # Create combined condition logic
        combined_condition = f"({conditions[0]}) and ({conditions[1]})"
        
        if root_step in control_map.conditionals:
            original_logic = control_map.conditionals[root_step]
            original_logic.condition_template = combined_condition
            logger.debug(f"Optimized nested conditional for step {root_step}")
            
    async def _optimize_fan_out(self, control_map: ControlFlowMap, pattern: Dict[str, Any]) -> None:
        """Mark fan-out patterns for parallel execution."""
        source_step = pattern['source_step']
        target_steps = pattern['target_steps']
        
        # Add metadata for parallel execution
        control_map.parallel_opportunities[source_step] = {
            'type': 'fan_out',
            'parallel_targets': target_steps,
            'estimated_speedup': len(target_steps) * 0.7
        }
        logger.debug(f"Optimized fan-out pattern for step {source_step}")
        
    async def _optimize_early_exit_loop(self, control_map: ControlFlowMap, pattern: Dict[str, Any]) -> None:
        """Optimize loops with early exit conditions."""
        loop_step = pattern['loop_step']
        exit_condition = pattern['exit_condition']
        
        if loop_step in control_map.loops:
            original_logic = control_map.loops[loop_step]
            # Add early termination logic
            original_logic.early_exit_condition = exit_condition
            logger.debug(f"Added early exit optimization for loop {loop_step}")
            
    async def create_langraph_control_edges(self, control_map: ControlFlowMap) -> List[Dict[str, Any]]:
        """Convert control flow analysis into LangGraph conditional edges."""
        langraph_edges = []
        
        # Convert conditionals to LangGraph edges
        for step_id, conditional_logic in control_map.conditionals.items():
            edge_config = {
                'source_node': step_id,
                'condition_function': conditional_logic.evaluator_function,
                'true_path': conditional_logic.true_path,
                'false_path': conditional_logic.false_path or 'END',
                'required_state_keys': conditional_logic.required_data
            }
            langraph_edges.append(edge_config)
            
        # Convert loops to LangGraph structures
        for step_id, loop_logic in control_map.loops.items():
            edge_config = {
                'source_node': step_id,
                'loop_type': loop_logic.loop_type.value,
                'condition_template': loop_logic.condition_template,
                'max_iterations': loop_logic.max_iterations,
                'substeps': loop_logic.substeps
            }
            langraph_edges.append(edge_config)
            
        logger.info(f"Created {len(langraph_edges)} LangGraph control edges")
        return langraph_edges