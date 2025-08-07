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