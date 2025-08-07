"""
ParallelExecutionDetector - Identifies parallel execution opportunities.

This module implements automatic parallel execution detection as specified in Issue #199.
It analyzes dependency graphs to identify steps that can execute concurrently without
requiring users to understand parallelization concepts.

Key Features:
- Automatic detection of independent step groups
- Parallel execution opportunity identification 
- Resource-aware parallelization strategies
- Performance optimization for concurrent execution
"""

from __future__ import annotations

import logging
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict

from .types import DependencyGraph, ParallelGroup, ParallelizationType, ParsedStep

logger = logging.getLogger(__name__)


class ParallelExecutionDetector:
    """
    Automatically identifies parallel execution opportunities as specified in Issue #199.
    Users don't need to understand parallelization - system optimizes automatically.
    """
    
    def __init__(self):
        logger.info("ParallelExecutionDetector initialized")
        
    async def detect_parallel_groups(self, dependency_graph: DependencyGraph) -> List[ParallelGroup]:
        """
        Identify steps that can execute in parallel based on data dependencies.
        Implements the "automatic parallel execution detection" from Issue #199.
        
        Args:
            dependency_graph: Resolved dependency graph
            
        Returns:
            List of parallel groups that can execute concurrently
        """
        logger.debug("Detecting parallel execution opportunities")
        
        parallel_groups = []
        
        # Get execution levels for parallel analysis
        execution_levels = dependency_graph.get_execution_levels()
        
        for level, steps_at_level in execution_levels.items():
            if len(steps_at_level) > 1:
                # Multiple steps at same level can potentially run in parallel
                independent_groups = await self._find_independent_groups(steps_at_level, dependency_graph)
                
                for group in independent_groups:
                    if len(group) > 1:  # Only create parallel groups for multiple steps
                        parallel_group = ParallelGroup(
                            steps=group,
                            execution_level=level,
                            parallelization_type=ParallelizationType.INDEPENDENT,
                            estimated_speedup=self._estimate_speedup(group)
                        )
                        parallel_groups.append(parallel_group)
                        
        # Handle explicit parallel_map steps
        parallel_map_groups = await self._identify_parallel_map_steps(dependency_graph)
        parallel_groups.extend(parallel_map_groups)
        
        logger.info(f"Detected {len(parallel_groups)} parallel execution opportunities")
        return parallel_groups
        
    async def _find_independent_groups(self, 
                                     steps_at_level: List[str], 
                                     dependency_graph: DependencyGraph) -> List[List[str]]:
        """
        Within a single execution level, find groups of steps that are completely independent.
        Two steps are independent if they don't share data dependencies.
        """
        # Stub implementation - will be enhanced
        if len(steps_at_level) <= 1:
            return [steps_at_level]
            
        # For now, treat all steps at same level as potentially parallel
        # This is a conservative approach that will be refined
        return [steps_at_level]
        
    async def _identify_parallel_map_steps(self, dependency_graph: DependencyGraph) -> List[ParallelGroup]:
        """Identify explicit parallel_map steps for parallel processing."""
        parallel_map_groups = []
        
        for step_id, step in dependency_graph.nodes.items():
            if hasattr(step, 'type') and step.type.value == 'parallel_map':
                parallel_group = ParallelGroup(
                    steps=[step_id],
                    execution_level=0,  # Will be calculated properly later
                    parallelization_type=ParallelizationType.MAP_REDUCE,
                    estimated_speedup=3.0  # Conservative estimate
                )
                parallel_map_groups.append(parallel_group)
                
        return parallel_map_groups
        
    def _estimate_speedup(self, group: List[str]) -> float:
        """Estimate performance speedup from parallel execution."""
        # Conservative speedup estimation
        # Real implementation would consider:
        # - I/O vs CPU bound operations
        # - Resource contention
        # - Overhead of parallelization
        
        if len(group) <= 1:
            return 1.0
        elif len(group) <= 2:
            return 1.5
        elif len(group) <= 4:
            return 2.0
        else:
            return min(len(group) * 0.7, 4.0)  # Cap at 4x speedup