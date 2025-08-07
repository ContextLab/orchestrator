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
        Two steps are independent if they don't share data dependencies or resources.
        
        Uses graph coloring algorithm to identify maximal independent sets.
        """
        if len(steps_at_level) <= 1:
            return [steps_at_level]
            
        # Build interference graph - edges between steps that cannot run in parallel
        interference_graph = self._build_interference_graph(steps_at_level, dependency_graph)
        
        # Find maximal independent sets using greedy coloring
        independent_groups = self._find_maximal_independent_sets(steps_at_level, interference_graph)
        
        # Filter groups by resource constraints
        filtered_groups = await self._apply_resource_constraints(independent_groups, dependency_graph)
        
        logger.debug(f"Found {len(filtered_groups)} independent groups from {len(steps_at_level)} steps")
        return filtered_groups
        
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
        
    def _build_interference_graph(self, steps: List[str], dependency_graph: DependencyGraph) -> Dict[str, Set[str]]:
        """
        Build interference graph where edges represent steps that cannot run in parallel.
        Steps interfere if they:
        1. Share input/output data dependencies
        2. Use same external resources (APIs, databases, files)
        3. Have ordering constraints
        """
        interference_graph = defaultdict(set)
        
        for i, step1 in enumerate(steps):
            for j, step2 in enumerate(steps):
                if i >= j:  # Avoid duplicate comparisons
                    continue
                    
                if self._steps_interfere(step1, step2, dependency_graph):
                    interference_graph[step1].add(step2)
                    interference_graph[step2].add(step1)
                    
        return interference_graph
        
    def _steps_interfere(self, step1: str, step2: str, dependency_graph: DependencyGraph) -> bool:
        """Check if two steps interfere and cannot run in parallel."""
        step1_node = dependency_graph.nodes.get(step1)
        step2_node = dependency_graph.nodes.get(step2)
        
        if not step1_node or not step2_node:
            return True  # Be conservative if nodes missing
            
        # Check for shared resource usage
        if self._share_resources(step1_node, step2_node):
            return True
            
        # Check for data dependencies between steps
        if self._have_data_dependency(step1, step2, dependency_graph):
            return True
            
        # Check for ordering constraints
        if self._have_ordering_constraint(step1, step2, dependency_graph):
            return True
            
        return False
        
    def _share_resources(self, step1: ParsedStep, step2: ParsedStep) -> bool:
        """Check if steps use the same external resources."""
        # Same tool/action type suggests potential resource conflict
        if step1.tool == step2.tool and step1.tool is not None:
            # Same external API/service - may have rate limits
            if step1.tool in ['web-search', 'api-call', 'database-query']:
                return True
                
        # Same action type
        if step1.action == step2.action and step1.action is not None:
            # File operations on same path
            if step1.action in ['file-read', 'file-write', 'file-append']:
                # Would need to check file paths from inputs
                pass
                
        return False
        
    def _have_data_dependency(self, step1: str, step2: str, dependency_graph: DependencyGraph) -> bool:
        """Check if steps have direct or indirect data dependencies."""
        # Check if step2 depends on step1's output or vice versa
        for edge in dependency_graph.edges:
            if (edge.source == step1 and edge.target == step2) or \
               (edge.source == step2 and edge.target == step1):
                return True
                
        return False
        
    def _have_ordering_constraint(self, step1: str, step2: str, dependency_graph: DependencyGraph) -> bool:
        """Check for implicit ordering constraints."""
        # For now, assume no ordering constraints beyond data dependencies
        # Future enhancement: analyze for side effects, state mutations, etc.
        return False
        
    def _find_maximal_independent_sets(self, steps: List[str], interference_graph: Dict[str, Set[str]]) -> List[List[str]]:
        """
        Find maximal independent sets using greedy graph coloring.
        Each independent set can run in parallel.
        """
        if not steps:
            return []
            
        # Start with all steps unassigned
        unassigned = set(steps)
        independent_groups = []
        
        while unassigned:
            # Greedy: pick step with least interference
            current_group = []
            candidates = set(unassigned)
            
            while candidates:
                # Pick node with minimum degree in remaining subgraph
                min_degree_node = min(candidates, 
                                    key=lambda node: len(interference_graph[node] & candidates))
                
                current_group.append(min_degree_node)
                unassigned.discard(min_degree_node)
                
                # Remove this node and all its neighbors from candidates
                to_remove = {min_degree_node} | interference_graph[min_degree_node]
                candidates -= to_remove
                
            if current_group:
                independent_groups.append(current_group)
                
        return independent_groups
        
    async def _apply_resource_constraints(self, groups: List[List[str]], dependency_graph: DependencyGraph) -> List[List[str]]:
        """Apply resource constraints to limit parallel group sizes."""
        filtered_groups = []
        
        for group in groups:
            # Apply maximum parallelism constraints
            max_parallel = await self._calculate_max_parallel(group, dependency_graph)
            
            if len(group) <= max_parallel:
                filtered_groups.append(group)
            else:
                # Split large groups respecting constraints
                split_groups = self._split_oversized_group(group, max_parallel)
                filtered_groups.extend(split_groups)
                
        return filtered_groups
        
    async def _calculate_max_parallel(self, group: List[str], dependency_graph: DependencyGraph) -> int:
        """Calculate maximum safe parallelism level for a group."""
        # Consider various constraints
        max_parallel = len(group)  # Start with group size
        
        # Resource-based limits
        for step_id in group:
            step = dependency_graph.nodes.get(step_id)
            if step:
                step_limit = self._get_step_parallelism_limit(step)
                max_parallel = min(max_parallel, step_limit)
                
        # System-wide limits (could be configurable)
        system_max = 8  # Conservative default
        max_parallel = min(max_parallel, system_max)
        
        return max(1, max_parallel)
        
    def _get_step_parallelism_limit(self, step: ParsedStep) -> int:
        """Get parallelism limit for individual step based on its characteristics."""
        # I/O bound operations can be more parallel
        if step.tool in ['web-search', 'api-call', 'fetch-url']:
            return 4
            
        # CPU bound operations more conservative
        if step.tool in ['analyze-text', 'process-data', 'compute']:
            return 2
            
        # File operations very conservative
        if step.action in ['file-write', 'database-update']:
            return 1
            
        return 3  # Default moderate parallelism
        
    def _split_oversized_group(self, group: List[str], max_size: int) -> List[List[str]]:
        """Split oversized parallel groups into smaller groups."""
        if max_size <= 0:
            return [[step] for step in group]  # Each step gets its own group
            
        split_groups = []
        for i in range(0, len(group), max_size):
            subgroup = group[i:i + max_size]
            split_groups.append(subgroup)
            
        return split_groups
        
    def _estimate_speedup(self, group: List[str]) -> float:
        """Estimate performance speedup from parallel execution."""
        # Enhanced speedup estimation considering realistic constraints
        base_speedup = len(group)
        
        # Apply Amdahl's law approximation
        # Assume 70% of work is parallelizable
        parallel_fraction = 0.7
        sequential_fraction = 1.0 - parallel_fraction
        
        # Amdahl's law: speedup = 1 / (sequential_fraction + parallel_fraction / parallel_workers)
        theoretical_speedup = 1.0 / (sequential_fraction + parallel_fraction / base_speedup)
        
        # Apply overhead penalty (parallelization isn't free)
        overhead_penalty = 0.15  # 15% overhead
        practical_speedup = theoretical_speedup * (1.0 - overhead_penalty)
        
        # Cap speedup at reasonable maximum
        return min(practical_speedup, 4.0)