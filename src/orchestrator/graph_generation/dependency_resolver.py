"""
EnhancedDependencyResolver - Sophisticated dependency resolution system.

This module implements comprehensive dependency resolution as outlined in Issue #199,
handling complex scenarios including explicit dependencies, implicit data dependencies,
conditional dependencies, and automatic execution order optimization.

Key Features:
- Explicit dependencies from depends_on arrays
- Implicit dependencies from template variable analysis
- Conditional dependencies for dynamic execution paths
- Circular dependency detection and prevention
- Execution order optimization
- Support for parallel execution planning
"""

from __future__ import annotations

import logging
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict

from ..core.exceptions import CircularDependencyError, InvalidDependencyError
from .types import (
    ParsedStep, DependencyGraph, DependencyEdge, DependencyType, 
    ValidationError
)
from .syntax_parser import DeclarativeSyntaxParser

logger = logging.getLogger(__name__)


class DependencyResolutionError(Exception):
    """Raised when dependency resolution fails."""
    pass


class EnhancedDependencyResolver:
    """
    Implements sophisticated dependency resolution as outlined in Issue #199.
    Handles complex scenarios including conditional dependencies and data flow analysis.
    """
    
    def __init__(self):
        self.syntax_parser = DeclarativeSyntaxParser()
        self._resolution_cache: Dict[str, DependencyGraph] = {}
        
        logger.info("EnhancedDependencyResolver initialized")
        
    async def resolve_dependencies(self, steps: List[ParsedStep]) -> DependencyGraph:
        """
        Build comprehensive dependency graph with advanced features.
        
        Creates a complete dependency graph including:
        - Explicit dependencies from depends_on arrays
        - Implicit dependencies from template variable references
        - Conditional dependencies for dynamic execution
        - Optimized execution ordering
        
        Args:
            steps: List of parsed pipeline steps
            
        Returns:
            Complete dependency graph with all relationships
            
        Raises:
            CircularDependencyError: If circular dependencies are detected
            DependencyResolutionError: If dependency resolution fails
        """
        try:
            logger.debug(f"Resolving dependencies for {len(steps)} steps")
            
            # Create dependency graph
            graph = DependencyGraph()
            
            # Add all steps as nodes first
            step_map = {}
            for step in steps:
                graph.add_node(step.id, step)
                step_map[step.id] = step
                
            # Phase 1: Add explicit dependencies
            await self._add_explicit_dependencies(graph, steps)
            
            # Phase 2: Add implicit data dependencies
            await self._add_implicit_dependencies(graph, steps)
            
            # Phase 3: Add conditional dependencies
            await self._add_conditional_dependencies(graph, steps)
            
            # Phase 4: Validate and detect circular dependencies
            if graph.has_cycles():
                cycles = graph.find_cycles()
                raise CircularDependencyError(
                    f"Circular dependencies detected: {cycles}"
                )
                
            # Phase 5: Optimize execution order
            optimized_order = await self._optimize_execution_order(graph)
            graph.set_execution_order(optimized_order)
            
            # Phase 6: Validate dependency consistency
            await self._validate_dependency_consistency(graph)
            
            logger.info(f"Successfully resolved {len(graph.edges)} dependencies for {len(steps)} steps")
            return graph
            
        except CircularDependencyError:
            # Re-raise circular dependency errors without wrapping
            raise
        except Exception as e:
            raise DependencyResolutionError(f"Failed to resolve dependencies: {e}") from e
            
    async def _add_explicit_dependencies(self, graph: DependencyGraph, steps: List[ParsedStep]) -> None:
        """Add explicit dependencies from depends_on arrays."""
        logger.debug("Adding explicit dependencies")
        
        for step in steps:
            for dep_id in step.depends_on:
                if not graph.has_node(dep_id):
                    raise InvalidDependencyError(
                        f"Step '{step.id}' depends on undefined step '{dep_id}'"
                    )
                    
                graph.add_edge(
                    source=dep_id,
                    target=step.id,
                    dependency_type=DependencyType.EXPLICIT,
                    weight=1.0
                )
                
        logger.debug(f"Added {sum(len(step.depends_on) for step in steps)} explicit dependencies")
        
    async def _add_implicit_dependencies(self, graph: DependencyGraph, steps: List[ParsedStep]) -> None:
        """
        Add implicit dependencies from template variable references.
        
        Analyzes step inputs for template variables to detect implicit dependencies.
        Examples: 
        - {{ web_search.results }} → depends on 'web_search'
        - {{ item.claim_text }} → depends on parallel iteration context
        """
        logger.debug("Analyzing implicit data dependencies")
        
        implicit_deps_count = 0
        
        for step in steps:
            implicit_deps = await self._analyze_data_dependencies(step)
            
            for dep_id in implicit_deps:
                if graph.has_node(dep_id):
                    # Only add if not already an explicit dependency
                    existing_edges = [
                        edge for edge in graph.edges 
                        if edge.source == dep_id and edge.target == step.id
                    ]
                    
                    if not existing_edges:
                        graph.add_edge(
                            source=dep_id,
                            target=step.id,
                            dependency_type=DependencyType.IMPLICIT,
                            weight=0.8  # Lower weight than explicit dependencies
                        )
                        implicit_deps_count += 1
                        
        logger.debug(f"Added {implicit_deps_count} implicit dependencies")
        
    async def _analyze_data_dependencies(self, step: ParsedStep) -> List[str]:
        """
        Analyze step inputs for template variables to detect implicit dependencies.
        
        Examples:
        - {{ web_search.results }} → depends on 'web_search'  
        - {{ analyze_results.claims }} → depends on 'analyze_results'
        - {{ item.claim_text }} → no dependency (special context variable)
        """
        dependencies = set()
        
        # Analyze all input values
        for input_value in step.inputs.values():
            if isinstance(input_value, str):
                template_vars = self.syntax_parser.extract_template_variables(input_value)
                for var in template_vars:
                    # Parse variable path to identify source step
                    source_step = var.split('.')[0]
                    
                    # Skip special context variables
                    if source_step not in ['inputs', 'item', 'loop', 'index']:
                        dependencies.add(source_step)
                        
        # Analyze condition if present
        if step.condition:
            condition_vars = self.syntax_parser.extract_template_variables(step.condition)
            for var in condition_vars:
                source_step = var.split('.')[0]
                if source_step not in ['inputs', 'item', 'loop', 'index']:
                    dependencies.add(source_step)
                    
        # Analyze parallel_map items expression
        if step.items:
            items_vars = self.syntax_parser.extract_template_variables(step.items)
            for var in items_vars:
                source_step = var.split('.')[0]
                if source_step not in ['inputs', 'item', 'loop', 'index']:
                    dependencies.add(source_step)
                    
        return list(dependencies)
        
    async def _add_conditional_dependencies(self, graph: DependencyGraph, steps: List[ParsedStep]) -> None:
        """Add conditional dependencies for dynamic execution paths."""
        logger.debug("Adding conditional dependencies")
        
        conditional_deps_count = 0
        
        for step in steps:
            if step.condition:
                # Steps with conditions create conditional dependencies
                conditional_deps = await self._analyze_conditional_dependencies(step)
                
                for dep_id, condition in conditional_deps.items():
                    if graph.has_node(dep_id):
                        graph.add_edge(
                            source=dep_id,
                            target=step.id,
                            dependency_type=DependencyType.CONDITIONAL,
                            condition=condition,
                            weight=0.5  # Lower weight for conditional deps
                        )
                        conditional_deps_count += 1
                        
            # Handle else_step relationships  
            if step.else_step:
                if graph.has_node(step.else_step):
                    # Create conditional edge to else step
                    graph.add_edge(
                        source=step.id,
                        target=step.else_step,
                        dependency_type=DependencyType.CONDITIONAL,
                        condition=f"not ({step.condition})" if step.condition else None,
                        weight=0.5
                    )
                    conditional_deps_count += 1
                    
        logger.debug(f"Added {conditional_deps_count} conditional dependencies")
        
    async def _analyze_conditional_dependencies(self, step: ParsedStep) -> Dict[str, str]:
        """
        Analyze conditional execution to identify conditional dependencies.
        
        Returns mapping of step_id -> condition for conditional dependencies.
        """
        conditional_deps = {}
        
        if not step.condition:
            return conditional_deps
            
        # Extract variables from condition to identify dependencies
        condition_vars = self.syntax_parser.extract_template_variables(step.condition)
        
        for var in condition_vars:
            source_step = var.split('.')[0]
            if source_step not in ['inputs', 'item', 'loop', 'index']:
                # This step conditionally depends on the source step
                conditional_deps[source_step] = step.condition
                
        return conditional_deps
        
    async def _optimize_execution_order(self, graph: DependencyGraph) -> List[str]:
        """
        Create optimized execution order based on dependencies and weights.
        
        Uses a weighted topological sort to optimize for:
        - Critical path reduction
        - Parallel execution opportunities
        - Resource utilization
        """
        logger.debug("Optimizing execution order")
        
        # Start with basic topological sort
        base_order = graph.topological_sort()
        
        # Calculate execution levels for parallel optimization
        execution_levels = graph.get_execution_levels()
        
        # Optimize within each level based on:
        # 1. Number of downstream dependencies (critical path)
        # 2. Resource requirements
        # 3. Estimated execution time
        optimized_order = []
        
        for level in sorted(execution_levels.keys()):
            level_steps = execution_levels[level]
            
            if len(level_steps) == 1:
                optimized_order.extend(level_steps)
            else:
                # Sort steps within level by priority
                prioritized_steps = await self._prioritize_steps_in_level(
                    level_steps, graph
                )
                optimized_order.extend(prioritized_steps)
                
        logger.debug(f"Optimized execution order: {' -> '.join(optimized_order)}")
        return optimized_order
        
    async def _prioritize_steps_in_level(self, 
                                       steps: List[str], 
                                       graph: DependencyGraph) -> List[str]:
        """
        Prioritize steps within an execution level for optimal ordering.
        
        Priority factors:
        - Number of downstream dependencies (critical path)
        - Estimated execution complexity
        - Resource requirements
        """
        step_priorities = []
        
        for step_id in steps:
            step = graph.nodes[step_id]
            
            # Calculate downstream dependency count (critical path indicator)
            downstream_count = len(graph._adjacency_list.get(step_id, []))
            
            # Estimate complexity based on step characteristics
            complexity_score = await self._estimate_step_complexity(step)
            
            # Combined priority score (higher = more critical)
            priority_score = (downstream_count * 2.0) + complexity_score
            
            step_priorities.append((step_id, priority_score))
            
        # Sort by priority (descending - most critical first)
        step_priorities.sort(key=lambda x: x[1], reverse=True)
        
        return [step_id for step_id, _ in step_priorities]
        
    async def _estimate_step_complexity(self, step: ParsedStep) -> float:
        """
        Estimate step execution complexity for prioritization.
        
        Factors:
        - Tool type (some tools are inherently slower)
        - Input data size indicators
        - Model requirements
        - Control flow complexity
        """
        complexity = 1.0  # Base complexity
        
        # Tool-based complexity estimation
        if step.tool:
            tool_complexity = {
                'web-search': 3.0,
                'headless-browser': 4.0, 
                'llm_analysis': 2.5,
                'filesystem': 1.0,
                'fact_checker': 3.5
            }
            complexity += tool_complexity.get(step.tool, 2.0)
            
        # Model requirements complexity
        if step.model_requirements:
            if isinstance(step.model_requirements, dict):
                min_size = step.model_requirements.get('min_size', '1B')
                if isinstance(min_size, str) and 'B' in min_size:
                    size_val = float(min_size.replace('B', ''))
                    complexity += size_val * 0.1  # Larger models = higher complexity
                    
        # Control flow complexity
        if step.type.value != 'standard':
            complexity += 1.5
            
        if step.condition:
            complexity += 0.5
            
        if step.substeps:
            complexity += len(step.substeps) * 0.3
            
        return complexity
        
    async def _validate_dependency_consistency(self, graph: DependencyGraph) -> None:
        """
        Validate that the dependency graph is consistent and well-formed.
        
        Checks for:
        - All referenced steps exist
        - No orphaned dependencies  
        - Conditional dependencies are valid
        - Execution order is feasible
        """
        logger.debug("Validating dependency consistency")
        
        validation_errors = []
        
        # Check all edges reference valid nodes
        for edge in graph.edges:
            if not graph.has_node(edge.source):
                validation_errors.append(f"Edge references non-existent source: {edge.source}")
            if not graph.has_node(edge.target):
                validation_errors.append(f"Edge references non-existent target: {edge.target}")
                
        # Check for orphaned conditional dependencies
        for edge in graph.edges:
            if edge.dependency_type == DependencyType.CONDITIONAL:
                if edge.condition:
                    # Validate condition template variables
                    condition_vars = self.syntax_parser.extract_template_variables(edge.condition)
                    for var in condition_vars:
                        var_step = var.split('.')[0] 
                        if var_step not in ['inputs', 'item', 'loop'] and not graph.has_node(var_step):
                            validation_errors.append(
                                f"Conditional dependency condition references undefined step: {var_step}"
                            )
                            
        # Verify execution order covers all nodes
        execution_order = graph.get_execution_order()
        all_nodes = set(graph.nodes.keys())
        ordered_nodes = set(execution_order)
        
        if all_nodes != ordered_nodes:
            missing = all_nodes - ordered_nodes
            extra = ordered_nodes - all_nodes
            if missing:
                validation_errors.append(f"Execution order missing steps: {missing}")
            if extra:
                validation_errors.append(f"Execution order has extra steps: {extra}")
                
        if validation_errors:
            raise DependencyResolutionError(
                f"Dependency graph validation failed: {'; '.join(validation_errors)}"
            )
            
        logger.debug("Dependency graph validation passed")
        
    def get_dependency_statistics(self, graph: DependencyGraph) -> Dict[str, Any]:
        """Get statistics about the dependency graph for analysis."""
        edge_types = defaultdict(int)
        for edge in graph.edges:
            edge_types[edge.dependency_type.value] += 1
            
        execution_levels = graph.get_execution_levels()
        max_parallelism = max(len(steps) for steps in execution_levels.values()) if execution_levels else 0
        
        return {
            'total_steps': len(graph.nodes),
            'total_dependencies': len(graph.edges),
            'dependency_types': dict(edge_types),
            'execution_levels': len(execution_levels),
            'max_parallel_steps': max_parallelism,
            'entry_points': len(graph.get_entry_points()),
            'terminal_points': len(graph.get_terminal_nodes()),
            'average_dependencies_per_step': len(graph.edges) / max(1, len(graph.nodes))
        }