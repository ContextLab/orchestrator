"""
Dependency graph validation for pipeline definitions.

This module provides comprehensive dependency validation including:
- Circular dependency detection
- Unreachable task detection
- Validation of all dependency types (dependencies, for_each, conditional, action_loop)
- Clear error messages showing dependency chains
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union, Tuple
import re

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

from ..core.exceptions import ValidationError

logger = logging.getLogger(__name__)


@dataclass
class DependencyIssue:
    """Represents a dependency validation issue."""
    
    issue_type: str
    severity: str  # 'error', 'warning'
    message: str
    involved_tasks: List[str] = field(default_factory=list)
    dependency_chain: Optional[List[str]] = None
    recommendation: Optional[str] = None


@dataclass 
class DependencyValidationResult:
    """Results of dependency validation."""
    
    is_valid: bool
    issues: List[DependencyIssue] = field(default_factory=list)
    dependency_graph: Optional[Any] = None  # NetworkX graph if available
    execution_order: Optional[List[str]] = None
    
    @property
    def errors(self) -> List[DependencyIssue]:
        """Get only error-level issues."""
        return [issue for issue in self.issues if issue.severity == 'error']
    
    @property
    def warnings(self) -> List[DependencyIssue]:
        """Get only warning-level issues."""
        return [issue for issue in self.issues if issue.severity == 'warning']
    
    @property
    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues."""
        return len(self.warnings) > 0


class DependencyValidator:
    """
    Validates dependency graphs in pipeline definitions.
    
    Features:
    - Builds complete dependency graph from pipeline
    - Checks for circular dependencies
    - Validates that all referenced tasks exist
    - Checks for unreachable tasks
    - Validates for_each and conditional dependencies
    - Supports networkx for advanced graph analysis or fallback to custom implementation
    """
    
    def __init__(self, development_mode: bool = False):
        """
        Initialize dependency validator.
        
        Args:
            development_mode: If True, allows some validation bypasses
        """
        self.development_mode = development_mode
        self.use_networkx = NETWORKX_AVAILABLE
        
        if not self.use_networkx:
            logger.warning("NetworkX not available, using custom graph implementation")
    
    def validate_pipeline_dependencies(
        self, 
        pipeline_def: Dict[str, Any]
    ) -> DependencyValidationResult:
        """
        Validate all dependencies in a pipeline definition.
        
        Args:
            pipeline_def: Pipeline definition dictionary
            
        Returns:
            DependencyValidationResult with validation results
        """
        logger.debug("Starting dependency validation for pipeline")
        
        issues = []
        dependency_graph = None
        execution_order = None
        
        try:
            # Extract all tasks from the pipeline
            tasks = self._extract_tasks(pipeline_def)
            
            if not tasks:
                issues.append(DependencyIssue(
                    issue_type="empty_pipeline",
                    severity="warning",
                    message="Pipeline contains no tasks",
                    recommendation="Add at least one task to the pipeline"
                ))
                return DependencyValidationResult(is_valid=True, issues=issues)
            
            # Build dependency graph
            dependency_graph = self._build_dependency_graph(tasks)
            
            # Validate unique task IDs
            issues.extend(self._validate_unique_task_ids(tasks))
            
            # Validate that all referenced dependencies exist
            issues.extend(self._validate_dependency_references(tasks))
            
            # Check for circular dependencies
            issues.extend(self._validate_circular_dependencies(dependency_graph, tasks))
            
            # Check for unreachable tasks
            issues.extend(self._validate_reachable_tasks(dependency_graph, tasks))
            
            # Validate special dependency types
            issues.extend(self._validate_control_flow_dependencies(tasks))
            
            # Determine execution order if possible
            if not any(issue.severity == 'error' for issue in issues):
                execution_order = self._compute_execution_order(dependency_graph, tasks)
            
        except Exception as e:
            logger.error(f"Error during dependency validation: {e}")
            issues.append(DependencyIssue(
                issue_type="validation_error",
                severity="error",
                message=f"Internal validation error: {e}",
                recommendation="Check pipeline syntax and structure"
            ))
        
        is_valid = not any(issue.severity == 'error' for issue in issues)
        
        # In development mode, convert some errors to warnings
        if self.development_mode:
            for issue in issues:
                if issue.issue_type in ["unreachable_task", "missing_dependency"]:
                    if issue.severity == "error":
                        issue.severity = "warning"
                        issue.message = f"[DEV MODE] {issue.message}"
            is_valid = True  # Allow pipeline to proceed in development mode
        
        logger.info(f"Dependency validation completed: {len(issues)} issues found")
        
        return DependencyValidationResult(
            is_valid=is_valid,
            issues=issues,
            dependency_graph=dependency_graph,
            execution_order=execution_order
        )
    
    def _extract_tasks(self, pipeline_def: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all tasks from pipeline definition."""
        tasks = []
        
        steps = pipeline_def.get("steps", [])
        for step in steps:
            if isinstance(step, dict):
                tasks.append(step)  # Include all steps, even those missing IDs for validation
        
        logger.debug(f"Extracted {len(tasks)} tasks from pipeline")
        return tasks
    
    def _build_dependency_graph(self, tasks: List[Dict[str, Any]]) -> Any:
        """
        Build dependency graph from tasks.
        
        Returns:
            NetworkX DiGraph if available, otherwise custom graph dict
        """
        if self.use_networkx:
            graph = nx.DiGraph()
            
            # Add all tasks as nodes (only tasks with IDs)
            for task in tasks:
                task_id = task.get("id")
                if task_id:
                    graph.add_node(task_id, **task)
            
            # Add dependency edges
            for task in tasks:
                task_id = task.get("id")
                if not task_id:
                    continue  # Skip tasks without IDs
                    
                dependencies = self._get_task_dependencies(task)
                
                for dep in dependencies:
                    if dep != task_id:  # Avoid self-loops for now, will validate separately
                        graph.add_edge(dep, task_id)
            
            return graph
        else:
            # Custom graph implementation
            graph = {
                "nodes": {task["id"]: task for task in tasks if "id" in task},
                "edges": {}
            }
            
            for task in tasks:
                task_id = task.get("id")
                if not task_id:
                    continue  # Skip tasks without IDs
                    
                dependencies = self._get_task_dependencies(task)
                graph["edges"][task_id] = dependencies
            
            return graph
    
    def _get_task_dependencies(self, task: Dict[str, Any]) -> List[str]:
        """Extract all dependencies from a task definition."""
        dependencies = []
        
        # Direct dependencies
        deps = task.get("dependencies", task.get("depends_on", []))
        if isinstance(deps, str):
            if "," in deps:
                dependencies.extend([d.strip() for d in deps.split(",") if d.strip()])
            else:
                if deps.strip():
                    dependencies.append(deps.strip())
        elif isinstance(deps, list):
            dependencies.extend(deps)
        
        # Control flow dependencies
        for_each = task.get("for_each", task.get("foreach"))
        if for_each:
            # Extract dependencies from for_each expressions
            deps = self._extract_template_dependencies(for_each)
            dependencies.extend(deps)
        
        # Conditional dependencies
        condition = task.get("condition", task.get("if"))
        if condition:
            deps = self._extract_template_dependencies(condition)
            dependencies.extend(deps)
        
        # While loop dependencies
        while_condition = task.get("while")
        if while_condition:
            deps = self._extract_template_dependencies(while_condition)
            dependencies.extend(deps)
        
        # Action loop dependencies
        if "action_loop" in task:
            # Check until/while conditions in action loops
            until = task.get("until")
            if until:
                deps = self._extract_template_dependencies(until)
                dependencies.extend(deps)
            
            while_cond = task.get("while")  # Can also be on action_loop
            if while_cond:
                deps = self._extract_template_dependencies(while_cond)
                dependencies.extend(deps)
        
        # Parallel queue dependencies
        if "create_parallel_queue" in task:
            queue_config = task["create_parallel_queue"]
            if isinstance(queue_config, dict):
                # Check 'on' field for dependencies
                on_expr = queue_config.get("on")
                if on_expr:
                    deps = self._extract_template_dependencies(on_expr)
                    dependencies.extend(deps)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_deps = []
        for dep in dependencies:
            if dep not in seen:
                seen.add(dep)
                unique_deps.append(dep)
        
        return unique_deps
    
    def _extract_template_dependencies(self, template_str: str) -> List[str]:
        """Extract task dependencies from template expressions."""
        if not isinstance(template_str, str):
            return []
        
        dependencies = set()  # Use set to avoid duplicates
        
        # Pattern to match task references like "task_id.result", "task_id.output", etc.
        # This is a simplified pattern - in practice, you'd want more sophisticated parsing
        task_ref_pattern = r'\b([a-zA-Z][a-zA-Z0-9_-]*)\.(result|output|data|content|status|metadata)'
        
        matches = re.finditer(task_ref_pattern, template_str)
        for match in matches:
            task_id = match.group(1)
            dependencies.add(task_id)
        
        # Also look for simple variable references that might be task IDs
        # Pattern for variables in templates like {{ task_id }} 
        var_pattern = r'\{\{\s*([a-zA-Z][a-zA-Z0-9_-]*)'
        var_matches = re.finditer(var_pattern, template_str)
        for match in var_matches:
            var_name = match.group(1)
            # Only include if it looks like a task ID (not built-in variables)
            if not var_name.startswith('$') and var_name not in ['item', 'index', 'is_first', 'is_last', 'iteration', 'loop']:
                # Don't add if already captured by task reference pattern
                if not any(var_name == dep for dep in dependencies):
                    dependencies.add(var_name)
        
        return list(dependencies)
    
    def _validate_unique_task_ids(self, tasks: List[Dict[str, Any]]) -> List[DependencyIssue]:
        """Validate that all task IDs are unique."""
        issues = []
        task_ids = []
        
        for task in tasks:
            task_id = task.get("id")
            if not task_id:
                issues.append(DependencyIssue(
                    issue_type="missing_task_id",
                    severity="error",
                    message="Task missing required 'id' field",
                    recommendation="Add unique 'id' field to all tasks"
                ))
                continue
            
            if task_id in task_ids:
                issues.append(DependencyIssue(
                    issue_type="duplicate_task_id",
                    severity="error",
                    message=f"Duplicate task ID: '{task_id}'",
                    involved_tasks=[task_id],
                    recommendation=f"Change one of the tasks with ID '{task_id}' to use a unique identifier"
                ))
            else:
                task_ids.append(task_id)
        
        return issues
    
    def _validate_dependency_references(self, tasks: List[Dict[str, Any]]) -> List[DependencyIssue]:
        """Validate that all referenced dependencies exist."""
        issues = []
        task_ids = {task["id"] for task in tasks if "id" in task}
        
        for task in tasks:
            task_id = task.get("id")
            if not task_id:
                continue
            
            dependencies = self._get_task_dependencies(task)
            
            for dep in dependencies:
                if dep not in task_ids:
                    issues.append(DependencyIssue(
                        issue_type="missing_dependency",
                        severity="error",
                        message=f"Task '{task_id}' depends on non-existent task '{dep}'",
                        involved_tasks=[task_id, dep],
                        recommendation=f"Either create task '{dep}' or remove the dependency from task '{task_id}'"
                    ))
                elif dep == task_id:
                    issues.append(DependencyIssue(
                        issue_type="self_dependency",
                        severity="error",
                        message=f"Task '{task_id}' cannot depend on itself",
                        involved_tasks=[task_id],
                        dependency_chain=[task_id, task_id],
                        recommendation=f"Remove self-dependency from task '{task_id}'"
                    ))
        
        return issues
    
    def _validate_circular_dependencies(
        self, 
        dependency_graph: Any, 
        tasks: List[Dict[str, Any]]
    ) -> List[DependencyIssue]:
        """Check for circular dependencies in the graph."""
        issues = []
        
        if self.use_networkx and dependency_graph:
            try:
                # Use NetworkX to find cycles
                cycles = list(nx.simple_cycles(dependency_graph))
                
                for cycle in cycles:
                    # Create a readable dependency chain
                    chain = cycle + [cycle[0]]  # Close the cycle for display
                    
                    issues.append(DependencyIssue(
                        issue_type="circular_dependency",
                        severity="error",
                        message=f"Circular dependency detected: {' -> '.join(chain)}",
                        involved_tasks=cycle,
                        dependency_chain=chain,
                        recommendation="Remove one or more dependencies to break the cycle"
                    ))
                    
            except Exception as e:
                logger.warning(f"Error detecting cycles with NetworkX: {e}")
                # Fall back to custom implementation
                issues.extend(self._validate_circular_dependencies_custom(tasks))
        else:
            # Use custom cycle detection
            issues.extend(self._validate_circular_dependencies_custom(tasks))
        
        return issues
    
    def _validate_circular_dependencies_custom(self, tasks: List[Dict[str, Any]]) -> List[DependencyIssue]:
        """Custom implementation of circular dependency detection."""
        issues = []
        
        # Build adjacency list
        graph = {}
        for task in tasks:
            task_id = task.get("id")
            if task_id:
                dependencies = self._get_task_dependencies(task)
                graph[task_id] = dependencies
        
        # DFS-based cycle detection
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]) -> bool:
            """DFS with cycle detection."""
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                
                issues.append(DependencyIssue(
                    issue_type="circular_dependency",
                    severity="error",
                    message=f"Circular dependency detected: {' -> '.join(cycle)}",
                    involved_tasks=cycle[:-1],  # Remove duplicate
                    dependency_chain=cycle,
                    recommendation="Remove one or more dependencies to break the cycle"
                ))
                return True
            
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            # Explore dependencies
            for dep in graph.get(node, []):
                if dfs(dep, path + [node]):
                    return True
            
            rec_stack.remove(node)
            return False
        
        # Check each node
        for task_id in graph:
            if task_id not in visited:
                dfs(task_id, [])
        
        return issues
    
    def _validate_reachable_tasks(
        self, 
        dependency_graph: Any, 
        tasks: List[Dict[str, Any]]
    ) -> List[DependencyIssue]:
        """Check for unreachable tasks in the pipeline."""
        issues = []
        
        task_ids = {task["id"] for task in tasks if "id" in task}
        
        if not task_ids:
            return issues
        
        if self.use_networkx and dependency_graph:
            try:
                # Find tasks with no incoming edges (potential entry points)
                entry_points = [node for node in dependency_graph.nodes() 
                               if dependency_graph.in_degree(node) == 0]
                
                if not entry_points:
                    # If no entry points, check if we already have a circular dependency error
                    # Don't add "all_tasks_cyclic" if we already detected circular dependencies
                    return issues  # Let circular dependency detection handle this
                
                # Find all reachable tasks from entry points
                reachable = set()
                for entry in entry_points:
                    reachable.update(nx.descendants(dependency_graph, entry))
                    reachable.add(entry)
                
                # Find unreachable tasks
                unreachable = task_ids - reachable
                
                for task_id in unreachable:
                    issues.append(DependencyIssue(
                        issue_type="unreachable_task",
                        severity="warning",  # Could be intentional
                        message=f"Task '{task_id}' is unreachable from pipeline entry points",
                        involved_tasks=[task_id],
                        recommendation=f"Add dependencies to connect task '{task_id}' to the main pipeline flow"
                    ))
                    
            except Exception as e:
                logger.warning(f"Error checking reachability with NetworkX: {e}")
                # Fall back to custom implementation
                issues.extend(self._validate_reachable_tasks_custom(tasks))
        else:
            # Use custom reachability analysis
            issues.extend(self._validate_reachable_tasks_custom(tasks))
        
        return issues
    
    def _validate_reachable_tasks_custom(self, tasks: List[Dict[str, Any]]) -> List[DependencyIssue]:
        """Custom implementation of reachability validation."""
        issues = []
        
        # Build adjacency lists (both forward and backward)
        forward_graph = {}  # task -> [dependents]
        backward_graph = {}  # task -> [dependencies]
        all_tasks = set()
        
        for task in tasks:
            task_id = task.get("id")
            if not task_id:
                continue
                
            all_tasks.add(task_id)
            dependencies = self._get_task_dependencies(task)
            
            backward_graph[task_id] = dependencies
            forward_graph[task_id] = forward_graph.get(task_id, [])
            
            # Add reverse edges
            for dep in dependencies:
                if dep not in forward_graph:
                    forward_graph[dep] = []
                forward_graph[dep].append(task_id)
        
        # Find entry points (tasks with no dependencies)
        entry_points = [task_id for task_id in all_tasks 
                       if not backward_graph.get(task_id, [])]
        
        if not entry_points:
            issues.append(DependencyIssue(
                issue_type="no_entry_points",
                severity="warning",
                message="No tasks found without dependencies - pipeline may have circular dependencies",
                involved_tasks=list(all_tasks),
                recommendation="Ensure at least one task has no dependencies to serve as an entry point"
            ))
            return issues
        
        # BFS to find all reachable tasks
        reachable = set()
        queue = entry_points.copy()
        
        while queue:
            current = queue.pop(0)
            if current in reachable:
                continue
                
            reachable.add(current)
            
            # Add all dependents to queue
            for dependent in forward_graph.get(current, []):
                if dependent not in reachable:
                    queue.append(dependent)
        
        # Find unreachable tasks
        unreachable = all_tasks - reachable
        
        for task_id in unreachable:
            issues.append(DependencyIssue(
                issue_type="unreachable_task",
                severity="warning",
                message=f"Task '{task_id}' is unreachable from pipeline entry points",
                involved_tasks=[task_id],
                recommendation=f"Add dependencies to connect task '{task_id}' to the main pipeline flow"
            ))
        
        return issues
    
    def _validate_control_flow_dependencies(self, tasks: List[Dict[str, Any]]) -> List[DependencyIssue]:
        """Validate control flow specific dependency patterns."""
        issues = []
        
        task_ids = {task["id"] for task in tasks if "id" in task}
        
        for task in tasks:
            task_id = task.get("id")
            if not task_id:
                continue
            
            # Validate for_each dependencies
            for_each = task.get("for_each", task.get("foreach"))
            if for_each:
                deps = self._extract_template_dependencies(for_each)
                for dep in deps:
                    if dep not in task_ids:
                        issues.append(DependencyIssue(
                            issue_type="invalid_foreach_dependency",
                            severity="error",
                            message=f"Task '{task_id}' for_each references non-existent task '{dep}'",
                            involved_tasks=[task_id, dep],
                            recommendation=f"Create task '{dep}' or fix the for_each expression in task '{task_id}'"
                        ))
            
            # Validate conditional dependencies
            condition = task.get("condition", task.get("if"))
            if condition:
                deps = self._extract_template_dependencies(condition)
                for dep in deps:
                    if dep not in task_ids:
                        issues.append(DependencyIssue(
                            issue_type="invalid_condition_dependency",
                            severity="error",
                            message=f"Task '{task_id}' condition references non-existent task '{dep}'",
                            involved_tasks=[task_id, dep],
                            recommendation=f"Create task '{dep}' or fix the condition in task '{task_id}'"
                        ))
            
            # Validate action_loop dependencies
            if "action_loop" in task:
                # Check until condition
                until = task.get("until")
                if until:
                    deps = self._extract_template_dependencies(until)
                    for dep in deps:
                        if dep not in task_ids:
                            issues.append(DependencyIssue(
                                issue_type="invalid_action_loop_dependency",
                                severity="error",
                                message=f"Task '{task_id}' action_loop until condition references non-existent task '{dep}'",
                                involved_tasks=[task_id, dep],
                                recommendation=f"Create task '{dep}' or fix the until condition in task '{task_id}'"
                            ))
                
                # Check while condition
                while_cond = task.get("while")
                if while_cond:
                    deps = self._extract_template_dependencies(while_cond)
                    for dep in deps:
                        if dep not in task_ids:
                            issues.append(DependencyIssue(
                                issue_type="invalid_action_loop_dependency",
                                severity="error",
                                message=f"Task '{task_id}' action_loop while condition references non-existent task '{dep}'",
                                involved_tasks=[task_id, dep],
                                recommendation=f"Create task '{dep}' or fix the while condition in task '{task_id}'"
                            ))
            
            # Validate parallel queue dependencies
            if "create_parallel_queue" in task:
                queue_config = task["create_parallel_queue"]
                if isinstance(queue_config, dict):
                    on_expr = queue_config.get("on")
                    if on_expr:
                        deps = self._extract_template_dependencies(on_expr)
                        for dep in deps:
                            if dep not in task_ids:
                                issues.append(DependencyIssue(
                                    issue_type="invalid_parallel_queue_dependency",
                                    severity="error",
                                    message=f"Task '{task_id}' parallel queue 'on' references non-existent task '{dep}'",
                                    involved_tasks=[task_id, dep],
                                    recommendation=f"Create task '{dep}' or fix the parallel queue configuration in task '{task_id}'"
                                ))
        
        return issues
    
    def _compute_execution_order(
        self, 
        dependency_graph: Any, 
        tasks: List[Dict[str, Any]]
    ) -> Optional[List[str]]:
        """Compute a valid execution order for tasks."""
        if self.use_networkx and dependency_graph:
            try:
                # Use topological sort
                return list(nx.topological_sort(dependency_graph))
            except nx.NetworkXError:
                # Graph has cycles, can't compute topological order
                return None
        else:
            # Custom topological sort implementation
            return self._topological_sort_custom(tasks)
    
    def _topological_sort_custom(self, tasks: List[Dict[str, Any]]) -> Optional[List[str]]:
        """Custom topological sort implementation."""
        # Build adjacency list and in-degree count
        graph = {}
        in_degree = {}
        all_tasks = set()
        
        for task in tasks:
            task_id = task.get("id")
            if not task_id:
                continue
            
            all_tasks.add(task_id)
            dependencies = self._get_task_dependencies(task)
            
            graph[task_id] = []
            in_degree[task_id] = in_degree.get(task_id, 0)
            
            for dep in dependencies:
                if dep not in graph:
                    graph[dep] = []
                graph[dep].append(task_id)
                in_degree[task_id] = in_degree.get(task_id, 0) + 1
        
        # Kahn's algorithm
        queue = [task_id for task_id in all_tasks if in_degree.get(task_id, 0) == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for neighbor in graph.get(current, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check if all tasks were processed
        if len(result) != len(all_tasks):
            # Cycle detected
            return None
        
        return result