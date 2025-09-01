"""Dependency Handling System - Issue #312 Stream C

Comprehensive dependency handling system providing:
- Dependency resolution algorithms with graph traversal
- Dependency chain validation and conflict detection
- Resource requirement management
- Integration with EnhancedToolRegistry and SetupSystem
- Automatic dependency installation orchestration
"""

import logging
from typing import Dict, List, Set, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio
from collections import defaultdict, deque

from .registry import (
    EnhancedToolRegistry, EnhancedToolMetadata, InstallationStatus, 
    CompatibilityRequirement, get_enhanced_registry
)
from .setup import SetupSystem, InstallationResult, get_setup_system

logger = logging.getLogger(__name__)


class DependencyType(Enum):
    """Types of dependencies."""
    REQUIRED = "required"           # Must be available
    OPTIONAL = "optional"           # Nice to have, but not required
    RUNTIME = "runtime"             # Required during execution
    BUILD = "build"                 # Required during setup/build
    TEST = "test"                   # Required for testing
    DEVELOPMENT = "development"     # Required for development


class ConflictSeverity(Enum):
    """Severity levels for dependency conflicts."""
    CRITICAL = "critical"           # Breaks functionality
    WARNING = "warning"             # May cause issues
    INFO = "info"                   # Informational only


@dataclass
class DependencyNode:
    """Represents a single dependency in the dependency graph."""
    name: str
    version: Optional[str] = None
    dependency_type: DependencyType = DependencyType.REQUIRED
    metadata: Optional[EnhancedToolMetadata] = None
    
    # Graph relationships
    dependencies: Set[str] = field(default_factory=set)  # What this depends on
    dependents: Set[str] = field(default_factory=set)    # What depends on this
    
    # Resolution state
    resolved: bool = False
    installation_status: InstallationStatus = InstallationStatus.NEEDS_INSTALL
    resolution_order: int = -1
    
    def __hash__(self) -> int:
        return hash((self.name, self.version))
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, DependencyNode):
            return False
        return self.name == other.name and self.version == other.version


@dataclass
class DependencyConflict:
    """Represents a conflict between dependencies."""
    tool1: str
    tool2: str
    conflict_type: str
    severity: ConflictSeverity
    description: str
    resolution_suggestions: List[str] = field(default_factory=list)


@dataclass
class DependencyChain:
    """Represents a chain of dependencies for a tool."""
    root_tool: str
    nodes: Dict[str, DependencyNode]
    conflicts: List[DependencyConflict] = field(default_factory=list)
    resolution_order: List[str] = field(default_factory=list)
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class ResolutionResult:
    """Result of dependency resolution."""
    success: bool
    resolved_tools: List[str] = field(default_factory=list)
    installation_plan: List[Tuple[str, InstallationStatus]] = field(default_factory=list)
    conflicts: List[DependencyConflict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    resolution_time: float = 0.0
    dependency_chain: Optional[DependencyChain] = None


class DependencyResolver:
    """Resolves dependencies using graph algorithms."""
    
    def __init__(self, 
                 registry: Optional[EnhancedToolRegistry] = None,
                 setup_system: Optional[SetupSystem] = None):
        self.registry = registry or get_enhanced_registry()
        self.setup_system = setup_system or get_setup_system()
        
        # Caching for performance
        self._dependency_cache: Dict[str, DependencyChain] = {}
        self._conflict_cache: Dict[Tuple[str, str], DependencyConflict] = {}
        
        logger.info("Dependency resolver initialized")
    
    def resolve_dependencies(self, tool_name: str, 
                           include_optional: bool = False) -> ResolutionResult:
        """Resolve all dependencies for a tool."""
        start_time = datetime.now()
        
        try:
            logger.info(f"Resolving dependencies for {tool_name}")
            
            # Check if tool exists
            if tool_name not in self.registry.enhanced_metadata:
                return ResolutionResult(
                    success=False,
                    errors=[f"Tool {tool_name} not found in registry"]
                )
            
            # Build dependency graph
            dependency_chain = self._build_dependency_graph(tool_name, include_optional)
            
            # Validate the chain
            self._validate_dependency_chain(dependency_chain)
            
            if not dependency_chain.is_valid:
                return ResolutionResult(
                    success=False,
                    errors=dependency_chain.validation_errors,
                    conflicts=dependency_chain.conflicts,
                    dependency_chain=dependency_chain
                )
            
            # Calculate resolution order
            resolution_order = self._calculate_resolution_order(dependency_chain)
            dependency_chain.resolution_order = resolution_order
            
            # Create installation plan
            installation_plan = self._create_installation_plan(dependency_chain)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            result = ResolutionResult(
                success=True,
                resolved_tools=resolution_order,
                installation_plan=installation_plan,
                conflicts=dependency_chain.conflicts,
                resolution_time=duration,
                dependency_chain=dependency_chain
            )
            
            logger.info(f"Dependency resolution completed for {tool_name} in {duration:.2f}s")
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Dependency resolution failed for {tool_name}: {e}")
            return ResolutionResult(
                success=False,
                errors=[str(e)],
                resolution_time=duration
            )
    
    def _build_dependency_graph(self, root_tool: str, 
                               include_optional: bool = False) -> DependencyChain:
        """Build a complete dependency graph for a tool."""
        
        # Check cache first
        cache_key = f"{root_tool}:{include_optional}"
        if cache_key in self._dependency_cache:
            return self._dependency_cache[cache_key]
        
        chain = DependencyChain(root_tool=root_tool, nodes={})
        visited = set()
        
        def visit_node(tool_name: str, depth: int = 0) -> None:
            if tool_name in visited or depth > 20:  # Prevent infinite loops
                return
            
            visited.add(tool_name)
            
            # Get metadata
            metadata = self.registry.enhanced_metadata.get(tool_name)
            if not metadata:
                chain.validation_errors.append(f"Tool {tool_name} not found in registry")
                return
            
            # Create dependency node
            node = DependencyNode(
                name=tool_name,
                metadata=metadata,
                installation_status=self.registry.installation_tracker.get(
                    tool_name, InstallationStatus.NEEDS_INSTALL
                )
            )
            chain.nodes[tool_name] = node
            
            # Process required dependencies
            for dep_name in metadata.requires:
                node.dependencies.add(dep_name)
                
                # Add reverse dependency
                if dep_name in chain.nodes:
                    chain.nodes[dep_name].dependents.add(tool_name)
                
                # Recursively visit dependency
                visit_node(dep_name, depth + 1)
            
            # Process compatibility requirements
            for req in metadata.compatibility_requirements:
                if req.required or include_optional:
                    node.dependencies.add(req.name)
                    
                    # Add reverse dependency
                    if req.name in chain.nodes:
                        chain.nodes[req.name].dependents.add(tool_name)
                    
                    # Recursively visit dependency
                    visit_node(req.name, depth + 1)
        
        # Start traversal from root tool
        visit_node(root_tool)
        
        # Cache the result
        self._dependency_cache[cache_key] = chain
        
        return chain
    
    def _validate_dependency_chain(self, chain: DependencyChain) -> None:
        """Validate a dependency chain for conflicts and issues."""
        
        # Check for circular dependencies
        circular_deps = self._detect_circular_dependencies(chain)
        if circular_deps:
            chain.is_valid = False
            chain.validation_errors.append(f"Circular dependencies detected: {' -> '.join(circular_deps)}")
        
        # Check for conflicts
        conflicts = self._detect_conflicts(chain)
        chain.conflicts.extend(conflicts)
        
        # Check for missing dependencies
        for node_name, node in chain.nodes.items():
            for dep_name in node.dependencies:
                if dep_name not in chain.nodes:
                    # Try to find the dependency in registry
                    if dep_name not in self.registry.enhanced_metadata:
                        chain.validation_errors.append(
                            f"Missing dependency: {dep_name} required by {node_name}"
                        )
                        chain.is_valid = False
        
        # Check version compatibility
        version_conflicts = self._check_version_compatibility(chain)
        chain.conflicts.extend(version_conflicts)
        
        # Mark as invalid if there are critical conflicts
        critical_conflicts = [c for c in chain.conflicts if c.severity == ConflictSeverity.CRITICAL]
        if critical_conflicts:
            chain.is_valid = False
            chain.validation_errors.extend([
                f"Critical conflict: {c.description}" for c in critical_conflicts
            ])
    
    def _detect_circular_dependencies(self, chain: DependencyChain) -> Optional[List[str]]:
        """Detect circular dependencies using DFS."""
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {node: WHITE for node in chain.nodes}
        path = []
        
        def dfs(node_name: str) -> Optional[List[str]]:
            if colors[node_name] == GRAY:  # Back edge found
                # Find the cycle
                cycle_start = path.index(node_name)
                return path[cycle_start:] + [node_name]
            
            if colors[node_name] == BLACK:
                return None
            
            colors[node_name] = GRAY
            path.append(node_name)
            
            node = chain.nodes[node_name]
            for dep_name in node.dependencies:
                if dep_name in chain.nodes:
                    cycle = dfs(dep_name)
                    if cycle:
                        return cycle
            
            path.pop()
            colors[node_name] = BLACK
            return None
        
        for node_name in chain.nodes:
            if colors[node_name] == WHITE:
                cycle = dfs(node_name)
                if cycle:
                    return cycle
        
        return None
    
    def _detect_conflicts(self, chain: DependencyChain) -> List[DependencyConflict]:
        """Detect conflicts between tools in the dependency chain."""
        conflicts = []
        
        for node_name, node in chain.nodes.items():
            if not node.metadata:
                continue
            
            # Check conflicts with other nodes
            for other_name, other_node in chain.nodes.items():
                if node_name == other_name or not other_node.metadata:
                    continue
                
                # Check if tools explicitly conflict
                if other_name in node.metadata.conflicts_with:
                    conflict = DependencyConflict(
                        tool1=node_name,
                        tool2=other_name,
                        conflict_type="explicit_conflict",
                        severity=ConflictSeverity.CRITICAL,
                        description=f"{node_name} explicitly conflicts with {other_name}",
                        resolution_suggestions=[
                            f"Remove one of the conflicting tools",
                            f"Find alternative tools that don't conflict"
                        ]
                    )
                    conflicts.append(conflict)
                
                # Check for version conflicts
                if node_name == other_name and node.version != other_node.version:
                    conflict = DependencyConflict(
                        tool1=node_name,
                        tool2=other_name,
                        conflict_type="version_conflict",
                        severity=ConflictSeverity.WARNING,
                        description=f"Version conflict for {node_name}: {node.version} vs {other_node.version}",
                        resolution_suggestions=[
                            f"Use a single version of {node_name}",
                            f"Check compatibility between versions"
                        ]
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    def _check_version_compatibility(self, chain: DependencyChain) -> List[DependencyConflict]:
        """Check version compatibility requirements."""
        conflicts = []
        
        for node_name, node in chain.nodes.items():
            if not node.metadata:
                continue
            
            for req in node.metadata.compatibility_requirements:
                if req.name in chain.nodes:
                    dep_node = chain.nodes[req.name]
                    if dep_node.metadata:
                        if not req.is_compatible(dep_node.metadata.version_info):
                            conflict = DependencyConflict(
                                tool1=node_name,
                                tool2=req.name,
                                conflict_type="version_incompatibility",
                                severity=ConflictSeverity.CRITICAL if req.required else ConflictSeverity.WARNING,
                                description=f"{node_name} requires {req.name} version {req.min_version}-{req.max_version}, but found {dep_node.metadata.version_info}",
                                resolution_suggestions=[
                                    f"Upgrade/downgrade {req.name} to compatible version",
                                    f"Find alternative version of {node_name}"
                                ]
                            )
                            conflicts.append(conflict)
        
        return conflicts
    
    def _calculate_resolution_order(self, chain: DependencyChain) -> List[str]:
        """Calculate the order in which dependencies should be resolved using topological sort."""
        
        # Build adjacency list
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        # Initialize all nodes
        for node_name in chain.nodes:
            in_degree[node_name] = 0
        
        # Build graph and calculate in-degrees
        for node_name, node in chain.nodes.items():
            for dep_name in node.dependencies:
                if dep_name in chain.nodes:  # Only include nodes in our chain
                    graph[dep_name].append(node_name)
                    in_degree[node_name] += 1
        
        # Topological sort using Kahn's algorithm
        queue = deque([node for node in chain.nodes if in_degree[node] == 0])
        result = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            # Update in-degrees of neighbors
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # If we couldn't process all nodes, there's a cycle
        if len(result) != len(chain.nodes):
            logger.warning(f"Could not establish complete resolution order for {chain.root_tool}")
            # Return partial order + remaining nodes
            remaining = [node for node in chain.nodes if node not in result]
            result.extend(remaining)
        
        return result
    
    def _create_installation_plan(self, chain: DependencyChain) -> List[Tuple[str, InstallationStatus]]:
        """Create an installation plan based on the dependency chain."""
        plan = []
        
        for tool_name in chain.resolution_order:
            if tool_name in chain.nodes:
                node = chain.nodes[tool_name]
                current_status = self.registry.installation_tracker.get(
                    tool_name, InstallationStatus.NEEDS_INSTALL
                )
                
                if current_status != InstallationStatus.AVAILABLE:
                    plan.append((tool_name, current_status))
        
        return plan
    
    async def install_dependencies(self, resolution_result: ResolutionResult) -> Dict[str, InstallationResult]:
        """Install dependencies according to the resolution plan."""
        if not resolution_result.success:
            logger.error("Cannot install dependencies from failed resolution")
            return {}
        
        logger.info(f"Installing {len(resolution_result.installation_plan)} dependencies")
        
        installation_results = {}
        
        # Install dependencies in resolution order
        for tool_name, status in resolution_result.installation_plan:
            try:
                logger.info(f"Installing dependency: {tool_name}")
                result = await self.setup_system.setup_tool(tool_name)
                installation_results[tool_name] = result
                
                if not result.success:
                    logger.error(f"Failed to install {tool_name}: {result.error}")
                    # Continue with other installations
                
            except Exception as e:
                logger.error(f"Exception during installation of {tool_name}: {e}")
                installation_results[tool_name] = InstallationResult(
                    success=False,
                    package_name=tool_name,
                    package_manager="unknown",
                    error=str(e)
                )
        
        return installation_results
    
    def get_dependency_tree(self, tool_name: str, max_depth: int = 10) -> Dict[str, Any]:
        """Get a hierarchical representation of the dependency tree."""
        
        def build_tree(node_name: str, depth: int = 0) -> Dict[str, Any]:
            if depth > max_depth or node_name not in self.registry.enhanced_metadata:
                return {"name": node_name, "error": "Not found or max depth reached"}
            
            metadata = self.registry.enhanced_metadata[node_name]
            status = self.registry.installation_tracker.get(node_name, InstallationStatus.NEEDS_INSTALL)
            
            tree = {
                "name": node_name,
                "version": str(metadata.version_info),
                "status": status.value,
                "dependencies": [],
                "provides": metadata.provides,
                "conflicts": metadata.conflicts_with
            }
            
            # Add direct dependencies
            for dep_name in metadata.requires:
                tree["dependencies"].append(build_tree(dep_name, depth + 1))
            
            return tree
        
        return build_tree(tool_name)
    
    def clear_cache(self) -> None:
        """Clear dependency resolution cache."""
        self._dependency_cache.clear()
        self._conflict_cache.clear()
        logger.info("Dependency resolver cache cleared")


class DependencyManager:
    """High-level dependency management interface."""
    
    def __init__(self):
        self.resolver = DependencyResolver()
        self.registry = get_enhanced_registry()
        self.setup_system = get_setup_system()
        
        logger.info("Dependency manager initialized")
    
    async def ensure_dependencies(self, tool_name: str, 
                                 install_missing: bool = True) -> Tuple[bool, List[str]]:
        """Ensure all dependencies for a tool are available."""
        logger.info(f"Ensuring dependencies for {tool_name}")
        
        # Resolve dependencies
        resolution = self.resolver.resolve_dependencies(tool_name)
        
        if not resolution.success:
            logger.error(f"Dependency resolution failed for {tool_name}: {resolution.errors}")
            return False, resolution.errors
        
        # Check if all dependencies are available
        missing_deps = []
        for tool, status in resolution.installation_plan:
            if status != InstallationStatus.AVAILABLE:
                missing_deps.append(tool)
        
        if not missing_deps:
            logger.info(f"All dependencies for {tool_name} are available")
            return True, []
        
        if not install_missing:
            logger.warning(f"Missing dependencies for {tool_name}: {missing_deps}")
            return False, [f"Missing dependencies: {', '.join(missing_deps)}"]
        
        # Install missing dependencies
        logger.info(f"Installing missing dependencies: {missing_deps}")
        installation_results = await self.resolver.install_dependencies(resolution)
        
        # Check installation results
        failed_installs = []
        for tool, result in installation_results.items():
            if not result.success:
                failed_installs.append(f"{tool}: {result.error}")
        
        if failed_installs:
            logger.error(f"Some dependencies failed to install: {failed_installs}")
            return False, failed_installs
        
        logger.info(f"Successfully ensured all dependencies for {tool_name}")
        return True, []
    
    def validate_tool_dependencies(self, tool_name: str) -> Tuple[bool, List[str]]:
        """Validate that a tool's dependencies are consistent."""
        resolution = self.resolver.resolve_dependencies(tool_name)
        
        if not resolution.success:
            return False, resolution.errors
        
        issues = []
        
        # Check for conflicts
        if resolution.conflicts:
            critical_conflicts = [c for c in resolution.conflicts if c.severity == ConflictSeverity.CRITICAL]
            if critical_conflicts:
                issues.extend([f"Critical conflict: {c.description}" for c in critical_conflicts])
            
            warning_conflicts = [c for c in resolution.conflicts if c.severity == ConflictSeverity.WARNING]
            if warning_conflicts:
                issues.extend([f"Warning: {c.description}" for c in warning_conflicts])
        
        return len(issues) == 0, issues
    
    def get_dependency_status(self, tool_name: str) -> Dict[str, Any]:
        """Get comprehensive dependency status for a tool."""
        resolution = self.resolver.resolve_dependencies(tool_name)
        
        status = {
            "tool_name": tool_name,
            "resolution_success": resolution.success,
            "total_dependencies": len(resolution.resolved_tools),
            "dependencies": {},
            "conflicts": [],
            "installation_needed": []
        }
        
        if resolution.success and resolution.dependency_chain:
            # Add dependency details
            for dep_name, node in resolution.dependency_chain.nodes.items():
                status["dependencies"][dep_name] = {
                    "status": node.installation_status.value,
                    "version": str(node.metadata.version_info) if node.metadata else "unknown",
                    "direct_dependencies": list(node.dependencies),
                    "dependents": list(node.dependents)
                }
        
        # Add conflicts
        status["conflicts"] = [
            {
                "tool1": c.tool1,
                "tool2": c.tool2,
                "type": c.conflict_type,
                "severity": c.severity.value,
                "description": c.description,
                "suggestions": c.resolution_suggestions
            }
            for c in resolution.conflicts
        ]
        
        # Add installation plan
        status["installation_needed"] = [
            {"tool": tool, "status": status.value}
            for tool, status in resolution.installation_plan
        ]
        
        return status
    
    def list_tools_requiring(self, dependency_name: str) -> List[str]:
        """List all tools that require a specific dependency."""
        requiring_tools = []
        
        for tool_name, metadata in self.registry.enhanced_metadata.items():
            if dependency_name in metadata.requires:
                requiring_tools.append(tool_name)
            
            # Check compatibility requirements
            for req in metadata.compatibility_requirements:
                if req.name == dependency_name and req.required:
                    requiring_tools.append(tool_name)
        
        return requiring_tools
    
    def find_dependency_conflicts(self) -> List[DependencyConflict]:
        """Find all dependency conflicts across registered tools."""
        all_conflicts = []
        
        for tool_name in self.registry.enhanced_metadata:
            resolution = self.resolver.resolve_dependencies(tool_name)
            all_conflicts.extend(resolution.conflicts)
        
        # Remove duplicates (conflicts might be found multiple times)
        unique_conflicts = []
        seen = set()
        
        for conflict in all_conflicts:
            conflict_key = tuple(sorted([conflict.tool1, conflict.tool2]) + [conflict.conflict_type])
            if conflict_key not in seen:
                seen.add(conflict_key)
                unique_conflicts.append(conflict)
        
        return unique_conflicts


# Global dependency manager instance
dependency_manager = DependencyManager()


def get_dependency_manager() -> DependencyManager:
    """Get the global dependency manager instance."""
    return dependency_manager


# Convenience functions
async def ensure_dependencies(tool_name: str, install_missing: bool = True) -> Tuple[bool, List[str]]:
    """Convenience function to ensure dependencies for a tool."""
    return await dependency_manager.ensure_dependencies(tool_name, install_missing)


def validate_dependencies(tool_name: str) -> Tuple[bool, List[str]]:
    """Convenience function to validate tool dependencies."""
    return dependency_manager.validate_tool_dependencies(tool_name)


def get_dependency_status(tool_name: str) -> Dict[str, Any]:
    """Convenience function to get dependency status."""
    return dependency_manager.get_dependency_status(tool_name)


__all__ = [
    "DependencyResolver",
    "DependencyManager", 
    "DependencyNode",
    "DependencyChain",
    "DependencyConflict",
    "ResolutionResult",
    "DependencyType",
    "ConflictSeverity",
    "dependency_manager",
    "get_dependency_manager",
    "ensure_dependencies",
    "validate_dependencies",
    "get_dependency_status"
]