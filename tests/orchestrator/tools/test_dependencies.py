"""Tests for dependency handling system."""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from src.orchestrator.tools.dependencies import (
    DependencyResolver, DependencyManager, DependencyNode, DependencyChain,
    DependencyConflict, ResolutionResult, DependencyType, ConflictSeverity,
    get_dependency_manager, ensure_dependencies, validate_dependencies
)
from src.orchestrator.tools.registry import (
    EnhancedToolRegistry, EnhancedToolMetadata, VersionInfo, 
    CompatibilityRequirement, InstallationStatus, ToolSource,
    ToolCategory, SecurityLevel
)
from src.orchestrator.tools.base import Tool as OrchestratorTool


class TestDependencyNode:
    """Test the DependencyNode class."""
    
    def test_dependency_node_creation(self):
        """Test creating a dependency node."""
        node = DependencyNode(
            name="test-tool",
            version="1.0.0",
            dependency_type=DependencyType.REQUIRED
        )
        
        assert node.name == "test-tool"
        assert node.version == "1.0.0"
        assert node.dependency_type == DependencyType.REQUIRED
        assert not node.resolved
        assert node.installation_status == InstallationStatus.NEEDS_INSTALL
        assert len(node.dependencies) == 0
        assert len(node.dependents) == 0
    
    def test_dependency_node_hash_and_equality(self):
        """Test hashing and equality of dependency nodes."""
        node1 = DependencyNode(name="test-tool", version="1.0.0")
        node2 = DependencyNode(name="test-tool", version="1.0.0")
        node3 = DependencyNode(name="test-tool", version="1.1.0")
        
        assert node1 == node2
        assert node1 != node3
        assert hash(node1) == hash(node2)
        assert hash(node1) != hash(node3)


class TestDependencyResolver:
    """Test the DependencyResolver class."""
    
    @pytest.fixture
    def mock_registry(self):
        """Create a mock registry with test tools."""
        registry = Mock(spec=EnhancedToolRegistry)
        
        # Create test metadata
        tool_a_metadata = EnhancedToolMetadata(
            name="tool-a",
            source=ToolSource.ORCHESTRATOR,
            category=ToolCategory.CUSTOM,
            description="Tool A",
            version_info=VersionInfo(1, 0, 0),
            requires=["tool-b"],
            compatibility_requirements=[
                CompatibilityRequirement(
                    name="tool-c",
                    min_version=VersionInfo(1, 0, 0),
                    required=True
                )
            ]
        )
        
        tool_b_metadata = EnhancedToolMetadata(
            name="tool-b",
            source=ToolSource.ORCHESTRATOR,
            category=ToolCategory.CUSTOM,
            description="Tool B",
            version_info=VersionInfo(1, 0, 0),
            requires=[]
        )
        
        tool_c_metadata = EnhancedToolMetadata(
            name="tool-c",
            source=ToolSource.ORCHESTRATOR,
            category=ToolCategory.CUSTOM,
            description="Tool C",
            version_info=VersionInfo(1, 0, 0),
            requires=[],
            conflicts_with=["tool-d"]
        )
        
        tool_d_metadata = EnhancedToolMetadata(
            name="tool-d",
            source=ToolSource.ORCHESTRATOR,
            category=ToolCategory.CUSTOM,
            description="Tool D",
            version_info=VersionInfo(1, 0, 0),
            requires=[]
        )
        
        registry.enhanced_metadata = {
            "tool-a": tool_a_metadata,
            "tool-b": tool_b_metadata,
            "tool-c": tool_c_metadata,
            "tool-d": tool_d_metadata
        }
        
        registry.installation_tracker = {
            "tool-a": InstallationStatus.NEEDS_INSTALL,
            "tool-b": InstallationStatus.AVAILABLE,
            "tool-c": InstallationStatus.NEEDS_INSTALL,
            "tool-d": InstallationStatus.AVAILABLE
        }
        
        return registry
    
    @pytest.fixture
    def mock_setup_system(self):
        """Create a mock setup system."""
        setup_system = Mock()
        return setup_system
    
    @pytest.fixture
    def resolver(self, mock_registry, mock_setup_system):
        """Create a dependency resolver with mocks."""
        return DependencyResolver(mock_registry, mock_setup_system)
    
    def test_resolve_dependencies_success(self, resolver):
        """Test successful dependency resolution."""
        result = resolver.resolve_dependencies("tool-a")
        
        assert result.success
        assert "tool-a" in result.resolved_tools
        assert "tool-b" in result.resolved_tools
        assert "tool-c" in result.resolved_tools
        assert len(result.errors) == 0
    
    def test_resolve_dependencies_tool_not_found(self, resolver):
        """Test dependency resolution for non-existent tool."""
        result = resolver.resolve_dependencies("non-existent-tool")
        
        assert not result.success
        assert "not found in registry" in result.errors[0]
    
    def test_build_dependency_graph(self, resolver):
        """Test building dependency graph."""
        chain = resolver._build_dependency_graph("tool-a")
        
        assert chain.root_tool == "tool-a"
        assert "tool-a" in chain.nodes
        assert "tool-b" in chain.nodes
        assert "tool-c" in chain.nodes
        
        # Check dependencies
        tool_a_node = chain.nodes["tool-a"]
        assert "tool-b" in tool_a_node.dependencies
        assert "tool-c" in tool_a_node.dependencies
    
    def test_detect_circular_dependencies(self, resolver, mock_registry):
        """Test detection of circular dependencies."""
        # Create circular dependency: tool-a -> tool-b -> tool-a
        mock_registry.enhanced_metadata["tool-b"].requires = ["tool-a"]
        
        chain = resolver._build_dependency_graph("tool-a")
        circular_deps = resolver._detect_circular_dependencies(chain)
        
        assert circular_deps is not None
        assert "tool-a" in circular_deps
        assert "tool-b" in circular_deps
    
    def test_detect_conflicts(self, resolver):
        """Test conflict detection."""
        # Create a chain that includes conflicting tools
        chain = resolver._build_dependency_graph("tool-a")
        # Manually add tool-d to create conflict with tool-c
        tool_d_node = DependencyNode(
            name="tool-d",
            metadata=resolver.registry.enhanced_metadata["tool-d"]
        )
        chain.nodes["tool-d"] = tool_d_node
        
        conflicts = resolver._detect_conflicts(chain)
        
        assert len(conflicts) > 0
        # Should find conflict between tool-c and tool-d
        conflict_found = any(
            (c.tool1 == "tool-c" and c.tool2 == "tool-d") or
            (c.tool1 == "tool-d" and c.tool2 == "tool-c")
            for c in conflicts
        )
        assert conflict_found
    
    def test_calculate_resolution_order(self, resolver):
        """Test calculation of resolution order using topological sort."""
        chain = resolver._build_dependency_graph("tool-a")
        order = resolver._calculate_resolution_order(chain)
        
        # tool-b and tool-c should come before tool-a
        tool_a_index = order.index("tool-a")
        tool_b_index = order.index("tool-b")
        tool_c_index = order.index("tool-c")
        
        assert tool_b_index < tool_a_index
        assert tool_c_index < tool_a_index
    
    @pytest.mark.asyncio
    async def test_install_dependencies(self, resolver):
        """Test dependency installation."""
        # Create a successful resolution result
        result = resolver.resolve_dependencies("tool-a")
        
        # Mock installation results
        mock_result = Mock()
        mock_result.success = True
        resolver.setup_system.setup_tool.return_value = mock_result
        
        installation_results = await resolver.install_dependencies(result)
        
        # Should have attempted to install tools that need installation
        assert len(installation_results) > 0
        resolver.setup_system.setup_tool.assert_called()
    
    def test_get_dependency_tree(self, resolver):
        """Test getting hierarchical dependency tree."""
        tree = resolver.get_dependency_tree("tool-a", max_depth=5)
        
        assert tree["name"] == "tool-a"
        assert "dependencies" in tree
        assert len(tree["dependencies"]) > 0
        
        # Check that dependencies are included
        dep_names = [dep["name"] for dep in tree["dependencies"]]
        assert "tool-b" in dep_names


class TestDependencyManager:
    """Test the DependencyManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create a dependency manager for testing."""
        with patch('src.orchestrator.tools.dependencies.get_enhanced_registry'), \
             patch('src.orchestrator.tools.dependencies.get_setup_system'):
            return DependencyManager()
    
    @pytest.mark.asyncio
    async def test_ensure_dependencies_success(self, manager):
        """Test successful dependency ensuring."""
        # Mock successful resolution
        manager.resolver.resolve_dependencies = Mock(return_value=ResolutionResult(
            success=True,
            resolved_tools=["tool-a", "tool-b"],
            installation_plan=[]  # No installation needed
        ))
        
        success, errors = await manager.ensure_dependencies("tool-a")
        
        assert success
        assert len(errors) == 0
    
    @pytest.mark.asyncio
    async def test_ensure_dependencies_with_installation(self, manager):
        """Test dependency ensuring with installation."""
        # Mock resolution with installation needed
        manager.resolver.resolve_dependencies = Mock(return_value=ResolutionResult(
            success=True,
            resolved_tools=["tool-a", "tool-b"],
            installation_plan=[("tool-b", InstallationStatus.NEEDS_INSTALL)]
        ))
        
        # Mock successful installation
        manager.resolver.install_dependencies = Mock(return_value={
            "tool-b": Mock(success=True)
        })
        
        success, errors = await manager.ensure_dependencies("tool-a", install_missing=True)
        
        assert success
        assert len(errors) == 0
        manager.resolver.install_dependencies.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ensure_dependencies_installation_failure(self, manager):
        """Test dependency ensuring with installation failure."""
        # Mock resolution with installation needed
        manager.resolver.resolve_dependencies = Mock(return_value=ResolutionResult(
            success=True,
            resolved_tools=["tool-a", "tool-b"],
            installation_plan=[("tool-b", InstallationStatus.NEEDS_INSTALL)]
        ))
        
        # Mock failed installation
        manager.resolver.install_dependencies = Mock(return_value={
            "tool-b": Mock(success=False, error="Installation failed")
        })
        
        success, errors = await manager.ensure_dependencies("tool-a", install_missing=True)
        
        assert not success
        assert len(errors) > 0
        assert "Installation failed" in errors[0]
    
    def test_validate_tool_dependencies_success(self, manager):
        """Test successful dependency validation."""
        # Mock successful resolution without conflicts
        manager.resolver.resolve_dependencies = Mock(return_value=ResolutionResult(
            success=True,
            conflicts=[]
        ))
        
        success, issues = manager.validate_tool_dependencies("tool-a")
        
        assert success
        assert len(issues) == 0
    
    def test_validate_tool_dependencies_with_conflicts(self, manager):
        """Test dependency validation with conflicts."""
        # Mock resolution with conflicts
        conflict = DependencyConflict(
            tool1="tool-c",
            tool2="tool-d",
            conflict_type="explicit_conflict",
            severity=ConflictSeverity.CRITICAL,
            description="Critical conflict between tools"
        )
        
        manager.resolver.resolve_dependencies = Mock(return_value=ResolutionResult(
            success=True,
            conflicts=[conflict]
        ))
        
        success, issues = manager.validate_tool_dependencies("tool-a")
        
        assert not success
        assert len(issues) > 0
        assert "Critical conflict" in issues[0]
    
    def test_get_dependency_status(self, manager):
        """Test getting comprehensive dependency status."""
        # Mock dependency chain
        mock_chain = Mock()
        mock_chain.nodes = {
            "tool-b": Mock(
                installation_status=InstallationStatus.AVAILABLE,
                metadata=Mock(version_info=VersionInfo(1, 0, 0)),
                dependencies=set(),
                dependents={"tool-a"}
            )
        }
        
        # Mock resolution result
        manager.resolver.resolve_dependencies = Mock(return_value=ResolutionResult(
            success=True,
            resolved_tools=["tool-a", "tool-b"],
            dependency_chain=mock_chain,
            conflicts=[],
            installation_plan=[]
        ))
        
        status = manager.get_dependency_status("tool-a")
        
        assert status["tool_name"] == "tool-a"
        assert status["resolution_success"]
        assert "dependencies" in status
        assert "conflicts" in status
        assert "installation_needed" in status
    
    def test_list_tools_requiring(self, manager):
        """Test listing tools that require a specific dependency."""
        # Mock registry with tools
        manager.registry.enhanced_metadata = {
            "tool-a": Mock(requires=["tool-b"], compatibility_requirements=[]),
            "tool-c": Mock(requires=["tool-b"], compatibility_requirements=[]),
            "tool-d": Mock(requires=[], compatibility_requirements=[
                Mock(name="tool-b", required=True)
            ])
        }
        
        requiring_tools = manager.list_tools_requiring("tool-b")
        
        assert "tool-a" in requiring_tools
        assert "tool-c" in requiring_tools
        assert "tool-d" in requiring_tools


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.mark.asyncio
    async def test_ensure_dependencies(self):
        """Test the convenience function for ensuring dependencies."""
        with patch('src.orchestrator.tools.dependencies.dependency_manager') as mock_manager:
            mock_manager.ensure_dependencies = Mock(return_value=(True, []))
            
            success, errors = await ensure_dependencies("test-tool")
            
            assert success
            assert len(errors) == 0
            mock_manager.ensure_dependencies.assert_called_once_with("test-tool", True)
    
    def test_validate_dependencies(self):
        """Test the convenience function for validating dependencies."""
        with patch('src.orchestrator.tools.dependencies.dependency_manager') as mock_manager:
            mock_manager.validate_tool_dependencies = Mock(return_value=(True, []))
            
            success, issues = validate_dependencies("test-tool")
            
            assert success
            assert len(issues) == 0
            mock_manager.validate_tool_dependencies.assert_called_once_with("test-tool")


class TestIntegration:
    """Integration tests for dependency management."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_dependency_resolution(self):
        """Test complete dependency resolution workflow."""
        # This would be a more complex integration test
        # involving real registry and setup system
        pass
    
    def test_dependency_caching(self):
        """Test that dependency resolution results are cached properly."""
        # Test caching behavior
        pass
    
    def test_concurrent_dependency_resolution(self):
        """Test concurrent dependency resolution for multiple tools."""
        # Test thread safety and concurrent access
        pass


if __name__ == "__main__":
    pytest.main([__file__])