"""Tests for resource management system."""

import pytest
import asyncio
import threading
import time
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime, timedelta

from src.orchestrator.tools.resources import (
    ResourceManager, ResourceMonitor, ResourcePool, ResourceHandle,
    ResourceMetrics, ResourceLimits, ResourceType, ResourceState,
    get_resource_manager, allocate_resource, release_resource,
    release_tool_resources, managed_resource
)


class TestResourceMetrics:
    """Test the ResourceMetrics class."""
    
    def test_resource_metrics_creation(self):
        """Test creating resource metrics."""
        now = datetime.now()
        metrics = ResourceMetrics(
            allocated_at=now,
            last_accessed=now,
            access_count=5,
            total_usage_time=120.5,
            peak_memory=1024 * 1024,
            peak_cpu=45.5,
            errors=2
        )
        
        assert metrics.allocated_at == now
        assert metrics.last_accessed == now
        assert metrics.access_count == 5
        assert metrics.total_usage_time == 120.5
        assert metrics.peak_memory == 1024 * 1024
        assert metrics.peak_cpu == 45.5
        assert metrics.errors == 2


class TestResourceLimits:
    """Test the ResourceLimits class."""
    
    def test_resource_limits_creation(self):
        """Test creating resource limits."""
        limits = ResourceLimits(
            max_memory_mb=512,
            max_cpu_percent=80.0,
            max_disk_mb=1024,
            max_lifetime_seconds=3600,
            max_idle_seconds=300,
            max_access_count=1000
        )
        
        assert limits.max_memory_mb == 512
        assert limits.max_cpu_percent == 80.0
        assert limits.max_disk_mb == 1024
        assert limits.max_lifetime_seconds == 3600
        assert limits.max_idle_seconds == 300
        assert limits.max_access_count == 1000
    
    def test_default_resource_limits(self):
        """Test default resource limits."""
        limits = ResourceLimits()
        
        assert limits.max_memory_mb is None
        assert limits.max_cpu_percent is None
        assert limits.max_disk_mb is None
        assert limits.max_lifetime_seconds is None
        assert limits.max_idle_seconds == 300  # Default 5 minutes
        assert limits.max_access_count is None


class TestResourceHandle:
    """Test the ResourceHandle class."""
    
    def test_resource_handle_creation(self):
        """Test creating a resource handle."""
        test_data = {"test": "data"}
        handle = ResourceHandle(
            resource_id="test-resource-1",
            resource_type=ResourceType.MEMORY,
            tool_name="test-tool",
            resource_data=test_data
        )
        
        assert handle.resource_id == "test-resource-1"
        assert handle.resource_type == ResourceType.MEMORY
        assert handle.tool_name == "test-tool"
        assert handle.resource_data == test_data
        assert handle.state == ResourceState.ALLOCATED
        assert isinstance(handle.metrics, ResourceMetrics)
        assert isinstance(handle.limits, ResourceLimits)
    
    def test_resource_handle_access(self):
        """Test accessing a resource."""
        handle = ResourceHandle(
            resource_id="test-resource-1",
            resource_type=ResourceType.MEMORY,
            tool_name="test-tool",
            resource_data={}
        )
        
        initial_access_count = handle.metrics.access_count
        initial_last_accessed = handle.metrics.last_accessed
        handle.state = ResourceState.IDLE
        
        # Wait a bit to ensure time difference
        time.sleep(0.01)
        
        handle.access()
        
        assert handle.metrics.access_count == initial_access_count + 1
        assert handle.metrics.last_accessed > initial_last_accessed
        assert handle.state == ResourceState.ACTIVE
    
    def test_resource_handle_is_expired(self):
        """Test checking if resource is expired."""
        # Test lifetime expiration
        limits = ResourceLimits(max_lifetime_seconds=1)
        handle = ResourceHandle(
            resource_id="test-resource-1",
            resource_type=ResourceType.MEMORY,
            tool_name="test-tool",
            resource_data={},
            limits=limits
        )
        
        # Should not be expired initially
        assert not handle.is_expired()
        
        # Mock old allocation time
        handle.metrics.allocated_at = datetime.now() - timedelta(seconds=2)
        assert handle.is_expired()
    
    def test_resource_handle_exceeds_limits(self):
        """Test checking if resource exceeds limits."""
        limits = ResourceLimits(
            max_memory_mb=100,
            max_cpu_percent=50.0
        )
        handle = ResourceHandle(
            resource_id="test-resource-1",
            resource_type=ResourceType.MEMORY,
            tool_name="test-tool",
            resource_data={},
            limits=limits
        )
        
        # Test within limits
        violations = handle.exceeds_limits(current_memory=50 * 1024 * 1024, current_cpu=25.0)
        assert len(violations) == 0
        
        # Test exceeding memory limit
        violations = handle.exceeds_limits(current_memory=200 * 1024 * 1024, current_cpu=25.0)
        assert len(violations) == 1
        assert "Memory usage" in violations[0]
        
        # Test exceeding CPU limit
        violations = handle.exceeds_limits(current_memory=50 * 1024 * 1024, current_cpu=75.0)
        assert len(violations) == 1
        assert "CPU usage" in violations[0]


class TestResourcePool:
    """Test the ResourcePool class."""
    
    def test_resource_pool_creation(self):
        """Test creating a resource pool."""
        pool = ResourcePool(ResourceType.MEMORY, max_size=5)
        
        assert pool.resource_type == ResourceType.MEMORY
        assert pool.max_size == 5
        assert len(pool._available) == 0
        assert len(pool._in_use) == 0
    
    def test_resource_pool_get_resource(self):
        """Test getting a resource from pool."""
        pool = ResourcePool(ResourceType.MEMORY, max_size=2)
        
        # Factory function to create test resources
        def factory():
            return {"created": datetime.now()}
        
        # Get first resource
        handle1 = pool.get_resource("tool-a", factory)
        assert handle1 is not None
        assert handle1.tool_name == "tool-a"
        assert handle1.resource_id in pool._in_use
        
        # Get second resource
        handle2 = pool.get_resource("tool-b", factory)
        assert handle2 is not None
        assert handle2.tool_name == "tool-b"
        assert len(pool._in_use) == 2
        
        # Try to get third resource (should fail due to max_size)
        handle3 = pool.get_resource("tool-c", factory)
        assert handle3 is None
    
    def test_resource_pool_return_resource(self):
        """Test returning a resource to pool."""
        pool = ResourcePool(ResourceType.MEMORY, max_size=2)
        
        def factory():
            return {"data": "test"}
        
        # Get and return resource
        handle = pool.get_resource("tool-a", factory)
        assert handle is not None
        assert len(pool._in_use) == 1
        assert len(pool._available) == 0
        
        pool.return_resource(handle)
        assert len(pool._in_use) == 0
        assert len(pool._available) == 1
        assert handle.state == ResourceState.IDLE
    
    def test_resource_pool_cleanup_expired(self):
        """Test cleaning up expired resources from pool."""
        pool = ResourcePool(ResourceType.MEMORY, max_size=5)
        
        def factory():
            return {"data": "test"}
        
        # Create resource with short lifetime
        limits = ResourceLimits(max_lifetime_seconds=1)
        handle = pool.get_resource("tool-a", factory)
        handle.limits = limits
        handle.metrics.allocated_at = datetime.now() - timedelta(seconds=2)
        
        # Return to pool
        pool.return_resource(handle)
        assert len(pool._available) == 1
        
        # Cleanup expired
        cleaned = pool.cleanup_expired()
        assert cleaned == 1
        assert len(pool._available) == 0
    
    def test_resource_pool_get_stats(self):
        """Test getting pool statistics."""
        pool = ResourcePool(ResourceType.MEMORY, max_size=5)
        
        def factory():
            return {"data": "test"}
        
        # Get some resources
        handle1 = pool.get_resource("tool-a", factory)
        handle2 = pool.get_resource("tool-b", factory)
        pool.return_resource(handle1)
        
        stats = pool.get_stats()
        assert stats["available"] == 1
        assert stats["in_use"] == 1
        assert stats["total"] == 2
        assert stats["max_size"] == 5


class TestResourceMonitor:
    """Test the ResourceMonitor class."""
    
    def test_resource_monitor_creation(self):
        """Test creating a resource monitor."""
        monitor = ResourceMonitor(check_interval=1.0)
        
        assert monitor.check_interval == 1.0
        assert len(monitor._resources) == 0
        assert not monitor._monitoring
        assert monitor._monitor_task is None
    
    def test_resource_monitor_register_unregister(self):
        """Test registering and unregistering resources."""
        monitor = ResourceMonitor()
        handle = ResourceHandle(
            resource_id="test-resource-1",
            resource_type=ResourceType.MEMORY,
            tool_name="test-tool",
            resource_data={}
        )
        
        # Register resource
        monitor.register_resource(handle)
        assert "test-resource-1" in monitor._resources
        
        # Unregister resource
        monitor.unregister_resource("test-resource-1")
        assert "test-resource-1" not in monitor._resources
    
    @pytest.mark.asyncio
    async def test_resource_monitor_start_stop(self):
        """Test starting and stopping resource monitoring."""
        monitor = ResourceMonitor(check_interval=0.1)
        
        # Start monitoring
        await monitor.start_monitoring()
        assert monitor._monitoring
        assert monitor._monitor_task is not None
        
        # Give it a moment to run
        await asyncio.sleep(0.2)
        
        # Stop monitoring
        await monitor.stop_monitoring()
        assert not monitor._monitoring


class TestResourceManager:
    """Test the ResourceManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create a resource manager for testing."""
        with patch('src.orchestrator.tools.resources.get_enhanced_registry'), \
             patch('src.orchestrator.tools.resources.get_dependency_manager'):
            return ResourceManager()
    
    def test_resource_manager_creation(self, manager):
        """Test creating a resource manager."""
        assert isinstance(manager.monitor, ResourceMonitor)
        assert len(manager._resources) == 0
        assert len(manager._tool_resources) == 0
        assert len(manager._resource_pools) == 0
    
    def test_allocate_resource(self, manager):
        """Test allocating a resource."""
        test_data = {"test": "data"}
        handle = manager.allocate_resource(
            ResourceType.MEMORY,
            "test-tool",
            test_data,
            use_pool=False
        )
        
        assert handle.resource_type == ResourceType.MEMORY
        assert handle.tool_name == "test-tool"
        assert handle.resource_data == test_data
        assert handle.resource_id in manager._resources
        assert "test-tool" in manager._tool_resources
        assert handle.resource_id in manager._tool_resources["test-tool"]
    
    def test_release_resource(self, manager):
        """Test releasing a resource."""
        test_data = {"test": "data"}
        handle = manager.allocate_resource(
            ResourceType.MEMORY,
            "test-tool",
            test_data,
            use_pool=False
        )
        
        resource_id = handle.resource_id
        assert resource_id in manager._resources
        
        success = manager.release_resource(resource_id)
        assert success
        assert resource_id not in manager._resources
    
    def test_release_tool_resources(self, manager):
        """Test releasing all resources for a tool."""
        test_data1 = {"test": "data1"}
        test_data2 = {"test": "data2"}
        
        handle1 = manager.allocate_resource(
            ResourceType.MEMORY,
            "test-tool",
            test_data1,
            use_pool=False
        )
        handle2 = manager.allocate_resource(
            ResourceType.CPU,
            "test-tool",
            test_data2,
            use_pool=False
        )
        
        assert len(manager._tool_resources["test-tool"]) == 2
        
        released = manager.release_tool_resources("test-tool")
        assert released == 2
        assert "test-tool" not in manager._tool_resources
        assert handle1.resource_id not in manager._resources
        assert handle2.resource_id not in manager._resources
    
    def test_create_resource_pool(self, manager):
        """Test creating a resource pool."""
        pool = manager.create_resource_pool(ResourceType.DATABASE_CONNECTION, max_size=10)
        
        assert ResourceType.DATABASE_CONNECTION in manager._resource_pools
        assert manager._resource_pools[ResourceType.DATABASE_CONNECTION] == pool
        assert pool.max_size == 10
    
    def test_allocate_resource_with_pool(self, manager):
        """Test allocating resource using pool."""
        # Create pool first
        manager.create_resource_pool(ResourceType.MEMORY, max_size=5)
        
        test_data = {"test": "data"}
        handle = manager.allocate_resource(
            ResourceType.MEMORY,
            "test-tool",
            test_data,
            use_pool=True
        )
        
        assert handle is not None
        assert handle.resource_type == ResourceType.MEMORY
        assert handle.tool_name == "test-tool"
    
    @pytest.mark.asyncio
    async def test_managed_resource_context(self, manager):
        """Test managed resource context manager."""
        test_data = {"test": "data"}
        
        async with manager.managed_resource(
            ResourceType.MEMORY,
            "test-tool",
            test_data
        ) as handle:
            assert handle.resource_type == ResourceType.MEMORY
            assert handle.tool_name == "test-tool"
            assert handle.resource_data == test_data
            assert handle.resource_id in manager._resources
        
        # Resource should be released after context exit
        assert handle.resource_id not in manager._resources
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_resources(self, manager):
        """Test cleaning up expired resources."""
        # Create resource with short lifetime
        limits = ResourceLimits(max_lifetime_seconds=1)
        handle = manager.allocate_resource(
            ResourceType.MEMORY,
            "test-tool",
            {"test": "data"},
            limits=limits,
            use_pool=False
        )
        
        # Make it expired
        handle.metrics.allocated_at = datetime.now() - timedelta(seconds=2)
        
        # Cleanup
        cleaned = await manager.cleanup_expired_resources()
        assert cleaned == 1
        assert handle.resource_id not in manager._resources
    
    @pytest.mark.asyncio
    async def test_cleanup_all_resources(self, manager):
        """Test cleaning up all resources."""
        # Create multiple resources
        handle1 = manager.allocate_resource(
            ResourceType.MEMORY,
            "tool-1",
            {"test": "data1"},
            use_pool=False
        )
        handle2 = manager.allocate_resource(
            ResourceType.CPU,
            "tool-2",
            {"test": "data2"},
            use_pool=False
        )
        
        assert len(manager._resources) == 2
        
        cleaned = await manager.cleanup_all_resources()
        assert cleaned == 2
        assert len(manager._resources) == 0
    
    def test_get_resource_stats(self, manager):
        """Test getting resource statistics."""
        # Create some resources
        handle1 = manager.allocate_resource(
            ResourceType.MEMORY,
            "tool-1",
            {"test": "data1"},
            use_pool=False
        )
        handle2 = manager.allocate_resource(
            ResourceType.MEMORY,
            "tool-2",
            {"test": "data2"},
            use_pool=False
        )
        handle3 = manager.allocate_resource(
            ResourceType.CPU,
            "tool-1",
            {"test": "data3"},
            use_pool=False
        )
        
        stats = manager.get_resource_stats()
        
        assert stats["total_resources"] == 3
        assert stats["resources_by_type"]["memory"] == 2
        assert stats["resources_by_type"]["cpu"] == 1
        assert stats["resources_by_tool"]["tool-1"] == 2
        assert stats["resources_by_tool"]["tool-2"] == 1
        assert stats["resources_by_state"]["allocated"] == 3
    
    def test_get_tool_resource_usage(self, manager):
        """Test getting resource usage for a specific tool."""
        # Create resources for tool
        handle1 = manager.allocate_resource(
            ResourceType.MEMORY,
            "test-tool",
            {"test": "data1"},
            use_pool=False
        )
        handle2 = manager.allocate_resource(
            ResourceType.CPU,
            "test-tool",
            {"test": "data2"},
            use_pool=False
        )
        
        # Access resources to update metrics
        handle1.access()
        handle2.access()
        
        usage = manager.get_tool_resource_usage("test-tool")
        
        assert usage["tool_name"] == "test-tool"
        assert usage["resource_count"] == 2
        assert len(usage["resources"]) == 2
        assert usage["total_access_count"] == 2


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_allocate_resource_convenience(self):
        """Test the convenience function for allocating resources."""
        with patch('src.orchestrator.tools.resources.resource_manager') as mock_manager:
            mock_handle = Mock()
            mock_manager.allocate_resource.return_value = mock_handle
            
            handle = allocate_resource(
                ResourceType.MEMORY,
                "test-tool",
                {"test": "data"}
            )
            
            assert handle == mock_handle
            mock_manager.allocate_resource.assert_called_once()
    
    def test_release_resource_convenience(self):
        """Test the convenience function for releasing resources."""
        with patch('src.orchestrator.tools.resources.resource_manager') as mock_manager:
            mock_manager.release_resource.return_value = True
            
            result = release_resource("test-resource-id")
            
            assert result is True
            mock_manager.release_resource.assert_called_once_with("test-resource-id")
    
    def test_release_tool_resources_convenience(self):
        """Test the convenience function for releasing tool resources."""
        with patch('src.orchestrator.tools.resources.resource_manager') as mock_manager:
            mock_manager.release_tool_resources.return_value = 3
            
            count = release_tool_resources("test-tool")
            
            assert count == 3
            mock_manager.release_tool_resources.assert_called_once_with("test-tool")
    
    @pytest.mark.asyncio
    async def test_managed_resource_convenience(self):
        """Test the convenience function for managed resources."""
        mock_handle = Mock()
        
        with patch('src.orchestrator.tools.resources.resource_manager') as mock_manager:
            # Set up the async context manager
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_handle
            mock_manager.managed_resource.return_value = mock_context
            
            async with managed_resource(
                ResourceType.MEMORY,
                "test-tool",
                {"test": "data"}
            ) as handle:
                assert handle == mock_handle


class TestIntegration:
    """Integration tests for resource management."""
    
    @pytest.mark.asyncio
    async def test_resource_lifecycle(self):
        """Test complete resource lifecycle."""
        with patch('src.orchestrator.tools.resources.get_enhanced_registry'), \
             patch('src.orchestrator.tools.resources.get_dependency_manager'):
            
            manager = ResourceManager()
            
            # Start manager
            await manager.start()
            
            # Create resource pool
            pool = manager.create_resource_pool(ResourceType.DATABASE_CONNECTION, max_size=2)
            
            # Allocate resources
            handle1 = manager.allocate_resource(
                ResourceType.DATABASE_CONNECTION,
                "tool-a",
                Mock(close=Mock()),
                use_pool=True
            )
            
            # Use managed resource context
            async with manager.managed_resource(
                ResourceType.TEMPORARY_FILE,
                "tool-b",
                "/tmp/test.txt"
            ) as handle2:
                assert handle2.resource_type == ResourceType.TEMPORARY_FILE
            
            # Check statistics
            stats = manager.get_resource_stats()
            assert stats["total_resources"] >= 1
            
            # Cleanup and stop
            await manager.cleanup_all_resources()
            await manager.stop()
    
    def test_concurrent_resource_access(self):
        """Test concurrent access to resource pools."""
        # Test thread safety of resource pools and manager
        pass
    
    @pytest.mark.asyncio
    async def test_resource_monitoring_integration(self):
        """Test integration between resource manager and monitor."""
        # Test that resources are properly monitored
        pass


class TestResourceCleanup:
    """Test resource cleanup functionality."""
    
    def test_temporary_file_cleanup(self):
        """Test cleanup of temporary file resources."""
        with patch('src.orchestrator.tools.resources.get_enhanced_registry'), \
             patch('src.orchestrator.tools.resources.get_dependency_manager'), \
             patch('os.path.exists', return_value=True), \
             patch('os.remove') as mock_remove:
            
            manager = ResourceManager()
            
            # Create temporary file resource
            handle = manager.allocate_resource(
                ResourceType.TEMPORARY_FILE,
                "test-tool",
                "/tmp/test.txt",
                use_pool=False
            )
            
            # Release resource (should trigger cleanup)
            manager.release_resource(handle.resource_id)
            
            # Should have attempted to remove the file
            mock_remove.assert_called_once_with("/tmp/test.txt")
    
    def test_database_connection_cleanup(self):
        """Test cleanup of database connection resources."""
        with patch('src.orchestrator.tools.resources.get_enhanced_registry'), \
             patch('src.orchestrator.tools.resources.get_dependency_manager'):
            
            manager = ResourceManager()
            
            # Create mock database connection
            mock_conn = Mock()
            mock_conn.close = Mock()
            
            handle = manager.allocate_resource(
                ResourceType.DATABASE_CONNECTION,
                "test-tool",
                mock_conn,
                use_pool=False
            )
            
            # Release resource (should trigger cleanup)
            manager.release_resource(handle.resource_id)
            
            # Should have called close on the connection
            mock_conn.close.assert_called_once()
    
    def test_process_cleanup(self):
        """Test cleanup of process resources."""
        with patch('src.orchestrator.tools.resources.get_enhanced_registry'), \
             patch('src.orchestrator.tools.resources.get_dependency_manager'):
            
            manager = ResourceManager()
            
            # Create mock process
            mock_process = Mock()
            mock_process.terminate = Mock()
            mock_process.wait = Mock()
            
            handle = manager.allocate_resource(
                ResourceType.PROCESS,
                "test-tool",
                mock_process,
                use_pool=False
            )
            
            # Release resource (should trigger cleanup)
            manager.release_resource(handle.resource_id)
            
            # Should have terminated the process
            mock_process.terminate.assert_called_once()
            mock_process.wait.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])