"""Tests for resource allocation functionality."""

import time

import pytest

from src.orchestrator.core.resource_allocator import (
    FairShareStrategy,
    PriorityBasedStrategy,
    ResourceAllocator,
    ResourcePool,
    ResourceQuota,
    ResourceRequest,
    ResourceType,
    ResourceUsage)


class TestResourceType:
    """Test cases for ResourceType enum."""

    def test_resource_type_values(self):
        """Test all resource type values."""
        assert ResourceType.CPU.value == "cpu"
        assert ResourceType.MEMORY.value == "memory"
        assert ResourceType.GPU.value == "gpu"
        assert ResourceType.NETWORK.value == "network"
        assert ResourceType.STORAGE.value == "storage"
        assert ResourceType.API_QUOTA.value == "api_quota"
        assert ResourceType.CUSTOM.value == "custom"


class TestResourceQuota:
    """Test cases for ResourceQuota class."""

    def test_resource_quota_creation_basic(self):
        """Test basic resource quota creation."""
        quota = ResourceQuota(resource_type=ResourceType.CPU, limit=100.0, unit="cores")

        assert quota.resource_type == ResourceType.CPU
        assert quota.limit == 100.0
        assert quota.unit == "cores"
        assert quota.renewable is True
        assert quota.renewal_period == 3600.0
        assert quota.burst_limit == 120.0  # 20% burst capacity

    def test_resource_quota_creation_custom(self):
        """Test resource quota creation with custom values."""
        quota = ResourceQuota(
            resource_type=ResourceType.MEMORY,
            limit=8.0,
            unit="GB",
            renewable=False,
            renewal_period=7200.0,
            burst_limit=10.0)

        assert quota.resource_type == ResourceType.MEMORY
        assert quota.limit == 8.0
        assert quota.unit == "GB"
        assert quota.renewable is False
        assert quota.renewal_period == 7200.0
        assert quota.burst_limit == 10.0

    def test_resource_quota_burst_limit_calculation(self):
        """Test automatic burst limit calculation."""
        quota = ResourceQuota(
            resource_type=ResourceType.API_QUOTA, limit=1000.0, unit="requests/hour"
        )

        assert quota.burst_limit == 1200.0  # 20% more than limit


class TestResourceUsage:
    """Test cases for ResourceUsage class."""

    def test_resource_usage_creation_defaults(self):
        """Test resource usage creation with defaults."""
        usage = ResourceUsage(resource_type=ResourceType.CPU)

        assert usage.resource_type == ResourceType.CPU
        assert usage.used == 0.0
        assert usage.reserved == 0.0
        assert usage.timestamp > 0
        assert usage.task_id is None

    def test_resource_usage_creation_custom(self):
        """Test resource usage creation with custom values."""
        custom_time = time.time() - 100
        usage = ResourceUsage(
            resource_type=ResourceType.MEMORY,
            used=4.0,
            reserved=2.0,
            timestamp=custom_time,
            task_id="task_123")

        assert usage.resource_type == ResourceType.MEMORY
        assert usage.used == 4.0
        assert usage.reserved == 2.0
        assert usage.timestamp == custom_time
        assert usage.task_id == "task_123"

    def test_total_used_property(self):
        """Test total_used property calculation."""
        usage = ResourceUsage(resource_type=ResourceType.CPU, used=3.0, reserved=1.5)

        assert usage.total_used == 4.5

    def test_update_usage(self):
        """Test updating usage values."""
        usage = ResourceUsage(resource_type=ResourceType.GPU)
        initial_time = usage.timestamp

        # Wait a bit to ensure timestamp changes
        time.sleep(0.01)

        usage.update_usage(2.5, "new_task")

        assert usage.used == 2.5
        assert usage.task_id == "new_task"
        assert usage.timestamp > initial_time

    def test_update_usage_without_task_id(self):
        """Test updating usage without task ID."""
        usage = ResourceUsage(resource_type=ResourceType.STORAGE, task_id="old_task")

        usage.update_usage(5.0)

        assert usage.used == 5.0
        assert usage.task_id == "old_task"  # Should remain unchanged


class TestResourceRequest:
    """Test cases for ResourceRequest class."""

    def test_resource_request_creation_basic(self):
        """Test basic resource request creation."""
        resources = {ResourceType.CPU: 2.0, ResourceType.MEMORY: 4.0}
        request = ResourceRequest(task_id="task_456", resources=resources)

        assert request.task_id == "task_456"
        assert request.resources == resources
        assert request.priority == 0
        assert request.timeout == 300.0
        # min_resources should be 50% of requested
        assert request.min_resources[ResourceType.CPU] == 1.0
        assert request.min_resources[ResourceType.MEMORY] == 2.0

    def test_resource_request_creation_custom(self):
        """Test resource request creation with custom values."""
        resources = {ResourceType.GPU: 1.0, ResourceType.API_QUOTA: 100.0}
        min_resources = {ResourceType.GPU: 0.5, ResourceType.API_QUOTA: 50.0}

        request = ResourceRequest(
            task_id="priority_task",
            resources=resources,
            priority=10,
            min_resources=min_resources)

        assert request.task_id == "priority_task"
        assert request.resources == resources
        assert request.priority == 10
        assert request.timeout == 600.0
        assert request.min_resources == min_resources

    def test_resource_request_min_resources_auto_calculation(self):
        """Test automatic calculation of minimum resources."""
        resources = {
            ResourceType.CPU: 8.0,
            ResourceType.MEMORY: 16.0,
            ResourceType.GPU: 2.0,
        }

        request = ResourceRequest(task_id="auto_min", resources=resources)

        assert request.min_resources[ResourceType.CPU] == 4.0
        assert request.min_resources[ResourceType.MEMORY] == 8.0
        assert request.min_resources[ResourceType.GPU] == 1.0


class TestFairShareStrategy:
    """Test cases for FairShareStrategy class."""

    @pytest.mark.asyncio
    async def test_fair_share_allocation_sufficient_resources(self):
        """Test fair share allocation with sufficient resources."""
        strategy = FairShareStrategy()

        request = ResourceRequest(
            task_id="test_task",
            resources={ResourceType.CPU: 4.0, ResourceType.MEMORY: 8.0})

        available = {ResourceType.CPU: 10.0, ResourceType.MEMORY: 16.0}
        current_usage = {
            ResourceType.CPU: ResourceUsage(ResourceType.CPU, used=2.0),
            ResourceType.MEMORY: ResourceUsage(ResourceType.MEMORY, used=4.0),
        }

        allocation = await strategy.allocate(request, available, current_usage)

        assert allocation[ResourceType.CPU] == 4.0
        assert allocation[ResourceType.MEMORY] == 8.0

    @pytest.mark.asyncio
    async def test_fair_share_allocation_insufficient_resources(self):
        """Test fair share allocation with insufficient resources."""
        strategy = FairShareStrategy()

        request = ResourceRequest(
            task_id="test_task",
            resources={ResourceType.CPU: 8.0, ResourceType.MEMORY: 16.0})

        available = {ResourceType.CPU: 4.0, ResourceType.MEMORY: 8.0}
        current_usage = {}

        allocation = await strategy.allocate(request, available, current_usage)

        # Should allocate what's available (proportionally)
        assert allocation[ResourceType.CPU] <= 4.0
        assert allocation[ResourceType.MEMORY] <= 8.0

    @pytest.mark.asyncio
    async def test_fair_share_allocation_partial_availability(self):
        """Test fair share allocation with partial resource availability."""
        strategy = FairShareStrategy()

        request = ResourceRequest(
            task_id="test_task",
            resources={ResourceType.CPU: 6.0, ResourceType.MEMORY: 12.0})

        available = {ResourceType.CPU: 8.0, ResourceType.MEMORY: 6.0}
        current_usage = {}

        allocation = await strategy.allocate(request, available, current_usage)

        # Should allocate proportionally to available resources
        assert allocation[ResourceType.CPU] <= 6.0
        assert allocation[ResourceType.MEMORY] <= 6.0


class TestPriorityBasedStrategy:
    """Test cases for PriorityBasedStrategy class."""

    @pytest.mark.asyncio
    async def test_priority_based_allocation_high_priority(self):
        """Test priority-based allocation for high priority task."""
        strategy = PriorityBasedStrategy()

        request = ResourceRequest(
            task_id="high_priority_task",
            resources={ResourceType.CPU: 4.0, ResourceType.MEMORY: 8.0},
            priority=10)

        available = {ResourceType.CPU: 10.0, ResourceType.MEMORY: 16.0}
        current_usage = {}

        allocation = await strategy.allocate(request, available, current_usage)

        # High priority should get full or enhanced allocation
        assert allocation[ResourceType.CPU] >= 4.0
        assert allocation[ResourceType.MEMORY] >= 8.0

    @pytest.mark.asyncio
    async def test_priority_based_allocation_low_priority(self):
        """Test priority-based allocation for low priority task."""
        strategy = PriorityBasedStrategy()

        request = ResourceRequest(
            task_id="low_priority_task",
            resources={ResourceType.CPU: 4.0, ResourceType.MEMORY: 8.0},
            priority=1)

        available = {ResourceType.CPU: 3.0, ResourceType.MEMORY: 6.0}
        current_usage = {}

        allocation = await strategy.allocate(request, available, current_usage)

        # Should get reduced allocation based on priority
        assert allocation[ResourceType.CPU] <= 3.0
        assert allocation[ResourceType.MEMORY] <= 6.0


class TestResourcePool:
    """Test cases for ResourcePool class."""

    def test_resource_pool_creation(self):
        """Test resource pool creation."""
        quota = ResourceQuota(ResourceType.CPU, 16.0, "cores")
        pool = ResourcePool(ResourceType.CPU, quota)

        assert pool.resource_type == ResourceType.CPU
        assert pool.quota == quota
        assert pool.allocations == {}
        assert pool.reservations == {}
        assert len(pool.usage_history) == 0

    def test_resource_pool_get_available(self):
        """Test getting available resources."""
        quota = ResourceQuota(ResourceType.CPU, 8.0, "cores")
        pool = ResourcePool(ResourceType.CPU, quota)

        # Allocate some resources
        pool.allocations["task1"] = 2.0
        pool.reservations["task2"] = 1.0

        available = pool.get_available()

        assert available == 5.0  # 8 - 2 - 1

    def test_resource_pool_allocate_method(self):
        """Test allocating resources using allocate method."""
        quota = ResourceQuota(ResourceType.CPU, 8.0, "cores")
        pool = ResourcePool(ResourceType.CPU, quota)

        # Should be able to allocate 3 cores
        success = pool.allocate("task1", 3.0)
        assert success is True
        assert pool.allocations["task1"] == 3.0

        # Should not be able to allocate 6 more cores (total would be 9)
        success = pool.allocate("task2", 6.0)
        assert success is False
        assert "task2" not in pool.allocations

    def test_resource_pool_reserve_method(self):
        """Test reserving resources using reserve method."""
        quota = ResourceQuota(ResourceType.MEMORY, 16.0, "GB")
        pool = ResourcePool(ResourceType.MEMORY, quota)

        # Should be able to reserve 8 GB
        success = pool.reserve("task1", 8.0)
        assert success is True
        assert pool.reservations["task1"] == 8.0

        # Should not be able to reserve 10 more GB (total would be 18)
        success = pool.reserve("task2", 10.0)
        assert success is False
        assert "task2" not in pool.reservations

    def test_resource_pool_release_method(self):
        """Test releasing resources back to pool."""
        quota = ResourceQuota(ResourceType.CPU, 8.0, "cores")
        pool = ResourcePool(ResourceType.CPU, quota)

        # First allocate and reserve resources
        pool.allocate("task1", 3.0)
        pool.reserve("task2", 2.0)

        # Release allocations for task1
        pool.release("task1")

        assert "task1" not in pool.allocations
        assert "task1" not in pool.reservations
        assert pool.reservations["task2"] == 2.0  # Should remain

    def test_resource_pool_get_usage(self):
        """Test getting usage statistics."""
        quota = ResourceQuota(ResourceType.GPU, 4.0, "cards")
        pool = ResourcePool(ResourceType.GPU, quota)

        # Allocate and reserve some resources
        pool.allocate("task1", 1.5)
        pool.reserve("task2", 1.0)

        usage = pool.get_usage()

        assert usage.resource_type == ResourceType.GPU
        assert usage.used == 1.5
        assert usage.reserved == 1.0
        assert usage.total_used == 2.5


class TestResourceAllocator:
    """Test cases for ResourceAllocator class."""

    def test_resource_allocator_creation(self):
        """Test resource allocator creation."""
        allocator = ResourceAllocator()

        assert allocator.pools == {}
        assert len(allocator.pending_requests) == 0  # It's a deque
        assert len(allocator.allocation_history) == 0  # It's a deque
        assert isinstance(allocator.strategy, FairShareStrategy)

    def test_resource_allocator_creation_with_strategy(self):
        """Test resource allocator creation with custom strategy."""
        strategy = PriorityBasedStrategy()
        allocator = ResourceAllocator(strategy=strategy)

        assert allocator.strategy is strategy

    def test_add_resource_pool(self):
        """Test adding resource pool."""
        allocator = ResourceAllocator()

        quota = ResourceQuota(ResourceType.CPU, 8.0, "cores")
        allocator.add_resource_pool(ResourceType.CPU, quota)

        assert ResourceType.CPU in allocator.pools
        assert allocator.pools[ResourceType.CPU].quota == quota

    def test_remove_resource_pool(self):
        """Test removing resource pool."""
        allocator = ResourceAllocator()

        quota = ResourceQuota(ResourceType.CPU, 8.0, "cores")
        allocator.add_resource_pool(ResourceType.CPU, quota)
        allocator.remove_resource_pool(ResourceType.CPU)

        assert ResourceType.CPU not in allocator.pools

    def test_get_task_allocation(self):
        """Test getting task allocation."""
        allocator = ResourceAllocator()

        # Initially no allocation
        allocation = allocator.get_task_allocation("non_existent_task")
        assert allocation == {}

    def test_get_resource_usage(self):
        """Test getting resource usage for specific type."""
        allocator = ResourceAllocator()

        quota = ResourceQuota(ResourceType.CPU, 8.0, "cores")
        allocator.add_resource_pool(ResourceType.CPU, quota)

        # Get usage for CPU
        usage = allocator.get_resource_usage(ResourceType.CPU)
        assert usage is not None
        assert usage.resource_type == ResourceType.CPU

        # Get usage for non-existent resource
        usage = allocator.get_resource_usage(ResourceType.GPU)
        assert usage is None

    @pytest.mark.asyncio
    async def test_request_resources_success(self):
        """Test successful resource request."""
        allocator = ResourceAllocator()

        quota = ResourceQuota(ResourceType.CPU, 8.0, "cores")
        allocator.add_resource_pool(ResourceType.CPU, quota)

        request = ResourceRequest(
            task_id="test_task", resources={ResourceType.CPU: 4.0}
        )

        success = await allocator.request_resources(request)

        assert success is True

        # Check that task allocation is tracked
        allocation = allocator.get_task_allocation("test_task")
        assert ResourceType.CPU in allocation

    @pytest.mark.asyncio
    async def test_request_resources_insufficient(self):
        """Test resource request with insufficient resources."""
        allocator = ResourceAllocator()

        quota = ResourceQuota(ResourceType.CPU, 2.0, "cores")
        allocator.add_resource_pool(ResourceType.CPU, quota)

        request = ResourceRequest(task_id="big_task", resources={ResourceType.CPU: 8.0})

        success = await allocator.request_resources(request)

        # Should fail since we don't have enough resources
        assert success is False

    @pytest.mark.asyncio
    async def test_release_resources(self):
        """Test releasing allocated resources."""
        allocator = ResourceAllocator()

        quota = ResourceQuota(ResourceType.CPU, 8.0, "cores")
        allocator.add_resource_pool(ResourceType.CPU, quota)

        # First allocate resources
        request = ResourceRequest(
            task_id="release_test", resources={ResourceType.CPU: 3.0}
        )
        await allocator.request_resources(request)

        # Check that allocation is tracked
        allocation = allocator.get_task_allocation("release_test")
        assert ResourceType.CPU in allocation

        # Then release them
        await allocator.release_resources("release_test")

        # Check that allocation is cleared
        allocation = allocator.get_task_allocation("release_test")
        assert allocation == {}

    def test_get_overall_statistics(self):
        """Test getting overall statistics."""
        allocator = ResourceAllocator()

        quota = ResourceQuota(ResourceType.CPU, 10.0, "cores")
        allocator.add_resource_pool(ResourceType.CPU, quota)

        stats = allocator.get_overall_statistics()

        assert "total_pools" in stats
        assert "active_tasks" in stats
        assert "overall_utilization" in stats
        assert stats["total_pools"] == 1
        assert stats["active_tasks"] == 0

    def test_set_allocation_strategy(self):
        """Test changing allocation strategy."""
        allocator = ResourceAllocator()

        # Start with default FairShareStrategy
        assert isinstance(allocator.strategy, FairShareStrategy)

        # Change to PriorityBasedStrategy
        new_strategy = PriorityBasedStrategy()
        allocator.set_allocation_strategy(new_strategy)

        assert allocator.strategy is new_strategy
        assert isinstance(allocator.strategy, PriorityBasedStrategy)

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test cleanup functionality."""
        allocator = ResourceAllocator()

        quota = ResourceQuota(ResourceType.CPU, 8.0, "cores")
        allocator.add_resource_pool(ResourceType.CPU, quota)

        # Add a pending request
        request = ResourceRequest(
            task_id="cleanup_test", resources={ResourceType.CPU: 4.0}
        )
        allocator.pending_requests.append(request)

        # Perform cleanup
        await allocator.cleanup()

        assert len(allocator.pending_requests) == 0
