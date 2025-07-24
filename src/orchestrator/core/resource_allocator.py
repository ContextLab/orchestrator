"""Resource allocation and management for the Orchestrator framework."""

import asyncio
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from .task import Task


class ResourceType(Enum):
    """Types of resources that can be allocated."""

    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    NETWORK = "network"
    STORAGE = "storage"
    API_QUOTA = "api_quota"
    CUSTOM = "custom"


@dataclass
class ResourceQuota:
    """Resource quota specification."""

    resource_type: ResourceType
    limit: float
    unit: str
    renewable: bool = True
    renewal_period: float = 3600.0  # 1 hour in seconds
    burst_limit: Optional[float] = None

    def __post_init__(self):
        if self.burst_limit is None:
            self.burst_limit = self.limit * 1.2  # 20% burst capacity


@dataclass
class ResourceUsage:
    """Resource usage tracking."""

    resource_type: ResourceType
    used: float = 0.0
    reserved: float = 0.0
    timestamp: float = field(default_factory=time.time)
    task_id: Optional[str] = None

    @property
    def total_used(self) -> float:
        """Total used including reserved."""
        return self.used + self.reserved

    def update_usage(self, amount: float, task_id: str = None):
        """Update usage amount."""
        self.used = amount
        self.timestamp = time.time()
        if task_id:
            self.task_id = task_id


@dataclass
class ResourceRequest:
    """Resource allocation request."""

    task_id: str
    resources: Dict[ResourceType, float]
    priority: int = 0
    timeout: float = 300.0  # 5 minutes
    min_resources: Optional[Dict[ResourceType, float]] = None

    def __post_init__(self):
        if self.min_resources is None:
            # Minimum is 50% of requested by default
            self.min_resources = {
                res_type: amount * 0.5 for res_type, amount in self.resources.items()
            }


class ResourceAllocationStrategy(ABC):
    """Abstract base class for resource allocation strategies."""

    @abstractmethod
    async def allocate(
        self,
        request: ResourceRequest,
        available: Dict[ResourceType, float],
        current_usage: Dict[ResourceType, ResourceUsage],
    ) -> Dict[ResourceType, float]:
        """Allocate resources based on strategy."""
        pass


class FairShareStrategy(ResourceAllocationStrategy):
    """Fair share resource allocation strategy."""

    async def allocate(
        self,
        request: ResourceRequest,
        available: Dict[ResourceType, float],
        current_usage: Dict[ResourceType, ResourceUsage],
    ) -> Dict[ResourceType, float]:
        """Allocate resources fairly across tasks."""
        allocation = {}

        for resource_type, requested in request.resources.items():
            available_amount = available.get(resource_type, 0.0)

            if available_amount >= requested:
                # Full allocation possible
                allocation[resource_type] = requested
            elif available_amount >= request.min_resources.get(resource_type, 0.0):
                # Partial allocation
                allocation[resource_type] = available_amount
            else:
                # Cannot meet minimum requirements
                allocation[resource_type] = 0.0

        return allocation


class PriorityBasedStrategy(ResourceAllocationStrategy):
    """Priority-based resource allocation strategy."""

    async def allocate(
        self,
        request: ResourceRequest,
        available: Dict[ResourceType, float],
        current_usage: Dict[ResourceType, ResourceUsage],
    ) -> Dict[ResourceType, float]:
        """Allocate resources based on priority."""
        allocation = {}

        # High priority tasks get preferential treatment
        priority_multiplier = 1.0 + (request.priority / 10.0)

        for resource_type, requested in request.resources.items():
            available_amount = available.get(resource_type, 0.0)

            # Adjust allocation based on priority
            adjusted_request = requested * priority_multiplier

            if available_amount >= adjusted_request:
                allocation[resource_type] = min(
                    adjusted_request, requested * 1.2
                )  # Max 20% bonus
            elif available_amount >= request.min_resources.get(resource_type, 0.0):
                allocation[resource_type] = available_amount
            else:
                allocation[resource_type] = 0.0

        return allocation


class ResourcePool:
    """Manages a pool of resources of a specific type."""

    def __init__(self, resource_type: ResourceType, quota: ResourceQuota):
        self.resource_type = resource_type
        self.quota = quota
        self.allocations: Dict[str, float] = {}  # task_id -> allocated amount
        self.reservations: Dict[str, float] = {}  # task_id -> reserved amount
        self.usage_history = deque(maxlen=1000)
        self.last_renewal = time.time()
        self._lock = threading.RLock()

    def get_available(self) -> float:
        """Get available resource amount."""
        with self._lock:
            self._check_renewal()

            total_allocated = sum(self.allocations.values())
            total_reserved = sum(self.reservations.values())

            return max(0.0, self.quota.limit - total_allocated - total_reserved)

    def get_usage(self) -> ResourceUsage:
        """Get current usage information."""
        with self._lock:
            total_used = sum(self.allocations.values())
            total_reserved = sum(self.reservations.values())

            return ResourceUsage(
                resource_type=self.resource_type,
                used=total_used,
                reserved=total_reserved,
                timestamp=time.time(),
            )

    def allocate(self, task_id: str, amount: float) -> bool:
        """Allocate resources to a task."""
        with self._lock:
            if self.get_available() >= amount:
                self.allocations[task_id] = amount
                self.usage_history.append(
                    {
                        "action": "allocate",
                        "task_id": task_id,
                        "amount": amount,
                        "timestamp": time.time(),
                    }
                )
                return True
            return False

    def reserve(self, task_id: str, amount: float) -> bool:
        """Reserve resources for a task."""
        with self._lock:
            if self.get_available() >= amount:
                self.reservations[task_id] = amount
                self.usage_history.append(
                    {
                        "action": "reserve",
                        "task_id": task_id,
                        "amount": amount,
                        "timestamp": time.time(),
                    }
                )
                return True
            return False

    def release(self, task_id: str) -> float:
        """Release all resources for a task."""
        with self._lock:
            released = 0.0

            if task_id in self.allocations:
                released += self.allocations.pop(task_id)

            if task_id in self.reservations:
                released += self.reservations.pop(task_id)

            if released > 0:
                self.usage_history.append(
                    {
                        "action": "release",
                        "task_id": task_id,
                        "amount": released,
                        "timestamp": time.time(),
                    }
                )

            return released

    def _check_renewal(self):
        """Check if resources should be renewed."""
        if not self.quota.renewable:
            return

        current_time = time.time()
        if current_time - self.last_renewal >= self.quota.renewal_period:
            # Reset allocations for renewable resources
            self.allocations.clear()
            self.reservations.clear()
            self.last_renewal = current_time

            self.usage_history.append({"action": "renewal", "timestamp": current_time})

    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            usage = self.get_usage()

            return {
                "resource_type": self.resource_type.value,
                "quota_limit": self.quota.limit,
                "quota_unit": self.quota.unit,
                "available": self.get_available(),
                "allocated": usage.used,
                "reserved": usage.reserved,
                "utilization": (
                    (usage.used + usage.reserved) / self.quota.limit
                    if self.quota.limit > 0
                    else 0
                ),
                "active_allocations": len(self.allocations),
                "active_reservations": len(self.reservations),
                "renewable": self.quota.renewable,
                "last_renewal": self.last_renewal,
            }


class ResourceAllocator:
    """Central resource allocator for the Orchestrator framework."""

    def __init__(self, strategy: ResourceAllocationStrategy = None):
        self.strategy = strategy or FairShareStrategy()
        self.pools: Dict[ResourceType, ResourcePool] = {}
        self.pending_requests: deque = deque()
        self.allocation_history: deque = deque(maxlen=1000)
        self.task_allocations: Dict[str, Dict[ResourceType, float]] = defaultdict(dict)
        self._lock = asyncio.Lock()

    def add_resource_pool(self, resource_type: ResourceType, quota: ResourceQuota):
        """Add a resource pool."""
        self.pools[resource_type] = ResourcePool(resource_type, quota)

    def remove_resource_pool(self, resource_type: ResourceType):
        """Remove a resource pool."""
        if resource_type in self.pools:
            # Release all allocations
            pool = self.pools[resource_type]
            for task_id in list(pool.allocations.keys()) + list(
                pool.reservations.keys()
            ):
                pool.release(task_id)

            del self.pools[resource_type]

    async def request_resources(self, request: ResourceRequest) -> bool:
        """Request resource allocation for a task."""
        async with self._lock:
            # Get current availability
            available = {}
            current_usage = {}

            for resource_type in request.resources:
                if resource_type in self.pools:
                    pool = self.pools[resource_type]
                    available[resource_type] = pool.get_available()
                    current_usage[resource_type] = pool.get_usage()
                else:
                    # Resource type not managed
                    available[resource_type] = float("inf")
                    current_usage[resource_type] = ResourceUsage(resource_type)

            # Use strategy to determine allocation
            allocation = await self.strategy.allocate(request, available, current_usage)

            # Check if allocation meets minimum requirements
            allocation_viable = all(
                allocation.get(res_type, 0.0)
                >= request.min_resources.get(res_type, 0.0)
                for res_type in request.resources
            )

            if allocation_viable:
                # Perform allocation
                success = True
                allocated = {}

                for resource_type, amount in allocation.items():
                    if amount > 0 and resource_type in self.pools:
                        pool = self.pools[resource_type]
                        if not pool.allocate(request.task_id, amount):
                            success = False
                            break
                        allocated[resource_type] = amount

                if success:
                    # Record successful allocation
                    self.task_allocations[request.task_id] = allocated
                    self.allocation_history.append(
                        {
                            "action": "allocate",
                            "task_id": request.task_id,
                            "request": request.resources,
                            "allocated": allocated,
                            "timestamp": time.time(),
                            "success": True,
                        }
                    )
                    return True
                else:
                    # Rollback partial allocations
                    for resource_type, amount in allocated.items():
                        if resource_type in self.pools:
                            self.pools[resource_type].release(request.task_id)

            # Add to pending queue if allocation failed
            self.pending_requests.append(request)
            self.allocation_history.append(
                {
                    "action": "queue",
                    "task_id": request.task_id,
                    "request": request.resources,
                    "timestamp": time.time(),
                    "success": False,
                }
            )

            return False

    async def release_resources(self, task_id: str):
        """Release all resources for a task."""
        async with self._lock:
            released = {}

            for resource_type, pool in self.pools.items():
                amount = pool.release(task_id)
                if amount > 0:
                    released[resource_type] = amount

            if task_id in self.task_allocations:
                del self.task_allocations[task_id]

            if released:
                self.allocation_history.append(
                    {
                        "action": "release",
                        "task_id": task_id,
                        "released": released,
                        "timestamp": time.time(),
                    }
                )

                # Process pending requests
                await self._process_pending_requests()

    async def _process_pending_requests(self):
        """Process pending resource requests."""
        processed = []

        while self.pending_requests:
            request = self.pending_requests.popleft()

            # Check if request has timed out
            if time.time() - request.timeout > 300:  # Default timeout
                self.allocation_history.append(
                    {
                        "action": "timeout",
                        "task_id": request.task_id,
                        "timestamp": time.time(),
                    }
                )
                continue

            # Try to allocate again
            success = await self.request_resources(request)
            if not success:
                # Put back in queue if still can't allocate
                processed.append(request)

        # Put back unprocessed requests
        self.pending_requests.extend(processed)

    def get_task_allocation(self, task_id: str) -> Dict[ResourceType, float]:
        """Get current resource allocation for a task."""
        return self.task_allocations.get(task_id, {})

    def get_resource_usage(
        self, resource_type: ResourceType
    ) -> Optional[ResourceUsage]:
        """Get usage for a specific resource type."""
        if resource_type in self.pools:
            return self.pools[resource_type].get_usage()
        return None

    def get_overall_statistics(self) -> Dict[str, Any]:
        """Get overall resource allocation statistics."""
        pool_stats = {}
        total_pools = len(self.pools)
        total_tasks = len(self.task_allocations)
        pending_requests = len(self.pending_requests)

        for resource_type, pool in self.pools.items():
            pool_stats[resource_type.value] = pool.get_statistics()

        # Calculate overall utilization
        total_utilization = 0.0
        if total_pools > 0:
            total_utilization = (
                sum(stats["utilization"] for stats in pool_stats.values()) / total_pools
            )

        return {
            "total_pools": total_pools,
            "active_tasks": total_tasks,
            "pending_requests": pending_requests,
            "overall_utilization": total_utilization,
            "pool_statistics": pool_stats,
            "allocation_history_size": len(self.allocation_history),
            "strategy": type(self.strategy).__name__,
        }

    async def get_utilization(self) -> Dict[str, Any]:
        """Get current resource utilization metrics."""
        async with self._lock:
            # Calculate CPU usage
            cpu_usage = 0.0
            if ResourceType.CPU in self.pools:
                cpu_pool = self.pools[ResourceType.CPU]
                cpu_stats = cpu_pool.get_statistics()
                cpu_usage = cpu_stats.get("utilization", 0.0)

            # Calculate memory usage
            memory_usage = 0.0
            if ResourceType.MEMORY in self.pools:
                memory_pool = self.pools[ResourceType.MEMORY]
                memory_stats = memory_pool.get_statistics()
                memory_usage = memory_stats.get("utilization", 0.0)

            # Calculate GPU usage if available
            gpu_usage = 0.0
            if ResourceType.GPU in self.pools:
                gpu_pool = self.pools[ResourceType.GPU]
                gpu_stats = gpu_pool.get_statistics()
                gpu_usage = gpu_stats.get("utilization", 0.0)

            # Calculate API quota usage
            api_usage = 0.0
            if ResourceType.API_QUOTA in self.pools:
                api_pool = self.pools[ResourceType.API_QUOTA]
                api_stats = api_pool.get_statistics()
                api_usage = api_stats.get("utilization", 0.0)

            return {
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "gpu_usage": gpu_usage,
                "api_usage": api_usage,
                "active_tasks": len(self.task_allocations),
                "pending_requests": len(self.pending_requests),
                "timestamp": time.time(),
            }

    def set_allocation_strategy(self, strategy: ResourceAllocationStrategy):
        """Change the allocation strategy."""
        self.strategy = strategy

    async def cleanup(self):
        """Clean up all allocations and pending requests."""
        async with self._lock:
            # Release all allocations
            for task_id in list(self.task_allocations.keys()):
                await self.release_resources(task_id)

            # Clear pending requests
            self.pending_requests.clear()

            # Clear history
            self.allocation_history.clear()


# Utility functions for common resource configurations


def create_default_resource_allocator() -> ResourceAllocator:
    """Create a resource allocator with default configurations."""
    allocator = ResourceAllocator(FairShareStrategy())

    # Add common resource pools
    allocator.add_resource_pool(
        ResourceType.CPU,
        ResourceQuota(ResourceType.CPU, limit=8.0, unit="cores", renewable=False),
    )

    allocator.add_resource_pool(
        ResourceType.MEMORY,
        ResourceQuota(ResourceType.MEMORY, limit=16.0, unit="GB", renewable=False),
    )

    allocator.add_resource_pool(
        ResourceType.API_QUOTA,
        ResourceQuota(
            ResourceType.API_QUOTA,
            limit=1000.0,
            unit="requests",
            renewable=True,
            renewal_period=3600.0,
        ),
    )

    return allocator


def create_resource_request_from_task(task: Task, priority: int = 0) -> ResourceRequest:
    """Create a resource request from a task."""
    # Extract resource requirements from task parameters
    resources = {}

    # Default resource requirements based on task type
    task_type = task.parameters.get("type", "default")

    if task_type == "llm_generation":
        resources[ResourceType.CPU] = task.parameters.get("cpu_cores", 1.0)
        resources[ResourceType.MEMORY] = task.parameters.get("memory_gb", 2.0)
        resources[ResourceType.API_QUOTA] = task.parameters.get("api_calls", 1.0)
    elif task_type == "data_processing":
        resources[ResourceType.CPU] = task.parameters.get("cpu_cores", 2.0)
        resources[ResourceType.MEMORY] = task.parameters.get("memory_gb", 4.0)
    elif task_type == "training":
        resources[ResourceType.CPU] = task.parameters.get("cpu_cores", 4.0)
        resources[ResourceType.MEMORY] = task.parameters.get("memory_gb", 8.0)
        if task.parameters.get("use_gpu", False):
            resources[ResourceType.GPU] = task.parameters.get("gpu_count", 1.0)
    else:
        # Default minimal resources
        resources[ResourceType.CPU] = 0.5
        resources[ResourceType.MEMORY] = 1.0

    return ResourceRequest(
        task_id=task.id,
        resources=resources,
        priority=priority,
        timeout=task.parameters.get("timeout", 300.0),
    )
