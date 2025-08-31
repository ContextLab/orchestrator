"""Resource Management System - Issue #312 Stream C

Comprehensive resource management for efficient tool lifecycle management:
- Resource allocation and cleanup tracking
- Memory and process monitoring
- Resource pools and sharing mechanisms
- Integration with dependency resolution and installation systems
- Automatic resource cleanup and garbage collection
"""

import logging
import asyncio
import psutil
import threading
import time
from typing import Dict, List, Set, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from weakref import WeakKeyDictionary
import gc

from .registry import EnhancedToolRegistry, get_enhanced_registry
from .dependencies import DependencyManager, get_dependency_manager

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources that can be managed."""
    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"
    NETWORK = "network"
    FILE_HANDLE = "file_handle"
    DATABASE_CONNECTION = "database_connection"
    EXTERNAL_SERVICE = "external_service"
    TEMPORARY_FILE = "temporary_file"
    PROCESS = "process"
    THREAD = "thread"


class ResourceState(Enum):
    """States of managed resources."""
    ALLOCATED = "allocated"
    ACTIVE = "active"
    IDLE = "idle"
    CLEANUP_PENDING = "cleanup_pending"
    RELEASED = "released"
    ERROR = "error"


@dataclass
class ResourceMetrics:
    """Metrics for resource usage."""
    allocated_at: datetime
    last_accessed: datetime
    access_count: int = 0
    total_usage_time: float = 0.0
    peak_memory: int = 0
    peak_cpu: float = 0.0
    errors: int = 0


@dataclass
class ResourceLimits:
    """Resource usage limits."""
    max_memory_mb: Optional[int] = None
    max_cpu_percent: Optional[float] = None
    max_disk_mb: Optional[int] = None
    max_lifetime_seconds: Optional[int] = None
    max_idle_seconds: Optional[int] = 300  # 5 minutes default
    max_access_count: Optional[int] = None


@dataclass
class ResourceHandle:
    """Handle to a managed resource."""
    resource_id: str
    resource_type: ResourceType
    tool_name: str
    resource_data: Any
    state: ResourceState = ResourceState.ALLOCATED
    metrics: ResourceMetrics = field(default_factory=lambda: ResourceMetrics(
        allocated_at=datetime.now(),
        last_accessed=datetime.now()
    ))
    limits: Optional[ResourceLimits] = None
    cleanup_callbacks: List[Callable] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize resource handle."""
        if self.limits is None:
            self.limits = ResourceLimits()
    
    def access(self) -> None:
        """Mark resource as accessed."""
        self.metrics.last_accessed = datetime.now()
        self.metrics.access_count += 1
        if self.state == ResourceState.IDLE:
            self.state = ResourceState.ACTIVE
    
    def is_expired(self) -> bool:
        """Check if resource has expired based on limits."""
        now = datetime.now()
        
        # Check lifetime limit
        if self.limits.max_lifetime_seconds:
            lifetime = (now - self.metrics.allocated_at).total_seconds()
            if lifetime > self.limits.max_lifetime_seconds:
                return True
        
        # Check idle limit
        if self.limits.max_idle_seconds:
            idle_time = (now - self.metrics.last_accessed).total_seconds()
            if idle_time > self.limits.max_idle_seconds:
                return True
        
        # Check access count limit
        if self.limits.max_access_count:
            if self.metrics.access_count >= self.limits.max_access_count:
                return True
        
        return False
    
    def exceeds_limits(self, current_memory: int = 0, current_cpu: float = 0.0) -> List[str]:
        """Check if resource exceeds its limits."""
        violations = []
        
        if self.limits.max_memory_mb and current_memory > self.limits.max_memory_mb * 1024 * 1024:
            violations.append(f"Memory usage {current_memory / 1024 / 1024:.1f}MB exceeds limit {self.limits.max_memory_mb}MB")
        
        if self.limits.max_cpu_percent and current_cpu > self.limits.max_cpu_percent:
            violations.append(f"CPU usage {current_cpu:.1f}% exceeds limit {self.limits.max_cpu_percent}%")
        
        return violations


class ResourcePool:
    """Pool for sharing and reusing resources."""
    
    def __init__(self, resource_type: ResourceType, max_size: int = 10):
        self.resource_type = resource_type
        self.max_size = max_size
        self._available: List[ResourceHandle] = []
        self._in_use: Set[str] = set()
        self._lock = threading.RLock()
    
    def get_resource(self, tool_name: str, factory_func: Callable) -> Optional[ResourceHandle]:
        """Get a resource from the pool or create a new one."""
        with self._lock:
            # Try to reuse an available resource
            for i, handle in enumerate(self._available):
                if not handle.is_expired():
                    self._available.pop(i)
                    self._in_use.add(handle.resource_id)
                    handle.tool_name = tool_name  # Update tool name
                    handle.access()
                    return handle
            
            # Create new resource if pool not full
            if len(self._in_use) < self.max_size:
                try:
                    resource_data = factory_func()
                    handle = ResourceHandle(
                        resource_id=f"{self.resource_type.value}_{int(time.time() * 1000)}",
                        resource_type=self.resource_type,
                        tool_name=tool_name,
                        resource_data=resource_data
                    )
                    self._in_use.add(handle.resource_id)
                    return handle
                except Exception as e:
                    logger.error(f"Failed to create resource in pool: {e}")
                    return None
            
            return None  # Pool is full
    
    def return_resource(self, handle: ResourceHandle) -> None:
        """Return a resource to the pool."""
        with self._lock:
            if handle.resource_id in self._in_use:
                self._in_use.remove(handle.resource_id)
                
                # Only return to pool if not expired and not in error state
                if not handle.is_expired() and handle.state != ResourceState.ERROR:
                    handle.state = ResourceState.IDLE
                    self._available.append(handle)
                else:
                    # Resource expired or in error state, clean it up
                    self._cleanup_resource(handle)
    
    def _cleanup_resource(self, handle: ResourceHandle) -> None:
        """Clean up a resource."""
        try:
            for callback in handle.cleanup_callbacks:
                callback(handle.resource_data)
        except Exception as e:
            logger.error(f"Error during resource cleanup: {e}")
        
        handle.state = ResourceState.RELEASED
    
    def cleanup_expired(self) -> int:
        """Clean up expired resources from the pool."""
        with self._lock:
            expired = [h for h in self._available if h.is_expired()]
            for handle in expired:
                self._available.remove(handle)
                self._cleanup_resource(handle)
            return len(expired)
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        with self._lock:
            return {
                "available": len(self._available),
                "in_use": len(self._in_use),
                "total": len(self._available) + len(self._in_use),
                "max_size": self.max_size
            }


class ResourceMonitor:
    """Monitors resource usage and enforces limits."""
    
    def __init__(self, check_interval: float = 5.0):
        self.check_interval = check_interval
        self._resources: Dict[str, ResourceHandle] = {}
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._lock = threading.RLock()
    
    def register_resource(self, handle: ResourceHandle) -> None:
        """Register a resource for monitoring."""
        with self._lock:
            self._resources[handle.resource_id] = handle
            logger.debug(f"Registered resource {handle.resource_id} for monitoring")
    
    def unregister_resource(self, resource_id: str) -> None:
        """Unregister a resource from monitoring."""
        with self._lock:
            if resource_id in self._resources:
                del self._resources[resource_id]
                logger.debug(f"Unregistered resource {resource_id} from monitoring")
    
    async def start_monitoring(self) -> None:
        """Start the resource monitoring loop."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Resource monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop the resource monitoring loop."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Resource monitoring stopped")
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                await self._check_resources()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in resource monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_resources(self) -> None:
        """Check all monitored resources."""
        with self._lock:
            resources_to_check = list(self._resources.values())
        
        for handle in resources_to_check:
            try:
                await self._check_resource(handle)
            except Exception as e:
                logger.error(f"Error checking resource {handle.resource_id}: {e}")
                handle.metrics.errors += 1
    
    async def _check_resource(self, handle: ResourceHandle) -> None:
        """Check a specific resource."""
        
        # Get current system metrics for the process/tool
        try:
            if handle.resource_type in [ResourceType.MEMORY, ResourceType.CPU, ResourceType.PROCESS]:
                # Get process metrics if available
                current_memory = 0
                current_cpu = 0.0
                
                # Try to get metrics for the current process
                try:
                    process = psutil.Process()
                    mem_info = process.memory_info()
                    current_memory = mem_info.rss
                    current_cpu = process.cpu_percent(interval=0.1)
                    
                    # Update peak metrics
                    handle.metrics.peak_memory = max(handle.metrics.peak_memory, current_memory)
                    handle.metrics.peak_cpu = max(handle.metrics.peak_cpu, current_cpu)
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                
                # Check if resource exceeds limits
                violations = handle.exceeds_limits(current_memory, current_cpu)
                if violations:
                    logger.warning(f"Resource {handle.resource_id} violations: {violations}")
                    handle.state = ResourceState.ERROR
                    handle.metrics.errors += 1
            
            # Check if resource is expired
            if handle.is_expired():
                logger.info(f"Resource {handle.resource_id} has expired")
                handle.state = ResourceState.CLEANUP_PENDING
        
        except Exception as e:
            logger.error(f"Failed to check resource {handle.resource_id}: {e}")
            handle.metrics.errors += 1


class ResourceManager:
    """Main resource management system."""
    
    def __init__(self):
        self.registry = get_enhanced_registry()
        self.dependency_manager = get_dependency_manager()
        
        # Resource tracking
        self._resources: Dict[str, ResourceHandle] = {}
        self._tool_resources: Dict[str, Set[str]] = {}  # tool_name -> resource_ids
        self._resource_pools: Dict[ResourceType, ResourcePool] = {}
        self._lock = threading.RLock()
        
        # Monitoring
        self.monitor = ResourceMonitor()
        
        # Cleanup tracking
        self._cleanup_queue: List[str] = []
        self._auto_cleanup = True
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Weak references to prevent memory leaks
        self._weak_refs: WeakKeyDictionary = WeakKeyDictionary()
        
        logger.info("Resource manager initialized")
    
    async def start(self) -> None:
        """Start the resource manager."""
        await self.monitor.start_monitoring()
        if self._auto_cleanup:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Resource manager started")
    
    async def stop(self) -> None:
        """Stop the resource manager and clean up all resources."""
        await self.monitor.stop_monitoring()
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clean up all remaining resources
        await self.cleanup_all_resources()
        logger.info("Resource manager stopped")
    
    def allocate_resource(self,
                         resource_type: ResourceType,
                         tool_name: str,
                         resource_data: Any,
                         limits: Optional[ResourceLimits] = None,
                         use_pool: bool = True,
                         **kwargs) -> ResourceHandle:
        """Allocate a new resource."""
        
        # Check if tool has dependencies that need resources
        try:
            dep_status = self.dependency_manager.get_dependency_status(tool_name)
            if not dep_status["resolution_success"]:
                logger.warning(f"Tool {tool_name} has unresolved dependencies, allocating resource anyway")
        except Exception as e:
            logger.warning(f"Could not check dependencies for {tool_name}: {e}")
        
        # Try to use pool if requested and available
        if use_pool and resource_type in self._resource_pools:
            pool = self._resource_pools[resource_type]
            handle = pool.get_resource(tool_name, lambda: resource_data)
            if handle:
                with self._lock:
                    self._resources[handle.resource_id] = handle
                    if tool_name not in self._tool_resources:
                        self._tool_resources[tool_name] = set()
                    self._tool_resources[tool_name].add(handle.resource_id)
                
                self.monitor.register_resource(handle)
                logger.debug(f"Allocated pooled resource {handle.resource_id} for {tool_name}")
                return handle
        
        # Create new resource
        resource_id = f"{resource_type.value}_{tool_name}_{int(time.time() * 1000)}"
        handle = ResourceHandle(
            resource_id=resource_id,
            resource_type=resource_type,
            tool_name=tool_name,
            resource_data=resource_data,
            limits=limits,
            tags=kwargs
        )
        
        with self._lock:
            self._resources[resource_id] = handle
            if tool_name not in self._tool_resources:
                self._tool_resources[tool_name] = set()
            self._tool_resources[tool_name].add(resource_id)
        
        self.monitor.register_resource(handle)
        logger.debug(f"Allocated new resource {resource_id} for {tool_name}")
        return handle
    
    def release_resource(self, resource_id: str) -> bool:
        """Release a specific resource."""
        with self._lock:
            if resource_id not in self._resources:
                logger.warning(f"Resource {resource_id} not found for release")
                return False
            
            handle = self._resources[resource_id]
            
            # Remove from tool tracking
            if handle.tool_name in self._tool_resources:
                self._tool_resources[handle.tool_name].discard(resource_id)
                if not self._tool_resources[handle.tool_name]:
                    del self._tool_resources[handle.tool_name]
        
        # Try to return to pool first
        if handle.resource_type in self._resource_pools:
            pool = self._resource_pools[handle.resource_type]
            pool.return_resource(handle)
            with self._lock:
                del self._resources[resource_id]
        else:
            # Direct cleanup
            self._cleanup_resource(handle)
            with self._lock:
                del self._resources[resource_id]
        
        self.monitor.unregister_resource(resource_id)
        logger.debug(f"Released resource {resource_id}")
        return True
    
    def release_tool_resources(self, tool_name: str) -> int:
        """Release all resources for a specific tool."""
        with self._lock:
            resource_ids = self._tool_resources.get(tool_name, set()).copy()
        
        released_count = 0
        for resource_id in resource_ids:
            if self.release_resource(resource_id):
                released_count += 1
        
        logger.info(f"Released {released_count} resources for tool {tool_name}")
        return released_count
    
    def _cleanup_resource(self, handle: ResourceHandle) -> None:
        """Internal resource cleanup."""
        try:
            handle.state = ResourceState.CLEANUP_PENDING
            
            # Run cleanup callbacks
            for callback in handle.cleanup_callbacks:
                try:
                    callback(handle.resource_data)
                except Exception as e:
                    logger.error(f"Error in cleanup callback for {handle.resource_id}: {e}")
            
            # Resource-specific cleanup
            if handle.resource_type == ResourceType.TEMPORARY_FILE:
                try:
                    import os
                    if isinstance(handle.resource_data, (str, os.PathLike)) and os.path.exists(handle.resource_data):
                        os.remove(handle.resource_data)
                except Exception as e:
                    logger.error(f"Failed to remove temporary file: {e}")
            
            elif handle.resource_type == ResourceType.DATABASE_CONNECTION:
                try:
                    if hasattr(handle.resource_data, 'close'):
                        handle.resource_data.close()
                except Exception as e:
                    logger.error(f"Failed to close database connection: {e}")
            
            elif handle.resource_type == ResourceType.PROCESS:
                try:
                    if hasattr(handle.resource_data, 'terminate'):
                        handle.resource_data.terminate()
                        # Give process time to terminate gracefully
                        if hasattr(handle.resource_data, 'wait'):
                            handle.resource_data.wait(timeout=5)
                except Exception as e:
                    logger.error(f"Failed to terminate process: {e}")
            
            handle.state = ResourceState.RELEASED
            
        except Exception as e:
            logger.error(f"Error during resource cleanup for {handle.resource_id}: {e}")
            handle.state = ResourceState.ERROR
    
    async def cleanup_expired_resources(self) -> int:
        """Clean up all expired resources."""
        expired_resources = []
        
        with self._lock:
            for resource_id, handle in self._resources.items():
                if handle.is_expired() or handle.state == ResourceState.CLEANUP_PENDING:
                    expired_resources.append(resource_id)
        
        cleaned_count = 0
        for resource_id in expired_resources:
            if self.release_resource(resource_id):
                cleaned_count += 1
        
        # Also clean up pools
        for pool in self._resource_pools.values():
            pool.cleanup_expired()
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired resources")
        
        return cleaned_count
    
    async def cleanup_all_resources(self) -> int:
        """Clean up all resources."""
        with self._lock:
            all_resource_ids = list(self._resources.keys())
        
        cleaned_count = 0
        for resource_id in all_resource_ids:
            if self.release_resource(resource_id):
                cleaned_count += 1
        
        logger.info(f"Cleaned up all {cleaned_count} resources")
        return cleaned_count
    
    async def _cleanup_loop(self) -> None:
        """Automatic cleanup loop."""
        while self._auto_cleanup:
            try:
                await self.cleanup_expired_resources()
                # Force garbage collection periodically
                gc.collect()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(30)
    
    def create_resource_pool(self, resource_type: ResourceType, max_size: int = 10) -> ResourcePool:
        """Create a resource pool for a specific resource type."""
        pool = ResourcePool(resource_type, max_size)
        self._resource_pools[resource_type] = pool
        logger.info(f"Created resource pool for {resource_type.value} with max size {max_size}")
        return pool
    
    @asynccontextmanager
    async def managed_resource(self,
                              resource_type: ResourceType,
                              tool_name: str,
                              resource_data: Any,
                              limits: Optional[ResourceLimits] = None,
                              **kwargs):
        """Context manager for automatic resource management."""
        handle = self.allocate_resource(resource_type, tool_name, resource_data, limits, **kwargs)
        try:
            yield handle
        finally:
            self.release_resource(handle.resource_id)
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get comprehensive resource statistics."""
        with self._lock:
            total_resources = len(self._resources)
            resources_by_type = {}
            resources_by_tool = {}
            resources_by_state = {}
            
            for handle in self._resources.values():
                # By type
                type_name = handle.resource_type.value
                resources_by_type[type_name] = resources_by_type.get(type_name, 0) + 1
                
                # By tool
                resources_by_tool[handle.tool_name] = resources_by_tool.get(handle.tool_name, 0) + 1
                
                # By state
                state_name = handle.state.value
                resources_by_state[state_name] = resources_by_state.get(state_name, 0) + 1
            
            # Pool stats
            pool_stats = {}
            for resource_type, pool in self._resource_pools.items():
                pool_stats[resource_type.value] = pool.get_stats()
        
        return {
            "total_resources": total_resources,
            "resources_by_type": resources_by_type,
            "resources_by_tool": resources_by_tool,
            "resources_by_state": resources_by_state,
            "pool_stats": pool_stats,
            "monitoring_active": self.monitor._monitoring,
            "auto_cleanup": self._auto_cleanup,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_tool_resource_usage(self, tool_name: str) -> Dict[str, Any]:
        """Get detailed resource usage for a specific tool."""
        with self._lock:
            tool_resource_ids = self._tool_resources.get(tool_name, set())
            tool_resources = [self._resources[rid] for rid in tool_resource_ids if rid in self._resources]
        
        if not tool_resources:
            return {"tool_name": tool_name, "resource_count": 0, "resources": []}
        
        resource_details = []
        total_memory = 0
        total_access_count = 0
        
        for handle in tool_resources:
            resource_details.append({
                "resource_id": handle.resource_id,
                "resource_type": handle.resource_type.value,
                "state": handle.state.value,
                "allocated_at": handle.metrics.allocated_at.isoformat(),
                "last_accessed": handle.metrics.last_accessed.isoformat(),
                "access_count": handle.metrics.access_count,
                "peak_memory_mb": handle.metrics.peak_memory / 1024 / 1024,
                "peak_cpu_percent": handle.metrics.peak_cpu,
                "errors": handle.metrics.errors,
                "is_expired": handle.is_expired(),
                "tags": handle.tags
            })
            
            total_memory += handle.metrics.peak_memory
            total_access_count += handle.metrics.access_count
        
        return {
            "tool_name": tool_name,
            "resource_count": len(tool_resources),
            "total_peak_memory_mb": total_memory / 1024 / 1024,
            "total_access_count": total_access_count,
            "resources": resource_details
        }


# Global resource manager instance
resource_manager = ResourceManager()


def get_resource_manager() -> ResourceManager:
    """Get the global resource manager instance."""
    return resource_manager


# Convenience functions
def allocate_resource(resource_type: ResourceType, 
                     tool_name: str,
                     resource_data: Any,
                     limits: Optional[ResourceLimits] = None,
                     **kwargs) -> ResourceHandle:
    """Convenience function to allocate a resource."""
    return resource_manager.allocate_resource(resource_type, tool_name, resource_data, limits, **kwargs)


def release_resource(resource_id: str) -> bool:
    """Convenience function to release a resource."""
    return resource_manager.release_resource(resource_id)


def release_tool_resources(tool_name: str) -> int:
    """Convenience function to release all resources for a tool."""
    return resource_manager.release_tool_resources(tool_name)


async def managed_resource(resource_type: ResourceType,
                          tool_name: str, 
                          resource_data: Any,
                          limits: Optional[ResourceLimits] = None,
                          **kwargs):
    """Convenience function for managed resource context."""
    async with resource_manager.managed_resource(resource_type, tool_name, resource_data, limits, **kwargs) as handle:
        yield handle


__all__ = [
    "ResourceManager",
    "ResourceMonitor",
    "ResourcePool",
    "ResourceHandle",
    "ResourceMetrics",
    "ResourceLimits",
    "ResourceType",
    "ResourceState",
    "resource_manager",
    "get_resource_manager",
    "allocate_resource",
    "release_resource",
    "release_tool_resources",
    "managed_resource"
]