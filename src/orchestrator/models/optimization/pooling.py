"""Connection pooling system for efficient model utilization."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from ...core.model import Model

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """Configuration for connection pool."""
    
    min_connections: int = 1
    max_connections: int = 10
    connection_timeout: float = 30.0
    idle_timeout: float = 300.0
    max_retries: int = 3


@dataclass
class PoolStats:
    """Statistics for connection pool performance."""
    
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    created_connections: int = 0
    destroyed_connections: int = 0
    requests_served: int = 0
    requests_queued: int = 0
    max_queue_time: float = 0.0
    average_queue_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_connections": self.total_connections,
            "active_connections": self.active_connections,
            "idle_connections": self.idle_connections,
            "created_connections": self.created_connections,
            "destroyed_connections": self.destroyed_connections,
            "requests_served": self.requests_served,
            "requests_queued": self.requests_queued,
            "max_queue_time": self.max_queue_time,
            "average_queue_time": self.average_queue_time,
            "utilization_rate": self.active_connections / max(self.total_connections, 1),
        }


@dataclass
class PoolConnection:
    """Represents a pooled connection/model instance."""
    
    model: Model
    provider: str
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    is_active: bool = False
    health_check_failures: int = 0
    
    def use(self) -> None:
        """Mark connection as used."""
        self.last_used = time.time()
        self.use_count += 1
        self.is_active = True
    
    def release(self) -> None:
        """Release connection back to pool."""
        self.is_active = False
    
    def is_stale(self, max_idle_time: float) -> bool:
        """Check if connection is stale (idle too long)."""
        return time.time() - self.last_used > max_idle_time
    
    def is_overused(self, max_uses: int) -> bool:
        """Check if connection has been used too many times."""
        return self.use_count >= max_uses


@dataclass
class QueuedRequest:
    """Represents a queued request waiting for a connection."""
    
    future: asyncio.Future
    queued_at: float = field(default_factory=time.time)
    model_name: str = ""
    method: str = ""
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    
    def get_queue_time(self) -> float:
        """Get time spent in queue."""
        return time.time() - self.queued_at


class ConnectionPool:
    """
    Connection pool for efficient model instance management.
    
    Features:
    - Connection reuse to avoid repeated initialization
    - Automatic connection health monitoring
    - Request queuing when pool is full
    - Connection lifecycle management
    - Performance statistics
    """
    
    def __init__(
        self,
        provider_name: str,
        min_connections: int = 1,
        max_connections: int = 10,
        max_idle_time: float = 300.0,  # 5 minutes
        max_uses_per_connection: int = 1000,
        health_check_interval: float = 60.0,  # 1 minute
        queue_timeout: float = 30.0,  # 30 seconds
    ):
        """
        Initialize connection pool.
        
        Args:
            provider_name: Name of the provider this pool serves
            min_connections: Minimum connections to maintain
            max_connections: Maximum connections allowed
            max_idle_time: Maximum idle time before connection cleanup
            max_uses_per_connection: Maximum uses before connection refresh
            health_check_interval: Interval between health checks
            queue_timeout: Maximum time to wait in queue
        """
        self.provider_name = provider_name
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.max_idle_time = max_idle_time
        self.max_uses_per_connection = max_uses_per_connection
        self.health_check_interval = health_check_interval
        self.queue_timeout = queue_timeout
        
        # Connection management
        self._connections: List[PoolConnection] = []
        self._connection_lock = asyncio.Lock()
        self._request_queue: List[QueuedRequest] = []
        self._queue_lock = asyncio.Lock()
        
        # Statistics
        self._stats = PoolStats()
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._initialized = False
        
        logger.info(f"ConnectionPool created for {provider_name} (min={min_connections}, max={max_connections})")
    
    async def initialize(self) -> None:
        """Initialize the connection pool."""
        if self._initialized:
            return
        
        # Start background tasks
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        self._health_check_task = asyncio.create_task(self._periodic_health_check())
        
        self._initialized = True
        logger.info(f"ConnectionPool initialized for {self.provider_name}")
    
    async def get_connection(self, model: Model) -> PoolConnection:
        """
        Get a connection from the pool.
        
        Args:
            model: Model to get connection for
            
        Returns:
            Pool connection
        """
        async with self._connection_lock:
            # Look for idle connection for this model
            for conn in self._connections:
                if (
                    not conn.is_active and
                    conn.model.name == model.name and
                    conn.provider == model.provider and
                    not conn.is_stale(self.max_idle_time) and
                    not conn.is_overused(self.max_uses_per_connection)
                ):
                    conn.use()
                    self._stats.active_connections += 1
                    self._stats.idle_connections -= 1
                    return conn
            
            # If no suitable connection and under max limit, create new one
            if len(self._connections) < self.max_connections:
                conn = await self._create_connection(model)
                self._connections.append(conn)
                conn.use()
                self._stats.total_connections += 1
                self._stats.active_connections += 1
                self._stats.created_connections += 1
                return conn
        
        # Pool is full, queue the request
        return await self._queue_request(model)
    
    async def release_connection(self, connection: PoolConnection) -> None:
        """
        Release a connection back to the pool.
        
        Args:
            connection: Connection to release
        """
        async with self._connection_lock:
            if connection in self._connections:
                connection.release()
                self._stats.active_connections -= 1
                self._stats.idle_connections += 1
                
                # Process queued requests
                await self._process_queue()
    
    async def execute_with_model(
        self,
        model: Model,
        method: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a method on a model using pooled connection.
        
        Args:
            model: Model to execute method on
            method: Method name to call
            *args: Method arguments
            **kwargs: Method keyword arguments
            
        Returns:
            Method result
        """
        connection = await self.get_connection(model)
        
        try:
            # Get the method from the model
            model_method = getattr(connection.model, method)
            if not callable(model_method):
                raise AttributeError(f"Method '{method}' is not callable on model {model.name}")
            
            # Execute the method
            result = await model_method(*args, **kwargs)
            self._stats.requests_served += 1
            
            return result
            
        except Exception as e:
            # Mark connection as potentially unhealthy
            connection.health_check_failures += 1
            logger.error(f"Error executing {method} on {model.name}: {e}")
            raise
            
        finally:
            await self.release_connection(connection)
    
    async def get_stats(self) -> PoolStats:
        """Get current pool statistics."""
        async with self._connection_lock:
            self._stats.total_connections = len(self._connections)
            self._stats.active_connections = sum(1 for conn in self._connections if conn.is_active)
            self._stats.idle_connections = self._stats.total_connections - self._stats.active_connections
            
        async with self._queue_lock:
            self._stats.requests_queued = len(self._request_queue)
        
        return self._stats
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all connections.
        
        Returns:
            Health check results
        """
        results = {
            "healthy_connections": 0,
            "unhealthy_connections": 0,
            "total_connections": 0,
            "connection_details": [],
        }
        
        async with self._connection_lock:
            for i, conn in enumerate(self._connections):
                try:
                    is_healthy = await conn.model.health_check()
                    if is_healthy:
                        results["healthy_connections"] += 1
                        conn.health_check_failures = 0  # Reset failure count
                    else:
                        results["unhealthy_connections"] += 1
                        conn.health_check_failures += 1
                    
                    results["connection_details"].append({
                        "index": i,
                        "model": conn.model.name,
                        "provider": conn.provider,
                        "healthy": is_healthy,
                        "active": conn.is_active,
                        "use_count": conn.use_count,
                        "age_seconds": time.time() - conn.created_at,
                        "idle_seconds": time.time() - conn.last_used,
                        "health_failures": conn.health_check_failures,
                    })
                    
                except Exception as e:
                    results["unhealthy_connections"] += 1
                    conn.health_check_failures += 1
                    results["connection_details"].append({
                        "index": i,
                        "model": conn.model.name,
                        "provider": conn.provider,
                        "healthy": False,
                        "error": str(e),
                        "health_failures": conn.health_check_failures,
                    })
            
            results["total_connections"] = len(self._connections)
        
        return results
    
    async def cleanup(self) -> None:
        """Clean up pool resources."""
        logger.info(f"Cleaning up ConnectionPool for {self.provider_name}")
        
        # Cancel background tasks
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Clean up connections
        async with self._connection_lock:
            for conn in self._connections:
                await self._destroy_connection(conn)
            self._connections.clear()
        
        # Clear queue
        async with self._queue_lock:
            for req in self._request_queue:
                if not req.future.done():
                    req.future.cancel()
            self._request_queue.clear()
        
        self._initialized = False
        logger.info(f"ConnectionPool cleanup completed for {self.provider_name}")
    
    async def _create_connection(self, model: Model) -> PoolConnection:
        """Create a new connection."""
        # For this implementation, we assume the model is already initialized
        # In practice, you might want to create a fresh instance or ensure proper state
        connection = PoolConnection(
            model=model,
            provider=model.provider,
        )
        
        logger.debug(f"Created new connection for {model.name}")
        return connection
    
    async def _destroy_connection(self, connection: PoolConnection) -> None:
        """Destroy a connection."""
        try:
            # Perform any cleanup needed for the model connection
            if hasattr(connection.model, 'cleanup'):
                await connection.model.cleanup()
            
            self._stats.destroyed_connections += 1
            logger.debug(f"Destroyed connection for {connection.model.name}")
            
        except Exception as e:
            logger.warning(f"Error destroying connection for {connection.model.name}: {e}")
    
    async def _queue_request(self, model: Model) -> PoolConnection:
        """Queue a request and wait for available connection."""
        future = asyncio.Future()
        
        request = QueuedRequest(
            future=future,
            model_name=model.name,
            method="get_connection",
            args=(model,),
        )
        
        async with self._queue_lock:
            self._request_queue.append(request)
        
        try:
            # Wait for connection with timeout
            connection = await asyncio.wait_for(future, timeout=self.queue_timeout)
            
            # Update queue statistics
            queue_time = request.get_queue_time()
            if queue_time > self._stats.max_queue_time:
                self._stats.max_queue_time = queue_time
            
            # Update average queue time (simple moving average)
            if self._stats.requests_served > 0:
                self._stats.average_queue_time = (
                    self._stats.average_queue_time * 0.9 + queue_time * 0.1
                )
            else:
                self._stats.average_queue_time = queue_time
            
            return connection
            
        except asyncio.TimeoutError:
            # Remove from queue
            async with self._queue_lock:
                if request in self._request_queue:
                    self._request_queue.remove(request)
            
            raise TimeoutError(f"Timeout waiting for connection to {model.name} after {self.queue_timeout}s")
    
    async def _process_queue(self) -> None:
        """Process queued requests."""
        async with self._queue_lock:
            if not self._request_queue:
                return
            
            # Find idle connections that can serve queued requests
            idle_connections = [
                conn for conn in self._connections 
                if not conn.is_active and not conn.is_stale(self.max_idle_time)
            ]
            
            processed = 0
            while self._request_queue and idle_connections and processed < len(idle_connections):
                request = self._request_queue.pop(0)
                connection = idle_connections[processed]
                
                if not request.future.done():
                    connection.use()
                    self._stats.active_connections += 1
                    self._stats.idle_connections -= 1
                    request.future.set_result(connection)
                
                processed += 1
    
    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup task."""
        while True:
            try:
                await asyncio.sleep(self.max_idle_time / 2)  # Cleanup every half of max idle time
                await self._cleanup_stale_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    async def _cleanup_stale_connections(self) -> None:
        """Remove stale and overused connections."""
        async with self._connection_lock:
            connections_to_remove = []
            
            for conn in self._connections:
                if conn.is_active:
                    continue  # Don't remove active connections
                
                should_remove = (
                    conn.is_stale(self.max_idle_time) or
                    conn.is_overused(self.max_uses_per_connection) or
                    conn.health_check_failures > 3
                )
                
                # Don't remove if it would go below minimum connections
                if should_remove and len(self._connections) > self.min_connections:
                    connections_to_remove.append(conn)
            
            # Remove stale connections
            for conn in connections_to_remove:
                self._connections.remove(conn)
                await self._destroy_connection(conn)
                self._stats.idle_connections -= 1
                self._stats.total_connections -= 1
            
            if connections_to_remove:
                logger.debug(f"Cleaned up {len(connections_to_remove)} stale connections")
    
    async def _periodic_health_check(self) -> None:
        """Periodic health check task."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self.health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic health check: {e}")
    
    def get_pool_info(self) -> Dict[str, Any]:
        """Get pool configuration and status information."""
        return {
            "provider_name": self.provider_name,
            "min_connections": self.min_connections,
            "max_connections": self.max_connections,
            "max_idle_time": self.max_idle_time,
            "max_uses_per_connection": self.max_uses_per_connection,
            "health_check_interval": self.health_check_interval,
            "queue_timeout": self.queue_timeout,
            "initialized": self._initialized,
            "stats": self._stats.to_dict() if hasattr(self._stats, 'to_dict') else str(self._stats),
        }
    
    def __str__(self) -> str:
        """String representation of pool."""
        return f"ConnectionPool({self.provider_name}, {len(self._connections)}/{self.max_connections})"