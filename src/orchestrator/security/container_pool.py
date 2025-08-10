"""Container Pool Management - Issue #206 Task 3.1

Advanced container pooling system that reuses containers for better performance,
resource efficiency, and faster execution times while maintaining security isolation.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from collections import defaultdict
import weakref

from .docker_manager import SecureContainer, ResourceLimits, SecurityConfig, EnhancedDockerManager

logger = logging.getLogger(__name__)


class PoolContainerState(Enum):
    """Container states in the pool."""
    AVAILABLE = "available"      # Ready for use
    IN_USE = "in_use"           # Currently executing
    WARMING = "warming"         # Being prepared
    COOLDOWN = "cooldown"       # Cooling down after use
    UNHEALTHY = "unhealthy"     # Failed health check
    EXPIRED = "expired"         # Needs replacement


@dataclass
class PoolConfiguration:
    """Configuration for container pool management."""
    # Pool sizing
    min_pool_size: int = 2              # Minimum containers per image
    max_pool_size: int = 10             # Maximum containers per image
    target_pool_size: int = 5           # Target containers per image
    
    # Container lifecycle
    max_container_age_seconds: int = 3600      # 1 hour max age
    max_executions_per_container: int = 50     # Max uses before replacement
    cooldown_period_seconds: int = 30          # Cooldown after execution
    warmup_time_seconds: int = 10              # Time to warm up container
    
    # Pool management
    cleanup_interval_seconds: int = 300        # Pool cleanup every 5 minutes
    health_check_interval_seconds: int = 60    # Health check every minute
    preemptive_creation_threshold: float = 0.8 # Create new when 80% capacity
    
    # Performance optimization
    enable_container_reuse: bool = True        # Enable container reuse
    enable_preemptive_creation: bool = True    # Create containers proactively
    enable_load_balancing: bool = True         # Distribute load across containers
    
    # Security
    isolation_reset_required: bool = True     # Reset isolation between uses
    security_scan_interval: int = 10          # Scan every 10 executions


@dataclass
class PooledContainer:
    """Container in the pool with metadata."""
    container: SecureContainer
    pool_id: str
    image_name: str
    created_at: float
    last_used_at: float
    execution_count: int = 0
    state: PoolContainerState = PoolContainerState.AVAILABLE
    total_execution_time: float = 0.0
    last_health_check: float = 0.0
    security_scan_count: int = 0
    assigned_executions: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        self.pool_id = self._generate_pool_id()
    
    def _generate_pool_id(self) -> str:
        """Generate unique pool ID."""
        identifier = f"{self.image_name}_{self.container.container_id}_{time.time()}"
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]
    
    @property
    def age_seconds(self) -> float:
        """Get container age in seconds."""
        return time.time() - self.created_at
    
    @property
    def idle_time_seconds(self) -> float:
        """Get time since last use."""
        return time.time() - self.last_used_at
    
    @property
    def average_execution_time(self) -> float:
        """Get average execution time."""
        if self.execution_count == 0:
            return 0.0
        return self.total_execution_time / self.execution_count
    
    def is_available(self, config: PoolConfiguration) -> bool:
        """Check if container is available for use."""
        if self.state != PoolContainerState.AVAILABLE:
            return False
        
        # Check age limits
        if self.age_seconds > config.max_container_age_seconds:
            self.state = PoolContainerState.EXPIRED
            return False
        
        # Check execution count limits
        if self.execution_count >= config.max_executions_per_container:
            self.state = PoolContainerState.EXPIRED
            return False
        
        # Check cooldown period
        if self.idle_time_seconds < config.cooldown_period_seconds:
            self.state = PoolContainerState.COOLDOWN
            return False
        
        return True
    
    def mark_in_use(self, execution_id: str):
        """Mark container as in use."""
        self.state = PoolContainerState.IN_USE
        self.assigned_executions.add(execution_id)
        self.last_used_at = time.time()
    
    def mark_available(self, execution_id: str, execution_time: float = 0.0):
        """Mark container as available."""
        self.state = PoolContainerState.AVAILABLE
        self.assigned_executions.discard(execution_id)
        self.execution_count += 1
        self.total_execution_time += execution_time
        self.last_used_at = time.time()


class ContainerPoolManager:
    """
    Advanced container pool manager providing efficient container reuse,
    performance optimization, and resource management.
    """
    
    def __init__(self, docker_manager: EnhancedDockerManager, config: Optional[PoolConfiguration] = None):
        self.docker_manager = docker_manager
        self.config = config or PoolConfiguration()
        
        # Pool storage
        self.pools: Dict[str, List[PooledContainer]] = defaultdict(list)  # image -> containers
        self.container_lookup: Dict[str, PooledContainer] = {}  # container_id -> pooled_container
        self.execution_assignments: Dict[str, str] = {}  # execution_id -> container_id
        
        # Pool statistics
        self.stats = {
            'containers_created': 0,
            'containers_reused': 0,
            'containers_destroyed': 0,
            'pool_hits': 0,
            'pool_misses': 0,
            'total_executions': 0,
            'average_wait_time': 0.0,
            'peak_pool_size': 0,
            'current_pool_size': 0
        }
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._preemptive_creation_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info("ContainerPoolManager initialized")
    
    async def start_background_tasks(self):
        """Start background pool management tasks."""
        if self._running:
            return
        
        self._running = True
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # Start health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        # Start preemptive creation task
        if self.config.enable_preemptive_creation:
            self._preemptive_creation_task = asyncio.create_task(self._preemptive_creation_loop())
        
        logger.info("Container pool background tasks started")
    
    async def stop_background_tasks(self):
        """Stop background pool management tasks."""
        self._running = False
        
        # Cancel tasks
        for task in [self._cleanup_task, self._health_check_task, self._preemptive_creation_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Container pool background tasks stopped")
    
    async def get_container(
        self, 
        image: str,
        name_prefix: str = "pooled",
        resource_limits: Optional[ResourceLimits] = None,
        security_config: Optional[SecurityConfig] = None,
        execution_id: Optional[str] = None
    ) -> SecureContainer:
        """
        Get a container from the pool or create a new one.
        
        Args:
            image: Docker image name
            name_prefix: Prefix for container name
            resource_limits: Resource limits for the container
            security_config: Security configuration
            execution_id: Unique execution identifier
            
        Returns:
            SecureContainer ready for use
        """
        start_time = time.time()
        execution_id = execution_id or f"exec_{int(time.time() * 1000000)}"
        
        try:
            # Try to get from pool first
            pooled_container = await self._get_from_pool(image, execution_id)
            
            if pooled_container:
                self.stats['pool_hits'] += 1
                self.stats['containers_reused'] += 1
                
                # Reset container state for reuse
                if self.config.isolation_reset_required:
                    await self._reset_container_isolation(pooled_container)
                
                container = pooled_container.container
            else:
                # Create new container
                self.stats['pool_misses'] += 1
                container = await self._create_pooled_container(
                    image, name_prefix, resource_limits, security_config, execution_id
                )
            
            # Track execution assignment
            self.execution_assignments[execution_id] = container.container_id
            self.stats['total_executions'] += 1
            
            # Update wait time statistics
            wait_time = time.time() - start_time
            self._update_average_wait_time(wait_time)
            
            return container
            
        except Exception as e:
            logger.error(f"Failed to get container from pool: {e}")
            raise
    
    async def return_container(
        self, 
        container: SecureContainer, 
        execution_id: Optional[str] = None,
        execution_time: float = 0.0,
        execution_successful: bool = True
    ):
        """
        Return a container to the pool for reuse.
        
        Args:
            container: Container to return
            execution_id: Execution identifier
            execution_time: Time taken for execution
            execution_successful: Whether execution was successful
        """
        try:
            container_id = container.container_id
            pooled_container = self.container_lookup.get(container_id)
            
            if not pooled_container:
                logger.warning(f"Container {container_id} not found in pool lookup")
                return
            
            # Clean up execution assignment
            if execution_id:
                self.execution_assignments.pop(execution_id, None)
            
            # Update container state
            if execution_successful and self.config.enable_container_reuse:
                # Mark as available for reuse
                pooled_container.mark_available(execution_id or "", execution_time)
                
                # Check if security scan is needed
                if (pooled_container.execution_count % self.config.security_scan_interval == 0):
                    await self._schedule_security_scan(pooled_container)
                
                logger.debug(f"Returned container {container_id} to pool (uses: {pooled_container.execution_count})")
            else:
                # Mark for destruction
                pooled_container.state = PoolContainerState.UNHEALTHY
                await self._remove_from_pool(pooled_container)
                logger.info(f"Container {container_id} marked for destruction due to failure")
            
        except Exception as e:
            logger.error(f"Failed to return container to pool: {e}")
    
    async def _get_from_pool(self, image: str, execution_id: str) -> Optional[PooledContainer]:
        """Get an available container from the pool."""
        pool = self.pools.get(image, [])
        
        for pooled_container in pool:
            if pooled_container.is_available(self.config):
                pooled_container.mark_in_use(execution_id)
                return pooled_container
        
        return None
    
    async def _create_pooled_container(
        self,
        image: str,
        name_prefix: str,
        resource_limits: Optional[ResourceLimits],
        security_config: Optional[SecurityConfig],
        execution_id: str
    ) -> SecureContainer:
        """Create a new container and add it to the pool."""
        # Generate unique name
        timestamp = int(time.time() * 1000)
        name = f"{name_prefix}_{image.replace(':', '_').replace('/', '_')}_{timestamp}"
        
        # Create container (bypass pool to avoid recursion)
        container = await self.docker_manager.create_secure_container(
            image=image,
            name=name,
            resource_limits=resource_limits,
            security_config=security_config,
            _pool_internal=True
        )
        
        # Create pooled container
        pooled_container = PooledContainer(
            container=container,
            pool_id="",  # Will be generated in __post_init__
            image_name=image,
            created_at=time.time(),
            last_used_at=time.time(),
            state=PoolContainerState.IN_USE
        )
        
        # Mark as in use for this execution
        pooled_container.mark_in_use(execution_id)
        
        # Add to pool
        self.pools[image].append(pooled_container)
        self.container_lookup[container.container_id] = pooled_container
        
        # Update statistics
        self.stats['containers_created'] += 1
        self._update_pool_size_stats()
        
        logger.info(f"Created new pooled container {container.container_id} for image {image}")
        return container
    
    async def _remove_from_pool(self, pooled_container: PooledContainer):
        """Remove a container from the pool and destroy it."""
        try:
            image = pooled_container.image_name
            container_id = pooled_container.container.container_id
            
            # Remove from pool
            if image in self.pools:
                self.pools[image] = [pc for pc in self.pools[image] if pc.pool_id != pooled_container.pool_id]
            
            # Remove from lookup
            self.container_lookup.pop(container_id, None)
            
            # Destroy container
            await self.docker_manager.destroy_container(pooled_container.container, force=True)
            
            # Update statistics
            self.stats['containers_destroyed'] += 1
            self._update_pool_size_stats()
            
            logger.info(f"Removed and destroyed container {container_id} from pool")
            
        except Exception as e:
            logger.error(f"Failed to remove container from pool: {e}")
    
    async def _reset_container_isolation(self, pooled_container: PooledContainer):
        """Reset container isolation and state for reuse."""
        try:
            container = pooled_container.container
            
            # Clear any temporary files
            await self.docker_manager.execute_in_container(
                container,
                "find /tmp -type f -delete 2>/dev/null || true",
                timeout=10
            )
            
            # Reset working directory
            await self.docker_manager.execute_in_container(
                container,
                "cd /",
                timeout=5
            )
            
            # Clear environment variables (if possible)
            await self.docker_manager.execute_in_container(
                container,
                "unset $(env | grep '^TEMP\\|^TMP' | cut -d= -f1) 2>/dev/null || true",
                timeout=5
            )
            
            logger.debug(f"Reset isolation for container {container.container_id}")
            
        except Exception as e:
            logger.warning(f"Failed to reset container isolation: {e}")
    
    async def _schedule_security_scan(self, pooled_container: PooledContainer):
        """Schedule a security scan for a container."""
        try:
            # Perform basic security checks
            container = pooled_container.container
            
            # Check for suspicious processes
            result = await self.docker_manager.execute_in_container(
                container,
                "ps aux | grep -v 'ps aux' | wc -l",
                timeout=10
            )
            
            if result.get('success') and result.get('output'):
                process_count = int(result['output'].strip())
                if process_count > 50:  # Arbitrary threshold
                    logger.warning(f"Container {container.container_id} has {process_count} processes")
                    pooled_container.state = PoolContainerState.UNHEALTHY
            
            pooled_container.security_scan_count += 1
            logger.debug(f"Security scan completed for container {container.container_id}")
            
        except Exception as e:
            logger.warning(f"Security scan failed for container: {e}")
    
    async def _cleanup_loop(self):
        """Background task for pool cleanup."""
        while self._running:
            try:
                await self._cleanup_expired_containers()
                await asyncio.sleep(self.config.cleanup_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Pool cleanup error: {e}")
                await asyncio.sleep(10)  # Short delay on error
    
    async def _cleanup_expired_containers(self):
        """Clean up expired and unhealthy containers."""
        cleanup_count = 0
        
        for image, pool in list(self.pools.items()):
            containers_to_remove = []
            
            for pooled_container in pool:
                # Check if container is expired or unhealthy
                if (pooled_container.state in [PoolContainerState.EXPIRED, PoolContainerState.UNHEALTHY] or
                    pooled_container.age_seconds > self.config.max_container_age_seconds or
                    pooled_container.execution_count >= self.config.max_executions_per_container):
                    
                    containers_to_remove.append(pooled_container)
            
            # Remove expired containers
            for pooled_container in containers_to_remove:
                await self._remove_from_pool(pooled_container)
                cleanup_count += 1
        
        if cleanup_count > 0:
            logger.info(f"Cleaned up {cleanup_count} expired containers")
    
    async def _health_check_loop(self):
        """Background task for container health checks."""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.health_check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(30)  # Longer delay on error
    
    async def _perform_health_checks(self):
        """Perform health checks on all pooled containers."""
        unhealthy_count = 0
        
        for pooled_container in list(self.container_lookup.values()):
            if pooled_container.state == PoolContainerState.IN_USE:
                continue  # Skip containers currently in use
            
            try:
                # Simple health check - execute a basic command
                result = await self.docker_manager.execute_in_container(
                    pooled_container.container,
                    "echo 'health_check'",
                    timeout=5
                )
                
                if result.get('success') and 'health_check' in result.get('output', ''):
                    pooled_container.last_health_check = time.time()
                else:
                    pooled_container.state = PoolContainerState.UNHEALTHY
                    unhealthy_count += 1
                    
            except Exception as e:
                logger.debug(f"Health check failed for container {pooled_container.container.container_id}: {e}")
                pooled_container.state = PoolContainerState.UNHEALTHY
                unhealthy_count += 1
        
        if unhealthy_count > 0:
            logger.info(f"Marked {unhealthy_count} containers as unhealthy")
    
    async def _preemptive_creation_loop(self):
        """Background task for preemptive container creation."""
        while self._running:
            try:
                await self._create_preemptive_containers()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Preemptive creation error: {e}")
                await asyncio.sleep(30)
    
    async def _create_preemptive_containers(self):
        """Create containers preemptively based on usage patterns."""
        for image, pool in self.pools.items():
            available_count = sum(1 for pc in pool if pc.state == PoolContainerState.AVAILABLE)
            total_count = len(pool)
            
            # Check if we need more containers
            if (total_count < self.config.max_pool_size and
                available_count < self.config.min_pool_size):
                
                try:
                    # Create a new container (bypass pool to avoid recursion)
                    container = await self.docker_manager.create_secure_container(
                        image=image,
                        name=f"preemptive_{image.replace(':', '_').replace('/', '_')}_{int(time.time())}",
                        _pool_internal=True
                    )
                    
                    # Add to pool
                    pooled_container = PooledContainer(
                        container=container,
                        pool_id="",
                        image_name=image,
                        created_at=time.time(),
                        last_used_at=time.time(),
                        state=PoolContainerState.AVAILABLE
                    )
                    
                    self.pools[image].append(pooled_container)
                    self.container_lookup[container.container_id] = pooled_container
                    
                    logger.info(f"Created preemptive container for image {image}")
                    
                except Exception as e:
                    logger.warning(f"Failed to create preemptive container for {image}: {e}")
    
    def _update_average_wait_time(self, wait_time: float):
        """Update average wait time statistics."""
        current_avg = self.stats['average_wait_time']
        total_execs = self.stats['total_executions']
        
        # Exponential moving average
        if total_execs == 1:
            self.stats['average_wait_time'] = wait_time
        else:
            alpha = 2.0 / (min(total_execs, 100) + 1)  # Decay factor
            self.stats['average_wait_time'] = alpha * wait_time + (1 - alpha) * current_avg
    
    def _update_pool_size_stats(self):
        """Update pool size statistics."""
        current_size = len(self.container_lookup)
        self.stats['current_pool_size'] = current_size
        if current_size > self.stats['peak_pool_size']:
            self.stats['peak_pool_size'] = current_size
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics."""
        stats = self.stats.copy()
        
        # Add pool-specific statistics
        pool_stats = {}
        for image, pool in self.pools.items():
            pool_stats[image] = {
                'total_containers': len(pool),
                'available_containers': sum(1 for pc in pool if pc.state == PoolContainerState.AVAILABLE),
                'in_use_containers': sum(1 for pc in pool if pc.state == PoolContainerState.IN_USE),
                'unhealthy_containers': sum(1 for pc in pool if pc.state == PoolContainerState.UNHEALTHY),
                'average_age': sum(pc.age_seconds for pc in pool) / len(pool) if pool else 0,
                'total_executions': sum(pc.execution_count for pc in pool),
                'average_execution_time': sum(pc.average_execution_time for pc in pool) / len(pool) if pool else 0
            }
        
        stats['pool_details'] = pool_stats
        stats['configuration'] = {
            'min_pool_size': self.config.min_pool_size,
            'max_pool_size': self.config.max_pool_size,
            'target_pool_size': self.config.target_pool_size,
            'container_reuse_enabled': self.config.enable_container_reuse,
            'preemptive_creation_enabled': self.config.enable_preemptive_creation
        }
        
        return stats
    
    async def shutdown(self):
        """Shutdown the container pool manager."""
        logger.info("Shutting down container pool manager...")
        
        # Stop background tasks
        await self.stop_background_tasks()
        
        # Clean up all containers
        cleanup_count = 0
        for pooled_container in list(self.container_lookup.values()):
            await self._remove_from_pool(pooled_container)
            cleanup_count += 1
        
        logger.info(f"Container pool manager shutdown complete. Cleaned up {cleanup_count} containers.")


# Export classes
__all__ = [
    'ContainerPoolManager',
    'PoolConfiguration',
    'PooledContainer',
    'PoolContainerState'
]