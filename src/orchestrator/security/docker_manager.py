"""Enhanced Docker Container Manager - Issue #206 Task 1.1

Production-grade Docker container management with advanced security controls,
container pooling, resource monitoring, and performance optimization.

Phase 3 Updates:
- Integrated with ContainerPoolManager for efficient container reuse
- Added performance monitoring and analytics integration
"""

import asyncio
import docker
import logging
import time
import json
import uuid
import psutil
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import hashlib
import threading
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ContainerStatus(Enum):
    """Container status states."""
    CREATING = "creating"
    RUNNING = "running" 
    STOPPED = "stopped"
    FAILED = "failed"
    POOLED = "pooled"
    DESTROYED = "destroyed"


class SandboxingLevel(Enum):
    """Sandboxing security levels."""
    DIRECT = "direct"           # No sandboxing (safe operations)
    SANDBOXED = "sandboxed"     # Standard sandboxing
    ISOLATED = "isolated"       # High security isolation
    BLOCKED = "blocked"         # Operation not allowed


@dataclass
class ResourceLimits:
    """Container resource limits configuration."""
    memory_mb: int = 512
    cpu_cores: float = 0.5
    cpu_quota: int = 50000  # 50% of 100000 (one core)
    cpu_period: int = 100000
    pids_limit: int = 100
    disk_limit_mb: int = 1024
    network_bandwidth_mbps: Optional[int] = None
    execution_timeout: int = 30
    
    def to_docker_config(self) -> Dict[str, Any]:
        """Convert to Docker container configuration."""
        return {
            'mem_limit': f"{self.memory_mb}m",
            'memswap_limit': f"{self.memory_mb}m",  # No swap
            'cpu_quota': self.cpu_quota,
            'cpu_period': self.cpu_period,
            'pids_limit': self.pids_limit,
        }


@dataclass
class SecurityConfig:
    """Container security configuration."""
    read_only_root: bool = True
    no_new_privileges: bool = True
    drop_all_capabilities: bool = True
    allowed_capabilities: List[str] = field(default_factory=lambda: ['SETUID', 'SETGID'])
    blocked_capabilities: List[str] = field(default_factory=lambda: ['SYS_ADMIN', 'NET_ADMIN', 'SYS_TIME'])
    user_namespace: bool = True
    network_isolation: bool = True
    apparmor_profile: str = "docker-default"
    seccomp_profile: Optional[str] = None
    
    def to_docker_config(self) -> Dict[str, Any]:
        """Convert to Docker security configuration."""
        config = {
            'read_only': self.read_only_root,
            'security_opt': [],
            'cap_drop': ['ALL'] if self.drop_all_capabilities else [],
            'cap_add': self.allowed_capabilities,
            'user': '1000:1000',  # Non-root user
        }
        
        if self.no_new_privileges:
            config['security_opt'].append('no-new-privileges:true')
        
        if self.apparmor_profile:
            config['security_opt'].append(f'apparmor:{self.apparmor_profile}')
        
        if self.seccomp_profile:
            config['security_opt'].append(f'seccomp:{self.seccomp_profile}')
        
        if self.network_isolation:
            config['network_mode'] = 'none'
        
        return config


@dataclass
class ContainerMetrics:
    """Container performance and resource usage metrics."""
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    memory_limit_mb: float = 0.0
    network_rx_bytes: int = 0
    network_tx_bytes: int = 0
    disk_read_bytes: int = 0
    disk_write_bytes: int = 0
    pids_current: int = 0
    pids_limit: int = 0
    uptime_seconds: float = 0.0
    
    @property
    def memory_usage_percent(self) -> float:
        """Calculate memory usage percentage."""
        if self.memory_limit_mb > 0:
            return (self.memory_usage_mb / self.memory_limit_mb) * 100
        return 0.0


@dataclass
class SecureContainer:
    """Represents a secure Docker container with enhanced management."""
    container_id: str
    name: str
    image: str
    status: ContainerStatus
    resource_limits: ResourceLimits
    security_config: SecurityConfig
    created_at: datetime
    docker_container: Any = None
    metrics: ContainerMetrics = field(default_factory=ContainerMetrics)
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    health_checks: List[Dict[str, Any]] = field(default_factory=list)
    is_pooled: bool = False
    pool_key: Optional[str] = None
    last_used: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now()
    
    def add_execution(self, execution_data: Dict[str, Any]):
        """Add execution record to history."""
        execution_data['timestamp'] = datetime.now().isoformat()
        self.execution_history.append(execution_data)
        self.last_used = datetime.now()
        
        # Keep only last 50 executions
        if len(self.execution_history) > 50:
            self.execution_history = self.execution_history[-50:]
    
    def add_health_check(self, health_data: Dict[str, Any]):
        """Add health check result to history."""
        health_data['timestamp'] = datetime.now().isoformat()
        self.health_checks.append(health_data)
        
        # Keep only last 20 health checks
        if len(self.health_checks) > 20:
            self.health_checks = self.health_checks[-20:]
    
    @property
    def age_seconds(self) -> float:
        """Get container age in seconds."""
        return (datetime.now() - self.created_at).total_seconds()
    
    @property
    def idle_seconds(self) -> float:
        """Get seconds since last use."""
        return (datetime.now() - self.last_used).total_seconds()


class ContainerPool:
    """Container pool for efficient container reuse and management."""
    
    def __init__(self, max_pool_size: int = 50, max_idle_time: int = 300):
        self.max_pool_size = max_pool_size
        self.max_idle_time = max_idle_time
        self.pools: Dict[str, List[SecureContainer]] = {}
        self.active_containers: Dict[str, SecureContainer] = {}
        self.pool_stats = {
            'created': 0,
            'reused': 0,
            'destroyed': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        self._lock = threading.RLock()
        self._cleanup_task: Optional[asyncio.Task] = None
        
    def generate_pool_key(
        self, 
        image: str, 
        resource_limits: ResourceLimits,
        security_config: SecurityConfig
    ) -> str:
        """Generate unique pool key for container configuration."""
        config_data = {
            'image': image,
            'memory_mb': resource_limits.memory_mb,
            'cpu_cores': resource_limits.cpu_cores,
            'read_only': security_config.read_only_root,
            'capabilities': sorted(security_config.allowed_capabilities),
            'network_isolation': security_config.network_isolation
        }
        
        config_str = json.dumps(config_data, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]
    
    def get_pool_container(
        self, 
        image: str,
        resource_limits: ResourceLimits,
        security_config: SecurityConfig
    ) -> Optional[SecureContainer]:
        """Get container from pool if available."""
        pool_key = self.generate_pool_key(image, resource_limits, security_config)
        
        with self._lock:
            if pool_key not in self.pools:
                self.pools[pool_key] = []
                self.pool_stats['cache_misses'] += 1
                return None
            
            pool = self.pools[pool_key]
            if not pool:
                self.pool_stats['cache_misses'] += 1
                return None
            
            # Get the most recently used container
            container = pool.pop()
            container.is_pooled = False
            container.status = ContainerStatus.RUNNING
            container.last_used = datetime.now()
            
            self.active_containers[container.container_id] = container
            self.pool_stats['cache_hits'] += 1
            self.pool_stats['reused'] += 1
            
            logger.info(f"Reused container {container.name} from pool {pool_key}")
            return container
    
    def return_container(self, container: SecureContainer) -> bool:
        """Return container to pool for reuse."""
        with self._lock:
            if container.container_id in self.active_containers:
                del self.active_containers[container.container_id]
            
            if not container.pool_key:
                logger.warning(f"Container {container.name} has no pool key, cannot return to pool")
                return False
            
            # Check if container is healthy for reuse
            if not self._is_container_healthy_for_reuse(container):
                logger.info(f"Container {container.name} not healthy for reuse, destroying")
                return False
            
            # Initialize pool if needed
            if container.pool_key not in self.pools:
                self.pools[container.pool_key] = []
            
            pool = self.pools[container.pool_key]
            
            # Check pool size limits
            if len(pool) >= self.max_pool_size:
                logger.info(f"Pool {container.pool_key} full, destroying container {container.name}")
                return False
            
            # Reset container for reuse
            container.is_pooled = True
            container.status = ContainerStatus.POOLED
            container.execution_history.clear()
            
            pool.append(container)
            logger.info(f"Returned container {container.name} to pool {container.pool_key}")
            return True
    
    def _is_container_healthy_for_reuse(self, container: SecureContainer) -> bool:
        """Check if container is healthy and suitable for reuse."""
        try:
            # Check container age
            if container.age_seconds > 3600:  # 1 hour max age
                return False
            
            # Check resource usage
            if container.metrics.memory_usage_percent > 80:
                return False
            
            if container.metrics.cpu_usage_percent > 90:
                return False
            
            # Check execution history for failures
            recent_executions = container.execution_history[-5:]  # Last 5 executions
            if len(recent_executions) > 0:
                failure_rate = len([e for e in recent_executions if not e.get('success', True)]) / len(recent_executions)
                if failure_rate > 0.4:  # More than 40% failures
                    return False
            
            # Check Docker container status
            if container.docker_container:
                try:
                    container.docker_container.reload()
                    if container.docker_container.status != 'running':
                        return False
                except Exception:
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Health check failed for container {container.name}: {e}")
            return False
    
    async def cleanup_idle_containers(self):
        """Clean up idle containers from pools."""
        with self._lock:
            containers_to_remove = []
            
            for pool_key, pool in self.pools.items():
                for i, container in enumerate(pool):
                    if container.idle_seconds > self.max_idle_time:
                        containers_to_remove.append((pool_key, i, container))
            
            # Remove from pools (reverse order to maintain indices)
            for pool_key, index, container in reversed(containers_to_remove):
                self.pools[pool_key].pop(index)
                self.pool_stats['destroyed'] += 1
                logger.info(f"Cleaned up idle container {container.name} from pool {pool_key}")
                
                # Destroy Docker container
                try:
                    if container.docker_container:
                        container.docker_container.remove(force=True)
                except Exception as e:
                    logger.warning(f"Error destroying idle container {container.name}: {e}")
    
    async def start_cleanup_task(self):
        """Start background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Started container pool cleanup task")
    
    async def stop_cleanup_task(self):
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Stopped container pool cleanup task")
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self.cleanup_idle_containers()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in container pool cleanup loop: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            total_pooled = sum(len(pool) for pool in self.pools.values())
            active_count = len(self.active_containers)
            
            return {
                'pools': len(self.pools),
                'total_pooled_containers': total_pooled,
                'active_containers': active_count,
                'max_pool_size': self.max_pool_size,
                'max_idle_time': self.max_idle_time,
                **self.pool_stats,
                'reuse_rate': self.pool_stats['reused'] / max(self.pool_stats['created'], 1),
                'cache_hit_rate': self.pool_stats['cache_hits'] / max(self.pool_stats['cache_hits'] + self.pool_stats['cache_misses'], 1)
            }


class EnhancedDockerManager:
    """Production-grade Docker container management with security and performance optimization."""
    
    def __init__(self, enable_container_pooling: bool = True, enable_advanced_pooling: bool = True, performance_monitor=None):
        self.docker_client: Optional[docker.DockerClient] = None
        
        # Performance monitoring integration
        self.performance_monitor = performance_monitor
        
        # Container pooling - support both simple and advanced pooling
        self.enable_advanced_pooling = enable_advanced_pooling
        if enable_advanced_pooling:
            # Import here to avoid circular imports
            from .container_pool import ContainerPoolManager, PoolConfiguration
            self.pool_manager: Optional['ContainerPoolManager'] = None
            self.pool_config = PoolConfiguration()
            self.container_pool = None  # Not used with advanced pooling
        else:
            self.container_pool = ContainerPool() if enable_container_pooling else None
            self.pool_manager = None
        
        self.active_containers: Dict[str, SecureContainer] = {}
        self.container_metrics: Dict[str, ContainerMetrics] = {}
        
        # Manager statistics
        self.stats = {
            'containers_created': 0,
            'containers_destroyed': 0,
            'executions_successful': 0,
            'executions_failed': 0,
            'security_violations': 0,
            'resource_violations': 0
        }
        
        # Initialize Docker client
        self._init_docker()
        
        # Background tasks
        self._metrics_task: Optional[asyncio.Task] = None
        
        logger.info("EnhancedDockerManager initialized")
    
    def _init_docker(self) -> None:
        """Initialize Docker client with error handling."""
        try:
            self.docker_client = docker.from_env()
            # Test Docker connectivity
            self.docker_client.ping()
            
            # Get Docker info
            docker_info = self.docker_client.info()
            logger.info(f"Docker client initialized - Version: {docker_info.get('ServerVersion', 'Unknown')}")
            logger.info(f"Docker security features: {docker_info.get('SecurityOptions', [])}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            self.docker_client = None
            raise RuntimeError(f"Docker initialization failed: {e}")
    
    async def create_secure_container(
        self,
        image: str,
        name: Optional[str] = None,
        command: Optional[List[str]] = None,
        resource_limits: Optional[ResourceLimits] = None,
        security_config: Optional[SecurityConfig] = None,
        environment: Optional[Dict[str, str]] = None,
        volumes: Optional[Dict[str, Dict[str, str]]] = None,
        working_dir: Optional[str] = None,
        try_pool: bool = True,
        _pool_internal: bool = False
    ) -> SecureContainer:
        """Create secure Docker container with advanced configuration."""
        
        if not self.docker_client:
            raise RuntimeError("Docker client not available")
        
        # Track container creation performance
        creation_start_time = time.time()
        
        # Use defaults if not provided
        if resource_limits is None:
            resource_limits = ResourceLimits()
        if security_config is None:
            security_config = SecurityConfig()
        
        # Try to get container from advanced pool first (but not if called from pool internally)
        if try_pool and self.pool_manager and not _pool_internal:
            try:
                pooled_container = await self.pool_manager.get_container(
                    image=image,
                    name_prefix=name or "secure",
                    resource_limits=resource_limits,
                    security_config=security_config
                )
                if pooled_container:
                    logger.info(f"Using advanced pooled container {pooled_container.name}")
                    return pooled_container
            except Exception as e:
                logger.warning(f"Failed to get container from advanced pool: {e}")
        
        # Fallback to simple pool
        elif try_pool and self.container_pool:
            pooled_container = self.container_pool.get_pool_container(
                image, resource_limits, security_config
            )
            if pooled_container:
                logger.info(f"Using pooled container {pooled_container.name}")
                return pooled_container
        
        # Generate unique container name
        if not name:
            name = f"sandbox-{uuid.uuid4().hex[:8]}"
        
        # Ensure image is available
        await self._ensure_image_available(image)
        
        # Build container configuration
        container_config = {
            'image': image,
            'name': name,
            'command': command or ['/bin/bash', '-c', 'sleep infinity'],
            'detach': True,
            'auto_remove': False,
            'stdin_open': True,
            'tty': True,
        }
        
        # Add resource limits
        container_config.update(resource_limits.to_docker_config())
        
        # Add security configuration
        container_config.update(security_config.to_docker_config())
        
        # Add optional configurations
        if environment:
            container_config['environment'] = environment
        
        if volumes:
            container_config['volumes'] = volumes
        else:
            # Default secure volume configuration
            container_config['tmpfs'] = {
                '/tmp': 'rw,noexec,nosuid,size=100m',
                '/var/tmp': 'rw,noexec,nosuid,size=50m'
            }
        
        if working_dir:
            container_config['working_dir'] = working_dir
        
        try:
            # Create Docker container
            docker_container = self.docker_client.containers.create(**container_config)
            
            # Create SecureContainer wrapper
            container = SecureContainer(
                container_id=docker_container.id,
                name=name,
                image=image,
                status=ContainerStatus.CREATING,
                resource_limits=resource_limits,
                security_config=security_config,
                created_at=datetime.now(),
                docker_container=docker_container,
                pool_key=self.container_pool.generate_pool_key(image, resource_limits, security_config) if self.container_pool else None
            )
            
            # Start container
            docker_container.start()
            container.status = ContainerStatus.RUNNING
            
            # Track container startup performance
            creation_time = time.time() - creation_start_time
            if self.performance_monitor:
                asyncio.create_task(self.performance_monitor.record_container_startup(
                    container_id=container.container_id,
                    startup_time=creation_time,
                    image=image,
                    success=True
                ))
            
            # Register container
            self.active_containers[container.container_id] = container
            self.stats['containers_created'] += 1
            
            # Initial health check
            await self._perform_container_health_check(container)
            
            logger.info(f"Created secure container {name} with ID {container.container_id[:12]}")
            return container
            
        except Exception as e:
            logger.error(f"Failed to create container {name}: {e}")
            self.stats['executions_failed'] += 1
            raise RuntimeError(f"Container creation failed: {e}")
    
    async def _ensure_image_available(self, image: str) -> None:
        """Ensure Docker image is available locally."""
        try:
            self.docker_client.images.get(image)
            logger.debug(f"Image {image} already available locally")
        except docker.errors.ImageNotFound:
            logger.info(f"Pulling image {image}...")
            try:
                self.docker_client.images.pull(image)
                logger.info(f"Successfully pulled image {image}")
            except Exception as e:
                raise RuntimeError(f"Failed to pull image {image}: {e}")
    
    async def execute_in_container(
        self,
        container: SecureContainer,
        command: str,
        timeout: Optional[int] = None,
        capture_output: bool = True,
        working_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute command in container with monitoring and security checks."""
        
        if container.status != ContainerStatus.RUNNING:
            raise RuntimeError(f"Container {container.name} is not running (status: {container.status})")
        
        if not container.docker_container:
            raise RuntimeError(f"Docker container reference missing for {container.name}")
        
        execution_start = time.time()
        execution_id = f"exec_{int(execution_start)}_{uuid.uuid4().hex[:8]}"
        
        try:
            # Update container metrics before execution
            await self._update_container_metrics(container)
            
            # Prepare execution
            exec_config = {
                'cmd': ['/bin/bash', '-c', command],
                'stdout': capture_output,
                'stderr': capture_output,
                'stdin': False,
                'tty': False,
                'privileged': False,
                'user': '1000:1000',  # Non-root user
            }
            
            if working_dir:
                exec_config['workdir'] = working_dir
            
            # Create exec instance using the Docker API
            exec_instance = container.docker_container.client.api.exec_create(
                container=container.docker_container.id,
                cmd=['/bin/bash', '-c', command],
                stdout=capture_output,
                stderr=capture_output,
                stdin=False,
                tty=False,
                privileged=False,
                user='1000:1000',
                workdir=working_dir
            )
            
            # Start the exec instance with async execution
            exec_stream = container.docker_container.client.api.exec_start(
                exec_id=exec_instance['Id'],
                detach=False,
                stream=True,
                socket=False
            )
            
            # Wait for execution with timeout
            timeout = timeout or container.resource_limits.execution_timeout
            
            try:
                # Monitor execution
                result = await self._monitor_execution_stream(
                    container, exec_instance, exec_stream, execution_id, timeout
                )
                
                execution_time = time.time() - execution_start
                
                # Record execution in container history
                execution_data = {
                    'execution_id': execution_id,
                    'command': command[:100],  # Truncate long commands
                    'success': result['exit_code'] == 0,
                    'execution_time': execution_time,
                    'exit_code': result['exit_code'],
                    'timeout': timeout
                }
                
                container.add_execution(execution_data)
                
                if result['exit_code'] == 0:
                    self.stats['executions_successful'] += 1
                else:
                    self.stats['executions_failed'] += 1
                
                return {
                    'success': result['exit_code'] == 0,
                    'output': result['output'],
                    'error': result['error'],
                    'exit_code': result['exit_code'],
                    'execution_time': execution_time,
                    'execution_id': execution_id,
                    'resource_usage': await self._get_execution_resource_usage(container)
                }
                
            except asyncio.TimeoutError:
                # Handle timeout
                logger.warning(f"Execution timeout in container {container.name} after {timeout}s")
                
                # Try to kill the execution
                try:
                    # This is a simplification - in practice you'd need more sophisticated timeout handling
                    container.docker_container.exec_run(['pkill', '-f', command[:50]], user='1000:1000')
                except Exception as kill_error:
                    logger.warning(f"Failed to kill timed out execution: {kill_error}")
                
                execution_data = {
                    'execution_id': execution_id,
                    'command': command[:100],
                    'success': False,
                    'execution_time': timeout,
                    'exit_code': -1,
                    'timeout': True
                }
                container.add_execution(execution_data)
                self.stats['executions_failed'] += 1
                
                return {
                    'success': False,
                    'output': '',
                    'error': f'Execution timeout after {timeout} seconds',
                    'exit_code': -1,
                    'execution_time': timeout,
                    'execution_id': execution_id,
                    'timeout': True
                }
                
        except Exception as e:
            execution_time = time.time() - execution_start
            logger.error(f"Execution error in container {container.name}: {e}")
            
            execution_data = {
                'execution_id': execution_id,
                'command': command[:100],
                'success': False,
                'execution_time': execution_time,
                'error': str(e)
            }
            container.add_execution(execution_data)
            self.stats['executions_failed'] += 1
            
            return {
                'success': False,
                'output': '',
                'error': str(e),
                'exit_code': -1,
                'execution_time': execution_time,
                'execution_id': execution_id
            }
    
    async def _monitor_execution_stream(
        self,
        container: SecureContainer,
        exec_instance: Any,
        exec_stream: Any,
        execution_id: str,
        timeout: int
    ) -> Dict[str, Any]:
        """Monitor command execution with streaming output and timeout."""
        
        start_time = time.time()
        output_chunks = []
        
        try:
            # Collect output with timeout
            while True:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    raise asyncio.TimeoutError()
                
                # Check if execution is still running
                try:
                    exec_info = container.docker_container.client.api.exec_inspect(exec_instance['Id'])
                    if not exec_info['Running']:
                        # Execution completed, collect any remaining output
                        try:
                            for chunk in exec_stream:
                                if isinstance(chunk, bytes):
                                    output_chunks.append(chunk.decode('utf-8', errors='ignore'))
                                else:
                                    output_chunks.append(str(chunk))
                        except Exception:
                            pass  # Stream might be closed
                        
                        # Combine all output
                        full_output = ''.join(output_chunks)
                        
                        # Separate stdout and stderr (simplified)
                        if 'Error:' in full_output or 'Traceback' in full_output:
                            error = full_output
                            output = ''
                        else:
                            error = ''
                            output = full_output
                        
                        return {
                            'output': output.strip(),
                            'error': error.strip(),
                            'exit_code': exec_info['ExitCode']
                        }
                    
                    # Check resource usage during execution
                    await self._update_container_metrics(container)
                    
                    # Check for resource violations
                    if container.metrics.memory_usage_percent > 95:
                        logger.warning(f"Memory usage critical in container {container.name}: {container.metrics.memory_usage_percent:.1f}%")
                        self.stats['resource_violations'] += 1
                    
                    if container.metrics.cpu_usage_percent > 95:
                        logger.warning(f"CPU usage critical in container {container.name}: {container.metrics.cpu_usage_percent:.1f}%")
                        self.stats['resource_violations'] += 1
                    
                    await asyncio.sleep(0.1)  # Short polling interval
                    
                except Exception as e:
                    logger.error(f"Error monitoring execution {execution_id}: {e}")
                    return {
                        'output': '',
                        'error': f'Monitoring error: {e}',
                        'exit_code': -1
                    }
                
        except asyncio.TimeoutError:
            # Try to get partial output
            full_output = ''.join(output_chunks)
            return {
                'output': full_output.strip(),
                'error': 'Execution timed out',
                'exit_code': -1
            }
        except Exception as e:
            logger.error(f"Error in execution stream {execution_id}: {e}")
            return {
                'output': '',
                'error': f'Stream error: {e}',
                'exit_code': -1
            }
    
    async def _update_container_metrics(self, container: SecureContainer) -> None:
        """Update container resource usage metrics."""
        try:
            if not container.docker_container:
                return
            
            # Get container stats
            stats = container.docker_container.stats(stream=False)
            
            # Parse memory stats
            memory_stats = stats.get('memory', {})
            memory_usage = memory_stats.get('usage', 0)
            memory_limit = memory_stats.get('limit', 0)
            
            container.metrics.memory_usage_mb = memory_usage / 1024 / 1024
            container.metrics.memory_limit_mb = memory_limit / 1024 / 1024
            
            # Parse CPU stats
            cpu_stats = stats.get('cpu_stats', {})
            precpu_stats = stats.get('precpu_stats', {})
            
            if cpu_stats and precpu_stats:
                cpu_usage_delta = cpu_stats.get('cpu_usage', {}).get('total_usage', 0) - precpu_stats.get('cpu_usage', {}).get('total_usage', 0)
                system_cpu_delta = cpu_stats.get('system_cpu_usage', 0) - precpu_stats.get('system_cpu_usage', 0)
                
                if system_cpu_delta > 0 and cpu_usage_delta >= 0:
                    num_cpus = len(cpu_stats.get('cpu_usage', {}).get('percpu_usage', [1]))
                    raw_cpu_percent = (cpu_usage_delta / system_cpu_delta) * num_cpus * 100
                    # Cap CPU usage at reasonable maximum to handle calculation errors
                    container.metrics.cpu_usage_percent = min(raw_cpu_percent, 200.0)
            
            # Parse network stats
            networks = stats.get('networks', {})
            if networks and isinstance(networks, dict):
                total_rx = sum(net.get('rx_bytes', 0) for net in networks.values() if isinstance(net, dict))
                total_tx = sum(net.get('tx_bytes', 0) for net in networks.values() if isinstance(net, dict))
                container.metrics.network_rx_bytes = total_rx
                container.metrics.network_tx_bytes = total_tx
            
            # Parse PIDs stats
            pids_stats = stats.get('pids', {})
            container.metrics.pids_current = pids_stats.get('current', 0)
            container.metrics.pids_limit = pids_stats.get('limit', 0)
            
            # Update uptime
            container.metrics.uptime_seconds = container.age_seconds
            
            # Store metrics for manager tracking
            self.container_metrics[container.container_id] = container.metrics
            
        except Exception as e:
            logger.warning(f"Failed to update metrics for container {container.name}: {e}")
    
    async def _get_execution_resource_usage(self, container: SecureContainer) -> Dict[str, Any]:
        """Get resource usage for last execution."""
        return {
            'cpu_usage_percent': container.metrics.cpu_usage_percent,
            'memory_usage_mb': container.metrics.memory_usage_mb,
            'memory_usage_percent': container.metrics.memory_usage_percent,
            'network_rx_bytes': container.metrics.network_rx_bytes,
            'network_tx_bytes': container.metrics.network_tx_bytes,
            'pids_current': container.metrics.pids_current,
            'uptime_seconds': container.metrics.uptime_seconds
        }
    
    async def _perform_container_health_check(self, container: SecureContainer) -> Dict[str, Any]:
        """Perform comprehensive health check on container."""
        health_data = {
            'timestamp': datetime.now().isoformat(),
            'container_responsive': False,
            'resource_usage_normal': False,
            'no_security_violations': False,
            'overall_healthy': False
        }
        
        try:
            # Check if container is responsive
            response = await self.execute_in_container(
                container, 
                'echo "health_check"', 
                timeout=5,
                capture_output=True
            )
            health_data['container_responsive'] = response['success'] and 'health_check' in response.get('output', '')
            
            # Update and check resource usage
            await self._update_container_metrics(container)
            
            # Be more lenient for new containers (less than 30 seconds old)
            is_new_container = container.age_seconds < 30
            cpu_threshold = 95 if is_new_container else 80
            memory_threshold = 90 if is_new_container else 80
            
            # Handle edge cases in metrics
            cpu_usage = container.metrics.cpu_usage_percent
            memory_usage = container.metrics.memory_usage_percent
            pids_usage = container.metrics.pids_current
            pids_limit = container.metrics.pids_limit
            
            # Validate CPU usage (cap at reasonable maximum)
            if cpu_usage > 200:  # Unreasonable value, likely calculation error
                cpu_usage = 0
            
            health_data['resource_usage_normal'] = (
                memory_usage < memory_threshold and
                cpu_usage < cpu_threshold and
                (pids_limit == 0 or pids_usage < pids_limit * 0.8)
            )
            
            # Check for security violations (simplified check)
            health_data['no_security_violations'] = len([
                exec_data for exec_data in container.execution_history[-10:]
                if not exec_data.get('success', True)
            ]) < 3
            
            # Overall health assessment
            health_data['overall_healthy'] = (
                health_data['container_responsive'] and
                health_data['resource_usage_normal'] and
                health_data['no_security_violations']
            )
            
            container.add_health_check(health_data)
            
            if health_data['overall_healthy']:
                logger.debug(f"Container {container.name} health check: HEALTHY")
            else:
                logger.warning(f"Container {container.name} health check: UNHEALTHY - {health_data}")
            
            return health_data
            
        except Exception as e:
            logger.error(f"Health check failed for container {container.name}: {e}")
            health_data['error'] = str(e)
            container.add_health_check(health_data)
            return health_data
    
    async def return_container_to_pool(
        self, 
        container: SecureContainer, 
        execution_id: Optional[str] = None,
        execution_time: float = 0.0,
        execution_successful: bool = True,
        force_destroy: bool = False
    ) -> bool:
        """Return container to pool or destroy if pooling not enabled."""
        try:
            # Try to return to advanced pool first
            if not force_destroy and self.pool_manager:
                await self.pool_manager.return_container(
                    container=container,
                    execution_id=execution_id,
                    execution_time=execution_time,
                    execution_successful=execution_successful
                )
                return True
            
            # Fallback to simple pool or destroy
            return await self.destroy_container(container, force=force_destroy)
            
        except Exception as e:
            logger.error(f"Failed to return container to pool: {e}")
            return await self.destroy_container(container, force=True)

    async def destroy_container(self, container: SecureContainer, force: bool = False) -> bool:
        """Destroy container and clean up resources."""
        try:
            # Remove from active containers
            if container.container_id in self.active_containers:
                del self.active_containers[container.container_id]
            
            # Remove metrics tracking
            if container.container_id in self.container_metrics:
                del self.container_metrics[container.container_id]
            
            if container.docker_container:
                try:
                    # Stop container gracefully first
                    if not force:
                        container.docker_container.stop(timeout=10)
                    
                    # Remove container
                    container.docker_container.remove(force=force)
                    
                    container.status = ContainerStatus.DESTROYED
                    self.stats['containers_destroyed'] += 1
                    
                    logger.info(f"Destroyed container {container.name}")
                    return True
                    
                except docker.errors.NotFound:
                    # Container already removed
                    container.status = ContainerStatus.DESTROYED
                    logger.info(f"Container {container.name} already removed")
                    return True
                    
                except Exception as e:
                    logger.error(f"Error destroying container {container.name}: {e}")
                    if force:
                        # Try force removal one more time
                        try:
                            container.docker_container.remove(force=True)
                            container.status = ContainerStatus.DESTROYED
                            return True
                        except Exception as force_error:
                            logger.error(f"Force destruction failed for {container.name}: {force_error}")
                    return False
            
            container.status = ContainerStatus.DESTROYED
            return True
            
        except Exception as e:
            logger.error(f"Unexpected error destroying container {container.name}: {e}")
            return False
    
    
    def get_container_by_id(self, container_id: str) -> Optional[SecureContainer]:
        """Get container by ID."""
        return self.active_containers.get(container_id)
    
    def list_active_containers(self) -> List[SecureContainer]:
        """List all active containers."""
        return list(self.active_containers.values())
    
    def get_manager_statistics(self) -> Dict[str, Any]:
        """Get comprehensive manager statistics."""
        pool_stats = self.container_pool.get_statistics() if self.container_pool else {}
        
        return {
            'manager_stats': self.stats.copy(),
            'active_containers': len(self.active_containers),
            'tracked_metrics': len(self.container_metrics),
            'pool_stats': pool_stats,
            'docker_info': self._get_docker_info()
        }
    
    def _get_docker_info(self) -> Dict[str, Any]:
        """Get Docker system information."""
        try:
            if self.docker_client:
                info = self.docker_client.info()
                return {
                    'version': info.get('ServerVersion', 'Unknown'),
                    'containers': info.get('Containers', 0),
                    'containers_running': info.get('ContainersRunning', 0),
                    'images': info.get('Images', 0),
                    'memory_total': info.get('MemTotal', 0),
                    'cpu_cores': info.get('NCPU', 0),
                    'security_options': info.get('SecurityOptions', [])
                }
        except Exception as e:
            logger.warning(f"Failed to get Docker info: {e}")
        
        return {'error': 'Docker info unavailable'}
    
    async def start_background_tasks(self):
        """Start background monitoring tasks."""
        # Initialize and start advanced pool manager
        if self.enable_advanced_pooling and self.pool_manager is None:
            from .container_pool import ContainerPoolManager
            self.pool_manager = ContainerPoolManager(self, self.pool_config)
            await self.pool_manager.start_background_tasks()
        elif self.container_pool:
            await self.container_pool.start_cleanup_task()
        
        if self._metrics_task is None:
            self._metrics_task = asyncio.create_task(self._metrics_update_loop())
            logger.info("Started Docker manager background tasks")
    
    async def stop_background_tasks(self):
        """Stop background monitoring tasks."""
        # Stop advanced pool manager
        if self.pool_manager:
            await self.pool_manager.stop_background_tasks()
        elif self.container_pool:
            await self.container_pool.stop_cleanup_task()
        
        if self._metrics_task:
            self._metrics_task.cancel()
            try:
                await self._metrics_task
            except asyncio.CancelledError:
                pass
            self._metrics_task = None
            logger.info("Stopped Docker manager background tasks")
    
    async def _metrics_update_loop(self):
        """Background loop for updating container metrics."""
        while True:
            try:
                await asyncio.sleep(30)  # Update every 30 seconds
                
                # Update metrics for all active containers
                for container in list(self.active_containers.values()):
                    try:
                        await self._update_container_metrics(container)
                        
                        # Perform periodic health checks
                        if container.age_seconds % 300 == 0:  # Every 5 minutes
                            await self._perform_container_health_check(container)
                            
                    except Exception as e:
                        logger.warning(f"Metrics update failed for container {container.name}: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics update loop: {e}")
    
    async def cleanup_all_containers(self, force: bool = False):
        """Clean up all managed containers."""
        logger.info("Cleaning up all containers...")
        
        containers_to_cleanup = list(self.active_containers.values())
        
        for container in containers_to_cleanup:
            try:
                await self.destroy_container(container, force=force)
            except Exception as e:
                logger.error(f"Error cleaning up container {container.name}: {e}")
        
        self.active_containers.clear()
        self.container_metrics.clear()
        
        logger.info(f"Cleaned up {len(containers_to_cleanup)} containers")
    
    async def shutdown(self):
        """Shutdown Docker manager and clean up all resources."""
        logger.info("Shutting down EnhancedDockerManager...")
        
        await self.stop_background_tasks()
        
        # Shutdown advanced pool manager
        if self.pool_manager:
            await self.pool_manager.shutdown()
        
        await self.cleanup_all_containers(force=True)
        
        if self.docker_client:
            self.docker_client.close()
        
        logger.info("EnhancedDockerManager shutdown complete")


# Utility functions
def create_default_resource_limits() -> ResourceLimits:
    """Create default resource limits."""
    return ResourceLimits()


def create_strict_security_config() -> SecurityConfig:
    """Create strict security configuration."""
    return SecurityConfig(
        read_only_root=True,
        no_new_privileges=True,
        drop_all_capabilities=True,
        allowed_capabilities=[],  # No capabilities
        user_namespace=True,
        network_isolation=True
    )


def create_moderate_security_config() -> SecurityConfig:
    """Create moderate security configuration."""
    return SecurityConfig()  # Uses defaults


# Export main classes
__all__ = [
    'EnhancedDockerManager',
    'SecureContainer',
    'ContainerPool',
    'ResourceLimits',
    'SecurityConfig',
    'ContainerMetrics',
    'ContainerStatus',
    'SandboxingLevel',
    'create_default_resource_limits',
    'create_strict_security_config',
    'create_moderate_security_config'
]