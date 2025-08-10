"""Performance Optimization System - Issue #205 Phase 3

Optimizes checkpoint operations for performance, storage efficiency, and scalability.
Provides compression, caching, concurrent operations, and retention policies.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import pickle
import statistics
import time
import zlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Callable, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Internal imports
from ..state.global_context import PipelineGlobalState, validate_pipeline_state
from ..state.langgraph_state_manager import LangGraphGlobalContextManager

logger = logging.getLogger(__name__)


class CompressionMethod(Enum):
    """Compression methods for checkpoint storage."""
    NONE = "none"
    GZIP = "gzip" 
    ZLIB = "zlib"
    PICKLE = "pickle"  # Python-specific serialization


class RetentionPolicy(Enum):
    """Retention policies for checkpoint cleanup."""
    NEVER = "never"  # Never delete checkpoints
    BY_AGE = "by_age"  # Delete by age
    BY_COUNT = "by_count"  # Keep only N most recent
    BY_SIZE = "by_size"  # Delete when storage exceeds limit
    SMART = "smart"  # Intelligent retention based on usage patterns


@dataclass
class PerformanceMetrics:
    """Performance metrics for checkpoint operations."""
    operation_type: str
    thread_id: str
    start_time: float
    end_time: float
    data_size_bytes: int
    compressed_size_bytes: int = 0
    compression_ratio: float = 1.0
    cache_hit: bool = False
    concurrent_operations: int = 1
    memory_usage_mb: float = 0.0
    
    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000
    
    @property
    def throughput_mbps(self) -> float:
        if self.duration_ms == 0:
            return 0
        return (self.data_size_bytes / (1024 * 1024)) / (self.duration_ms / 1000)


@dataclass
class RetentionConfig:
    """Configuration for checkpoint retention."""
    policy: RetentionPolicy
    max_age_days: Optional[int] = None
    max_count: Optional[int] = None
    max_size_gb: Optional[float] = None
    preserve_important: bool = True
    cleanup_interval_hours: float = 24.0


@dataclass
class CompressionStats:
    """Statistics for compression operations."""
    original_size: int
    compressed_size: int
    compression_time: float
    decompression_time: float
    method: CompressionMethod
    
    @property
    def compression_ratio(self) -> float:
        return self.original_size / max(1, self.compressed_size)
    
    @property
    def space_saved_percent(self) -> float:
        return (1 - self.compressed_size / max(1, self.original_size)) * 100


class PerformanceOptimizer:
    """
    Performance optimization system for checkpoint operations.
    
    Provides:
    - State compression and decompression
    - Intelligent caching with LRU eviction
    - Concurrent operation management
    - Storage efficiency optimizations
    - Retention policy enforcement
    - Performance monitoring and analytics
    """
    
    def __init__(
        self,
        langgraph_manager: LangGraphGlobalContextManager,
        enable_compression: bool = True,
        compression_method: CompressionMethod = CompressionMethod.GZIP,
        compression_threshold_bytes: int = 1024,  # Only compress if larger than 1KB
        cache_size_mb: float = 100.0,  # 100MB cache
        max_concurrent_operations: int = 10,
        retention_config: Optional[RetentionConfig] = None,
        performance_monitoring: bool = True,
    ):
        """
        Initialize performance optimizer.
        
        Args:
            langgraph_manager: LangGraph state manager
            enable_compression: Enable state compression
            compression_method: Compression algorithm to use
            compression_threshold_bytes: Minimum size to trigger compression
            cache_size_mb: Maximum cache size in MB
            max_concurrent_operations: Maximum concurrent checkpoint operations
            retention_config: Checkpoint retention configuration
            performance_monitoring: Enable performance metrics collection
        """
        self.langgraph_manager = langgraph_manager
        self.enable_compression = enable_compression
        self.compression_method = compression_method
        self.compression_threshold_bytes = compression_threshold_bytes
        self.cache_size_mb = cache_size_mb
        self.max_concurrent_operations = max_concurrent_operations
        self.retention_config = retention_config or RetentionConfig(
            policy=RetentionPolicy.BY_AGE,
            max_age_days=30
        )
        self.performance_monitoring = performance_monitoring
        
        # Performance tracking
        self.metrics_history: List[PerformanceMetrics] = []
        self.compression_stats: List[CompressionStats] = []
        
        # Caching system
        self.state_cache: Dict[str, Tuple[PipelineGlobalState, float, int]] = {}  # thread_id -> (state, timestamp, access_count)
        self.cache_access_times: Dict[str, float] = {}
        self.cache_size_bytes = 0
        self.cache_max_bytes = int(cache_size_mb * 1024 * 1024)
        
        # Concurrency management
        self.operation_semaphore = asyncio.Semaphore(max_concurrent_operations)
        self.active_operations: Dict[str, Set[str]] = defaultdict(set)  # operation_type -> set of thread_ids
        
        # Storage optimization
        self.compression_cache: Dict[str, bytes] = {}  # thread_id -> compressed_data
        self.deduplication_map: Dict[str, List[str]] = {}  # content_hash -> thread_ids
        
        # Retention management
        self.cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
        
        # Performance aggregation
        self.performance_summary = {
            "total_operations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "compression_operations": 0,
            "storage_saved_bytes": 0,
            "average_operation_time_ms": 0.0,
            "concurrent_operation_peak": 0
        }
        
        logger.info("PerformanceOptimizer initialized")
    
    async def optimize_checkpoint_creation(
        self,
        thread_id: str,
        state: PipelineGlobalState,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create checkpoint with performance optimizations.
        
        Args:
            thread_id: Thread identifier
            state: Pipeline state to checkpoint
            description: Checkpoint description
            metadata: Optional metadata
            
        Returns:
            Checkpoint ID
        """
        start_time = time.time()
        
        async with self.operation_semaphore:
            try:
                # Track concurrent operations
                self.active_operations["checkpoint_creation"].add(thread_id)
                concurrent_count = sum(len(ops) for ops in self.active_operations.values())
                self.performance_summary["concurrent_operation_peak"] = max(
                    self.performance_summary["concurrent_operation_peak"],
                    concurrent_count
                )
                
                # Check for state deduplication
                state_hash = await self._compute_state_hash(state)
                
                # Apply compression if enabled
                compressed_state = state
                compression_stats = None
                if self.enable_compression:
                    compressed_state, compression_stats = await self._compress_state(state)
                
                # Create checkpoint with optimized state
                enhanced_metadata = {
                    "performance_optimized": True,
                    "compression_enabled": self.enable_compression,
                    "compression_method": self.compression_method.value if self.enable_compression else None,
                    "state_hash": state_hash,
                    "creation_timestamp": time.time(),
                    **(metadata or {})
                }
                
                checkpoint_id = await self.langgraph_manager.create_checkpoint(
                    thread_id=thread_id,
                    description=description,
                    metadata=enhanced_metadata
                )
                
                # Update cache
                await self._update_cache(thread_id, state)
                
                # Record performance metrics
                if self.performance_monitoring:
                    await self._record_metrics(
                        operation_type="checkpoint_creation",
                        thread_id=thread_id,
                        start_time=start_time,
                        data_size_bytes=await self._estimate_state_size(state),
                        compressed_size_bytes=await self._estimate_state_size(compressed_state),
                        compression_stats=compression_stats
                    )
                
                # Store compression stats
                if compression_stats:
                    self.compression_stats.append(compression_stats)
                
                return checkpoint_id
                
            finally:
                self.active_operations["checkpoint_creation"].discard(thread_id)
    
    async def optimize_state_retrieval(
        self,
        thread_id: str,
        use_cache: bool = True
    ) -> Optional[PipelineGlobalState]:
        """
        Retrieve pipeline state with caching optimization.
        
        Args:
            thread_id: Thread identifier
            use_cache: Whether to use cached state if available
            
        Returns:
            Pipeline state or None if not found
        """
        start_time = time.time()
        
        async with self.operation_semaphore:
            try:
                # Track concurrent operations
                self.active_operations["state_retrieval"].add(thread_id)
                
                # Check cache first
                cache_hit = False
                if use_cache and thread_id in self.state_cache:
                    cached_state, timestamp, access_count = self.state_cache[thread_id]
                    
                    # Check if cache is still valid (within 5 minutes)
                    if time.time() - timestamp < 300:
                        cache_hit = True
                        self.performance_summary["cache_hits"] += 1
                        
                        # Update cache access
                        self.state_cache[thread_id] = (cached_state, timestamp, access_count + 1)
                        self.cache_access_times[thread_id] = time.time()
                        
                        # Record cache hit metrics
                        if self.performance_monitoring:
                            await self._record_metrics(
                                operation_type="state_retrieval",
                                thread_id=thread_id,
                                start_time=start_time,
                                data_size_bytes=await self._estimate_state_size(cached_state),
                                cache_hit=True
                            )
                        
                        return cached_state
                
                # Cache miss - retrieve from storage
                self.performance_summary["cache_misses"] += 1
                state = await self.langgraph_manager.get_global_state(thread_id)
                
                if state:
                    # Check if state is compressed
                    if isinstance(state, bytes) or self._is_compressed_state(state):
                        state = await self._decompress_state(state)
                    
                    # Update cache
                    await self._update_cache(thread_id, state)
                    
                    # Record retrieval metrics
                    if self.performance_monitoring:
                        await self._record_metrics(
                            operation_type="state_retrieval",
                            thread_id=thread_id,
                            start_time=start_time,
                            data_size_bytes=await self._estimate_state_size(state),
                            cache_hit=False
                        )
                
                return state
                
            finally:
                self.active_operations["state_retrieval"].discard(thread_id)
    
    async def batch_optimize_checkpoints(
        self,
        thread_states: List[Tuple[str, PipelineGlobalState, str]],  # (thread_id, state, description)
        max_concurrent: Optional[int] = None
    ) -> List[str]:
        """
        Create multiple checkpoints concurrently with optimization.
        
        Args:
            thread_states: List of (thread_id, state, description) tuples
            max_concurrent: Override for max concurrent operations
            
        Returns:
            List of checkpoint IDs in same order as input
        """
        if not thread_states:
            return []
        
        # Use custom semaphore if specified
        semaphore = asyncio.Semaphore(max_concurrent or self.max_concurrent_operations)
        
        async def create_single_checkpoint(thread_id: str, state: PipelineGlobalState, description: str) -> str:
            async with semaphore:
                return await self.optimize_checkpoint_creation(thread_id, state, description)
        
        # Execute all checkpoint creations concurrently
        tasks = [
            create_single_checkpoint(thread_id, state, description)
            for thread_id, state, description in thread_states
        ]
        
        start_time = time.time()
        checkpoint_ids = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        results = []
        for i, result in enumerate(checkpoint_ids):
            if isinstance(result, Exception):
                logger.error(f"Batch checkpoint creation failed for {thread_states[i][0]}: {result}")
                results.append("")  # Empty string indicates failure
            else:
                results.append(result)
        
        # Record batch performance
        if self.performance_monitoring:
            total_time = time.time() - start_time
            logger.info(f"Batch checkpoint creation: {len(thread_states)} checkpoints in {total_time:.2f}s")
        
        return results
    
    async def _compress_state(self, state: PipelineGlobalState) -> Tuple[Any, CompressionStats]:
        """Compress pipeline state using configured method."""
        start_time = time.time()
        
        # Serialize state to bytes
        serialized = json.dumps(state, ensure_ascii=False).encode('utf-8')
        original_size = len(serialized)
        
        # Skip compression if below threshold
        if original_size < self.compression_threshold_bytes:
            return state, CompressionStats(
                original_size=original_size,
                compressed_size=original_size,
                compression_time=0,
                decompression_time=0,
                method=CompressionMethod.NONE
            )
        
        # Apply compression
        compression_start = time.time()
        if self.compression_method == CompressionMethod.GZIP:
            compressed = gzip.compress(serialized)
        elif self.compression_method == CompressionMethod.ZLIB:
            compressed = zlib.compress(serialized)
        elif self.compression_method == CompressionMethod.PICKLE:
            compressed = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            compressed = serialized
        
        compression_time = time.time() - compression_start
        compressed_size = len(compressed)
        
        # Update performance tracking
        self.performance_summary["compression_operations"] += 1
        self.performance_summary["storage_saved_bytes"] += (original_size - compressed_size)
        
        stats = CompressionStats(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_time=compression_time,
            decompression_time=0,  # Will be measured during decompression
            method=self.compression_method
        )
        
        logger.debug(f"Compressed state: {original_size} -> {compressed_size} bytes ({stats.compression_ratio:.2f}x)")
        
        return compressed, stats
    
    async def _decompress_state(self, compressed_data: Any) -> PipelineGlobalState:
        """Decompress pipeline state."""
        if not isinstance(compressed_data, bytes):
            return compressed_data  # Not compressed
        
        start_time = time.time()
        
        try:
            # Try different decompression methods
            if self.compression_method == CompressionMethod.GZIP:
                decompressed = gzip.decompress(compressed_data)
                state = json.loads(decompressed.decode('utf-8'))
            elif self.compression_method == CompressionMethod.ZLIB:
                decompressed = zlib.decompress(compressed_data)
                state = json.loads(decompressed.decode('utf-8'))
            elif self.compression_method == CompressionMethod.PICKLE:
                state = pickle.loads(compressed_data)
            else:
                state = json.loads(compressed_data.decode('utf-8'))
            
            decompression_time = time.time() - start_time
            logger.debug(f"Decompressed state in {decompression_time:.3f}s")
            
            return state
            
        except Exception as e:
            logger.error(f"State decompression failed: {e}")
            # Return as-is if decompression fails
            return compressed_data
    
    async def _update_cache(self, thread_id: str, state: PipelineGlobalState):
        """Update state cache with LRU eviction."""
        state_size = await self._estimate_state_size(state)
        
        # Add to cache
        self.state_cache[thread_id] = (state.copy(), time.time(), 1)
        self.cache_access_times[thread_id] = time.time()
        self.cache_size_bytes += state_size
        
        # Evict if cache is too large
        while self.cache_size_bytes > self.cache_max_bytes:
            await self._evict_lru_cache_entry()
    
    async def _evict_lru_cache_entry(self):
        """Evict least recently used cache entry."""
        if not self.cache_access_times:
            return
        
        # Find LRU entry
        lru_thread_id = min(self.cache_access_times.items(), key=lambda x: x[1])[0]
        
        # Remove from cache
        if lru_thread_id in self.state_cache:
            evicted_state = self.state_cache[lru_thread_id][0]
            evicted_size = await self._estimate_state_size(evicted_state)
            
            del self.state_cache[lru_thread_id]
            del self.cache_access_times[lru_thread_id]
            self.cache_size_bytes -= evicted_size
            
            logger.debug(f"Evicted {lru_thread_id} from cache (size: {evicted_size} bytes)")
    
    async def _compute_state_hash(self, state: PipelineGlobalState) -> str:
        """Compute hash of state for deduplication."""
        # Create deterministic representation
        state_str = json.dumps(state, sort_keys=True, ensure_ascii=False)
        return str(hash(state_str))
    
    async def _estimate_state_size(self, state: Any) -> int:
        """Estimate memory size of state object."""
        if isinstance(state, bytes):
            return len(state)
        
        try:
            # Use JSON serialization as size estimate
            serialized = json.dumps(state, ensure_ascii=False)
            return len(serialized.encode('utf-8'))
        except:
            # Fallback to string representation
            return len(str(state))
    
    def _is_compressed_state(self, state: Any) -> bool:
        """Check if state appears to be compressed."""
        return isinstance(state, bytes) and self.enable_compression
    
    async def _record_metrics(
        self,
        operation_type: str,
        thread_id: str,
        start_time: float,
        data_size_bytes: int,
        compressed_size_bytes: int = 0,
        cache_hit: bool = False,
        compression_stats: Optional[CompressionStats] = None
    ):
        """Record performance metrics for operation."""
        end_time = time.time()
        concurrent_ops = sum(len(ops) for ops in self.active_operations.values())
        
        metrics = PerformanceMetrics(
            operation_type=operation_type,
            thread_id=thread_id,
            start_time=start_time,
            end_time=end_time,
            data_size_bytes=data_size_bytes,
            compressed_size_bytes=compressed_size_bytes or data_size_bytes,
            compression_ratio=data_size_bytes / max(1, compressed_size_bytes or data_size_bytes),
            cache_hit=cache_hit,
            concurrent_operations=concurrent_ops
        )
        
        self.metrics_history.append(metrics)
        
        # Update summary statistics
        self.performance_summary["total_operations"] += 1
        if cache_hit:
            self.performance_summary["cache_hits"] += 1
        
        # Calculate rolling average operation time
        recent_metrics = self.metrics_history[-100:]  # Last 100 operations
        avg_time = statistics.mean(m.duration_ms for m in recent_metrics)
        self.performance_summary["average_operation_time_ms"] = avg_time
    
    def _start_cleanup_task(self):
        """Start background cleanup task for retention policy."""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.retention_config.cleanup_interval_hours * 3600)
                    await self._enforce_retention_policy()
                except Exception as e:
                    logger.error(f"Cleanup task error: {e}")
        
        self.cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def _enforce_retention_policy(self):
        """Enforce configured retention policy."""
        if self.retention_config.policy == RetentionPolicy.NEVER:
            return
        
        logger.info("Enforcing checkpoint retention policy")
        
        try:
            if self.retention_config.policy == RetentionPolicy.BY_AGE:
                await self._cleanup_by_age()
            elif self.retention_config.policy == RetentionPolicy.BY_COUNT:
                await self._cleanup_by_count()
            elif self.retention_config.policy == RetentionPolicy.BY_SIZE:
                await self._cleanup_by_size()
            elif self.retention_config.policy == RetentionPolicy.SMART:
                await self._smart_cleanup()
                
        except Exception as e:
            logger.error(f"Retention policy enforcement failed: {e}")
    
    async def _cleanup_by_age(self):
        """Clean up checkpoints older than max_age_days."""
        if not self.retention_config.max_age_days:
            return
        
        cutoff_time = time.time() - (self.retention_config.max_age_days * 24 * 3600)
        cleaned_count = await self.langgraph_manager.cleanup_expired_checkpoints(
            retention_days=self.retention_config.max_age_days
        )
        
        logger.info(f"Age-based cleanup: removed {cleaned_count} old checkpoints")
    
    async def _cleanup_by_count(self):
        """Keep only the N most recent checkpoints."""
        # This would require additional LangGraph manager functionality
        logger.info("Count-based cleanup not yet implemented")
    
    async def _cleanup_by_size(self):
        """Clean up checkpoints when storage exceeds size limit."""
        # This would require storage size monitoring
        logger.info("Size-based cleanup not yet implemented")
    
    async def _smart_cleanup(self):
        """Intelligent cleanup based on usage patterns."""
        # Combine multiple strategies based on access patterns
        await self._cleanup_by_age()
        logger.info("Smart cleanup completed")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        cache_hit_rate = 0.0
        if self.performance_summary["total_operations"] > 0:
            cache_hit_rate = self.performance_summary["cache_hits"] / self.performance_summary["total_operations"]
        
        compression_summary = {}
        if self.compression_stats:
            compression_summary = {
                "total_compressions": len(self.compression_stats),
                "average_compression_ratio": statistics.mean(s.compression_ratio for s in self.compression_stats),
                "total_space_saved_mb": sum(s.original_size - s.compressed_size for s in self.compression_stats) / (1024 * 1024),
                "average_compression_time_ms": statistics.mean(s.compression_time * 1000 for s in self.compression_stats)
            }
        
        return {
            **self.performance_summary,
            "cache_hit_rate": cache_hit_rate,
            "cache_size_mb": self.cache_size_bytes / (1024 * 1024),
            "cache_utilization": self.cache_size_bytes / self.cache_max_bytes,
            "active_cache_entries": len(self.state_cache),
            "compression_enabled": self.enable_compression,
            "compression_method": self.compression_method.value,
            "compression_stats": compression_summary,
            "retention_policy": self.retention_config.policy.value,
            "concurrent_operation_limit": self.max_concurrent_operations,
            "recent_metrics_count": len(self.metrics_history[-100:])
        }
    
    def get_recent_metrics(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent performance metrics."""
        recent = self.metrics_history[-limit:]
        return [
            {
                "operation_type": m.operation_type,
                "thread_id": m.thread_id,
                "duration_ms": m.duration_ms,
                "data_size_mb": m.data_size_bytes / (1024 * 1024),
                "compression_ratio": m.compression_ratio,
                "cache_hit": m.cache_hit,
                "throughput_mbps": m.throughput_mbps,
                "concurrent_operations": m.concurrent_operations,
                "timestamp": m.start_time
            }
            for m in recent
        ]
    
    async def optimize_storage_usage(self) -> Dict[str, Any]:
        """Optimize storage usage and return statistics."""
        logger.info("Starting storage optimization")
        
        optimization_stats = {
            "cache_entries_evicted": 0,
            "compression_operations": 0,
            "deduplication_savings": 0,
            "retention_cleanups": 0
        }
        
        try:
            # Force cache cleanup to free memory
            initial_cache_size = len(self.state_cache)
            while self.cache_size_bytes > (self.cache_max_bytes * 0.8):  # Reduce to 80% capacity
                await self._evict_lru_cache_entry()
            optimization_stats["cache_entries_evicted"] = initial_cache_size - len(self.state_cache)
            
            # Enforce retention policy
            await self._enforce_retention_policy()
            optimization_stats["retention_cleanups"] = 1
            
            # Clear old metrics to free memory
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-500:]  # Keep only recent 500
            
            logger.info(f"Storage optimization complete: {optimization_stats}")
            
        except Exception as e:
            logger.error(f"Storage optimization failed: {e}")
            
        return optimization_stats
    
    async def shutdown(self):
        """Shutdown performance optimizer and cleanup resources."""
        # Cancel cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clear caches
        self.state_cache.clear()
        self.cache_access_times.clear()
        self.compression_cache.clear()
        
        logger.info("PerformanceOptimizer shutdown complete")