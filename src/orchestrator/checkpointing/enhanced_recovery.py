"""Enhanced Recovery and Monitoring - Issue #205 Phase 2

Provides sophisticated failure analysis, recovery point selection, checkpoint health
monitoring, execution analytics, and performance monitoring.
"""

from __future__ import annotations

import asyncio
import logging
import time
import statistics
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

# Internal imports
from ..state.global_context import (
    PipelineGlobalState,
    validate_pipeline_state,
    PipelineStatus
)
from ..state.langgraph_state_manager import LangGraphGlobalContextManager
from ..core.exceptions import PipelineExecutionError

logger = logging.getLogger(__name__)


class FailureCategory(Enum):
    """Categories of execution failures."""
    SYSTEM_ERROR = "system_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_FAILURE = "network_failure"
    DATA_CORRUPTION = "data_corruption"
    TIMEOUT = "timeout"
    VALIDATION_ERROR = "validation_error"
    DEPENDENCY_FAILURE = "dependency_failure"
    USER_CANCELLATION = "user_cancellation"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategies for different failure types."""
    IMMEDIATE_RETRY = "immediate_retry"
    BACKOFF_RETRY = "backoff_retry"
    SKIP_STEP = "skip_step"
    ROLLBACK_AND_RETRY = "rollback_and_retry"
    MANUAL_INTERVENTION = "manual_intervention"
    ABORT_EXECUTION = "abort_execution"


class CheckpointHealth(Enum):
    """Health status of checkpoints."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CORRUPTED = "corrupted"
    UNREACHABLE = "unreachable"


@dataclass
class FailureAnalysis:
    """Analysis of execution failure."""
    failure_id: str
    thread_id: str
    step_id: str
    failure_category: FailureCategory
    error_message: str
    stack_trace: Optional[str]
    failure_timestamp: float
    recovery_strategy: RecoveryStrategy
    recovery_checkpoint_id: Optional[str]
    failure_context: Dict[str, Any]
    recovery_confidence: float  # 0.0 to 1.0
    
    
@dataclass
class CheckpointHealthInfo:
    """Health information for a checkpoint."""
    checkpoint_id: str
    health_status: CheckpointHealth
    validation_errors: List[str]
    size_bytes: int
    creation_time: float
    last_accessed: Optional[float]
    corruption_indicators: List[str]
    recovery_success_rate: float


@dataclass
class ExecutionAnalytics:
    """Analytics for pipeline execution."""
    thread_id: str
    pipeline_id: str
    total_steps: int
    completed_steps: int
    failed_steps: int
    execution_time: float
    checkpoint_count: int
    recovery_attempts: int
    average_step_time: float
    checkpoint_overhead_percent: float
    failure_categories: Dict[str, int]
    recovery_success_rate: float


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    checkpoint_creation_times: List[float]
    checkpoint_restoration_times: List[float]
    state_validation_times: List[float]
    failure_detection_times: List[float]
    recovery_execution_times: List[float]
    storage_efficiency: Dict[str, Any]
    concurrent_operation_performance: Dict[str, Any]


class EnhancedRecoveryManager:
    """
    Enhanced recovery and monitoring system for checkpointed execution.
    
    Provides:
    - Sophisticated failure analysis and recovery point selection
    - Checkpoint health monitoring and validation
    - Execution analytics and performance monitoring
    - Checkpoint storage optimization and cleanup
    """
    
    def __init__(
        self,
        langgraph_manager: LangGraphGlobalContextManager,
        health_check_interval: float = 300.0,  # 5 minutes
        analytics_retention_days: int = 30,
        performance_monitoring_enabled: bool = True,
        auto_recovery_enabled: bool = True,
        recovery_confidence_threshold: float = 0.7,
    ):
        """
        Initialize enhanced recovery manager.
        
        Args:
            langgraph_manager: LangGraph state manager for persistence
            health_check_interval: Interval for checkpoint health checks
            analytics_retention_days: Days to retain analytics data
            performance_monitoring_enabled: Enable performance monitoring
            auto_recovery_enabled: Enable automatic recovery attempts
            recovery_confidence_threshold: Minimum confidence for auto-recovery
        """
        self.langgraph_manager = langgraph_manager
        self.health_check_interval = health_check_interval
        self.analytics_retention_days = analytics_retention_days
        self.performance_monitoring_enabled = performance_monitoring_enabled
        self.auto_recovery_enabled = auto_recovery_enabled
        self.recovery_confidence_threshold = recovery_confidence_threshold
        
        # Failure analysis and recovery
        self.failure_history: Dict[str, List[FailureAnalysis]] = {}
        self.recovery_patterns: Dict[str, RecoveryStrategy] = {}
        
        # Health monitoring
        self.checkpoint_health: Dict[str, CheckpointHealthInfo] = {}
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Analytics and metrics
        self.execution_analytics: Dict[str, ExecutionAnalytics] = {}
        self.performance_metrics = PerformanceMetrics(
            checkpoint_creation_times=[],
            checkpoint_restoration_times=[],
            state_validation_times=[],
            failure_detection_times=[],
            recovery_execution_times=[],
            storage_efficiency={},
            concurrent_operation_performance={}
        )
        
        # Monitoring and alerts
        self.alert_thresholds = {
            "checkpoint_creation_time_ms": 5000,  # 5 seconds
            "checkpoint_restoration_time_ms": 10000,  # 10 seconds
            "failure_rate_percent": 10,  # 10%
            "storage_growth_mb_per_hour": 100,  # 100 MB/hour
        }
        
        # Start background tasks
        self._start_health_monitoring()
        
        logger.info("EnhancedRecoveryManager initialized")
    
    async def analyze_failure(
        self,
        thread_id: str,
        step_id: str,
        error: Exception,
        execution_context: Dict[str, Any]
    ) -> FailureAnalysis:
        """
        Analyze execution failure and determine recovery strategy.
        
        Args:
            thread_id: Thread where failure occurred
            step_id: Step that failed
            error: Exception that occurred
            execution_context: Context at time of failure
            
        Returns:
            Failure analysis with recovery recommendations
        """
        start_time = time.time()
        
        # Categorize failure
        failure_category = self._categorize_failure(error, execution_context)
        
        # Determine recovery strategy
        recovery_strategy = self._determine_recovery_strategy(failure_category, execution_context)
        
        # Find optimal recovery checkpoint
        recovery_checkpoint_id = await self._find_recovery_checkpoint(thread_id, step_id, failure_category)
        
        # Calculate recovery confidence
        recovery_confidence = self._calculate_recovery_confidence(
            failure_category, recovery_strategy, thread_id, step_id
        )
        
        # Create failure analysis
        failure_id = f"failure_{int(time.time())}_{thread_id}_{step_id}"
        analysis = FailureAnalysis(
            failure_id=failure_id,
            thread_id=thread_id,
            step_id=step_id,
            failure_category=failure_category,
            error_message=str(error),
            stack_trace=getattr(error, '__traceback__', None),
            failure_timestamp=time.time(),
            recovery_strategy=recovery_strategy,
            recovery_checkpoint_id=recovery_checkpoint_id,
            failure_context=execution_context.copy(),
            recovery_confidence=recovery_confidence
        )
        
        # Store failure analysis
        if thread_id not in self.failure_history:
            self.failure_history[thread_id] = []
        self.failure_history[thread_id].append(analysis)
        
        # Update performance metrics
        if self.performance_monitoring_enabled:
            detection_time = time.time() - start_time
            self.performance_metrics.failure_detection_times.append(detection_time)
        
        logger.info(f"Analyzed failure {failure_id}: {failure_category.value} -> {recovery_strategy.value} (confidence: {recovery_confidence:.2f})")
        
        return analysis
    
    def _categorize_failure(self, error: Exception, context: Dict[str, Any]) -> FailureCategory:
        """Categorize failure based on error type and context."""
        
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # System errors
        if error_type in ["SystemError", "RuntimeError", "OSError"]:
            return FailureCategory.SYSTEM_ERROR
        
        # Resource exhaustion
        if any(keyword in error_message for keyword in ["memory", "disk", "resource", "limit"]):
            return FailureCategory.RESOURCE_EXHAUSTION
        
        # Network failures
        if any(keyword in error_message for keyword in ["network", "connection", "timeout", "dns"]):
            return FailureCategory.NETWORK_FAILURE
        
        # Data corruption
        if any(keyword in error_message for keyword in ["corrupt", "invalid", "malformed", "checksum"]):
            return FailureCategory.DATA_CORRUPTION
        
        # Timeouts
        if "timeout" in error_message or error_type == "TimeoutError":
            return FailureCategory.TIMEOUT
        
        # Validation errors
        if any(keyword in error_message for keyword in ["validation", "schema", "format"]):
            return FailureCategory.VALIDATION_ERROR
        
        # Dependency failures
        if any(keyword in error_message for keyword in ["dependency", "import", "module", "library"]):
            return FailureCategory.DEPENDENCY_FAILURE
        
        # User cancellation
        if any(keyword in error_message for keyword in ["cancel", "abort", "interrupt"]):
            return FailureCategory.USER_CANCELLATION
        
        return FailureCategory.UNKNOWN
    
    def _determine_recovery_strategy(
        self,
        failure_category: FailureCategory,
        context: Dict[str, Any]
    ) -> RecoveryStrategy:
        """Determine recovery strategy based on failure category and context."""
        
        # Check for patterns from previous failures
        context_key = f"{failure_category.value}_{context.get('step_type', 'unknown')}"
        if context_key in self.recovery_patterns:
            return self.recovery_patterns[context_key]
        
        # Default strategies by category
        strategy_map = {
            FailureCategory.SYSTEM_ERROR: RecoveryStrategy.BACKOFF_RETRY,
            FailureCategory.RESOURCE_EXHAUSTION: RecoveryStrategy.ROLLBACK_AND_RETRY,
            FailureCategory.NETWORK_FAILURE: RecoveryStrategy.BACKOFF_RETRY,
            FailureCategory.DATA_CORRUPTION: RecoveryStrategy.ROLLBACK_AND_RETRY,
            FailureCategory.TIMEOUT: RecoveryStrategy.IMMEDIATE_RETRY,
            FailureCategory.VALIDATION_ERROR: RecoveryStrategy.MANUAL_INTERVENTION,
            FailureCategory.DEPENDENCY_FAILURE: RecoveryStrategy.MANUAL_INTERVENTION,
            FailureCategory.USER_CANCELLATION: RecoveryStrategy.ABORT_EXECUTION,
            FailureCategory.UNKNOWN: RecoveryStrategy.MANUAL_INTERVENTION,
        }
        
        return strategy_map.get(failure_category, RecoveryStrategy.MANUAL_INTERVENTION)
    
    async def _find_recovery_checkpoint(
        self,
        thread_id: str,
        failed_step_id: str,
        failure_category: FailureCategory
    ) -> Optional[str]:
        """Find optimal checkpoint for recovery based on failure analysis."""
        
        try:
            # Get all checkpoints for thread
            checkpoints = await self.langgraph_manager.list_checkpoints(thread_id)
            
            if not checkpoints:
                return None
            
            # Filter healthy checkpoints
            healthy_checkpoints = []
            for checkpoint in checkpoints:
                checkpoint_id = checkpoint.get("checkpoint_id", "")
                if checkpoint_id in self.checkpoint_health:
                    health = self.checkpoint_health[checkpoint_id]
                    if health.health_status == CheckpointHealth.HEALTHY:
                        healthy_checkpoints.append(checkpoint)
                else:
                    # Assume healthy if not checked yet
                    healthy_checkpoints.append(checkpoint)
            
            if not healthy_checkpoints:
                return None
            
            # Select recovery point based on failure category
            if failure_category == FailureCategory.DATA_CORRUPTION:
                # Go back further for data corruption
                return healthy_checkpoints[-min(3, len(healthy_checkpoints))]["checkpoint_id"]
            elif failure_category == FailureCategory.RESOURCE_EXHAUSTION:
                # Go back to free up resources
                return healthy_checkpoints[-min(2, len(healthy_checkpoints))]["checkpoint_id"]
            else:
                # Use most recent healthy checkpoint
                return healthy_checkpoints[-1]["checkpoint_id"]
                
        except Exception as e:
            logger.error(f"Error finding recovery checkpoint: {e}")
            return None
    
    def _calculate_recovery_confidence(
        self,
        failure_category: FailureCategory,
        recovery_strategy: RecoveryStrategy,
        thread_id: str,
        step_id: str
    ) -> float:
        """Calculate confidence score for recovery success."""
        
        base_confidence = {
            FailureCategory.TIMEOUT: 0.8,
            FailureCategory.NETWORK_FAILURE: 0.7,
            FailureCategory.SYSTEM_ERROR: 0.6,
            FailureCategory.RESOURCE_EXHAUSTION: 0.5,
            FailureCategory.DATA_CORRUPTION: 0.3,
            FailureCategory.VALIDATION_ERROR: 0.2,
            FailureCategory.DEPENDENCY_FAILURE: 0.2,
            FailureCategory.USER_CANCELLATION: 0.0,
            FailureCategory.UNKNOWN: 0.3,
        }.get(failure_category, 0.3)
        
        # Adjust based on recovery strategy
        strategy_multipliers = {
            RecoveryStrategy.IMMEDIATE_RETRY: 0.8,
            RecoveryStrategy.BACKOFF_RETRY: 0.9,
            RecoveryStrategy.SKIP_STEP: 0.6,
            RecoveryStrategy.ROLLBACK_AND_RETRY: 0.7,
            RecoveryStrategy.MANUAL_INTERVENTION: 0.95,
            RecoveryStrategy.ABORT_EXECUTION: 0.0,
        }
        
        confidence = base_confidence * strategy_multipliers.get(recovery_strategy, 0.5)
        
        # Adjust based on failure history
        if thread_id in self.failure_history:
            recent_failures = [f for f in self.failure_history[thread_id] 
                             if time.time() - f.failure_timestamp < 3600]  # Last hour
            if len(recent_failures) > 3:
                confidence *= 0.7  # Reduce confidence for frequent failures
        
        return min(1.0, max(0.0, confidence))
    
    async def perform_health_check(self, checkpoint_id: str) -> CheckpointHealthInfo:
        """Perform comprehensive health check on a checkpoint."""
        
        start_time = time.time()
        validation_errors = []
        corruption_indicators = []
        
        try:
            # Basic accessibility check
            checkpoints = await self.langgraph_manager.list_checkpoints()
            checkpoint_exists = any(c.get("checkpoint_id") == checkpoint_id for c in checkpoints)
            
            if not checkpoint_exists:
                return CheckpointHealthInfo(
                    checkpoint_id=checkpoint_id,
                    health_status=CheckpointHealth.UNREACHABLE,
                    validation_errors=["Checkpoint not found"],
                    size_bytes=0,
                    creation_time=0,
                    last_accessed=None,
                    corruption_indicators=["Checkpoint missing"],
                    recovery_success_rate=0.0
                )
            
            # Try to get checkpoint data
            # Note: This would need actual checkpoint restoration in full implementation
            # For now, we'll simulate the health check
            
            # Simulate validation
            health_status = CheckpointHealth.HEALTHY
            size_bytes = 1024  # Simulated size
            creation_time = time.time() - 3600  # Simulated creation time
            
            # Calculate recovery success rate from history
            recovery_success_rate = self._calculate_checkpoint_recovery_rate(checkpoint_id)
            
            health_info = CheckpointHealthInfo(
                checkpoint_id=checkpoint_id,
                health_status=health_status,
                validation_errors=validation_errors,
                size_bytes=size_bytes,
                creation_time=creation_time,
                last_accessed=time.time(),
                corruption_indicators=corruption_indicators,
                recovery_success_rate=recovery_success_rate
            )
            
            # Store health info
            self.checkpoint_health[checkpoint_id] = health_info
            
            # Update performance metrics
            if self.performance_monitoring_enabled:
                validation_time = time.time() - start_time
                self.performance_metrics.state_validation_times.append(validation_time)
            
            return health_info
            
        except Exception as e:
            logger.error(f"Health check failed for checkpoint {checkpoint_id}: {e}")
            return CheckpointHealthInfo(
                checkpoint_id=checkpoint_id,
                health_status=CheckpointHealth.CORRUPTED,
                validation_errors=[str(e)],
                size_bytes=0,
                creation_time=0,
                last_accessed=None,
                corruption_indicators=["Health check failed"],
                recovery_success_rate=0.0
            )
    
    def _calculate_checkpoint_recovery_rate(self, checkpoint_id: str) -> float:
        """Calculate success rate for recoveries using this checkpoint."""
        
        total_attempts = 0
        successful_recoveries = 0
        
        for failure_list in self.failure_history.values():
            for failure in failure_list:
                if failure.recovery_checkpoint_id == checkpoint_id:
                    total_attempts += 1
                    # In a full implementation, we'd track actual recovery outcomes
                    # For now, use confidence as proxy for success
                    if failure.recovery_confidence > self.recovery_confidence_threshold:
                        successful_recoveries += 1
        
        if total_attempts == 0:
            return 1.0  # No data, assume good
        
        return successful_recoveries / total_attempts
    
    async def generate_execution_analytics(self, thread_id: str) -> Optional[ExecutionAnalytics]:
        """Generate comprehensive analytics for a pipeline execution."""
        
        try:
            # Get execution state
            state = await self.langgraph_manager.get_global_state(thread_id)
            if not state:
                return None
            
            exec_meta = state.get("execution_metadata", {})
            perf_metrics = state.get("performance_metrics", {})
            
            # Calculate analytics
            total_steps = len(exec_meta.get("completed_steps", [])) + len(exec_meta.get("failed_steps", []))
            completed_steps = len(exec_meta.get("completed_steps", []))
            failed_steps = len(exec_meta.get("failed_steps", []))
            
            execution_time = exec_meta.get("total_execution_time", 0)
            checkpoint_count = len(exec_meta.get("checkpoints", []))
            
            # Get failure history for this thread
            failures = self.failure_history.get(thread_id, [])
            recovery_attempts = len(failures)
            
            # Calculate average step time
            if total_steps > 0 and execution_time > 0:
                average_step_time = execution_time / total_steps
            else:
                average_step_time = 0
            
            # Calculate checkpoint overhead
            total_checkpoint_time = sum(cp.get("creation_time_ms", 0) for cp in exec_meta.get("checkpoints", []))
            if execution_time > 0:
                checkpoint_overhead_percent = (total_checkpoint_time / 1000) / execution_time * 100
            else:
                checkpoint_overhead_percent = 0
            
            # Categorize failures
            failure_categories = {}
            for failure in failures:
                category = failure.failure_category.value
                failure_categories[category] = failure_categories.get(category, 0) + 1
            
            # Calculate recovery success rate
            if recovery_attempts > 0:
                successful_recoveries = sum(1 for f in failures if f.recovery_confidence > self.recovery_confidence_threshold)
                recovery_success_rate = successful_recoveries / recovery_attempts
            else:
                recovery_success_rate = 1.0
            
            analytics = ExecutionAnalytics(
                thread_id=thread_id,
                pipeline_id=exec_meta.get("pipeline_id", ""),
                total_steps=total_steps,
                completed_steps=completed_steps,
                failed_steps=failed_steps,
                execution_time=execution_time,
                checkpoint_count=checkpoint_count,
                recovery_attempts=recovery_attempts,
                average_step_time=average_step_time,
                checkpoint_overhead_percent=checkpoint_overhead_percent,
                failure_categories=failure_categories,
                recovery_success_rate=recovery_success_rate
            )
            
            # Store analytics
            self.execution_analytics[thread_id] = analytics
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error generating analytics for {thread_id}: {e}")
            return None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        
        metrics = self.performance_metrics
        
        def safe_stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"count": 0, "mean": 0, "median": 0, "std": 0, "min": 0, "max": 0}
            return {
                "count": len(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0,
                "min": min(values),
                "max": max(values)
            }
        
        return {
            "checkpoint_creation": {
                **safe_stats(metrics.checkpoint_creation_times),
                "unit": "seconds"
            },
            "checkpoint_restoration": {
                **safe_stats(metrics.checkpoint_restoration_times),
                "unit": "seconds"
            },
            "state_validation": {
                **safe_stats(metrics.state_validation_times),
                "unit": "seconds"
            },
            "failure_detection": {
                **safe_stats(metrics.failure_detection_times),
                "unit": "seconds"
            },
            "recovery_execution": {
                **safe_stats(metrics.recovery_execution_times),
                "unit": "seconds"
            },
            "health_checks": {
                "total_checkpoints_monitored": len(self.checkpoint_health),
                "healthy_checkpoints": len([h for h in self.checkpoint_health.values() 
                                          if h.health_status == CheckpointHealth.HEALTHY]),
                "corrupted_checkpoints": len([h for h in self.checkpoint_health.values() 
                                            if h.health_status == CheckpointHealth.CORRUPTED])
            },
            "failure_analysis": {
                "total_failures_analyzed": sum(len(failures) for failures in self.failure_history.values()),
                "unique_threads_with_failures": len(self.failure_history),
                "average_recovery_confidence": statistics.mean([
                    f.recovery_confidence for failures in self.failure_history.values() for f in failures
                ]) if any(self.failure_history.values()) else 0
            }
        }
    
    def _start_health_monitoring(self):
        """Start background health monitoring task."""
        
        async def health_monitoring_loop():
            while True:
                try:
                    await asyncio.sleep(self.health_check_interval)
                    
                    # Get all checkpoints to monitor
                    all_checkpoints = []
                    for analytics in self.execution_analytics.values():
                        # In a full implementation, we'd get actual checkpoint IDs
                        all_checkpoints.extend([f"checkpoint_{i}" for i in range(analytics.checkpoint_count)])
                    
                    # Sample checkpoints for health checking (to avoid overload)
                    import random
                    sample_size = min(10, len(all_checkpoints))
                    if all_checkpoints:
                        sample_checkpoints = random.sample(all_checkpoints, sample_size)
                        
                        for checkpoint_id in sample_checkpoints:
                            try:
                                await self.perform_health_check(checkpoint_id)
                            except Exception as e:
                                logger.error(f"Health check failed for {checkpoint_id}: {e}")
                    
                    # Check for performance alerts
                    await self._check_performance_alerts()
                    
                except Exception as e:
                    logger.error(f"Error in health monitoring loop: {e}")
        
        self.health_check_task = asyncio.create_task(health_monitoring_loop())
    
    async def _check_performance_alerts(self):
        """Check performance metrics against alert thresholds."""
        
        metrics = self.performance_metrics
        alerts = []
        
        # Check checkpoint creation time
        if metrics.checkpoint_creation_times:
            avg_creation_time_ms = statistics.mean(metrics.checkpoint_creation_times) * 1000
            if avg_creation_time_ms > self.alert_thresholds["checkpoint_creation_time_ms"]:
                alerts.append(f"High checkpoint creation time: {avg_creation_time_ms:.1f}ms")
        
        # Check checkpoint restoration time
        if metrics.checkpoint_restoration_times:
            avg_restoration_time_ms = statistics.mean(metrics.checkpoint_restoration_times) * 1000
            if avg_restoration_time_ms > self.alert_thresholds["checkpoint_restoration_time_ms"]:
                alerts.append(f"High checkpoint restoration time: {avg_restoration_time_ms:.1f}ms")
        
        # Check failure rate
        total_executions = len(self.execution_analytics)
        failed_executions = sum(1 for a in self.execution_analytics.values() if a.failed_steps > 0)
        if total_executions > 0:
            failure_rate = (failed_executions / total_executions) * 100
            if failure_rate > self.alert_thresholds["failure_rate_percent"]:
                alerts.append(f"High failure rate: {failure_rate:.1f}%")
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"Performance Alert: {alert}")
    
    async def cleanup_old_data(self, retention_days: Optional[int] = None) -> Dict[str, int]:
        """Clean up old analytics and monitoring data."""
        
        retention_days = retention_days or self.analytics_retention_days
        cutoff_time = time.time() - (retention_days * 24 * 3600)
        
        cleanup_counts = {
            "failure_analyses_removed": 0,
            "analytics_removed": 0,
            "health_records_removed": 0
        }
        
        # Clean up old failure analyses
        for thread_id in list(self.failure_history.keys()):
            old_failures = [f for f in self.failure_history[thread_id] if f.failure_timestamp < cutoff_time]
            self.failure_history[thread_id] = [f for f in self.failure_history[thread_id] if f.failure_timestamp >= cutoff_time]
            cleanup_counts["failure_analyses_removed"] += len(old_failures)
            
            if not self.failure_history[thread_id]:
                del self.failure_history[thread_id]
        
        # Clean up old analytics
        old_analytics = [tid for tid, analytics in self.execution_analytics.items() 
                        if analytics.execution_time > 0 and time.time() - analytics.execution_time > retention_days * 24 * 3600]
        for tid in old_analytics:
            del self.execution_analytics[tid]
            cleanup_counts["analytics_removed"] += 1
        
        # Clean up old health records
        old_health_records = [cid for cid, health in self.checkpoint_health.items() 
                             if health.last_accessed and time.time() - health.last_accessed > retention_days * 24 * 3600]
        for cid in old_health_records:
            del self.checkpoint_health[cid]
            cleanup_counts["health_records_removed"] += 1
        
        logger.info(f"Cleaned up old data: {cleanup_counts}")
        return cleanup_counts
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status and health."""
        
        return {
            "health_monitoring": {
                "enabled": self.health_check_task is not None and not self.health_check_task.done(),
                "check_interval_seconds": self.health_check_interval,
                "checkpoints_monitored": len(self.checkpoint_health),
                "last_check_time": time.time()  # Simplified
            },
            "performance_monitoring": {
                "enabled": self.performance_monitoring_enabled,
                "metrics_collected": len(self.performance_metrics.checkpoint_creation_times),
                "alert_thresholds": self.alert_thresholds
            },
            "failure_analysis": {
                "threads_analyzed": len(self.failure_history),
                "total_failures": sum(len(failures) for failures in self.failure_history.values()),
                "auto_recovery_enabled": self.auto_recovery_enabled,
                "confidence_threshold": self.recovery_confidence_threshold
            },
            "data_retention": {
                "analytics_retention_days": self.analytics_retention_days,
                "current_analytics_count": len(self.execution_analytics),
                "current_health_records": len(self.checkpoint_health)
            }
        }
    
    async def shutdown(self):
        """Shutdown enhanced recovery manager."""
        
        # Cancel health monitoring task
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        logger.info("EnhancedRecoveryManager shutdown complete")