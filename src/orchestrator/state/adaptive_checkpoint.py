"""Adaptive checkpointing strategies for the Orchestrator framework."""

import asyncio
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from ..core.pipeline import Pipeline
from ..core.task import Task

if TYPE_CHECKING:
    from .state_manager import StateManager


class CheckpointTrigger(Enum):
    """Triggers for creating checkpoints."""

    TIME_BASED = "time_based"
    TASK_COMPLETION = "task_completion"
    ERROR_DETECTION = "error_detection"
    RESOURCE_USAGE = "resource_usage"
    MANUAL = "manual"
    PIPELINE_MILESTONE = "pipeline_milestone"


@dataclass
class CheckpointConfig:
    """Configuration for adaptive checkpointing."""

    max_checkpoints: int = 10
    min_interval: float = 60.0  # 1 minute
    max_interval: float = 3600.0  # 1 hour
    compression_enabled: bool = True
    cleanup_policy: str = "keep_latest"  # "keep_latest", "keep_all", "time_based"
    retention_days: int = 7

    # Adaptive parameters
    error_rate_threshold: float = (
        0.1  # 10% error rate triggers more frequent checkpoints
    )
    task_duration_multiplier: float = 2.0  # Checkpoint every N*average_task_duration
    memory_usage_threshold: float = 0.8  # 80% memory usage triggers checkpoint


@dataclass
class CheckpointMetrics:
    """Metrics for checkpoint decision making."""

    execution_time: float = 0.0
    completed_tasks: int = 0
    failed_tasks: int = 0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    last_checkpoint_time: float = 0.0
    checkpoint_count: int = 0
    average_task_duration: float = 0.0
    error_rate: float = 0.0
    progress_percentage: float = 0.0


class CheckpointStrategy(ABC):
    """Abstract base class for checkpoint strategies."""

    @abstractmethod
    def should_checkpoint(
        self,
        metrics: CheckpointMetrics,
        config: CheckpointConfig,
        trigger: CheckpointTrigger,
    ) -> bool:
        """Determine if a checkpoint should be created."""
        pass

    @abstractmethod
    def get_next_checkpoint_time(
        self, metrics: CheckpointMetrics, config: CheckpointConfig
    ) -> float:
        """Get time until next checkpoint."""
        pass


class TimeBasedStrategy(CheckpointStrategy):
    """Simple time-based checkpointing strategy."""

    def should_checkpoint(
        self,
        metrics: CheckpointMetrics,
        config: CheckpointConfig,
        trigger: CheckpointTrigger,
    ) -> bool:
        """Check if enough time has passed for a checkpoint."""
        if trigger == CheckpointTrigger.TIME_BASED:
            time_since_last = time.time() - metrics.last_checkpoint_time
            return time_since_last >= config.min_interval

        # Always checkpoint on errors
        if trigger == CheckpointTrigger.ERROR_DETECTION:
            return True

        return False

    def get_next_checkpoint_time(
        self, metrics: CheckpointMetrics, config: CheckpointConfig
    ) -> float:
        """Return fixed interval."""
        return config.min_interval


class AdaptiveStrategy(CheckpointStrategy):
    """Adaptive checkpointing strategy based on execution patterns."""

    def __init__(self, checkpoint_interval: float = 60.0):
        """Initialize adaptive strategy.

        Args:
            checkpoint_interval: Default checkpoint interval in seconds
        """
        self.checkpoint_interval = checkpoint_interval

    def should_checkpoint(
        self,
        metrics: CheckpointMetrics,
        config: CheckpointConfig,
        trigger: CheckpointTrigger,
    ) -> bool:
        """Adaptive checkpoint decision."""
        current_time = time.time()
        time_since_last = current_time - metrics.last_checkpoint_time

        # Always checkpoint on certain triggers
        if trigger in [CheckpointTrigger.ERROR_DETECTION, CheckpointTrigger.MANUAL]:
            return True

        # Don't checkpoint too frequently
        if time_since_last < config.min_interval:
            return False

        # High error rate - more frequent checkpoints
        if metrics.error_rate > config.error_rate_threshold:
            return time_since_last >= config.min_interval

        # High resource usage - create checkpoint for safety
        if (
            metrics.memory_usage > config.memory_usage_threshold
            or metrics.cpu_usage > 0.9
        ):
            return time_since_last >= config.min_interval * 0.5

        # Task completion milestones
        if trigger == CheckpointTrigger.TASK_COMPLETION:
            # Checkpoint every N completed tasks
            checkpoint_interval = max(5, int(10 * (1 - metrics.error_rate)))
            return metrics.completed_tasks % checkpoint_interval == 0

        # Progress-based checkpointing
        if trigger == CheckpointTrigger.PIPELINE_MILESTONE:
            # Checkpoint at 25%, 50%, 75%, 90% completion
            milestones = [0.25, 0.5, 0.75, 0.9]
            for milestone in milestones:
                if (
                    metrics.progress_percentage >= milestone
                    and metrics.progress_percentage < milestone + 0.05
                ):  # 5% tolerance
                    return True

        # Adaptive time-based
        if trigger == CheckpointTrigger.TIME_BASED:
            adaptive_interval = self._calculate_adaptive_interval(metrics, config)
            return time_since_last >= adaptive_interval

        return False

    def get_next_checkpoint_time(
        self, metrics: CheckpointMetrics, config: CheckpointConfig
    ) -> float:
        """Calculate adaptive checkpoint interval."""
        return self._calculate_adaptive_interval(metrics, config)

    def _calculate_adaptive_interval(
        self, metrics: CheckpointMetrics, config: CheckpointConfig
    ) -> float:
        """Calculate adaptive checkpoint interval based on metrics."""
        base_interval = config.min_interval

        # Adjust based on error rate
        error_factor = 1.0 - min(
            metrics.error_rate * 2, 0.8
        )  # More errors = shorter interval

        # Adjust based on task duration
        if metrics.average_task_duration > 0:
            duration_factor = (
                config.task_duration_multiplier * metrics.average_task_duration
            )
            duration_factor = min(duration_factor, config.max_interval)
        else:
            duration_factor = base_interval

        # Adjust based on progress
        progress_factor = 1.0 + (
            metrics.progress_percentage * 0.5
        )  # Longer intervals as we progress

        # Combine factors
        adaptive_interval = base_interval * error_factor * progress_factor
        adaptive_interval = min(adaptive_interval, duration_factor)

        # Ensure within bounds
        return max(config.min_interval, min(adaptive_interval, config.max_interval))


class ProgressBasedStrategy(CheckpointStrategy):
    """Progress-based checkpointing strategy."""

    def __init__(self, milestone_percentages: List[float] = None):
        self.milestones = milestone_percentages or [0.1, 0.25, 0.5, 0.75, 0.9]
        self.reached_milestones: Set[float] = set()

    def should_checkpoint(
        self,
        metrics: CheckpointMetrics,
        config: CheckpointConfig,
        trigger: CheckpointTrigger,
    ) -> bool:
        """Checkpoint at progress milestones."""
        # Always checkpoint on errors and manual triggers
        if trigger in [CheckpointTrigger.ERROR_DETECTION, CheckpointTrigger.MANUAL]:
            return True

        # Check progress milestones
        for milestone in self.milestones:
            if (
                milestone not in self.reached_milestones
                and metrics.progress_percentage >= milestone
            ):
                self.reached_milestones.add(milestone)
                return True

        # Fallback to time-based
        time_since_last = time.time() - metrics.last_checkpoint_time
        return time_since_last >= config.max_interval

    def get_next_checkpoint_time(
        self, metrics: CheckpointMetrics, config: CheckpointConfig
    ) -> float:
        """Time until next milestone or max interval."""
        next_milestone = None
        for milestone in self.milestones:
            if milestone > metrics.progress_percentage:
                next_milestone = milestone
                break

        if next_milestone:
            # Estimate time to next milestone
            if metrics.progress_percentage > 0:
                progress_rate = metrics.progress_percentage / metrics.execution_time
                remaining_progress = next_milestone - metrics.progress_percentage
                return (
                    remaining_progress / progress_rate
                    if progress_rate > 0
                    else config.max_interval
                )

        return config.max_interval


class AdaptiveCheckpointManager:
    """Manages adaptive checkpointing for pipeline executions."""

    def __init__(
        self,
        state_manager: "StateManager",
        config: CheckpointConfig = None,
        strategy: CheckpointStrategy = None,
    ):
        self.state_manager = state_manager
        self.config = config or CheckpointConfig()
        self.strategy = strategy or AdaptiveStrategy()

        # Execution tracking
        self.execution_metrics: Dict[str, CheckpointMetrics] = {}
        self.checkpoint_history: Dict[str, List[str]] = (
            {}
        )  # execution_id -> checkpoint_ids
        self.task_durations: Dict[str, deque] = {}  # execution_id -> task durations

        # Background checkpoint scheduler
        self._checkpoint_tasks: Dict[str, asyncio.Task] = {}
        self._shutdown = False

    async def start_execution(self, execution_id: str, pipeline: Pipeline):
        """Start tracking an execution for adaptive checkpointing."""
        metrics = CheckpointMetrics()
        metrics.last_checkpoint_time = time.time()
        self.execution_metrics[execution_id] = metrics
        self.checkpoint_history[execution_id] = []
        self.task_durations[execution_id] = deque(maxlen=100)

        # Start background checkpoint scheduler
        task = asyncio.create_task(self._checkpoint_scheduler(execution_id, pipeline))
        self._checkpoint_tasks[execution_id] = task

    async def stop_execution(self, execution_id: str):
        """Stop tracking an execution."""
        # Cancel background task
        if execution_id in self._checkpoint_tasks:
            self._checkpoint_tasks[execution_id].cancel()
            try:
                await self._checkpoint_tasks[execution_id]
            except asyncio.CancelledError:
                pass
            del self._checkpoint_tasks[execution_id]

        # Cleanup metrics
        if execution_id in self.execution_metrics:
            del self.execution_metrics[execution_id]
        if execution_id in self.checkpoint_history:
            del self.checkpoint_history[execution_id]
        if execution_id in self.task_durations:
            del self.task_durations[execution_id]

    async def update_metrics(
        self,
        execution_id: str,
        completed_tasks: int = 0,
        failed_tasks: int = 0,
        memory_usage: float = 0.0,
        cpu_usage: float = 0.0,
        progress_percentage: float = 0.0,
    ):
        """Update execution metrics."""
        if execution_id not in self.execution_metrics:
            return

        metrics = self.execution_metrics[execution_id]
        metrics.execution_time = time.time() - (
            metrics.last_checkpoint_time - metrics.execution_time
        )
        metrics.completed_tasks = completed_tasks
        metrics.failed_tasks = failed_tasks
        metrics.memory_usage = memory_usage
        metrics.cpu_usage = cpu_usage
        metrics.progress_percentage = progress_percentage

        # Calculate error rate
        total_tasks = completed_tasks + failed_tasks
        metrics.error_rate = failed_tasks / total_tasks if total_tasks > 0 else 0.0

        # Update average task duration
        if execution_id in self.task_durations and self.task_durations[execution_id]:
            durations = self.task_durations[execution_id]
            metrics.average_task_duration = sum(durations) / len(durations)

    async def record_task_completion(
        self, execution_id: str, task: Task, duration: float
    ):
        """Record task completion for metrics."""
        if execution_id in self.task_durations:
            self.task_durations[execution_id].append(duration)

        await self.update_metrics(execution_id)

        # Check if checkpoint should be created
        await self.check_checkpoint(execution_id, CheckpointTrigger.TASK_COMPLETION)

    async def check_checkpoint(
        self,
        execution_id: str,
        trigger: CheckpointTrigger,
        state: Dict[str, Any] = None,
    ) -> Optional[str]:
        """Check if a checkpoint should be created and create it if needed."""
        if execution_id not in self.execution_metrics:
            return None

        metrics = self.execution_metrics[execution_id]

        if self.strategy.should_checkpoint(metrics, self.config, trigger):
            return await self.create_checkpoint(execution_id, state, trigger)

        return None

    async def create_checkpoint(
        self,
        execution_id: str,
        state: Dict[str, Any] = None,
        trigger: CheckpointTrigger = CheckpointTrigger.MANUAL,
    ) -> str:
        """Create a checkpoint."""
        if execution_id not in self.execution_metrics:
            raise ValueError(f"Execution {execution_id} not tracked")

        metrics = self.execution_metrics[execution_id]

        # Create checkpoint
        metadata = {
            "trigger": trigger.value,
            "metrics": {
                "execution_time": metrics.execution_time,
                "completed_tasks": metrics.completed_tasks,
                "failed_tasks": metrics.failed_tasks,
                "progress_percentage": metrics.progress_percentage,
                "error_rate": metrics.error_rate,
            },
        }

        checkpoint_id = await self.state_manager.save_checkpoint(
            execution_id, state or {}, metadata
        )

        # Update metrics
        metrics.last_checkpoint_time = time.time()
        metrics.checkpoint_count += 1

        # Track checkpoint
        self.checkpoint_history[execution_id].append(checkpoint_id)

        # Cleanup old checkpoints if needed
        await self._cleanup_checkpoints(execution_id)

        return checkpoint_id

    async def _checkpoint_scheduler(self, execution_id: str, pipeline: Pipeline):
        """Background task for scheduled checkpointing."""
        while not self._shutdown and execution_id in self.execution_metrics:
            try:
                metrics = self.execution_metrics[execution_id]
                next_checkpoint_time = self.strategy.get_next_checkpoint_time(
                    metrics, self.config
                )

                # Wait for next checkpoint time
                await asyncio.sleep(next_checkpoint_time)

                # Check if checkpoint is needed
                await self.check_checkpoint(execution_id, CheckpointTrigger.TIME_BASED)

            except asyncio.CancelledError:
                break
            except Exception:
                # Log error and continue
                await asyncio.sleep(60)  # Wait before retrying

    async def _cleanup_checkpoints(self, execution_id: str):
        """Clean up old checkpoints based on cleanup policy."""
        if execution_id not in self.checkpoint_history:
            return

        checkpoints = self.checkpoint_history[execution_id]

        if self.config.cleanup_policy == "keep_latest":
            # Keep only the latest N checkpoints
            if len(checkpoints) > self.config.max_checkpoints:
                to_remove = checkpoints[: -self.config.max_checkpoints]
                for checkpoint_id in to_remove:
                    await self._remove_checkpoint(checkpoint_id)
                self.checkpoint_history[execution_id] = checkpoints[
                    -self.config.max_checkpoints :
                ]

        elif self.config.cleanup_policy == "time_based":
            # Remove checkpoints older than retention period
            current_time = time.time()
            retention_seconds = self.config.retention_days * 24 * 3600

            to_remove = []
            for checkpoint_id in checkpoints:
                # Extract timestamp from checkpoint_id (format: execution_id_timestamp)
                try:
                    timestamp = int(checkpoint_id.split("_")[-1])
                    if current_time - timestamp > retention_seconds:
                        to_remove.append(checkpoint_id)
                except (ValueError, IndexError):
                    continue

            for checkpoint_id in to_remove:
                await self._remove_checkpoint(checkpoint_id)
                checkpoints.remove(checkpoint_id)

    async def _remove_checkpoint(self, checkpoint_id: str):
        """Remove a checkpoint file."""
        import os

        filename = f"{checkpoint_id}.json"
        filepath = os.path.join(self.state_manager.storage_path, filename)

        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except OSError:
            pass  # Log error in production

    def get_execution_statistics(self, execution_id: str) -> Dict[str, Any]:
        """Get statistics for an execution."""
        if execution_id not in self.execution_metrics:
            return {}

        metrics = self.execution_metrics[execution_id]
        checkpoint_count = len(self.checkpoint_history.get(execution_id, []))

        return {
            "execution_id": execution_id,
            "execution_time": metrics.execution_time,
            "completed_tasks": metrics.completed_tasks,
            "failed_tasks": metrics.failed_tasks,
            "error_rate": metrics.error_rate,
            "progress_percentage": metrics.progress_percentage,
            "checkpoint_count": checkpoint_count,
            "last_checkpoint_time": metrics.last_checkpoint_time,
            "average_task_duration": metrics.average_task_duration,
            "memory_usage": metrics.memory_usage,
            "cpu_usage": metrics.cpu_usage,
            "strategy": type(self.strategy).__name__,
        }

    async def shutdown(self):
        """Shutdown the checkpoint manager."""
        self._shutdown = True

        # Cancel all background tasks
        for task in self._checkpoint_tasks.values():
            task.cancel()

        # Wait for tasks to complete
        if self._checkpoint_tasks:
            await asyncio.gather(
                *self._checkpoint_tasks.values(), return_exceptions=True
            )

        self._checkpoint_tasks.clear()


# Utility functions for creating common configurations


def create_development_config() -> CheckpointConfig:
    """Create checkpoint configuration suitable for development."""
    return CheckpointConfig(
        max_checkpoints=3,
        min_interval=30.0,  # 30 seconds
        max_interval=300.0,  # 5 minutes
        cleanup_policy="keep_latest",
        retention_days=1,
    )


def create_production_config() -> CheckpointConfig:
    """Create checkpoint configuration suitable for production."""
    return CheckpointConfig(
        max_checkpoints=10,
        min_interval=300.0,  # 5 minutes
        max_interval=1800.0,  # 30 minutes
        cleanup_policy="time_based",
        retention_days=7,
        error_rate_threshold=0.05,  # Lower threshold for production
    )


class AdaptiveCheckpointStrategy(CheckpointStrategy):
    """Advanced adaptive checkpointing strategy with task history and pattern matching."""

    def __init__(
        self,
        checkpoint_interval: int = 5,
        critical_task_patterns: List[str] = None,
        time_threshold_minutes: float = 10.0,
        failure_rate_threshold: float = 0.2,
    ):
        """Initialize adaptive checkpoint strategy.

        Args:
            checkpoint_interval: Number of tasks before creating checkpoint
            critical_task_patterns: Regex patterns for critical tasks
            time_threshold_minutes: Max time before forcing checkpoint
            failure_rate_threshold: Failure rate threshold for checkpointing
        """
        self.checkpoint_interval = checkpoint_interval
        self.critical_task_patterns = critical_task_patterns or []
        self.time_threshold_minutes = time_threshold_minutes
        self.failure_rate_threshold = failure_rate_threshold

        # Track task execution history per pipeline
        self.task_history: Dict[str, List[Dict[str, Any]]] = {}
        self.last_checkpoint_time: Dict[str, float] = {}

    def should_checkpoint(self, *args, **kwargs) -> bool:
        """Polymorphic should_checkpoint method that handles both interfaces."""
        # Handle the test interface: should_checkpoint(pipeline_id, task_id)
        if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], str):
            return self._should_checkpoint_for_task(args[0], args[1])

        # Handle the abstract interface: should_checkpoint(metrics, config, trigger)
        elif len(args) == 3:
            return self._should_checkpoint_with_metrics(args[0], args[1], args[2])

        # Handle keyword arguments
        if "pipeline_id" in kwargs and "task_id" in kwargs:
            return self._should_checkpoint_for_task(
                kwargs["pipeline_id"], kwargs["task_id"]
            )

        return False

    def _should_checkpoint_for_task(self, pipeline_id: str, task_id: str) -> bool:
        """Determine if checkpoint should be created for specific task."""
        import re

        # Track this task call (for interval counting)
        if pipeline_id not in self.task_history:
            self.task_history[pipeline_id] = []

        # Add a minimal record for counting purposes
        self.task_history[pipeline_id].append(
            {"task_id": task_id, "timestamp": time.time()}
        )

        task_count = len(self.task_history[pipeline_id])

        # Check if task matches critical patterns
        for pattern in self.critical_task_patterns:
            if re.match(pattern, task_id):
                return True

        # Check interval-based checkpointing
        if task_count % self.checkpoint_interval == 0:
            return True

        # Check time-based checkpointing
        current_time = time.time()
        if pipeline_id in self.last_checkpoint_time:
            time_since_last = current_time - self.last_checkpoint_time[pipeline_id]
            if time_since_last > (self.time_threshold_minutes * 60):
                return True

        # Check failure rate (for recorded executions with success/failure data)
        failure_rate = self._get_failure_rate(pipeline_id)
        if failure_rate > self.failure_rate_threshold:
            return True

        return False

    def record_task_execution(
        self, pipeline_id: str, task_id: str, success: bool, duration: float
    ) -> None:
        """Record task execution result."""
        if pipeline_id not in self.task_history:
            self.task_history[pipeline_id] = []

        execution_record = {
            "task_id": task_id,
            "success": success,
            "duration": duration,
            "timestamp": time.time(),
        }

        self.task_history[pipeline_id].append(execution_record)

    def get_checkpoint_priority(self, pipeline_id: str, task_id: str) -> float:
        """Calculate checkpoint priority for given task."""
        import re

        base_priority = 0.5  # Start with lower base priority for normal tasks

        # Higher priority for critical tasks
        is_critical = False
        for pattern in self.critical_task_patterns:
            if re.match(pattern, task_id):
                base_priority = 1.0  # Critical tasks get higher base priority
                is_critical = True
                break

        # Additional boost for critical tasks
        if is_critical:
            base_priority *= 1.5  # Results in 1.5 for critical tasks

        # Increase priority based on failure rate
        failure_rate = self._get_failure_rate(pipeline_id)
        base_priority *= 1.0 + failure_rate * 0.5  # Less aggressive multiplier

        # Increase priority based on time since last checkpoint
        if pipeline_id in self.last_checkpoint_time:
            time_factor = (time.time() - self.last_checkpoint_time[pipeline_id]) / (
                self.time_threshold_minutes * 60
            )
            base_priority *= max(1.0, time_factor * 0.5)  # Less aggressive time factor

        return min(base_priority, 2.0)  # Cap at 2.0

    def optimize_checkpoint_frequency(self, pipeline_id: str) -> None:
        """Optimize checkpoint frequency based on pipeline performance."""
        if (
            pipeline_id not in self.task_history
            or len(self.task_history[pipeline_id]) < 10
        ):
            return  # Need sufficient data

        failure_rate = self._get_failure_rate(pipeline_id)
        avg_duration = self._get_average_duration(pipeline_id)

        # Store original interval for comparison

        # Adjust interval based on stability
        if failure_rate < 0.05:  # Very stable (< 5% failure rate)
            self.checkpoint_interval = min(10, self.checkpoint_interval + 1)
        elif (
            failure_rate > 0.3
        ):  # High instability (> 30% failure rate) - be aggressive
            self.checkpoint_interval = max(
                1, self.checkpoint_interval - 2
            )  # Decrease by 2
        elif failure_rate > 0.2:  # Moderate instability (> 20% failure rate)
            self.checkpoint_interval = max(
                1, self.checkpoint_interval - 1
            )  # Decrease by 1
        elif failure_rate > 0.1:  # Low instability (> 10% failure rate)
            self.checkpoint_interval = max(2, self.checkpoint_interval - 1)

        # For very high failure rates, be even more aggressive
        if failure_rate > 0.5:  # > 50% failure rate
            self.checkpoint_interval = max(1, int(self.checkpoint_interval * 0.5))

        # Adjust time threshold based on task duration
        if avg_duration > 300:  # Long-running tasks (5+ minutes)
            self.time_threshold_minutes = max(5.0, self.time_threshold_minutes * 0.8)
        elif avg_duration < 30:  # Fast tasks
            self.time_threshold_minutes = min(30.0, self.time_threshold_minutes * 1.2)

    def get_statistics(self) -> Dict[str, Any]:
        """Get strategy performance statistics."""
        total_pipelines = len(self.task_history)
        total_executions = sum(len(history) for history in self.task_history.values())

        if total_executions == 0:
            return {
                "total_pipelines": total_pipelines,
                "total_executions": 0,
                "success_rate": 0.0,
                "average_duration": 0.0,
                "checkpoint_interval": self.checkpoint_interval,
                "checkpoint_frequency": self.checkpoint_interval,  # Add this alias
                "critical_patterns": len(self.critical_task_patterns),
            }

        # Calculate overall success rate
        total_successful = 0
        total_duration = 0.0

        for history in self.task_history.values():
            for record in history:
                if record["success"]:
                    total_successful += 1
                total_duration += record["duration"]

        success_rate = (
            total_successful / total_executions if total_executions > 0 else 0.0
        )
        avg_duration = (
            total_duration / total_executions if total_executions > 0 else 0.0
        )

        return {
            "total_pipelines": total_pipelines,
            "total_executions": total_executions,
            "success_rate": success_rate,
            "average_duration": avg_duration,
            "checkpoint_interval": self.checkpoint_interval,
            "checkpoint_frequency": self.checkpoint_interval,  # Add this alias
            "critical_patterns": len(self.critical_task_patterns),
        }

    def _get_failure_rate(self, pipeline_id: str) -> float:
        """Calculate failure rate for pipeline."""
        if pipeline_id not in self.task_history or not self.task_history[pipeline_id]:
            return 0.0

        history = self.task_history[pipeline_id]
        # Only count records that have success information
        records_with_status = [r for r in history if "success" in r]

        if not records_with_status:
            return 0.0

        failed_count = sum(1 for record in records_with_status if not record["success"])
        return failed_count / len(records_with_status)

    def _get_average_duration(self, pipeline_id: str) -> float:
        """Calculate average task duration for pipeline."""
        if pipeline_id not in self.task_history or not self.task_history[pipeline_id]:
            return 0.0

        history = self.task_history[pipeline_id]
        # Only count records that have duration information
        records_with_duration = [r for r in history if "duration" in r]

        if not records_with_duration:
            return 0.0

        total_duration = sum(record["duration"] for record in records_with_duration)
        return total_duration / len(records_with_duration)

    def _should_checkpoint_with_metrics(
        self,
        metrics: CheckpointMetrics,
        config: CheckpointConfig,
        trigger: CheckpointTrigger,
    ) -> bool:
        """Implement abstract method for compatibility."""
        current_time = time.time()
        time_since_last = current_time - metrics.last_checkpoint_time

        # Always checkpoint on certain triggers
        if trigger in [CheckpointTrigger.ERROR_DETECTION, CheckpointTrigger.MANUAL]:
            return True

        # Don't checkpoint too frequently
        if time_since_last < config.min_interval:
            return False

        # High error rate - more frequent checkpoints
        if metrics.error_rate > config.error_rate_threshold:
            return time_since_last >= config.min_interval

        # Task completion milestones
        if trigger == CheckpointTrigger.TASK_COMPLETION:
            return metrics.completed_tasks % self.checkpoint_interval == 0

        return False

    def get_next_checkpoint_time(
        self, metrics: CheckpointMetrics, config: CheckpointConfig
    ) -> float:
        """Calculate next checkpoint time."""
        current_time = time.time()

        # Base interval from config
        base_interval = (config.min_interval + config.max_interval) / 2

        # Adjust based on current checkpoint interval setting
        adjusted_interval = base_interval * (
            self.checkpoint_interval / 5.0
        )  # 5 is default

        # Adjust based on error rate
        if metrics.error_rate > self.failure_rate_threshold:
            adjusted_interval *= 0.5  # More frequent if error rate is high

        return current_time + adjusted_interval

    def get_checkpoint_metadata(
        self, metrics: CheckpointMetrics, config: CheckpointConfig
    ) -> Dict[str, Any]:
        """Get checkpoint metadata."""
        return {
            "strategy": "adaptive",
            "checkpoint_interval": self.checkpoint_interval,
            "failure_rate_threshold": self.failure_rate_threshold,
            "time_threshold_minutes": self.time_threshold_minutes,
            "critical_patterns_count": len(self.critical_task_patterns),
            "trigger": "adaptive_strategy",
        }

    def update_config(self, config: CheckpointConfig) -> None:
        """Update strategy configuration."""
        # Update thresholds from config if available
        if hasattr(config, "error_rate_threshold"):
            self.failure_rate_threshold = config.error_rate_threshold
