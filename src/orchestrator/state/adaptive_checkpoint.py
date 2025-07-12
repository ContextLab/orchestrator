"""Adaptive checkpointing strategies for the Orchestrator framework."""

import asyncio
import time
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Set, Callable
from collections import deque

from ..core.task import Task, TaskStatus
from ..core.pipeline import Pipeline


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
    error_rate_threshold: float = 0.1  # 10% error rate triggers more frequent checkpoints
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
    def should_checkpoint(self, 
                         metrics: CheckpointMetrics, 
                         config: CheckpointConfig,
                         trigger: CheckpointTrigger) -> bool:
        """Determine if a checkpoint should be created."""
        pass
    
    @abstractmethod
    def get_next_checkpoint_time(self, 
                               metrics: CheckpointMetrics, 
                               config: CheckpointConfig) -> float:
        """Get time until next checkpoint."""
        pass


class TimeBasedStrategy(CheckpointStrategy):
    """Simple time-based checkpointing strategy."""
    
    def should_checkpoint(self, 
                         metrics: CheckpointMetrics, 
                         config: CheckpointConfig,
                         trigger: CheckpointTrigger) -> bool:
        """Check if enough time has passed for a checkpoint."""
        if trigger == CheckpointTrigger.TIME_BASED:
            time_since_last = time.time() - metrics.last_checkpoint_time
            return time_since_last >= config.min_interval
        
        # Always checkpoint on errors
        if trigger == CheckpointTrigger.ERROR_DETECTION:
            return True
        
        return False
    
    def get_next_checkpoint_time(self, 
                               metrics: CheckpointMetrics, 
                               config: CheckpointConfig) -> float:
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
    
    def should_checkpoint(self, 
                         metrics: CheckpointMetrics, 
                         config: CheckpointConfig,
                         trigger: CheckpointTrigger) -> bool:
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
        if (metrics.memory_usage > config.memory_usage_threshold or 
            metrics.cpu_usage > 0.9):
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
                if (metrics.progress_percentage >= milestone and 
                    metrics.progress_percentage < milestone + 0.05):  # 5% tolerance
                    return True
        
        # Adaptive time-based
        if trigger == CheckpointTrigger.TIME_BASED:
            adaptive_interval = self._calculate_adaptive_interval(metrics, config)
            return time_since_last >= adaptive_interval
        
        return False
    
    def get_next_checkpoint_time(self, 
                               metrics: CheckpointMetrics, 
                               config: CheckpointConfig) -> float:
        """Calculate adaptive checkpoint interval."""
        return self._calculate_adaptive_interval(metrics, config)
    
    def _calculate_adaptive_interval(self, 
                                   metrics: CheckpointMetrics, 
                                   config: CheckpointConfig) -> float:
        """Calculate adaptive checkpoint interval based on metrics."""
        base_interval = config.min_interval
        
        # Adjust based on error rate
        error_factor = 1.0 - min(metrics.error_rate * 2, 0.8)  # More errors = shorter interval
        
        # Adjust based on task duration
        if metrics.average_task_duration > 0:
            duration_factor = config.task_duration_multiplier * metrics.average_task_duration
            duration_factor = min(duration_factor, config.max_interval)
        else:
            duration_factor = base_interval
        
        # Adjust based on progress
        progress_factor = 1.0 + (metrics.progress_percentage * 0.5)  # Longer intervals as we progress
        
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
    
    def should_checkpoint(self, 
                         metrics: CheckpointMetrics, 
                         config: CheckpointConfig,
                         trigger: CheckpointTrigger) -> bool:
        """Checkpoint at progress milestones."""
        # Always checkpoint on errors and manual triggers
        if trigger in [CheckpointTrigger.ERROR_DETECTION, CheckpointTrigger.MANUAL]:
            return True
        
        # Check progress milestones
        for milestone in self.milestones:
            if (milestone not in self.reached_milestones and 
                metrics.progress_percentage >= milestone):
                self.reached_milestones.add(milestone)
                return True
        
        # Fallback to time-based
        time_since_last = time.time() - metrics.last_checkpoint_time
        return time_since_last >= config.max_interval
    
    def get_next_checkpoint_time(self, 
                               metrics: CheckpointMetrics, 
                               config: CheckpointConfig) -> float:
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
                return remaining_progress / progress_rate if progress_rate > 0 else config.max_interval
        
        return config.max_interval


class AdaptiveCheckpointManager:
    """Manages adaptive checkpointing for pipeline executions."""
    
    def __init__(self, 
                 state_manager: "StateManager",
                 config: CheckpointConfig = None,
                 strategy: CheckpointStrategy = None):
        self.state_manager = state_manager
        self.config = config or CheckpointConfig()
        self.strategy = strategy or AdaptiveStrategy()
        
        # Execution tracking
        self.execution_metrics: Dict[str, CheckpointMetrics] = {}
        self.checkpoint_history: Dict[str, List[str]] = {}  # execution_id -> checkpoint_ids
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
        task = asyncio.create_task(
            self._checkpoint_scheduler(execution_id, pipeline)
        )
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
    
    async def update_metrics(self, 
                           execution_id: str, 
                           completed_tasks: int = 0,
                           failed_tasks: int = 0,
                           memory_usage: float = 0.0,
                           cpu_usage: float = 0.0,
                           progress_percentage: float = 0.0):
        """Update execution metrics."""
        if execution_id not in self.execution_metrics:
            return
        
        metrics = self.execution_metrics[execution_id]
        metrics.execution_time = time.time() - (metrics.last_checkpoint_time - metrics.execution_time)
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
    
    async def record_task_completion(self, execution_id: str, task: Task, duration: float):
        """Record task completion for metrics."""
        if execution_id in self.task_durations:
            self.task_durations[execution_id].append(duration)
        
        await self.update_metrics(execution_id)
        
        # Check if checkpoint should be created
        await self.check_checkpoint(execution_id, CheckpointTrigger.TASK_COMPLETION)
    
    async def check_checkpoint(self, 
                             execution_id: str, 
                             trigger: CheckpointTrigger,
                             state: Dict[str, Any] = None) -> Optional[str]:
        """Check if a checkpoint should be created and create it if needed."""
        if execution_id not in self.execution_metrics:
            return None
        
        metrics = self.execution_metrics[execution_id]
        
        if self.strategy.should_checkpoint(metrics, self.config, trigger):
            return await self.create_checkpoint(execution_id, state, trigger)
        
        return None
    
    async def create_checkpoint(self, 
                              execution_id: str, 
                              state: Dict[str, Any] = None,
                              trigger: CheckpointTrigger = CheckpointTrigger.MANUAL) -> str:
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
                "error_rate": metrics.error_rate
            }
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
            except Exception as e:
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
                to_remove = checkpoints[:-self.config.max_checkpoints]
                for checkpoint_id in to_remove:
                    await self._remove_checkpoint(checkpoint_id)
                self.checkpoint_history[execution_id] = checkpoints[-self.config.max_checkpoints:]
        
        elif self.config.cleanup_policy == "time_based":
            # Remove checkpoints older than retention period
            current_time = time.time()
            retention_seconds = self.config.retention_days * 24 * 3600
            
            to_remove = []
            for checkpoint_id in checkpoints:
                # Extract timestamp from checkpoint_id (format: execution_id_timestamp)
                try:
                    timestamp = int(checkpoint_id.split('_')[-1])
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
            "strategy": type(self.strategy).__name__
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
                *self._checkpoint_tasks.values(),
                return_exceptions=True
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
        retention_days=1
    )


def create_production_config() -> CheckpointConfig:
    """Create checkpoint configuration suitable for production."""
    return CheckpointConfig(
        max_checkpoints=10,
        min_interval=300.0,  # 5 minutes
        max_interval=1800.0,  # 30 minutes
        cleanup_policy="time_based",
        retention_days=7,
        error_rate_threshold=0.05  # Lower threshold for production
    )


# Alias for backward compatibility
AdaptiveCheckpointStrategy = AdaptiveStrategy