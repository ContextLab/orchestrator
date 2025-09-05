"""Tests for adaptive checkpointing functionality."""

import time

import pytest

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider
from src.orchestrator.state.adaptive_checkpoint import (

    AdaptiveCheckpointManager,
    AdaptiveCheckpointStrategy,
    AdaptiveStrategy,
    CheckpointConfig,
    CheckpointMetrics,
    CheckpointStrategy,
    CheckpointTrigger,
    ProgressBasedStrategy,
    TimeBasedStrategy)


class TestCheckpointConfig:
    """Test cases for CheckpointConfig class."""

    def test_config_creation_defaults(self):
        """Test default configuration creation."""
        config = CheckpointConfig()

        assert config.max_checkpoints == 10
        assert config.min_interval == 60.0
        assert config.max_interval == 3600.0
        assert config.compression_enabled is True
        assert config.cleanup_policy == "keep_latest"
        assert config.retention_days == 7
        assert config.error_rate_threshold == 0.1
        assert config.task_duration_multiplier == 2.0
        assert config.memory_usage_threshold == 0.8

    def test_config_creation_custom(self):
        """Test custom configuration creation."""
        config = CheckpointConfig(
            max_checkpoints=20,
            min_interval=30.0,
            max_interval=7200.0,
            compression_enabled=False,
            cleanup_policy="time_based",
            retention_days=14,
            error_rate_threshold=0.05,
            task_duration_multiplier=3.0,
            memory_usage_threshold=0.9)

        assert config.max_checkpoints == 20
        assert config.min_interval == 30.0
        assert config.max_interval == 7200.0
        assert config.compression_enabled is False
        assert config.cleanup_policy == "time_based"
        assert config.retention_days == 14
        assert config.error_rate_threshold == 0.05
        assert config.task_duration_multiplier == 3.0
        assert config.memory_usage_threshold == 0.9


class TestCheckpointMetrics:
    """Test cases for CheckpointMetrics class."""

    def test_metrics_creation_defaults(self):
        """Test default metrics creation."""
        metrics = CheckpointMetrics()

        assert metrics.execution_time == 0.0
        assert metrics.completed_tasks == 0
        assert metrics.failed_tasks == 0
        assert metrics.memory_usage == 0.0
        assert metrics.cpu_usage == 0.0
        assert metrics.last_checkpoint_time == 0.0
        assert metrics.checkpoint_count == 0
        assert metrics.average_task_duration == 0.0
        assert metrics.error_rate == 0.0
        assert metrics.progress_percentage == 0.0

    def test_metrics_creation_custom(self):
        """Test custom metrics creation."""
        metrics = CheckpointMetrics(
            execution_time=120.5,
            completed_tasks=15,
            failed_tasks=2,
            memory_usage=0.7,
            cpu_usage=0.8,
            last_checkpoint_time=1234567890.0,
            checkpoint_count=3,
            average_task_duration=8.0,
            error_rate=0.13,
            progress_percentage=0.65)

        assert metrics.execution_time == 120.5
        assert metrics.completed_tasks == 15
        assert metrics.failed_tasks == 2
        assert metrics.memory_usage == 0.7
        assert metrics.cpu_usage == 0.8
        assert metrics.last_checkpoint_time == 1234567890.0
        assert metrics.checkpoint_count == 3
        assert metrics.average_task_duration == 8.0
        assert metrics.error_rate == 0.13
        assert metrics.progress_percentage == 0.65


class TestTimeBasedStrategy:
    """Test cases for TimeBasedStrategy class."""

    def test_time_based_strategy_creation(self):
        """Test time-based strategy creation."""
        strategy = TimeBasedStrategy()
        assert isinstance(strategy, CheckpointStrategy)

    def test_should_checkpoint_time_based_trigger(self):
        """Test checkpointing with time-based trigger."""
        strategy = TimeBasedStrategy()
        config = CheckpointConfig(min_interval=60.0)

        # Recent checkpoint - should not checkpoint
        metrics = CheckpointMetrics(last_checkpoint_time=time.time())
        assert (
            strategy.should_checkpoint(metrics, config, CheckpointTrigger.TIME_BASED)
            is False
        )

        # Old checkpoint - should checkpoint
        metrics = CheckpointMetrics(last_checkpoint_time=time.time() - 120.0)
        assert (
            strategy.should_checkpoint(metrics, config, CheckpointTrigger.TIME_BASED)
            is True
        )

    def test_should_checkpoint_error_trigger(self):
        """Test checkpointing with error trigger."""
        strategy = TimeBasedStrategy()
        config = CheckpointConfig()
        metrics = CheckpointMetrics(last_checkpoint_time=time.time())

        # Error detection should always trigger checkpoint
        assert (
            strategy.should_checkpoint(
                metrics, config, CheckpointTrigger.ERROR_DETECTION
            )
            is True
        )

    def test_should_checkpoint_other_triggers(self):
        """Test checkpointing with other triggers."""
        strategy = TimeBasedStrategy()
        config = CheckpointConfig()
        metrics = CheckpointMetrics()

        # Other triggers should not trigger in time-based strategy
        assert (
            strategy.should_checkpoint(
                metrics, config, CheckpointTrigger.TASK_COMPLETION
            )
            is False
        )
        assert (
            strategy.should_checkpoint(
                metrics, config, CheckpointTrigger.RESOURCE_USAGE
            )
            is False
        )
        assert (
            strategy.should_checkpoint(metrics, config, CheckpointTrigger.MANUAL)
            is False
        )

    def test_get_next_checkpoint_time(self):
        """Test getting next checkpoint time."""
        strategy = TimeBasedStrategy()
        config = CheckpointConfig(min_interval=120.0)
        metrics = CheckpointMetrics()

        next_time = strategy.get_next_checkpoint_time(metrics, config)
        assert next_time == 120.0


class TestAdaptiveStrategy:
    """Test cases for AdaptiveStrategy class."""

    def test_adaptive_strategy_creation(self):
        """Test adaptive strategy creation."""
        strategy = AdaptiveStrategy()
        assert strategy.checkpoint_interval == 60.0

        strategy_custom = AdaptiveStrategy(checkpoint_interval=120.0)
        assert strategy_custom.checkpoint_interval == 120.0

    def test_should_checkpoint_error_detection(self):
        """Test checkpointing with error detection trigger."""
        strategy = AdaptiveStrategy()
        config = CheckpointConfig()
        metrics = CheckpointMetrics()

        # Error detection should always trigger
        assert (
            strategy.should_checkpoint(
                metrics, config, CheckpointTrigger.ERROR_DETECTION
            )
            is True
        )

    def test_should_checkpoint_manual_trigger(self):
        """Test checkpointing with manual trigger."""
        strategy = AdaptiveStrategy()
        config = CheckpointConfig()
        metrics = CheckpointMetrics()

        # Manual trigger should always trigger
        assert (
            strategy.should_checkpoint(metrics, config, CheckpointTrigger.MANUAL)
            is True
        )

    def test_should_checkpoint_too_frequent(self):
        """Test prevention of too frequent checkpoints."""
        strategy = AdaptiveStrategy()
        config = CheckpointConfig(min_interval=60.0)
        metrics = CheckpointMetrics(last_checkpoint_time=time.time())

        # Should not checkpoint if too recent
        assert (
            strategy.should_checkpoint(metrics, config, CheckpointTrigger.TIME_BASED)
            is False
        )

    def test_should_checkpoint_high_error_rate(self):
        """Test checkpointing with high error rate."""
        strategy = AdaptiveStrategy()
        config = CheckpointConfig(min_interval=60.0, error_rate_threshold=0.1)
        metrics = CheckpointMetrics(
            last_checkpoint_time=time.time() - 70.0, error_rate=0.15
        )

        # High error rate should trigger more frequent checkpoints
        assert (
            strategy.should_checkpoint(metrics, config, CheckpointTrigger.TIME_BASED)
            is True
        )

    def test_should_checkpoint_high_resource_usage(self):
        """Test checkpointing with high resource usage."""
        strategy = AdaptiveStrategy()
        config = CheckpointConfig(min_interval=60.0, memory_usage_threshold=0.8)

        # High memory usage - must pass min_interval check first (60 seconds)
        metrics = CheckpointMetrics(
            last_checkpoint_time=time.time() - 61.0, memory_usage=0.85  # 61 seconds ago
        )
        assert (
            strategy.should_checkpoint(metrics, config, CheckpointTrigger.TIME_BASED)
            is True
        )

        # High CPU usage
        metrics = CheckpointMetrics(
            last_checkpoint_time=time.time() - 61.0, cpu_usage=0.95
        )
        assert (
            strategy.should_checkpoint(metrics, config, CheckpointTrigger.TIME_BASED)
            is True
        )

    def test_should_checkpoint_task_completion(self):
        """Test checkpointing on task completion."""
        strategy = AdaptiveStrategy()
        config = CheckpointConfig()

        # Must pass min_interval check first (60 seconds by default)
        old_time = time.time() - 61.0

        # Low error rate - checkpoint every 9 tasks (max(5, int(10 * (1 - 0.05))))
        metrics = CheckpointMetrics(
            completed_tasks=9, error_rate=0.05, last_checkpoint_time=old_time
        )
        assert (
            strategy.should_checkpoint(
                metrics, config, CheckpointTrigger.TASK_COMPLETION
            )
            is True
        )

        metrics = CheckpointMetrics(
            completed_tasks=10, error_rate=0.05, last_checkpoint_time=old_time
        )
        assert (
            strategy.should_checkpoint(
                metrics, config, CheckpointTrigger.TASK_COMPLETION
            )
            is False
        )

        # High error rate - checkpoint every 5 tasks
        metrics = CheckpointMetrics(
            completed_tasks=5, error_rate=0.2, last_checkpoint_time=old_time
        )
        assert (
            strategy.should_checkpoint(
                metrics, config, CheckpointTrigger.TASK_COMPLETION
            )
            is True
        )

    def test_should_checkpoint_pipeline_milestones(self):
        """Test checkpointing at pipeline milestones."""
        strategy = AdaptiveStrategy()
        config = CheckpointConfig()

        # Test 25% milestone
        metrics = CheckpointMetrics(progress_percentage=0.26)
        assert (
            strategy.should_checkpoint(
                metrics, config, CheckpointTrigger.PIPELINE_MILESTONE
            )
            is True
        )

        # Test 50% milestone
        metrics = CheckpointMetrics(progress_percentage=0.51)
        assert (
            strategy.should_checkpoint(
                metrics, config, CheckpointTrigger.PIPELINE_MILESTONE
            )
            is True
        )

        # Test 75% milestone
        metrics = CheckpointMetrics(progress_percentage=0.76)
        assert (
            strategy.should_checkpoint(
                metrics, config, CheckpointTrigger.PIPELINE_MILESTONE
            )
            is True
        )

        # Test 90% milestone
        metrics = CheckpointMetrics(progress_percentage=0.91)
        assert (
            strategy.should_checkpoint(
                metrics, config, CheckpointTrigger.PIPELINE_MILESTONE
            )
            is True
        )

        # Test non-milestone
        metrics = CheckpointMetrics(progress_percentage=0.35)
        assert (
            strategy.should_checkpoint(
                metrics, config, CheckpointTrigger.PIPELINE_MILESTONE
            )
            is False
        )

    def test_get_next_checkpoint_time_adaptive(self):
        """Test adaptive next checkpoint time calculation."""
        strategy = AdaptiveStrategy()
        config = CheckpointConfig(min_interval=60.0, max_interval=3600.0)

        # Low error rate - longer interval
        metrics = CheckpointMetrics(error_rate=0.02, average_task_duration=10.0)
        next_time = strategy.get_next_checkpoint_time(metrics, config)
        assert next_time >= config.min_interval
        assert next_time <= config.max_interval

        # High error rate - shorter interval
        metrics = CheckpointMetrics(error_rate=0.15, average_task_duration=10.0)
        next_time_high_error = strategy.get_next_checkpoint_time(metrics, config)
        assert next_time_high_error >= config.min_interval
        assert next_time_high_error <= next_time  # Should be shorter


class TestProgressBasedStrategy:
    """Test cases for ProgressBasedStrategy class."""

    def test_progress_based_strategy_creation(self):
        """Test progress-based strategy creation."""
        strategy = ProgressBasedStrategy()
        assert isinstance(strategy, CheckpointStrategy)

    def test_should_checkpoint_milestones(self):
        """Test checkpointing at progress milestones."""
        strategy = ProgressBasedStrategy()
        config = CheckpointConfig()

        # Test different progress percentages
        for progress in [0.25, 0.5, 0.75, 0.9]:
            metrics = CheckpointMetrics(progress_percentage=progress)
            result = strategy.should_checkpoint(
                metrics, config, CheckpointTrigger.PIPELINE_MILESTONE
            )
            assert isinstance(result, bool)

    def test_get_next_checkpoint_time_progress(self):
        """Test getting next checkpoint time for progress strategy."""
        strategy = ProgressBasedStrategy()
        config = CheckpointConfig()
        metrics = CheckpointMetrics()

        next_time = strategy.get_next_checkpoint_time(metrics, config)
        assert isinstance(next_time, float)
        assert next_time > 0


class TestAdaptiveCheckpointManager:
    """Test cases for AdaptiveCheckpointManager class."""

    def test_manager_creation(self):
        """Test adaptive checkpoint manager creation."""
        from src.orchestrator.state.state_manager import StateManager

        state_manager = StateManager(backend_type="memory")

        manager = AdaptiveCheckpointManager(state_manager)
        assert hasattr(manager, "state_manager")
        assert hasattr(manager, "config")
        assert hasattr(manager, "strategy")

    @pytest.mark.asyncio
    async def test_manager_operations(self):
        """Test basic manager operations."""
        from src.orchestrator.state.state_manager import StateManager

        state_manager = StateManager(backend_type="memory")

        manager = AdaptiveCheckpointManager(state_manager)

        # Test that manager can be used for basic operations
        # This tests the interface without relying on specific implementation
        assert manager is not None

        # Test configuration access
        assert manager.config is not None
        assert manager.state_manager is not None


class TestAdaptiveCheckpointStrategyIntegration:
    """Test cases for AdaptiveCheckpointStrategy integration."""

    def test_strategy_creation_with_params(self):
        """Test strategy creation with parameters."""
        strategy = AdaptiveCheckpointStrategy(
            checkpoint_interval=10, critical_task_patterns=[".*_critical"]
        )

        assert hasattr(strategy, "checkpoint_interval")
        assert strategy.checkpoint_interval == 10

    def test_strategy_checkpoint_decisions(self):
        """Test strategy checkpoint decisions."""
        strategy = AdaptiveCheckpointStrategy()

        # Test basic checkpoint decision logic
        result = strategy.should_checkpoint("test_pipeline", "test_task")
        assert isinstance(result, bool)

    def test_strategy_with_critical_patterns(self):
        """Test strategy with critical task patterns."""
        strategy = AdaptiveCheckpointStrategy(
            critical_task_patterns=[".*_critical", "important_.*"]
        )

        # Critical tasks should trigger checkpoint
        assert strategy.should_checkpoint("pipeline", "task_critical") is True
        assert strategy.should_checkpoint("pipeline", "important_task") is True

        # Non-critical should follow normal logic
        result = strategy.should_checkpoint("pipeline", "normal_task")
        assert isinstance(result, bool)
