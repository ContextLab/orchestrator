"""
State Management and Execution Context for Pipeline Execution.

This module provides execution context management, state persistence capabilities,
and coordination with the variable management system for isolated pipeline runs.
"""

from __future__ import annotations

import logging
import json
import pickle
import gzip
from typing import Any, Dict, List, Optional, Set, Union, Protocol
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import hashlib
import threading
import asyncio
from contextlib import asynccontextmanager, contextmanager

from .variables import VariableManager, VariableScope, VariableType

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of pipeline execution."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PersistenceFormat(Enum):
    """State persistence format options."""
    JSON = "json"
    PICKLE = "pickle"
    COMPRESSED_JSON = "compressed_json"
    COMPRESSED_PICKLE = "compressed_pickle"


@dataclass
class ExecutionMetrics:
    """Metrics for pipeline execution."""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration: Optional[timedelta] = None
    steps_completed: int = 0
    steps_failed: int = 0
    steps_skipped: int = 0
    steps_total: int = 0
    variables_created: int = 0
    variables_updated: int = 0
    memory_peak_mb: float = 0.0
    cpu_time_seconds: float = 0.0
    
    def mark_completed(self):
        """Mark execution as completed and calculate duration."""
        self.end_time = datetime.now()
        self.duration = self.end_time - self.start_time
    
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.steps_total == 0:
            return 0.0
        return (self.steps_completed / self.steps_total) * 100.0
    
    def success_rate(self) -> float:
        """Calculate success rate of completed steps."""
        total_attempted = self.steps_completed + self.steps_failed
        if total_attempted == 0:
            return 0.0
        return (self.steps_completed / total_attempted) * 100.0


@dataclass
class Checkpoint:
    """Represents a state checkpoint for recovery."""
    id: str
    timestamp: datetime
    execution_id: str
    step_id: Optional[str] = None
    status: ExecutionStatus = ExecutionStatus.RUNNING
    variables_snapshot: Dict[str, Any] = field(default_factory=dict)
    metrics_snapshot: Optional[ExecutionMetrics] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate checkpoint ID if not provided."""
        if not self.id:
            content = f"{self.execution_id}_{self.timestamp.isoformat()}"
            if self.step_id:
                content += f"_{self.step_id}"
            self.id = hashlib.md5(content.encode()).hexdigest()[:8]


class StateManager(Protocol):
    """Protocol for state management implementations."""
    
    def save_state(self, execution_context: 'ExecutionContext') -> bool:
        """Save execution context state."""
        ...
    
    def load_state(self, execution_id: str) -> Optional['ExecutionContext']:
        """Load execution context state."""
        ...
    
    def list_states(self) -> List[str]:
        """List available execution states."""
        ...
    
    def delete_state(self, execution_id: str) -> bool:
        """Delete a saved execution state."""
        ...


class FileStateManager:
    """
    File-based state persistence manager.
    
    Provides state persistence to local filesystem with support for
    multiple formats and compression.
    """
    
    def __init__(
        self,
        state_dir: Union[str, Path] = "./pipeline_states",
        format: PersistenceFormat = PersistenceFormat.JSON,
        auto_cleanup_days: int = 30
    ):
        """
        Initialize file state manager.
        
        Args:
            state_dir: Directory to store state files
            format: Persistence format to use
            auto_cleanup_days: Days after which to auto-cleanup old states
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.format = format
        self.auto_cleanup_days = auto_cleanup_days
        self._lock = threading.RLock()
        
        logger.info(f"Initialized FileStateManager at {self.state_dir} using {format.value}")
    
    def save_state(self, execution_context: 'ExecutionContext') -> bool:
        """
        Save execution context to file.
        
        Args:
            execution_context: Context to save
            
        Returns:
            True if saved successfully
        """
        try:
            with self._lock:
                # Export context state
                state_data = execution_context.export_state()
                
                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{execution_context.execution_id}_{timestamp}"
                
                if self.format == PersistenceFormat.JSON:
                    filepath = self.state_dir / f"{filename}.json"
                    self._save_json(state_data, filepath)
                elif self.format == PersistenceFormat.PICKLE:
                    filepath = self.state_dir / f"{filename}.pkl"
                    self._save_pickle(state_data, filepath)
                elif self.format == PersistenceFormat.COMPRESSED_JSON:
                    filepath = self.state_dir / f"{filename}.json.gz"
                    self._save_compressed_json(state_data, filepath)
                elif self.format == PersistenceFormat.COMPRESSED_PICKLE:
                    filepath = self.state_dir / f"{filename}.pkl.gz"
                    self._save_compressed_pickle(state_data, filepath)
                
                logger.info(f"Saved execution state to {filepath}")
                
                # Auto-cleanup if enabled
                if self.auto_cleanup_days > 0:
                    self._cleanup_old_states()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to save state for {execution_context.execution_id}: {e}")
            return False
    
    def load_state(self, execution_id: str) -> Optional['ExecutionContext']:
        """
        Load execution context from file.
        
        Args:
            execution_id: Execution ID to load
            
        Returns:
            Loaded execution context or None if not found
        """
        try:
            with self._lock:
                # Find the most recent state file for this execution ID
                pattern = f"{execution_id}_*"
                matching_files = list(self.state_dir.glob(pattern + ".*"))
                
                if not matching_files:
                    logger.warning(f"No state files found for execution ID: {execution_id}")
                    return None
                
                # Sort by modification time, newest first
                matching_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                filepath = matching_files[0]
                
                # Load based on file extension
                if filepath.suffix == '.json':
                    state_data = self._load_json(filepath)
                elif filepath.suffix == '.pkl':
                    state_data = self._load_pickle(filepath)
                elif filepath.name.endswith('.json.gz'):
                    state_data = self._load_compressed_json(filepath)
                elif filepath.name.endswith('.pkl.gz'):
                    state_data = self._load_compressed_pickle(filepath)
                else:
                    logger.error(f"Unknown file format: {filepath}")
                    return None
                
                # Create execution context from loaded data
                context = ExecutionContext._from_state_data(state_data)
                logger.info(f"Loaded execution state from {filepath}")
                return context
                
        except Exception as e:
            logger.error(f"Failed to load state for {execution_id}: {e}")
            return None
    
    def list_states(self) -> List[str]:
        """
        List all available execution states.
        
        Returns:
            List of execution IDs
        """
        execution_ids = set()
        
        for filepath in self.state_dir.iterdir():
            if filepath.is_file():
                # Extract execution ID from filename
                name_parts = filepath.stem.split('_')
                if len(name_parts) >= 3:  # execution_id might contain underscores
                    # Take everything except the last 2 parts (date_time)
                    execution_id = '_'.join(name_parts[:-2])
                    execution_ids.add(execution_id)
        
        return sorted(list(execution_ids))
    
    def delete_state(self, execution_id: str) -> bool:
        """
        Delete all state files for an execution ID.
        
        Args:
            execution_id: Execution ID to delete
            
        Returns:
            True if any files were deleted
        """
        try:
            with self._lock:
                pattern = f"{execution_id}_*"
                matching_files = list(self.state_dir.glob(pattern + ".*"))
                
                deleted_count = 0
                for filepath in matching_files:
                    filepath.unlink()
                    deleted_count += 1
                
                if deleted_count > 0:
                    logger.info(f"Deleted {deleted_count} state files for execution ID: {execution_id}")
                    return True
                else:
                    logger.warning(f"No state files found to delete for execution ID: {execution_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to delete states for {execution_id}: {e}")
            return False
    
    def _save_json(self, data: Dict[str, Any], filepath: Path) -> None:
        """Save data as JSON."""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _load_json(self, filepath: Path) -> Dict[str, Any]:
        """Load data from JSON."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def _save_pickle(self, data: Dict[str, Any], filepath: Path) -> None:
        """Save data as pickle."""
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _load_pickle(self, filepath: Path) -> Dict[str, Any]:
        """Load data from pickle."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def _save_compressed_json(self, data: Dict[str, Any], filepath: Path) -> None:
        """Save data as compressed JSON."""
        json_str = json.dumps(data, default=str)
        with gzip.open(filepath, 'wt') as f:
            f.write(json_str)
    
    def _load_compressed_json(self, filepath: Path) -> Dict[str, Any]:
        """Load data from compressed JSON."""
        with gzip.open(filepath, 'rt') as f:
            return json.load(f)
    
    def _save_compressed_pickle(self, data: Dict[str, Any], filepath: Path) -> None:
        """Save data as compressed pickle."""
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _load_compressed_pickle(self, filepath: Path) -> Dict[str, Any]:
        """Load data from compressed pickle."""
        with gzip.open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def _cleanup_old_states(self) -> None:
        """Clean up old state files."""
        cutoff_time = datetime.now() - timedelta(days=self.auto_cleanup_days)
        
        deleted_count = 0
        for filepath in self.state_dir.iterdir():
            if filepath.is_file():
                # Check file modification time
                mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
                if mtime < cutoff_time:
                    filepath.unlink()
                    deleted_count += 1
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old state files")


class ExecutionContext:
    """
    Comprehensive execution context for pipeline runs.
    
    Provides isolated execution environment with variable management,
    state persistence, checkpoint/recovery capabilities, and metrics tracking.
    """
    
    def __init__(
        self,
        execution_id: str,
        pipeline_id: str,
        parent_context: Optional['ExecutionContext'] = None
    ):
        """
        Initialize execution context.
        
        Args:
            execution_id: Unique execution identifier
            pipeline_id: Pipeline identifier this execution belongs to
            parent_context: Parent context for nested executions
        """
        self.execution_id = execution_id
        self.pipeline_id = pipeline_id
        self.parent_context = parent_context
        
        # Core components
        self.variable_manager = VariableManager(pipeline_id=pipeline_id)
        self.state_manager: Optional[StateManager] = None
        
        # Execution state
        self.status = ExecutionStatus.PENDING
        self.metrics = ExecutionMetrics()
        self.current_step_id: Optional[str] = None
        
        # Checkpoint management
        self.checkpoints: List[Checkpoint] = []
        self.max_checkpoints = 10  # Keep last N checkpoints
        self.auto_checkpoint_steps = 5  # Create checkpoint every N steps
        self._checkpoint_counter = 0
        
        # Context isolation
        self.context_variables: Dict[str, Any] = {}
        self.nested_contexts: Dict[str, 'ExecutionContext'] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Event handlers
        self._step_handlers: List[callable] = []
        self._status_handlers: List[callable] = []
        
        logger.info(f"Initialized ExecutionContext: {execution_id} for pipeline: {pipeline_id}")
        
        # Register for variable events
        self.variable_manager.on_variable_created(self._on_variable_created)
        self.variable_manager.on_variable_changed(self._on_variable_changed)
    
    def set_state_manager(self, state_manager: StateManager) -> None:
        """
        Set the state manager for persistence.
        
        Args:
            state_manager: State manager instance
        """
        self.state_manager = state_manager
        logger.debug(f"Set state manager: {type(state_manager).__name__}")
    
    def start_execution(self) -> None:
        """Start pipeline execution."""
        with self._lock:
            self.status = ExecutionStatus.RUNNING
            self.metrics.start_time = datetime.now()
            logger.info(f"Started execution: {self.execution_id}")
            self._notify_status_handlers()
    
    def complete_execution(self, success: bool = True) -> None:
        """
        Complete pipeline execution.
        
        Args:
            success: Whether execution was successful
        """
        with self._lock:
            self.status = ExecutionStatus.COMPLETED if success else ExecutionStatus.FAILED
            self.metrics.mark_completed()
            
            # Create final checkpoint
            self.create_checkpoint("execution_completed")
            
            logger.info(
                f"Completed execution: {self.execution_id} "
                f"({'success' if success else 'failed'}) "
                f"in {self.metrics.duration}"
            )
            self._notify_status_handlers()
    
    def pause_execution(self) -> bool:
        """
        Pause pipeline execution.
        
        Returns:
            True if paused successfully
        """
        with self._lock:
            if self.status == ExecutionStatus.RUNNING:
                self.status = ExecutionStatus.PAUSED
                self.create_checkpoint("execution_paused")
                logger.info(f"Paused execution: {self.execution_id}")
                self._notify_status_handlers()
                return True
            return False
    
    def resume_execution(self) -> bool:
        """
        Resume paused execution.
        
        Returns:
            True if resumed successfully
        """
        with self._lock:
            if self.status == ExecutionStatus.PAUSED:
                self.status = ExecutionStatus.RUNNING
                logger.info(f"Resumed execution: {self.execution_id}")
                self._notify_status_handlers()
                return True
            return False
    
    def cancel_execution(self) -> None:
        """Cancel pipeline execution."""
        with self._lock:
            self.status = ExecutionStatus.CANCELLED
            self.metrics.mark_completed()
            logger.info(f"Cancelled execution: {self.execution_id}")
            self._notify_status_handlers()
    
    def start_step(self, step_id: str) -> None:
        """
        Start executing a pipeline step.
        
        Args:
            step_id: Step identifier
        """
        with self._lock:
            self.current_step_id = step_id
            self.metrics.steps_total = max(self.metrics.steps_total, 
                                         self.metrics.steps_completed + 
                                         self.metrics.steps_failed + 1)
            
            logger.debug(f"Started step: {step_id}")
            self._notify_step_handlers(step_id, "started")
            
            # Auto-checkpoint if configured
            self._checkpoint_counter += 1
            if self._checkpoint_counter % self.auto_checkpoint_steps == 0:
                self.create_checkpoint(f"step_{step_id}")
    
    def complete_step(self, step_id: str, success: bool = True) -> None:
        """
        Complete a pipeline step.
        
        Args:
            step_id: Step identifier
            success: Whether step completed successfully
        """
        with self._lock:
            if success:
                self.metrics.steps_completed += 1
            else:
                self.metrics.steps_failed += 1
            
            logger.debug(f"Completed step: {step_id} ({'success' if success else 'failed'})")
            self._notify_step_handlers(step_id, "completed" if success else "failed")
            
            # Clear current step
            if self.current_step_id == step_id:
                self.current_step_id = None
    
    def skip_step(self, step_id: str) -> None:
        """
        Skip a pipeline step.
        
        Args:
            step_id: Step identifier
        """
        with self._lock:
            self.metrics.steps_skipped += 1
            logger.debug(f"Skipped step: {step_id}")
            self._notify_step_handlers(step_id, "skipped")
    
    def create_checkpoint(self, description: str = None) -> Checkpoint:
        """
        Create a state checkpoint.
        
        Args:
            description: Optional description for the checkpoint
            
        Returns:
            Created checkpoint
        """
        with self._lock:
            checkpoint = Checkpoint(
                id="",  # Will be auto-generated
                timestamp=datetime.now(),
                execution_id=self.execution_id,
                step_id=self.current_step_id,
                status=self.status,
                variables_snapshot=self.variable_manager.export_state(),
                metrics_snapshot=ExecutionMetrics(**asdict(self.metrics)),
                metadata={"description": description} if description else {}
            )
            
            self.checkpoints.append(checkpoint)
            
            # Limit checkpoint history
            if len(self.checkpoints) > self.max_checkpoints:
                removed = self.checkpoints.pop(0)
                logger.debug(f"Removed old checkpoint: {removed.id}")
            
            logger.debug(f"Created checkpoint: {checkpoint.id}")
            return checkpoint
    
    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Restore from a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID to restore from
            
        Returns:
            True if restored successfully
        """
        with self._lock:
            # Find checkpoint
            checkpoint = None
            for cp in self.checkpoints:
                if cp.id == checkpoint_id:
                    checkpoint = cp
                    break
            
            if not checkpoint:
                logger.error(f"Checkpoint not found: {checkpoint_id}")
                return False
            
            try:
                # Restore status and metrics
                self.status = checkpoint.status
                if checkpoint.metrics_snapshot:
                    self.metrics = checkpoint.metrics_snapshot
                self.current_step_id = checkpoint.step_id
                
                # Restore variables
                if checkpoint.variables_snapshot:
                    self.variable_manager.import_state(checkpoint.variables_snapshot)
                
                logger.info(f"Restored from checkpoint: {checkpoint_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to restore checkpoint {checkpoint_id}: {e}")
                return False
    
    def create_nested_context(self, context_id: str) -> 'ExecutionContext':
        """
        Create a nested execution context.
        
        Args:
            context_id: Identifier for the nested context
            
        Returns:
            Nested execution context
        """
        nested_execution_id = f"{self.execution_id}.{context_id}"
        nested_context = ExecutionContext(
            execution_id=nested_execution_id,
            pipeline_id=self.pipeline_id,
            parent_context=self
        )
        
        self.nested_contexts[context_id] = nested_context
        logger.debug(f"Created nested context: {context_id}")
        return nested_context
    
    def destroy_nested_context(self, context_id: str) -> None:
        """
        Destroy a nested context.
        
        Args:
            context_id: Context to destroy
        """
        if context_id in self.nested_contexts:
            del self.nested_contexts[context_id]
            logger.debug(f"Destroyed nested context: {context_id}")
    
    def save_state(self) -> bool:
        """
        Save execution state using configured state manager.
        
        Returns:
            True if saved successfully
        """
        if not self.state_manager:
            logger.warning("No state manager configured for state saving")
            return False
        
        return self.state_manager.save_state(self)
    
    @classmethod
    def load_state(
        cls,
        execution_id: str,
        state_manager: StateManager
    ) -> Optional['ExecutionContext']:
        """
        Load execution state using state manager.
        
        Args:
            execution_id: Execution ID to load
            state_manager: State manager to use
            
        Returns:
            Loaded execution context or None if not found
        """
        return state_manager.load_state(execution_id)
    
    def export_state(self) -> Dict[str, Any]:
        """
        Export complete execution state.
        
        Returns:
            Complete state as dictionary
        """
        return {
            'execution_id': self.execution_id,
            'pipeline_id': self.pipeline_id,
            'status': self.status.value,
            'current_step_id': self.current_step_id,
            'metrics': asdict(self.metrics),
            'variables': self.variable_manager.export_state(),
            'checkpoints': [asdict(cp) for cp in self.checkpoints],
            'context_variables': dict(self.context_variables),
            'nested_contexts': {
                ctx_id: ctx.export_state() 
                for ctx_id, ctx in self.nested_contexts.items()
            },
            'timestamp': datetime.now().isoformat(),
        }
    
    @classmethod
    def _from_state_data(cls, state_data: Dict[str, Any]) -> 'ExecutionContext':
        """
        Create execution context from state data.
        
        Args:
            state_data: State dictionary
            
        Returns:
            Execution context instance
        """
        context = cls(
            execution_id=state_data['execution_id'],
            pipeline_id=state_data['pipeline_id']
        )
        
        # Restore status and metrics
        context.status = ExecutionStatus(state_data['status'])
        context.current_step_id = state_data.get('current_step_id')
        
        # Restore metrics
        metrics_data = state_data.get('metrics', {})
        context.metrics = ExecutionMetrics(**metrics_data)
        
        # Restore variables
        variables_data = state_data.get('variables', {})
        if variables_data:
            context.variable_manager.import_state(variables_data)
        
        # Restore checkpoints
        for cp_data in state_data.get('checkpoints', []):
            checkpoint = Checkpoint(**cp_data)
            context.checkpoints.append(checkpoint)
        
        # Restore context variables
        context.context_variables = dict(state_data.get('context_variables', {}))
        
        # Restore nested contexts
        for ctx_id, ctx_data in state_data.get('nested_contexts', {}).items():
            nested_ctx = cls._from_state_data(ctx_data)
            nested_ctx.parent_context = context
            context.nested_contexts[ctx_id] = nested_ctx
        
        return context
    
    def on_step_event(self, handler: callable) -> None:
        """
        Register handler for step events.
        
        Args:
            handler: Function that takes (step_id, event_type)
        """
        self._step_handlers.append(handler)
    
    def on_status_change(self, handler: callable) -> None:
        """
        Register handler for status change events.
        
        Args:
            handler: Function that takes (execution_context)
        """
        self._status_handlers.append(handler)
    
    def _on_variable_created(self, name: str, value: Any) -> None:
        """Handle variable creation events."""
        self.metrics.variables_created += 1
    
    def _on_variable_changed(self, name: str, old_value: Any, new_value: Any) -> None:
        """Handle variable change events."""
        self.metrics.variables_updated += 1
    
    def _notify_step_handlers(self, step_id: str, event_type: str) -> None:
        """Notify step event handlers."""
        for handler in self._step_handlers:
            try:
                handler(step_id, event_type)
            except Exception as e:
                logger.error(f"Error in step handler: {e}")
    
    def _notify_status_handlers(self) -> None:
        """Notify status change handlers."""
        for handler in self._status_handlers:
            try:
                handler(self)
            except Exception as e:
                logger.error(f"Error in status handler: {e}")
    
    @contextmanager
    def variable_scope(self):
        """Context manager for variable scoping."""
        from .variables import VariableContext
        with VariableContext(self.variable_manager) as ctx:
            yield ctx
    
    def __enter__(self) -> 'ExecutionContext':
        """Enter execution context."""
        self.start_execution()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit execution context."""
        if exc_type is not None:
            self.complete_execution(success=False)
        else:
            self.complete_execution(success=True)


# Convenience functions for common operations

def create_execution_context(
    execution_id: str,
    pipeline_id: str,
    state_dir: Optional[Union[str, Path]] = None,
    auto_save: bool = True
) -> ExecutionContext:
    """
    Create a fully configured execution context.
    
    Args:
        execution_id: Unique execution identifier
        pipeline_id: Pipeline identifier
        state_dir: Directory for state persistence
        auto_save: Whether to auto-save state
        
    Returns:
        Configured execution context
    """
    context = ExecutionContext(execution_id, pipeline_id)
    
    if state_dir or auto_save:
        state_manager = FileStateManager(state_dir or "./pipeline_states")
        context.set_state_manager(state_manager)
    
    return context


def load_execution_context(
    execution_id: str,
    state_dir: Union[str, Path] = "./pipeline_states"
) -> Optional[ExecutionContext]:
    """
    Load execution context from saved state.
    
    Args:
        execution_id: Execution ID to load
        state_dir: Directory containing saved states
        
    Returns:
        Loaded execution context or None if not found
    """
    state_manager = FileStateManager(state_dir)
    return ExecutionContext.load_state(execution_id, state_manager)