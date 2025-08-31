"""
Integration module for connecting Variable & State Management with existing runtime.

This module provides bridges and adapters to integrate the new variable management
system, progress tracking, and recovery mechanisms with the existing 
PipelineExecutionState from the runtime module.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Set, Callable
from .variables import VariableManager, VariableScope, VariableType
from .state import ExecutionContext
from .progress import ProgressTracker, ProgressEvent, create_progress_tracker
from .recovery import RecoveryManager, ErrorInfo, create_recovery_manager
from ..runtime.execution_state import PipelineExecutionState

logger = logging.getLogger(__name__)


class ExecutionStateBridge:
    """
    Bridge between new ExecutionContext and existing PipelineExecutionState.
    
    Provides seamless integration between the legacy runtime state management
    and the new comprehensive variable and execution context system.
    """
    
    def __init__(
        self,
        execution_context: ExecutionContext,
        pipeline_execution_state: Optional[PipelineExecutionState] = None
    ):
        """
        Initialize the bridge.
        
        Args:
            execution_context: New execution context
            pipeline_execution_state: Existing pipeline state (optional)
        """
        self.execution_context = execution_context
        self.pipeline_state = pipeline_execution_state
        
        # Set up bidirectional synchronization if pipeline state provided
        if self.pipeline_state:
            self._setup_synchronization()
        
        logger.info(f"Initialized ExecutionStateBridge for {execution_context.execution_id}")
    
    def set_pipeline_state(self, pipeline_state: PipelineExecutionState) -> None:
        """
        Set the pipeline execution state and enable synchronization.
        
        Args:
            pipeline_state: Pipeline execution state to bridge with
        """
        self.pipeline_state = pipeline_state
        self._setup_synchronization()
        logger.info(f"Connected pipeline state: {pipeline_state.pipeline_id}")
    
    def sync_to_pipeline_state(self) -> None:
        """Sync variables from ExecutionContext to PipelineExecutionState."""
        if not self.pipeline_state:
            return
        
        # Get all variables from the variable manager
        variables = self.execution_context.variable_manager.list_variables(
            include_metadata=True
        )
        
        for name, var_info in variables.items():
            value = var_info['value'] if isinstance(var_info, dict) else var_info
            metadata = var_info.get('metadata') if isinstance(var_info, dict) else None
            
            # Determine appropriate registration method based on variable type
            if metadata:
                var_type = metadata.var_type
                if var_type == VariableType.OUTPUT:
                    # Register as task result if it came from a step
                    if metadata.source_step:
                        self.pipeline_state.register_result(metadata.source_step, value)
                    else:
                        self.pipeline_state.register_variable(name, value)
                elif var_type == VariableType.CONFIGURATION:
                    # Register as variable
                    self.pipeline_state.register_variable(name, value)
                else:
                    # Default to variable registration
                    self.pipeline_state.register_variable(name, value)
            else:
                # No metadata, register as variable
                self.pipeline_state.register_variable(name, value)
    
    def sync_from_pipeline_state(self) -> None:
        """Sync variables from PipelineExecutionState to ExecutionContext."""
        if not self.pipeline_state:
            return
        
        # Get available context from pipeline state
        context = self.pipeline_state.get_available_context()
        
        for name, value in context.items():
            # Skip system variables that are already managed
            if name in ['pipeline_id', 'execution_time', 'timestamp']:
                continue
            
            # Determine variable type and scope
            var_type = VariableType.INTERMEDIATE
            scope = VariableScope.GLOBAL
            source_step = None
            
            # Check if this is a task result
            if name in self.pipeline_state.global_context.get('results', {}):
                var_type = VariableType.OUTPUT
                # Try to determine source step
                if name in self.pipeline_state.executed_tasks:
                    source_step = name
            elif name in self.pipeline_state.global_context.get('variables', {}):
                var_type = VariableType.CONFIGURATION
            
            # Set variable in the new system
            self.execution_context.variable_manager.set_variable(
                name=name,
                value=value,
                var_type=var_type,
                scope=scope,
                source_step=source_step,
                description=f"Synced from pipeline state"
            )
    
    def register_variable(
        self,
        name: str,
        value: Any,
        var_type: VariableType = VariableType.INTERMEDIATE,
        source_step: Optional[str] = None
    ) -> None:
        """
        Register a variable in both systems.
        
        Args:
            name: Variable name
            value: Variable value
            var_type: Variable type classification
            source_step: Source step identifier
        """
        # Register in new system
        self.execution_context.variable_manager.set_variable(
            name=name,
            value=value,
            var_type=var_type,
            source_step=source_step
        )
        
        # Register in legacy system if available
        if self.pipeline_state:
            if var_type == VariableType.OUTPUT and source_step:
                self.pipeline_state.register_result(source_step, value)
            else:
                self.pipeline_state.register_variable(name, value)
    
    def register_step_result(self, step_id: str, result: Any) -> None:
        """
        Register a step execution result in both systems.
        
        Args:
            step_id: Step identifier
            result: Step execution result
        """
        # Register in new system
        self.execution_context.variable_manager.set_variable(
            name=step_id,
            value=result,
            var_type=VariableType.OUTPUT,
            source_step=step_id,
            description=f"Result from step {step_id}"
        )
        
        # Complete step in execution context
        self.execution_context.complete_step(step_id, success=True)
        
        # Register in legacy system if available
        if self.pipeline_state:
            self.pipeline_state.register_result(step_id, result)
    
    def start_step(self, step_id: str) -> None:
        """
        Start a step in both systems.
        
        Args:
            step_id: Step identifier
        """
        # Start in new system
        self.execution_context.start_step(step_id)
        
        # Mark as pending in legacy system if available
        if self.pipeline_state:
            self.pipeline_state.pending_tasks.add(step_id)
    
    def fail_step(self, step_id: str, error: str) -> None:
        """
        Mark a step as failed in both systems.
        
        Args:
            step_id: Step identifier
            error: Error message
        """
        # Fail in new system
        self.execution_context.complete_step(step_id, success=False)
        
        # Fail in legacy system if available
        if self.pipeline_state:
            self.pipeline_state.mark_task_failed(step_id, error)
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """
        Get a variable value, checking both systems.
        
        Args:
            name: Variable name
            default: Default value if not found
            
        Returns:
            Variable value
        """
        # Try new system first
        value = self.execution_context.variable_manager.get_variable(name)
        if value is not None:
            return value
        
        # Fall back to legacy system
        if self.pipeline_state:
            context = self.pipeline_state.get_available_context()
            return context.get(name, default)
        
        return default
    
    def has_variable(self, name: str) -> bool:
        """
        Check if a variable exists in either system.
        
        Args:
            name: Variable name
            
        Returns:
            True if variable exists
        """
        # Check new system
        if self.execution_context.variable_manager.has_variable(name):
            return True
        
        # Check legacy system
        if self.pipeline_state:
            context = self.pipeline_state.get_available_context()
            return name in context
        
        return False
    
    def export_combined_state(self) -> Dict[str, Any]:
        """
        Export combined state from both systems.
        
        Returns:
            Combined state dictionary
        """
        combined_state = {
            'execution_context': self.execution_context.export_state(),
            'pipeline_state': None,
            'bridge_metadata': {
                'created_at': self.execution_context.metrics.start_time.isoformat(),
                'pipeline_id': self.execution_context.pipeline_id,
                'execution_id': self.execution_context.execution_id,
            }
        }
        
        if self.pipeline_state:
            combined_state['pipeline_state'] = self.pipeline_state.export_state()
        
        return combined_state
    
    def _setup_synchronization(self) -> None:
        """Set up bidirectional synchronization between systems."""
        # Initial sync from pipeline state to execution context
        self.sync_from_pipeline_state()
        
        # Set up change handlers for ongoing synchronization
        self.execution_context.variable_manager.on_variable_created(
            self._on_variable_created
        )
        self.execution_context.variable_manager.on_variable_changed(
            self._on_variable_changed
        )
    
    def _on_variable_created(self, name: str, value: Any) -> None:
        """Handle variable creation in the new system."""
        if self.pipeline_state and name not in self.pipeline_state.get_available_context():
            self.pipeline_state.register_variable(name, value)
    
    def _on_variable_changed(self, name: str, old_value: Any, new_value: Any) -> None:
        """Handle variable changes in the new system."""
        if self.pipeline_state:
            self.pipeline_state.register_variable(name, new_value)


class VariableManagerAdapter:
    """
    Adapter to make VariableManager compatible with expected interfaces.
    
    This allows the VariableManager to be plugged into existing systems
    that expect different interfaces.
    """
    
    def __init__(self, variable_manager: VariableManager):
        """
        Initialize the adapter.
        
        Args:
            variable_manager: Variable manager to adapt
        """
        self.variable_manager = variable_manager
    
    def set(self, key: str, value: Any) -> None:
        """Set a variable (dict-like interface)."""
        self.variable_manager.set_variable(key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a variable (dict-like interface)."""
        return self.variable_manager.get_variable(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Get variable using bracket notation."""
        value = self.variable_manager.get_variable(key)
        if value is None:
            raise KeyError(key)
        return value
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set variable using bracket notation."""
        self.variable_manager.set_variable(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if variable exists."""
        return self.variable_manager.has_variable(key)
    
    def keys(self):
        """Get all variable names."""
        variables = self.variable_manager.list_variables()
        return variables.keys()
    
    def values(self):
        """Get all variable values."""
        variables = self.variable_manager.list_variables()
        return variables.values()
    
    def items(self):
        """Get all variable name-value pairs."""
        variables = self.variable_manager.list_variables()
        return variables.items()
    
    def update(self, other: Dict[str, Any]) -> None:
        """Update multiple variables at once."""
        for key, value in other.items():
            self.variable_manager.set_variable(key, value)


class ComprehensiveExecutionManager:
    """
    Comprehensive integration manager that brings together all execution components.
    
    This class integrates ExecutionContext, VariableManager, ProgressTracker,
    and RecoveryManager to provide a unified execution management system for
    the StateGraphEngine.
    """
    
    def __init__(
        self,
        execution_id: str,
        pipeline_id: str,
        execution_context: Optional[ExecutionContext] = None,
        variable_manager: Optional[VariableManager] = None,
        progress_tracker: Optional[ProgressTracker] = None,
        recovery_manager: Optional[RecoveryManager] = None
    ):
        """
        Initialize comprehensive execution manager.
        
        Args:
            execution_id: Unique execution identifier
            pipeline_id: Pipeline identifier
            execution_context: Optional execution context (will create if None)
            variable_manager: Optional variable manager (will create if None)
            progress_tracker: Optional progress tracker (will create if None)
            recovery_manager: Optional recovery manager (will create if None)
        """
        self.execution_id = execution_id
        self.pipeline_id = pipeline_id
        
        # Initialize core components
        if execution_context:
            self.execution_context = execution_context
        else:
            self.execution_context = ExecutionContext(execution_id, pipeline_id)
        
        # Get variable manager from execution context
        self.variable_manager = self.execution_context.variable_manager
        
        # Initialize progress tracker
        if progress_tracker:
            self.progress_tracker = progress_tracker
        else:
            self.progress_tracker = create_progress_tracker(
                execution_context=self.execution_context,
                variable_manager=self.variable_manager
            )
        
        # Initialize recovery manager
        if recovery_manager:
            self.recovery_manager = recovery_manager
        else:
            self.recovery_manager = create_recovery_manager(
                execution_context=self.execution_context,
                progress_tracker=self.progress_tracker
            )
        
        # Set up cross-component integration
        self._setup_integration()
        
        logger.info(f"Initialized ComprehensiveExecutionManager for execution {execution_id}")
    
    def _setup_integration(self) -> None:
        """Set up integration between all components."""
        # Set up progress tracking for execution events
        self.execution_context.add_status_handler(self._handle_execution_status_change)
        
        # Set up recovery handlers for common error patterns
        self._setup_default_recovery_handlers()
        
        # Set up progress event handlers for checkpoint integration
        self.progress_tracker.add_event_handler(self._handle_progress_event)
    
    def _setup_default_recovery_handlers(self) -> None:
        """Set up default recovery handlers."""
        from .recovery import (
            network_error_handler,
            timeout_error_handler,
            critical_error_handler,
            ErrorCategory
        )
        
        # Register standard error handlers
        self.recovery_manager.register_error_handler(ErrorCategory.NETWORK, network_error_handler)
        self.recovery_manager.register_error_handler(ErrorCategory.TIMEOUT, timeout_error_handler)
        self.recovery_manager.register_global_error_handler(critical_error_handler)
    
    def _handle_execution_status_change(self, execution_id: str, status: Any) -> None:
        """Handle execution status changes."""
        from .state import ExecutionStatus
        
        if status == ExecutionStatus.RUNNING:
            # Get total steps from execution context if available
            total_steps = getattr(self.execution_context, 'total_steps', 1)
            self.progress_tracker.start_execution(execution_id, total_steps)
        
        elif status in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED):
            success = status == ExecutionStatus.COMPLETED
            self.progress_tracker.complete_execution(execution_id, success)
    
    def _handle_progress_event(self, event: ProgressEvent) -> None:
        """Handle progress events for checkpoint integration."""
        from .progress import ProgressEventType
        
        # Create checkpoints at key progress milestones
        if event.event_type == ProgressEventType.STEP_COMPLETED and event.step_id:
            # Create checkpoint every 5 steps or on important steps
            step_num = self.execution_context.completed_steps + 1
            if step_num % 5 == 0 or event.data.get('important', False):
                checkpoint = self.execution_context.create_checkpoint(
                    f"step_completed_{event.step_id}"
                )
                self.progress_tracker.create_checkpoint_event(
                    event.execution_id,
                    checkpoint.id,
                    event.step_id
                )
    
    def start_execution(self, total_steps: int = 1) -> None:
        """Start the execution with all systems coordinated."""
        logger.info(f"Starting execution {self.execution_id} with {total_steps} steps")
        
        # Start execution context
        self.execution_context.total_steps = total_steps
        self.execution_context.start()
        
        # Create initial checkpoint
        self.execution_context.create_checkpoint("execution_started")
        
        logger.info(f"Execution {self.execution_id} started successfully")
    
    def complete_execution(self, success: bool = True) -> None:
        """Complete the execution with all systems coordinated."""
        logger.info(f"Completing execution {self.execution_id} (success: {success})")
        
        # Complete execution context
        self.execution_context.complete(success)
        
        # Clean up if successful
        if success:
            # Keep only final checkpoint
            checkpoints = self.execution_context.checkpoints
            if len(checkpoints) > 1:
                final_checkpoint = checkpoints[-1]
                self.execution_context.checkpoints = [final_checkpoint]
        
        logger.info(f"Execution {self.execution_id} completed")
    
    def start_step(self, step_id: str, step_name: str) -> None:
        """Start a step with coordinated tracking."""
        logger.debug(f"Starting step {step_id} ({step_name})")
        
        # Start in execution context
        self.execution_context.start_step(step_id)
        
        # Start in progress tracker
        self.progress_tracker.start_step(self.execution_id, step_id, step_name)
    
    def complete_step(
        self,
        step_id: str,
        success: bool = True,
        error: Optional[Exception] = None,
        progress_percentage: float = 100.0
    ) -> None:
        """Complete a step with coordinated tracking."""
        logger.debug(f"Completing step {step_id} (success: {success})")
        
        # Complete in execution context
        self.execution_context.complete_step(step_id, success)
        
        # Complete in progress tracker
        error_message = str(error) if error else None
        self.progress_tracker.complete_step(
            self.execution_id,
            step_id,
            success,
            error_message,
            progress_percentage
        )
    
    def handle_step_error(
        self,
        step_id: str,
        step_name: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:  # Returns RecoveryPlan
        """Handle step error with integrated recovery."""
        logger.warning(f"Handling error in step {step_id}: {error}")
        
        # Use recovery manager to handle the error
        recovery_plan = self.recovery_manager.handle_error(
            error, step_id, step_name, context
        )
        
        return recovery_plan
    
    async def execute_step_with_recovery(
        self,
        step_id: str,
        step_name: str,
        step_executor: Callable[[], Any]
    ) -> bool:
        """Execute a step with integrated error handling and recovery."""
        self.start_step(step_id, step_name)
        
        try:
            # Execute the step
            result = await step_executor()
            self.complete_step(step_id, success=True)
            return True
            
        except Exception as error:
            # Handle the error and get recovery plan
            recovery_plan = self.handle_step_error(step_id, step_name, error)
            
            # Execute recovery if possible
            if recovery_plan.is_automated():
                logger.info(f"Attempting automated recovery for step {step_id}")
                try:
                    recovery_success = await self.recovery_manager.execute_recovery(
                        step_id, recovery_plan, step_executor
                    )
                    
                    if recovery_success:
                        self.complete_step(step_id, success=True)
                        return True
                    else:
                        self.complete_step(step_id, success=False, error=error)
                        return False
                        
                except Exception as recovery_error:
                    logger.error(f"Recovery failed for step {step_id}: {recovery_error}")
                    self.complete_step(step_id, success=False, error=recovery_error)
                    return False
            else:
                # Manual intervention required
                self.complete_step(step_id, success=False, error=error)
                return False
    
    def update_step_progress(
        self,
        step_id: str,
        progress_percentage: float,
        message: Optional[str] = None
    ) -> None:
        """Update step progress across all systems."""
        self.progress_tracker.update_step_progress(
            self.execution_id, step_id, progress_percentage, message
        )
        
        # Update variable for integration with variable change tracking
        progress_var_name = f"progress.{step_id}.percentage"
        self.variable_manager.set_variable(
            progress_var_name,
            progress_percentage,
            scope=VariableScope.STEP,
            var_type=VariableType.SYSTEM
        )
    
    def create_checkpoint(self, description: str = None) -> Any:  # Returns Checkpoint
        """Create a checkpoint with progress tracking integration."""
        checkpoint = self.execution_context.create_checkpoint(description)
        
        # Notify progress tracker
        self.progress_tracker.create_checkpoint_event(
            self.execution_id,
            checkpoint.id
        )
        
        return checkpoint
    
    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore from checkpoint with integrated systems."""
        logger.info(f"Restoring from checkpoint {checkpoint_id}")
        
        # Restore execution context
        success = self.execution_context.restore_checkpoint(checkpoint_id)
        
        if success:
            # Reset recovery state for restored step
            if self.execution_context.current_step_id:
                self.recovery_manager.reset_retry_count(self.execution_context.current_step_id)
            
            # Emit checkpoint restore event
            self.progress_tracker.create_checkpoint_event(
                self.execution_id,
                checkpoint_id,
                self.execution_context.current_step_id
            )
        
        return success
    
    def get_execution_status(self) -> Dict[str, Any]:
        """Get comprehensive execution status."""
        execution_progress = self.progress_tracker.get_execution_progress(self.execution_id)
        recovery_status = self.recovery_manager.get_recovery_status()
        
        return {
            "execution_id": self.execution_id,
            "pipeline_id": self.pipeline_id,
            "status": self.execution_context.status.value,
            "progress": {
                "percentage": execution_progress.progress_percentage if execution_progress else 0,
                "completed_steps": execution_progress.completed_steps if execution_progress else 0,
                "total_steps": execution_progress.total_steps if execution_progress else 0,
                "running_steps": execution_progress.running_steps if execution_progress else 0,
                "failed_steps": execution_progress.failed_steps if execution_progress else 0
            },
            "recovery": recovery_status,
            "checkpoints": len(self.execution_context.checkpoints),
            "metrics": {
                "start_time": self.execution_context.metrics.start_time.isoformat(),
                "duration": self.execution_context.metrics.duration.total_seconds() if self.execution_context.metrics.duration else None,
                "steps_completed": self.execution_context.completed_steps,
                "steps_failed": self.execution_context.failed_steps
            }
        }
    
    def cleanup(self) -> None:
        """Clean up all systems."""
        logger.info(f"Cleaning up execution {self.execution_id}")
        
        self.progress_tracker.cleanup(self.execution_id)
        self.recovery_manager.cleanup(self.execution_id)
        
        # Clear execution context but keep final state
        if hasattr(self.execution_context, 'cleanup'):
            self.execution_context.cleanup()
        
        logger.info(f"Cleanup completed for execution {self.execution_id}")
    
    def shutdown(self) -> None:
        """Shutdown all systems."""
        logger.info(f"Shutting down execution manager for {self.execution_id}")
        
        self.progress_tracker.shutdown()
        self.recovery_manager.shutdown()
        
        logger.info(f"Shutdown completed for execution {self.execution_id}")


def create_comprehensive_execution_manager(
    execution_id: str,
    pipeline_id: str
) -> ComprehensiveExecutionManager:
    """
    Create a fully integrated execution manager with all Stream C components.
    
    Args:
        execution_id: Unique execution identifier
        pipeline_id: Pipeline identifier
    
    Returns:
        Configured ComprehensiveExecutionManager
    """
    return ComprehensiveExecutionManager(
        execution_id=execution_id,
        pipeline_id=pipeline_id
    )