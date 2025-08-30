"""
Integration module for connecting Variable & State Management with existing runtime.

This module provides bridges and adapters to integrate the new variable management
system with the existing PipelineExecutionState from the runtime module.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Set
from .variables import VariableManager, VariableScope, VariableType
from .state import ExecutionContext
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