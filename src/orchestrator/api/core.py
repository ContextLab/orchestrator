"""
Core API Interface for the Orchestrator Framework.

This module provides the main user-facing API for pipeline compilation and execution,
integrating all foundation components into a clean, intuitive interface.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Import foundation components
from ..compiler import YAMLCompiler, YAMLCompilerError
from ..execution import (
    create_comprehensive_execution_manager,
    ComprehensiveExecutionManager,
    ExecutionStatus
)
from ..models import (
    get_model_registry,
    set_model_registry,
    ModelRegistry,
    UCBModelSelector
)
from ..core.pipeline import Pipeline
from ..core.task import Task

logger = logging.getLogger(__name__)


class PipelineAPIError(Exception):
    """Base exception for Pipeline API errors."""
    pass


class CompilationError(PipelineAPIError):
    """Exception raised during pipeline compilation."""
    pass


class ExecutionError(PipelineAPIError):
    """Exception raised during pipeline execution."""
    pass


class PipelineAPI:
    """
    Main API interface for pipeline operations.
    
    Provides clean, intuitive methods for:
    - Compiling YAML pipeline specifications into executable pipelines
    - Executing pipelines with real-time status tracking
    - Managing pipeline state and monitoring execution progress
    - Comprehensive error handling and recovery
    
    Example:
        >>> api = PipelineAPI()
        >>> pipeline = await api.compile_pipeline(yaml_content, context)
        >>> execution = await api.execute_pipeline(pipeline)
        >>> status = api.get_execution_status(execution.execution_id)
    """
    
    def __init__(
        self,
        model_registry: Optional[ModelRegistry] = None,
        development_mode: bool = False,
        validation_level: str = "strict"
    ):
        """
        Initialize the Pipeline API.
        
        Args:
            model_registry: Optional model registry for AUTO tag resolution
            development_mode: Enable development mode (relaxed validation)
            validation_level: Validation strictness ("strict", "permissive", "development")
        """
        self.development_mode = development_mode
        self.validation_level = validation_level
        
        # Initialize model registry
        if model_registry:
            set_model_registry(model_registry)
            self.model_registry = model_registry
        else:
            self.model_registry = get_model_registry()
        
        # Initialize YAML compiler with integrated validation
        self.compiler = YAMLCompiler(
            model_registry=self.model_registry,
            development_mode=development_mode,
            validation_level=self._get_validation_level_enum(validation_level),
            enable_validation_report=True,
            validate_templates=True,
            validate_tools=True,
            validate_models=True,
            validate_data_flow=True
        )
        
        # Track active executions
        self._active_executions: Dict[str, ComprehensiveExecutionManager] = {}
        
        logger.info(f"PipelineAPI initialized (development_mode={development_mode}, validation_level={validation_level})")
    
    def _get_validation_level_enum(self, level: str):
        """Convert validation level string to enum."""
        try:
            from ..validation.validation_report import ValidationLevel
            level_map = {
                "strict": ValidationLevel.STRICT,
                "permissive": ValidationLevel.PERMISSIVE,
                "development": ValidationLevel.DEVELOPMENT
            }
            return level_map.get(level.lower(), ValidationLevel.STRICT)
        except ImportError:
            # Fallback if validation_report not available
            return None
    
    async def compile_pipeline(
        self,
        yaml_content: Union[str, Path],
        context: Optional[Dict[str, Any]] = None,
        resolve_ambiguities: bool = True,
        validate: bool = True
    ) -> Pipeline:
        """
        Compile a YAML pipeline specification into an executable Pipeline.
        
        Args:
            yaml_content: YAML content as string or path to YAML file
            context: Template context variables for compilation
            resolve_ambiguities: Whether to resolve AUTO tags during compilation
            validate: Whether to perform comprehensive validation
            
        Returns:
            Compiled Pipeline object ready for execution
            
        Raises:
            CompilationError: If compilation fails
            FileNotFoundError: If YAML file path does not exist
        """
        try:
            # Handle file path input
            if isinstance(yaml_content, (str, Path)) and Path(yaml_content).exists():
                yaml_file = Path(yaml_content)
                logger.info(f"Loading YAML pipeline from file: {yaml_file}")
                yaml_content = yaml_file.read_text(encoding='utf-8')
            elif isinstance(yaml_content, Path):
                raise FileNotFoundError(f"YAML file not found: {yaml_content}")
            
            # Ensure we have string content
            if not isinstance(yaml_content, str):
                raise CompilationError(f"Invalid YAML content type: {type(yaml_content)}")
            
            logger.info("Starting pipeline compilation...")
            
            # Compile using YAML compiler with full validation
            pipeline = await self.compiler.compile(
                yaml_content=yaml_content,
                context=context or {},
                resolve_ambiguities=resolve_ambiguities
            )
            
            logger.info(f"Pipeline '{pipeline.id}' compiled successfully")
            
            # Log validation results if available
            validation_report = self.compiler.get_validation_report()
            if validation_report and validation_report.has_issues:
                logger.info(f"Compilation completed with {validation_report.stats.total_issues} validation issues")
                if validation_report.has_warnings:
                    logger.warning(f"Validation warnings: {validation_report.stats.warnings}")
            else:
                logger.info("Pipeline validation passed without issues")
            
            return pipeline
            
        except YAMLCompilerError as e:
            logger.error(f"Pipeline compilation failed: {e}")
            raise CompilationError(f"Failed to compile pipeline: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during compilation: {e}")
            raise CompilationError(f"Unexpected compilation error: {e}") from e
    
    async def execute_pipeline(
        self,
        pipeline: Union[Pipeline, str, Path],
        context: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None
    ) -> ComprehensiveExecutionManager:
        """
        Execute a compiled pipeline with comprehensive monitoring.
        
        Args:
            pipeline: Pipeline object, YAML content, or path to YAML file
            context: Additional execution context variables
            execution_id: Optional custom execution ID
            
        Returns:
            ComprehensiveExecutionManager for monitoring and control
            
        Raises:
            ExecutionError: If execution initialization fails
            CompilationError: If pipeline compilation fails (for non-Pipeline inputs)
        """
        try:
            # Compile pipeline if needed
            if not isinstance(pipeline, Pipeline):
                logger.info("Compiling pipeline from provided content...")
                pipeline = await self.compile_pipeline(pipeline, context)
            
            # Generate execution ID
            if not execution_id:
                execution_id = f"{pipeline.id}_{uuid.uuid4().hex[:8]}"
            
            logger.info(f"Starting execution {execution_id} for pipeline '{pipeline.id}'")
            
            # Create comprehensive execution manager
            execution_manager = create_comprehensive_execution_manager(
                execution_id=execution_id,
                pipeline_id=pipeline.id
            )
            
            # Add execution context if provided
            if context:
                for key, value in context.items():
                    execution_manager.variable_manager.set_variable(key, value)
            
            # Add pipeline context to execution
            if pipeline.context:
                for key, value in pipeline.context.items():
                    execution_manager.variable_manager.set_variable(key, value)
            
            # Store active execution
            self._active_executions[execution_id] = execution_manager
            
            # Start execution with step count
            total_steps = len(pipeline.tasks)
            execution_manager.start_execution(total_steps)
            
            logger.info(f"Execution {execution_id} initialized with {total_steps} steps")
            
            return execution_manager
            
        except CompilationError:
            # Re-raise compilation errors as-is
            raise
        except Exception as e:
            logger.error(f"Failed to initialize pipeline execution: {e}")
            raise ExecutionError(f"Execution initialization failed: {e}") from e
    
    def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """
        Get comprehensive status for a pipeline execution.
        
        Args:
            execution_id: Unique execution identifier
            
        Returns:
            Dictionary containing execution status, progress, metrics, and recovery info
            
        Raises:
            ExecutionError: If execution not found
        """
        if execution_id not in self._active_executions:
            raise ExecutionError(f"Execution not found: {execution_id}")
        
        execution_manager = self._active_executions[execution_id]
        return execution_manager.get_execution_status()
    
    def stop_execution(self, execution_id: str, graceful: bool = True) -> bool:
        """
        Stop a running pipeline execution.
        
        Args:
            execution_id: Unique execution identifier
            graceful: Whether to wait for current step to complete
            
        Returns:
            True if execution was stopped successfully
            
        Raises:
            ExecutionError: If execution not found
        """
        if execution_id not in self._active_executions:
            raise ExecutionError(f"Execution not found: {execution_id}")
        
        execution_manager = self._active_executions[execution_id]
        
        try:
            logger.info(f"Stopping execution {execution_id} (graceful={graceful})")
            
            # Complete execution as failed/stopped
            execution_manager.complete_execution(success=False)
            
            # Clean up execution
            execution_manager.cleanup()
            
            # Remove from active executions
            del self._active_executions[execution_id]
            
            logger.info(f"Execution {execution_id} stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop execution {execution_id}: {e}")
            return False
    
    def list_active_executions(self) -> List[str]:
        """
        Get list of currently active execution IDs.
        
        Returns:
            List of active execution identifiers
        """
        return list(self._active_executions.keys())
    
    def get_compilation_report(self) -> Optional[Dict[str, Any]]:
        """
        Get the validation report from the last compilation.
        
        Returns:
            Dictionary containing validation results, or None if no report available
        """
        validation_report = self.compiler.get_validation_report()
        if validation_report:
            try:
                from ..validation.validation_report import OutputFormat
                return {
                    "summary": validation_report.format_report(OutputFormat.SUMMARY),
                    "details": validation_report.format_report(OutputFormat.JSON),
                    "stats": {
                        "total_issues": validation_report.stats.total_issues,
                        "errors": validation_report.stats.errors,
                        "warnings": validation_report.stats.warnings,
                        "info": validation_report.stats.info
                    },
                    "has_errors": validation_report.has_errors,
                    "has_warnings": validation_report.has_warnings
                }
            except ImportError:
                # Fallback if validation report module not available
                return None
        return None
    
    def validate_yaml(self, yaml_content: Union[str, Path]) -> bool:
        """
        Validate YAML pipeline specification without full compilation.
        
        Args:
            yaml_content: YAML content as string or path to YAML file
            
        Returns:
            True if YAML is valid, False otherwise
        """
        try:
            # Handle file path input
            if isinstance(yaml_content, (str, Path)) and Path(yaml_content).exists():
                yaml_file = Path(yaml_content)
                yaml_content = yaml_file.read_text(encoding='utf-8')
            elif isinstance(yaml_content, Path):
                return False  # File doesn't exist
            
            return self.compiler.validate_yaml(yaml_content)
            
        except Exception as e:
            logger.debug(f"YAML validation failed: {e}")
            return False
    
    def get_template_variables(self, yaml_content: Union[str, Path]) -> List[str]:
        """
        Extract template variables from YAML pipeline specification.
        
        Args:
            yaml_content: YAML content as string or path to YAML file
            
        Returns:
            List of template variable names found in the YAML
        """
        try:
            # Handle file path input
            if isinstance(yaml_content, (str, Path)) and Path(yaml_content).exists():
                yaml_file = Path(yaml_content)
                yaml_content = yaml_file.read_text(encoding='utf-8')
            elif isinstance(yaml_content, Path):
                return []  # File doesn't exist
            
            return self.compiler.get_template_variables(yaml_content)
            
        except Exception as e:
            logger.debug(f"Template variable extraction failed: {e}")
            return []
    
    def cleanup_execution(self, execution_id: str) -> bool:
        """
        Clean up resources for a completed or failed execution.
        
        Args:
            execution_id: Unique execution identifier
            
        Returns:
            True if cleanup was successful
        """
        if execution_id not in self._active_executions:
            logger.warning(f"Execution {execution_id} not found for cleanup")
            return False
        
        try:
            execution_manager = self._active_executions[execution_id]
            execution_manager.cleanup()
            del self._active_executions[execution_id]
            
            logger.info(f"Execution {execution_id} cleaned up successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup execution {execution_id}: {e}")
            return False
    
    def shutdown(self) -> None:
        """
        Shutdown the API and clean up all active executions.
        
        This should be called when the application is shutting down to ensure
        proper cleanup of resources and graceful termination of executions.
        """
        logger.info("Shutting down PipelineAPI...")
        
        # Stop all active executions
        active_ids = list(self._active_executions.keys())
        for execution_id in active_ids:
            try:
                self.stop_execution(execution_id, graceful=False)
            except Exception as e:
                logger.error(f"Error stopping execution {execution_id} during shutdown: {e}")
        
        # Shutdown execution managers
        for execution_manager in list(self._active_executions.values()):
            try:
                execution_manager.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down execution manager: {e}")
        
        self._active_executions.clear()
        
        logger.info("PipelineAPI shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.shutdown()


# Convenience function for quick API access
def create_pipeline_api(
    model_registry: Optional[ModelRegistry] = None,
    development_mode: bool = False,
    validation_level: str = "strict"
) -> PipelineAPI:
    """
    Create a PipelineAPI instance with the specified configuration.
    
    Args:
        model_registry: Optional model registry for AUTO tag resolution
        development_mode: Enable development mode (relaxed validation)
        validation_level: Validation strictness ("strict", "permissive", "development")
        
    Returns:
        Configured PipelineAPI instance
    """
    return PipelineAPI(
        model_registry=model_registry,
        development_mode=development_mode,
        validation_level=validation_level
    )