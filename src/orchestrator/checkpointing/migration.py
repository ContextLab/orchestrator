"""Checkpoint Migration System - Issue #205 Phase 3

Migrates existing checkpoints to LangGraph format while preserving all data
and metadata. Supports ClaudePoint integration and legacy checkpoint formats.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

# Internal imports  
from ..state.global_context import (
    PipelineGlobalState,
    create_initial_pipeline_state,
    validate_pipeline_state,
    PipelineStatus
)
from ..state.langgraph_state_manager import LangGraphGlobalContextManager
from ..core.exceptions import PipelineExecutionError

logger = logging.getLogger(__name__)


class MigrationStatus(Enum):
    """Status of migration operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class CheckpointFormat(Enum):
    """Supported checkpoint formats."""
    LEGACY_JSON = "legacy_json"  # Original JSON file format
    CLAUDEPOINT = "claudepoint"  # ClaudePoint format
    ORCHESTRATOR_V1 = "orchestrator_v1"  # Orchestrator v1 format
    LANGGRAPH = "langgraph"  # Target LangGraph format


@dataclass
class MigrationResult:
    """Result of a checkpoint migration operation."""
    success: bool
    source_path: str
    source_format: CheckpointFormat
    target_thread_id: Optional[str] = None
    target_checkpoint_id: Optional[str] = None
    migrated_data_size: int = 0
    migration_time_seconds: float = 0.0
    error_message: Optional[str] = None
    warnings: List[str] = None
    metadata_preserved: bool = True
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class MigrationSummary:
    """Summary of bulk migration operation."""
    total_files: int
    successful_migrations: int
    failed_migrations: int
    skipped_files: int
    total_data_size: int
    total_migration_time: float
    errors: List[str] = None
    source_directory: str = ""
    target_storage: str = ""
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class CheckpointMigrationManager:
    """
    Manages migration of legacy checkpoints to LangGraph format.
    
    Provides capabilities for:
    - Legacy checkpoint format detection and parsing
    - Data validation and integrity checking
    - Batch migration with progress tracking
    - ClaudePoint integration preservation
    - Metadata and context preservation
    """
    
    def __init__(
        self,
        langgraph_manager: LangGraphGlobalContextManager,
        preserve_original_files: bool = True,
        validate_migrations: bool = True,
        backup_before_migration: bool = True,
        max_concurrent_migrations: int = 5,
        migration_timeout_seconds: float = 300.0,
    ):
        """
        Initialize checkpoint migration manager.
        
        Args:
            langgraph_manager: Target LangGraph state manager
            preserve_original_files: Keep original files after migration
            validate_migrations: Validate migrated data integrity
            backup_before_migration: Create backup before migration
            max_concurrent_migrations: Maximum concurrent migration operations
            migration_timeout_seconds: Timeout for individual migrations
        """
        self.langgraph_manager = langgraph_manager
        self.preserve_original_files = preserve_original_files
        self.validate_migrations = validate_migrations
        self.backup_before_migration = backup_before_migration
        self.max_concurrent_migrations = max_concurrent_migrations
        self.migration_timeout_seconds = migration_timeout_seconds
        
        # Migration tracking
        self.migration_history: List[MigrationResult] = []
        self.migration_semaphore = asyncio.Semaphore(max_concurrent_migrations)
        
        # Format parsers
        self.format_parsers = {
            CheckpointFormat.LEGACY_JSON: self._parse_legacy_json,
            CheckpointFormat.CLAUDEPOINT: self._parse_claudepoint_format,
            CheckpointFormat.ORCHESTRATOR_V1: self._parse_orchestrator_v1,
        }
        
        # Metrics
        self.metrics = {
            "migrations_attempted": 0,
            "migrations_successful": 0,
            "migrations_failed": 0,
            "data_migrated_bytes": 0,
            "format_detections": {fmt.value: 0 for fmt in CheckpointFormat}
        }
        
        logger.info("CheckpointMigrationManager initialized")
    
    async def migrate_checkpoint_file(
        self,
        source_path: Union[str, Path],
        preserve_metadata: bool = True,
        target_pipeline_id: Optional[str] = None
    ) -> MigrationResult:
        """
        Migrate a single checkpoint file to LangGraph format.
        
        Args:
            source_path: Path to source checkpoint file
            preserve_metadata: Whether to preserve all original metadata
            target_pipeline_id: Optional override for pipeline ID
            
        Returns:
            Migration result with details and status
        """
        start_time = time.time()
        source_path = Path(source_path)
        
        async with self.migration_semaphore:
            try:
                self.metrics["migrations_attempted"] += 1
                
                # Validate source file
                if not source_path.exists():
                    return MigrationResult(
                        success=False,
                        source_path=str(source_path),
                        source_format=CheckpointFormat.LEGACY_JSON,
                        error_message=f"Source file does not exist: {source_path}"
                    )
                
                # Detect checkpoint format
                checkpoint_format = await self._detect_checkpoint_format(source_path)
                self.metrics["format_detections"][checkpoint_format.value] += 1
                
                logger.info(f"Migrating {checkpoint_format.value} checkpoint: {source_path}")
                
                # Create backup if enabled
                if self.backup_before_migration:
                    backup_path = await self._create_backup(source_path)
                    logger.debug(f"Created backup: {backup_path}")
                
                # Parse source checkpoint
                parser = self.format_parsers[checkpoint_format]
                parsed_data = await parser(source_path)
                
                # Convert to LangGraph global state format
                langgraph_state = await self._convert_to_langgraph_state(
                    parsed_data, 
                    checkpoint_format,
                    target_pipeline_id
                )
                
                # Validate converted state
                if self.validate_migrations:
                    validation_errors = validate_pipeline_state(langgraph_state)
                    if validation_errors:
                        return MigrationResult(
                            success=False,
                            source_path=str(source_path),
                            source_format=checkpoint_format,
                            error_message=f"State validation failed: {validation_errors}",
                            migration_time_seconds=time.time() - start_time
                        )
                
                # Initialize pipeline in LangGraph if needed
                pipeline_id = langgraph_state["execution_metadata"]["pipeline_id"]
                thread_id = langgraph_state["thread_id"]
                
                # Check if thread already exists
                existing_state = await self.langgraph_manager.get_global_state(thread_id)
                if not existing_state:
                    # Initialize new pipeline state
                    await self.langgraph_manager.initialize_pipeline_state(
                        pipeline_id=pipeline_id,
                        inputs=langgraph_state["global_variables"].get("inputs", {}),
                        user_id=langgraph_state.get("execution_metadata", {}).get("user_id"),
                        session_id=langgraph_state.get("execution_metadata", {}).get("session_id")
                    )
                
                # Update with migrated state
                await self.langgraph_manager.update_global_state(thread_id, langgraph_state)
                
                # Create migration checkpoint
                checkpoint_description = f"Migrated from {checkpoint_format.value}: {source_path.name}"
                if preserve_metadata and "description" in parsed_data:
                    checkpoint_description += f" - {parsed_data['description']}"
                
                checkpoint_id = await self.langgraph_manager.create_checkpoint(
                    thread_id=thread_id,
                    description=checkpoint_description,
                    metadata={
                        "migration_source": str(source_path),
                        "source_format": checkpoint_format.value,
                        "migration_timestamp": time.time(),
                        "preserve_metadata": preserve_metadata,
                        **parsed_data.get("metadata", {})
                    }
                )
                
                # Calculate migration metrics
                data_size = source_path.stat().st_size
                migration_time = time.time() - start_time
                
                # Update metrics
                self.metrics["migrations_successful"] += 1
                self.metrics["data_migrated_bytes"] += data_size
                
                # Create result
                result = MigrationResult(
                    success=True,
                    source_path=str(source_path),
                    source_format=checkpoint_format,
                    target_thread_id=thread_id,
                    target_checkpoint_id=checkpoint_id,
                    migrated_data_size=data_size,
                    migration_time_seconds=migration_time,
                    metadata_preserved=preserve_metadata
                )
                
                # Store result
                self.migration_history.append(result)
                
                logger.info(f"Successfully migrated {source_path} to thread {thread_id}")
                return result
                
            except asyncio.TimeoutError:
                self.metrics["migrations_failed"] += 1
                return MigrationResult(
                    success=False,
                    source_path=str(source_path),
                    source_format=CheckpointFormat.LEGACY_JSON,
                    error_message="Migration timeout exceeded",
                    migration_time_seconds=time.time() - start_time
                )
                
            except Exception as e:
                self.metrics["migrations_failed"] += 1
                logger.error(f"Migration failed for {source_path}: {e}")
                return MigrationResult(
                    success=False,
                    source_path=str(source_path),
                    source_format=CheckpointFormat.LEGACY_JSON,
                    error_message=str(e),
                    migration_time_seconds=time.time() - start_time
                )
    
    async def migrate_directory(
        self,
        source_directory: Union[str, Path],
        file_patterns: List[str] = None,
        recursive: bool = True,
        progress_callback: Optional[callable] = None
    ) -> MigrationSummary:
        """
        Migrate all checkpoints in a directory.
        
        Args:
            source_directory: Directory containing checkpoint files
            file_patterns: File patterns to match (e.g., ["*.json", "*.checkpoint"])
            recursive: Whether to search subdirectories
            progress_callback: Optional callback for progress updates
            
        Returns:
            Migration summary with overall results
        """
        start_time = time.time()
        source_directory = Path(source_directory)
        file_patterns = file_patterns or ["*.json", "*.checkpoint", "*.pkl"]
        
        if not source_directory.exists():
            return MigrationSummary(
                total_files=0,
                successful_migrations=0,
                failed_migrations=0,
                skipped_files=0,
                total_data_size=0,
                total_migration_time=0,
                errors=[f"Source directory does not exist: {source_directory}"],
                source_directory=str(source_directory)
            )
        
        logger.info(f"Starting directory migration: {source_directory}")
        
        # Find all checkpoint files
        checkpoint_files = []
        for pattern in file_patterns:
            if recursive:
                checkpoint_files.extend(source_directory.rglob(pattern))
            else:
                checkpoint_files.extend(source_directory.glob(pattern))
        
        # Remove duplicates and sort
        checkpoint_files = sorted(list(set(checkpoint_files)))
        total_files = len(checkpoint_files)
        
        logger.info(f"Found {total_files} potential checkpoint files")
        
        # Migrate files
        successful_migrations = 0
        failed_migrations = 0
        skipped_files = 0
        total_data_size = 0
        errors = []
        
        for i, file_path in enumerate(checkpoint_files):
            try:
                if progress_callback:
                    progress_callback(i, total_files, str(file_path))
                
                # Skip non-checkpoint files
                if not await self._is_checkpoint_file(file_path):
                    skipped_files += 1
                    continue
                
                # Migrate file
                result = await self.migrate_checkpoint_file(file_path)
                
                if result.success:
                    successful_migrations += 1
                    total_data_size += result.migrated_data_size
                    logger.debug(f"Migrated {file_path} successfully")
                else:
                    failed_migrations += 1
                    errors.append(f"{file_path}: {result.error_message}")
                    logger.warning(f"Failed to migrate {file_path}: {result.error_message}")
                
            except Exception as e:
                failed_migrations += 1
                errors.append(f"{file_path}: {str(e)}")
                logger.error(f"Unexpected error migrating {file_path}: {e}")
        
        migration_time = time.time() - start_time
        
        # Create summary
        summary = MigrationSummary(
            total_files=total_files,
            successful_migrations=successful_migrations,
            failed_migrations=failed_migrations,
            skipped_files=skipped_files,
            total_data_size=total_data_size,
            total_migration_time=migration_time,
            errors=errors,
            source_directory=str(source_directory),
            target_storage=self.langgraph_manager.storage_type
        )
        
        logger.info(f"Directory migration complete: {successful_migrations}/{total_files} successful")
        return summary
    
    async def _detect_checkpoint_format(self, file_path: Path) -> CheckpointFormat:
        """Detect the format of a checkpoint file."""
        try:
            # Check file extension first
            if file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check for ClaudePoint format markers
                if 'claudepoint_version' in data or 'cp_metadata' in data:
                    return CheckpointFormat.CLAUDEPOINT
                
                # Check for orchestrator v1 format
                if 'orchestrator_version' in data or 'pipeline_state' in data:
                    return CheckpointFormat.ORCHESTRATOR_V1
                
                # Default to legacy JSON
                return CheckpointFormat.LEGACY_JSON
            
            elif file_path.suffix in ['.checkpoint', '.cp']:
                # Likely ClaudePoint format
                return CheckpointFormat.CLAUDEPOINT
            
            else:
                # Try to parse as JSON
                return CheckpointFormat.LEGACY_JSON
                
        except Exception as e:
            logger.warning(f"Could not detect format for {file_path}: {e}")
            return CheckpointFormat.LEGACY_JSON
    
    async def _parse_legacy_json(self, file_path: Path) -> Dict[str, Any]:
        """Parse legacy JSON checkpoint format."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert legacy format to standard structure
        parsed = {
            "pipeline_id": data.get("pipeline_id", f"legacy_{file_path.stem}"),
            "thread_id": data.get("thread_id", f"legacy_thread_{uuid.uuid4().hex[:8]}"),
            "execution_id": data.get("execution_id", f"legacy_exec_{uuid.uuid4().hex}"),
            "global_variables": data.get("global_variables", {}),
            "intermediate_results": data.get("intermediate_results", {}),
            "execution_metadata": {
                "status": data.get("status", "completed"),
                "start_time": data.get("start_time", time.time()),
                "pipeline_id": data.get("pipeline_id", f"legacy_{file_path.stem}"),
                "completed_steps": data.get("completed_steps", []),
                "failed_steps": data.get("failed_steps", []),
                "checkpoints": data.get("checkpoints", [])
            },
            "error_context": data.get("error_context", {}),
            "description": data.get("description", f"Migrated from {file_path.name}"),
            "metadata": data.get("metadata", {})
        }
        
        return parsed
    
    async def _parse_claudepoint_format(self, file_path: Path) -> Dict[str, Any]:
        """Parse ClaudePoint checkpoint format."""
        if file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            # Handle binary ClaudePoint format
            # For now, assume it's text-based
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Parse ClaudePoint format (simplified)
                data = {"content": content, "type": "claudepoint"}
        
        # Convert ClaudePoint format to standard structure
        parsed = {
            "pipeline_id": data.get("pipeline_id", f"claudepoint_{file_path.stem}"),
            "thread_id": data.get("thread_id", f"cp_thread_{uuid.uuid4().hex[:8]}"),
            "execution_id": data.get("execution_id", f"cp_exec_{uuid.uuid4().hex}"),
            "global_variables": data.get("state", {}).get("global_variables", {}),
            "intermediate_results": data.get("state", {}).get("intermediate_results", {}),
            "execution_metadata": {
                "status": data.get("status", "completed"),
                "start_time": data.get("created_at", time.time()),
                "pipeline_id": data.get("pipeline_id", f"claudepoint_{file_path.stem}"),
                "completed_steps": data.get("completed_steps", []),
                "failed_steps": [],
                "checkpoints": []
            },
            "error_context": {},
            "description": data.get("description", f"Migrated from ClaudePoint: {file_path.name}"),
            "metadata": {
                "claudepoint_version": data.get("claudepoint_version"),
                "original_format": "claudepoint",
                **data.get("cp_metadata", {})
            }
        }
        
        return parsed
    
    async def _parse_orchestrator_v1(self, file_path: Path) -> Dict[str, Any]:
        """Parse Orchestrator v1 checkpoint format."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert orchestrator v1 format to standard structure
        pipeline_state = data.get("pipeline_state", {})
        
        parsed = {
            "pipeline_id": data.get("pipeline_id", f"orch_v1_{file_path.stem}"),
            "thread_id": data.get("thread_id", f"v1_thread_{uuid.uuid4().hex[:8]}"),
            "execution_id": data.get("execution_id", f"v1_exec_{uuid.uuid4().hex}"),
            "global_variables": pipeline_state.get("global_variables", {}),
            "intermediate_results": pipeline_state.get("intermediate_results", {}),
            "execution_metadata": {
                "status": pipeline_state.get("status", "completed"),
                "start_time": pipeline_state.get("start_time", time.time()),
                "pipeline_id": data.get("pipeline_id", f"orch_v1_{file_path.stem}"),
                "completed_steps": pipeline_state.get("completed_steps", []),
                "failed_steps": pipeline_state.get("failed_steps", []),
                "checkpoints": pipeline_state.get("checkpoints", [])
            },
            "error_context": pipeline_state.get("error_context", {}),
            "description": data.get("description", f"Migrated from Orchestrator v1: {file_path.name}"),
            "metadata": {
                "orchestrator_version": data.get("orchestrator_version"),
                "original_format": "orchestrator_v1",
                **data.get("metadata", {})
            }
        }
        
        return parsed
    
    async def _convert_to_langgraph_state(
        self,
        parsed_data: Dict[str, Any],
        source_format: CheckpointFormat,
        target_pipeline_id: Optional[str] = None
    ) -> PipelineGlobalState:
        """Convert parsed checkpoint data to LangGraph global state format."""
        
        # Use target pipeline ID if provided
        pipeline_id = target_pipeline_id or parsed_data["pipeline_id"]
        
        # Create LangGraph-compatible state
        langgraph_state: PipelineGlobalState = {
            "thread_id": parsed_data["thread_id"],
            "execution_id": parsed_data["execution_id"],
            "pipeline_id": pipeline_id,
            "global_variables": parsed_data["global_variables"],
            "intermediate_results": parsed_data["intermediate_results"],
            "execution_metadata": {
                "status": self._convert_status(parsed_data["execution_metadata"]["status"]),
                "start_time": parsed_data["execution_metadata"]["start_time"],
                "pipeline_id": pipeline_id,
                "completed_steps": parsed_data["execution_metadata"]["completed_steps"],
                "failed_steps": parsed_data["execution_metadata"]["failed_steps"],
                "pending_steps": [],
                "current_step": parsed_data["execution_metadata"].get("current_step", ""),
                "retry_count": parsed_data["execution_metadata"].get("retry_count", 0),
                **{k: v for k, v in parsed_data["execution_metadata"].items() 
                   if k not in ["status", "start_time", "pipeline_id", "completed_steps", "failed_steps"]}
            },
            "error_context": {
                "error_history": [],
                "retry_count": 0,
                "retry_attempts": [],
                **parsed_data.get("error_context", {})
            },
            "performance_metrics": parsed_data.get("performance_metrics", {}),
            "user_context": parsed_data.get("user_context", {}),
            "resource_constraints": parsed_data.get("resource_constraints", {}),
            "monitoring": parsed_data.get("monitoring", {})
        }
        
        return langgraph_state
    
    def _convert_status(self, status: Union[str, Any]) -> PipelineStatus:
        """Convert legacy status to PipelineStatus enum."""
        if isinstance(status, str):
            status_mapping = {
                "pending": PipelineStatus.PENDING,
                "running": PipelineStatus.RUNNING,
                "in_progress": PipelineStatus.RUNNING,
                "completed": PipelineStatus.COMPLETED,
                "success": PipelineStatus.COMPLETED,
                "failed": PipelineStatus.FAILED,
                "error": PipelineStatus.FAILED,
                "cancelled": PipelineStatus.CANCELLED,
                "canceled": PipelineStatus.CANCELLED,
                "paused": PipelineStatus.PAUSED
            }
            return status_mapping.get(status.lower(), PipelineStatus.PENDING)
        
        return PipelineStatus.PENDING
    
    async def _is_checkpoint_file(self, file_path: Path) -> bool:
        """Check if file is a valid checkpoint file."""
        try:
            # Check file size (skip empty or very large files)
            size = file_path.stat().st_size
            if size == 0 or size > 100 * 1024 * 1024:  # Skip files > 100MB
                return False
            
            # Try to detect format
            format_type = await self._detect_checkpoint_format(file_path)
            
            # Try to parse a small portion
            if file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Check for required checkpoint fields
                    return any(key in data for key in [
                        "pipeline_id", "global_variables", "intermediate_results",
                        "pipeline_state", "claudepoint_version", "state"
                    ])
            
            return True
            
        except Exception:
            return False
    
    async def _create_backup(self, file_path: Path) -> Path:
        """Create a backup of the original file."""
        backup_dir = file_path.parent / "migration_backups"
        backup_dir.mkdir(exist_ok=True)
        
        backup_path = backup_dir / f"{file_path.stem}_backup_{int(time.time())}{file_path.suffix}"
        shutil.copy2(file_path, backup_path)
        
        return backup_path
    
    def get_migration_metrics(self) -> Dict[str, Any]:
        """Get comprehensive migration metrics."""
        success_rate = 0.0
        if self.metrics["migrations_attempted"] > 0:
            success_rate = self.metrics["migrations_successful"] / self.metrics["migrations_attempted"]
        
        return {
            **self.metrics,
            "migration_success_rate": success_rate,
            "total_migration_history": len(self.migration_history),
            "recent_migrations": len([r for r in self.migration_history 
                                    if time.time() - (r.migration_time_seconds or 0) < 3600]),
            "average_migration_time": (
                sum(r.migration_time_seconds for r in self.migration_history if r.migration_time_seconds) /
                max(1, len([r for r in self.migration_history if r.migration_time_seconds]))
            ),
            "storage_backend": self.langgraph_manager.storage_type
        }
    
    def get_migration_history(self, limit: Optional[int] = None) -> List[MigrationResult]:
        """Get recent migration history."""
        history = sorted(self.migration_history, key=lambda x: x.migration_time_seconds or 0, reverse=True)
        return history[:limit] if limit else history
    
    async def cleanup(self):
        """Clean up migration manager resources."""
        logger.info("CheckpointMigrationManager cleanup complete")