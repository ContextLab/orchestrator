"""Integration Tools and Enhanced Utilities - Issue #205 Phase 3

Provides enhanced tools, CLI utilities, monitoring interfaces, and API endpoints
for comprehensive LangGraph checkpoint management and integration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

# Internal imports
from .migration import CheckpointMigrationManager, MigrationResult, MigrationSummary
from .performance_optimizer import PerformanceOptimizer, PerformanceMetrics
from .human_interaction import HumanInteractionManager, InteractionType, SessionStatus
from .branching import CheckpointBranchingManager, BranchStatus, MergeStrategy
from .enhanced_recovery import EnhancedRecoveryManager, FailureCategory, RecoveryStrategy
from ..state.langgraph_state_manager import LangGraphGlobalContextManager
from ..state.global_context import PipelineGlobalState, PipelineStatus

logger = logging.getLogger(__name__)


class ToolOperation(Enum):
    """Available tool operations."""
    LIST_CHECKPOINTS = "list_checkpoints"
    INSPECT_CHECKPOINT = "inspect_checkpoint"
    CREATE_CHECKPOINT = "create_checkpoint"
    RESTORE_CHECKPOINT = "restore_checkpoint"
    MIGRATE_CHECKPOINTS = "migrate_checkpoints"
    ANALYZE_PERFORMANCE = "analyze_performance"
    MANAGE_BRANCHES = "manage_branches"
    MONITOR_HEALTH = "monitor_health"
    CLEANUP_STORAGE = "cleanup_storage"
    EXPORT_DATA = "export_data"


@dataclass
class CheckpointInfo:
    """Enhanced checkpoint information."""
    checkpoint_id: str
    thread_id: str
    description: str
    created_at: float
    data_size_bytes: int
    metadata: Dict[str, Any]
    pipeline_id: str
    status: str
    step_count: int
    is_compressed: bool
    health_status: str = "unknown"
    
    @property
    def created_at_iso(self) -> str:
        return datetime.fromtimestamp(self.created_at).isoformat()
    
    @property
    def data_size_mb(self) -> float:
        return self.data_size_bytes / (1024 * 1024)


@dataclass
class SystemHealth:
    """System health status information."""
    overall_status: str  # "healthy", "warning", "critical"
    active_sessions: int
    active_branches: int
    pending_migrations: int
    cache_utilization: float
    storage_usage_mb: float
    recent_failures: int
    performance_issues: List[str]
    recommendations: List[str]
    last_updated: float
    
    @property
    def last_updated_iso(self) -> str:
        return datetime.fromtimestamp(self.last_updated).isoformat()


class IntegratedCheckpointManager:
    """
    Integrated checkpoint management system combining all Phase 2 and 3 components.
    
    Provides a unified interface for:
    - Checkpoint CRUD operations with optimization
    - Migration from legacy formats
    - Performance monitoring and optimization
    - Human-in-the-loop interactions
    - Branch management and merging
    - Enhanced recovery and health monitoring
    """
    
    def __init__(
        self,
        langgraph_manager: LangGraphGlobalContextManager,
        enable_performance_optimization: bool = True,
        enable_migration_support: bool = True,
        enable_human_interaction: bool = True,
        enable_branching: bool = True,
        enable_enhanced_recovery: bool = True,
        auto_optimize_performance: bool = True,
    ):
        """
        Initialize integrated checkpoint manager.
        
        Args:
            langgraph_manager: Core LangGraph state manager
            enable_performance_optimization: Enable performance optimizations
            enable_migration_support: Enable checkpoint migration features
            enable_human_interaction: Enable human-in-the-loop features  
            enable_branching: Enable checkpoint branching features
            enable_enhanced_recovery: Enable enhanced recovery features
            auto_optimize_performance: Automatically optimize operations
        """
        self.langgraph_manager = langgraph_manager
        self.enable_performance_optimization = enable_performance_optimization
        self.enable_migration_support = enable_migration_support
        self.enable_human_interaction = enable_human_interaction
        self.enable_branching = enable_branching
        self.enable_enhanced_recovery = enable_enhanced_recovery
        self.auto_optimize_performance = auto_optimize_performance
        
        # Initialize component managers
        self.performance_optimizer: Optional[PerformanceOptimizer] = None
        self.migration_manager: Optional[CheckpointMigrationManager] = None
        self.human_interaction_manager: Optional[HumanInteractionManager] = None
        self.branching_manager: Optional[CheckpointBranchingManager] = None
        self.recovery_manager: Optional[EnhancedRecoveryManager] = None
        
        # Initialize enabled components
        self._initialize_components()
        
        # Integration metrics
        self.integration_metrics = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "component_usage": {
                "performance_optimizer": 0,
                "migration_manager": 0,
                "human_interaction": 0,
                "branching_manager": 0,
                "recovery_manager": 0
            },
            "operation_types": {},
            "average_response_time_ms": 0.0
        }
        
        logger.info("IntegratedCheckpointManager initialized")
    
    def _initialize_components(self):
        """Initialize enabled component managers."""
        try:
            if self.enable_performance_optimization:
                self.performance_optimizer = PerformanceOptimizer(
                    langgraph_manager=self.langgraph_manager,
                    enable_compression=True,
                    cache_size_mb=100.0,
                    max_concurrent_operations=10
                )
                logger.info("Performance optimizer enabled")
            
            if self.enable_migration_support:
                self.migration_manager = CheckpointMigrationManager(
                    langgraph_manager=self.langgraph_manager,
                    preserve_original_files=True,
                    validate_migrations=True
                )
                logger.info("Migration manager enabled")
            
            if self.enable_human_interaction:
                self.human_interaction_manager = HumanInteractionManager(
                    langgraph_manager=self.langgraph_manager,
                    enable_approval_workflows=True
                )
                logger.info("Human interaction manager enabled")
            
            if self.enable_branching:
                self.branching_manager = CheckpointBranchingManager(
                    langgraph_manager=self.langgraph_manager,
                    auto_cleanup_abandoned_branches=True
                )
                logger.info("Branching manager enabled")
            
            if self.enable_enhanced_recovery:
                self.recovery_manager = EnhancedRecoveryManager(
                    langgraph_manager=self.langgraph_manager,
                    performance_monitoring_enabled=True,
                    auto_recovery_enabled=True
                )
                logger.info("Enhanced recovery manager enabled")
        
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise
    
    async def create_optimized_checkpoint(
        self,
        thread_id: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        use_performance_optimization: bool = None
    ) -> str:
        """
        Create checkpoint with integrated optimizations.
        
        Args:
            thread_id: Thread identifier
            description: Checkpoint description
            metadata: Optional metadata
            use_performance_optimization: Override performance optimization setting
            
        Returns:
            Checkpoint ID
        """
        start_time = time.time()
        operation_type = ToolOperation.CREATE_CHECKPOINT.value
        
        try:
            self.integration_metrics["total_operations"] += 1
            self.integration_metrics["operation_types"][operation_type] = (
                self.integration_metrics["operation_types"].get(operation_type, 0) + 1
            )
            
            # Get current state
            current_state = await self.langgraph_manager.get_global_state(thread_id)
            if not current_state:
                raise ValueError(f"No state found for thread: {thread_id}")
            
            # Use performance optimization if enabled
            use_optimization = (
                use_performance_optimization if use_performance_optimization is not None
                else (self.auto_optimize_performance and self.performance_optimizer is not None)
            )
            
            if use_optimization and self.performance_optimizer:
                self.integration_metrics["component_usage"]["performance_optimizer"] += 1
                checkpoint_id = await self.performance_optimizer.optimize_checkpoint_creation(
                    thread_id=thread_id,
                    state=current_state,
                    description=description,
                    metadata=metadata
                )
            else:
                checkpoint_id = await self.langgraph_manager.create_checkpoint(
                    thread_id=thread_id,
                    description=description,
                    metadata=metadata
                )
            
            self.integration_metrics["successful_operations"] += 1
            self._update_response_time(time.time() - start_time)
            
            logger.info(f"Created optimized checkpoint {checkpoint_id} for thread {thread_id}")
            return checkpoint_id
            
        except Exception as e:
            self.integration_metrics["failed_operations"] += 1
            logger.error(f"Failed to create optimized checkpoint: {e}")
            raise
    
    async def get_checkpoint_info(self, checkpoint_id: str) -> Optional[CheckpointInfo]:
        """Get comprehensive checkpoint information."""
        try:
            # We need to search through all active threads to find the checkpoint
            # since we don't know which thread the checkpoint belongs to
            target_checkpoint = None
            thread_id = ""
            
            # First, try to find the checkpoint in any active session
            for active_thread_id in self.langgraph_manager.active_sessions.keys():
                checkpoints = await self.langgraph_manager.list_checkpoints(active_thread_id)
                for cp in checkpoints:
                    if cp.get("checkpoint_id") == checkpoint_id:
                        target_checkpoint = cp
                        thread_id = active_thread_id
                        break
                if target_checkpoint:
                    break
            
            # If not found in active sessions, the checkpoint might not exist
            if not target_checkpoint:
                logger.warning(f"Checkpoint {checkpoint_id} not found in any active thread")
                return None
            
            # Extract information
            metadata = target_checkpoint.get("metadata", {})
            
            # Get state size
            data_size_bytes = metadata.get("data_size_bytes", 0)
            if data_size_bytes == 0:
                # Estimate size if not stored
                state = await self.langgraph_manager.get_global_state(thread_id)
                if state:
                    try:
                        data_size_bytes = len(json.dumps(state, default=str).encode('utf-8'))
                    except Exception:
                        data_size_bytes = 1024  # Default size estimate
            
            # Check health status
            health_status = "healthy"
            if self.recovery_manager:
                try:
                    health_info = await self.recovery_manager.perform_health_check(checkpoint_id)
                    health_status = health_info.health_status.value
                except Exception as e:
                    logger.warning(f"Health check failed for {checkpoint_id}: {e}")
                    health_status = "unknown"
            
            # Get pipeline state to extract additional information
            current_state = await self.langgraph_manager.get_global_state(thread_id)
            
            return CheckpointInfo(
                checkpoint_id=checkpoint_id,
                thread_id=thread_id,
                description=target_checkpoint.get("description", ""),
                created_at=target_checkpoint.get("timestamp", time.time()),
                data_size_bytes=data_size_bytes,
                metadata=metadata,
                pipeline_id=current_state.get("pipeline_id", "") if current_state else "",
                status=current_state.get("execution_metadata", {}).get("status", "completed") if current_state else "completed",
                step_count=len(current_state.get("execution_metadata", {}).get("completed_steps", [])) if current_state else 0,
                is_compressed=metadata.get("compression_enabled", False),
                health_status=health_status
            )
            
        except Exception as e:
            logger.error(f"Failed to get checkpoint info for {checkpoint_id}: {e}")
            return None
    
    async def list_enhanced_checkpoints(
        self,
        thread_id: Optional[str] = None,
        limit: Optional[int] = None,
        include_health: bool = True
    ) -> List[CheckpointInfo]:
        """List checkpoints with enhanced information."""
        try:
            checkpoints = await self.langgraph_manager.list_checkpoints(thread_id)
            
            enhanced_checkpoints = []
            for cp in checkpoints[:limit] if limit else checkpoints:
                checkpoint_id = cp.get("checkpoint_id", "")
                if checkpoint_id:
                    info = await self.get_checkpoint_info(checkpoint_id)
                    if info:
                        enhanced_checkpoints.append(info)
            
            return enhanced_checkpoints
            
        except Exception as e:
            logger.error(f"Failed to list enhanced checkpoints: {e}")
            return []
    
    async def migrate_legacy_checkpoints(
        self,
        source_directory: Union[str, Path],
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> MigrationSummary:
        """Migrate legacy checkpoints with progress tracking."""
        if not self.migration_manager:
            raise RuntimeError("Migration support is disabled")
        
        try:
            self.integration_metrics["component_usage"]["migration_manager"] += 1
            
            summary = await self.migration_manager.migrate_directory(
                source_directory=source_directory,
                progress_callback=progress_callback
            )
            
            logger.info(f"Migration complete: {summary.successful_migrations}/{summary.total_files} successful")
            return summary
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
    
    async def start_human_inspection(
        self,
        thread_id: str,
        step_id: str,
        interaction_type: InteractionType = InteractionType.INSPECTION,
        user_id: Optional[str] = None
    ) -> str:
        """Start human inspection session."""
        if not self.human_interaction_manager:
            raise RuntimeError("Human interaction support is disabled")
        
        try:
            self.integration_metrics["component_usage"]["human_interaction"] += 1
            
            session = await self.human_interaction_manager.pause_for_inspection(
                thread_id=thread_id,
                step_id=step_id,
                interaction_type=interaction_type,
                user_id=user_id
            )
            
            logger.info(f"Started human inspection session: {session.session_id}")
            return session.session_id
            
        except Exception as e:
            logger.error(f"Failed to start human inspection: {e}")
            raise
    
    async def create_execution_branch(
        self,
        source_thread_id: str,
        source_checkpoint_id: str,
        branch_name: str,
        description: str = ""
    ) -> str:
        """Create execution branch from checkpoint."""
        if not self.branching_manager:
            raise RuntimeError("Branching support is disabled")
        
        try:
            self.integration_metrics["component_usage"]["branching_manager"] += 1
            
            branch_info = await self.branching_manager.create_branch(
                source_thread_id=source_thread_id,
                source_checkpoint_id=source_checkpoint_id,
                branch_name=branch_name,
                description=description
            )
            
            logger.info(f"Created execution branch: {branch_info.branch_id}")
            return branch_info.branch_id
            
        except Exception as e:
            logger.error(f"Failed to create execution branch: {e}")
            raise
    
    async def analyze_system_performance(self) -> Dict[str, Any]:
        """Analyze overall system performance."""
        performance_analysis = {
            "timestamp": time.time(),
            "integration_metrics": self.integration_metrics.copy(),
            "component_performance": {},
            "recommendations": [],
            "health_status": "healthy"
        }
        
        try:
            # Performance optimizer metrics
            if self.performance_optimizer:
                perf_summary = self.performance_optimizer.get_performance_summary()
                performance_analysis["component_performance"]["performance_optimizer"] = perf_summary
                
                # Add recommendations based on performance
                if perf_summary.get("cache_hit_rate", 0) < 0.5:
                    performance_analysis["recommendations"].append(
                        "Consider increasing cache size - low cache hit rate detected"
                    )
                
                if perf_summary.get("average_operation_time_ms", 0) > 1000:
                    performance_analysis["recommendations"].append(
                        "High operation latency detected - consider performance optimization"
                    )
            
            # Recovery manager metrics
            if self.recovery_manager:
                recovery_summary = self.recovery_manager.get_performance_summary()
                performance_analysis["component_performance"]["recovery_manager"] = recovery_summary
                
                failure_count = recovery_summary.get("failure_analysis", {}).get("total_failures_analyzed", 0)
                if failure_count > 10:
                    performance_analysis["health_status"] = "warning"
                    performance_analysis["recommendations"].append(
                        f"High failure rate detected ({failure_count} failures) - investigate root causes"
                    )
            
            # Human interaction metrics
            if self.human_interaction_manager:
                human_metrics = self.human_interaction_manager.get_metrics()
                performance_analysis["component_performance"]["human_interaction"] = human_metrics
                
                if human_metrics.get("expired_sessions", 0) > 5:
                    performance_analysis["recommendations"].append(
                        "Multiple expired human interaction sessions - consider increasing timeouts"
                    )
            
            # Branching metrics
            if self.branching_manager:
                branch_metrics = self.branching_manager.get_metrics()
                performance_analysis["component_performance"]["branching_manager"] = branch_metrics
                
                if branch_metrics.get("abandoned_branches", 0) > 10:
                    performance_analysis["recommendations"].append(
                        "High number of abandoned branches - review branch management practices"
                    )
            
            return performance_analysis
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            performance_analysis["health_status"] = "critical"
            performance_analysis["recommendations"].append(f"System analysis error: {e}")
            return performance_analysis
    
    async def get_system_health(self) -> SystemHealth:
        """Get comprehensive system health status."""
        try:
            # Collect health metrics from all components
            active_sessions = 0
            active_branches = 0
            pending_migrations = 0
            cache_utilization = 0.0
            storage_usage_mb = 0.0
            recent_failures = 0
            performance_issues = []
            recommendations = []
            
            if self.human_interaction_manager:
                active_sessions = len(self.human_interaction_manager.get_active_sessions())
            
            if self.branching_manager:
                active_branches = len(self.branching_manager.get_active_branches())
            
            if self.performance_optimizer:
                perf_summary = self.performance_optimizer.get_performance_summary()
                cache_utilization = perf_summary.get("cache_utilization", 0.0)
                storage_usage_mb = perf_summary.get("cache_size_mb", 0.0)
                
                if cache_utilization > 0.9:
                    performance_issues.append("Cache utilization > 90%")
                    recommendations.append("Consider increasing cache size or optimizing cache usage")
            
            if self.recovery_manager:
                recovery_summary = self.recovery_manager.get_performance_summary()
                recent_failures = recovery_summary.get("failure_analysis", {}).get("total_failures_analyzed", 0)
                
                if recent_failures > 5:
                    performance_issues.append(f"High failure rate: {recent_failures} recent failures")
                    recommendations.append("Investigate and address recurring failure patterns")
            
            # Determine overall status
            overall_status = "healthy"
            if performance_issues:
                overall_status = "warning"
            if recent_failures > 20 or cache_utilization > 0.95:
                overall_status = "critical"
            
            return SystemHealth(
                overall_status=overall_status,
                active_sessions=active_sessions,
                active_branches=active_branches,
                pending_migrations=pending_migrations,
                cache_utilization=cache_utilization,
                storage_usage_mb=storage_usage_mb,
                recent_failures=recent_failures,
                performance_issues=performance_issues,
                recommendations=recommendations,
                last_updated=time.time()
            )
            
        except Exception as e:
            logger.error(f"System health check failed: {e}")
            return SystemHealth(
                overall_status="critical",
                active_sessions=0,
                active_branches=0,
                pending_migrations=0,
                cache_utilization=0.0,
                storage_usage_mb=0.0,
                recent_failures=0,
                performance_issues=[f"Health check failed: {e}"],
                recommendations=["Investigate system health check failures"],
                last_updated=time.time()
            )
    
    async def optimize_system_storage(self) -> Dict[str, Any]:
        """Optimize system storage usage."""
        optimization_results = {
            "timestamp": time.time(),
            "operations_performed": [],
            "storage_saved_mb": 0.0,
            "performance_improvement": {},
            "recommendations": []
        }
        
        try:
            # Performance optimizer storage optimization
            if self.performance_optimizer:
                storage_stats = await self.performance_optimizer.optimize_storage_usage()
                optimization_results["operations_performed"].append("performance_cache_optimization")
                optimization_results["performance_improvement"]["cache_optimization"] = storage_stats
            
            # Recovery manager cleanup
            if self.recovery_manager:
                cleanup_stats = await self.recovery_manager.cleanup_old_data()
                optimization_results["operations_performed"].append("recovery_data_cleanup")
                optimization_results["performance_improvement"]["recovery_cleanup"] = cleanup_stats
            
            logger.info("System storage optimization completed")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Storage optimization failed: {e}")
            optimization_results["error"] = str(e)
            return optimization_results
    
    def _update_response_time(self, response_time: float):
        """Update average response time metric."""
        current_avg = self.integration_metrics["average_response_time_ms"]
        total_ops = self.integration_metrics["total_operations"]
        
        if total_ops == 1:
            self.integration_metrics["average_response_time_ms"] = response_time * 1000
        else:
            # Rolling average
            new_avg = ((current_avg * (total_ops - 1)) + (response_time * 1000)) / total_ops
            self.integration_metrics["average_response_time_ms"] = new_avg
    
    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get integration and usage metrics."""
        return {
            **self.integration_metrics,
            "enabled_components": {
                "performance_optimizer": self.performance_optimizer is not None,
                "migration_manager": self.migration_manager is not None,
                "human_interaction": self.human_interaction_manager is not None,
                "branching_manager": self.branching_manager is not None,
                "recovery_manager": self.recovery_manager is not None
            },
            "component_count": sum(1 for comp in [
                self.performance_optimizer,
                self.migration_manager,
                self.human_interaction_manager,
                self.branching_manager,
                self.recovery_manager
            ] if comp is not None)
        }
    
    async def export_system_data(
        self,
        output_file: Union[str, Path],
        include_checkpoints: bool = True,
        include_metrics: bool = True,
        include_health: bool = True
    ) -> Dict[str, Any]:
        """Export comprehensive system data."""
        export_data = {
            "timestamp": time.time(),
            "export_metadata": {
                "include_checkpoints": include_checkpoints,
                "include_metrics": include_metrics,
                "include_health": include_health
            }
        }
        
        try:
            if include_checkpoints:
                checkpoints = await self.list_enhanced_checkpoints(limit=100)
                export_data["checkpoints"] = [asdict(cp) for cp in checkpoints]
            
            if include_metrics:
                export_data["integration_metrics"] = self.get_integration_metrics()
                export_data["performance_analysis"] = await self.analyze_system_performance()
            
            if include_health:
                system_health = await self.get_system_health()
                export_data["system_health"] = asdict(system_health)
            
            # Write to file
            output_path = Path(output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"System data exported to {output_path}")
            return {"success": True, "output_file": str(output_path), "data_size": len(json.dumps(export_data))}
            
        except Exception as e:
            logger.error(f"Data export failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def shutdown(self):
        """Shutdown integrated checkpoint manager and all components."""
        logger.info("Shutting down IntegratedCheckpointManager...")
        
        shutdown_results = []
        
        # Shutdown all components
        if self.performance_optimizer:
            try:
                await self.performance_optimizer.shutdown()
                shutdown_results.append("performance_optimizer: success")
            except Exception as e:
                shutdown_results.append(f"performance_optimizer: error - {e}")
        
        if self.migration_manager:
            try:
                await self.migration_manager.cleanup()
                shutdown_results.append("migration_manager: success")
            except Exception as e:
                shutdown_results.append(f"migration_manager: error - {e}")
        
        if self.human_interaction_manager:
            try:
                await self.human_interaction_manager.shutdown()
                shutdown_results.append("human_interaction: success")
            except Exception as e:
                shutdown_results.append(f"human_interaction: error - {e}")
        
        if self.branching_manager:
            try:
                await self.branching_manager.shutdown()
                shutdown_results.append("branching_manager: success")
            except Exception as e:
                shutdown_results.append(f"branching_manager: error - {e}")
        
        if self.recovery_manager:
            try:
                await self.recovery_manager.shutdown()
                shutdown_results.append("recovery_manager: success")
            except Exception as e:
                shutdown_results.append(f"recovery_manager: error - {e}")
        
        logger.info(f"IntegratedCheckpointManager shutdown complete: {shutdown_results}")


class CheckpointCLITools:
    """
    Command-line interface tools for checkpoint management.
    
    Provides CLI utilities for common checkpoint operations.
    """
    
    def __init__(self, integrated_manager: IntegratedCheckpointManager):
        self.manager = integrated_manager
    
    async def cli_list_checkpoints(self, thread_id: Optional[str] = None, limit: int = 20) -> None:
        """CLI command to list checkpoints."""
        print(f"Listing checkpoints (limit: {limit})...")
        
        checkpoints = await self.manager.list_enhanced_checkpoints(thread_id=thread_id, limit=limit)
        
        if not checkpoints:
            print("No checkpoints found.")
            return
        
        print(f"\nFound {len(checkpoints)} checkpoints:")
        print("-" * 80)
        print(f"{'ID':<20} {'Thread':<20} {'Created':<20} {'Size':<10} {'Status':<10}")
        print("-" * 80)
        
        for cp in checkpoints:
            print(f"{cp.checkpoint_id[:18]:<20} {cp.thread_id[:18]:<20} {cp.created_at_iso[:19]:<20} {cp.data_size_mb:.1f}MB{'':<6} {cp.health_status:<10}")
    
    async def cli_system_health(self) -> None:
        """CLI command to show system health."""
        print("Checking system health...")
        
        health = await self.manager.get_system_health()
        
        print(f"\n{'='*50}")
        print(f"SYSTEM HEALTH STATUS: {health.overall_status.upper()}")
        print(f"{'='*50}")
        print(f"Active Sessions: {health.active_sessions}")
        print(f"Active Branches: {health.active_branches}")
        print(f"Cache Utilization: {health.cache_utilization:.1%}")
        print(f"Storage Usage: {health.storage_usage_mb:.1f} MB")
        print(f"Recent Failures: {health.recent_failures}")
        print(f"Last Updated: {health.last_updated_iso}")
        
        if health.performance_issues:
            print(f"\n⚠️  Performance Issues:")
            for issue in health.performance_issues:
                print(f"  - {issue}")
        
        if health.recommendations:
            print(f"\n💡 Recommendations:")
            for rec in health.recommendations:
                print(f"  - {rec}")
    
    async def cli_migrate_checkpoints(self, source_dir: str) -> None:
        """CLI command to migrate checkpoints."""
        print(f"Migrating checkpoints from: {source_dir}")
        
        def progress_callback(current: int, total: int, filename: str):
            progress = (current / total) * 100 if total > 0 else 0
            print(f"Progress: {progress:.1f}% - {filename}")
        
        summary = await self.manager.migrate_legacy_checkpoints(
            source_directory=source_dir,
            progress_callback=progress_callback
        )
        
        print(f"\nMigration Summary:")
        print(f"Total Files: {summary.total_files}")
        print(f"Successful: {summary.successful_migrations}")
        print(f"Failed: {summary.failed_migrations}")
        print(f"Skipped: {summary.skipped_files}")
        print(f"Data Size: {summary.total_data_size / (1024*1024):.1f} MB")
        print(f"Time: {summary.total_migration_time:.2f}s")
        
        if summary.errors:
            print(f"\nErrors ({len(summary.errors)}):")
            for error in summary.errors[:5]:  # Show first 5 errors
                print(f"  - {error}")


# Export main integration class for external use
__all__ = [
    "IntegratedCheckpointManager",
    "CheckpointInfo", 
    "SystemHealth",
    "ToolOperation",
    "CheckpointCLITools"
]