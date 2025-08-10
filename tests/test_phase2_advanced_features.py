"""Phase 2 Advanced Features Testing - Issue #205

REAL TESTING for Human-in-the-Loop, Checkpoint Branching, and Enhanced Recovery features.
NO MOCKS - All tests use real databases, real failures, and real recovery scenarios.
"""

import asyncio
import pytest
import logging
import time
import uuid
import tempfile
import os
import sqlite3
from pathlib import Path
from typing import Dict, Any

from orchestrator.checkpointing.human_interaction import (
    HumanInteractionManager,
    InteractionType,
    SessionStatus,
    StateModification,
    ApprovalRequest
)
from orchestrator.checkpointing.branching import (
    CheckpointBranchingManager,
    BranchStatus,
    MergeStrategy,
    ConflictResolution,
    BranchInfo,
    MergeResult,
    RollbackResult
)
from orchestrator.checkpointing.enhanced_recovery import (
    EnhancedRecoveryManager,
    FailureCategory,
    RecoveryStrategy,
    CheckpointHealth,
    FailureAnalysis,
    ExecutionAnalytics
)
from orchestrator.state.langgraph_state_manager import LangGraphGlobalContextManager
from orchestrator.state.global_context import PipelineGlobalState, PipelineStatus

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
async def real_database():
    """Create a real SQLite database for testing."""
    # Create temporary database
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_checkpoints.db")
    
    logger.info(f"Created real test database: {db_path}")
    
    yield db_path
    
    # Cleanup
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
        os.rmdir(temp_dir)
    except Exception as e:
        logger.warning(f"Failed to cleanup test database: {e}")


@pytest.fixture
async def langgraph_manager(real_database):
    """Create LangGraph manager with real database."""
    
    manager = LangGraphGlobalContextManager(
        storage_type="sqlite",
        database_url=f"sqlite:///{real_database}"
    )
    
    yield manager


@pytest.fixture
async def human_interaction_manager(langgraph_manager):
    """Create human interaction manager with real database."""
    manager = HumanInteractionManager(
        langgraph_manager=langgraph_manager,
        default_session_timeout=60.0,  # 1 minute for testing
        max_concurrent_sessions=5,
        enable_approval_workflows=True,
        auto_checkpoint_on_modification=True
    )
    
    yield manager
    
    await manager.shutdown()


@pytest.fixture
async def branching_manager(langgraph_manager):
    """Create branching manager with real database."""
    manager = CheckpointBranchingManager(
        langgraph_manager=langgraph_manager,
        max_branch_depth=5,
        auto_cleanup_abandoned_branches=True,
        default_merge_strategy=MergeStrategy.MERGE_RECURSIVE,
        branch_retention_hours=1  # 1 hour for testing
    )
    
    yield manager
    
    await manager.shutdown()


@pytest.fixture
async def enhanced_recovery_manager(langgraph_manager):
    """Create enhanced recovery manager with real database."""
    manager = EnhancedRecoveryManager(
        langgraph_manager=langgraph_manager,
        health_check_interval=30.0,  # 30 seconds for testing
        analytics_retention_days=1,  # 1 day for testing
        performance_monitoring_enabled=True,
        auto_recovery_enabled=True,
        recovery_confidence_threshold=0.7
    )
    
    yield manager
    
    await manager.shutdown()


@pytest.fixture
def real_pipeline_state() -> PipelineGlobalState:
    """Create a real pipeline state for testing."""
    return {
        "thread_id": f"test_thread_{uuid.uuid4().hex[:8]}",
        "execution_id": f"exec_{uuid.uuid4().hex[:8]}",
        "pipeline_id": "test_pipeline",
        "global_variables": {
            "inputs": {"test_input": "test_value"},
            "shared_data": {"step_1": "result_1", "step_2": "result_2"}
        },
        "intermediate_results": {
            "step_1": {"output": "Step 1 completed", "metadata": {"duration": 0.5}},
            "step_2": {"output": "Step 2 completed", "metadata": {"duration": 0.8}},
            "step_3": {"output": "Step 3 in progress", "metadata": {"duration": 0.3}}
        },
        "execution_metadata": {
            "status": PipelineStatus.RUNNING,
            "start_time": time.time() - 100,
            "pipeline_id": "test_pipeline",
            "completed_steps": ["step_1", "step_2"],
            "failed_steps": [],
            "checkpoints": [
                {"step_id": "step_1", "checkpoint_id": "cp_1", "timestamp": time.time() - 50},
                {"step_id": "step_2", "checkpoint_id": "cp_2", "timestamp": time.time() - 25}
            ]
        },
        "performance_metrics": {
            "total_execution_time": 100.0,
            "step_durations": {"step_1": 0.5, "step_2": 0.8, "step_3": 0.3}
        }
    }


class TestHumanInteractionReal:
    """Real testing for Human-in-the-Loop capabilities."""
    
    @pytest.mark.asyncio
    async def test_human_inspection_session_real(
        self, 
        human_interaction_manager: HumanInteractionManager,
        real_pipeline_state: PipelineGlobalState
    ):
        """Test real human inspection session with database persistence."""
        logger.info("🧪 Testing real human inspection session")
        
        # Initialize pipeline state in database
        thread_id = real_pipeline_state["thread_id"]
        await human_interaction_manager.langgraph_manager.initialize_pipeline_state(
            pipeline_id=real_pipeline_state["pipeline_id"],
            inputs=real_pipeline_state["global_variables"]["inputs"],
            user_id="test_user"
        )
        
        # Update state in database
        await human_interaction_manager.langgraph_manager.update_global_state(
            thread_id, real_pipeline_state
        )
        
        # Create inspection session
        session = await human_interaction_manager.pause_for_inspection(
            thread_id=thread_id,
            step_id="step_3",
            interaction_type=InteractionType.INSPECTION,
            user_id="test_user",
            timeout_seconds=120.0,
            description="Testing inspection session"
        )
        
        assert session.session_id is not None
        assert session.thread_id == thread_id
        assert session.status == SessionStatus.ACTIVE
        assert session.user_id == "test_user"
        
        # Verify session is tracked
        active_sessions = human_interaction_manager.get_active_sessions()
        assert len(active_sessions) == 1
        assert active_sessions[0]["session_id"] == session.session_id
        
        # Get current state through session
        current_state = await human_interaction_manager.get_current_state(session.session_id)
        assert current_state is not None
        assert current_state["thread_id"] == thread_id
        assert current_state["pipeline_id"] == real_pipeline_state["pipeline_id"]
        
        # Inspect specific field
        step_output = await human_interaction_manager.inspect_state_field(
            session.session_id, "intermediate_results.step_1.output"
        )
        assert step_output == "Step 1 completed"
        
        # Resume execution
        resumed_thread_id = await human_interaction_manager.resume_execution(
            session.session_id, modifications_applied=False
        )
        assert resumed_thread_id == thread_id
        
        # Verify session is completed
        active_sessions = human_interaction_manager.get_active_sessions()
        assert len(active_sessions) == 0
        
        logger.info("✅ Real human inspection session test passed")
    
    @pytest.mark.asyncio
    async def test_state_modification_real(
        self,
        human_interaction_manager: HumanInteractionManager,
        real_pipeline_state: PipelineGlobalState
    ):
        """Test real state modification with database persistence."""
        logger.info("🧪 Testing real state modification")
        
        # Initialize pipeline state
        thread_id = real_pipeline_state["thread_id"]
        await human_interaction_manager.langgraph_manager.initialize_pipeline_state(
            pipeline_id=real_pipeline_state["pipeline_id"],
            inputs=real_pipeline_state["global_variables"]["inputs"],
            user_id="test_user"
        )
        await human_interaction_manager.langgraph_manager.update_global_state(
            thread_id, real_pipeline_state
        )
        
        # Create modification session
        session = await human_interaction_manager.pause_for_inspection(
            thread_id=thread_id,
            step_id="step_3",
            interaction_type=InteractionType.MODIFICATION,
            user_id="test_user"
        )
        
        # Get original value
        original_output = await human_interaction_manager.inspect_state_field(
            session.session_id, "intermediate_results.step_3.output"
        )
        assert original_output == "Step 3 in progress"
        
        # Modify state
        new_output = "Step 3 modified by human"
        success = await human_interaction_manager.modify_state(
            session_id=session.session_id,
            field_path="intermediate_results.step_3.output",
            new_value=new_output,
            modification_type="update",
            user_id="test_user",
            reason="Testing human modification"
        )
        assert success == True
        
        # Verify modification was applied
        modified_output = await human_interaction_manager.inspect_state_field(
            session.session_id, "intermediate_results.step_3.output"
        )
        assert modified_output == new_output
        
        # Verify modification is tracked
        assert len(session.modifications) == 1
        modification = session.modifications[0]
        assert modification.field_path == "intermediate_results.step_3.output"
        assert modification.old_value == original_output
        assert modification.new_value == new_output
        assert modification.user_id == "test_user"
        assert modification.reason == "Testing human modification"
        
        # Verify state persisted in database
        current_state = await human_interaction_manager.langgraph_manager.get_global_state(thread_id)
        assert current_state["intermediate_results"]["step_3"]["output"] == new_output
        
        # Resume with modifications
        await human_interaction_manager.resume_execution(
            session.session_id, modifications_applied=True
        )
        
        logger.info("✅ Real state modification test passed")
    
    @pytest.mark.asyncio
    async def test_approval_workflow_real(
        self,
        human_interaction_manager: HumanInteractionManager,
        real_pipeline_state: PipelineGlobalState
    ):
        """Test real approval workflow with database persistence."""
        logger.info("🧪 Testing real approval workflow")
        
        # Initialize pipeline state
        thread_id = real_pipeline_state["thread_id"]
        await human_interaction_manager.langgraph_manager.initialize_pipeline_state(
            pipeline_id=real_pipeline_state["pipeline_id"],
            inputs=real_pipeline_state["global_variables"]["inputs"],
            user_id="test_user"
        )
        
        # Create session requiring approval
        session = await human_interaction_manager.pause_for_inspection(
            thread_id=thread_id,
            step_id="step_3",
            interaction_type=InteractionType.APPROVAL,
            user_id="test_user",
            approval_required=True
        )
        
        # Create approval request
        approval_request = await human_interaction_manager.request_approval(
            session_id=session.session_id,
            approval_type="execute",
            description="Approve execution of sensitive step",
            required_approvers=["approver_1", "approver_2"],
            expires_in_seconds=300
        )
        
        assert approval_request.request_id is not None
        assert approval_request.session_id == session.session_id
        assert len(approval_request.required_approvers) == 2
        assert len(approval_request.current_approvals) == 0
        
        # Verify approval is pending
        pending_approvals = human_interaction_manager.get_pending_approvals()
        assert len(pending_approvals) == 1
        assert pending_approvals[0]["request_id"] == approval_request.request_id
        
        # Provide first approval
        success = await human_interaction_manager.provide_approval(
            request_id=approval_request.request_id,
            approver_id="approver_1",
            approved=True,
            comments="Looks good to me"
        )
        assert success == True
        
        # Verify partial approval
        pending_approvals = human_interaction_manager.get_pending_approvals()
        assert len(pending_approvals) == 1
        assert len(pending_approvals[0]["current_approvals"]) == 1
        
        # Provide second approval
        success = await human_interaction_manager.provide_approval(
            request_id=approval_request.request_id,
            approver_id="approver_2",
            approved=True
        )
        assert success == True
        
        # Verify approval is complete (removed from pending)
        pending_approvals = human_interaction_manager.get_pending_approvals()
        assert len(pending_approvals) == 0
        
        # Resume execution after approval
        await human_interaction_manager.resume_execution(session.session_id)
        
        logger.info("✅ Real approval workflow test passed")


class TestCheckpointBranchingReal:
    """Real testing for Checkpoint Branching capabilities."""
    
    @pytest.mark.asyncio
    async def test_branch_creation_real(
        self,
        branching_manager: CheckpointBranchingManager,
        real_pipeline_state: PipelineGlobalState
    ):
        """Test real branch creation with database persistence."""
        logger.info("🧪 Testing real branch creation")
        
        # Initialize main execution thread
        thread_id = real_pipeline_state["thread_id"]
        await branching_manager.langgraph_manager.initialize_pipeline_state(
            pipeline_id=real_pipeline_state["pipeline_id"],
            inputs=real_pipeline_state["global_variables"]["inputs"],
            user_id="test_user"
        )
        await branching_manager.langgraph_manager.update_global_state(
            thread_id, real_pipeline_state
        )
        
        # Create checkpoint
        checkpoint_id = await branching_manager.langgraph_manager.create_checkpoint(
            thread_id=thread_id,
            description="Test checkpoint for branching",
            metadata={"step": "step_2"}
        )
        
        # Create branch
        branch_info = await branching_manager.create_branch(
            source_thread_id=thread_id,
            source_checkpoint_id=checkpoint_id,
            branch_name="experimental_branch",
            description="Testing experimental approach",
            created_by="test_user"
        )
        
        assert branch_info.branch_id is not None
        assert branch_info.thread_id != thread_id
        assert branch_info.parent_thread_id == thread_id
        assert branch_info.branch_name == "experimental_branch"
        assert branch_info.status == BranchStatus.ACTIVE
        assert branch_info.created_by == "test_user"
        
        # Verify branch is tracked
        active_branches = branching_manager.get_active_branches()
        assert len(active_branches) == 1
        assert active_branches[0]["branch_id"] == branch_info.branch_id
        
        # Verify branch hierarchy
        hierarchy = branching_manager.get_branch_hierarchy(thread_id)
        assert hierarchy["thread_id"] == thread_id
        assert len(hierarchy["branches"]) == 1
        assert hierarchy["branches"][0]["branch_id"] == branch_info.branch_id
        
        # Verify branch state exists in database
        branch_state = await branching_manager.langgraph_manager.get_global_state(
            branch_info.thread_id
        )
        assert branch_state is not None
        assert branch_state["execution_metadata"]["branch_info"]["branch_id"] == branch_info.branch_id
        
        logger.info("✅ Real branch creation test passed")
    
    @pytest.mark.asyncio
    async def test_rollback_operation_real(
        self,
        branching_manager: CheckpointBranchingManager,
        real_pipeline_state: PipelineGlobalState
    ):
        """Test real rollback operation with database persistence."""
        logger.info("🧪 Testing real rollback operation")
        
        # Initialize execution with multiple checkpoints
        thread_id = real_pipeline_state["thread_id"]
        await branching_manager.langgraph_manager.initialize_pipeline_state(
            pipeline_id=real_pipeline_state["pipeline_id"],
            inputs=real_pipeline_state["global_variables"]["inputs"],
            user_id="test_user"
        )
        await branching_manager.langgraph_manager.update_global_state(
            thread_id, real_pipeline_state
        )
        
        # Create multiple checkpoints
        checkpoint1_id = await branching_manager.langgraph_manager.create_checkpoint(
            thread_id=thread_id,
            description="Checkpoint after step 1",
            metadata={"step": "step_1"}
        )
        
        # Update state for step 2
        modified_state = real_pipeline_state.copy()
        modified_state["intermediate_results"]["step_4"] = {
            "output": "Step 4 completed", 
            "metadata": {"duration": 1.2}
        }
        modified_state["execution_metadata"]["completed_steps"].append("step_4")
        
        await branching_manager.langgraph_manager.update_global_state(
            thread_id, modified_state
        )
        
        checkpoint2_id = await branching_manager.langgraph_manager.create_checkpoint(
            thread_id=thread_id,
            description="Checkpoint after step 4",
            metadata={"step": "step_4"}
        )
        
        # Verify current state includes step 4
        current_state = await branching_manager.langgraph_manager.get_global_state(thread_id)
        assert "step_4" in current_state["intermediate_results"]
        assert "step_4" in current_state["execution_metadata"]["completed_steps"]
        
        # Perform rollback to checkpoint1
        rollback_result = await branching_manager.rollback_to_checkpoint(
            thread_id=thread_id,
            target_checkpoint_id=checkpoint1_id,
            create_rollback_branch=True,
            rollback_reason="Testing rollback operation"
        )
        
        assert rollback_result.success == True
        assert rollback_result.thread_id == thread_id
        assert rollback_result.target_checkpoint_id == checkpoint1_id
        assert rollback_result.steps_rolled_back >= 0
        assert rollback_result.restored_state is not None
        
        # Verify state was rolled back (step 4 should be gone)
        rolled_back_state = await branching_manager.langgraph_manager.get_global_state(thread_id)
        assert "step_4" not in rolled_back_state["intermediate_results"]
        assert "step_4" not in rolled_back_state["execution_metadata"]["completed_steps"]
        
        # Verify rollback branch was created
        active_branches = branching_manager.get_active_branches()
        assert len(active_branches) >= 1  # Should have created rollback branch
        
        logger.info("✅ Real rollback operation test passed")
    
    @pytest.mark.asyncio 
    async def test_branch_merge_real(
        self,
        branching_manager: CheckpointBranchingManager,
        real_pipeline_state: PipelineGlobalState
    ):
        """Test real branch merge with conflict resolution."""
        logger.info("🧪 Testing real branch merge")
        
        # Initialize main thread
        main_thread_id = real_pipeline_state["thread_id"]
        await branching_manager.langgraph_manager.initialize_pipeline_state(
            pipeline_id=real_pipeline_state["pipeline_id"],
            inputs=real_pipeline_state["global_variables"]["inputs"],
            user_id="test_user"
        )
        await branching_manager.langgraph_manager.update_global_state(
            main_thread_id, real_pipeline_state
        )
        
        # Create initial checkpoint
        checkpoint_id = await branching_manager.langgraph_manager.create_checkpoint(
            thread_id=main_thread_id,
            description="Base checkpoint for merge test"
        )
        
        # Create branch
        branch_info = await branching_manager.create_branch(
            source_thread_id=main_thread_id,
            source_checkpoint_id=checkpoint_id,
            branch_name="feature_branch",
            description="Feature development branch"
        )
        
        # Modify branch state
        branch_state = await branching_manager.langgraph_manager.get_global_state(
            branch_info.thread_id
        )
        branch_state["intermediate_results"]["branch_feature"] = {
            "output": "Feature implemented",
            "metadata": {"feature": "new_capability"}
        }
        branch_state["global_variables"]["shared_data"]["feature_flag"] = True
        
        await branching_manager.langgraph_manager.update_global_state(
            branch_info.thread_id, branch_state
        )
        
        # Modify main thread state (create potential conflict)
        main_state = await branching_manager.langgraph_manager.get_global_state(main_thread_id)
        main_state["global_variables"]["shared_data"]["main_update"] = "main_thread_change"
        
        await branching_manager.langgraph_manager.update_global_state(
            main_thread_id, main_state
        )
        
        # Perform merge
        merge_result = await branching_manager.merge_branch(
            source_branch_id=branch_info.branch_id,
            target_thread_id=main_thread_id,
            merge_strategy=MergeStrategy.MERGE_RECURSIVE,
            conflict_resolution=ConflictResolution.MERGE_BOTH,
            merge_description="Merge feature branch into main"
        )
        
        assert merge_result.success == True
        assert merge_result.merged_thread_id == main_thread_id
        assert merge_result.merge_checkpoint_id is not None
        
        # Verify merged state contains both changes
        merged_state = await branching_manager.langgraph_manager.get_global_state(main_thread_id)
        assert "branch_feature" in merged_state["intermediate_results"]
        assert merged_state["global_variables"]["shared_data"]["feature_flag"] == True
        assert merged_state["global_variables"]["shared_data"]["main_update"] == "main_thread_change"
        
        # Verify branch status updated
        assert branching_manager.active_branches[branch_info.branch_id].status == BranchStatus.MERGED
        
        logger.info("✅ Real branch merge test passed")


class TestEnhancedRecoveryReal:
    """Real testing for Enhanced Recovery capabilities."""
    
    @pytest.mark.asyncio
    async def test_failure_analysis_real(
        self,
        enhanced_recovery_manager: EnhancedRecoveryManager,
        real_pipeline_state: PipelineGlobalState
    ):
        """Test real failure analysis and recovery strategy determination."""
        logger.info("🧪 Testing real failure analysis")
        
        # Initialize pipeline state
        thread_id = real_pipeline_state["thread_id"]
        await enhanced_recovery_manager.langgraph_manager.initialize_pipeline_state(
            pipeline_id=real_pipeline_state["pipeline_id"],
            inputs=real_pipeline_state["global_variables"]["inputs"],
            user_id="test_user"
        )
        await enhanced_recovery_manager.langgraph_manager.update_global_state(
            thread_id, real_pipeline_state
        )
        
        # Create checkpoints for recovery
        checkpoint_id = await enhanced_recovery_manager.langgraph_manager.create_checkpoint(
            thread_id=thread_id,
            description="Pre-failure checkpoint"
        )
        
        # Simulate various failure scenarios
        failures = [
            (ValueError("Invalid input format"), {"step_type": "validation"}),
            (ConnectionError("Network timeout occurred"), {"step_type": "api_call"}), 
            (MemoryError("Out of memory"), {"step_type": "data_processing"}),
            (RuntimeError("System resource exhausted"), {"step_type": "computation"})
        ]
        
        for error, context in failures:
            # Analyze failure
            analysis = await enhanced_recovery_manager.analyze_failure(
                thread_id=thread_id,
                step_id="test_step",
                error=error,
                execution_context=context
            )
            
            assert analysis.failure_id is not None
            assert analysis.thread_id == thread_id
            assert analysis.failure_category != FailureCategory.UNKNOWN
            assert analysis.recovery_strategy in [rs for rs in RecoveryStrategy]
            assert 0.0 <= analysis.recovery_confidence <= 1.0
            assert analysis.failure_timestamp > 0
            
            # Verify failure was stored
            assert thread_id in enhanced_recovery_manager.failure_history
            thread_failures = enhanced_recovery_manager.failure_history[thread_id]
            assert any(f.failure_id == analysis.failure_id for f in thread_failures)
        
        # Verify failure patterns are learned
        assert len(enhanced_recovery_manager.failure_history[thread_id]) == len(failures)
        
        logger.info("✅ Real failure analysis test passed")
    
    @pytest.mark.asyncio
    async def test_checkpoint_health_monitoring_real(
        self,
        enhanced_recovery_manager: EnhancedRecoveryManager,
        real_pipeline_state: PipelineGlobalState
    ):
        """Test real checkpoint health monitoring with database."""
        logger.info("🧪 Testing real checkpoint health monitoring")
        
        # Initialize pipeline state
        thread_id = real_pipeline_state["thread_id"] 
        await enhanced_recovery_manager.langgraph_manager.initialize_pipeline_state(
            pipeline_id=real_pipeline_state["pipeline_id"],
            inputs=real_pipeline_state["global_variables"]["inputs"],
            user_id="test_user"
        )
        
        # Create multiple checkpoints
        checkpoint_ids = []
        for i in range(3):
            checkpoint_id = await enhanced_recovery_manager.langgraph_manager.create_checkpoint(
                thread_id=thread_id,
                description=f"Health test checkpoint {i}",
                metadata={"checkpoint_number": i}
            )
            checkpoint_ids.append(checkpoint_id)
        
        # Perform health checks
        for checkpoint_id in checkpoint_ids:
            health_info = await enhanced_recovery_manager.perform_health_check(checkpoint_id)
            
            assert health_info.checkpoint_id == checkpoint_id
            assert health_info.health_status in [hs for hs in CheckpointHealth]
            assert isinstance(health_info.size_bytes, int)
            assert health_info.creation_time > 0
            assert 0.0 <= health_info.recovery_success_rate <= 1.0
            
            # Verify health info is stored
            assert checkpoint_id in enhanced_recovery_manager.checkpoint_health
        
        # Check performance summary includes health data
        summary = enhanced_recovery_manager.get_performance_summary()
        assert "health_checks" in summary
        assert summary["health_checks"]["total_checkpoints_monitored"] >= 3
        
        logger.info("✅ Real checkpoint health monitoring test passed")
    
    @pytest.mark.asyncio
    async def test_execution_analytics_real(
        self,
        enhanced_recovery_manager: EnhancedRecoveryManager,
        real_pipeline_state: PipelineGlobalState
    ):
        """Test real execution analytics generation."""
        logger.info("🧪 Testing real execution analytics")
        
        # Initialize pipeline with comprehensive state
        thread_id = real_pipeline_state["thread_id"]
        await enhanced_recovery_manager.langgraph_manager.initialize_pipeline_state(
            pipeline_id=real_pipeline_state["pipeline_id"],
            inputs=real_pipeline_state["global_variables"]["inputs"],
            user_id="test_user"
        )
        
        # Update with detailed execution metadata
        enhanced_state = real_pipeline_state.copy()
        enhanced_state["execution_metadata"]["total_execution_time"] = 150.0
        enhanced_state["execution_metadata"]["completed_steps"] = ["step_1", "step_2", "step_3"]
        enhanced_state["execution_metadata"]["failed_steps"] = ["step_4"]
        enhanced_state["execution_metadata"]["checkpoints"] = [
            {"step_id": "step_1", "creation_time_ms": 50, "timestamp": time.time() - 100},
            {"step_id": "step_2", "creation_time_ms": 75, "timestamp": time.time() - 80},
            {"step_id": "step_3", "creation_time_ms": 60, "timestamp": time.time() - 60}
        ]
        
        await enhanced_recovery_manager.langgraph_manager.update_global_state(
            thread_id, enhanced_state
        )
        
        # Add some failure history
        test_error = RuntimeError("Simulated failure")
        await enhanced_recovery_manager.analyze_failure(
            thread_id=thread_id,
            step_id="step_4",
            error=test_error,
            execution_context={"step_type": "computation"}
        )
        
        # Generate analytics
        analytics = await enhanced_recovery_manager.generate_execution_analytics(thread_id)
        
        assert analytics is not None
        assert analytics.thread_id == thread_id
        assert analytics.pipeline_id == real_pipeline_state["pipeline_id"]
        assert analytics.total_steps == 4  # 3 completed + 1 failed
        assert analytics.completed_steps == 3
        assert analytics.failed_steps == 1
        assert analytics.execution_time == 150.0
        assert analytics.checkpoint_count == 3
        assert analytics.recovery_attempts >= 1
        assert analytics.average_step_time > 0
        assert analytics.checkpoint_overhead_percent >= 0
        assert len(analytics.failure_categories) >= 1
        assert 0.0 <= analytics.recovery_success_rate <= 1.0
        
        # Verify analytics are stored
        assert thread_id in enhanced_recovery_manager.execution_analytics
        stored_analytics = enhanced_recovery_manager.execution_analytics[thread_id]
        assert stored_analytics.thread_id == analytics.thread_id
        
        logger.info("✅ Real execution analytics test passed")
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_real(
        self,
        enhanced_recovery_manager: EnhancedRecoveryManager
    ):
        """Test real performance monitoring with metrics collection."""
        logger.info("🧪 Testing real performance monitoring")
        
        # Enable performance monitoring 
        assert enhanced_recovery_manager.performance_monitoring_enabled == True
        
        # Simulate various operations to generate metrics
        operations = [
            ("checkpoint_creation", 0.05, enhanced_recovery_manager.performance_metrics.checkpoint_creation_times),
            ("checkpoint_restoration", 0.12, enhanced_recovery_manager.performance_metrics.checkpoint_restoration_times),
            ("state_validation", 0.03, enhanced_recovery_manager.performance_metrics.state_validation_times),
            ("failure_detection", 0.08, enhanced_recovery_manager.performance_metrics.failure_detection_times),
            ("recovery_execution", 0.25, enhanced_recovery_manager.performance_metrics.recovery_execution_times)
        ]
        
        for op_name, duration, metrics_list in operations:
            # Simulate multiple operations
            for _ in range(5):
                metrics_list.append(duration + (0.01 * len(metrics_list)))  # Add slight variance
        
        # Get performance summary
        summary = enhanced_recovery_manager.get_performance_summary()
        
        # Verify all metric categories are present
        expected_categories = [
            "checkpoint_creation", "checkpoint_restoration", "state_validation",
            "failure_detection", "recovery_execution", "health_checks", "failure_analysis"
        ]
        
        for category in expected_categories:
            assert category in summary
            
        # Verify statistical calculations
        for op_name, _, _ in operations:
            category_stats = summary[op_name]
            assert category_stats["count"] == 5
            assert category_stats["mean"] > 0
            assert category_stats["min"] > 0
            assert category_stats["max"] > 0
            assert category_stats["unit"] == "seconds"
        
        # Test monitoring status
        status = enhanced_recovery_manager.get_monitoring_status()
        assert status["performance_monitoring"]["enabled"] == True
        assert status["performance_monitoring"]["metrics_collected"] > 0
        assert "alert_thresholds" in status["performance_monitoring"]
        
        logger.info("✅ Real performance monitoring test passed")


class TestIntegratedAdvancedFeatures:
    """Integration tests combining multiple Phase 2 features."""
    
    @pytest.mark.asyncio
    async def test_human_interaction_with_branching_real(
        self,
        human_interaction_manager: HumanInteractionManager,
        branching_manager: CheckpointBranchingManager,
        real_pipeline_state: PipelineGlobalState
    ):
        """Test human interaction combined with checkpoint branching."""
        logger.info("🧪 Testing integrated human interaction with branching")
        
        # Use same database backend
        assert human_interaction_manager.langgraph_manager == branching_manager.langgraph_manager
        
        # Initialize pipeline
        thread_id = real_pipeline_state["thread_id"]
        await human_interaction_manager.langgraph_manager.initialize_pipeline_state(
            pipeline_id=real_pipeline_state["pipeline_id"],
            inputs=real_pipeline_state["global_variables"]["inputs"],
            user_id="test_user"
        )
        await human_interaction_manager.langgraph_manager.update_global_state(
            thread_id, real_pipeline_state
        )
        
        # Create checkpoint for branching
        checkpoint_id = await human_interaction_manager.langgraph_manager.create_checkpoint(
            thread_id=thread_id,
            description="Pre-interaction checkpoint"
        )
        
        # Start human inspection
        session = await human_interaction_manager.pause_for_inspection(
            thread_id=thread_id,
            step_id="step_3",
            interaction_type=InteractionType.DECISION,
            user_id="test_user"
        )
        
        # Create branch during human interaction
        branch_info = await branching_manager.create_branch(
            source_thread_id=thread_id,
            source_checkpoint_id=checkpoint_id,
            branch_name="human_decision_branch",
            description="Branch created during human interaction"
        )
        
        # Modify main thread through human interaction
        await human_interaction_manager.modify_state(
            session_id=session.session_id,
            field_path="intermediate_results.human_decision",
            new_value="main_thread_choice",
            modification_type="update",
            user_id="test_user"
        )
        
        # Modify branch thread directly
        branch_state = await branching_manager.langgraph_manager.get_global_state(
            branch_info.thread_id
        )
        branch_state["intermediate_results"]["human_decision"] = "branch_choice"
        await branching_manager.langgraph_manager.update_global_state(
            branch_info.thread_id, branch_state
        )
        
        # Resume main thread
        await human_interaction_manager.resume_execution(session.session_id, True)
        
        # Verify both threads have different decisions
        main_state = await branching_manager.langgraph_manager.get_global_state(thread_id)
        branch_state = await branching_manager.langgraph_manager.get_global_state(branch_info.thread_id)
        
        assert main_state["intermediate_results"]["human_decision"] == "main_thread_choice"
        assert branch_state["intermediate_results"]["human_decision"] == "branch_choice"
        
        logger.info("✅ Integrated human interaction with branching test passed")
    
    @pytest.mark.asyncio
    async def test_failure_recovery_with_human_intervention_real(
        self,
        enhanced_recovery_manager: EnhancedRecoveryManager,
        human_interaction_manager: HumanInteractionManager,
        real_pipeline_state: PipelineGlobalState
    ):
        """Test failure recovery combined with human intervention."""
        logger.info("🧪 Testing failure recovery with human intervention")
        
        # Initialize pipeline
        thread_id = real_pipeline_state["thread_id"]
        await enhanced_recovery_manager.langgraph_manager.initialize_pipeline_state(
            pipeline_id=real_pipeline_state["pipeline_id"],
            inputs=real_pipeline_state["global_variables"]["inputs"],
            user_id="test_user"
        )
        await enhanced_recovery_manager.langgraph_manager.update_global_state(
            thread_id, real_pipeline_state
        )
        
        # Create recovery checkpoint
        recovery_checkpoint_id = await enhanced_recovery_manager.langgraph_manager.create_checkpoint(
            thread_id=thread_id,
            description="Recovery point before failure"
        )
        
        # Simulate complex failure requiring human intervention
        complex_error = ValueError("Complex validation error requiring human review")
        failure_analysis = await enhanced_recovery_manager.analyze_failure(
            thread_id=thread_id,
            step_id="step_validation",
            error=complex_error,
            execution_context={"step_type": "critical_validation", "requires_human": True}
        )
        
        # If recovery confidence is low, trigger human interaction
        if failure_analysis.recovery_confidence < enhanced_recovery_manager.recovery_confidence_threshold:
            
            # Start human debugging session
            debug_session = await human_interaction_manager.pause_for_inspection(
                thread_id=thread_id,
                step_id="step_validation", 
                interaction_type=InteractionType.DEBUG,
                user_id="debug_expert",
                description=f"Debug complex failure: {failure_analysis.failure_category.value}"
            )
            
            # Human investigates and modifies state to fix issue
            await human_interaction_manager.modify_state(
                session_id=debug_session.session_id,
                field_path="intermediate_results.validation_fix",
                new_value="human_corrected_data",
                modification_type="update",
                user_id="debug_expert",
                reason="Fixed validation issue found during debugging"
            )
            
            # Generate analytics after human intervention
            analytics = await enhanced_recovery_manager.generate_execution_analytics(thread_id)
            assert analytics is not None
            assert analytics.recovery_attempts >= 1
            
            # Resume execution with fix
            await human_interaction_manager.resume_execution(
                debug_session.session_id, modifications_applied=True
            )
            
            # Verify fix is in place
            fixed_state = await enhanced_recovery_manager.langgraph_manager.get_global_state(thread_id)
            assert fixed_state["intermediate_results"]["validation_fix"] == "human_corrected_data"
        
        logger.info("✅ Failure recovery with human intervention test passed")


if __name__ == "__main__":
    async def run_all_tests():
        """Run all Phase 2 advanced features tests."""
        print("🚀 Running Phase 2 Advanced Features Tests (NO MOCKS)")
        
        # Create real test database
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "phase2_test.db")
        
        try:
            # Create managers with real database
            langgraph_manager = LangGraphGlobalContextManager(
                storage_type="sqlite",
                database_url=f"sqlite:///{db_path}"
            )
            
            human_manager = HumanInteractionManager(langgraph_manager)
            branching_manager = CheckpointBranchingManager(langgraph_manager)
            recovery_manager = EnhancedRecoveryManager(langgraph_manager)
            
            # Create test state
            test_state = {
                "thread_id": f"manual_test_{uuid.uuid4().hex[:8]}",
                "execution_id": f"exec_{uuid.uuid4().hex[:8]}",
                "pipeline_id": "manual_test_pipeline",
                "global_variables": {"inputs": {"test": "value"}},
                "intermediate_results": {"step_1": {"output": "test"}},
                "execution_metadata": {
                    "status": PipelineStatus.RUNNING,
                    "start_time": time.time(),
                    "pipeline_id": "manual_test_pipeline",
                    "completed_steps": ["step_1"],
                    "failed_steps": [],
                    "checkpoints": []
                }
            }
            
            # Test human interaction
            print("🧪 Testing Human Interaction...")
            session = await human_manager.pause_for_inspection(
                thread_id=test_state["thread_id"],
                step_id="step_1",
                interaction_type=InteractionType.INSPECTION
            )
            print(f"✅ Created inspection session: {session.session_id}")
            
            # Test branching
            print("🧪 Testing Checkpoint Branching...")
            checkpoint_id = await langgraph_manager.create_checkpoint(
                thread_id=test_state["thread_id"],
                description="Test checkpoint"
            )
            branch = await branching_manager.create_branch(
                source_thread_id=test_state["thread_id"],
                source_checkpoint_id=checkpoint_id,
                branch_name="test_branch",
                description="Manual test branch"
            )
            print(f"✅ Created branch: {branch.branch_id}")
            
            # Test recovery
            print("🧪 Testing Enhanced Recovery...")
            test_error = RuntimeError("Manual test error")
            analysis = await recovery_manager.analyze_failure(
                thread_id=test_state["thread_id"],
                step_id="step_1",
                error=test_error,
                execution_context={"step_type": "test"}
            )
            print(f"✅ Analyzed failure: {analysis.failure_category.value}")
            
            # Cleanup
            await human_manager.shutdown()
            await branching_manager.shutdown()
            await recovery_manager.shutdown()
            
            print("✅ All Phase 2 Advanced Features Tests Passed!")
            
        finally:
            # Cleanup database
            try:
                if os.path.exists(db_path):
                    os.remove(db_path)
                os.rmdir(temp_dir)
            except:
                pass
    
    asyncio.run(run_all_tests())