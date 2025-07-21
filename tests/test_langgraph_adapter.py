"""Tests for LangGraph adapter functionality."""

import asyncio

import pytest

from src.orchestrator.adapters.langgraph_adapter import (
    LangGraphAdapter,
    LangGraphEdge,
    LangGraphNode,
    LangGraphState,
    LangGraphWorkflow,
)
from src.orchestrator.core.control_system import ControlAction
from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.task import Task


class TestLangGraphNode:
    """Test cases for LangGraphNode class."""

    def test_langgraph_node_creation_basic(self):
        """Test basic LangGraph node creation."""

        def dummy_function(state, **kwargs):
            return {"result": "test"}

        node = LangGraphNode(
            name="test_node",
            function=dummy_function,
            inputs=["input1", "input2"],
            outputs=["output1"],
        )

        assert node.name == "test_node"
        assert node.function == dummy_function
        assert node.inputs == ["input1", "input2"]
        assert node.outputs == ["output1"]
        assert node.metadata == {}

    def test_langgraph_node_creation_with_metadata(self):
        """Test LangGraph node creation with metadata."""

        def dummy_function(state, **kwargs):
            return {"result": "test"}

        metadata = {"type": "processor", "version": "1.0"}
        node = LangGraphNode(
            name="test_node",
            function=dummy_function,
            inputs=["input1"],
            outputs=["output1"],
            metadata=metadata,
        )

        assert node.metadata == metadata

    def test_langgraph_node_post_init(self):
        """Test LangGraph node post-init behavior."""

        def dummy_function(state, **kwargs):
            return {"result": "test"}

        node = LangGraphNode(
            name="test_node",
            function=dummy_function,
            inputs=["input1"],
            outputs=["output1"],
            metadata=None,
        )

        # Should initialize empty metadata dict
        assert node.metadata == {}


class TestLangGraphEdge:
    """Test cases for LangGraphEdge class."""

    def test_langgraph_edge_creation_basic(self):
        """Test basic LangGraph edge creation."""
        edge = LangGraphEdge(source="node1", target="node2")

        assert edge.source == "node1"
        assert edge.target == "node2"
        assert edge.condition is None
        assert edge.metadata == {}

    def test_langgraph_edge_creation_with_condition(self):
        """Test LangGraph edge creation with condition."""

        def condition_func(state):
            return state.get("should_continue", True)

        metadata = {"type": "conditional"}
        edge = LangGraphEdge(
            source="node1", target="node2", condition=condition_func, metadata=metadata
        )

        assert edge.condition == condition_func
        assert edge.metadata == metadata

    def test_langgraph_edge_post_init(self):
        """Test LangGraph edge post-init behavior."""
        edge = LangGraphEdge(source="node1", target="node2", metadata=None)

        # Should initialize empty metadata dict
        assert edge.metadata == {}


class TestLangGraphState:
    """Test cases for LangGraphState class."""

    def test_langgraph_state_creation(self):
        """Test LangGraph state creation."""
        state = LangGraphState()

        assert state.data == {}
        assert state.history == []
        assert state.current_node is None

    def test_langgraph_state_get_set(self):
        """Test getting and setting values in state."""
        state = LangGraphState()

        # Test get with default
        assert state.get("nonexistent") is None
        assert state.get("nonexistent", "default") == "default"

        # Test set
        state.set("key1", "value1")
        assert state.get("key1") == "value1"
        assert len(state.history) == 1
        assert state.history[0] == {"action": "set", "key": "key1", "value": "value1"}

    def test_langgraph_state_update(self):
        """Test updating multiple values in state."""
        state = LangGraphState()

        updates = {"key1": "value1", "key2": "value2"}
        state.update(updates)

        assert state.get("key1") == "value1"
        assert state.get("key2") == "value2"
        assert len(state.history) == 1
        assert state.history[0] == {"action": "update", "updates": updates}

    def test_langgraph_state_to_dict(self):
        """Test converting state to dictionary."""
        state = LangGraphState()
        state.set("key1", "value1")
        state.current_node = "node1"

        result = state.to_dict()

        assert result["data"] == {"key1": "value1"}
        assert result["current_node"] == "node1"
        assert result["history_length"] == 1


class TestLangGraphWorkflow:
    """Test cases for LangGraphWorkflow class."""

    def test_langgraph_workflow_creation(self):
        """Test LangGraph workflow creation."""
        workflow = LangGraphWorkflow("test_workflow")

        assert workflow.name == "test_workflow"
        assert workflow.nodes == {}
        assert workflow.edges == []
        assert workflow.entry_point is None
        assert workflow.checkpoints == []
        assert workflow.metadata == {}

    def test_langgraph_workflow_add_node(self):
        """Test adding nodes to workflow."""
        workflow = LangGraphWorkflow("test_workflow")

        def node_func(state, **kwargs):
            return {"result": "test"}

        node1 = LangGraphNode("node1", node_func, ["input1"], ["output1"])
        node2 = LangGraphNode("node2", node_func, ["input2"], ["output2"])

        workflow.add_node(node1)
        assert "node1" in workflow.nodes
        assert workflow.entry_point == "node1"  # First node becomes entry point

        workflow.add_node(node2)
        assert "node2" in workflow.nodes
        assert workflow.entry_point == "node1"  # Entry point remains the same

    def test_langgraph_workflow_add_edge_valid(self):
        """Test adding valid edges to workflow."""
        workflow = LangGraphWorkflow("test_workflow")

        def node_func(state, **kwargs):
            return {"result": "test"}

        node1 = LangGraphNode("node1", node_func, ["input1"], ["output1"])
        node2 = LangGraphNode("node2", node_func, ["input2"], ["output2"])

        workflow.add_node(node1)
        workflow.add_node(node2)

        edge = LangGraphEdge("node1", "node2")
        workflow.add_edge(edge)

        assert len(workflow.edges) == 1
        assert workflow.edges[0] == edge

    def test_langgraph_workflow_add_edge_invalid_source(self):
        """Test adding edge with invalid source node."""
        workflow = LangGraphWorkflow("test_workflow")

        def node_func(state, **kwargs):
            return {"result": "test"}

        node2 = LangGraphNode("node2", node_func, ["input2"], ["output2"])
        workflow.add_node(node2)

        edge = LangGraphEdge("nonexistent", "node2")

        with pytest.raises(ValueError, match="Source node 'nonexistent' not found"):
            workflow.add_edge(edge)

    def test_langgraph_workflow_add_edge_invalid_target(self):
        """Test adding edge with invalid target node."""
        workflow = LangGraphWorkflow("test_workflow")

        def node_func(state, **kwargs):
            return {"result": "test"}

        node1 = LangGraphNode("node1", node_func, ["input1"], ["output1"])
        workflow.add_node(node1)

        edge = LangGraphEdge("node1", "nonexistent")

        with pytest.raises(ValueError, match="Target node 'nonexistent' not found"):
            workflow.add_edge(edge)

    def test_langgraph_workflow_get_next_nodes_no_condition(self):
        """Test getting next nodes without conditions."""
        workflow = LangGraphWorkflow("test_workflow")

        def node_func(state, **kwargs):
            return {"result": "test"}

        node1 = LangGraphNode("node1", node_func, [], [])
        node2 = LangGraphNode("node2", node_func, [], [])
        node3 = LangGraphNode("node3", node_func, [], [])

        workflow.add_node(node1)
        workflow.add_node(node2)
        workflow.add_node(node3)

        edge1 = LangGraphEdge("node1", "node2")
        edge2 = LangGraphEdge("node1", "node3")

        workflow.add_edge(edge1)
        workflow.add_edge(edge2)

        state = LangGraphState()
        next_nodes = workflow.get_next_nodes("node1", state)

        assert set(next_nodes) == {"node2", "node3"}

    def test_langgraph_workflow_get_next_nodes_with_condition(self):
        """Test getting next nodes with conditions."""
        workflow = LangGraphWorkflow("test_workflow")

        def node_func(state, **kwargs):
            return {"result": "test"}

        def condition_true(state):
            return True

        def condition_false(state):
            return False

        node1 = LangGraphNode("node1", node_func, [], [])
        node2 = LangGraphNode("node2", node_func, [], [])
        node3 = LangGraphNode("node3", node_func, [], [])

        workflow.add_node(node1)
        workflow.add_node(node2)
        workflow.add_node(node3)

        edge1 = LangGraphEdge("node1", "node2", condition_true)
        edge2 = LangGraphEdge("node1", "node3", condition_false)

        workflow.add_edge(edge1)
        workflow.add_edge(edge2)

        state = LangGraphState()
        next_nodes = workflow.get_next_nodes("node1", state)

        assert next_nodes == ["node2"]  # Only condition_true passes

    @pytest.mark.asyncio
    async def test_langgraph_workflow_execute_simple(self):
        """Test executing a simple workflow."""
        workflow = LangGraphWorkflow("test_workflow")

        def node1_func(state, **kwargs):
            state.set("step1_done", True)
            return {"output1": "result1"}

        def node2_func(state, **kwargs):
            state.set("step2_done", True)
            return {"output2": "result2"}

        node1 = LangGraphNode("node1", node1_func, [], ["output1"])
        node2 = LangGraphNode("node2", node2_func, ["output1"], ["output2"])

        workflow.add_node(node1)
        workflow.add_node(node2)
        workflow.add_edge(LangGraphEdge("node1", "node2"))

        initial_state = {"initial": "value"}
        result_state = await workflow.execute(initial_state)

        assert result_state.get("initial") == "value"
        assert result_state.get("step1_done") is True
        assert result_state.get("step2_done") is True
        assert result_state.get("output1") == "result1"
        assert result_state.get("output2") == "result2"

    @pytest.mark.asyncio
    async def test_langgraph_workflow_execute_async_function(self):
        """Test executing workflow with async node function."""
        workflow = LangGraphWorkflow("test_workflow")

        async def async_node_func(state, **kwargs):
            await asyncio.sleep(0.01)  # Simulate async work
            state.set("async_done", True)
            return {"async_result": "success"}

        node = LangGraphNode("async_node", async_node_func, [], ["async_result"])
        workflow.add_node(node)

        result_state = await workflow.execute()

        assert result_state.get("async_done") is True
        assert result_state.get("async_result") == "success"

    @pytest.mark.asyncio
    async def test_langgraph_workflow_execute_with_error(self):
        """Test executing workflow with node error."""
        workflow = LangGraphWorkflow("test_workflow")

        def error_node_func(state, **kwargs):
            raise ValueError("Test error")

        error_node = LangGraphNode("error_node", error_node_func, [], ["error_output"])
        workflow.add_node(error_node)

        result_state = await workflow.execute()

        # Error should be captured in state
        assert "error_error_node" in result_state.data
        assert "Test error" in result_state.get("error_error_node")

        # Workflow should complete despite error
        assert result_state.current_node == "error_node"

    @pytest.mark.asyncio
    async def test_langgraph_workflow_execute_no_entry_point(self):
        """Test executing workflow with no entry point."""
        workflow = LangGraphWorkflow("test_workflow")
        workflow.entry_point = None

        result_state = await workflow.execute()

        # Should return empty state
        assert result_state.data == {}


class TestLangGraphAdapter:
    """Test cases for LangGraphAdapter class."""

    def test_langgraph_adapter_creation_default(self):
        """Test LangGraph adapter creation with default config."""
        adapter = LangGraphAdapter()

        assert adapter.config == {"name": "langgraph_adapter"}
        assert adapter.workflows == {}
        assert adapter.active_executions == {}

    def test_langgraph_adapter_creation_custom_config(self):
        """Test LangGraph adapter creation with custom config."""
        config = {"name": "custom_adapter", "setting": "value"}
        adapter = LangGraphAdapter(config)

        assert adapter.config == config
        assert adapter.workflows == {}
        assert adapter.active_executions == {}

    def test_langgraph_adapter_register_workflow(self):
        """Test registering a workflow."""
        adapter = LangGraphAdapter()
        workflow = LangGraphWorkflow("test_workflow")

        adapter.register_workflow(workflow)

        assert "test_workflow" in adapter.workflows
        assert adapter.workflows["test_workflow"] == workflow

    def test_langgraph_adapter_create_workflow_from_pipeline(self):
        """Test creating workflow from pipeline."""
        adapter = LangGraphAdapter()

        # Create test pipeline
        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="test_action")
        task2 = Task(
            id="task2", name="Task 2", action="test_action", dependencies=["task1"]
        )

        pipeline.add_task(task1)
        pipeline.add_task(task2)

        workflow = adapter.create_workflow_from_pipeline(pipeline)

        assert workflow.name == "test_pipeline"
        assert "task1" in workflow.nodes
        assert "task2" in workflow.nodes
        assert len(workflow.edges) == 1
        assert workflow.edges[0].source == "task1"
        assert workflow.edges[0].target == "task2"

    @pytest.mark.asyncio
    async def test_langgraph_adapter_execute_task(self):
        """Test executing a single task."""
        # This test requires real model execution - skip if models aren't available
        pytest.skip("This test requires real model execution with API keys configured")

    @pytest.mark.asyncio
    async def test_langgraph_adapter_execute_pipeline(self):
        """Test executing a pipeline."""
        adapter = LangGraphAdapter()

        # Create test pipeline
        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="test_action")
        pipeline.add_task(task1)

        # Store original method
        original_execute_task = adapter._execute_task
        execute_called = False

        # Replace with test implementation
        async def test_execute_task(task, context):
            nonlocal execute_called
            execute_called = True
            return "task_result"

        adapter._execute_task = test_execute_task

        try:
            result = await adapter.execute_pipeline(pipeline)

            assert isinstance(result, dict)
            assert execute_called
        finally:
            # Restore original method
            adapter._execute_task = original_execute_task

    def test_langgraph_adapter_get_capabilities(self):
        """Test getting adapter capabilities."""
        adapter = LangGraphAdapter()

        capabilities = adapter.get_capabilities()

        assert capabilities["supports_workflows"] is True
        assert capabilities["supports_conditional_execution"] is True
        assert capabilities["supports_parallel_execution"] is True
        assert "supported_actions" in capabilities
        assert "generate" in capabilities["supported_actions"]

    @pytest.mark.asyncio
    async def test_langgraph_adapter_health_check(self):
        """Test health check."""
        # This test requires real model registry - skip if models aren't available
        pytest.skip("This test requires real model registry with API keys configured")

    @pytest.mark.asyncio
    async def test_langgraph_adapter_execute_workflow_success(self):
        """Test executing a registered workflow."""
        adapter = LangGraphAdapter()

        # Create and register workflow
        workflow = LangGraphWorkflow("test_workflow")

        def node_func(state, **kwargs):
            return {"result": "success"}

        node = LangGraphNode("node1", node_func, [], ["result"])
        workflow.add_node(node)
        adapter.register_workflow(workflow)

        result_state = await adapter.execute_workflow(
            "test_workflow", {"initial": "data"}
        )

        assert result_state.get("initial") == "data"
        assert result_state.get("result") == "success"
        assert len(adapter.active_executions) == 1

    @pytest.mark.asyncio
    async def test_langgraph_adapter_execute_workflow_not_found(self):
        """Test executing non-existent workflow."""
        adapter = LangGraphAdapter()

        with pytest.raises(ValueError, match="Workflow 'nonexistent' not found"):
            await adapter.execute_workflow("nonexistent")

    @pytest.mark.asyncio
    async def test_langgraph_adapter_decide_action_execute(self):
        """Test deciding action when inputs are ready."""
        adapter = LangGraphAdapter()

        # Create workflow with node
        workflow = LangGraphWorkflow("test_workflow")

        def node_func(state, **kwargs):
            return {"result": "test"}

        node = LangGraphNode("task1", node_func, ["input1"], ["output1"])
        workflow.add_node(node)
        adapter.register_workflow(workflow)

        # Create state with required input
        state = LangGraphState()
        state.set("input1", "value1")
        execution_id = "test_workflow_0"
        adapter.active_executions[execution_id] = state

        task = Task(id="task1", name="Task 1", action="test_action")
        context = {"workflow_name": "test_workflow", "execution_id": execution_id}

        action = await adapter.decide_action(task, context)
        assert action == ControlAction.EXECUTE

    @pytest.mark.asyncio
    async def test_langgraph_adapter_decide_action_wait(self):
        """Test deciding action when inputs are not ready."""
        adapter = LangGraphAdapter()

        # Create workflow with node
        workflow = LangGraphWorkflow("test_workflow")

        def node_func(state, **kwargs):
            return {"result": "test"}

        node = LangGraphNode("task1", node_func, ["input1"], ["output1"])
        workflow.add_node(node)
        adapter.register_workflow(workflow)

        # Create state without required input
        state = LangGraphState()
        execution_id = "test_workflow_0"
        adapter.active_executions[execution_id] = state

        task = Task(id="task1", name="Task 1", action="test_action")
        context = {"workflow_name": "test_workflow", "execution_id": execution_id}

        action = await adapter.decide_action(task, context)
        assert action == ControlAction.WAIT

    @pytest.mark.asyncio
    async def test_langgraph_adapter_decide_action_default(self):
        """Test deciding action with default behavior."""
        adapter = LangGraphAdapter()
        task = Task(id="task1", name="Task 1", action="test_action")
        context = {}

        action = await adapter.decide_action(task, context)
        assert action == ControlAction.EXECUTE

    def test_langgraph_adapter_get_workflow_status_existing(self):
        """Test getting status of existing workflow."""
        adapter = LangGraphAdapter()

        # Create and register workflow
        workflow = LangGraphWorkflow("test_workflow")

        def node_func(state, **kwargs):
            return {"result": "test"}

        node1 = LangGraphNode("node1", node_func, [], [])
        node2 = LangGraphNode("node2", node_func, [], [])
        workflow.add_node(node1)
        workflow.add_node(node2)
        workflow.add_edge(LangGraphEdge("node1", "node2"))
        workflow.metadata = {"version": "1.0"}

        adapter.register_workflow(workflow)

        # Add active execution
        adapter.active_executions["test_workflow_0"] = LangGraphState()

        status = adapter.get_workflow_status("test_workflow")

        assert status["name"] == "test_workflow"
        assert status["nodes"] == 2
        assert status["edges"] == 1
        assert status["entry_point"] == "node1"
        assert status["active_executions"] == 1
        assert status["metadata"] == {"version": "1.0"}

    def test_langgraph_adapter_get_workflow_status_not_found(self):
        """Test getting status of non-existent workflow."""
        adapter = LangGraphAdapter()

        status = adapter.get_workflow_status("nonexistent")
        assert status["error"] == "Workflow not found"

    def test_langgraph_adapter_get_execution_status_existing(self):
        """Test getting status of existing execution."""
        adapter = LangGraphAdapter()

        state = LangGraphState()
        state.set("key", "value")
        state.current_node = "node1"

        execution_id = "test_execution"
        adapter.active_executions[execution_id] = state

        status = adapter.get_execution_status(execution_id)

        assert status["execution_id"] == execution_id
        assert status["current_node"] == "node1"
        assert status["state_data"] == {"key": "value"}
        assert status["history_length"] == 1
        assert status["status"] == "active"

    def test_langgraph_adapter_get_execution_status_not_found(self):
        """Test getting status of non-existent execution."""
        adapter = LangGraphAdapter()

        status = adapter.get_execution_status("nonexistent")
        assert status["error"] == "Execution not found"

    def test_langgraph_adapter_cleanup_execution_success(self):
        """Test cleaning up existing execution."""
        adapter = LangGraphAdapter()

        execution_id = "test_execution"
        adapter.active_executions[execution_id] = LangGraphState()

        result = adapter.cleanup_execution(execution_id)

        assert result is True
        assert execution_id not in adapter.active_executions

    def test_langgraph_adapter_cleanup_execution_not_found(self):
        """Test cleaning up non-existent execution."""
        adapter = LangGraphAdapter()

        result = adapter.cleanup_execution("nonexistent")
        assert result is False

    def test_langgraph_adapter_get_statistics(self):
        """Test getting adapter statistics."""
        adapter = LangGraphAdapter()

        # Create and register workflow
        workflow1 = LangGraphWorkflow("workflow1")
        workflow2 = LangGraphWorkflow("workflow2")

        def node_func(state, **kwargs):
            return {"result": "test"}

        # Add nodes and edges to workflows
        node1 = LangGraphNode("node1", node_func, [], [])
        node2 = LangGraphNode("node2", node_func, [], [])
        workflow1.add_node(node1)
        workflow1.add_node(node2)
        workflow1.add_edge(LangGraphEdge("node1", "node2"))

        node3 = LangGraphNode("node3", node_func, [], [])
        workflow2.add_node(node3)

        adapter.register_workflow(workflow1)
        adapter.register_workflow(workflow2)

        # Add active executions
        adapter.active_executions["exec1"] = LangGraphState()
        adapter.active_executions["exec2"] = LangGraphState()

        stats = adapter.get_statistics()

        assert stats["workflows_registered"] == 2
        assert stats["active_executions"] == 2
        assert stats["total_nodes"] == 3  # 2 from workflow1 + 1 from workflow2
        assert stats["total_edges"] == 1  # 1 from workflow1


class TestLangGraphAdapterIntegration:
    """Integration tests for LangGraph adapter."""

    @pytest.mark.asyncio
    async def test_full_pipeline_to_workflow_execution(self):
        """Test complete pipeline to workflow conversion and execution."""
        adapter = LangGraphAdapter()

        # Create a pipeline with dependencies
        pipeline = Pipeline(id="integration_test", name="Integration Test Pipeline")

        task_a = Task(
            id="task_a", name="Task A", action="process", parameters={"data": "input_a"}
        )
        task_b = Task(
            id="task_b", name="Task B", action="process", parameters={"data": "input_b"}
        )
        task_c = Task(
            id="task_c",
            name="Task C",
            action="combine",
            parameters={"inputs": ["a", "b"]},
            dependencies=["task_a", "task_b"],
        )

        pipeline.add_task(task_a)
        pipeline.add_task(task_b)
        pipeline.add_task(task_c)

        # Store original method
        original_execute_task = adapter._execute_task

        # Replace with test implementation
        async def test_execute_task(task, state_data):
            if task.id == "task_a":
                return "result_a"
            elif task.id == "task_b":
                return "result_b"
            elif task.id == "task_c":
                return f"combined_{state_data.get('output', 'unknown')}"
            return f"result_{task.id}"

        adapter._execute_task = test_execute_task

        try:
            result = await adapter.execute_pipeline(pipeline)

            # Verify execution completed
            assert isinstance(result, dict)

            # Verify workflow was created properly
            workflow = adapter.create_workflow_from_pipeline(pipeline)
            assert len(workflow.nodes) == 3
            assert len(workflow.edges) == 2  # task_a->task_c, task_b->task_c
        finally:
            # Restore original method
            adapter._execute_task = original_execute_task

    @pytest.mark.asyncio
    async def test_conditional_workflow_execution(self):
        """Test workflow execution with conditional edges."""
        adapter = LangGraphAdapter()

        workflow = LangGraphWorkflow("conditional_test")

        def start_node(state, **kwargs):
            state.set("decision", "go_right")
            return {"decision": "go_right"}

        def left_node(state, **kwargs):
            state.set("path", "left")
            return {"result": "left_result"}

        def right_node(state, **kwargs):
            state.set("path", "right")
            return {"result": "right_result"}

        def go_right_condition(state):
            return state.get("decision") == "go_right"

        def go_left_condition(state):
            return state.get("decision") == "go_left"

        # Create nodes
        start = LangGraphNode("start", start_node, [], ["decision"])
        left = LangGraphNode("left", left_node, [], ["result"])
        right = LangGraphNode("right", right_node, [], ["result"])

        workflow.add_node(start)
        workflow.add_node(left)
        workflow.add_node(right)

        # Add conditional edges
        workflow.add_edge(LangGraphEdge("start", "left", go_left_condition))
        workflow.add_edge(LangGraphEdge("start", "right", go_right_condition))

        adapter.register_workflow(workflow)

        result_state = await adapter.execute_workflow("conditional_test")

        # Should take the right path
        assert result_state.get("decision") == "go_right"
        assert result_state.get("path") == "right"
        assert result_state.get("result") == "right_result"
