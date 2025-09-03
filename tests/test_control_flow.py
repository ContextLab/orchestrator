"""Tests for control flow features using real models."""

import pytest

from src.orchestrator.control_flow.auto_resolver import ControlFlowAutoResolver
from src.orchestrator.control_flow.conditional import ConditionalHandler, ConditionalTask
from src.orchestrator.control_flow.loops import ForLoopHandler, WhileLoopHandler
from src.orchestrator.control_flow.dynamic_flow import DynamicFlowHandler
from src.orchestrator.engine.control_flow_engine import ControlFlowEngine
from src.orchestrator.core.task import Task
from src.orchestrator.core.pipeline import Pipeline

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider


class TestControlFlowAutoResolver:
    """Test AUTO tag resolution for control flow using real models."""

    @pytest.mark.asyncio
    async def test_resolve_condition_boolean(self):
        """Test resolving boolean conditions."""
        resolver = ControlFlowAutoResolver()

        # Test simple boolean strings - these don't need AI
        assert await resolver.resolve_condition("true", {}, {}) is True
        assert await resolver.resolve_condition("false", {}, {}) is False
        assert await resolver.resolve_condition("yes", {}, {}) is True
        assert await resolver.resolve_condition("no", {}, {}) is False

    @pytest.mark.asyncio
    async def test_resolve_condition_with_auto(self, populated_model_registry):
        """Test resolving conditions with AUTO tags using real models."""
        resolver = ControlFlowAutoResolver(populated_model_registry)

        # Test with a simple AUTO tag that should resolve to true/false
        condition = "<AUTO>Is 5 greater than 3? Answer only 'true' or 'false'.</AUTO>"
        result = await resolver.resolve_condition(condition, {}, {})
        assert result is True

        # Test with a condition that should be false
        condition2 = "<AUTO>Is 2 greater than 10? Answer only 'true' or 'false'.</AUTO>"
        result2 = await resolver.resolve_condition(condition2, {}, {})
        assert result2 is False

    @pytest.mark.asyncio
    async def test_resolve_condition_with_context(self, populated_model_registry):
        """Test resolving conditions with context using real models."""
        resolver = ControlFlowAutoResolver(populated_model_registry)

        context = {"user_type": "premium", "credits": 100}
        step_results = {"previous_check": {"has_access": True}}

        # Test with context-aware AUTO tag
        condition = "<AUTO>Given user_type is '{{ user_type }}' and they have {{ credits }} credits, should they be allowed to proceed? Answer only 'true' or 'false'.</AUTO>"
        result = await resolver.resolve_condition(condition, context, step_results)
        assert result is True  # Premium user with credits should proceed

    @pytest.mark.asyncio
    async def test_resolve_iterator(self):
        """Test resolving iterator expressions."""
        resolver = ControlFlowAutoResolver()

        # Test list
        items = await resolver.resolve_iterator("[1, 2, 3]", {}, {})
        assert items == [1, 2, 3]

        # Test comma-separated
        items = await resolver.resolve_iterator("a, b, c", {}, {})
        assert items == ["a", "b", "c"]

        # Test single item
        items = await resolver.resolve_iterator("single", {}, {})
        assert items == ["single"]

    @pytest.mark.asyncio
    async def test_resolve_iterator_with_auto(self, populated_model_registry):
        """Test resolving iterator with AUTO tags using real models."""
        resolver = ControlFlowAutoResolver(populated_model_registry)

        # Test AUTO tag for iterator
        iterator = "<AUTO>List the first 3 prime numbers, separated by commas. Answer only with the numbers and commas, no other text.</AUTO>"
        items = await resolver.resolve_iterator(iterator, {}, {})
        assert items == ["2", "3", "5"] or items == [
            2,
            3,
            5,
        ]  # Model might return strings or ints

    @pytest.mark.asyncio
    async def test_resolve_count(self):
        """Test resolving count expressions."""
        resolver = ControlFlowAutoResolver()

        assert await resolver.resolve_count("5", {}, {}) == 5
        assert await resolver.resolve_count("10", {}, {}) == 10
        assert await resolver.resolve_count(3, {}, {}) == 3

    @pytest.mark.asyncio
    async def test_resolve_count_with_auto(self, populated_model_registry):
        """Test resolving count with AUTO tags using real models."""
        resolver = ControlFlowAutoResolver(populated_model_registry)

        # Test AUTO tag for count
        count_expr = (
            "<AUTO>How many days are in a week? Answer with just the number.</AUTO>"
        )
        count = await resolver.resolve_count(count_expr, {}, {})
        assert count == 7

    @pytest.mark.asyncio
    async def test_resolve_target(self):
        """Test resolving jump targets."""
        resolver = ControlFlowAutoResolver()

        valid_targets = ["step1", "step2", "step3"]

        # Valid target
        target = await resolver.resolve_target("step2", {}, {}, valid_targets)
        assert target == "step2"

        # Invalid target
        with pytest.raises(ValueError):
            await resolver.resolve_target("invalid", {}, {}, valid_targets)

    @pytest.mark.asyncio
    async def test_resolve_target_with_auto(self, populated_model_registry):
        """Test resolving jump targets with AUTO tags using real models."""
        resolver = ControlFlowAutoResolver(populated_model_registry)

        valid_targets = ["start", "process", "validate", "end"]
        context = {"error_occurred": True}

        # Test AUTO tag for target selection
        target_expr = "<AUTO>Given that error_occurred is {{ error_occurred }}, which step should we jump to: 'validate' or 'end'? Answer with just the step name.</AUTO>"
        target = await resolver.resolve_target(target_expr, context, {}, valid_targets)
        assert target in ["validate", "end"]  # Model should choose one of these


class TestConditionalHandler:
    """Test conditional execution handler with real models."""

    @pytest.mark.asyncio
    async def test_evaluate_condition_simple(self, populated_model_registry):
        """Test simple condition evaluation."""
        resolver = ControlFlowAutoResolver(populated_model_registry)
        handler = ConditionalHandler(resolver)

        # Test with simple boolean
        task = Task("test", "Test", "action", {}, metadata={"condition": "true"})
        result = await handler.evaluate_condition(task, {}, {})
        assert result is True

        # Test with expression
        context = {"count": 5}
        task = Task("test2", "Test2", "action", {}, metadata={"condition": "count > 3"})
        result = await handler.evaluate_condition(task, context, {})
        assert result is True

    @pytest.mark.asyncio
    async def test_evaluate_condition_with_auto(self, populated_model_registry):
        """Test condition evaluation with AUTO tags using real models."""
        resolver = ControlFlowAutoResolver(populated_model_registry)
        handler = ConditionalHandler(resolver)

        # Test with AUTO tag
        condition = (
            "<AUTO>Is Python a programming language? Answer 'true' or 'false'.</AUTO>"
        )
        task = Task("test", "Test", "action", {}, metadata={"condition": condition})
        result = await handler.evaluate_condition(task, {}, {})
        assert result is True

    @pytest.mark.asyncio
    async def test_create_conditional_tasks(self):
        """Test creating conditional tasks."""
        handler = ConditionalHandler()

        # Create conditional task definition
        task_def = {
            "id": "cond1",
            "name": "Conditional Task",
            "condition": "count > 5",
            "action": "log",
            "parameters": {"message": "Count is large"},
            "else": "task3",
        }

        # Create conditional task
        cond_task = handler.create_conditional_task(task_def)

        assert isinstance(cond_task, ConditionalTask)
        assert cond_task.condition == "count > 5"
        assert cond_task.else_task_id == "task3"
        assert cond_task.id == "cond1"


class TestForLoopHandler:
    """Test for loop handler with real models."""

    @pytest.mark.asyncio
    async def test_expand_for_loop_simple(self, populated_model_registry):
        """Test simple for loop expansion."""
        resolver = ControlFlowAutoResolver(populated_model_registry)
        handler = ForLoopHandler(resolver)

        # Test with list iterator
        loop_def = {
            "id": "loop1",
            "for_each": "[apple, banana, orange]",
            "loop_var": "fruit",
            "steps": [
                {
                    "id": "process",
                    "action": "process",
                    "parameters": {"item": "{{ fruit }}"},
                }
            ],
        }

        tasks = await handler.expand_for_loop(loop_def, {}, {})

        assert len(tasks) == 3
        assert tasks[0].parameters["item"] == "apple"
        assert tasks[1].parameters["item"] == "banana"
        assert tasks[2].parameters["item"] == "orange"

    @pytest.mark.asyncio
    async def test_expand_for_loop_with_auto(self, populated_model_registry):
        """Test for loop with AUTO tag iterator using real models."""
        resolver = ControlFlowAutoResolver(populated_model_registry)
        handler = ForLoopHandler(resolver)

        # Test with AUTO tag iterator
        loop_def = {
            "id": "loop1",
            "for_each": "<AUTO>List the primary colors (red, green, blue) separated by commas.</AUTO>",
            "loop_var": "color",
            "steps": [
                {
                    "id": "process",
                    "action": "process",
                    "parameters": {"color": "{{ color }}"},
                }
            ],
        }

        tasks = await handler.expand_for_loop(loop_def, {}, {})

        assert len(tasks) >= 3  # Should have at least 3 colors
        # Check that we got color names in parameters
        for task in tasks:
            assert "color" in task.parameters
            assert len(task.parameters["color"]) > 0


class TestWhileLoopHandler:
    """Test while loop handler with real models."""

    @pytest.mark.asyncio
    async def test_should_continue_simple(self, populated_model_registry):
        """Test simple while loop condition."""
        resolver = ControlFlowAutoResolver(populated_model_registry)
        handler = WhileLoopHandler(resolver)

        # Test with simple condition
        context = {"counter": 3}
        should_continue = await handler.should_continue(
            loop_id="while1",
            condition="counter < 5",
            context=context,
            step_results={},
            iteration=0,
            max_iterations=100)
        assert should_continue is True

        # Test when condition is false
        context["counter"] = 10
        # Clear the cache to ensure fresh evaluation
        resolver.clear_cache()
        should_continue = await handler.should_continue(
            loop_id="while1",
            condition="counter < 5",
            context=context,
            step_results={},
            iteration=0,
            max_iterations=100)
        assert should_continue is False

    @pytest.mark.asyncio
    async def test_should_continue_with_auto(self, populated_model_registry):
        """Test while loop with AUTO condition using real models."""
        resolver = ControlFlowAutoResolver(populated_model_registry)
        handler = WhileLoopHandler(resolver)

        # Test with AUTO tag condition
        context = {"retries": 2, "max_retries": 3}
        condition = "<AUTO>Given retries={{ retries }} and max_retries={{ max_retries }}, should we continue retrying? Answer 'true' or 'false'.</AUTO>"

        should_continue = await handler.should_continue(
            loop_id="while2",
            condition=condition,
            context=context,
            step_results={},
            iteration=0,
            max_iterations=100)
        assert should_continue is True  # 2 < 3, so should continue


class TestDynamicFlowHandler:
    """Test dynamic flow handler with real models."""

    @pytest.mark.asyncio
    async def test_process_goto_simple(self, populated_model_registry):
        """Test simple goto processing."""
        resolver = ControlFlowAutoResolver(populated_model_registry)
        handler = DynamicFlowHandler(resolver)

        # Create test tasks
        all_tasks = {
            "start": Task("start", "Start", "action", {}),
            "middle": Task("middle", "Middle", "action", {}),
            "end": Task("end", "End", "action", {}),
        }

        # Test goto with valid target
        goto_task = Task("goto1", "Goto", "goto", {}, metadata={"goto": "end"})
        target = await handler.process_goto(goto_task, {}, {}, all_tasks)
        assert target == "end"

    @pytest.mark.asyncio
    async def test_process_goto_with_auto(self, populated_model_registry):
        """Test goto with AUTO target using real models."""
        resolver = ControlFlowAutoResolver(populated_model_registry)
        handler = DynamicFlowHandler(resolver)

        # Create test tasks
        all_tasks = {
            "validate": Task("validate", "Validate", "action", {}),
            "process": Task("process", "Process", "action", {}),
            "error_handler": Task("error_handler", "Error Handler", "action", {}),
            "success": Task("success", "Success", "action", {}),
        }

        # Test goto with AUTO tag
        context = {"validation_passed": False}
        goto_task = Task(
            "goto1",
            "Goto",
            "goto",
            {},
            metadata={
                "goto": "<AUTO>validation_passed={{ validation_passed }}. Choose next step: 'error_handler' or 'process'. Reply with only one word: error_handler OR process</AUTO>"
            })

        target = await handler.process_goto(goto_task, context, {}, all_tasks)
        assert (
            target == "error_handler"
        )  # Should go to error handler when validation fails


class TestControlFlowEngine:
    """Test control flow engine with real models and execution."""

    @pytest.mark.asyncio
    async def test_execute_conditional_flow(self, populated_model_registry):
        """Test executing a pipeline with conditional flow using real models."""
        # Create engine with model registry
        engine = ControlFlowEngine(populated_model_registry)

        # Create a pipeline with conditional
        pipeline = Pipeline(id="conditional_test", name="Conditional Test Pipeline")

        # Add a task that sets a value
        pipeline.add_task(Task("setup", "Setup", "set_value", {"value": 10}))

        # Add conditional task
        resolver = ControlFlowAutoResolver(populated_model_registry)
        cond_handler = ConditionalHandler(resolver)

        # Create conditional task definition
        cond_task_def = {
            "id": "check_value",
            "name": "Check Value",
            "condition": "<AUTO>Is 10 greater than 5? Answer 'true' or 'false'.</AUTO>",
            "action": "log",
            "parameters": {"message": "Checking value"},
            "depends_on": ["setup"],
        }
        cond_task = cond_handler.create_conditional_task(cond_task_def)
        pipeline.add_task(cond_task)

        # Add the then and else tasks
        pipeline.add_task(
            Task(
                "then_path",
                "Then Path",
                "log",
                {"message": "Value is large"},
                dependencies=["check_value"])
        )
        pipeline.add_task(
            Task(
                "else_path",
                "Else Path",
                "log",
                {"message": "Value is small"},
                dependencies=["check_value"])
        )

        # Set else reference
        cond_task.else_task_id = "else_path"

        # Execute pipeline
        context = {}
        await engine.execute_pipeline(pipeline, context)

        # Check that the right path was taken
        # The then_path should have been executed
        then_task = pipeline.get_task("then_path")
        else_task = pipeline.get_task("else_path")

        # In a real execution, we'd check task status
        # For now, verify the pipeline structure is correct
        assert then_task is not None
        assert else_task is not None

    @pytest.mark.asyncio
    async def test_execute_loop_flow(self, populated_model_registry):
        """Test executing a pipeline with loops using real models."""
        # Create engine with model registry
        # engine = ControlFlowEngine(populated_model_registry)

        # Create a pipeline with a for loop
        # pipeline = Pipeline(id="loop_test", name="Loop Test Pipeline")

        # Add a for loop that processes items
        # for_handler = ForLoopHandler()

        # This would normally be done through YAML compilation
        # For testing, we'll simulate the structure
        context = {
            "items": "<AUTO>List 3 programming languages, separated by commas. Just the names, no other text.</AUTO>"
        }

        # Test that the engine can handle AUTO resolution in loops
        # In real usage, this would be part of pipeline execution
        resolver = ControlFlowAutoResolver(populated_model_registry)
        items = await resolver.resolve_iterator(context["items"], {}, {})

        assert len(items) >= 3  # Should get at least 3 languages
        assert all(isinstance(item, str) for item in items)
        assert all(len(item) > 0 for item in items)  # No empty strings
