"""
Comprehensive tests for the Variable Management System.

Tests cover all aspects of variable management including scoping,
dependency tracking, template resolution, and thread safety.
"""

import pytest
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from src.orchestrator.execution.variables import (
    VariableManager,
    VariableContext,
    VariableScope,
    VariableType,
    Variable,
    VariableMetadata
)


class TestVariableMetadata:
    """Tests for VariableMetadata class."""
    
    def test_metadata_creation(self):
        """Test basic metadata creation."""
        metadata = VariableMetadata(
            name="test_var",
            scope=VariableScope.GLOBAL,
            var_type=VariableType.INPUT,
            description="Test variable"
        )
        
        assert metadata.name == "test_var"
        assert metadata.scope == VariableScope.GLOBAL
        assert metadata.var_type == VariableType.INPUT
        assert metadata.description == "Test variable"
        assert metadata.version == 1
        assert isinstance(metadata.created_at, datetime)
        assert isinstance(metadata.updated_at, datetime)
    
    def test_update_timestamp(self):
        """Test timestamp updating."""
        metadata = VariableMetadata(
            name="test_var",
            scope=VariableScope.GLOBAL,
            var_type=VariableType.INPUT
        )
        
        original_version = metadata.version
        original_updated = metadata.updated_at
        
        time.sleep(0.01)  # Small delay to ensure timestamp difference
        metadata.update_timestamp()
        
        assert metadata.version == original_version + 1
        assert metadata.updated_at > original_updated


class TestVariable:
    """Tests for Variable class."""
    
    def test_variable_creation(self):
        """Test variable creation with metadata."""
        metadata = VariableMetadata(
            name="test_var",
            scope=VariableScope.GLOBAL,
            var_type=VariableType.INPUT
        )
        
        variable = Variable(
            name="test_var",
            value="test_value",
            metadata=metadata
        )
        
        assert variable.name == "test_var"
        assert variable.value == "test_value"
        assert variable.metadata.name == "test_var"  # Auto-synced
    
    def test_variable_copy(self):
        """Test deep copying of variables."""
        metadata = VariableMetadata(
            name="test_var",
            scope=VariableScope.GLOBAL,
            var_type=VariableType.INPUT
        )
        
        original = Variable(
            name="test_var",
            value={"nested": {"data": [1, 2, 3]}},
            metadata=metadata
        )
        
        copy = original.copy()
        
        assert copy.name == original.name
        assert copy.value == original.value
        assert copy.value is not original.value  # Deep copy
        assert copy.metadata is not original.metadata  # Deep copy
    
    def test_update_value(self):
        """Test value updating with metadata tracking."""
        metadata = VariableMetadata(
            name="test_var",
            scope=VariableScope.GLOBAL,
            var_type=VariableType.INPUT
        )
        
        variable = Variable(
            name="test_var",
            value="initial_value",
            metadata=metadata
        )
        
        original_version = variable.metadata.version
        variable.update_value("new_value", source_step="step1")
        
        assert variable.value == "new_value"
        assert variable.metadata.source_step == "step1"
        assert variable.metadata.version > original_version


class TestVariableManager:
    """Tests for VariableManager class."""
    
    @pytest.fixture
    def variable_manager(self):
        """Create a fresh VariableManager for each test."""
        return VariableManager(pipeline_id="test_pipeline")
    
    def test_initialization(self, variable_manager):
        """Test VariableManager initialization."""
        assert variable_manager.pipeline_id == "test_pipeline"
        assert len(variable_manager._variables) == 0
        assert len(variable_manager._scope_stack) == 0
    
    def test_set_and_get_variable(self, variable_manager):
        """Test basic variable setting and getting."""
        variable_manager.set_variable(
            "test_var",
            "test_value",
            scope=VariableScope.GLOBAL,
            var_type=VariableType.INPUT,
            description="Test variable"
        )
        
        value = variable_manager.get_variable("test_var")
        assert value == "test_value"
        
        # Test default value
        default_value = variable_manager.get_variable("nonexistent", "default")
        assert default_value == "default"
    
    def test_variable_metadata_retrieval(self, variable_manager):
        """Test retrieving variable metadata."""
        variable_manager.set_variable(
            "test_var",
            "test_value",
            scope=VariableScope.GLOBAL,
            var_type=VariableType.INPUT,
            description="Test variable",
            tags={"important", "test"}
        )
        
        metadata = variable_manager.get_variable_metadata("test_var")
        assert metadata is not None
        assert metadata.name == "test_var"
        assert metadata.scope == VariableScope.GLOBAL
        assert metadata.var_type == VariableType.INPUT
        assert metadata.description == "Test variable"
        assert "important" in metadata.tags
        assert "test" in metadata.tags
    
    def test_has_variable(self, variable_manager):
        """Test variable existence checking."""
        assert not variable_manager.has_variable("nonexistent")
        
        variable_manager.set_variable("existing", "value")
        assert variable_manager.has_variable("existing")
    
    def test_delete_variable(self, variable_manager):
        """Test variable deletion."""
        variable_manager.set_variable("to_delete", "value")
        assert variable_manager.has_variable("to_delete")
        
        result = variable_manager.delete_variable("to_delete")
        assert result is True
        assert not variable_manager.has_variable("to_delete")
        
        # Test deleting non-existent variable
        result = variable_manager.delete_variable("nonexistent")
        assert result is False
    
    def test_list_variables(self, variable_manager):
        """Test variable listing with filters."""
        variable_manager.set_variable(
            "global_input", "value1",
            scope=VariableScope.GLOBAL,
            var_type=VariableType.INPUT
        )
        variable_manager.set_variable(
            "global_output", "value2",
            scope=VariableScope.GLOBAL,
            var_type=VariableType.OUTPUT
        )
        variable_manager.set_variable(
            "step_var", "value3",
            scope=VariableScope.STEP,
            var_type=VariableType.INTERMEDIATE
        )
        
        # Test listing all variables
        all_vars = variable_manager.list_variables()
        assert len(all_vars) == 3
        assert "global_input" in all_vars
        assert "global_output" in all_vars
        assert "step_var" in all_vars
        
        # Test filtering by scope
        global_vars = variable_manager.list_variables(scope=VariableScope.GLOBAL)
        assert len(global_vars) == 2
        assert "global_input" in global_vars
        assert "global_output" in global_vars
        
        # Test filtering by type
        input_vars = variable_manager.list_variables(var_type=VariableType.INPUT)
        assert len(input_vars) == 1
        assert "global_input" in input_vars
        
        # Test including metadata
        vars_with_metadata = variable_manager.list_variables(include_metadata=True)
        assert len(vars_with_metadata) == 3
        assert "value" in vars_with_metadata["global_input"]
        assert "metadata" in vars_with_metadata["global_input"]
    
    def test_context_management(self, variable_manager):
        """Test context creation and isolation."""
        # Set global variable
        variable_manager.set_variable("global_var", "global_value")
        
        # Create context and set context variable
        context_id = variable_manager.create_context()
        variable_manager.set_variable(
            "context_var", "context_value", 
            context_id=context_id
        )
        
        # Test global variable access
        assert variable_manager.get_variable("global_var") == "global_value"
        
        # Test context variable access
        assert variable_manager.get_variable("context_var", context_id=context_id) == "context_value"
        assert variable_manager.get_variable("context_var") is None  # Not in global
        
        # Test context destruction
        variable_manager.destroy_context(context_id)
        assert variable_manager.get_variable("context_var", context_id=context_id) is None
    
    def test_scope_stack(self, variable_manager):
        """Test scope stack management."""
        variable_manager.set_variable("real_var", "real_value")
        
        # Push scope mapping
        scope_mapping = {"alias_var": "real_var"}
        variable_manager.push_scope(scope_mapping)
        
        # Test scope resolution
        value = variable_manager.get_variable("alias_var")
        assert value == "real_value"
        
        # Pop scope
        popped_mapping = variable_manager.pop_scope()
        assert popped_mapping == scope_mapping
        
        # Scope should no longer resolve
        assert variable_manager.get_variable("alias_var") is None
    
    def test_template_resolution(self, variable_manager):
        """Test template resolution functionality."""
        variable_manager.set_variable("name", "World")
        variable_manager.set_variable("greeting", "Hello")
        
        # Test simple template
        result = variable_manager.resolve_template("${greeting}, ${name}!")
        assert result == "Hello, World!"
        
        # Test template with missing variable
        result = variable_manager.resolve_template("${greeting}, ${unknown}!")
        assert result == "Hello, ${unknown}!"
        
        # Test with context
        context = {"unknown": "Claude"}
        result = variable_manager.resolve_template("${greeting}, ${unknown}!", context)
        assert result == "Hello, Claude!"
    
    def test_template_variables(self, variable_manager):
        """Test template variable setting and resolution."""
        variable_manager.set_variable("base", "Hello")
        variable_manager.set_template("dynamic", "${base}, World!")
        
        # Template should resolve when accessed
        result = variable_manager.get_variable("dynamic")
        assert result == "Hello, World!"
        
        # Update base variable and check template updates
        variable_manager.set_variable("base", "Hi")
        result = variable_manager.get_variable("dynamic")
        assert result == "Hi, World!"
    
    def test_dependency_tracking(self, variable_manager):
        """Test variable dependency tracking."""
        variable_manager.add_dependency("derived", {"base1", "base2"})
        
        deps = variable_manager.get_dependencies("derived")
        assert deps == {"base1", "base2"}
        
        # Test getting dependents
        dependents = variable_manager.get_dependents("base1")
        assert "derived" in dependents
    
    def test_event_handlers(self, variable_manager):
        """Test variable event handlers."""
        created_vars = []
        changed_vars = []
        
        def on_created(name, value):
            created_vars.append((name, value))
        
        def on_changed(name, old_value, new_value):
            changed_vars.append((name, old_value, new_value))
        
        variable_manager.on_variable_created(on_created)
        variable_manager.on_variable_changed(on_changed)
        
        # Create variable
        variable_manager.set_variable("test_var", "initial")
        assert len(created_vars) == 1
        assert created_vars[0] == ("test_var", "initial")
        
        # Change variable
        variable_manager.set_variable("test_var", "updated")
        assert len(changed_vars) == 1
        assert changed_vars[0] == ("test_var", "initial", "updated")
    
    def test_state_export_import(self, variable_manager):
        """Test state export and import."""
        # Set up variables
        variable_manager.set_variable(
            "test_var1", "value1",
            var_type=VariableType.INPUT,
            description="Test variable 1"
        )
        variable_manager.set_variable("test_var2", {"complex": "data"})
        
        # Add dependencies and templates
        variable_manager.add_dependency("derived", {"test_var1"})
        variable_manager.set_template("template_var", "${test_var1}_suffix")
        
        # Export state
        state = variable_manager.export_state()
        assert "global_variables" in state
        assert "dependency_graph" in state
        assert "variable_templates" in state
        assert state["pipeline_id"] == "test_pipeline"
        
        # Create new manager and import state
        new_manager = VariableManager("imported_pipeline")
        new_manager.import_state(state)
        
        # Verify imported data
        assert new_manager.pipeline_id == "test_pipeline"
        assert new_manager.get_variable("test_var1") == "value1"
        assert new_manager.get_variable("test_var2") == {"complex": "data"}
        assert new_manager.get_dependencies("derived") == {"test_var1"}
    
    def test_thread_safety(self, variable_manager):
        """Test thread safety of variable operations."""
        num_threads = 10
        num_operations = 100
        results = {}
        
        def worker_function(thread_id):
            for i in range(num_operations):
                var_name = f"thread_{thread_id}_var_{i}"
                variable_manager.set_variable(var_name, f"value_{i}")
                value = variable_manager.get_variable(var_name)
                results[var_name] = value
        
        # Start threads
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(results) == num_threads * num_operations
        for thread_id in range(num_threads):
            for i in range(num_operations):
                var_name = f"thread_{thread_id}_var_{i}"
                assert results[var_name] == f"value_{i}"


class TestVariableContext:
    """Tests for VariableContext context manager."""
    
    def test_context_manager(self):
        """Test variable context as context manager."""
        manager = VariableManager("test_pipeline")
        
        with VariableContext(manager) as ctx:
            # Set variable in context
            ctx.set_variable("context_var", "context_value")
            
            # Verify variable exists in context
            assert ctx.get_variable("context_var") == "context_value"
            
            # Verify context ID is set
            assert ctx.context_id is not None
        
        # Context should be cleaned up after exit
        assert ctx.context_id not in manager._context_variables
    
    def test_context_with_scope(self):
        """Test variable context with scope mapping."""
        manager = VariableManager("test_pipeline")
        manager.set_variable("global_var", "global_value")
        
        with VariableContext(manager) as ctx:
            # Set up scope mapping
            ctx.with_scope({"local_alias": "global_var"})
            
            # Test scope resolution
            value = manager.get_variable("local_alias")
            assert value == "global_value"
        
        # Scope should be cleaned up
        assert len(manager._scope_stack) == 0
    
    def test_context_isolation(self):
        """Test isolation between different contexts."""
        manager = VariableManager("test_pipeline")
        
        # Create two contexts
        with VariableContext(manager) as ctx1:
            ctx1.set_variable("var", "value1")
            
            with VariableContext(manager) as ctx2:
                ctx2.set_variable("var", "value2")
                
                # Variables should be isolated
                assert ctx1.get_variable("var") == "value1"
                assert ctx2.get_variable("var") == "value2"


class TestIntegrationScenarios:
    """Integration tests for complex scenarios."""
    
    def test_pipeline_simulation(self):
        """Test complete pipeline execution simulation."""
        manager = VariableManager("simulation_pipeline")
        
        # Set pipeline inputs
        manager.set_variable(
            "input_data", {"items": [1, 2, 3]},
            var_type=VariableType.INPUT,
            description="Pipeline input data"
        )
        
        # Simulate step execution with variable dependencies
        # Step 1: Process input
        input_data = manager.get_variable("input_data")
        processed_data = [item * 2 for item in input_data["items"]]
        
        manager.set_variable(
            "processed_data", processed_data,
            var_type=VariableType.INTERMEDIATE,
            source_step="step1"
        )
        
        # Step 2: Aggregate results
        processed_data = manager.get_variable("processed_data")
        total = sum(processed_data)
        
        manager.set_variable(
            "final_result", total,
            var_type=VariableType.OUTPUT,
            source_step="step2"
        )
        
        # Verify final results
        assert manager.get_variable("final_result") == 12  # (1+2+3) * 2 = 12
        
        # Check variable metadata
        result_metadata = manager.get_variable_metadata("final_result")
        assert result_metadata.var_type == VariableType.OUTPUT
        assert result_metadata.source_step == "step2"
    
    def test_template_dependencies(self):
        """Test complex template dependencies."""
        manager = VariableManager("template_pipeline")
        
        # Set base variables
        manager.set_variable("project_name", "MyProject")
        manager.set_variable("version", "1.0.0")
        manager.set_variable("environment", "production")
        
        # Set template variables with dependencies
        manager.set_template("docker_image", "${project_name}:${version}")
        manager.set_template("deployment_name", "${project_name}-${environment}")
        manager.set_template("full_spec", "${deployment_name}@${docker_image}")
        
        # Test template resolution
        assert manager.get_variable("docker_image") == "MyProject:1.0.0"
        assert manager.get_variable("deployment_name") == "MyProject-production"
        assert manager.get_variable("full_spec") == "MyProject-production@MyProject:1.0.0"
        
        # Test template updates when base variables change
        manager.set_variable("version", "1.1.0")
        assert manager.get_variable("docker_image") == "MyProject:1.1.0"
        assert manager.get_variable("full_spec") == "MyProject-production@MyProject:1.1.0"


if __name__ == "__main__":
    pytest.main([__file__])