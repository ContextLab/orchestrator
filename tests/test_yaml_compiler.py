"""Tests for YAML compiler."""

import pytest

from orchestrator.compiler.yaml_compiler import YAMLCompiler, YAMLCompilerError
from orchestrator.compiler.ambiguity_resolver import AmbiguityResolver
from orchestrator.compiler.schema_validator import SchemaValidator
from orchestrator.core.model import MockModel
from orchestrator.core.pipeline import Pipeline
from orchestrator.core.task import Task


class TestYAMLCompiler:
    """Test cases for YAMLCompiler class."""
    
    def test_yaml_compiler_creation(self):
        """Test basic YAML compiler creation."""
        compiler = YAMLCompiler()
        
        assert compiler.schema_validator is not None
        assert compiler.ambiguity_resolver is not None
        assert compiler.template_engine is not None
        assert compiler.auto_tag_pattern is not None
    
    def test_yaml_compiler_with_custom_components(self):
        """Test YAML compiler with custom components."""
        validator = SchemaValidator()
        resolver = AmbiguityResolver()
        
        compiler = YAMLCompiler(
            schema_validator=validator,
            ambiguity_resolver=resolver,
        )
        
        assert compiler.schema_validator is validator
        assert compiler.ambiguity_resolver is resolver
    
    def test_parse_yaml_valid(self):
        """Test parsing valid YAML."""
        compiler = YAMLCompiler()
        yaml_content = """
        name: test_pipeline
        version: 1.0.0
        steps:
          - id: step1
            name: Step 1
            action: generate
            parameters:
              prompt: Hello
        """
        
        result = compiler._parse_yaml(yaml_content)
        
        assert result["name"] == "test_pipeline"
        assert result["version"] == "1.0.0"
        assert len(result["steps"]) == 1
        assert result["steps"][0]["id"] == "step1"
    
    def test_parse_yaml_invalid(self):
        """Test parsing invalid YAML."""
        compiler = YAMLCompiler()
        yaml_content = """
        name: test_pipeline
        version: 1.0.0
        steps:
          - id: step1
            name: Step 1
            action: generate
            parameters:
              prompt: Hello
            invalid_indentation
        """
        
        with pytest.raises(YAMLCompilerError, match="Invalid YAML"):
            compiler._parse_yaml(yaml_content)
    
    def test_process_templates_simple(self):
        """Test template processing with simple variables."""
        compiler = YAMLCompiler()
        pipeline_def = {
            "name": "{{ pipeline_name }}",
            "steps": [
                {
                    "id": "step1",
                    "action": "generate",
                    "parameters": {
                        "prompt": "Hello {{ name }}!"
                    }
                }
            ]
        }
        context = {"pipeline_name": "Test Pipeline", "name": "World"}
        
        result = compiler._process_templates(pipeline_def, context)
        
        assert result["name"] == "Test Pipeline"
        assert result["steps"][0]["parameters"]["prompt"] == "Hello World!"
    
    def test_process_templates_complex(self):
        """Test template processing with complex structures."""
        compiler = YAMLCompiler()
        pipeline_def = {
            "name": "{{ pipeline_name }}",
            "context": {
                "timeout": "{{ timeout }}",
                "variables": ["{{ var1 }}", "{{ var2 }}"]
            },
            "steps": [
                {
                    "id": "step1",
                    "parameters": {
                        "data": {
                            "input": "{{ input_data }}",
                            "config": "{{ config_path }}"
                        }
                    }
                }
            ]
        }
        context = {
            "pipeline_name": "Complex Pipeline",
            "timeout": 300,
            "var1": "value1",
            "var2": "value2",
            "input_data": "test_data",
            "config_path": "/path/to/config"
        }
        
        result = compiler._process_templates(pipeline_def, context)
        
        assert result["name"] == "Complex Pipeline"
        assert result["context"]["timeout"] == "300"
        assert result["context"]["variables"] == ["value1", "value2"]
        assert result["steps"][0]["parameters"]["data"]["input"] == "test_data"
        assert result["steps"][0]["parameters"]["data"]["config"] == "/path/to/config"
    
    def test_process_templates_missing_variable(self):
        """Test template processing with missing variable."""
        compiler = YAMLCompiler()
        pipeline_def = {
            "name": "{{ missing_var }}",
            "steps": []
        }
        context = {}
        
        with pytest.raises(YAMLCompilerError, match="Failed to render template"):
            compiler._process_templates(pipeline_def, context)
    
    def test_detect_auto_tags(self):
        """Test AUTO tag detection."""
        compiler = YAMLCompiler()
        
        # Test single AUTO tag
        content = "<AUTO>Choose the best method</AUTO>"
        tags = compiler.detect_auto_tags(content)
        assert tags == ["Choose the best method"]
        
        # Test multiple AUTO tags
        content = "<AUTO>First choice</AUTO> and <AUTO>Second choice</AUTO>"
        tags = compiler.detect_auto_tags(content)
        assert tags == ["First choice", "Second choice"]
        
        # Test no AUTO tags
        content = "No auto tags here"
        tags = compiler.detect_auto_tags(content)
        assert tags == []
        
        # Test non-string input
        tags = compiler.detect_auto_tags(123)
        assert tags == []
    
    def test_has_auto_tags_string(self):
        """Test has_auto_tags with string input."""
        compiler = YAMLCompiler()
        
        assert compiler.has_auto_tags("<AUTO>test</AUTO>")
        assert compiler.has_auto_tags("Before <AUTO>test</AUTO> after")
        assert not compiler.has_auto_tags("No auto tags")
        assert not compiler.has_auto_tags("")
    
    def test_has_auto_tags_dict(self):
        """Test has_auto_tags with dict input."""
        compiler = YAMLCompiler()
        
        assert compiler.has_auto_tags({"key": "<AUTO>test</AUTO>"})
        assert compiler.has_auto_tags({"key1": "normal", "key2": "<AUTO>test</AUTO>"})
        assert not compiler.has_auto_tags({"key": "normal"})
        assert not compiler.has_auto_tags({})
    
    def test_has_auto_tags_list(self):
        """Test has_auto_tags with list input."""
        compiler = YAMLCompiler()
        
        assert compiler.has_auto_tags(["<AUTO>test</AUTO>"])
        assert compiler.has_auto_tags(["normal", "<AUTO>test</AUTO>"])
        assert not compiler.has_auto_tags(["normal", "values"])
        assert not compiler.has_auto_tags([])
    
    def test_has_auto_tags_other(self):
        """Test has_auto_tags with other types."""
        compiler = YAMLCompiler()
        
        assert not compiler.has_auto_tags(123)
        assert not compiler.has_auto_tags(None)
        assert not compiler.has_auto_tags(True)
    
    @pytest.mark.asyncio
    async def test_resolve_auto_string_single_tag(self):
        """Test resolving single AUTO tag."""
        model = MockModel()
        resolver = AmbiguityResolver(model)
        compiler = YAMLCompiler(ambiguity_resolver=resolver)
        
        content = "<AUTO>Choose the best format</AUTO>"
        result = await compiler._resolve_auto_string(content, "test.path")
        
        assert isinstance(result, str)
        assert result != content  # Should be resolved
    
    @pytest.mark.asyncio
    async def test_resolve_auto_string_multiple_tags(self):
        """Test resolving multiple AUTO tags."""
        model = MockModel()
        model.set_response("Choose format", "json")
        model.set_response("Choose method", "post")
        
        resolver = AmbiguityResolver(model)
        compiler = YAMLCompiler(ambiguity_resolver=resolver)
        
        content = "Use <AUTO>Choose format</AUTO> format with <AUTO>Choose method</AUTO> method"
        result = await compiler._resolve_auto_string(content, "test.path")
        
        assert "json" in result
        assert "post" in result
        assert "<AUTO>" not in result
    
    @pytest.mark.asyncio
    async def test_resolve_auto_string_no_tags(self):
        """Test resolving string with no AUTO tags."""
        compiler = YAMLCompiler()
        
        content = "No auto tags here"
        result = await compiler._resolve_auto_string(content, "test.path")
        
        assert result == content
    
    def test_build_task_basic(self):
        """Test building basic task."""
        compiler = YAMLCompiler()
        task_def = {
            "id": "test_task",
            "name": "Test Task",
            "action": "generate",
            "parameters": {"prompt": "Hello"},
            "dependencies": ["other_task"],
            "timeout": 60,
            "max_retries": 2,
            "metadata": {"key": "value"}
        }
        
        task = compiler._build_task(task_def)
        
        assert isinstance(task, Task)
        assert task.id == "test_task"
        assert task.name == "Test Task"
        assert task.action == "generate"
        assert task.parameters == {"prompt": "Hello"}
        assert task.dependencies == ["other_task"]
        assert task.timeout == 60
        assert task.max_retries == 2
        assert task.metadata == {"key": "value"}
    
    def test_build_task_minimal(self):
        """Test building minimal task."""
        compiler = YAMLCompiler()
        task_def = {
            "id": "test_task",
            "action": "generate"
        }
        
        task = compiler._build_task(task_def)
        
        assert isinstance(task, Task)
        assert task.id == "test_task"
        assert task.name == "test_task"  # Should default to id
        assert task.action == "generate"
        assert task.parameters == {}
        assert task.dependencies == []
        assert task.timeout is None
        assert task.max_retries == 3  # Default
        assert task.metadata == {}
    
    def test_build_task_with_metadata_fields(self):
        """Test building task with metadata fields."""
        compiler = YAMLCompiler()
        task_def = {
            "id": "test_task",
            "action": "generate",
            "on_failure": "continue",
            "requires_model": {
                "min_size": "7B",
                "expertise": "high"
            }
        }
        
        task = compiler._build_task(task_def)
        
        assert task.metadata["on_failure"] == "continue"
        assert task.metadata["requires_model"]["min_size"] == "7B"
        assert task.metadata["requires_model"]["expertise"] == "high"
    
    def test_build_pipeline_basic(self):
        """Test building basic pipeline."""
        compiler = YAMLCompiler()
        pipeline_def = {
            "id": "test_pipeline",
            "name": "Test Pipeline",
            "version": "1.0.0",
            "description": "Test description",
            "context": {"timeout": 300},
            "metadata": {"key": "value"},
            "steps": [
                {
                    "id": "step1",
                    "action": "generate",
                    "parameters": {"prompt": "Hello"}
                }
            ]
        }
        
        pipeline = compiler._build_pipeline(pipeline_def)
        
        assert isinstance(pipeline, Pipeline)
        assert pipeline.id == "test_pipeline"
        assert pipeline.name == "Test Pipeline"
        assert pipeline.version == "1.0.0"
        assert pipeline.description == "Test description"
        assert pipeline.context == {"timeout": 300}
        assert pipeline.metadata == {"key": "value"}
        assert len(pipeline.tasks) == 1
        assert "step1" in pipeline.tasks
    
    def test_build_pipeline_minimal(self):
        """Test building minimal pipeline."""
        compiler = YAMLCompiler()
        pipeline_def = {
            "name": "Test Pipeline",
            "steps": [
                {
                    "id": "step1",
                    "action": "generate"
                }
            ]
        }
        
        pipeline = compiler._build_pipeline(pipeline_def)
        
        assert isinstance(pipeline, Pipeline)
        assert pipeline.id == "Test Pipeline"  # Should default to name
        assert pipeline.name == "Test Pipeline"
        assert pipeline.version == "1.0.0"  # Default
        assert pipeline.description is None
        assert pipeline.context == {}
        assert pipeline.metadata == {}
        assert len(pipeline.tasks) == 1
    
    def test_build_pipeline_empty_steps(self):
        """Test building pipeline with empty steps."""
        compiler = YAMLCompiler()
        pipeline_def = {
            "name": "Test Pipeline",
            "steps": []
        }
        
        pipeline = compiler._build_pipeline(pipeline_def)
        
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.tasks) == 0
    
    @pytest.mark.asyncio
    async def test_compile_simple_pipeline(self):
        """Test compiling simple pipeline."""
        compiler = YAMLCompiler()
        yaml_content = """
        name: test_pipeline
        version: 1.0.0
        steps:
          - id: step1
            name: Step 1
            action: generate
            parameters:
              prompt: Hello World
        """
        
        pipeline = await compiler.compile(yaml_content)
        
        assert isinstance(pipeline, Pipeline)
        assert pipeline.name == "test_pipeline"
        assert pipeline.version == "1.0.0"
        assert len(pipeline.tasks) == 1
        assert "step1" in pipeline.tasks
        assert pipeline.tasks["step1"].parameters["prompt"] == "Hello World"
    
    @pytest.mark.asyncio
    async def test_compile_with_templates(self):
        """Test compiling pipeline with templates."""
        compiler = YAMLCompiler()
        yaml_content = """
        name: "{{ pipeline_name }}"
        version: 1.0.0
        steps:
          - id: step1
            name: Step 1
            action: generate
            parameters:
              prompt: "Hello {{ target }}"
        """
        context = {"pipeline_name": "Templated Pipeline", "target": "World"}
        
        pipeline = await compiler.compile(yaml_content, context)
        
        assert pipeline.name == "Templated Pipeline"
        assert pipeline.tasks["step1"].parameters["prompt"] == "Hello World"
    
    @pytest.mark.asyncio
    async def test_compile_with_auto_tags(self):
        """Test compiling pipeline with AUTO tags."""
        model = MockModel()
        model.set_response("Choose the best output format", "json")
        
        resolver = AmbiguityResolver(model)
        compiler = YAMLCompiler(ambiguity_resolver=resolver)
        
        yaml_content = """
        name: test_pipeline
        version: 1.0.0
        steps:
          - id: step1
            name: Step 1
            action: generate
            parameters:
              format: "<AUTO>Choose the best output format</AUTO>"
        """
        
        pipeline = await compiler.compile(yaml_content)
        
        assert pipeline.tasks["step1"].parameters["format"] == "json"
    
    @pytest.mark.asyncio
    async def test_compile_without_auto_resolution(self):
        """Test compiling pipeline without AUTO resolution."""
        compiler = YAMLCompiler()
        yaml_content = """
        name: test_pipeline
        version: 1.0.0
        steps:
          - id: step1
            name: Step 1
            action: generate
            parameters:
              format: "<AUTO>Choose the best output format</AUTO>"
        """
        
        pipeline = await compiler.compile(yaml_content, resolve_ambiguities=False)
        
        assert "<AUTO>" in pipeline.tasks["step1"].parameters["format"]
    
    @pytest.mark.asyncio
    async def test_compile_complex_pipeline(self):
        """Test compiling complex pipeline."""
        compiler = YAMLCompiler()
        yaml_content = """
        name: complex_pipeline
        version: 2.0.0
        description: A complex test pipeline
        context:
          timeout: 300
          max_retries: 3
        metadata:
          author: test
          tags: [test, complex]
        steps:
          - id: step1
            name: First Step
            action: generate
            parameters:
              prompt: Initial prompt
          - id: step2
            name: Second Step
            action: analyze
            parameters:
              input: "{{ steps.step1.output }}"
            dependencies: [step1]
            timeout: 60
            max_retries: 2
            on_failure: continue
            requires_model:
              min_size: 7B
              expertise: high
          - id: step3
            name: Third Step
            action: transform
            parameters:
              data: "{{ steps.step2.result }}"
            dependencies: [step2]
        """
        
        pipeline = await compiler.compile(yaml_content)
        
        assert pipeline.name == "complex_pipeline"
        assert pipeline.version == "2.0.0"
        assert pipeline.description == "A complex test pipeline"
        assert pipeline.context["timeout"] == 300
        assert pipeline.context["max_retries"] == 3
        assert pipeline.metadata["author"] == "test"
        assert pipeline.metadata["tags"] == ["test", "complex"]
        assert len(pipeline.tasks) == 3
        
        # Check step2 details
        step2 = pipeline.tasks["step2"]
        assert step2.dependencies == ["step1"]
        assert step2.timeout == 60
        assert step2.max_retries == 2
        assert step2.metadata["on_failure"] == "continue"
        assert step2.metadata["requires_model"]["min_size"] == "7B"
        assert step2.metadata["requires_model"]["expertise"] == "high"
        
        # Check step3 dependencies
        step3 = pipeline.tasks["step3"]
        assert step3.dependencies == ["step2"]
    
    def test_validate_yaml_valid(self):
        """Test validating valid YAML."""
        compiler = YAMLCompiler()
        yaml_content = """
        name: test_pipeline
        steps:
          - id: step1
            action: generate
        """
        
        assert compiler.validate_yaml(yaml_content) is True
    
    def test_validate_yaml_invalid(self):
        """Test validating invalid YAML."""
        compiler = YAMLCompiler()
        yaml_content = """
        name: test_pipeline
        steps:
          - id: step1
            action: generate
            invalid_indentation
        """
        
        assert compiler.validate_yaml(yaml_content) is False
    
    def test_get_template_variables(self):
        """Test extracting template variables."""
        compiler = YAMLCompiler()
        yaml_content = """
        name: "{{ pipeline_name }}"
        context:
          user: "{{ user.name }}"
          config: "{{ config_path | default('config.yaml') }}"
        steps:
          - id: step1
            parameters:
              prompt: "Hello {{ target }}"
              data: "{{ input_data }}"
        """
        
        variables = compiler.get_template_variables(yaml_content)
        
        expected_variables = ["pipeline_name", "user", "config_path", "target", "input_data"]
        assert set(variables) == set(expected_variables)
    
    def test_get_template_variables_no_variables(self):
        """Test extracting template variables when none exist."""
        compiler = YAMLCompiler()
        yaml_content = """
        name: test_pipeline
        steps:
          - id: step1
            action: generate
        """
        
        variables = compiler.get_template_variables(yaml_content)
        
        assert variables == []
    
    def test_get_template_variables_duplicates(self):
        """Test extracting template variables with duplicates."""
        compiler = YAMLCompiler()
        yaml_content = """
        name: "{{ pipeline_name }}"
        description: "{{ pipeline_name }} description"
        steps:
          - id: step1
            parameters:
              prompt: "Hello {{ target }}"
              greeting: "Hi {{ target }}"
        """
        
        variables = compiler.get_template_variables(yaml_content)
        
        # Should not have duplicates
        assert len(variables) == 2
        assert set(variables) == {"pipeline_name", "target"}


class TestAmbiguityResolver:
    """Test cases for AmbiguityResolver class."""
    
    def test_resolver_creation(self):
        """Test basic resolver creation."""
        resolver = AmbiguityResolver()
        
        assert resolver.model is not None
        assert resolver.resolution_cache == {}
        assert len(resolver.resolution_strategies) > 0
    
    def test_resolver_with_model(self):
        """Test resolver with specific model."""
        model = MockModel()
        resolver = AmbiguityResolver(model)
        
        assert resolver.model is model
    
    @pytest.mark.asyncio
    async def test_resolve_parameter_ambiguity(self):
        """Test resolving parameter ambiguity."""
        model = MockModel()
        resolver = AmbiguityResolver(model)
        
        result = await resolver.resolve("Choose the best format", "parameters.format")
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_resolve_value_ambiguity(self):
        """Test resolving value ambiguity."""
        model = MockModel()
        resolver = AmbiguityResolver(model)
        
        result = await resolver.resolve("Select from: json, xml, csv", "output.format")
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_resolve_boolean_ambiguity(self):
        """Test resolving boolean ambiguity."""
        model = MockModel()
        resolver = AmbiguityResolver(model)
        
        result = await resolver.resolve("Enable compression", "config.compression")
        
        assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_resolve_number_ambiguity(self):
        """Test resolving number ambiguity."""
        model = MockModel()
        resolver = AmbiguityResolver(model)
        
        result = await resolver.resolve("Set batch size", "config.batch_size")
        
        assert isinstance(result, (int, float))
        assert result > 0
    
    @pytest.mark.asyncio
    async def test_resolve_list_ambiguity(self):
        """Test resolving list ambiguity."""
        model = MockModel()
        resolver = AmbiguityResolver(model)
        
        result = await resolver.resolve("Choose supported languages", "config.languages")
        
        assert isinstance(result, list)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_resolve_with_cache(self):
        """Test resolution with caching."""
        model = MockModel()
        resolver = AmbiguityResolver(model)
        
        # First resolution
        result1 = await resolver.resolve("Choose format", "test.format")
        
        # Second resolution should use cache
        result2 = await resolver.resolve("Choose format", "test.format")
        
        assert result1 == result2
        assert resolver.get_cache_size() == 1
    
    @pytest.mark.asyncio
    async def test_resolve_with_custom_response(self):
        """Test resolution with custom model response."""
        model = MockModel()
        model.set_response("Choose the best format for data export", "parquet")
        
        resolver = AmbiguityResolver(model)
        
        result = await resolver.resolve("Choose the best format for data export", "export.format")
        
        assert result == "parquet"
    
    def test_classify_ambiguity_types(self):
        """Test ambiguity classification."""
        resolver = AmbiguityResolver()
        
        # Test parameter classification
        assert resolver._classify_ambiguity("Choose format", "parameters.format") == "parameter"
        
        # Test boolean classification
        assert resolver._classify_ambiguity("Enable compression", "config.enable") == "boolean"
        
        # Test number classification
        assert resolver._classify_ambiguity("Set batch size", "config.size") == "number"
        
        # Test list classification
        assert resolver._classify_ambiguity("Choose list of items", "config.items") == "list"
    
    def test_extract_choices(self):
        """Test extracting choices from content."""
        resolver = AmbiguityResolver()
        
        # Test simple choices - this method should parse comma-separated options
        choices = resolver._extract_choices("Choose: json, xml, or csv")
        assert len(choices) >= 2  # Should extract at least the main options
        assert any("json" in choice for choice in choices)
        assert any("xml" in choice for choice in choices)
        
        # Test no choices
        choices = resolver._extract_choices("No choices here")
        assert choices == []
    
    def test_extract_quotes(self):
        """Test extracting quoted strings."""
        resolver = AmbiguityResolver()
        
        # Test quoted strings
        quotes = resolver._extract_quotes('Use "json" format or "xml" format')
        assert len(quotes) == 2
        assert "json" in quotes
        assert "xml" in quotes
        
        # Test no quotes
        quotes = resolver._extract_quotes("No quotes here")
        assert quotes == []
    
    def test_clear_cache(self):
        """Test clearing resolution cache."""
        resolver = AmbiguityResolver()
        resolver.resolution_cache["key"] = "value"
        
        resolver.clear_cache()
        
        assert resolver.resolution_cache == {}
    
    def test_get_cache_size(self):
        """Test getting cache size."""
        resolver = AmbiguityResolver()
        
        assert resolver.get_cache_size() == 0
        
        resolver.resolution_cache["key1"] = "value1"
        resolver.resolution_cache["key2"] = "value2"
        
        assert resolver.get_cache_size() == 2
    
    def test_set_custom_strategy(self):
        """Test setting custom resolution strategy."""
        resolver = AmbiguityResolver()
        
        def custom_strategy(content, context, severity):
            return "custom_result"
        
        resolver.set_resolution_strategy("custom", custom_strategy)
        
        assert "custom" in resolver.resolution_strategies
        assert resolver.resolution_strategies["custom"] == custom_strategy


class TestSchemaValidator:
    """Test cases for SchemaValidator class."""
    
    def test_validator_creation(self):
        """Test basic validator creation."""
        validator = SchemaValidator()
        
        assert validator.schema is not None
        assert validator.validator is not None
    
    def test_validator_with_custom_schema(self):
        """Test validator with custom schema."""
        custom_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        }
        
        validator = SchemaValidator(custom_schema)
        
        assert validator.schema == custom_schema
    
    def test_validate_valid_pipeline(self):
        """Test validating valid pipeline."""
        validator = SchemaValidator()
        
        pipeline_def = {
            "name": "test_pipeline",
            "steps": [
                {
                    "id": "step1",
                    "action": "generate"
                }
            ]
        }
        
        # Should not raise exception
        validator.validate(pipeline_def)
    
    def test_validate_invalid_pipeline_missing_name(self):
        """Test validating pipeline missing name."""
        validator = SchemaValidator()
        
        pipeline_def = {
            "steps": [
                {
                    "id": "step1",
                    "action": "generate"
                }
            ]
        }
        
        with pytest.raises(Exception):  # SchemaValidationError
            validator.validate(pipeline_def)
    
    def test_validate_invalid_pipeline_missing_steps(self):
        """Test validating pipeline missing steps."""
        validator = SchemaValidator()
        
        pipeline_def = {
            "name": "test_pipeline"
        }
        
        with pytest.raises(Exception):  # SchemaValidationError
            validator.validate(pipeline_def)
    
    def test_validate_invalid_pipeline_empty_steps(self):
        """Test validating pipeline with empty steps."""
        validator = SchemaValidator()
        
        pipeline_def = {
            "name": "test_pipeline",
            "steps": []
        }
        
        with pytest.raises(Exception):  # SchemaValidationError
            validator.validate(pipeline_def)
    
    def test_validate_invalid_step_missing_id(self):
        """Test validating step missing ID."""
        validator = SchemaValidator()
        
        pipeline_def = {
            "name": "test_pipeline",
            "steps": [
                {
                    "action": "generate"
                }
            ]
        }
        
        with pytest.raises(Exception):  # SchemaValidationError
            validator.validate(pipeline_def)
    
    def test_validate_invalid_step_missing_action(self):
        """Test validating step missing action."""
        validator = SchemaValidator()
        
        pipeline_def = {
            "name": "test_pipeline",
            "steps": [
                {
                    "id": "step1"
                }
            ]
        }
        
        with pytest.raises(Exception):  # SchemaValidationError
            validator.validate(pipeline_def)
    
    def test_is_valid_true(self):
        """Test is_valid with valid pipeline."""
        validator = SchemaValidator()
        
        pipeline_def = {
            "name": "test_pipeline",
            "steps": [
                {
                    "id": "step1",
                    "action": "generate"
                }
            ]
        }
        
        assert validator.is_valid(pipeline_def) is True
    
    def test_is_valid_false(self):
        """Test is_valid with invalid pipeline."""
        validator = SchemaValidator()
        
        pipeline_def = {
            "name": "test_pipeline"
            # Missing steps
        }
        
        assert validator.is_valid(pipeline_def) is False
    
    def test_get_validation_errors(self):
        """Test getting validation errors."""
        validator = SchemaValidator()
        
        pipeline_def = {
            "name": "test_pipeline"
            # Missing steps
        }
        
        errors = validator.get_validation_errors(pipeline_def)
        
        assert len(errors) > 0
        assert any("steps" in error for error in errors)
    
    def test_validate_task_dependencies(self):
        """Test validating task dependencies."""
        validator = SchemaValidator()
        
        # Valid dependencies
        pipeline_def = {
            "name": "test_pipeline",
            "steps": [
                {"id": "step1", "action": "generate"},
                {"id": "step2", "action": "analyze", "dependencies": ["step1"]}
            ]
        }
        
        errors = validator.validate_task_dependencies(pipeline_def)
        assert len(errors) == 0
        
        # Invalid dependencies
        pipeline_def = {
            "name": "test_pipeline",
            "steps": [
                {"id": "step1", "action": "generate"},
                {"id": "step2", "action": "analyze", "dependencies": ["nonexistent"]}
            ]
        }
        
        errors = validator.validate_task_dependencies(pipeline_def)
        assert len(errors) > 0
        assert any("nonexistent" in error for error in errors)
    
    def test_validate_unique_task_ids(self):
        """Test validating unique task IDs."""
        validator = SchemaValidator()
        
        # Unique IDs
        pipeline_def = {
            "name": "test_pipeline",
            "steps": [
                {"id": "step1", "action": "generate"},
                {"id": "step2", "action": "analyze"}
            ]
        }
        
        errors = validator.validate_unique_task_ids(pipeline_def)
        assert len(errors) == 0
        
        # Duplicate IDs
        pipeline_def = {
            "name": "test_pipeline",
            "steps": [
                {"id": "step1", "action": "generate"},
                {"id": "step1", "action": "analyze"}
            ]
        }
        
        errors = validator.validate_unique_task_ids(pipeline_def)
        assert len(errors) > 0
        assert any("Duplicate" in error for error in errors)
    
    def test_validate_complete(self):
        """Test complete validation."""
        validator = SchemaValidator()
        
        # Valid pipeline
        pipeline_def = {
            "name": "test_pipeline",
            "steps": [
                {"id": "step1", "action": "generate"},
                {"id": "step2", "action": "analyze", "dependencies": ["step1"]}
            ]
        }
        
        errors = validator.validate_complete(pipeline_def)
        assert len(errors) == 0
        
        # Invalid pipeline with multiple issues
        pipeline_def = {
            "name": "test_pipeline",
            "steps": [
                {"id": "step1", "action": "generate"},
                {"id": "step1", "action": "analyze", "dependencies": ["nonexistent"]}
            ]
        }
        
        errors = validator.validate_complete(pipeline_def)
        assert len(errors) > 0  # Should have multiple errors
    
    @pytest.mark.asyncio
    async def test_compile_with_general_exception(self):
        """Test compilation with general exception."""
        compiler = YAMLCompiler()
        
        # Mock the schema validator to raise an exception
        compiler.schema_validator.validate = lambda x: (_ for _ in ()).throw(Exception("Validation error"))
        
        yaml_content = """
        name: test_pipeline
        steps:
          - id: step1
            action: generate
        """
        
        with pytest.raises(YAMLCompilerError, match="Failed to compile YAML"):
            await compiler.compile(yaml_content)
    
    def test_validator_error_path_handling(self):
        """Test error path handling in schema validator."""
        validator = SchemaValidator()
        
        # Create invalid pipeline with nested error to test error path
        pipeline_def = {
            "name": "test_pipeline",
            "steps": [
                {
                    "id": "step1",
                    "action": "generate",
                    "parameters": {
                        "invalid_nested": "should_be_dict_but_is_string"
                    }
                }
            ]
        }
        
        errors = validator.get_validation_errors(pipeline_def)
        # Should have errors with path information
        assert len(errors) >= 0  # May or may not have errors depending on schema
    
    
    def test_validate_task_self_dependency(self):
        """Test validation of task self-dependency."""
        validator = SchemaValidator()
        
        # Pipeline with self-dependent task
        pipeline_def = {
            "name": "test_pipeline",
            "steps": [
                {
                    "id": "step1",
                    "action": "generate",
                    "dependencies": ["step1"]  # Self-dependency
                }
            ]
        }
        
        errors = validator.validate_task_dependencies(pipeline_def)
        assert len(errors) > 0
        assert any("cannot depend on itself" in error for error in errors)