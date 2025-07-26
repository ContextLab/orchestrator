"""Tests for template resolution in AUTO tags."""

import pytest
from orchestrator.compiler.yaml_compiler import YAMLCompiler
from orchestrator import init_models


@pytest.fixture(scope="module")
def model_registry():
    """Initialize models once for all tests in this module."""
    return init_models()


@pytest.fixture
def yaml_compiler(model_registry):
    """Create YAML compiler with model registry."""
    return YAMLCompiler(model_registry=model_registry)


class TestTemplateResolution:
    """Test template resolution before AUTO tag processing."""

    @pytest.mark.asyncio
    async def test_compile_time_template_resolution(self, yaml_compiler):
        """Test that compile-time templates are resolved before AUTO tags."""
        # Track what the ambiguity resolver receives
        captured_prompts = []
        original_resolve = yaml_compiler.ambiguity_resolver.resolve

        async def capture_resolve(content, context_path):
            captured_prompts.append(
                {
                    "path": context_path,
                    "content": content,
                    "has_unresolved_template": "{{" in content,
                }
            )
            # Return mock value
            if "batch_size" in context_path:
                return 32
            elif "workers" in context_path:
                return 4
            return "resolved"

        yaml_compiler.ambiguity_resolver.resolve = capture_resolve

        # Test YAML with compile-time templates in AUTO tags
        yaml_content = """
id: test-compile-time
name: Test Compile Time Templates
inputs:
  batch_size: 100
  environment: production

steps:
  - id: configure
    action: process
    parameters:
      # Regular template (should resolve to "100")
      size: "{{batch_size}}"
      
      # AUTO tag with compile-time template (should resolve before AI)
      workers: <AUTO>For batch size {{batch_size}}, how many workers?</AUTO>
      
      # AUTO tag with multiple templates
      strategy: <AUTO>In {{environment}} with batch size {{batch_size}}, what strategy?</AUTO>
"""

        # Compile the pipeline
        pipeline = await yaml_compiler.compile(yaml_content)

        # Verify pipeline compiled
        assert pipeline is not None
        assert pipeline.name == "Test Compile Time Templates"

        # Verify AUTO tags received resolved values
        assert len(captured_prompts) >= 2

        # Check workers AUTO tag
        workers_prompt = next(p for p in captured_prompts if "workers" in p["path"])
        assert workers_prompt["content"] == "For batch size 100, how many workers?"
        assert not workers_prompt["has_unresolved_template"]

        # Check strategy AUTO tag
        strategy_prompt = next(p for p in captured_prompts if "strategy" in p["path"])
        assert "production" in strategy_prompt["content"]
        assert "100" in strategy_prompt["content"]
        assert not strategy_prompt["has_unresolved_template"]

    @pytest.mark.asyncio
    async def test_runtime_template_preservation(self, yaml_compiler):
        """Test that runtime templates are preserved for later resolution."""
        captured_prompts = []
        original_resolve = yaml_compiler.ambiguity_resolver.resolve

        async def capture_resolve(content, context_path):
            captured_prompts.append(content)
            return "mock_value"

        yaml_compiler.ambiguity_resolver.resolve = capture_resolve

        yaml_content = """
id: test-runtime
name: Test Runtime Templates
inputs:
  initial_value: 42

steps:
  - id: analyze
    action: analyze
    parameters:
      value: "{{initial_value}}"  # Should resolve at compile time
      
  - id: process
    action: process
    parameters:
      # Runtime reference - should be preserved
      based_on: "{{analyze.result}}"
      # AUTO tag with runtime reference
      decision: <AUTO>Based on {{analyze.result}}, what next?</AUTO>
    depends_on: [analyze]
"""

        pipeline = await yaml_compiler.compile(yaml_content)

        # Check that compile-time value was resolved
        analyze_task = pipeline.tasks.get("analyze")
        assert analyze_task is not None
        assert analyze_task.parameters["value"] == "42"

        # Check that runtime reference was preserved
        process_task = pipeline.tasks.get("process")
        assert process_task is not None
        assert "{{analyze.result}}" in process_task.parameters["based_on"]

        # AUTO tag with runtime reference should be captured
        assert any("{{analyze.result}}" in prompt for prompt in captured_prompts)

    @pytest.mark.asyncio
    async def test_input_default_resolution(self, yaml_compiler):
        """Test that input defaults are properly resolved."""
        captured = []
        original_resolve = yaml_compiler.ambiguity_resolver.resolve

        async def capture_resolve(content, context_path):
            captured.append(content)
            return 10

        yaml_compiler.ambiguity_resolver.resolve = capture_resolve

        yaml_content = """
id: test-defaults
name: Test Input Defaults
inputs:
  # Direct value
  direct_value: 100
  # Default in dict
  with_default:
    type: integer
    default: 200
    description: "Test value"

steps:
  - id: test
    action: process
    parameters:
      auto1: <AUTO>Direct value is {{direct_value}}</AUTO>
      auto2: <AUTO>Default value is {{with_default}}</AUTO>
"""

        pipeline = await yaml_compiler.compile(yaml_content)

        # Both formats should work
        assert len(captured) == 2
        assert "Direct value is 100" in captured
        assert "Default value is 200" in captured

    @pytest.mark.asyncio
    async def test_nested_input_resolution(self, yaml_compiler):
        """Test resolution of nested input values."""
        captured = []
        original_resolve = yaml_compiler.ambiguity_resolver.resolve

        async def capture_resolve(content, context_path):
            captured.append(content)
            return "resolved"

        yaml_compiler.ambiguity_resolver.resolve = capture_resolve

        yaml_content = """
id: test-nested
name: Test Nested Inputs
inputs:
  config:
    database:
      host: "localhost"
      port: 5432
    cache:
      enabled: true
      ttl: 3600

steps:
  - id: configure
    action: setup
    parameters:
      db_auto: <AUTO>Connect to {{config.database.host}}:{{config.database.port}}</AUTO>
      cache_auto: <AUTO>Cache enabled: {{config.cache.enabled}}, TTL: {{config.cache.ttl}}</AUTO>
"""

        pipeline = await yaml_compiler.compile(yaml_content)

        # Check nested values were resolved
        assert len(captured) == 2
        assert "Connect to localhost:5432" in captured
        assert "Cache enabled: True, TTL: 3600" in captured

    @pytest.mark.asyncio
    async def test_no_auto_tag_resolution_without_models(self):
        """Test that YAML compiler raises error when no models available."""
        with pytest.raises(ValueError, match="No model registry provided"):
            YAMLCompiler()

    @pytest.mark.asyncio
    async def test_mixed_compile_and_runtime_templates(self, yaml_compiler):
        """Test AUTO tags with both compile-time and runtime templates."""
        captured = []
        original_resolve = yaml_compiler.ambiguity_resolver.resolve

        async def capture_resolve(content, context_path):
            captured.append(content)
            return "optimized"

        yaml_compiler.ambiguity_resolver.resolve = capture_resolve

        yaml_content = """
id: test-mixed
name: Test Mixed Templates
inputs:
  timeout: 30
  max_retries: 3

steps:
  - id: fetch
    action: fetch
    parameters:
      timeout: "{{timeout}}"
      
  - id: process
    action: process
    parameters:
      # Mix of compile-time and runtime templates
      strategy: <AUTO>With timeout {{timeout}}s and data {{fetch.result}}, how to process?</AUTO>
    depends_on: [fetch]
"""

        pipeline = await yaml_compiler.compile(yaml_content)

        # The AUTO tag should have compile-time value resolved but runtime preserved
        assert len(captured) == 1
        auto_content = captured[0]

        # Compile-time template should be resolved
        assert "With timeout 30s" in auto_content
        # Runtime template should be preserved
        assert "{{fetch.result}}" in auto_content
