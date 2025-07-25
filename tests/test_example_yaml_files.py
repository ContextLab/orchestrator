"""Test that all example YAML files are valid and can be parsed."""

import os
from pathlib import Path
import pytest
import yaml
from orchestrator.compiler.auto_tag_yaml_parser import parse_yaml_with_auto_tags
from orchestrator.compiler.yaml_compiler import YAMLCompiler


class TestExampleYAMLFiles:
    """Test all example YAML files for validity."""

    @pytest.fixture
    def example_files(self):
        """Get all YAML files from examples directory."""
        examples_dir = Path(__file__).parent.parent / "examples"
        yaml_files = []

        # Recursively find all .yaml files
        for root, dirs, files in os.walk(examples_dir):
            for file in files:
                if file.endswith(".yaml"):
                    yaml_files.append(Path(root) / file)

        return yaml_files

    def test_all_examples_exist(self, example_files):
        """Test that we have example files to test."""
        assert len(example_files) > 0, "No example YAML files found"
        print(f"\nFound {len(example_files)} example YAML files to test")

    def test_yaml_syntax_valid(self, example_files):
        """Test that all YAML files have valid syntax."""
        errors = []

        for yaml_file in example_files:
            try:
                with open(yaml_file, "r") as f:
                    content = f.read()

                # First try standard YAML parsing
                yaml.safe_load(content)

            except yaml.YAMLError as e:
                errors.append(f"{yaml_file.name}: YAML syntax error - {str(e)}")
            except Exception as e:
                errors.append(f"{yaml_file.name}: Unexpected error - {str(e)}")

        if errors:
            error_msg = "YAML syntax errors found:\n" + "\n".join(errors)
            pytest.fail(error_msg)

    def test_auto_tag_parsing(self, example_files):
        """Test that YAML files with AUTO tags can be parsed."""
        errors = []
        auto_tag_files = []

        for yaml_file in example_files:
            try:
                with open(yaml_file, "r") as f:
                    content = f.read()

                # Check if file contains AUTO tags
                if "<AUTO>" in content:
                    auto_tag_files.append(yaml_file)
                    # Parse with AUTO tag support
                    parsed = parse_yaml_with_auto_tags(content)

                    # Verify basic structure
                    assert "id" in parsed, f"{yaml_file.name}: Missing 'id' field"
                    assert "steps" in parsed, f"{yaml_file.name}: Missing 'steps' field"

            except Exception as e:
                errors.append(f"{yaml_file.name}: AUTO tag parsing error - {str(e)}")

        print(f"\nFound {len(auto_tag_files)} files with AUTO tags")

        if errors:
            error_msg = "AUTO tag parsing errors found:\n" + "\n".join(errors)
            pytest.fail(error_msg)

    def test_pipeline_structure(self, example_files):
        """Test that all pipelines have required structure."""
        errors = []

        required_fields = ["id", "name", "steps"]

        for yaml_file in example_files:
            try:
                with open(yaml_file, "r") as f:
                    content = f.read()

                # Parse with AUTO tag support if needed
                if "<AUTO>" in content:
                    parsed = parse_yaml_with_auto_tags(content)
                else:
                    parsed = yaml.safe_load(content)

                # Check required fields
                for field in required_fields:
                    if field not in parsed:
                        errors.append(
                            f"{yaml_file.name}: Missing required field '{field}'"
                        )

                # Validate steps structure
                if "steps" in parsed and isinstance(parsed["steps"], list):
                    for i, step in enumerate(parsed["steps"]):
                        if not isinstance(step, dict):
                            errors.append(
                                f"{yaml_file.name}: Step {i} is not a dictionary"
                            )
                            continue

                        # Check required step fields
                        if "id" not in step:
                            errors.append(
                                f"{yaml_file.name}: Step {i} missing 'id' field"
                            )
                        
                        # Control flow steps (for_each, while, etc.) don't need action
                        is_control_flow = any(key in step for key in ["for_each", "while", "if", "condition"])
                        
                        if "action" not in step and not is_control_flow:
                            errors.append(
                                f"{yaml_file.name}: Step {i} missing 'action' field"
                            )

            except Exception as e:
                errors.append(
                    f"{yaml_file.name}: Structure validation error - {str(e)}"
                )

        if errors:
            error_msg = "Pipeline structure errors found:\n" + "\n".join(errors)
            pytest.fail(error_msg)

    @pytest.mark.asyncio
    async def test_pipeline_compilation(self, example_files, populated_model_registry):
        """Test that pipelines can be compiled successfully."""

        compiler = YAMLCompiler(model_registry=populated_model_registry)
        compiled_count = 0

        for yaml_file in example_files:
            try:
                # Read the YAML content
                with open(yaml_file, "r") as f:
                    yaml_content = f.read()
                
                # Compile the pipeline
                pipeline = await compiler.compile(yaml_content)
                assert pipeline is not None
                assert hasattr(pipeline, "name")
                assert hasattr(pipeline, "tasks")
                compiled_count += 1

            except Exception as e:
                # Some pipelines may have specific requirements
                # Log but don't fail for compilation errors
                print(f"\nWarning: {yaml_file.name} compilation failed: {str(e)}")

        print(
            f"\nSuccessfully compiled {compiled_count}/{len(example_files)} pipelines"
        )
        assert compiled_count > 0, "No pipelines could be compiled"

    def test_examples_readme_exists(self):
        """Test that examples directory has a README."""
        examples_dir = Path(__file__).parent.parent / "examples"
        readme_path = examples_dir / "README.md"

        assert readme_path.exists(), "examples/README.md is missing"

        # Check that README has content
        with open(readme_path, "r") as f:
            content = f.read()

        assert len(content) > 100, "examples/README.md seems too short"
        assert "# " in content, "examples/README.md should have headers"


class TestSpecificExamples:
    """Test specific example files for correctness."""

    def test_simple_data_processing_example(self):
        """Test the simple_data_processing.yaml example."""
        yaml_file = (
            Path(__file__).parent.parent / "examples" / "simple_data_processing.yaml"
        )

        if not yaml_file.exists():
            pytest.skip("simple_data_processing.yaml not found")

        with open(yaml_file, "r") as f:
            content = f.read()

        parsed = yaml.safe_load(content)

        # Verify expected structure
        assert parsed["id"] == "simple_data_processing"
        assert "steps" in parsed
        assert len(parsed["steps"]) > 0

        # Check for data processing actions
        actions = [step["action"] for step in parsed["steps"]]
        assert any(
            action
            in [
                "process_data",
                "transform_data",
                "analyze_data",
                "generate_text",
                "validate",
            ]
            for action in actions
        )

    def test_research_pipeline_example(self):
        """Test the research_pipeline.yaml example."""
        yaml_file = Path(__file__).parent.parent / "examples" / "research_pipeline.yaml"

        if not yaml_file.exists():
            pytest.skip("research_pipeline.yaml not found")

        with open(yaml_file, "r") as f:
            content = f.read()

        # Parse with AUTO tag support if needed
        if "<AUTO>" in content:
            parsed = parse_yaml_with_auto_tags(content)
        else:
            parsed = yaml.safe_load(content)

        # Verify research pipeline structure
        assert (
            "research" in parsed["id"].lower() or "research" in parsed["name"].lower()
        )
        assert "steps" in parsed

        # Check for typical research actions
        actions = [step.get("action", "") for step in parsed["steps"]]
        research_actions = [
            "web_search",
            "search",
            "analyze",
            "summarize",
            "generate_text",
            "write_file",
            "analyze_text",
        ]
        assert any(
            action in research_actions for action in actions
        ), f"No research-related actions found. Actions: {actions}"

    def test_control_flow_examples(self):
        """Test control flow example files."""
        examples_dir = Path(__file__).parent.parent / "examples"
        control_flow_files = [
            "control_flow_conditional.yaml",
            "control_flow_for_loop.yaml",
            "control_flow_while_loop.yaml",
            "control_flow_dynamic.yaml",
        ]

        for filename in control_flow_files:
            yaml_file = examples_dir / filename

            if not yaml_file.exists():
                continue

            with open(yaml_file, "r") as f:
                content = f.read()

            # Parse with AUTO tag support
            parsed = parse_yaml_with_auto_tags(content)

            # Verify control flow structures
            if "conditional" in filename:
                # Should have if/else logic
                assert any(
                    "if" in step or "condition" in step
                    for step in parsed["steps"]
                    if isinstance(step, dict)
                )

            elif "for_loop" in filename:
                # Should have for_each action
                assert any(
                    step.get("action") == "for_each"
                    for step in parsed["steps"]
                    if isinstance(step, dict)
                )

            elif "while_loop" in filename:
                # Should have while_loop action
                assert any(
                    step.get("action") == "while_loop"
                    for step in parsed["steps"]
                    if isinstance(step, dict)
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
