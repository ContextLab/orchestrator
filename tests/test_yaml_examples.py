"""
Tests for YAML example files in the examples directory.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from orchestrator.compiler.yaml_compiler import YAMLCompiler
from orchestrator.control_systems.model_based_control_system import ModelBasedControlSystem
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.compiler.yaml_compiler import YAMLCompilerError


class TestYAMLExamples:
    """Test all YAML examples to ensure they compile and execute correctly."""
    
    @pytest.fixture
    def setup(self):
        """Setup test environment with mock model."""
        compiler = YAMLCompiler()
        model_registry = ModelRegistry()
        
        # Create a mock model that returns predictable results
        model = Mock()
        model.name = "test-model"
        
        async def mock_generate(prompt, **kwargs):
            # Return different results based on prompt content
            prompt_lower = prompt.lower()
            
            if "research" in prompt_lower or "plan" in prompt_lower:
                return "Research plan: 1. Gather data 2. Analyze 3. Report"
            elif "gather" in prompt_lower or "collect" in prompt_lower:
                return "Data collected: Various sources analyzed"
            elif "analyze" in prompt_lower or "process" in prompt_lower:
                return "Analysis complete: Key insights identified"
            elif "report" in prompt_lower or "summary" in prompt_lower:
                return "Report: Comprehensive findings documented"
            elif "generate" in prompt_lower or "create" in prompt_lower:
                return "Generated content based on requirements"
            elif "validate" in prompt_lower or "check" in prompt_lower:
                return {"valid": True, "score": 0.95}
            elif "transform" in prompt_lower or "convert" in prompt_lower:
                return "Data transformed successfully"
            else:
                return "Task completed successfully"
        
        model.generate = mock_generate
        model.capabilities = {
            "tasks": ["generate", "analyze", "transform"],
            "context_window": 8192,
            "output_tokens": 4096
        }
        
        model_registry.register_model(model)
        control_system = ModelBasedControlSystem(model_registry)
        
        return compiler, control_system
    
    def get_example_files(self):
        """Get all YAML files from examples directory."""
        examples_dir = Path("/Users/jmanning/orchestrator/examples")
        if not examples_dir.exists():
            return []
        
        return list(examples_dir.glob("*.yaml"))
    
    @pytest.mark.asyncio
    async def test_research_assistant(self, setup):
        """Test the research_assistant.yaml example."""
        compiler, control_system = setup
        
        yaml_path = Path("/Users/jmanning/orchestrator/examples/research_assistant.yaml")
        if not yaml_path.exists():
            pytest.skip("research_assistant.yaml not found")
        
        with open(yaml_path, 'r') as f:
            yaml_content = f.read()
        
        inputs = {
            "topic": "AI Ethics",
            "depth": "comprehensive",
            "output_format": "markdown"
        }
        
        pipeline = await compiler.compile(yaml_content, inputs)
        results = await control_system.execute_pipeline(pipeline)
        
        # Verify key steps executed
        assert "create_research_plan" in results
        assert "gather_information" in results
        assert "analyze_findings" in results
        assert "generate_report" in results
    
    @pytest.mark.asyncio
    async def test_data_processing_workflow(self, setup):
        """Test the data_processing_workflow.yaml example."""
        compiler, control_system = setup
        
        yaml_path = Path("/Users/jmanning/orchestrator/examples/data_processing_workflow.yaml")
        if not yaml_path.exists():
            pytest.skip("data_processing_workflow.yaml not found")
        
        with open(yaml_path, 'r') as f:
            yaml_content = f.read()
        
        inputs = {
            "input_data": {"source": "test_data.csv", "format": "csv"},
            "processing_config": {
                "remove_duplicates": True,
                "normalize": True,
                "validation_threshold": 0.95
            },
            "output_format": "json"
        }
        
        pipeline = await compiler.compile(yaml_content, inputs)
        results = await control_system.execute_pipeline(pipeline)
        
        # Verify pipeline steps
        assert "load_data" in results
        assert "validate_data" in results
        assert "clean_data" in results
        assert "transform_data" in results
    
    @pytest.mark.asyncio
    async def test_content_creation_pipeline(self, setup):
        """Test the content_creation_pipeline.yaml example."""
        compiler, control_system = setup
        
        yaml_path = Path("/Users/jmanning/orchestrator/examples/content_creation_pipeline.yaml")
        if not yaml_path.exists():
            pytest.skip("content_creation_pipeline.yaml not found")
        
        with open(yaml_path, 'r') as f:
            yaml_content = f.read()
        
        inputs = {
            "topic": "Machine Learning",
            "content_type": "blog_post",
            "target_audience": "developers",
            "tone": "professional",
            "length": "medium"
        }
        
        pipeline = await compiler.compile(yaml_content, inputs)
        results = await control_system.execute_pipeline(pipeline)
        
        # Verify content creation steps
        assert "research_topic" in results
        assert "create_outline" in results
        assert "write_content" in results
        assert "review_and_edit" in results
    
    @pytest.mark.asyncio
    async def test_multi_agent_collaboration(self, setup):
        """Test the multi_agent_collaboration.yaml example."""
        compiler, control_system = setup
        
        yaml_path = Path("/Users/jmanning/orchestrator/examples/multi_agent_collaboration.yaml")
        if not yaml_path.exists():
            pytest.skip("multi_agent_collaboration.yaml not found")
        
        with open(yaml_path, 'r') as f:
            yaml_content = f.read()
        
        inputs = {
            "project_description": "Build a recommendation system",
            "requirements": ["scalable", "real-time", "personalized"],
            "constraints": {"budget": 50000, "timeline": "3 months"}
        }
        
        pipeline = await compiler.compile(yaml_content, inputs)
        results = await control_system.execute_pipeline(pipeline)
        
        # Verify collaboration steps
        assert "project_analysis" in results
        assert "technical_design" in results
        assert "implementation_plan" in results
    
    @pytest.mark.asyncio
    async def test_creative_writing_assistant(self, setup):
        """Test the creative_writing_assistant.yaml example."""
        compiler, control_system = setup
        
        yaml_path = Path("/Users/jmanning/orchestrator/examples/creative_writing_assistant.yaml")
        if not yaml_path.exists():
            pytest.skip("creative_writing_assistant.yaml not found")
        
        with open(yaml_path, 'r') as f:
            yaml_content = f.read()
        
        inputs = {
            "genre": "science fiction",
            "theme": "time travel",
            "length": "short story",
            "style": "descriptive"
        }
        
        pipeline = await compiler.compile(yaml_content, inputs)
        results = await control_system.execute_pipeline(pipeline)
        
        # Verify creative writing steps
        assert "develop_concept" in results
        assert "create_characters" in results
        assert "plot_outline" in results
        assert "write_story" in results
    
    @pytest.mark.asyncio
    async def test_all_examples_compile(self, setup):
        """Test that all example YAML files compile without errors."""
        compiler, control_system = setup
        
        example_files = self.get_example_files()
        if not example_files:
            pytest.skip("No example files found")
        
        failed_files = []
        
        for yaml_file in example_files:
            try:
                with open(yaml_file, 'r') as f:
                    yaml_content = f.read()
                
                # Use minimal inputs to test compilation
                inputs = self.get_default_inputs_for_file(yaml_file.name)
                
                pipeline = await compiler.compile(yaml_content, inputs)
                assert pipeline is not None
                assert len(pipeline.tasks) > 0
                
            except Exception as e:
                failed_files.append((yaml_file.name, str(e)))
        
        if failed_files:
            failure_msg = "\n".join([f"{name}: {error}" for name, error in failed_files])
            pytest.fail(f"Failed to compile the following files:\n{failure_msg}")
    
    def get_default_inputs_for_file(self, filename):
        """Get default inputs for a specific YAML file."""
        defaults = {
            "research_assistant.yaml": {
                "topic": "Test Topic",
                "depth": "basic",
                "output_format": "text"
            },
            "data_processing_workflow.yaml": {
                "input_data": {"source": "test.csv"},
                "processing_config": {"normalize": True},
                "output_format": "json"
            },
            "content_creation_pipeline.yaml": {
                "topic": "Test",
                "content_type": "article",
                "target_audience": "general",
                "tone": "neutral",
                "length": "short"
            },
            "multi_agent_collaboration.yaml": {
                "project_description": "Test project",
                "requirements": ["basic"],
                "constraints": {}
            },
            "code_analysis_suite.yaml": {
                "repository_path": "/test/repo",
                "analysis_config": {"checks": ["style"]},
                "output_format": "json"
            },
            "customer_support_automation.yaml": {
                "customer_query": "Help with product",
                "customer_id": "12345",
                "context": {}
            },
            "automated_testing_system.yaml": {
                "test_suite": "unit",
                "target": "module",
                "config": {}
            },
            "interactive_chat_bot.yaml": {
                "user_message": "Hello",
                "session_id": "123",
                "context": {}
            },
            "creative_writing_assistant.yaml": {
                "genre": "fiction",
                "theme": "adventure",
                "length": "short",
                "style": "simple"
            },
            "scalable_customer_service_agent.yaml": {
                "interaction_id": "INT-001",
                "customer_id": "CUST-001",
                "channel": "chat",
                "content": "Help needed",
                "metadata": {},
                "languages": ["en"],
                "sla_targets": {}
            },
            "document_intelligence.yaml": {
                "documents": [{"name": "test.pdf", "path": "/test.pdf"}],
                "operation": "extract",
                "config": {"format": "json"}
            },
            "financial_analysis_bot.yaml": {
                "query": "stock analysis",
                "symbols": ["TEST"],
                "timeframe": "1d",
                "analysis_type": "basic"
            }
        }
        
        return defaults.get(filename, {})
    
    @pytest.mark.asyncio
    async def test_example_outputs(self, setup):
        """Test that example pipelines produce expected output structure."""
        compiler, control_system = setup
        
        # Test a specific example with known structure
        yaml_path = Path("/Users/jmanning/orchestrator/examples/research_assistant.yaml")
        if not yaml_path.exists():
            pytest.skip("research_assistant.yaml not found")
        
        with open(yaml_path, 'r') as f:
            yaml_content = f.read()
        
        inputs = {
            "topic": "Quantum Computing",
            "depth": "detailed",
            "output_format": "markdown"
        }
        
        pipeline = await compiler.compile(yaml_content, inputs)
        results = await control_system.execute_pipeline(pipeline)
        
        # Get pipeline outputs
        outputs = pipeline.get_outputs(results)
        
        # Verify output structure
        assert "research_report" in outputs
        assert "key_findings" in outputs
        assert "research_plan" in outputs
        
        # Verify outputs have content
        assert outputs["research_report"] is not None
        assert len(str(outputs["research_report"])) > 0


class TestYAMLExampleValidation:
    """Validate YAML examples against best practices."""
    
    def test_examples_follow_naming_convention(self):
        """Test that all examples follow naming conventions."""
        examples_dir = Path("/Users/jmanning/orchestrator/examples")
        if not examples_dir.exists():
            pytest.skip("Examples directory not found")
        
        yaml_files = list(examples_dir.glob("*.yaml"))
        
        for yaml_file in yaml_files:
            # Check filename is lowercase with underscores
            assert yaml_file.stem.islower() or "_" in yaml_file.stem
            assert not yaml_file.stem.startswith("_")
            assert not yaml_file.stem.endswith("_")
    
    def test_examples_have_required_sections(self):
        """Test that all examples have required sections."""
        examples_dir = Path("/Users/jmanning/orchestrator/examples")
        if not examples_dir.exists():
            pytest.skip("Examples directory not found")
        
        yaml_files = list(examples_dir.glob("*.yaml"))
        compiler = YAMLCompiler()
        
        for yaml_file in yaml_files:
            with open(yaml_file, 'r') as f:
                yaml_content = f.read()
            
            try:
                # Parse YAML to check structure
                import yaml
                data = yaml.safe_load(yaml_content)
                
                # Check required sections
                assert "name" in data, f"{yaml_file.name} missing 'name'"
                assert "description" in data, f"{yaml_file.name} missing 'description'"
                assert "steps" in data, f"{yaml_file.name} missing 'steps'"
                assert len(data["steps"]) > 0, f"{yaml_file.name} has no steps"
                
                # Check each step has required fields
                for i, step in enumerate(data["steps"]):
                    assert "id" in step, f"{yaml_file.name} step {i} missing 'id'"
                    assert "action" in step, f"{yaml_file.name} step {i} missing 'action'"
                
            except Exception as e:
                pytest.fail(f"Failed to validate {yaml_file.name}: {str(e)}")
    
    def test_examples_have_descriptive_actions(self):
        """Test that all example steps have descriptive actions."""
        examples_dir = Path("/Users/jmanning/orchestrator/examples")
        if not examples_dir.exists():
            pytest.skip("Examples directory not found")
        
        yaml_files = list(examples_dir.glob("*.yaml"))
        
        for yaml_file in yaml_files:
            with open(yaml_file, 'r') as f:
                yaml_content = f.read()
            
            import yaml
            data = yaml.safe_load(yaml_content)
            
            for step in data.get("steps", []):
                action = step.get("action", "")
                
                # Check action is descriptive (not too short)
                assert len(action) > 10, f"{yaml_file.name} step {step.get('id')} has too short action"
                
                # Check action doesn't have placeholder text
                placeholder_words = ["todo", "placeholder", "fixme", "xxx"]
                action_lower = action.lower()
                for word in placeholder_words:
                    assert word not in action_lower, f"{yaml_file.name} has placeholder text in step {step.get('id')}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])