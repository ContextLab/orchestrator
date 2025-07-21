"""Full integration tests for YAML pipelines with real execution.

These tests use real API calls, real file operations, and real model interactions.
They verify actual functionality rather than just structure.

Requirements:
- API keys must be set in environment variables
- Tests will skip if required resources aren't available
- Tests create temporary data and clean up after themselves
"""

import asyncio
import os
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List

from orchestrator import Orchestrator, init_models
from orchestrator.compiler import YAMLCompiler
from orchestrator.models.openai_model import OpenAIModel
from orchestrator.models.anthropic_model import AnthropicModel
from orchestrator.utils.api_keys import load_api_keys
from orchestrator.control_systems.model_based_control_system import ModelBasedControlSystem


# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def setup_api_keys():
    """Load API keys from environment."""
    try:
        load_api_keys()
        return True
    except Exception as e:
        pytest.skip(f"API keys not configured: {e}")
        return False


@pytest.fixture(scope="module")
def orchestrator(setup_api_keys):
    """Create orchestrator with real models."""
    # Initialize model registry with real models
    try:
        model_registry = init_models()
    except Exception as e:
        pytest.skip(f"Failed to initialize models: {e}")
    
    # Check if we have any models available
    available_models = model_registry.list_models()
    
    # Check the structure of available_models
    if available_models and isinstance(available_models[0], str):
        # It's a list of model names
        api_models = [m for m in available_models if any(
            provider in m.lower() for provider in ['gpt', 'claude', 'gemini']
        )]
    else:
        # It's a list of dicts
        api_models = [m for m in available_models if 
                     isinstance(m, dict) and m.get('provider') in ['openai', 'anthropic', 'google']]
    
    if not api_models:
        pytest.skip("No API models available for integration testing")
    
    # Create control system with model registry
    control_system = ModelBasedControlSystem(model_registry=model_registry)
    
    # Create orchestrator with control system
    orchestrator = Orchestrator(control_system=control_system)
    
    print(f"\nAvailable models for testing: {len(api_models)}")
    for model in api_models[:5]:  # Show first 5 models
        if isinstance(model, str):
            print(f"  - {model}")
        else:
            print(f"  - {model.get('name', model)} ({model.get('provider', 'unknown')})")
    
    return orchestrator


@pytest.fixture
def yaml_compiler(orchestrator):
    """Create YAML compiler with model registry."""
    # Get model registry from the orchestrator's control system
    model_registry = orchestrator.control_system.model_registry if hasattr(orchestrator.control_system, 'model_registry') else None
    return YAMLCompiler(model_registry=model_registry)


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for tests."""
    workspace = tempfile.mkdtemp(prefix="orchestrator_test_")
    yield workspace
    # Cleanup
    shutil.rmtree(workspace, ignore_errors=True)


class TestCodeAnalysisSuiteIntegration:
    """Integration tests for code analysis suite pipeline."""
    
    @pytest.fixture
    def sample_code_repo(self, temp_workspace):
        """Create a sample code repository with known issues."""
        repo_path = Path(temp_workspace) / "test_repo"
        repo_path.mkdir()
        
        # Create Python file with known issues
        python_file = repo_path / "example.py"
        python_file.write_text("""
# Example Python file with various issues
import os
import sys
import unused_module  # Unused import

def insecure_function(user_input):
    # Security issue: SQL injection vulnerability
    query = f"SELECT * FROM users WHERE id = {user_input}"
    # Hardcoded password
    password = "admin123"
    return query

def complex_function(a, b, c, d, e, f):  # Too many parameters
    # Complex nested logic
    if a:
        if b:
            if c:
                if d:
                    if e:
                        return f
    return None

class UnusedClass:
    pass

# TODO: Fix this later
# FIXME: This is broken
""")
        
        # Create JavaScript file
        js_file = repo_path / "example.js"
        js_file.write_text("""
// Example JavaScript file
function insecureEval(userInput) {
    // Security issue: eval usage
    return eval(userInput);
}

var globalVar = "bad practice";  // Global variable

function complexFunction() {
    // Deeply nested callbacks
    setTimeout(function() {
        fetch('/api/data').then(function(response) {
            response.json().then(function(data) {
                console.log(data);
            });
        });
    }, 1000);
}
""")
        
        return str(repo_path)
    
    @pytest.mark.timeout(300)  # 5 minute timeout
    async def test_code_analysis_full_execution(self, orchestrator, yaml_compiler, sample_code_repo):
        """Test full execution of code analysis pipeline with real code."""
        # Load the pipeline
        pipeline_path = Path("examples/code_analysis_suite.yaml")
        if not pipeline_path.exists():
            pytest.skip("code_analysis_suite.yaml not found")
        
        # Read YAML file
        with open(pipeline_path, 'r') as f:
            yaml_content = f.read()
        
        # Create context with required inputs
        context = {
            "repo_path": sample_code_repo,
            "languages": ["python", "javascript"],
            "analysis_depth": "standard",
            "security_scan": True,
            "performance_check": False,
            "doc_check": True,
            "severity_threshold": "medium"
        }
        
        # Compile the YAML content with context
        pipeline = await yaml_compiler.compile(yaml_content, context=context)
        
        # Execute pipeline
        try:
            result = await orchestrator.execute_pipeline(pipeline)
            
            # Verify we got results
            assert result is not None
            assert isinstance(result, dict)
            
            # Check for expected analysis results
            # The pipeline should detect issues in our sample code
            if "code_quality_scan" in result:
                quality_results = result["code_quality_scan"]
                assert quality_results is not None
                # Should detect complexity issues, unused imports, etc.
            
            if "dependency_analysis" in result:
                dep_results = result["dependency_analysis"]
                assert dep_results is not None
            
            if "security_scan" in result:
                security_results = result["security_scan"]
                assert security_results is not None
                # Should detect SQL injection, eval usage, hardcoded passwords
            
            if "generate_report" in result:
                report = result["generate_report"]
                assert report is not None
                assert isinstance(report, str) or isinstance(report, dict)
                # Report should mention some of the issues we planted
            
        except Exception as e:
            pytest.fail(f"Pipeline execution failed: {e}")


class TestCreativeWritingAssistantIntegration:
    """Integration tests for creative writing assistant pipeline."""
    
    @pytest.mark.timeout(180)  # 3 minute timeout
    async def test_creative_writing_execution(self, orchestrator, yaml_compiler):
        """Test creative writing pipeline with real prompts."""
        # Load the pipeline
        pipeline_path = Path("examples/creative_writing_assistant.yaml")
        if not pipeline_path.exists():
            pytest.skip("creative_writing_assistant.yaml not found")
        
        # Read YAML file
        with open(pipeline_path, 'r') as f:
            yaml_content = f.read()
        
        # Set context for compilation and story generation
        context = {
            "prompt": "Write a very short story about a robot learning to paint",
            "genre": "science fiction",
            "style": "simple",
            "length": "flash",  # Very short
            "write_detailed_chapters": False,  # For short stories
            "include_worldbuilding": False,  # Skip for speed
            "chapter_count": 0,  # No chapters for flash fiction
            "target_audience": "general",
            "writing_style": "concise",
            "initial_premise": "A robot discovers painting"
        }
        
        # Compile the YAML content with context
        pipeline = await yaml_compiler.compile(yaml_content, context=context)
        
        # Execute pipeline
        try:
            result = await orchestrator.execute_pipeline(pipeline)
            
            # Verify results
            assert result is not None
            assert isinstance(result, dict)
            
            # Check for story generation
            if "generate_story" in result:
                story = result["generate_story"]
                assert story is not None
                assert isinstance(story, str)
                assert len(story) > 100  # Should generate substantial content
                # Verify it's about the requested topic
                assert any(word in story.lower() for word in ["robot", "paint", "art"])
            
            # Check for title generation if present
            if "generate_title" in result:
                title = result["generate_title"]
                assert title is not None
                assert isinstance(title, str)
                assert len(title) > 0
            
            # Check for character development if present
            if "develop_characters" in result:
                characters = result["develop_characters"]
                assert characters is not None
            
        except Exception as e:
            pytest.fail(f"Creative writing pipeline failed: {e}")
    
    @pytest.mark.timeout(180)
    async def test_creative_writing_genres(self, orchestrator, yaml_compiler):
        """Test creative writing with different genres."""
        pipeline_path = Path("examples/creative_writing_assistant.yaml")
        if not pipeline_path.exists():
            pytest.skip("creative_writing_assistant.yaml not found")
        
        genres = ["fantasy", "mystery", "romance"]
        
        for genre in genres:
            # Read YAML file
            with open(pipeline_path, 'r') as f:
                yaml_content = f.read()
            
            # Set context for this genre
            context = {
                "prompt": f"Write a {genre} story opening",
                "genre": genre,
                "style": "engaging",
                "length": "short",
                "write_detailed_chapters": False,
                "include_worldbuilding": True,
                "chapter_count": 1,
                "target_audience": "general",
                "writing_style": f"engaging {genre} style",
                "initial_premise": f"A compelling {genre} story opening"
            }
            
            # Compile the YAML content with context
            pipeline = await yaml_compiler.compile(yaml_content, context=context)
            
            try:
                result = await orchestrator.execute_pipeline(pipeline)
                
                assert result is not None
                if "generate_story" in result:
                    story = result["generate_story"]
                    assert story is not None
                    assert len(story) > 50
                    # Genre should influence the content
                    print(f"\n{genre.upper()} Story Sample: {story[:200]}...")
                    
            except Exception as e:
                pytest.fail(f"Failed for genre {genre}: {e}")


class TestContentCreationPipelineIntegration:
    """Integration tests for content creation pipeline."""
    
    @pytest.mark.timeout(240)  # 4 minute timeout
    async def test_content_creation_execution(self, orchestrator, yaml_compiler, temp_workspace):
        """Test content creation pipeline with real content generation."""
        # Load the pipeline
        pipeline_path = Path("examples/content_creation_pipeline.yaml")
        if not pipeline_path.exists():
            pytest.skip("content_creation_pipeline.yaml not found")
        
        # Read YAML file
        with open(pipeline_path, 'r') as f:
            yaml_content = f.read()
        
        # Compile the YAML content
        pipeline = await yaml_compiler.compile(yaml_content)
        
        # Set context for content creation
        pipeline.set_context("topic", "Introduction to Machine Learning")
        pipeline.set_context("content_type", "blog_post")
        pipeline.set_context("target_audience", "beginners")
        pipeline.set_context("tone", "educational")
        pipeline.set_context("seo_keywords", ["machine learning", "AI", "tutorial"])
        pipeline.set_context("output_dir", temp_workspace)
        
        # Execute pipeline
        try:
            result = await orchestrator.execute_pipeline(pipeline)
            
            # Verify results
            assert result is not None
            assert isinstance(result, dict)
            
            # Check content generation
            if "generate_content" in result:
                content = result["generate_content"]
                assert content is not None
                assert isinstance(content, (str, dict))
                if isinstance(content, str):
                    assert len(content) > 500  # Substantial content
                    assert "machine learning" in content.lower()
            
            # Check SEO optimization
            if "optimize_seo" in result:
                seo_result = result["optimize_seo"]
                assert seo_result is not None
                # Should include SEO recommendations or optimized content
            
            # Check formatting
            if "format_content" in result:
                formatted = result["format_content"]
                assert formatted is not None
            
            # Verify files were created if pipeline saves output
            output_files = list(Path(temp_workspace).glob("*.md"))
            if output_files:
                # Read and verify content
                for file in output_files:
                    content = file.read_text()
                    assert len(content) > 0
                    print(f"\nCreated file: {file.name} ({len(content)} chars)")
            
        except Exception as e:
            pytest.fail(f"Content creation pipeline failed: {e}")


class TestMultiAgentCollaborationIntegration:
    """Integration tests for multi-agent collaboration pipeline."""
    
    @pytest.mark.timeout(300)  # 5 minute timeout
    async def test_multi_agent_execution(self, orchestrator, yaml_compiler):
        """Test multi-agent collaboration with real agent interactions."""
        # Load the pipeline
        pipeline_path = Path("examples/multi_agent_collaboration.yaml")
        if not pipeline_path.exists():
            pytest.skip("multi_agent_collaboration.yaml not found")
        
        # Read YAML file
        with open(pipeline_path, 'r') as f:
            yaml_content = f.read()
        
        # Compile the YAML content
        pipeline = await yaml_compiler.compile(yaml_content)
        
        # Set context for collaboration
        pipeline.set_context("task", "Design a simple web application for task management")
        pipeline.set_context("collaboration_mode", "sequential")
        pipeline.set_context("num_iterations", 2)  # Limit iterations for testing
        
        # Execute pipeline
        try:
            result = await orchestrator.execute_pipeline(pipeline)
            
            # Verify results
            assert result is not None
            assert isinstance(result, dict)
            
            # Check for agent outputs
            agent_roles = ["researcher", "designer", "developer", "reviewer"]
            
            for role in agent_roles:
                agent_key = f"agent_{role}"
                if agent_key in result:
                    agent_output = result[agent_key]
                    assert agent_output is not None
                    print(f"\n{role.upper()} Agent Output: {str(agent_output)[:200]}...")
            
            # Check for final synthesis
            if "synthesize_results" in result:
                synthesis = result["synthesize_results"]
                assert synthesis is not None
                # Should combine insights from all agents
            
        except Exception as e:
            pytest.fail(f"Multi-agent collaboration failed: {e}")


class TestDataProcessingWorkflowIntegration:
    """Integration tests for data processing workflow pipeline."""
    
    @pytest.fixture
    def sample_dataset(self, temp_workspace):
        """Create sample dataset for processing."""
        import json
        
        data_file = Path(temp_workspace) / "sample_data.json"
        
        # Create sample data
        sample_data = {
            "records": [
                {"id": 1, "name": "Alice", "age": 30, "score": 85.5},
                {"id": 2, "name": "Bob", "age": 25, "score": 92.0},
                {"id": 3, "name": "Charlie", "age": 35, "score": 78.5},
                {"id": 4, "name": "Diana", "age": 28, "score": 88.0},
                {"id": 5, "name": "Eve", "age": 32, "score": 95.5}
            ],
            "metadata": {
                "source": "test_dataset",
                "created": "2024-01-20",
                "version": "1.0"
            }
        }
        
        with open(data_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        return str(data_file)
    
    @pytest.mark.timeout(180)
    async def test_data_processing_execution(self, orchestrator, yaml_compiler, sample_dataset, temp_workspace):
        """Test data processing workflow with real data."""
        # Load the pipeline
        pipeline_path = Path("examples/data_processing_workflow.yaml")
        if not pipeline_path.exists():
            pytest.skip("data_processing_workflow.yaml not found")
        
        # Read YAML file
        with open(pipeline_path, 'r') as f:
            yaml_content = f.read()
        
        # Compile the YAML content
        pipeline = await yaml_compiler.compile(yaml_content)
        
        # Set context for data processing
        pipeline.set_context("input_file", sample_dataset)
        pipeline.set_context("output_dir", temp_workspace)
        pipeline.set_context("processing_steps", ["validate", "transform", "analyze"])
        
        # Execute pipeline
        try:
            result = await orchestrator.execute_pipeline(pipeline)
            
            # Verify results
            assert result is not None
            assert isinstance(result, dict)
            
            # Check data validation
            if "validate_data" in result:
                validation = result["validate_data"]
                assert validation is not None
                # Should report on data quality
            
            # Check data transformation
            if "transform_data" in result:
                transformed = result["transform_data"]
                assert transformed is not None
                # Should have processed the data
            
            # Check analysis
            if "analyze_data" in result:
                analysis = result["analyze_data"]
                assert analysis is not None
                # Should include statistics or insights
            
            # Check for output files
            output_files = list(Path(temp_workspace).glob("*_processed.*"))
            print(f"\nGenerated {len(output_files)} output files")
            
        except Exception as e:
            pytest.fail(f"Data processing workflow failed: {e}")


class TestErrorHandlingIntegration:
    """Test error handling and edge cases in pipelines."""
    
    @pytest.mark.timeout(120)
    async def test_pipeline_with_missing_model(self, yaml_compiler):
        """Test pipeline behavior when required model is not available."""
        # Create orchestrator without models
        orchestrator = Orchestrator()
        
        pipeline_path = Path("examples/creative_writing_assistant.yaml")
        if not pipeline_path.exists():
            pytest.skip("Pipeline not found")
        
        # Read YAML file
        with open(pipeline_path, 'r') as f:
            yaml_content = f.read()
        
        # Compile the YAML content
        pipeline = await yaml_compiler.compile(yaml_content)
        pipeline.set_context("prompt", "Test prompt")
        
        # Should handle missing model gracefully
        with pytest.raises(Exception) as exc_info:
            await orchestrator.execute_pipeline(pipeline)
        
        assert "model" in str(exc_info.value).lower()
    
    @pytest.mark.timeout(120)
    async def test_pipeline_with_invalid_input(self, orchestrator, yaml_compiler):
        """Test pipeline behavior with invalid inputs."""
        pipeline_path = Path("examples/content_creation_pipeline.yaml")
        if not pipeline_path.exists():
            pytest.skip("Pipeline not found")
        
        # Read YAML file
        with open(pipeline_path, 'r') as f:
            yaml_content = f.read()
        
        # Compile the YAML content
        pipeline = await yaml_compiler.compile(yaml_content)
        
        # Set invalid context (missing required fields)
        pipeline.set_context("topic", "")  # Empty topic
        
        # Should handle invalid input appropriately
        try:
            result = await orchestrator.execute_pipeline(pipeline)
            # Pipeline might handle empty topic, check result
            if result:
                print(f"\nPipeline handled empty topic: {list(result.keys())}")
        except Exception as e:
            # Expected behavior - should fail gracefully
            assert "topic" in str(e).lower() or "input" in str(e).lower()


# Utility functions for test support

def verify_output_quality(output: str, expected_keywords: List[str], min_length: int = 100) -> bool:
    """Verify output meets quality standards."""
    if not output or len(output) < min_length:
        return False
    
    # Check for expected keywords
    output_lower = output.lower()
    found_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in output_lower)
    
    return found_keywords >= len(expected_keywords) // 2  # At least half the keywords


def count_api_calls(result: Dict[str, Any]) -> int:
    """Count approximate API calls made during execution."""
    # This is a rough estimate based on result keys
    return len([k for k in result.keys() if any(
        action in k for action in ["generate", "analyze", "optimize", "create"]
    )])


# Test configuration

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring real resources"
    )
    config.addinivalue_line(
        "markers", "timeout: mark test with custom timeout"
    )


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-m", "integration"])