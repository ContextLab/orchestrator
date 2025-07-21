"""Tests for content_creation_pipeline.yaml example.

This test file follows the NO MOCKS policy. Tests use real orchestration
when API keys are available, otherwise they skip gracefully.
"""
import pytest
from .test_base import BaseExampleTest


class TestContentCreationPipelineYAML(BaseExampleTest):
    """Test the content creation pipeline YAML configuration."""
    
    @pytest.fixture
    def pipeline_name(self):
        return "content_creation_pipeline.yaml"
    
    @pytest.fixture
    def sample_inputs(self):
        return {
            "topic": "AI in Healthcare",
            "content_type": "blog_post",
            "target_audience": "healthcare professionals",
            "tone": "professional",
            "length": 1500,
            "keywords": ["AI", "healthcare", "medical technology", "patient care"],
            "include_images": True,
            "seo_optimization": True,
            "language": "en"
        }
    
    def test_pipeline_structure(self, pipeline_name):
        """Test that the pipeline has valid structure."""
        self.validate_pipeline_structure(pipeline_name)
        
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check content creation steps
        step_ids = [step['id'] for step in config['steps']]
        required_steps = [
            'research_topic',
            'create_outline',
            'write_content',
            'optimize_seo',
            'generate_images',
            'review_content',
            'format_output'
        ]
        
        for step in required_steps:
            assert step in step_ids, f"Missing required step: {step}"
    
    def test_content_type_configuration(self, pipeline_name):
        """Test content type specific configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check content type is used in writing
        write_step = next(s for s in config['steps'] if s['id'] == 'write_content')
        assert 'content_type' in str(write_step), \
            "Content type should be referenced in writing step"
    
    def test_seo_optimization_conditional(self, pipeline_name):
        """Test conditional SEO optimization."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check SEO optimization has condition
        seo_step = next(s for s in config['steps'] if s['id'] == 'optimize_seo')
        assert 'condition' in seo_step or 'seo_optimization' in str(seo_step), \
            "SEO optimization should be conditional on seo_optimization flag"
    
    def test_image_generation_conditional(self, pipeline_name):
        """Test conditional image generation."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check image generation has condition
        image_step = next(s for s in config['steps'] if s['id'] == 'generate_images')
        assert 'condition' in image_step or 'include_images' in str(image_step), \
            "Image generation should be conditional on include_images flag"
    
    def test_auto_tags_present(self, pipeline_name):
        """Test that AUTO tags are used for content decisions."""
        # Load raw YAML to check for AUTO tags
        from pathlib import Path
        example_dir = Path(__file__).parent.parent.parent / "examples"
        pipeline_path = example_dir / pipeline_name
        
        with open(pipeline_path, 'r') as f:
            content = f.read()
        
        # Check for AUTO tags
        assert '<AUTO>' in content, "Pipeline should use AUTO tags for content decisions"
        assert '</AUTO>' in content, "AUTO tags should be properly closed"
    
    @pytest.mark.asyncio
    async def test_basic_execution_structure(self, orchestrator, pipeline_name, sample_inputs):
        """Test basic pipeline execution structure without full execution."""
        # This test verifies the pipeline can be loaded and initialized
        # Full execution would require significant model usage
        
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Verify the pipeline can be parsed and validated
        assert config is not None
        assert 'steps' in config
        assert len(config['steps']) > 0
        
        # Verify all required inputs are defined
        if 'inputs' in config:
            required_inputs = config['inputs']
            for input_key in required_inputs:
                assert input_key in sample_inputs, f"Missing required input: {input_key}"
    
    def test_target_audience_usage(self, pipeline_name):
        """Test that target audience is properly used."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check audience is used in content creation
        has_audience_reference = any(
            'target_audience' in str(step) or 'audience' in str(step) 
            for step in config['steps']
        )
        assert has_audience_reference, \
            "Target audience should be referenced in content creation"
    
    def test_keyword_integration(self, pipeline_name):
        """Test keyword integration in content."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check keywords are used
        has_keyword_reference = any(
            'keywords' in str(step) or 'keyword' in str(step)
            for step in config['steps']
        )
        assert has_keyword_reference, "Keywords should be integrated in content creation"
    
    def test_review_process(self, pipeline_name):
        """Test content review process."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check review depends on content creation
        review_step = next(s for s in config['steps'] if s['id'] == 'review_content')
        if 'depends_on' in review_step:
            assert 'write_content' in review_step['depends_on'], \
                "Review should depend on content writing"
    
    def test_output_formatting(self, pipeline_name):
        """Test output formatting configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check format output is final step
        format_step = next(s for s in config['steps'] if s['id'] == 'format_output')
        if 'depends_on' in format_step:
            # Should depend on review or other final steps
            deps = format_step['depends_on']
            assert 'review_content' in deps or len(deps) > 0, \
                "Format output should depend on previous steps"
    
    def test_language_configuration(self, pipeline_name):
        """Test language configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check language is used
        has_language_reference = any(
            'language' in str(step) for step in config['steps']
        )
        assert has_language_reference, "Language should be configured for content"
    
    def test_length_constraints(self, pipeline_name):
        """Test content length constraints."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check length is used in writing
        write_step = next(s for s in config['steps'] if s['id'] == 'write_content')
        assert 'length' in str(write_step) or 'word_count' in str(write_step), \
            "Content length should be constrained"
    
    @pytest.mark.asyncio
    async def test_error_handling_structure(self, pipeline_name):
        """Test error handling for content creation."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check error handling configuration
        if 'error_handling' in config:
            error_config = config['error_handling']
            assert 'strategy' in error_config, "Error handling should define a strategy"
            
        # Critical steps should have error handling
        critical_steps = ['write_content', 'format_output']
        for step_id in critical_steps:
            step = next((s for s in config['steps'] if s['id'] == step_id), None)
            if step and 'error_handling' in step:
                assert 'on_error' in step['error_handling'], \
                    f"Step {step_id} should define error behavior"


# Note: Full integration tests that would generate complete content
# with images and SEO optimization are not included here as they
# would require significant model usage and external services.
# The tests above verify the pipeline structure and configuration
# without mocks.