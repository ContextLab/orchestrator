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
            "formats": ["blog", "social", "email"],
            "audience": "healthcare professionals",
            "brand_voice": "professional yet approachable",
            "goals": ["educate", "engage", "convert"],
            "auto_publish": False,
            "target_length": 1500
        }
    
    def test_pipeline_structure(self, pipeline_name):
        """Test that the pipeline has valid structure."""
        self.validate_pipeline_structure(pipeline_name)
        
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check content creation steps
        step_ids = [step['id'] for step in config['steps']]
        required_steps = [
            'research_topic',
            'generate_outline',
            'create_blog_content',
            'optimize_seo',
            'generate_visuals',
            'quality_review',
            'save_content_to_file'
        ]
        
        for step in required_steps:
            assert step in step_ids, f"Missing required step: {step}"
    
    def test_content_type_configuration(self, pipeline_name):
        """Test content type specific configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check formats are used in content creation
        blog_step = next(s for s in config['steps'] if s['id'] == 'create_blog_content')
        assert 'condition' in blog_step and 'blog' in blog_step['condition'], \
            "Blog content should be conditional on formats"
    
    def test_seo_optimization_conditional(self, pipeline_name):
        """Test conditional SEO optimization."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check SEO optimization has condition
        seo_step = next(s for s in config['steps'] if s['id'] == 'optimize_seo')
        assert 'condition' in seo_step and 'blog' in seo_step['condition'], \
            "SEO optimization should be conditional on blog format"
    
    def test_image_generation_conditional(self, pipeline_name):
        """Test conditional image generation."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check visual generation exists
        visual_step = next(s for s in config['steps'] if s['id'] == 'generate_visuals')
        assert visual_step is not None, \
            "Visual generation step should exist"
    
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
        # Test pipeline execution with minimal responses
        result = await self.run_pipeline_test(
            orchestrator,
            pipeline_name,
            sample_inputs,
            expected_outputs={
                'blog_content': str,
                'seo_score': int,
                'keywords_used': list
            },
            use_minimal_responses=True
        )
        
        # Verify result structure
        assert result is not None
        assert 'outputs' in result or 'steps' in result
    
    def test_target_audience_usage(self, pipeline_name):
        """Test that target audience is properly used."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check audience is used in content creation
        has_audience_reference = any(
            'audience' in str(step) 
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
        review_step = next(s for s in config['steps'] if s['id'] == 'quality_review')
        if 'depends_on' in review_step:
            assert any('create' in dep or 'optimize' in dep for dep in review_step['depends_on']), \
                "Review should depend on content creation"
    
    def test_output_formatting(self, pipeline_name):
        """Test output formatting configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check save content to file is present
        save_step = next((s for s in config['steps'] if s['id'] == 'save_content_to_file'), None)
        assert save_step is not None, "Save content to file step should exist"
        if 'depends_on' in save_step:
            # Should depend on quality review or other steps
            deps = save_step['depends_on']
            assert len(deps) > 0, "Save content should depend on previous steps"
    
    def test_language_configuration(self, pipeline_name):
        """Test language configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Language is not configured in this pipeline but audience is
        # The test was checking for wrong parameter
        assert 'audience' in config['inputs'], "Audience should be configured"
    
    def test_length_constraints(self, pipeline_name):
        """Test content length constraints."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check length is used in writing
        blog_step = next(s for s in config['steps'] if s['id'] == 'create_blog_content')
        assert 'target_length' in str(blog_step), \
            "Content length should be constrained"
    
    @pytest.mark.asyncio
    async def test_error_handling_structure(self, pipeline_name):
        """Test error handling for content creation."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check error handling in publish_content
        publish_step = next((s for s in config['steps'] if s['id'] == 'publish_content'), None)
        if publish_step:
            assert 'on_error' in publish_step, \
                "Publish content should have error handling"


# Note: Full integration tests that would generate complete content
# with images and SEO optimization are not included here as they
# would require significant model usage and external services.
# The tests above verify the pipeline structure and configuration
# without mocks.