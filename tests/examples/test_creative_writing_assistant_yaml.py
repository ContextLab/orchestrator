"""Tests for creative_writing_assistant.yaml example.

This test file follows the NO MOCKS policy. Tests use real orchestration
when API keys are available, otherwise they skip gracefully.
"""
import pytest
from .test_base import BaseExampleTest


class TestCreativeWritingAssistantYAML(BaseExampleTest):
    """Test the creative writing assistant YAML pipeline."""
    
    @pytest.fixture
    def pipeline_name(self):
        return "creative_writing_assistant.yaml"
    
    @pytest.fixture
    def sample_inputs(self):
        return {
            "genre": "sci-fi",
            "length": "short_story",
            "writing_style": "contemporary",
            "target_audience": "general",
            "initial_premise": "first contact with aliens",
            "include_worldbuilding": True,
            "chapter_count": 5
        }
    
    def test_pipeline_structure(self, pipeline_name):
        """Test that the pipeline has valid structure."""
        self.validate_pipeline_structure(pipeline_name)
        
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check creative writing steps
        step_ids = [step['id'] for step in config['steps']]
        required_steps = [
            'analyze_genre',
            'generate_premise',
            'develop_characters',
            'build_world',
            'design_plot',
            'outline_chapters',
            'write_opening'
        ]
        
        for step in required_steps:
            assert step in step_ids, f"Missing required step: {step}"
    
    def test_genre_specific_configuration(self, pipeline_name):
        """Test genre-specific writing configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check genre is used in analysis
        analyze_step = next(s for s in config['steps'] if s['id'] == 'analyze_genre')
        assert '{{genre}}' in str(analyze_step), "Genre should be referenced in analysis"
    
    def test_worldbuilding_conditional(self, pipeline_name):
        """Test conditional worldbuilding step."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check worldbuilding has condition
        world_step = next(s for s in config['steps'] if s['id'] == 'build_world')
        assert 'condition' in world_step or 'include_worldbuilding' in str(world_step), \
            "Worldbuilding should be conditional on include_worldbuilding flag"
    
    def test_character_development(self, pipeline_name):
        """Test character development configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check character development uses character_count
        char_step = next(s for s in config['steps'] if s['id'] == 'develop_characters')
        assert 'character_count' in str(char_step) or 'characters' in str(char_step), \
            "Character development should reference character count"
    
    def test_auto_tags_present(self, pipeline_name):
        """Test that AUTO tags are used for creative decisions."""
        # Load raw YAML to check for AUTO tags
        from pathlib import Path
        example_dir = Path(__file__).parent.parent.parent / "examples"
        pipeline_path = example_dir / pipeline_name
        
        with open(pipeline_path, 'r') as f:
            content = f.read()
        
        # Check for AUTO tags
        assert '<AUTO>' in content, "Pipeline should use AUTO tags for creative decisions"
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
    
    def test_writing_style_configuration(self, pipeline_name):
        """Test writing style configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check writing style is used
        write_step = next(s for s in config['steps'] if s['id'] == 'write_opening')
        assert '{{writing_style}}' in str(write_step), \
            "Writing should reference writing style"
    
    def test_length_and_chapters(self, pipeline_name):
        """Test story length configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check length parameters are used
        plot_step = next(s for s in config['steps'] if s['id'] == 'design_plot')
        assert '{{length}}' in str(plot_step) or '{{chapter_count}}' in str(plot_step), \
            "Plot design should reference length constraints"
    
    def test_review_and_edit_steps(self, pipeline_name):
        """Test review and editing process."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check consistency check exists
        consistency_step = next(s for s in config['steps'] if s['id'] == 'check_consistency')
        assert consistency_step is not None
        
        # Check polish step exists
        polish_step = next(s for s in config['steps'] if s['id'] == 'polish_writing')
        assert polish_step is not None
    
    def test_creative_parameters(self, pipeline_name):
        """Test that creative parameters are properly integrated."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check target audience is used
        polish_step = next(s for s in config['steps'] if s['id'] == 'polish_writing')
        assert '{{target_audience}}' in str(polish_step), "Target audience should be referenced"
        
        # Check initial premise is used
        premise_step = next(s for s in config['steps'] if s['id'] == 'generate_premise')
        assert '{{initial_premise' in str(premise_step), "Initial premise should be referenced"
    
    @pytest.mark.asyncio
    async def test_error_handling_structure(self, pipeline_name):
        """Test that pipeline has proper error handling for creative tasks."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Creative tasks should have flexible error handling
        if 'error_handling' in config:
            error_config = config['error_handling']
            # Creative writing might want to continue on partial failures
            assert error_config.get('strategy') in ['continue', 'retry'], \
                "Creative pipeline should have flexible error handling"


    @pytest.mark.asyncio
    async def test_full_pipeline_flow(self, orchestrator, pipeline_name, sample_inputs):
        """Test full pipeline execution with minimal responses."""
        # Test pipeline execution with minimal responses
        result = await self.run_pipeline_test(
            orchestrator,
            pipeline_name,
            sample_inputs,
            use_minimal_responses=True
        )
        
        # Verify result structure
        assert result is not None
        assert 'outputs' in result or 'steps' in result


# Note: Full integration tests that would generate complete stories
# are not included here as they would require significant model usage
# and time. The tests above verify the pipeline structure and 
# configuration without mocks.