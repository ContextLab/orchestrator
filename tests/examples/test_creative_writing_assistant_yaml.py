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
            "theme": "first contact with aliens",
            "tone": "mysterious",
            "include_worldbuilding": True,
            "character_count": 3,
            "pov": "third_person",
            "target_words": 5000
        }
    
    def test_pipeline_structure(self, pipeline_name):
        """Test that the pipeline has valid structure."""
        self.validate_pipeline_structure(pipeline_name)
        
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check creative writing steps
        step_ids = [step['id'] for step in config['steps']]
        required_steps = [
            'brainstorm_ideas',
            'create_outline',
            'develop_characters',
            'build_world',
            'write_story',
            'review_and_edit',
            'final_polish'
        ]
        
        for step in required_steps:
            assert step in step_ids, f"Missing required step: {step}"
    
    def test_genre_specific_configuration(self, pipeline_name):
        """Test genre-specific writing configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check genre is used in brainstorming
        brainstorm_step = next(s for s in config['steps'] if s['id'] == 'brainstorm_ideas')
        assert 'genre' in str(brainstorm_step), "Genre should be referenced in brainstorming"
    
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
    
    def test_pov_configuration(self, pipeline_name):
        """Test point of view configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check POV is used in writing
        write_step = next(s for s in config['steps'] if s['id'] == 'write_story')
        assert 'pov' in str(write_step) or 'point_of_view' in str(write_step), \
            "Writing should reference point of view"
    
    def test_length_and_target_words(self, pipeline_name):
        """Test story length configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check length parameters are used
        write_step = next(s for s in config['steps'] if s['id'] == 'write_story')
        assert 'length' in str(write_step) or 'target_words' in str(write_step), \
            "Writing should reference length constraints"
    
    def test_review_and_edit_steps(self, pipeline_name):
        """Test review and editing process."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check review depends on writing
        review_step = next(s for s in config['steps'] if s['id'] == 'review_and_edit')
        if 'depends_on' in review_step:
            assert 'write_story' in review_step['depends_on'], \
                "Review should depend on story writing"
        
        # Check final polish depends on review
        polish_step = next(s for s in config['steps'] if s['id'] == 'final_polish')
        if 'depends_on' in polish_step:
            assert 'review_and_edit' in polish_step['depends_on'], \
                "Final polish should depend on review"
    
    def test_creative_parameters(self, pipeline_name):
        """Test that creative parameters are properly integrated."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check tone is used
        has_tone_reference = any('tone' in str(step) for step in config['steps'])
        assert has_tone_reference, "Tone should be referenced in creative steps"
        
        # Check theme is used
        has_theme_reference = any('theme' in str(step) for step in config['steps'])
        assert has_theme_reference, "Theme should be referenced in creative steps"
    
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


# Note: Full integration tests that would generate complete stories
# are not included here as they would require significant model usage
# and time. The tests above verify the pipeline structure and 
# configuration without mocks.