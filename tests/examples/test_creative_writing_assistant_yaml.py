"""Tests for creative_writing_assistant.yaml example."""
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
        
        # Check genre is used throughout
        brainstorm_step = next(s for s in config['steps'] if s['id'] == 'generate_premise')
        assert '{{genre}}' in str(brainstorm_step)
        
        # Check worldbuilding condition for specific genres
        world_step = next(s for s in config['steps'] if s['id'] == 'build_world')
        assert 'condition' in world_step
        assert 'fantasy' in world_step['condition']
        assert 'sci-fi' in world_step['condition']
    
    @pytest.mark.asyncio
    async def test_brainstorming_and_outlining(self, orchestrator, pipeline_name):
        """Test idea generation and story outlining."""
        # Test pipeline structure
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Validate relevant configuration
        assert 'steps' in config
        assert len(config['steps']) > 0
    async def mock_step_execution(step, context, state):
                step_id = step.get('id')
                
                if step_id == 'generate_premise':
                    return {
                        'result': {
                            'core_concept': 'Aliens communicate through music',
                            'plot_points': [
                                'Discovery of alien signal',
                                'Decoding musical patterns',
                                'First contact through symphony'
                            ],
                            'conflicts': ['Language barrier', 'Fear vs curiosity'],
                            'themes': ['Universal language of music', 'Unity through art']
                        }
                    }
                elif step_id == 'outline_chapters':
                    return {
                        'result': {
                            'chapters': [
                                {'title': 'The Signal', 'scenes': 3},
                                {'title': 'Decoding', 'scenes': 4},
                                {'title': 'First Notes', 'scenes': 3}
                            ],
                            'story_arc': 'discovery → understanding → connection',
                            'estimated_words': 4800
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=sample_inputs
            )
            
            # Verify brainstorming and outlining
            brainstorm_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'generate_premise'
            ]
            outline_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'outline_chapters'
            ]
            
            assert len(brainstorm_calls) == 1
            assert len(outline_calls) == 1
    
    @pytest.mark.asyncio
    async def test_character_development(self, orchestrator, pipeline_name):
        """Test character creation and development."""
        # Test pipeline structure
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Validate relevant configuration
        assert 'steps' in config
        assert len(config['steps']) > 0
    async def mock_step_execution(step, context, state):
                if step.get('id') == 'develop_characters':
                    return {
                        'result': {
                            'characters': [
                                {
                                    'name': 'Detective Sarah Chen',
                                    'role': 'protagonist',
                                    'personality': 'methodical, intuitive',
                                    'backstory': 'Former forensic scientist',
                                    'arc': 'learning to trust instincts'
                                },
                                {
                                    'name': 'Marcus Webb',
                                    'role': 'suspect',
                                    'personality': 'charming, secretive',
                                    'backstory': 'Art dealer with hidden past',
                                    'arc': 'revealing true motives'
                                }
                            ],
                            'relationships': [
                                {'between': ['Sarah', 'Marcus'], 'type': 'suspicion/attraction'}
                            ]
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify character development
            character_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'develop_characters'
            ]
            assert len(character_calls) > 0
    
    @pytest.mark.asyncio
    async def test_conditional_worldbuilding(self, orchestrator, pipeline_name):
        """Test conditional worldbuilding for appropriate genres."""
        # Test pipeline structure
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Validate relevant configuration
        assert 'steps' in config
        assert len(config['steps']) > 0
    async def mock_step_execution(step, context, state):
                if step.get('id') == 'build_world':
                    return {
                        'result': {
                            'setting': 'Mars colony, 2157',
                            'technology': ['Neural interfaces', 'Quantum communication'],
                            'society': 'Post-scarcity economy',
                            'physics_rules': 'FTL travel via wormholes',
                            'cultures': ['Martian-born', 'Earth refugees']
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=scifi_inputs
            )
            
            # Check worldbuilding was called
            world_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'build_world'
            ]
            assert len(world_calls) > 0
        
        # Test with contemporary (should skip worldbuilding)
        contemporary_inputs = {
            "genre": "contemporary",
            "include_worldbuilding": True,
            "length": "short_story"
        }
        
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {'result': {}}
            
            await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=contemporary_inputs
            )
            
            # Check worldbuilding was NOT called
            world_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'build_world'
            ]
            assert len(world_calls) == 0
    
    @pytest.mark.asyncio
    async def test_story_writing_chapters(self, orchestrator, pipeline_name):
        """Test chapter-by-chapter story writing."""
        # Test pipeline structure
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Validate relevant configuration
        assert 'steps' in config
        assert len(config['steps']) > 0
    async def mock_step_execution(step, context, state):
                nonlocal chapters_written
                step_id = step.get('id')
                
                if step_id == 'outline_chapters':
                    return {
                        'result': {
                            'chapters': [
                                {'title': 'The Prophecy', 'scenes': 3},
                                {'title': 'The Journey Begins', 'scenes': 4},
                                {'title': 'The Final Battle', 'scenes': 5}
                            ]
                        }
                    }
                elif step_id == 'write_key_scenes':
                    chapters_written += 1
                    return {
                        'result': {
                            'chapter_text': f'Chapter {chapters_written} content...',
                            'word_count': 6500,
                            'chapter_number': chapters_written
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Should write multiple chapters
            assert chapters_written >= 3
    
    @pytest.mark.asyncio
    async def test_review_and_editing(self, orchestrator, pipeline_name):
        """Test story review and editing process."""
        # Test pipeline structure
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Validate relevant configuration
        assert 'steps' in config
        assert len(config['steps']) > 0
    async def mock_step_execution(step, context, state):
                step_id = step.get('id')
                
                if step_id == 'check_consistency':
                    return {
                        'result': {
                            'consistency_issues': [
                                'Character name changed from John to Jon in chapter 2'
                            ],
                            'pacing_notes': 'Middle section needs tightening',
                            'tone_adjustments': 'Increase tension in climax',
                            'suggested_edits': 15
                        }
                    }
                elif step_id == 'apply_fixes':
                    return {
                        'result': {
                            'edits_applied': 15,
                            'word_count_change': -200,
                            'improved_sections': ['climax', 'character_introductions']
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify review and editing
            review_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'check_consistency'
            ]
            edit_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'apply_fixes'
            ]
            
            assert len(review_calls) > 0
            assert len(edit_calls) > 0
    
    def test_pov_configuration(self, pipeline_name):
        """Test point of view handling."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check POV is used in writing
        write_step = next(
            (s for s in config['steps'] if 'write' in s['id'] and 'story' in s['id']),
            None
        )
        
        assert write_step is not None
        assert '{{pov}}' in str(write_step)
    
    def test_output_completeness(self, pipeline_name):
        """Test that all writing outputs are defined."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        expected_outputs = [
            'story_text',
            'word_count',
            'character_profiles',
            'story_summary',
            'themes'
        ]
        
        for output in expected_outputs:
            assert output in config['outputs'], f"Missing output: {output}"