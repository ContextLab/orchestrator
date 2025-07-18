"""Tests for content_creation_pipeline.yaml example."""
import pytest
from unittest.mock import AsyncMock, patch
from .test_base import BaseExampleTest


class TestContentCreationPipelineYAML(BaseExampleTest):
    """Test the content creation pipeline YAML pipeline."""
    
    @pytest.fixture
    def pipeline_name(self):
        return "content_creation_pipeline.yaml"
    
    @pytest.fixture
    def sample_inputs(self):
        return {
            "topic": "The Future of Renewable Energy",
            "content_type": "blog_series",
            "target_audience": "technology professionals",
            "tone": "informative",
            "num_pieces": 5,
            "include_images": True,
            "optimize_seo": True
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
            'create_content',
            'review_content',
            'generate_images',
            'optimize_seo',
            'finalize_content'
        ]
        
        for step in required_steps:
            assert step in step_ids, f"Missing required step: {step}"
    
    def test_content_type_handling(self, pipeline_name):
        """Test different content type configurations."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check content type is used in generation
        create_step = next(s for s in config['steps'] if s['id'] == 'create_blog_content')
        assert '{{content_type}}' in str(create_step)
        
        # Check loop for multiple pieces
        assert 'loop' in create_step
        assert create_step['loop']['max_iterations'] == '{{num_pieces}}'
    
    @pytest.mark.asyncio
    async def test_research_and_outline(self, orchestrator, pipeline_name, sample_inputs):
        """Test research and outline generation."""
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            async def mock_step_execution(step, context, state):
                step_id = step.get('id')
                
                if step_id == 'research_topic':
                    return {
                        'result': {
                            'key_points': [
                                'Solar energy advancements',
                                'Wind power innovations',
                                'Energy storage solutions'
                            ],
                            'sources': ['source1.com', 'source2.org'],
                            'statistics': {'growth_rate': '15%', 'market_size': '$1.5T'}
                        }
                    }
                elif step_id == 'generate_outline':
                    return {
                        'result': {
                            'outline': [
                                {'title': 'Introduction to Renewable Energy', 'sections': 3},
                                {'title': 'Solar Power Revolution', 'sections': 4},
                                {'title': 'Wind Energy Innovations', 'sections': 3},
                                {'title': 'Energy Storage Breakthroughs', 'sections': 4},
                                {'title': 'Future Outlook', 'sections': 2}
                            ]
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=sample_inputs
            )
            
            # Verify research and outline were called
            research_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'research_topic'
            ]
            outline_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'generate_outline'
            ]
            
            assert len(research_calls) == 1
            assert len(outline_calls) == 1
    
    @pytest.mark.asyncio
    async def test_content_creation_loop(self, orchestrator, pipeline_name):
        """Test content creation for multiple pieces."""
        inputs = {
            "topic": "AI in Healthcare",
            "content_type": "article_series",
            "num_pieces": 3,
            "include_images": False
        }
        
        content_pieces_created = 0
        
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            async def mock_step_execution(step, context, state):
                nonlocal content_pieces_created
                step_id = step.get('id')
                
                if step_id == 'create_blog_content':
                    content_pieces_created += 1
                    return {
                        'result': {
                            'content': f'Article {content_pieces_created} content',
                            'word_count': 1500,
                            'piece_number': content_pieces_created
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify correct number of content pieces were created
            assert content_pieces_created == inputs['num_pieces']
    
    @pytest.mark.asyncio
    async def test_conditional_image_generation(self, orchestrator, pipeline_name):
        """Test conditional image generation."""
        # Test with images enabled
        inputs_with_images = {
            "topic": "Test Topic",
            "content_type": "blog",
            "include_images": True
        }
        
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            async def mock_step_execution(step, context, state):
                if step.get('id') == 'generate_visuals':
                    return {
                        'result': {
                            'images': [
                                {'id': 'img1', 'alt_text': 'Renewable energy diagram'},
                                {'id': 'img2', 'alt_text': 'Solar panel installation'}
                            ]
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs_with_images
            )
            
            # Check image generation was called
            image_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'generate_visuals'
            ]
            assert len(image_calls) > 0
        
        # Test with images disabled
        inputs_no_images = {
            "topic": "Test Topic",
            "content_type": "blog",
            "include_images": False
        }
        
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {'result': {}}
            
            await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs_no_images
            )
            
            # Check image generation was NOT called
            image_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'generate_visuals'
            ]
            assert len(image_calls) == 0
    
    @pytest.mark.asyncio
    async def test_seo_optimization(self, orchestrator, pipeline_name):
        """Test SEO optimization functionality."""
        inputs = {
            "topic": "Digital Marketing Trends",
            "content_type": "blog",
            "optimize_seo": True
        }
        
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            async def mock_step_execution(step, context, state):
                if step.get('id') == 'optimize_seo':
                    return {
                        'result': {
                            'meta_title': 'Digital Marketing Trends 2024: Complete Guide',
                            'meta_description': 'Discover the latest digital marketing trends...',
                            'keywords': ['digital marketing', 'marketing trends', '2024'],
                            'readability_score': 85,
                            'seo_score': 92
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify SEO optimization was performed
            seo_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'optimize_seo'
            ]
            assert len(seo_calls) > 0
    
    def test_quality_review_step(self, pipeline_name):
        """Test quality review configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Find review step
        review_step = next(s for s in config['steps'] if s['id'] == 'review_content')
        
        # Check review criteria
        assert 'grammar' in review_step['action'].lower()
        assert 'accuracy' in review_step['action'].lower()
        assert 'tone' in review_step['action'].lower()
    
    def test_output_completeness(self, pipeline_name):
        """Test that all content outputs are defined."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        expected_outputs = [
            'content_pieces',
            'total_word_count',
            'seo_metadata',
            'publication_schedule'
        ]
        
        for output in expected_outputs:
            assert output in config['outputs'], f"Missing output: {output}"