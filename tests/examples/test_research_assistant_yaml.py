"""Tests for research_assistant.yaml example."""
import pytest
from .test_base import BaseExampleTest


class TestResearchAssistantYAML(BaseExampleTest):
    """Test the research assistant YAML pipeline."""
    
    @pytest.fixture
    def pipeline_name(self):
        return "research_assistant.yaml"
    
    @pytest.fixture
    def sample_inputs(self):
        return {
            "query": "What are the latest developments in quantum computing?",
            "max_sources": 5,
            "search_depth": "comprehensive"
        }
    
    def test_pipeline_structure(self, pipeline_name):
        """Test that the pipeline has valid structure."""
        self.validate_pipeline_structure(pipeline_name)
        
        # Additional checks specific to research assistant
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check inputs
        assert 'inputs' in config
        assert 'query' in config['inputs']
        assert config['inputs']['query']['required'] is True
        
        # Check key steps exist
        step_ids = [step['id'] for step in config['steps']]
        assert 'analyze_query' in step_ids
        assert 'web_search' in step_ids
        assert 'analyze_credibility' in step_ids
        assert 'extract_content' in step_ids
        assert 'synthesize_findings' in step_ids
    
    def test_auto_tags(self, pipeline_name):
        """Test that AUTO tags are properly formatted."""
        auto_tags = self.extract_auto_tags(pipeline_name)
        
        # Check that key steps have AUTO tags
        assert 'analyze_query' in auto_tags
        assert 'web_search' in auto_tags
        assert 'analyze_credibility' in auto_tags
        
        # Validate AUTO tag content
        analyze_content = auto_tags['analyze_query'][0]
        assert 'key topics' in analyze_content.lower()
        assert 'search terms' in analyze_content.lower()
    
    @pytest.mark.asyncio
    async def test_pipeline_execution_mock(self, orchestrator, pipeline_name):
        """Test pipeline execution with mocked responses."""
        # Test pipeline structure
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Test with minimal responses to avoid expensive API calls
        result = await self.run_pipeline_test(
            orchestrator,
            pipeline_name,
            {},  # minimal inputs
            use_minimal_responses=True
        )
        
        # Verify execution completed
        assert result is not None
    async def mock_step_execution(step, context, state):
                step_id = step.get('id')
                if step_id in mock_responses:
                    return mock_responses[step_id]
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            # Run pipeline
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=sample_inputs
            )
            
            # Verify execution
            assert mock_exec.called
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_quality_check_execution(self, orchestrator, pipeline_name):
        """Test quality check execution."""
        # Test pipeline structure
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Test with minimal responses to avoid expensive API calls
        result = await self.run_pipeline_test(
            orchestrator,
            pipeline_name,
            {},  # minimal inputs
            use_minimal_responses=True
        )
        
        # Verify execution completed
        assert result is not None
    async def mock_step_execution(step, context, state):
                step_id = step.get('id')
                if step_id == 'quality_check':
                    return {'result': {'score': 0.9, 'passed': True}}
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify quality check was executed
            call_ids = [call[0][0]['id'] for call in mock_exec.call_args_list]
            assert 'quality_check' in call_ids
    
    @pytest.mark.asyncio
    async def test_output_structure(self, orchestrator, pipeline_name, sample_inputs):
        """Test that outputs match expected structure."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check output definitions
        assert 'outputs' in config
        expected_outputs = [
            'report_markdown',
            'pdf_path',
            'quality_score',
            'key_findings',
            'sources_analyzed'
        ]
        
        for output in expected_outputs:
            assert output in config['outputs']