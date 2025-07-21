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
    
    def test_research_steps_configuration(self, pipeline_name):
        """Test that research steps are properly configured."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check that key research steps exist and have proper actions
        analyze_step = next((s for s in config['steps'] if s['id'] == 'analyze_query'), None)
        assert analyze_step is not None
        assert 'action' in analyze_step
        
        # Check web search step
        web_search_step = next((s for s in config['steps'] if s['id'] == 'web_search'), None)
        assert web_search_step is not None
        
        # Check credibility analysis
        cred_step = next((s for s in config['steps'] if s['id'] == 'analyze_credibility'), None)
        assert cred_step is not None
    
    @pytest.mark.asyncio
    async def test_pipeline_execution(self, orchestrator, pipeline_name, sample_inputs):
        """Test pipeline execution with minimal responses."""
        # Test with minimal responses to avoid expensive API calls
        result = await self.run_pipeline_test(
            orchestrator,
            pipeline_name,
            sample_inputs,
            expected_outputs={
                'report_markdown': str,
                'key_findings': list
            },
            use_minimal_responses=True
        )
        
        # Verify execution completed
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_quality_check_execution(self, orchestrator, pipeline_name, sample_inputs):
        """Test quality check execution."""
        # Load and validate pipeline structure
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Find quality check step
        quality_step = next((s for s in config['steps'] if s['id'] == 'quality_check'), None)
        assert quality_step is not None
        
        # Test with minimal responses
        result = await self.run_pipeline_test(
            orchestrator,
            pipeline_name,
            sample_inputs,
            use_minimal_responses=True
        )
        
        # Verify execution completed
        assert result is not None
    
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
    
    @pytest.mark.asyncio
    async def test_web_search_configuration(self, orchestrator, pipeline_name):
        """Test web search step configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Find web search step
        web_search_step = next((s for s in config['steps'] if s['id'] == 'web_search'), None)
        assert web_search_step is not None
        
        # Verify it uses search queries from analyze_query
        assert '{{analyze_query' in str(web_search_step) or 'search_queries' in str(web_search_step)
    
    @pytest.mark.asyncio
    async def test_credibility_analysis(self, orchestrator, pipeline_name, sample_inputs):
        """Test credibility analysis functionality."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Find credibility analysis step
        cred_step = next((s for s in config['steps'] if s['id'] == 'analyze_credibility'), None)
        assert cred_step is not None
        
        # Test pipeline execution
        result = await self.run_pipeline_test(
            orchestrator,
            pipeline_name,
            sample_inputs,
            use_minimal_responses=True
        )
        
        assert result is not None