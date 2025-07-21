"""Tests for scalable_customer_service_agent.yaml example.

This test file follows the NO MOCKS policy. Tests use real orchestration
when API keys are available, otherwise they skip gracefully.
"""
import pytest
from .test_base import BaseExampleTest


class TestScalableCustomerServiceAgentYAML(BaseExampleTest):
    """Test the scalable customer service agent YAML pipeline."""
    
    @pytest.fixture
    def pipeline_name(self):
        return "scalable_customer_service_agent.yaml"
    
    @pytest.fixture
    def sample_inputs(self):
        return {
            "interaction_id": "INT-123456",
            "customer_id": "CUST-789456",
            "channel": "email",
            "content": "My premium subscription isn't working properly",
            "metadata": {},
            "business_hours": {"start": "09:00", "end": "17:00", "timezone": "UTC"},
            "sla_targets": {"first_response": 60, "resolution": 3600},
            "languages": ["en", "es", "fr"]
        }
    
    def test_pipeline_structure(self, pipeline_name):
        """Test that the pipeline has valid structure."""
        self.validate_pipeline_structure(pipeline_name)
        
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check service agent steps
        step_ids = [step['id'] for step in config['steps']]
        required_steps = [
            'receive_interaction',
            'identify_customer',
            'analyze_sentiment',
            'classify_intent',
            'search_knowledge_base',
            'determine_routing',
            'generate_response',
            'quality_check'
        ]
        
        for step in required_steps:
            assert step in step_ids, f"Missing required step: {step}"
    
    def test_multi_channel_support(self, pipeline_name):
        """Test multi-channel configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check channel routing
        receive_step = next(s for s in config['steps'] if s['id'] == 'receive_interaction')
        assert '{{channel}}' in str(receive_step)
        
        # Check send response step formats for channel
        send_step = next(s for s in config['steps'] if s['id'] == 'send_response')
        assert send_step is not None
        assert '{{channel}}' in str(send_step)
    
    @pytest.mark.asyncio
    async def test_customer_data_loading(self, orchestrator, pipeline_name, sample_inputs):
        """Test customer data retrieval and analysis."""
        # Test pipeline execution with minimal responses
        result = await self.run_pipeline_test(
            orchestrator,
            pipeline_name,
            sample_inputs,
            expected_outputs={
                'interaction_id': str,
                'resolution_type': str,
                'response_sent': (bool, str)
            },
            use_minimal_responses=True
        )
        
        # Verify result structure
        assert result is not None
        assert 'outputs' in result or 'steps' in result
    
    def test_sla_monitoring(self, pipeline_name):
        """Test SLA checking and prioritization configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check SLA step exists
        sla_step = next(s for s in config['steps'] if s['id'] == 'check_sla')
        assert sla_step is not None
        assert '{{sla_targets}}' in str(sla_step)
        assert 'First response time' in sla_step['action']
    
    def test_automated_solution_search(self, pipeline_name):
        """Test automated solution finding configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check knowledge base search step
        kb_step = next(s for s in config['steps'] if s['id'] == 'search_knowledge_base')
        assert kb_step is not None
        assert '{{content}}' in str(kb_step)
        assert '{{classify_intent.result.primary_intent}}' in str(kb_step)
        
        # Check automation eligibility step
        auto_step = next(s for s in config['steps'] if s['id'] == 'check_automation')
        assert auto_step is not None
        assert 'condition' in auto_step
        assert auto_step['condition'] == '{{determine_routing.result.decision}} == \'automated\''
    
    def test_quality_assurance(self, pipeline_name):
        """Test response quality checking configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check quality check step
        quality_step = next(s for s in config['steps'] if s['id'] == 'quality_check')
        assert quality_step is not None
        assert 'Compliance with policies' in quality_step['action']
        assert 'Brand voice consistency' in quality_step['action']
        
        # Check send response has quality condition
        send_step = next(s for s in config['steps'] if s['id'] == 'send_response')
        assert 'condition' in send_step
        assert 'qa_score' in send_step['condition']
    
    def test_performance_tracking(self, pipeline_name):
        """Test performance metrics collection configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check analytics logging step
        analytics_step = next(s for s in config['steps'] if s['id'] == 'log_analytics')
        assert analytics_step is not None
        assert 'Response time' in analytics_step['action']
        assert 'Automation success rate' in analytics_step['action']
    
    def test_customer_tier_handling(self, pipeline_name):
        """Test customer tier based prioritization."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check customer tier usage
        steps_with_tier = [
            s for s in config['steps'] 
            if 'customer_profile.tier' in str(s)
        ]
        
        assert len(steps_with_tier) > 0
    
    def test_multi_language_support(self, pipeline_name):
        """Test language handling configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Find translation step
        translate_step = next(
            (s for s in config['steps'] if 'translate' in s.get('id', '')),
            None
        )
        
        assert translate_step is not None
        assert 'customer_profile.language' in str(translate_step)
    
    def test_output_completeness(self, pipeline_name):
        """Test that all service outputs are defined."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        expected_outputs = [
            'interaction_id',
            'resolution_type',
            'response_sent',
            'sla_status',
            'response_time_seconds',
            'quality_score'
        ]
        
        for output in expected_outputs:
            assert output in config['outputs'], f"Missing output: {output}"
    
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