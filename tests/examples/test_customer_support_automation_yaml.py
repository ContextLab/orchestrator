"""Tests for customer_support_automation.yaml example.

This test file follows the NO MOCKS policy. Tests use real orchestration
when API keys are available, otherwise they skip gracefully.
"""
import pytest
from .test_base import BaseExampleTest


class TestCustomerSupportAutomationYAML(BaseExampleTest):
    """Test the customer support automation YAML pipeline."""
    
    @pytest.fixture
    def pipeline_name(self):
        return "customer_support_automation.yaml"
    
    @pytest.fixture
    def sample_inputs(self):
        return {
            "ticket_id": "TICKET-12345",
            "ticketing_system": "zendesk",
            "auto_respond": True,
            "languages": ["en", "es", "fr"],
            "escalation_threshold": -0.5,
            "kb_confidence_threshold": 0.75
        }
    
    def test_pipeline_structure(self, pipeline_name):
        """Test that the pipeline has valid structure."""
        self.validate_pipeline_structure(pipeline_name)
        
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check support workflow steps
        step_ids = [step['id'] for step in config['steps']]
        required_steps = [
            'receive_ticket',
            'detect_language',
            'analyze_sentiment',
            'extract_entities',
            'classify_ticket',
            'search_knowledge_base',
            'check_automation_eligibility',
            'generate_response',
            'assign_to_agent'
        ]
        
        for step in required_steps:
            assert step in step_ids, f"Missing required step: {step}"
    
    def test_language_detection(self, pipeline_name):
        """Test language handling in the pipeline."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check language is used in response generation
        response_step = next(s for s in config['steps'] if s['id'] == 'generate_response')
        assert 'Language:' in str(response_step) or 'language' in str(response_step)
    
    @pytest.mark.asyncio
    async def test_ticket_analysis(self, orchestrator, pipeline_name, sample_inputs):
        """Test ticket analysis and categorization."""
        # Test pipeline execution with minimal responses
        result = await self.run_pipeline_test(
            orchestrator,
            pipeline_name,
            sample_inputs,
            expected_outputs={
                'category': str,
                'sentiment_score': (int, float, str),
                'automation_status': str
            },
            use_minimal_responses=True
        )
        
        # Verify result structure
        assert result is not None
        assert 'outputs' in result or 'steps' in result
    
    def test_customer_history_configuration(self, pipeline_name):
        """Test customer history lookup configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check that receive_ticket retrieves customer history
        receive_step = next(s for s in config['steps'] if s['id'] == 'receive_ticket')
        assert 'customer information and history' in str(receive_step)
        
        # Check that extract_entities processes the information
        extract_step = next(s for s in config['steps'] if s['id'] == 'extract_entities')
        assert extract_step is not None
    
    def test_knowledge_base_search_configuration(self, pipeline_name):
        """Test knowledge base solution search configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check that search_knowledge_base exists
        kb_step = next(s for s in config['steps'] if s['id'] == 'search_knowledge_base')
        assert kb_step is not None
        assert 'knowledge base for solutions' in str(kb_step)
        
        # Check that KB search depends on classification
        if 'depends_on' in kb_step:
            assert 'classify_ticket' in kb_step['depends_on']
    
    def test_escalation_logic(self, pipeline_name):
        """Test escalation decision making configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check automation eligibility step
        eligibility_step = next(s for s in config['steps'] if s['id'] == 'check_automation_eligibility')
        assert eligibility_step is not None
        assert 'escalation_threshold' in str(eligibility_step)
        
        # Check assign_to_agent step has condition
        assign_step = next(s for s in config['steps'] if s['id'] == 'assign_to_agent')
        assert 'condition' in assign_step
        assert 'can_automate}} == false' in assign_step['condition']
    
    def test_sentiment_analysis_configuration(self, pipeline_name):
        """Test sentiment checking and response adjustment configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check sentiment analysis step exists
        sentiment_step = next(s for s in config['steps'] if s['id'] == 'analyze_sentiment')
        assert sentiment_step is not None
        assert 'customer sentiment and emotion' in str(sentiment_step)
        
        # Check that response generation considers sentiment
        response_step = next(s for s in config['steps'] if s['id'] == 'generate_response')
        assert 'empathy matching customer sentiment' in str(response_step)
    
    def test_conditional_translation(self, pipeline_name):
        """Test conditional translation step."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Find translation step
        translate_step = next(
            (s for s in config['steps'] if s['id'] == 'translate_response'), 
            None
        )
        
        assert translate_step is not None
        assert 'condition' in translate_step
        assert "!= 'en'" in translate_step['condition']
    
    def test_output_completeness(self, pipeline_name):
        """Test that all support outputs are defined."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        expected_outputs = [
            'ticket_id',
            'automation_status',
            'category',
            'sentiment_score',
            'language'
        ]
        
        for output in expected_outputs:
            assert output in config['outputs'], f"Missing output: {output}"
    
    @pytest.mark.asyncio 
    async def test_full_automation_flow(self, orchestrator, pipeline_name, sample_inputs):
        """Test full automation flow with minimal responses."""
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