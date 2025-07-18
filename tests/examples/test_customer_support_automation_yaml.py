"""Tests for customer_support_automation.yaml example."""
import pytest
from unittest.mock import AsyncMock, patch
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
            "customer_message": "My account was charged twice for the same order",
            "customer_id": "CUST-98765",
            "priority": "high",
            "category": "billing",
            "language": "en",
            "escalate_threshold": 0.8
        }
    
    def test_pipeline_structure(self, pipeline_name):
        """Test that the pipeline has valid structure."""
        self.validate_pipeline_structure(pipeline_name)
        
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check support workflow steps
        step_ids = [step['id'] for step in config['steps']]
        required_steps = [
            'analyze_ticket',
            'retrieve_customer_history',
            'categorize_issue',
            'check_knowledge_base',
            'generate_response',
            'sentiment_check',
            'escalation_check'
        ]
        
        for step in required_steps:
            assert step in step_ids, f"Missing required step: {step}"
    
    def test_language_detection(self, pipeline_name):
        """Test language handling in the pipeline."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check language is used in response generation
        response_step = next(s for s in config['steps'] if s['id'] == 'generate_response')
        assert '{{language}}' in str(response_step)
    
    @pytest.mark.asyncio
    async def test_ticket_analysis(self, orchestrator, pipeline_name, sample_inputs):
        """Test ticket analysis and categorization."""
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            async def mock_step_execution(step, context, state):
                step_id = step.get('id')
                
                if step_id == 'analyze_sentiment':
                    return {
                        'result': {
                            'intent': 'billing_dispute',
                            'sentiment': 'negative',
                            'urgency': 'high',
                            'key_entities': ['double charge', 'same order']
                        }
                    }
                elif step_id == 'classify_ticket':
                    return {
                        'result': {
                            'primary_category': 'billing',
                            'subcategory': 'duplicate_charge',
                            'confidence': 0.95
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=sample_inputs
            )
            
            # Verify analysis was performed
            analysis_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'analyze_sentiment'
            ]
            assert len(analysis_calls) == 1
    
    @pytest.mark.asyncio
    async def test_customer_history_retrieval(self, orchestrator, pipeline_name):
        """Test customer history lookup."""
        inputs = {
            "ticket_id": "TICKET-001",
            "customer_id": "CUST-123",
            "customer_message": "Need help with my order"
        }
        
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            async def mock_step_execution(step, context, state):
                if step.get('id') == 'extract_entities':
                    return {
                        'result': {
                            'previous_tickets': 3,
                            'account_status': 'premium',
                            'lifetime_value': 2500,
                            'recent_issues': [
                                {'date': '2024-01-15', 'type': 'shipping_delay'}
                            ]
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify history was retrieved
            history_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'extract_entities'
            ]
            assert len(history_calls) > 0
    
    @pytest.mark.asyncio
    async def test_knowledge_base_search(self, orchestrator, pipeline_name):
        """Test knowledge base solution search."""
        inputs = {
            "ticket_id": "TICKET-002",
            "customer_message": "How do I reset my password?",
            "category": "account"
        }
        
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            async def mock_step_execution(step, context, state):
                if step.get('id') == 'search_knowledge_base':
                    return {
                        'result': {
                            'found_solution': True,
                            'articles': [
                                {
                                    'id': 'KB-001',
                                    'title': 'Password Reset Guide',
                                    'relevance': 0.98
                                }
                            ],
                            'suggested_steps': [
                                'Click "Forgot Password" on login page',
                                'Enter your email address',
                                'Check email for reset link'
                            ]
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify KB was searched
            kb_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'search_knowledge_base'
            ]
            assert len(kb_calls) > 0
    
    @pytest.mark.asyncio
    async def test_escalation_logic(self, orchestrator, pipeline_name):
        """Test escalation decision making."""
        # Test high complexity requiring escalation
        inputs_escalate = {
            "ticket_id": "TICKET-003",
            "customer_message": "Legal action if not resolved",
            "priority": "critical",
            "escalate_threshold": 0.7
        }
        
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            async def mock_step_execution(step, context, state):
                step_id = step.get('id')
                
                if step_id == 'check_automation_eligibility':
                    return {
                        'result': {
                            'complexity_score': 0.85,
                            'should_escalate': True,
                            'reason': 'Legal threat detected'
                        }
                    }
                elif step_id == 'assign_to_agent':
                    return {
                        'result': {
                            'assigned_to': 'senior_agent_01',
                            'escalation_notes': 'Customer threatening legal action'
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs_escalate
            )
            
            # Verify escalation was triggered
            escalation_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'assign_to_agent'
            ]
            assert len(escalation_calls) > 0
    
    @pytest.mark.asyncio
    async def test_sentiment_analysis(self, orchestrator, pipeline_name):
        """Test sentiment checking and response adjustment."""
        inputs = {
            "ticket_id": "TICKET-004",
            "customer_message": "Extremely frustrated with your service!",
            "category": "general"
        }
        
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            async def mock_step_execution(step, context, state):
                step_id = step.get('id')
                
                if step_id == 'analyze_sentiment':
                    return {
                        'result': {
                            'sentiment_score': -0.8,
                            'emotion': 'angry',
                            'requires_empathy': True
                        }
                    }
                elif step_id == 'translate_response':
                    return {
                        'result': {
                            'adjusted_response': 'I sincerely apologize for the frustration...',
                            'empathy_level': 'high',
                            'offers_made': ['discount', 'priority_support']
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify sentiment analysis and tone adjustment
            sentiment_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'analyze_sentiment'
            ]
            tone_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'translate_response'
            ]
            
            assert len(sentiment_calls) > 0
            assert len(tone_calls) > 0
    
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
        assert "{{language}} != 'en'" in translate_step['condition']
    
    def test_output_completeness(self, pipeline_name):
        """Test that all support outputs are defined."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        expected_outputs = [
            'ticket_status',
            'response_text',
            'resolution_time',
            'customer_satisfaction_prediction',
            'follow_up_required'
        ]
        
        for output in expected_outputs:
            assert output in config['outputs'], f"Missing output: {output}"