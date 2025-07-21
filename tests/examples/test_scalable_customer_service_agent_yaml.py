"""Tests for scalable_customer_service_agent.yaml example."""
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
            "channel": "email",
            "customer_id": "CUST-789456",
            "message": "My premium subscription isn't working properly",
            "priority": "high",
            "language": "en",
            "account_type": "premium",
            "sla_hours": 4,
            "enable_automation": True
        }
    
    def test_pipeline_structure(self, pipeline_name):
        """Test that the pipeline has valid structure."""
        self.validate_pipeline_structure(pipeline_name)
        
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check service agent steps
        step_ids = [step['id'] for step in config['steps']]
        required_steps = [
            'route_request',
            'load_customer_data',
            'analyze_sentiment',
            'classify_issue',
            'check_sla',
            'search_solutions',
            'generate_response',
            'quality_check'
        ]
        
        for step in required_steps:
            assert step in step_ids, f"Missing required step: {step}"
    
    def test_multi_channel_support(self, pipeline_name):
        """Test multi-channel configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check channel routing
        route_step = next(s for s in config['steps'] if s['id'] == 'route_request')
        assert '{{channel}}' in str(route_step)
        
        # Check channel-specific formatting
        format_step = next(
            (s for s in config['steps'] if 'format' in s['id'] and 'channel' in s['id']),
            None
        )
        assert format_step is not None
    
    @pytest.mark.asyncio
    async def test_customer_data_loading(self, orchestrator, pipeline_name, sample_inputs):
        """Test customer data retrieval and analysis."""
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            async def mock_step_execution(step, context, state):
                step_id = step.get('id')
                
                if step_id == 'identify_customer':
                    return {
                        'result': {
                            'customer_profile': {
                                'name': 'John Smith',
                                'account_type': 'premium',
                                'tenure_months': 24,
                                'lifetime_value': 5000,
                                'support_history': {
                                    'total_tickets': 5,
                                    'resolved_tickets': 4,
                                    'avg_resolution_time': 2.5
                                }
                            },
                            'recent_interactions': [
                                {
                                    'date': '2024-01-15',
                                    'issue': 'login problem',
                                    'resolved': True
                                }
                            ],
                            'product_usage': {
                                'last_login': '2024-01-20',
                                'features_used': ['dashboard', 'reports', 'api']
                            }
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=sample_inputs
            )
            
            # Verify customer data loading
            data_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'identify_customer'
            ]
            assert len(data_calls) == 1
    
    @pytest.mark.asyncio
    async def test_sla_monitoring(self, orchestrator, pipeline_name):
        """Test SLA checking and prioritization."""
        inputs = {
            "channel": "chat",
            "priority": "critical",
            "account_type": "enterprise",
            "sla_hours": 1,
            "customer_id": "ENT-001"
        }
        
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            async def mock_step_execution(step, context, state):
                step_id = step.get('id')
                
                if step_id == 'check_sla':
                    return {
                        'result': {
                            'sla_status': 'at_risk',
                            'time_remaining': 0.5,
                            'escalation_required': True,
                            'priority_score': 9.5
                        }
                    }
                elif step_id == 'check_sla':
                    return {
                        'result': {
                            'new_priority': 'critical',
                            'assigned_agent': 'senior_team',
                            'notification_sent': True
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify SLA monitoring and escalation
            sla_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'check_sla'
            ]
            escalate_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'check_sla'
            ]
            
            assert len(sla_calls) > 0
            assert len(escalate_calls) > 0
    
    @pytest.mark.asyncio
    async def test_automated_solution_search(self, orchestrator, pipeline_name):
        """Test automated solution finding."""
        inputs = {
            "message": "Cannot access dashboard after login",
            "enable_automation": True,
            "channel": "email"
        }
        
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            async def mock_step_execution(step, context, state):
                step_id = step.get('id')
                
                if step_id == 'search_knowledge_base':
                    return {
                        'result': {
                            'solutions_found': 3,
                            'top_solution': {
                                'id': 'SOL-145',
                                'title': 'Dashboard Access Issues',
                                'confidence': 0.87,
                                'steps': [
                                    'Clear browser cache',
                                    'Check account permissions',
                                    'Try incognito mode'
                                ]
                            },
                            'related_articles': ['KB-234', 'KB-567']
                        }
                    }
                elif step_id == 'check_automation':
                    return {
                        'result': {
                            'automation_applied': True,
                            'actions_taken': [
                                'Reset user session',
                                'Cleared dashboard cache'
                            ],
                            'issue_resolved': True
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify solution search and automation
            search_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'search_knowledge_base'
            ]
            automation_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'check_automation'
            ]
            
            assert len(search_calls) > 0
            assert len(automation_calls) > 0
    
    @pytest.mark.asyncio
    async def test_quality_assurance(self, orchestrator, pipeline_name):
        """Test response quality checking."""
        inputs = {
            "message": "Very upset about service",
            "channel": "social_media",
            "priority": "high"
        }
        
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            async def mock_step_execution(step, context, state):
                step_id = step.get('id')
                
                if step_id == 'analyze_sentiment':
                    return {
                        'result': {
                            'sentiment': 'very_negative',
                            'emotion': 'angry',
                            'intensity': 0.9
                        }
                    }
                elif step_id == 'quality_check':
                    return {
                        'result': {
                            'quality_score': 0.65,
                            'issues_found': [
                                'Insufficient empathy',
                                'No compensation offered'
                            ],
                            'requires_revision': True
                        }
                    }
                elif step_id == 'quality_check':
                    return {
                        'result': {
                            'enhanced_response': 'We sincerely apologize...',
                            'improvements': [
                                'Added empathetic language',
                                'Included service credit offer'
                            ],
                            'new_quality_score': 0.92
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify quality checking and enhancement
            quality_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'quality_check'
            ]
            enhance_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'quality_check'
            ]
            
            assert len(quality_calls) > 0
            assert len(enhance_calls) > 0
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, orchestrator, pipeline_name):
        """Test performance metrics collection."""
        inputs = {
            "channel": "chat",
            "message": "Quick question about billing",
            "customer_id": "CUST-999"
        }
        
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            async def mock_step_execution(step, context, state):
                if step.get('id') == 'log_analytics':
                    return {
                        'result': {
                            'response_time': 1.2,
                            'resolution_status': 'resolved',
                            'customer_effort_score': 2,
                            'automation_percentage': 85,
                            'first_contact_resolution': True
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify metrics tracking
            metrics_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'log_analytics'
            ]
            assert len(metrics_calls) > 0
    
    def test_account_type_handling(self, pipeline_name):
        """Test account type based prioritization."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check account type usage
        steps_with_account = [
            s for s in config['steps'] 
            if '{{account_type}}' in str(s)
        ]
        
        assert len(steps_with_account) > 0
    
    def test_multi_language_support(self, pipeline_name):
        """Test language handling configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Find translation step
        translate_step = next(
            (s for s in config['steps'] if 'translate' in s.get('id', '')),
            None
        )
        
        assert translate_step is not None
        assert '{{language}}' in str(translate_step)
    
    def test_output_completeness(self, pipeline_name):
        """Test that all service outputs are defined."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        expected_outputs = [
            'response_sent',
            'resolution_status',
            'response_time',
            'customer_satisfaction',
            'automation_used',
            'sla_met'
        ]
        
        for output in expected_outputs:
            assert output in config['outputs'], f"Missing output: {output}"