"""Tests for interactive_chat_bot.yaml example."""
import pytest
from unittest.mock import AsyncMock, patch
from .test_base import BaseExampleTest


class TestInteractiveChatBotYAML(BaseExampleTest):
    """Test the interactive chat bot YAML pipeline."""
    
    @pytest.fixture
    def pipeline_name(self):
        return "interactive_chat_bot.yaml"
    
    @pytest.fixture
    def sample_inputs(self):
        return {
            "user_message": "What's the weather like today?",
            "session_id": "session_123",
            "user_id": "user_456",
            "context_window": 10,
            "enable_tools": True,
            "personality": "helpful",
            "stream_response": True
        }
    
    def test_pipeline_structure(self, pipeline_name):
        """Test that the pipeline has valid structure."""
        self.validate_pipeline_structure(pipeline_name)
        
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check chat bot steps
        step_ids = [step['id'] for step in config['steps']]
        required_steps = [
            'load_context',
            'analyze_intent',
            'check_tools',
            'generate_response',
            'update_memory',
            'stream_output'
        ]
        
        for step in required_steps:
            assert step in step_ids, f"Missing required step: {step}"
    
    def test_streaming_configuration(self, pipeline_name):
        """Test streaming response configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Find stream output step
        stream_step = next(s for s in config['steps'] if s['id'] == 'prepare_streaming')
        
        # Check streaming condition
        assert 'condition' in stream_step
        assert '{{stream_response}}' in stream_step['condition']
    
    @pytest.mark.asyncio
    async def test_context_loading(self, orchestrator, pipeline_name, sample_inputs):
        """Test conversation context loading."""
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            async def mock_step_execution(step, context, state):
                step_id = step.get('id')
                
                if step_id == 'retrieve_context':
                    return {
                        'result': {
                            'conversation_history': [
                                {'role': 'user', 'content': 'Hello'},
                                {'role': 'assistant', 'content': 'Hi! How can I help?'},
                                {'role': 'user', 'content': 'What time is it?'},
                                {'role': 'assistant', 'content': 'It\'s 2:30 PM'}
                            ],
                            'user_preferences': {
                                'timezone': 'EST',
                                'language': 'en'
                            },
                            'session_data': {
                                'start_time': '2024-01-20T14:00:00',
                                'message_count': 4
                            }
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=sample_inputs
            )
            
            # Verify context loading
            context_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'retrieve_context'
            ]
            assert len(context_calls) == 1
    
    @pytest.mark.asyncio
    async def test_intent_analysis(self, orchestrator, pipeline_name):
        """Test user intent analysis."""
        inputs = {
            "user_message": "Book a flight to New York next Friday",
            "session_id": "session_789",
            "enable_tools": True
        }
        
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            async def mock_step_execution(step, context, state):
                if step.get('id') == 'classify_intent':
                    return {
                        'result': {
                            'primary_intent': 'booking_request',
                            'entities': {
                                'destination': 'New York',
                                'date': 'next Friday',
                                'service': 'flight'
                            },
                            'confidence': 0.92,
                            'requires_tool': True,
                            'tool_needed': 'flight_booking'
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify intent analysis
            intent_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'classify_intent'
            ]
            assert len(intent_calls) > 0
    
    @pytest.mark.asyncio
    async def test_tool_usage(self, orchestrator, pipeline_name):
        """Test conditional tool usage."""
        inputs = {
            "user_message": "Calculate 25% of 840",
            "enable_tools": True,
            "session_id": "calc_session"
        }
        
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            async def mock_step_execution(step, context, state):
                step_id = step.get('id')
                
                if step_id == 'classify_intent':
                    return {
                        'result': {
                            'primary_intent': 'calculation',
                            'requires_tool': True,
                            'tool_needed': 'calculator'
                        }
                    }
                elif step_id == 'select_tools':
                    return {
                        'result': {
                            'tool_available': True,
                            'tool_name': 'calculator',
                            'can_execute': True
                        }
                    }
                elif step_id == 'execute_tools':
                    return {
                        'result': {
                            'tool_output': '210',
                            'calculation': '25% of 840 = 210',
                            'execution_time': 0.05
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify tool execution
            tool_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'execute_tools'
            ]
            assert len(tool_calls) > 0
    
    @pytest.mark.asyncio
    async def test_memory_update(self, orchestrator, pipeline_name):
        """Test conversation memory management."""
        inputs = {
            "user_message": "My name is Alice",
            "session_id": "memory_test",
            "context_window": 5
        }
        
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            async def mock_step_execution(step, context, state):
                step_id = step.get('id')
                
                if step_id == 'update_memory':
                    return {
                        'result': {
                            'facts_extracted': [
                                {'type': 'user_name', 'value': 'Alice'}
                            ],
                            'context_updated': True,
                            'messages_stored': 1,
                            'old_messages_pruned': 0
                        }
                    }
                elif step_id == 'update_memory':
                    return {
                        'result': {
                            'messages_kept': 5,
                            'messages_removed': 2,
                            'summary_created': True
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify memory updates
            memory_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'update_memory'
            ]
            assert len(memory_calls) > 0
    
    @pytest.mark.asyncio
    async def test_streaming_response(self, orchestrator, pipeline_name):
        """Test streaming response generation."""
        inputs = {
            "user_message": "Tell me a story",
            "stream_response": True,
            "session_id": "stream_test"
        }
        
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            async def mock_step_execution(step, context, state):
                step_id = step.get('id')
                
                if step_id == 'generate_response':
                    return {
                        'result': {
                            'response': 'Once upon a time, in a land far away...',
                            'tokens': 12,
                            'completion_time': 1.2
                        }
                    }
                elif step_id == 'prepare_streaming':
                    return {
                        'result': {
                            'chunks_sent': 5,
                            'streaming_time': 0.8,
                            'average_chunk_size': 2.4
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify streaming was triggered
            stream_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'prepare_streaming'
            ]
            assert len(stream_calls) > 0
    
    def test_personality_configuration(self, pipeline_name):
        """Test personality handling in responses."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Find response generation step
        response_step = next(s for s in config['steps'] if s['id'] == 'generate_response')
        
        # Check personality is used
        assert '{{personality}}' in str(response_step)
    
    def test_fallback_handling(self, pipeline_name):
        """Test fallback response configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Find fallback step
        fallback_step = next(
            (s for s in config['steps'] if s['id'] == 'generate_fallback'),
            None
        )
        
        assert fallback_step is not None
        assert 'error_handler' in fallback_step or 'condition' in fallback_step
    
    def test_output_completeness(self, pipeline_name):
        """Test that all chat outputs are defined."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        expected_outputs = [
            'response',
            'intent',
            'confidence',
            'tools_used',
            'session_updated'
        ]
        
        for output in expected_outputs:
            assert output in config['outputs'], f"Missing output: {output}"