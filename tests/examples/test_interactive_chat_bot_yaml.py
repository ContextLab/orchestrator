"""Tests for interactive_chat_bot.yaml example.

This test file follows the NO MOCKS policy. Tests use real orchestration
when API keys are available, otherwise they skip gracefully.
"""
import pytest
from .test_base import BaseExampleTest


class TestInteractiveChatBotYAML(BaseExampleTest):
    """Test the interactive chat bot YAML pipeline."""
    
    @pytest.fixture
    def pipeline_name(self):
        return "interactive_chat_bot.yaml"
    
    @pytest.fixture
    def sample_inputs(self):
        return {
            "message": "What's the weather like today?",
            "conversation_id": "conv_123",
            "persona": "helpful-assistant",
            "enable_streaming": False,
            "safety_level": "moderate",
            "available_tools": ["web_search", "calculator", "weather"],
            "max_response_length": 500
        }
    
    def test_pipeline_structure(self, pipeline_name):
        """Test that the pipeline has valid structure."""
        self.validate_pipeline_structure(pipeline_name)
        
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check chat bot steps
        step_ids = [step['id'] for step in config['steps']]
        required_steps = [
            'process_input',
            'safety_check',
            'retrieve_context',
            'classify_intent',
            'generate_response',
            'update_memory'
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
        assert '{{enable_streaming}}' in stream_step['condition']
    
    @pytest.mark.asyncio
    async def test_context_loading(self, orchestrator, pipeline_name, sample_inputs):
        """Test conversation context loading."""
        # Test pipeline execution with minimal responses
        result = await self.run_pipeline_test(
            orchestrator,
            pipeline_name,
            sample_inputs,
            expected_outputs={
                'response': str,
                'intent': str,
                'conversation_id': str
            },
            use_minimal_responses=True
        )
        
        # Verify result structure
        assert result is not None
        assert 'outputs' in result or 'steps' in result
    
    def test_intent_analysis(self, pipeline_name):
        """Test user intent analysis configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check intent classification step
        intent_step = next(s for s in config['steps'] if s['id'] == 'classify_intent')
        assert intent_step is not None
        assert 'Primary intent' in intent_step['action']
        assert 'confidence scores' in intent_step['action']
    
    def test_tool_usage(self, pipeline_name):
        """Test conditional tool usage configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check tool selection step
        select_step = next(s for s in config['steps'] if s['id'] == 'select_tools')
        assert select_step is not None
        assert '{{available_tools}}' in str(select_step)
        
        # Check tool execution step
        execute_step = next(s for s in config['steps'] if s['id'] == 'execute_tools')
        assert 'condition' in execute_step
        assert execute_step['condition'] == '{{select_tools.result.tools|length}} > 0'
    
    def test_memory_update(self, pipeline_name):
        """Test conversation memory management configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check fact extraction step
        extract_step = next(s for s in config['steps'] if s['id'] == 'extract_facts')
        assert extract_step is not None
        assert 'User preferences' in extract_step['action']
        
        # Check memory update step
        memory_step = next(s for s in config['steps'] if s['id'] == 'update_memory')
        assert memory_step is not None
        assert '{{conversation_id}}' in str(memory_step)
    
    def test_streaming_response(self, pipeline_name):
        """Test streaming response generation configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check prepare streaming step
        stream_step = next(s for s in config['steps'] if s['id'] == 'prepare_streaming')
        assert stream_step is not None
        assert 'condition' in stream_step
        assert stream_step['condition'] == '{{enable_streaming}} == true'
    
    def test_personality_configuration(self, pipeline_name):
        """Test personality handling in responses."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Find response generation step
        response_step = next(s for s in config['steps'] if s['id'] == 'generate_response')
        
        # Check persona is used
        assert '{{persona}}' in str(response_step)
    
    def test_safety_configuration(self, pipeline_name):
        """Test safety check configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check safety check step
        safety_step = next(s for s in config['steps'] if s['id'] == 'safety_check')
        assert safety_step is not None
        assert '{{safety_level}}' in str(safety_step)
        
        # Check that retrieve_context depends on safety
        context_step = next(s for s in config['steps'] if s['id'] == 'retrieve_context')
        assert 'condition' in context_step
        assert 'is_safe' in context_step['condition']
    
    def test_output_completeness(self, pipeline_name):
        """Test that all chat outputs are defined."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        expected_outputs = [
            'response',
            'intent',
            'confidence',
            'tools_used',
            'conversation_id'
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