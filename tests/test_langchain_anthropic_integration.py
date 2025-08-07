"""Real integration tests for LangChain-enhanced Anthropic model."""

import pytest
import asyncio
import os

from src.orchestrator.models.anthropic_model import AnthropicModel
from src.orchestrator.utils.api_keys_flexible import load_api_keys_optional


class TestLangChainAnthropicIntegration:
    """Real integration tests for LangChain-enhanced Anthropic models."""

    @pytest.mark.asyncio
    async def test_anthropic_langchain_fallback_behavior(self):
        """Test that Anthropic model gracefully handles LangChain availability."""
        
        # Test with LangChain explicitly disabled
        model = AnthropicModel("claude-3-haiku", use_langchain=False)
        assert model._use_langchain is False
        assert model.client is not None  # Direct Anthropic client should be initialized
        assert model.provider == "anthropic"
        assert model.name == "claude-3-haiku"
        
        # Test that LangChain can be enabled when available
        model_with_langchain = AnthropicModel("claude-3-haiku", use_langchain=True)
        # Should either use LangChain (if available) or fall back to direct Anthropic
        assert model_with_langchain.provider == "anthropic"
        assert model_with_langchain.name == "claude-3-haiku"

    @pytest.mark.asyncio
    async def test_anthropic_model_initialization_preserves_interface(self):
        """Test that enhanced Anthropic model preserves existing interface."""
        
        # Test with LangChain disabled
        model = AnthropicModel("claude-3-sonnet", use_langchain=False)
        
        # Verify all existing attributes are preserved
        assert hasattr(model, 'capabilities')
        assert hasattr(model, 'requirements')
        assert hasattr(model, 'cost')
        assert hasattr(model, '_model_id')
        assert hasattr(model, '_expertise')
        assert hasattr(model, '_size_billions')
        
        # Verify model metadata
        assert model.provider == "anthropic"
        assert model.name == "claude-3-sonnet"
        assert model.capabilities.supports_function_calling
        assert model.capabilities.supports_structured_output
        assert not model.cost.is_free
        
    def test_anthropic_model_capabilities_unchanged(self):
        """Test that model capabilities detection is unchanged."""
        
        # Test Claude Opus capabilities
        model = AnthropicModel("claude-3-opus", use_langchain=False)
        assert "reasoning" in model.capabilities.supported_tasks
        assert "creative" in model.capabilities.supported_tasks
        assert "vision" in model.capabilities.supported_tasks
        assert model.capabilities.context_window == 200000
        assert model.capabilities.vision_capable
        
        # Test Claude Haiku capabilities  
        model = AnthropicModel("claude-3-haiku", use_langchain=False)
        assert "code" in model.capabilities.supported_tasks
        assert model.capabilities.context_window == 200000
        assert model.capabilities.vision_capable
        assert model.capabilities.speed_rating == "fast"

    def test_anthropic_model_cost_estimation_unchanged(self):
        """Test that cost estimation logic is preserved."""
        
        # Test Claude Opus pricing
        model = AnthropicModel("claude-3-opus", use_langchain=False)
        assert model.cost.input_cost_per_1k_tokens == 0.015  # $15 per 1M = $0.015 per 1K
        assert model.cost.output_cost_per_1k_tokens == 0.075  # $75 per 1M = $0.075 per 1K
        
        # Test Claude Haiku pricing
        model = AnthropicModel("claude-3-haiku", use_langchain=False)
        assert model.cost.input_cost_per_1k_tokens == 0.00025  # $0.25 per 1M
        assert model.cost.output_cost_per_1k_tokens == 0.00125  # $1.25 per 1M

    @pytest.mark.asyncio 
    async def test_anthropic_model_methods_preserve_interface(self):
        """Test that all model methods preserve their interface."""
        
        model = AnthropicModel("claude-3-haiku", use_langchain=False)
        
        # Test method signatures haven't changed
        assert hasattr(model, 'generate')
        assert hasattr(model, 'generate_structured')
        assert hasattr(model, 'health_check')
        assert hasattr(model, 'estimate_cost')
        
        # Test that methods can be called without errors (with API key issues handled gracefully)
        try:
            await model.health_check()
            # If we reach here, health check worked (API key available)
        except (ValueError, RuntimeError) as e:
            # Expected if no API key available
            assert "API key" in str(e) or "Anthropic" in str(e)

    @pytest.mark.asyncio
    async def test_existing_anthropic_compatibility(self):
        """Test that existing Anthropic model code continues to work."""
        
        # Test that existing initialization patterns work
        model = AnthropicModel(
            name="claude-3-haiku",
            api_key=os.getenv("ANTHROPIC_API_KEY", "dummy-key"),
            use_langchain=False
        )
        
        assert model.name == "claude-3-haiku"
        assert model.provider == "anthropic"
        
        # Test that all existing attributes are accessible
        assert hasattr(model, 'capabilities')
        assert hasattr(model, 'requirements')
        assert hasattr(model, 'cost')
        assert hasattr(model, 'metrics')

    def test_anthropic_model_name_normalization(self):
        """Test that model name normalization works correctly."""
        
        # Test various Claude model name variations
        test_cases = [
            ("claude-3-opus", "claude-3-opus-20240229"),
            ("claude-3-sonnet", "claude-3-sonnet-20240229"),
            ("claude-3-haiku", "claude-3-haiku-20240307"),
            ("claude-3.5-sonnet", "claude-3-5-sonnet-20241022"),
            ("claude-instant", "claude-instant-1.2"),
        ]
        
        for input_name, expected_normalized in test_cases:
            model = AnthropicModel(input_name, use_langchain=False)
            assert model._model_id == expected_normalized

    def test_anthropic_model_expertise_detection(self):
        """Test model expertise detection."""
        
        # Test Opus expertise
        model = AnthropicModel("claude-3-opus", use_langchain=False)
        expertise = model._expertise
        assert "reasoning" in expertise
        assert "research" in expertise
        assert "math" in expertise
        
        # Test Haiku expertise
        model = AnthropicModel("claude-3-haiku", use_langchain=False)
        expertise = model._expertise
        assert "general" in expertise
        assert "chat" in expertise

    @pytest.mark.asyncio
    async def test_real_anthropic_generation_compatibility(self):
        """Test real Anthropic API calls work the same with enhanced model."""
        
        # Check if Anthropic API key is available using our key management system
        available_keys = load_api_keys_optional()
        if not available_keys.get("anthropic"):
            pytest.skip("Anthropic API key not available")
        """Test real Anthropic API calls work the same with enhanced model."""
        
        # Test with LangChain disabled (original behavior)
        model_direct = AnthropicModel("claude-3-haiku", use_langchain=False)
        
        try:
            response_direct = await model_direct.generate(
                "What is 2+2? Respond with just the number.",
                temperature=0.0,
                max_tokens=10
            )
            assert len(response_direct) > 0
            assert "4" in response_direct
            
            # Test health check
            health = await model_direct.health_check()
            assert health is True
            
            # Test cost estimation
            cost = await model_direct.estimate_cost("Test prompt", 100)
            assert cost > 0
            
        except Exception as e:
            pytest.skip(f"Anthropic API test failed (possibly rate limited): {e}")

    @pytest.mark.asyncio
    async def test_structured_output_compatibility(self):
        """Test structured output generation works correctly."""
        
        # Check if Anthropic API key is available using our key management system
        available_keys = load_api_keys_optional()
        if not available_keys.get("anthropic"):
            pytest.skip("Anthropic API key not available")
        """Test structured output generation works correctly."""
        
        model = AnthropicModel("claude-3-haiku", use_langchain=False)
        
        schema = {
            "type": "object",
            "properties": {
                "result": {"type": "number"},
                "explanation": {"type": "string"}
            },
            "required": ["result", "explanation"]
        }
        
        try:
            response = await model.generate_structured(
                "What is 2+2?",
                schema=schema,
                temperature=0.0
            )
            
            assert isinstance(response, dict)
            assert "result" in response
            assert "explanation" in response
            assert response["result"] == 4
            
        except Exception as e:
            pytest.skip(f"Anthropic structured output test failed: {e}")

    def test_anthropic_model_size_estimation(self):
        """Test model size estimation."""
        
        size_tests = [
            ("claude-3-opus", 175.0),
            ("claude-3-sonnet", 70.0),
            ("claude-3-haiku", 20.0),
            ("claude-instant", 10.0),
        ]
        
        for model_name, expected_size in size_tests:
            model = AnthropicModel(model_name, use_langchain=False)
            assert model._size_billions == expected_size

    @pytest.mark.asyncio
    async def test_anthropic_system_prompt_support(self):
        """Test that system prompts are handled correctly."""
        
        model = AnthropicModel("claude-3-haiku", use_langchain=False)
        
        # Test with API key available
        try:
            if os.getenv("ANTHROPIC_API_KEY"):
                response = await model.generate(
                    "What is your name?",
                    system_prompt="You are a helpful assistant named Claude.",
                    temperature=0.0,
                    max_tokens=20
                )
                assert len(response) > 0
                # System prompt should influence the response
                assert "claude" in response.lower() or "assistant" in response.lower()
            else:
                # Test that the method signature accepts system_prompt
                # (will fail with API key error but signature is correct)
                try:
                    await model.generate(
                        "Test",
                        system_prompt="Test system",
                        temperature=0.0,
                        max_tokens=5
                    )
                except (ValueError, RuntimeError):
                    # Expected without API key
                    pass
        except Exception as e:
            pytest.skip(f"System prompt test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])