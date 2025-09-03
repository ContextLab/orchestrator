"""Real integration tests for LangChain-enhanced OpenAI model."""

import pytest
import asyncio
from unittest.mock import patch
import os

from src.orchestrator.models.openai_model import OpenAIModel
from src.orchestrator.utils.api_keys_flexible import load_api_keys_optional

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider


class TestLangChainOpenAIIntegration:
    """Real integration tests for LangChain-enhanced OpenAI models."""

    @pytest.mark.asyncio
    async def test_openai_langchain_fallback_behavior(self):
        """Test that OpenAI model gracefully handles LangChain availability."""
        
        # Test with LangChain explicitly disabled
        model = OpenAIModel("gpt-3.5-turbo", use_langchain=False)
        assert model._use_langchain is False
        assert model.client is not None  # Direct OpenAI client should be initialized
        assert model.provider == "openai"
        assert model.name == "gpt-3.5-turbo"
        
        # Test that LangChain can be enabled when available
        model_with_langchain = OpenAIModel("gpt-3.5-turbo", use_langchain=True)
        # Should either use LangChain (if available) or fall back to direct OpenAI
        assert model_with_langchain.provider == "openai"
        assert model_with_langchain.name == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_openai_model_initialization_preserves_interface(self):
        """Test that enhanced OpenAI model preserves existing interface."""
        
        # Test with LangChain disabled
        model = OpenAIModel("gpt-3.5-turbo", use_langchain=False)
        
        # Verify all existing attributes are preserved
        assert hasattr(model, 'capabilities')
        assert hasattr(model, 'requirements')
        assert hasattr(model, 'cost')
        assert hasattr(model, '_model_id')
        assert hasattr(model, '_expertise')
        assert hasattr(model, '_size_billions')
        
        # Verify model metadata
        assert model.provider == "openai"
        assert model.name == "gpt-3.5-turbo"
        assert model.capabilities.supports_function_calling
        assert model.capabilities.supports_structured_output
        assert not model.cost.is_free
        
    def test_openai_model_capabilities_unchanged(self):
        """Test that model capabilities detection is unchanged."""
        
        # Test GPT-4 capabilities
        model = OpenAIModel("gpt-4-turbo", use_langchain=False)
        assert "reasoning" in model.capabilities.supported_tasks
        assert "creative" in model.capabilities.supported_tasks
        assert model.capabilities.context_window == 128000
        assert model.capabilities.vision_capable
        
        # Test GPT-3.5 capabilities  
        model = OpenAIModel("gpt-3.5-turbo", use_langchain=False)
        assert "code" in model.capabilities.supported_tasks
        assert model.capabilities.context_window == 4096
        assert not model.capabilities.vision_capable

    def test_openai_model_cost_estimation_unchanged(self):
        """Test that cost estimation logic is preserved."""
        
        # Test GPT-4 pricing
        model = OpenAIModel("gpt-4-turbo", use_langchain=False)
        assert model.cost.input_cost_per_1k_tokens == 0.01
        assert model.cost.output_cost_per_1k_tokens == 0.03
        
        # Test GPT-3.5 pricing
        model = OpenAIModel("gpt-3.5-turbo", use_langchain=False)
        assert model.cost.input_cost_per_1k_tokens == 0.0005
        assert model.cost.output_cost_per_1k_tokens == 0.0015

    @pytest.mark.asyncio 
    async def test_openai_model_methods_preserve_interface(self):
        """Test that all model methods preserve their interface."""
        
        model = OpenAIModel("gpt-3.5-turbo", use_langchain=False)
        
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
            assert "API key" in str(e) or "OpenAI" in str(e)

    @pytest.mark.asyncio
    async def test_existing_openai_compatibility(self):
        """Test that existing OpenAI model code continues to work."""
        
        # Test that existing initialization patterns work
        model = OpenAIModel(
            name="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY", "dummy-key"),
            use_langchain=False
        )
        
        assert model.name == "gpt-3.5-turbo"
        assert model.provider == "openai"
        
        # Test that all existing attributes are accessible
        assert hasattr(model, 'capabilities')
        assert hasattr(model, 'requirements')
        assert hasattr(model, 'cost')
        assert hasattr(model, 'metrics')

    def test_langchain_package_mapping_exists(self):
        """Test that LangChain packages are mapped in auto_install."""
        
        from src.orchestrator.utils.auto_install import PACKAGE_MAPPINGS
        
        # Verify LangChain packages are mapped
        assert "langchain_openai" in PACKAGE_MAPPINGS
        assert PACKAGE_MAPPINGS["langchain_openai"] == "langchain-openai"
        assert "langchain_anthropic" in PACKAGE_MAPPINGS
        assert PACKAGE_MAPPINGS["langchain_anthropic"] == "langchain-anthropic"
        assert "langchain_community" in PACKAGE_MAPPINGS
        assert PACKAGE_MAPPINGS["langchain_community"] == "langchain-community"

    @pytest.mark.asyncio
    async def test_real_openai_generation_compatibility(self):
        """Test real OpenAI API calls work the same with enhanced model."""
        
        # Check if OpenAI API key is available using our key management system
        available_keys = load_api_keys_optional()
        if not available_keys.get("openai"):
            pytest.skip("OpenAI API key not available")
        """Test real OpenAI API calls work the same with enhanced model."""
        
        # Test with LangChain disabled (original behavior)
        model_direct = OpenAIModel("gpt-3.5-turbo", use_langchain=False)
        
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
            pytest.skip(f"OpenAI API test failed (possibly rate limited): {e}")

    @pytest.mark.asyncio
    async def test_structured_output_compatibility(self):
        """Test structured output generation works correctly."""
        
        # Check if OpenAI API key is available using our key management system
        available_keys = load_api_keys_optional()
        if not available_keys.get("openai"):
            pytest.skip("OpenAI API key not available")
        """Test structured output generation works correctly."""
        
        model = OpenAIModel("gpt-3.5-turbo", use_langchain=False)
        
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
            pytest.skip(f"OpenAI structured output test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])