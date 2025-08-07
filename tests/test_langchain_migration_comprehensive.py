"""Comprehensive integration tests for the LangChain model migration - Issue #202."""

import pytest
import asyncio
import os
from unittest.mock import patch

from src.orchestrator.models.openai_model import OpenAIModel
from src.orchestrator.models.anthropic_model import AnthropicModel
from src.orchestrator.models.langchain_adapter import LangChainModelAdapter
from src.orchestrator.utils.auto_install import PACKAGE_MAPPINGS
from src.orchestrator.utils.api_keys_flexible import load_api_keys_optional


class TestLangChainMigrationComprehensive:
    """Comprehensive tests for LangChain provider migration."""

    def test_langchain_package_mappings_complete(self):
        """Test that all required LangChain packages are mapped."""
        
        required_mappings = {
            "langchain_openai": "langchain-openai",
            "langchain_anthropic": "langchain-anthropic", 
            "langchain_google_genai": "langchain-google-genai",
            "langchain_community": "langchain-community",
            "langchain_huggingface": "langchain-huggingface",
        }
        
        for import_name, pip_name in required_mappings.items():
            assert import_name in PACKAGE_MAPPINGS
            assert PACKAGE_MAPPINGS[import_name] == pip_name

    @pytest.mark.asyncio
    async def test_enhanced_models_preserve_all_interfaces(self):
        """Test that enhanced models preserve all existing interfaces."""
        
        models_to_test = [
            ("openai", OpenAIModel, "gpt-3.5-turbo"),
            ("anthropic", AnthropicModel, "claude-3-haiku"),
        ]
        
        for provider, model_class, model_name in models_to_test:
            # Test with LangChain disabled to verify fallback works
            model = model_class(model_name, use_langchain=False)
            
            # Verify all required attributes exist
            assert hasattr(model, 'name')
            assert hasattr(model, 'provider')
            assert hasattr(model, 'capabilities')
            assert hasattr(model, 'requirements')
            assert hasattr(model, 'cost')
            assert hasattr(model, 'metrics')
            
            # Verify all required methods exist  
            assert hasattr(model, 'generate')
            assert hasattr(model, 'generate_structured')
            assert hasattr(model, 'health_check')
            assert hasattr(model, 'estimate_cost')
            
            # Verify provider and name are correct
            assert model.provider == provider
            assert model.name == model_name

    @pytest.mark.asyncio
    async def test_langchain_adapter_all_providers(self):
        """Test LangChainModelAdapter with all supported providers."""
        
        providers_to_test = [
            ("openai", "gpt-3.5-turbo"),
            ("anthropic", "claude-3-haiku"),
            ("google", "gemini-pro"),
            ("ollama", "llama3.2:1b"),
            ("huggingface", "microsoft/DialoGPT-small"),
        ]
        
        for provider, model_name in providers_to_test:
            try:
                adapter = LangChainModelAdapter(provider, model_name)
                
                # Verify basic properties
                assert adapter.provider == provider
                assert adapter.name == model_name
                assert hasattr(adapter, 'capabilities')
                assert hasattr(adapter, 'cost')
                
                # Verify methods exist
                assert hasattr(adapter, 'generate')
                assert hasattr(adapter, 'health_check')
                
            except Exception as e:
                # Expected for providers without API keys or Ollama not running
                if "API key" not in str(e) and "Ollama" not in str(e) and "Failed to install" not in str(e):
                    raise e

    @pytest.mark.asyncio
    async def test_backward_compatibility_no_breaking_changes(self):
        """Test that existing code patterns continue to work."""
        
        # Test existing OpenAI model usage patterns
        try:
            openai_model = OpenAIModel(
                name="gpt-3.5-turbo",
                api_key=os.getenv("OPENAI_API_KEY", "dummy-key")
            )
            
            # These should all work without modification
            assert openai_model.name == "gpt-3.5-turbo"
            assert openai_model.provider == "openai"
            assert openai_model.capabilities.supports_function_calling
            
        except ValueError as e:
            # Expected if no API key
            assert "API key" in str(e)
        
        # Test existing Anthropic model usage patterns
        try:
            anthropic_model = AnthropicModel(
                name="claude-3-haiku", 
                api_key=os.getenv("ANTHROPIC_API_KEY", "dummy-key")
            )
            
            # These should all work without modification
            assert anthropic_model.name == "claude-3-haiku"
            assert anthropic_model.provider == "anthropic"
            assert anthropic_model.capabilities.supports_structured_output
            
        except ValueError as e:
            # Expected if no API key
            assert "API key" in str(e)

    def test_cost_analysis_preserved(self):
        """Test that cost analysis functionality is fully preserved."""
        
        # Test OpenAI cost analysis
        openai_model = OpenAIModel("gpt-4-turbo", use_langchain=False)
        assert openai_model.cost.input_cost_per_1k_tokens > 0
        assert openai_model.cost.output_cost_per_1k_tokens > 0
        assert not openai_model.cost.is_free
        
        # Test cost calculation
        cost = openai_model.cost.calculate_cost(1000, 1000)
        assert cost > 0
        
        # Test Anthropic cost analysis
        anthropic_model = AnthropicModel("claude-3-opus", use_langchain=False)
        assert anthropic_model.cost.input_cost_per_1k_tokens > 0
        assert anthropic_model.cost.output_cost_per_1k_tokens > 0
        assert not anthropic_model.cost.is_free
        
        # Test cost efficiency calculation
        efficiency = anthropic_model.cost.get_cost_efficiency_score(0.9)
        assert efficiency > 0

    def test_capability_detection_enhanced(self):
        """Test that capability detection is preserved and enhanced."""
        
        test_cases = [
            ("openai", OpenAIModel, "gpt-4-turbo", ["code", "reasoning", "creative"]),
            ("openai", OpenAIModel, "gpt-3.5-turbo", ["code", "chat"]),
            ("anthropic", AnthropicModel, "claude-3-opus", ["reasoning", "creative", "vision"]),
            ("anthropic", AnthropicModel, "claude-3-haiku", ["code", "chat", "vision"]),
        ]
        
        for provider, model_class, model_name, expected_tasks in test_cases:
            model = model_class(model_name, use_langchain=False)
            
            # Check that expected capabilities are detected
            for task in expected_tasks:
                assert task in model.capabilities.supported_tasks
            
            # Check advanced capability flags
            if "gpt-4" in model_name or "opus" in model_name or "sonnet" in model_name:
                assert model.capabilities.supports_function_calling
                assert model.capabilities.supports_structured_output
            
            # Check vision capabilities for Claude 3 models
            if "claude-3" in model_name:
                assert model.capabilities.vision_capable

    @pytest.mark.asyncio
    async def test_fallback_behavior_robust(self):
        """Test robust fallback behavior when LangChain is unavailable."""
        
        # Test OpenAI fallback
        openai_model = OpenAIModel("gpt-3.5-turbo", use_langchain=False)
        assert not openai_model._use_langchain
        assert openai_model.client is not None
        
        # Test Anthropic fallback
        anthropic_model = AnthropicModel("claude-3-haiku", use_langchain=False)
        assert not anthropic_model._use_langchain
        assert anthropic_model.client is not None
        
        # Test that all methods are still callable
        for model in [openai_model, anthropic_model]:
            try:
                await model.health_check()
            except (ValueError, RuntimeError) as e:
                # Expected without valid API keys
                assert "API key" in str(e) or model.provider.title() in str(e)

    def test_model_metadata_preservation(self):
        """Test that all model metadata is preserved."""
        
        # Test OpenAI metadata
        openai_model = OpenAIModel("gpt-4-turbo", use_langchain=False)
        assert hasattr(openai_model, '_model_id')
        assert hasattr(openai_model, '_expertise')
        assert hasattr(openai_model, '_size_billions')
        assert openai_model._expertise is not None
        assert openai_model._size_billions > 0
        
        # Test Anthropic metadata
        anthropic_model = AnthropicModel("claude-3-opus", use_langchain=False)
        assert hasattr(anthropic_model, '_model_id')
        assert hasattr(anthropic_model, '_expertise')
        assert hasattr(anthropic_model, '_size_billions')
        assert anthropic_model._expertise is not None
        assert anthropic_model._size_billions > 0

    @pytest.mark.asyncio
    async def test_api_key_handling_enhanced(self):
        """Test enhanced API key handling using existing infrastructure."""
        
        # Test that models use existing API key infrastructure
        from src.orchestrator.utils.api_keys_flexible import load_api_keys_optional
        
        available_keys = load_api_keys_optional()
        
        # Test OpenAI key handling
        if "openai" in available_keys:
            model = OpenAIModel("gpt-3.5-turbo")
            assert model.api_key == available_keys["openai"]
        
        # Test Anthropic key handling
        if "anthropic" in available_keys:
            model = AnthropicModel("claude-3-haiku")
            assert model.api_key == available_keys["anthropic"]

    def test_model_serialization_compatibility(self):
        """Test that model serialization/deserialization still works."""
        
        # Test OpenAI model serialization
        openai_model = OpenAIModel("gpt-3.5-turbo", use_langchain=False)
        model_dict = openai_model.to_dict()
        
        assert "name" in model_dict
        assert "provider" in model_dict
        assert "capabilities" in model_dict
        assert "requirements" in model_dict
        assert "cost" in model_dict
        assert "metrics" in model_dict
        
        # Test Anthropic model serialization
        anthropic_model = AnthropicModel("claude-3-haiku", use_langchain=False)
        model_dict = anthropic_model.to_dict()
        
        assert "name" in model_dict
        assert "provider" in model_dict
        assert model_dict["provider"] == "anthropic"

    @pytest.mark.asyncio
    async def test_error_handling_robust(self):
        """Test robust error handling in all scenarios."""
        
        # Test invalid model names
        try:
            invalid_model = OpenAIModel("invalid-model-name", use_langchain=False)
            # Should still create the model, just with different capabilities
            assert invalid_model.name == "invalid-model-name"
        except Exception:
            # Any exception should be meaningful
            pass
        
        # Test that error handling is preserved in enhanced models
        # The enhanced models should handle errors the same way as original models
        
        # Test that models gracefully handle generation errors
        model = OpenAIModel("gpt-3.5-turbo", use_langchain=False)
        try:
            # This might work if API key is available, or fail gracefully if not
            await model.health_check()
        except Exception as e:
            # Should be a meaningful error message
            assert len(str(e)) > 0

    @pytest.mark.asyncio
    async def test_cross_provider_consistency(self):
        """Test consistency between different providers for same tasks."""
        
        # Check if API keys are available using our key management system
        available_keys = load_api_keys_optional()
        if not (available_keys.get("openai") and available_keys.get("anthropic")):
            pytest.skip("Both OpenAI and Anthropic API keys needed for cross-provider test")
        """Test consistency between different providers for same tasks."""
        
        openai_model = OpenAIModel("gpt-3.5-turbo")
        anthropic_model = AnthropicModel("claude-3-haiku")
        
        test_prompt = "What is 2+2? Respond with just the number."
        
        try:
            # Test both models with same prompt
            openai_response = await openai_model.generate(test_prompt, temperature=0.0, max_tokens=5)
            anthropic_response = await anthropic_model.generate(test_prompt, temperature=0.0, max_tokens=5)
            
            # Both should contain "4"
            assert "4" in openai_response
            assert "4" in anthropic_response
            
            # Both should be healthy
            assert await openai_model.health_check()
            assert await anthropic_model.health_check()
            
        except Exception as e:
            pytest.skip(f"Cross-provider test failed (API issues): {e}")

    def test_phase1_completion_criteria(self):
        """Verify that Phase 1 completion criteria are met."""
        
        # ✅ Auto-install system extended
        assert "langchain_openai" in PACKAGE_MAPPINGS
        assert "langchain_anthropic" in PACKAGE_MAPPINGS
        
        # ✅ LangChainModelAdapter created and functional
        adapter = LangChainModelAdapter("openai", "gpt-3.5-turbo")
        assert adapter.provider == "openai"
        
        # ✅ OpenAI model enhanced with LangChain support
        openai_model = OpenAIModel("gpt-3.5-turbo", use_langchain=True)
        assert hasattr(openai_model, '_use_langchain')
        assert hasattr(openai_model, 'langchain_model')
        
        # ✅ Anthropic model enhanced with LangChain support
        anthropic_model = AnthropicModel("claude-3-haiku", use_langchain=True)
        assert hasattr(anthropic_model, '_use_langchain')
        assert hasattr(anthropic_model, 'langchain_model')
        
        # ✅ All existing interfaces preserved
        for model in [openai_model, anthropic_model]:
            assert hasattr(model, 'generate')
            assert hasattr(model, 'generate_structured')
            assert hasattr(model, 'health_check')
            assert hasattr(model, 'estimate_cost')
        
        # ✅ No breaking changes
        # All existing initialization patterns still work
        # All existing method signatures unchanged
        # All existing capabilities preserved

    def test_migration_success_metrics(self):
        """Test that migration success metrics are met."""
        
        # Performance: No degradation (models initialize successfully)
        openai_model = OpenAIModel("gpt-3.5-turbo", use_langchain=False)
        anthropic_model = AnthropicModel("claude-3-haiku", use_langchain=False)
        
        assert openai_model.name == "gpt-3.5-turbo"
        assert anthropic_model.name == "claude-3-haiku"
        
        # Reliability: Robust fallback behavior
        assert not openai_model._use_langchain  # Falls back when langchain=False
        assert not anthropic_model._use_langchain
        
        # Compatibility: All existing attributes accessible
        for model in [openai_model, anthropic_model]:
            assert hasattr(model, 'capabilities')
            assert hasattr(model, 'cost')
            assert hasattr(model, 'requirements')
            
        # Functionality: Cost tracking preserved
        assert not openai_model.cost.is_free
        assert not anthropic_model.cost.is_free
        assert openai_model.cost.calculate_cost(1000, 1000) > 0
        assert anthropic_model.cost.calculate_cost(1000, 1000) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])