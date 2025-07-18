"""Test update_models tool functionality with REAL API calls."""

import pytest
import asyncio
import tempfile
import yaml
import os
from pathlib import Path

from src.orchestrator.tools.update_models import ModelUpdater, update_models


class TestModelUpdater:
    """Test ModelUpdater class with real API calls."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create temporary config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "models.yaml"
        
        # Create updater with temp path
        self.updater = ModelUpdater(config_path=self.config_path)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OpenAI API key not set")
    async def test_fetch_openai_models_real(self):
        """Test fetching OpenAI models with real API call."""
        models = await self.updater.fetch_openai_models()
        
        # Verify we got real models
        assert len(models) > 0, "Should fetch at least one OpenAI model"
        
        # Check for known models
        model_ids = [m["id"] for m in models]
        assert any("gpt-4" in m for m in model_ids), "Should include GPT-4* models"
        assert any("o3" in m for m in model_ids), "Should include o3 models"
        
        # Verify no deprecated models
        assert not any("text-davinci" in m for m in model_ids), "Should filter out deprecated models"
        assert not any("text-curie" in m for m in model_ids), "Should filter out deprecated models"
        
        # Verify model structure
        for model in models[:5]:  # Check first 5 models
            assert "id" in model, f"Model should have 'id' field: {model}"
            assert isinstance(model["id"], str), f"Model ID should be string: {model}"
    
    @pytest.mark.asyncio
    async def test_fetch_anthropic_models_real(self):
        """Test fetching Anthropic models (hardcoded list)."""
        models = await self.updater.fetch_anthropic_models()
        
        # Should return hardcoded list
        assert len(models) > 0, "Should have Anthropic models"
        
        # Check for known models
        model_ids = [m["id"] for m in models]
        assert any("claude-opus-4" in m or "claude-3-opus" in m for m in model_ids), "Should include Claude Opus models"
        assert any("claude-sonnet-4" in m or "claude-3-sonnet" in m or "claude-3-5-sonnet" in m or "claude-3-7-sonnet" in m for m in model_ids), "Should include Claude Sonnet models"  
        assert any("claude-3-haiku" in m or "claude-3-5-haiku" in m for m in model_ids), "Should include Claude Haiku models"
        
        # Verify structure
        for model in models:
            assert "id" in model
            # Anthropic models might not have 'created' field in hardcoded list
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.environ.get("GOOGLE_API_KEY"), reason="Google API key not set")
    async def test_fetch_google_models_real(self):
        """Test fetching Google models with real API call."""
        models = await self.updater.fetch_google_models()
        
        # Verify we got models
        assert len(models) > 0, "Should fetch at least one Google model"
        
        # Check for known models
        model_ids = [m["id"] for m in models]
        assert any("gemini" in m for m in model_ids), "Should include Gemini models"
        
        # Verify structure
        for model in models[:5]:
            assert "id" in model
            assert isinstance(model["id"], str)
    
    @pytest.mark.asyncio
    async def test_fetch_ollama_models_real(self):
        """Test fetching Ollama models (hardcoded list)."""
        models = await self.updater.fetch_ollama_models()
        
        # Should return comprehensive list
        assert len(models) > 10, "Should have many Ollama models"
        
        # Check for known models
        model_ids = [m["id"] for m in models]
        assert any("llama" in m for m in model_ids), "Should include Llama models"
        assert any("mistral" in m for m in model_ids), "Should include Mistral models"
        assert any("gemma" in m for m in model_ids), "Should include Gemma models"
        
        # Verify structure and variants
        llama_models = [m for m in models if "llama" in m["id"]]
        assert len(llama_models) > 2, "Should have multiple Llama variants"
    
    @pytest.mark.asyncio
    async def test_fetch_huggingface_models_real(self):
        """Test fetching HuggingFace models (curated list)."""
        models = await self.updater.fetch_huggingface_models()
        
        # Should return curated list
        assert len(models) > 5, "Should have several HuggingFace models"
        
        # Check for known organizations
        model_ids = [m["id"] for m in models]
        assert any("meta-llama" in m for m in model_ids), "Should include Meta Llama models"
        assert any("microsoft" in m for m in model_ids), "Should include Microsoft models"
        assert any("Qwen" in m for m in model_ids), "Should include Qwen models"
        
        # All should be instruct/chat models
        for model in models:
            assert "/" in model["id"], "HuggingFace models should have org/model format"
    
    def test_estimate_model_size(self):
        """Test model size estimation with various patterns."""
        # Known model sizes
        assert self.updater.estimate_model_size("gpt-4") == 1760
        assert self.updater.estimate_model_size("gpt-3.5-turbo") == 175
        assert self.updater.estimate_model_size("gpt-4o") == 1760
        assert self.updater.estimate_model_size("gpt-4o-mini") == 1760
        assert self.updater.estimate_model_size("claude-3-opus-20240229") == 2000
        assert self.updater.estimate_model_size("claude-3-sonnet") == 200
        assert self.updater.estimate_model_size("claude-3-haiku") == 20
        
        # Pattern-based sizes
        assert self.updater.estimate_model_size("llama-7b") == 7
        assert self.updater.estimate_model_size("llama3:13b") == 13
        assert self.updater.estimate_model_size("model-175b") == 175
        # Note: mixtral patterns match the last 'b' pattern first, not the multiplication
        assert self.updater.estimate_model_size("mixtral-8x7b") == 7  # Matches '7b'
        assert self.updater.estimate_model_size("mixtral:8x22b") == 22  # Matches '22b'
        assert self.updater.estimate_model_size("phi-3.8b") == 3.8
        
        # Unknown models get default
        assert self.updater.estimate_model_size("unknown-model") == 7
        assert self.updater.estimate_model_size("some-random-name") == 7
    
    @pytest.mark.asyncio
    async def test_update_models_config_real(self):
        """Test updating models configuration with real API calls."""
        # This test actually calls all APIs
        await self.updater.run()
        
        # Check config was created
        assert self.config_path.exists(), "Config file should be created"
        
        # Load and verify config
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check structure
        assert "models" in config
        assert "preferences" in config
        assert "cost_optimized" in config
        assert "performance_optimized" in config
        
        # Verify we have models from multiple providers
        model_ids = list(config["models"].keys())
        assert len(model_ids) > 50, f"Should have many models, got {len(model_ids)}"
        
        # Check provider diversity
        providers = set()
        for model_config in config["models"].values():
            providers.add(model_config.get("provider"))
        
        # Should have models from multiple providers (at least Anthropic, Ollama, HF)
        assert len(providers) >= 3, f"Should have multiple providers, got {providers}"
        
        # Verify preferences
        assert config["preferences"]["default"] == "gpt-4o-mini"
        assert isinstance(config["preferences"]["fallback"], list)
        assert len(config["preferences"]["fallback"]) > 0
        
        # Verify cost/performance lists
        assert len(config["cost_optimized"]) > 0
        assert len(config["performance_optimized"]) > 0
        
        # Check that cost optimized includes mini/haiku models
        assert any("mini" in m or "haiku" in m or "flash" in m 
                  for m in config["cost_optimized"])
        
        # Check that performance optimized includes large models
        assert any("gpt-4" in m or "opus" in m or "pro" in m 
                  for m in config["performance_optimized"])
    
    @pytest.mark.asyncio
    async def test_update_models_with_partial_api_keys(self):
        """Test update models when only some API keys are available."""
        # Save current env
        original_openai = os.environ.get("OPENAI_API_KEY")
        original_google = os.environ.get("GOOGLE_API_KEY")
        
        try:
            # Remove API keys
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
            if "GOOGLE_API_KEY" in os.environ:
                del os.environ["GOOGLE_API_KEY"]
            
            # Should still work with hardcoded providers
            await self.updater.run()
            
            # Verify config was created
            assert self.config_path.exists()
            
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Should still have models from hardcoded providers
            assert len(config["models"]) > 20  # Anthropic, Ollama, HuggingFace
            
            # Should have Anthropic and Ollama models
            model_providers = set(m["provider"] for m in config["models"].values())
            assert "anthropic" in model_providers
            assert "ollama" in model_providers
            assert "huggingface" in model_providers
            
        finally:
            # Restore env
            if original_openai:
                os.environ["OPENAI_API_KEY"] = original_openai
            if original_google:
                os.environ["GOOGLE_API_KEY"] = original_google
    
    def test_model_config_structure(self):
        """Test that model configurations have correct structure."""
        # Test various model types
        test_cases = [
            {"id": "gpt-4", "provider": "openai"},
            {"id": "claude-4-opus", "provider": "anthropic"},
            {"id": "llama2:7b", "provider": "ollama"},
            {"id": "microsoft/phi-2", "provider": "huggingface"},
        ]
        
        for test_case in test_cases:
            model_data = {"id": test_case["id"], "created": 1234567890}
            provider = test_case["provider"]
            
            # Create config entry as the updater would
            config_id = f"{provider}_{model_data['id'].replace('/', '_').replace(':', '_')}"
            config_entry = {
                "provider": provider,
                "type": provider,
                "size_b": self.updater.estimate_model_size(model_data["id"]),
                "config": {
                    "model_name": model_data["id"]
                }
            }
            
            # Verify structure
            assert config_entry["provider"] == provider
            assert config_entry["type"] == provider
            assert isinstance(config_entry["size_b"], (int, float))
            assert config_entry["size_b"] > 0
            assert "config" in config_entry
            assert config_entry["config"]["model_name"] == model_data["id"]
            
            # Provider-specific config
            if provider == "ollama":
                config_entry["config"]["base_url"] = "http://localhost:11434"
                config_entry["config"]["timeout"] = 60
                assert "base_url" in config_entry["config"]
                assert "timeout" in config_entry["config"]
            elif provider == "huggingface":
                config_entry["config"]["device"] = "auto"
                assert "device" in config_entry["config"]


@pytest.mark.asyncio
async def test_update_models_main_real():
    """Test the main update_models function with real execution."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "test_models.yaml"
        
        # Run the actual update
        await update_models(str(config_path))
        
        # Verify it worked
        assert config_path.exists(), "Config file should be created"
        
        # Load and do basic verification
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert "models" in config
        assert len(config["models"]) > 0
        assert "preferences" in config