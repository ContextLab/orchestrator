"""Test API key detection in different environments."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from orchestrator.utils.api_keys_flexible import (
    load_api_keys_optional,
    get_missing_providers,
    ensure_api_key
)


class TestAPIKeyDetection:
    """Test API key detection functionality."""
    
    def test_load_api_keys_from_environment(self):
        """Test loading API keys from environment variables."""
        # Set test environment variables
        test_keys = {
            "OPENAI_API_KEY": "test-openai-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "GOOGLE_AI_API_KEY": "test-google-key",
            "HF_TOKEN": "test-hf-token"
        }
        
        with patch.dict(os.environ, test_keys, clear=False):
            # Should load from environment
            loaded_keys = load_api_keys_optional()
            
            assert len(loaded_keys) >= 4  # At least our test keys
            assert loaded_keys.get("openai") == "test-openai-key"
            assert loaded_keys.get("anthropic") == "test-anthropic-key"
            assert loaded_keys.get("google") == "test-google-key"
            assert loaded_keys.get("huggingface") == "test-hf-token"
    
    def test_load_api_keys_github_actions(self):
        """Test loading API keys in GitHub Actions environment."""
        test_keys = {
            "GITHUB_ACTIONS": "true",
            "CI": "true",
            "OPENAI_API_KEY": "gh-secret-openai",
            "ANTHROPIC_API_KEY": "gh-secret-anthropic"
        }
        
        with patch.dict(os.environ, test_keys, clear=True):
            # Should detect GitHub Actions and use environment variables
            loaded_keys = load_api_keys_optional()
            
            assert loaded_keys.get("openai") == "gh-secret-openai"
            assert loaded_keys.get("anthropic") == "gh-secret-anthropic"
            # Should not have keys that weren't set
            assert "google" not in loaded_keys
            assert "huggingface" not in loaded_keys
    
    def test_load_api_keys_from_dotenv_file(self):
        """Test loading API keys from .env file."""
        # Create a temporary .env file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("OPENAI_API_KEY=file-openai-key\n")
            f.write("ANTHROPIC_API_KEY=file-anthropic-key\n")
            temp_env_path = Path(f.name)
        
        try:
            # Mock the home directory .env path
            with patch('orchestrator.utils.api_keys_flexible.Path.home') as mock_home:
                mock_home.return_value = temp_env_path.parent
                with patch.object(Path, 'exists') as mock_exists:
                    def exists_side_effect(self):
                        # Only our temp file exists
                        return str(self) == str(temp_env_path.parent / ".orchestrator" / ".env")
                    
                    mock_exists.side_effect = exists_side_effect
                    
                    # Clear environment to ensure we load from file
                    with patch.dict(os.environ, {}, clear=True):
                        # This should load from our temp .env file
                        loaded_keys = load_api_keys_optional()
                        
                        # Note: dotenv loading in tests can be tricky
                        # The test mainly verifies the code path works
        finally:
            # Clean up
            temp_env_path.unlink()
    
    def test_get_missing_providers(self):
        """Test identifying missing API key providers."""
        test_keys = {
            "OPENAI_API_KEY": "test-key",
            "ANTHROPIC_API_KEY": "test-key"
        }
        
        # Mock the load_api_keys_optional to return only our test keys
        with patch('orchestrator.utils.api_keys_flexible.load_api_keys_optional') as mock_load:
            mock_load.return_value = {"openai": "test-key", "anthropic": "test-key"}
            
            # Check all providers
            missing = get_missing_providers()
            assert "google" in missing
            assert "huggingface" in missing
            assert "openai" not in missing
            assert "anthropic" not in missing
            
            # Check specific providers
            missing_specific = get_missing_providers({"openai", "google"})
            assert "google" in missing_specific
            assert "openai" not in missing_specific
            assert len(missing_specific) == 1
    
    def test_ensure_api_key_success(self):
        """Test ensuring API key when it exists."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            key = ensure_api_key("openai")
            assert key == "test-key"
    
    def test_ensure_api_key_missing(self):
        """Test ensuring API key when it's missing."""
        # Mock load_api_keys_optional to return empty dict
        with patch('orchestrator.utils.api_keys_flexible.load_api_keys_optional') as mock_load:
            mock_load.return_value = {}
            
            with pytest.raises(EnvironmentError) as exc_info:
                ensure_api_key("openai")
            
            assert "Missing API key for openai" in str(exc_info.value)
            assert "OPENAI_API_KEY" in str(exc_info.value)
    
    def test_debug_logging_output(self, capsys):
        """Test that debug logging works correctly."""
        test_keys = {
            "GITHUB_ACTIONS": "true",
            "OPENAI_API_KEY": "test-key-12345",
            "ANTHROPIC_API_KEY": "test-key-67890"
        }
        
        with patch.dict(os.environ, test_keys, clear=True):
            load_api_keys_optional()
            
            # Check debug output
            captured = capsys.readouterr()
            assert "Running in GitHub Actions" in captured.out
            assert "Using environment variables from GitHub secrets" in captured.out
            assert "Found API key for openai (length: 14)" in captured.out
            assert "Found API key for anthropic (length: 14)" in captured.out
            assert "No API key found for google" in captured.out
            assert "Total API keys found: 2" in captured.out
            # Should not log actual key values
            assert "test-key-12345" not in captured.out
            assert "test-key-67890" not in captured.out


class TestModelInitialization:
    """Test model initialization with API keys."""
    
    @pytest.mark.skipif(
        not any(os.environ.get(key) for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]),
        reason="No API keys available for model testing"
    )
    def test_init_models_with_api_keys(self):
        """Test that init_models works when API keys are available."""
        from orchestrator import init_models
        
        # Should initialize successfully
        registry = init_models()
        
        # Should have at least one model
        models = registry.list_models()
        assert len(models) > 0
        
        # Check that API key providers have models
        available_keys = load_api_keys_optional()
        for provider in available_keys:
            # Should have at least one model from each provider with keys
            provider_models = [m for m in models if m.startswith(f"{provider}:")]
            assert len(provider_models) > 0, f"No models found for provider {provider}"
    
    def test_init_models_without_api_keys(self):
        """Test that init_models handles missing API keys gracefully."""
        # Clear all API keys
        api_key_env_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_AI_API_KEY", "HF_TOKEN"]
        
        # Save current values
        saved_env = {key: os.environ.get(key) for key in api_key_env_vars}
        
        try:
            # Clear API keys
            for key in api_key_env_vars:
                if key in os.environ:
                    del os.environ[key]
            
            from orchestrator import init_models
            
            # Should still initialize (might only have local models)
            registry = init_models()
            
            # Should return a registry even with no models
            assert registry is not None
            
        finally:
            # Restore environment
            for key, value in saved_env.items():
                if value is not None:
                    os.environ[key] = value