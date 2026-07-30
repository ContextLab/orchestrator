"""Test API key detection in different environments using real functionality."""

import os
import pytest
from pathlib import Path
import tempfile
import shutil

from orchestrator.utils.api_keys_flexible import (
    load_api_keys_optional,
    get_missing_providers,
    ensure_api_key
)


class TestAPIKeyDetection:
    """Test API key detection functionality with real operations."""
    
    def test_load_api_keys_from_real_environment(self):
        """Test loading API keys from actual environment variables."""
        # Save current environment
        saved_keys = {}
        key_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_AI_API_KEY", "HF_TOKEN"]
        for var in key_vars:
            saved_keys[var] = os.environ.get(var)
        
        try:
            # Set real test environment variables
            os.environ["OPENAI_API_KEY"] = "test-openai-key-real"
            os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key-real"
            os.environ["GOOGLE_AI_API_KEY"] = "test-google-key-real"
            os.environ["HF_TOKEN"] = "test-hf-token-real"
            
            # Load keys - this actually reads from environment
            loaded_keys = load_api_keys_optional()
            
            assert len(loaded_keys) >= 4
            assert loaded_keys.get("openai") == "test-openai-key-real"
            assert loaded_keys.get("anthropic") == "test-anthropic-key-real"
            assert loaded_keys.get("google") == "test-google-key-real"
            assert loaded_keys.get("huggingface") == "test-hf-token-real"
            
        finally:
            # Restore original environment
            for var, value in saved_keys.items():
                if value is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = value
    
    def test_environment_credentials_win_without_a_ci_special_case(self, tmp_path, monkeypatch):
        """Credentials in the environment take precedence over the .env file.

        ``load_api_keys_optional`` used to branch on ``GITHUB_ACTIONS`` and skip
        the file entirely when set. That branch was load-bearing for nothing:
        ``load_dotenv`` never overrides an already-set variable, so injected
        secrets already win. It only served to make CI behave differently from
        a developer machine, which is how an eager-credential-read defect went
        unnoticed in CI. It is gone; this pins the behaviour that replaced it.
        """
        home = tmp_path / "home"
        (home / ".orchestrator").mkdir(parents=True)
        (home / ".orchestrator" / ".env").write_text(
            "OPENAI_API_KEY=from-file\nGOOGLE_AI_API_KEY=google-from-file\n"
        )
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

        for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_AI_API_KEY", "HF_TOKEN"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("OPENAI_API_KEY", "gh-secret-openai-real")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "gh-secret-anthropic-real")

        loaded_keys = load_api_keys_optional()

        # Injected secrets win over the file, with or without GITHUB_ACTIONS.
        assert loaded_keys.get("openai") == "gh-secret-openai-real"
        assert loaded_keys.get("anthropic") == "gh-secret-anthropic-real"
        # A provider only present in the file is still discovered.
        assert loaded_keys.get("google") == "google-from-file"
        # A provider configured nowhere is absent.
        assert "huggingface" not in loaded_keys

    def test_load_api_keys_from_real_dotenv_file(self):
        """Test loading API keys from actual .env file."""
        # Create a temporary directory to simulate home
        with tempfile.TemporaryDirectory() as temp_home:
            # Create .orchestrator directory
            orchestrator_dir = Path(temp_home) / ".orchestrator"
            orchestrator_dir.mkdir(parents=True)
            
            # Create real .env file
            env_file = orchestrator_dir / ".env"
            with open(env_file, 'w') as f:
                f.write("OPENAI_API_KEY=file-openai-key-real\n")
                f.write("ANTHROPIC_API_KEY=file-anthropic-key-real\n")
                f.write("GOOGLE_AI_API_KEY=file-google-key-real\n")
            
            # Save current environment
            saved_home = os.environ.get("HOME")
            saved_keys = {}
            key_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_AI_API_KEY", "HF_TOKEN"]
            for var in key_vars:
                saved_keys[var] = os.environ.get(var)
                os.environ.pop(var, None)  # Clear keys
            
            try:
                # Temporarily change HOME to our test directory
                os.environ["HOME"] = temp_home
                
                # Load keys - this should find our .env file
                loaded_keys = load_api_keys_optional()
                
                # Verify keys were loaded from file
                assert loaded_keys.get("openai") == "file-openai-key-real"
                assert loaded_keys.get("anthropic") == "file-anthropic-key-real"
                assert loaded_keys.get("google") == "file-google-key-real"
                
            finally:
                # Restore environment
                if saved_home:
                    os.environ["HOME"] = saved_home
                else:
                    os.environ.pop("HOME", None)
                    
                for var, value in saved_keys.items():
                    if value is not None:
                        os.environ[var] = value
    
    def test_get_missing_providers_real(self):
        """Test identifying missing API key providers with real environment."""
        # Get current state of providers
        available_keys = load_api_keys_optional()
        all_providers = {"openai", "anthropic", "google", "huggingface"}
        
        # Check all providers - this actually loads from environment or .env file
        missing = get_missing_providers()
        
        # The missing providers should be exactly those not in available_keys
        expected_missing = all_providers - set(available_keys.keys())
        assert missing == expected_missing
        
        # For providers we have keys for, they should not be missing
        for provider in available_keys:
            assert provider not in missing
        
        # Check specific providers
        test_providers = {"openai", "google"}
        missing_specific = get_missing_providers(test_providers)
        
        # The missing ones should be those in test_providers but not in available_keys
        expected_missing_specific = test_providers - set(available_keys.keys())
        assert missing_specific == expected_missing_specific
        
        # If we have all API keys, missing should be empty
        if len(available_keys) == 4:
            assert len(missing) == 0
            assert len(get_missing_providers({"openai"})) == 0
    
    def test_ensure_api_key_success_real(self):
        """Test ensuring API key when it exists in real environment."""
        # Save current key
        saved_key = os.environ.get("OPENAI_API_KEY")
        
        try:
            # Set real environment variable
            os.environ["OPENAI_API_KEY"] = "test-key-ensure"
            
            # This actually reads from environment
            key = ensure_api_key("openai")
            assert key == "test-key-ensure"
            
        finally:
            # Restore
            if saved_key:
                os.environ["OPENAI_API_KEY"] = saved_key
            else:
                os.environ.pop("OPENAI_API_KEY", None)
    
    def test_ensure_api_key_missing_real(self):
        """Test ensuring API key when it's missing from real environment."""
        # First check what keys are available
        available_keys = load_api_keys_optional()
        
        # Find a provider that doesn't have a key
        test_providers = ["openai", "anthropic", "google", "huggingface"]
        missing_provider = None
        
        for provider in test_providers:
            if provider not in available_keys:
                missing_provider = provider
                break
        
        if missing_provider:
            # Test with a provider that's actually missing
            with pytest.raises(EnvironmentError) as exc_info:
                ensure_api_key(missing_provider)
            
            assert f"Missing API key for {missing_provider}" in str(exc_info.value)
            
            # Map provider to env var name
            env_var_map = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "google": "GOOGLE_AI_API_KEY",
                "huggingface": "HF_TOKEN"
            }
            expected_env_var = env_var_map[missing_provider]
            assert expected_env_var in str(exc_info.value)
        else:
            # All keys are present, test with a fake provider
            with pytest.raises(EnvironmentError) as exc_info:
                ensure_api_key("fake_provider")
            
            assert "Missing API key for fake_provider" in str(exc_info.value)
            assert "FAKE_PROVIDER_API_KEY" in str(exc_info.value)
    
    def test_loading_credentials_prints_nothing_and_leaks_nothing(
        self, tmp_path, monkeypatch, capsys, caplog
    ):
        """Credential discovery is silent on stdout and never echoes a key.

        It used to print a running commentary -- which provider files it read,
        which keys it found and how long each was -- on every run. Key lengths
        are themselves a disclosure, and the commentary was a large part of the
        CLI's default noise. What remains goes to the logger at DEBUG.
        """
        import logging

        home = tmp_path / "home"
        (home / ".orchestrator").mkdir(parents=True)
        (home / ".orchestrator" / ".env").write_text("HF_TOKEN=hf-secret-value\n")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

        for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_AI_API_KEY", "HF_TOKEN"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")

        with caplog.at_level(logging.DEBUG, logger="orchestrator.utils.api_keys_flexible"):
            loaded = load_api_keys_optional()

        assert loaded["openai"] == "test-key-12345"
        assert loaded["huggingface"] == "hf-secret-value"

        captured = capsys.readouterr()
        assert captured.out == "", f"credential loading printed to stdout: {captured.out!r}"

        logged = captured.out + captured.err + caplog.text
        # Neither the values nor their lengths.
        assert "test-key-12345" not in logged
        assert "hf-secret-value" not in logged
        assert "length" not in logged.lower()
        # The providers that were found are still traceable at DEBUG.
        assert "openai" in caplog.text


class TestModelInitialization:
    """Test model initialization with real API keys."""
    
    def test_init_models_with_real_api_keys(self):
        """Test that init_models works with actual API keys from environment."""
        from orchestrator import init_models
        
        # Initialize with real API keys from environment
        registry = init_models()
        
        # Should have at least one model
        models = registry.list_models()
        assert len(models) > 0
        
        # Check that API key providers have models
        available_keys = load_api_keys_optional()
        for provider in available_keys:
            # Should have at least one model from each provider with keys
            provider_models = [m for m in models if provider in m]
            assert len(provider_models) > 0, f"No models found for provider {provider}"
    
    def test_init_models_without_api_keys_real(self):
        """Test that init_models handles missing API keys gracefully in real environment."""
        # Save current API keys
        api_key_env_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_AI_API_KEY", "HF_TOKEN"]
        saved_env = {key: os.environ.get(key) for key in api_key_env_vars}
        
        try:
            # Clear API keys from real environment
            for key in api_key_env_vars:
                os.environ.pop(key, None)
            
            from orchestrator import init_models
            
            # Should still initialize (might only have local models like Ollama or HuggingFace)
            registry = init_models()
            
            # Should return a registry even with no API-based models
            assert registry is not None
            
            # May still have some models (Ollama, HuggingFace without token)
            models = registry.list_models()
            print(f"Models available without API keys: {models}")
            
        finally:
            # Restore real environment
            for key, value in saved_env.items():
                if value is not None:
                    os.environ[key] = value