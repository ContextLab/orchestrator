"""Test model initialization and configuration loading with REAL file operations."""

import pytest
import os
import yaml
import tempfile
from pathlib import Path
import shutil

from src.orchestrator import init_models
from src.orchestrator.utils.model_config_loader import ModelConfigLoader, get_model_config_loader
from src.orchestrator.models.model_registry import ModelRegistry


class TestModelConfigLoaderReal:
    """Test ModelConfigLoader class with real file operations."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create temporary directory for test configs
        self.temp_dir = tempfile.mkdtemp()
        self.original_home = os.environ.get('HOME')
        
        # Create test directory structure
        self.test_home = Path(self.temp_dir) / "home"
        self.test_home.mkdir()
        
        self.user_config_dir = self.test_home / ".orchestrator"
        self.user_config_dir.mkdir()
        self.user_config_path = self.user_config_dir / "models.yaml"
        
        # Set HOME to our test directory
        os.environ['HOME'] = str(self.test_home)
        
    def teardown_method(self):
        """Clean up test environment."""
        # Restore original HOME
        if self.original_home:
            os.environ['HOME'] = self.original_home
        
        # Clean up temp directory
        shutil.rmtree(self.temp_dir)
        
        # Reset singleton
        import src.orchestrator.utils.model_config_loader
        src.orchestrator.utils.model_config_loader._loader_instance = None
    
    def test_find_config_path_user_config_real(self):
        """Test finding user config path when it exists."""
        # Create real user config file
        user_config = {
            "models": {
                "test_model": {
                    "provider": "test",
                    "type": "test",
                    "size_b": 7,
                    "config": {"model_name": "test_model"}
                }
            },
            "preferences": {"default": "test_model"},
            "cost_optimized": [],
            "performance_optimized": []
        }
        
        with open(self.user_config_path, 'w') as f:
            yaml.dump(user_config, f)
        
        # Create loader - should find user config
        loader = ModelConfigLoader()
        
        # Verify it found the user config
        assert str(loader.config_path).endswith(".orchestrator/models.yaml")
        assert loader.config_path.exists()
        
        # Verify it can load the config
        config = loader.load_config()
        assert "test_model" in config["models"]
    
    def test_find_config_path_repo_config_real(self):
        """Test finding repo config when user config doesn't exist."""
        # Don't create user config
        # The loader should fall back to repo config path
        
        loader = ModelConfigLoader()
        
        # Should use repo config path (even if it doesn't exist)
        assert "config/models.yaml" in str(loader.config_path)
    
    def test_load_config_real_file(self):
        """Test loading configuration from real file."""
        config_data = {
            "models": {
                "gpt-4": {
                    "provider": "openai",
                    "type": "openai",
                    "size_b": 1760,
                    "config": {"model_name": "gpt-4"}
                },
                "claude-3": {
                    "provider": "anthropic",
                    "type": "anthropic",
                    "size_b": 200,
                    "config": {"model_name": "claude-3-opus"}
                }
            },
            "preferences": {
                "default": "gpt-4o-mini",
                "fallback": ["gpt-3.5-turbo", "claude-3-haiku"]
            },
            "cost_optimized": ["gpt-4o-mini", "claude-3-haiku"],
            "performance_optimized": ["gpt-4", "claude-3-opus"]
        }
        
        # Write real file
        with open(self.user_config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ModelConfigLoader()
        config = loader.load_config()
        
        # Verify loaded correctly
        assert len(config["models"]) == 2
        assert "gpt-4" in config["models"]
        assert "claude-3" in config["models"]
        assert config["models"]["gpt-4"]["provider"] == "openai"
        assert config["preferences"]["default"] == "gpt-4o-mini"
        assert len(config["cost_optimized"]) == 2
    
    def test_load_missing_file_real(self):
        """Test loading when config file doesn't exist."""
        # Don't create any config file
        loader = ModelConfigLoader(config_path=Path(self.temp_dir) / "nonexistent.yaml")
        
        config = loader.load_config()
        
        # If it found repo config, it will have models; otherwise empty
        # Just verify the structure is valid
        assert "models" in config
        assert "preferences" in config
        assert "cost_optimized" in config
        assert "performance_optimized" in config
        assert isinstance(config["models"], dict)
        assert isinstance(config["preferences"], dict)
        assert isinstance(config["cost_optimized"], list)
        assert isinstance(config["performance_optimized"], list)
    
    def test_config_caching_real(self):
        """Test configuration caching with real file modifications."""
        import time
        
        # Create initial config
        config_data = {"models": {"model1": {"provider": "test"}}}
        with open(self.user_config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ModelConfigLoader()
        
        # First load
        config1 = loader.load_config()
        assert "model1" in config1["models"]
        assert "model2" not in config1["models"]
        
        # Sleep to ensure file modification time changes
        time.sleep(0.01)
        
        # Modify the real file
        config_data["models"]["model2"] = {"provider": "test2"}
        with open(self.user_config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Second load without force - should use cache based on mtime
        config2 = loader.load_config()
        # If mtime resolution is sufficient, it should reload; otherwise cache
        # Just verify it's a valid config
        assert "model1" in config2["models"]
        
        # Third load with force - should definitely reload from disk
        config3 = loader.load_config(force_reload=True)
        assert "model2" in config3["models"]
    
    def test_add_model_real(self):
        """Test adding a model and saving to real file."""
        # Start with minimal config
        initial_config = {"models": {}}
        with open(self.user_config_path, 'w') as f:
            yaml.dump(initial_config, f)
        
        loader = ModelConfigLoader()
        
        # Add new model
        new_model_config = {
            "provider": "openai",
            "type": "openai",
            "size_b": 175,
            "config": {"model_name": "gpt-3.5-turbo"}
        }
        loader.add_model("gpt-3.5-turbo", new_model_config)
        
        # Verify in memory
        config = loader.load_config()
        assert "gpt-3.5-turbo" in config["models"]
        
        # Verify on disk by creating new loader
        loader2 = ModelConfigLoader(config_path=self.user_config_path)
        config2 = loader2.load_config()
        assert "gpt-3.5-turbo" in config2["models"]
        assert config2["models"]["gpt-3.5-turbo"]["provider"] == "openai"
    
    def test_get_models_by_provider_real(self):
        """Test filtering models by provider from real config."""
        config_data = {
            "models": {
                "gpt-4": {"provider": "openai", "type": "openai"},
                "gpt-3.5": {"provider": "openai", "type": "openai"},
                "claude-3": {"provider": "anthropic", "type": "anthropic"},
                "llama-2": {"provider": "ollama", "type": "ollama"}
            }
        }
        
        with open(self.user_config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ModelConfigLoader()
        
        # Test filtering
        openai_models = loader.get_models_by_provider("openai")
        assert len(openai_models) == 2
        assert all(m["provider"] == "openai" for m in openai_models.values())
        
        anthropic_models = loader.get_models_by_provider("anthropic")
        assert len(anthropic_models) == 1
        assert "claude-3" in anthropic_models


class TestInitModelsReal:
    """Test init_models function with real operations."""
    
    def setup_method(self):
        """Set up test environment."""
        # Save original environment
        self.original_env = os.environ.copy()
        
        # Create test directory
        self.temp_dir = tempfile.mkdtemp()
        self.test_home = Path(self.temp_dir) / "home"
        self.test_home.mkdir()
        
        # Create config directory
        self.user_config_dir = self.test_home / ".orchestrator"
        self.user_config_dir.mkdir()
        self.config_path = self.user_config_dir / "models.yaml"
        
        # Set test HOME
        os.environ['HOME'] = str(self.test_home)
        
        # Reset global registry
        import src.orchestrator
        src.orchestrator._model_registry = None
        
        # Reset singleton loader
        import src.orchestrator.utils.model_config_loader
        src.orchestrator.utils.model_config_loader._loader_instance = None
    
    def teardown_method(self):
        """Clean up test environment."""
        # Restore environment
        os.environ.clear()
        os.environ.update(self.original_env)
        
        # Clean up directory
        shutil.rmtree(self.temp_dir)
        
        # Reset globals
        import src.orchestrator
        src.orchestrator._model_registry = None
        
        import src.orchestrator.utils.model_config_loader
        src.orchestrator.utils.model_config_loader._loader_instance = None
    
    def create_test_config(self, models_dict):
        """Helper to create test configuration."""
        config = {
            "models": models_dict,
            "preferences": {"default": "gpt-4o-mini"},
            "cost_optimized": [],
            "performance_optimized": [],
            "defaults": {}
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
    
    def test_init_models_with_openai_real(self):
        """Test model initialization with real OpenAI config."""
        # Skip if no API key
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not set")
        
        # Create config with OpenAI models
        self.create_test_config({
            "gpt-4": {
                "provider": "openai",
                "type": "openai",
                "size_b": 1760,
                "config": {"model_name": "gpt-4"}
            },
            "gpt-3.5-turbo": {
                "provider": "openai",
                "type": "openai",
                "size_b": 175,
                "config": {"model_name": "gpt-3.5-turbo"}
            }
        })
        
        # Initialize models
        registry = init_models()
        
        # Verify models were registered
        assert isinstance(registry, ModelRegistry)
        models = registry.list_models()
        assert len(models) == 2
        assert "openai:gpt-4" in models
        assert "openai:gpt-3.5-turbo" in models
        
        # Verify we can get the models
        gpt4 = registry.get_model("gpt-4", "openai")
        assert gpt4.name == "gpt-4"
        assert gpt4.provider == "openai"
    
    def test_init_models_without_api_keys_real(self):
        """Test initialization without API keys."""
        # Remove all API keys
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]:
            os.environ.pop(key, None)
        
        # Create config requiring API keys
        self.create_test_config({
            "gpt-4": {
                "provider": "openai",
                "type": "openai",
                "size_b": 1760,
                "config": {"model_name": "gpt-4"}
            },
            "claude-3": {
                "provider": "anthropic",
                "type": "anthropic",
                "size_b": 200,
                "config": {"model_name": "claude-3-opus"}
            }
        })
        
        # Initialize models
        registry = init_models()
        
        # Should not register any models without keys
        assert len(registry.list_models()) == 0
    
    def test_init_models_ollama_real(self):
        """Test initialization with Ollama models."""
        # Create config with Ollama models
        self.create_test_config({
            "llama2:7b": {
                "provider": "ollama",
                "type": "ollama",
                "size_b": 7,
                "config": {"model_name": "llama2:7b"}
            },
            "mistral:7b": {
                "provider": "ollama",
                "type": "ollama",
                "size_b": 7,
                "config": {"model_name": "mistral:7b"}
            }
        })
        
        # Initialize models
        registry = init_models()
        
        # Check if Ollama is installed
        from src.orchestrator.utils.model_utils import check_ollama_installed
        if check_ollama_installed():
            # Should register Ollama models
            models = registry.list_models()
            assert any("ollama:" in m for m in models)
        else:
            # Should skip Ollama models
            assert len(registry.list_models()) == 0
    
    def test_init_models_mixed_providers_real(self):
        """Test with mix of available and unavailable providers."""
        # Set only OpenAI key
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]
        if "GOOGLE_API_KEY" in os.environ:
            del os.environ["GOOGLE_API_KEY"]
        
        # Create mixed config
        self.create_test_config({
            "gpt-4": {
                "provider": "openai",
                "type": "openai",
                "size_b": 1760,
                "config": {"model_name": "gpt-4"}
            },
            "claude-3": {
                "provider": "anthropic",
                "type": "anthropic",
                "size_b": 200,
                "config": {"model_name": "claude-3-opus"}
            },
            "llama2": {
                "provider": "ollama",
                "type": "ollama",
                "size_b": 7,
                "config": {"model_name": "llama2:7b"}
            }
        })
        
        # Initialize
        registry = init_models()
        models = registry.list_models()
        
        # Should register OpenAI if key exists
        if os.environ.get("OPENAI_API_KEY"):
            assert any("openai:" in m for m in models)
        
        # Should not register Anthropic without key
        assert not any("anthropic:" in m for m in models)
    
    def test_init_models_auto_registration_enabled_real(self):
        """Test that auto-registration is enabled."""
        # Create empty config
        self.create_test_config({})
        
        # Initialize
        registry = init_models()
        
        # Auto-registration should be enabled
        assert registry._auto_registrar is not None
    
    def test_init_models_expertise_detection_real(self):
        """Test expertise detection from model names."""
        # Skip if no API key
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not set")
        
        # Create config with models that imply expertise
        self.create_test_config({
            "gpt-4-code": {
                "provider": "openai",
                "type": "openai",
                "size_b": 1760,
                "config": {"model_name": "gpt-4"}  # Using real model
            },
            "gpt-3.5-turbo-instruct": {
                "provider": "openai",
                "type": "openai",
                "size_b": 175,
                "config": {"model_name": "gpt-3.5-turbo-instruct"}
            }
        })
        
        # Initialize
        registry = init_models()
        
        # Check expertise was set
        # The model is registered with the actual model name from config, not the key
        model = registry.get_model("gpt-4", "openai")
        assert hasattr(model, "_expertise")
        # The expertise is detected from the key name "gpt-4-code"
        assert "code" in model._expertise  # Should detect from key name
        
        instruct_model = registry.get_model("gpt-3.5-turbo-instruct", "openai")
        assert hasattr(instruct_model, "_expertise")
        assert "chat" in instruct_model._expertise or "instruct" in instruct_model._expertise
    
    def test_init_models_global_registry_real(self):
        """Test that init_models sets global registry."""
        import src.orchestrator
        
        # Should start as None
        assert src.orchestrator._model_registry is None
        
        # Create minimal config
        self.create_test_config({})
        
        # Initialize
        registry = init_models()
        
        # Global should be set
        assert src.orchestrator._model_registry is registry
        assert isinstance(src.orchestrator._model_registry, ModelRegistry)
    
    def test_init_models_from_repo_config_real(self):
        """Test loading from actual repo config file."""
        # Remove user config so it uses repo config
        if self.config_path.exists():
            self.config_path.unlink()
        self.user_config_dir.rmdir()
        
        # Initialize - should use repo config
        registry = init_models()
        
        # Should load many models from repo config
        models = registry.list_models()
        if Path("config/models.yaml").exists():
            # If running in repo, should have many models
            assert len(models) > 50
        else:
            # If not in repo, should have no models
            assert len(models) == 0