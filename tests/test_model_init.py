"""Test model initialization and configuration loading."""

import pytest
import os
import yaml
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
import shutil

from src.orchestrator import init_models
from src.orchestrator.utils.model_config_loader import ModelConfigLoader, get_model_config_loader
from src.orchestrator.models.model_registry import ModelRegistry


class TestModelConfigLoader:
    """Test ModelConfigLoader class."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create temporary directory for test configs
        self.temp_dir = tempfile.mkdtemp()
        self.user_config_dir = Path(self.temp_dir) / ".orchestrator"
        self.user_config_dir.mkdir()
        self.user_config_path = self.user_config_dir / "models.yaml"
        
        # Create repo config path
        self.repo_config_dir = Path(self.temp_dir) / "config"
        self.repo_config_dir.mkdir()
        self.repo_config_path = self.repo_config_dir / "models.yaml"
        
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_find_config_path_user_config(self):
        """Test finding user config path when it exists."""
        # Create user config
        user_config = {
            "models": {"test_model": {"provider": "test"}},
            "preferences": {"default": "test_model"}
        }
        with open(self.user_config_path, 'w') as f:
            yaml.dump(user_config, f)
        
        # Mock Path.home() to return our temp directory
        with patch('pathlib.Path.home', return_value=Path(self.temp_dir)):
            loader = ModelConfigLoader()
            # The loader should prefer user config
            assert str(loader.config_path).endswith(".orchestrator/models.yaml")
    
    def test_find_config_path_repo_config(self):
        """Test finding repo config when user config doesn't exist."""
        # Create repo config only
        repo_config = {
            "models": {"repo_model": {"provider": "test"}},
            "preferences": {"default": "repo_model"}
        }
        with open(self.repo_config_path, 'w') as f:
            yaml.dump(repo_config, f)
        
        # Mock paths
        with patch('pathlib.Path.home', return_value=Path(self.temp_dir)):
            with patch('pathlib.Path', side_effect=lambda x: Path(x) if isinstance(x, str) else x):
                with patch('src.orchestrator.utils.model_config_loader.__file__', 
                          str(self.temp_dir / "src" / "orchestrator" / "utils" / "model_config_loader.py")):
                    loader = ModelConfigLoader()
                    # Should fall back to repo config
                    assert "config/models.yaml" in str(loader.config_path)
    
    def test_find_config_path_custom(self):
        """Test using custom config path."""
        custom_path = Path(self.temp_dir) / "custom_models.yaml"
        custom_config = {"models": {}, "preferences": {}}
        with open(custom_path, 'w') as f:
            yaml.dump(custom_config, f)
        
        loader = ModelConfigLoader(config_path=custom_path)
        assert loader.config_path == custom_path
    
    def test_load_config_valid(self):
        """Test loading valid configuration."""
        config_data = {
            "models": {
                "gpt-4": {
                    "provider": "openai",
                    "type": "openai",
                    "size_b": 1760,
                    "config": {"model_name": "gpt-4"}
                }
            },
            "preferences": {
                "default": "gpt-4o-mini",
                "fallback": ["gpt-3.5-turbo", "claude-3-haiku"]
            },
            "cost_optimized": ["gpt-4o-mini", "claude-3-haiku"],
            "performance_optimized": ["gpt-4", "claude-3-opus"]
        }
        
        with open(self.repo_config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ModelConfigLoader(config_path=self.repo_config_path)
        config = loader.load_config()
        
        assert "models" in config
        assert "gpt-4" in config["models"]
        assert config["models"]["gpt-4"]["provider"] == "openai"
        assert config["preferences"]["default"] == "gpt-4o-mini"
        assert len(config["cost_optimized"]) == 2
        assert len(config["performance_optimized"]) == 2
    
    def test_load_config_missing_file(self):
        """Test loading when config file doesn't exist."""
        loader = ModelConfigLoader(config_path=Path(self.temp_dir) / "nonexistent.yaml")
        config = loader.load_config()
        
        # Should return empty but valid structure
        assert config["models"] == {}
        assert config["preferences"] == {}
        assert config["cost_optimized"] == []
        assert config["performance_optimized"] == []
    
    def test_load_config_invalid_yaml(self):
        """Test loading invalid YAML."""
        with open(self.repo_config_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        loader = ModelConfigLoader(config_path=self.repo_config_path)
        config = loader.load_config()
        
        # Should return empty structure on error
        assert config["models"] == {}
        assert config["preferences"] == {}
    
    def test_config_caching(self):
        """Test configuration caching."""
        config_data = {"models": {"model1": {"provider": "test"}}}
        with open(self.repo_config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ModelConfigLoader(config_path=self.repo_config_path)
        
        # First load
        config1 = loader.load_config()
        assert "model1" in config1["models"]
        
        # Modify file
        config_data["models"]["model2"] = {"provider": "test2"}
        with open(self.repo_config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Second load without force - should return cached
        config2 = loader.load_config()
        assert "model2" not in config2["models"]
        
        # Third load with force - should reload
        config3 = loader.load_config(force_reload=True)
        assert "model2" in config3["models"]
    
    def test_get_models_by_provider(self):
        """Test getting models by provider."""
        config_data = {
            "models": {
                "gpt-4": {"provider": "openai", "type": "openai"},
                "gpt-3.5": {"provider": "openai", "type": "openai"},
                "claude-3": {"provider": "anthropic", "type": "anthropic"},
                "llama-2": {"provider": "ollama", "type": "ollama"}
            }
        }
        with open(self.repo_config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ModelConfigLoader(config_path=self.repo_config_path)
        
        openai_models = loader.get_models_by_provider("openai")
        assert len(openai_models) == 2
        assert "gpt-4" in openai_models
        assert "gpt-3.5" in openai_models
        
        anthropic_models = loader.get_models_by_provider("anthropic")
        assert len(anthropic_models) == 1
        assert "claude-3" in anthropic_models
    
    def test_get_model_config(self):
        """Test getting specific model configuration."""
        config_data = {
            "models": {
                "test-model": {
                    "provider": "test",
                    "size_b": 7,
                    "config": {"param": "value"}
                }
            }
        }
        with open(self.repo_config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ModelConfigLoader(config_path=self.repo_config_path)
        
        model_config = loader.get_model_config("test-model")
        assert model_config is not None
        assert model_config["provider"] == "test"
        assert model_config["size_b"] == 7
        
        # Non-existent model
        assert loader.get_model_config("nonexistent") is None
    
    def test_add_model(self):
        """Test adding a new model to configuration."""
        # Start with empty config
        config_data = {"models": {}}
        with open(self.repo_config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ModelConfigLoader(config_path=self.repo_config_path)
        
        # Add new model
        new_model_config = {
            "provider": "openai",
            "type": "openai",
            "size_b": 175,
            "config": {"model_name": "gpt-3.5-turbo"}
        }
        loader.add_model("gpt-3.5-turbo", new_model_config)
        
        # Verify it was added
        config = loader.load_config()
        assert "gpt-3.5-turbo" in config["models"]
        assert config["models"]["gpt-3.5-turbo"]["provider"] == "openai"
        
        # Verify it was saved to file
        with open(self.repo_config_path, 'r') as f:
            saved_config = yaml.safe_load(f)
        assert "gpt-3.5-turbo" in saved_config["models"]
    
    def test_model_exists(self):
        """Test checking if model exists."""
        config_data = {
            "models": {
                "existing-model": {"provider": "test"}
            }
        }
        with open(self.repo_config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ModelConfigLoader(config_path=self.repo_config_path)
        
        assert loader.model_exists("existing-model") is True
        assert loader.model_exists("nonexistent-model") is False
    
    def test_get_preferences(self):
        """Test getting preferences."""
        config_data = {
            "models": {},
            "preferences": {
                "default": "gpt-4o-mini",
                "code": "gpt-4",
                "reasoning": "claude-3-opus"
            }
        }
        with open(self.repo_config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        loader = ModelConfigLoader(config_path=self.repo_config_path)
        prefs = loader.get_preferences()
        
        assert prefs["default"] == "gpt-4o-mini"
        assert prefs["code"] == "gpt-4"
        assert prefs["reasoning"] == "claude-3-opus"
    
    def test_singleton_instance(self):
        """Test singleton pattern for loader."""
        loader1 = get_model_config_loader()
        loader2 = get_model_config_loader()
        
        assert loader1 is loader2


class TestInitModels:
    """Test init_models function."""
    
    def setup_method(self):
        """Set up test environment."""
        # Save original environment variables
        self.original_env = os.environ.copy()
        
        # Create test config
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "models.yaml"
        
        # Reset global model registry
        import src.orchestrator
        src.orchestrator._model_registry = None
    
    def teardown_method(self):
        """Clean up test environment."""
        # Restore environment variables
        os.environ.clear()
        os.environ.update(self.original_env)
        
        # Clean up temp directory
        shutil.rmtree(self.temp_dir)
    
    def create_test_config(self, models_dict):
        """Helper to create test configuration."""
        config = {
            "models": models_dict,
            "preferences": {"default": "gpt-4o-mini"},
            "cost_optimized": [],
            "performance_optimized": []
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
        return self.config_path
    
    @patch('src.orchestrator.utils.model_config_loader.get_model_config_loader')
    @patch('src.orchestrator.utils.model_utils.check_ollama_installed')
    def test_init_models_basic(self, mock_ollama_check, mock_get_loader):
        """Test basic model initialization."""
        # Mock Ollama as not installed
        mock_ollama_check.return_value = False
        
        # Mock config loader
        mock_loader = MagicMock()
        mock_loader.load_config.return_value = {
            "models": {
                "gpt-4": {
                    "provider": "openai",
                    "type": "openai",
                    "size_b": 1760,
                    "config": {"model_name": "gpt-4"}
                }
            },
            "defaults": {}
        }
        mock_get_loader.return_value = mock_loader
        
        # Set API key
        os.environ["OPENAI_API_KEY"] = "test-key"
        
        # Initialize models
        registry = init_models()
        
        assert isinstance(registry, ModelRegistry)
        assert len(registry.list_models()) == 1
        assert "openai:gpt-4" in registry.list_models()
    
    @patch('src.orchestrator.utils.model_config_loader.get_model_config_loader')
    @patch('src.orchestrator.utils.model_utils.check_ollama_installed')
    def test_init_models_multiple_providers(self, mock_ollama_check, mock_get_loader):
        """Test initialization with multiple providers."""
        mock_ollama_check.return_value = True
        
        # Mock config with multiple providers
        mock_loader = MagicMock()
        mock_loader.load_config.return_value = {
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
                },
                "llama2": {
                    "provider": "ollama",
                    "type": "ollama",
                    "size_b": 7,
                    "config": {"model_name": "llama2:7b"}
                }
            },
            "defaults": {}
        }
        mock_get_loader.return_value = mock_loader
        
        # Set API keys
        os.environ["OPENAI_API_KEY"] = "test-openai-key"
        os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"
        
        registry = init_models()
        
        # Should register OpenAI and Anthropic models (and Ollama if available)
        models = registry.list_models()
        assert len(models) >= 2  # At least OpenAI and Anthropic
        assert any("openai:" in m for m in models)
        assert any("anthropic:" in m for m in models)
    
    @patch('src.orchestrator.utils.model_config_loader.get_model_config_loader')
    @patch('src.orchestrator.utils.model_utils.check_ollama_installed')
    def test_init_models_no_api_keys(self, mock_ollama_check, mock_get_loader):
        """Test initialization without API keys."""
        mock_ollama_check.return_value = False
        
        # Remove API keys
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("GOOGLE_API_KEY", None)
        
        # Mock config
        mock_loader = MagicMock()
        mock_loader.load_config.return_value = {
            "models": {
                "gpt-4": {
                    "provider": "openai",
                    "type": "openai",
                    "size_b": 1760,
                    "config": {"model_name": "gpt-4"}
                }
            },
            "defaults": {}
        }
        mock_get_loader.return_value = mock_loader
        
        registry = init_models()
        
        # Should not register any models without API keys
        assert len(registry.list_models()) == 0
    
    @patch('src.orchestrator.utils.model_config_loader.get_model_config_loader')
    @patch('src.orchestrator.utils.model_utils.check_ollama_installed')
    def test_init_models_ollama_available(self, mock_ollama_check, mock_get_loader):
        """Test initialization with Ollama available."""
        mock_ollama_check.return_value = True
        
        # Mock config with Ollama models
        mock_loader = MagicMock()
        mock_loader.load_config.return_value = {
            "models": {
                "llama2": {
                    "provider": "ollama",
                    "type": "ollama",
                    "size_b": 7,
                    "config": {"model_name": "llama2:7b"}
                },
                "mistral": {
                    "provider": "ollama",
                    "type": "ollama",
                    "size_b": 7,
                    "config": {"model_name": "mistral:7b"}
                }
            },
            "defaults": {}
        }
        mock_get_loader.return_value = mock_loader
        
        registry = init_models()
        
        # Should register Ollama models
        models = registry.list_models()
        assert len(models) == 2
        assert all("ollama:" in m for m in models)
    
    @patch('src.orchestrator.utils.model_config_loader.get_model_config_loader')
    @patch('src.orchestrator.utils.model_utils.check_ollama_installed')
    def test_init_models_huggingface(self, mock_ollama_check, mock_get_loader):
        """Test initialization with HuggingFace models."""
        mock_ollama_check.return_value = False
        
        # Mock config with HuggingFace models
        mock_loader = MagicMock()
        mock_loader.load_config.return_value = {
            "models": {
                "tiny-llama": {
                    "provider": "huggingface",
                    "type": "huggingface",
                    "size_b": 1.1,
                    "config": {"model_name": "TinyLlama/TinyLlama-1.1B"}
                }
            },
            "defaults": {}
        }
        mock_get_loader.return_value = mock_loader
        
        # Mock transformers availability
        with patch('importlib.util.find_spec') as mock_find_spec:
            mock_find_spec.return_value = MagicMock()  # Transformers is available
            
            registry = init_models()
            
            # Should register HuggingFace model
            models = registry.list_models()
            assert len(models) == 1
            assert "huggingface:" in models[0]
    
    @patch('src.orchestrator.utils.model_config_loader.get_model_config_loader')
    @patch('src.orchestrator.utils.model_utils.check_ollama_installed')
    def test_init_models_auto_registration_enabled(self, mock_ollama_check, mock_get_loader):
        """Test that auto-registration is enabled."""
        mock_ollama_check.return_value = False
        
        # Mock empty config
        mock_loader = MagicMock()
        mock_loader.load_config.return_value = {
            "models": {},
            "defaults": {}
        }
        mock_get_loader.return_value = mock_loader
        
        registry = init_models()
        
        # Auto-registration should be enabled
        assert registry._auto_registrar is not None
    
    @patch('src.orchestrator.utils.model_config_loader.get_model_config_loader')
    @patch('src.orchestrator.utils.model_utils.check_ollama_installed')
    def test_init_models_expertise_detection(self, mock_ollama_check, mock_get_loader):
        """Test expertise detection from model names."""
        mock_ollama_check.return_value = False
        
        # Mock config with models that should have expertise detected
        mock_loader = MagicMock()
        mock_loader.load_config.return_value = {
            "models": {
                "gpt-4-code": {
                    "provider": "openai",
                    "type": "openai",
                    "size_b": 1760,
                    "config": {"model_name": "gpt-4-code"}
                },
                "claude-chat": {
                    "provider": "anthropic", 
                    "type": "anthropic",
                    "size_b": 200,
                    "config": {"model_name": "claude-3-chat"}
                }
            },
            "defaults": {}
        }
        mock_get_loader.return_value = mock_loader
        
        # Set API keys
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        
        registry = init_models()
        
        # Get models and check expertise
        openai_model = registry.get_model("gpt-4-code", "openai")
        assert hasattr(openai_model, "_expertise")
        assert "code" in openai_model._expertise
        
        anthropic_model = registry.get_model("claude-3-chat", "anthropic")
        assert hasattr(anthropic_model, "_expertise")
        assert "chat" in anthropic_model._expertise
    
    @patch('src.orchestrator.utils.model_config_loader.get_model_config_loader')
    @patch('src.orchestrator.utils.model_utils.check_ollama_installed')
    def test_init_models_size_setting(self, mock_ollama_check, mock_get_loader):
        """Test that model sizes are properly set."""
        mock_ollama_check.return_value = False
        
        # Mock config with specific sizes
        mock_loader = MagicMock()
        mock_loader.load_config.return_value = {
            "models": {
                "small-model": {
                    "provider": "openai",
                    "type": "openai",
                    "size_b": 7,
                    "config": {"model_name": "small-model"}
                },
                "large-model": {
                    "provider": "openai",
                    "type": "openai",
                    "size_b": 175,
                    "config": {"model_name": "large-model"}
                }
            },
            "defaults": {}
        }
        mock_get_loader.return_value = mock_loader
        
        os.environ["OPENAI_API_KEY"] = "test-key"
        
        registry = init_models()
        
        # Check sizes
        small_model = registry.get_model("small-model", "openai")
        assert hasattr(small_model, "_size_billions")
        assert small_model._size_billions == 7
        
        large_model = registry.get_model("large-model", "openai")
        assert hasattr(large_model, "_size_billions")
        assert large_model._size_billions == 175
    
    @patch('src.orchestrator.utils.model_config_loader.get_model_config_loader')
    @patch('src.orchestrator.utils.model_utils.check_ollama_installed')
    def test_init_models_error_handling(self, mock_ollama_check, mock_get_loader):
        """Test error handling during model registration."""
        mock_ollama_check.return_value = False
        
        # Mock config
        mock_loader = MagicMock()
        mock_loader.load_config.return_value = {
            "models": {
                "error-model": {
                    "provider": "openai",
                    "type": "openai",
                    "size_b": 7,
                    "config": {"model_name": "error-model"}
                }
            },
            "defaults": {}
        }
        mock_get_loader.return_value = mock_loader
        
        os.environ["OPENAI_API_KEY"] = "test-key"
        
        # Mock OpenAIModel to raise error
        with patch('src.orchestrator.integrations.openai_model.OpenAIModel') as mock_openai:
            mock_openai.side_effect = Exception("Registration error")
            
            # Should not crash, just skip the model
            registry = init_models()
            assert len(registry.list_models()) == 0
    
    def test_init_models_global_registry(self):
        """Test that init_models sets global registry."""
        import src.orchestrator
        
        # Initially should be None
        assert src.orchestrator._model_registry is None
        
        with patch('src.orchestrator.utils.model_config_loader.get_model_config_loader'):
            with patch('src.orchestrator.utils.model_utils.check_ollama_installed'):
                registry = init_models()
                
                # Global registry should be set
                assert src.orchestrator._model_registry is registry