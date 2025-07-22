"""Automatic model registration for new models not in models.yaml."""

import logging
from typing import Optional
from ..core.model import Model
from ..utils.model_config_loader import get_model_config_loader
from ..utils.model_utils import parse_model_size

logger = logging.getLogger(__name__)


class AutoModelRegistrar:
    """Automatically register new models when they're successfully used."""
    
    def __init__(self, model_registry):
        """Initialize the auto-registrar.
        
        Args:
            model_registry: The ModelRegistry instance to register models with
        """
        self.model_registry = model_registry
        self.config_loader = get_model_config_loader()
        self._pending_models = {}  # Track models being tested
    
    async def try_register_model(self, model_name: str, provider: str) -> Optional[Model]:
        """Try to register a new model that's not in models.yaml.
        
        Args:
            model_name: Name of the model
            provider: Provider name (openai, anthropic, google, ollama, huggingface)
            
        Returns:
            Registered Model instance if successful, None otherwise
        """
        # Check if already registered
        model_key = f"{provider}:{model_name}"
        if model_key in self.model_registry.models:
            return self.model_registry.models[model_key]
        
        # Check if model exists in config
        if self.config_loader.model_exists(model_name):
            return None
        
        # Check if we're already trying to register this model
        if model_key in self._pending_models:
            return None
        
        logger.info(f"Attempting to auto-register new model: {model_key}")
        self._pending_models[model_key] = True
        
        try:
            # Create model instance based on provider
            model = await self._create_model_instance(model_name, provider)
            if not model:
                return None
            
            # Test the model with a simple prompt
            success = await self._test_model(model)
            if not success:
                logger.warning(f"Model {model_key} failed test, not registering")
                return None
            
            # Register the model
            self.model_registry.register_model(model)
            logger.info(f"Successfully auto-registered model: {model_key}")
            
            # Add to models.yaml for persistence
            await self._add_to_config(model_name, provider, model)
            
            return model
            
        except Exception as e:
            logger.error(f"Error auto-registering model {model_key}: {e}")
            return None
        finally:
            # Remove from pending
            self._pending_models.pop(model_key, None)
    
    async def _create_model_instance(self, model_name: str, provider: str) -> Optional[Model]:
        """Create a model instance for the given provider.
        
        Args:
            model_name: Model name
            provider: Provider name
            
        Returns:
            Model instance or None if creation failed
        """
        try:
            if provider == "ollama":
                from ..integrations.lazy_ollama_model import LazyOllamaModel
                model = LazyOllamaModel(model_name=model_name)
                # Set dynamic attributes
                setattr(model, "_expertise", ["general"])
                setattr(model, "_size_billions", parse_model_size(model_name))
                return model
                
            elif provider == "huggingface":
                from ..integrations.lazy_huggingface_model import LazyHuggingFaceModel
                model = LazyHuggingFaceModel(model_name=model_name)
                # Set dynamic attributes
                setattr(model, "_expertise", ["general"])
                setattr(model, "_size_billions", parse_model_size(model_name))
                return model
                
            elif provider == "openai":
                import os
                if not os.environ.get("OPENAI_API_KEY"):
                    logger.warning("OpenAI API key not set, cannot auto-register OpenAI models")
                    return None
                from ..integrations.openai_model import OpenAIModel
                model = OpenAIModel(model_name=model_name)
                setattr(model, "_expertise", ["general"])
                setattr(model, "_size_billions", parse_model_size(model_name))
                return model
                
            elif provider == "anthropic":
                import os
                if not os.environ.get("ANTHROPIC_API_KEY"):
                    logger.warning("Anthropic API key not set, cannot auto-register Anthropic models")
                    return None
                from ..integrations.anthropic_model import AnthropicModel
                model = AnthropicModel(model_name=model_name)
                setattr(model, "_expertise", ["general"])
                setattr(model, "_size_billions", parse_model_size(model_name))
                return model
                
            elif provider == "google":
                import os
                if not os.environ.get("GOOGLE_API_KEY"):
                    logger.warning("Google API key not set, cannot auto-register Google models")
                    return None
                from ..integrations.google_model import GoogleModel
                model = GoogleModel(model_name=model_name)
                setattr(model, "_expertise", ["general"])
                setattr(model, "_size_billions", parse_model_size(model_name))
                return model
                
            else:
                logger.warning(f"Unknown provider: {provider}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating model instance for {provider}:{model_name}: {e}")
            return None
    
    async def _test_model(self, model: Model) -> bool:
        """Test if a model works with a simple prompt.
        
        Args:
            model: Model instance to test
            
        Returns:
            True if model works, False otherwise
        """
        try:
            # Simple test prompt - use a more conversational prompt for chat models
            test_prompts = [
                "What is 2+2?",
                "Hello! Please respond with a simple greeting.",
                "Say 'test successful' if you can understand this message."
            ]
            
            # Try multiple prompts to handle different model styles
            for prompt in test_prompts:
                try:
                    response = await model.generate(prompt)
                    
                    # Check if we got a reasonable response
                    if response and len(response.strip()) > 0:
                        logger.info(f"Model test successful with prompt '{prompt[:30]}...', response: {response[:50]}...")
                        return True
                except Exception as e:
                    logger.debug(f"Test failed with prompt '{prompt[:30]}...': {e}")
                    continue
            
            logger.warning("Model test failed: no successful responses from any test prompt")
            return False
                
        except Exception as e:
            logger.error(f"Model test failed with error: {e}")
            return False
    
    async def _add_to_config(self, model_name: str, provider: str, model: Model) -> None:
        """Add successfully registered model to models.yaml.
        
        Args:
            model_name: Model name
            provider: Provider name
            model: Model instance
        """
        try:
            # Get model size
            size_b = getattr(model, "_size_billions", 1.0)
            
            # Determine expertise
            getattr(model, "_expertise", ["general"])
            
            # Create model configuration
            model_config = {
                "provider": provider,
                "type": provider,
                "size_b": size_b,
                "config": {
                    "model_name": model_name
                }
            }
            
            # Add any provider-specific config
            if provider == "ollama":
                model_config["config"]["base_url"] = "http://localhost:11434"
                model_config["config"]["timeout"] = 60
            elif provider == "huggingface":
                model_config["config"]["device"] = "auto"
            
            # Add to configuration
            model_id = f"{provider}_{model_name.replace('/', '_').replace(':', '_')}"
            self.config_loader.add_model(model_id, model_config)
            
            logger.info(f"Added model {model_id} to models.yaml")
            
        except Exception as e:
            logger.error(f"Error adding model to config: {e}")


def get_provider_from_model_name(model_name: str) -> Optional[str]:
    """Guess provider from model name patterns.
    
    Args:
        model_name: Model name
        
    Returns:
        Provider name or None
    """
    # Check for explicit provider prefixes
    if model_name.startswith("gpt-") or model_name.startswith("o1-"):
        return "openai"
    elif model_name.startswith("claude-"):
        return "anthropic"
    elif model_name.startswith("gemini-") or model_name.startswith("models/"):
        return "google"
    elif "/" in model_name and not model_name.startswith("models/"):
        # Likely a HuggingFace model (org/model format)
        return "huggingface"
    elif ":" in model_name or model_name in ["llama3.1", "llama3.2", "mistral", "gemma", "qwen"]:
        # Likely an Ollama model
        return "ollama"
    else:
        # Default to ollama for simple names
        return "ollama"