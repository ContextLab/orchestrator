"""Load model configurations from models.yaml."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class ModelConfigLoader:
    """Load and manage model configurations from YAML files."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the model config loader.
        
        Args:
            config_path: Path to models.yaml. If not provided, searches in:
                1. ~/.orchestrator/models.yaml (user config)
                2. config/models.yaml (repository config)
        """
        self.config_path = self._find_config_path(config_path)
        self._config_cache = None
        self._last_modified = None
    
    def _find_config_path(self, config_path: Optional[Path]) -> Path:
        """Find the models.yaml configuration file."""
        if config_path and config_path.exists():
            return config_path
        
        # Search paths in order of preference
        search_paths = [
            Path.home() / ".orchestrator" / "models.yaml",  # User config
            Path(__file__).parent.parent.parent.parent / "config" / "models.yaml",  # Repo config
        ]
        
        for path in search_paths:
            if path.exists():
                logger.info(f"Using models config from: {path}")
                return path
        
        # Fallback to repo config path even if it doesn't exist yet
        default_path = search_paths[1]
        logger.warning(f"No models.yaml found, will use default path: {default_path}")
        return default_path
    
    def load_config(self, force_reload: bool = False) -> Dict[str, Any]:
        """Load the models configuration from YAML.
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            Dictionary containing model configurations
        """
        # Check if we need to reload
        if not force_reload and self._config_cache is not None:
            # Check if file was modified
            if self.config_path.exists():
                current_mtime = self.config_path.stat().st_mtime
                if current_mtime == self._last_modified:
                    return self._config_cache
        
        # Load configuration
        if not self.config_path.exists():
            logger.warning(f"Models config not found at {self.config_path}, using empty config")
            return {"models": {}, "preferences": {}, "cost_optimized": [], "performance_optimized": []}
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            
            # Ensure required sections exist
            if "models" not in config:
                config["models"] = {}
            if "preferences" not in config:
                config["preferences"] = {"default": "gpt-4o-mini", "fallback": []}
            if "cost_optimized" not in config:
                config["cost_optimized"] = []
            if "performance_optimized" not in config:
                config["performance_optimized"] = []
            
            # Cache the config
            self._config_cache = config
            if self.config_path.exists():
                self._last_modified = self.config_path.stat().st_mtime
            
            logger.info(f"Loaded {len(config['models'])} models from config")
            return config
            
        except Exception as e:
            logger.error(f"Error loading models config: {e}")
            return {"models": {}, "preferences": {}, "cost_optimized": [], "performance_optimized": []}
    
    def get_models_by_provider(self, provider: str) -> Dict[str, Dict[str, Any]]:
        """Get all models for a specific provider.
        
        Args:
            provider: Provider name (e.g., "openai", "anthropic", "ollama")
            
        Returns:
            Dictionary of model configurations for the provider
        """
        config = self.load_config()
        models = {}
        
        for model_id, model_config in config["models"].items():
            if model_config.get("provider") == provider:
                models[model_id] = model_config
        
        return models
    
    def get_model_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model configuration or None if not found
        """
        config = self.load_config()
        return config["models"].get(model_id)
    
    def add_model(self, model_id: str, model_config: Dict[str, Any]) -> None:
        """Add a new model to the configuration.
        
        Args:
            model_id: Model identifier
            model_config: Model configuration dictionary
        """
        config = self.load_config(force_reload=True)
        config["models"][model_id] = model_config
        self.save_config(config)
        logger.info(f"Added model {model_id} to config")
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration back to YAML file.
        
        Args:
            config: Configuration dictionary to save
        """
        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write configuration
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        # Update cache
        self._config_cache = config
        self._last_modified = self.config_path.stat().st_mtime
        
        logger.info(f"Saved models config to {self.config_path}")
    
    def get_preferences(self) -> Dict[str, Any]:
        """Get model preferences configuration."""
        config = self.load_config()
        return config.get("preferences", {})
    
    def get_cost_optimized_models(self) -> List[str]:
        """Get list of cost-optimized model IDs."""
        config = self.load_config()
        return config.get("cost_optimized", [])
    
    def get_performance_optimized_models(self) -> List[str]:
        """Get list of performance-optimized model IDs."""
        config = self.load_config()
        return config.get("performance_optimized", [])
    
    def model_exists(self, model_id: str) -> bool:
        """Check if a model exists in the configuration.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if model exists in config
        """
        config = self.load_config()
        return model_id in config["models"]


# Singleton instance
_loader_instance = None


def get_model_config_loader() -> ModelConfigLoader:
    """Get the singleton model config loader instance."""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = ModelConfigLoader()
    return _loader_instance