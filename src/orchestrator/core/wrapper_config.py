"""
Unified configuration management system for wrapper architecture.

This module provides standardized configuration management for all wrappers
including validation, inheritance, environment overrides, and runtime updates.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, get_type_hints

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='BaseWrapperConfig')


class ConfigSource(Enum):
    """Sources for configuration values."""
    
    DEFAULT = "default"
    FILE = "file"
    ENVIRONMENT = "environment"
    RUNTIME = "runtime"
    OVERRIDE = "override"


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


@dataclass
class ConfigField:
    """Metadata for configuration fields."""
    
    name: str
    field_type: Type
    default_value: Any
    description: str = ""
    required: bool = False
    sensitive: bool = False  # For secrets/API keys
    environment_var: Optional[str] = None
    validator: Optional[callable] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    
    def validate(self, value: Any) -> bool:
        """
        Validate a configuration value.
        
        Args:
            value: Value to validate
            
        Returns:
            True if valid, False otherwise
        """
        if value is None and self.required:
            return False
        
        if value is not None:
            # Type validation
            if not isinstance(value, self.field_type):
                try:
                    # Try to convert
                    value = self.field_type(value)
                except (ValueError, TypeError):
                    return False
            
            # Range validation
            if isinstance(value, (int, float)):
                if self.min_value is not None and value < self.min_value:
                    return False
                if self.max_value is not None and value > self.max_value:
                    return False
            
            # Allowed values validation
            if self.allowed_values is not None and value not in self.allowed_values:
                return False
            
            # Custom validator
            if self.validator and not self.validator(value):
                return False
        
        return True


@dataclass
class BaseWrapperConfig(ABC):
    """
    Abstract base configuration for all wrappers.
    
    Provides standard configuration fields that all wrappers need:
    - Enable/disable control
    - Timeout and retry settings  
    - Monitoring configuration
    - Feature flag integration
    """
    
    # Core wrapper settings
    enabled: bool = False
    fallback_enabled: bool = True
    max_retry_attempts: int = 3
    timeout_seconds: float = 30.0
    
    # Monitoring and metrics
    monitoring_enabled: bool = True
    metrics_retention_days: int = 30
    debug_logging: bool = False
    
    # Feature flag integration
    feature_flag_prefix: str = ""
    
    # Runtime metadata (not user configurable)
    _config_source: ConfigSource = field(default=ConfigSource.DEFAULT, init=False)
    _last_updated: datetime = field(default_factory=datetime.utcnow, init=False)
    _updated_by: str = field(default="system", init=False)
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        self.validate()
    
    @abstractmethod
    def get_config_fields(self) -> Dict[str, ConfigField]:
        """
        Get configuration field metadata.
        
        Subclasses must implement this to define their specific configuration
        fields with validation rules and metadata.
        
        Returns:
            Dictionary mapping field names to ConfigField instances
        """
        pass
    
    def validate(self) -> None:
        """
        Validate entire configuration.
        
        Raises:
            ConfigValidationError: If any configuration values are invalid
        """
        config_fields = self.get_config_fields()
        errors = []
        
        for field_name, config_field in config_fields.items():
            value = getattr(self, field_name, None)
            if not config_field.validate(value):
                errors.append(f"Invalid value for {field_name}: {value}")
        
        # Base validation
        if self.timeout_seconds <= 0:
            errors.append("timeout_seconds must be positive")
        
        if self.max_retry_attempts < 0:
            errors.append("max_retry_attempts must be non-negative")
        
        if self.metrics_retention_days <= 0:
            errors.append("metrics_retention_days must be positive")
        
        if errors:
            raise ConfigValidationError(f"Configuration validation failed: {'; '.join(errors)}")
    
    @property
    def is_valid(self) -> bool:
        """
        Check if configuration is valid.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            self.validate()
            return True
        except ConfigValidationError:
            return False
    
    def get_feature_flag_name(self, flag: str) -> str:
        """
        Get the full feature flag name for this wrapper.
        
        Args:
            flag: Flag name suffix
            
        Returns:
            Full feature flag name with wrapper prefix
        """
        prefix = self.feature_flag_prefix or self._get_default_prefix()
        return f"{prefix}_{flag}" if prefix else flag
    
    def _get_default_prefix(self) -> str:
        """Get default feature flag prefix from class name."""
        class_name = self.__class__.__name__
        # Convert CamelCase to snake_case
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', class_name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower().replace('_config', '')
    
    def update(self, updates: Dict[str, Any], updated_by: str = "system") -> None:
        """
        Update configuration values.
        
        Args:
            updates: Dictionary of field names to new values
            updated_by: Who made the update (for audit trail)
        """
        config_fields = self.get_config_fields()
        
        for field_name, value in updates.items():
            if field_name.startswith('_'):
                continue  # Skip private fields
            
            if not hasattr(self, field_name):
                logger.warning(f"Unknown configuration field: {field_name}")
                continue
            
            # Validate individual field if metadata available
            if field_name in config_fields:
                config_field = config_fields[field_name]
                if not config_field.validate(value):
                    raise ConfigValidationError(f"Invalid value for {field_name}: {value}")
            
            setattr(self, field_name, value)
        
        self._last_updated = datetime.utcnow()
        self._updated_by = updated_by
        self._config_source = ConfigSource.RUNTIME
        
        # Validate entire config after updates
        self.validate()
    
    def to_dict(self, include_metadata: bool = False, mask_sensitive: bool = True) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Args:
            include_metadata: Whether to include runtime metadata
            mask_sensitive: Whether to mask sensitive values
            
        Returns:
            Dictionary representation of configuration
        """
        config_dict = {}
        config_fields = self.get_config_fields()
        
        for field in fields(self):
            field_name = field.name
            
            # Skip private fields unless metadata is requested
            if field_name.startswith('_') and not include_metadata:
                continue
            
            value = getattr(self, field_name)
            
            # Mask sensitive fields
            if mask_sensitive and field_name in config_fields:
                config_field = config_fields[field_name]
                if config_field.sensitive and value is not None:
                    value = "*" * 8  # Mask sensitive values
            
            # Convert datetime to ISO string
            if isinstance(value, datetime):
                value = value.isoformat()
            
            # Convert enums to values
            if hasattr(value, 'value'):
                value = value.value
            
            config_dict[field_name] = value
        
        return config_dict
    
    @classmethod
    def from_dict(cls: Type[T], config_dict: Dict[str, Any]) -> T:
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            New configuration instance
        """
        # Filter out unknown fields and private fields
        valid_fields = {f.name for f in fields(cls) if not f.name.startswith('_')}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        return cls(**filtered_dict)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get configuration summary with key information.
        
        Returns:
            Summary dictionary with essential configuration info
        """
        return {
            "enabled": self.enabled,
            "fallback_enabled": self.fallback_enabled,
            "monitoring_enabled": self.monitoring_enabled,
            "timeout_seconds": self.timeout_seconds,
            "max_retry_attempts": self.max_retry_attempts,
            "is_valid": self.is_valid,
            "last_updated": self._last_updated.isoformat(),
            "config_source": self._config_source.value
        }


class ConfigurationManager:
    """
    Manages configuration for multiple wrappers.
    
    Provides centralized configuration management including:
    - Loading from multiple sources (files, environment, runtime)
    - Configuration validation and inheritance
    - Environment-specific overrides
    - Runtime configuration updates
    """
    
    def __init__(
        self,
        config_dir: Optional[Path] = None,
        environment: str = "production"
    ):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
            environment: Current environment (development, staging, production)
        """
        self.config_dir = config_dir or Path("config")
        self.environment = environment
        self._configs: Dict[str, BaseWrapperConfig] = {}
        self._config_types: Dict[str, Type[BaseWrapperConfig]] = {}
        
        # Load configurations
        self._load_configurations()
    
    def register_config_type(
        self, 
        wrapper_name: str, 
        config_type: Type[BaseWrapperConfig]
    ) -> None:
        """
        Register a configuration type for a wrapper.
        
        Args:
            wrapper_name: Name of the wrapper
            config_type: Configuration class for the wrapper
        """
        self._config_types[wrapper_name] = config_type
        logger.debug(f"Registered config type for {wrapper_name}: {config_type.__name__}")
    
    def get_config(self, wrapper_name: str) -> Optional[BaseWrapperConfig]:
        """
        Get configuration for a wrapper.
        
        Args:
            wrapper_name: Name of the wrapper
            
        Returns:
            Configuration instance or None if not found
        """
        return self._configs.get(wrapper_name)
    
    def update_config(
        self,
        wrapper_name: str,
        updates: Dict[str, Any],
        updated_by: str = "system",
        save_to_file: bool = True
    ) -> bool:
        """
        Update configuration for a wrapper.
        
        Args:
            wrapper_name: Name of the wrapper
            updates: Dictionary of configuration updates
            updated_by: Who made the update
            save_to_file: Whether to persist changes to file
            
        Returns:
            True if update was successful
        """
        if wrapper_name not in self._configs:
            logger.error(f"No configuration found for wrapper: {wrapper_name}")
            return False
        
        config = self._configs[wrapper_name]
        
        try:
            config.update(updates, updated_by)
            
            if save_to_file:
                self._save_config(wrapper_name, config)
            
            logger.info(f"Updated configuration for {wrapper_name} by {updated_by}")
            return True
            
        except ConfigValidationError as e:
            logger.error(f"Configuration update failed for {wrapper_name}: {e}")
            return False
    
    def create_config(
        self,
        wrapper_name: str,
        config_data: Optional[Dict[str, Any]] = None
    ) -> Optional[BaseWrapperConfig]:
        """
        Create configuration for a wrapper.
        
        Args:
            wrapper_name: Name of the wrapper
            config_data: Initial configuration data
            
        Returns:
            Created configuration instance or None if failed
        """
        if wrapper_name not in self._config_types:
            logger.error(f"No config type registered for wrapper: {wrapper_name}")
            return None
        
        config_type = self._config_types[wrapper_name]
        config_data = config_data or {}
        
        try:
            # Load environment-specific overrides
            env_overrides = self._load_environment_overrides(wrapper_name)
            if env_overrides:
                config_data.update(env_overrides)
            
            config = config_type.from_dict(config_data)
            config._config_source = ConfigSource.FILE
            
            self._configs[wrapper_name] = config
            logger.info(f"Created configuration for {wrapper_name}")
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to create configuration for {wrapper_name}: {e}")
            return None
    
    def validate_all_configs(self) -> Dict[str, bool]:
        """
        Validate all registered configurations.
        
        Returns:
            Dictionary mapping wrapper names to validation status
        """
        validation_results = {}
        
        for wrapper_name, config in self._configs.items():
            try:
                config.validate()
                validation_results[wrapper_name] = True
            except ConfigValidationError as e:
                logger.error(f"Validation failed for {wrapper_name}: {e}")
                validation_results[wrapper_name] = False
        
        return validation_results
    
    def get_all_configs(self) -> Dict[str, BaseWrapperConfig]:
        """Get all registered configurations."""
        return self._configs.copy()
    
    def get_system_summary(self) -> Dict[str, Any]:
        """
        Get summary of all configurations.
        
        Returns:
            System-wide configuration summary
        """
        total_configs = len(self._configs)
        enabled_configs = len([c for c in self._configs.values() if c.enabled])
        valid_configs = len([c for c in self._configs.values() if c.is_valid])
        
        return {
            "total_configs": total_configs,
            "enabled_configs": enabled_configs,
            "disabled_configs": total_configs - enabled_configs,
            "valid_configs": valid_configs,
            "invalid_configs": total_configs - valid_configs,
            "environment": self.environment,
            "config_directory": str(self.config_dir)
        }
    
    def _load_configurations(self) -> None:
        """Load configurations from files."""
        if not self.config_dir.exists():
            logger.info(f"Configuration directory not found: {self.config_dir}")
            return
        
        # Load base configuration file
        base_config_file = self.config_dir / "wrappers.json"
        if base_config_file.exists():
            self._load_config_file(base_config_file)
        
        # Load environment-specific configuration
        env_config_file = self.config_dir / f"wrappers-{self.environment}.json"
        if env_config_file.exists():
            self._load_config_file(env_config_file, is_override=True)
    
    def _load_config_file(self, config_file: Path, is_override: bool = False) -> None:
        """Load configuration from a specific file."""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            wrappers_config = config_data.get('wrappers', {})
            
            for wrapper_name, wrapper_config in wrappers_config.items():
                if is_override:
                    # Update existing configuration
                    if wrapper_name in self._configs:
                        self._configs[wrapper_name].update(wrapper_config, "config_file")
                else:
                    # Create new configuration
                    self.create_config(wrapper_name, wrapper_config)
            
            logger.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_file}: {e}")
    
    def _load_environment_overrides(self, wrapper_name: str) -> Dict[str, Any]:
        """Load configuration overrides from environment variables."""
        overrides = {}
        prefix = f"WRAPPER_{wrapper_name.upper()}_"
        
        for env_var, value in os.environ.items():
            if env_var.startswith(prefix):
                field_name = env_var[len(prefix):].lower()
                
                # Try to convert to appropriate type
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif '.' in value and all(part.replace('.', '').isdigit() for part in value.split('.', 1)):
                    value = float(value)
                
                overrides[field_name] = value
        
        return overrides
    
    def _save_config(self, wrapper_name: str, config: BaseWrapperConfig) -> None:
        """Save configuration to file."""
        config_file = self.config_dir / f"{wrapper_name}.json"
        
        try:
            # Ensure directory exists
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            config_data = {
                "wrapper_name": wrapper_name,
                "config": config.to_dict(include_metadata=True, mask_sensitive=False)
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            
            logger.debug(f"Saved configuration for {wrapper_name} to {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration for {wrapper_name}: {e}")


# Example concrete configuration classes

@dataclass
class RouteLLMWrapperConfig(BaseWrapperConfig):
    """Configuration for RouteLLM wrapper."""
    
    router_type: str = "mf"
    threshold: float = 0.11593
    strong_model: str = "gpt-4-1106-preview"
    weak_model: str = "gpt-3.5-turbo"
    cost_optimization_target: float = 0.5
    
    def get_config_fields(self) -> Dict[str, ConfigField]:
        """Get RouteLLM-specific configuration fields."""
        return {
            "enabled": ConfigField("enabled", bool, False, "Enable RouteLLM integration"),
            "router_type": ConfigField(
                "router_type", str, "mf", 
                "Router type for RouteLLM",
                allowed_values=["mf", "bert", "causal_llm", "sw_ranking", "random"]
            ),
            "threshold": ConfigField(
                "threshold", float, 0.11593,
                "Routing threshold",
                min_value=0.0, max_value=1.0
            ),
            "strong_model": ConfigField("strong_model", str, "gpt-4-1106-preview", "Strong model name"),
            "weak_model": ConfigField("weak_model", str, "gpt-3.5-turbo", "Weak model name"),
            "cost_optimization_target": ConfigField(
                "cost_optimization_target", float, 0.5,
                "Target cost reduction percentage",
                min_value=0.0, max_value=1.0
            ),
            "timeout_seconds": ConfigField(
                "timeout_seconds", float, 30.0,
                "Operation timeout in seconds",
                min_value=1.0, max_value=300.0
            ),
            "max_retry_attempts": ConfigField(
                "max_retry_attempts", int, 3,
                "Maximum retry attempts",
                min_value=0, max_value=10
            )
        }


@dataclass  
class POMLWrapperConfig(BaseWrapperConfig):
    """Configuration for POML template wrapper."""
    
    enable_poml: bool = True
    enable_hybrid_templates: bool = True
    migration_analysis_enabled: bool = False
    
    def get_config_fields(self) -> Dict[str, ConfigField]:
        """Get POML-specific configuration fields."""
        return {
            "enabled": ConfigField("enabled", bool, False, "Enable POML integration"),
            "enable_poml": ConfigField("enable_poml", bool, True, "Enable POML template processing"),
            "enable_hybrid_templates": ConfigField(
                "enable_hybrid_templates", bool, True,
                "Enable hybrid Jinja2/POML templates"
            ),
            "migration_analysis_enabled": ConfigField(
                "migration_analysis_enabled", bool, False,
                "Enable migration analysis tools"
            ),
            "timeout_seconds": ConfigField(
                "timeout_seconds", float, 30.0,
                "Template processing timeout",
                min_value=1.0, max_value=300.0
            ),
            "max_retry_attempts": ConfigField(
                "max_retry_attempts", int, 3,
                "Maximum retry attempts",
                min_value=0, max_value=10
            )
        }