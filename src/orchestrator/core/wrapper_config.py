"""
Unified configuration management system for wrapper architecture.

This module provides standardized configuration management for all wrappers
including validation, inheritance, environment overrides, and runtime updates.
Extended with external tool configuration support for Issue #251.
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


# External Tool Configuration Classes

@dataclass
class ExternalToolConfig(BaseWrapperConfig):
    """Configuration for external tool integrations."""
    
    # API Configuration
    api_endpoint: str = ""
    api_key: str = ""
    api_version: str = "v1"
    
    # Authentication
    auth_type: str = "bearer"  # bearer, api_key, oauth
    oauth_client_id: Optional[str] = None
    oauth_client_secret: Optional[str] = None
    
    # Rate Limiting
    rate_limit_requests_per_minute: int = 60
    rate_limit_tokens_per_minute: int = 10000
    rate_limit_burst_size: int = 10
    
    # Connection Settings
    connection_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 30.0
    max_connections: int = 100
    max_keepalive_connections: int = 20
    
    # Cost and Budget Settings
    daily_budget: Optional[float] = None
    monthly_budget: Optional[float] = None
    budget_alert_thresholds: Dict[str, float] = field(default_factory=lambda: {"warning": 0.8, "critical": 0.95})
    cost_tracking_enabled: bool = True
    
    def get_config_fields(self) -> Dict[str, ConfigField]:
        """Get external tool specific configuration fields."""
        return {
            "api_endpoint": ConfigField(
                "api_endpoint", str, "", 
                "API endpoint URL", 
                required=True,
                validator=lambda x: x.startswith(('http://', 'https://')) if x else False
            ),
            "api_key": ConfigField(
                "api_key", str, "", 
                "API key for authentication",
                sensitive=True,
                environment_var="EXTERNAL_API_KEY"
            ),
            "cost_tracking_enabled": ConfigField(
                "cost_tracking_enabled", bool, True,
                "Enable cost tracking and monitoring"
            )
        }


# Factory function for easy instantiation
def create_external_tool_config(**kwargs) -> ExternalToolConfig:
    """Create an external tool configuration."""
    return ExternalToolConfig(**kwargs)


class ConfigurationManager:
    """Configuration manager for wrapper configurations."""
    
    def __init__(self):
        self.configurations: Dict[str, BaseWrapperConfig] = {}
    
    def register_config(self, name: str, config: BaseWrapperConfig) -> None:
        """Register a configuration."""
        self.configurations[name] = config
    
    def get_config(self, name: str) -> Optional[BaseWrapperConfig]:
        """Get a configuration by name."""
        return self.configurations.get(name)
    
    def list_configs(self) -> List[str]:
        """List all configuration names."""
        return list(self.configurations.keys())