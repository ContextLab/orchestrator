"""
Environment-specific configuration management system.

This module provides environment-specific configuration overrides,
validation, and management for wrapper integrations across different
deployment environments (development, staging, production).
"""

from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .wrapper_config import BaseWrapperConfig, ConfigValidationError

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Supported deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    
    @classmethod
    def from_string(cls, env_str: str) -> Environment:
        """Create Environment from string, case-insensitive."""
        try:
            return cls(env_str.lower())
        except ValueError:
            logger.warning(f"Unknown environment '{env_str}', defaulting to development")
            return cls.DEVELOPMENT


@dataclass
class EnvironmentOverride:
    """Environment-specific configuration override."""
    
    environment: Environment
    config_path: str  # dot notation: "wrapper.api.timeout_seconds"
    value: Any
    reason: str = ""
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        """Check if override has expired."""
        return self.expires_at is not None and self.expires_at < datetime.utcnow()
    
    def is_applicable(self, environment: Environment) -> bool:
        """Check if override applies to the given environment."""
        return self.environment == environment and not self.is_expired()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert override to dictionary."""
        return {
            "environment": self.environment.value,
            "config_path": self.config_path,
            "value": self.value,
            "reason": self.reason,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EnvironmentOverride:
        """Create override from dictionary."""
        return cls(
            environment=Environment(data["environment"]),
            config_path=data["config_path"],
            value=data["value"],
            reason=data.get("reason", ""),
            created_by=data.get("created_by", "system"),
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            tags=data.get("tags", [])
        )


@dataclass
class EnvironmentProfile:
    """Environment-specific configuration profile."""
    
    name: str
    environment: Environment
    description: str = ""
    base_overrides: Dict[str, Any] = field(default_factory=dict)
    inherit_from: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "name": self.name,
            "environment": self.environment.value,
            "description": self.description,
            "base_overrides": self.base_overrides,
            "inherit_from": self.inherit_from,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EnvironmentProfile:
        """Create profile from dictionary."""
        return cls(
            name=data["name"],
            environment=Environment(data["environment"]),
            description=data.get("description", ""),
            base_overrides=data.get("base_overrides", {}),
            inherit_from=data.get("inherit_from"),
            created_at=datetime.fromisoformat(data["created_at"]),
            tags=data.get("tags", [])
        )


class EnvironmentConfigManager:
    """Manages environment-specific configuration overrides."""
    
    def __init__(
        self, 
        current_environment: Union[Environment, str],
        config_dir: Optional[Path] = None
    ):
        """
        Initialize environment configuration manager.
        
        Args:
            current_environment: Current deployment environment
            config_dir: Directory for configuration files
        """
        if isinstance(current_environment, str):
            self.current_environment = Environment.from_string(current_environment)
        else:
            self.current_environment = current_environment
        
        self.config_dir = config_dir or Path("config") / "environments"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for overrides and profiles
        self._overrides: Dict[str, List[EnvironmentOverride]] = {}
        self._profiles: Dict[str, EnvironmentProfile] = {}
        self._base_configs: Dict[str, BaseWrapperConfig] = {}
        
        # Load configuration
        self._load_configuration()
        
        logger.info(f"Initialized environment config manager for {self.current_environment.value}")
    
    def register_base_config(
        self, 
        wrapper_name: str, 
        config: BaseWrapperConfig
    ) -> None:
        """Register a base configuration for environment overrides."""
        self._base_configs[wrapper_name] = config
        
        # Load wrapper-specific overrides
        self._load_wrapper_overrides(wrapper_name)
        
        logger.debug(f"Registered base config for {wrapper_name}")
    
    def add_override(
        self, 
        wrapper_name: str, 
        override: EnvironmentOverride
    ) -> None:
        """Add an environment-specific override."""
        if wrapper_name not in self._overrides:
            self._overrides[wrapper_name] = []
        
        # Remove any existing override for the same path and environment
        self._overrides[wrapper_name] = [
            o for o in self._overrides[wrapper_name] 
            if not (o.environment == override.environment and o.config_path == override.config_path)
        ]
        
        self._overrides[wrapper_name].append(override)
        
        # Save to file
        self._save_wrapper_overrides(wrapper_name)
        
        logger.info(f"Added override for {wrapper_name}: {override.config_path} = {override.value}")
    
    def remove_override(
        self, 
        wrapper_name: str,
        environment: Environment,
        config_path: str
    ) -> bool:
        """Remove an environment-specific override."""
        if wrapper_name not in self._overrides:
            return False
        
        original_count = len(self._overrides[wrapper_name])
        self._overrides[wrapper_name] = [
            o for o in self._overrides[wrapper_name]
            if not (o.environment == environment and o.config_path == config_path)
        ]
        
        removed = len(self._overrides[wrapper_name]) < original_count
        
        if removed:
            self._save_wrapper_overrides(wrapper_name)
            logger.info(f"Removed override for {wrapper_name}: {config_path}")
        
        return removed
    
    def get_effective_config(
        self, 
        wrapper_name: str,
        environment: Optional[Environment] = None
    ) -> Optional[BaseWrapperConfig]:
        """Get configuration with environment overrides applied."""
        if wrapper_name not in self._base_configs:
            logger.warning(f"No base configuration found for {wrapper_name}")
            return None
        
        environment = environment or self.current_environment
        
        # Start with a deep copy of base configuration
        config = copy.deepcopy(self._base_configs[wrapper_name])
        
        # Apply profile overrides first
        self._apply_profile_overrides(config, environment, wrapper_name)
        
        # Apply specific overrides
        overrides = self._overrides.get(wrapper_name, [])
        applicable_overrides = [
            o for o in overrides 
            if o.is_applicable(environment)
        ]
        
        # Sort by creation time to apply in order
        applicable_overrides.sort(key=lambda x: x.created_at)
        
        for override in applicable_overrides:
            try:
                self._apply_override_to_config(config, override)
            except Exception as e:
                logger.error(f"Failed to apply override {override.config_path}: {e}")
        
        # Apply environment variables
        self._apply_environment_variables(config, wrapper_name)
        
        return config
    
    def create_profile(
        self,
        profile_name: str,
        environment: Environment,
        description: str = "",
        base_overrides: Optional[Dict[str, Any]] = None,
        inherit_from: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> EnvironmentProfile:
        """Create an environment profile."""
        profile = EnvironmentProfile(
            name=profile_name,
            environment=environment,
            description=description,
            base_overrides=base_overrides or {},
            inherit_from=inherit_from,
            tags=tags or []
        )
        
        self._profiles[profile_name] = profile
        self._save_profiles()
        
        logger.info(f"Created environment profile: {profile_name}")
        return profile
    
    def get_profile(self, profile_name: str) -> Optional[EnvironmentProfile]:
        """Get an environment profile."""
        return self._profiles.get(profile_name)
    
    def list_profiles(self, environment: Optional[Environment] = None) -> List[EnvironmentProfile]:
        """List environment profiles."""
        profiles = list(self._profiles.values())
        
        if environment:
            profiles = [p for p in profiles if p.environment == environment]
        
        return profiles
    
    def get_overrides(
        self,
        wrapper_name: Optional[str] = None,
        environment: Optional[Environment] = None,
        include_expired: bool = False
    ) -> Dict[str, List[EnvironmentOverride]]:
        """Get environment overrides."""
        result = {}
        
        wrappers_to_check = [wrapper_name] if wrapper_name else list(self._overrides.keys())
        
        for wrapper in wrappers_to_check:
            if wrapper not in self._overrides:
                continue
            
            overrides = self._overrides[wrapper]
            
            # Filter by environment
            if environment:
                overrides = [o for o in overrides if o.environment == environment]
            
            # Filter expired
            if not include_expired:
                overrides = [o for o in overrides if not o.is_expired()]
            
            if overrides:
                result[wrapper] = overrides
        
        return result
    
    def validate_config(
        self,
        wrapper_name: str,
        environment: Optional[Environment] = None
    ) -> Dict[str, Any]:
        """Validate configuration for environment."""
        validation_result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "config": None
        }
        
        try:
            config = self.get_effective_config(wrapper_name, environment)
            
            if config is None:
                validation_result["errors"].append(f"No configuration found for {wrapper_name}")
                return validation_result
            
            # Validate the configuration
            config.validate()
            
            validation_result["valid"] = True
            validation_result["config"] = config.to_dict(mask_sensitive=True)
            
            # Check for warnings
            overrides = self._overrides.get(wrapper_name, [])
            env_to_check = environment or self.current_environment
            
            applicable_overrides = [o for o in overrides if o.is_applicable(env_to_check)]
            
            if applicable_overrides:
                validation_result["warnings"].append(
                    f"{len(applicable_overrides)} environment overrides applied"
                )
            
            # Check for soon-to-expire overrides
            soon_expiring = [
                o for o in applicable_overrides
                if o.expires_at and o.expires_at < datetime.utcnow() + timedelta(days=7)
            ]
            
            if soon_expiring:
                validation_result["warnings"].append(
                    f"{len(soon_expiring)} overrides expiring within 7 days"
                )
            
        except ConfigValidationError as e:
            validation_result["errors"].append(str(e))
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {e}")
        
        return validation_result
    
    def cleanup_expired_overrides(self) -> int:
        """Remove expired overrides."""
        total_cleaned = 0
        
        for wrapper_name in list(self._overrides.keys()):
            original_count = len(self._overrides[wrapper_name])
            
            # Filter out expired overrides
            self._overrides[wrapper_name] = [
                o for o in self._overrides[wrapper_name]
                if not o.is_expired()
            ]
            
            cleaned = original_count - len(self._overrides[wrapper_name])
            if cleaned > 0:
                self._save_wrapper_overrides(wrapper_name)
                total_cleaned += cleaned
        
        if total_cleaned > 0:
            logger.info(f"Cleaned up {total_cleaned} expired overrides")
        
        return total_cleaned
    
    def get_environment_summary(self) -> Dict[str, Any]:
        """Get summary of environment configuration."""
        total_wrappers = len(self._base_configs)
        total_overrides = sum(len(overrides) for overrides in self._overrides.values())
        
        # Count active overrides for current environment
        current_env_overrides = 0
        for overrides in self._overrides.values():
            current_env_overrides += len([
                o for o in overrides 
                if o.is_applicable(self.current_environment)
            ])
        
        # Count overrides by environment
        overrides_by_env = {}
        for env in Environment:
            count = 0
            for overrides in self._overrides.values():
                count += len([o for o in overrides if o.environment == env and not o.is_expired()])
            overrides_by_env[env.value] = count
        
        return {
            "current_environment": self.current_environment.value,
            "total_wrappers": total_wrappers,
            "total_overrides": total_overrides,
            "current_env_overrides": current_env_overrides,
            "overrides_by_environment": overrides_by_env,
            "total_profiles": len(self._profiles),
            "config_directory": str(self.config_dir)
        }
    
    def _apply_override_to_config(
        self, 
        config: BaseWrapperConfig, 
        override: EnvironmentOverride
    ) -> None:
        """Apply a single override to a configuration object."""
        path_parts = override.config_path.split('.')
        obj = config
        
        # Navigate to the parent object
        for part in path_parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                logger.warning(f"Invalid override path: {override.config_path}")
                return
        
        # Set the final value
        final_key = path_parts[-1]
        if hasattr(obj, final_key):
            # Type conversion if needed
            original_type = type(getattr(obj, final_key))
            if original_type != type(override.value) and original_type != type(None):
                try:
                    converted_value = original_type(override.value)
                    setattr(obj, final_key, converted_value)
                except (ValueError, TypeError):
                    logger.warning(f"Type conversion failed for {override.config_path}")
                    setattr(obj, final_key, override.value)
            else:
                setattr(obj, final_key, override.value)
            
            logger.debug(f"Applied override: {override.config_path} = {override.value}")
        else:
            logger.warning(f"Invalid override key: {final_key}")
    
    def _apply_profile_overrides(
        self,
        config: BaseWrapperConfig,
        environment: Environment,
        wrapper_name: str
    ) -> None:
        """Apply profile-based overrides."""
        # Find profiles for this environment
        applicable_profiles = [
            p for p in self._profiles.values()
            if p.environment == environment
        ]
        
        # Apply profile overrides
        for profile in applicable_profiles:
            for path, value in profile.base_overrides.items():
                # Create temporary override
                temp_override = EnvironmentOverride(
                    environment=environment,
                    config_path=path,
                    value=value,
                    reason=f"Profile: {profile.name}"
                )
                try:
                    self._apply_override_to_config(config, temp_override)
                except Exception as e:
                    logger.error(f"Failed to apply profile override {path}: {e}")
    
    def _apply_environment_variables(
        self,
        config: BaseWrapperConfig,
        wrapper_name: str
    ) -> None:
        """Apply environment variable overrides."""
        prefix = f"ORCHESTRATOR_{wrapper_name.upper()}_"
        
        for env_var, value in os.environ.items():
            if env_var.startswith(prefix):
                # Convert environment variable name to config path
                config_key = env_var[len(prefix):].lower()
                
                # Try to convert value to appropriate type
                converted_value = self._convert_env_value(value)
                
                # Apply as override
                temp_override = EnvironmentOverride(
                    environment=self.current_environment,
                    config_path=config_key,
                    value=converted_value,
                    reason="Environment variable"
                )
                
                try:
                    self._apply_override_to_config(config, temp_override)
                    logger.debug(f"Applied env var override: {config_key} = {converted_value}")
                except Exception as e:
                    logger.debug(f"Failed to apply env var override {config_key}: {e}")
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean values
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer values
        if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            return int(value)
        
        # Float values
        try:
            if '.' in value:
                return float(value)
        except ValueError:
            pass
        
        # JSON values
        if value.startswith(('{', '[', '"')):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # Return as string
        return value
    
    def _load_configuration(self) -> None:
        """Load configuration files."""
        self._load_profiles()
        
        # Load overrides for all wrapper configs
        for wrapper_name in self._base_configs.keys():
            self._load_wrapper_overrides(wrapper_name)
    
    def _load_profiles(self) -> None:
        """Load environment profiles."""
        profiles_file = self.config_dir / "profiles.json"
        
        if profiles_file.exists():
            try:
                with open(profiles_file, 'r') as f:
                    profiles_data = json.load(f)
                
                self._profiles = {
                    name: EnvironmentProfile.from_dict(data)
                    for name, data in profiles_data.items()
                }
                
                logger.debug(f"Loaded {len(self._profiles)} environment profiles")
                
            except Exception as e:
                logger.error(f"Failed to load environment profiles: {e}")
                self._profiles = {}
    
    def _save_profiles(self) -> None:
        """Save environment profiles."""
        profiles_file = self.config_dir / "profiles.json"
        
        try:
            profiles_data = {
                name: profile.to_dict()
                for name, profile in self._profiles.items()
            }
            
            with open(profiles_file, 'w') as f:
                json.dump(profiles_data, f, indent=2, default=str)
            
            logger.debug("Saved environment profiles")
            
        except Exception as e:
            logger.error(f"Failed to save environment profiles: {e}")
    
    def _load_wrapper_overrides(self, wrapper_name: str) -> None:
        """Load overrides for a specific wrapper."""
        overrides_file = self.config_dir / f"{wrapper_name}_overrides.json"
        
        if overrides_file.exists():
            try:
                with open(overrides_file, 'r') as f:
                    overrides_data = json.load(f)
                
                self._overrides[wrapper_name] = [
                    EnvironmentOverride.from_dict(data)
                    for data in overrides_data
                ]
                
                logger.debug(f"Loaded {len(self._overrides[wrapper_name])} overrides for {wrapper_name}")
                
            except Exception as e:
                logger.error(f"Failed to load overrides for {wrapper_name}: {e}")
                self._overrides[wrapper_name] = []
    
    def _save_wrapper_overrides(self, wrapper_name: str) -> None:
        """Save overrides for a specific wrapper."""
        overrides_file = self.config_dir / f"{wrapper_name}_overrides.json"
        
        try:
            overrides_data = [
                override.to_dict()
                for override in self._overrides.get(wrapper_name, [])
            ]
            
            with open(overrides_file, 'w') as f:
                json.dump(overrides_data, f, indent=2, default=str)
            
            logger.debug(f"Saved overrides for {wrapper_name}")
            
        except Exception as e:
            logger.error(f"Failed to save overrides for {wrapper_name}: {e}")


# Factory functions for easy instantiation
def create_environment_manager(
    environment: Union[Environment, str] = Environment.DEVELOPMENT,
    config_dir: Optional[Path] = None
) -> EnvironmentConfigManager:
    """Create an environment configuration manager."""
    return EnvironmentConfigManager(environment, config_dir)


def get_current_environment() -> Environment:
    """Get current environment from environment variable or default."""
    env_str = os.environ.get('ORCHESTRATOR_ENVIRONMENT', 'development')
    return Environment.from_string(env_str)