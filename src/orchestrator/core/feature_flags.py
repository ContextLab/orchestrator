"""
Unified feature flag management system for wrapper architecture.

This module provides a comprehensive feature flag system that supports:
- Hierarchical flag dependencies
- Per-wrapper and per-domain flag management
- Runtime flag updates
- A/B testing capabilities
- Configuration integration
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureFlagScope(Enum):
    """Scope levels for feature flags."""
    
    GLOBAL = "global"           # System-wide flags
    WRAPPER = "wrapper"         # Wrapper-specific flags  
    DOMAIN = "domain"          # Domain/context specific flags
    USER = "user"              # User-specific flags
    SESSION = "session"        # Session-specific flags
    EXPERIMENTAL = "experimental"  # Experimental feature flags


class FeatureFlagStrategy(Enum):
    """Strategies for evaluating feature flags."""
    
    BOOLEAN = "boolean"         # Simple on/off
    PERCENTAGE = "percentage"   # Percentage-based rollout
    WHITELIST = "whitelist"     # Explicit whitelist of users/contexts
    BLACKLIST = "blacklist"     # Explicit blacklist of users/contexts
    CUSTOM = "custom"          # Custom evaluation function


@dataclass
class FeatureFlag:
    """
    Comprehensive feature flag definition.
    
    Supports various rollout strategies, dependencies, and metadata
    for sophisticated feature flag management.
    """
    
    name: str
    enabled: bool = False
    scope: FeatureFlagScope = FeatureFlagScope.WRAPPER
    strategy: FeatureFlagStrategy = FeatureFlagStrategy.BOOLEAN
    description: str = ""
    
    # Dependency management
    dependencies: List[str] = field(default_factory=list)
    incompatible_with: List[str] = field(default_factory=list)
    
    # Rollout controls
    rollout_percentage: float = 100.0  # For percentage-based rollout
    whitelist: List[str] = field(default_factory=list)  # For whitelist strategy
    blacklist: List[str] = field(default_factory=list)  # For blacklist strategy
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    tags: List[str] = field(default_factory=list)
    
    # Evaluation function for custom strategy
    custom_evaluator: Optional[Callable[[Dict[str, Any]], bool]] = None
    
    def __post_init__(self):
        """Validate flag configuration after initialization."""
        if self.rollout_percentage < 0 or self.rollout_percentage > 100:
            raise ValueError(f"Invalid rollout percentage: {self.rollout_percentage}")
        
        if self.strategy == FeatureFlagStrategy.CUSTOM and self.custom_evaluator is None:
            raise ValueError("Custom strategy requires custom_evaluator function")
    
    def evaluate(
        self, 
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> bool:
        """
        Evaluate flag based on strategy and context.
        
        Args:
            context: Evaluation context with relevant metadata
            user_id: User ID for user-specific evaluation
            session_id: Session ID for session-specific evaluation
            
        Returns:
            True if flag should be enabled for this context
        """
        if not self.enabled:
            return False
        
        # Strategy-based evaluation
        if self.strategy == FeatureFlagStrategy.BOOLEAN:
            return True
        
        elif self.strategy == FeatureFlagStrategy.PERCENTAGE:
            # Use consistent hash-based percentage rollout
            hash_source = user_id or session_id or str(context.get('request_id', ''))
            if not hash_source:
                return self.rollout_percentage >= 100.0
            
            # Simple hash-based percentage calculation
            hash_value = abs(hash(f"{self.name}:{hash_source}")) % 100
            return hash_value < self.rollout_percentage
        
        elif self.strategy == FeatureFlagStrategy.WHITELIST:
            if user_id and user_id in self.whitelist:
                return True
            if session_id and session_id in self.whitelist:
                return True
            if context:
                return any(str(v) in self.whitelist for v in context.values())
            return False
        
        elif self.strategy == FeatureFlagStrategy.BLACKLIST:
            if user_id and user_id in self.blacklist:
                return False
            if session_id and session_id in self.blacklist:
                return False
            if context:
                if any(str(v) in self.blacklist for v in context.values()):
                    return False
            return True
        
        elif self.strategy == FeatureFlagStrategy.CUSTOM:
            if self.custom_evaluator:
                eval_context = {
                    'user_id': user_id,
                    'session_id': session_id,
                    'context': context or {},
                    'flag': self
                }
                return self.custom_evaluator(eval_context)
            return False
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert flag to dictionary representation."""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'scope': self.scope.value,
            'strategy': self.strategy.value,
            'description': self.description,
            'dependencies': self.dependencies,
            'incompatible_with': self.incompatible_with,
            'rollout_percentage': self.rollout_percentage,
            'whitelist': self.whitelist,
            'blacklist': self.blacklist,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'created_by': self.created_by,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FeatureFlag:
        """Create flag from dictionary representation."""
        # Convert datetime strings back to datetime objects
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        # Convert enum strings back to enums
        if 'scope' in data and isinstance(data['scope'], str):
            data['scope'] = FeatureFlagScope(data['scope'])
        
        if 'strategy' in data and isinstance(data['strategy'], str):
            data['strategy'] = FeatureFlagStrategy(data['strategy'])
        
        # Remove custom_evaluator from dict (can't be serialized)
        data.pop('custom_evaluator', None)
        
        return cls(**data)


class FeatureFlagManager:
    """
    Unified feature flag management system.
    
    Provides comprehensive flag management including:
    - Flag registration and evaluation
    - Dependency resolution
    - Runtime updates
    - Persistence and loading
    - A/B testing support
    """
    
    def __init__(
        self, 
        config_source: Optional[Union[Dict[str, Any], str, Path]] = None,
        auto_save: bool = True
    ):
        """
        Initialize feature flag manager.
        
        Args:
            config_source: Configuration source (dict, file path, or None)
            auto_save: Whether to automatically save changes to file
        """
        self._flags: Dict[str, FeatureFlag] = {}
        self._flag_hierarchy: Dict[str, Set[str]] = {}  # parent -> children
        self._evaluation_cache: Dict[str, Dict[str, bool]] = {}
        self._cache_ttl_seconds = 60  # Cache evaluations for 1 minute
        self._config_file: Optional[Path] = None
        self._auto_save = auto_save
        
        # Load configuration
        if config_source:
            self.load_configuration(config_source)
        
        # Register default wrapper flags
        self._register_default_flags()
    
    def _register_default_flags(self) -> None:
        """Register default feature flags for common wrapper functionality."""
        
        # Global wrapper system flags
        self.register_flag(FeatureFlag(
            name="wrapper_system_enabled",
            enabled=True,
            scope=FeatureFlagScope.GLOBAL,
            description="Master switch for entire wrapper system"
        ))
        
        self.register_flag(FeatureFlag(
            name="wrapper_monitoring_enabled", 
            enabled=True,
            scope=FeatureFlagScope.GLOBAL,
            description="Enable monitoring for all wrappers"
        ))
        
        self.register_flag(FeatureFlag(
            name="wrapper_fallback_enabled",
            enabled=True,
            scope=FeatureFlagScope.GLOBAL,
            description="Enable fallback mechanisms for all wrappers"
        ))
        
        # Experimental features
        self.register_flag(FeatureFlag(
            name="wrapper_a_b_testing",
            enabled=False,
            scope=FeatureFlagScope.EXPERIMENTAL,
            strategy=FeatureFlagStrategy.PERCENTAGE,
            rollout_percentage=10.0,
            description="Enable A/B testing for wrapper performance"
        ))
        
        self.register_flag(FeatureFlag(
            name="wrapper_advanced_metrics",
            enabled=False,
            scope=FeatureFlagScope.EXPERIMENTAL,
            description="Enable advanced metrics collection"
        ))
    
    def register_flag(
        self,
        flag: FeatureFlag,
        parent_flags: Optional[List[str]] = None
    ) -> None:
        """
        Register a feature flag with optional parent dependencies.
        
        Args:
            flag: Feature flag to register
            parent_flags: List of parent flag names that control this flag
        """
        self._flags[flag.name] = flag
        
        # Update hierarchy
        if parent_flags:
            for parent in parent_flags:
                if parent not in self._flag_hierarchy:
                    self._flag_hierarchy[parent] = set()
                self._flag_hierarchy[parent].add(flag.name)
        
        logger.debug(f"Registered feature flag: {flag.name}")
        
        if self._auto_save:
            self._save_configuration()
    
    def register_wrapper_flags(
        self,
        wrapper_name: str,
        additional_flags: Optional[List[FeatureFlag]] = None
    ) -> None:
        """
        Register standard flags for a wrapper.
        
        Creates common flags that every wrapper needs:
        - {wrapper_name}_enabled
        - {wrapper_name}_monitoring
        - {wrapper_name}_fallback
        
        Args:
            wrapper_name: Name of the wrapper
            additional_flags: Additional wrapper-specific flags
        """
        base_flags = [
            FeatureFlag(
                name=f"{wrapper_name}_enabled",
                enabled=False,
                scope=FeatureFlagScope.WRAPPER,
                dependencies=["wrapper_system_enabled"],
                description=f"Enable {wrapper_name} wrapper"
            ),
            FeatureFlag(
                name=f"{wrapper_name}_monitoring",
                enabled=True,
                scope=FeatureFlagScope.WRAPPER,
                dependencies=[f"{wrapper_name}_enabled", "wrapper_monitoring_enabled"],
                description=f"Enable monitoring for {wrapper_name} wrapper"
            ),
            FeatureFlag(
                name=f"{wrapper_name}_fallback",
                enabled=True,
                scope=FeatureFlagScope.WRAPPER,
                dependencies=[f"{wrapper_name}_enabled", "wrapper_fallback_enabled"],
                description=f"Enable fallback for {wrapper_name} wrapper"
            )
        ]
        
        # Register base flags
        for flag in base_flags:
            self.register_flag(flag)
        
        # Register additional flags
        if additional_flags:
            for flag in additional_flags:
                self.register_flag(flag, parent_flags=[f"{wrapper_name}_enabled"])
    
    def is_enabled(
        self,
        flag_name: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        use_cache: bool = True
    ) -> bool:
        """
        Check if a feature flag is enabled.
        
        Considers flag hierarchy, dependencies, and evaluation strategy.
        
        Args:
            flag_name: Name of the flag to check
            context: Evaluation context
            user_id: User ID for user-specific evaluation
            session_id: Session ID for session-specific evaluation
            use_cache: Whether to use cached evaluation results
            
        Returns:
            True if flag is enabled for this context
        """
        # Check cache first if enabled
        if use_cache:
            cache_key = self._get_cache_key(flag_name, context, user_id, session_id)
            if cache_key in self._evaluation_cache:
                cache_entry = self._evaluation_cache[cache_key]
                if 'expires_at' in cache_entry and datetime.utcnow() < cache_entry['expires_at']:
                    return cache_entry.get('result', False)
        
        # Evaluate flag
        result = self._evaluate_flag(flag_name, context, user_id, session_id)
        
        # Cache result if caching is enabled
        if use_cache:
            cache_key = self._get_cache_key(flag_name, context, user_id, session_id)
            self._evaluation_cache[cache_key] = {
                'result': result,
                'expires_at': datetime.utcnow() + timedelta(seconds=self._cache_ttl_seconds)
            }
        
        return result
    
    def _evaluate_flag(
        self,
        flag_name: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> bool:
        """Internal flag evaluation with dependency resolution."""
        
        if flag_name not in self._flags:
            logger.warning(f"Feature flag not found: {flag_name}")
            return False
        
        flag = self._flags[flag_name]
        
        # Check dependencies first
        for dependency in flag.dependencies:
            if not self._evaluate_flag(dependency, context, user_id, session_id):
                logger.debug(f"Flag {flag_name} disabled due to dependency: {dependency}")
                return False
        
        # Check incompatible flags
        for incompatible in flag.incompatible_with:
            if self._evaluate_flag(incompatible, context, user_id, session_id):
                logger.debug(f"Flag {flag_name} disabled due to incompatible flag: {incompatible}")
                return False
        
        # Evaluate the flag itself
        return flag.evaluate(context, user_id, session_id)
    
    def _get_cache_key(
        self,
        flag_name: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """Generate cache key for flag evaluation."""
        key_parts = [flag_name]
        
        if user_id:
            key_parts.append(f"user:{user_id}")
        
        if session_id:
            key_parts.append(f"session:{session_id}")
        
        if context:
            # Sort context items for consistent key generation
            context_str = ",".join(f"{k}:{v}" for k, v in sorted(context.items()))
            key_parts.append(f"context:{context_str}")
        
        return "|".join(key_parts)
    
    def enable_flag(
        self,
        flag_name: str,
        enable_dependencies: bool = True,
        enabled_by: str = "system"
    ) -> bool:
        """
        Enable a feature flag.
        
        Args:
            flag_name: Name of flag to enable
            enable_dependencies: Whether to enable required dependencies
            enabled_by: Who enabled the flag (for audit trail)
            
        Returns:
            True if flag was successfully enabled
        """
        if flag_name not in self._flags:
            logger.error(f"Cannot enable unknown flag: {flag_name}")
            return False
        
        flag = self._flags[flag_name]
        
        # Enable dependencies first if requested
        if enable_dependencies:
            for dependency in flag.dependencies:
                if not self.enable_flag(dependency, enable_dependencies=True, enabled_by=enabled_by):
                    logger.error(f"Cannot enable {flag_name}: dependency {dependency} failed")
                    return False
        
        # Check for incompatible flags
        for incompatible in flag.incompatible_with:
            if self._flags.get(incompatible, {}).enabled:
                logger.warning(f"Cannot enable {flag_name}: incompatible with active flag {incompatible}")
                return False
        
        # Enable the flag
        flag.enabled = True
        flag.updated_at = datetime.utcnow()
        flag.created_by = enabled_by
        
        # Clear evaluation cache
        self._clear_cache()
        
        logger.info(f"Feature flag enabled: {flag_name} by {enabled_by}")
        
        if self._auto_save:
            self._save_configuration()
        
        return True
    
    def disable_flag(
        self,
        flag_name: str,
        disable_dependents: bool = True,
        disabled_by: str = "system"
    ) -> bool:
        """
        Disable a feature flag.
        
        Args:
            flag_name: Name of flag to disable
            disable_dependents: Whether to disable flags that depend on this one
            disabled_by: Who disabled the flag (for audit trail)
            
        Returns:
            True if flag was successfully disabled
        """
        if flag_name not in self._flags:
            logger.error(f"Cannot disable unknown flag: {flag_name}")
            return False
        
        # Disable dependent flags first if requested
        if disable_dependents and flag_name in self._flag_hierarchy:
            for dependent in self._flag_hierarchy[flag_name]:
                self.disable_flag(dependent, disable_dependents=True, disabled_by=disabled_by)
        
        # Disable the flag
        flag = self._flags[flag_name]
        flag.enabled = False
        flag.updated_at = datetime.utcnow()
        flag.created_by = disabled_by
        
        # Clear evaluation cache
        self._clear_cache()
        
        logger.info(f"Feature flag disabled: {flag_name} by {disabled_by}")
        
        if self._auto_save:
            self._save_configuration()
        
        return True
    
    def update_flag(
        self,
        flag_name: str,
        updates: Dict[str, Any],
        updated_by: str = "system"
    ) -> bool:
        """
        Update flag properties.
        
        Args:
            flag_name: Name of flag to update
            updates: Dictionary of properties to update
            updated_by: Who updated the flag
            
        Returns:
            True if flag was successfully updated
        """
        if flag_name not in self._flags:
            logger.error(f"Cannot update unknown flag: {flag_name}")
            return False
        
        flag = self._flags[flag_name]
        
        # Update allowed properties
        allowed_updates = [
            'enabled', 'description', 'rollout_percentage', 
            'whitelist', 'blacklist', 'tags'
        ]
        
        for key, value in updates.items():
            if key in allowed_updates:
                setattr(flag, key, value)
            else:
                logger.warning(f"Ignoring update to protected property: {key}")
        
        flag.updated_at = datetime.utcnow()
        flag.created_by = updated_by
        
        # Clear evaluation cache
        self._clear_cache()
        
        logger.info(f"Feature flag updated: {flag_name} by {updated_by}")
        
        if self._auto_save:
            self._save_configuration()
        
        return True
    
    def get_flag_status(self, wrapper_name: str) -> Dict[str, Any]:
        """
        Get status of all flags for a wrapper.
        
        Args:
            wrapper_name: Name of the wrapper
            
        Returns:
            Dictionary containing flag status information
        """
        prefix = f"{wrapper_name}_"
        wrapper_flags = {
            name: flag for name, flag in self._flags.items()
            if name.startswith(prefix)
        }
        
        status = {
            "wrapper_name": wrapper_name,
            "total_flags": len(wrapper_flags),
            "enabled_flags": len([f for f in wrapper_flags.values() if f.enabled]),
            "flags": {}
        }
        
        for name, flag in wrapper_flags.items():
            status["flags"][name] = {
                "enabled": flag.enabled,
                "scope": flag.scope.value,
                "strategy": flag.strategy.value,
                "description": flag.description,
                "dependencies": flag.dependencies,
                "updated_at": flag.updated_at.isoformat()
            }
        
        return status
    
    def get_all_flags(self) -> Dict[str, FeatureFlag]:
        """Get all registered flags."""
        return self._flags.copy()
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status for all flags.
        
        Returns:
            Dictionary containing system-wide flag status
        """
        total_flags = len(self._flags)
        enabled_flags = len([f for f in self._flags.values() if f.enabled])
        
        scope_breakdown = {}
        strategy_breakdown = {}
        
        for flag in self._flags.values():
            scope = flag.scope.value
            strategy = flag.strategy.value
            
            scope_breakdown[scope] = scope_breakdown.get(scope, 0) + 1
            strategy_breakdown[strategy] = strategy_breakdown.get(strategy, 0) + 1
        
        return {
            "total_flags": total_flags,
            "enabled_flags": enabled_flags,
            "disabled_flags": total_flags - enabled_flags,
            "scope_breakdown": scope_breakdown,
            "strategy_breakdown": strategy_breakdown,
            "cache_size": len(self._evaluation_cache),
            "hierarchy_size": len(self._flag_hierarchy)
        }
    
    def load_configuration(self, config_source: Union[Dict[str, Any], str, Path]) -> None:
        """
        Load feature flag configuration from source.
        
        Args:
            config_source: Dictionary, file path, or Path object
        """
        if isinstance(config_source, dict):
            self._load_from_dict(config_source)
        elif isinstance(config_source, (str, Path)):
            self._load_from_file(Path(config_source))
        else:
            raise ValueError(f"Unsupported configuration source type: {type(config_source)}")
    
    def _load_from_dict(self, config: Dict[str, Any]) -> None:
        """Load flags from dictionary configuration."""
        flags_config = config.get('feature_flags', {})
        
        for flag_name, flag_data in flags_config.items():
            try:
                flag_data['name'] = flag_name  # Ensure name is set
                flag = FeatureFlag.from_dict(flag_data)
                self._flags[flag_name] = flag
            except Exception as e:
                logger.error(f"Failed to load flag {flag_name}: {e}")
        
        logger.info(f"Loaded {len(flags_config)} feature flags from configuration")
    
    def _load_from_file(self, file_path: Path) -> None:
        """Load flags from JSON file."""
        self._config_file = file_path
        
        if not file_path.exists():
            logger.info(f"Feature flag config file not found: {file_path}")
            return
        
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
            self._load_from_dict(config)
            logger.info(f"Loaded feature flags from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load feature flags from {file_path}: {e}")
    
    def _save_configuration(self) -> None:
        """Save current flags to configuration file."""
        if not self._config_file:
            return
        
        try:
            config = {
                "feature_flags": {
                    name: flag.to_dict() 
                    for name, flag in self._flags.items()
                }
            }
            
            # Ensure directory exists
            self._config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self._config_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            logger.debug(f"Saved feature flags to {self._config_file}")
        except Exception as e:
            logger.error(f"Failed to save feature flags: {e}")
    
    def _clear_cache(self) -> None:
        """Clear evaluation cache."""
        self._evaluation_cache.clear()
        logger.debug("Cleared feature flag evaluation cache")
    
    def export_configuration(self) -> Dict[str, Any]:
        """
        Export current configuration as dictionary.
        
        Returns:
            Dictionary containing all flags and their configurations
        """
        return {
            "feature_flags": {
                name: flag.to_dict() 
                for name, flag in self._flags.items()
            }
        }