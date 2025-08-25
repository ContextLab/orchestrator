"""
Administrative interface for configuration management.

This module provides web-based and programmatic interfaces for managing
wrapper configurations, credentials, environment settings, and system
administration tasks.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from threading import Lock

# Optional Flask imports
try:
    from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    logger.warning("Flask not available - admin web interface disabled")

from ..core.wrapper_config import (
    ConfigurationManager, BaseWrapperConfig, ExternalToolConfig,
    ConfigValidationError, ConfigSource
)
from ..core.credential_manager import CredentialManager, CredentialConfig
from ..core.environment_config import (
    EnvironmentConfigManager, Environment, EnvironmentOverride, EnvironmentProfile
)

logger = logging.getLogger(__name__)


@dataclass
class AdminConfig:
    """Configuration for admin interface."""
    
    host: str = "127.0.0.1"  # More restrictive default for admin
    port: int = 5001
    debug: bool = False
    auth_required: bool = True
    secret_key: str = "admin-secret-change-in-production"
    session_timeout_minutes: int = 60
    audit_logging: bool = True
    max_config_history: int = 50
    backup_retention_days: int = 30


class ConfigurationAdminInterface:
    """Administrative interface for configuration management."""
    
    def __init__(
        self,
        config_manager: ConfigurationManager,
        credential_manager: Optional[CredentialManager] = None,
        environment_manager: Optional[EnvironmentConfigManager] = None,
        config: Optional[AdminConfig] = None
    ):
        """
        Initialize configuration admin interface.
        
        Args:
            config_manager: Configuration manager instance
            credential_manager: Optional credential manager
            environment_manager: Optional environment configuration manager
            config: Admin interface configuration
        """
        self.config_manager = config_manager
        self.credential_manager = credential_manager
        self.environment_manager = environment_manager
        self.config = config or AdminConfig()
        
        # Flask app for web interface
        self.app: Optional[Flask] = None
        if HAS_FLASK:
            self._setup_flask_app()
        
        # Audit logging
        self._audit_log: List[Dict[str, Any]] = []
        self._audit_lock = Lock()
        
        # Configuration history
        self._config_history: Dict[str, List[Dict[str, Any]]] = {}
        self._history_lock = Lock()
        
        logger.info("Initialized configuration admin interface")
    
    def start_web_interface(self) -> None:
        """Start the web-based admin interface."""
        
        if not self.app:
            raise RuntimeError("Flask not available - cannot start web interface")
        
        logger.info(f"Starting admin interface on {self.config.host}:{self.config.port}")
        
        self.app.run(
            host=self.config.host,
            port=self.config.port,
            debug=self.config.debug
        )
    
    # Programmatic Configuration Management API
    
    def list_wrappers(self) -> Dict[str, Any]:
        """List all registered wrapper configurations."""
        
        try:
            configs = self.config_manager.get_all_configs()
            system_summary = self.config_manager.get_system_summary()
            
            wrapper_info = {}
            for name, config in configs.items():
                wrapper_info[name] = {
                    "config_type": config.__class__.__name__,
                    "enabled": config.enabled,
                    "is_valid": config.is_valid,
                    "last_updated": config._last_updated.isoformat(),
                    "updated_by": config._updated_by,
                    "config_source": config._config_source.value
                }
            
            return {
                "wrappers": wrapper_info,
                "system_summary": system_summary,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to list wrappers: {e}")
            return {"error": str(e)}
    
    def get_wrapper_config(
        self, 
        wrapper_name: str,
        include_sensitive: bool = False,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Get configuration for a specific wrapper."""
        
        try:
            config = self.config_manager.get_config(wrapper_name)
            
            if not config:
                return {"error": f"Configuration not found for wrapper: {wrapper_name}"}
            
            config_dict = config.to_dict(
                include_metadata=include_metadata,
                mask_sensitive=not include_sensitive
            )
            
            # Add validation status
            validation_result = self._validate_wrapper_config(wrapper_name)
            
            result = {
                "wrapper_name": wrapper_name,
                "config": config_dict,
                "validation": validation_result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add environment overrides if available
            if self.environment_manager:
                env_status = self.environment_manager.validate_config(wrapper_name)
                result["environment_status"] = env_status
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get config for {wrapper_name}: {e}")
            return {"error": str(e)}
    
    def update_wrapper_config(
        self,
        wrapper_name: str,
        updates: Dict[str, Any],
        updated_by: str = "admin",
        create_backup: bool = True
    ) -> Dict[str, Any]:
        """Update configuration for a wrapper."""
        
        try:
            # Create backup if requested
            if create_backup:
                self._create_config_backup(wrapper_name, updated_by)
            
            # Record audit log
            self._record_audit_action(
                action="update_config",
                wrapper_name=wrapper_name,
                updated_by=updated_by,
                details={"updates": updates}
            )
            
            # Update configuration
            success = self.config_manager.update_config(
                wrapper_name=wrapper_name,
                updates=updates,
                updated_by=updated_by,
                save_to_file=True
            )
            
            if success:
                # Record in history
                self._record_config_history(wrapper_name, updates, updated_by)
                
                result = {
                    "success": True,
                    "message": f"Configuration updated for {wrapper_name}",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Validate updated configuration
                validation_result = self._validate_wrapper_config(wrapper_name)
                result["validation"] = validation_result
                
                return result
            else:
                return {
                    "success": False,
                    "error": "Failed to update configuration",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to update config for {wrapper_name}: {e}")
            return {"success": False, "error": str(e)}
    
    def create_wrapper_config(
        self,
        wrapper_name: str,
        config_type: str,
        config_data: Dict[str, Any],
        created_by: str = "admin"
    ) -> Dict[str, Any]:
        """Create a new wrapper configuration."""
        
        try:
            # Map config type to class
            config_classes = {
                "ExternalToolConfig": ExternalToolConfig,
                "BaseWrapperConfig": BaseWrapperConfig
            }
            
            if config_type not in config_classes:
                return {
                    "success": False,
                    "error": f"Unknown configuration type: {config_type}"
                }
            
            # Register config type with manager
            self.config_manager.register_config_type(wrapper_name, config_classes[config_type])
            
            # Create configuration
            config = self.config_manager.create_config(wrapper_name, config_data)
            
            if config:
                # Record audit log
                self._record_audit_action(
                    action="create_config",
                    wrapper_name=wrapper_name,
                    updated_by=created_by,
                    details={"config_type": config_type, "config_data": config_data}
                )
                
                result = {
                    "success": True,
                    "message": f"Configuration created for {wrapper_name}",
                    "config_type": config_type,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Validate new configuration
                validation_result = self._validate_wrapper_config(wrapper_name)
                result["validation"] = validation_result
                
                return result
            else:
                return {
                    "success": False,
                    "error": "Failed to create configuration",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to create config for {wrapper_name}: {e}")
            return {"success": False, "error": str(e)}
    
    def validate_all_configurations(self) -> Dict[str, Any]:
        """Validate all wrapper configurations."""
        
        try:
            validation_results = self.config_manager.validate_all_configs()
            
            # Get detailed validation for each wrapper
            detailed_results = {}
            for wrapper_name in validation_results:
                detailed_results[wrapper_name] = self._validate_wrapper_config(wrapper_name)
            
            summary = {
                "total_wrappers": len(validation_results),
                "valid_wrappers": sum(1 for v in validation_results.values() if v),
                "invalid_wrappers": sum(1 for v in validation_results.values() if not v),
                "validation_timestamp": datetime.utcnow().isoformat()
            }
            
            return {
                "summary": summary,
                "results": validation_results,
                "detailed_results": detailed_results
            }
            
        except Exception as e:
            logger.error(f"Failed to validate configurations: {e}")
            return {"error": str(e)}
    
    # Credential Management API
    
    def manage_credentials(
        self,
        action: str,
        service: str,
        key: str,
        value: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Manage credentials through admin interface."""
        
        if not self.credential_manager:
            return {"error": "Credential manager not available"}
        
        try:
            if action == "store":
                if not value:
                    return {"error": "Value required for store action"}
                
                success = self.credential_manager.store_credential(
                    service=service,
                    key=key,
                    value=value,
                    description=kwargs.get("description", ""),
                    tags=kwargs.get("tags", []),
                    expires_in_days=kwargs.get("expires_in_days"),
                    auto_rotate=kwargs.get("auto_rotate", False)
                )
                
                self._record_audit_action(
                    action="store_credential",
                    wrapper_name=service,
                    updated_by=kwargs.get("updated_by", "admin"),
                    details={"key": key, "description": kwargs.get("description", "")}
                )
                
                return {
                    "success": success,
                    "message": f"Credential {'stored' if success else 'failed to store'}",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            elif action == "retrieve":
                value = self.credential_manager.retrieve_credential(service, key)
                
                return {
                    "success": value is not None,
                    "has_value": value is not None,
                    "message": f"Credential {'found' if value else 'not found'}",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            elif action == "delete":
                success = self.credential_manager.delete_credential(service, key)
                
                self._record_audit_action(
                    action="delete_credential",
                    wrapper_name=service,
                    updated_by=kwargs.get("updated_by", "admin"),
                    details={"key": key}
                )
                
                return {
                    "success": success,
                    "message": f"Credential {'deleted' if success else 'failed to delete'}",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            elif action == "list":
                credentials = self.credential_manager.list_credentials(
                    service=service if service != "*" else None,
                    include_expired=kwargs.get("include_expired", False)
                )
                
                return {
                    "success": True,
                    "credentials": [asdict(cred) for cred in credentials],
                    "count": len(credentials),
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            else:
                return {"error": f"Unknown credential action: {action}"}
                
        except Exception as e:
            logger.error(f"Failed credential operation {action}: {e}")
            return {"error": str(e)}
    
    # Environment Configuration Management
    
    def manage_environment_overrides(
        self,
        action: str,
        wrapper_name: str,
        environment: Optional[str] = None,
        config_path: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Manage environment-specific configuration overrides."""
        
        if not self.environment_manager:
            return {"error": "Environment manager not available"}
        
        try:
            if action == "add":
                if not all([environment, config_path, value is not None]):
                    return {"error": "environment, config_path, and value required for add action"}
                
                env_enum = Environment.from_string(environment)
                expires_at = None
                if kwargs.get("expires_in_hours"):
                    expires_at = datetime.utcnow() + timedelta(hours=kwargs["expires_in_hours"])
                
                override = EnvironmentOverride(
                    environment=env_enum,
                    config_path=config_path,
                    value=value,
                    reason=kwargs.get("reason", "Admin override"),
                    created_by=kwargs.get("created_by", "admin"),
                    expires_at=expires_at,
                    tags=kwargs.get("tags", [])
                )
                
                self.environment_manager.add_override(wrapper_name, override)
                
                self._record_audit_action(
                    action="add_env_override",
                    wrapper_name=wrapper_name,
                    updated_by=kwargs.get("created_by", "admin"),
                    details={
                        "environment": environment,
                        "config_path": config_path,
                        "value": value,
                        "reason": override.reason
                    }
                )
                
                return {
                    "success": True,
                    "message": f"Environment override added for {wrapper_name}",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            elif action == "remove":
                if not all([environment, config_path]):
                    return {"error": "environment and config_path required for remove action"}
                
                env_enum = Environment.from_string(environment)
                success = self.environment_manager.remove_override(wrapper_name, env_enum, config_path)
                
                if success:
                    self._record_audit_action(
                        action="remove_env_override",
                        wrapper_name=wrapper_name,
                        updated_by=kwargs.get("updated_by", "admin"),
                        details={"environment": environment, "config_path": config_path}
                    )
                
                return {
                    "success": success,
                    "message": f"Environment override {'removed' if success else 'not found'}",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            elif action == "list":
                env_filter = Environment.from_string(environment) if environment else None
                overrides = self.environment_manager.get_overrides(
                    wrapper_name=wrapper_name if wrapper_name != "*" else None,
                    environment=env_filter,
                    include_expired=kwargs.get("include_expired", False)
                )
                
                # Convert to serializable format
                serializable_overrides = {}
                for wrapper, wrapper_overrides in overrides.items():
                    serializable_overrides[wrapper] = [
                        override.to_dict() for override in wrapper_overrides
                    ]
                
                return {
                    "success": True,
                    "overrides": serializable_overrides,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            elif action == "validate":
                env_filter = Environment.from_string(environment) if environment else None
                validation_result = self.environment_manager.validate_config(wrapper_name, env_filter)
                
                return {
                    "success": True,
                    "validation": validation_result,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            else:
                return {"error": f"Unknown environment action: {action}"}
                
        except Exception as e:
            logger.error(f"Failed environment operation {action}: {e}")
            return {"error": str(e)}
    
    # Audit and History
    
    def get_audit_log(
        self,
        limit: Optional[int] = None,
        wrapper_name: Optional[str] = None,
        action_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get audit log entries."""
        
        try:
            with self._audit_lock:
                entries = self._audit_log.copy()
            
            # Apply filters
            if wrapper_name:
                entries = [e for e in entries if e.get("wrapper_name") == wrapper_name]
            
            if action_filter:
                entries = [e for e in entries if e.get("action") == action_filter]
            
            # Sort by timestamp (newest first)
            entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            if limit:
                entries = entries[:limit]
            
            return {
                "success": True,
                "entries": entries,
                "total_entries": len(self._audit_log),
                "filtered_entries": len(entries),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get audit log: {e}")
            return {"error": str(e)}
    
    def get_config_history(self, wrapper_name: str) -> Dict[str, Any]:
        """Get configuration change history for a wrapper."""
        
        try:
            with self._history_lock:
                history = self._config_history.get(wrapper_name, [])
            
            return {
                "success": True,
                "wrapper_name": wrapper_name,
                "history": history,
                "total_changes": len(history),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get config history for {wrapper_name}: {e}")
            return {"error": str(e)}
    
    # System Administration
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        try:
            # Configuration manager status
            config_summary = self.config_manager.get_system_summary()
            
            # Environment manager status
            env_summary = {}
            if self.environment_manager:
                env_summary = self.environment_manager.get_environment_summary()
            
            # Credential manager status
            cred_summary = {}
            if self.credential_manager:
                try:
                    credentials = self.credential_manager.list_credentials()
                    cred_summary = {
                        "total_credentials": len(credentials),
                        "expired_credentials": len([c for c in credentials if c.expires_at and c.expires_at < datetime.utcnow()]),
                        "auto_rotate_enabled": len([c for c in credentials if c.auto_rotate])
                    }
                except Exception as e:
                    cred_summary = {"error": str(e)}
            
            # Audit log status
            with self._audit_lock:
                audit_summary = {
                    "total_audit_entries": len(self._audit_log),
                    "recent_entries": len([
                        e for e in self._audit_log
                        if datetime.fromisoformat(e.get("timestamp", "1970-01-01")) > 
                        datetime.utcnow() - timedelta(hours=24)
                    ])
                }
            
            # Config history status
            with self._history_lock:
                history_summary = {
                    "wrappers_with_history": len(self._config_history),
                    "total_history_entries": sum(len(h) for h in self._config_history.values())
                }
            
            return {
                "success": True,
                "timestamp": datetime.utcnow().isoformat(),
                "configuration_manager": config_summary,
                "environment_manager": env_summary,
                "credential_manager": cred_summary,
                "audit_log": audit_summary,
                "config_history": history_summary,
                "admin_interface": {
                    "version": "1.0.0",
                    "host": self.config.host,
                    "port": self.config.port,
                    "auth_required": self.config.auth_required,
                    "audit_logging": self.config.audit_logging
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {"error": str(e)}
    
    # Helper methods
    
    def _validate_wrapper_config(self, wrapper_name: str) -> Dict[str, Any]:
        """Validate a wrapper configuration."""
        
        try:
            config = self.config_manager.get_config(wrapper_name)
            
            if not config:
                return {
                    "valid": False,
                    "errors": [f"Configuration not found for {wrapper_name}"]
                }
            
            config.validate()
            
            return {
                "valid": True,
                "errors": [],
                "warnings": [],
                "config_type": config.__class__.__name__
            }
            
        except ConfigValidationError as e:
            return {
                "valid": False,
                "errors": [str(e)],
                "warnings": [],
                "config_type": config.__class__.__name__ if config else "Unknown"
            }
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation failed: {e}"],
                "warnings": [],
                "config_type": "Unknown"
            }
    
    def _record_audit_action(
        self,
        action: str,
        wrapper_name: str,
        updated_by: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record an audit log entry."""
        
        if not self.config.audit_logging:
            return
        
        with self._audit_lock:
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "action": action,
                "wrapper_name": wrapper_name,
                "updated_by": updated_by,
                "details": details or {}
            }
            
            self._audit_log.append(entry)
            
            # Keep only recent entries to manage memory
            max_entries = 10000
            if len(self._audit_log) > max_entries:
                self._audit_log = self._audit_log[-max_entries:]
    
    def _record_config_history(
        self,
        wrapper_name: str,
        updates: Dict[str, Any],
        updated_by: str
    ) -> None:
        """Record configuration change in history."""
        
        with self._history_lock:
            if wrapper_name not in self._config_history:
                self._config_history[wrapper_name] = []
            
            history_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "updates": updates,
                "updated_by": updated_by
            }
            
            self._config_history[wrapper_name].append(history_entry)
            
            # Keep only recent history entries
            if len(self._config_history[wrapper_name]) > self.config.max_config_history:
                self._config_history[wrapper_name] = self._config_history[wrapper_name][-self.config.max_config_history:]
    
    def _create_config_backup(self, wrapper_name: str, created_by: str) -> None:
        """Create a backup of current configuration."""
        
        try:
            config = self.config_manager.get_config(wrapper_name)
            if not config:
                return
            
            backup_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "wrapper_name": wrapper_name,
                "created_by": created_by,
                "config": config.to_dict(include_metadata=True, mask_sensitive=False)
            }
            
            # Save backup to file
            backup_dir = Path("config") / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            backup_file = backup_dir / f"{wrapper_name}_{int(datetime.utcnow().timestamp())}.json"
            
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            logger.info(f"Created configuration backup: {backup_file}")
            
        except Exception as e:
            logger.error(f"Failed to create config backup for {wrapper_name}: {e}")
    
    def _setup_flask_app(self) -> None:
        """Setup Flask application for web interface."""
        
        self.app = Flask(__name__, template_folder='templates', static_folder='static')
        self.app.secret_key = self.config.secret_key
        
        # Basic routes - full implementation would include authentication, forms, etc.
        @self.app.route('/')
        def admin_home():
            """Admin home page."""
            return render_template('admin_home.html')
        
        @self.app.route('/api/wrappers')
        def api_list_wrappers():
            """API endpoint to list wrappers."""
            return jsonify(self.list_wrappers())
        
        @self.app.route('/api/wrapper/<wrapper_name>')
        def api_get_wrapper_config(wrapper_name: str):
            """API endpoint to get wrapper configuration."""
            include_sensitive = request.args.get('include_sensitive', 'false').lower() == 'true'
            return jsonify(self.get_wrapper_config(wrapper_name, include_sensitive))
        
        @self.app.route('/api/system-status')
        def api_system_status():
            """API endpoint for system status."""
            return jsonify(self.get_system_status())
        
        logger.debug("Flask app setup completed for admin interface")


# Factory function for easy instantiation
def create_config_admin_interface(
    config_manager: ConfigurationManager,
    credential_manager: Optional[CredentialManager] = None,
    environment_manager: Optional[EnvironmentConfigManager] = None,
    config: Optional[AdminConfig] = None
) -> ConfigurationAdminInterface:
    """Create a configuration admin interface."""
    return ConfigurationAdminInterface(
        config_manager, credential_manager, environment_manager, config
    )