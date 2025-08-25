"""
Secure credential management system for external tool integrations.

This module provides secure storage, retrieval, and management of API keys,
tokens, and other sensitive credentials used by wrapper integrations.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import optional dependencies with fallbacks
try:
    from cryptography.fernet import Fernet
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False
    logger.warning("cryptography package not available - credentials will not be encrypted")

try:
    import keyring
    HAS_KEYRING = True
except ImportError:
    HAS_KEYRING = False
    logger.warning("keyring package not available - falling back to file storage")

logger = logging.getLogger(__name__)


class CredentialError(Exception):
    """Raised when credential operations fail."""
    pass


@dataclass
class CredentialMetadata:
    """Metadata for stored credentials."""
    
    service: str
    key: str
    created_at: datetime
    last_accessed: datetime
    expires_at: Optional[datetime] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)
    auto_rotate: bool = False
    rotation_days: int = 90


@dataclass
class CredentialConfig:
    """Configuration for credential management."""
    
    storage_backend: str = "auto"  # auto, keyring, file, environment
    encryption_enabled: bool = True
    key_rotation_days: int = 90
    audit_logging: bool = True
    secure_file_path: Optional[Path] = None
    auto_cleanup_days: int = 365
    
    def __post_init__(self):
        """Validate and setup configuration."""
        if self.storage_backend == "auto":
            if HAS_KEYRING:
                self.storage_backend = "keyring"
            elif HAS_CRYPTOGRAPHY:
                self.storage_backend = "file"
            else:
                self.storage_backend = "environment"
                logger.warning("No secure storage available - using environment variables")
        
        if self.encryption_enabled and not HAS_CRYPTOGRAPHY:
            self.encryption_enabled = False
            logger.warning("Cryptography not available - encryption disabled")
        
        if self.secure_file_path is None:
            self.secure_file_path = Path.home() / ".orchestrator" / "credentials"


class CredentialStorage(ABC):
    """Abstract base class for credential storage backends."""
    
    @abstractmethod
    def store(self, service: str, key: str, value: str, metadata: CredentialMetadata) -> bool:
        """Store a credential."""
        pass
    
    @abstractmethod
    def retrieve(self, service: str, key: str) -> Optional[str]:
        """Retrieve a credential."""
        pass
    
    @abstractmethod
    def delete(self, service: str, key: str) -> bool:
        """Delete a credential."""
        pass
    
    @abstractmethod
    def list_credentials(self, service: Optional[str] = None) -> List[CredentialMetadata]:
        """List stored credentials."""
        pass


class KeyringStorage(CredentialStorage):
    """Keyring-based credential storage."""
    
    def __init__(self, service_prefix: str = "orchestrator"):
        if not HAS_KEYRING:
            raise CredentialError("Keyring not available")
        self.service_prefix = service_prefix
        self._metadata_cache: Dict[str, CredentialMetadata] = {}
    
    def store(self, service: str, key: str, value: str, metadata: CredentialMetadata) -> bool:
        """Store credential in keyring."""
        try:
            full_service = f"{self.service_prefix}:{service}"
            keyring.set_password(full_service, key, value)
            
            # Store metadata separately
            metadata_key = f"{key}_metadata"
            metadata_json = json.dumps({
                "service": metadata.service,
                "key": metadata.key,
                "created_at": metadata.created_at.isoformat(),
                "last_accessed": metadata.last_accessed.isoformat(),
                "expires_at": metadata.expires_at.isoformat() if metadata.expires_at else None,
                "description": metadata.description,
                "tags": metadata.tags,
                "auto_rotate": metadata.auto_rotate,
                "rotation_days": metadata.rotation_days
            })
            keyring.set_password(full_service, metadata_key, metadata_json)
            
            self._metadata_cache[f"{service}:{key}"] = metadata
            return True
            
        except Exception as e:
            logger.error(f"Failed to store credential in keyring: {e}")
            return False
    
    def retrieve(self, service: str, key: str) -> Optional[str]:
        """Retrieve credential from keyring."""
        try:
            full_service = f"{self.service_prefix}:{service}"
            credential = keyring.get_password(full_service, key)
            
            if credential:
                # Update last accessed time
                self._update_access_time(service, key)
            
            return credential
            
        except Exception as e:
            logger.error(f"Failed to retrieve credential from keyring: {e}")
            return None
    
    def delete(self, service: str, key: str) -> bool:
        """Delete credential from keyring."""
        try:
            full_service = f"{self.service_prefix}:{service}"
            keyring.delete_password(full_service, key)
            keyring.delete_password(full_service, f"{key}_metadata")
            
            cache_key = f"{service}:{key}"
            if cache_key in self._metadata_cache:
                del self._metadata_cache[cache_key]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete credential from keyring: {e}")
            return False
    
    def list_credentials(self, service: Optional[str] = None) -> List[CredentialMetadata]:
        """List stored credentials (limited functionality with keyring)."""
        # Keyring doesn't provide a way to list all credentials
        # Return cached metadata
        results = []
        for cache_key, metadata in self._metadata_cache.items():
            if service is None or metadata.service == service:
                results.append(metadata)
        return results
    
    def _update_access_time(self, service: str, key: str) -> None:
        """Update last accessed time for credential."""
        cache_key = f"{service}:{key}"
        if cache_key in self._metadata_cache:
            self._metadata_cache[cache_key].last_accessed = datetime.utcnow()


class FileStorage(CredentialStorage):
    """File-based credential storage with encryption."""
    
    def __init__(self, file_path: Path, encryption_key: Optional[bytes] = None):
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._encryption_key = encryption_key
        if self._encryption_key and HAS_CRYPTOGRAPHY:
            self._cipher = Fernet(self._encryption_key)
        else:
            self._cipher = None
        
        self._data: Dict[str, Any] = {}
        self._load_data()
    
    def store(self, service: str, key: str, value: str, metadata: CredentialMetadata) -> bool:
        """Store credential in encrypted file."""
        try:
            # Encrypt value if encryption is enabled
            if self._cipher:
                encrypted_value = self._cipher.encrypt(value.encode()).decode()
            else:
                encrypted_value = value
            
            # Store credential and metadata
            service_key = f"{service}:{key}"
            self._data[service_key] = {
                "value": encrypted_value,
                "metadata": {
                    "service": metadata.service,
                    "key": metadata.key,
                    "created_at": metadata.created_at.isoformat(),
                    "last_accessed": metadata.last_accessed.isoformat(),
                    "expires_at": metadata.expires_at.isoformat() if metadata.expires_at else None,
                    "description": metadata.description,
                    "tags": metadata.tags,
                    "auto_rotate": metadata.auto_rotate,
                    "rotation_days": metadata.rotation_days
                }
            }
            
            self._save_data()
            return True
            
        except Exception as e:
            logger.error(f"Failed to store credential in file: {e}")
            return False
    
    def retrieve(self, service: str, key: str) -> Optional[str]:
        """Retrieve credential from encrypted file."""
        try:
            service_key = f"{service}:{key}"
            if service_key not in self._data:
                return None
            
            credential_data = self._data[service_key]
            encrypted_value = credential_data["value"]
            
            # Decrypt value if encryption is enabled
            if self._cipher and encrypted_value:
                try:
                    value = self._cipher.decrypt(encrypted_value.encode()).decode()
                except Exception:
                    # Value might not be encrypted (backwards compatibility)
                    value = encrypted_value
            else:
                value = encrypted_value
            
            # Update last accessed time
            credential_data["metadata"]["last_accessed"] = datetime.utcnow().isoformat()
            self._save_data()
            
            return value
            
        except Exception as e:
            logger.error(f"Failed to retrieve credential from file: {e}")
            return None
    
    def delete(self, service: str, key: str) -> bool:
        """Delete credential from file."""
        try:
            service_key = f"{service}:{key}"
            if service_key in self._data:
                del self._data[service_key]
                self._save_data()
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete credential from file: {e}")
            return False
    
    def list_credentials(self, service: Optional[str] = None) -> List[CredentialMetadata]:
        """List stored credentials."""
        results = []
        
        for service_key, credential_data in self._data.items():
            metadata_dict = credential_data["metadata"]
            
            if service is None or metadata_dict["service"] == service:
                metadata = CredentialMetadata(
                    service=metadata_dict["service"],
                    key=metadata_dict["key"],
                    created_at=datetime.fromisoformat(metadata_dict["created_at"]),
                    last_accessed=datetime.fromisoformat(metadata_dict["last_accessed"]),
                    expires_at=datetime.fromisoformat(metadata_dict["expires_at"]) if metadata_dict.get("expires_at") else None,
                    description=metadata_dict.get("description", ""),
                    tags=metadata_dict.get("tags", []),
                    auto_rotate=metadata_dict.get("auto_rotate", False),
                    rotation_days=metadata_dict.get("rotation_days", 90)
                )
                results.append(metadata)
        
        return results
    
    def _load_data(self) -> None:
        """Load credential data from file."""
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r') as f:
                    self._data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load credential file: {e}")
                self._data = {}
        else:
            self._data = {}
    
    def _save_data(self) -> None:
        """Save credential data to file."""
        try:
            # Ensure directory exists
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write with restricted permissions
            with open(self.file_path, 'w') as f:
                json.dump(self._data, f, indent=2)
            
            # Set restrictive file permissions (owner read/write only)
            os.chmod(self.file_path, 0o600)
            
        except Exception as e:
            logger.error(f"Failed to save credential file: {e}")


class EnvironmentStorage(CredentialStorage):
    """Environment variable based credential storage."""
    
    def __init__(self, prefix: str = "ORCHESTRATOR"):
        self.prefix = prefix
        self._metadata: Dict[str, CredentialMetadata] = {}
    
    def store(self, service: str, key: str, value: str, metadata: CredentialMetadata) -> bool:
        """Store credential as environment variable."""
        try:
            env_var = f"{self.prefix}_{service.upper()}_{key.upper()}"
            os.environ[env_var] = value
            
            # Store metadata in memory only
            cache_key = f"{service}:{key}"
            self._metadata[cache_key] = metadata
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store credential in environment: {e}")
            return False
    
    def retrieve(self, service: str, key: str) -> Optional[str]:
        """Retrieve credential from environment variable."""
        try:
            env_var = f"{self.prefix}_{service.upper()}_{key.upper()}"
            value = os.environ.get(env_var)
            
            if value:
                # Update access time
                cache_key = f"{service}:{key}"
                if cache_key in self._metadata:
                    self._metadata[cache_key].last_accessed = datetime.utcnow()
            
            return value
            
        except Exception as e:
            logger.error(f"Failed to retrieve credential from environment: {e}")
            return None
    
    def delete(self, service: str, key: str) -> bool:
        """Delete credential from environment variable."""
        try:
            env_var = f"{self.prefix}_{service.upper()}_{key.upper()}"
            if env_var in os.environ:
                del os.environ[env_var]
            
            cache_key = f"{service}:{key}"
            if cache_key in self._metadata:
                del self._metadata[cache_key]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete credential from environment: {e}")
            return False
    
    def list_credentials(self, service: Optional[str] = None) -> List[CredentialMetadata]:
        """List stored credentials."""
        results = []
        for cache_key, metadata in self._metadata.items():
            if service is None or metadata.service == service:
                results.append(metadata)
        return results


class CredentialManager:
    """Comprehensive credential management system."""
    
    def __init__(self, config: CredentialConfig):
        self.config = config
        self._audit_log: List[Dict[str, Any]] = []
        
        # Initialize storage backend
        self._storage = self._create_storage_backend()
        
        # Initialize encryption key if needed
        self._encryption_key = self._get_or_create_encryption_key()
        
        logger.info(f"Initialized credential manager with {config.storage_backend} backend")
    
    def store_credential(
        self,
        service: str,
        key: str,
        value: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
        auto_rotate: bool = False
    ) -> bool:
        """Store a credential securely."""
        try:
            # Create metadata
            now = datetime.utcnow()
            expires_at = now + timedelta(days=expires_in_days) if expires_in_days else None
            
            metadata = CredentialMetadata(
                service=service,
                key=key,
                created_at=now,
                last_accessed=now,
                expires_at=expires_at,
                description=description,
                tags=tags or [],
                auto_rotate=auto_rotate,
                rotation_days=self.config.key_rotation_days
            )
            
            # Store credential
            success = self._storage.store(service, key, value, metadata)
            
            if success:
                # Audit logging
                if self.config.audit_logging:
                    self._audit_log.append({
                        "action": "store",
                        "service": service,
                        "key": key,
                        "timestamp": now.isoformat(),
                        "description": description,
                        "tags": tags or []
                    })
                
                logger.info(f"Stored credential for {service}/{key}")
            else:
                logger.error(f"Failed to store credential for {service}/{key}")
            
            return success
            
        except Exception as e:
            logger.error(f"Exception storing credential for {service}/{key}: {e}")
            return False
    
    def retrieve_credential(self, service: str, key: str) -> Optional[str]:
        """Retrieve a credential."""
        try:
            value = self._storage.retrieve(service, key)
            
            if value and self.config.audit_logging:
                self._audit_log.append({
                    "action": "retrieve",
                    "service": service,
                    "key": key,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            return value
            
        except Exception as e:
            logger.error(f"Exception retrieving credential for {service}/{key}: {e}")
            return None
    
    def delete_credential(self, service: str, key: str) -> bool:
        """Delete a credential."""
        try:
            success = self._storage.delete(service, key)
            
            if success and self.config.audit_logging:
                self._audit_log.append({
                    "action": "delete",
                    "service": service,
                    "key": key,
                    "timestamp": datetime.utcnow().isoformat()
                })
                logger.info(f"Deleted credential for {service}/{key}")
            
            return success
            
        except Exception as e:
            logger.error(f"Exception deleting credential for {service}/{key}: {e}")
            return False
    
    def list_credentials(
        self, 
        service: Optional[str] = None,
        include_expired: bool = False
    ) -> List[CredentialMetadata]:
        """List stored credentials."""
        try:
            credentials = self._storage.list_credentials(service)
            
            if not include_expired:
                now = datetime.utcnow()
                credentials = [
                    c for c in credentials 
                    if c.expires_at is None or c.expires_at > now
                ]
            
            return credentials
            
        except Exception as e:
            logger.error(f"Exception listing credentials: {e}")
            return []
    
    def rotate_credentials(self, service: Optional[str] = None) -> Dict[str, bool]:
        """Rotate credentials that are due for rotation."""
        results = {}
        
        try:
            credentials = self.list_credentials(service)
            now = datetime.utcnow()
            
            for credential in credentials:
                if credential.auto_rotate:
                    # Check if rotation is due
                    rotation_due = (
                        now - credential.created_at > 
                        timedelta(days=credential.rotation_days)
                    )
                    
                    if rotation_due:
                        # For now, just log that rotation is needed
                        # In a real implementation, this would integrate with external
                        # tool APIs to generate new credentials
                        logger.info(f"Credential rotation due for {credential.service}/{credential.key}")
                        results[f"{credential.service}:{credential.key}"] = False
                    else:
                        results[f"{credential.service}:{credential.key}"] = True
            
        except Exception as e:
            logger.error(f"Exception during credential rotation: {e}")
        
        return results
    
    def cleanup_expired_credentials(self) -> int:
        """Remove expired credentials."""
        cleaned = 0
        
        try:
            credentials = self.list_credentials(include_expired=True)
            now = datetime.utcnow()
            
            for credential in credentials:
                if credential.expires_at and credential.expires_at < now:
                    if self.delete_credential(credential.service, credential.key):
                        cleaned += 1
            
            logger.info(f"Cleaned up {cleaned} expired credentials")
            
        except Exception as e:
            logger.error(f"Exception during credential cleanup: {e}")
        
        return cleaned
    
    def get_audit_log(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        if limit:
            return self._audit_log[-limit:]
        return self._audit_log.copy()
    
    def _create_storage_backend(self) -> CredentialStorage:
        """Create appropriate storage backend."""
        if self.config.storage_backend == "keyring":
            return KeyringStorage()
        elif self.config.storage_backend == "file":
            return FileStorage(
                self.config.secure_file_path,
                self._get_or_create_encryption_key() if self.config.encryption_enabled else None
            )
        elif self.config.storage_backend == "environment":
            return EnvironmentStorage()
        else:
            raise CredentialError(f"Unknown storage backend: {self.config.storage_backend}")
    
    def _get_or_create_encryption_key(self) -> Optional[bytes]:
        """Get or create encryption key for credential storage."""
        if not self.config.encryption_enabled or not HAS_CRYPTOGRAPHY:
            return None
        
        key_file = self.config.secure_file_path.parent / ".encryption_key"
        
        try:
            if key_file.exists():
                # Load existing key
                with open(key_file, 'rb') as f:
                    return f.read()
            else:
                # Generate new key
                key = Fernet.generate_key()
                
                # Save key with restricted permissions
                key_file.parent.mkdir(parents=True, exist_ok=True)
                with open(key_file, 'wb') as f:
                    f.write(key)
                os.chmod(key_file, 0o600)
                
                logger.info("Generated new encryption key for credentials")
                return key
                
        except Exception as e:
            logger.error(f"Failed to handle encryption key: {e}")
            return None


# Factory function for easy instantiation
def create_credential_manager(
    storage_backend: str = "auto",
    encryption_enabled: bool = True,
    **kwargs
) -> CredentialManager:
    """Create a credential manager with the specified configuration."""
    config = CredentialConfig(
        storage_backend=storage_backend,
        encryption_enabled=encryption_enabled,
        **kwargs
    )
    return CredentialManager(config)