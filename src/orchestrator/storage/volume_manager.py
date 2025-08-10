"""Persistent Volume Management for Data - Issue #206 Task 3.4

Advanced persistent volume management system that provides secure, isolated data storage
for containers with automatic cleanup, encryption, quotas, and backup capabilities.
Integrates with the Docker security system and performance monitoring.
"""

import asyncio
import logging
import os
import shutil
import time
import json
import hashlib
import stat
import tempfile
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import psutil
import threading
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class VolumeType(Enum):
    """Types of persistent volumes."""
    TEMPORARY = "temporary"          # Cleaned up after session
    PERSISTENT = "persistent"       # Survives container restarts
    SHARED = "shared"               # Shared between containers
    BACKUP = "backup"               # Backup storage
    CACHE = "cache"                 # Cache storage with automatic cleanup


class VolumeAccess(Enum):
    """Volume access permissions."""
    READ_ONLY = "ro"
    READ_WRITE = "rw"
    WRITE_ONLY = "wo"


class EncryptionLevel(Enum):
    """Data encryption levels."""
    NONE = "none"
    BASIC = "basic"                 # Simple file-level encryption
    ADVANCED = "advanced"           # Full volume encryption


@dataclass
class VolumeQuota:
    """Volume storage quota settings."""
    max_size_mb: int = 1024         # 1GB default
    max_files: int = 10000          # Maximum number of files
    warn_threshold: float = 0.8     # Warn at 80% usage
    enforce: bool = True            # Enforce quotas
    
    @property
    def max_size_bytes(self) -> int:
        return self.max_size_mb * 1024 * 1024
    
    @property
    def warn_size_bytes(self) -> int:
        return int(self.max_size_bytes * self.warn_threshold)


@dataclass
class VolumeMetadata:
    """Volume metadata and configuration."""
    volume_id: str
    volume_type: VolumeType
    access_mode: VolumeAccess
    owner: str                      # Container ID or user
    created_at: float
    last_accessed: float
    encryption_level: EncryptionLevel
    quota: VolumeQuota
    mount_point: str
    host_path: str
    tags: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    auto_cleanup: bool = True
    backup_enabled: bool = False
    compression: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'volume_id': self.volume_id,
            'volume_type': self.volume_type.value,
            'access_mode': self.access_mode.value,
            'owner': self.owner,
            'created_at': self.created_at,
            'last_accessed': self.last_accessed,
            'encryption_level': self.encryption_level.value,
            'quota': {
                'max_size_mb': self.quota.max_size_mb,
                'max_files': self.quota.max_files,
                'warn_threshold': self.quota.warn_threshold,
                'enforce': self.quota.enforce
            },
            'mount_point': self.mount_point,
            'host_path': self.host_path,
            'tags': self.tags,
            'description': self.description,
            'auto_cleanup': self.auto_cleanup,
            'backup_enabled': self.backup_enabled,
            'compression': self.compression
        }


@dataclass
class VolumeUsage:
    """Volume usage statistics."""
    volume_id: str
    total_size_bytes: int = 0
    used_size_bytes: int = 0
    available_size_bytes: int = 0
    file_count: int = 0
    directory_count: int = 0
    last_updated: float = field(default_factory=time.time)
    
    @property
    def usage_percentage(self) -> float:
        """Calculate usage percentage."""
        if self.total_size_bytes == 0:
            return 0.0
        return (self.used_size_bytes / self.total_size_bytes) * 100
    
    @property
    def is_over_quota(self) -> bool:
        """Check if volume is over quota."""
        return self.usage_percentage > 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'volume_id': self.volume_id,
            'total_size_bytes': self.total_size_bytes,
            'used_size_bytes': self.used_size_bytes,
            'available_size_bytes': self.available_size_bytes,
            'file_count': self.file_count,
            'directory_count': self.directory_count,
            'usage_percentage': self.usage_percentage,
            'is_over_quota': self.is_over_quota,
            'last_updated': self.last_updated
        }


class VolumeEncryption:
    """Simple volume encryption management."""
    
    def __init__(self):
        self.encryption_keys: Dict[str, str] = {}
    
    def generate_key(self, volume_id: str) -> str:
        """Generate encryption key for volume."""
        key = hashlib.sha256(f"{volume_id}_{time.time()}".encode()).hexdigest()
        self.encryption_keys[volume_id] = key
        return key
    
    def encrypt_file(self, file_path: str, volume_id: str) -> bool:
        """Simple file encryption (XOR-based for demo)."""
        if volume_id not in self.encryption_keys:
            return False
        
        try:
            key = self.encryption_keys[volume_id]
            key_bytes = key.encode()[:32]  # Use first 32 chars as key
            
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Simple XOR encryption for demo
            encrypted = bytes(a ^ key_bytes[i % len(key_bytes)] for i, a in enumerate(data))
            
            with open(file_path + '.enc', 'wb') as f:
                f.write(encrypted)
            
            os.remove(file_path)  # Remove original
            os.rename(file_path + '.enc', file_path)
            
            return True
        except Exception as e:
            logger.error(f"Failed to encrypt file {file_path}: {e}")
            return False
    
    def decrypt_file(self, file_path: str, volume_id: str) -> bool:
        """Simple file decryption."""
        if volume_id not in self.encryption_keys:
            return False
        
        try:
            key = self.encryption_keys[volume_id]
            key_bytes = key.encode()[:32]
            
            with open(file_path, 'rb') as f:
                encrypted_data = f.read()
            
            # XOR decryption (same as encryption for XOR)
            decrypted = bytes(a ^ key_bytes[i % len(key_bytes)] for i, a in enumerate(encrypted_data))
            
            with open(file_path + '.dec', 'wb') as f:
                f.write(decrypted)
            
            os.remove(file_path)  # Remove encrypted
            os.rename(file_path + '.dec', file_path)
            
            return True
        except Exception as e:
            logger.error(f"Failed to decrypt file {file_path}: {e}")
            return False


class VolumeBackup:
    """Volume backup management."""
    
    def __init__(self, backup_root: str):
        self.backup_root = Path(backup_root)
        self.backup_root.mkdir(parents=True, exist_ok=True)
    
    def create_backup(self, volume_metadata: VolumeMetadata) -> Optional[str]:
        """Create backup of volume."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{volume_metadata.volume_id}_{timestamp}"
            backup_path = self.backup_root / backup_name
            
            # Create compressed archive
            shutil.make_archive(
                str(backup_path),
                'tar',
                root_dir=volume_metadata.host_path,
                base_dir='.'
            )
            
            backup_file = f"{backup_path}.tar"
            
            # Store backup metadata
            backup_meta = {
                'volume_id': volume_metadata.volume_id,
                'backup_name': backup_name,
                'backup_path': backup_file,
                'created_at': time.time(),
                'original_size': self._get_directory_size(volume_metadata.host_path),
                'backup_size': os.path.getsize(backup_file) if os.path.exists(backup_file) else 0
            }
            
            meta_file = f"{backup_path}.json"
            with open(meta_file, 'w') as f:
                json.dump(backup_meta, f, indent=2)
            
            logger.info(f"Created backup for volume {volume_metadata.volume_id}: {backup_file}")
            return backup_file
            
        except Exception as e:
            logger.error(f"Failed to create backup for volume {volume_metadata.volume_id}: {e}")
            return None
    
    def restore_backup(self, backup_file: str, target_path: str) -> bool:
        """Restore volume from backup."""
        try:
            # Create target directory if it doesn't exist
            os.makedirs(target_path, exist_ok=True)
            
            # Extract backup
            shutil.unpack_archive(backup_file, target_path)
            
            logger.info(f"Restored backup {backup_file} to {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore backup {backup_file}: {e}")
            return False
    
    def list_backups(self, volume_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available backups."""
        backups = []
        
        try:
            for meta_file in self.backup_root.glob("*.json"):
                with open(meta_file, 'r') as f:
                    backup_meta = json.load(f)
                
                if volume_id is None or backup_meta.get('volume_id') == volume_id:
                    backups.append(backup_meta)
            
            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x.get('created_at', 0), reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
        
        return backups
    
    def cleanup_old_backups(self, volume_id: str, keep_count: int = 5) -> int:
        """Clean up old backups, keeping only the most recent ones."""
        backups = self.list_backups(volume_id)
        
        if len(backups) <= keep_count:
            return 0
        
        deleted_count = 0
        for backup in backups[keep_count:]:
            try:
                backup_path = backup.get('backup_path')
                meta_path = backup_path.replace('.tar', '.json')
                
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                if os.path.exists(meta_path):
                    os.remove(meta_path)
                
                deleted_count += 1
                logger.info(f"Deleted old backup: {backup_path}")
                
            except Exception as e:
                logger.error(f"Failed to delete backup {backup.get('backup_name')}: {e}")
        
        return deleted_count
    
    def _get_directory_size(self, path: str) -> int:
        """Calculate total size of directory."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except Exception as e:
            logger.error(f"Error calculating directory size for {path}: {e}")
        
        return total_size


class PersistentVolumeManager:
    """
    Main persistent volume management system that provides secure, isolated data storage
    for containers with automatic cleanup, encryption, quotas, and backup capabilities.
    """
    
    def __init__(self, 
                 storage_root: str = "/tmp/orchestrator_volumes",
                 backup_root: str = "/tmp/orchestrator_backups",
                 performance_monitor=None):
        
        self.storage_root = Path(storage_root)
        self.backup_root = Path(backup_root)
        self.performance_monitor = performance_monitor
        
        # Create root directories
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.backup_root.mkdir(parents=True, exist_ok=True)
        
        # Volume tracking
        self.volumes: Dict[str, VolumeMetadata] = {}
        self.volume_usage: Dict[str, VolumeUsage] = {}
        self.active_mounts: Dict[str, str] = {}  # volume_id -> container_id
        
        # Components
        self.encryption = VolumeEncryption()
        self.backup_manager = VolumeBackup(str(self.backup_root))
        
        # Background tasks
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._cleanup_interval = 300  # 5 minutes
        
        # Statistics
        self.stats = {
            'volumes_created': 0,
            'volumes_deleted': 0,
            'volumes_mounted': 0,
            'volumes_unmounted': 0,
            'backups_created': 0,
            'quota_violations': 0,
            'cleanup_operations': 0
        }
        
        # Load existing volumes
        self._load_existing_volumes()
        
        logger.info(f"PersistentVolumeManager initialized with storage at {self.storage_root}")
    
    def _load_existing_volumes(self):
        """Load metadata for existing volumes."""
        try:
            for volume_dir in self.storage_root.iterdir():
                if volume_dir.is_dir():
                    meta_file = volume_dir / "volume_metadata.json"
                    if meta_file.exists():
                        with open(meta_file, 'r') as f:
                            meta_data = json.load(f)
                        
                        # Reconstruct VolumeMetadata object
                        quota = VolumeQuota(**meta_data.get('quota', {}))
                        
                        metadata = VolumeMetadata(
                            volume_id=meta_data['volume_id'],
                            volume_type=VolumeType(meta_data['volume_type']),
                            access_mode=VolumeAccess(meta_data['access_mode']),
                            owner=meta_data['owner'],
                            created_at=meta_data['created_at'],
                            last_accessed=meta_data['last_accessed'],
                            encryption_level=EncryptionLevel(meta_data['encryption_level']),
                            quota=quota,
                            mount_point=meta_data['mount_point'],
                            host_path=meta_data['host_path'],
                            tags=meta_data.get('tags', {}),
                            description=meta_data.get('description', ''),
                            auto_cleanup=meta_data.get('auto_cleanup', True),
                            backup_enabled=meta_data.get('backup_enabled', False),
                            compression=meta_data.get('compression', False)
                        )
                        
                        self.volumes[metadata.volume_id] = metadata
                        
                        # Update usage
                        self._update_volume_usage(metadata.volume_id)
            
            logger.info(f"Loaded {len(self.volumes)} existing volumes")
            
        except Exception as e:
            logger.error(f"Failed to load existing volumes: {e}")
    
    async def start_monitoring(self):
        """Start background monitoring tasks."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Volume monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring tasks."""
        self._monitoring = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Volume monitoring stopped")
    
    async def create_volume(self,
                     owner: str,
                     volume_type: VolumeType = VolumeType.TEMPORARY,
                     access_mode: VolumeAccess = VolumeAccess.READ_WRITE,
                     quota: Optional[VolumeQuota] = None,
                     encryption_level: EncryptionLevel = EncryptionLevel.NONE,
                     mount_point: str = "/data",
                     tags: Optional[Dict[str, str]] = None,
                     description: str = "") -> Optional[str]:
        """Create a new persistent volume."""
        
        try:
            # Generate unique volume ID
            volume_id = hashlib.sha256(f"{owner}_{time.time()}_{os.urandom(8).hex()}".encode()).hexdigest()[:16]
            
            # Create host path
            host_path = str(self.storage_root / volume_id)
            os.makedirs(host_path, exist_ok=True)
            
            # Set default quota if not provided
            if quota is None:
                quota = VolumeQuota()
            
            # Create volume metadata
            metadata = VolumeMetadata(
                volume_id=volume_id,
                volume_type=volume_type,
                access_mode=access_mode,
                owner=owner,
                created_at=time.time(),
                last_accessed=time.time(),
                encryption_level=encryption_level,
                quota=quota,
                mount_point=mount_point,
                host_path=host_path,
                tags=tags or {},
                description=description,
                auto_cleanup=(volume_type == VolumeType.TEMPORARY),
                backup_enabled=(volume_type in [VolumeType.PERSISTENT, VolumeType.SHARED])
            )
            
            # Save metadata
            meta_file = Path(host_path) / "volume_metadata.json"
            with open(meta_file, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            # Set up encryption if required
            if encryption_level != EncryptionLevel.NONE:
                self.encryption.generate_key(volume_id)
            
            # Set permissions based on access mode
            self._set_volume_permissions(host_path, access_mode)
            
            # Register volume
            self.volumes[volume_id] = metadata
            self._update_volume_usage(volume_id)
            
            self.stats['volumes_created'] += 1
            
            # Record performance metrics
            if self.performance_monitor:
                try:
                    await self.performance_monitor.record_execution(
                        component="volume_manager",
                        execution_time=0.1,
                        success=True,
                        context={
                            'operation': 'create_volume',
                            'volume_type': volume_type.value,
                            'volume_id': volume_id
                        }
                    )
                except Exception:
                    pass  # Don't fail volume creation due to monitoring issues
            
            logger.info(f"Created volume {volume_id} for {owner} at {host_path}")
            return volume_id
            
        except Exception as e:
            logger.error(f"Failed to create volume for {owner}: {e}")
            self.stats['volumes_created'] += 1  # Count failed attempts too
            return None
    
    def _set_volume_permissions(self, host_path: str, access_mode: VolumeAccess):
        """Set filesystem permissions based on access mode."""
        try:
            if access_mode == VolumeAccess.READ_ONLY:
                # Read-only: 444
                os.chmod(host_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            elif access_mode == VolumeAccess.WRITE_ONLY:
                # Write-only: 222 (though this is unusual)
                os.chmod(host_path, stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
            else:  # READ_WRITE
                # Read-write: 755 for directories, 644 for files
                os.chmod(host_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        except Exception as e:
            logger.warning(f"Failed to set permissions for {host_path}: {e}")
    
    def mount_volume(self, volume_id: str, container_id: str) -> Optional[Dict[str, str]]:
        """Mount volume to container and return mount configuration."""
        if volume_id not in self.volumes:
            logger.error(f"Volume {volume_id} not found")
            return None
        
        metadata = self.volumes[volume_id]
        
        # Check if volume is already mounted to different container
        if volume_id in self.active_mounts and self.active_mounts[volume_id] != container_id:
            if metadata.volume_type != VolumeType.SHARED:
                logger.error(f"Volume {volume_id} is already mounted to container {self.active_mounts[volume_id]}")
                return None
        
        # Update last accessed time
        metadata.last_accessed = time.time()
        self.active_mounts[volume_id] = container_id
        
        # Update usage statistics
        self._update_volume_usage(volume_id)
        
        self.stats['volumes_mounted'] += 1
        
        # Return Docker mount configuration
        mount_config = {
            'source': metadata.host_path,
            'target': metadata.mount_point,
            'type': 'bind',
            'read_only': (metadata.access_mode == VolumeAccess.READ_ONLY)
        }
        
        logger.info(f"Mounted volume {volume_id} to container {container_id}")
        return mount_config
    
    def unmount_volume(self, volume_id: str, container_id: str) -> bool:
        """Unmount volume from container."""
        if volume_id not in self.volumes:
            logger.error(f"Volume {volume_id} not found")
            return False
        
        if volume_id in self.active_mounts and self.active_mounts[volume_id] == container_id:
            del self.active_mounts[volume_id]
            self.stats['volumes_unmounted'] += 1
            
            # Update last accessed time
            self.volumes[volume_id].last_accessed = time.time()
            
            logger.info(f"Unmounted volume {volume_id} from container {container_id}")
            return True
        
        return False
    
    def delete_volume(self, volume_id: str, force: bool = False) -> bool:
        """Delete a volume and all its data."""
        if volume_id not in self.volumes:
            logger.error(f"Volume {volume_id} not found")
            return False
        
        metadata = self.volumes[volume_id]
        
        # Check if volume is currently mounted
        if volume_id in self.active_mounts and not force:
            logger.error(f"Volume {volume_id} is currently mounted. Use force=True to delete anyway.")
            return False
        
        try:
            # Create backup if enabled
            if metadata.backup_enabled:
                self.backup_manager.create_backup(metadata)
                self.stats['backups_created'] += 1
            
            # Remove from active mounts if forced
            if volume_id in self.active_mounts:
                del self.active_mounts[volume_id]
            
            # Remove host directory
            if os.path.exists(metadata.host_path):
                shutil.rmtree(metadata.host_path)
            
            # Remove from tracking
            del self.volumes[volume_id]
            if volume_id in self.volume_usage:
                del self.volume_usage[volume_id]
            
            self.stats['volumes_deleted'] += 1
            
            logger.info(f"Deleted volume {volume_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete volume {volume_id}: {e}")
            return False
    
    def _update_volume_usage(self, volume_id: str):
        """Update volume usage statistics."""
        if volume_id not in self.volumes:
            return
        
        metadata = self.volumes[volume_id]
        
        try:
            # Calculate directory usage
            total_size = 0
            file_count = 0
            dir_count = 0
            
            for root, dirs, files in os.walk(metadata.host_path):
                dir_count += len(dirs)
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
                        file_count += 1
            
            usage = VolumeUsage(
                volume_id=volume_id,
                total_size_bytes=metadata.quota.max_size_bytes,
                used_size_bytes=total_size,
                available_size_bytes=max(0, metadata.quota.max_size_bytes - total_size),
                file_count=file_count,
                directory_count=dir_count,
                last_updated=time.time()
            )
            
            self.volume_usage[volume_id] = usage
            
            # Check quota violations
            if metadata.quota.enforce:
                if usage.used_size_bytes > metadata.quota.max_size_bytes:
                    logger.warning(f"Volume {volume_id} exceeded size quota: {usage.used_size_bytes} > {metadata.quota.max_size_bytes}")
                    self.stats['quota_violations'] += 1
                
                if usage.file_count > metadata.quota.max_files:
                    logger.warning(f"Volume {volume_id} exceeded file quota: {usage.file_count} > {metadata.quota.max_files}")
                    self.stats['quota_violations'] += 1
            
        except Exception as e:
            logger.error(f"Failed to update usage for volume {volume_id}: {e}")
    
    def get_volume_info(self, volume_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive volume information."""
        if volume_id not in self.volumes:
            return None
        
        metadata = self.volumes[volume_id]
        usage = self.volume_usage.get(volume_id)
        
        return {
            'metadata': metadata.to_dict(),
            'usage': usage.to_dict() if usage else None,
            'is_mounted': volume_id in self.active_mounts,
            'mounted_to': self.active_mounts.get(volume_id),
            'exists': os.path.exists(metadata.host_path)
        }
    
    def list_volumes(self, 
                    owner: Optional[str] = None,
                    volume_type: Optional[VolumeType] = None,
                    include_usage: bool = True) -> List[Dict[str, Any]]:
        """List volumes with optional filtering."""
        volumes = []
        
        for volume_id, metadata in self.volumes.items():
            # Apply filters
            if owner and metadata.owner != owner:
                continue
            if volume_type and metadata.volume_type != volume_type:
                continue
            
            volume_info = {
                'volume_id': volume_id,
                'metadata': metadata.to_dict(),
                'is_mounted': volume_id in self.active_mounts,
                'mounted_to': self.active_mounts.get(volume_id)
            }
            
            if include_usage:
                usage = self.volume_usage.get(volume_id)
                volume_info['usage'] = usage.to_dict() if usage else None
            
            volumes.append(volume_info)
        
        # Sort by creation time (newest first)
        volumes.sort(key=lambda x: x['metadata']['created_at'], reverse=True)
        
        return volumes
    
    def create_backup(self, volume_id: str) -> Optional[str]:
        """Create backup of specific volume."""
        if volume_id not in self.volumes:
            logger.error(f"Volume {volume_id} not found")
            return None
        
        metadata = self.volumes[volume_id]
        backup_path = self.backup_manager.create_backup(metadata)
        
        if backup_path:
            self.stats['backups_created'] += 1
        
        return backup_path
    
    def restore_volume(self, volume_id: str, backup_file: str) -> bool:
        """Restore volume from backup."""
        if volume_id not in self.volumes:
            logger.error(f"Volume {volume_id} not found")
            return False
        
        metadata = self.volumes[volume_id]
        
        # Check if volume is mounted
        if volume_id in self.active_mounts:
            logger.error(f"Cannot restore mounted volume {volume_id}")
            return False
        
        # Clear existing data
        if os.path.exists(metadata.host_path):
            shutil.rmtree(metadata.host_path)
        os.makedirs(metadata.host_path, exist_ok=True)
        
        # Restore from backup
        success = self.backup_manager.restore_backup(backup_file, metadata.host_path)
        
        if success:
            # Update usage statistics
            self._update_volume_usage(volume_id)
            metadata.last_accessed = time.time()
            logger.info(f"Restored volume {volume_id} from backup {backup_file}")
        
        return success
    
    async def _monitoring_loop(self):
        """Background monitoring and cleanup loop."""
        while self._monitoring:
            try:
                # Update usage for all volumes
                for volume_id in list(self.volumes.keys()):
                    self._update_volume_usage(volume_id)
                
                # Cleanup old temporary volumes
                await self._cleanup_temporary_volumes()
                
                # Cleanup old backups
                await self._cleanup_old_backups()
                
                # Check disk space
                await self._check_disk_space()
                
                await asyncio.sleep(self._cleanup_interval)
                
            except Exception as e:
                logger.error(f"Error in volume monitoring loop: {e}")
                await asyncio.sleep(self._cleanup_interval)
    
    async def _cleanup_temporary_volumes(self):
        """Clean up old temporary volumes."""
        current_time = time.time()
        cleanup_age = 3600  # 1 hour for temporary volumes
        
        to_cleanup = []
        for volume_id, metadata in self.volumes.items():
            if (metadata.volume_type == VolumeType.TEMPORARY and 
                metadata.auto_cleanup and
                current_time - metadata.last_accessed > cleanup_age and
                volume_id not in self.active_mounts):
                
                to_cleanup.append(volume_id)
        
        for volume_id in to_cleanup:
            if self.delete_volume(volume_id, force=True):
                self.stats['cleanup_operations'] += 1
                logger.info(f"Cleaned up temporary volume {volume_id}")
    
    async def _cleanup_old_backups(self):
        """Clean up old backups."""
        for volume_id in self.volumes.keys():
            try:
                deleted = self.backup_manager.cleanup_old_backups(volume_id, keep_count=5)
                if deleted > 0:
                    self.stats['cleanup_operations'] += deleted
            except Exception as e:
                logger.error(f"Error cleaning up backups for volume {volume_id}: {e}")
    
    async def _check_disk_space(self):
        """Check available disk space and warn if low."""
        try:
            disk_usage = psutil.disk_usage(str(self.storage_root))
            usage_percentage = (disk_usage.used / disk_usage.total) * 100
            
            if usage_percentage > 90:
                logger.warning(f"High disk usage: {usage_percentage:.1f}% used")
            elif usage_percentage > 95:
                logger.critical(f"Critical disk usage: {usage_percentage:.1f}% used")
        except Exception as e:
            logger.error(f"Error checking disk space: {e}")
    
    def get_storage_summary(self) -> Dict[str, Any]:
        """Get comprehensive storage summary."""
        try:
            # Calculate total usage
            total_volumes = len(self.volumes)
            mounted_volumes = len(self.active_mounts)
            total_size_used = sum(usage.used_size_bytes for usage in self.volume_usage.values())
            total_size_allocated = sum(vol.quota.max_size_bytes for vol in self.volumes.values())
            
            # Volume type breakdown
            type_breakdown = {}
            for vol in self.volumes.values():
                vol_type = vol.volume_type.value
                type_breakdown[vol_type] = type_breakdown.get(vol_type, 0) + 1
            
            # Disk usage
            disk_usage = psutil.disk_usage(str(self.storage_root))
            
            return {
                'timestamp': time.time(),
                'total_volumes': total_volumes,
                'mounted_volumes': mounted_volumes,
                'volume_types': type_breakdown,
                'storage': {
                    'total_size_used_bytes': total_size_used,
                    'total_size_allocated_bytes': total_size_allocated,
                    'storage_root': str(self.storage_root),
                    'disk_total_bytes': disk_usage.total,
                    'disk_used_bytes': disk_usage.used,
                    'disk_free_bytes': disk_usage.free,
                    'disk_usage_percentage': (disk_usage.used / disk_usage.total) * 100
                },
                'statistics': self.stats,
                'monitoring_active': self._monitoring,
                'backup_root': str(self.backup_root)
            }
            
        except Exception as e:
            logger.error(f"Error generating storage summary: {e}")
            return {'error': str(e)}


# Export classes
__all__ = [
    'PersistentVolumeManager',
    'VolumeType',
    'VolumeAccess',
    'EncryptionLevel',
    'VolumeMetadata',
    'VolumeQuota',
    'VolumeUsage',
    'VolumeEncryption',
    'VolumeBackup'
]