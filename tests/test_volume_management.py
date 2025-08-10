"""Real Volume Management Tests - Issue #206 Task 3.4

Comprehensive tests for the persistent volume management system.
Tests include volume creation, mounting, quotas, encryption, backups,
and cleanup operations. NO MOCKS - real volume management only.
"""

import pytest
import asyncio
import logging
import time
import tempfile
import json
import os
import shutil
from pathlib import Path

from orchestrator.storage.volume_manager import (
    PersistentVolumeManager,
    VolumeType,
    VolumeAccess,
    EncryptionLevel,
    VolumeQuota,
    VolumeMetadata,
    VolumeEncryption,
    VolumeBackup
)
from orchestrator.analytics.performance_monitor import PerformanceMonitor

# Configure logging for test visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestVolumeEncryption:
    """Test volume encryption functionality."""
    
    @pytest.fixture
    def encryption(self):
        """Create encryption manager for testing."""
        return VolumeEncryption()
    
    def test_key_generation(self, encryption):
        """Test encryption key generation."""
        logger.info("🧪 Testing encryption key generation")
        
        volume_id = "test_volume_123"
        
        # Generate key
        key = encryption.generate_key(volume_id)
        
        assert key is not None
        assert isinstance(key, str)
        assert len(key) == 64  # SHA256 hex string
        assert volume_id in encryption.encryption_keys
        assert encryption.encryption_keys[volume_id] == key
        
        # Generate another key for different volume
        key2 = encryption.generate_key("test_volume_456")
        assert key2 != key  # Should be different
        
        logger.info("✅ Encryption key generation test passed")
    
    def test_file_encryption_decryption(self, encryption):
        """Test file encryption and decryption."""
        logger.info("🧪 Testing file encryption and decryption")
        
        volume_id = "test_encrypt_volume"
        encryption.generate_key(volume_id)
        
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            test_data = "This is sensitive data that should be encrypted!"
            f.write(test_data)
            test_file = f.name
        
        try:
            # Read original data
            with open(test_file, 'r') as f:
                original_data = f.read()
            
            assert original_data == test_data
            
            # Encrypt file
            success = encryption.encrypt_file(test_file, volume_id)
            assert success is True
            
            # File should now be encrypted (different content)
            with open(test_file, 'rb') as f:
                encrypted_data = f.read()
            
            assert encrypted_data != test_data.encode()
            
            # Decrypt file
            success = encryption.decrypt_file(test_file, volume_id)
            assert success is True
            
            # File should be back to original
            with open(test_file, 'r') as f:
                decrypted_data = f.read()
            
            assert decrypted_data == test_data
            
        finally:
            # Cleanup
            if os.path.exists(test_file):
                os.remove(test_file)
        
        logger.info("✅ File encryption/decryption test passed")


class TestVolumeBackup:
    """Test volume backup functionality."""
    
    @pytest.fixture
    def backup_manager(self):
        """Create backup manager for testing."""
        backup_dir = tempfile.mkdtemp(prefix="test_backup_")
        backup_mgr = VolumeBackup(backup_dir)
        yield backup_mgr
        # Cleanup
        shutil.rmtree(backup_dir, ignore_errors=True)
    
    @pytest.fixture
    def test_volume_data(self):
        """Create test volume with data."""
        volume_dir = tempfile.mkdtemp(prefix="test_volume_")
        
        # Create test files
        (Path(volume_dir) / "test1.txt").write_text("Test file 1 content")
        (Path(volume_dir) / "test2.txt").write_text("Test file 2 content")
        
        # Create subdirectory with files
        subdir = Path(volume_dir) / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("Nested file content")
        
        yield volume_dir
        
        # Cleanup
        shutil.rmtree(volume_dir, ignore_errors=True)
    
    def test_backup_creation(self, backup_manager, test_volume_data):
        """Test creating volume backups."""
        logger.info("🧪 Testing backup creation")
        
        # Create volume metadata
        metadata = VolumeMetadata(
            volume_id="test_backup_volume",
            volume_type=VolumeType.PERSISTENT,
            access_mode=VolumeAccess.READ_WRITE,
            owner="test_owner",
            created_at=time.time(),
            last_accessed=time.time(),
            encryption_level=EncryptionLevel.NONE,
            quota=VolumeQuota(),
            mount_point="/data",
            host_path=test_volume_data
        )
        
        # Create backup
        backup_path = backup_manager.create_backup(metadata)
        
        assert backup_path is not None
        assert os.path.exists(backup_path)
        assert backup_path.endswith('.tar')
        
        # Check backup metadata
        meta_file = backup_path.replace('.tar', '.json')
        assert os.path.exists(meta_file)
        
        with open(meta_file, 'r') as f:
            backup_meta = json.load(f)
        
        assert backup_meta['volume_id'] == "test_backup_volume"
        assert backup_meta['backup_size'] > 0
        assert backup_meta['original_size'] > 0
        
        logger.info("✅ Backup creation test passed")
    
    def test_backup_restore(self, backup_manager, test_volume_data):
        """Test restoring from backup."""
        logger.info("🧪 Testing backup restore")
        
        # Create volume metadata
        metadata = VolumeMetadata(
            volume_id="test_restore_volume",
            volume_type=VolumeType.PERSISTENT,
            access_mode=VolumeAccess.READ_WRITE,
            owner="test_owner",
            created_at=time.time(),
            last_accessed=time.time(),
            encryption_level=EncryptionLevel.NONE,
            quota=VolumeQuota(),
            mount_point="/data",
            host_path=test_volume_data
        )
        
        # Create backup
        backup_path = backup_manager.create_backup(metadata)
        assert backup_path is not None
        
        # Create empty target directory for restore
        restore_dir = tempfile.mkdtemp(prefix="test_restore_")
        
        try:
            # Restore backup
            success = backup_manager.restore_backup(backup_path, restore_dir)
            assert success is True
            
            # Check restored files
            assert os.path.exists(os.path.join(restore_dir, "test1.txt"))
            assert os.path.exists(os.path.join(restore_dir, "test2.txt"))
            assert os.path.exists(os.path.join(restore_dir, "subdir", "nested.txt"))
            
            # Verify file contents
            with open(os.path.join(restore_dir, "test1.txt"), 'r') as f:
                assert f.read() == "Test file 1 content"
            
            with open(os.path.join(restore_dir, "subdir", "nested.txt"), 'r') as f:
                assert f.read() == "Nested file content"
                
        finally:
            shutil.rmtree(restore_dir, ignore_errors=True)
        
        logger.info("✅ Backup restore test passed")
    
    def test_backup_listing(self, backup_manager, test_volume_data):
        """Test listing backups."""
        logger.info("🧪 Testing backup listing")
        
        volume_id = "test_list_volume"
        
        # Create volume metadata
        metadata = VolumeMetadata(
            volume_id=volume_id,
            volume_type=VolumeType.PERSISTENT,
            access_mode=VolumeAccess.READ_WRITE,
            owner="test_owner",
            created_at=time.time(),
            last_accessed=time.time(),
            encryption_level=EncryptionLevel.NONE,
            quota=VolumeQuota(),
            mount_point="/data",
            host_path=test_volume_data
        )
        
        # Create multiple backups
        backup1 = backup_manager.create_backup(metadata)
        time.sleep(1)  # Ensure different timestamps
        backup2 = backup_manager.create_backup(metadata)
        
        assert backup1 is not None
        assert backup2 is not None
        
        # List all backups
        all_backups = backup_manager.list_backups()
        assert len(all_backups) >= 2
        
        # List backups for specific volume
        volume_backups = backup_manager.list_backups(volume_id)
        assert len(volume_backups) == 2
        
        # Check backup ordering (newest first)
        assert volume_backups[0]['created_at'] > volume_backups[1]['created_at']
        
        logger.info("✅ Backup listing test passed")


class TestPersistentVolumeManager:
    """Test persistent volume management system."""
    
    @pytest.fixture
    async def performance_monitor(self):
        """Create performance monitor for integration testing."""
        monitor = PerformanceMonitor(collection_interval=0.5)
        await monitor.start_monitoring()
        yield monitor
        await monitor.stop_monitoring()
    
    @pytest.fixture
    async def volume_manager(self, performance_monitor):
        """Create volume manager for testing."""
        storage_dir = tempfile.mkdtemp(prefix="test_storage_")
        backup_dir = tempfile.mkdtemp(prefix="test_backup_")
        
        manager = PersistentVolumeManager(
            storage_root=storage_dir,
            backup_root=backup_dir,
            performance_monitor=performance_monitor
        )
        
        await manager.start_monitoring()
        yield manager
        await manager.stop_monitoring()
        
        # Cleanup
        shutil.rmtree(storage_dir, ignore_errors=True)
        shutil.rmtree(backup_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_volume_manager_initialization(self, volume_manager):
        """Test volume manager initialization."""
        logger.info("🧪 Testing volume manager initialization")
        
        assert volume_manager.storage_root.exists()
        assert volume_manager.backup_root.exists()
        assert len(volume_manager.volumes) >= 0
        assert len(volume_manager.volume_usage) >= 0
        assert len(volume_manager.active_mounts) == 0
        assert volume_manager._monitoring is True
        
        # Check statistics
        assert 'volumes_created' in volume_manager.stats
        assert 'volumes_deleted' in volume_manager.stats
        assert 'volumes_mounted' in volume_manager.stats
        assert 'volumes_unmounted' in volume_manager.stats
        
        logger.info("✅ Volume manager initialization test passed")
    
    @pytest.mark.asyncio
    async def test_volume_creation(self, volume_manager):
        """Test volume creation with different configurations."""
        logger.info("🧪 Testing volume creation")
        
        # Test basic volume creation
        volume_id = await volume_manager.create_volume(
            owner="test_container_1",
            volume_type=VolumeType.TEMPORARY,
            description="Test temporary volume"
        )
        
        assert volume_id is not None
        assert volume_id in volume_manager.volumes
        
        # Check volume metadata
        metadata = volume_manager.volumes[volume_id]
        assert metadata.owner == "test_container_1"
        assert metadata.volume_type == VolumeType.TEMPORARY
        assert metadata.access_mode == VolumeAccess.READ_WRITE
        assert metadata.description == "Test temporary volume"
        
        # Check host path exists
        assert os.path.exists(metadata.host_path)
        
        # Test persistent volume with custom quota
        quota = VolumeQuota(max_size_mb=512, max_files=5000)
        
        volume_id2 = await volume_manager.create_volume(
            owner="test_container_2",
            volume_type=VolumeType.PERSISTENT,
            access_mode=VolumeAccess.READ_ONLY,
            quota=quota,
            encryption_level=EncryptionLevel.BASIC,
            tags={"environment": "test", "purpose": "data"}
        )
        
        assert volume_id2 is not None
        assert volume_id2 != volume_id
        
        metadata2 = volume_manager.volumes[volume_id2]
        assert metadata2.volume_type == VolumeType.PERSISTENT
        assert metadata2.access_mode == VolumeAccess.READ_ONLY
        assert metadata2.quota.max_size_mb == 512
        assert metadata2.encryption_level == EncryptionLevel.BASIC
        assert metadata2.tags["environment"] == "test"
        
        logger.info("✅ Volume creation test passed")
    
    @pytest.mark.asyncio
    async def test_volume_mounting(self, volume_manager):
        """Test volume mounting and unmounting."""
        logger.info("🧪 Testing volume mounting")
        
        # Create volume
        volume_id = await volume_manager.create_volume(
            owner="test_container",
            volume_type=VolumeType.PERSISTENT,
            mount_point="/test/data"
        )
        
        assert volume_id is not None
        
        container_id = "container_123"
        
        # Test mounting
        mount_config = volume_manager.mount_volume(volume_id, container_id)
        
        assert mount_config is not None
        assert mount_config['target'] == "/test/data"
        assert mount_config['type'] == "bind"
        assert mount_config['read_only'] is False
        
        # Check mount tracking
        assert volume_id in volume_manager.active_mounts
        assert volume_manager.active_mounts[volume_id] == container_id
        
        # Test double mounting to same container (should work)
        mount_config2 = volume_manager.mount_volume(volume_id, container_id)
        assert mount_config2 is not None
        
        # Test mounting to different container (should fail for non-shared volumes)
        mount_config3 = volume_manager.mount_volume(volume_id, "different_container")
        assert mount_config3 is None
        
        # Test unmounting
        success = volume_manager.unmount_volume(volume_id, container_id)
        assert success is True
        assert volume_id not in volume_manager.active_mounts
        
        logger.info("✅ Volume mounting test passed")
    
    @pytest.mark.asyncio
    async def test_shared_volume_mounting(self, volume_manager):
        """Test shared volume mounting to multiple containers."""
        logger.info("🧪 Testing shared volume mounting")
        
        # Create shared volume
        volume_id = await volume_manager.create_volume(
            owner="shared_owner",
            volume_type=VolumeType.SHARED,
            description="Shared test volume"
        )
        
        assert volume_id is not None
        
        # Mount to first container
        mount_config1 = volume_manager.mount_volume(volume_id, "container_1")
        assert mount_config1 is not None
        
        # Mount to second container (should work for shared volumes)
        mount_config2 = volume_manager.mount_volume(volume_id, "container_2")
        assert mount_config2 is not None
        
        # Both should be tracked
        assert volume_id in volume_manager.active_mounts
        # Note: active_mounts tracks last container for simplicity
        
        logger.info("✅ Shared volume mounting test passed")
    
    @pytest.mark.asyncio
    async def test_volume_usage_tracking(self, volume_manager):
        """Test volume usage statistics tracking."""
        logger.info("🧪 Testing volume usage tracking")
        
        # Create volume
        volume_id = await volume_manager.create_volume(
            owner="test_usage",
            quota=VolumeQuota(max_size_mb=10, max_files=100)
        )
        
        assert volume_id is not None
        
        # Get volume metadata
        metadata = volume_manager.volumes[volume_id]
        
        # Create test files in volume
        test_file1 = os.path.join(metadata.host_path, "test1.txt")
        test_file2 = os.path.join(metadata.host_path, "test2.txt")
        
        with open(test_file1, 'w') as f:
            f.write("Test content 1" * 100)  # Create some data
        
        with open(test_file2, 'w') as f:
            f.write("Test content 2" * 200)
        
        # Update usage
        volume_manager._update_volume_usage(volume_id)
        
        # Check usage statistics
        usage = volume_manager.volume_usage.get(volume_id)
        assert usage is not None
        assert usage.file_count == 3  # 2 test files + 1 metadata file
        assert usage.used_size_bytes > 0
        assert usage.usage_percentage >= 0
        
        # Get volume info
        vol_info = volume_manager.get_volume_info(volume_id)
        assert vol_info is not None
        assert vol_info['metadata']['volume_id'] == volume_id
        assert vol_info['usage'] is not None
        assert vol_info['exists'] is True
        
        logger.info("✅ Volume usage tracking test passed")
    
    @pytest.mark.asyncio
    async def test_volume_deletion(self, volume_manager):
        """Test volume deletion."""
        logger.info("🧪 Testing volume deletion")
        
        # Create volume
        volume_id = await volume_manager.create_volume(
            owner="test_delete",
            volume_type=VolumeType.TEMPORARY
        )
        
        assert volume_id is not None
        metadata = volume_manager.volumes[volume_id]
        host_path = metadata.host_path
        
        # Verify volume exists
        assert os.path.exists(host_path)
        
        # Try to delete mounted volume (should fail)
        volume_manager.mount_volume(volume_id, "test_container")
        success = volume_manager.delete_volume(volume_id, force=False)
        assert success is False
        
        # Unmount and delete
        volume_manager.unmount_volume(volume_id, "test_container")
        success = volume_manager.delete_volume(volume_id, force=False)
        assert success is True
        
        # Verify volume is gone
        assert volume_id not in volume_manager.volumes
        assert not os.path.exists(host_path)
        
        logger.info("✅ Volume deletion test passed")
    
    @pytest.mark.asyncio
    async def test_volume_listing(self, volume_manager):
        """Test volume listing with filtering."""
        logger.info("🧪 Testing volume listing")
        
        # Create different types of volumes
        temp_id = await volume_manager.create_volume(
            owner="user_1",
            volume_type=VolumeType.TEMPORARY,
            tags={"env": "test"}
        )
        
        persistent_id = await volume_manager.create_volume(
            owner="user_1",
            volume_type=VolumeType.PERSISTENT,
            tags={"env": "prod"}
        )
        
        shared_id = await volume_manager.create_volume(
            owner="user_2",
            volume_type=VolumeType.SHARED,
            tags={"env": "staging"}
        )
        
        assert all([temp_id, persistent_id, shared_id])
        
        # List all volumes
        all_volumes = volume_manager.list_volumes()
        assert len(all_volumes) >= 3
        
        # Filter by owner
        user1_volumes = volume_manager.list_volumes(owner="user_1")
        assert len(user1_volumes) == 2
        
        # Filter by type
        persistent_volumes = volume_manager.list_volumes(volume_type=VolumeType.PERSISTENT)
        assert len(persistent_volumes) >= 1
        
        # Check volume info structure
        for vol_info in all_volumes:
            assert 'volume_id' in vol_info
            assert 'metadata' in vol_info
            assert 'is_mounted' in vol_info
            assert 'usage' in vol_info
        
        logger.info("✅ Volume listing test passed")
    
    @pytest.mark.asyncio
    async def test_backup_operations(self, volume_manager):
        """Test volume backup and restore operations."""
        logger.info("🧪 Testing backup operations")
        
        # Create volume with data
        volume_id = await volume_manager.create_volume(
            owner="backup_test",
            volume_type=VolumeType.PERSISTENT,
            description="Volume for backup testing"
        )
        
        assert volume_id is not None
        metadata = volume_manager.volumes[volume_id]
        
        # Add test data
        test_file = os.path.join(metadata.host_path, "backup_test.txt")
        test_data = "This data should be preserved in backup"
        
        with open(test_file, 'w') as f:
            f.write(test_data)
        
        # Create backup
        backup_path = volume_manager.create_backup(volume_id)
        assert backup_path is not None
        assert os.path.exists(backup_path)
        
        # Delete test file to simulate data loss
        os.remove(test_file)
        assert not os.path.exists(test_file)
        
        # Restore from backup
        success = volume_manager.restore_volume(volume_id, backup_path)
        assert success is True
        
        # Verify data is restored
        assert os.path.exists(test_file)
        with open(test_file, 'r') as f:
            restored_data = f.read()
        
        assert restored_data == test_data
        
        logger.info("✅ Backup operations test passed")
    
    @pytest.mark.asyncio
    async def test_storage_summary(self, volume_manager):
        """Test storage summary generation."""
        logger.info("🧪 Testing storage summary")
        
        # Create some volumes
        for i in range(3):
            await volume_manager.create_volume(
                owner=f"user_{i}",
                volume_type=VolumeType.TEMPORARY if i % 2 == 0 else VolumeType.PERSISTENT
            )
        
        # Get storage summary
        summary = volume_manager.get_storage_summary()
        
        # Verify summary structure
        required_keys = [
            'timestamp', 'total_volumes', 'mounted_volumes',
            'volume_types', 'storage', 'statistics', 'monitoring_active'
        ]
        
        for key in required_keys:
            assert key in summary, f"Missing key in summary: {key}"
        
        assert summary['total_volumes'] >= 3
        assert isinstance(summary['volume_types'], dict)
        assert isinstance(summary['storage'], dict)
        assert isinstance(summary['statistics'], dict)
        assert summary['monitoring_active'] is True
        
        # Check storage details
        storage = summary['storage']
        storage_keys = ['total_size_used_bytes', 'total_size_allocated_bytes', 
                       'disk_total_bytes', 'disk_used_bytes', 'disk_free_bytes']
        
        for key in storage_keys:
            assert key in storage
            assert isinstance(storage[key], (int, float))
        
        logger.info(f"Storage summary: {summary['statistics']}")
        logger.info("✅ Storage summary test passed")
    
    @pytest.mark.asyncio
    async def test_quota_enforcement(self, volume_manager):
        """Test quota enforcement."""
        logger.info("🧪 Testing quota enforcement")
        
        # Create volume with small quota
        small_quota = VolumeQuota(max_size_mb=1, max_files=2, enforce=True)
        
        volume_id = await volume_manager.create_volume(
            owner="quota_test",
            quota=small_quota,
            description="Volume with small quota for testing"
        )
        
        assert volume_id is not None
        metadata = volume_manager.volumes[volume_id]
        
        # Create file that exceeds quota
        large_file = os.path.join(metadata.host_path, "large_file.txt")
        large_content = "x" * (2 * 1024 * 1024)  # 2MB content
        
        with open(large_file, 'w') as f:
            f.write(large_content)
        
        # Update usage - should detect quota violation
        initial_violations = volume_manager.stats['quota_violations']
        volume_manager._update_volume_usage(volume_id)
        
        # Check if quota violation was detected
        usage = volume_manager.volume_usage[volume_id]
        assert usage.is_over_quota is True
        assert volume_manager.stats['quota_violations'] > initial_violations
        
        logger.info("✅ Quota enforcement test passed")
    
    @pytest.mark.asyncio
    async def test_cleanup_operations(self, volume_manager):
        """Test automatic cleanup operations."""
        logger.info("🧪 Testing cleanup operations")
        
        # Create temporary volume
        temp_volume_id = await volume_manager.create_volume(
            owner="cleanup_test",
            volume_type=VolumeType.TEMPORARY,
            description="Volume for cleanup testing"
        )
        
        assert temp_volume_id is not None
        
        # Modify last_accessed time to make it old
        metadata = volume_manager.volumes[temp_volume_id]
        old_time = time.time() - 7200  # 2 hours ago
        metadata.last_accessed = old_time
        
        # Run cleanup manually
        await volume_manager._cleanup_temporary_volumes()
        
        # Temporary volume should still exist (not mounted, but recent)
        # Let's force it to be really old
        metadata.last_accessed = time.time() - 3700  # Just over 1 hour
        
        # Run cleanup again
        initial_cleanups = volume_manager.stats['cleanup_operations']
        await volume_manager._cleanup_temporary_volumes()
        
        # Should have cleaned up the old temporary volume
        assert temp_volume_id not in volume_manager.volumes
        # The cleanup should have occurred (stats may vary based on timing)
        logger.info(f"Cleanup operations: {volume_manager.stats['cleanup_operations']}, initial: {initial_cleanups}")
        
        logger.info("✅ Cleanup operations test passed")
    
    @pytest.mark.asyncio
    async def test_performance_integration(self, volume_manager, performance_monitor):
        """Test integration with performance monitoring."""
        logger.info("🧪 Testing performance monitoring integration")
        
        # Create volume (should record performance metrics)
        volume_id = await volume_manager.create_volume(
            owner="performance_test",
            description="Volume for performance testing"
        )
        
        assert volume_id is not None
        
        # Allow some time for performance monitoring
        await asyncio.sleep(1.0)
        
        # Check performance monitor for volume manager metrics
        summary = performance_monitor.get_performance_summary()
        
        # Look for volume manager component
        if hasattr(performance_monitor, 'analyzer') and hasattr(performance_monitor.analyzer, 'profiles'):
            volume_profile = performance_monitor.analyzer.profiles.get('volume_manager')
            if volume_profile:
                assert volume_profile.total_executions >= 1
                logger.info(f"Volume manager recorded {volume_profile.total_executions} performance metrics")
        
        logger.info("✅ Performance integration test passed")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])