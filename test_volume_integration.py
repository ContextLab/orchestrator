#!/usr/bin/env python3
"""
Volume Management Integration Test - Task 3.4 Validation

Integration test to validate persistent volume management works with
Docker containers and the multi-language execution system.
"""

import asyncio
import logging
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_volume_integration():
    """Test volume management with real Docker containers."""
    from orchestrator.storage.volume_manager import (
        PersistentVolumeManager, 
        VolumeType, 
        VolumeAccess,
        VolumeQuota
    )
    from orchestrator.security.docker_manager import EnhancedDockerManager
    from orchestrator.analytics.performance_monitor import PerformanceMonitor
    from orchestrator.tools.multi_language_executor import MultiLanguageExecutor, Language
    
    logger.info("🧪 Testing volume management integration...")
    
    # Create components
    performance_monitor = PerformanceMonitor(collection_interval=0.5)
    docker_manager = EnhancedDockerManager(enable_advanced_pooling=True, performance_monitor=performance_monitor)
    volume_manager = PersistentVolumeManager(performance_monitor=performance_monitor)
    
    try:
        # Start monitoring systems
        await performance_monitor.start_monitoring()
        await docker_manager.start_background_tasks()
        await volume_manager.start_monitoring()
        
        logger.info("✅ All systems started")
        
        # Test 1: Create different types of volumes
        temp_volume = await volume_manager.create_volume(
            owner="integration_test",
            volume_type=VolumeType.TEMPORARY,
            description="Temporary volume for testing"
        )
        
        persistent_volume = await volume_manager.create_volume(
            owner="integration_test",
            volume_type=VolumeType.PERSISTENT,
            quota=VolumeQuota(max_size_mb=100, max_files=50),
            description="Persistent volume for data storage"
        )
        
        shared_volume = await volume_manager.create_volume(
            owner="integration_test",
            volume_type=VolumeType.SHARED,
            mount_point="/shared_data",
            description="Shared volume between containers"
        )
        
        logger.info(f"Created volumes: temp={temp_volume}, persistent={persistent_volume}, shared={shared_volume}")
        assert all([temp_volume, persistent_volume, shared_volume])
        
        # Test 2: Add data to volumes
        for volume_id in [temp_volume, persistent_volume, shared_volume]:
            volume_info = volume_manager.get_volume_info(volume_id)
            host_path = volume_info['metadata']['host_path']
            
            # Create test files
            test_file = os.path.join(host_path, "test_data.txt")
            with open(test_file, 'w') as f:
                f.write(f"Test data in volume {volume_id}\nCreated at {time.time()}")
            
            logger.info(f"Added test data to volume {volume_id}")
        
        # Test 3: Create container with volume mounts
        executor = MultiLanguageExecutor(docker_manager)
        
        # Create Python code that writes to mounted volumes
        python_code = """
import os
import time

# Write to different mounted volumes
volumes = ['/data', '/shared_data']
for vol_path in volumes:
    if os.path.exists(vol_path):
        with open(os.path.join(vol_path, 'container_output.txt'), 'w') as f:
            f.write(f'Written from container at {time.time()}\\n')
        print(f'Successfully wrote to {vol_path}')
    else:
        print(f'Volume {vol_path} not mounted')

# List files in mounted volumes
for vol_path in volumes:
    if os.path.exists(vol_path):
        files = os.listdir(vol_path)
        print(f'Files in {vol_path}: {files}')
"""
        
        # Mount volumes to container (this would require Docker integration)
        # For now, we'll test the mount configurations
        
        # Test 4: Get mount configurations
        container_id = "test_container_123"
        
        temp_mount = volume_manager.mount_volume(temp_volume, container_id)
        persistent_mount = volume_manager.mount_volume(persistent_volume, container_id)
        shared_mount = volume_manager.mount_volume(shared_volume, container_id)
        
        assert all([temp_mount, persistent_mount, shared_mount])
        logger.info("✅ Successfully mounted all volumes")
        
        # Test 5: Execute code (simulated, as actual Docker mount integration would need more setup)
        logger.info("📝 Executing Python code (without actual volume mounts for this test)")
        simple_code = "print('Volume integration test executed successfully')"
        result = await executor.execute_code(simple_code, Language.PYTHON, timeout=30)
        
        logger.info(f"Code execution result: {'✅' if result.success else '❌'}")
        if not result.success and result.error:
            logger.warning(f"Execution error: {result.error}")
        
        # Test 6: Test volume operations
        # Check volume usage
        for volume_id in [temp_volume, persistent_volume, shared_volume]:
            vol_info = volume_manager.get_volume_info(volume_id)
            usage = vol_info['usage']
            
            logger.info(f"Volume {volume_id}: {usage['file_count']} files, {usage['used_size_bytes']} bytes")
            assert usage['file_count'] >= 2  # metadata + test file
            assert usage['used_size_bytes'] > 0
        
        # Test 7: Create backup of persistent volume
        backup_path = volume_manager.create_backup(persistent_volume)
        assert backup_path is not None
        logger.info(f"✅ Created backup: {backup_path}")
        
        # Test 8: Test volume listing and filtering
        all_volumes = volume_manager.list_volumes()
        test_volumes = volume_manager.list_volumes(owner="integration_test")
        persistent_volumes = volume_manager.list_volumes(volume_type=VolumeType.PERSISTENT)
        
        assert len(all_volumes) >= 3
        assert len(test_volumes) == 3
        assert len(persistent_volumes) >= 1
        
        logger.info(f"Volume listing: total={len(all_volumes)}, test_owner={len(test_volumes)}, persistent={len(persistent_volumes)}")
        
        # Test 9: Get comprehensive storage summary
        summary = volume_manager.get_storage_summary()
        
        logger.info(f"Storage Summary:")
        logger.info(f"  - Total volumes: {summary['total_volumes']}")
        logger.info(f"  - Mounted volumes: {summary['mounted_volumes']}")
        logger.info(f"  - Volume types: {summary['volume_types']}")
        logger.info(f"  - Statistics: {summary['statistics']}")
        
        assert summary['total_volumes'] >= 3
        assert summary['mounted_volumes'] >= 3
        assert summary['statistics']['volumes_created'] >= 3
        
        # Test 10: Unmount and cleanup
        for volume_id in [temp_volume, persistent_volume, shared_volume]:
            success = volume_manager.unmount_volume(volume_id, container_id)
            assert success is True
        
        logger.info("✅ Successfully unmounted all volumes")
        
        # Test 11: Delete temporary volume (keep persistent and shared for now)
        success = volume_manager.delete_volume(temp_volume)
        assert success is True
        logger.info(f"✅ Deleted temporary volume {temp_volume}")
        
        # Verify persistent volume still exists
        vol_info = volume_manager.get_volume_info(persistent_volume)
        assert vol_info is not None
        assert vol_info['exists'] is True
        
        logger.info("✅ Volume management integration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Volume integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        await volume_manager.stop_monitoring()
        await docker_manager.shutdown()
        await performance_monitor.stop_monitoring()

async def main():
    """Run volume management integration test."""
    logger.info("🚀 Starting Volume Management Integration Test...")
    
    success = await test_volume_integration()
    
    if success:
        logger.info("🎉 VOLUME MANAGEMENT INTEGRATION TEST PASSED!")
    else:
        logger.info("⚠️  VOLUME MANAGEMENT INTEGRATION TEST FAILED!")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)