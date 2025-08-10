#!/usr/bin/env python3
"""
Test File Write Fix - Test the new printf-based file writing
"""

import asyncio
import logging
from orchestrator.security.docker_manager import EnhancedDockerManager
from orchestrator.analytics.performance_monitor import PerformanceMonitor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_file_write():
    """Test file writing with new method."""
    logger.info("🧪 Testing file write fix...")
    
    # Create components
    performance_monitor = PerformanceMonitor(collection_interval=0.5)
    docker_manager = EnhancedDockerManager(enable_advanced_pooling=False, performance_monitor=performance_monitor)
    
    try:
        # Start monitoring systems
        await performance_monitor.start_monitoring()
        await docker_manager.start_background_tasks()
        
        # Create a simple Python container
        container = await docker_manager.create_secure_container(
            image="python:3.11-slim",
            name="test_file_write"
        )
        
        # Test the new printf method
        logger.info("🧪 Testing printf file write method...")
        
        test_code = "print('Hello from file!')\nprint('Line 2')"
        # Escape single quotes and backslashes for safe shell execution
        escaped_code = test_code.replace('\\\\', '\\\\\\\\').replace("'", "'\"'\"'")
        
        result1 = await docker_manager.execute_in_container(
            container,
            f"printf '%s' '{escaped_code}' > test.py",
            timeout=10
        )
        
        logger.info(f"Write result: {result1}")
        
        # Test reading the file back
        result2 = await docker_manager.execute_in_container(
            container,
            "cat test.py",
            timeout=10
        )
        
        logger.info(f"Read result: {result2}")
        
        # Test executing the file
        result3 = await docker_manager.execute_in_container(
            container,
            "python3 test.py",
            timeout=10
        )
        
        logger.info(f"Execute result: {result3}")
        
        return result3.get('success', False)
        
    except Exception as e:
        logger.error(f"❌ File write test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        await docker_manager.shutdown()
        await performance_monitor.stop_monitoring()

async def main():
    """Run file write test.""" 
    success = await test_file_write()
    logger.info(f"✅ Test {'PASSED' if success else 'FAILED'}")
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)