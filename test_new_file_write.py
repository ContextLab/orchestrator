#!/usr/bin/env python3
"""
Test New File Write - Test the updated file writing approach
"""

import asyncio
import logging
from orchestrator.security.docker_manager import EnhancedDockerManager
from orchestrator.analytics.performance_monitor import PerformanceMonitor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_new_file_write():
    """Test new file writing approaches."""
    logger.info("🧪 Testing new file write approaches...")
    
    # Create components
    performance_monitor = PerformanceMonitor(collection_interval=0.5)
    docker_manager = EnhancedDockerManager(enable_advanced_pooling=False, performance_monitor=performance_monitor)
    
    try:
        # Start monitoring systems
        await performance_monitor.start_monitoring()
        await docker_manager.start_background_tasks()
        
        # Test 1: Python container with Python write method
        container1 = await docker_manager.create_secure_container(
            image="python:3.11-slim",
            name="test_python_write"
        )
        
        logger.info("🧪 Testing Python file write method...")
        
        test_code = "print('Hello from Python!')\nprint('Line 2')"
        python_write_cmd = f"python3 -c \"import sys; open('test.py', 'w').write(sys.stdin.read())\" << 'EOFCODE'\n{test_code}\nEOFCODE"
        
        result1 = await docker_manager.execute_in_container(
            container1,
            python_write_cmd,
            timeout=10
        )
        
        logger.info(f"Python write result: {result1}")
        
        # Test reading the file back
        result1_read = await docker_manager.execute_in_container(
            container1,
            "cat test.py",
            timeout=10
        )
        
        logger.info(f"Python read result: {result1_read}")
        
        # Test executing the file
        result1_exec = await docker_manager.execute_in_container(
            container1,
            "python3 test.py",
            timeout=10
        )
        
        logger.info(f"Python execute result: {result1_exec}")
        
        # Test 2: Ubuntu container with tee method
        container2 = await docker_manager.create_secure_container(
            image="ubuntu:22.04",
            name="test_bash_write"
        )
        
        logger.info("🧪 Testing Bash file write method...")
        
        bash_code = "echo 'Hello from Bash!'\necho 'Line 2'"
        tee_write_cmd = f"tee test.sh << 'EOFCODE'\n{bash_code}\nEOFCODE"
        
        result2 = await docker_manager.execute_in_container(
            container2,
            tee_write_cmd,
            timeout=10
        )
        
        logger.info(f"Bash write result: {result2}")
        
        # Test reading the file back
        result2_read = await docker_manager.execute_in_container(
            container2,
            "cat test.sh",
            timeout=10
        )
        
        logger.info(f"Bash read result: {result2_read}")
        
        # Test executing the file
        result2_exec = await docker_manager.execute_in_container(
            container2,
            "bash test.sh",
            timeout=10
        )
        
        logger.info(f"Bash execute result: {result2_exec}")
        
        return result1_exec.get('success', False) and result2_exec.get('success', False)
        
    except Exception as e:
        logger.error(f"❌ New file write test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        await docker_manager.shutdown()
        await performance_monitor.stop_monitoring()

async def main():
    """Run new file write test.""" 
    success = await test_new_file_write()
    logger.info(f"✅ Test {'PASSED' if success else 'FAILED'}")
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)