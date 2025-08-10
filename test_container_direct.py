#!/usr/bin/env python3
"""
Direct Container Execution Test - Debug base level execution issues
"""

import asyncio
import logging
from orchestrator.security.docker_manager import EnhancedDockerManager
from orchestrator.analytics.performance_monitor import PerformanceMonitor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_direct_execution():
    """Test direct container execution."""
    logger.info("🧪 Testing direct container execution...")
    
    # Create components
    performance_monitor = PerformanceMonitor(collection_interval=0.5)
    docker_manager = EnhancedDockerManager(enable_advanced_pooling=False, performance_monitor=performance_monitor)
    
    try:
        # Start monitoring systems
        await performance_monitor.start_monitoring()
        await docker_manager.start_background_tasks()
        
        logger.info("✅ Systems started")
        
        # Create a simple Python container
        container = await docker_manager.create_secure_container(
            image="python:3.11-slim",
            name="test_direct_python"
        )
        
        logger.info(f"✅ Created container: {container.container_id}")
        
        # Test 1: Simple echo command
        logger.info("🧪 Testing simple echo command...")
        result1 = await docker_manager.execute_in_container(
            container,
            'echo "Hello World"',
            timeout=10
        )
        
        logger.info(f"Echo result: {result1}")
        
        # Test 2: Python command
        logger.info("🧪 Testing Python command...")
        result2 = await docker_manager.execute_in_container(
            container,
            'python3 -c "print(\'Hello from Python\')"',
            timeout=10
        )
        
        logger.info(f"Python result: {result2}")
        
        # Test 3: File write with base64 (as in multi-language executor)
        logger.info("🧪 Testing base64 file write...")
        
        import base64
        test_code = "print('Hello from file!')"
        code_b64 = base64.b64encode(test_code.encode('utf-8')).decode('ascii')
        
        result3 = await docker_manager.execute_in_container(
            container,
            f"echo '{code_b64}' | base64 -d > test.py",
            timeout=10
        )
        
        logger.info(f"Base64 write result: {result3}")
        
        # Test 4: Execute the written file
        logger.info("🧪 Testing execute written file...")
        result4 = await docker_manager.execute_in_container(
            container,
            "python3 test.py",
            timeout=10
        )
        
        logger.info(f"Execute file result: {result4}")
        
        return result4.get('success', False)
        
    except Exception as e:
        logger.error(f"❌ Direct execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        await docker_manager.shutdown()
        await performance_monitor.stop_monitoring()

async def main():
    """Run direct execution test.""" 
    success = await test_direct_execution()
    logger.info(f"✅ Test {'PASSED' if success else 'FAILED'}")
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)