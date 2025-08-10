#!/usr/bin/env python3
"""
Debug Container Timing - Debug why non-Python containers are failing
"""

import asyncio
import logging
import sys
import time
sys.path.insert(0, '/Users/jmanning/orchestrator/src')

from orchestrator.security.docker_manager import EnhancedDockerManager
from orchestrator.security.docker_manager import ResourceLimits, SecurityConfig

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def debug_container_timing():
    """Debug container timing issues."""
    logger.info("🧪 Debugging container timing...")
    
    # Create minimal docker manager
    docker_manager = EnhancedDockerManager(enable_advanced_pooling=False, performance_monitor=None)
    
    try:
        await docker_manager.start_background_tasks()
        logger.info("✅ Docker manager started")
        
        # Test different container types with timing
        images_to_test = [
            "python:3.11-slim",
            "node:18-alpine", 
            "ubuntu:22.04"
        ]
        
        for image in images_to_test:
            logger.info(f"🧪 Testing {image}...")
            
            start_time = time.time()
            container = await docker_manager.create_secure_container(
                image=image,
                name=f"debug_{image.replace(':', '_').replace('/', '_')}",
                resource_limits=ResourceLimits(memory_mb=128, cpu_cores=0.5, execution_timeout=30, pids_limit=10),
                security_config=SecurityConfig()
            )
            create_time = time.time() - start_time
            logger.info(f"✅ Created {image} container in {create_time:.2f}s: {container.container_id}")
            
            # Wait a moment for container to fully start
            await asyncio.sleep(2)
            
            # Test simple command
            start_time = time.time()
            result = await docker_manager.execute_in_container(
                container,
                'echo "Hello World"',
                timeout=10
            )
            exec_time = time.time() - start_time
            logger.info(f"📊 Echo command result ({exec_time:.2f}s): success={result.get('success')}, output={result.get('output', '')[:50]}")
            
            if not result.get('success'):
                logger.error(f"❌ Echo failed for {image}: {result.get('error')}")
            
            # Test base64 command availability
            if result.get('success'):
                start_time = time.time()
                result2 = await docker_manager.execute_in_container(
                    container,
                    'echo "dGVzdA==" | base64 -d',
                    timeout=10
                )
                base64_time = time.time() - start_time
                logger.info(f"📊 Base64 decode result ({base64_time:.2f}s): success={result2.get('success')}, output={result2.get('output', '')[:50]}")
                
                if not result2.get('success'):
                    logger.error(f"❌ Base64 failed for {image}: {result2.get('error')}")
            
            # Check container health
            health = await docker_manager._check_container_health(container)
            logger.info(f"🏥 Container health: {health}")
            
            logger.info(f"✅ Completed testing {image}")
            logger.info("-" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Container timing debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await docker_manager.shutdown()

async def main():
    """Run container timing debug.""" 
    success = await debug_container_timing()
    logger.info(f"✅ Debug {'COMPLETED' if success else 'FAILED'}")
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)