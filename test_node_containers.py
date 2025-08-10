#!/usr/bin/env python3
"""
Test Node Container Issues - Debug Node.js container startup problems
"""

import asyncio
import logging
import sys
import time
sys.path.insert(0, '/Users/jmanning/orchestrator/src')

from orchestrator.security.docker_manager import EnhancedDockerManager
from orchestrator.security.docker_manager import ResourceLimits, SecurityConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_node_containers():
    """Test different Node.js container images."""
    logger.info("🧪 Testing Node.js containers...")
    
    # Create minimal docker manager
    docker_manager = EnhancedDockerManager(enable_advanced_pooling=False, performance_monitor=None)
    
    try:
        await docker_manager.start_background_tasks()
        
        # Test different Node.js images
        node_images = [
            "node:18-alpine",
            "node:18-slim",
            "node:18"
        ]
        
        for image in node_images:
            logger.info(f"🧪 Testing {image}...")
            
            try:
                container = await docker_manager.create_secure_container(
                    image=image,
                    name=f"test_{image.replace(':', '_').replace('-', '_')}",
                    resource_limits=ResourceLimits(memory_mb=256, cpu_cores=0.5, execution_timeout=60, pids_limit=20),
                    security_config=SecurityConfig()
                )
                
                logger.info(f"✅ Created container: {container.container_id}")
                
                # Wait longer for Node.js containers to fully start
                await asyncio.sleep(5)
                
                # Check if container is still running
                try:
                    result = await docker_manager.execute_in_container(
                        container,
                        'echo "Container alive"',
                        timeout=10
                    )
                    
                    if result.get('success'):
                        logger.info(f"✅ {image} container is responsive: {result.get('output', '')}")
                        
                        # Test Node.js specific command
                        result2 = await docker_manager.execute_in_container(
                            container,
                            'node --version',
                            timeout=10
                        )
                        
                        if result2.get('success'):
                            logger.info(f"✅ Node.js version: {result2.get('output', '')}")
                        else:
                            logger.error(f"❌ Node.js command failed: {result2.get('error', '')}")
                    else:
                        logger.error(f"❌ Container not responsive: {result.get('error', '')}")
                        
                except Exception as e:
                    logger.error(f"❌ Container execution failed: {e}")
                
            except Exception as e:
                logger.error(f"❌ Failed to create {image} container: {e}")
            
            logger.info("-" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Node container test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await docker_manager.shutdown()

async def main():
    """Run node container test.""" 
    success = await test_node_containers()
    logger.info(f"✅ Test {'COMPLETED' if success else 'FAILED'}")
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)