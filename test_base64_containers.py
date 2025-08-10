#!/usr/bin/env python3
"""
Test Base64 in Containers - Test base64 availability and functionality
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

async def test_base64_containers():
    """Test base64 functionality in different containers."""
    logger.info("🧪 Testing base64 in containers...")
    
    # Create minimal docker manager
    docker_manager = EnhancedDockerManager(enable_advanced_pooling=False, performance_monitor=None)
    
    try:
        await docker_manager.start_background_tasks()
        
        # Test different container types
        test_configs = [
            {
                "image": "python:3.11-slim",
                "name": "test_python",
                "test_commands": [
                    'echo "Hello World"',
                    'which base64',
                    'echo "dGVzdA==" | base64 -d',
                    'echo "test" | base64',
                    'python3 -c "import base64; print(base64.b64decode(\'dGVzdA==\').decode())"'
                ]
            },
            {
                "image": "node:18-alpine",
                "name": "test_node",
                "test_commands": [
                    'echo "Hello World"',
                    'which base64',
                    'echo "dGVzdA==" | base64 -d',
                    'echo "test" | base64'
                ]
            },
            {
                "image": "ubuntu:22.04",
                "name": "test_ubuntu", 
                "test_commands": [
                    'echo "Hello World"',
                    'which base64',
                    'echo "dGVzdA==" | base64 -d',
                    'echo "test" | base64'
                ]
            }
        ]
        
        for config in test_configs:
            logger.info(f"🧪 Testing {config['image']}...")
            
            container = await docker_manager.create_secure_container(
                image=config['image'],
                name=config['name'],
                resource_limits=ResourceLimits(memory_mb=128, cpu_cores=0.5, execution_timeout=60, pids_limit=10),
                security_config=SecurityConfig()
            )
            
            logger.info(f"✅ Created container: {container.container_id}")
            
            # Wait for container to be fully ready
            await asyncio.sleep(3)
            
            for cmd in config['test_commands']:
                logger.info(f"🔍 Testing: {cmd}")
                result = await docker_manager.execute_in_container(
                    container,
                    cmd,
                    timeout=10
                )
                
                success = result.get('success', False)
                output = result.get('output', '').strip()
                error = result.get('error', '').strip()
                
                if success:
                    logger.info(f"✅ Success: {output}")
                else:
                    logger.error(f"❌ Failed: {error}")
            
            logger.info("-" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Base64 container test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await docker_manager.shutdown()

async def main():
    """Run base64 container test.""" 
    success = await test_base64_containers()
    logger.info(f"✅ Test {'COMPLETED' if success else 'FAILED'}")
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)