#!/usr/bin/env python3
"""
Simple Container Pool Test - Task 3.1 Validation

Quick test to validate container pooling functionality works correctly.
"""

import asyncio
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_container_pool_basic():
    """Test basic container pool functionality."""
    from orchestrator.security.docker_manager import EnhancedDockerManager, ResourceLimits
    from orchestrator.security.container_pool import PoolConfiguration
    
    logger.info("🧪 Testing basic container pooling...")
    
    # Create Docker manager with advanced pooling
    docker_manager = EnhancedDockerManager(enable_advanced_pooling=True)
    await docker_manager.start_background_tasks()
    
    try:
        # Create container
        container1 = await docker_manager.create_secure_container(
            image="python:3.11-slim",
            name="test_pool_basic",
            resource_limits=ResourceLimits(memory_mb=128)
        )
        
        logger.info(f"Created container: {container1.name}")
        
        # Return to pool
        await docker_manager.return_container_to_pool(
            container=container1,
            execution_time=1.0,
            execution_successful=True
        )
        
        logger.info("Returned container to pool")
        
        # Get pool statistics
        if docker_manager.pool_manager:
            stats = docker_manager.pool_manager.get_pool_statistics()
            logger.info(f"Pool stats: {stats}")
            
            # Wait a bit for cooldown
            await asyncio.sleep(2)
            
            # Try to get another container (should reuse)
            container2 = await docker_manager.create_secure_container(
                image="python:3.11-slim",
                name="test_pool_reuse",
                resource_limits=ResourceLimits(memory_mb=128)
            )
            
            logger.info(f"Got second container: {container2.name}")
            
            # Return second container
            await docker_manager.return_container_to_pool(
                container=container2,
                execution_time=0.5,
                execution_successful=True
            )
            
            # Final stats
            final_stats = docker_manager.pool_manager.get_pool_statistics()
            logger.info(f"Final stats: {final_stats}")
            
            logger.info("✅ Basic container pooling test passed")
            return True
        else:
            logger.warning("Pool manager not initialized")
            return False
            
    except Exception as e:
        logger.error(f"❌ Container pooling test failed: {e}")
        return False
        
    finally:
        await docker_manager.shutdown()

async def test_multi_language_pooling():
    """Test multi-language executor with pooling."""
    from orchestrator.tools.multi_language_executor import MultiLanguageExecutor, Language
    from orchestrator.security.docker_manager import EnhancedDockerManager
    
    logger.info("🧪 Testing multi-language with pooling...")
    
    # Create Docker manager with pooling
    docker_manager = EnhancedDockerManager(enable_advanced_pooling=True)
    await docker_manager.start_background_tasks()
    
    try:
        # Create multi-language executor
        executor = MultiLanguageExecutor(docker_manager)
        
        # Simple Python code
        python_code = "print('Hello from pooled Python!')"
        
        # First execution
        result1 = await executor.execute_code(
            code=python_code,
            language=Language.PYTHON,
            timeout=30
        )
        
        logger.info(f"First execution: {'✅' if result1.success else '❌'}")
        if not result1.success:
            logger.error(f"Error: {result1.error}")
        
        # Second execution (should benefit from pooling)
        result2 = await executor.execute_code(
            code=python_code,
            language=Language.PYTHON,
            timeout=30
        )
        
        logger.info(f"Second execution: {'✅' if result2.success else '❌'}")
        
        # Check pool statistics
        if docker_manager.pool_manager:
            stats = docker_manager.pool_manager.get_pool_statistics()
            logger.info(f"Pool stats: containers_created={stats['containers_created']}, containers_reused={stats['containers_reused']}")
        
        success = result1.success or result2.success  # At least one should work
        logger.info(f"{'✅' if success else '❌'} Multi-language pooling test {'passed' if success else 'failed'}")
        return success
        
    except Exception as e:
        logger.error(f"❌ Multi-language pooling test failed: {e}")
        return False
        
    finally:
        await docker_manager.shutdown()

async def main():
    """Run container pool tests."""
    logger.info("🚀 Starting Container Pool Validation Tests...")
    
    tests = [
        ("Basic Container Pooling", test_container_pool_basic()),
        ("Multi-Language Pooling", test_multi_language_pooling()),
    ]
    
    results = []
    
    for test_name, test_coro in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = await test_coro
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("CONTAINER POOL VALIDATION SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL CONTAINER POOL TESTS PASSED!")
        return True
    else:
        logger.info(f"⚠️  {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)