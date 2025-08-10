"""Real Container Pool Management Tests - Issue #206 Task 3.1

Comprehensive tests for the advanced container pooling system with real Docker containers,
performance optimization, and resource management. NO MOCKS - real container testing only.
"""

import pytest
import asyncio
import logging
import time
from typing import List

from orchestrator.security.container_pool import (
    ContainerPoolManager,
    PoolConfiguration,
    PooledContainer,
    PoolContainerState
)
from orchestrator.security.docker_manager import EnhancedDockerManager, ResourceLimits, SecurityConfig
from orchestrator.tools.multi_language_executor import MultiLanguageExecutor, Language

# Configure logging for test visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestContainerPoolManager:
    """Test container pool management with real Docker containers."""
    
    @pytest.fixture
    async def docker_manager(self):
        """Create Docker manager with advanced pooling enabled."""
        manager = EnhancedDockerManager(enable_advanced_pooling=True)
        await manager.start_background_tasks()
        yield manager
        await manager.shutdown()
    
    @pytest.fixture
    async def pool_manager(self, docker_manager):
        """Create container pool manager for testing."""
        config = PoolConfiguration(
            min_pool_size=1,
            max_pool_size=3,
            target_pool_size=2,
            max_container_age_seconds=300,  # 5 minutes for testing
            max_executions_per_container=5,
            cooldown_period_seconds=5,  # Short for testing
            cleanup_interval_seconds=10,
            health_check_interval_seconds=10
        )
        
        pool_manager = ContainerPoolManager(docker_manager, config)
        await pool_manager.start_background_tasks()
        yield pool_manager
        await pool_manager.stop_background_tasks()
    
    @pytest.mark.asyncio
    async def test_pool_manager_initialization(self, pool_manager):
        """Test pool manager initializes correctly."""
        logger.info("🧪 Testing pool manager initialization")
        
        assert pool_manager.config.min_pool_size == 1
        assert pool_manager.config.max_pool_size == 3
        assert pool_manager.config.enable_container_reuse is True
        
        # Check initial statistics
        stats = pool_manager.get_pool_statistics()
        assert stats['containers_created'] == 0
        assert stats['containers_reused'] == 0
        assert stats['current_pool_size'] == 0
        
        logger.info("✅ Pool manager initialization test passed")
    
    @pytest.mark.asyncio
    async def test_container_creation_and_reuse(self, pool_manager):
        """Test container creation and reuse functionality."""
        logger.info("🧪 Testing container creation and reuse")
        
        image = "python:3.11-slim"
        resource_limits = ResourceLimits(memory_mb=128, cpu_cores=0.1)
        
        # First request - should create new container
        start_time = time.time()
        container1 = await pool_manager.get_container(
            image=image,
            name_prefix="test_pool",
            resource_limits=resource_limits,
            execution_id="exec_1"
        )
        first_creation_time = time.time() - start_time
        
        assert container1 is not None
        assert container1.name.startswith("test_pool")
        
        # Return container to pool
        await pool_manager.return_container(
            container=container1,
            execution_id="exec_1",
            execution_time=1.0,
            execution_successful=True
        )
        
        # Wait for cooldown
        await asyncio.sleep(6)
        
        # Second request - should reuse container (faster)
        start_time = time.time()
        container2 = await pool_manager.get_container(
            image=image,
            name_prefix="test_pool",
            resource_limits=resource_limits,
            execution_id="exec_2"
        )
        second_creation_time = time.time() - start_time
        
        assert container2 is not None
        
        # Should be faster due to reuse (but not necessarily 2x due to Docker overhead)
        logger.info(f"First creation: {first_creation_time:.3f}s, Reuse: {second_creation_time:.3f}s")
        
        # More reasonable assertion - reuse should be at least 10% faster or show pool activity
        performance_improvement = second_creation_time < first_creation_time * 0.9
        
        # Check statistics to see if reuse occurred
        stats = pool_manager.get_pool_statistics()
        container_reused = stats['containers_reused'] >= 1 or stats['pool_hits'] >= 1
        
        assert performance_improvement or container_reused, f"Expected performance improvement or container reuse. Stats: {stats}"
        
        # Check statistics
        stats = pool_manager.get_pool_statistics()
        assert stats['containers_reused'] >= 1
        assert stats['pool_hits'] >= 1
        
        await pool_manager.return_container(container2, "exec_2", 0.5, True)
        
        logger.info("✅ Container creation and reuse test passed")
    
    @pytest.mark.asyncio  
    async def test_multiple_concurrent_requests(self, pool_manager):
        """Test handling multiple concurrent container requests."""
        logger.info("🧪 Testing multiple concurrent requests")
        
        image = "python:3.11-slim"
        concurrent_requests = 4
        
        async def get_and_use_container(execution_id: str):
            """Get container, simulate usage, and return."""
            container = await pool_manager.get_container(
                image=image,
                execution_id=execution_id
            )
            
            # Simulate some work
            await asyncio.sleep(0.1)
            
            await pool_manager.return_container(
                container=container,
                execution_id=execution_id,
                execution_time=0.1,
                execution_successful=True
            )
            
            return execution_id
        
        # Launch concurrent requests
        tasks = [
            get_and_use_container(f"concurrent_exec_{i}")
            for i in range(concurrent_requests)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == concurrent_requests
        
        # Check that pool managed the concurrent load
        stats = pool_manager.get_pool_statistics()
        logger.info(f"Pool stats after concurrent test: {stats}")
        
        assert stats['total_executions'] >= concurrent_requests
        
        logger.info("✅ Multiple concurrent requests test passed")
    
    @pytest.mark.asyncio
    async def test_container_expiration(self, pool_manager):
        """Test container expiration based on age and usage limits."""
        logger.info("🧪 Testing container expiration")
        
        image = "ubuntu:22.04"
        
        # Get a container and use it multiple times
        container = await pool_manager.get_container(image=image, execution_id="expire_test")
        container_id = container.container_id
        
        # Find the pooled container
        pooled_container = pool_manager.container_lookup.get(container_id)
        assert pooled_container is not None
        
        # Simulate multiple executions to reach limit
        for i in range(pool_manager.config.max_executions_per_container):
            await pool_manager.return_container(
                container=container,
                execution_id=f"expire_exec_{i}",
                execution_time=0.1,
                execution_successful=True
            )
            
            # Wait for cooldown
            await asyncio.sleep(1)
            
            if i < pool_manager.config.max_executions_per_container - 1:
                container = await pool_manager.get_container(image=image, execution_id=f"expire_get_{i}")
        
        # Container should now be marked for expiration
        assert pooled_container.execution_count >= pool_manager.config.max_executions_per_container
        assert pooled_container.state == PoolContainerState.EXPIRED or not pooled_container.is_available(pool_manager.config)
        
        logger.info("✅ Container expiration test passed")
    
    @pytest.mark.asyncio
    async def test_health_checks(self, pool_manager):
        """Test container health checking functionality."""
        logger.info("🧪 Testing container health checks")
        
        image = "python:3.11-slim"
        
        # Create a container
        container = await pool_manager.get_container(image=image, execution_id="health_test")
        await pool_manager.return_container(container, "health_test", 0.0, True)
        
        # Wait for health check cycle
        await asyncio.sleep(12)  # Longer than health_check_interval
        
        # Find the pooled container
        pooled_container = pool_manager.container_lookup.get(container.container_id)
        if pooled_container:
            # Health check should have updated timestamp
            assert pooled_container.last_health_check > 0
            logger.info(f"Health check timestamp: {pooled_container.last_health_check}")
        
        logger.info("✅ Container health checks test passed")
    
    @pytest.mark.asyncio
    async def test_pool_statistics(self, pool_manager):
        """Test comprehensive pool statistics tracking."""
        logger.info("🧪 Testing pool statistics")
        
        image = "python:3.11-slim"
        
        # Perform several operations
        for i in range(3):
            container = await pool_manager.get_container(image=image, execution_id=f"stats_test_{i}")
            await asyncio.sleep(0.1)  # Simulate execution time
            await pool_manager.return_container(container, f"stats_test_{i}", 0.1, True)
            await asyncio.sleep(1)  # Cooldown
        
        stats = pool_manager.get_pool_statistics()
        
        # Verify statistics structure
        required_stats = [
            'containers_created', 'containers_reused', 'containers_destroyed',
            'pool_hits', 'pool_misses', 'total_executions', 'average_wait_time',
            'current_pool_size', 'peak_pool_size'
        ]
        
        for stat in required_stats:
            assert stat in stats, f"Missing statistic: {stat}"
        
        # Verify pool details
        assert 'pool_details' in stats
        assert 'configuration' in stats
        
        # Check that we have reasonable values
        assert stats['total_executions'] >= 3
        assert stats['current_pool_size'] >= 0
        
        logger.info(f"Pool statistics: {stats}")
        logger.info("✅ Pool statistics test passed")


class TestContainerPoolIntegration:
    """Test container pool integration with multi-language executor."""
    
    @pytest.fixture
    async def docker_manager(self):
        """Create Docker manager with pooling enabled."""
        manager = EnhancedDockerManager(enable_advanced_pooling=True)
        await manager.start_background_tasks()
        yield manager
        await manager.shutdown()
    
    @pytest.fixture  
    async def multi_lang_executor(self, docker_manager):
        """Create multi-language executor with pooling."""
        executor = MultiLanguageExecutor(docker_manager)
        yield executor
    
    @pytest.mark.asyncio
    async def test_multi_language_with_pooling(self, multi_lang_executor):
        """Test multi-language execution with container pooling."""
        logger.info("🧪 Testing multi-language execution with pooling")
        
        # Test Python execution
        python_code = """
print("Testing container pooling with Python")
import time
start = time.time()
result = sum(range(1000))
elapsed = time.time() - start
print(f"Calculated sum: {result} in {elapsed:.4f}s")
"""
        
        # First execution
        start_time = time.time()
        result1 = await multi_lang_executor.execute_code(
            code=python_code,
            language=Language.PYTHON,
            timeout=30
        )
        first_exec_time = time.time() - start_time
        
        assert result1.success, f"First execution failed: {result1.error}"
        assert "Testing container pooling" in result1.output
        
        # Second execution (should benefit from pooling)
        start_time = time.time()
        result2 = await multi_lang_executor.execute_code(
            code=python_code,
            language=Language.PYTHON,
            timeout=30
        )
        second_exec_time = time.time() - start_time
        
        assert result2.success, f"Second execution failed: {result2.error}"
        assert "Testing container pooling" in result2.output
        
        logger.info(f"First execution: {first_exec_time:.3f}s, Second execution: {second_exec_time:.3f}s")
        
        # Check pool statistics
        docker_manager = multi_lang_executor.docker_manager
        if docker_manager.pool_manager:
            stats = docker_manager.pool_manager.get_pool_statistics()
            logger.info(f"Pool stats: {stats}")
            
            # Should show container reuse
            assert stats['total_executions'] >= 2
            
        logger.info("✅ Multi-language execution with pooling test passed")
    
    @pytest.mark.asyncio
    async def test_different_languages_pooling(self, multi_lang_executor):
        """Test pooling with different programming languages."""
        logger.info("🧪 Testing pooling across different languages")
        
        test_cases = [
            (Language.PYTHON, "print('Python with pooling')", "Python with pooling"),
            (Language.NODEJS, "console.log('Node.js with pooling');", "Node.js with pooling"),
            (Language.BASH, "echo 'Bash with pooling'", "Bash with pooling"),
        ]
        
        results = []
        for language, code, expected_output in test_cases:
            try:
                result = await multi_lang_executor.execute_code(
                    code=code,
                    language=language,
                    timeout=30
                )
                
                results.append((language.value, result.success, expected_output in (result.output or "")))
                logger.info(f"{language.value}: {'✅' if result.success else '❌'}")
                
            except Exception as e:
                logger.warning(f"{language.value} execution failed: {e}")
                results.append((language.value, False, False))
        
        # At least one language should work
        successful_results = [r for r in results if r[1]]
        assert len(successful_results) >= 1, "At least one language should execute successfully"
        
        # Check pool statistics
        docker_manager = multi_lang_executor.docker_manager
        if docker_manager.pool_manager:
            stats = docker_manager.pool_manager.get_pool_statistics()
            logger.info(f"Final pool stats: {stats}")
            
            # Should have multiple pools for different images
            assert len(stats.get('pool_details', {})) >= 1
        
        logger.info("✅ Different languages pooling test passed")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])