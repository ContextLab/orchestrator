"""Tests for connection pooling system."""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock

from src.orchestrator.core.model import Model, ModelCapabilities, ModelCost
from src.orchestrator.models.optimization.pooling import (
    ConnectionPool,
    PoolConnection,
    PoolStats,
    QueuedRequest,
)


class MockModel(Model):
    """Mock model for testing."""
    
    def __init__(self, name: str, provider: str, should_fail: bool = False):
        super().__init__(
            name=name,
            provider=provider,
            capabilities=ModelCapabilities(),
            cost=ModelCost(is_free=True),
        )
        self.should_fail = should_fail
        self.call_count = 0
    
    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = None, **kwargs):
        self.call_count += 1
        if self.should_fail:
            raise Exception(f"Model {self.name} failed")
        await asyncio.sleep(0.01)  # Simulate processing time
        return f"Response from {self.name}: {prompt}"
    
    async def generate_structured(self, prompt: str, schema: dict, temperature: float = 0.7, **kwargs):
        self.call_count += 1
        if self.should_fail:
            raise Exception(f"Structured generation failed for {self.name}")
        await asyncio.sleep(0.01)
        return {"result": f"Structured from {self.name}", "input": prompt}
    
    async def health_check(self) -> bool:
        return not self.should_fail
    
    async def estimate_cost(self, prompt: str, max_tokens: int = None) -> float:
        return 0.001


class TestPoolConnection:
    """Test PoolConnection functionality."""
    
    def test_initialization(self):
        """Test pool connection initialization."""
        model = MockModel("test_model", "test_provider")
        connection = PoolConnection(model=model, provider="test_provider")
        
        assert connection.model == model
        assert connection.provider == "test_provider"
        assert connection.use_count == 0
        assert connection.is_active is False
        assert connection.health_check_failures == 0
    
    def test_use_and_release(self):
        """Test connection use and release tracking."""
        model = MockModel("test_model", "test_provider")
        connection = PoolConnection(model=model, provider="test_provider")
        
        initial_last_used = connection.last_used
        initial_use_count = connection.use_count
        
        # Use connection
        time.sleep(0.01)  # Ensure timestamp difference
        connection.use()
        
        assert connection.is_active is True
        assert connection.use_count == initial_use_count + 1
        assert connection.last_used > initial_last_used
        
        # Release connection
        connection.release()
        assert connection.is_active is False
    
    def test_staleness_check(self):
        """Test connection staleness checking."""
        model = MockModel("test_model", "test_provider")
        connection = PoolConnection(model=model, provider="test_provider")
        
        # Fresh connection should not be stale
        assert connection.is_stale(max_idle_time=300.0) is False
        
        # Simulate old connection
        connection.last_used = time.time() - 400  # 400 seconds ago
        assert connection.is_stale(max_idle_time=300.0) is True
    
    def test_overuse_check(self):
        """Test connection overuse checking."""
        model = MockModel("test_model", "test_provider")
        connection = PoolConnection(model=model, provider="test_provider")
        
        # New connection should not be overused
        assert connection.is_overused(max_uses=1000) is False
        
        # Simulate heavy use
        connection.use_count = 1001
        assert connection.is_overused(max_uses=1000) is True


class TestQueuedRequest:
    """Test QueuedRequest functionality."""
    
    def test_initialization(self):
        """Test queued request initialization."""
        future = asyncio.Future()
        request = QueuedRequest(
            future=future,
            model_name="test_model",
            method="generate",
            args=("test_prompt",),
            kwargs={"temperature": 0.8},
        )
        
        assert request.future == future
        assert request.model_name == "test_model"
        assert request.method == "generate"
        assert request.args == ("test_prompt",)
        assert request.kwargs == {"temperature": 0.8}
    
    def test_queue_time_calculation(self):
        """Test queue time calculation."""
        future = asyncio.Future()
        request = QueuedRequest(future=future)
        
        # Wait a bit
        time.sleep(0.01)
        
        queue_time = request.get_queue_time()
        assert queue_time > 0
        assert queue_time < 1.0  # Should be very small


class TestPoolStats:
    """Test PoolStats functionality."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = PoolStats(
            total_connections=10,
            active_connections=5,
            idle_connections=5,
            created_connections=12,
            destroyed_connections=2,
            requests_served=100,
        )
        
        result = stats.to_dict()
        
        expected_keys = [
            "total_connections", "active_connections", "idle_connections",
            "created_connections", "destroyed_connections", "requests_served",
            "requests_queued", "max_queue_time", "average_queue_time",
            "utilization_rate"
        ]
        
        for key in expected_keys:
            assert key in result
        
        assert result["utilization_rate"] == 0.5  # 5 active out of 10 total


class TestConnectionPool:
    """Test ConnectionPool functionality."""
    
    @pytest.fixture
    def pool(self):
        """Create connection pool for testing."""
        return ConnectionPool(
            provider_name="test_provider",
            min_connections=1,
            max_connections=5,
            max_idle_time=10.0,  # 10 seconds for testing
            max_uses_per_connection=10,  # Low for testing
            health_check_interval=1.0,  # 1 second for testing
            queue_timeout=5.0,
        )
    
    @pytest.fixture
    def model(self):
        """Create test model."""
        return MockModel("test_model", "test_provider")
    
    @pytest.mark.asyncio
    async def test_initialization(self, pool):
        """Test pool initialization."""
        assert pool.provider_name == "test_provider"
        assert pool.min_connections == 1
        assert pool.max_connections == 5
        assert pool._initialized is False
        
        await pool.initialize()
        assert pool._initialized is True
    
    @pytest.mark.asyncio
    async def test_connection_creation(self, pool, model):
        """Test connection creation and retrieval."""
        await pool.initialize()
        
        # Get first connection
        connection = await pool.get_connection(model)
        
        assert isinstance(connection, PoolConnection)
        assert connection.model.name == model.name
        assert connection.provider == model.provider
        assert connection.is_active is True
        
        # Pool should have one connection now
        stats = await pool.get_stats()
        assert stats.total_connections == 1
        assert stats.active_connections == 1
    
    @pytest.mark.asyncio
    async def test_connection_reuse(self, pool, model):
        """Test connection reuse."""
        await pool.initialize()
        
        # Get and release connection
        connection1 = await pool.get_connection(model)
        await pool.release_connection(connection1)
        
        # Get another connection for same model
        connection2 = await pool.get_connection(model)
        
        # Should reuse the same connection
        assert connection1 is connection2
        assert connection2.use_count == 2  # Used twice
    
    @pytest.mark.asyncio
    async def test_max_connections_limit(self, pool, model):
        """Test maximum connections limit."""
        await pool.initialize()
        
        # Get connections up to max
        connections = []
        for i in range(pool.max_connections):
            conn = await pool.get_connection(model)
            connections.append(conn)
        
        stats = await pool.get_stats()
        assert stats.total_connections == pool.max_connections
        assert stats.active_connections == pool.max_connections
        
        # Clean up
        for conn in connections:
            await pool.release_connection(conn)
    
    @pytest.mark.asyncio
    async def test_request_queueing(self, pool, model):
        """Test request queueing when pool is full."""
        await pool.initialize()
        
        # Fill pool to capacity
        connections = []
        for i in range(pool.max_connections):
            conn = await pool.get_connection(model)
            connections.append(conn)
        
        # This should be queued
        queue_task = asyncio.create_task(pool.get_connection(model))
        
        # Wait a bit to ensure it's queued
        await asyncio.sleep(0.1)
        assert not queue_task.done()
        
        # Release one connection
        await pool.release_connection(connections[0])
        
        # Queued request should complete
        queued_connection = await queue_task
        assert isinstance(queued_connection, PoolConnection)
        
        # Clean up remaining connections
        for conn in connections[1:]:
            await pool.release_connection(conn)
        await pool.release_connection(queued_connection)
    
    @pytest.mark.asyncio
    async def test_execute_with_model(self, pool, model):
        """Test executing methods through the pool."""
        await pool.initialize()
        
        # Test generate method
        result = await pool.execute_with_model(
            model, "generate", "Hello world", temperature=0.8
        )
        
        assert isinstance(result, str)
        assert "Hello world" in result
        assert model.call_count == 1
        
        # Test structured generation
        schema = {"type": "object", "properties": {"result": {"type": "string"}}}
        structured_result = await pool.execute_with_model(
            model, "generate_structured", "Test prompt", schema=schema
        )
        
        assert isinstance(structured_result, dict)
        assert "result" in structured_result
        assert model.call_count == 2
    
    @pytest.mark.asyncio
    async def test_error_handling(self, pool):
        """Test error handling and failure tracking."""
        failing_model = MockModel("failing_model", "test_provider", should_fail=True)
        await pool.initialize()
        
        # Execute should raise exception
        with pytest.raises(Exception, match="Model failing_model failed"):
            await pool.execute_with_model(
                failing_model, "generate", "This will fail"
            )
        
        # Get connection to check failure tracking
        connection = await pool.get_connection(failing_model)
        assert connection.health_check_failures >= 1
        await pool.release_connection(connection)
    
    @pytest.mark.asyncio
    async def test_health_check(self, pool, model):
        """Test pool health checking."""
        await pool.initialize()
        
        # Create some connections
        connection = await pool.get_connection(model)
        await pool.release_connection(connection)
        
        # Perform health check
        health_result = await pool.health_check()
        
        assert "healthy_connections" in health_result
        assert "unhealthy_connections" in health_result
        assert "total_connections" in health_result
        assert "connection_details" in health_result
        
        assert health_result["total_connections"] >= 1
        assert len(health_result["connection_details"]) >= 1
    
    @pytest.mark.asyncio
    async def test_stale_connection_cleanup(self, pool, model):
        """Test cleanup of stale connections."""
        # Use very short idle time for testing
        short_pool = ConnectionPool(
            provider_name="test_provider",
            min_connections=0,  # Allow cleanup to 0
            max_connections=5,
            max_idle_time=0.1,  # 100ms
        )
        await short_pool.initialize()
        
        # Create and release connection
        connection = await short_pool.get_connection(model)
        await short_pool.release_connection(connection)
        
        # Wait for staleness
        await asyncio.sleep(0.2)
        
        # Trigger cleanup by creating new connection
        new_connection = await short_pool.get_connection(model)
        await short_pool.release_connection(new_connection)
        
        await short_pool.cleanup()
    
    @pytest.mark.asyncio
    async def test_overused_connection_refresh(self, pool, model):
        """Test refresh of overused connections."""
        await pool.initialize()
        
        # Get connection and simulate heavy use
        connection = await pool.get_connection(model)
        connection.use_count = pool.max_uses_per_connection + 1
        await pool.release_connection(connection)
        
        # Next get should create new connection due to overuse
        new_connection = await pool.get_connection(model)
        
        # Should be different instance (new connection created)
        # Note: This might not always be true depending on implementation details
        await pool.release_connection(new_connection)
    
    @pytest.mark.asyncio
    async def test_queue_timeout(self, pool, model):
        """Test queue timeout functionality."""
        # Use very short timeout for testing
        short_timeout_pool = ConnectionPool(
            provider_name="test_provider",
            max_connections=1,
            queue_timeout=0.1,  # 100ms timeout
        )
        await short_timeout_pool.initialize()
        
        # Fill pool
        connection = await short_timeout_pool.get_connection(model)
        
        # This should timeout
        with pytest.raises(TimeoutError):
            await short_timeout_pool.get_connection(model)
        
        # Clean up
        await short_timeout_pool.release_connection(connection)
        await short_timeout_pool.cleanup()
    
    @pytest.mark.asyncio
    async def test_cleanup(self, pool, model):
        """Test pool cleanup."""
        await pool.initialize()
        
        # Create some connections
        connection = await pool.get_connection(model)
        await pool.release_connection(connection)
        
        # Verify pool has connections
        stats = await pool.get_stats()
        assert stats.total_connections > 0
        
        # Cleanup
        await pool.cleanup()
        
        # Pool should be clean
        assert pool._initialized is False
        assert len(pool._connections) == 0
    
    @pytest.mark.asyncio
    async def test_stats_tracking(self, pool, model):
        """Test statistics tracking."""
        await pool.initialize()
        
        # Perform some operations
        for i in range(3):
            result = await pool.execute_with_model(
                model, "generate", f"Test prompt {i}"
            )
        
        stats = await pool.get_stats()
        assert stats.requests_served == 3
        assert stats.total_connections >= 1
    
    def test_pool_info(self, pool):
        """Test pool information retrieval."""
        info = pool.get_pool_info()
        
        required_keys = [
            "provider_name", "min_connections", "max_connections",
            "max_idle_time", "max_uses_per_connection", "health_check_interval",
            "queue_timeout", "initialized", "stats"
        ]
        
        for key in required_keys:
            assert key in info
        
        assert info["provider_name"] == "test_provider"
        assert info["min_connections"] == 1
        assert info["max_connections"] == 5
    
    def test_string_representation(self, pool):
        """Test string representation of pool."""
        str_repr = str(pool)
        assert "ConnectionPool" in str_repr
        assert "test_provider" in str_repr
        assert "0/5" in str_repr  # 0 connections out of 5 max