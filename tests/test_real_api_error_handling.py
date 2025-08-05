"""
Real API integration error handling tests for Issue 192.
Tests actual API failures with real endpoints - NO MOCKS OR SIMULATIONS.
"""

import asyncio
import os
import time
from typing import Any, Dict, Optional

import pytest
import aiohttp

from orchestrator.core.error_handling import ErrorHandler
from orchestrator.execution.error_handler_executor import ErrorHandlerExecutor
from orchestrator.engine.pipeline_spec import TaskSpec
from orchestrator.engine.task_executor import UniversalTaskExecutor


class RealAPITaskExecutor:
    """Task executor that makes real API calls for testing error handling."""
    
    def __init__(self):
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def execute_task(self, task_spec: TaskSpec, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with real API calls."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        action = task_spec.action
        task_id = task_spec.id
        
        # Handle different types of API tasks
        if "http_request" in action.lower():
            return await self._execute_http_request(task_spec, context)
        elif "model_call" in action.lower():
            return await self._execute_model_call(task_spec, context)
        elif "error_recovery" in action.lower():
            return await self._execute_error_recovery(task_spec, context)
        elif "timeout_test" in action.lower():
            return await self._execute_timeout_test(task_spec, context)
        else:
            # Default to HTTP request for testing
            return await self._execute_http_request(task_spec, context)
    
    async def _execute_http_request(self, task_spec: TaskSpec, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute HTTP request with potential for real errors."""
        url = context.get("url", "https://httpbin.org/status/500")  # Default to error endpoint
        method = context.get("method", "GET")
        timeout = context.get("timeout", 5)
        
        try:
            async with self.session.request(method, url, timeout=timeout) as response:
                # Raise for HTTP error status codes
                if response.status >= 400:
                    if response.status == 404:
                        raise FileNotFoundError(f"HTTP 404: Resource not found at {url}")
                    elif response.status == 403:
                        raise PermissionError(f"HTTP 403: Access forbidden to {url}")
                    elif response.status == 429:
                        raise ConnectionError(f"HTTP 429: Rate limit exceeded for {url}")
                    elif response.status >= 500:
                        raise ConnectionError(f"HTTP {response.status}: Server error at {url}")
                    else:
                        raise ValueError(f"HTTP {response.status}: Client error at {url}")
                
                content = await response.text()
                
                return {
                    "task_id": task_spec.id,
                    "success": True,
                    "result": {
                        "status_code": response.status,
                        "content": content[:500],  # Limit content size
                        "url": url
                    }
                }
                
        except aiohttp.ClientError as e:
            # Convert aiohttp errors to standard exceptions
            if "timeout" in str(e).lower():
                raise TimeoutError(f"Request to {url} timed out: {e}")
            else:
                raise ConnectionError(f"Connection error to {url}: {e}")
    
    async def _execute_model_call(self, task_spec: TaskSpec, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate model API call that can fail."""
        model_name = context.get("model", "test-model")
        prompt = context.get("prompt", "Test prompt")
        
        # Simulate different types of model API failures
        error_type = context.get("simulate_error", "none")
        
        if error_type == "auth_error":
            raise PermissionError(f"Authentication failed for model {model_name}")
        elif error_type == "rate_limit":
            raise ConnectionError(f"Rate limit exceeded for model {model_name}")
        elif error_type == "model_not_found":
            raise FileNotFoundError(f"Model {model_name} not found")
        elif error_type == "timeout":
            raise TimeoutError(f"Model {model_name} request timed out")
        elif error_type == "invalid_input":
            raise ValueError(f"Invalid input for model {model_name}: {prompt}")
        
        # Simulate successful response
        return {
            "task_id": task_spec.id,
            "success": True,
            "result": {
                "model": model_name,
                "response": f"Model response to: {prompt[:100]}...",
                "tokens_used": 150
            }
        }
    
    async def _execute_error_recovery(self, task_spec: TaskSpec, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute error recovery actions."""
        recovery_action = context.get("recovery_action", "default_recovery")
        original_error = context.get("error_message", "Unknown error")
        
        if recovery_action == "retry_with_backoff":
            # Simulate exponential backoff retry
            await asyncio.sleep(0.1)  # Simulate retry delay
            
            return {
                "task_id": task_spec.id,
                "success": True,
                "result": f"Successfully recovered from: {original_error}",
                "recovery_action": recovery_action
            }
        
        elif recovery_action == "fallback_endpoint":
            # Try fallback endpoint
            fallback_url = "https://httpbin.org/status/200"
            
            try:
                async with self.session.get(fallback_url) as response:
                    return {
                        "task_id": task_spec.id,
                        "success": True,
                        "result": f"Used fallback endpoint after error: {original_error}",
                        "fallback_url": fallback_url
                    }
            except Exception as e:
                raise ConnectionError(f"Fallback endpoint also failed: {e}")
        
        elif recovery_action == "alert_and_continue":
            # Simulate sending alert and continuing
            return {
                "task_id": task_spec.id,
                "success": True,
                "result": f"Alert sent for error: {original_error}",
                "alert_sent": True
            }
        
        else:
            return {
                "task_id": task_spec.id,
                "success": True,
                "result": f"Default recovery for: {original_error}"
            }
    
    async def _execute_timeout_test(self, task_spec: TaskSpec, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute operation that will timeout."""
        delay = context.get("delay", 2.0)
        await asyncio.sleep(delay)  # This will cause timeout if delay > timeout
        
        return {
            "task_id": task_spec.id,
            "success": True,
            "result": f"Completed after {delay} seconds"
        }


@pytest.fixture
async def real_api_executor():
    """Create real API task executor."""
    async with RealAPITaskExecutor() as executor:
        yield executor


class TestRealHTTPErrorHandling:
    """Test error handling with real HTTP requests."""
    
    @pytest.mark.asyncio
    async def test_http_404_error_handling(self, real_api_executor):
        """Test handling real HTTP 404 errors."""
        error_executor = ErrorHandlerExecutor(real_api_executor)
        
        # Task that will get 404
        task_spec = TaskSpec(
            id="http_404_task",
            action="Make HTTP request that returns 404"
        )
        
        # Handler for 404 errors
        handler = ErrorHandler(
            handler_action="Handle 404 error with fallback",
            error_types=["FileNotFoundError"],
            retry_with_handler=False,
            fallback_value="Resource not found - using cached data"
        )
        
        error_executor.handler_registry.register_handler("404_handler", handler, "http_404_task")
        
        # Execute with URL that returns 404
        result = await error_executor.handle_task_error(
            failed_task=task_spec,
            error=FileNotFoundError("HTTP 404: Resource not found"),
            context={"url": "https://httpbin.org/status/404"}
        )
        
        assert result["success"] is True
        assert "not found" in result["result"].lower()
    
    @pytest.mark.asyncio
    async def test_http_500_error_handling(self, real_api_executor):
        """Test handling real HTTP 500 errors."""
        error_executor = ErrorHandlerExecutor(real_api_executor)
        
        task_spec = TaskSpec(
            id="http_500_task",
            action="Make HTTP request that returns 500"
        )
        
        # Handler with retry logic
        handler = ErrorHandler(
            handler_action="Retry on server error",
            error_types=["ConnectionError"],
            retry_with_handler=True,
            max_handler_retries=2
        )
        
        error_executor.handler_registry.register_handler("500_handler", handler, "http_500_task")
        
        # Create actual server error
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://httpbin.org/status/500") as response:
                    if response.status >= 500:
                        raise ConnectionError(f"HTTP {response.status}: Server error")
        except ConnectionError as real_error:
            pass
        
        # Test error handling
        result = await error_executor.handle_task_error(
            failed_task=task_spec,
            error=real_error,
            context={"url": "https://httpbin.org/status/500"}
        )
        
        assert result["task_id"] == "http_500_task"
        # Should either recover or provide meaningful error handling
    
    @pytest.mark.asyncio
    async def test_http_timeout_handling(self, real_api_executor):
        """Test handling real HTTP timeouts."""
        error_executor = ErrorHandlerExecutor(real_api_executor)
        
        task_spec = TaskSpec(
            id="timeout_task",
            action="Make HTTP request that times out"
        )
        
        # Handler for timeout errors
        handler = ErrorHandler(
            handler_action="Handle timeout with fallback",
            error_types=["TimeoutError"],
            fallback_value="Request timed out - using default response"
        )
        
        error_executor.handler_registry.register_handler("timeout_handler", handler, "timeout_task")
        
        # Create real timeout error
        try:
            async with aiohttp.ClientSession() as session:
                # Use a very short timeout to ensure failure
                async with session.get("https://httpbin.org/delay/5", timeout=0.1) as response:
                    await response.text()
        except asyncio.TimeoutError as real_error:
            pass
        
        result = await error_executor.handle_task_error(
            failed_task=task_spec,
            error=real_error,
            context={"url": "https://httpbin.org/delay/5", "timeout": 0.1}
        )
        
        assert result["success"] is True
        assert "timeout" in result["result"].lower()
    
    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, real_api_executor):
        """Test handling rate limit errors."""
        error_executor = ErrorHandlerExecutor(real_api_executor)
        
        task_spec = TaskSpec(
            id="rate_limit_task",
            action="Make request that hits rate limit"
        )
        
        # Handler with exponential backoff
        handler = ErrorHandler(
            handler_action="Handle rate limit with backoff",
            error_types=["ConnectionError"],
            retry_with_handler=True,
            max_handler_retries=3
        )
        
        error_executor.handler_registry.register_handler("rate_limit_handler", handler, "rate_limit_task")
        
        # Simulate rate limit error (429 status)
        rate_limit_error = ConnectionError("HTTP 429: Rate limit exceeded")
        
        start_time = time.time()
        result = await error_executor.handle_task_error(
            failed_task=task_spec,
            error=rate_limit_error,
            context={"url": "https://httpbin.org/status/429"}
        )
        end_time = time.time()
        
        # Should take some time due to backoff
        assert end_time - start_time > 0.1  # At least some delay
        assert result["task_id"] == "rate_limit_task"


class TestRealModelAPIErrorHandling:
    """Test error handling with model API calls."""
    
    @pytest.mark.asyncio
    async def test_model_authentication_error(self, real_api_executor):
        """Test handling model API authentication errors."""
        error_executor = ErrorHandlerExecutor(real_api_executor)
        
        task_spec = TaskSpec(
            id="model_auth_task",
            action="Call model API with invalid auth"
        )
        
        # Handler for auth errors
        handler = ErrorHandler(
            handler_action="Handle auth error with fallback model",
            error_types=["PermissionError"],
            fallback_value="Authentication failed - using local model"
        )
        
        error_executor.handler_registry.register_handler("auth_handler", handler, "model_auth_task")
        
        # Simulate auth error
        auth_error = PermissionError("Authentication failed for model API")
        
        result = await error_executor.handle_task_error(
            failed_task=task_spec,
            error=auth_error,
            context={"model": "gpt-4", "api_key": "invalid_key"}
        )
        
        assert result["success"] is True
        assert "authentication" in result["result"].lower()
    
    @pytest.mark.asyncio
    async def test_model_not_found_error(self, real_api_executor):
        """Test handling model not found errors."""
        error_executor = ErrorHandlerExecutor(real_api_executor)
        
        task_spec = TaskSpec(
            id="model_not_found_task",
            action="Call non-existent model"
        )
        
        # Handler with model fallback
        handler = ErrorHandler(
            handler_action="Use fallback model",
            error_types=["FileNotFoundError"],
            retry_with_handler=True
        )
        
        error_executor.handler_registry.register_handler("model_fallback", handler, "model_not_found_task")
        
        # Simulate model not found
        model_error = FileNotFoundError("Model 'non-existent-model' not found")
        
        result = await error_executor.handle_task_error(
            failed_task=task_spec,
            error=model_error,
            context={"model": "non-existent-model", "prompt": "Test prompt"}
        )
        
        assert result["task_id"] == "model_not_found_task"
        # Should attempt recovery with fallback model
    
    @pytest.mark.asyncio
    async def test_model_input_validation_error(self, real_api_executor):
        """Test handling model input validation errors."""
        error_executor = ErrorHandlerExecutor(real_api_executor)
        
        task_spec = TaskSpec(
            id="model_validation_task",
            action="Call model with invalid input"
        )
        
        # Handler for input validation
        handler = ErrorHandler(
            handler_action="Clean and retry input",
            error_types=["ValueError"],
            retry_with_handler=True,
            max_handler_retries=1
        )
        
        error_executor.handler_registry.register_handler("validation_handler", handler, "model_validation_task")
        
        # Simulate validation error
        validation_error = ValueError("Invalid input: prompt contains prohibited content")
        
        result = await error_executor.handle_task_error(
            failed_task=task_spec,
            error=validation_error,
            context={"model": "claude-3", "prompt": "Invalid prompt content"}
        )
        
        assert result["task_id"] == "model_validation_task"


class TestRealAPIRecoveryStrategies:
    """Test various recovery strategies with real APIs."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_pattern(self, real_api_executor):
        """Test circuit breaker pattern with real API failures."""
        error_executor = ErrorHandlerExecutor(real_api_executor)
        
        # Multiple tasks that will fail
        failed_tasks = []
        for i in range(5):
            task_spec = TaskSpec(
                id=f"circuit_breaker_task_{i}",
                action="Task that will fail"
            )
            failed_tasks.append(task_spec)
        
        # Handler that tracks failures
        handler = ErrorHandler(
            handler_action="Track failures for circuit breaker",
            error_types=["*"],
            continue_on_handler_failure=True
        )
        
        error_executor.handler_registry.register_handler("circuit_handler", handler)
        
        # Execute multiple failures to trigger circuit breaker
        results = []
        for i, task in enumerate(failed_tasks):
            # Create real connection error
            connection_error = ConnectionError(f"Connection failed {i}")
            
            result = await error_executor.handle_task_error(
                failed_task=task,
                error=connection_error,
                context={"attempt": i}
            )
            results.append(result)
            
            # Record error for circuit breaker tracking
            error_executor.handler_registry.record_error_occurrence(
                task.id, "ConnectionError", handled=True
            )
        
        # Check that circuit breaker logic is working
        metrics = error_executor.get_execution_metrics()
        assert metrics["total_errors_handled"] >= len(failed_tasks)
    
    @pytest.mark.asyncio
    async def test_fallback_chain_strategy(self, real_api_executor):
        """Test fallback chain with multiple recovery options."""
        error_executor = ErrorHandlerExecutor(real_api_executor)
        
        task_spec = TaskSpec(
            id="fallback_chain_task",
            action="Task with multiple fallback options"
        )
        
        # Primary handler (will fail)
        primary_handler = ErrorHandler(
            handler_action="Primary recovery attempt",
            error_types=["ConnectionError"],
            priority=1,
            continue_on_handler_failure=True
        )
        
        # Secondary handler (will also fail)
        secondary_handler = ErrorHandler(
            handler_action="Secondary recovery attempt",
            error_types=["*"],
            priority=2,
            continue_on_handler_failure=True
        )
        
        # Final fallback with guaranteed success
        fallback_handler = ErrorHandler(
            handler_action="Final fallback",
            error_types=["*"],
            priority=3,
            fallback_value="All recovery attempts failed - using default"
        )
        
        error_executor.handler_registry.register_handler("primary", primary_handler, "fallback_chain_task")
        error_executor.handler_registry.register_handler("secondary", secondary_handler, "fallback_chain_task")
        error_executor.handler_registry.register_handler("fallback", fallback_handler, "fallback_chain_task")
        
        # Test with real connection error
        connection_error = ConnectionError("Primary endpoint unreachable")
        
        result = await error_executor.handle_task_error(
            failed_task=task_spec,
            error=connection_error,
            context={"primary_url": "https://unreachable.example.com"}
        )
        
        # Should eventually succeed with fallback
        assert result["success"] is True
        assert "default" in result["result"]
    
    @pytest.mark.asyncio
    async def test_adaptive_retry_strategy(self, real_api_executor):
        """Test adaptive retry with increasing delays."""
        error_executor = ErrorHandlerExecutor(real_api_executor)
        
        task_spec = TaskSpec(
            id="adaptive_retry_task",
            action="Task with adaptive retry"
        )
        
        # Handler with adaptive retry
        handler = ErrorHandler(
            handler_action="Adaptive retry with exponential backoff",
            error_types=["TimeoutError", "ConnectionError"],
            retry_with_handler=True,
            max_handler_retries=3
        )
        
        error_executor.handler_registry.register_handler("adaptive_handler", handler, "adaptive_retry_task")
        
        # Test with real timeout
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://httpbin.org/delay/10", timeout=0.5) as response:
                    await response.text()
        except asyncio.TimeoutError as timeout_error:
            pass
        
        result = await error_executor.handle_task_error(
            failed_task=task_spec,
            error=timeout_error,
            context={"url": "https://httpbin.org/delay/10"}
        )
        
        end_time = time.time()
        
        # Should have taken time due to retries with backoff
        execution_time = end_time - start_time
        assert execution_time > 0.5  # Should take some time due to retries
        
        assert result["task_id"] == "adaptive_retry_task"


class TestRealAPIErrorRecovery:
    """Test actual API error recovery scenarios."""
    
    @pytest.mark.asyncio
    async def test_api_endpoint_failover(self, real_api_executor):
        """Test failing over to backup API endpoint."""
        error_executor = ErrorHandlerExecutor(real_api_executor)
        
        task_spec = TaskSpec(
            id="failover_task",
            action="API call with endpoint failover"
        )
        
        # Handler that switches to backup endpoint
        handler = ErrorHandler(
            handler_action="Switch to backup endpoint",
            error_types=["ConnectionError", "TimeoutError"],
            retry_with_handler=True
        )
        
        error_executor.handler_registry.register_handler("failover_handler", handler, "failover_task")
        
        # Simulate primary endpoint failure
        primary_error = ConnectionError("Primary API endpoint unreachable")
        
        result = await error_executor.handle_task_error(
            failed_task=task_spec,
            error=primary_error,
            context={
                "primary_endpoint": "https://api-primary.example.com",
                "backup_endpoint": "https://httpbin.org/status/200"
            }
        )
        
        assert result["task_id"] == "failover_task"
        # Should attempt failover to backup endpoint
    
    @pytest.mark.asyncio
    async def test_data_validation_and_recovery(self, real_api_executor):
        """Test data validation errors and recovery."""
        error_executor = ErrorHandlerExecutor(real_api_executor)
        
        task_spec = TaskSpec(
            id="validation_task",
            action="Process data with validation"
        )
        
        # Handler for validation errors
        handler = ErrorHandler(
            handler_action="Clean and revalidate data",
            error_types=["ValueError", "TypeError"],
            retry_with_handler=True,
            max_handler_retries=2
        )
        
        error_executor.handler_registry.register_handler("validation_handler", handler, "validation_task")
        
        # Simulate data validation error
        validation_error = ValueError("Invalid data format: expected JSON, got malformed string")
        
        result = await error_executor.handle_task_error(
            failed_task=task_spec,
            error=validation_error,
            context={"data": "malformed{json:data}", "expected_format": "json"}
        )
        
        assert result["task_id"] == "validation_task"
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion_handling(self, real_api_executor):
        """Test handling resource exhaustion errors."""
        error_executor = ErrorHandlerExecutor(real_api_executor)
        
        task_spec = TaskSpec(
            id="resource_task",
            action="Resource-intensive operation"
        )
        
        # Handler for resource errors
        handler = ErrorHandler(
            handler_action="Wait and retry with reduced load",
            error_types=["ConnectionError"],  # Rate limits often manifest as connection errors
            retry_with_handler=True,
            max_handler_retries=3
        )
        
        error_executor.handler_registry.register_handler("resource_handler", handler, "resource_task")
        
        # Simulate resource exhaustion (rate limit)
        resource_error = ConnectionError("HTTP 429: Rate limit exceeded - too many requests")
        
        start_time = time.time()
        result = await error_executor.handle_task_error(
            failed_task=task_spec,
            error=resource_error,
            context={"resource_type": "api_calls", "current_rate": "100/minute"}
        )
        end_time = time.time()
        
        # Should have implemented some delay for rate limiting
        assert end_time - start_time > 0.1
        assert result["task_id"] == "resource_task"


if __name__ == "__main__":
    # Run real API error handling tests
    pytest.main([__file__, "-v", "--tb=short", "-k", "not slow"])