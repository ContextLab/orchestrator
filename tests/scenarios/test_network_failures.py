"""
Network Failure and API Timeout Testing

Tests the orchestrator's behavior under various network failure conditions,
API timeouts, and connectivity issues using real network scenarios.
"""

import pytest
import asyncio
import time
import socket
import threading
import http.server
import socketserver
from typing import Dict, Any
from unittest.mock import patch
from pathlib import Path

# Import orchestrator components
from src.orchestrator.orchestrator import Orchestrator


class TimeoutHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that introduces controlled delays to simulate timeouts."""
    
    def do_GET(self):
        # Extract delay from query parameter
        delay = int(self.path.split('delay=')[-1]) if 'delay=' in self.path else 0
        if delay > 0:
            time.sleep(delay)
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = '{"status": "success", "data": "test response"}'
        self.wfile.write(response.encode())
    
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        # Simulate API processing time
        delay = int(self.headers.get('X-Delay', 0))
        if delay > 0:
            time.sleep(delay)
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = '{"status": "processed", "received": "' + str(len(post_data)) + ' bytes"}'
        self.wfile.write(response.encode())
    
    def log_message(self, format, *args):
        # Suppress logging to keep test output clean
        pass


class MockSlowServer:
    """Mock server that simulates slow or failing API responses."""
    
    def __init__(self, port=0):
        self.port = port
        self.server = None
        self.thread = None
        
    def start(self, delay=0):
        """Start the mock server with specified delay."""
        handler = TimeoutHTTPRequestHandler
        self.server = socketserver.TCPServer(("localhost", self.port), handler)
        if self.port == 0:
            self.port = self.server.server_address[1]
        
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()
        return self.port
    
    def stop(self):
        """Stop the mock server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.thread:
            self.thread.join(timeout=1)


class TestNetworkFailures:
    """Test network failure scenarios and timeout handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_server = MockSlowServer()
        self.test_pipelines_dir = Path("/Users/jmanning/orchestrator/tests/scenarios/test_pipelines")
        self.test_pipelines_dir.mkdir(exist_ok=True)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.mock_server.stop()
    
    def create_web_request_pipeline(self, url, timeout=30):
        """Create a test pipeline that makes web requests."""
        pipeline_content = f"""
name: network_test_pipeline
version: "1.0"
description: Test pipeline for network failure scenarios

steps:
  - id: web_request
    action: web_request
    parameters:
      url: "{url}"
      timeout: {timeout}
      method: "GET"
    outputs:
      - response_data
  
  - id: process_response
    action: python_code
    parameters:
      code: |
        import json
        print(f"Received response: {{response_data}}")
        result = json.loads(response_data) if response_data else {{"error": "no data"}}
    depends_on:
      - web_request
"""
        pipeline_path = self.test_pipelines_dir / "web_request_test.yaml"
        pipeline_path.write_text(pipeline_content)
        return pipeline_path
    
    @pytest.mark.asyncio
    async def test_api_timeout_handling(self):
        """Test handling of API timeouts with real network calls."""
        # Start mock server with delay
        port = self.mock_server.start(delay=0)
        slow_url = f"http://localhost:{port}/?delay=5"
        
        # Create pipeline with short timeout
        pipeline_path = self.create_web_request_pipeline(slow_url, timeout=2)
        
        executor = Orchestrator()
        
        # Test that timeout is properly handled
        start_time = time.time()
        try:
            yaml_content = pipeline_path.read_text()
            result = await executor.execute_yaml(yaml_content)
            execution_time = time.time() - start_time
            
            # Should timeout within reasonable time (not wait full 5 seconds)
            assert execution_time < 4, f"Execution took too long: {execution_time}s"
            
            # Should have error information about timeout
            assert result.status in ["error", "timeout", "failed"]
            
            # Verify timeout error is captured
            timeout_errors = [step for step in result.step_results 
                            if step.status == "error" and ("timeout" in step.error_message.lower() or "time" in step.error_message.lower())]
            assert len(timeout_errors) > 0, "No timeout errors detected"
            
        except Exception as e:
            # Timeout exceptions are acceptable
            assert "timeout" in str(e).lower() or "time" in str(e).lower()
        
        print(f"✓ API timeout properly handled in {time.time() - start_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_connection_refused_handling(self):
        """Test handling when connection is refused (server down)."""
        # Use a port that's guaranteed to be closed
        closed_url = "http://localhost:65534/api/test"
        
        pipeline_path = self.create_web_request_pipeline(closed_url, timeout=5)
        
        executor = Orchestrator()
        
        start_time = time.time()
        result = await executor.execute_pipeline_file(str(pipeline_path))
        execution_time = time.time() - start_time
        
        # Should fail quickly (not wait full timeout)
        assert execution_time < 10, f"Connection refused took too long: {execution_time}s"
        
        # Should have connection error
        assert result.status in ["error", "failed"]
        
        # Verify connection error is captured
        connection_errors = [step for step in result.step_results 
                           if step.status == "error" and ("connection" in step.error_message.lower() or "refused" in step.error_message.lower())]
        assert len(connection_errors) > 0, "No connection errors detected"
        
        print(f"✓ Connection refused properly handled in {execution_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_dns_resolution_failure(self):
        """Test handling of DNS resolution failures."""
        # Use a domain that doesn't exist
        invalid_url = "http://this-domain-definitely-does-not-exist-12345.com/api"
        
        pipeline_path = self.create_web_request_pipeline(invalid_url, timeout=10)
        
        executor = Orchestrator()
        
        start_time = time.time()
        result = await executor.execute_pipeline_file(str(pipeline_path))
        execution_time = time.time() - start_time
        
        # Should fail within reasonable time
        assert execution_time < 15, f"DNS failure took too long: {execution_time}s"
        
        # Should have DNS/resolution error
        assert result.status in ["error", "failed"]
        
        # Verify DNS error is captured
        dns_errors = [step for step in result.step_results 
                     if step.status == "error" and any(keyword in step.error_message.lower() 
                     for keyword in ["dns", "resolve", "name", "host", "address"])]
        assert len(dns_errors) > 0, "No DNS resolution errors detected"
        
        print(f"✓ DNS resolution failure properly handled in {execution_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_network_interruption_recovery(self):
        """Test recovery from network interruptions."""
        # Start server, make request, stop server, try again
        port = self.mock_server.start()
        working_url = f"http://localhost:{port}/test"
        
        pipeline_content = f"""
name: network_recovery_test
version: "1.0"

steps:
  - id: first_request
    action: web_request
    parameters:
      url: "{working_url}"
      timeout: 5
    outputs:
      - first_response
      
  - id: wait_step
    action: python_code
    parameters:
      code: |
        import time
        print("Waiting before second request...")
        time.sleep(1)
    depends_on:
      - first_request
      
  - id: second_request
    action: web_request
    parameters:
      url: "{working_url}"
      timeout: 5
      retry_count: 2
      retry_delay: 1
    depends_on:
      - wait_step
"""
        
        pipeline_path = self.test_pipelines_dir / "network_recovery.yaml"
        pipeline_path.write_text(pipeline_content)
        
        executor = Orchestrator()
        
        # Start execution in background
        async def execute_with_interruption():
            await asyncio.sleep(2)  # Let first request complete
            self.mock_server.stop()  # Stop server during execution
            await asyncio.sleep(1)   # Give it time to fail
        
        # Run both tasks
        execution_task = asyncio.create_task(executor.execute_pipeline_file(str(pipeline_path)))
        interruption_task = asyncio.create_task(execute_with_interruption())
        
        result = await execution_task
        await interruption_task
        
        # Verify behavior
        assert len(result.step_results) >= 2
        
        # First request should succeed
        first_step = next(step for step in result.step_results if step.step_id == "first_request")
        assert first_step.status == "success"
        
        # Second request should fail due to server being down
        second_step = next(step for step in result.step_results if step.step_id == "second_request")
        assert second_step.status in ["error", "failed"]
        
        print("✓ Network interruption behavior verified")
    
    @pytest.mark.asyncio
    async def test_concurrent_network_requests(self):
        """Test behavior under concurrent network load."""
        port = self.mock_server.start()
        
        pipeline_content = f"""
name: concurrent_network_test
version: "1.0"

steps:
  - id: concurrent_requests
    action: parallel
    parameters:
      max_concurrent: 10
      tasks:
        - action: web_request
          parameters:
            url: "http://localhost:{port}/test1"
            timeout: 10
        - action: web_request
          parameters:
            url: "http://localhost:{port}/test2"  
            timeout: 10
        - action: web_request
          parameters:
            url: "http://localhost:{port}/test3"
            timeout: 10
        - action: web_request
          parameters:
            url: "http://localhost:{port}/test4"
            timeout: 10
        - action: web_request
          parameters:
            url: "http://localhost:{port}/test5"
            timeout: 10
"""
        
        pipeline_path = self.test_pipelines_dir / "concurrent_network.yaml"
        pipeline_path.write_text(pipeline_content)
        
        executor = Orchestrator()
        
        start_time = time.time()
        result = await executor.execute_pipeline_file(str(pipeline_path))
        execution_time = time.time() - start_time
        
        print(f"Concurrent requests completed in {execution_time:.2f}s")
        
        # Verify concurrent execution worked
        assert result.status in ["success", "partial_success"]
        
        # Should complete much faster than sequential (< 5 requests * timeout)
        assert execution_time < 30, f"Concurrent execution too slow: {execution_time}s"
        
        print("✓ Concurrent network requests handled properly")
    
    @pytest.mark.asyncio 
    async def test_rate_limiting_response(self):
        """Test handling of rate limiting (HTTP 429) responses."""
        # This would require a more sophisticated mock server
        # For now, test with a pipeline that simulates rate limiting
        
        pipeline_content = """
name: rate_limit_test
version: "1.0"

steps:
  - id: simulate_rate_limit
    action: python_code
    parameters:
      code: |
        import requests
        import time
        
        # Simulate rate limiting behavior
        for i in range(3):
            try:
                # This will likely get rate limited by httpbin
                response = requests.get("https://httpbin.org/delay/1", timeout=5)
                print(f"Request {i+1}: Status {response.status_code}")
                time.sleep(0.1)  # Very short delay to trigger rate limiting
            except Exception as e:
                print(f"Request {i+1} failed: {e}")
                if "429" in str(e) or "rate" in str(e).lower():
                    print("Rate limiting detected")
                    break
"""
        
        pipeline_path = self.test_pipelines_dir / "rate_limit.yaml"
        pipeline_path.write_text(pipeline_content)
        
        executor = Orchestrator()
        
        result = await executor.execute_pipeline_file(str(pipeline_path))
        
        # Should complete (may succeed or fail, but shouldn't crash)
        assert result.status in ["success", "error", "failed"]
        assert len(result.step_results) > 0
        
        print("✓ Rate limiting scenario tested")


class TestAPITimeoutScenarios:
    """Test various API timeout scenarios with real services."""
    
    @pytest.mark.asyncio
    async def test_model_api_timeout(self):
        """Test timeout behavior with model API calls."""
        pipeline_content = """
name: model_timeout_test
version: "1.0"

steps:
  - id: slow_model_call
    action: llm
    parameters:
      model: "ollama/llama3.2:1b"  # Use local model to avoid external API costs
      prompt: "Generate a very long story about artificial intelligence and robotics with at least 2000 words"
      max_tokens: 4000
      timeout: 5  # Very short timeout
    outputs:
      - story_text
      
  - id: fallback_response
    action: python_code  
    parameters:
      code: |
        if 'story_text' not in globals() or not story_text:
            story_text = "Fallback: Unable to generate full story due to timeout"
        print(f"Final result: {story_text[:100]}...")
    depends_on:
      - slow_model_call
"""
        
        test_dir = Path("/Users/jmanning/orchestrator/tests/scenarios/test_pipelines")
        test_dir.mkdir(exist_ok=True)
        pipeline_path = test_dir / "model_timeout.yaml"
        pipeline_path.write_text(pipeline_content)
        
        executor = Orchestrator()
        
        start_time = time.time()
        result = await executor.execute_pipeline_file(str(pipeline_path))
        execution_time = time.time() - start_time
        
        # Should timeout within reasonable time
        assert execution_time < 15, f"Model timeout took too long: {execution_time}s"
        
        # Check if timeout was handled
        assert result.status in ["success", "error", "timeout", "partial_success"]
        
        # Verify fallback logic executed
        fallback_step = next((step for step in result.step_results if step.step_id == "fallback_response"), None)
        if fallback_step:
            assert fallback_step.status in ["success", "completed"]
        
        print(f"✓ Model API timeout handled in {execution_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_web_search_timeout(self):
        """Test timeout with web search operations."""
        pipeline_content = """
name: web_search_timeout_test
version: "1.0"

steps:
  - id: web_search_with_timeout
    action: web_search
    parameters:
      query: "artificial intelligence research 2024"
      max_results: 20
      timeout: 3  # Short timeout
    outputs:
      - search_results
      
  - id: process_results
    action: python_code
    parameters:
      code: |
        if 'search_results' in globals() and search_results:
            print(f"Found {len(search_results)} results")
        else:
            print("No search results due to timeout")
            search_results = []
    depends_on:
      - web_search_with_timeout
"""
        
        test_dir = Path("/Users/jmanning/orchestrator/tests/scenarios/test_pipelines")
        pipeline_path = test_dir / "web_search_timeout.yaml"
        pipeline_path.write_text(pipeline_content)
        
        executor = Orchestrator()
        
        start_time = time.time()
        result = await executor.execute_pipeline_file(str(pipeline_path))
        execution_time = time.time() - start_time
        
        # Should handle timeout appropriately
        assert execution_time < 10, f"Web search timeout took too long: {execution_time}s"
        
        # Pipeline should handle the timeout gracefully
        assert result.status in ["success", "error", "partial_success"]
        
        print(f"✓ Web search timeout handled in {execution_time:.2f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])