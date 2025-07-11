"""Integration tests for Docker containerization and sandboxed execution.

These tests verify:
1. Docker daemon connectivity
2. Container creation, execution, and cleanup
3. Resource limits and security constraints
4. File system isolation and volume mounts
5. Network isolation
6. Error handling for container failures

Note: Requires Docker daemon running.
"""

import pytest
import asyncio
import os
import tempfile
import time
from typing import Dict, Any, Optional, List

# Check Docker availability
def check_docker_available():
    """Check if Docker is available for testing."""
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except (ImportError, Exception):
        return False

HAS_DOCKER = check_docker_available()


class DockerExecutor:
    """Docker-based sandboxed code executor for testing."""
    
    def __init__(self, docker_client=None):
        """Initialize Docker executor."""
        if not HAS_DOCKER:
            pytest.skip("Docker not available")
        
        import docker
        self.docker = docker_client or docker.from_env()
        self.containers = {}
        self.resource_limits = {
            "memory": "128m",
            "cpu_quota": 50000,  # 50% of one CPU
            "pids_limit": 100,
            "network_mode": "none"  # No network access
        }
    
    async def execute_code(self, code: str, language: str = "python", 
                          timeout: int = 30, **kwargs) -> Dict[str, Any]:
        """Execute code in a sandboxed Docker container."""
        image = self._get_image_for_language(language)
        
        # Create temporary file for code
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{language}', delete=False) as f:
            f.write(code)
            code_file = f.name
        
        try:
            # Create container
            container = self.docker.containers.create(
                image=image,
                command=self._get_command_for_language(language, os.path.basename(code_file)),
                volumes={
                    os.path.dirname(code_file): {
                        'bind': '/code',
                        'mode': 'ro'  # Read-only
                    }
                },
                working_dir='/code',
                mem_limit=self.resource_limits["memory"],
                cpu_quota=self.resource_limits["cpu_quota"],
                pids_limit=self.resource_limits["pids_limit"],
                network_mode=self.resource_limits["network_mode"],
                remove=True,  # Auto-remove after execution
                detach=True,
                stdout=True,
                stderr=True
            )
            
            # Start container and wait for completion
            start_time = time.time()
            container.start()
            
            # Wait for container to finish or timeout
            try:
                result = container.wait(timeout=timeout)
                execution_time = time.time() - start_time
                
                # Get output
                logs = container.logs(stdout=True, stderr=True).decode('utf-8')
                
                return {
                    "success": result["StatusCode"] == 0,
                    "exit_code": result["StatusCode"],
                    "output": logs,
                    "execution_time": execution_time,
                    "timeout": timeout
                }
                
            except Exception as e:
                # Handle timeout or other execution errors
                try:
                    container.kill()
                except:
                    pass
                
                return {
                    "success": False,
                    "error": str(e),
                    "timeout": timeout,
                    "execution_time": time.time() - start_time
                }
        
        finally:
            # Cleanup
            try:
                os.unlink(code_file)
            except:
                pass
    
    def _get_image_for_language(self, language: str) -> str:
        """Get Docker image for programming language."""
        images = {
            "python": "python:3.11-slim",
            "javascript": "node:18-slim",
            "bash": "bash:5.1"
        }
        return images.get(language, "ubuntu:22.04")
    
    def _get_command_for_language(self, language: str, filename: str) -> List[str]:
        """Get execution command for programming language."""
        commands = {
            "python": ["python", filename],
            "javascript": ["node", filename],
            "bash": ["bash", filename]
        }
        return commands.get(language, ["cat", filename])
    
    def create_test_container(self, image: str = "alpine:latest", **kwargs) -> Any:
        """Create a test container for integration testing."""
        return self.docker.containers.create(
            image=image,
            command=kwargs.get("command", ["sleep", "60"]),
            **kwargs
        )
    
    def cleanup_test_containers(self):
        """Clean up any test containers."""
        for container in self.docker.containers.list(all=True):
            if container.name and "test-" in container.name:
                try:
                    container.remove(force=True)
                except:
                    pass
    
    def get_docker_info(self) -> Dict[str, Any]:
        """Get Docker system information."""
        info = self.docker.info()
        return {
            "docker_version": info.get("ServerVersion"),
            "containers_running": info.get("ContainersRunning", 0),
            "containers_total": info.get("Containers", 0),
            "images": info.get("Images", 0),
            "memory_total": info.get("MemTotal", 0),
            "cpus": info.get("NCPU", 0)
        }


@pytest.mark.skipif(not HAS_DOCKER, reason="Docker not available")
class TestDockerIntegration:
    """Integration tests for Docker functionality."""
    
    @pytest.fixture
    def docker_executor(self):
        """Create Docker executor instance."""
        executor = DockerExecutor()
        yield executor
        executor.cleanup_test_containers()
    
    def test_docker_connection(self, docker_executor):
        """Test basic Docker daemon connection."""
        info = docker_executor.get_docker_info()
        
        assert "docker_version" in info
        assert isinstance(info["containers_total"], int)
        assert isinstance(info["images"], int)
    
    @pytest.mark.asyncio
    async def test_python_code_execution(self, docker_executor):
        """Test executing Python code in container."""
        code = """
print("Hello from Docker!")
result = 2 + 2
print(f"2 + 2 = {result}")
"""
        
        result = await docker_executor.execute_code(code, "python")
        
        assert result["success"] is True
        assert result["exit_code"] == 0
        assert "Hello from Docker!" in result["output"]
        assert "2 + 2 = 4" in result["output"]
        assert result["execution_time"] > 0
    
    @pytest.mark.asyncio
    async def test_javascript_code_execution(self, docker_executor):
        """Test executing JavaScript code in container."""
        code = """
console.log("Hello from Node.js!");
const result = 3 * 7;
console.log(`3 * 7 = ${result}`);
"""
        
        result = await docker_executor.execute_code(code, "javascript")
        
        assert result["success"] is True
        assert result["exit_code"] == 0
        assert "Hello from Node.js!" in result["output"]
        assert "3 * 7 = 21" in result["output"]
    
    @pytest.mark.asyncio
    async def test_code_execution_with_error(self, docker_executor):
        """Test executing code that produces errors."""
        code = """
print("Before error")
x = 1 / 0  # Division by zero
print("After error")
"""
        
        result = await docker_executor.execute_code(code, "python")
        
        assert result["success"] is False
        assert result["exit_code"] != 0
        assert "Before error" in result["output"]
        assert "ZeroDivisionError" in result["output"]
        assert "After error" not in result["output"]
    
    @pytest.mark.asyncio
    async def test_code_execution_timeout(self, docker_executor):
        """Test code execution timeout."""
        code = """
import time
time.sleep(10)  # Sleep longer than timeout
print("This should not appear")
"""
        
        result = await docker_executor.execute_code(code, "python", timeout=2)
        
        assert result["success"] is False
        assert "timeout" in result.get("error", "").lower() or result["execution_time"] >= 2
    
    @pytest.mark.asyncio
    async def test_memory_limit_enforcement(self, docker_executor):
        """Test that memory limits are enforced."""
        # Try to allocate more memory than allowed
        code = """
try:
    # Try to allocate 256MB (more than 128MB limit)
    large_list = [0] * (256 * 1024 * 1024)
    print("MEMORY LIMIT NOT ENFORCED!")
except MemoryError:
    print("Memory limit enforced correctly")
except Exception as e:
    print(f"Memory allocation failed: {e}")
"""
        
        result = await docker_executor.execute_code(code, "python")
        
        # Should either fail or report memory limit enforcement
        assert "MEMORY LIMIT NOT ENFORCED!" not in result["output"]
        # Check that some kind of memory constraint was hit
        assert (result["success"] is False or 
                "Memory limit enforced" in result["output"] or
                "Memory allocation failed" in result["output"])
    
    @pytest.mark.asyncio
    async def test_network_isolation(self, docker_executor):
        """Test that network access is blocked."""
        code = """
import urllib.request
try:
    response = urllib.request.urlopen("https://httpbin.org/ip", timeout=5)
    print("NETWORK ACCESS ALLOWED!")
except Exception as e:
    print(f"Network access blocked: {e}")
"""
        
        result = await docker_executor.execute_code(code, "python")
        
        # Network should be blocked
        assert "NETWORK ACCESS ALLOWED!" not in result["output"]
        assert "Network access blocked" in result["output"]
    
    @pytest.mark.asyncio
    async def test_file_system_isolation(self, docker_executor):
        """Test file system access restrictions."""
        code = """
import os
try:
    # Try to access host file system
    with open("/etc/passwd", "r") as f:
        content = f.read()
    print("HOST FILE ACCESS ALLOWED!")
except Exception as e:
    print(f"Host file access blocked: {e}")

# Test writing to container filesystem
try:
    with open("/tmp/test.txt", "w") as f:
        f.write("test")
    print("Container filesystem write successful")
except Exception as e:
    print(f"Container filesystem write failed: {e}")
"""
        
        result = await docker_executor.execute_code(code, "python")
        
        # Should not be able to access host files
        assert "HOST FILE ACCESS ALLOWED!" not in result["output"]
        # Should be able to write to container filesystem
        assert "Container filesystem write successful" in result["output"]
    
    def test_container_creation_and_cleanup(self, docker_executor):
        """Test container creation and cleanup."""
        # Create test container
        container = docker_executor.create_test_container(
            name="test-integration-container",
            command=["echo", "Hello World"]
        )
        
        assert container.name == "test-integration-container"
        
        # Start and wait for completion
        container.start()
        result = container.wait()
        
        assert result["StatusCode"] == 0
        
        # Get output
        logs = container.logs().decode('utf-8')
        assert "Hello World" in logs
        
        # Cleanup
        container.remove()
    
    def test_container_resource_limits(self, docker_executor):
        """Test that resource limits are properly set."""
        container = docker_executor.create_test_container(
            name="test-resource-limits",
            mem_limit="64m",
            cpu_quota=25000,  # 25% CPU
            pids_limit=50
        )
        
        container.start()
        
        # Get container info
        container.reload()
        host_config = container.attrs["HostConfig"]
        
        assert host_config["Memory"] == 64 * 1024 * 1024  # 64MB in bytes
        assert host_config["CpuQuota"] == 25000
        assert host_config["PidsLimit"] == 50
        
        container.stop()
        container.remove()
    
    @pytest.mark.asyncio
    async def test_concurrent_container_execution(self, docker_executor):
        """Test multiple containers running concurrently."""
        async def execute_task(task_id):
            code = f"""
import time
time.sleep(0.1)
print("Task {task_id} completed")
"""
            return await docker_executor.execute_code(code, "python")
        
        # Run multiple tasks concurrently
        tasks = [execute_task(i) for i in range(3)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(result["success"] for result in results)
        assert all(f"Task {i} completed" in results[i]["output"] for i in range(3))
    
    def test_docker_image_availability(self, docker_executor):
        """Test that required Docker images are available."""
        required_images = ["python:3.11-slim", "node:18-slim", "alpine:latest"]
        
        for image_name in required_images:
            try:
                # Try to pull image if not available
                image = docker_executor.docker.images.get(image_name)
                assert image is not None
            except Exception:
                # Try to pull the image
                try:
                    docker_executor.docker.images.pull(image_name)
                except Exception as e:
                    pytest.fail(f"Could not pull required image {image_name}: {e}")
    
    @pytest.mark.asyncio
    async def test_container_error_handling(self, docker_executor):
        """Test error handling when container operations fail."""
        # Test with non-existent image
        try:
            container = docker_executor.docker.containers.create(
                "nonexistent-image-12345",
                command=["echo", "hello"]
            )
            pytest.fail("Should have raised exception for non-existent image")
        except Exception as e:
            assert "image" in str(e).lower() or "not found" in str(e).lower()
    
    def test_docker_system_resources(self, docker_executor):
        """Test Docker system resource monitoring."""
        info = docker_executor.get_docker_info()
        
        # Basic sanity checks
        assert info["memory_total"] > 0
        assert info["cpus"] > 0
        assert info["containers_total"] >= 0
        assert info["images"] >= 0
    
    @pytest.mark.asyncio
    async def test_volume_mount_security(self, docker_executor):
        """Test that volume mounts work securely."""
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test data")
            test_file = f.name
        
        try:
            # Mount file as read-only
            container = docker_executor.docker.containers.create(
                "alpine:latest",
                command=["cat", "/mounted/file"],
                volumes={
                    test_file: {
                        'bind': '/mounted/file',
                        'mode': 'ro'
                    }
                },
                name="test-volume-mount"
            )
            
            container.start()
            result = container.wait()
            logs = container.logs().decode('utf-8')
            
            assert result["StatusCode"] == 0
            assert "test data" in logs
            
            container.remove()
            
        finally:
            os.unlink(test_file)
    
    @pytest.mark.asyncio
    async def test_security_no_privileged_access(self, docker_executor):
        """Test that containers don't have privileged access."""
        code = """
import os
import subprocess

# Try to access privileged operations
try:
    # Try to mount filesystem
    result = subprocess.run(['mount'], capture_output=True, text=True)
    if result.returncode == 0:
        print("PRIVILEGED ACCESS DETECTED!")
    else:
        print("Mount command failed as expected")
except Exception as e:
    print(f"Privileged operation blocked: {e}")

# Check if running as root (should be, but without privileges)
print(f"Running as UID: {os.getuid()}")
"""
        
        result = await docker_executor.execute_code(code, "python")
        
        # Should not have privileged access even if running as root
        assert "PRIVILEGED ACCESS DETECTED!" not in result["output"]
        assert "Mount command failed as expected" in result["output"] or "Privileged operation blocked" in result["output"]


if __name__ == "__main__":
    # Print Docker availability for debugging
    print("Docker integration test requirements:")
    print(f"Docker available: {'✓' if HAS_DOCKER else '✗'}")
    
    if not HAS_DOCKER:
        print("\nDocker not available. Try:")
        print("1. Install Docker Desktop or Docker Engine")
        print("2. Start Docker daemon")
        print("3. Ensure current user has Docker permissions")
        print("4. Test with: docker run hello-world")
    
    pytest.main([__file__, "-v"])