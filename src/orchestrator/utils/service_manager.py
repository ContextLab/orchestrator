"""Service management utilities for automatic startup of required services."""

import subprocess
import time
import os
import logging
import asyncio
from typing import Dict, Optional, Callable, Any, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ServiceManager(ABC):
    """Abstract base class for service managers."""
    
    @abstractmethod
    def is_installed(self) -> bool:
        """Check if the service is installed."""
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """Check if the service is running."""
        pass
    
    @abstractmethod
    def start(self) -> bool:
        """Start the service."""
        pass
    
    @abstractmethod
    def stop(self) -> bool:
        """Stop the service."""
        pass
    
    def ensure_running(self) -> bool:
        """Ensure the service is running, starting it if necessary."""
        if not self.is_installed():
            logger.warning(f"{self.__class__.__name__}: Service not installed")
            return False
            
        if self.is_running():
            logger.info(f"{self.__class__.__name__}: Service already running")
            return True
            
        logger.info(f"{self.__class__.__name__}: Starting service...")
        return self.start()


class OllamaServiceManager(ServiceManager):
    """Manage Ollama service with enhanced model download capabilities."""
    
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 30):
        """Initialize Ollama service manager.
        
        Args:
            base_url: Ollama server URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._model_cache: Optional[List[str]] = None
        self._cache_ttl = 300  # 5 minutes
        self._last_cache_update = 0
    
    def is_installed(self) -> bool:
        """Check if Ollama is installed."""
        try:
            result = subprocess.run(
                ["which", "ollama"], 
                capture_output=True, 
                text=True, 
                timeout=1
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def is_running(self) -> bool:
        """Check if Ollama server is running."""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=1)
            return response.status_code == 200
        except Exception:
            return False
    
    def start(self) -> bool:
        """Start Ollama server."""
        try:
            # Start Ollama server in the background
            process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True  # Detach from parent process
            )
            
            # Wait for server to start
            for i in range(10):  # Try for up to 10 seconds
                time.sleep(1)
                if self.is_running():
                    logger.info("Ollama server started successfully")
                    return True
                    
            logger.error("Ollama server failed to start within timeout")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start Ollama server: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop Ollama server."""
        try:
            # Try graceful shutdown first
            subprocess.run(["pkill", "-TERM", "ollama"], check=False)
            time.sleep(2)
            
            # Force kill if still running
            if self.is_running():
                subprocess.run(["pkill", "-KILL", "ollama"], check=False)
                
            return not self.is_running()
            
        except Exception as e:
            logger.error(f"Failed to stop Ollama server: {e}")
            return False
    
    def get_available_models(self, force_refresh: bool = False) -> List[str]:
        """Get list of available Ollama models with caching.
        
        Args:
            force_refresh: Force refresh the model cache
            
        Returns:
            List of available model names
        """
        current_time = time.time()
        
        # Use cache if it's fresh and not forcing refresh
        if (not force_refresh and 
            self._model_cache is not None and 
            (current_time - self._last_cache_update) < self._cache_ttl):
            return self._model_cache
            
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            if response.status_code == 200:
                models = response.json().get("models", [])
                self._model_cache = [model["name"] for model in models]
                self._last_cache_update = current_time
                logger.info(f"Found {len(self._model_cache)} Ollama models")
                return self._model_cache
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            
        # Return empty list if service not available
        if self._model_cache is None:
            self._model_cache = []
        return self._model_cache
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available locally.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model is available locally
        """
        available_models = self.get_available_models()
        return model_name in available_models
    
    def ensure_model_available(self, model_name: str, auto_pull: bool = True) -> bool:
        """Ensure a model is available, downloading if necessary.
        
        Args:
            model_name: Name of the model to ensure is available
            auto_pull: Whether to automatically pull the model if not available
            
        Returns:
            True if model is available after operation
        """
        if not self.ensure_running():
            logger.error("Cannot ensure model availability: Ollama service not running")
            return False
            
        # Check if model is already available
        if self.is_model_available(model_name):
            logger.info(f"Model {model_name} already available")
            return True
            
        if not auto_pull:
            logger.warning(f"Model {model_name} not available and auto_pull disabled")
            return False
            
        # Pull the model
        return self.pull_model(model_name)
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model using Ollama CLI.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if model was pulled successfully
        """
        try:
            logger.info(f"Pulling Ollama model: {model_name}")
            result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes timeout for model pull
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully pulled model: {model_name}")
                # Invalidate cache to force refresh
                self._model_cache = None
                return True
            else:
                logger.error(f"Failed to pull model {model_name}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout pulling model {model_name}")
            return False
        except FileNotFoundError:
            logger.error("Ollama CLI not found - cannot pull models")
            return False
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    def remove_model(self, model_name: str) -> bool:
        """Remove a model using Ollama CLI.
        
        Args:
            model_name: Name of the model to remove
            
        Returns:
            True if model was removed successfully
        """
        try:
            logger.info(f"Removing Ollama model: {model_name}")
            result = subprocess.run(
                ["ollama", "rm", model_name],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully removed model: {model_name}")
                # Invalidate cache to force refresh
                self._model_cache = None
                return True
            else:
                logger.error(f"Failed to remove model {model_name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error removing model {model_name}: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information or None if not found
        """
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            if response.status_code == 200:
                models = response.json().get("models", [])
                for model in models:
                    if model["name"] == model_name:
                        return model
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            
        return None
    
    async def health_check_model(self, model_name: str) -> bool:
        """Perform health check on a specific model.
        
        Args:
            model_name: Name of the model to health check
            
        Returns:
            True if model is healthy and responsive
        """
        if not self.is_running():
            return False
            
        if not self.is_model_available(model_name):
            return False
            
        try:
            import requests
            # Test generate with minimal request
            payload = {
                "model": model_name,
                "prompt": "Test",
                "stream": False,
                "options": {
                    "num_predict": 1,
                    "temperature": 0.0,
                }
            }
            
            response = await asyncio.to_thread(
                requests.post,
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Health check failed for model {model_name}: {e}")
            return False


class DockerServiceManager(ServiceManager):
    """Manage Docker service with containerized model support."""
    
    def __init__(self, timeout: int = 30):
        """Initialize Docker service manager.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self._container_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_cache_update = 0
    
    def is_installed(self) -> bool:
        """Check if Docker is installed."""
        try:
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                timeout=1
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def is_running(self) -> bool:
        """Check if Docker daemon is running."""
        try:
            result = subprocess.run(
                ["docker", "info"], 
                capture_output=True, 
                timeout=2
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def start(self) -> bool:
        """Start Docker daemon."""
        try:
            # Try different methods based on OS
            if os.path.exists("/usr/bin/systemctl"):
                # Linux with systemd
                result = subprocess.run(
                    ["sudo", "systemctl", "start", "docker"],
                    capture_output=True
                )
                if result.returncode == 0:
                    time.sleep(2)
                    return self.is_running()
                    
            elif os.path.exists("/Applications/Docker.app"):
                # macOS
                subprocess.run([
                    "open", "-a", "Docker"
                ], check=False)
                
                # Wait for Docker to start
                for i in range(30):  # Up to 30 seconds
                    time.sleep(1)
                    if self.is_running():
                        return True
                        
            return False
            
        except Exception as e:
            logger.error(f"Failed to start Docker: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop Docker daemon."""
        try:
            if os.path.exists("/usr/bin/systemctl"):
                # Linux with systemd
                subprocess.run(
                    ["sudo", "systemctl", "stop", "docker"],
                    check=False
                )
            elif os.path.exists("/Applications/Docker.app"):
                # macOS
                subprocess.run([
                    "osascript", "-e", 'quit app "Docker"'
                ], check=False)
                
            time.sleep(2)
            return not self.is_running()
            
        except Exception as e:
            logger.error(f"Failed to stop Docker: {e}")
            return False
    
    def get_running_containers(self, force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """Get list of running containers with caching.
        
        Args:
            force_refresh: Force refresh the container cache
            
        Returns:
            Dictionary of container info keyed by container name
        """
        current_time = time.time()
        
        # Use cache if it's fresh and not forcing refresh
        if (not force_refresh and 
            self._container_cache and 
            (current_time - self._last_cache_update) < self._cache_ttl):
            return self._container_cache
            
        containers = {}
        try:
            result = subprocess.run(
                ["docker", "ps", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                # Parse each line as JSON
                for line in result.stdout.strip().split('\n'):
                    if line:
                        import json
                        container = json.loads(line)
                        name = container.get("Names", "")
                        if name:
                            containers[name] = container
                            
                self._container_cache = containers
                self._last_cache_update = current_time
                logger.info(f"Found {len(containers)} running Docker containers")
                
        except Exception as e:
            logger.error(f"Failed to get running containers: {e}")
            
        return containers
    
    def is_container_running(self, container_name: str) -> bool:
        """Check if a specific container is running.
        
        Args:
            container_name: Name of the container to check
            
        Returns:
            True if container is running
        """
        running_containers = self.get_running_containers()
        return container_name in running_containers
    
    def ensure_container_running(self, container_config: Dict[str, Any]) -> bool:
        """Ensure a container is running with the given configuration.
        
        Args:
            container_config: Container configuration with keys:
                - name: Container name
                - image: Docker image to use
                - ports: Port mappings (optional)
                - environment: Environment variables (optional)
                - volumes: Volume mounts (optional)
                
        Returns:
            True if container is running after operation
        """
        if not self.ensure_running():
            logger.error("Cannot ensure container: Docker service not running")
            return False
            
        container_name = container_config["name"]
        
        # Check if container is already running
        if self.is_container_running(container_name):
            logger.info(f"Container {container_name} already running")
            return True
            
        # Try to start existing container first
        if self.start_container(container_name):
            return True
            
        # If that fails, create and run a new container
        return self.run_container(container_config)
    
    def start_container(self, container_name: str) -> bool:
        """Start an existing container.
        
        Args:
            container_name: Name of the container to start
            
        Returns:
            True if container was started successfully
        """
        try:
            logger.info(f"Starting Docker container: {container_name}")
            result = subprocess.run(
                ["docker", "start", container_name],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully started container: {container_name}")
                # Invalidate cache to force refresh
                self._container_cache = {}
                return True
            else:
                logger.error(f"Failed to start container {container_name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting container {container_name}: {e}")
            return False
    
    def run_container(self, container_config: Dict[str, Any]) -> bool:
        """Run a new container with the given configuration.
        
        Args:
            container_config: Container configuration
            
        Returns:
            True if container was created and started successfully
        """
        try:
            container_name = container_config["name"]
            image = container_config["image"]
            
            # Build docker run command
            cmd = ["docker", "run", "-d", "--name", container_name]
            
            # Add port mappings
            if "ports" in container_config:
                for port_mapping in container_config["ports"]:
                    cmd.extend(["-p", port_mapping])
                    
            # Add environment variables
            if "environment" in container_config:
                for env_var in container_config["environment"]:
                    cmd.extend(["-e", env_var])
                    
            # Add volume mounts
            if "volumes" in container_config:
                for volume in container_config["volumes"]:
                    cmd.extend(["-v", volume])
                    
            # Add the image
            cmd.append(image)
            
            # Add any command arguments
            if "args" in container_config:
                cmd.extend(container_config["args"])
                
            logger.info(f"Creating Docker container: {container_name} from {image}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout * 3  # Allow more time for image pull
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully created and started container: {container_name}")
                # Invalidate cache to force refresh
                self._container_cache = {}
                return True
            else:
                logger.error(f"Failed to run container {container_name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error running container {container_name}: {e}")
            return False
    
    def stop_container(self, container_name: str) -> bool:
        """Stop a running container.
        
        Args:
            container_name: Name of the container to stop
            
        Returns:
            True if container was stopped successfully
        """
        try:
            logger.info(f"Stopping Docker container: {container_name}")
            result = subprocess.run(
                ["docker", "stop", container_name],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully stopped container: {container_name}")
                # Invalidate cache to force refresh
                self._container_cache = {}
                return True
            else:
                logger.error(f"Failed to stop container {container_name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error stopping container {container_name}: {e}")
            return False
    
    def remove_container(self, container_name: str, force: bool = False) -> bool:
        """Remove a container.
        
        Args:
            container_name: Name of the container to remove
            force: Whether to force removal (stops running container first)
            
        Returns:
            True if container was removed successfully
        """
        try:
            cmd = ["docker", "rm"]
            if force:
                cmd.append("-f")
            cmd.append(container_name)
            
            logger.info(f"Removing Docker container: {container_name}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully removed container: {container_name}")
                # Invalidate cache to force refresh
                self._container_cache = {}
                return True
            else:
                logger.error(f"Failed to remove container {container_name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error removing container {container_name}: {e}")
            return False
    
    def get_container_logs(self, container_name: str, tail_lines: int = 50) -> str:
        """Get logs from a container.
        
        Args:
            container_name: Name of the container
            tail_lines: Number of recent lines to return
            
        Returns:
            Container logs as string
        """
        try:
            result = subprocess.run(
                ["docker", "logs", "--tail", str(tail_lines), container_name],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                logger.error(f"Failed to get logs for container {container_name}: {result.stderr}")
                return ""
                
        except Exception as e:
            logger.error(f"Error getting logs for container {container_name}: {e}")
            return ""
    
    async def health_check_container(self, container_name: str, health_endpoint: Optional[str] = None) -> bool:
        """Perform health check on a container.
        
        Args:
            container_name: Name of the container
            health_endpoint: Optional HTTP endpoint to check (e.g., "http://localhost:8080/health")
            
        Returns:
            True if container is healthy
        """
        # Check if container is running
        if not self.is_container_running(container_name):
            return False
            
        # If health endpoint provided, check it
        if health_endpoint:
            try:
                import requests
                response = await asyncio.to_thread(
                    requests.get, 
                    health_endpoint, 
                    timeout=self.timeout
                )
                return response.status_code == 200
            except Exception as e:
                logger.error(f"Health check failed for container {container_name} at {health_endpoint}: {e}")
                return False
                
        # Default: just check if container is running
        return True


# Global service registry
SERVICE_MANAGERS: Dict[str, ServiceManager] = {
    "ollama": OllamaServiceManager(),
    "docker": DockerServiceManager(),
}


def ensure_service_running(service_name: str) -> bool:
    """
    Ensure a service is running.
    
    Args:
        service_name: Name of the service (e.g., "ollama", "docker")
        
    Returns:
        True if service is running (or was started successfully)
    """
    manager = SERVICE_MANAGERS.get(service_name)
    if not manager:
        logger.error(f"Unknown service: {service_name}")
        return False
        
    return manager.ensure_running()


def register_service_manager(name: str, manager: ServiceManager) -> None:
    """
    Register a custom service manager.
    
    Args:
        name: Service name
        manager: ServiceManager instance
    """
    SERVICE_MANAGERS[name] = manager


def get_service_status(service_name: str) -> Dict[str, bool]:
    """
    Get the status of a service.
    
    Args:
        service_name: Name of the service
        
    Returns:
        Dict with 'installed' and 'running' status
    """
    manager = SERVICE_MANAGERS.get(service_name)
    if not manager:
        return {"installed": False, "running": False}
        
    return {
        "installed": manager.is_installed(),
        "running": manager.is_running()
    }