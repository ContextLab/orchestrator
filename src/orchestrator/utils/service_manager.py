"""Service management utilities for automatic startup of required services."""

import subprocess
import time
import os
import logging
from typing import Dict, Optional, Callable, Any
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
    """Manage Ollama service."""
    
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


class DockerServiceManager(ServiceManager):
    """Manage Docker service."""
    
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