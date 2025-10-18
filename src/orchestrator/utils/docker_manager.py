"""Automatic Docker installation and management."""

import logging
import os
import platform
import subprocess
import time

logger = logging.getLogger(__name__)


class DockerManager:
    """Manages Docker installation and startup automatically."""

    @staticmethod
    def is_installed() -> bool:
        """Check if Docker is installed.

        Returns:
            True if Docker is installed
        """
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def is_running() -> bool:
        """Check if Docker daemon is running.

        Returns:
            True if Docker is running
        """
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def install() -> bool:
        """Install Docker automatically.

        Returns:
            True if installation successful or already installed
        """
        if DockerManager.is_installed():
            logger.info("Docker is already installed")
            return True

        system = platform.system()
        logger.info(f"Installing Docker on {system}...")

        try:
            if system == "Darwin":  # macOS
                return DockerManager._install_macos()
            elif system == "Linux":
                return DockerManager._install_linux()
            else:
                logger.error(f"Automatic Docker installation not supported on {system}")
                return False
        except Exception as e:
            logger.error(f"Failed to install Docker: {e}")
            return False

    @staticmethod
    def _install_macos() -> bool:
        """Install Docker on macOS using Homebrew.

        Returns:
            True if successful
        """
        logger.info("Installing Docker Desktop for Mac via Homebrew...")

        # Check if Homebrew is installed
        try:
            subprocess.run(["brew", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.error("Homebrew not installed. Please install from https://brew.sh")
            return False

        # Install Docker Desktop
        try:
            subprocess.run(
                ["brew", "install", "--cask", "docker"],
                check=True,
                timeout=600  # 10 minutes
            )
            logger.info("Docker Desktop installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install Docker Desktop: {e}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Docker installation timed out")
            return False

    @staticmethod
    def _install_linux() -> bool:
        """Install Docker on Linux.

        Returns:
            True if successful
        """
        logger.info("Installing Docker on Linux...")

        # Use official Docker installation script
        try:
            # Download and run the convenience script
            subprocess.run(
                ["curl", "-fsSL", "https://get.docker.com", "-o", "/tmp/get-docker.sh"],
                check=True,
                timeout=60
            )
            subprocess.run(
                ["sudo", "sh", "/tmp/get-docker.sh"],
                check=True,
                timeout=600
            )

            # Add current user to docker group
            username = os.getenv("USER")
            if username:
                subprocess.run(
                    ["sudo", "usermod", "-aG", "docker", username],
                    check=True
                )
                logger.info(f"Added {username} to docker group (logout/login required)")

            logger.info("Docker installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install Docker: {e}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Docker installation timed out")
            return False

    @staticmethod
    def start() -> bool:
        """Start Docker daemon.

        Returns:
            True if started successfully or already running
        """
        if DockerManager.is_running():
            logger.info("Docker is already running")
            return True

        system = platform.system()
        logger.info(f"Starting Docker on {system}...")

        try:
            if system == "Darwin":  # macOS
                return DockerManager._start_macos()
            elif system == "Linux":
                return DockerManager._start_linux()
            else:
                logger.error(f"Automatic Docker start not supported on {system}")
                return False
        except Exception as e:
            logger.error(f"Failed to start Docker: {e}")
            return False

    @staticmethod
    def _start_macos() -> bool:
        """Start Docker Desktop on macOS.

        Returns:
            True if successful
        """
        logger.info("Starting Docker Desktop...")

        try:
            # Open Docker Desktop app
            subprocess.run(
                ["open", "-a", "Docker"],
                check=True,
                timeout=10
            )

            # Wait for Docker to start (up to 60 seconds)
            for i in range(60):
                time.sleep(2)
                if DockerManager.is_running():
                    logger.info(f"Docker started successfully (after {i*2}s)")
                    return True

            logger.error("Docker did not start within 60 seconds")
            return False

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start Docker Desktop: {e}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Docker start command timed out")
            return False

    @staticmethod
    def _start_linux() -> bool:
        """Start Docker daemon on Linux.

        Returns:
            True if successful
        """
        logger.info("Starting Docker daemon...")

        try:
            # Start docker service
            subprocess.run(
                ["sudo", "systemctl", "start", "docker"],
                check=True,
                timeout=30
            )

            # Wait for Docker to be ready
            for i in range(30):
                time.sleep(1)
                if DockerManager.is_running():
                    logger.info(f"Docker started successfully (after {i}s)")
                    return True

            logger.error("Docker did not start within 30 seconds")
            return False

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start Docker: {e}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Docker start timed out")
            return False

    @staticmethod
    def ensure_docker_ready(install_if_missing: bool = True, start_if_stopped: bool = True) -> bool:
        """Ensure Docker is installed and running.

        This is the main function to call before any Docker operations.

        Args:
            install_if_missing: Automatically install if not present
            start_if_stopped: Automatically start if not running

        Returns:
            True if Docker is ready to use

        Raises:
            RuntimeError: If Docker cannot be made ready
        """
        # Check installation
        if not DockerManager.is_installed():
            if install_if_missing:
                logger.info("Docker not found, installing automatically...")
                if not DockerManager.install():
                    raise RuntimeError(
                        "Docker is not installed and automatic installation failed. "
                        "Please install Docker manually from https://www.docker.com/get-started"
                    )
            else:
                raise RuntimeError("Docker is not installed. Please install from https://www.docker.com/get-started")

        # Check if running
        if not DockerManager.is_running():
            if start_if_stopped:
                logger.info("Docker is not running, starting automatically...")
                if not DockerManager.start():
                    raise RuntimeError(
                        "Docker is installed but not running and automatic start failed. "
                        "Please start Docker manually."
                    )
            else:
                raise RuntimeError("Docker is not running. Please start Docker.")

        logger.info("✅ Docker is ready")
        return True

    @staticmethod
    def get_status() -> dict:
        """Get Docker status information.

        Returns:
            Status dictionary with installation and running state
        """
        return {
            "installed": DockerManager.is_installed(),
            "running": DockerManager.is_running(),
            "platform": platform.system()
        }


def ensure_docker_ready(install: bool = True, start: bool = True) -> bool:
    """Convenience function to ensure Docker is ready.

    Args:
        install: Auto-install if missing
        start: Auto-start if stopped

    Returns:
        True if Docker is ready
    """
    return DockerManager.ensure_docker_ready(install, start)
