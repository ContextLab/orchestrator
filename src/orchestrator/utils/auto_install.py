"""Automatic package installation utilities."""

import subprocess
import sys
import importlib
import logging
from typing import List, Optional, Dict, Any
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def is_package_installed(package_name: str) -> bool:
    """
    Check if a package is installed.
    
    Args:
        package_name: Name of the package to check
        
    Returns:
        True if package is installed, False otherwise
    """
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def install_package(package_name: str, pip_name: Optional[str] = None) -> bool:
    """
    Install a package using pip.
    
    Args:
        package_name: Import name of the package
        pip_name: Name to use with pip (if different from import name)
        
    Returns:
        True if installation successful, False otherwise
    """
    if is_package_installed(package_name):
        return True
    
    install_name = pip_name or package_name
    
    logger.info(f"Package '{package_name}' not found. Installing '{install_name}'...")
    
    try:
        # Use the same Python interpreter that's running this script
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", install_name],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            logger.info(f"Successfully installed '{install_name}'")
            # Clear the cache since we installed a new package
            is_package_installed.cache_clear()
            return True
        else:
            logger.error(f"Failed to install '{install_name}': {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error installing '{install_name}': {e}")
        return False


def ensure_packages(requirements: Dict[str, Optional[str]]) -> Dict[str, bool]:
    """
    Ensure multiple packages are installed.
    
    Args:
        requirements: Dict mapping import names to pip names (None if same)
        
    Returns:
        Dict mapping package names to installation success status
    """
    results = {}
    
    for package_name, pip_name in requirements.items():
        if is_package_installed(package_name):
            results[package_name] = True
        else:
            results[package_name] = install_package(package_name, pip_name)
    
    return results


# Common package mappings (import name -> pip name)
PACKAGE_MAPPINGS = {
    # AI/ML packages
    "transformers": "transformers",
    "torch": "torch",
    "tensorflow": "tensorflow",
    "anthropic": "anthropic",
    "openai": "openai",
    "google.generativeai": "google-generativeai",
    
    # LangChain providers
    "langchain_openai": "langchain-openai",
    "langchain_anthropic": "langchain-anthropic",
    "langchain_google_genai": "langchain-google-genai",
    "langchain_community": "langchain-community",
    "langchain_huggingface": "langchain-huggingface",
    
    # Data processing
    "pandas": "pandas",
    "numpy": "numpy",
    "PIL": "pillow",
    "cv2": "opencv-python",
    
    # Web/Network
    "aiohttp": "aiohttp",
    "requests": "requests",
    "beautifulsoup4": "beautifulsoup4",
    "selenium": "selenium",
    
    # Document processing
    "pypdf": "pypdf",
    "docx": "python-docx",
    "openpyxl": "openpyxl",
    
    # Audio/Video
    "moviepy": "moviepy",
    "pydub": "pydub",
    "speech_recognition": "SpeechRecognition",
    
    # Utilities
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
    "jinja2": "jinja2",
}


def auto_install_for_import(module_name: str) -> bool:
    """
    Automatically install a package when importing fails.
    
    Args:
        module_name: Name of the module to import
        
    Returns:
        True if module is available (already installed or successfully installed)
    """
    if is_package_installed(module_name):
        return True
    
    # Check if we know the pip name for this module
    pip_name = PACKAGE_MAPPINGS.get(module_name)
    
    if pip_name:
        return install_package(module_name, pip_name)
    else:
        # Try using the module name as pip name
        return install_package(module_name)


def safe_import(module_name: str, auto_install: bool = True) -> Optional[Any]:
    """
    Safely import a module with optional auto-installation.
    
    Args:
        module_name: Name of the module to import
        auto_install: Whether to try installing if import fails
        
    Returns:
        Imported module or None if import fails
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        if auto_install and auto_install_for_import(module_name):
            try:
                return importlib.import_module(module_name)
            except ImportError:
                pass
    
    return None