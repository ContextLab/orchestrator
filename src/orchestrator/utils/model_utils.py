"""Utility functions for model management."""

import re
import subprocess
from typing import Any, Dict, Optional

from .model_config_loader import get_model_config_loader


def parse_model_size(model_name: str, size_str: Optional[str] = None) -> float:
    """
    Parse model size from string representation to float (in billions of parameters).

    Args:
        model_name: The model name which may contain size information
        size_str: Explicit size string (e.g., "7b", "175b", "1.5t")

    Returns:
        Size in billions of parameters as float
    """
    if size_str:
        # Parse explicit size string
        size_match = re.search(r"(\d+\.?\d*)([kmbt])?", size_str.lower())
        if size_match:
            number = float(size_match.group(1))
            unit = size_match.group(2) or "b"

            multipliers = {
                "k": 0.001,  # thousands -> billions
                "m": 0.001,  # millions -> billions
                "b": 1.0,  # billions
                "t": 1000.0,  # trillions -> billions
            }
            return number * multipliers.get(unit, 1.0)

    # Try to extract size from model name
    size_patterns = [
        r"(\d+\.?\d*)b",  # e.g., "7b", "13b", "2.7b"
        r"(\d+)B",  # e.g., "7B", "13B"
        r"-(\d+\.?\d*)b-",  # e.g., "llama-7b-chat"
        r":(\d+)b",  # e.g., "gemma2:27b"
    ]

    for pattern in size_patterns:
        match = re.search(pattern, model_name, re.IGNORECASE)
        if match:
            return float(match.group(1))

    # Try to get size from models.yaml configuration
    try:
        loader = get_model_config_loader()
        config = loader.load_config()
        if "models" in config:
            # Check if model exists in config
            if model_name in config["models"]:
                model_config = config["models"][model_name]
                if "size_b" in model_config:
                    return float(model_config["size_b"])
            
            # Also check for partial matches
            for model_id, model_config in config["models"].items():
                if (
                    model_id.lower() in model_name.lower()
                    or model_name.lower() in model_id.lower()
                ):
                    if "size_b" in model_config:
                        return float(model_config["size_b"])
    except Exception:
        # If loading config fails, continue with default
        pass

    # Default to small size if unknown
    return 1.0


def check_ollama_installed() -> bool:
    """Check if Ollama is installed and available."""
    try:
        # Use simple check without capturing output to avoid hanging
        result = subprocess.run(
            ["which", "ollama"], 
            capture_output=True, 
            text=True, 
            timeout=1
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        return False


def check_ollama_model(model_name: str) -> bool:
    """Check if an Ollama model is available locally."""
    if not check_ollama_installed():
        return False

    try:
        result = subprocess.run(
            ["ollama", "list"], 
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True, 
            timeout=3  # Reduced timeout
        )
        if result.returncode == 0:
            # Check if model is in the list
            return model_name in result.stdout
    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
        pass
    return False


def install_ollama_model(model_name: str) -> bool:
    """
    Attempt to install an Ollama model.

    Returns:
        True if installation successful, False otherwise
    """
    if not check_ollama_installed():
        print(f">>   ⚠️  Ollama not installed, cannot auto-install {model_name}")
        return False

    print(f">>   📥 Installing Ollama model: {model_name}")
    try:
        # Run ollama pull command
        result = subprocess.run(
            ["ollama", "pull", model_name], capture_output=True, text=True, timeout=600
        )  # 10 min timeout

        if result.returncode == 0:
            print(f">>   ✅ Successfully installed {model_name}")
            return True
        else:
            print(f">>   ❌ Failed to install {model_name}: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f">>   ⏱️  Installation of {model_name} timed out")
        return False
    except subprocess.SubprocessError as e:
        print(f">>   ❌ Error installing {model_name}: {e}")
        return False


def load_model_config(config_path: str = "models.yaml") -> Dict[str, Any]:
    """Load model configuration from YAML file.
    
    This is a compatibility wrapper that uses the new ModelConfigLoader.
    """
    
    # Use the new config loader
    loader = get_model_config_loader()
    config = loader.load_config()
    
    # Transform to old format for compatibility if needed
    # The new format has models as a dict, old format expected a list
    if "models" in config and isinstance(config["models"], dict):
        # Keep the new format - callers will be updated
        return config
    
    # Return default configuration if no file found
    return {
        "models": [
            {
                "source": "ollama",
                "name": "llama3.1:8b",
                "expertise": ["general", "reasoning", "multilingual"],
                "size": "8b",
            },
            {
                "source": "ollama",
                "name": "llama3.2:1b",
                "expertise": ["general", "fast"],
                "size": "1b",
            },
            {
                "source": "huggingface",
                "name": "microsoft/Phi-3.5-mini-instruct",
                "expertise": ["reasoning", "code", "compact"],
                "size": "3.8b",
            },
        ],
        "defaults": {
            "expertise_preferences": {
                "code": "qwen2.5-coder:7b",
                "reasoning": "deepseek-r1:8b",
                "fast": "llama3.2:1b",
                "general": "llama3.1:8b",
            },
            "fallback_chain": [
                "llama3.1:8b",
                "mistral:7b",
                "llama3.2:1b",
                "microsoft/Phi-3.5-mini-instruct",
            ],
        },
    }
