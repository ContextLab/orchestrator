"""Utility functions for model management."""

import re
from typing import Optional, Dict, Any
import subprocess
import os


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
        size_match = re.search(r'(\d+\.?\d*)([kmbt])?', size_str.lower())
        if size_match:
            number = float(size_match.group(1))
            unit = size_match.group(2) or 'b'
            
            multipliers = {
                'k': 0.001,  # thousands -> billions
                'm': 0.001,  # millions -> billions  
                'b': 1.0,    # billions
                't': 1000.0  # trillions -> billions
            }
            return number * multipliers.get(unit, 1.0)
    
    # Try to extract size from model name
    size_patterns = [
        r'(\d+\.?\d*)b',  # e.g., "7b", "13b", "2.7b"
        r'(\d+)B',         # e.g., "7B", "13B"
        r'-(\d+\.?\d*)b-', # e.g., "llama-7b-chat"
        r':(\d+)b',        # e.g., "gemma2:27b"
    ]
    
    for pattern in size_patterns:
        match = re.search(pattern, model_name, re.IGNORECASE)
        if match:
            return float(match.group(1))
    
    # Default sizes for known models
    known_sizes = {
        'gpt-4o': 1760.0,  # Estimated
        'gpt-4.1': 1760.0,  # Estimated
        'o3': 2000.0,  # Estimated
        'o4-mini': 70.0,  # Estimated
        'claude-4-opus': 2500.0,  # Estimated
        'claude-4-sonnet': 600.0,  # Estimated
        'claude-3.7-sonnet': 400.0,  # Estimated
        'claude-3.5-sonnet': 200.0,  # Estimated
        'gemini-2.5-pro': 1500.0,  # Estimated
        'gemini-2.5-flash': 80.0,  # Estimated
        'gemini-2.0-pro': 1000.0,  # Estimated
        'tinyllama': 1.1,  # 1.1B parameters
        'phi-2': 2.7,  # 2.7B parameters
        'phi-3.5': 3.8,  # 3.8B parameters
        'gpt2': 0.117,  # 117M parameters
    }
    
    for known_model, size in known_sizes.items():
        if known_model in model_name.lower():
            return size
    
    # Default to small size if unknown
    return 1.0


def check_ollama_installed() -> bool:
    """Check if Ollama is installed and available."""
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def check_ollama_model(model_name: str) -> bool:
    """Check if an Ollama model is available locally."""
    if not check_ollama_installed():
        return False
        
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            # Check if model is in the list
            return model_name in result.stdout
    except subprocess.SubprocessError:
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
        result = subprocess.run(['ollama', 'pull', model_name],
                              capture_output=True, text=True, timeout=600)  # 10 min timeout
        
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
    """Load model configuration from YAML file."""
    import yaml
    from pathlib import Path
    
    # Try multiple locations for the config file
    search_paths = [
        Path(config_path),  # Current directory
        Path.home() / ".orchestrator" / config_path,  # User config directory
        Path(__file__).parent.parent.parent / config_path,  # Project root
        Path(os.environ.get("ORCHESTRATOR_HOME", ".")) / config_path,  # Env variable
    ]
    
    for path in search_paths:
        if path.exists():
            with open(path, 'r') as f:
                return yaml.safe_load(f)
    
    # Return default configuration if no file found
    return {
        "models": [
            {
                "source": "ollama",
                "name": "llama3.1:8b",
                "expertise": ["general", "reasoning", "multilingual"],
                "size": "8b"
            },
            {
                "source": "ollama", 
                "name": "llama3.2:1b",
                "expertise": ["general", "fast"],
                "size": "1b"
            },
            {
                "source": "huggingface",
                "name": "microsoft/Phi-3.5-mini-instruct",
                "expertise": ["reasoning", "code", "compact"],
                "size": "3.8b"
            }
        ],
        "defaults": {
            "expertise_preferences": {
                "code": "qwen2.5-coder:7b",
                "reasoning": "deepseek-r1:8b",
                "fast": "llama3.2:1b",
                "general": "llama3.1:8b"
            },
            "fallback_chain": [
                "llama3.1:8b",
                "mistral:7b",
                "llama3.2:1b",
                "microsoft/Phi-3.5-mini-instruct"
            ]
        }
    }