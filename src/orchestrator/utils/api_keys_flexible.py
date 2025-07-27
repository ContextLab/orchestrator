"""Flexible API key loading that doesn't require all keys to be present."""

import os
from pathlib import Path
from typing import Dict, Set, Optional

from dotenv import load_dotenv


def load_api_keys_optional() -> Dict[str, str]:
    """Load available API keys from ~/.orchestrator/.env or environment.

    Unlike load_api_keys(), this doesn't raise errors for missing keys.
    It returns a dict of available keys.

    Returns:
        Dict mapping provider names to their API keys (if available)
    """
    # Debug logging
    is_github_actions = os.getenv("GITHUB_ACTIONS")
    if is_github_actions:
        print(f">> Running in GitHub Actions (GITHUB_ACTIONS={is_github_actions})")
    
    # Check if running in GitHub Actions
    if is_github_actions:
        # Use environment variables directly - they're injected as secrets
        print(">> Using environment variables from GitHub secrets")
    else:
        # Load from ~/.orchestrator/.env for local development
        env_path = Path.home() / ".orchestrator" / ".env"
        if env_path.exists():
            print(f">> Loading API keys from {env_path}")
            load_dotenv(env_path)
        else:
            # Try legacy location
            legacy_path = Path(".env")
            if legacy_path.exists():
                print(
                    f"Warning: Found .env in current directory. Please move it to {env_path}"
                )
                load_dotenv(legacy_path)

    # Collect available keys
    provider_keys = {
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_AI_API_KEY",
        "huggingface": "HF_TOKEN",
        "openai": "OPENAI_API_KEY",
    }

    available = {}
    for provider, env_var in provider_keys.items():
        value = os.getenv(env_var)
        if value:
            # Don't log the actual key value for security
            available[provider] = value
            print(f">> Found API key for {provider} (length: {len(value)})")
        else:
            print(f">> No API key found for {provider} ({env_var})")

    print(f">> Total API keys found: {len(available)}")
    return available


def get_missing_providers(required: Optional[Set[str]] = None) -> Set[str]:
    """Get set of providers that are missing API keys.

    Args:
        required: Set of required provider names. If None, checks all known providers.

    Returns:
        Set of provider names that are missing API keys
    """
    all_providers = {"anthropic", "google", "huggingface", "openai"}
    providers_to_check = required if required else all_providers

    available = load_api_keys_optional()
    return providers_to_check - set(available.keys())


def ensure_api_key(provider: str) -> str:
    """Ensure an API key is available for a specific provider.

    Args:
        provider: Provider name (anthropic, google, huggingface, openai)

    Returns:
        The API key value

    Raises:
        EnvironmentError: If the key is not available
    """
    available = load_api_keys_optional()

    if provider in available:
        return available[provider]

    provider_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_AI_API_KEY",
        "huggingface": "HF_TOKEN",
        "openai": "OPENAI_API_KEY",
    }

    env_var = provider_map.get(provider, provider.upper() + "_API_KEY")
    env_path = Path.home() / ".orchestrator" / ".env"

    raise EnvironmentError(
        f"Missing API key for {provider} ({env_var})\n"
        f"Please configure it in {env_path} or set as an environment variable.\n"
        f"Run 'orchestrator keys setup' for interactive setup."
    )
