"""API key management for orchestrator."""

import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv


def load_api_keys() -> None:
    """Load API keys from ~/.orchestrator/.env or environment.

    Raises:
        EnvironmentError: If required API keys are missing
    """
    # Check if running in GitHub Actions
    if os.getenv("GITHUB_ACTIONS"):
        # Use environment variables directly - they're injected as secrets
        pass
    else:
        # Load from ~/.orchestrator/.env for local development
        env_path = Path.home() / ".orchestrator" / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        else:
            # Try legacy location
            legacy_path = Path(".env")
            if legacy_path.exists():
                print(f"Warning: Found .env in current directory. Please move it to {env_path}")
                load_dotenv(legacy_path)

    # Verify all required keys are present
    required_keys = {
        "ANTHROPIC_API_KEY": "Anthropic",
        "GOOGLE_AI_API_KEY": "Google AI",
        "HF_TOKEN": "Hugging Face",
        "OPENAI_API_KEY": "OpenAI",
    }

    missing = []
    for key, provider in required_keys.items():
        if not os.getenv(key):
            missing.append(f"{provider} ({key})")

    if missing:
        env_path = Path.home() / ".orchestrator" / ".env"
        raise EnvironmentError(
            f"Missing required API keys: {', '.join(missing)}\n"
            f"Please configure them in {env_path}\n"
            f"Run 'orchestrator keys setup' for interactive setup."
        )


def get_configured_providers() -> List[str]:
    """Get list of providers that have API keys configured.

    Returns:
        List of provider names with configured keys
    """
    load_api_keys()

    providers = []
    provider_keys = {
        "Anthropic": "ANTHROPIC_API_KEY",
        "Google AI": "GOOGLE_AI_API_KEY",
        "Hugging Face": "HF_TOKEN",
        "OpenAI": "OPENAI_API_KEY",
    }

    for provider, key in provider_keys.items():
        if os.getenv(key):
            providers.append(provider)

    return providers


def add_api_key(provider: str, key: str) -> None:
    """Add or update an API key for a provider.

    Args:
        provider: Provider name (anthropic, google, huggingface, openai)
        key: API key value
    """
    # Map provider names to env var names
    provider_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_AI_API_KEY",
        "huggingface": "HF_TOKEN",
        "openai": "OPENAI_API_KEY",
    }

    env_var = provider_map.get(provider.lower())
    if not env_var:
        raise ValueError(f"Unknown provider: {provider}")

    # Load existing keys
    env_path = Path.home() / ".orchestrator" / ".env"
    env_path.parent.mkdir(exist_ok=True)

    # Read existing content
    existing = {}
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    existing[k] = v.strip('"')

    # Update with new key
    existing[env_var] = key

    # Write back
    with open(env_path, "w") as f:
        f.write("# Orchestrator API Keys\n")
        f.write("# This file contains sensitive API keys - DO NOT COMMIT TO GIT\n\n")

        if "OPENAI_API_KEY" in existing:
            f.write("# OpenAI API Key\n")
            f.write(f'OPENAI_API_KEY="{existing["OPENAI_API_KEY"]}"\n\n')

        if "ANTHROPIC_API_KEY" in existing:
            f.write("# Anthropic API Key\n")
            f.write(f'ANTHROPIC_API_KEY="{existing["ANTHROPIC_API_KEY"]}"\n\n')

        if "GOOGLE_AI_API_KEY" in existing:
            f.write("# Google AI/Gemini API Key\n")
            f.write(f'GOOGLE_AI_API_KEY="{existing["GOOGLE_AI_API_KEY"]}"\n\n')

        if "GOOGLE_API_KEY" in existing:
            f.write("# Also set the alternative Google env var that some libraries use\n")
            f.write(f'GOOGLE_API_KEY="{existing["GOOGLE_API_KEY"]}"\n\n')

        if "HF_TOKEN" in existing:
            f.write("# Hugging Face Token\n")
            f.write(f'HF_TOKEN="{existing["HF_TOKEN"]}"\n')

    # Set secure permissions
    env_path.chmod(0o600)


def validate_api_keys() -> Dict[str, bool]:
    """Validate that configured API keys work.

    Returns:
        Dict mapping provider names to validation status
    """
    # This will be implemented to actually test each API
    # For now, just check they exist
    load_api_keys()

    results = {}
    provider_keys = {
        "Anthropic": "ANTHROPIC_API_KEY",
        "Google AI": "GOOGLE_AI_API_KEY",
        "Hugging Face": "HF_TOKEN",
        "OpenAI": "OPENAI_API_KEY",
    }

    for provider, key in provider_keys.items():
        results[provider] = bool(os.getenv(key))

    return results
