#!/usr/bin/env python3
"""Interactive setup script for orchestrator API keys."""

import sys
from pathlib import Path
from getpass import getpass


def setup_api_keys():
    """Interactive setup for API keys."""
    print("Orchestrator API Key Setup")
    print("=" * 50)
    print("\nThis will configure API keys in ~/.orchestrator/.env")
    print("Leave blank to skip any provider.\n")
    
    env_path = Path.home() / '.orchestrator' / '.env'
    env_path.parent.mkdir(exist_ok=True)
    
    # Read existing keys if any
    existing_keys = {}
    if env_path.exists():
        print(f"Found existing configuration at {env_path}")
        print("Press Enter to keep existing values, or enter new ones.\n")
        
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    existing_keys[key] = value.strip('"')
    
    keys = {}
    
    # Prompt for each key
    providers = [
        ("Anthropic", "ANTHROPIC_API_KEY", "sk-ant-..."),
        ("Google AI", "GOOGLE_AI_API_KEY", "AIza..."),
        ("Hugging Face", "HF_TOKEN", "hf_..."),
        ("OpenAI", "OPENAI_API_KEY", "sk-..."),
    ]
    
    for provider, env_var, example in providers:
        current = existing_keys.get(env_var)
        if current:
            # Mask the current key
            masked = current[:10] + "..." + current[-4:] if len(current) > 14 else "***"
            prompt = f"{provider} API Key (current: {masked}): "
        else:
            prompt = f"{provider} API Key (e.g., {example}): "
        
        # Use getpass to hide input
        value = getpass(prompt).strip()
        
        if value:
            keys[env_var] = value
        elif current:
            keys[env_var] = current
    
    # Also handle GOOGLE_API_KEY for compatibility
    if keys.get('GOOGLE_AI_API_KEY'):
        keys['GOOGLE_API_KEY'] = keys['GOOGLE_AI_API_KEY']
    
    # Write to file
    with open(env_path, 'w') as f:
        f.write("# Orchestrator API Keys\n")
        f.write("# This file contains sensitive API keys - DO NOT COMMIT TO GIT\n\n")
        
        if 'OPENAI_API_KEY' in keys:
            f.write("# OpenAI API Key\n")
            f.write(f'OPENAI_API_KEY="{keys["OPENAI_API_KEY"]}"\n\n')
        
        if 'ANTHROPIC_API_KEY' in keys:
            f.write("# Anthropic API Key\n")
            f.write(f'ANTHROPIC_API_KEY="{keys["ANTHROPIC_API_KEY"]}"\n\n')
        
        if 'GOOGLE_AI_API_KEY' in keys:
            f.write("# Google AI/Gemini API Key\n")
            f.write(f'GOOGLE_AI_API_KEY="{keys["GOOGLE_AI_API_KEY"]}"\n\n')
            
        if 'GOOGLE_API_KEY' in keys:
            f.write("# Also set the alternative Google env var that some libraries use\n")
            f.write(f'GOOGLE_API_KEY="{keys["GOOGLE_API_KEY"]}"\n\n')
        
        if 'HF_TOKEN' in keys:
            f.write("# Hugging Face Token\n")
            f.write(f'HF_TOKEN="{keys["HF_TOKEN"]}"\n')
    
    # Set secure permissions
    env_path.chmod(0o600)
    
    print(f"\n✓ API keys saved to {env_path}")
    
    # Validate configuration
    print("\nValidating configuration...")
    
    configured = []
    missing = []
    
    for provider, env_var, _ in providers:
        if env_var in keys:
            configured.append(provider)
        else:
            missing.append(provider)
    
    if configured:
        print(f"✓ Configured: {', '.join(configured)}")
    
    if missing:
        print(f"⚠️  Not configured: {', '.join(missing)}")
        print("\nYou can add missing keys later by running this script again.")
    
    print("\nTo test your configuration, run:")
    print("  python -m orchestrator.utils.api_keys")


if __name__ == "__main__":
    try:
        setup_api_keys()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)