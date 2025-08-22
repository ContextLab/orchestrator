"""Command-line interface for Orchestrator."""

import click
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from .utils.api_keys import get_configured_providers, add_api_key, validate_api_keys


def setup_logging():
    """Configure logging based on LOG_LEVEL environment variable."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Map string levels to logging constants
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    
    level = level_map.get(log_level, logging.INFO)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Set orchestrator logger to the specified level
    logger = logging.getLogger("orchestrator")
    logger.setLevel(level)


@click.group()
@click.option("--log-level", 
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                               case_sensitive=False),
              help="Set logging level (overrides LOG_LEVEL environment variable)")
def cli(log_level):
    """Orchestrator - AI pipeline orchestration framework."""
    # Set up logging - apply CLI parameter if provided
    if log_level:
        os.environ["LOG_LEVEL"] = log_level.upper()
    setup_logging()


@cli.group()
def keys():
    """Manage API keys for AI providers."""
    pass


@keys.command()
def setup():
    """Run interactive setup for API keys."""
    # Import here to avoid circular imports
    import subprocess

    setup_script = Path(__file__).parent.parent.parent / "scripts" / "setup_api_keys.py"

    if not setup_script.exists():
        click.echo(f"Error: Setup script not found at {setup_script}", err=True)
        sys.exit(1)

    # Run the setup script
    subprocess.run([sys.executable, str(setup_script)])


@keys.command()
def list():
    """Show configured providers (not the keys)."""
    try:
        providers = get_configured_providers()
        if providers:
            click.echo("Configured providers:")
            for provider in providers:
                click.echo(f"  ✓ {provider}")
        else:
            click.echo("No providers configured.")
            click.echo("Run 'orchestrator keys setup' to configure API keys.")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@keys.command()
@click.argument(
    "provider",
    type=click.Choice(
        ["anthropic", "google", "huggingface", "openai"], case_sensitive=False
    ),
)
def add(provider: str):
    """Add single key interactively for a specific provider."""
    from getpass import getpass

    provider_map = {
        "anthropic": ("Anthropic", "sk-ant-..."),
        "google": ("Google AI", "AIza..."),
        "huggingface": ("Hugging Face", "hf_..."),
        "openai": ("OpenAI", "sk-..."),
    }

    provider_name, example = provider_map[provider.lower()]

    click.echo(f"Adding API key for {provider_name}")
    key = getpass(f"Enter API key (e.g., {example}): ").strip()

    if not key:
        click.echo("No key provided. Aborting.")
        return

    try:
        add_api_key(provider, key)
        click.echo(f"✓ API key for {provider_name} saved successfully.")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@keys.command()
def validate():
    """Test all configured keys work."""
    click.echo("Validating API keys...")

    try:
        results = validate_api_keys()

        all_valid = True
        for provider, is_valid in results.items():
            if is_valid:
                click.echo(f"  ✓ {provider}: Configured")
            else:
                click.echo(f"  ✗ {provider}: Not configured")
                all_valid = False

        if not all_valid:
            click.echo("\nSome providers are not configured.")
            click.echo("Run 'orchestrator keys setup' to configure missing keys.")
            sys.exit(1)
        else:
            click.echo("\nAll providers are configured!")
            click.echo("\nNote: This currently only checks if keys exist.")
            click.echo("Future versions will validate keys by making test API calls.")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# Add more commands as needed
@cli.command()
@click.argument("pipeline_file", type=click.Path(exists=True))
@click.option("--context", "-c", help="Context JSON file", type=click.Path(exists=True))
def run(pipeline_file: str, context: Optional[str]):
    """Run a pipeline from a YAML file."""
    click.echo(f"Running pipeline: {pipeline_file}")
    # This would integrate with the existing run_pipeline.py functionality
    click.echo("Note: Pipeline execution not yet integrated into CLI")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
