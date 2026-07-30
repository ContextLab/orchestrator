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


# Exit codes (see docs/adr/0001-product-contract.md):
#   0   success
#   1   execution failure
#   2   validation / compilation failure
#   130 interrupted
EXIT_OK = 0
EXIT_EXECUTION_ERROR = 1
EXIT_VALIDATION_ERROR = 2
EXIT_INTERRUPTED = 130


def _build_orchestrator():
    """Construct an Orchestrator suited to the available environment.

    A pipeline built only from deterministic local tools (filesystem,
    data-processing, validation, ...) needs no model provider at all. The
    default ``Orchestrator()`` refuses to start without a populated model
    registry, which made such pipelines impossible to run without credentials.

    So: register models when credentials/local models are present, and
    otherwise fall back to the tool-only control system. The fallback is
    announced rather than silent -- a step that genuinely needs a model then
    fails with that step's own error, instead of being masked here.
    """
    from .control_systems.tool_integrated_control_system import (
        ToolIntegratedControlSystem,
    )
    from .orchestrator import Orchestrator

    try:
        from . import init_models

        registry = init_models()
        if registry.models:
            return Orchestrator(model_registry=registry)
    except Exception as exc:  # noqa: BLE001 - degrade to the tool-only path
        click.echo(f"Model initialization skipped ({type(exc).__name__}: {exc})", err=True)

    click.echo(
        "No models available - running with deterministic local tools only.",
        err=True,
    )
    return Orchestrator(control_system=ToolIntegratedControlSystem())


def _load_context(context_file: Optional[str], inputs: tuple) -> dict:
    """Build the pipeline context from a JSON file plus -i key=value overrides.

    Raises:
        click.ClickException: if the file is not valid JSON, is not an object,
            or an -i argument is not in ``key=value`` form.
    """
    import json

    ctx: dict = {}
    if context_file:
        try:
            with open(context_file) as fh:
                loaded = json.load(fh)
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"{context_file} is not valid JSON: {exc}")
        if not isinstance(loaded, dict):
            raise click.ClickException(
                f"{context_file} must contain a JSON object, got {type(loaded).__name__}"
            )
        ctx.update(loaded)

    for item in inputs:
        if "=" not in item:
            raise click.ClickException(
                f"--input expects key=value, got {item!r}"
            )
        key, _, value = item.partition("=")
        # Accept JSON scalars/objects so that -i n=3 gives an int, while a bare
        # word stays a string.
        try:
            ctx[key] = json.loads(value)
        except json.JSONDecodeError:
            ctx[key] = value
    return ctx


@cli.command()
@click.argument("pipeline_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--context", "-c", help="Context JSON file", type=click.Path(exists=True))
@click.option(
    "--input",
    "-i",
    "inputs",
    multiple=True,
    help="Pipeline input as key=value (repeatable). Values parse as JSON when possible.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False),
    help="Write results as JSON to this file instead of stdout.",
)
def run(pipeline_file: str, context: Optional[str], inputs: tuple, output: Optional[str]):
    """Run a pipeline from a YAML file.

    Compiles PIPELINE_FILE and executes it, printing the results as JSON.
    """
    import asyncio
    import json

    ctx = _load_context(context, inputs)

    async def _execute():
        orchestrator = _build_orchestrator()
        try:
            return await orchestrator.execute_yaml_file(pipeline_file, ctx)
        finally:
            # Release background workers/connections even when execution fails.
            await orchestrator.shutdown()

    try:
        results = asyncio.run(_execute())
    except KeyboardInterrupt:
        click.echo("Interrupted.", err=True)
        sys.exit(EXIT_INTERRUPTED)
    except Exception as exc:
        # Compilation/validation problems are a distinct, actionable class from
        # a pipeline that compiled but failed while running.
        name = type(exc).__name__
        is_validation = any(
            token in name
            for token in ("Validation", "Compil", "YAML", "Schema", "CircularDependency")
        )
        click.echo(f"{name}: {exc}", err=True)
        sys.exit(EXIT_VALIDATION_ERROR if is_validation else EXIT_EXECUTION_ERROR)

    rendered = json.dumps(results, indent=2, default=str)
    if output:
        Path(output).write_text(rendered)
        click.echo(f"Results written to {output}")
    else:
        click.echo(rendered)

    # A step can fail without raising: the executor records the failure in the
    # step's result and continues. Reporting exit 0 in that case would tell a
    # caller (a shell script, CI, a parent pipeline) that a pipeline succeeded
    # when one of its steps did not.
    failed = _failed_steps(results)
    if failed:
        click.echo(
            f"Pipeline completed with {len(failed)} failed step(s): "
            + ", ".join(sorted(failed)),
            err=True,
        )
        sys.exit(EXIT_EXECUTION_ERROR)

    sys.exit(EXIT_OK)


def _failed_steps(results) -> list:
    """Return the ids of steps whose result reports failure.

    Steps report themselves as ``{"success": bool, "error": ...}``. Anything
    that does not look like a step result is ignored rather than guessed at.
    """
    if not isinstance(results, dict):
        return []
    failed = []
    for step_id, value in results.items():
        if isinstance(value, dict) and value.get("success") is False:
            failed.append(str(step_id))
    return failed


@cli.command("validate")
@click.argument("pipeline_file", type=click.Path(exists=True, dir_okay=False))
def validate_pipeline(pipeline_file: str):
    """Compile a pipeline without running it, and report its task graph."""
    import asyncio

    from .compiler.yaml_compiler import YAMLCompiler

    async def _compile():
        compiler = YAMLCompiler()
        with open(pipeline_file) as fh:
            return await compiler.compile(fh.read(), {})

    try:
        pipeline = asyncio.run(_compile())
    except KeyboardInterrupt:
        click.echo("Interrupted.", err=True)
        sys.exit(EXIT_INTERRUPTED)
    except Exception as exc:
        click.echo(f"{type(exc).__name__}: {exc}", err=True)
        sys.exit(EXIT_VALIDATION_ERROR)

    tasks = getattr(pipeline, "tasks", {}) or {}
    click.echo(f"✓ {pipeline_file} is valid")
    click.echo(f"  pipeline: {getattr(pipeline, 'id', '<unnamed>')}")
    click.echo(f"  tasks: {len(tasks)}")
    for task_id in tasks:
        task = tasks[task_id]
        deps = getattr(task, "dependencies", []) or []
        suffix = f"  <- {', '.join(deps)}" if deps else ""
        click.echo(f"    - {task_id}{suffix}")
    sys.exit(EXIT_OK)


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
