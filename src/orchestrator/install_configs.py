"""Install default configuration files to user's home directory.

The default configs are packaged resources under ``orchestrator.config`` and are
read through :mod:`importlib.resources`. They must never be located by walking
``__file__`` upwards: inside an installed wheel that walk escapes the package
and silently resolves to nothing, which made ``orchestrator-install-configs``
install no configuration at all.
"""

from importlib.resources import as_file, files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Dict

CONFIG_PACKAGE = "orchestrator.config"

# Packaged default configuration files and what each one is for.
CONFIG_FILES: Dict[str, str] = {
    "orchestrator.yaml": "Default orchestrator configuration",
    "models.yaml": "Model configuration and registry",
}

README_CONTENT = """# Orchestrator Configuration Directory

This directory contains configuration files for the Orchestrator framework.

## Configuration Files

### models.yaml
Defines available AI models and their properties. You can:
- Add new models from Ollama, HuggingFace, or cloud providers
- Set model expertise areas and size information
- Configure default model preferences
- Define fallback chains for model selection

### orchestrator.yaml
Main configuration for the Orchestrator framework. You can:
- Set default execution parameters
- Configure resource limits
- Customize error handling behavior
- Set up monitoring and logging preferences

## Customization

Feel free to edit these files to customize Orchestrator's behavior. The framework
will automatically pick up changes on the next run.

For more information, see: https://orc.readthedocs.io/en/latest/user_guide/configuration.html
"""


def packaged_config_resource(filename: str) -> Traversable:
    """Return the packaged default config resource for ``filename``.

    Args:
        filename: Name of the resource inside :data:`CONFIG_PACKAGE`.

    Returns:
        The resource, guaranteed to exist.

    Raises:
        FileNotFoundError: If the resource is missing from the installed
            package. That is a packaging failure and must be loud rather than
            degrading to a silent no-op.
    """
    resource = files(CONFIG_PACKAGE) / filename
    if not resource.is_file():
        raise FileNotFoundError(
            f"Packaged default config {filename!r} is missing from the "
            f"{CONFIG_PACKAGE} package; the installation is incomplete."
        )
    return resource


def packaged_config_path(filename: str) -> Path:
    """Return a filesystem path to a packaged default config file.

    Orchestrator is distributed as a regular (unzipped) wheel, so the resource
    always has a stable location on disk.

    Args:
        filename: Name of the resource inside :data:`CONFIG_PACKAGE`.

    Returns:
        Path to the resource on disk.

    Raises:
        FileNotFoundError: If the resource is missing from the package.
    """
    with as_file(packaged_config_resource(filename)) as path:
        return Path(path)


def user_config_dir() -> Path:
    """Return the user configuration directory (``~/.orchestrator``)."""
    return Path.home() / ".orchestrator"


def install_default_configs() -> None:
    """Copy default configuration files to ~/.orchestrator/ if they don't exist.

    Existing user files are never overwritten, so repeated runs are idempotent.
    Returns ``None``: this is the ``orchestrator-install-configs`` console
    script entry point, and console scripts pass their return value to
    ``sys.exit``.

    Raises:
        FileNotFoundError: If a packaged default config is missing.
    """
    config_dir = user_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)

    for filename in CONFIG_FILES:
        resource = packaged_config_resource(filename)
        dst_file = config_dir / filename

        if dst_file.exists():
            print(f"Keeping existing {filename} at {dst_file}")
            continue

        dst_file.write_bytes(resource.read_bytes())
        print(f"Installed {filename} to {dst_file}")

    readme_path = config_dir / "README.md"
    if not readme_path.exists():
        readme_path.write_text(README_CONTENT)
        print(f"Created README at {readme_path}")

    print(f"\nConfiguration files installed to: {config_dir}")
    print("You can customize these files to change Orchestrator's behavior.")


if __name__ == "__main__":
    install_default_configs()
