"""Registry installer for managing ~/.orchestrator directory structure."""

import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import yaml

logger = logging.getLogger(__name__)


class RegistryInstaller:
    """Manages installation and setup of registries in user home directory."""

    def __init__(self, home_dir: Optional[Path] = None):
        """Initialize registry installer.

        Args:
            home_dir: Override home directory (for testing). Defaults to ~/.orchestrator
        """
        self.home_dir = home_dir or (Path.home() / ".orchestrator")
        self.skills_dir = self.home_dir / "skills"
        self.models_dir = self.home_dir / "models"

        # Package registry location (source)
        self.package_root = Path(__file__).parent.parent.parent.parent  # Up to orchestrator root
        self.package_registry = self.package_root / "orchestrator" / "registry"

    def is_installed(self) -> bool:
        """Check if registries are already installed."""
        return (
            self.home_dir.exists()
            and (self.skills_dir / "registry.yaml").exists()
            and (self.models_dir / "registry.yaml").exists()
        )

    def install(self, force: bool = False) -> bool:
        """Install default registries to user home.

        Args:
            force: Force reinstall even if already exists

        Returns:
            True if installation successful
        """
        try:
            # Create directories
            self.home_dir.mkdir(exist_ok=True)
            self.skills_dir.mkdir(exist_ok=True)
            self.models_dir.mkdir(exist_ok=True)

            # Install skills registry
            if force or not (self.skills_dir / "registry.yaml").exists():
                self._install_skills_registry()

            # Install models registry
            if force or not (self.models_dir / "registry.yaml").exists():
                self._install_models_registry()

            # Create .env file if it doesn't exist
            env_file = self.home_dir / ".env"
            if not env_file.exists():
                self._create_env_template(env_file)

            logger.info(f"Registry installed successfully at {self.home_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to install registry: {e}")
            return False

    def _install_skills_registry(self) -> None:
        """Install default skills registry."""
        source = self.package_registry / "skills" / "default_registry.yaml"
        dest = self.skills_dir / "registry.yaml"

        if source.exists():
            shutil.copy(source, dest)
            logger.info(f"Installed skills registry to {dest}")

            # Copy default skills if they exist
            default_skills_src = self.package_registry / "skills" / "default_skills"
            if default_skills_src.exists():
                default_skills_dest = self.skills_dir / "default_skills"
                if not default_skills_dest.exists():
                    shutil.copytree(default_skills_src, default_skills_dest)
                    logger.info(f"Installed default skills to {default_skills_dest}")
        else:
            # Create minimal registry if source doesn't exist
            minimal_registry = {
                "version": "1.0.0",
                "skills": {},
                "metadata": {
                    "description": "User skills registry",
                    "created": "2024-01-15"
                }
            }
            with open(dest, 'w') as f:
                yaml.dump(minimal_registry, f, default_flow_style=False)
            logger.info(f"Created minimal skills registry at {dest}")

    def _install_models_registry(self) -> None:
        """Install default models registry."""
        source = self.package_registry / "models" / "default_registry.yaml"
        dest = self.models_dir / "registry.yaml"

        if source.exists():
            shutil.copy(source, dest)
            logger.info(f"Installed models registry to {dest}")
        else:
            # Create minimal registry if source doesn't exist
            minimal_registry = {
                "version": "1.0.0",
                "providers": {
                    "anthropic": {
                        "enabled": True,
                        "api_key_env": "ANTHROPIC_API_KEY"
                    }
                },
                "models": {},
                "metadata": {
                    "description": "User models registry",
                    "created": "2024-01-15"
                }
            }
            with open(dest, 'w') as f:
                yaml.dump(minimal_registry, f, default_flow_style=False)
            logger.info(f"Created minimal models registry at {dest}")

    def _create_env_template(self, env_file: Path) -> None:
        """Create template .env file."""
        template = """# Orchestrator API Keys
# Add your API keys here

# Anthropic (Required for Claude Skills)
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Optional: Other providers (for future extensions)
# OPENAI_API_KEY=your-openai-api-key-here
# GOOGLE_AI_API_KEY=your-google-api-key-here
# HF_TOKEN=your-huggingface-token-here
"""
        with open(env_file, 'w') as f:
            f.write(template)
        logger.info(f"Created .env template at {env_file}")

    def update_skill_registry(self, skill_name: str, skill_data: Dict[str, Any]) -> bool:
        """Add or update a skill in the registry.

        Args:
            skill_name: Name of the skill
            skill_data: Skill configuration data

        Returns:
            True if successful
        """
        try:
            registry_path = self.skills_dir / "registry.yaml"

            # Load existing registry
            with open(registry_path, 'r') as f:
                registry = yaml.safe_load(f)

            # Update skill
            if 'skills' not in registry:
                registry['skills'] = {}
            registry['skills'][skill_name] = skill_data

            # Save updated registry
            with open(registry_path, 'w') as f:
                yaml.dump(registry, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Updated skill '{skill_name}' in registry")
            return True

        except Exception as e:
            logger.error(f"Failed to update skill registry: {e}")
            return False

    def get_skills_registry(self) -> Dict[str, Any]:
        """Load and return the skills registry."""
        registry_path = self.skills_dir / "registry.yaml"
        if not registry_path.exists():
            return {}

        with open(registry_path, 'r') as f:
            return yaml.safe_load(f)

    def get_models_registry(self) -> Dict[str, Any]:
        """Load and return the models registry."""
        registry_path = self.models_dir / "registry.yaml"
        if not registry_path.exists():
            return {}

        with open(registry_path, 'r') as f:
            return yaml.safe_load(f)

    def create_skill_directory(self, skill_name: str) -> Path:
        """Create directory for a new skill.

        Args:
            skill_name: Name of the skill

        Returns:
            Path to the created skill directory
        """
        skill_dir = self.skills_dir / skill_name
        skill_dir.mkdir(exist_ok=True)

        # Create standard skill structure
        (skill_dir / "examples").mkdir(exist_ok=True)
        (skill_dir / "tests").mkdir(exist_ok=True)

        return skill_dir

    def verify_installation(self) -> Dict[str, bool]:
        """Verify the installation status.

        Returns:
            Dictionary with verification results
        """
        return {
            "home_dir_exists": self.home_dir.exists(),
            "skills_dir_exists": self.skills_dir.exists(),
            "models_dir_exists": self.models_dir.exists(),
            "skills_registry_exists": (self.skills_dir / "registry.yaml").exists(),
            "models_registry_exists": (self.models_dir / "registry.yaml").exists(),
            "env_file_exists": (self.home_dir / ".env").exists(),
        }

    def __str__(self) -> str:
        """String representation."""
        return f"RegistryInstaller(home={self.home_dir})"


def ensure_registry_installed() -> RegistryInstaller:
    """Ensure registry is installed and return installer instance.

    Returns:
        Configured RegistryInstaller instance
    """
    installer = RegistryInstaller()

    if not installer.is_installed():
        logger.info("Registry not found, installing defaults...")
        installer.install()

    return installer
