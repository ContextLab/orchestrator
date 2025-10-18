"""Skills registry management for the Orchestrator framework."""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

import yaml

from .installer import RegistryInstaller

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Manages the registry of available skills."""

    def __init__(self, registry_dir: Optional[Path] = None):
        """Initialize skill registry.

        Args:
            registry_dir: Override registry directory (defaults to ~/.orchestrator/skills)
        """
        self.installer = RegistryInstaller(registry_dir.parent if registry_dir else None)
        self.registry_dir = registry_dir or self.installer.skills_dir
        self.registry_file = self.registry_dir / "registry.yaml"
        self._cache = None
        self._ensure_registry_exists()

    def _ensure_registry_exists(self) -> None:
        """Ensure registry is installed."""
        if not self.registry_file.exists():
            logger.info("Registry not found, installing...")
            self.installer.install()

    def _load_registry(self) -> Dict[str, Any]:
        """Load the registry from disk.

        Returns:
            Registry data
        """
        if self._cache is not None:
            return self._cache

        with open(self.registry_file, 'r') as f:
            self._cache = yaml.safe_load(f) or {"version": "1.0.0", "skills": {}}

        return self._cache

    def _save_registry(self, registry: Dict[str, Any]) -> None:
        """Save the registry to disk.

        Args:
            registry: Registry data to save
        """
        with open(self.registry_file, 'w') as f:
            yaml.dump(registry, f, default_flow_style=False, sort_keys=False)
        self._cache = registry

    def register(self, skill_name: str, skill_data: Dict[str, Any]) -> bool:
        """Register a new skill or update existing one.

        Args:
            skill_name: Name of the skill
            skill_data: Skill configuration and metadata

        Returns:
            True if successful
        """
        try:
            registry = self._load_registry()

            if "skills" not in registry:
                registry["skills"] = {}

            registry["skills"][skill_name] = skill_data
            self._save_registry(registry)

            logger.info(f"Registered skill: {skill_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to register skill {skill_name}: {e}")
            return False

    def unregister(self, skill_name: str) -> bool:
        """Remove a skill from the registry.

        Args:
            skill_name: Name of the skill to remove

        Returns:
            True if successful
        """
        try:
            registry = self._load_registry()

            if skill_name in registry.get("skills", {}):
                del registry["skills"][skill_name]
                self._save_registry(registry)
                logger.info(f"Unregistered skill: {skill_name}")
                return True
            else:
                logger.warning(f"Skill not found: {skill_name}")
                return False

        except Exception as e:
            logger.error(f"Failed to unregister skill {skill_name}: {e}")
            return False

    def get(self, skill_name: str) -> Optional[Dict[str, Any]]:
        """Get a skill by name.

        Args:
            skill_name: Name of the skill

        Returns:
            Skill data if found, None otherwise
        """
        registry = self._load_registry()
        return registry.get("skills", {}).get(skill_name)

    def exists(self, skill_name: str) -> bool:
        """Check if a skill exists.

        Args:
            skill_name: Name of the skill

        Returns:
            True if skill exists
        """
        return self.get(skill_name) is not None

    def list_skills(self) -> List[str]:
        """List all registered skill names.

        Returns:
            List of skill names
        """
        registry = self._load_registry()
        return list(registry.get("skills", {}).keys())

    def get_all_skills(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered skills.

        Returns:
            Dictionary of all skills
        """
        registry = self._load_registry()
        return registry.get("skills", {})

    def find_by_capability(self, capability: str) -> List[str]:
        """Find skills that have a specific capability.

        Args:
            capability: Capability to search for

        Returns:
            List of skill names with the capability
        """
        matching_skills = []
        registry = self._load_registry()

        for skill_name, skill_data in registry.get("skills", {}).items():
            capabilities = skill_data.get("capabilities", [])
            if capability in capabilities:
                matching_skills.append(skill_name)

        return matching_skills

    def find_by_category(self, category: str) -> List[str]:
        """Find skills in a specific category.

        Args:
            category: Category name

        Returns:
            List of skill names in the category
        """
        registry = self._load_registry()
        categories = registry.get("categories", {})

        if category in categories:
            return categories[category].get("skills", [])

        return []

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search for skills by query string.

        Args:
            query: Search query (searches name, description, capabilities)

        Returns:
            List of matching skills with their data
        """
        query_lower = query.lower()
        matching_skills = []
        registry = self._load_registry()

        for skill_name, skill_data in registry.get("skills", {}).items():
            # Search in name
            if query_lower in skill_name.lower():
                matching_skills.append({
                    "name": skill_name,
                    "match_field": "name",
                    **skill_data
                })
                continue

            # Search in description
            description = skill_data.get("description", "").lower()
            if query_lower in description:
                matching_skills.append({
                    "name": skill_name,
                    "match_field": "description",
                    **skill_data
                })
                continue

            # Search in capabilities
            capabilities = skill_data.get("capabilities", [])
            if any(query_lower in cap.lower() for cap in capabilities):
                matching_skills.append({
                    "name": skill_name,
                    "match_field": "capabilities",
                    **skill_data
                })

        return matching_skills

    def get_skill_path(self, skill_name: str) -> Optional[Path]:
        """Get the filesystem path for a skill.

        Args:
            skill_name: Name of the skill

        Returns:
            Path to skill directory if exists
        """
        skill = self.get(skill_name)
        if not skill:
            return None

        # Check if skill has a path
        if "path" in skill:
            skill_path = self.registry_dir / skill["path"]
        else:
            skill_path = self.registry_dir / skill_name

        return skill_path if skill_path.exists() else None

    def load_skill_implementation(self, skill_name: str) -> Optional[str]:
        """Load the Python implementation for a skill.

        Args:
            skill_name: Name of the skill

        Returns:
            Python code if found
        """
        skill_path = self.get_skill_path(skill_name)
        if not skill_path:
            return None

        # Check for implementation.py
        impl_file = skill_path / "implementation.py"
        if impl_file.exists():
            with open(impl_file, 'r') as f:
                return f.read()

        # Check for skill.yaml with embedded implementation
        skill_file = skill_path / "skill.yaml"
        if skill_file.exists():
            with open(skill_file, 'r') as f:
                skill_data = yaml.safe_load(f)
                if skill_data and "implementation" in skill_data:
                    impl = skill_data["implementation"]
                    if isinstance(impl, dict) and "code" in impl:
                        return impl["code"]

        return None

    def get_skill_examples(self, skill_name: str) -> List[Dict[str, Any]]:
        """Get examples for a skill.

        Args:
            skill_name: Name of the skill

        Returns:
            List of examples
        """
        skill = self.get(skill_name)
        if not skill:
            return []

        return skill.get("examples", [])

    def get_required_capabilities(self, skill_names: List[str]) -> Set[str]:
        """Get all capabilities required by a set of skills.

        Args:
            skill_names: List of skill names

        Returns:
            Set of required capabilities
        """
        capabilities = set()

        for skill_name in skill_names:
            skill = self.get(skill_name)
            if skill:
                capabilities.update(skill.get("capabilities", []))

        return capabilities

    def validate_skill(self, skill_data: Dict[str, Any]) -> List[str]:
        """Validate skill data structure.

        Args:
            skill_data: Skill data to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check required fields
        required_fields = ["name", "description", "version"]
        for field in required_fields:
            if field not in skill_data:
                errors.append(f"Missing required field: {field}")

        # Check name format
        if "name" in skill_data:
            name = skill_data["name"]
            if not name.replace("-", "").replace("_", "").isalnum():
                errors.append("Skill name must be alphanumeric with hyphens/underscores")

        # Check version format
        if "version" in skill_data:
            version = skill_data["version"]
            if not self._is_valid_version(version):
                errors.append(f"Invalid version format: {version}")

        # Check parameters structure
        if "parameters" in skill_data:
            params = skill_data["parameters"]
            if not isinstance(params, dict):
                errors.append("Parameters must be a dictionary")
            else:
                for param_name, param_spec in params.items():
                    if not isinstance(param_spec, dict):
                        errors.append(f"Parameter '{param_name}' must be a dictionary")
                    elif "type" not in param_spec:
                        errors.append(f"Parameter '{param_name}' missing 'type' field")

        return errors

    def _is_valid_version(self, version: str) -> bool:
        """Check if version string is valid.

        Args:
            version: Version string to check

        Returns:
            True if valid semantic version
        """
        parts = version.split(".")
        if len(parts) != 3:
            return False

        try:
            for part in parts:
                int(part)
            return True
        except ValueError:
            return False

    def add_to_category(self, skill_name: str, category: str) -> bool:
        """Add a skill to a category.

        Args:
            skill_name: Name of the skill
            category: Category name

        Returns:
            True if successful
        """
        try:
            registry = self._load_registry()

            # Ensure categories exist
            if "categories" not in registry:
                registry["categories"] = {}

            # Ensure category exists
            if category not in registry["categories"]:
                registry["categories"][category] = {
                    "description": f"Skills in category: {category}",
                    "skills": []
                }

            # Add skill to category if not already there
            if skill_name not in registry["categories"][category]["skills"]:
                registry["categories"][category]["skills"].append(skill_name)
                self._save_registry(registry)
                logger.info(f"Added skill '{skill_name}' to category '{category}'")

            return True

        except Exception as e:
            logger.error(f"Failed to add skill to category: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the registry.

        Returns:
            Statistics dictionary
        """
        registry = self._load_registry()
        skills = registry.get("skills", {})

        # Count skills by status
        status_counts = {}
        for skill_data in skills.values():
            status = skill_data.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        # Count skills by capabilities
        capability_counts = {}
        for skill_data in skills.values():
            for cap in skill_data.get("capabilities", []):
                capability_counts[cap] = capability_counts.get(cap, 0) + 1

        return {
            "total_skills": len(skills),
            "categories": len(registry.get("categories", {})),
            "status_breakdown": status_counts,
            "capability_breakdown": capability_counts,
            "registry_version": registry.get("version", "unknown")
        }

    def export_skill(self, skill_name: str, output_path: Path) -> bool:
        """Export a skill to a file.

        Args:
            skill_name: Name of the skill to export
            output_path: Path to export to

        Returns:
            True if successful
        """
        try:
            skill_data = self.get(skill_name)
            if not skill_data:
                logger.error(f"Skill not found: {skill_name}")
                return False

            # Include implementation if available
            implementation = self.load_skill_implementation(skill_name)
            if implementation:
                skill_data["implementation"] = {
                    "type": "python",
                    "code": implementation
                }

            # Save to file
            with open(output_path, 'w') as f:
                yaml.dump(skill_data, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Exported skill '{skill_name}' to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export skill: {e}")
            return False

    def import_skill(self, skill_file: Path, overwrite: bool = False) -> bool:
        """Import a skill from a file.

        Args:
            skill_file: Path to skill YAML file
            overwrite: Whether to overwrite existing skill

        Returns:
            True if successful
        """
        try:
            with open(skill_file, 'r') as f:
                skill_data = yaml.safe_load(f)

            # Validate skill data
            errors = self.validate_skill(skill_data)
            if errors:
                logger.error(f"Invalid skill data: {', '.join(errors)}")
                return False

            skill_name = skill_data["name"]

            # Check if skill exists
            if self.exists(skill_name) and not overwrite:
                logger.error(f"Skill '{skill_name}' already exists (use overwrite=True)")
                return False

            # Extract implementation if embedded
            implementation = None
            if "implementation" in skill_data:
                implementation = skill_data["implementation"]
                # Remove from registry data to avoid duplication
                del skill_data["implementation"]

            # Register skill
            if not self.register(skill_name, skill_data):
                return False

            # Save implementation if provided
            if implementation and isinstance(implementation, dict):
                skill_path = self.registry_dir / skill_name
                skill_path.mkdir(exist_ok=True)

                if "code" in implementation:
                    impl_file = skill_path / "implementation.py"
                    with open(impl_file, 'w') as f:
                        f.write(implementation["code"])

            logger.info(f"Imported skill '{skill_name}' from {skill_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to import skill: {e}")
            return False
