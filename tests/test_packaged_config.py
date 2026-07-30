"""Tests that the default config and registry resources are actually packaged.

These are regression tests for a packaging defect: the default configs used to
live outside `src/`, were located by walking `__file__` out of the installed
package, and therefore shipped in no wheel at all. `orchestrator-install-configs`
then installed nothing while reporting success.

Everything here touches the real filesystem and the real installed package. No
mocks: a mocked resource loader would pass even with an empty wheel, which is
exactly the failure these tests exist to catch.
"""

from importlib.resources import files
from pathlib import Path

import pytest
import yaml

from orchestrator.install_configs import (
    CONFIG_FILES,
    CONFIG_PACKAGE,
    install_default_configs,
    packaged_config_path,
    packaged_config_resource,
    user_config_dir,
)
from orchestrator.skills.installer import (
    REGISTRY_PACKAGE,
    RegistryInstaller,
    packaged_registry_resource,
)

REGISTRY_RESOURCES = [
    ("models", "default_registry.yaml"),
    ("skills", "default_registry.yaml"),
]


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Point Path.home() at a throwaway directory."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    assert Path.home() == home
    return home


class TestPackagedResourcesExist:
    """The yaml resources must be reachable through importlib.resources."""

    @pytest.mark.parametrize("filename", sorted(CONFIG_FILES))
    def test_config_resource_is_locatable(self, filename):
        resource = files(CONFIG_PACKAGE) / filename
        assert resource.is_file(), f"{CONFIG_PACKAGE}/{filename} is not packaged"

        text = resource.read_text()
        assert text.strip(), f"{filename} is empty"
        parsed = yaml.safe_load(text)
        assert isinstance(parsed, dict) and parsed, f"{filename} is not a yaml mapping"

    @pytest.mark.parametrize("parts", REGISTRY_RESOURCES, ids=lambda p: "/".join(p))
    def test_registry_resource_is_locatable(self, parts):
        resource = packaged_registry_resource(*parts)
        assert resource.is_file()

        parsed = yaml.safe_load(resource.read_text())
        assert isinstance(parsed, dict) and parsed, (
            f"{REGISTRY_PACKAGE}/{'/'.join(parts)} is not a yaml mapping"
        )

    def test_packaged_config_path_points_inside_the_package(self):
        path = packaged_config_path("models.yaml")
        package_dir = Path(str(files("orchestrator")))
        assert path.is_file()
        assert package_dir in path.parents, (
            f"{path} is outside the installed package {package_dir}; a wheel "
            f"cannot ship it"
        )

    def test_missing_resource_raises_instead_of_returning_nothing(self):
        with pytest.raises(FileNotFoundError):
            packaged_config_resource("definitely-not-a-config.yaml")
        with pytest.raises(FileNotFoundError):
            packaged_registry_resource("models", "definitely-not-a-registry.yaml")


class TestInstallDefaultConfigs:
    """`orchestrator-install-configs` must write real files."""

    def test_returns_none_so_the_console_script_exits_zero(self, fake_home):
        """console_scripts pass the return value to sys.exit()."""
        assert install_default_configs() is None

    def test_writes_real_files_matching_the_packaged_defaults(self, fake_home):
        install_default_configs()
        config_dir = user_config_dir()

        assert config_dir == fake_home / ".orchestrator"
        assert config_dir.is_dir()

        for filename in CONFIG_FILES:
            installed = config_dir / filename
            assert installed.is_file(), f"{filename} was not installed"
            assert installed.stat().st_size > 0, f"{filename} was installed empty"
            assert installed.read_bytes() == packaged_config_resource(
                filename
            ).read_bytes()
            assert isinstance(yaml.safe_load(installed.read_text()), dict)

        readme = config_dir / "README.md"
        assert readme.is_file() and readme.read_text().strip()

    def test_does_not_silently_no_op(self, fake_home):
        """A no-op install is the exact bug being guarded against."""
        install_default_configs()

        written = sorted(p.name for p in (fake_home / ".orchestrator").iterdir())
        assert written == sorted([*CONFIG_FILES, "README.md"]), (
            f"installer produced {written}"
        )

    def test_is_idempotent_and_preserves_user_edits(self, fake_home):
        install_default_configs()
        config_dir = user_config_dir()

        edited = config_dir / "models.yaml"
        edited.write_text("models: {}\n# user edit\n")
        before = edited.read_text()

        install_default_configs()

        assert edited.read_text() == before, "installer clobbered a user config"
        # Untouched file is still the packaged default.
        assert (config_dir / "orchestrator.yaml").read_bytes() == (
            packaged_config_resource("orchestrator.yaml").read_bytes()
        )

    def test_restores_a_deleted_file_on_rerun(self, fake_home):
        install_default_configs()
        config_dir = user_config_dir()
        (config_dir / "models.yaml").unlink()

        install_default_configs()

        assert (config_dir / "models.yaml").read_bytes() == (
            packaged_config_resource("models.yaml").read_bytes()
        )


class TestRegistryInstaller:
    """Default registries must be installed from packaged resources."""

    def test_install_writes_registries_from_the_package(self, tmp_path):
        installer = RegistryInstaller(home_dir=tmp_path / ".orchestrator")

        assert installer.install() is True
        assert installer.is_installed()

        skills_registry = installer.skills_dir / "registry.yaml"
        models_registry = installer.models_dir / "registry.yaml"

        assert skills_registry.read_bytes() == packaged_registry_resource(
            "skills", "default_registry.yaml"
        ).read_bytes()
        assert models_registry.read_bytes() == packaged_registry_resource(
            "models", "default_registry.yaml"
        ).read_bytes()

        assert all(installer.verify_installation().values())
        assert installer.get_skills_registry().get("version")
        assert installer.get_models_registry().get("version")

    def test_install_is_idempotent(self, tmp_path):
        installer = RegistryInstaller(home_dir=tmp_path / ".orchestrator")
        assert installer.install() is True

        (installer.skills_dir / "registry.yaml").write_text("version: user\nskills: {}\n")

        assert installer.install() is True
        assert yaml.safe_load(
            (installer.skills_dir / "registry.yaml").read_text()
        ) == {"version": "user", "skills": {}}

    def test_force_reinstall_restores_defaults(self, tmp_path):
        installer = RegistryInstaller(home_dir=tmp_path / ".orchestrator")
        assert installer.install() is True

        (installer.skills_dir / "registry.yaml").write_text("version: user\nskills: {}\n")

        assert installer.install(force=True) is True
        assert (installer.skills_dir / "registry.yaml").read_bytes() == (
            packaged_registry_resource("skills", "default_registry.yaml").read_bytes()
        )
