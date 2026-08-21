"""Contract tests for the provider retirement (#430).

The product's providers are Dartmouth Chat and the HuggingFace Inference API
(ADR 0001). The Anthropic / OpenAI / Google / Ollama / local-HuggingFace
adapters are retired: not importable, not exported, not offered by the
packaged default model pool, and never silently constructed as a fallback.

A ``models.yaml`` written before the retirement may still name those sources;
population must skip each such entry with a warning that names the source,
and must never raise -- an old config file is not an error.
"""

import logging
import os

import pytest
import yaml

pytestmark = pytest.mark.unit


_RETIRED_SOURCES = {"ollama", "openai", "anthropic", "google", "huggingface"}


def test_packaged_models_yaml_offers_no_retired_providers():
    """The shipped default model pool must not offer retired providers."""
    from orchestrator.install_configs import packaged_config_path

    path = packaged_config_path("models.yaml")
    assert path.exists(), f"packaged models.yaml missing at {path}"
    config = yaml.safe_load(path.read_text()) or {}
    models = config.get("models") or []
    assert isinstance(models, list)
    offered = {m.get("source") for m in models if isinstance(m, dict)}
    assert not (offered & _RETIRED_SOURCES), (
        f"packaged models.yaml still offers retired providers: "
        f"{sorted(offered & _RETIRED_SOURCES)}"
    )


def _seal_credentials(monkeypatch, tmp_path):
    """Cut every provider credential source, including import-time paths.

    The credential modules compute their credential-file paths from
    ``Path.home()`` at import time, so patching ``HOME`` afterwards changes
    nothing -- the module constants must be redirected instead.
    """
    from orchestrator.models import dartmouth_credentials, huggingface_credentials

    monkeypatch.delenv("DARTMOUTH_CHAT_API_KEY", raising=False)
    monkeypatch.setattr(
        dartmouth_credentials, "_ORCHESTRATOR_ENV_FILE", tmp_path / "nope.env"
    )
    monkeypatch.setattr(
        dartmouth_credentials, "_LLMXIVE_CREDENTIALS_FILE", tmp_path / "nope.toml"
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        huggingface_credentials, "_ORCHESTRATOR_ENV_FILE", tmp_path / "nope.env"
    )
    monkeypatch.setattr(
        huggingface_credentials, "_HF_CLI_TOKEN_FILE", tmp_path / "nope-token"
    )


def test_populate_skips_retired_sources_with_a_warning(tmp_path, monkeypatch, caplog):
    """A pre-retirement models.yaml is skipped entry-by-entry, never raised on."""
    from orchestrator._api import populate_model_registry
    from orchestrator.models.model_registry import ModelRegistry
    from orchestrator.utils import model_config_loader as loader_module

    config_file = tmp_path / "models.yaml"
    config_file.write_text(
        "models:\n"
        + "".join(
            f"  - source: {source}\n    name: some-model\n    size: 1b\n"
            for source in sorted(_RETIRED_SOURCES)
        )
    )
    loader = loader_module.ModelConfigLoader(config_path=config_file)
    monkeypatch.setattr(loader_module, "get_model_config_loader", lambda: loader)
    _seal_credentials(monkeypatch, tmp_path)

    registry = ModelRegistry()
    with caplog.at_level(logging.WARNING):
        populate_model_registry(registry)

    assert registry.list_models() == []
    warned = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    for source in sorted(_RETIRED_SOURCES):
        assert any(source in message for message in warned), (
            f"no warning named retired source {source!r}: {warned}"
        )


def test_populate_without_any_credentials_registers_nothing(tmp_path, monkeypatch):
    """No keys and an empty pool is a valid state, not an error."""
    from orchestrator._api import populate_model_registry
    from orchestrator.models.model_registry import ModelRegistry
    from orchestrator.utils import model_config_loader as loader_module

    config_file = tmp_path / "models.yaml"
    config_file.write_text("models: []\n")
    loader = loader_module.ModelConfigLoader(config_path=config_file)
    monkeypatch.setattr(loader_module, "get_model_config_loader", lambda: loader)
    _seal_credentials(monkeypatch, tmp_path)

    registry = ModelRegistry()
    populate_model_registry(registry)
    assert registry.list_models() == []


def test_hybrid_control_system_without_registry_never_constructs_a_provider():
    """The retired 'no registry? build a gpt-4 client' fallback must stay gone."""
    import asyncio

    from orchestrator.control_systems.hybrid_control_system import (
        HybridControlSystem,
    )
    from orchestrator.core.task import Task

    control_system = HybridControlSystem(model_registry=None)
    task = Task(id="t1", name="t1", action="analyze_text", parameters={"text": "x"})
    result = asyncio.run(control_system._handle_analyze_text(task, {}))
    assert result["success"] is False
    assert "No suitable model" in result["error"]


def test_public_surface_drops_retired_model_exports():
    """Retired adapters are not reachable from the public package surface."""
    import orchestrator
    from orchestrator.models import providers

    for name in ("HuggingFaceModel", "OllamaModel"):
        assert not hasattr(orchestrator, name), (
            f"orchestrator.{name} still resolves after retirement"
        )
    assert not hasattr(providers, "AnthropicProvider"), (
        "models.providers.AnthropicProvider still resolves after retirement"
    )
