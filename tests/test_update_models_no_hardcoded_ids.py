"""Model ids must come from provider APIs, never from a list in the source.

This repository has hardcoded Anthropic model ids three times, and all three
rotted the same way:

1. ids frozen at 2024 dates, which the API had since retired;
2. invented ``-latest`` aliases, none of which existed (every call 404'd);
3. ``update_models.fetch_anthropic_models`` returning claude-2, claude-2.1 and
   claude-instant-1.2, on the stated belief that "Anthropic doesn't have a
   models.list() endpoint" -- it does.

A list of model ids in source is a dated snapshot presented as fact. These
tests exist to make the fourth attempt fail loudly.
"""

import inspect
import re

import pytest

from orchestrator.tools import update_models as update_models_module

pytestmark = pytest.mark.unit

#: Matches a concrete, dated or versioned Claude model id in source text.
_CLAUDE_ID = re.compile(r"claude[-\w.]*?(\d{8}|-\d+\.\d+|-latest)", re.IGNORECASE)


def test_anthropic_fetcher_contains_no_hardcoded_model_ids():
    source = inspect.getsource(update_models_module.ModelUpdater.fetch_anthropic_models)
    # The docstring names the retired ids on purpose, as the explanation for
    # why this rule exists. Only the executable body is checked.
    body = source.split('"""')[-1]
    found = _CLAUDE_ID.findall(body)
    assert not found, (
        f"hardcoded Claude model ids are back in fetch_anthropic_models: "
        f"{found}. Read ids from client.models.list() instead -- a static "
        f"list here has rotted three times."
    )


def test_anthropic_fetcher_asks_the_api():
    source = inspect.getsource(update_models_module.ModelUpdater.fetch_anthropic_models)
    assert "models.list()" in source, (
        "fetch_anthropic_models must query the live listing endpoint"
    )


@pytest.mark.asyncio
async def test_anthropic_fetcher_degrades_to_empty_without_a_key(monkeypatch):
    """Skipping a provider is honest; inventing its catalogue is not."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    updater = update_models_module.ModelUpdater()
    assert await updater.fetch_anthropic_models() == []


@pytest.mark.asyncio
async def test_dartmouth_fetcher_degrades_to_empty_without_a_credential(monkeypatch, tmp_path):
    """Same contract for Dartmouth: no credential means no models, not guesses."""
    monkeypatch.delenv("DARTMOUTH_CHAT_API_KEY", raising=False)
    # Point the on-disk credential lookups somewhere empty so a developer's
    # real credential cannot make this test pass or fail by accident.
    monkeypatch.setattr(
        "orchestrator.models.dartmouth_credentials._ORCHESTRATOR_ENV_FILE",
        tmp_path / "absent.env",
    )
    monkeypatch.setattr(
        "orchestrator.models.dartmouth_credentials._LLMXIVE_CREDENTIALS_FILE",
        tmp_path / "absent.toml",
    )
    updater = update_models_module.ModelUpdater()
    assert await updater.fetch_dartmouth_models() == []


def test_dartmouth_fetcher_registers_only_free_models():
    """Writing paid ids into models.yaml would make them selectable by default."""
    source = inspect.getsource(update_models_module.ModelUpdater.fetch_dartmouth_models)
    assert "list_free_models()" in source
    assert "list_paid_models" not in source
