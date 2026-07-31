# Contributing

Thanks for your interest in the project. It is **alpha**: a small core is
verified and the wider legacy surface is not. Read [the product
contract](docs/adr/0001-product-contract.md) before making architectural
changes — it names which implementation is canonical and which are frozen.

## Setup

```bash
git clone https://github.com/ContextLab/orchestrator.git
cd orchestrator
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Optional capabilities live behind extras (`anthropic`, `web`, `multimedia`,
`viz`, `infra`, `crypto`, `langgraph`). The core install deliberately pulls in
only 12 dependencies — please keep it that way. If your change needs a new
third-party package, put it in an extra and import it lazily.

Verify the install:

```bash
python -c "import orchestrator; print(orchestrator.__version__)"
pytest -m "unit or contract or e2e" -q      # the blocking gate
```

## Test layers

Tests are separated by marker, and the separation is load-bearing — CI gates
on some layers and not others.

| Marker | Meaning | Runs by default | Gates CI |
|-|-|-|-|
| `unit` | Hermetic. No network, no services, no API keys. | yes | **yes** |
| `contract` | Verifies a published interface. Hermetic. | yes | **yes** |
| `e2e` | Full pipeline through the installed CLI/API. | yes | **yes** |
| `integration` | Needs local services (Docker/Redis/Postgres). | no | no |
| `live` | Calls a real provider API and **costs money**. | no | no |

Opt in to the gated-off layers explicitly:

```bash
ORCHESTRATOR_RUN_INTEGRATION=1 pytest -m integration
ANTHROPIC_API_KEY=... pytest -m live            # costs money
DARTMOUTH_CHAT_API_KEY=... pytest -m live -k dartmouth   # costs nothing
```

The Dartmouth live tests are **free** — they only use models the live catalog
reports at zero cost per token, and assert that before making a request. If
you have a Dartmouth Chat account, run them: they are the cheapest real-model
coverage available, and they exercise the same provider contract.

Two things about them are easy to get wrong:

- **Free model endpoints flap.** Each is served from its own cluster endpoint
  and they go down independently (`Cannot connect to host
  vllm-qwen35...`). A live test must **skip** on `ModelUnavailable`, not fail
  — that is an upstream outage, not a defect. Prefer
  `provider.generate_free()`, which falls through the free set.
- **Several free models are reasoning models.** They spend tokens on
  `reasoning_content` before emitting any `content`, so a small `max_tokens`
  yields `content: null` — a truncation, not an empty answer. That raises
  `ReasoningTruncated` rather than returning `""`.

Tests requiring absent prerequisites **skip with a reason naming what is
missing**. They must never silently pass. If you add a test that needs
something optional, gate it the same way (see `tests/conftest.py`).

New tests should carry a marker. `--strict-markers` is on, so an invented
marker is an error rather than a silent no-op.

## Testing rules

These are firm, and reviewers will hold you to them:

- **No mock objects. Ever.** Not for external APIs, not as a fallback. If real
  functionality cannot be exercised, the test must fail or skip with a clear
  reason — never pass against a simulation. Write real files to `tmp_path`,
  make real connections, run real models.
- **No cheater tests.** A test that cannot fail is worse than no test. Design
  each one to reveal a specific defect.
- **Tests are debugging tools.** Make them verbose. When one fails, its output
  should be enough to diagnose the problem without re-running it.
- **Never weaken a test to make it pass.** If a test fails, fix the code. If
  the test itself is wrong, say so explicitly in the commit message and
  explain why — do not quietly relax an assertion.

## Security rules

The project has had two confirmed remote-code-execution defects, both from the
same root cause: treating pipeline content as trusted code. Pipeline content
is **data**.

- **Never call `eval()` or `exec()` on pipeline content.** Use
  `orchestrator.core.expressions.evaluate_expression` /
  `evaluate_condition`, which allowlist over the parsed AST.
- **Restricting `__builtins__` is not a sandbox.** Any function object in
  scope carries `__globals__`, a live reference to the real builtins:
  `json.loads.__globals__["__builtins__"]["__import__"]` defeats it entirely.
  This is not theoretical — it was exploitable here.
- **Guards fail closed.** A condition that is malformed, unsupported, or
  undefined must *not* run the step it gates.
- **Never interpolate caller-supplied strings into generated code or shell
  commands.** Validate, then pass an argument vector.
- **Widening the expression evaluator requires adversarial review.** Every
  capability added to it has introduced a defect on the first attempt —
  including a memory-amplification DoS from methods that were individually
  safe. Add attack tests alongside the capability.

## Making changes

- Fix root causes. Do not paper over a failure with a special case, and do not
  simplify production code to make a test pass.
- No partial implementations, no dead code, no duplicated logic — search for
  an existing helper before writing a new one.
- Match the surrounding code's naming and structure.
- Update the docs in the same change. If you change a test or example, update
  what documents it.
- Check for credentials and personal information before committing.

## Commits and pull requests

Commit messages should explain **why**, not just what. If a change alters
user-visible behaviour, say so explicitly — for example, a guard that used to
fail open and now fails closed will stop firing for some existing pipelines,
and that belongs in the message.

Reference issues as `Issue #123: description` where applicable.

Before opening a PR, run the full local gate:

```bash
pytest -m "unit or contract or e2e" -q
ruff check src/orchestrator
python -m build && twine check dist/*
```

CI runs lint, the blocking test matrix across Python 3.11–3.13 on Ubuntu and
macOS, a wheel smoke test, and the legacy suite as a **non-blocking** job. The
legacy job is expected to be red; it exists to keep the backlog visible rather
than hidden. Do not "fix" it by deleting or skipping tests — see
[#354](https://github.com/ContextLab/orchestrator/issues/354).

## Reporting bugs

Include the pipeline YAML that reproduces it, the exact command, the full
output, and your OS/Python version. A reproduction that fits in one file is
worth more than a long description.
