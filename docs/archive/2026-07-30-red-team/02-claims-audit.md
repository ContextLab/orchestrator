# Red-team audit: claims on branch `recovery/phase-0-4`

Repo: /Users/jmanning/orchestrator @ HEAD 987d597 (5 commits ahead of main).
Date: 2026-07-30. Clean env used for hermetic checks:
`/private/tmp/.../scratchpad/cleanenv/bin/python` (Python 3.12.10) with exactly the
12 core deps + pytest tooling installed — matches what `ci.yml` installs (`pip install -e .`).

Verdict legend: SUPPORTED / OVERSTATED / FALSE. "Verified" = I re-ran it; "inferred" = read-only reasoning.

---

## Headline

Most concrete, measurable claims are **SUPPORTED** and were independently reproduced.
This branch is markedly more honest than the prior "completion narratives." Two
findings need flagging:

1. **The Phase-4 security claim does not protect the shipped code path (FALSE as
   worded).** `core/expressions.py` is real, adversarially tested, and fail-closed —
   but the canonical executor (`Orchestrator.execute_pipeline`) never calls it. Step
   conditions on the real path still run through `ControlFlowAutoResolver._safe_eval`
   → `eval()` on textually-substituted strings — the exact defect the new module's
   own docstring says it replaces. The only importer of `core/expressions.py` in the
   tree is `execution/engine.py`, which the ADR itself lists as an *unsupported /
   competing* module.

2. **"Passing CI" is achieved by scoping the blocking gate to 69 of 2860 tests.**
   This is disclosed (ADR test-layers + a job literally named "Legacy suite
   (non-blocking backlog)"), so it is defensible, not deceptive — but a reader of the
   README line "exercised by hermetic tests in CI on every commit" could over-read it.
   The broad selection the audit prompt gave me is the *non-blocking* legacy job, and
   it fails heavily (≥212 failed + ≥67 errored within the first 36%, then wedges on a
   hung async test).

---

## Claim-by-claim

### 1. "import orchestrator takes ~4ms and pulls in zero heavy dependencies" — SUPPORTED
Measured (clean env, `PYTHONPATH=src`, cold subprocess):
- `python -X importtime`: cumulative `orchestrator` self+children = **2.29–4.26 ms** across 10 runs.
- Module closure: `import orchestrator` adds **12 modules total**, top-level only
  `{collections, contextlib, importlib, typing, warnings, _typing, orchestrator}`.
- Heavy-dep probe: **none** of numpy/pydantic/aiohttp/jinja2/jsonschema/networkx/
  click/requests/psutil/yaml/anthropic/openai present in `sys.modules` after import.
"~4ms" is the honest upper end of the cumulative importtime; PEP 562 lazy facade is genuine.

### 2. "Core dependencies 40 -> 12" — SUPPORTED (exact)
- `git show main:pyproject.toml` `dependencies` block: **40** uncommented entries.
- HEAD `dependencies`: **12** — pydantic, pyyaml, jinja2, jsonschema, networkx, click,
  python-dotenv, aiofiles, aiohttp, requests, psutil, numpy. Matches ADR §Dependency policy.
- Dropped-from-core (psycopg2-binary, asyncio-mqtt, six, imageio-ffmpeg, lxml) confirmed absent from `src/` references per commit; extras (`anthropic/openai/google/langgraph/web/multimedia/viz/infra/crypto/dev/...`) exist in `[project.optional-dependencies]`.

### 3. "All correctness-class lint findings fixed: 45 F821 + 12 others -> 0" — SUPPORTED
`uv tool run --from ruff ruff check src/orchestrator --select F821,F823,F601,F811 --output-format concise`
→ **"All checks passed!"** (0 findings). The count "45+12" is historical and unverifiable
against main here, but the current-state claim (→ 0) is true.

### 4. "0 collection errors; 2860 tests collect" — SUPPORTED
`PYTHONPATH=src pytest --collect-only -q` → **2860 tests collected in ~1.7s, 0 errors**,
5 module-level SKIPs (multimedia extra, infra+docker, POML x2, web extra) — all guarding
genuinely-absent optional extras, not core deps.
- **Not neutered (subagent-verified):** 0 test files deleted; 0 assertions removed/
  weakened; no `assert True`/`xfail`/emptied bodies/swallowed exceptions. The bulk diff is
  the mechanical `src.orchestrator` → `orchestrator` rewrite (≈741 removed / 776 added
  import lines; only 26 of 229 changed test files have any non-import change, 13 of which
  are committed CSV/JSON artifacts under `tests/performance/results/`). All 11 added
  skips guard dependencies confirmed absent by import in both clean and dev envs.

### 5. "The CLI now actually executes pipelines" — SUPPORTED
Ran the installed console path from **outside** any repo context via
`python -m orchestrator.cli` in a scratch dir:
- `validate basic.yaml` → exit 0, prints resolved task graph ("✓ basic.yaml is valid", 2 tasks).
- `run basic.yaml -i greeting=redteam` → exit 0, wrote `golden_out/greeting.txt` = `redteam world`,
  printed typed JSON with `read_back.result.content == "redteam world"`, wrote a checkpoint.
The Phase-3 "Pipeline execution not yet integrated" stub is gone.
- Minor: even a tool-only run emits noisy `⚠️ Error registering <provider> model ... Failed
  to install` lines and (see finding A) *attempts a network pip install* per provider model.

### 6. "66/67 tests pass hermetically" (golden + expressions) — SUPPORTED
`pytest tests/test_golden_pipelines.py tests/test_expressions.py` → **67 passed in ~7.5s**
(clean env, no network/keys/docker). Current count is 67/67.
Tests are **meaningful, not tautological**:
- `test_expressions.py`: 18 adversarial escape cases (`__import__`, `open`, `__subclasses__`,
  `getattr`, comprehension/lambda/walrus/f-string, `2**10**10` DoS) each asserted to raise;
  fail-closed cases assert `evaluate_condition(...) is False`; a regression test pins the old
  textual-substitution bug (`max(a,10)` with `a=3` → 10). Would catch real regressions.
- `test_golden_pipelines.py`: asserts exact file contents, typed JSON payload fields,
  CLI==API agreement, exit codes (0/2), and that `{{`/`}}` never reach output. Real.
- **Gap vs ADR:** ADR §Golden pipelines requires the control-flow golden to include "a
  deliberately failing step to verify failure propagation and exit codes." `tests/golden/
  control_flow.yaml` has **no failing step**, and **no test asserts a step-failure → exit 1**.
  Exit-1 propagation is therefore unverified by the golden layer. (OVERSTATED sub-claim.)

### 7. README "verified by hermetic tests in CI on every commit" — SUPPORTED, with caveats
IMPORTANT: the audit prompt's command
`pytest -m "not live and not integration and not docker and not slow"` is **NOT the CI
blocking gate**. The committed `.github/workflows/ci.yml` (172 lines, == working tree,
verified via `git show HEAD`) splits tests:
- **BLOCKING** `test` job, py3.11/3.12/3.13 × ubuntu/macos:
  `pytest -m "(unit or contract or e2e) and not live and not integration and not docker"`.
  I reran this exact selection: **69 passed, 13 skipped, 2793 deselected, exit 0** in ~19s.
  → The gate genuinely passes. The 5 capabilities the README lists (compile, sequential/
  parallel exec + dep ordering, template interpolation, deterministic tools, `run`/`validate`
  + Python API) are covered by these tests (golden=`e2e`, expressions=`unit`). **SUPPORTED.**
- **NON-BLOCKING** `legacy-suite` job (`continue-on-error: true`, `|| true`,
  `timeout-minutes: 20`, `--timeout-method=signal --tb=no`): runs the broad selection the
  prompt gave me — the 2708 unmarked legacy tests. This is the one that fails.

My reruns of that **broad/legacy** selection (clean env):
- thread-method timeout: **wedged at ~36%** on a hung async test; partial tally to that
  point = **636 passed, 212 failed, 67 errored, 103 skipped** (then pytest-timeout dumped a
  Timeout traceback and aborted with no summary — reproduced twice). CI dodges the wedge by
  using `--timeout-method=signal` + a 20-min job cap, and swallows the result with `|| true`.
- **Full legacy tally with the exact CI flags** (`--timeout=60 --timeout-method=signal
  --tb=no -rN`, 10m11s): **1921 passed, 539 failed, 110 errored, 151 skipped, 154
  deselected**. So the non-blocking backlog = **649 non-passing tests** (~25% of the ~2610
  it actually ran). This is real and large, but by design cannot fail the build.

So: the README/CI claim is **true as scoped** and the scoping is openly labeled ("Legacy
suite (non-blocking backlog)", ADR test-layer table). It is **not FALSE**. The honest caveat
is that "hermetic tests in CI" = **69 blocking of 2860**; ~2708 run non-blocking and many
fail. Calling the project **alpha** (README + ADR) is consistent with this.

NOTE (transient inconsistency, flagged for honesty): my *first* Read of `ci.yml` returned a
different 133-line version whose BLOCKING step used the broad `not live and not integration
and not docker and not slow` selection with no legacy job. Every subsequent check —
`git show HEAD:.github/workflows/ci.yml`, `wc -l`, `git diff HEAD` (empty) — shows the
172-line marker-gated version. I treat HEAD/on-disk as authoritative. If a broad-blocking
variant were ever the committed gate, that gate would be RED (per the tallies above); the
committed HEAD gate is green.

### 8. ADR "Canonical implementations" table — MIXED
All 8 named canonical files/symbols **exist**. Failures are all in column 3 ("competing /
not supported"), where several listed modules are in fact imported by the canonical path
(closure traced from `import orchestrator` + `Orchestrator`/`YAMLCompiler` + CLI):

| Role | Verdict | Evidence |
|-|-|-|
| Compiler (`compiler/yaml_compiler.py::YAMLCompiler`) | SUPPORTED | exists; competitors `enhanced_yaml_compiler.py`, `graph_generation/` not in closure |
| Control-flow compiler | SUPPORTED | `compiler/control_flow_compiler.py` exists, used |
| Domain model (`core/pipeline.py`,`core/task.py`) | SUPPORTED | exist, canonical |
| Tool registry (`tools/base.py::ToolRegistry`) | SUPPORTED | competitors not on closure |
| Executor (`orchestrator.py::Orchestrator`) | OVERSTATED | `executor/` and `runtime/` are listed as *competing* but are imported by `orchestrator.py:21` (`executor.parallel_executor`) and `:28`/`:291` (`runtime`). Only `engine/`,`execution/`,`api/` are truly off-closure. |
| Model registry (`models/model_registry.py`) | OVERSTATED | `models/__init__.py:16` unconditionally `from .registry import ModelRegistry` — the "competing" unified registry loads on every import; two same-named `ModelRegistry` classes coexist |
| State (`state/state_manager.py`) | OVERSTATED | `orchestrator.py:25` imports `LangGraphGlobalContextManager` at module scope; only *instantiation* is opt-in |
| Expression evaluation (`core/expressions.py`) | **FALSE** | see finding A below |

---

## Finding A (most important): the fail-closed expression language is not on the product path

Commit fa07cd6 / README bullet "🔒 Fail-closed conditions … never `eval()`".

- `core/expressions.py` exists, is AST-allowlist based, fail-closed, and has 55/adversarial
  tests that pass (claim #6). In isolation it is correct.
- **But its only importer in the entire tree is `execution/engine.py:460`** — and the ADR's
  own table lists `execution/` as a *competing, unsupported* implementation. `execution/
  engine.py`'s sole importer is `api/execution.py`, itself listed as competing. So the new,
  safe evaluator is reachable only through the unsupported subtree.
- The **canonical** condition path, traced live:
  `Orchestrator._execute_task_with_resources` (orchestrator.py:1721-1768) →
  `ConditionalTask.should_execute` → `ControlFlowAutoResolver` →
  **`control_flow/auto_resolver.py:747  result = eval(code, {"__builtins__": {}}, context)`**,
  after regex/textual rewrites (`auto_resolver.py:711`+). This is the same class of textual
  substitution the new module's docstring cites as the bug it fixes.
- **≥4 `eval()` sites remain reachable from pipeline content:** `auto_resolver.py:747`,
  `runtime/dependency_resolver.py:281` (via `runtime/loop_expander.py` for `while:`),
  `control_flow/enhanced_condition_evaluator.py:297`, `control_systems/hybrid_control_system.py:722`.
- Live probe (real CLI run, ORCHESTRATOR_AUTO_INSTALL=0) of a step with
  `condition: "().__class__.__bases__[0].__subclasses__()[0]...__import__('os').system('touch /tmp/PWNED_REDTEAM')==0"`:
  the guarded step was **skipped** and no side-effect file was created — BUT the log shows it
  failed closed via `auto_resolver`'s generic exception handler
  (`{"__builtins__": {}}` made the specific gadget raise `AttributeError`), **not** via
  `core/expressions.py`. So today's canonical path fails closed for *this* gadget by luck of an
  empty-builtins `eval`, not by the audited allowlist. The security posture the commit/README
  advertise is real for `core/expressions.py` and **not wired into the shipped executor**.

Net: README's fail-closed bullet is **OVERSTATED**; the ADR "Expression evaluation" canonical
row is **FALSE** as written (migration described as done; it touched one non-pipeline-facing
call site).

---

## Finding B: runtime pip-install on an ordinary tool-only run (contradicts Phase-1/4 spirit)

Even a deterministic, no-model `orchestrator run` triggers, per registered provider model,
`anthropic_model.py:160-162` / `openai_model.py:271-273` / `google_model.py:165-167`:
`print("<X> library not found. Installing..."); subprocess.check_call([sys.executable,"-m","pip","install", ...])`.
These call sites are **NOT gated by `ORCHESTRATOR_AUTO_INSTALL`** (only `utils/auto_install.py`
is). Observed live: repeated `pip install anthropic/openai/google-generativeai` attempts during
a golden-basic run (they failed only because the clean env has no network/build). The commit
claim "Runtime package installation is now opt-in" is therefore **OVERSTATED** — `utils/
auto_install.py` was gated, but these three provider adapters still shell out to pip on any run
that registers their models. CI is protected only because it sets `ORCHESTRATOR_AUTO_INSTALL=0`
(which these sites ignore) AND has no models registered in the hermetic gate.

---

## Other verified cleanup claims (Phase 7) — SUPPORTED
- `.ccpm_backup`: **1388** files on main → **0** at HEAD.
- Tracked files: main **3362** → HEAD **1989** (commit says "1975"; off by 14, −40.8% vs claimed −41% — negligible OVERSTATE).
- 4 duplicate modules (auto_debugger_backup/_clean, anthropic/openai_model_original) removed: confirmed 1→0 each.
- Workflows: removed coverage/organization-validation/pipeline-tests/test-models/tests/wrapper-validation (6); added ci.yml, live-tests.yml. `claude.yml` untouched. Confirmed.

## Loose ends worth fixing (not claims)
- 13 committed test-artifact files under `tests/performance/results/` (timestamped CSV/JSON) should be gitignored.
- Golden control-flow lacks the ADR-required deliberately-failing step (exit-1 propagation unverified).
- Legacy/non-blocking backlog is 649 non-passing tests (539 failed + 110 errored of ~2610 run) — visible but unaddressed.
