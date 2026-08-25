# ADR 0003: sherpa — an evidence-driven recursive agentic runtime (MVP)

- Status: Accepted (experimental)
- Date: 2026-08-23
- Issue: [#492](https://github.com/ContextLab/orchestrator/issues/492) (synthesizes #485 discussion, #486 simulations)
- Supersedes: nothing; runs alongside [ADR 0001](0001-product-contract.md)

## Context

Issue #492 asks whether Orchestrator can, for a goal not directly solvable:
recursively turn it into a bounded typed plan; prove claimed leaf operations are
actually executable with available capabilities; coordinate logical workers
through durable state and messages; review plans and results independently
against frozen criteria; recover from interruption without losing lineage or
repeating completed work; and manage more source material than one model context
while keeping every claim traceable — and answer with **measured executions**,
not attractive planning transcripts.

Three design reviews on #485 plus the checked-in simulations (#486) and the
`scripts/prototypes/minikernel/` prototype established six findings this MVP
encodes directly, except where noted:

1. The solution cache must store **negative** results, or learning cannot start
   in exactly the regime where it matters. **Not yet implemented.** The store
   supports negative entries and refuses budget-death entries, but the kernel
   only ever writes `{"status_class": "solved", ...}`; no failure is cached
   today. Recorded here as a known gap rather than a delivered property.
2. An escalating sibling must not cancel siblings that could still run.
3. Wildcard signatures defeat two-key retrieval; untyped solutions are never
   published.
4. Budget exhaustion must **not** be recorded as evidence that a plan is bad.
5. A planner must be a function of its inputs for any cache over it to mean
   anything.
6. `atomic` must be a **checked claim**, not a planner label (35–44% overclaim
   was measured).

## Decision

We add a new experimental package `src/sherpa/` beside the frozen supported
`orchestrator/` path. Nothing under `src/orchestrator/` imports sherpa and vice
versa except one permitted boundary: `sherpa.channel.LiveChannel` may lazily
import orchestrator's supported providers (Dartmouth Chat, HuggingFace Inference
API). Supported examples stay green; sherpa ships its own hermetic suite.

### Identities (kept separate; immutable/versioned where noted)

| Identity | Representation | Mutability |
|-|-|-|
| Problem specification | `ir.ProblemSpec` (goal, inputs, output schema, acceptance checks, budgets, authority) | immutable once a run starts |
| Plan definition/version | `ir.Plan` (`id`, integer `version`) | append new versions, never edit |
| Run / node run / attempt / session | rows + events keyed by `run_id`, `(run_id,node_key)`, attempt seq, session id | event-sourced state machine |
| Artifact/evidence | sha256-addressed blobs + spans | immutable |
| Capability/version | `capabilities.CapabilitySpec` | versioned by name@version |

A logical node is not an LLM instance: worker sessions are leased per attempt,
and one session may execute many nodes while a node may see several sessions
across attempts/resumes.

### Event semantics

One append-only SQLite table (`events`) is the source of truth. Event kinds are
a closed vocabulary (`events.EVENT_KINDS`); payloads are JSON; each event may
cite its causal predecessor (`causal_seq`), which is persisted in the `events`
table and round-trips across a reopen
(`tests/sherpa/test_store.py::TestEventLogIntegrity`). Large artifacts never enter the log; they live in a
content-addressed blob store (`store.BlobStore`) and events carry hashes.

The *run* projection is maintained transactionally alongside appends and is
independently rebuildable by replay: `Store.replay_projection` reconstructs run
status and error, `parent_run_id`, every node (state, `owner_session`, `depth`,
`parent_key`), usage totals, the pending-message count, and the run's finding
ids, and the reconstruction is compared field-for-field against the live
projection in the demos and the durability tests. The *document* substrate is
not replay-rebuildable: `chunks_meta`/`chunks_fts`, `summaries`, and
`solution_cache` are written directly and are not reconstructed from the log,
even though `chunk_indexed` and `summary_created` events are recorded. Leases
and messages are recovered as counts and states, not as full row sets.

Terminal states are loud: `completed`, `failed`, `blocked`, `escalated`,
`cancelled`, `budget_exhausted`. There is no silently-partial success.

### Plan IR

Typed Pydantic v2 models are the semantic contract; YAML is an optional
import/export surface only. Node variants: `InvokeCapability`,
`InvokePlan`, `Decompose`, `Branch`, bounded `While`, `Parallel`,
`AskUser`, `Return`, `Fail`. Control flow is structural — there is no `goto`
to compile. `ir.validate_plan` is *structural only*: it rejects duplicate node
ids, an else-case that is not last in a `Branch`, a `While` with no guard, an
empty `Parallel` branch, declared fan-out over caps, estimated depth over
budget, and unregistered capability names. It performs **no** authority
check. Authority containment is enforced elsewhere — see below. Guard/branch
expressions use a fail-closed AST-whitelist evaluator (`sherpa.expr`); there is
no `eval()` anywhere in the package.

### Budgets and authority

Every plan carries `Budgets` (nodes, attempts/node, depth, fan-out, tokens,
cost, wall seconds). Usage is checkpointed into the event stream; enforcement
happens at checkpoints; exhaustion yields terminal `budget_exhausted`. The cache
rule is *negative*, not affirmative: nothing at all is written to the solution
cache on a budget death, and `Store.cache_put` raises rather than accept an
entry marked `status_class="failed"` with `inconclusive=True`. So caches cannot
learn "this plan is bad" from a budget death — but no `inconclusive` entry is
recorded either.

`sherpa.authority` is the single implementation of every authority question;
`ir.Authority.allows`, `capabilities.assert_authority`, and
`admission.assert_authority_granted` all delegate to it. Grants are POSIX-style
patterns: `src`/`src/` a prefix subtree, `src/*` one segment, `src/**` any
depth, `src/a.txt` one file, `**` everything beneath the *workspace*, and `/**`
the entire filesystem — `/**` being the only way to ask for it. Relative grants are anchored
at the run's workspace and paths are fully resolved (`..`, absolute paths,
symlinks) *before* the comparison, so escapes are visible to the check rather
than hidden by it. Empty grants deny.

Two questions are kept apart: **delegation** (`authority_covers`, over pattern
sets) and **access** (`path_within_grants`, over one resolved path).
Authority flows from the problem specification to the root plan and can only
narrow on delegation: `kernel._author_child` refuses an authored child plan
whose `Authority` is not covered by the parent's. `admission` re-checks before a
leaf runs, and `capabilities.run_capability` performs the per-resource check on
every invocation. `CapabilitySpec.requires` names authority *dimensions*, not
patterns: the coarse gate requires the dimension be granted something, and the
per-resource check happens per invocation, so scoped grants are usable.

For MVP runs are attended-at-root only. Depth is now tracked through the
execution stack (it was previously a local pinned at 0), so the guard means what
it says: `AskUser` at depth 0 in an attended run pauses the run into `blocked`
awaiting an enqueued answer, and `AskUser` anywhere below the root — or in an
unattended run — terminates the run `escalated` with
`ask_user outside attended root: {node_key}`. Children can never widen
authority.

### Recursive decomposition with admission control

Unknown operations become `Decompose` nodes executed by a planner that must be
deterministic given its inputs (StubPlanner rules or LLM-authored IR validated
against the same schema). Before any `InvokeCapability` leaf executes, an
independent `AdmissionChecker` verifies (a) the named capability exists, (b)
resolved inputs type-check against its declared input schema, (c) required
authority is granted, and (d) **executable evidence** exists that the capability
can satisfy this step (`Capability.probe` runs now; its bytes are hashed into
the blob store). Rejected atomic claims are reclassified for decomposition or
escalated — never silently run. Declared vs corrected fan-out, ambiguity, and
admission outcomes are logged as events so corrected branching
`m = b·f` and overclaim rate are measured per run, not assumed. Corrected
branching is derived from `admission_checked` events; a ratio whose denominator
is zero is reported as `None`, never `0.0`, so an unmeasurable metric is
distinguishable from a measured zero.

`capabilities.run_capability` is the only invocation path. It validates resolved
inputs against the capability's declared `input_schema` **and** the returned
value against its `output_schema`, raising `CapabilityContractError` on either
violation rather than propagating an undeclared value. Capability inputs are
redacted before they reach the event log. Crash injection is opt-in per engine:
the SIGKILL hook fires only for `Engine(..., fault_injection=True)`, so
`SHERPA_KILL_AFTER_EVENTS` alone can no longer kill a run.

### Bounded independent review

Review applies at MVP to generated plans and final outputs. The reviewer session
is derived deterministically as `reviewer::{author_session}` and asserted
distinct from the author's by `Reviewer.ensure_separate`
(`SeparationOfDutyError` otherwise). Separation is falsifiable rather than
merely declared: because `RecordedChannel` is keyed by session,
`tests/sherpa/test_review_metrics.py::TestSeparation` stocks a response under
the *author's* key alone and asserts the review completes 0 rounds with
`escalated_review_incomplete`, stocks one under `reviewer::a` and asserts it
completes 1 round, and asserts the emitted `review_round` event carries
`reviewer_session == "reviewer::a"`.

Concerns come from a ledger frozen at review start — the problem's acceptance
checks plus fixed generic concerns — hashed by `ledger_sha` and recorded on the
report. The ledger is genuinely injected, not just hashed: `ledger_prompt`
renders the concerns and the hash verbatim into the reviewer's prompt for every
round, and it gates adjudication — a finding whose criterion is outside
`ledger_criteria` is downgraded to non-blocking with a rationale naming the
drift, so an off-ledger concern cannot block a plan. A finding blocks
only with a criterion plus reproducible evidence; unevidenced reviewer concerns
are downgraded to non-blocking residual risks. Findings have stable identity and
dispositions `fixed | accepted_risk | invalid | deferred | superseded`. Rounds,
tokens, and time are capped; unresolved blocking findings escalate the run. A
pass verdict is `pass_with_risk` — review never claims simply "clean".
Reviewer adjudications persist as findings/events so false-positive rate and
defect recall can be measured later.

### Context guarantee

The substrate stores concise operational journal entries (`intent`, `decision`,
`observation`, `assumption`, `blocker`, `result`) — not required private chain
of thought. There is no global scratchpad mutex: journal writes are appends;
compare-and-swap protects only node/run state transitions. Documents become
immutable structural chunks (exact char spans, content hashes) indexed in
SQLite FTS5; optional summary layers form a DAG where every summary points to
**all** children with exact source spans/hashes. The guarantee is bounded
overview + lossless source addressability + on-demand retrieval — not lossless
compression into a context window. Retrieval is FTS-first; embeddings are out of
scope until seeded-needle measurement shows they are needed.

### Model boundary

`ModelChannel` abstracts model access: `RecordedChannel` replays recorded
responses deterministically (the issue-sanctioned mechanism for external
boundaries), `LiveChannel` lazily uses orchestrator's supported providers when
credentials exist, `EchoChannel` refuses use so deterministic demos fail loudly
if they unexpectedly need a model. Deterministic local capabilities (real
filesystem ops, real pytest subprocesses, real patch application) keep scenarios
hermetic: the acceptance suite runs with no network and no API keys.

## Explicit non-goals (MVP)

Distributed workers/messaging/backpressure (#487); self-authored capability
lifecycle and sandbox enforcement (#488); governed cross-run memory and reuse
(#489); calibrated risk-tiered review beyond plan/final gates (#490);
unattended side-effecting production runs, approval UX, compensation, quotas,
SLOs, DR (#491). Bit-reproducible model calls (lineage is required,
bit-reproducibility is not).

## Consequences

- The blocking CI gate stays hermetic; sherpa adds its own marked tests under
  `tests/sherpa/`.
- Benchmarks (durable-semantics fixture crash/resume; repository repair;
  oversized-corpus synthesis) produce the preregistered measurements and the
  go/no-go report; thresholds are MVP decisions, not product claims.
- If a gate fails, we retain negative evidence and name the failed assumption
  rather than widen scope until the demo passes.
- What the current fixture distribution does **not** establish is written down
  in [docs/sherpa.md](../sherpa.md#limitations) against the regenerated
  `benchmarks/artifacts/suite.json`: all 41 `m_values` are `0.0` on a
  distribution with no ambiguous children, so the branching gate is bounded
  trivially; `total_tokens` is `null` with `runs_with_token_measurements: 0`
  because hermetic runs consult no model; the deepest persisted node `depth`
  across all 41 runs is 1; #485's organization tree is only partially realized
  (linkage and depth persist, but there is no per-node agent identity and
  delivery is exact-key, addressed by the caller); and #485's shared
  scratchpad / insights pool / shared tool pool are not implemented.
