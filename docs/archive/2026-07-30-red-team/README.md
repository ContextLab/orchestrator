# Red-team reports, 2026-07-30

Two adversarial reviews of the `recovery/phase-0-4` branch, kept verbatim
because their value is partly in what they got *right about work that had
already claimed to be done*.

- `01-code-review.md` — attacked the new AST expression evaluator and the lazy
  import facade. Achieved real code execution through an `eval()` site the
  first migration pass missed, and found a cold-import regression introduced by
  the lazy facade.
- `02-claims-audit.md` — re-ran every measurable claim in the branch's commit
  messages, README and ADR. Most were reproduced; it found that failed steps
  exited 0, that runtime `pip install` was only partly gated, and that the
  ADR's canonical-implementation table overstated how separable the competing
  subsystems were.

Both were written against intermediate commits, so some findings ("the
evaluator is not wired into the canonical path") were already resolved by the
time they landed. The disposition of every finding is recorded in
`notes/2026-07-30_recovery_execution_session.md`.
