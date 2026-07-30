# Archived refactor and status documents (2025)

These files are kept as a **historical record only**. They are not current
documentation and several of them are contradicted by the code they describe.

They were moved out of the repository root on 2026-07-30 because they were
being read as current status. Specifically:

- `CLAUDE_SKILLS_REFACTOR_COMPLETE.md` declares the refactor complete and
  production-ready, while `IMPLEMENTATION_STATUS.md` in the same set says CI
  was still being debugged and awaiting confirmation. No later successful
  confirmation exists in the history.
- Several describe an Anthropic-only architecture while the then-current README
  advertised multi-provider support.
- The completion claims predate any evidence that the advertised execution path
  worked; `orchestrator run` was in fact a stub that printed
  "Pipeline execution not yet integrated into CLI".

For what the project actually supports today, and the rules that govern how
anything gets promoted from "present in the tree" to "supported", see:

- [`docs/adr/0001-product-contract.md`](../../adr/0001-product-contract.md) — the product and architecture contract
- [`README.md`](../../../README.md) — current status, verified surface, quickstart
- [`notes/2026-07-30_repository_recovery_audit_and_plan.md`](../../../notes/2026-07-30_repository_recovery_audit_and_plan.md) — the audit that prompted the recovery

Nothing here should be cited as evidence that a feature works.
