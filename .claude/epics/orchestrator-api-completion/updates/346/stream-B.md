---
issue: 346
stream: Experts Field & Tool-Model Assignment
agent: parallel-worker
started: 2025-09-02T12:45:15Z
status: in_progress
---

# Stream B: Experts Field & Tool-Model Assignment

## Scope
Implement experts field in pipeline specifications enabling tool-specific model assignments. Create expert assignment engine that maps tools to optimal models based on specialization and requirements.

## Files
- `src/orchestrator/tools/experts.py` (new)
- `src/orchestrator/models/assignment.py` (new)
- `src/orchestrator/api/types.py` (update)
- `tests/tools/test_experts.py` (new)

## Progress
- Starting implementation