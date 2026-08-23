# sherpa measurement report

Raw projections from real runs; no assumed numbers.

- runs aggregated: 41
- task success rate: 100.0% (CI95 1.00..1.00)
- atomic overclaim rate: 0.0% (CI95 0.00..0.00)
- corrected m observed max: 0.000 — subcritical (<1) on this fixture distribution
- total tokens: 0

| run | status | admissions | overclaim | m_corrected | tokens |
|-|-|-|-|-|-|
| None | completed | 3 | 0.0% | 0.000 | 0 |
| None | completed | 3 | 0.0% | 0.000 | 0 |
| None | completed | 3 | 0.0% | 0.000 | 0 |
| None | completed | 3 | 0.0% | 0.000 | 0 |
| None | completed | 3 | 0.0% | 0.000 | 0 |
| None | completed | 3 | 0.0% | 0.000 | 0 |
| None | completed | 3 | 0.0% | 0.000 | 0 |
| None | completed | 3 | 0.0% | 0.000 | 0 |
| None | completed | 3 | 0.0% | 0.000 | 0 |
| None | completed | 3 | 0.0% | 0.000 | 0 |
| None | completed | 3 | 0.0% | 0.000 | 0 |
| None | completed | 3 | 0.0% | 0.000 | 0 |
| None | completed | 3 | 0.0% | 0.000 | 0 |
| None | completed | 3 | 0.0% | 0.000 | 0 |
| None | completed | 3 | 0.0% | 0.000 | 0 |
| None | completed | 3 | 0.0% | 0.000 | 0 |
| run_206745e2dd7c | completed | 2 | 0.0% | 0.000 | 0 |
| run_113d270608c9 | completed | 2 | 0.0% | 0.000 | 0 |
| run_89699176f9d8 | completed | 2 | 0.0% | 0.000 | 0 |
| run_49fa235b9063 | completed | 2 | 0.0% | 0.000 | 0 |
| run_1b4966e8e3b6 | completed | 2 | 0.0% | 0.000 | 0 |
| run_d42d5e352ea6 | completed | 2 | 0.0% | 0.000 | 0 |
| run_f335c01f39e7 | completed | 2 | 0.0% | 0.000 | 0 |
| run_203774aedfa3 | completed | 2 | 0.0% | 0.000 | 0 |
| run_b25b14b76761 | completed | 2 | 0.0% | 0.000 | 0 |
| run_be92bffa3ea3 | completed | 2 | 0.0% | 0.000 | 0 |
| run_d2d2cc5a2cc1 | completed | 2 | 0.0% | 0.000 | 0 |
| run_ce45a57a2cf9 | completed | 2 | 0.0% | 0.000 | 0 |
| run_458996392aca | completed | 2 | 0.0% | 0.000 | 0 |
| run_a8d0dbcc1acb | completed | 2 | 0.0% | 0.000 | 0 |
| run_31771659f2e0 | completed | 2 | 0.0% | 0.000 | 0 |
| run_7bf1721eac8c | completed | 2 | 0.0% | 0.000 | 0 |
| run_998d8c107fb8 | completed | 2 | 0.0% | 0.000 | 0 |
| run_ca1b78a49360 | completed | 2 | 0.0% | 0.000 | 0 |
| run_5efdf83d06a3 | completed | 2 | 0.0% | 0.000 | 0 |
| run_f5f9cfdbaf8f | completed | 2 | 0.0% | 0.000 | 0 |
| run_41d41325e12d | completed | 2 | 0.0% | 0.000 | 0 |
| run_53762ab1b8fb | completed | 2 | 0.0% | 0.000 | 0 |
| run_455791041766 | completed | 2 | 0.0% | 0.000 | 0 |
| run_a8bd76217158 | completed | 2 | 0.0% | 0.000 | 0 |
| run_3d496e00d1ca | completed | 2 | 0.0% | 0.000 | 0 |

## Preregistered gates

| gate | threshold | observed | verdict |
|-|-|-|-|
| decomposition decisions with admission outcomes | >= 50 | 66 | PASS |
| claimed-atomic steps admitted/rejected independently | >= 30 | 98 | PASS |
| held-out repair/corpus tasks externally verified within budgets | >= 80% | 1.0 | PASS |
| corrected m upper bound on fixture distribution | < 1.0 | 0.0 | PASS |
| seeded-needle retrieval recall | >= 95% | 1.0 | PASS |
| crash/resume preserves projections; no repeated effects | required | met | PASS |
| repair results verified by REAL pytest outside the runtime | 100% | 1.0 | PASS |

**GO**: all preregistered MVP gates met on this fixture distribution. Thresholds are MVP decisions, not product claims.
