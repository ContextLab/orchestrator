# sherpa measurement report

Raw projections from real runs; no assumed numbers.

- runs aggregated: 41
- task success rate: 100.0% (CI95 1.00..1.00)
- atomic overclaim rate: 0.0% (CI95 0.00..0.00)
- corrected m observed max: 0.000 — subcritical (<1) on this fixture distribution
- total tokens: n/a

| run | status | admissions | overclaim | m_corrected | tokens |
|-|-|-|-|-|-|
| None | completed | 3 | 0.0% | 0.000 | n/a |
| None | completed | 3 | 0.0% | 0.000 | n/a |
| None | completed | 3 | 0.0% | 0.000 | n/a |
| None | completed | 3 | 0.0% | 0.000 | n/a |
| None | completed | 3 | 0.0% | 0.000 | n/a |
| None | completed | 3 | 0.0% | 0.000 | n/a |
| None | completed | 3 | 0.0% | 0.000 | n/a |
| None | completed | 3 | 0.0% | 0.000 | n/a |
| None | completed | 3 | 0.0% | 0.000 | n/a |
| None | completed | 3 | 0.0% | 0.000 | n/a |
| None | completed | 3 | 0.0% | 0.000 | n/a |
| None | completed | 3 | 0.0% | 0.000 | n/a |
| None | completed | 3 | 0.0% | 0.000 | n/a |
| None | completed | 3 | 0.0% | 0.000 | n/a |
| None | completed | 3 | 0.0% | 0.000 | n/a |
| None | completed | 3 | 0.0% | 0.000 | n/a |
| run_a09af9cf0eee | completed | 2 | 0.0% | 0.000 | n/a |
| run_edc1e7d2102c | completed | 2 | 0.0% | 0.000 | n/a |
| run_32de57ad56a2 | completed | 2 | 0.0% | 0.000 | n/a |
| run_bf9488f845a3 | completed | 2 | 0.0% | 0.000 | n/a |
| run_0519cf61c7ee | completed | 2 | 0.0% | 0.000 | n/a |
| run_51d396009023 | completed | 2 | 0.0% | 0.000 | n/a |
| run_871bb348bd66 | completed | 2 | 0.0% | 0.000 | n/a |
| run_629e2f725d07 | completed | 2 | 0.0% | 0.000 | n/a |
| run_51ea60434c29 | completed | 2 | 0.0% | 0.000 | n/a |
| run_f8bb3c597fce | completed | 2 | 0.0% | 0.000 | n/a |
| run_b2160b2418f0 | completed | 2 | 0.0% | 0.000 | n/a |
| run_4c3a7a18578a | completed | 2 | 0.0% | 0.000 | n/a |
| run_a12f5ccbdb6f | completed | 2 | 0.0% | 0.000 | n/a |
| run_96e5eb892c90 | completed | 2 | 0.0% | 0.000 | n/a |
| run_2465e3b4cf65 | completed | 2 | 0.0% | 0.000 | n/a |
| run_521bb82d6540 | completed | 2 | 0.0% | 0.000 | n/a |
| run_2397b2c9c103 | completed | 2 | 0.0% | 0.000 | n/a |
| run_a0cfd5717809 | completed | 2 | 0.0% | 0.000 | n/a |
| run_debcf2d81d95 | completed | 2 | 0.0% | 0.000 | n/a |
| run_b1e570f38964 | completed | 2 | 0.0% | 0.000 | n/a |
| run_6420253017fc | completed | 2 | 0.0% | 0.000 | n/a |
| run_e5371bf55366 | completed | 2 | 0.0% | 0.000 | n/a |
| run_c19fb73f7b5e | completed | 2 | 0.0% | 0.000 | n/a |
| run_4853cd58b8d0 | completed | 2 | 0.0% | 0.000 | n/a |
| run_e213ce415187 | completed | 2 | 0.0% | 0.000 | n/a |

## Preregistered gates

| gate | threshold | observed | verdict |
|-|-|-|-|
| decomposition decisions with admission outcomes | >= 50 | 107 | PASS |
| claimed-atomic steps admitted/rejected independently | >= 30 | 98 | PASS |
| held-out repair/corpus tasks externally verified within budgets | >= 80% | 1.0 | PASS |
| corrected m upper bound on fixture distribution | < 1.0 | 0.0 | PASS |
| seeded-needle retrieval recall | >= 95% | 1.0 | PASS |
| crash/resume preserves projections; no repeated effects | required | met | PASS |
| repair results verified by REAL pytest outside the runtime | 100% | 1.0 | PASS |
| seeded defect classes named by the planner from the sources | 100% | 1.0 | PASS |

**GO**: all preregistered MVP gates met on this fixture distribution. Thresholds are MVP decisions, not product claims.
