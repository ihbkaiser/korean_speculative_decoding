# Experiment Tracker

Ngày: 2026-10-06. Trạng thái từ artifacts đang có; chưa chạy mới trong phiên review này.

| Run ID | Milestone | Purpose | Variant/data | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|
| R001 | M0 | Validate linguistic measurement | Two blind Korean annotators; accepted/rejected/ambiguous | Class/environment accuracy, agreement, exclusion sensitivity | MUST | TODO | Existing 200 rejected-only sheet unannotated; expand to 300–500 |
| R002 | M0 | Separate retrospective/online features | Target vs actual draft; strict-prefix/full-block | Context agreement, available feature audit | MUST | PARTIAL | Existing projection analysis; deployable covariates still need revision |
| R003 | M1 | Confirmation generation | Extra Wiki/P1 | Parity, actual proposal traces | MUST | COLLECTED | 1,000 rows; independent target parity audited on first 32 |
| R004 | M1 | Confirmation generation | Extra Wiki/P2 | Same | MUST | PARTIAL | 32/1,000 rows in saved trace |
| R005 | M1 | Confirmation generation | FLORES dev/P1 | Same | MUST | PARTIAL | 165/997 rows |
| R006 | M1 | Confirmation generation | FLORES dev/P2 | Same | MUST | PARTIAL | 32/997 rows |
| R007 | M1 | Replicate all environment effects | Actual P1/P2 proposals, discovery then confirmation | Raw/adjusted RD, support, interactions | MUST | PARTIAL | Existing report exports N→P; full environment profile required |
| R008 | M1 | Check comparable cases | Prespecified common-support estimand | Coverage, SMD, matched/weighted RD | MUST | INCONCLUSIVE | Old primary 44/29 pairs, poor balance; do not claim confirmed |
| R009 | M2 | Test information beyond confidence | Same predictor family: confidence/+F1/+direction/+environment | Held-out log-loss, Brier, AUPRC, risk–coverage | MUST | TODO | Source-grouped splits, no target entropy/future target structure |
| R010 | M2 | Generalization vs memorization | Cross-workload; held-out nominal/token identity | Same metrics with support | MUST | TODO | Hold same source across pair variants together |
| R011 | M3 | Diagnose disagreement content | Expert subset, same-prefix draft/target distributions | Confidence, rank/overlap, content labels | MUST FOR ORAL AMBITION | TODO | Do not infer particle-choice errors without annotation |
| R012 | M3 | Beyond Qwen | One second-family compatible draft–target pair | Boundary RD, online utility | MUST FOR BROADER CLAIM | TODO | Select after pilot and compute constraints |
| R013 | M3 | Test N→P suppression | Existing replacement intervention | Immediate acceptance delta | REUSE | NEGATIVE | Wiki pooled -67.87 pp; FLORES -46.52 pp; no rerun just to find positive |
| R014 | M4 | Cost/headroom gate | Profile + offline cost-aware oracle | Avoidable work, extra rounds, detector budget | CONDITIONAL MUST | TODO | Rarity and full-block detector may eliminate headroom |
| R015 | M4 | Standard acceleration baseline | Batched verify + efficient draft cache | Parity, stable logits, real latency | CONDITIONAL MUST | BLOCKING TECHNICAL GAP | Current batched prototype fails 1/8; sequential baseline inadequate for speed claim |
| R016 | M4 | Validate scheduling correctness | All policies × Wiki/FLORES × P1/P2/(P3) | Exact IDs and stopping | CONDITIONAL MUST | PARTIAL | Saved 1,700 correctness rows Wiki only, zero failures |
| R017 | M4 | Operational method pilot | Best fixed-K, entropy, entropy+morphology | Synchronized paired latency + overhead | CONDITIONAL MUST | TODO | >=3 repeats; typical workloads; random/all-boundary specificity controls |
| R018 | M4 | Full timing and robustness | Frozen held-out tests, K/context checks | Same + CI + repeat variability | CONDITIONAL | TODO | Only if correctness, headroom and pilot pass |

No numeric GPU budget for new-family or controller runs until a pilot provides measurements. Detailed protocol and negative-result interpretation: `EXPERIMENT_PLAN.md`. Evidence audit: `../RESEARCH_REVIEW.md`.
