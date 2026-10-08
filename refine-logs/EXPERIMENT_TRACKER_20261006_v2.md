# Concrete experiment-card tracker v2

Date: 2026-10-06. Detailed specifications in `NAACL_EXPERIMENT_BLUEPRINT.md`; machine-readable design in `../configs/paper_experiment_matrix_v2.yaml`. New v2 runs are proposed, not launched. Existing artifacts can be reused only if revisions, prompts, precision and feature semantics match.

| Card | Goal | Proposed config | Output | Priority | Current evidence / next status |
|---|---|---|---|---|---|
| E00 | Tokenizer/cache/numerical validity | Each new pair, 64W/32F/32N; BF16, K4 | A1 validity | MUST | Qwen historical checks exist; new generalized/batched/Gemma path TODO |
| E01 | Human morphology audit | 600 cases, two Korean annotators | F1, A2 | MUST | Existing rejected-only sheet unannotated; TODO |
| E02 | Main actual-proposal signature | Q1/Q2/Q3; W5k/F997/N550; BF16, output256 | MAIN T1, F2 | MUST | Existing smaller/FP16 signatures are discovery; new test TODO |
| E03 | Data-scale stability | Nested W test500/1k/2k/5k; CPU reuse | F3a | MUST | TODO after E02; no extra generation |
| E04 | Capacity grid | Q1–Q6, identical W2k; same-prefix Qwen scoring1k | F3b/c, A3 | MUST FOR SCALE CLAIM | 8B-Base checkpoint verified available; TODO |
| E05 | Second family/tokenizer | Gemma3-4b-pt→12b-pt; W2k/F997/N550 | T1 G1 panel | MUST FOR BROADER SCOPE | Check access/tokenizers/model class and Korean output; TODO |
| E06 | IT/task transfer | QI1/GI1; KoAlpaca1k +XL-Sum summary550 | A4 / main panel if substantive | STRONG | TODO; templates and thinking mode must be explicit |
| E07 | Common support | Overlap weighting[.05,.95]; matched sensitivity | A5, T1 if space | MUST | Old strict match inconclusive/poor balance; new analysis TODO |
| E08 | Frequency/identity | Train-only counts, smoothed token lookup, identity-held-out | A6, T2 | MUST | H4 discovery available; online held-out tests TODO |
| E09 | Online incremental information | Q2 W4k/1k/5k; frozen F/N transfer; logistic +HistGB | MAIN T2, F4 | MUST | TODO; remove target/future leakage |
| E10 | Content/mechanism diagnostic | 400 natural audit cases +1,200 validated frame contexts; Q2/Q5/G1 | A7, qualitative main panel | STRONG | Requires expert labels; TODO |
| E11 | K/context/length/precision | Q2/Q5 K1/2/4/8/16; Q5 lengths128/256/512; BF16/FP16 | A8 | MUST FOCUSED | TODO; reuse matching K4 cells |
| E12 | Sampling scope | Distribution overlap2k prefixes, T.7/1; optional exact SD Q2 200W/200F×3seeds | A9 | STRONG DIAGNOSTIC / OPTIONAL IMPLEMENTATION | Current repo implements greedy only |
| E13 | Cost/headroom gate | Q4/Q5 200W/200F dev; true batched verifier | Cost panel | MUST BEFORE METHOD | Old CPU opportunities exist; real cost-aware headroom TODO |
| E14 | Runtime controller | Q4/Q5;500W/500F frozen; best fixed/entropy/confidence/morph;3repeats | MAIN T3 | CONDITIONAL MUST | Existing sequential/full-block guard incomplete; TODO after gates |
| E15 | Controller ablations | Q4 W500; matched random/all-boundary/drop-morph/lookahead | T3 or A10 | CONDITIONAL MUST | TODO only if E14 positive |
| E16 | Negative intervention/limits | Existing suppression/matching/P3/parity artifacts | A11, short main paragraph | MUST REUSE | Negative suppression confirmed in stored reports; no rerun needed |
| E17 | Cross-lingual generalization | Optional Japanese/English controls | Appendix | OPTIONAL | Outside Korean-only minimum scope |
| E18 | Morphology-aware draft distillation | Optional separate equal-budget training study | Future method study | OPTIONAL / DEFER | Not required for the proposed two-analysis paper |

Recommended gates: E00/E01 → Q2 development E07/E09 → E02/E03/E04/E05 → E06/E08/E10/E11 → E13 → E14/E15. No GPU-hour budget until a per-pair pilot is measured on actual hardware.
