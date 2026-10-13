# AGENTS.md — korean_speculative_decoding

## Layout
- `results/naacl_experiments/` — frozen experiment outputs. NEVER modify or regenerate files here.
  - `simulated/<PAIR>/<split>/events_k{1,2,4,8,16}.parquet` — greedy-SD replay events (`eligible`, `rejected`, `slot`, `block_id`, `invalidated_after`).
  - `teacher_forced/generated/<target>/<model>/<split>/shard-*.parquet` — per-position scores (`argmax_id`, `entropy`, `top2_margin`, `logp_argmax`; target self-scores also have `top20_ids`, `top20_logp`).
  - `teacher_forced/human/<target>/<model>/common/` — same, on human-written continuations.
  - `aligned/kiwi/<target>/<split or human/common>/shard-*.parquet` — Kiwi token–morpheme alignment (`alignment_class`, `token_index_in_eojeol`, `token_start`, `morpheme_char_spans`, `global_token_char_span`, ...).
- `configs/naacl_experiments.yaml` — pair definitions (Q1..Q6, G1, M1: draft, target, split).
- `paper/` — the LaTeX paper (Overleaf project). `paper/scripts/` builds everything:
  - `build_decision_point_tables.py`, `analyze_decision_points.py`, `standardize_by_margin.py`,
    `simulate_stopping.py`, `summarize_stopping.py`, `control_ladder.py`, `segmentation_competitors.py`
  - `make_tables.py` → `paper/numbers.tex` (macros) and `paper/tables/*.tex`; `make_figures.py` → `paper/figures/`.
  - All numbers in the paper come from `paper/data/*` through these scripts.

## Rules (must follow)
1. Never type a result number into `main.tex` by hand. Add it to `paper/data/` from a script, expose it as a macro in `make_tables.py`, and use the macro.
2. Never invent or estimate a result. If something cannot be computed in this environment (e.g. needs a GPU), write the script, document how to run it, and put a `\pending{...}` placeholder in the paper.
3. Report what the data show, including results that weaken the claim. Do not drop pairs or rows to improve a result.
4. Do not mention excluded or faulty data cells in the paper. Use only: generated text for Q1–Q6, G1, M1; human text for Q1–Q6 and M1.
5. Run Python on repo data with `python -I`. Prefer the common 20k split (pool_index < 20000) when memory is tight; Q2 "all" needs about 6 GB RAM.
6. After any change: `cd paper && python scripts/make_tables.py && python scripts/make_figures.py && pdflatex main && bibtex main && pdflatex main && pdflatex main`. The build must finish with no errors, no undefined references and no overfull boxes. The main text (through Conclusion) must stay within 8 pages.
7. Keep the writing style: short sentences, no hype, numbers via macros, claims scoped to Korean.
8. Each PR must include: what was run (commands), new data files, a short table of new numbers, and the diff of `main.tex`.

## Current checkout note
This checkout currently has `paper.tex` and `refs.bib` at the repository root, not under `paper/`; `paper/scripts/make_tables.py`, `paper/scripts/make_figures.py`, `pdflatex`, and `bibtex` are unavailable here. Before applying rule 6 or claiming a paper build passed, identify the actual paper build workflow and toolchain for this checkout.
