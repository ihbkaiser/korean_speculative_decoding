# Historical P3 model revision recovery

**Result: unresolved for both P3 weight revisions.** No revision is inferred from current Hugging Face `main` or silently copied from the E2 run.

- P3 run ID: `20260926T184145Z_pilot1000`; approximate start from run ID: `2026-09-26T18:41:45+00:00`; recorded elapsed: `30934.50831770897` seconds; estimated end: `2026-09-27T03:17:19.508318+00:00`.
- Hugging Face cache path inspected: `/ephemeral/cache/huggingface/hub`; `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, and `TRANSFORMERS_CACHE` are unset in this process. The home cache path resolves to `/ephemeral/cache/huggingface/hub`.
- P3 files tracked by repository Git: 39; local P3 `.log`/`.out`/`.err` files: 0; readable shell-history matches: 0.

| P3 model | Recovered revision | Confidence | Local evidence |
|---|---|---|---|
| Qwen/Qwen3-0.6B-Base (draft) | unresolved | unresolved | accessible cache snapshot(s): da87bfb608c14b7cf20ba1ce41287e8de496c0cd @ 2026-09-28T18:21:43.156161+00:00 |
| Qwen/Qwen3-4B-Base (target) | unresolved | unresolved | accessible cache snapshot(s): 906bfd4b4dc7f14ee4320094d8b41684abff8539 @ 2026-09-28T18:23:51.521998+00:00 |

## Evidence inspected

- P3 `config.yaml` has model IDs but no `revision` fields; P3 `run_metadata.json` has no model revision or snapshot path.
- The accessible cache currently contains only the listed snapshots. Their snapshot and weight-blob modification times are after the approximate P3 completion time, so they are evidence of later availability, not P3 selection.
- The cached config/tokenizer hashes and weight file sizes/timestamps are recorded in `p3_revision_recovery.json`. No corresponding P3-time model config/weight hashes were saved for comparison.
- E1 recorded tokenizer revision `906bfd4b4dc7f14ee4320094d8b41684abff8539` for the 4B tokenizer. This does not identify the 4B weights loaded by P3.
- No matching local shell history entries or P3 run logs were found. Git tracks 39 files under the P3 run path.

The historical P3 0.6B revision remains unresolved. The historical 4B model-weight revision also remains unresolved; its later tokenizer revision is not sufficient evidence.

**Recommended for future reproducibility:** one future model-pair rerun with immutable revisions for both models recorded in run metadata. No rerun is performed as part of this housekeeping task.
