# Rebuttal analyses on the released logs (CPU only)

Replays of the teacher-forced logs in `results/naacl_experiments/` (read-only). No model is run.
Place this folder at `scripts/rebuttal/cpu/` in the repository. Outputs go to `results/rebuttal/cpu/`.
Set `KSD_REPO=/path/to/repo` to run from elsewhere.

    pip install pandas pyarrow numpy scipy scikit-learn numba
    python -I scripts/rebuttal/cpu/expA_policies.py     # baselines, oracle, held-out thresholds, document bootstrap
    python -I scripts/rebuttal/cpu/expA2_cost.py        # cost-ratio sensitivity, fixed k up to 16, 1-nat threshold
    python -I scripts/rebuttal/cpu/expB_thirdparty.py   # margin of a model outside the pair
    python -I scripts/rebuttal/cpu/expC_gemma.py        # fork 2x2 table, repetition, confidence strata
    python -I scripts/rebuttal/cpu/expC2_repetition.py  # results on non-repeated positions

`reference_out/` holds the outputs of the run reported in the rebuttal plan, for comparison.

Notes
- `policysim.py` reproduces Table 5 for Q2: 3.45 tokens per call at k=4, 3.90 entropy, 4.04 margin
  (paper: 3.45, 3.89, 4.00). After a rejection the draft's stop signals on its own continuation are
  not in the logs; the replay uses the on-path values.
- Position types in `ksdlib.read_aligned` are re-derived from the alignment tables. Fused and internal
  rates match the paper (23.8%, 7.6% for Q2); word-onset and morpheme-onset rates are 22.9% and 17.5%
  against the paper's 26.6% and 15.1%. Swap in the paper's labels before quoting position-type numbers.
- Policy results use the odd-`pool_index` half of the 20k common documents as the test set; thresholds
  for the held-out check and the learned stopper come from the even half.
- Peak memory is about 6 GB (Q2 on all 40k documents in `expB_thirdparty.py`).
