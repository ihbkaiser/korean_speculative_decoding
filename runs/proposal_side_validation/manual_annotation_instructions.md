# Blind Korean morphology annotation audit

This sheet contains 200 rejected draft proposals sampled with equal allocation across Wiki/FLORES × P1/P2 × the detector’s two primary classes. The annotator-facing sheet does not expose detector labels or the Kiwi segmentation used by the computational measurement. `manual_annotation_key.csv` is a separate answer key; keep it hidden until annotation is complete.

For each row, classify the marked candidate token span in its eojeol from Korean linguistic judgment. The candidate is shown between `〔...〕` when its surface has one unique literal match in the eojeol. Rows with repeated or absent literal matches are flagged; inspect the surface carefully and use `OTHER_OR_AMBIGUOUS` if the alignment cannot be decided from the provided context.

Use these labels:
- `CROSS_MORPHEME`: the candidate span crosses one or more morpheme boundaries within one eojeol.
- `WITHIN_SPLIT`: the candidate span is a strict substring of exactly one morpheme.
- `EXACT`: the candidate span exactly matches one morpheme.
- `OTHER_OR_AMBIGUOUS`: the span is not reliably classifiable from the supplied context.

Fill `expert_label`, confidence (1–5), and notes. This audit sheet is prepared but has not been annotated. For a paper-quality validation, use at least one Korean linguistically competent annotator, preferably two independent annotators with adjudication; report agreement and class-specific precision/recall against the blinded labels.
