# Prompt overlap audit

The model-pair prompt sets below are the prompts with at least one eligible primary-contrast row under that pair's unchanged H2 filter. The full input set is reported separately.

- Full prompt inputs: 1000 per pair; exact ID sets match across P1, P2, and P3.
- Eligible prompts: P1=995, P2=997, P3=997.
- Eligible union: 999 prompts.
- Pairwise eligible intersections: P1∩P2=993, P1∩P3=993, P2∩P3=997.
- Three-way eligible intersection: 993 prompts.

Thus the previous pooled count of 999 is the eligible-prompt union, not a count of prompts represented in every pair. Full prompt inputs were still identical across pairs.

Missing eligible IDs by pair within the union:
- P1: [268, 327, 818, 928]
- P2: [453, 894]
- P3: [453, 894]
- No eligible rows in any pair: [283]
