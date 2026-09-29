# P2: 1.7B → 4B

## Data and alignment
- Generated tokens: 127,245; eligible primary contrast tokens: 73,257 across 997 prompts.
- Exact visible spans: 127,092/127,232 non-special generated tokens.
- CROSS_EOJEOL excluded: 0; KIWI_COMPLEX excluded: 9,698; ambiguous exclusion: 7.62%.
- Target/speculative outputs matched on all 1,000/1,000 prompts; singleton reference fallbacks: 160.
- Re-tokenization exact for 920/1,000 prompts.

## Primary models and raw rates

| Class | N tokens | N prompts | Raw rejection rate |
|---|---:|---:|---:|
| WITHIN_SPLIT | 69,495 | 997 | 14.27% |
| CROSS_MORPHEME | 3,762 | 730 | 19.32% |

Model formulas, estimates, two-sided tests, and prompt-clustered summaries are in `regression.txt`.
