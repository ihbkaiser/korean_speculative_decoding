# P1: 0.6B → 1.7B

## Data and alignment
- Generated tokens: 127,118; eligible primary contrast tokens: 73,001 across 995 prompts.
- Exact visible spans: 126,770/127,108 non-special generated tokens.
- CROSS_EOJEOL excluded: 0; KIWI_COMPLEX excluded: 10,145; ambiguous exclusion: 7.98%.
- Target/speculative outputs matched on all 1,000/1,000 prompts; singleton reference fallbacks: 0.
- Re-tokenization exact for 898/1,000 prompts.

## Primary models and raw rates

| Class | N tokens | N prompts | Raw rejection rate |
|---|---:|---:|---:|
| WITHIN_SPLIT | 69,547 | 993 | 12.78% |
| CROSS_MORPHEME | 3,454 | 652 | 20.15% |

Model formulas, estimates, two-sided tests, and prompt-clustered summaries are in `regression.txt`.
