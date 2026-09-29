# P1 provisional analysis: 0.6B → 1.7B

Computed from complete P1 artifacts while P2 is still running. These are pair-specific estimates under the prespecified E2 models; pooled inference and the final E2 decision await P2.

- Generated tokens: 127,118; eligible tokens: 73,001; eligible prompts: 995.
- Exact target/speculative continuation match: True across 1,000 prompts; target fallbacks: 0.
- CROSS_EOJEOL excluded: 0; KIWI_COMPLEX excluded: 10,145; exact retokenization failures: 338; ambiguous exclusion rate: 7.98%.

| Class | Tokens | Prompts | Raw rejection rate |
|---|---:|---:|---:|
| WITHIN_SPLIT | 69,547 | 993 | 12.78% |
| CROSS_MORPHEME | 3,454 | 652 | 20.15% |

| Model | OR (95% CI) | p | N |
|---|---:|---:|---:|
| M1 | 1.609 (1.450–1.784) | 2.29e-19 | 73,001 |
| M2 | 2.527 (2.081–3.069) | 8.93e-21 | 73,001 |
| M3 | 2.002 (1.604–2.499) | 8.6e-10 | 73,001 |
| M4 | 1.963 (1.566–2.461) | 4.92e-09 | 73,001 |
| M5 | 2.086 (1.653–2.633) | 5.85e-10 | 73,001 |
