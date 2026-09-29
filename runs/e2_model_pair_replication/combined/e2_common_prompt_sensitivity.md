# Common-prompt pooled sensitivity

Eligible prompt sets have sizes P1=995, P2=997, and P3=997. Their union has 999 prompts; the exact three-way intersection has **993 prompts**.

The original pooled eligible result uses the union of prompts with any eligible rows, not a common-prompt intersection. The sensitivity refits pooled M3 and frequency-controlled M5 using only the exact three-way intersection. Formulas, fixed effects, controls, and prompt-clustered SEs match the original pooled models.
The original-union refits match the saved E2 regression coefficients, OR intervals, and p-values to numerical tolerance.

Classification: **unchanged**. This is based on direction and effect magnitude: unchanged means every OR changes by at most 5%; minor means the maximum change is between 5% and 10%; material means any sign reversal or a change of 10% or more. Maximum absolute log-OR change: 0.0125.

| Model | Pair | Original union OR (95% CI) | Common-only OR (95% CI) | Relative OR change | Common N tokens / prompts | p |
|---|---|---:|---:|---:|---:|---:|
| M3 | P1 (0.6B → 1.7B) | 2.190 (1.840–2.606) | 2.205 (1.851–2.625) | +0.7% | 72,843 / 993 | 7.19e-19 |
| M3 | P2 (1.7B → 4B) | 1.680 (1.417–1.991) | 1.662 (1.402–1.970) | -1.1% | 72,910 / 993 | 4.83e-09 |
| M3 | P3 (0.6B → 4B) | 1.770 (1.498–2.092) | 1.759 (1.488–2.079) | -0.6% | 72,807 / 993 | 3.52e-11 |
| M5 | P1 (0.6B → 1.7B) | 2.394 (2.010–2.852) | 2.406 (2.019–2.867) | +0.5% | 72,843 / 993 | 9.45e-23 |
| M5 | P2 (1.7B → 4B) | 1.840 (1.550–2.185) | 1.817 (1.531–2.158) | -1.2% | 72,910 / 993 | 9.31e-12 |
| M5 | P3 (0.6B → 4B) | 1.940 (1.637–2.298) | 1.924 (1.623–2.280) | -0.8% | 72,807 / 993 | 4.43e-14 |

This is a population-overlap sensitivity analysis only; it does not redefine the primary E2 estimates.
