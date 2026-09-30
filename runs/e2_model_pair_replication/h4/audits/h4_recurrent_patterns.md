# H4 recurrent-pattern audit

Patterns are exact nominal surface × particle surface × token surface, selected by counts only. Display threshold: at least 20 CROSS tokens and 10 prompts. All patterns remain in the CSV.

## Count-ranked concentration (outcome-independent ranking)

| pair | top_k_patterns | patterns_available | patterns_in_top_k | total_cross_tokens | top_k_cross_tokens | share_of_cross_tokens | top_k_raw_rejection | top_k_share_of_rejected_cross | ranked_by                              |
| ---- | -------------- | ------------------ | ----------------- | ------------------ | ------------------ | --------------------- | ------------------- | ----------------------------- | -------------------------------------- |
| P1   | 1              | 75                 | 1                 | 211                | 15                 | 0.071                 | 0.133               | 0.036                         | n_cross only; no outcome-based ranking |
| P1   | 5              | 75                 | 5                 | 211                | 52                 | 0.246                 | 0.231               | 0.214                         | n_cross only; no outcome-based ranking |
| P1   | 10             | 75                 | 10                | 211                | 83                 | 0.393                 | 0.193               | 0.286                         | n_cross only; no outcome-based ranking |
| P2   | 1              | 87                 | 1                 | 197                | 12                 | 0.061                 | 0.000               | 0.000                         | n_cross only; no outcome-based ranking |
| P2   | 5              | 87                 | 5                 | 197                | 32                 | 0.162                 | 0.062               | 0.045                         | n_cross only; no outcome-based ranking |
| P2   | 10             | 87                 | 10                | 197                | 56                 | 0.284                 | 0.071               | 0.091                         | n_cross only; no outcome-based ranking |
| P3   | 1              | 90                 | 1                 | 201                | 12                 | 0.060                 | 0.083               | 0.014                         | n_cross only; no outcome-based ranking |
| P3   | 5              | 90                 | 5                 | 201                | 35                 | 0.174                 | 0.114               | 0.056                         | n_cross only; no outcome-based ranking |
| P3   | 10             | 90                 | 10                | 201                | 60                 | 0.299                 | 0.200               | 0.169                         | n_cross only; no outcome-based ranking |

## Frequent supported patterns

(no rows)
