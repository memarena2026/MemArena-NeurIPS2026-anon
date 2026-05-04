# Instance Statistics (10k corpus, 2026-03-11)

## Per-Dimension Counts

| Dimension | Count | Jaccard (mean±std) | max |
|---|---|---|---|
| d1_conflict | 170 | 0.215±0.055 (n=116) | 0.300 |
| d2_anaphora | 170 | 0.229±0.052 (n=110) | 0.300 |
| d3_confabulation | 160 | N/A | N/A |
| d4_permission | 156 | 0.117±0.082 (n=142) | 0.280 |
| d5_cloze | 198 | N/A | N/A |
| d6_metadata | 170 | 0.016±0.032 (n=170) | 0.138 |
| d7_qa | 170 | N/A | N/A |
| d8_temporal | 170 | N/A | N/A |
| d9_negation | 119 | N/A | N/A |
| d10_counterfactual | 170 | N/A | N/A |
| d11_exception | 36 | N/A | N/A |
| **TOTAL** | **1,689** | | |

## Jaccard Overlap Distribution (hardened queries only)

- Mean: 0.1293
- Std: 0.1049
- Max: 0.3000
- Below 0.3 threshold: 538/538 (100.0%)

## Notes

- Dimensions without Jaccard scores (d3, d5, d7-d10) use non-paraphrased
  query types (MCQ, temporal, negation, counterfactual) where Jaccard
  hardening is not applicable
- d11_exception has only 36 instances (low count due to anomaly generation
  constraints)
- Will be regenerated with new corpus on H200
