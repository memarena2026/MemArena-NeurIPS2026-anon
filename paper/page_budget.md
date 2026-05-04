# Page Budget Plan — 9 page target (NeurIPS 2026 E&D)

## Current estimate: ~12-13 content pages (need to cut ~3-4 pages)

| Section | Current (lines) | Target (lines) | Action |
|---|---|---|---|
| Abstract | 11 | 11 | Keep — already tight |
| 1. Introduction | 41 | 35 | Trim ego-centric gap paragraph slightly (-6) |
| 2. Related Work | 173 | 90 | **Heavy cut.** Table 1 stays (essential). Move data-gen-methodology paragraph to appendix. Compress benchmark descriptions to 1-2 sentences each. Remove concurrent work list or move to appendix. Target: ~1.5 pages |
| 3. Benchmark Design | 60 | 50 | Compress dimension descriptions — Table 2 carries the detail. Cut "diagnostic hierarchy" paragraph. Move "ego-centric only dimensions" and "non-independence" paragraphs to appendix (-10) |
| 4. Data Generation | 97 | 70 | GT types table can move to appendix. Compress PA pipeline numbered list to inline. Merge simulation engineering paragraphs tighter (-27) |
| 5. Scoring | 29 | 25 | Light trim — compress cross-family judging (-4) |
| 6. Experiments | 133 | 100 | Move efficiency analysis details to appendix. Compress research hypotheses to 3 (drop H3, H5, H6 or move to appendix). Tighten main results text (-33) |
| 7. Discussion | 60 | 40 | Compress 3 expected-insights paragraphs. Move synthetic-data and limitations paragraphs partly to appendix (-20) |
| 8. Conclusion | 12 | 10 | Light trim (-2) |
| **Total** | **616** | **431** | **~7.8 pages text + tables/figures ≈ 9 pages** |

## Tables/Figures space estimate
- Table 1 (comparison): ~0.6 page
- Table 2 (dimensions): ~0.3 page
- Figure 1 (architecture): ~0.4 page
- Table 3 (corpus stats): ~0.2 page
- Results tables (S, L): ~0.5 page
- GT types table: move to appendix to save ~0.2 page
- Total tables/figures: ~2.0 pages

## So: ~7.8 pages text + ~2.0 pages tables/figures ≈ 9.8 pages → tight but feasible with careful trimming

## Priority cuts (biggest savings):
1. **Related Work** (-83 lines, saves ~1.5 pages) — biggest win
2. **Experiments** (-33 lines, saves ~0.6 page) — move efficiency details + hypotheses to appendix
3. **Data Generation** (-27 lines, saves ~0.5 page) — compress pipeline description
4. **Discussion** (-20 lines, saves ~0.4 page) — compress expected insights
