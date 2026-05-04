# NeurIPS 9-page Main-Body Compression Plan

Author: drafted 2026-04-27 cycle 41 via subagent audit of preprint.pdf
Target: 13.2 pp main body (abstract + §1-§9 before References) -> 9.0 pp
Deficit: 4.2 pp to cut.
Constraint: every cut goes either to (a) tighter prose carrying same info,
or (b) appendix with a forward reference. No content lost.

## Execution priority (impact/risk ordered)

| # | Action | File:scope | Est. pp saved | Risk | Notes |
|---|--------|------------|--------------:|------|-------|
| 1 | Tighten Intro contributions paragraph | chapters/01:14-17 | 0.20 | low | Same info, cleaner structure |
| 2 | Move Table:ablation-summary to appendix; replace with 2-sentence forward ref | tables/main_ablation_summary.tex; chapters/07 | 0.25 | medium | Appendix table must stay prominent |
| 3 | Condense D6 decomposition detail in Finding 1 | chapters/07 (Finding 1 block, lines ~28-35) | 0.35 | low | Headline stays; detail to app:localA-refusal-sensitivity |
| 4 | Compress Table:comparison to 2-line stub; move full grid to appendix | tables/comparison.tex; chapters/02 | 0.35 | medium | Appendix table must be identical, prominent |
| 5 | Compress Limitations paragraph | chapters/08 (Limitations block, lines ~6-7) | 0.30 | low | Append "Full analysis in app:limitations" |
| 6 | Trim Mistral temporal-ceiling tangent in Finding 2 | chapters/07 (Finding 2 block, lines ~41-42) | 0.25 | low | Substantive claim survives |
| 7 | Cut defensive query-time-substitutes list in Finding 3c | chapters/07 (Finding 3c, line ~53) | 0.25 | low | List is in appendix tables already |
| 8 | Tighten Statistical Protocol paragraph in §6 | chapters/06 (lines ~12-13) | 0.15 | low | Appendix tab:pvalues forward ref |
| 9 | Remove Manual Quality Audit paragraph from §4 | chapters/04 (lines 20-21) | 0.08 | low | Move to app:cost or leave in appendix |
| 10 | Move Fig:d6 sub-panel (a) scenarios to appendix figure | chapters/07 inline figure | 0.15 | medium | Keep (b,c) as Finding 1 evidence |
| 11 | Shorten Figure:finding3 caption | figures/fig_finding3.tex | 0.15 | low | Amortization detail is in app:ingest-amortization |
| 12 | Trim long footnote on workload calibration | chapters/07 around K=5,170 cite | 0.10 | low | Detail is in app:ingest-amortization |
| **firm subtotal** | | | **2.58** | | |
| 13 | Inline reword pass: cut "broadly", "we believe", "suggests"; tighten parentheticals; remove redundant citations | all of §1-§9 | ~0.8 | low | Stretch buffer; do as a single sweep at the end |
| **total** | | | **~3.4 firm + 0.8 stretch** | | |

## Non-negotiables (DO NOT cut)

- Abstract
- Three Findings (i)/(ii)/(iii) headline claims in §7
- Table:dimensions (D1-D6 definitions, used throughout §5-7)
- Table:results-L (main results)
- Figure:finding3 (Finding 3 Pareto anchor)
- Figure:three_gaps (Intro motivational anchor)
- Figure:architecture (§4 MASim pipeline anchor)
- §2 LifeBench/OrgForge positioning sentence (just added; reviewer-defensibility)
- Section §9 Conclusion three-Findings recap with break-even number

## Per-section page budget (current vs target)

| Section | Current pp | Target pp | Cut |
|---------|-----------:|----------:|----:|
| §1 Introduction | 1.6 | 1.4 | 0.20 |
| §2 Related Work | 1.2 | 0.85 | 0.35 |
| §3 Benchmark Design | 0.75 | 0.75 | 0 |
| §4 Data Generation | 1.2 | 1.12 | 0.08 |
| §5 Scoring/Eval | 1.45 | 1.45 | 0 |
| §6 Experimental Setup | 0.95 | 0.80 | 0.15 |
| §7 Results & Analysis | 4.5 | 3.5 | 1.00 (#3+#6+#7+#10+#11+#12 = 1.05) |
| §8 Discussion | 1.85 | 1.55 | 0.30 |
| §9 Conclusion | 0.9 | 0.9 | 0 |
| inline reword sweep | spread | -0.8 | 0.80 |
| **total** | **13.5** | **~9.2** | **~3.4** |

## Sequencing recommendation

Don't cut all at once -- compress in 3 batches over consecutive /loop cycles
so each batch can be verified by a build + diff:

- Batch A (low-risk, 0.93 pp): #1 + #5 + #6 + #7 + #8 + #9
- Batch B (medium-risk, 1.10 pp): #2 + #4 (both involve table relocation; do after Batch A so the appendix already has the destination structure)
- Batch C (figure-level + reword sweep, ~1.4 pp): #3 + #10 + #11 + #12 + #13

After Batch A, rebuild preprint and re-measure pages. If §7 Results
already reads tightly, may not need full Batch C.

## Risk map

- Highest risk: #2 Table:comparison move and #4 Table:ablation-summary move.
  Both rely on appendix table remaining easy to find. Plan: ensure appendix
  has clear sub-section headers and forward refs use \Cref{} not just page
  numbers, so the relocated content is one click away.
- Lowest risk: #5 Limitations compression and #11 caption shortening. Pure
  prose with no claim loss.
- The reword sweep (#13) gives ~0.8 pp without any structural changes; if
  Batches A+B already bring us to <=10 pp, the sweep alone closes the gap.

## Verification plan after each batch

1. `bash ~/.claude/skills/paper-build/build_preprint.sh` to rebuild preprint.pdf
2. Run `python3 -c "import subprocess; out = subprocess.run(['pdftotext','preprint.pdf','-'], capture_output=True, text=True).stdout; pages = out.split('\f'); print(len([p for i,p in enumerate(pages) if i < next(j for j,q in enumerate(pages) if 'References' in q[:300] and j > 0)]))"` to count main-body pages
3. Read the body around each cut to confirm no broken \ref or hanging clause
4. Commit + push the batch
