# Finding 3 Compression Plan (50% target)

Drafted 2026-04-27 cycle 43 via subagent audit of `chapters/07_results_and_analysis.tex` lines 47-58.

**Current**: 482 words across 4 sub-blocks (a)/(b)/(c)/Ingest overhead.
**Target**: ~241 words (50% reduction).
**Achievable**: ~215 words (56% reduction with safety margin).

## Per-block word counts and target

| Block | Current | Target | Cut |
|---|---:|---:|---:|
| (a) Serving-time Pareto | 161 | 85 | 76 |
| (b) Energy workload-conditional | 102 + 90 footnote = 192 | 50 | 142 |
| (c) Why structured memory wins | 153 | 80 | 73 |
| Ingest overhead | 66 | 45 | 21 |
| **Subtotal (per-block tightening)** | **482** | **260** | **222** |
| Plus (a)+(b) merge into single "Operating point" para | -50 |
| **Final** | | **~210** | **~272** |

## Concrete moves

### Block (a) — delete + footnote
- DELETE `"strictly dominates ... on every axis"` and the `+22.3 pp / ~1500x / 27°C cooler` restatement (the bare numbers say it).
- MOVE Vanilla 8K truncation justification (`"Vanilla here applies the deterministic 8,192-token truncation ... do not close the gap to Memobase"`) → footnote.
- KEEP: 58.3 vs 36.0, 0.56 vs 891.6 J/query, 55°C vs 82°C, NVML caveat, fig:finding3(a)(b) refs, the Q3-8B within-reader anchor (Memobase ties Oracle, Vanilla/RAG trail 24-41pp).

### Block (b) — drop footnote, drop Oracle defensiveness
- MOVE entire OpenAI/NBER ChatGPT-usage calibration footnote (~90 words) → `Appendix~\ref{app:ingest-amortization}` standalone para. Replace inline with `"workload-volume calibration in App.~\ref{app:ingest-amortization}"`.
- DELETE `"We report break-even against Oracle as the most conservative threshold (Oracle has zero ingest cost ... but is an analysis upper bound rather than a deployable baseline); against Vanilla or RAG the break-even is tighter by construction."` — this is editorial defensiveness; fig caption already explains the choice.
- KEEP: K≈5,170 number, 52.5 kJ ingest cost, "below threshold = capability argument; above = lifecycle energy" two-clause statement.

### Block (c) — compress query-time list
- DELETE the full enumeration `(LLM summarization, BGE-M3 dense retrieval at matched k=5, expanded context, MS-MARCO cross-encoder reranking)` and the appendix-table list `(Tab h2-llm-sum, p1c-rag, p1a-budget)`.
- KEEP the cross-encoder counter-example (`-20.6 pp on BM25 RAG`) as the single concrete number; it's a reviewer-magnetic robustness fact.
- DELETE `"(i)"` and `"(ii)"` consequence labels; fold into prose.
- KEEP: prompt-length contrast (Memobase ~1.5K vs Vanilla ~4.8K tokens), monotonic Avg-lift with reader scale, Mistral schema-failure (Tab:mistral-structured ref).

### Ingest overhead — citation cleanup
- DELETE non-load-bearing cites `\citep{kwon2023vllm, zheng2024sglang}` (keep `memobase2025, memos2025`).
- DELETE `"on the same backbone"` redundancy.
- KEEP: Memobase 1712s / 52.5 kJ vs MemOS 6917s / 240.5 kJ contrast, ~4× extractor-time gap, fig:deployment-triptych ref.

## Structural recommendation

Currently 4 blocks: (a)/(b)/(c) + Ingest overhead.

Proposed: **merge (a)+(b) into single "Operating point and amortization" paragraph**, keep (c), close with single-sentence Ingest overhead. Saves ~50 more words via reduced topic-sentence overhead. Result: 3 paragraphs total.

## Non-negotiables (verify each kept after cuts)

- 58.3 vs 36.0 Avg accuracy
- 0.56 vs 891.6 J/query GPU energy
- 55°C vs 82°C peak temperature
- "GPU-only via NVML, CPU/DRAM/I/O excluded" caveat (at least once)
- K≈5,170 break-even number
- 52.5 kJ ingest cost
- Memobase vs MemOS ingest contrast (1712s vs 6917s; 52.5 kJ vs 240.5 kJ)
- Memobase ~1.5K vs Vanilla ~4.8K prompt tokens
- Mistral schema-compliance failure as concrete robustness counter-case
- fig:finding3 forward ref + Appendix:ingest-amortization forward ref
- The "structured memory beats scaling the reader" headline claim

## Risk callouts

| Cut | Risk | Mitigation |
|---|---|---|
| Vanilla truncation → footnote | Low | Fig (a) + Table:results-L show Vanilla absolute numbers anyway |
| OpenAI footnote → Appendix | Low | Calibration is defensive; K≈5,170 is the fact, fig caption mentions it |
| Drop full query-time list | Medium | Reviewer might ask "did you test dense retrieval?" — Ablation §7.3 lists them; reranking -20.6pp counter stays inline |
| Merge (a)+(b) | Low | Logical flow improves (operating point → mechanism → constraint) |
| Drop "strictly dominates" | None | Numbers speak |
| Drop Oracle defensiveness | Low | Fig caption clarifies the choice |

## Execution checklist for the next /loop cycle

1. Apply Block (a) cuts (delete redundancy + move truncation prose to footnote). Verify 58.3/36.0/0.56/891.6/55/82 numbers all stay.
2. Apply Block (b) cuts (move footnote to Appendix:ingest-amortization, delete Oracle defensiveness). Verify K≈5,170 + 52.5 kJ stay.
3. Apply Block (c) cuts (compress query-time list, keep -20.6 pp). Verify Memobase 1.5K vs Vanilla 4.8K + Mistral failure stay.
4. Cleanup Ingest overhead cites.
5. Merge (a)+(b) topic sentences into one operating-point paragraph.
6. Build, recount Finding 3 word count, recount main body pp.
7. Push.
