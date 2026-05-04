# GPT-4o-mini Secondary Judge Scoring Plan

## Total inventory

### S corpus (10k) — 14 files × 1,271 judge instances = 17,794 total
All on LOCAL machine. Yesterday's run failed (rate limited after 23 instances).

| # | File | Status |
|---|---|---|
| S1 | inmem/rag_0_6b | 23 scored (partial, REDO) |
| S2 | inmem/rag_8b | REDO |
| S3 | inmem/rag_32b | REDO |
| S4 | llm/oracle_4b | REDO |
| S5 | oracle/oracle_0_6b | REDO |
| S6 | oracle/oracle_3b | REDO |
| S7 | oracle/oracle_7b | REDO |
| S8 | oracle/oracle_8b | REDO |
| S9 | oracle/oracle_32b | REDO |
| S10 | vanilla/vanilla_0_6b | REDO |
| S11 | vanilla/vanilla_3b | REDO |
| S12 | vanilla/vanilla_7b | REDO |
| S13 | vanilla/vanilla_8b | REDO |
| S14 | vanilla/vanilla_32b | REDO |

### M corpus (100k) — 15 files × ~1,271 judge instances = ~19,065 total
All on SPARK machine. Yesterday's run also failed.

| # | File |
|---|---|
| M1 | inmem/rag_0_6b |
| M2 | inmem/rag_3b |
| M3 | inmem/rag_7b |
| M4 | inmem/rag_8b |
| M5 | inmem/rag_32b |
| M6 | oracle/oracle_0_6b |
| M7 | oracle/oracle_3b |
| M8 | oracle/oracle_7b |
| M9 | oracle/oracle_8b |
| M10 | oracle/oracle_32b |
| M11 | vanilla/vanilla_0_6b |
| M12 | vanilla/vanilla_3b |
| M13 | vanilla/vanilla_7b |
| M14 | vanilla/vanilla_8b |
| M15 | vanilla/vanilla_32b |

### Q4 ablation — 2 files × 1,271 = 2,542 total
On SPARK machine.

| # | File |
|---|---|
| A1 | ablations/oracle/q4_oracle_8b |
| A2 | ablations/omniscient/q4_omniscient_8b |

## Total: 14 + 15 + 2 = 31 files, ~39,401 judge instances

## API capacity (per day)

| Route | Endpoint | RPD limit | Usable instances |
|---|---|---|---|
| OpenAI key 1 | api.openai.com | 10,000 | ~9,000 (safety margin) |
| OpenAI key 2 | api.openai.com | 10,000 | ~9,000 |
| OpenRouter key 1 | openrouter.ai | ~10,000* | ~9,000 |
| OpenRouter key 2 | openrouter.ai | ~10,000* | ~9,000 |
| **Total/day** | | | **~36,000** |

*OpenRouter limits vary, start conservative.

## Assignment — 2 machines, 4 keys, NO overlap

### LOCAL machine (WS009585) — S corpus
Uses 2 keys: OpenAI key 1 + OpenRouter key 1

| Batch | Key | Files | Instances |
|---|---|---|---|
| L-batch-1 | OpenAI key 1 | S1-S7 (rag_0.6b/8b/32b, oracle_4b/0.6b/3b/7b) | 7 × 1,271 = 8,897 |
| L-batch-2 | OpenRouter key 1 | S8-S14 (oracle_8b/32b, vanilla_all) | 7 × 1,271 = 8,897 |

### SPARK machine (WS009586) — M corpus + Q4 ablation
Uses 2 keys: OpenAI key 2 + OpenRouter key 2

| Batch | Key | Files | Instances |
|---|---|---|---|
| S-batch-1 | OpenAI key 2 | M1-M7 + A1-A2 (rag_all + oracle_0.6b/3b + ablations) | 9 × 1,271 = 11,439... too many! |

Revised split for Spark:

| Batch | Key | Files | Instances |
|---|---|---|---|
| S-batch-1 | OpenAI key 2 | M1-M7 (rag_all + oracle_0.6b/3b) | 7 × 1,271 = 8,897 |
| S-batch-2 | OpenRouter key 2 | M8-M15 + A1-A2 (oracle_7b-32b + vanilla_all + ablations) | 10 × 1,271 = 12,710 |

## Execution

Each batch runs `run_secondary_judge.py` with:
- `--concurrency 16` (conservative, avoid rate spikes)
- `--backends` flag to select specific files
- Different `--api-key` and `--endpoint` per batch

Batches within a machine run SEQUENTIALLY (not parallel) to avoid key confusion.

## Dedup check
- S corpus files are ONLY on local
- M corpus files are ONLY on Spark
- Q4 ablation files are ONLY on Spark
- No file appears in two batches
