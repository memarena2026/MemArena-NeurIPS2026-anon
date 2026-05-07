# Hugging Face Metadata Cheat Sheet

Use these values for the public MemArena-L Hugging Face dataset page.

## Repository Metadata

| Field | Value |
|---|---|
| Repository type | Dataset |
| Suggested name | `MemArena-L` or `memarena-l` |
| License | `cc-by-4.0` |
| Language | `en` |
| Tags | `memory`, `agents`, `long-context`, `multi-user`, `ego-centric`, `on-device`, `open-weight`, `permission-aware`, `benchmark`, `synthetic` |
| Paper | Final arXiv / proceedings URL, when available |
| Code | Public MemArena GitHub URL |

## Files

For the expanded Hugging Face layout, upload:

```text
README.md                  # copied from docs/DATASET_CARD.md
LICENSE                    # copied from docs/LICENSE
CITATION.cff               # copied from docs/CITATION.cff
croissant.json             # copied from docs/memarena-croissant.json with final URL
agents_personas.jsonl.gz
agent_schedules.jsonl.gz
corpus_sessions.jsonl.gz
events.jsonl
groups.json
locations.json
graph_edges.jsonl
ego_projections.json
ego_session_map.json
day_checkpoint.json
pipeline_report.json
eval_instances/d1_conflict.jsonl
eval_instances/d2_anaphora.jsonl
eval_instances/d3_confabulation.jsonl
eval_instances/d4_permission.jsonl
eval_instances/d5_cloze.jsonl
eval_instances/d6_metadata.jsonl
eval_instances/d7_qa.jsonl
eval_instances/d8_temporal.jsonl
eval_instances/d10_counterfactual.jsonl
```

For an archive-first release, upload:

```text
memarena-l-v1.0.0.tar.gz
memarena-l-v1.0.0.SHA256SUMS
LICENSE
CITATION.cff
README.md
croissant.json
```

## Public Citation

The dataset card and `CITATION.cff` should use the final camera-ready paper
title, author list, proceedings/arXiv URL, and dataset URL once available.

Before public release, replace these placeholders everywhere:

- `<ANONYMOUS-PAPER-URL>`
- `https://huggingface.co/datasets/anonymous/memarena-l`
- any remaining placeholder Hugging Face dataset URL
- any remaining placeholder citation fields
