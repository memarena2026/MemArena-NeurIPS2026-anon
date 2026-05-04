# Multi-machine data merge protocol — 2026-05-03

Three machines run different slices of the MemArena experiments. This doc
defines the branch / path / file conventions so all three can push to the
same GitHub repo and a single integration step on H100 can merge them
without conflicts.

```
┌────────┬──────────────────────────────────────────┬─────────────────────────┐
│ Machine│ Slice                                    │ Branch                  │
├────────┼──────────────────────────────────────────┼─────────────────────────┤
│ H100   │ main-L.tex memobase + mem0 (5 models)    │ data-h100-structured    │
│ H200   │ main-L.tex vanilla + oracle + rag +      │ data-h200-baselines     │
│        │   ablations (Tables/Figures)             │                         │
│ Spark  │ latency benchmarks                       │ data-spark-latency      │
└────────┴──────────────────────────────────────────┴─────────────────────────┘
```

After all three branches are pushed, **H100** merges them into
`data-merged-2026-05-03` and runs the aggregation scripts.

---

## 1. Output path discipline (no path collisions)

Each machine MUST write under a non-overlapping `out/` subtree.

| Machine | Allowed paths under `out/` |
|---|---|
| **H100** | `accuracy_memarena_l_${model}/` — memobase + mem0 cells, `${model}` ∈ {0_6b, llama3b, 7b, 8b, 32b}.  `accuracy_memarena_l_mem0_${model}/` for mem0-only cells. |
| **H200** | `accuracy_memarena_l_baselines_${model}/` — vanilla + oracle + baseline_simplerag.  `ablation_*/` — ablation runs. |
| **Spark** | `latency_spark_${model}_${trial}/` — `run_latency_spark.sh` outputs.  `out/spark_paper_hooks/` for any paper-table latency dumps. |

Per-cell directory layout (same for everyone):
```
out/<top>/
  eval_results_${trial}/memory_cache/
    answer_results_${backend}_${model}_${trial}.json     # per-question latency + answer
    evaluation_results_${backend}_${model}_${trial}_judge_remote.json   # accuracy + per-Q judge
    run_meta_${backend}_${model}_${trial}.json           # config metadata
  memory_cache/${backend}/${model}/${trial}/
    memcache_${backend}_A_paired_${model}_${trial}.jsonl  # local cache (DO NOT COMMIT)
  wrapper_logs/
    ${backend}_${model}_${trial}.log                     # per-day cadence + errors
  run_eval_matrix_manifest.tsv
  run_accuracy_manifest.json
```

---

## 2. What to commit per machine

**Whitelist** (small enough to live in git):
- `out/<top>/eval_results_*/memory_cache/evaluation_results_*judge*.json`
- `out/<top>/eval_results_*/memory_cache/answer_results_*.json` (NOT `*_light_*`)
- `out/<top>/eval_results_*/memory_cache/run_meta_*.json`
- `out/<top>/wrapper_logs/*.log`
- `out/<top>/*manifest*`

**Blacklist** (regenerable / huge):
- `memcache_*.jsonl` (multi-GB, regenerable from the corpus)
- `masim_transcript_*.jsonl` (~30 MB each, derivable from `data/benchmark/`)
- `masim_qa_*.json` (derivable from `data/benchmark/eval_instances/`)
- `openclaw_cleanup_*.json`, `input_output_*.json`, `search_results_*.json`
- `answer_results_light_*.json` (duplicate of full version)
- Anything under `runs/` snapshot subdirs

The repo's `.gitignore` already excludes `out/`, so every commit MUST use
`git add -f` to override it. There's a one-shot helper in `scripts/`:

```bash
bash scripts/git_add_results.sh out/accuracy_memarena_l_baselines_8b
```
(provided by H100 in this branch — see §6)

---

## 3. Per-machine push protocol

### H100 (this machine — memobase + mem0)

```bash
git fetch origin
git checkout -B data-h100-structured origin/main
# regenerate or symlink result dirs as needed
for d in out/accuracy_memarena_l_{0_6b,llama3b,7b,8b,32b} \
         out/accuracy_memarena_l_mem0_{0_6b,llama3b,7b,8b,32b}; do
  [[ -d "$d" ]] || continue
  bash scripts/git_add_results.sh "$d"
done
git -c user.email=anonymous@example.com \
    -c user.name='Anonymous (H100)' \
    commit -m "H100 structured backends snapshot ($(date -u +%F))"
git push -u origin data-h100-structured
```

### H200 (vanilla + oracle + rag + ablations)

```bash
git fetch origin
git checkout -B data-h200-baselines origin/main
for d in out/accuracy_memarena_l_baselines_*/ out/ablation_*/; do
  [[ -d "$d" ]] || continue
  bash scripts/git_add_results.sh "$d"
done
git -c user.email=anonymous@example.com \
    -c user.name='Anonymous (H200)' \
    commit -m "H200 baselines + ablations snapshot ($(date -u +%F))"
git push -u origin data-h200-baselines
```

### Spark (latency)

```bash
git fetch origin
git checkout -B data-spark-latency origin/main
for d in out/latency_spark_*/ out/spark_paper_hooks/; do
  [[ -d "$d" ]] || continue
  bash scripts/git_add_results.sh "$d"
done
git -c user.email=anonymous@example.com \
    -c user.name='Anonymous (Spark)' \
    commit -m "Spark latency snapshot ($(date -u +%F))"
git push -u origin data-spark-latency
```

---

## 4. Integration on H100

Once all three branches are pushed:

```bash
git fetch origin
git checkout -B data-merged-2026-05-03 origin/main
# Octopus merge — works because the three branches touch disjoint paths.
git merge --no-ff origin/data-h100-structured \
                  origin/data-h200-baselines  \
                  origin/data-spark-latency
# If a conflict appears it must be a path-discipline violation in §1.
```

After merge:
```bash
ls out/  # should show all three machines' outputs side by side
```

---

## 5. Aggregation + analysis

Once merged, generate the unified report:

```bash
.venv/bin/python memarena/figures/paper_data.py \
  --out-dir out/ \
  --output reports/main_L_aggregated_2026-05-03.json \
  --include-backends vanilla,oracle,baseline_simplerag,memobase,mem0
```

`paper_data.py` walks `out/accuracy_memarena_l_*` and emits a per
(backend, model, trial) table with accuracy + per-dim breakdown, ready
for the paper figures.

For latency:
```bash
.venv/bin/python -c "
import json, glob, statistics
for f in glob.glob('out/latency_spark_*/eval_results_*/memory_cache/answer_results_*.json'):
    rows = json.load(open(f))
    ttft = sorted(r.get('ttft_ms', 0) or 0 for r in rows)
    ans  = sorted(r.get('answer_time_ms', 0) or 0 for r in rows)
    print(f, 'p50_ttft', ttft[len(ttft)//2], 'p50_ans', ans[len(ans)//2])
"
```

For accuracy summaries by backend:
```bash
for f in out/accuracy_memarena_l_*/eval_results_*/memory_cache/evaluation_results_*judge*.json; do
  jq -r '[input_filename, .summary.accuracy] | @tsv' "$f"
done | sort -k1
```

---

## 6. The whitelist helper (`scripts/git_add_results.sh`)

A small wrapper that takes a results dir and force-adds the whitelisted
files (since `out/` is gitignored).

```bash
#!/usr/bin/env bash
# Usage: scripts/git_add_results.sh out/accuracy_memarena_l_baselines_7b
set -euo pipefail
[[ -d "${1:-}" ]] || { echo "usage: $0 <out/dir>"; exit 2; }
d="$1"

find "$d" -name "evaluation_results_*judge*.json" -exec git add -f {} +
find "$d" -name "answer_results_*.json" -not -name "*_light_*" -exec git add -f {} +
find "$d" -name "run_meta_*.json"        -exec git add -f {} +
find "$d/wrapper_logs" -name "*.log"     -exec git add -f {} + 2>/dev/null
find "$d" -maxdepth 1 -name "*manifest*" -exec git add -f {} +

echo "[git_add_results] staged whitelisted files under $d"
```

---

## 7. Conventions & gotchas

- **Branch lifetime**: each `data-<machine>-*` branch is for ONE day's
  snapshot. If a machine pushes again, force-update or open a new
  date-suffixed branch (`data-h100-structured-2026-05-04`).
- **Don't merge into `main`**. Merges land on `data-merged-YYYY-MM-DD`
  so `main` stays code-only.
- **Paths are case-sensitive on Linux**. Use exactly the model tag
  conventions: `0_6b`, `llama3b`, `7b`, `8b`, `32b`.
- **Trials are `s2`, `s3`, `s4`** (s1 is reserved for unit tests).
- **DO NOT commit anything under `~/.mem0/`, `/tmp/mem0_qdrant_*`,
  `/tmp/graphiti_qdrant_*`, or `/tmp/mem0_history_*`** — those are
  process-local scratch and don't belong in git.
- **Latency dirs (`out/latency_spark_*/`)** also have an
  `_elapsed_seconds.txt` summary file the wrapper writes. Include it.
- If two machines accidentally write to the SAME `out/<top>` dir, the
  merge will silently let the LATER push win for any colliding file.
  Stick to §1.

---

## 8. Status today (2026-05-03)

What's already been pushed (data on existing branches):

| branch | content |
|---|---|
| `data-snapshot-2026-05-03` | H100 dump: 0_6b/3b/7b/8b/32b memobase+memos cells (memos partially broken on 8b/32b — see notes), exp_async_llama3b_s2_sync experiment |
| `main` | code only |

Not yet pushed:
- H100 mem0_0_6b cells (currently building)
- H200 vanilla + oracle + rag + ablations
- Spark latency

Once all three machines push their respective branches per §3, run §4
on H100.
