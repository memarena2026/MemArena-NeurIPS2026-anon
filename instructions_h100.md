# H100 self-probe runbook (D6 paired-probe access-control test)

## What this is

The current D6 evaluation asks each access-control item with **a third-party
asker** and observes the model's response. We cannot tell from one observation
whether a "I don't have access to X" answer means:
- (a) recall failure: the system never had the fact, or
- (b) access control: the system has the fact but refuses to share with a
  third party.

This run adds the **second half** of a paired-probe protocol: re-ask each D6
item with `asker = ego_agent` (the user themselves). Cross-tabbing the two
probes per item lets us measure a true behavioural access-control rate:
**how often does the system refuse a third party for an item it would have
disclosed to ego?**

The repository now ships:
- `--d6-probe-mode self_ego` flag in `eval/cli.py` (overrides the asker for
  d4_permission items at inference time only; nothing else changes).
- `scripts/judge_d6_self_probe.py` — uses the existing 5-label rubric to
  judge the self-probe answer files.

You'll do two things:
1. **Inference**: run `--d6-probe-mode self_ego` on each (backend, reader,
   seed) cell. Output goes into a fresh directory tree
   (`out/d6_self_probe/<cell>/eval_results_<seed>/...`) so the existing
   third-party answer files are NOT overwritten.
2. **Judging**: after inference completes, run
   `python3 scripts/judge_d6_self_probe.py` on H100 with
   `OPENROUTER_API_KEY` set. The judge labels each self-probe response and
   writes the labels back into the same JSON files in place
   (with `*_legacy.json` backups).

You do NOT need to modify any figure, table, or paper code. After judging,
ping back so the analysis side joins the two probe sets.

---

## Step 0 — pull latest code

```bash
cd /path/to/MemArena   # whatever path on H100
git pull
```

Verify the new flag is present:

```bash
python3 eval/cli.py --help 2>&1 | grep -A4 "d6-probe-mode"
```

You should see:

```
  --d6-probe-mode {third_party,self_ego}
                        D6 paired-probe protocol mode (only affects
                        d4_permission items): 'third_party' (default) keeps
                        the original asker; 'self_ego' overrides asker to
                        the ego agent. ...
```

## Step 1 — sanity test: 1 cell, 1 backend, 1 seed

Pick the cheapest sane combination first to confirm the flag plumbing
works end-to-end before committing 75 cells of compute.

**Recommended sanity cell**: `vanilla × Qwen3-8B × s2`. Vanilla skips ingest,
so this is the fastest cell and decoupled from any backend service.

```bash
cd /path/to/MemArena

# Adjust these to match your local sglang serving setup:
ENDPOINT="http://localhost:30000/v1"
MODEL_TAG="Qwen3-8B"
RUN_DIR="MASim/runs/l_20260408_111046"
OUT_DIR="out/d6_self_probe/sanity_vanilla_8b_s2"

mkdir -p "$OUT_DIR"

python3 eval/cli.py \
    --run-dir "$RUN_DIR" \
    --output-dir "$OUT_DIR" \
    --backend vanilla \
    --model "$MODEL_TAG" \
    --endpoint "$ENDPOINT" \
    --trial-name s2 \
    --stages answer evaluate \
    --d6-arm B \
    --d6-probe-mode self_ego \
    --judge-model openai/gpt-4o-mini \
    --judge-endpoint https://openrouter.ai/api/v1 \
    --judge-api-key "$OPENROUTER_API_KEY"
```

Check that:
1. Output ends up under `out/d6_self_probe/sanity_vanilla_8b_s2/`.
2. The original `out/accuracy_memarena_l_baselines_8b/` etc. are untouched.
3. `evaluation_results_*.json` exists in the sanity output dir and has D6
   records with non-empty predictions (e.g., the model answers as if asked
   by the ego agent rather than by a third party).

Quick spot-check from Python:

```bash
python3 - <<'PY'
import json, glob
files = glob.glob("out/d6_self_probe/sanity_vanilla_8b_s2/**/evaluation_results_*.json", recursive=True)
assert files, "no evaluation_results files produced -- check inference logs"
for p in files:
    d = json.loads(open(p).read())
    d6 = [x for x in d.get("details", []) if x.get("question_id","").startswith("d4_perm")]
    print(p, "d4_perm records:", len(d6))
    for r in d6[:1]:
        print("  example pred:", repr((r.get("prediction") or "")[:120]))
        print("  policy_category:", r.get("policy_category"))
PY
```

Expected: ~200 d4_perm records, predictions look plausible, every record has
a `policy_category` set to one of `DISCLOSE_CORRECT`, `DISCLOSE_WRONG`,
`DONT_KNOW`, `REFUSE`, `OTHER`, or `PARSE_ERROR`. If any of those are wrong,
stop here and ping back.

## Step 2 — full self-probe inference (75 cells)

Once the sanity cell looks right, scale to all 75 cells. The matrix is the
same as the existing main grid:

| backends            | readers                                   | trials       |
|---------------------|-------------------------------------------|--------------|
| vanilla, oracle, baseline_simplerag (=BM25-RAG), memobase, memsearch | Qwen3-0.6B, Llama-3.2-3B, Mistral-7B-v0.3, Qwen3-8B, Qwen3-32B-AWQ | s2, s3, s4 |

Two practical paths:

### Path A — adapt the existing `run_l_baselines.sh` / `run_l_all_models.sh`

These already iterate over the full matrix. Add `--d6-probe-mode self_ego`
to the underlying `eval/cli.py` invocation (look for the `python3 eval/cli.py`
block in `scripts/run_eval_matrix.sh` — that's the inner loop) AND change
the `OUT_DIR` to point under `out/d6_self_probe/`. Concretely, the simplest
diff is to copy `run_l_baselines.sh` and `run_l_all_models.sh` to
`run_l_baselines_self_probe.sh` / `run_l_all_models_self_probe.sh`, then add:

- to the inner cli.py call: `--d6-probe-mode self_ego --d6-arm B`
- to the output-dir derivation: prefix with `out/d6_self_probe/...` instead
  of `out/accuracy_memarena_l_*`

### Path B — explicit per-cell loop (simpler if you only need one pass)

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /path/to/MemArena

declare -A ENDPOINTS
ENDPOINTS[Qwen3-0.6B]="http://localhost:30000/v1"
ENDPOINTS[Llama-3.2-3B]="http://localhost:30001/v1"
ENDPOINTS[Mistral-7B]="http://localhost:30002/v1"
ENDPOINTS[Qwen3-8B]="http://localhost:30003/v1"
ENDPOINTS[Qwen3-32B-AWQ]="http://localhost:30004/v1"

RUN_DIR="MASim/runs/l_20260408_111046"

for model in Qwen3-0.6B Llama-3.2-3B Mistral-7B Qwen3-8B Qwen3-32B-AWQ; do
  for backend in vanilla oracle baseline_simplerag memobase memsearch; do
    for trial in s2 s3 s4; do
      out="out/d6_self_probe/${backend}_${model//[\.-]/_}/${trial}"
      mkdir -p "$out"
      python3 eval/cli.py \
        --run-dir "$RUN_DIR" \
        --output-dir "$out" \
        --backend "$backend" \
        --model "$model" \
        --endpoint "${ENDPOINTS[$model]}" \
        --trial-name "$trial" \
        --stages answer \
        --d6-arm B \
        --d6-probe-mode self_ego &
    done
    wait
  done
done
```

Notes:
- We pass `--stages answer` only (no `evaluate`); the judge runs separately
  in step 3 via the dedicated rubric script.
- Use whatever serving topology you already use for the main eval; the
  example above is a placeholder.
- Memobase / MemSearch may need their per-cell isolated stacks; reuse the
  pattern from `run_l_all_models.sh` for those backends.

Time estimate (your hardware): each cell ~5-30 min depending on reader and
backend; full run a few hours to a day. Logs to `nohup` recommended.

## Step 3 — judging (cheap, on H100, can be in foreground)

After all inference completes, judge the self-probe answers:

```bash
cd /path/to/MemArena
export OPENROUTER_API_KEY=<your key>   # or already in .env
python3 scripts/judge_d6_self_probe.py
```

This:
- Discovers all `evaluation_results_*.json` under `out/d6_self_probe/`.
- For each, calls openai/gpt-4o-mini through OpenRouter on the d4_perm
  records and writes `policy_category`, `leaked_fact_in_output`,
  `correct`, `score`, and `d6_probe_mode="self_ego"` back into each record.
- Backs up each original file to `*_legacy.json` once on first touch.
- Idempotent: re-running skips already-judged records.

Cost: ~$3 in OpenRouter credits for the full 15K records. Wall time ~25 min
with default `--workers 16`.

To dry-run first (no API calls):

```bash
python3 scripts/judge_d6_self_probe.py --dry-run --max-files 3
```

## Step 4 — verify and report back

After judging, paste the final summary block (the line starting with
`label_counts={...}`) back to the analysis side. Expected shape (rough
order-of-magnitude only — the actual numbers are what we want to learn):
- DISCLOSE_CORRECT (the system disclosed to ego): meaningful share for
  Oracle (~50-95%), low for non-Oracle backends.
- DONT_KNOW: large share for Vanilla/RAG/MemSearch (recall failure dominates).
- REFUSE on the self-probe: should be near zero (model should not refuse
  ego). If REFUSE on self-probe is high, that itself is a finding.

Also include:
1. Number of cells completed vs expected (75) and any failed/skipped cells.
2. Wall time per phase.
3. Any cells where the inference logs showed errors or the JSON files are
   suspiciously small (<10 KB).

We will then join the third_party set with the self_ego set on
`(qid, backend, reader, seed)` and update Finding 1, the case-study tables,
and the relevant figures.

## Troubleshooting

- **`KeyError: ego_agent_id`**: the metadata loader is not surfacing
  `ego_agent_id`. Check `eval/src/masim_loader.py` line ~589 and confirm
  `ego_agent_id` is in the allowed metadata fields. The fallback chain in
  the new code (`ego_agent_id` -> `query_agent` -> `asker_agent_id` ->
  `instance_query_agent`) means this should not crash even if `ego_agent_id`
  is missing, but it would silently produce a probe identical to
  `third_party`. Sanity check from the prediction text whether the model
  appears to think the asker is ego.
- **`OPENROUTER_API_KEY not found`**: put it in `.env` at repo root or
  export it in your shell.
- **Judge labels look wrong**: the rubric is in
  `eval/src/scoring.py:_ARMB_JUDGE_SYSTEM`. Sanity test it by running
  `python3 scripts/sanity_d6_judge.py` (12 hand-picked cases, expects 12/12
  pass).
