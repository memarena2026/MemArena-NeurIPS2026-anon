#!/usr/bin/env python3
"""Build a 500-instance stratified sample for human-judge calibration.

Source of truth: ``experiments_index.csv`` (the canonical 75-cell main_5x5x3
registry). For each cell we read both the eval_results JSON (judge labels)
and the sibling answer_results JSON (the user-facing question/gold text)
and JOIN them by ``question_id``.

Stratification has two tracks:

* **Binary track** (D2/D3/D4/D5-answer, scored by the LLM judge):
  buckets = (dim, judge_correct), budget = 300.
* **D6 5-label track** (d4_perm_*, judged via the new 5-label rubric):
  buckets = (policy_category, expected_answer_mode), budget = 200.
  Excluded by the old `scoring_method == 'judge'` filter — pulled in here.

Output: ``memarena/human_calibration/sample.json`` consumed by ``server.py``.
We deliberately do NOT regenerate the embedded ``index.html``; the new
HTML is a thin renderer that fetches sample.json + /api/* at runtime.

Usage:
    python3 -m memarena.human_calibration.build_human_calibration \\
        --n 500 --seed 20260504
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_CSV = REPO_ROOT / "experiments_index.csv"

D6_LABELS = ("DISCLOSE_CORRECT", "DISCLOSE_WRONG", "DONT_KNOW", "REFUSE", "OTHER")

# Map legacy qid prefix → paper-facing dimension. The paper consolidates the
# 10 simulator dimensions into D1..D6; only D2..D6 are LLM-judged and thus
# belong in the binary calibration track. D1 (`d5_cloze`) is rule-based
# (`mcq_exact`) and is excluded from human kappa.
LEGACY_TO_PAPER_DIM: dict[str, str] = {
    "d5":  "D1",   # cloze fidelity — rule-based, excluded
    "d6":  "D2",   # metadata completeness
    "d7":  "D3",   # factual QA (standard)
    "d8":  "D3",   # factual QA (temporal)
    "d10": "D3",   # factual QA (counterfactual)
    "d1":  "D4",   # cross-session: conflict
    "d2":  "D4",   # cross-session: anaphora
    "d3":  "D5",   # abstention (only the LLM-judged half is included)
    # d4_perm handled separately as D6.
}


def _legacy_prefix(qid: str) -> str:
    """Extract the legacy ``dN`` prefix (treats ``d4_perm_*`` separately)."""
    if qid.startswith("d4_perm"):
        return "d4_perm"
    m = re.match(r"(d\d+)_", qid)
    return m.group(1) if m else ""


def _paper_dim(qid: str) -> str:
    """Paper-facing dimension or '' if not in scope."""
    if qid.startswith("d4_perm"):
        return "D6"
    return LEGACY_TO_PAPER_DIM.get(_legacy_prefix(qid), "")


def _cell_tag(row: dict) -> str:
    """Compact cell name: <backend>_<model_tag>_<seed_tag>."""
    return f"{row['backend']}_{row['model_tag']}_{row['trial']}"


def load_pool(table: str = "main_5x5x3") -> list[dict]:
    """Walk experiments_index.csv and return a flat list of normalized rows.

    Each row has: id, cell, dim, dim_kind ('binary'|'d6'), query, gold,
    prediction, judge_correct, judge_label (D6 only), judge_reason,
    policy_expected (D6 only), expected_answer_mode (D6 only).
    """
    rows: list[dict] = []
    if not INDEX_CSV.exists():
        raise SystemExit(f"missing {INDEX_CSV}")

    with INDEX_CSV.open() as fh:
        cells = [r for r in csv.DictReader(fh) if r["table"] == table]

    if not cells:
        raise SystemExit(f"no rows for table={table} in {INDEX_CSV}")

    for cell in cells:
        eval_path = REPO_ROOT / cell["json_path"]
        if not eval_path.exists():
            print(f"  skip missing {cell['json_path']}", file=sys.stderr)
            continue
        ans_path = eval_path.parent / eval_path.name.replace(
            "evaluation_results", "answer_results"
        ).replace("_judge_remote", "").replace("_judge_local", "")
        # Some cells use a slightly different naming — fall back to glob.
        if not ans_path.exists():
            cands = list(eval_path.parent.glob("answer_results_*.json"))
            ans_path = cands[0] if cands else None
        if ans_path is None or not ans_path.exists():
            print(f"  skip {cell['json_path']}: no answer_results sibling", file=sys.stderr)
            continue

        try:
            ev = json.loads(eval_path.read_text())
            ans_list = json.loads(ans_path.read_text())
        except Exception as exc:
            print(f"  skip {eval_path}: {exc}", file=sys.stderr)
            continue

        ans_by_qid = {r["question_id"]: r for r in ans_list if r.get("question_id")}
        cell_tag = _cell_tag(cell)

        for r in ev.get("details", []):
            qid = r.get("question_id")
            if not qid:
                continue
            ans = ans_by_qid.get(qid, {})
            paper_dim = _paper_dim(qid)
            if not paper_dim or paper_dim == "D1":
                # Out of paper scope, or D1 cloze (rule-based — excluded from kappa).
                continue
            is_d6 = paper_dim == "D6"

            if is_d6:
                pol = str(r.get("policy_category") or "").upper()
                if pol not in D6_LABELS:
                    continue  # un-judged or malformed
                if not r.get("rationale_v2"):
                    continue
                rows.append({
                    "id": qid,
                    "cell": cell_tag,
                    "dim": "D6",
                    "dim_kind": "d6",
                    "query": ans.get("question") or "",
                    "gold": r.get("gold_answer") or ans.get("answer") or "",
                    "prediction": r.get("prediction") or ans.get("prediction") or "",
                    "judge_correct": bool(r.get("correct")),
                    "judge_label": pol,
                    "judge_reason": r.get("rationale_v2") or r.get("reason") or "",
                    "policy_expected": str(r.get("policy_expected") or "").upper(),
                    "expected_answer_mode": str(r.get("expected_answer_mode") or "").lower(),
                    "leaked_fact_in_output": bool(r.get("leaked_fact_in_output", False)),
                })
                continue

            # Binary track: keep only LLM-judged items. The reason field is the
            # only reliable signal in the current schema (`scoring_method` is
            # absent / always None). LLM judging emits ``evidence_judge:...``
            # while rule paths emit ``mcq_exact``, ``refusal_detection``, etc.
            reason = str(r.get("reason") or "")
            if not reason.startswith("evidence_judge"):
                continue
            correct = r.get("correct")
            if correct is None:
                continue
            rows.append({
                "id": qid,
                "cell": cell_tag,
                "dim": paper_dim,
                "dim_kind": "binary",
                "query": ans.get("question") or "",
                "gold": r.get("gold_answer") or ans.get("answer") or "",
                "prediction": r.get("prediction") or ans.get("prediction") or "",
                "judge_correct": bool(correct),
                "judge_label": "CORRECT" if correct else "INCORRECT",
                "judge_reason": reason,
                "policy_expected": "",
                "expected_answer_mode": "",
                "leaked_fact_in_output": False,
            })
    return rows


def stratified_sample(
    rows: list[dict],
    rng: random.Random,
    *,
    n_d6: int,
    n_binary: int,
) -> list[dict]:
    """Two-track stratified sample with global qid dedup.

    A given question_id may appear in multiple cells (different
    backend/reader/seed combinations of the same question). We dedupe by id
    first (random cell tie-break) so the label store, which is keyed by id,
    has no collisions.
    """
    by_id: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_id[r["id"]].append(r)
    deduped = [rng.choice(v) for v in by_id.values()]

    d6 = [r for r in deduped if r["dim_kind"] == "d6"]
    bn = [r for r in deduped if r["dim_kind"] == "binary"]

    def _stratify(pool: list[dict], key, budget: int) -> list[dict]:
        buckets = defaultdict(list)
        for r in pool:
            buckets[key(r)].append(r)
        keys = sorted(buckets.keys())
        if not keys:
            return []
        per = max(1, budget // len(keys))
        chosen: list[dict] = []
        leftovers: list[dict] = []
        for k in keys:
            rng.shuffle(buckets[k])
            take = min(per, len(buckets[k]))
            chosen.extend(buckets[k][:take])
            if len(buckets[k]) > take:
                leftovers.extend(buckets[k][take:])
        rng.shuffle(leftovers)
        if len(chosen) < budget:
            chosen.extend(leftovers[: budget - len(chosen)])
        rng.shuffle(chosen)
        return chosen[:budget]

    d6_sample = _stratify(
        d6,
        lambda r: (r["judge_label"], r["expected_answer_mode"]),
        n_d6,
    )
    bn_sample = _stratify(
        bn,
        lambda r: (r["dim"], bool(r["judge_correct"])),
        n_binary,
    )

    sample = d6_sample + bn_sample
    rng.shuffle(sample)
    return sample


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--n-d6", type=int, default=200,
                    help="D6 5-label track budget (default 200 = 5 labels x ALLOW/DENY x 20)")
    ap.add_argument("--seed", type=int, default=20260504)
    ap.add_argument("--out", default=None,
                    help="default: memarena/human_calibration/sample.json")
    ap.add_argument("--table", default="main_5x5x3")
    args = ap.parse_args()

    if args.n_d6 > args.n:
        raise SystemExit("--n-d6 cannot exceed --n")
    n_binary = args.n - args.n_d6

    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "memarena/human_calibration/sample.json"
    )

    rows = load_pool(table=args.table)
    print(f"loaded {len(rows)} rows from table={args.table}", file=sys.stderr)
    counts = defaultdict(int)
    for r in rows:
        counts[r["dim"]] += 1
    print(f"  per-dim: {dict(counts)}", file=sys.stderr)

    rng = random.Random(args.seed)
    sample = stratified_sample(rows, rng, n_d6=args.n_d6, n_binary=n_binary)
    print(f"sampled {len(sample)} rows "
          f"(d6={sum(1 for r in sample if r['dim_kind']=='d6')}, "
          f"binary={sum(1 for r in sample if r['dim_kind']=='binary')})",
          file=sys.stderr)

    metadata = {
        "table": args.table,
        "n": len(sample),
        "n_d6": sum(1 for r in sample if r["dim_kind"] == "d6"),
        "n_binary": sum(1 for r in sample if r["dim_kind"] == "binary"),
        "seed": args.seed,
        "stratification_d6": "(policy_category, expected_answer_mode)",
        "stratification_binary": "(dim, judge_correct)",
        "d6_labels": list(D6_LABELS),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"metadata": metadata, "sample": sample}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out_path} ({out_path.stat().st_size/1024:.1f} KB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
