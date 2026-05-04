"""Phase 3: full D6 rerun on the 75 main_5x5x3 cells using the new 5-label rubric.

Reads predictions from existing ``evaluation_results_*.json`` files (the
``prediction`` field), runs the new judge, writes back updated label / reason /
correctness / fact_in_output for every d4_perm_* record. Backs up the
original file to ``*_legacy.json`` once per cell on first touch.

Inputs:
  - ``experiments_index.csv`` resolved via paper_data.load_all_cells(
      include_ablation=False).
  - ``OPENROUTER_API_KEY`` / ``OPENROUTER_BASE_URL`` from .env.

Concurrency: 16 worker threads, retry up to 3x with backoff. Empty
predictions are short-circuited (no API call) per ``_score_d4_armB``.

Idempotency: if a record already has the new vocabulary in ``policy_category``
(one of D6_LABELS) and a non-empty ``rationale_v2``, skip it. So the script
is safe to rerun on partial output.

Run:
    python3 scripts/rerun_d6_judge.py [--max-cells N] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_env() -> dict[str, str]:
    env_path = REPO_ROOT / ".env"
    out: dict[str, str] = {}
    if not env_path.exists():
        return out
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _make_client():
    env = _load_env()
    api_key = env.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    base_url = env.get("OPENROUTER_BASE_URL") or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not found in .env or env vars")
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=base_url)


def _judge_record(
    client: Any,
    model: str,
    rec: dict,
    gold_fact: str,
    max_retries: int = 3,
) -> tuple[str, bool, str, str]:
    """Run new D6 judge on one record. Returns (label, fact_in_output,
    rationale, reason_token). Empty prediction is short-circuited to
    OTHER without API call."""
    from eval.src.scoring import (
        _ARMB_JUDGE_SYSTEM,
        _build_armB_user,
        _parse_armB_response,
        _strip_thinking,
        _armB_correctness,
    )
    raw_pred = rec.get("prediction") or ""
    pred = _strip_thinking(raw_pred.strip())
    expected_mode = str(rec.get("expected_answer_mode") or "").lower()

    if not pred:
        ok, reason = _armB_correctness("OTHER", expected_mode)
        return "OTHER", False, "", reason

    user_msg = _build_armB_user(
        prediction=pred,
        question=rec.get("question_id", ""),  # qid in lieu of original question text
        gt={"fact": gold_fact},
    )
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _ARMB_JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            raw = resp.choices[0].message.content or ""
            parsed = _parse_armB_response(raw)
            label = parsed["category"]
            fact_in = bool(parsed["leaked_fact_in_output"])
            rationale = parsed["rationale"]
            ok, reason = _armB_correctness(label, expected_mode)
            return label, fact_in, rationale, reason
        except Exception as e:
            last_err = e
            time.sleep(1.5 ** attempt)
    # All retries failed
    ok, reason = _armB_correctness("PARSE_ERROR", expected_mode)
    return "PARSE_ERROR", False, f"judge_error: {last_err}"[:200], reason


@dataclass
class CellStats:
    cell_key: tuple
    n_total: int = 0
    n_done: int = 0
    n_skipped: int = 0
    n_error: int = 0
    label_counts: dict = None

    def __post_init__(self):
        self.label_counts = {}


def process_cell(
    json_path: Path,
    cell_key: tuple,
    client: Any,
    model: str,
    dry_run: bool = False,
) -> CellStats:
    """Re-judge all d4_perm_* records in one evaluation_results JSON file.
    Backs up original to *_legacy.json on first touch. Idempotent."""
    from eval.src.scoring import D6_LABELS

    stats = CellStats(cell_key=cell_key)

    backup_path = json_path.with_name(json_path.stem + "_legacy.json")
    data = json.loads(json_path.read_text())
    details = data.get("details", [])

    # Identify d4_perm records
    d4_records = [
        (i, d) for i, d in enumerate(details)
        if str(d.get("question_id", "")).startswith("d4_perm")
    ]
    stats.n_total = len(d4_records)
    if not d4_records:
        return stats

    # First-touch backup (only if not already backed up)
    if not backup_path.exists() and not dry_run:
        shutil.copyfile(json_path, backup_path)

    # Process each record
    for i, rec in d4_records:
        # Idempotency: skip if already in new vocabulary
        existing_cat = str(rec.get("policy_category") or "").upper()
        if existing_cat in D6_LABELS and rec.get("rationale_v2"):
            stats.n_skipped += 1
            stats.label_counts[existing_cat] = stats.label_counts.get(existing_cat, 0) + 1
            continue

        # Pull gold fact from record (it's already in the existing details)
        gold_fact = str(rec.get("gold_answer") or "")
        if dry_run:
            label = "DRY_RUN"
            fact_in = False
            rationale = ""
            reason = ""
        else:
            label, fact_in, rationale, reason = _judge_record(client, model, rec, gold_fact)

        # Compute correctness
        from eval.src.scoring import _armB_correctness
        expected_mode = str(rec.get("expected_answer_mode") or "").lower()
        ok, _reason = _armB_correctness(label, expected_mode)

        # Update record in place
        if not dry_run:
            rec["policy_category"] = label  # new value vocabulary
            rec["leaked_fact_in_output"] = fact_in
            rec["rationale_v2"] = rationale  # new diagnostic field
            rec["reason"] = reason  # update reason token to new vocabulary
            rec["correct"] = ok
            rec["score"] = 1.0 if ok else 0.0

        stats.n_done += 1
        if label == "PARSE_ERROR":
            stats.n_error += 1
        stats.label_counts[label] = stats.label_counts.get(label, 0) + 1

    if not dry_run:
        json_path.write_text(json.dumps(data, indent=2))

    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cells", type=int, default=None,
                        help="Limit to first N cells (for debugging)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't call API, don't write files")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    from memarena.figures.paper_data import load_all_cells
    grid = load_all_cells(include_ablation=False)
    cells = sorted(grid.items())
    if args.max_cells:
        cells = cells[: args.max_cells]
    print(f"Re-judging {len(cells)} cells via openai/gpt-4o-mini (workers={args.workers}, dry_run={args.dry_run})")

    if args.dry_run:
        client = None
    else:
        client = _make_client()
    model = "openai/gpt-4o-mini"

    t0 = time.time()
    overall = {
        "cells_done": 0, "records_done": 0, "records_skipped": 0,
        "records_error": 0, "label_counts": {},
    }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_cell, cell.source_path, key, client, model, args.dry_run): key
            for key, cell in cells
        }
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                s: CellStats = fut.result()
            except Exception as e:
                print(f"  cell {key} FAILED with {type(e).__name__}: {e}")
                continue
            overall["cells_done"] += 1
            overall["records_done"] += s.n_done
            overall["records_skipped"] += s.n_skipped
            overall["records_error"] += s.n_error
            for k, v in (s.label_counts or {}).items():
                overall["label_counts"][k] = overall["label_counts"].get(k, 0) + v

            elapsed = time.time() - t0
            rate = overall["records_done"] / elapsed if elapsed > 0 else 0
            est_total = (overall["records_done"] + (len(cells) - overall["cells_done"]) * 200)
            eta_s = (est_total - overall["records_done"]) / rate if rate > 0 else 0
            print(
                f"[{overall['cells_done']:>2}/{len(cells)}] {key}  "
                f"+{s.n_done}done +{s.n_skipped}skip +{s.n_error}err  "
                f"labels={s.label_counts}  "
                f"elapsed={elapsed:.0f}s rate={rate:.1f}/s eta={eta_s:.0f}s"
            )

    elapsed = time.time() - t0
    print()
    print("=" * 70)
    print(f"DONE in {elapsed:.0f}s")
    print(f"  cells_done={overall['cells_done']}/{len(cells)}")
    print(f"  records_done={overall['records_done']}")
    print(f"  records_skipped={overall['records_skipped']}")
    print(f"  records_error={overall['records_error']}")
    print(f"  label_counts={overall['label_counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
