#!/usr/bin/env python3
"""Verify the output-level failure split behind Finding 1.

This is not an internal causal trace of the reader or retriever.  It checks the
auditable claim used in the paper: on DENY queries, do recall-poor backends avoid
disclosure mainly by producing ``NONE`` (information absent / off topic) rather
than privacy-grounded ``REFUSAL``?

Input: per-item Arm-B D6 re-judgements for seeds s2/s3/s4.
Output: ``paper/history/d6_failure_mechanisms.json`` plus a console table.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    REPO_ROOT
    / "MASim/runs/l_20260408_111046/eval_results/d6_rebuild_2026-04-27"
)
OUT_PATH = Path(__file__).resolve().parent / "history" / "d6_failure_mechanisms.json"

SEEDS = ("s2", "s3", "s4")
BACKEND_ORDER = ("vanilla", "inmem", "oracle", "memobase")
BACKEND_LABEL = {
    "vanilla": "Vanilla",
    "inmem": "BM25-RAG",
    "oracle": "Oracle",
    "memobase": "Memobase",
}
VALID_CATEGORIES = ("REFUSAL", "NONE", "COMPLY")


def _load_counts() -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for seed in SEEDS:
        per_item_dir = SOURCE_ROOT / f"per_item_armB_{seed}"
        if not per_item_dir.exists():
            raise FileNotFoundError(per_item_dir)
        for path in sorted(per_item_dir.glob("*.jsonl")):
            backend = path.name.split("_", 1)[0]
            if backend not in BACKEND_ORDER:
                continue
            with path.open() as fh:
                for line in fh:
                    row = json.loads(line)
                    if row.get("expected_answer_mode") != "deny":
                        continue
                    category = row.get("category", "MISSING")
                    counts[backend]["TOTAL"] += 1
                    counts[backend][category] += 1
    return counts


def _pct(num: int, den: int) -> float:
    return round(100.0 * num / den, 2) if den else 0.0


def _summarize(counts: dict[str, Counter[str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for backend in BACKEND_ORDER:
        c = counts[backend]
        total = c["TOTAL"]
        non_disclosure = c["REFUSAL"] + c["NONE"]
        row = {
            "backend": backend,
            "label": BACKEND_LABEL[backend],
            "n_deny": total,
            "counts": {cat: c[cat] for cat in VALID_CATEGORIES},
            "category_pct": {cat: _pct(c[cat], total) for cat in VALID_CATEGORIES},
            "non_disclosure_pct": _pct(non_disclosure, total),
            "none_share_of_non_disclosures_pct": _pct(c["NONE"], non_disclosure),
        }
        rows.append(row)
    return rows


def main() -> None:
    rows = _summarize(_load_counts())
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "source": str(SOURCE_ROOT.relative_to(REPO_ROOT)),
        "seeds": list(SEEDS),
        "interpretation": (
            "On DENY queries, REFUSAL is a privacy-grounded refusal; NONE is an "
            "information-absent/off-topic response. A high NONE share therefore "
            "supports the output-level claim that non-disclosure is mostly recall "
            "failure rather than access-control compliance."
        ),
        "rows": rows,
    }, indent=2) + "\n")

    print("D6 DENY-pool failure mechanism audit (Arm-B, seeds s2/s3/s4)")
    print("Rates are percentages of DENY queries; NONE share is within non-disclosures.")
    print()
    print(f"{'Backend':<10} {'n':>5} {'REFUSAL':>8} {'NONE':>8} {'COMPLY':>8} {'NONE/non-disc':>14}")
    for row in rows:
        pct = row["category_pct"]
        print(
            f"{row['label']:<10} {row['n_deny']:>5} "
            f"{pct['REFUSAL']:>8.1f} {pct['NONE']:>8.1f} {pct['COMPLY']:>8.1f} "
            f"{row['none_share_of_non_disclosures_pct']:>14.1f}"
        )
    print()
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
