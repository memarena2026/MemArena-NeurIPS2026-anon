#!/usr/bin/env python3
"""F2 mechanism quantification: per-paper-D, do Vanilla→Oracle and
Vanilla→Memobase gaps tell the same story?

Hypothesis: If Memobase's gain over Vanilla TRACKS Oracle's gain
over Vanilla (parallel curves), Memobase is just a retrieval improvement.
If Memobase's gain MATCHES Oracle on direct-lookup dims but EXCEEDS Oracle
on integrative dims, Memobase's lift is specifically from cross-session
aggregation that even per-instance Oracle evidence cannot match.

Output: stdout markdown table; ``paper/history/f2_mechanism.json``.

Run from paper/ ::

    python3 analyze_f2_mechanism.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_data import (  # noqa: E402
    DEFAULT_RUN, DIM_KEYS_PAPER, DIM_SHORT, MODEL_ORDER, MODEL_TEX,
    SEEDS, load_all_cells,
)

OUT = Path(__file__).resolve().parent / "history" / "f2_mechanism.json"

PAPER_DIM_ORDER = DIM_KEYS_PAPER[:6]


def _cell_dim(grid, backend, model):
    accs = defaultdict(list)
    for seed in SEEDS:
        cell = grid.get((seed, backend, model))
        if cell is None:
            continue
        for d, v in cell.accuracy_by_paper_dim().items():
            if v is not None:
                accs[d].append(v)
    return {d: sum(v) / len(v) for d, v in accs.items() if v} if accs else None


def _gap(grid, model, dim_idx, ref_backend, target_backend):
    ref = _cell_dim(grid, ref_backend, model)
    tgt = _cell_dim(grid, target_backend, model)
    if ref is None or tgt is None:
        return None
    d = PAPER_DIM_ORDER[dim_idx]
    if d not in ref or d not in tgt:
        return None
    return (tgt[d] - ref[d]) * 100  # pp


def main() -> None:
    grid = load_all_cells(DEFAULT_RUN)

    # For each (model, paper-D), compute:
    #   van_to_oracle = Oracle - Vanilla  (retrieval value)
    #   van_to_memo   = Memobase - Vanilla (structured-memory value)
    #   memo_minus_oracle = Memobase - Oracle (does pre-aggregation
    #     beat per-instance Oracle?)

    rows = []
    for model in MODEL_ORDER:
        for i, d in enumerate(PAPER_DIM_ORDER):
            van_to_or = _gap(grid, model, i, "vanilla", "oracle")
            van_to_mem = _gap(grid, model, i, "vanilla", "memobase")
            mem_minus_or = _gap(grid, model, i, "oracle", "memobase")
            if any(x is None for x in (van_to_or, van_to_mem, mem_minus_or)):
                continue
            rows.append({
                "model": MODEL_TEX[model],
                "dim": DIM_SHORT[i],
                "van_to_oracle_pp": round(van_to_or, 2),
                "van_to_memo_pp": round(van_to_mem, 2),
                "memo_minus_oracle_pp": round(mem_minus_or, 2),
            })

    # Per-dim aggregates across capacity-sufficient readers
    # (exclude Mistral due to schema-compliance regression).
    capacity_sufficient = [MODEL_TEX[m] for m in MODEL_ORDER if m != "7b"]
    dim_summary = {}
    for i, d in enumerate(DIM_SHORT[:6]):
        d_rows = [r for r in rows
                  if r["dim"] == d and r["model"] in capacity_sufficient]
        if not d_rows:
            continue
        dim_summary[d] = {
            "n_models": len(d_rows),
            "van_to_oracle_mean": round(statistics.mean(r["van_to_oracle_pp"] for r in d_rows), 2),
            "van_to_memo_mean":   round(statistics.mean(r["van_to_memo_pp"]   for r in d_rows), 2),
            "memo_minus_oracle_mean": round(statistics.mean(r["memo_minus_oracle_pp"] for r in d_rows), 2),
        }

    # Print markdown.
    print("\n## Per-dim gap analysis (capacity-sufficient readers, excludes Mistral)\n")
    print("| Dim | Vanilla→Oracle | Vanilla→Memobase | Memobase−Oracle | family |")
    print("|---|---|---|---|---|")
    family_lookup = {
        "D1": "direct-lookup",
        "D2": "integrative",
        "D3": "mostly direct-lookup",
        "D4": "integrative",
        "D5": "trust",
        "D6": "trust",
    }
    for d in DIM_SHORT[:6]:
        s = dim_summary.get(d)
        if s is None:
            continue
        flag = "★" if s["memo_minus_oracle_mean"] > 5 else ""
        print(f"| **{d}** | +{s['van_to_oracle_mean']:>6.1f} pp | +{s['van_to_memo_mean']:>6.1f} pp "
              f"| {s['memo_minus_oracle_mean']:+6.1f} pp {flag} | {family_lookup[d]} |")

    # Aggregates across families.
    INTEGRATIVE = ["D2", "D4"]   # paper-D D2 metadata + D4 cross-session
    DIRECT      = ["D1", "D3"]   # paper-D D1 cloze + D3 factual QA

    def _avg(family, key):
        vals = [dim_summary[d][key] for d in family if d in dim_summary]
        return statistics.mean(vals) if vals else None

    family_gap_summary = {}
    for fam_name, fam_dims in [("integrative", INTEGRATIVE), ("direct_lookup", DIRECT)]:
        family_gap_summary[fam_name] = {
            "dims": fam_dims,
            "van_to_oracle_mean": round(_avg(fam_dims, "van_to_oracle_mean"), 2),
            "van_to_memo_mean":   round(_avg(fam_dims, "van_to_memo_mean"),   2),
            "memo_minus_oracle_mean": round(_avg(fam_dims, "memo_minus_oracle_mean"), 2),
        }

    print("\n## Family aggregates\n")
    for fam, s in family_gap_summary.items():
        print(f"  **{fam}** ({', '.join(s['dims'])}):")
        print(f"    Vanilla → Oracle  mean lift = +{s['van_to_oracle_mean']} pp")
        print(f"    Vanilla → Memobase mean lift = +{s['van_to_memo_mean']} pp")
        print(f"    Memobase − Oracle gap        = {s['memo_minus_oracle_mean']:+.1f} pp")

    payload = {
        "rows": rows,
        "dim_summary_capacity_sufficient": dim_summary,
        "family_gap_summary_capacity_sufficient": family_gap_summary,
        "note": "Mistral-7B excluded from aggregates due to schema-compliance regression in Memobase's extractor (App. memsys-impl).",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\n[f2-mech] wrote {OUT}")


if __name__ == "__main__":
    main()
