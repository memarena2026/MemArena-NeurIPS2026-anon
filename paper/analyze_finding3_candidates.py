#!/usr/bin/env python3
"""Investigate candidate "core Finding 2" directions on the existing corpus.

Three independent analyses on
``l_20260408_111046`` results:

A. **Six-dim coupling.** Pairwise Spearman correlation between D1..D6
   accuracies across every (backend, reader, seed) cell.  If the six
   axes are weakly coupled, no single scalar (e.g. Avg) can summarise
   memory ability --- a methodological finding.

B. **Memobase exceeds Oracle.** Enumerate (reader, dim) cells where
   Memobase mean accuracy is *above* Oracle mean accuracy on the same
   reader.  Magnitudes and counts.  Tests the "profile aggregation
   creates information density beyond per-instance evidence" hypothesis.

C. **Reading-bound dims.** Per-paper-dim Oracle accuracy across the
   five readers.  Identify dims that *Oracle Q3-32B* still misses ---
   these are reader-capacity ceilings, not retrieval failures.

Outputs
-------
Stdout markdown summary;  ``paper/history/finding3_candidates.json``
machine-readable.

Run from the paper directory::

    python3 analyze_finding3_candidates.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_data import (  # noqa: E402
    BACKEND_TEX,
    DEFAULT_RUN,
    DIM_KEYS_PAPER,
    DIM_SHORT,
    MODEL_ORDER,
    MODEL_TEX,
    SEEDS,
    aggregate_seeds,
    load_all_cells,
)

OUT_JSON = Path(__file__).resolve().parent / "history" / "finding3_candidates.json"

BACKENDS_FOR_COUPLING = ["vanilla", "rag", "memobase", "memos", "oracle"]
PAPER_DIM_ORDER = DIM_KEYS_PAPER[:6]   # D1..D6 (D7 = legacy)


# --- helpers ---------------------------------------------------------------


def _rank(xs: list[float]) -> list[float]:
    """Average-rank for tied values (Spearman convention)."""
    paired = sorted(enumerate(xs), key=lambda p: p[1])
    out = [0.0] * len(xs)
    i = 0
    while i < len(paired):
        j = i
        while j + 1 < len(paired) and paired[j + 1][1] == paired[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[paired[k][0]] = avg_rank
        i = j + 1
    return out


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx2 = sum((xs[i] - mx) ** 2 for i in range(n))
    dy2 = sum((ys[i] - my) ** 2 for i in range(n))
    if dx2 == 0 or dy2 == 0:
        return float("nan")
    return num / (dx2 ** 0.5 * dy2 ** 0.5)


def _spearman(xs: list[float], ys: list[float]) -> float:
    return _pearson(_rank(xs), _rank(ys))


def _cell_mean_per_dim(grid, backend, model):
    """{paper_dim -> mean accuracy across stochastic seeds}, or None."""
    accs_by_dim: dict[str, list[float]] = defaultdict(list)
    for seed in SEEDS:
        cell = grid.get((seed, backend, model))
        if cell is None:
            continue
        for dim, acc in cell.accuracy_by_paper_dim().items():
            if acc is not None:
                accs_by_dim[dim].append(acc)
    if not accs_by_dim:
        return None
    return {dim: (sum(v) / len(v)) for dim, v in accs_by_dim.items() if v}


# --- A. Six-dim coupling ---------------------------------------------------


def analyse_dim_coupling(grid):
    """Pairwise Spearman correlation between paper-dim accuracies across cells."""
    cells = []
    for backend in BACKENDS_FOR_COUPLING:
        for model in MODEL_ORDER:
            per_dim = _cell_mean_per_dim(grid, backend, model)
            if per_dim is None:
                continue
            row = [per_dim.get(d) for d in PAPER_DIM_ORDER]
            if any(v is None for v in row):
                continue
            cells.append({"backend": backend, "model": model, "dim": row})

    n = len(cells)
    matrix: dict[tuple[str, str], float] = {}
    for i, j in combinations(range(6), 2):
        xs = [c["dim"][i] for c in cells]
        ys = [c["dim"][j] for c in cells]
        rho = _spearman(xs, ys)
        matrix[(DIM_SHORT[i], DIM_SHORT[j])] = rho

    # Per-dim "isolation": max |rho| of that dim with any other dim.
    isolation = {}
    for i in range(6):
        rhos = []
        for j in range(6):
            if i == j:
                continue
            key = (DIM_SHORT[min(i, j)], DIM_SHORT[max(i, j)])
            rhos.append(abs(matrix[key]))
        isolation[DIM_SHORT[i]] = max(rhos)

    return {"n_cells": n, "pairwise_spearman": matrix, "isolation": isolation}


# --- B. Memobase exceeds Oracle ---------------------------------------------


def analyse_memo_vs_oracle(grid):
    """Cells where Memobase mean Avg > Oracle mean Avg on the same reader."""
    findings = []
    for model in MODEL_ORDER:
        memo = _cell_mean_per_dim(grid, "memobase", model)
        oracle = _cell_mean_per_dim(grid, "oracle", model)
        if memo is None or oracle is None:
            continue

        # paper-D Avg (mean over D1..D5; legacy convention from main_SML.tex)
        memo_avg = statistics.mean(memo[d] for d in PAPER_DIM_ORDER[:5])
        oracle_avg = statistics.mean(oracle[d] for d in PAPER_DIM_ORDER[:5])

        delta_avg = (memo_avg - oracle_avg) * 100  # pp
        per_dim_deltas = {
            DIM_SHORT[i]: (memo[d] - oracle[d]) * 100
            for i, d in enumerate(PAPER_DIM_ORDER)
        }
        findings.append({
            "model":   MODEL_TEX[model],
            "memo_avg_pp":   round(memo_avg * 100, 2),
            "oracle_avg_pp": round(oracle_avg * 100, 2),
            "delta_avg_pp":  round(delta_avg, 2),
            "per_dim_delta_pp": {k: round(v, 2) for k, v in per_dim_deltas.items()},
        })
    return findings


# --- C. Reading-bound dims --------------------------------------------------


def analyse_reading_bound(grid):
    """Per-paper-dim Oracle accuracy across readers + within-largest gap."""
    rows = []
    for model in MODEL_ORDER:
        per_dim = _cell_mean_per_dim(grid, "oracle", model)
        if per_dim is None:
            continue
        rows.append({
            "model": MODEL_TEX[model],
            "per_dim_pp": {
                DIM_SHORT[i]: round(per_dim[d] * 100, 2)
                for i, d in enumerate(PAPER_DIM_ORDER)
            },
        })

    # For each dim, what's Oracle accuracy at the LARGEST reader (Q3-32B)?
    largest_model_label = MODEL_TEX[MODEL_ORDER[-1]]
    largest = next((r for r in rows if r["model"] == largest_model_label), None)
    if largest is None:
        return {"per_reader": rows, "largest_reader_ceiling": None}

    # Sort dims by Oracle Q3-32B accuracy ascending — bottom = reader-bound.
    ranked = sorted(largest["per_dim_pp"].items(), key=lambda x: x[1])
    return {
        "per_reader": rows,
        "largest_reader_ceiling": {
            "model": largest_model_label,
            "ranked_low_to_high": ranked,
        },
    }


# --- main ------------------------------------------------------------------


def _print_md(out: dict) -> None:
    A = out["A_dim_coupling"]
    print(f"\n## A. Six-dim coupling  ({A['n_cells']} cells)")
    print()
    print("Pairwise Spearman ρ (D1..D6):")
    print()
    print("| | D1 | D2 | D3 | D4 | D5 | D6 |")
    print("|---|---|---|---|---|---|---|")
    for i, d in enumerate(DIM_SHORT[:6]):
        row = f"| **{d}** "
        for j, dj in enumerate(DIM_SHORT[:6]):
            if i == j:
                row += "| --- "
            else:
                key = (DIM_SHORT[min(i, j)], DIM_SHORT[max(i, j)])
                rho = A["pairwise_spearman"][key]
                row += f"| {rho:+.2f} "
        row += "|"
        print(row)
    print()
    print(f"Per-dim isolation (max |ρ| with any other dim):")
    for d, m in A["isolation"].items():
        flag = "  ← isolated" if m < 0.30 else ""
        print(f"  {d}: max |ρ| = {m:.2f}{flag}")

    B = out["B_memo_vs_oracle"]
    print(f"\n\n## B. Memobase vs Oracle (per-reader, mean over stochastic seeds)\n")
    print("| Reader | Memobase Avg | Oracle Avg | ΔAvg (Memo−Oracle) | per-dim Δ (D1..D6) |")
    print("|---|---|---|---|---|")
    for r in B:
        d = r["per_dim_delta_pp"]
        per_dim_str = " / ".join(f"{d[k]:+.1f}" for k in DIM_SHORT[:6])
        flag = " ★" if r["delta_avg_pp"] > 0 else ""
        print(f"| {r['model']} | {r['memo_avg_pp']:.1f} | {r['oracle_avg_pp']:.1f} "
              f"| {r['delta_avg_pp']:+.2f}{flag} | {per_dim_str} |")

    C = out["C_reading_bound"]
    print(f"\n\n## C. Reading-bound dims (Oracle accuracy by reader, %)\n")
    print("| Reader | " + " | ".join(DIM_SHORT[:6]) + " |")
    print("|---|" + "|".join("---" for _ in range(6)) + "|")
    for r in C["per_reader"]:
        d = r["per_dim_pp"]
        print(f"| {r['model']} | " + " | ".join(f"{d[k]:.1f}" for k in DIM_SHORT[:6]) + " |")
    print()
    if C["largest_reader_ceiling"]:
        ranked = C["largest_reader_ceiling"]["ranked_low_to_high"]
        print(f"At {C['largest_reader_ceiling']['model']} Oracle, ranked low-to-high:")
        for d, v in ranked:
            flag = "  ← reader-bound (<70%)" if v < 70.0 else ""
            print(f"  {d}: {v:.1f}%{flag}")


def main() -> None:
    grid = load_all_cells(DEFAULT_RUN)
    out = {
        "A_dim_coupling":     analyse_dim_coupling(grid),
        "B_memo_vs_oracle":   analyse_memo_vs_oracle(grid),
        "C_reading_bound":    analyse_reading_bound(grid),
    }
    _print_md(out)

    # Coerce tuple keys to strings for JSON.
    out_serial = {
        "A_dim_coupling": {
            "n_cells": out["A_dim_coupling"]["n_cells"],
            "pairwise_spearman": {
                f"{k[0]}-{k[1]}": v
                for k, v in out["A_dim_coupling"]["pairwise_spearman"].items()
            },
            "isolation": out["A_dim_coupling"]["isolation"],
        },
        "B_memo_vs_oracle": out["B_memo_vs_oracle"],
        "C_reading_bound": out["C_reading_bound"],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out_serial, indent=2))
    print(f"\n[finding3] wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
