#!/usr/bin/env python3
"""Empirical verification of Finding C — six-dimensional decoupling across cells.

Hypothesis (Finding C, candidate)
---------------------------------
MemArena's six dimensions D1..D6 are sufficiently decoupled across (backend,
reader) cells that no single scalar can summarise a memory system's behaviour.
Backends can be near-saturated on one dimension while near-zero on another,
exposing systems that look "capable" on aggregate metrics while failing on a
critical axis (esp. trustworthiness / permission).

Sub-claims
----------
1. Pairwise correlations across D1..D6 over all (backend, reader) cells:
   max |rho| <= 0.40.
2. D6 (permission) is near-orthogonal: |rho(D6, D_i)| <= 0.10 for all i != 6.
3. There exists at least one backend (probably Vanilla and BM25-RAG) with high
   D1 (cloze >= 0.5) and near-zero D6 (<= 0.10), demonstrating that recall-
   strong systems can be permission-broken.
4. Conversely, structured-memory backends do NOT pay a strong recall penalty
   for higher D6 -- i.e., the orthogonality is real, not an artifact of a
   single dominant tradeoff.
5. Quantify the consequence: if we ranked backends by D1 alone vs by D6 alone,
   would the rankings disagree? Compute Kendall's tau.

Unit of analysis
----------------
The existing reference table ``paper/tables/dim_correlation_0419.tex`` reports
correlations across *agents* within one cell (Oracle-32B). This script
computes correlations across *(backend, reader) cells* aggregated over seeds
s2/s3/s4. Both are valid -- they answer different questions -- and we surface
both numbers in the stdout for cross-check.

Usage
-----
    python paper/verify_findingC_dim_orthogonality.py

Stdlib + numpy + pandas + scipy only.
"""

from __future__ import annotations

import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Make the in-repo paper module importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "paper"))

from paper_data import (  # noqa: E402
    BACKEND_ORDER,
    DIM_KEYS_PAPER,
    MODEL_ORDER,
    NEW_FROM_OLD,
    SEEDS,
    aggregate_seeds,
    load_all_cells,
)

# Paper-facing dimension labels (D1..D6 only -- drop D7 legacy bucket).
PAPER_DIMS = DIM_KEYS_PAPER[:6]
PAPER_DIM_TEX = {
    "d1_cloze": "D1 (Cloze)",
    "d2_metadata": "D2 (Metadata)",
    "d3_factual_qa": "D3 (Factual QA)",
    "d4_cross_session": "D4 (Cross-Sess)",
    "d5_abstention": "D5 (Abstention)",
    "d6_permission": "D6 (Permission)",
}
SHORT = ["D1", "D2", "D3", "D4", "D5", "D6"]


def hr(width: int = 80) -> str:
    return "-" * width


def heading(text: str, char: str = "=") -> None:
    print(char * 80)
    print(text)
    print(char * 80)


def build_score_matrix() -> pd.DataFrame:
    """Return DataFrame indexed by (backend, model) with columns D1..D6.

    Each cell is the mean over seeds s2/s3/s4 of the per-paper-dim macro-mean
    accuracy. Cells where every contributing seed is missing are dropped.
    """
    grid = load_all_cells()
    rows = []
    for backend in BACKEND_ORDER:
        for model in MODEL_ORDER:
            row = {"backend": backend, "model": model}
            any_present = False
            for paper_dim in PAPER_DIMS:
                stat = aggregate_seeds(
                    grid,
                    backend,
                    model,
                    lambda c, pd=paper_dim: c.accuracy_by_paper_dim().get(pd),
                )
                if stat.n == 0 or math.isnan(stat.mean):
                    row[paper_dim] = np.nan
                else:
                    row[paper_dim] = stat.mean
                    any_present = True
            row["n_seeds"] = sum(
                1
                for seed in SEEDS
                if grid.get((seed, backend, model)) is not None
            )
            if any_present:
                rows.append(row)
    df = pd.DataFrame(rows)
    return df


def pairwise_corr(df: pd.DataFrame, method: str) -> pd.DataFrame:
    cols = PAPER_DIMS
    M = df[cols].to_numpy()
    out = np.full((len(cols), len(cols)), np.nan)
    p_out = np.full_like(out, np.nan)
    for i in range(len(cols)):
        for j in range(len(cols)):
            xi = M[:, i]
            xj = M[:, j]
            mask = ~(np.isnan(xi) | np.isnan(xj))
            if mask.sum() < 3:
                continue
            if method == "pearson":
                r, p = stats.pearsonr(xi[mask], xj[mask])
            elif method == "spearman":
                r, p = stats.spearmanr(xi[mask], xj[mask])
            else:
                raise ValueError(method)
            out[i, j] = r
            p_out[i, j] = p
    rho = pd.DataFrame(out, index=SHORT, columns=SHORT)
    pmat = pd.DataFrame(p_out, index=SHORT, columns=SHORT)
    return rho, pmat


def max_offdiag(rho: pd.DataFrame) -> tuple[str, str, float]:
    best = ("", "", 0.0)
    for i, a in enumerate(SHORT):
        for j, b in enumerate(SHORT):
            if i >= j:
                continue
            v = rho.iloc[i, j]
            if pd.isna(v):
                continue
            if abs(v) > abs(best[2]):
                best = (a, b, float(v))
    return best


def min_offdiag(rho: pd.DataFrame) -> tuple[str, str, float]:
    best = (None, None, None)
    for i, a in enumerate(SHORT):
        for j, b in enumerate(SHORT):
            if i >= j:
                continue
            v = rho.iloc[i, j]
            if pd.isna(v):
                continue
            if best[2] is None or abs(v) < abs(best[2]):
                best = (a, b, float(v))
    return best


def bootstrap_ci(
    x: np.ndarray, y: np.ndarray, method: str, B: int = 2000, alpha: float = 0.05
) -> tuple[float, float]:
    rng = np.random.default_rng(20260428)
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    rs = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        # Skip degenerate resamples.
        if np.std(xb) == 0 or np.std(yb) == 0:
            continue
        if method == "pearson":
            r, _ = stats.pearsonr(xb, yb)
        else:
            r, _ = stats.spearmanr(xb, yb)
        if not math.isnan(r):
            rs.append(r)
    if not rs:
        return float("nan"), float("nan")
    lo = float(np.quantile(rs, alpha / 2))
    hi = float(np.quantile(rs, 1 - alpha / 2))
    return lo, hi


def main() -> int:
    pd.set_option("display.float_format", lambda x: f"{x:6.3f}")

    heading("Finding C verification - dimension orthogonality across cells")

    df = build_score_matrix()
    print(f"Loaded {len(df)} (backend, reader) cells (seeds aggregated over {SEEDS}).")
    print()
    print("Per-cell D1..D6 macro-means:")
    show = df.copy()
    show.columns = [c if c not in PAPER_DIM_TEX else SHORT[PAPER_DIMS.index(c)]
                    for c in show.columns]
    print(show.to_string(index=False))
    print()

    n_total = len(df)
    n_full = df.dropna(subset=PAPER_DIMS).shape[0]
    print(f"Cells with all six dims present: {n_full} / {n_total}")
    if n_full < n_total:
        missing_summary = []
        for _, r in df.iterrows():
            miss = [SHORT[PAPER_DIMS.index(d)]
                    for d in PAPER_DIMS if pd.isna(r[d])]
            if miss:
                missing_summary.append(f"  {r['backend']:9s} x {r['model']:8s}: missing {miss}")
        print("\n".join(missing_summary))
    print()

    # Use the full-data subset (all 6 dims present) for clean correlation.
    df_full = df.dropna(subset=PAPER_DIMS).reset_index(drop=True)

    # ------------------------------------------------------------------
    heading("Pearson correlation matrix (across cells, n = {})".format(len(df_full)), char="-")
    rho_p, p_p = pairwise_corr(df_full, "pearson")
    print(rho_p.to_string())
    print()
    print("p-values:")
    print(p_p.to_string())
    print()

    heading("Spearman correlation matrix (across cells, n = {})".format(len(df_full)), char="-")
    rho_s, p_s = pairwise_corr(df_full, "spearman")
    print(rho_s.to_string())
    print()
    print("p-values:")
    print(p_s.to_string())
    print()

    # ------------------------------------------------------------------
    # Sub-claim 1: max off-diagonal |rho| <= 0.40
    heading("SUB-CLAIM 1: max |rho| over off-diagonal <= 0.40", char="-")
    for label, rho in [("Pearson", rho_p), ("Spearman", rho_s)]:
        a, b, r = max_offdiag(rho)
        verdict = "HOLDS" if abs(r) <= 0.40 else "VIOLATED"
        print(f"  {label:9s}: max |rho| = {abs(r):.3f} at ({a}, {b}) [signed {r:+.3f}]  -> {verdict}")
        a2, b2, r2 = min_offdiag(rho)
        print(f"  {label:9s}: min |rho| = {abs(r2):.3f} at ({a2}, {b2}) [signed {r2:+.3f}]  (most decoupled pair)")
    print()

    # ------------------------------------------------------------------
    # Sub-claim 2: D6 near-orthogonal to all other dims (|rho| <= 0.10)
    heading("SUB-CLAIM 2: |rho(D6, D_i)| <= 0.10 for all i != 6", char="-")
    for label, rho, p in [("Pearson", rho_p, p_p), ("Spearman", rho_s, p_s)]:
        print(f"  {label}:")
        max_abs = 0.0
        for d in SHORT:
            if d == "D6":
                continue
            r = rho.loc["D6", d]
            pv = p.loc["D6", d]
            # Bootstrap CI for D6 vs d
            i = SHORT.index("D6")
            j = SHORT.index(d)
            x = df_full[PAPER_DIMS[i]].to_numpy()
            y = df_full[PAPER_DIMS[j]].to_numpy()
            lo, hi = bootstrap_ci(x, y, method=label.lower())
            print(f"    rho(D6, {d}) = {r:+.3f}  [95% CI {lo:+.3f}, {hi:+.3f}]  p={pv:.3f}")
            max_abs = max(max_abs, abs(r))
        verdict = "HOLDS" if max_abs <= 0.10 else "VIOLATED"
        print(f"  {label}: max |rho(D6, *)| = {max_abs:.3f} -> {verdict}")
    print()

    # ------------------------------------------------------------------
    # Sub-claim 3 + 4: high D1 + low D6 vs high D6 backends
    heading("SUB-CLAIM 3+4: cells with extreme D1 vs D6 mismatch", char="-")
    df_full = df_full.copy()
    df_full["D1"] = df_full["d1_cloze"]
    df_full["D6"] = df_full["d6_permission"]
    df_full["spread"] = df_full[PAPER_DIMS].max(axis=1) - df_full[PAPER_DIMS].min(axis=1)
    print("D1 high (>=0.5), D6 low (<=0.10) cells (recall-strong, permission-broken):")
    sel = df_full[(df_full["D1"] >= 0.5) & (df_full["D6"] <= 0.10)]
    if sel.empty:
        print("  (none)")
    else:
        print(sel[["backend", "model", "D1", "D6"]].to_string(index=False))
    print()

    # The opposite: D6 high (>=0.5) while D1 not collapsed (>=0.5)
    print("D6 high (>=0.5) AND D1 high (>=0.5) cells (no recall penalty for permission):")
    sel2 = df_full[(df_full["D6"] >= 0.5) & (df_full["D1"] >= 0.5)]
    if sel2.empty:
        print("  (none)")
    else:
        print(sel2[["backend", "model", "D1", "D6"]].to_string(index=False))
    print()

    print("Top-5 cells by intra-cell dimension spread (max - min over D1..D6):")
    top_spread = df_full.sort_values("spread", ascending=False).head(5)
    print(top_spread[["backend", "model"] + PAPER_DIMS + ["spread"]].to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # Sub-claim 5: Kendall's tau between rank-by-D1 and rank-by-D6
    heading("SUB-CLAIM 5: Kendall's tau between cell rankings on different dims", char="-")
    print("Across all (backend, reader) cells:")
    print()
    print("  Pair          tau     p-value   #cells")
    pair_taus = []
    for a, b in combinations(SHORT, 2):
        ai = SHORT.index(a)
        bi = SHORT.index(b)
        xa = df_full[PAPER_DIMS[ai]].to_numpy()
        xb = df_full[PAPER_DIMS[bi]].to_numpy()
        mask = ~(np.isnan(xa) | np.isnan(xb))
        if mask.sum() < 3:
            continue
        tau, p = stats.kendalltau(xa[mask], xb[mask])
        pair_taus.append(((a, b), float(tau), float(p)))
        print(f"  {a}-{b}      {tau:+6.3f}  {p:7.3f}    {mask.sum():3d}")
    print()
    # Highlight D1 vs D6 specifically
    for (a, b), tau, p in pair_taus:
        if {a, b} == {"D1", "D6"}:
            print(f"  HIGHLIGHT  Kendall tau(D1, D6) = {tau:+.3f}  (p={p:.3f})")
            if abs(tau) <= 0.20:
                print(f"             -> Rankings by D1 vs D6 are essentially independent.")
            else:
                print(f"             -> Rankings show meaningful dependence.")
    print()

    # ------------------------------------------------------------------
    # Single-scalar test: how often does mean-rank disagree with D6-rank?
    heading("SUB-CLAIM 5 / single-scalar test: mean(D1..D6) vs D6 rank", char="-")
    df_full = df_full.copy()
    df_full["mean_dim"] = df_full[PAPER_DIMS].mean(axis=1)
    n = len(df_full)
    mean_rank = stats.rankdata(df_full["mean_dim"].to_numpy())
    d6_rank = stats.rankdata(df_full["d6_permission"].to_numpy())
    inversions = 0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairs += 1
            sign_mean = np.sign(mean_rank[i] - mean_rank[j])
            sign_d6 = np.sign(d6_rank[i] - d6_rank[j])
            if sign_mean != 0 and sign_d6 != 0 and sign_mean != sign_d6:
                inversions += 1
    frac = inversions / pairs if pairs else float("nan")
    print(f"  cell-pairs total : {pairs}")
    print(f"  inversions (sign(mean-rank) != sign(D6-rank)): {inversions}")
    print(f"  fraction inverted : {frac:.3f}")
    tau_md, p_md = stats.kendalltau(df_full["mean_dim"].to_numpy(),
                                    df_full["d6_permission"].to_numpy())
    print(f"  Kendall tau(mean_dim, D6) = {tau_md:+.3f} (p={p_md:.3f})")
    print()

    # ------------------------------------------------------------------
    # Cross-check vs paper/tables/dim_correlation_0419.tex (per-agent matrix)
    heading("CROSS-CHECK vs paper/tables/dim_correlation_0419.tex", char="-")
    print("Reference table (per-agent within Oracle-32B, seeds s2-s4): ")
    print("  largest |rho| = 0.40 at D3-D5; D2-D5 = 0.35; D6 row max |rho| = 0.10 at D1-D6.")
    print()
    print("Our matrix (across (backend, reader) cells, seeds s2-s4 averaged):")
    a, b, r = max_offdiag(rho_s)
    print(f"  largest |rho_spearman| = {abs(r):.3f} at ({a}, {b}) [signed {r:+.3f}]")
    d6_row = rho_s.loc["D6"].drop("D6")
    print(f"  D6 row |rho_spearman|: max={d6_row.abs().max():.3f} "
          f"(at {d6_row.abs().idxmax()}); all values: "
          f"{ {k: round(float(v),3) for k,v in d6_row.items()} }")
    print()
    print("Note: unit of analysis differs. The reference table treats each ego")
    print("agent as a row; this script treats each (backend, reader) cell as a row.")
    print("If both come out near-zero on D6, the orthogonality claim generalises;")
    print("if they diverge, the cross-cell story is what we should report.")

    # ------------------------------------------------------------------
    heading("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
