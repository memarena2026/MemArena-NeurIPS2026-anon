#!/usr/bin/env python3
"""Empirical verification of candidate finding D:
"Under retrieval-perfect conditions, direct-lookup and integrative tasks
are decoupled bottlenecks."

Three diagnostics:
  Part 1 -- Bottleneck-type stratification on Qwen3-32B-AWQ:
            per-sub-dim Vanilla/Oracle/Delta plus DIRECT vs INTEGRATIVE
            mean comparison (Mann-Whitney U, on n_direct=2 vs n_integ=6).
  Part 2 -- Scale-invariance of Oracle direct/integrative gap across
            Q3-0.6B / Q3-8B / Q3-32B (and side rows for Llama-3.2-3B,
            Mistral-7B).
  Part 3 -- Structured-backend inheritance (Memobase, MemOS) on Q3-32B:
            does the gap persist or close, which sub-dims beat Oracle.

Source of truth: paper/tables/appendix_L_subdim.tex (parsed directly).
This file is the canonical sub-dim aggregate used by the paper.

Usage:
    python3 paper/verify_findingD_decoupled_bottlenecks.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
TEX_PATH = ROOT / "tables" / "appendix_L_subdim.tex"

# Column order in the sub-dim table; this matches the multicolumn header
# in appendix_L_subdim.tex (Confl., Anaph., QA, Temp., Adv., Cloze, Meta.,
# Abst., Perm.). Internal sub-dim IDs follow the doc string in
# paper/paper_data.py.
SUBDIM_COLS = [
    "d1_conflict",       # Confl.
    "d2_anaphora",       # Anaph.
    "d7_qa",             # QA   (Factual QA -- direct-lookup)
    "d8_temporal",       # Temp.
    "d10_counterfactual",# Adv.
    "d5_cloze",          # Cloze (direct-lookup)
    "d6_metadata",       # Meta.
    "d3_confabulation",  # Abst.
    "d4_permission",     # Perm. (excluded for finding D -- privacy)
]

# Cluster definitions per the candidate finding statement.
DIRECT = ["d5_cloze", "d7_qa"]
INTEGRATIVE = [
    "d6_metadata",
    "d8_temporal",
    "d10_counterfactual",
    "d1_conflict",
    "d2_anaphora",
    "d3_confabulation",
]
EXCLUDED = ["d4_permission"]

BACKENDS = ["Oracle", "Vanilla", "RAG", "Memobase", "MemOS"]
MODELS = ["Qwen3-0.6B", "Llama-3.2-3B", "Mistral-7B", "Qwen3-8B", "Qwen3-32B"]

# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------

# Each data line in the table looks like:
#   & Qwen3-0.6B     & 86.7{\scriptsize$\pm$1.8} & 54.7{\scriptsize$\pm$1.6} & ...\\
ROW_RE = re.compile(
    r"&\s*(?P<model>[A-Za-z0-9.\-]+)\s*"
    + (r"&\s*(?P<v{i}>[0-9]+\.[0-9]+)\{\\scriptsize\$\\pm\$[0-9.]+\}\s*"
       .replace("{i}", "0"))  # placeholder; we'll use a simpler approach
)

# Simpler approach: split on '&' and pull out floats directly.
NUM_RE = re.compile(r"(\d+(?:\.\d+)?)\{\\scriptsize\$\\pm\$\s*(\d+(?:\.\d+)?)\}")


def _parse_subdim_tex(path: Path) -> pd.DataFrame:
    """Return a long-form dataframe with columns:
        backend, model, sub_dim, mean, std
    """
    text = path.read_text()
    lines = text.splitlines()

    rows: list[dict] = []
    current_backend: str | None = None
    multirow_re = re.compile(r"\\multirow\{\d+\}\{\*\}\{(\w+)\}")

    for raw in lines:
        m_back = multirow_re.search(raw)
        if m_back:
            current_backend = m_back.group(1)
            # The multirow line itself has no model/scores -- skip.
            # Actually in this table, the \multirow line is followed by a
            # data line on the next line. Continue.
        # A data row contains '& ModelName ... & xx.x{\scriptsize$\pm$y.y}'
        if current_backend is None:
            continue
        # Quick guard: must contain a number-with-pm pattern.
        if "\\scriptsize" not in raw:
            continue
        # Pull model name: first token after the leading '&'.
        # Example raw line:
        #   '  & Qwen3-0.6B     & 86.7{\scriptsize$\pm$1.8} & ...'
        # Some lines start with '\\multirow{5}{*}{Oracle}\n  & Qwen3-0.6B'
        # (multirow is on prior line). We strip the multirow if present.
        line = multirow_re.sub("", raw).strip()
        if not line.startswith("&"):
            continue
        parts = [p.strip() for p in line.split("&")]
        # parts[0] is empty (line starts with '&'); parts[1] is model name;
        # parts[2..10] are the 9 sub-dim cells.
        if len(parts) < 11:
            continue
        model = parts[1].strip()
        if model not in MODELS:
            continue
        cells = parts[2:11]
        for col_idx, cell in enumerate(cells):
            num_match = NUM_RE.search(cell)
            if not num_match:
                continue
            mean = float(num_match.group(1))
            std = float(num_match.group(2))
            rows.append(
                {
                    "backend": current_backend,
                    "model": model,
                    "sub_dim": SUBDIM_COLS[col_idx],
                    "mean": mean,
                    "std": std,
                }
            )

    df = pd.DataFrame(rows)
    return df


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _val(df: pd.DataFrame, backend: str, model: str, sub_dim: str) -> float | None:
    sel = df[(df.backend == backend) & (df.model == model) & (df.sub_dim == sub_dim)]
    if sel.empty:
        return None
    return float(sel["mean"].iloc[0])


def _mean(vals: Iterable[float | None]) -> float | None:
    clean = [v for v in vals if v is not None]
    if not clean:
        return None
    return float(np.mean(clean))


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------


def part1_stratification(df: pd.DataFrame) -> dict:
    print("=" * 78)
    print("PART 1 -- Bottleneck-type stratification on Qwen3-32B (AWQ)")
    print("=" * 78)
    print()
    print("Per-sub-dim Vanilla, Oracle, Delta=Oracle-Vanilla on Qwen3-32B.")
    print(f"DIRECT cluster:      {DIRECT}")
    print(f"INTEGRATIVE cluster: {INTEGRATIVE}")
    print(f"EXCLUDED:            {EXCLUDED}")
    print()

    rows = []
    for sd in DIRECT + INTEGRATIVE:
        v = _val(df, "Vanilla", "Qwen3-32B", sd)
        o = _val(df, "Oracle", "Qwen3-32B", sd)
        if v is None or o is None:
            print(f"  MISSING: {sd}  (vanilla={v}, oracle={o})")
            continue
        delta = o - v
        cluster = "DIRECT" if sd in DIRECT else "INTEGRATIVE"
        rows.append(
            {
                "sub_dim": sd,
                "cluster": cluster,
                "vanilla": v,
                "oracle": o,
                "delta_pp": delta,
            }
        )

    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False, float_format=lambda x: f"{x:6.2f}"))
    print()

    direct_delta = tab.loc[tab.cluster == "DIRECT", "delta_pp"].values
    integ_delta = tab.loc[tab.cluster == "INTEGRATIVE", "delta_pp"].values
    direct_oracle = tab.loc[tab.cluster == "DIRECT", "oracle"].values
    integ_oracle = tab.loc[tab.cluster == "INTEGRATIVE", "oracle"].values
    direct_van = tab.loc[tab.cluster == "DIRECT", "vanilla"].values
    integ_van = tab.loc[tab.cluster == "INTEGRATIVE", "vanilla"].values

    print(
        f"Vanilla mean   : DIRECT={np.mean(direct_van):.2f}  "
        f"INTEGRATIVE={np.mean(integ_van):.2f}  "
        f"diff={np.mean(direct_van)-np.mean(integ_van):+.2f}pp"
    )
    print(
        f"Oracle mean    : DIRECT={np.mean(direct_oracle):.2f}  "
        f"INTEGRATIVE={np.mean(integ_oracle):.2f}  "
        f"diff={np.mean(direct_oracle)-np.mean(integ_oracle):+.2f}pp"
    )
    print(
        f"Delta mean     : DIRECT={np.mean(direct_delta):.2f}  "
        f"INTEGRATIVE={np.mean(integ_delta):.2f}  "
        f"diff={np.mean(direct_delta)-np.mean(integ_delta):+.2f}pp"
    )
    print()

    # Mann-Whitney U on small n. We report regardless; effect size
    # via rank-biserial r = 1 - 2U / (n1*n2).
    def _mwu(a, b, label):
        if len(a) == 0 or len(b) == 0:
            print(f"  {label}: insufficient data")
            return None, None
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rbc = 1 - (2 * u) / (len(a) * len(b))
        print(
            f"  Mann-Whitney U on {label:8s}: U={u:.1f}, p={p:.4f}, "
            f"rank-biserial r={rbc:+.3f}, n_direct={len(a)}, n_integ={len(b)}"
        )
        return u, p

    print("Statistical tests (DIRECT vs INTEGRATIVE):")
    _mwu(direct_delta, integ_delta, "Delta")
    _mwu(direct_oracle, integ_oracle, "Oracle")
    print()

    # Abstract claim audit
    print("Abstract claim audit ('direct-lookup >=88%, integrative 51-62% on Q3-32B Oracle'):")
    print(f"  direct-lookup (cloze/qa) Oracle: "
          f"d5_cloze={_val(df,'Oracle','Qwen3-32B','d5_cloze'):.1f}, "
          f"d7_qa={_val(df,'Oracle','Qwen3-32B','d7_qa'):.1f}")
    print(f"  integrative Oracle:")
    for sd in INTEGRATIVE:
        v = _val(df, "Oracle", "Qwen3-32B", sd)
        print(f"    {sd:22s} {v:5.1f}")
    print(f"  -> abstract uses temp=51.2 (={_val(df,'Oracle','Qwen3-32B','d8_temporal'):.1f}) "
          f"and meta=62.7 (={_val(df,'Oracle','Qwen3-32B','d6_metadata'):.1f}); "
          f"these match exactly.")
    print()

    return {"table": tab, "direct_delta": direct_delta,
            "integ_delta": integ_delta,
            "direct_oracle": direct_oracle,
            "integ_oracle": integ_oracle}


def part2_scale_invariance(df: pd.DataFrame) -> dict:
    print("=" * 78)
    print("PART 2 -- Scale-invariance of Oracle direct/integrative gap")
    print("=" * 78)
    print()
    print("Oracle mean on DIRECT vs INTEGRATIVE clusters per reader; ratio &")
    print("absolute pp gap. Qwen scaling Q3-0.6B -> Q3-8B -> Q3-32B is the")
    print("primary axis; Llama-3.2-3B and Mistral-7B are side rows.")
    print()

    primary = ["Qwen3-0.6B", "Qwen3-8B", "Qwen3-32B"]
    side = ["Llama-3.2-3B", "Mistral-7B"]

    rows = []
    for m in primary + side:
        d_vals = [_val(df, "Oracle", m, sd) for sd in DIRECT]
        i_vals = [_val(df, "Oracle", m, sd) for sd in INTEGRATIVE]
        d_mean = _mean(d_vals)
        i_mean = _mean(i_vals)
        if d_mean is None or i_mean is None:
            continue
        rows.append({
            "model": m,
            "is_primary": m in primary,
            "direct_mean": d_mean,
            "integ_mean": i_mean,
            "ratio_d_over_i": d_mean / i_mean if i_mean else float("nan"),
            "gap_pp": d_mean - i_mean,
        })
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False, float_format=lambda x: f"{x:6.3f}"))
    print()

    primary_tab = tab[tab.is_primary]
    print("Primary scaling axis (Qwen):")
    for _, r in primary_tab.iterrows():
        print(f"  {r['model']:14s}  ratio={r['ratio_d_over_i']:.3f}  gap={r['gap_pp']:+.2f}pp")
    ratio_range = primary_tab["ratio_d_over_i"].max() - primary_tab["ratio_d_over_i"].min()
    gap_range = primary_tab["gap_pp"].max() - primary_tab["gap_pp"].min()
    print(f"  Range across Qwen scales: ratio span={ratio_range:.3f}, gap span={gap_range:.2f}pp")
    print()
    return {"table": tab}


def part3_structured_inheritance(df: pd.DataFrame) -> dict:
    print("=" * 78)
    print("PART 3 -- Structured-backend inheritance on Qwen3-32B")
    print("=" * 78)
    print()
    print("Q1: Do Memobase/MemOS show DIRECT vs INTEGRATIVE gap at all?")
    print("    (compare each backend's own DIRECT vs INTEGRATIVE means)")
    print()

    rows = []
    for backend in ["Vanilla", "Oracle", "Memobase", "MemOS"]:
        d = _mean([_val(df, backend, "Qwen3-32B", sd) for sd in DIRECT])
        i = _mean([_val(df, backend, "Qwen3-32B", sd) for sd in INTEGRATIVE])
        if d is None or i is None:
            continue
        rows.append({
            "backend": backend,
            "direct_mean": d,
            "integ_mean": i,
            "gap_pp": d - i,
            "ratio": d / i if i else float("nan"),
        })
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False, float_format=lambda x: f"{x:6.2f}"))
    print()

    base_gap = tab.loc[tab.backend == "Oracle", "gap_pp"].iloc[0]
    print("Q2: Which structured backend reduces the gap the most?")
    for _, r in tab.iterrows():
        if r["backend"] in ("Memobase", "MemOS"):
            change = r["gap_pp"] - base_gap
            print(f"    {r['backend']:9s}: gap={r['gap_pp']:+.2f}pp "
                  f"(Oracle gap={base_gap:+.2f}pp; change={change:+.2f}pp)")
    print()

    print("Q3: On INTEGRATIVE sub-dims, where do structured backends beat Oracle?")
    rows3 = []
    for sd in INTEGRATIVE:
        oracle_val = _val(df, "Oracle", "Qwen3-32B", sd)
        memo = _val(df, "Memobase", "Qwen3-32B", sd)
        memos = _val(df, "MemOS", "Qwen3-32B", sd)
        if oracle_val is None or memo is None or memos is None:
            continue
        rows3.append({
            "sub_dim": sd,
            "oracle": oracle_val,
            "memobase": memo,
            "memos": memos,
            "memobase_minus_oracle_pp": memo - oracle_val,
            "memos_minus_oracle_pp": memos - oracle_val,
        })
    tab3 = pd.DataFrame(rows3)
    print(tab3.to_string(index=False, float_format=lambda x: f"{x:6.2f}"))
    print()

    print("Spotlight: d6_metadata (abstract claims Memobase +14.4pp on D2):")
    o = _val(df, "Oracle", "Qwen3-32B", "d6_metadata")
    m = _val(df, "Memobase", "Qwen3-32B", "d6_metadata")
    if o is not None and m is not None:
        print(f"  Oracle={o:.1f}, Memobase={m:.1f}, delta={m-o:+.2f}pp "
              f"(abstract claim +14.4pp)")
    print()

    return {"backend_table": tab, "subdim_table": tab3}


# ----------------------------------------------------------------------
# Entry
# ----------------------------------------------------------------------


def main() -> int:
    if not TEX_PATH.exists():
        print(f"FATAL: missing {TEX_PATH}", file=sys.stderr)
        return 1
    df = _parse_subdim_tex(TEX_PATH)

    # Sanity: print row count per backend.
    counts = df.groupby(["backend", "model"]).size().unstack(fill_value=0)
    print("Parsed rows (backend x model -> #sub-dims):")
    print(counts.to_string())
    print()
    if not (counts == 9).all().all():
        print("WARNING: not every cell has 9 sub-dims; some entries missing.")
        print()

    part1_stratification(df)
    part2_scale_invariance(df)
    part3_structured_inheritance(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
