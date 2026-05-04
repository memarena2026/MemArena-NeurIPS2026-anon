"""Paired analysis of D6 third-party (TP) vs self-ego probe labels.

For each (backend, reader, seed, question_id) we have two judgments:
  L_tp   = TP_record["policy_category"]      (1 of 6 D6 labels)
  L_ego  = EGO_record["policy_category"]
  leak_tp / leak_ego = leaked_fact_in_output (deterministic substring match)
  mode = TP_record["expected_answer_mode"] in {"disclose","abstain"}

ALL analysis is stratified by ground-truth pool:
  DENY pool  = mode == "abstain"   (80 items / cell × 75 cells = 6,000)
  ALLOW pool = mode == "disclose"  (120 items / cell × 75 cells = 9,000)

Outputs:
  out/d6_paired/marginal.csv           label distribution per pool × probe
  out/d6_paired/transitions_deny.csv   5x6 paired matrix (DENY pool, global)
  out/d6_paired/transitions_allow.csv  same, ALLOW pool
  out/d6_paired/cell_deltas.csv        25 rows: ΔRefuse / ΔLeak / ΔF1_PU
  out/d6_paired/per_cell_full.json     full per-(seed,backend,reader) breakdown

Run:
    python3 scripts/analyze_d6_paired_probe.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# Label vocabulary (canonical order for tables / matrices).
D6_LABELS = [
    "DISCLOSE_CORRECT",
    "DISCLOSE_WRONG",
    "DONT_KNOW",
    "REFUSE",
    "OTHER",
    "PARSE_ERROR",
]
LABEL_SHORT = {
    "DISCLOSE_CORRECT": "DC",
    "DISCLOSE_WRONG":   "DW",
    "DONT_KNOW":        "DK",
    "REFUSE":           "REF",
    "OTHER":            "OTH",
    "PARSE_ERROR":      "ERR",
}

# Self-probe directory tree uses the original backend tokens; map to canonical.
SELF_PROBE_BACKEND = {
    "vanilla":           "vanilla",
    "baseline_simplerag": "rag",
    "oracle":            "oracle",
    "memobase":          "memobase",
    "memsearch":         "memsearch",
}
READERS = {"0_6b", "llama3b", "7b", "8b", "32b"}
SEEDS = ["s2", "s3", "s4"]
BACKENDS_ORDER = ["vanilla", "rag", "oracle", "memobase", "memsearch"]
READERS_ORDER = ["0_6b", "llama3b", "7b", "8b", "32b"]


# -----------------------------------------------------------------------------
# Discovery
# -----------------------------------------------------------------------------

def _split_top_dir(top_name: str) -> tuple[str, str] | None:
    """Split ``<backend_raw>_<reader>`` where reader is one of READERS.
    e.g. "vanilla_0_6b" → ("vanilla", "0_6b"); "baseline_simplerag_8b" → ("baseline_simplerag", "8b")."""
    for r in READERS:
        suffix = "_" + r
        if top_name.endswith(suffix):
            be_raw = top_name[: -len(suffix)]
            return be_raw, r
    return None


def discover_self_probe_files(root: Path) -> list[tuple[Path, str, str, str]]:
    """Return [(json_path, backend, reader, seed), ...] one per cell file."""
    out: list[tuple[Path, str, str, str]] = []
    if not root.exists():
        return out
    for top in sorted(root.iterdir()):
        if not top.is_dir():
            continue
        parsed = _split_top_dir(top.name)
        if not parsed:
            continue
        be_raw, reader = parsed
        backend = SELF_PROBE_BACKEND.get(be_raw)
        if backend is None:
            continue
        for seed in SEEDS:
            seed_dir = top / seed / f"eval_results_{seed}"
            if not seed_dir.exists():
                continue
            # Inside seed_dir: one or more inner subdirs (vanilla / memory_cache / etc.).
            # We expect exactly one evaluation_results_run.json across them.
            cands = list(seed_dir.rglob("evaluation_results_run.json"))
            if not cands:
                continue
            # Prefer the one whose parent dir name maps back to this backend.
            preferred = [p for p in cands if p.parent.name in (be_raw, "memory_cache")]
            chosen = (preferred or cands)[0]
            out.append((chosen, backend, reader, seed))
    return out


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def load_tp_grid() -> dict[tuple[str, str, str], dict[str, dict]]:
    """Return {(seed, backend, reader): {qid: record}} for d4_perm records."""
    from memarena.figures.paper_data import load_all_cells
    grid = load_all_cells(include_ablation=False)
    out: dict[tuple[str, str, str], dict[str, dict]] = {}
    for (seed, backend, reader), cell in grid.items():
        if backend not in SELF_PROBE_BACKEND.values():
            continue
        if reader not in READERS:
            continue
        if seed not in SEEDS:
            continue
        try:
            data = json.loads(Path(cell.source_path).read_text())
        except Exception:
            continue
        recs = {}
        for d in data.get("details", []):
            qid = str(d.get("question_id", ""))
            if qid.startswith("d4_perm"):
                recs[qid] = d
        out[(seed, backend, reader)] = recs
    return out


def load_ego_grid(root: Path) -> dict[tuple[str, str, str], dict[str, dict]]:
    out: dict[tuple[str, str, str], dict[str, dict]] = {}
    for path, backend, reader, seed in discover_self_probe_files(root):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        recs = {}
        for d in data.get("details", []):
            qid = str(d.get("question_id", ""))
            if qid.startswith("d4_perm"):
                recs[qid] = d
        out[(seed, backend, reader)] = recs
    return out


# -----------------------------------------------------------------------------
# Per-cell paired computation
# -----------------------------------------------------------------------------

def _pool_of(rec: dict) -> str | None:
    """Return 'deny' for abstain, 'allow' for disclose, else None."""
    mode = str(rec.get("expected_answer_mode") or "").lower()
    if mode == "abstain":
        return "deny"
    if mode == "disclose":
        return "allow"
    return None


def _label(rec: dict) -> str:
    lbl = str(rec.get("policy_category") or "").upper()
    return lbl if lbl in D6_LABELS else "OTHER"


def per_cell_paired(
    tp_recs: dict[str, dict],
    ego_recs: dict[str, dict],
) -> dict:
    """Compute per-pool marginal label counts, transition counts, and scalar metrics."""
    qids = sorted(set(tp_recs) & set(ego_recs))
    if not qids:
        return {"n_paired": 0}

    # marginal[pool][probe][label] = count
    marginal = {p: {"tp": defaultdict(int), "ego": defaultdict(int)} for p in ("deny", "allow")}
    # transitions[pool][L_tp][L_ego] = count
    transitions = {p: defaultdict(lambda: defaultdict(int)) for p in ("deny", "allow")}
    # leak counts (deterministic), per pool & probe
    leak = {p: {"tp": 0, "ego": 0} for p in ("deny", "allow")}
    n_pool = {"deny": 0, "allow": 0}

    for qid in qids:
        tp = tp_recs[qid]
        ego = ego_recs[qid]
        pool = _pool_of(tp) or _pool_of(ego)
        if pool is None:
            continue
        n_pool[pool] += 1
        l_tp = _label(tp)
        l_ego = _label(ego)
        marginal[pool]["tp"][l_tp] += 1
        marginal[pool]["ego"][l_ego] += 1
        transitions[pool][l_tp][l_ego] += 1
        if bool(tp.get("leaked_fact_in_output", False)):
            leak[pool]["tp"] += 1
        if bool(ego.get("leaked_fact_in_output", False)):
            leak[pool]["ego"] += 1

    # Convert defaultdicts → plain dicts
    out_marg = {
        p: {probe: {lbl: marginal[p][probe].get(lbl, 0) for lbl in D6_LABELS}
            for probe in ("tp", "ego")}
        for p in ("deny", "allow")
    }
    out_trans = {
        p: {l_tp: {l_ego: transitions[p][l_tp].get(l_ego, 0) for l_ego in D6_LABELS}
            for l_tp in D6_LABELS}
        for p in ("deny", "allow")
    }

    return {
        "n_paired": len(qids),
        "n_pool": dict(n_pool),
        "marginal": out_marg,
        "transitions": out_trans,
        "leak": leak,
    }


def cell_scalars(per_cell: dict) -> dict:
    """Derive headline rates per probe from a per_cell dict."""
    if per_cell.get("n_paired", 0) == 0:
        return {}
    out = {}
    for pool in ("deny", "allow"):
        n = per_cell["n_pool"][pool]
        if n == 0:
            continue
        m = per_cell["marginal"][pool]
        leak = per_cell["leak"][pool]
        for probe in ("tp", "ego"):
            out[f"refuse_{pool}_{probe}"]   = m[probe]["REFUSE"] / n
            out[f"comply_{pool}_{probe}"]   = m[probe]["DISCLOSE_CORRECT"] / n
            out[f"dont_know_{pool}_{probe}"] = m[probe]["DONT_KNOW"] / n
            out[f"leak_{pool}_{probe}"]     = leak[probe] / n
    # F1_PU per probe (P fact-based, U category-based) — DENY supplies P, ALLOW supplies U.
    for probe in ("tp", "ego"):
        if "leak_deny_" + probe in out and "comply_allow_" + probe in out:
            p = 1 - out[f"leak_deny_{probe}"]
            u = out[f"comply_allow_{probe}"]
            out[f"f1_pu_{probe}"] = (2 * p * u / (p + u)) if (p + u) > 0 else 0.0
    return out


# -----------------------------------------------------------------------------
# Aggregation across seeds
# -----------------------------------------------------------------------------

def aggregate_to_br(per_cell: dict[tuple[str, str, str], dict]) -> dict[tuple[str, str], dict]:
    """Aggregate marginals/transitions/leak across seeds → keyed by (backend, reader)."""
    by_br: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for (seed, b, r), m in per_cell.items():
        by_br[(b, r)].append(m)

    agg: dict[tuple[str, str], dict] = {}
    for (b, r), ms in by_br.items():
        marg = {p: {pr: {lbl: 0 for lbl in D6_LABELS} for pr in ("tp", "ego")}
                for p in ("deny", "allow")}
        trans = {p: {l_tp: {l_ego: 0 for l_ego in D6_LABELS} for l_tp in D6_LABELS}
                 for p in ("deny", "allow")}
        leak = {p: {"tp": 0, "ego": 0} for p in ("deny", "allow")}
        n_pool = {"deny": 0, "allow": 0}
        for m in ms:
            if m.get("n_paired", 0) == 0:
                continue
            for p in ("deny", "allow"):
                n_pool[p] += m["n_pool"][p]
                for pr in ("tp", "ego"):
                    leak[p][pr] += m["leak"][p][pr]
                    for lbl in D6_LABELS:
                        marg[p][pr][lbl] += m["marginal"][p][pr][lbl]
                for l_tp in D6_LABELS:
                    for l_ego in D6_LABELS:
                        trans[p][l_tp][l_ego] += m["transitions"][p][l_tp][l_ego]
        cell_combined = {"n_paired": sum(m["n_pool"]["deny"] + m["n_pool"]["allow"] for m in ms),
                         "n_pool": n_pool, "marginal": marg, "transitions": trans, "leak": leak}
        agg[(b, r)] = {
            "n_seeds": sum(1 for m in ms if m.get("n_paired", 0) > 0),
            "combined": cell_combined,
            "scalars": cell_scalars(cell_combined),
        }
    return agg


# -----------------------------------------------------------------------------
# Reporting + CSV
# -----------------------------------------------------------------------------

def print_marginal_global(global_per_pool: dict) -> None:
    print()
    print("=" * 78)
    print("STEP 2 — Per-pool marginal label distribution (global, all 25 cells × 3 seeds)")
    print("=" * 78)
    for pool in ("deny", "allow"):
        n = global_per_pool["n_pool"][pool]
        marg = global_per_pool["marginal"][pool]
        leak = global_per_pool["leak"][pool]
        title = "DENY (should withhold)" if pool == "deny" else "ALLOW (should disclose)"
        print(f"\nPool: {title}   N = {n}")
        print(f"  {'label':18s} {'TP%':>7s} {'EGO%':>7s} {'Δ (pp)':>8s}")
        for lbl in D6_LABELS:
            tp = marg["tp"][lbl] / n * 100 if n else 0
            ego = marg["ego"][lbl] / n * 100 if n else 0
            d = ego - tp
            print(f"  {lbl:18s} {tp:6.2f}  {ego:6.2f}  {d:+7.2f}")
        ltp = leak["tp"] / n * 100 if n else 0
        lego = leak["ego"] / n * 100 if n else 0
        print(f"  {'(fact-leak)':18s} {ltp:6.2f}  {lego:6.2f}  {lego-ltp:+7.2f}  [deterministic]")


def print_transitions(global_per_pool: dict) -> None:
    print()
    print("=" * 78)
    print("STEP 3 — Paired 5×5 transition matrices  (rows = TP label, cols = EGO label)")
    print("=" * 78)
    for pool in ("deny", "allow"):
        n = global_per_pool["n_pool"][pool]
        title = "DENY pool" if pool == "deny" else "ALLOW pool"
        print(f"\n{title}   N = {n}   (cell = % of pool; top = count)")
        cols = D6_LABELS
        print("  " + " " * 6 + "  ".join(f"{LABEL_SHORT[c]:>5s}" for c in cols))
        for l_tp in D6_LABELS:
            row = global_per_pool["transitions"][pool][l_tp]
            row_total = sum(row[c] for c in cols)
            cells = []
            for c in cols:
                v = row[c]
                pct = v / n * 100 if n else 0
                cells.append(f"{pct:5.1f}")
            print(f"  {LABEL_SHORT[l_tp]:>4s}  " + "  ".join(cells) + f"   | row_n={row_total}")


def print_cell_deltas(agg_br: dict[tuple[str, str], dict]) -> None:
    print()
    print("=" * 95)
    print("STEP 4 — Per-cell deltas  (EGO − TP), pooled over seeds")
    print("=" * 95)
    print(f"  {'backend':10s} {'reader':10s}   {'ΔRef_DENY':>10s}  "
          f"{'ΔRef_ALLOW':>11s}  {'ΔLeak_DENY':>11s}  "
          f"{'F1_TP':>6s}  {'F1_EGO':>7s}  {'ΔF1_PU':>7s}")
    print("-" * 95)
    for b in BACKENDS_ORDER:
        for r in READERS_ORDER:
            v = agg_br.get((b, r))
            if v is None:
                continue
            s = v["scalars"]
            d_ref_d = (s.get("refuse_deny_ego", 0) - s.get("refuse_deny_tp", 0)) * 100
            d_ref_a = (s.get("refuse_allow_ego", 0) - s.get("refuse_allow_tp", 0)) * 100
            d_leak  = (s.get("leak_deny_ego", 0) - s.get("leak_deny_tp", 0)) * 100
            f1_tp   = s.get("f1_pu_tp", 0) * 100
            f1_ego  = s.get("f1_pu_ego", 0) * 100
            d_f1    = f1_ego - f1_tp
            print(f"  {b:10s} {r:10s}   {d_ref_d:+9.2f}  {d_ref_a:+10.2f}  {d_leak:+10.2f}  "
                  f"{f1_tp:5.1f}  {f1_ego:6.1f}  {d_f1:+6.1f}")


# -----------------------------------------------------------------------------
# CSV writers
# -----------------------------------------------------------------------------

def _write_marginal_csv(out_path: Path, global_per_pool: dict) -> None:
    rows = [["pool", "label", "tp_count", "ego_count", "tp_pct", "ego_pct", "delta_pp", "n_pool"]]
    for pool in ("deny", "allow"):
        n = global_per_pool["n_pool"][pool]
        marg = global_per_pool["marginal"][pool]
        for lbl in D6_LABELS:
            tp_c = marg["tp"][lbl]
            ego_c = marg["ego"][lbl]
            tp_p = tp_c / n * 100 if n else 0
            ego_p = ego_c / n * 100 if n else 0
            rows.append([pool, lbl, tp_c, ego_c, f"{tp_p:.4f}", f"{ego_p:.4f}",
                         f"{ego_p - tp_p:+.4f}", n])
        leak = global_per_pool["leak"][pool]
        rows.append([pool, "_FACT_LEAK_", leak["tp"], leak["ego"],
                     f"{leak['tp']/n*100:.4f}" if n else "0",
                     f"{leak['ego']/n*100:.4f}" if n else "0",
                     f"{(leak['ego']-leak['tp'])/n*100:+.4f}" if n else "0", n])
    with out_path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)


def _write_transitions_csv(out_path: Path, transitions: dict, n: int) -> None:
    rows = [["tp_label", *[f"ego_{l}" for l in D6_LABELS], "row_n", "n_pool"]]
    for l_tp in D6_LABELS:
        row = transitions[l_tp]
        cells = [row[l_ego] for l_ego in D6_LABELS]
        row_n = sum(cells)
        rows.append([l_tp, *cells, row_n, n])
    with out_path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)


def _write_cell_deltas_csv(out_path: Path, agg_br: dict) -> None:
    cols = ["backend", "reader", "n_seeds",
            "refuse_deny_tp", "refuse_deny_ego", "delta_refuse_deny",
            "refuse_allow_tp", "refuse_allow_ego", "delta_refuse_allow",
            "leak_deny_tp", "leak_deny_ego", "delta_leak_deny",
            "comply_allow_tp", "comply_allow_ego", "delta_comply_allow",
            "f1_pu_tp", "f1_pu_ego", "delta_f1_pu"]
    rows = [cols]
    for b in BACKENDS_ORDER:
        for r in READERS_ORDER:
            v = agg_br.get((b, r))
            if v is None:
                continue
            s = v["scalars"]
            def pct(k):
                return f"{s.get(k, 0)*100:.4f}"
            r_dt = s.get("refuse_deny_tp", 0); r_de = s.get("refuse_deny_ego", 0)
            r_at = s.get("refuse_allow_tp", 0); r_ae = s.get("refuse_allow_ego", 0)
            l_dt = s.get("leak_deny_tp", 0); l_de = s.get("leak_deny_ego", 0)
            c_at = s.get("comply_allow_tp", 0); c_ae = s.get("comply_allow_ego", 0)
            f_t = s.get("f1_pu_tp", 0); f_e = s.get("f1_pu_ego", 0)
            rows.append([
                b, r, v["n_seeds"],
                pct("refuse_deny_tp"), pct("refuse_deny_ego"), f"{(r_de-r_dt)*100:+.4f}",
                pct("refuse_allow_tp"), pct("refuse_allow_ego"), f"{(r_ae-r_at)*100:+.4f}",
                pct("leak_deny_tp"), pct("leak_deny_ego"), f"{(l_de-l_dt)*100:+.4f}",
                pct("comply_allow_tp"), pct("comply_allow_ego"), f"{(c_ae-c_at)*100:+.4f}",
                pct("f1_pu_tp"), pct("f1_pu_ego"), f"{(f_e-f_t)*100:+.4f}",
            ])
    with out_path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-probe-root", type=Path,
                        default=REPO_ROOT / "data" / "d6_self_probe")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "out" / "d6_paired")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading TP grid via paper_data.load_all_cells() ...")
    tp_grid = load_tp_grid()
    print(f"  TP cells: {len(tp_grid)}")

    print(f"Loading EGO grid from {args.self_probe_root} ...")
    ego_grid = load_ego_grid(args.self_probe_root)
    print(f"  EGO cells: {len(ego_grid)}")

    common = sorted(set(tp_grid) & set(ego_grid))
    print(f"  Joinable cells: {len(common)}")
    missing_ego = sorted(set(tp_grid) - set(ego_grid))
    missing_tp = sorted(set(ego_grid) - set(tp_grid))
    if missing_ego:
        print(f"  WARNING: {len(missing_ego)} TP cells with no EGO data:")
        for c in missing_ego[:5]:
            print(f"    {c}")
    if missing_tp:
        print(f"  WARNING: {len(missing_tp)} EGO cells with no TP data:")
        for c in missing_tp[:5]:
            print(f"    {c}")

    # Per-cell paired computation
    per_cell: dict[tuple[str, str, str], dict] = {}
    for key in common:
        per_cell[key] = per_cell_paired(tp_grid[key], ego_grid[key])

    # Global pooling (all 75 cells together) for marginal + transitions
    global_per_pool = {
        "n_pool": {"deny": 0, "allow": 0},
        "marginal": {p: {pr: {lbl: 0 for lbl in D6_LABELS} for pr in ("tp", "ego")}
                     for p in ("deny", "allow")},
        "transitions": {p: {l: {l2: 0 for l2 in D6_LABELS} for l in D6_LABELS}
                        for p in ("deny", "allow")},
        "leak": {p: {"tp": 0, "ego": 0} for p in ("deny", "allow")},
    }
    for m in per_cell.values():
        if m.get("n_paired", 0) == 0:
            continue
        for p in ("deny", "allow"):
            global_per_pool["n_pool"][p] += m["n_pool"][p]
            for pr in ("tp", "ego"):
                global_per_pool["leak"][p][pr] += m["leak"][p][pr]
                for lbl in D6_LABELS:
                    global_per_pool["marginal"][p][pr][lbl] += m["marginal"][p][pr][lbl]
            for l_tp in D6_LABELS:
                for l_ego in D6_LABELS:
                    global_per_pool["transitions"][p][l_tp][l_ego] += \
                        m["transitions"][p][l_tp][l_ego]

    # Per-(backend, reader) aggregation
    agg_br = aggregate_to_br(per_cell)

    # Print
    print_marginal_global(global_per_pool)
    print_transitions(global_per_pool)
    print_cell_deltas(agg_br)

    # CSV outputs
    _write_marginal_csv(args.out_dir / "marginal.csv", global_per_pool)
    _write_transitions_csv(
        args.out_dir / "transitions_deny.csv",
        global_per_pool["transitions"]["deny"],
        global_per_pool["n_pool"]["deny"],
    )
    _write_transitions_csv(
        args.out_dir / "transitions_allow.csv",
        global_per_pool["transitions"]["allow"],
        global_per_pool["n_pool"]["allow"],
    )
    _write_cell_deltas_csv(args.out_dir / "cell_deltas.csv", agg_br)

    # Full per-cell JSON (per seed × backend × reader)
    full = {}
    for (s, b, r), m in per_cell.items():
        if m.get("n_paired", 0) == 0:
            continue
        full[f"{s}/{b}/{r}"] = {
            "n_paired": m["n_paired"],
            "n_pool": m["n_pool"],
            "marginal": m["marginal"],
            "transitions": m["transitions"],
            "leak": m["leak"],
            "scalars": cell_scalars(m),
        }
    (args.out_dir / "per_cell_full.json").write_text(json.dumps(full, indent=2))

    print()
    print(f"Wrote outputs under {args.out_dir}")
    for f in sorted(args.out_dir.glob("*")):
        print(f"  {f.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
