"""Fit per-reader LLM-generation latency model + summarize per-backend search overhead.

Inputs (auto-discovered):
  out/latency2_answer/<backend>_<reader>_s2/eval_results_s2/<backend>/answer_results_<backend>_<reader>_s2.json
    - per-question records with prompt_tokens, completion_tokens, ttft_ms, answer_time_ms, search_time_ms
    - measured under concurrency=1, vanilla=512 budget

Outputs:
  out/latency_selection/llm_fit_models.json
    {
      "fits": {
        "0_6b":   {"alpha_ms":..., "beta_ms_per_prompt_tok":..., "gamma_ms_per_completion_tok":..., "n_obs":...},
        "llama3b": ...,
        ...
      },
      "search_overhead_ms": {
        "vanilla":   {"median": ..., "p95": ..., "n": ...},   # should be ~0
        "oracle":    ...,                                      # should be ~0
        "inmem":     {"median": ..., "p95": ..., "n": ...},
        "memobase":  ...,   (filled in by separate phase2 run)
        "memsearch": ...,
      },
      "calibration_R2": { ... },
      "validation":   { ... }      # cell-level "predicted vs measured" diagnostics
    }

Model:
  T_LLM_gen(reader, N_p, M_c)  =  alpha_r + beta_r * N_p + gamma_r * M_c          (ms)
  T_total(reader, backend, q)  =  T_LLM_gen(reader, N_p_q, M_c_q) + T_search(backend)

  Fit (alpha, beta, gamma) by ordinary least squares per reader using
  answer_time_ms = alpha + beta * N_p + gamma * M_c    (ignores search_time which is tiny for vanilla/oracle/inmem)

Diagnostics:
  - R^2 per reader
  - Residual percentiles per reader (sanity: |resid| < 100ms typically)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import statistics

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ANS_ROOT = REPO / "out" / "latency2_answer"
ING_ROOT = REPO / "out" / "latency2_ingest"
SPARK_LEGACY_ROOT = REPO / "out"     # latency_spark_<reader>_s2 (May 3 caches)
OUT = REPO / "out" / "latency_selection" / "llm_fit_models.json"

READERS = ["0_6b", "llama3b", "7b", "8b", "32b"]
PHASE1_BACKENDS = ["vanilla", "oracle", "inmem"]
PHASE2_BACKENDS = ["memobase", "memsearch"]
ALL_BACKENDS = PHASE1_BACKENDS + PHASE2_BACKENDS


def _load_retrieve_latency_map(backend: str, reader: str, trial: str = "s2") -> dict[str, float]:
    """Look up retrieve_latency_ms per instance_id from the memcache jsonl.

    Tried locations:
      1. out/latency2_ingest/<be>_<reader>_<trial>/memcache_<be>_A_paired_<reader>_<trial>.jsonl
      2. out/latency_spark_<reader>_<trial>/memory_cache/<be>/<reader>/<trial>/memcache_<be>_A_paired_<reader>_<trial>.jsonl  (May 3 fallback)
    """
    candidates = [
        ING_ROOT / f"{backend}_{reader}_{trial}" / f"memcache_{backend}_A_paired_{reader}_{trial}.jsonl",
        SPARK_LEGACY_ROOT / f"latency_spark_{reader}_{trial}" / "memory_cache" / backend / reader / trial / f"memcache_{backend}_A_paired_{reader}_{trial}.jsonl",
    ]
    for p in candidates:
        if p.exists():
            mp = {}
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("error") is None and d.get("retrieve_latency_ms") is not None:
                    mp[d["instance_id"]] = float(d["retrieve_latency_ms"])
            print(f"  retrieve_latency: {backend}/{reader} ← {p.name} ({len(mp)} ids)")
            return mp
    print(f"  retrieve_latency: {backend}/{reader} ← NO CACHE FOUND")
    return {}


def find_answer_files():
    """Yield (reader, backend, path) for every cell on disk."""
    if not ANS_ROOT.exists():
        return
    for cell_dir in sorted(ANS_ROOT.iterdir()):
        # cell name pattern: <backend>_<reader>_s2
        name = cell_dir.name
        # split by last 2 underscores: "...{backend}_{reader}_s2" → backend, reader
        if not name.endswith("_s2"):
            continue
        for backend in ALL_BACKENDS:
            if name.startswith(backend + "_"):
                reader = name[len(backend) + 1:-3]  # strip "<backend>_" and "_s2"
                if reader in READERS:
                    cands = list(cell_dir.rglob(f"answer_results_{backend}_{reader}_s2.json"))
                    cands = [c for c in cands if "/runs/" not in str(c)]
                    if not cands:
                        cands = list(cell_dir.rglob(f"answer_results_{backend}_{reader}_s2.json"))
                    if cands:
                        yield reader, backend, cands[0]
                break


def load_records():
    rows = []
    for reader, backend, path in find_answer_files():
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            print(f"  WARN: failed to read {path}: {e}")
            continue
        # memsearch / memobase replays from cache → answer-phase search_time_ms is 0.
        # Pull true retrieval latency from the memcache jsonl, joined by instance_id.
        retrieve_map = _load_retrieve_latency_map(backend, reader) if backend in PHASE2_BACKENDS else {}
        rs = data if isinstance(data, list) else (data.get("results") or data.get("details") or [])
        for r in rs:
            if not isinstance(r, dict):
                continue
            np_, mc, ttft, ans = (
                r.get("prompt_tokens"),
                r.get("completion_tokens"),
                r.get("ttft_ms"),
                r.get("answer_time_ms"),
            )
            if any(x is None for x in (np_, mc, ttft, ans)):
                continue
            if mc < 1:
                continue
            if backend in PHASE2_BACKENDS:
                qid = r.get("question_id") or r.get("instance_id") or r.get("id")
                st = retrieve_map.get(qid)
                if st is None:
                    continue  # drop questions with unknown retrieval latency
            else:
                st = r.get("search_time_ms") or 0.0
            rows.append({
                "reader":  reader,
                "backend": backend,
                "n_p":     int(np_),
                "n_c":     int(mc),
                "ttft":    float(ttft),
                "ans":     float(ans),
                "search":  float(st),
            })
    return rows


def fit_one_reader(rs):
    """OLS fit answer_time_ms ≈ α + β·N_p + γ·M_c."""
    X = np.array([[1.0, r["n_p"], r["n_c"]] for r in rs])
    y = np.array([r["ans"] for r in rs])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    resid = y - yhat
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "alpha_ms":                    float(coef[0]),
        "beta_ms_per_prompt_tok":      float(coef[1]),
        "gamma_ms_per_completion_tok": float(coef[2]),
        "n_obs":                       len(rs),
        "R2":                          float(r2),
        "resid_p50_abs_ms":            float(np.percentile(np.abs(resid), 50)),
        "resid_p95_abs_ms":            float(np.percentile(np.abs(resid), 95)),
        "n_p_range":                   [int(min(r["n_p"] for r in rs)),
                                        int(max(r["n_p"] for r in rs))],
        "n_c_range":                   [int(min(r["n_c"] for r in rs)),
                                        int(max(r["n_c"] for r in rs))],
        "ans_range_ms":                [round(min(r["ans"] for r in rs), 1),
                                        round(max(r["ans"] for r in rs), 1)],
    }


def fit_per_reader(rows):
    out = {}
    for reader in READERS:
        rs = [r for r in rows if r["reader"] == reader and r["backend"] in PHASE1_BACKENDS]
        if len(rs) < 10:
            print(f"  reader={reader}: only {len(rs)} obs — fit skipped")
            continue
        out[reader] = fit_one_reader(rs)
    return out


def fit_one_reader_ttft(rs):
    """OLS fit ttft_ms ≈ α' + β'·N_p (no completion-token term: TTFT is prefill-only)."""
    X = np.array([[1.0, r["n_p"]] for r in rs])
    y = np.array([r["ttft"] for r in rs])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    resid = y - yhat
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "alpha_ttft_ms":               float(coef[0]),
        "beta_ttft_ms_per_prompt_tok": float(coef[1]),
        "n_obs":                       len(rs),
        "R2":                          float(r2),
        "resid_p50_abs_ms":            float(np.percentile(np.abs(resid), 50)),
        "resid_p95_abs_ms":            float(np.percentile(np.abs(resid), 95)),
        "ttft_range_ms":               [round(min(r["ttft"] for r in rs), 1),
                                        round(max(r["ttft"] for r in rs), 1)],
    }


def fit_ttft_per_reader(rows):
    out = {}
    for reader in READERS:
        rs = [r for r in rows if r["reader"] == reader and r["backend"] in PHASE1_BACKENDS]
        if len(rs) < 10:
            continue
        out[reader] = fit_one_reader_ttft(rs)
    return out


def search_overhead_per_backend(rows):
    out = {}
    for backend in ALL_BACKENDS:
        ss = [r["search"] for r in rows if r["backend"] == backend]
        if not ss:
            continue
        ss_sorted = sorted(ss)
        n = len(ss_sorted)
        out[backend] = {
            "median": float(ss_sorted[n // 2]),
            "p95":    float(ss_sorted[max(0, int(n * 0.95) - 1)]),
            "mean":   float(sum(ss_sorted) / n),
            "n":      n,
        }
    return out


def cell_predictions(rows, fits):
    """Per-cell predicted vs measured (sanity check). Group by (reader, backend) and report median."""
    out = {}
    cells = defaultdict(list)
    for r in rows:
        cells[(r["reader"], r["backend"])].append(r)
    for (reader, backend), rs in cells.items():
        if reader not in fits:
            continue
        f = fits[reader]
        pred = [f["alpha_ms"] + f["beta_ms_per_prompt_tok"] * r["n_p"] + f["gamma_ms_per_completion_tok"] * r["n_c"] for r in rs]
        meas = [r["ans"] for r in rs]
        rel_err = [abs(p - m) / m for p, m in zip(pred, meas) if m > 0]
        out[f"{backend}_{reader}"] = {
            "n":         len(rs),
            "pred_median_ms": float(statistics.median(pred)),
            "meas_median_ms": float(statistics.median(meas)),
            "rel_err_p50": float(statistics.median(rel_err)) if rel_err else None,
            "rel_err_p95": float(np.percentile(rel_err, 95)) if rel_err else None,
        }
    return out


def predict_all_cells(fits, search_oh, typical_lengths_per_cell):
    """Compose final latency predictions for the 5x5 grid using fitted LLM gen + measured search.

    For memsearch/memobase we only ran the answer phase on 0_6b (cache + LLM gen
    is reader-orthogonal under the assumption that retrieved snippets are the
    same regardless of the reader). To predict their latency on heavier readers
    we substitute the 0_6b prompt-length distribution and apply that reader's
    fitted (α, β, γ).
    """
    predicted = {}
    for reader in READERS:
        if reader not in fits:
            continue
        f = fits[reader]
        for backend in ALL_BACKENDS:
            t = typical_lengths_per_cell.get((reader, backend))
            extrapolated = False
            if not t and backend in PHASE2_BACKENDS:
                # fall back to the 0_6b prompt distribution for memsearch/memobase
                t = typical_lengths_per_cell.get(("0_6b", backend))
                extrapolated = bool(t)
            if not t:
                continue
            n_p, n_c = t["n_p_p50"], t["n_c_p50"]
            t_llm   = f["alpha_ms"] + f["beta_ms_per_prompt_tok"] * n_p + f["gamma_ms_per_completion_tok"] * n_c
            t_search = (search_oh.get(backend) or {}).get("median", 0.0)
            predicted[f"{backend}_{reader}"] = {
                "n_p_p50":          n_p,
                "n_c_p50":          n_c,
                "t_llm_pred_ms":    round(t_llm, 1),
                "t_search_meas_ms": round(t_search, 1),
                "t_total_pred_ms":  round(t_llm + t_search, 1),
                "prompt_extrapolated_from_0_6b": extrapolated,
            }
    return predicted


def typical_lengths(rows):
    """Per (reader, backend), the median (n_p, n_c) used for prediction in the final table."""
    cells = defaultdict(list)
    for r in rows:
        cells[(r["reader"], r["backend"])].append(r)
    out = {}
    for k, rs in cells.items():
        n_ps = sorted(r["n_p"] for r in rs)
        n_cs = sorted(r["n_c"] for r in rs)
        out[k] = {
            "n_p_p50": int(n_ps[len(n_ps) // 2]),
            "n_c_p50": int(n_cs[len(n_cs) // 2]),
            "n_obs":   len(rs),
        }
    return out


def measured_ttft_per_cell(rows):
    """Per (reader, backend) median measured ttft_ms (from answer_results)."""
    cells = defaultdict(list)
    for r in rows:
        cells[(r["reader"], r["backend"])].append(r["ttft"])
    out = {}
    for k, ts in cells.items():
        ts_sorted = sorted(ts)
        out[k] = {
            "ttft_p50_ms": float(ts_sorted[len(ts_sorted) // 2]),
            "n":           len(ts_sorted),
        }
    return out


def predict_ttft_cells(ttft_fits, search_oh, typical_lengths_per_cell, measured_ttft):
    """4×5 TTFT predictions: T_TTFT_total = T_search + (α' + β'·N_p_p50).

    For cells we measured directly use measured ttft median + T_search; for
    Phase-2 readers we lack a direct measurement so substitute the 0_6b
    prompt distribution into the per-reader ttft fit and add T_search.
    """
    predicted = {}
    for reader in READERS:
        if reader not in ttft_fits:
            continue
        f = ttft_fits[reader]
        for backend in ALL_BACKENDS:
            t = typical_lengths_per_cell.get((reader, backend))
            extrapolated = False
            if not t and backend in PHASE2_BACKENDS:
                t = typical_lengths_per_cell.get(("0_6b", backend))
                extrapolated = bool(t)
            if not t:
                continue
            t_search = (search_oh.get(backend) or {}).get("median", 0.0)
            n_p = t["n_p_p50"]
            t_ttft_llm_pred = f["alpha_ttft_ms"] + f["beta_ttft_ms_per_prompt_tok"] * n_p
            measured = measured_ttft.get((reader, backend))
            if measured and not extrapolated:
                t_ttft_llm = measured["ttft_p50_ms"]
                source = "measured"
            else:
                t_ttft_llm = t_ttft_llm_pred
                source = "predicted"
            predicted[f"{backend}_{reader}"] = {
                "n_p_p50":            n_p,
                "t_search_meas_ms":   round(t_search, 1),
                "t_ttft_llm_ms":      round(t_ttft_llm, 1),
                "t_ttft_total_ms":    round(t_ttft_llm + t_search, 1),
                "source":             source,
                "extrapolated":       extrapolated,
                "t_ttft_llm_pred_ms": round(t_ttft_llm_pred, 1),
            }
    return predicted


def main():
    rows = load_records()
    print(f"Loaded {len(rows)} records across cells.")
    if not rows:
        return
    fits      = fit_per_reader(rows)
    ttft_fits = fit_ttft_per_reader(rows)
    search_oh = search_overhead_per_backend(rows)
    val       = cell_predictions(rows, fits)
    typ       = typical_lengths(rows)
    pred      = predict_all_cells(fits, search_oh,
                                  {(r, b): t for (r, b), t in typ.items()})
    meas_ttft = measured_ttft_per_cell(rows)
    pred_ttft = predict_ttft_cells(ttft_fits, search_oh,
                                    {(r, b): t for (r, b), t in typ.items()},
                                    meas_ttft)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fits":               fits,
        "ttft_fits":          ttft_fits,
        "search_overhead_ms": search_oh,
        "validation":         val,
        "typical_lengths":    {f"{b}_{r}": v for (r, b), v in typ.items()},
        "predicted_cells":    pred,
        "measured_ttft_cells": {f"{b}_{r}": v for (r, b), v in meas_ttft.items()},
        "predicted_ttft_cells": pred_ttft,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT}")

    # Pretty stdout summary
    print("\n=== Per-reader LLM-gen fit ===")
    print(f"{'reader':<8} {'α(ms)':>8} {'β(ms/p_tok)':>14} {'γ(ms/c_tok)':>14} {'R²':>6} {'n':>5} {'resid_p95':>10}")
    for r in READERS:
        f = fits.get(r)
        if not f: continue
        print(f"{r:<8} {f['alpha_ms']:>8.1f} {f['beta_ms_per_prompt_tok']*1000:>14.2f} {f['gamma_ms_per_completion_tok']:>14.2f} {f['R2']:>6.3f} {f['n_obs']:>5} {f['resid_p95_abs_ms']:>10.1f}")

    print("\n=== Per-backend search overhead ===")
    for b in ALL_BACKENDS:
        s = search_oh.get(b)
        if not s: continue
        print(f"  {b:<10} median={s['median']:>7.2f}ms  p95={s['p95']:>7.2f}ms  mean={s['mean']:>7.2f}ms  (n={s['n']})")

    print("\n=== Predicted final latency table (median per cell) ===")
    print(f"{'cell':<22} {'n_p':>6} {'n_c':>5} {'t_llm':>8} {'t_search':>10} {'t_total':>10}")
    for r in READERS:
        for b in ALL_BACKENDS:
            p = pred.get(f"{b}_{r}")
            if not p: continue
            print(f"{b}_{r:<14} {p['n_p_p50']:>6} {p['n_c_p50']:>5} {p['t_llm_pred_ms']:>8.1f} {p['t_search_meas_ms']:>10.1f} {p['t_total_pred_ms']:>10.1f}")


if __name__ == "__main__":
    main()
