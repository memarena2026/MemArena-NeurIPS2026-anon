#!/usr/bin/env python3
"""Shared data loader for all paper figure/table generators.

Every generator script in this repo (`paper/gen_tables.py`, root-level
`analyze_*.py`, `gen_*.py`) ultimately needs the same thing: given a run
directory, enumerate every judged cell and return per-dimension accuracies.
This module is that common layer.

## Directory layout (as of 2026-04-18)

The canonical run directory is ``MASim/runs/l_20260408_111046/``, with the
following sub-structure that this loader understands:

    eval_results/           # seed s1  (original single-seed pass)
      vanilla/evaluation_results_vanilla_{model}_4omini.json
      inmem/evaluation_results_rag_{model}_4omini.json
      oracle/evaluation_results_oracle_{model}_4omini.json
      memory_cache/memory_cache/
        evaluation_results_memcache_{backend}_A_paired_{model_raw}.json
    eval_results_s2/        # seeds s2/s3/s4 (stochastic re-runs @ T=0.3)
    eval_results_s3/
    eval_results_s4/
        # same subdirs; note baselines cover only {0_6b, 7b, 8b, 32b}
        # (Llama-3.2-3B was not re-answered for stochastic baselines)

``{model}`` is one of ``0_6b | 7b | 8b | 32b | llama3b`` for baselines.
``{model_raw}`` for structured files follows Memobase/MemOS naming
(``qwen3_0_6b``, ``qwen3_32b_awq``, ``llama3_2_3b``); this module normalises
all of these to the baseline ``{model}`` vocabulary so downstream code can
join them.

## Dimension remap

The simulator emits eval dimensions ``d1..d11``. The paper reports the six
consolidated dimensions D1..D6 (plus a legacy D7 that is no longer shown in
the main table). The mapping lives in ``NEW_FROM_OLD``.

## Why a single accuracy number per cell is not enough

Downstream generators need three different slices of the same data:

* **Category-level** (Rec / Rea / Conf): the main results table
  (``tab:results-L``) averages dimensions inside each category.
* **Per-dimension** (D1..D7): the appendix tables (``tab:results-L-full``
  and ``tab:results-L-subdim``) show each dimension individually.
* **Per-sub-dimension** (d1..d11): the appendix sub-dim table and D6 analysis
  need access to raw eval dims without aggregation.

``CellResult`` exposes all three; ``aggregate_seeds`` turns a list of
``CellResult`` (one per seed) into mean ± std.

The loader is deliberately tolerant: cells with missing files are represented
as ``None`` in the returned grid so that each downstream script can decide
whether to print a dash, fall back to s1, or skip the row.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Constants shared across all paper generators
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN = PROJECT_ROOT / "MASim" / "runs" / "l_20260408_111046"

# Paper dimensions D1..D6 and the simulator eval dimensions they aggregate.
# A legacy D7 exception bucket is kept only so older helper scripts can still
# parse archived outputs, but it is not part of the active paper mapping.
NEW_FROM_OLD: dict[str, list[str]] = {
    "d1_cloze":         ["d5_cloze"],                                 # Recall: cloze fidelity
    "d2_metadata":      ["d6_metadata"],                              # Recall: metadata completeness
    "d3_factual_qa":    ["d7_qa", "d8_temporal", "d10_counterfactual"],  # Reasoning: factual QA (3 sub-types)
    "d4_cross_session": ["d1_conflict", "d2_anaphora"],               # Reasoning: cross-session reasoning
    "d5_abstention":    ["d3_confabulation"],                         # Trust: balanced abstention / true-claim foil scoring
    "d6_permission":    ["d4_permission"],                            # Trust: permission-aware access
    "d7_exception":     ["d11_exception"],                            # Trust: legacy, not in main table
}
DIM_KEYS_PAPER: list[str] = list(NEW_FROM_OLD.keys())  # D1..D7 order
DIM_SHORT: list[str] = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]

# Category indices into DIM_KEYS_PAPER.
CAT_RECALL    = [0, 1]      # D1 Cloze + D2 Metadata
CAT_REASONING = [2, 3]      # D3 Factual QA + D4 Cross-Session
CAT_TRUST     = [4]         # D5 Abstention only (D6 analysed separately in §core-findings)

# Model vocabulary. Keys are the canonical baseline identifier; values are the
# LaTeX string used in table headers.
MODEL_TEX: dict[str, str] = {
    "0_6b":    "Qwen3-0.6B",
    "llama3b": "Llama-3.2-3B",
    "7b":      "Mistral-7B",
    "8b":      "Qwen3-8B",
    "32b":     "Qwen3-32B",
}
# Canonical ordering when rendering model columns in the main table.
MODEL_ORDER: list[str] = ["0_6b", "llama3b", "7b", "8b", "32b"]

# Backend vocabulary. Directory-level keys on the left, paper-facing labels on
# the right.  ``memory_cache`` is the simulator's umbrella directory; we flatten
# it into the two structured-memory backends (memobase / memos).
BACKEND_TEX: dict[str, str] = {
    "vanilla":  "Vanilla",
    "oracle":   "Oracle",
    "rag":      "RAG",
    "memobase": "Memobase",
    "memos":    "MemOS",
}
BACKEND_ORDER: list[str] = ["vanilla", "oracle", "rag", "memobase", "memos"]

# Seeds we expect to find on disk.  s1 came first (single-seed pass) but was
# run with an older answering-pipeline snapshot (pre-Mistral chat-template fix,
# possibly other prompt drift), so s1 baseline scores diverge systematically
# from s2/s3/s4 by ~20pp on some cells.  The paper therefore reports mean±std
# over the three *stochastic* seeds only; s1 is retained on disk for the
# cross-judge appendix (where it supplies both _qwen3 and _4omini files) but
# does NOT enter any main-text aggregate.
#
# ``SEEDS_ALL`` lists every seed the loader can find; ``SEEDS`` is the default
# set that downstream aggregators pull from.  Pass an explicit ``seeds=``
# argument to ``aggregate_seeds`` to override.
SEEDS_ALL: list[str] = ["s1", "s2", "s3", "s4"]
SEEDS: list[str] = ["s2", "s3", "s4"]


# ---------------------------------------------------------------------------
# Filename parsers
# ---------------------------------------------------------------------------

# Canonicalises the model fragment that appears inside a structured memcache
# filename (``qwen3_0_6b``, ``qwen3_8b_awq``, ...) into the short baseline
# identifier used in MODEL_ORDER. Returns ``None`` if the fragment is not a
# model we track (e.g. ``gpt41mini`` under the Config-B closed-API path).
def _normalise_memcache_model(raw: str) -> str | None:
    # Order matters: strip the longest variant suffix first.
    mapping = [
        ("qwen3_0_6b",         "0_6b"),
        ("qwen3_8b_awq",       "8b"),
        ("qwen3_8b",           "8b"),
        ("qwen3_32b_awq",      "32b"),
        ("qwen3_32b",          "32b"),
        ("llama3_2_3b",        "llama3b"),
        ("mistral_7b",         "7b"),
    ]
    for needle, canonical in mapping:
        if raw == needle or raw.startswith(needle):
            return canonical
    return None


# Regex for baseline judge files: ``evaluation_results_{backend}_{model}_4omini.json``.
_BASELINE_RE = re.compile(
    r"^evaluation_results_(?P<backend>vanilla|rag|oracle)_"
    r"(?P<model>0_6b|32b|8b|7b|llama3b)_4omini\.json$"
)

# Regex for structured memcache judge files. Note that these do **not** carry a
# ``_4omini`` suffix in the current snapshot; judge-vs-self is encoded in the
# ``summary`` field inside the JSON.
_MEMCACHE_RE = re.compile(
    r"^evaluation_results_memcache_(?P<backend>memobase|memos)_"
    r"(?P<config>A_paired|B_remote)_(?P<model_raw>.+?)\.json$"
)


def _seed_from_dir(seed_dir: str) -> str:
    # ``eval_results`` → s1 (historical); ``eval_results_s2`` → s2 etc.
    return "s1" if seed_dir == "eval_results" else seed_dir.split("_")[-1]


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

@dataclass
class CellResult:
    """One (seed, backend, model) evaluation.

    Attributes
    ----------
    seed, backend, model
        Identifiers; ``backend`` uses the ``BACKEND_TEX`` keyset, not the
        on-disk directory name (``rag`` rather than ``inmem``).
    per_sub_dim
        ``{sub_dim_id: (n_correct, n_total)}`` at the finest granularity the
        simulator reports (d1..d11 + any new schema variants that show up in
        the JSON).  Downstream aggregations (`accuracy_by_paper_dim`, etc.)
        derive from this single dict to avoid double-counting.
    source_path
        Absolute path to the loaded JSON. Useful for debugging a cell that
        looks off.
    """

    seed: str
    backend: str
    model: str
    per_sub_dim: dict[str, tuple[int, int]]
    source_path: Path

    # --- derived quantities -------------------------------------------------

    def total(self) -> tuple[int, int]:
        c = sum(x[0] for x in self.per_sub_dim.values())
        n = sum(x[1] for x in self.per_sub_dim.values())
        return c, n

    def accuracy(self) -> float | None:
        c, n = self.total()
        return c / n if n else None

    def accuracy_by_paper_dim(self) -> dict[str, float | None]:
        """Return ``{'d1_cloze': acc, ..., 'd7_exception': acc}``.

        Macro-mean across sub-dims: each sub-type contributes equally
        regardless of instance count. This matches the §5 spec
        ("each per-dim cell is itself a macro-mean over its internal
        sub-types") and keeps appendix per-dim cells consistent with
        the macro-mean Rec/Rea/Avg in the main table.
        """
        out: dict[str, float | None] = {}
        for paper_dim, sub_dims in NEW_FROM_OLD.items():
            sub_accs: list[float] = []
            for sd in sub_dims:
                c, n = self.per_sub_dim.get(sd, (0, 0))
                if n > 0:
                    sub_accs.append(c / n)
            out[paper_dim] = sum(sub_accs) / len(sub_accs) if sub_accs else None
        return out

    def accuracy_by_sub_dim(self) -> dict[str, float | None]:
        return {sd: (c / n if n else None) for sd, (c, n) in self.per_sub_dim.items()}


# ---------------------------------------------------------------------------
# Low-level JSON reader
# ---------------------------------------------------------------------------

# Map from simulator ``question_id`` prefix (``d1``, ``d10``) to the canonical
# sub-dimension identifier used by NEW_FROM_OLD.  Structured-memcache JSON
# dumps do NOT set a ``dimension`` field, so we must derive it from the qid.
# Keep this in lock-step with ``NEW_FROM_OLD``.
_QID_PREFIX_TO_SUB_DIM: dict[str, str] = {
    "d1":  "d1_conflict",
    "d2":  "d2_anaphora",
    "d3":  "d3_confabulation",
    "d4":  "d4_permission",
    "d5":  "d5_cloze",
    "d6":  "d6_metadata",
    "d7":  "d7_qa",
    "d8":  "d8_temporal",
    "d9":  "d9_negation",
    "d10": "d10_counterfactual",
    "d11": "d11_exception",
}


def _sub_dim_from_qid(qid: str) -> str | None:
    """``'d10_a1b2c3' → 'd10_counterfactual'``. Returns None on malformed input."""
    if not qid:
        return None
    prefix = qid.split("_", 1)[0]
    return _QID_PREFIX_TO_SUB_DIM.get(prefix)


def _load_cell_json(path: Path) -> dict[str, tuple[int, int]]:
    """Parse a single evaluation_results JSON into {sub_dim: (correct, total)}.

    The scoring pipeline uses two different correctness fields depending on the
    schema version:

    * Early s1 dumps store ``judge_correct`` (bool) when ``scoring_method ==
      "judge"`` and ``correct`` otherwise (deterministic D6).
    * s2/s3/s4 dumps just use ``correct`` uniformly; ``judge_correct`` is
      absent.

    We honour both. For rows the scorer flagged as
    ``answer_scoring_skipped`` (LLM_ERROR, TEXT_SESSIONS_ERROR, etc.) we drop
    the row entirely so an infrastructure failure does not look like a wrong
    answer.  The sub-dimension identifier is pulled from ``dimension`` if
    present (baseline files), else derived from the ``question_id`` prefix
    (structured-memcache files); see :func:`_sub_dim_from_qid`.
    """
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    details = data.get("details", [])
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for d in details:
        # Filter rows the judge could not score. ``answer_scored == False`` is
        # the canonical flag; legacy dumps mark ``raw_response`` with
        # ``LLM_ERROR`` or ``TEXT_SESSIONS_ERROR`` strings instead.
        if d.get("answer_scored") is False:
            continue
        raw = d.get("raw_response") or ""
        if isinstance(raw, str) and (raw.startswith("LLM_ERROR") or raw.startswith("TEXT_SESSIONS_ERROR")):
            continue

        # Extract sub-dim identifier. Baseline dumps carry a canonical
        # ``dimension`` field; structured-memcache dumps omit it and we have
        # to reconstruct from the qid prefix (``d10_...`` →
        # ``d10_counterfactual``, etc.).
        sub = d.get("dimension")
        if not sub:
            sub = _sub_dim_from_qid(d.get("question_id", ""))
        if not sub:
            continue

        # Resolve correctness across both schema versions.
        if d.get("scoring_method") == "judge":
            val = d.get("judge_correct")
            if val is None:
                val = d.get("correct", False)
        else:
            val = d.get("correct", False)

        bucket = out[sub]
        bucket[1] += 1
        bucket[0] += int(bool(val))
    return {k: (v[0], v[1]) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_all_cells(run_dir: Path = DEFAULT_RUN) -> dict[tuple[str, str, str], CellResult]:
    """Scan ``run_dir`` and return every judged cell we can find.

    Returns
    -------
    grid : ``{(seed, backend, model): CellResult}``
        ``seed`` ∈ SEEDS; ``backend`` ∈ BACKEND_ORDER; ``model`` is the short
        identifier (``'0_6b'``, ``'llama3b'``, ...). Missing cells simply
        don't appear in the mapping; callers should use ``grid.get(...)``.

    Notes
    -----
    * Under A_paired memcache files we always trust the top-level
      ``evaluation_results_memcache_*.json`` (the one outside the per-run
      timestamped subfolder) as the canonical snapshot.
    * Config-B files (``B_remote_gpt41mini*``) are skipped: they belong to the
      closed-API comparison referenced in a footnote, not the main table.
    """
    grid: dict[tuple[str, str, str], CellResult] = {}
    for seed_dir in ("eval_results", "eval_results_s2", "eval_results_s3", "eval_results_s4"):
        seed = _seed_from_dir(seed_dir)
        seed_root = run_dir / seed_dir
        if not seed_root.exists():
            continue

        # ---- Baseline (vanilla / rag / oracle) ----
        backend_dir_map = {"vanilla": "vanilla", "inmem": "rag", "oracle": "oracle"}
        for on_disk, backend in backend_dir_map.items():
            bdir = seed_root / on_disk
            if not bdir.exists():
                continue
            for f in bdir.glob("evaluation_results_*_4omini.json"):
                m = _BASELINE_RE.match(f.name)
                if not m:
                    continue
                if m.group("backend") not in {"vanilla", "rag", "oracle"}:
                    continue
                model = m.group("model")
                cell = CellResult(
                    seed=seed, backend=backend, model=model,
                    per_sub_dim=_load_cell_json(f), source_path=f,
                )
                if cell.per_sub_dim:
                    grid[(seed, backend, model)] = cell

        # ---- Structured memcache (memobase / memos) ----
        # The memcache judge dumps live two directories deep.  We also
        # deliberately ignore the timestamped ``runs/*/`` replica files: those
        # are intermediate snapshots the eval harness writes before promoting
        # to the top-level result.
        mc = seed_root / "memory_cache" / "memory_cache"
        if mc.exists():
            for f in mc.glob("evaluation_results_memcache_*.json"):
                # Skip timestamped run replicas: they duplicate the top-level file.
                # We look for ``runs`` *relative to mc*, not in the whole path
                # (the canonical run lives under ``MASim/runs/…`` which would
                # otherwise match and discard every structured cell).
                if "runs" in f.relative_to(mc).parts:
                    continue
                m = _MEMCACHE_RE.match(f.name)
                if not m:
                    continue
                if m.group("config") != "A_paired":
                    # Config-B is the closed-API (GPT-4.1-mini) comparison; out of scope.
                    continue
                model = _normalise_memcache_model(m.group("model_raw"))
                if model is None:
                    continue
                backend = m.group("backend")
                cell = CellResult(
                    seed=seed, backend=backend, model=model,
                    per_sub_dim=_load_cell_json(f), source_path=f,
                )
                if cell.per_sub_dim:
                    grid[(seed, backend, model)] = cell
    return grid


# ---------------------------------------------------------------------------
# Seed aggregation
# ---------------------------------------------------------------------------

@dataclass
class Stat:
    """Mean ± (sample) standard deviation across seeds, plus book-keeping.

    ``n`` is the number of seeds that actually contributed. Tables should
    render a dash when ``n == 0`` and fall back to point estimate (no ±) when
    ``n == 1``.
    """

    mean: float
    std: float
    n: int

    @classmethod
    def from_values(cls, values: Iterable[float | None]) -> "Stat":
        xs = [float(v) for v in values if v is not None and not math.isnan(float(v))]
        if not xs:
            return cls(mean=float("nan"), std=float("nan"), n=0)
        mean = sum(xs) / len(xs)
        if len(xs) < 2:
            return cls(mean=mean, std=0.0, n=len(xs))
        var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
        return cls(mean=mean, std=math.sqrt(var), n=len(xs))


def aggregate_seeds(
    grid: dict[tuple[str, str, str], CellResult],
    backend: str, model: str,
    dim_selector,  # callable(CellResult) -> float | None
    seeds: Iterable[str] = SEEDS,
) -> Stat:
    """Aggregate one scalar across the requested seeds.

    Parameters
    ----------
    dim_selector : function
        Given a ``CellResult``, return the scalar to aggregate (accuracy for
        one paper dim, a category average, or anything else). Return
        ``None`` to say "this cell does not have this quantity".

    Example::

        rec_stat = aggregate_seeds(
            grid, "memobase", "0_6b",
            lambda c: _category_acc(c, CAT_RECALL),
        )
    """
    values: list[float | None] = []
    for seed in seeds:
        cell = grid.get((seed, backend, model))
        if cell is None:
            continue
        values.append(dim_selector(cell))
    return Stat.from_values(values)


def category_accuracy(cell: CellResult, indices: Iterable[int]) -> float | None:
    """Category accuracy = two-level macro-mean over paper-dim indices.

    For each paper-dim index (e.g. D2 = Metadata), the dim's score is the
    macro-mean over its sub-types (each sub-type contributes equally,
    regardless of instance count). For multiple indices (e.g. Rec = [0, 1]),
    the returned value is the macro-mean of the per-dim macro-means.

    Returns ``None`` when every contributing sub-dim is absent.
    """
    dim_means: list[float] = []
    for i in indices:
        paper_dim = DIM_KEYS_PAPER[i]
        sub_accs: list[float] = []
        for sub in NEW_FROM_OLD[paper_dim]:
            c, n = cell.per_sub_dim.get(sub, (0, 0))
            if n > 0:
                sub_accs.append(c / n)
        if sub_accs:
            dim_means.append(sum(sub_accs) / len(sub_accs))
    return sum(dim_means) / len(dim_means) if dim_means else None


# ---------------------------------------------------------------------------
# LaTeX formatting helpers
# ---------------------------------------------------------------------------

def fmt_pct(val: float | None, bold: bool = False, decimals: int = 1) -> str:
    """Render a fraction as a percentage.  ``None`` becomes ``--``."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "--"
    s = f"{val * 100:.{decimals}f}"
    return f"\\textbf{{{s}}}" if bold else s


def fmt_stat(stat: Stat, bold: bool = False, decimals: int = 1) -> str:
    """Render a ``Stat`` as ``mean±std`` (percentage points).

    * ``n == 0``   → ``--``
    * ``n == 1``   → point estimate only, no ``±`` (honest: one sample can't
      have an std deviation); print a footnote hint via caller if needed.
    * ``n >= 2``   → ``mean ± std``.

    Bolding applies to the mean only.
    """
    if stat.n == 0:
        return "--"
    m = f"{stat.mean * 100:.{decimals}f}"
    if stat.n == 1:
        return f"\\textbf{{{m}}}" if bold else m
    s = f"{stat.std * 100:.{decimals}f}"
    body = f"{m}{{\\scriptsize$\\pm${s}}}"
    return f"\\textbf{{{body}}}" if bold else body


# ---------------------------------------------------------------------------
# Diagnostic helper: quick coverage report
# ---------------------------------------------------------------------------

def print_coverage(grid: dict[tuple[str, str, str], CellResult]) -> None:
    """Print one line per (backend, model) listing which seeds are present.

    Handy when debugging a missing cell before running a generator that will
    silently render the cell as a dash.
    """
    by_bm: dict[tuple[str, str], list[str]] = defaultdict(list)
    for (seed, backend, model), _ in grid.items():
        by_bm[(backend, model)].append(seed)
    for (backend, model) in sorted(by_bm):
        seeds = sorted(by_bm[(backend, model)])
        print(f"{backend:10s} {model:10s} seeds={seeds}")


if __name__ == "__main__":
    # Invoke directly to sanity-check the loader on the canonical run.
    grid = load_all_cells()
    print(f"Loaded {len(grid)} cells from {DEFAULT_RUN}")
    print_coverage(grid)
